"""Cross-check sealed native DN with GDAL SAFE; never score or replace inputs."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rasterio
from rasterio.windows import Window
from scipy.interpolate import RegularGridInterpolator

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.hh_hv_transfer import summary
from agents.reserved_sar import DOWNLOAD, Cdse
from agents.source_annotations import replay_prepared
from agents.source_reservation import checked, ref
from agents.statoil_baseline import available_commit

POLICY = REPO_ROOT / "configs/reservations/source_crosscheck_v010.json"


def fresh(path: Path) -> None:
    if path.exists() or not path.resolve().is_relative_to(
        (REPO_ROOT / "data").resolve()
    ):
        raise ValueError("Fresh project data directory required")
    path.mkdir(parents=True)


def fetch_metadata(prepared: Path, output: Path) -> None:
    replay_prepared(prepared)
    manifest = json.loads(prepared.read_text())
    fresh(output)
    client = Cdse()
    records = []
    for product in manifest["products"]:
        directory = output / product["product_id"]
        directory.mkdir()
        base = f"{DOWNLOAD}({product['product_id']})/Nodes({product['name']})"
        path = directory / "manifest.safe"
        path.write_bytes(client.fetch(base + "/Nodes(manifest.safe)/$value"))
        noises = []
        for band in product["bands"]:
            filename = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
            noise_name = "noise-" + filename[:-5] + ".xml"
            noise = directory / noise_name
            noise.write_bytes(
                client.fetch(
                    base
                    + "/Nodes(annotation)/Nodes(calibration)"
                    + f"/Nodes({noise_name})/$value"
                )
            )
            tree = ET.parse(noise)
            if (
                tree.findtext(".//adsHeader/polarisation", "").upper()
                != band["polarization"]
            ):
                raise ValueError("Noise polarization mismatch")
            noises.append({"polarization": band["polarization"], "source": ref(noise)})
        records.append(
            {
                "product_id": product["product_id"],
                "manifest": ref(path),
                "noise": noises,
            }
        )
        save(
            output / f"progress_{len(records)}.json",
            {"products": records, "network_bytes": client.bytes},
        )
    save(
        output / "metadata.json",
        {"prepared": ref(prepared), "products": records, "network_bytes": client.bytes},
    )
    print(
        json.dumps(
            {"metadata": ref(output / "metadata.json"), "network_bytes": client.bytes}
        )
    )


def sparse_measurement(
    path: Path, width: int, height: int, windows: list[tuple[list[int], Any]]
) -> None:
    """Only saved native windows exist; zeros elsewhere MUST NOT be evaluated."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint16",
        tiled=False,
        blockysize=1,
        SPARSE_OK=True,
    ) as dst:
        for bounds, dn in windows:
            col, row, w, h = bounds
            if (
                dn.shape != (h, w)
                or min(col, row) < 0
                or col + w > width
                or row + h > height
            ):
                raise ValueError("Sparse native crop dimensions mismatch")
            dst.write(dn, 1, window=Window(col, row, w, h))


def reference_db(xml: Path, dn: Any, bounds: list[int]) -> Any:
    """Separate float64 XML parser and SciPy interpolation of ESA's equation."""
    vectors = ET.parse(xml).findall(".//calibrationVector")
    lines = np.array([float(v.findtext("line", "nan")) for v in vectors])
    pixels = [np.fromstring(v.findtext("pixel", ""), sep=" ") for v in vectors]
    values = [np.fromstring(v.findtext("sigmaNought", ""), sep=" ") for v in vectors]
    if (
        len(lines) < 2
        or any(not np.array_equal(p, pixels[0]) for p in pixels)
        or len(pixels[0]) < 2
        or any(len(v) != len(pixels[0]) for v in values)
    ):
        raise ValueError("Reference requires a complete regular calibration grid")
    col, row, width, height = bounds
    rr, cc = np.meshgrid(
        np.clip(np.arange(row, row + height), lines[0], lines[-1]),
        np.clip(np.arange(col, col + width), pixels[0][0], pixels[0][-1]),
        indexing="ij",
    )
    lut = RegularGridInterpolator((lines, pixels[0]), np.stack(values))(
        np.stack([rr, cc], axis=-1)
    )
    if not (lut > 0).all() or not (dn > 0).all():
        raise ValueError("Positive native pixels and calibration required")
    return 20 * np.log10(dn.astype("float64") / lut)


def gdal_check(
    safe: Path, pol: str, bounds: list[int], dn: Any, expected: Any
) -> dict[str, Any]:
    """Execute GDAL's calibration code independently of our Python calibrator."""
    window = Window(*bounds)
    manifest = (safe / "manifest.safe").as_posix()
    with rasterio.open(f"SENTINEL1_CALIB:UNCALIB:{manifest}:IW_{pol}:AMPLITUDE") as raw:
        if not np.array_equal(raw.read(1, window=window), dn):
            raise ValueError("GDAL reference did not read identical source pixels")
    try:
        with rasterio.open(
            f"SENTINEL1_CALIB:SIGMA0:{manifest}:IW_{pol}:INTENSITY"
        ) as calibrated:
            linear = calibrated.read(1, window=window).astype("float64")
            gcps, _ = calibrated.gcps
    except rasterio.errors.RasterioIOError as error:
        return {
            "raw_pixels_equal": True,
            "calibration_executed": False,
            "failure": str(error),
            "passed": False,
        }
    if not np.isfinite(linear).all() or not (linear > 0).all():
        raise ValueError("Invalid GDAL calibration")
    delta = 10 * np.log10(linear) - expected
    return {
        "raw_pixels_equal": True,
        "calibration_executed": True,
        "max_absolute_db": float(np.abs(delta).max()),
        "mean_signed_db": float(delta.mean()),
        "rmse_db": float(np.sqrt(np.square(delta).mean())),
        "pixels": int(linear.size),
        "gdal_first_gcp_pixel_row": [gcps[0].col, gcps[0].row] if gcps else None,
    }


def run(
    prepared: Path, metadata_path: Path, collection_path: Path, output: Path
) -> None:
    if available_commit() < 1024**3 or psutil.virtual_memory().available < 1024**3:
        raise RuntimeError("One GiB memory headroom required for bounded cross-check")
    if shutil.disk_usage(REPO_ROOT).free < 2 * 1024**3:
        raise ValueError("Two GiB free required for bounded sparse reference wrappers")
    replay_prepared(prepared)
    manifest = json.loads(prepared.read_text())
    metadata = json.loads(metadata_path.read_text())
    if metadata["prepared"] != ref(prepared):
        raise ValueError("Reference metadata preparation mismatch")
    collection = json.loads(collection_path.read_text())
    if collection["prepared"] != ref(prepared):
        raise ValueError("Collection preparation mismatch")
    policy = json.loads(POLICY.read_text())
    fresh(output)
    results, noise_inventory = [], []
    for product in manifest["products"]:
        source = next(
            p for p in metadata["products"] if p["product_id"] == product["product_id"]
        )
        safe = output / product["name"]
        (safe / "measurement").mkdir(parents=True)
        (safe / "annotation/calibration").mkdir(parents=True)
        shutil.copyfile(checked(source["manifest"]), safe / "manifest.safe")
        for index, band in enumerate(product["bands"]):
            filename = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
            annotation = checked(band["annotation"])
            shutil.copyfile(annotation, safe / "annotation" / (filename[:-5] + ".xml"))
            shutil.copyfile(
                checked(band["calibration"]),
                safe
                / "annotation/calibration"
                / ("calibration-" + filename[:-5] + ".xml"),
            )
            noise = next(
                n for n in source["noise"] if n["polarization"] == band["polarization"]
            )
            npth = checked(noise["source"])
            shutil.copyfile(npth, safe / "annotation/calibration" / npth.name)
            tree = ET.parse(npth)
            noise_inventory.append(
                {
                    "product_id": product["product_id"],
                    "polarization": band["polarization"],
                    "source": noise["source"],
                    "legacy_vectors": len(tree.findall(".//noiseVector")),
                    "range_vectors": len(tree.findall(".//noiseRangeVector")),
                    "azimuth_vectors": len(tree.findall(".//noiseAzimuthVector")),
                    "thermal_noise_corrected": band["thermal_noise_corrected"],
                    "correction_applied_by_this_audit": False,
                }
            )
            windows = [
                (
                    a["window"],
                    np.load(checked(a["bands"][index]["dn"]), allow_pickle=False),
                )
                for a in product["areas"]
            ]
            sparse_measurement(
                safe / "measurement" / filename, band["width"], band["height"], windows
            )
        for area in product["areas"]:
            with rasterio.open(checked(area["chip"])) as dst:
                expected = dst.read()
            for index, band in enumerate(product["bands"]):
                dn = np.load(checked(area["bands"][index]["dn"]), allow_pickle=False)
                gdal = gdal_check(
                    safe, band["polarization"], area["window"], dn, expected[index]
                )
                reference = reference_db(
                    checked(band["calibration"]), dn, area["window"]
                )
                delta = reference - expected[index]
                result: dict[str, Any] = {
                    "max_absolute_db": float(np.abs(delta).max()),
                    "mean_signed_db": float(delta.mean()),
                    "rmse_db": float(np.sqrt(np.square(delta).mean())),
                    "pixels": int(delta.size),
                    "gdal": gdal,
                }
                result.update(
                    area_id=area["id"],
                    polarization=band["polarization"],
                    source_dn=area["bands"][index]["dn"],
                    passed=result["max_absolute_db"]
                    <= policy["calibration_tolerance_db"],
                )
                results.append(result)
        print(json.dumps({"checked_product": product["product_id"]}), flush=True)
    # Publisher training inputs only; no Kaggle competition test or NL locked pixels.
    frozen_summary = json.loads(checked(policy["training_summary"]).read_text())
    training = frozen_summary["kaggle_training_reference"]
    if (
        training["partition"] != "train_only"
        or sum(c["chips"] for c in training["classes"].values()) != 1122
    ):
        raise ValueError("Frozen training-only reference required")
    checked(training["source"])
    checked(training["split"])
    local = json.loads(checked(collection["local_report"]).read_text())
    samples = [
        np.load(checked(s["chip"]), allow_pickle=False)
        for s in local["samples"]
        if s.get("chip")
    ]
    report = {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "prepared": ref(prepared),
        "metadata": ref(metadata_path),
        "collection": ref(collection_path),
        "gdal_version": rasterio.__gdal_version__,
        "rasterio_version": rasterio.__version__,
        "calibration": results,
        "numerical_reference_all_passed": all(r["passed"] for r in results),
        "gdal_calibration_all_passed": all(
            r["gdal"].get("calibration_executed")
            and r["gdal"].get("max_absolute_db", 1)
            <= policy["calibration_tolerance_db"]
            for r in results
        ),
        "noise_inventory": noise_inventory,
        "kaggle_training": training,
        "training_summary_reused": policy["training_summary"],
        "distribution": {
            "kaggle_ships": training["classes"]["ship"],
            "kaggle_icebergs": training["classes"]["iceberg"],
            "native_mixed_provisional": summary(samples),
        },
        "compatibility_established": False,
        "training_approved": False,
        "independent_review": False,
        "classifier_scored": False,
        "reference_scope": "Separate float64 XML parser and SciPy interpolation "
        "of ESA sigma0 equation; GDAL SAFE raw-pixel equality and attempted "
        "calibration on sparse full-dimension wrappers",
        "limitations": [
            "No thermal noise correction, terrain correction or resampling",
            "External calibrated view execution is reported separately; no SNAP run",
            "Kaggle physical scale and complete processing chain unknown",
            "Distribution overlap cannot establish compatibility or iceberg truth",
        ],
    }
    save(output / "report.json", report)
    print(
        json.dumps(
            {
                "report": ref(output / "report.json"),
                "numerical_reference_all_passed": report[
                    "numerical_reference_all_passed"
                ],
                "gdal_calibration_all_passed": report["gdal_calibration_all_passed"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--collection", type=Path)
    args = parser.parse_args()
    if args.fetch_metadata:
        fetch_metadata(args.prepared, args.output)
    elif args.metadata and args.collection:
        run(args.prepared, args.metadata, args.collection, args.output)
    else:
        parser.error("Choose metadata acquisition or cross-check with collection")
