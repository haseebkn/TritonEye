"""Replay native metadata noise power at four provisional development targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.native_nl_input import SPEC, prepare
from agents.nl_benchmark import require_development_scene
from agents.region import load_region
from agents.snap_reference import bands
from agents.source_preprocessing import reference_db
from agents.source_reservation import checked, ref

POLICY = REPO_ROOT / "configs/reservations/target_noise_v010.json"
OUTPUT = REPO_ROOT / "docs/evidence/target_noise_v010.json"
LOCK = REPO_ROOT / "configs/reservations/target_noise_v010.lock.json"


def interpolate(nodes: Any, values: Any, query: Any) -> Any:
    nodes, values = np.asarray(nodes), np.asarray(values)
    if (
        nodes.ndim != 1
        or values.shape != nodes.shape
        or len(nodes) == 0
        or not np.isfinite(nodes).all()
        or not np.isfinite(values).all()
        or (np.diff(nodes) <= 0).any()
    ):
        raise ValueError("Finite ordered equal-length LUT vectors required")
    if len(nodes) == 1:
        return np.full(np.shape(query), values[0], dtype="float64")
    return interp1d(nodes, values, fill_value="extrapolate")(query)


def noise_power(
    xml: Path, bounds: list[int], polarization: str, snap_precision: bool = False
) -> Any:
    """Range/azimuth product in original DN-power units at native offsets.

    Supports the modern GRD noise schema used in the four pinned acquisitions.
    Missing or overlapping coverage is an error, never guessed noise or zero.
    """
    root = ET.parse(xml)
    if root.findtext(".//adsHeader/polarisation") != polarization:
        raise ValueError("Noise polarization mismatch")
    x, y, width, height = bounds
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Positive native bounds required")
    columns, rows = np.arange(x, x + width), np.arange(y, y + height)
    vectors = root.findall(".//noiseRangeVector")
    if not vectors or root.find(".//noiseVectorList") is not None:
        raise ValueError("Modern range/azimuth noise schema required")
    lines, profiles = [], []
    for vector in vectors:
        lines.append(float(vector.findtext("line", "nan")))
        profiles.append(
            interpolate(
                np.fromstring(vector.findtext("pixel", ""), sep=" "),
                np.fromstring(vector.findtext("noiseRangeLut", ""), sep=" ")
                .astype("float32" if snap_precision else "float64")
                .astype("float64"),
                columns,
            )
        )
    # Each native range column is interpolated in original azimuth coordinates.
    range_power = np.stack(
        [interpolate(lines, np.stack(profiles)[:, c], rows) for c in range(width)],
        axis=1,
    )
    result = np.full((height, width), np.nan)
    coverage = np.zeros(result.shape, dtype="int32")
    for vector in root.findall(".//noiseAzimuthVector"):
        x0, x1, y0, y1 = [
            int(vector.findtext(name, "-1"))
            for name in (
                "firstRangeSample",
                "lastRangeSample",
                "firstAzimuthLine",
                "lastAzimuthLine",
            )
        ]
        inside = (
            (rows[:, None] >= y0)
            & (rows[:, None] <= y1)
            & (columns[None, :] >= x0)
            & (columns[None, :] <= x1)
        )
        if not inside.any():
            continue
        factor = interpolate(
            np.fromstring(vector.findtext("line", ""), sep=" "),
            np.fromstring(vector.findtext("noiseAzimuthLut", ""), sep=" ")
            .astype("float32" if snap_precision else "float64")
            .astype("float64"),
            rows,
        )
        combined = range_power * factor[:, None]
        result[inside] = combined[inside]
        coverage[inside] += 1
    if (coverage != 1).any() or not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Incomplete, overlapping or nonpositive noise coverage")
    return result


def masks(
    shape: tuple[int, int],
    boxes: list[list[int]],
    index: int,
    guard: int,
    outer: int,
) -> tuple[Any, Any]:
    if not 0 <= guard < outer:
        raise ValueError("Background outer margin must exceed guard")
    yy, xx = np.indices(shape)

    def box_mask(b: list[int], margin: int) -> Any:
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError("Target box outside native context")
        return (
            (xx >= x0 - margin)
            & (xx < x1 + margin)
            & (yy >= y0 - margin)
            & (yy < y1 + margin)
        )

    target = box_mask(boxes[index], 0)
    background = box_mask(boxes[index], outer)
    for box in boxes:
        background &= ~box_mask(box, guard)
    if not target.any() or not background.any():
        raise ValueError("Nonempty target and surrounding context required")
    return target, background


def quantiles(values: Any) -> list[float]:
    if not np.isfinite(values).all() or not np.size(values):
        raise ValueError("Finite nonempty diagnostic values required")
    return [round(float(v), 6) for v in np.quantile(values, [0.05, 0.5, 0.95])]


def region_stats(
    dn: Any, noise: Any, nominal: Any, corrected: Any, mask: Any
) -> dict[str, Any]:
    power = dn.astype("float64")[mask] ** 2
    estimated = noise[mask]
    residual = power - estimated
    return {
        "pixels": int(mask.sum()),
        "dn_power_q05_50_95": quantiles(power),
        "metadata_noise_power_q05_50_95": quantiles(estimated),
        "residual_power_q05_50_95": quantiles(residual),
        "observed_to_noise_db_q05_50_95": quantiles(10 * np.log10(power / estimated)),
        "nonpositive_residual_fraction": round(float(np.mean(residual <= 0)), 6),
        "nominal_sigma0_db_q05_50_95": quantiles(nominal[mask]),
        "corrected_sigma0_db_q05_50_95": quantiles(corrected[mask]),
        "corrected_delta_db_q05_50_95": quantiles(corrected[mask] - nominal[mask]),
    }


def require_uncorrected(annotation: ET.ElementTree[Any]) -> None:
    if annotation.findtext(".//thermalNoiseCorrectionPerformed") != "false":
        raise ValueError("Uncorrected source required; refuse double subtraction")


def replay() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text())
    public = json.loads(checked(policy["reference"]).read_text())
    collection = json.loads(checked(policy["collection"]).read_text())
    snap = json.loads(checked(public["private_inputs"]["snap"]).read_text())
    prepared = json.loads(checked(snap["prepared"]).read_text())
    metadata = json.loads(checked(snap["metadata"]).read_text())
    artifacts = {r["path"]: r for r in public["snap_outputs"]}
    used: dict[str, dict[str, str]] = {}

    def artifact(reference: dict[str, str]) -> Path:
        path = checked(reference)
        used[reference["path"]] = reference
        return path

    results = []
    for identity in policy["area_ids"]:
        label = next(a for a in collection["areas"] if a["id"] == identity)
        if label["split"] not in {"train", "validation"} or any(
            o["provisional"] is not True or o["ignore_for_accuracy"] is not True
            for o in label["objects"]
        ):
            raise ValueError("Provisional development objects required")
        product = next(
            p for p in prepared["products"] if p["product_id"] == label["product_id"]
        )
        native = next(a for a in product["areas"] if a["id"] == identity)
        reference = next(a for a in snap["areas"] if a["id"] == identity)
        source = next(
            p for p in metadata["products"] if p["product_id"] == label["product_id"]
        )
        require_development_scene(
            product["product_id"],
            {
                "name": product["name"],
                "split": label["split"],
                "acquisition_group": native["native_acquisition_group"],
                "geographic_group": native["geographic_group"],
                "study_roi": native["geometry"],
            },
        )
        load_region(native["geometry"])
        if (
            reference["window"] != native["window"]
            or reference["source_chip"] != native["chip"]
        ):
            raise ValueError("SNAP native offsets or source differ")
        artifact(native["chip"])
        manifest = ET.parse(artifact(source["manifest"]))
        ipf = sorted(
            {
                e.attrib["version"]
                for e in manifest.iter()
                if e.tag.split("}")[-1] == "software"
                and e.attrib.get("name") == "Sentinel-1 IPF"
            }
        )
        directory = artifact(reference["variants"]["raw"]["graph"]).parent
        # Bind all sidecars used by bands(), not only numerical image files.
        for path, item in artifacts.items():
            if Path(path).parent == directory.relative_to(REPO_ROOT) or path.startswith(
                directory.relative_to(REPO_ROOT).as_posix() + "/"
            ):
                artifact(item)
        raw = bands(directory / "raw.dim", raw=True)
        nominal, corrected = bands(directory / "calibration.dim"), bands(
            directory / "noise.dim"
        )
        boxes = [o["bbox_px"] for o in label["objects"]]
        band_rows = []
        for i, pol in enumerate(("HH", "HV")):
            band = product["bands"][i]
            if band["polarization"] != pol or native["bands"][i]["polarization"] != pol:
                raise ValueError("Native band order differs")
            annotation = ET.parse(artifact(band["annotation"]))
            require_uncorrected(annotation)
            if annotation.findtext(".//adsHeader/polarisation") != pol:
                raise ValueError("Annotation polarization mismatch")
            if (
                band["mode"] != "IW"
                or band["range_spacing_m"] != 10
                or band["azimuth_spacing_m"] != 10
                or annotation.findtext(".//adsHeader/mode") != "IW"
                or annotation.findtext(".//adsHeader/productType") != "GRD"
                or float(annotation.findtext(".//rangePixelSpacing", "nan")) != 10
                or float(annotation.findtext(".//azimuthPixelSpacing", "nan")) != 10
            ):
                raise ValueError("Native IW GRD 10m metadata required")
            dn = np.load(artifact(native["bands"][i]["dn"]), allow_pickle=False)
            if not np.array_equal(raw[i], dn):
                raise ValueError("Original DN and SNAP pixels differ")
            expected = reference_db(artifact(band["calibration"]), dn, native["window"])
            noise_ref = next(
                n["source"] for n in source["noise"] if n["polarization"] == pol
            )
            for related in (band["calibration"], noise_ref):
                header = ET.parse(artifact(related))
                for field in (
                    "missionId",
                    "polarisation",
                    "mode",
                    "startTime",
                    "stopTime",
                    "absoluteOrbitNumber",
                ):
                    value = annotation.findtext(f".//adsHeader/{field}")
                    if (
                        value is None
                        or header.findtext(f".//adsHeader/{field}") != value
                    ):
                        raise ValueError(
                            "Annotation/calibration/noise identity differs"
                        )
            estimated = noise_power(artifact(noise_ref), native["window"], pol)
            residual = dn.astype("float64") ** 2 - estimated
            surviving = np.where(residual < 0, policy["snap_floor_power"], residual)
            if (surviving <= 0).any():
                raise ValueError("Zero residual cannot be represented as finite dB")
            predicted = expected + 10 * np.log10(surviving / dn.astype("float64") ** 2)
            snap_estimated = noise_power(
                artifact(noise_ref), native["window"], pol, snap_precision=True
            )
            snap_residual = dn.astype("float64") ** 2 - snap_estimated
            snap_surviving = np.where(
                snap_residual < 0, policy["snap_floor_power"], snap_residual
            )
            if (snap_surviving <= 0).any():
                raise ValueError("Zero SNAP residual cannot be represented as dB")
            snap_prediction = expected + 10 * np.log10(
                snap_surviving / dn.astype("float64") ** 2
            )
            nominal_error = float(np.abs(expected - nominal[i]).max())
            corrected_error = float(np.abs(snap_prediction - corrected[i]).max())
            observed_power = dn.astype("float64") ** 2
            snap_power = observed_power * 10 ** ((corrected[i] - expected) / 10)
            linear_error = np.abs(snap_power - snap_surviving)
            peak = np.unravel_index(
                np.abs(snap_prediction - corrected[i]).argmax(), dn.shape
            )
            reconstruction = {
                "nominal_max_error_db": round(nominal_error, 9),
                "corrected_max_error_db": round(corrected_error, 9),
                "float64_lut_corrected_max_error_db": round(
                    float(np.abs(predicted - corrected[i]).max()), 9
                ),
                "float64_lut_corrected_error_db_q05_50_95": quantiles(
                    np.abs(predicted - corrected[i])
                ),
                "noise_lut_precision_for_snap_comparison": (
                    "float32 LUT nodes, float64 interpolation"
                ),
                "metadata_precision_floor_mask_disagreements": int(
                    np.sum((residual < 0) != (snap_residual < 0))
                ),
                "maximum_power_error_fraction_of_observed_power": float(
                    np.max(linear_error / observed_power)
                ),
                "largest_db_error_pixel": {
                    "local_row_col": [int(v) for v in peak],
                    "original_row_col": [
                        int(peak[0] + native["window"][1]),
                        int(peak[1] + native["window"][0]),
                    ],
                    "observed_power": float(observed_power[peak]),
                    "estimated_noise_power_snap_precision": float(snap_estimated[peak]),
                    "replayed_residual_power": float(snap_residual[peak]),
                    "snap_reconstructed_power": float(snap_power[peak]),
                },
                "passes": max(nominal_error, corrected_error)
                <= policy["comparison_tolerance_db"],
            }
            objects = []
            for j, obj in enumerate(label["objects"]):
                target, background = masks(
                    dn.shape,
                    boxes,
                    j,
                    policy["background_guard_px"],
                    policy["background_outer_px"],
                )
                objects.append(
                    {
                        "object_id": obj["id"],
                        "bbox_px": obj["bbox_px"],
                        "target": region_stats(
                            dn, estimated, nominal[i], corrected[i], target
                        ),
                        "surrounding_context": region_stats(
                            dn, estimated, nominal[i], corrected[i], background
                        ),
                        "nominal_target_background_median_db": round(
                            float(
                                np.median(nominal[i][target])
                                - np.median(nominal[i][background])
                            ),
                            6,
                        ),
                        "corrected_target_background_median_db": round(
                            float(
                                np.median(corrected[i][target])
                                - np.median(corrected[i][background])
                            ),
                            6,
                        ),
                    }
                )
            band_rows.append(
                {
                    "polarization": pol,
                    "source_noise_corrected": False,
                    "schema": "range_times_azimuth",
                    "reconstruction": reconstruction,
                    "objects": objects,
                }
            )
        b0 = product["bands"][0]
        input_metadata = {
            "mission": b0["mission"],
            "mode": b0["mode"],
            "product_type": "GRD",
            "band_order": ["HH", "HV"],
            "pixel_spacing_m": [b0["range_spacing_m"], b0["azimuth_spacing_m"]],
            "thermal_noise_corrected": False,
            "unit": "sigma0_db",
            "incidence_center_deg": native["native_context"]["center_incidence_deg"],
        }
        tensor_checks = []
        for obj in label["objects"]:
            result = prepare(nominal, obj["bbox_px"], input_metadata, label["split"])
            tensor_checks.append(
                {
                    "object_id": obj["id"],
                    "status": result["status"],
                    "reason": result["reason"],
                    "shape": (
                        list(result["tensor"].shape)
                        if result["tensor"] is not None
                        else None
                    ),
                    "tensor_sha256_float32_c_order": (
                        hashlib.sha256(result["tensor"].tobytes(order="C")).hexdigest()
                        if result["tensor"] is not None
                        else None
                    ),
                }
            )
        results.append(
            {
                "id": identity,
                "product_id": label["product_id"],
                "split": label["split"],
                "window": native["window"],
                "ipf_versions": ipf,
                "input_metadata": input_metadata,
                "bands": band_rows,
                "native_input_checks": tensor_checks,
            }
        )
    return {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "specification": ref(SPEC),
        "collection": policy["collection"],
        "reference": policy["reference"],
        "local_inputs": [
            ref(checked(r))
            for r in (
                public["private_inputs"]["snap"],
                snap["prepared"],
                snap["metadata"],
            )
        ],
        "artifacts": sorted(used.values(), key=lambda r: r["path"]),
        "areas": results,
        "reconstruction_all_passed": all(
            b["reconstruction"]["passes"] for a in results for b in a["bands"]
        ),
        "training_approved": False,
        "independent_review": False,
        "labels_modified": False,
        "classifier_scored": False,
        "compatibility_established": False,
        "promotion_eligible": False,
    }


def verify(local: bool = False) -> dict[str, Any]:
    report = json.loads(checked(json.loads(LOCK.read_text())["evidence"]).read_text())
    for key in ("policy", "specification", "collection", "reference"):
        checked(report[key])
    if local and report != replay():
        raise ValueError("Target noise replay differs")
    return {
        "verified": True,
        "local": local,
        "evidence": ref(OUTPUT),
        "training_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.publish and not args.verify and not args.local:
        if OUTPUT.exists() or LOCK.exists():
            raise ValueError("Existing release immutable; use a new version")
        save(OUTPUT, replay())
        save(LOCK, {"evidence": ref(OUTPUT)})
    elif args.verify and not args.publish:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose --publish or --verify [--local]")


if __name__ == "__main__":
    main()
