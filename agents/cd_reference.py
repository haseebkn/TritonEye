"""Bounded C/D reference comparison on existing training development windows.

The sparse wrappers are for pointwise operators only. This does not validate
absolute geolocation, extend the frozen A/B contract or admit labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil
import rasterio
from rasterio.transform import GCPTransformer
from scipy.interpolate import RegularGridInterpolator

from agents.artifacts import REPO_ROOT
from agents.association import GEOD
from agents.calibration import Calibrator
from agents.hh_hv_reservation import require_scope, save
from agents.nl_benchmark import require_development_scene
from agents.reserved_sar import DOWNLOAD, Cdse, annotation_metadata
from agents.snap_reference import (
    bands,
    compare,
    docker_arguments,
    execute,
    graph,
    wrapper,
)
from agents.source_followup import fresh
from agents.source_preprocessing import reference_db
from agents.source_reservation import checked, ref
from agents.source_windows import local_context
from agents.statoil_baseline import available_commit
from agents.target_noise_audit import noise_power

POLICY = REPO_ROOT / "configs/reservations/cd_reference_v010.json"
GUARDS = {
    "training_approved": False,
    "independent_review": False,
    "promotion_eligible": False,
    "input_contract_extended": False,
    "classifier_scored": False,
    "detector_scored": False,
}


def inputs() -> list[dict[str, Any]]:
    policy = json.loads(POLICY.read_text())
    manifest = json.loads(checked(policy["prepared"]).read_text())
    reservation = json.loads(
        (REPO_ROOT / "reservations/hh_hv/v0.1.0/manifest.json").read_text()
    )
    scenes = [
        s
        for s in manifest["scenes"]
        if s["scene"]["product_id"] in policy["product_ids"]
    ]
    if len(scenes) != 2 or {s["scene"]["name"][:3] for s in scenes} != {"S1C", "S1D"}:
        raise ValueError("Exact C and D development inputs required")
    for record in scenes:
        scene = record["scene"]
        original = next(
            s for s in reservation["scenes"] if s["product_id"] == scene["product_id"]
        )
        if scene != original or scene["split"] != "train":
            raise ValueError("Original training assignment required")
        if {r["regime"] for r in record["rois"]} != {"coastal", "offshore"} or len(
            record["rois"]
        ) != 2:
            raise ValueError("Both frozen regimes required")
        if [b["polarization"] for b in record["native_bands"]] != ["HH", "HV"]:
            raise ValueError("Original HH/HV order required")
        for roi in record["rois"]:
            require_scope(
                reservation, scene, roi | {"group": scene["geographic_group"]}
            )
            require_development_scene(
                scene["product_id"],
                {"name": scene["name"], "split": "train", "study_roi": roi["geometry"]},
            )
            if roi["window"][2:] != [512, 512]:
                raise ValueError("Original 512 pixel context required")
    return scenes


def header(
    annotation: bytes, calibration: bytes, noise: bytes, mission: str, pol: str
) -> dict[str, Any]:
    roots = [ET.fromstring(x) for x in (annotation, calibration, noise)]
    identities = []
    for root in roots:
        h = root.find(".//adsHeader")
        if h is None:
            raise ValueError("Missing native identity")
        identities.append(
            tuple(
                h.findtext(k)
                for k in (
                    "missionId",
                    "productType",
                    "polarisation",
                    "mode",
                    "startTime",
                    "stopTime",
                    "absoluteOrbitNumber",
                    "missionDataTakeId",
                )
            )
        )
    if len(set(identities)) != 1 or identities[0][:4] != (mission, "GRD", pol, "IW"):
        raise ValueError("Annotation/calibration/noise identity mismatch")
    root = roots[0]
    result = {
        "mission": mission,
        "polarization": pol,
        "range_spacing_m": float(root.findtext(".//rangePixelSpacing", "nan")),
        "azimuth_spacing_m": float(root.findtext(".//azimuthPixelSpacing", "nan")),
        "thermal_noise_corrected": root.findtext(".//thermalNoiseCorrectionPerformed"),
        "start": identities[0][4],
        "stop": identities[0][5],
    }
    if (
        result["range_spacing_m"] != 10
        or result["azimuth_spacing_m"] != 10
        or result["thermal_noise_corrected"] != "false"
    ):
        raise ValueError("Pilot requires original nominal ten-metre HH/HV inputs")
    return result


def prepare(output: Path) -> None:
    records = inputs()
    fresh(output)
    client = Cdse()
    products = []
    for record in records:
        scene = record["scene"]
        directory = output / scene["product_id"]
        directory.mkdir()
        base = f"{DOWNLOAD}({scene['product_id']})/Nodes({scene['name']})"
        manifest = directory / "manifest.safe"
        manifest.write_bytes(client.fetch(base + "/Nodes(manifest.safe)/$value"))
        native, noises, headers = [], [], []
        for band in record["native_bands"]:
            xml = checked(band["annotation"]).read_bytes()
            cal = checked(band["calibration"]).read_bytes()
            name = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
            noise = directory / ("noise-" + name[:-5] + ".xml")
            noise.write_bytes(
                client.fetch(
                    base
                    + "/Nodes(annotation)/Nodes(calibration)"
                    + f"/Nodes({noise.name})/$value"
                )
            )
            headers.append(
                header(
                    xml,
                    cal,
                    noise.read_bytes(),
                    scene["name"][:3],
                    band["polarization"],
                )
            )
            root = ET.fromstring(xml)
            native.append(
                band
                | {
                    "width": int(root.findtext(".//numberOfSamples", "0")),
                    "height": int(root.findtext(".//numberOfLines", "0")),
                }
            )
            noises.append({"polarization": band["polarization"], "source": ref(noise)})
        root = ET.parse(manifest).getroot()
        software = [
            dict(e.attrib) for e in root.iter() if e.tag.split("}")[-1] == "software"
        ]
        auxiliary = [
            dict(e.attrib) for e in root.iter() if e.tag.split("}")[-1] == "resource"
        ]
        areas = []
        for roi in record["rois"]:
            x, y, _, _ = roi["window"]
            area = {
                "id": roi["id"],
                "window": [x + 64, y + 64, 384, 384],
                "split": "train",
                "regime": roi["regime"],
                "bands": [],
                "original": roi,
            }
            with rasterio.open(checked(roi["chip"])) as ds:
                values = ds.read()[:, 64:448, 64:448]
            for i, band in enumerate(native):
                dn = np.load(checked(roi["bands"][i]["dn"]), allow_pickle=False)[
                    64:448, 64:448
                ].copy()
                db = (
                    Calibrator(str(checked(band["calibration"])))
                    .to_sigma0_db(dn, y + 64, x + 64)
                    .astype("float32")
                )
                if not np.array_equal(db, values[i]) or not np.isfinite(db).all():
                    raise ValueError("Original calibration does not replay")
                path = directory / f"{roi['id']}_{i}_dn.npy"
                np.save(path, dn, allow_pickle=False)
                area["bands"].append({"dn": ref(path)})
            chip = directory / f"{roi['id']}_sigma0.npy"
            np.save(chip, values, allow_pickle=False)
            area["chip"] = ref(chip)
            area["native_context"] = local_context(
                checked(native[0]["annotation"]).read_bytes(), x + 64, y + 64
            )
            areas.append(area)
        products.append(
            {
                "product_id": scene["product_id"],
                "name": scene["name"],
                "bands": native,
                "areas": areas,
                "metadata": {"manifest": ref(manifest), "noise": noises},
                "headers": headers,
                "software": software,
                "auxiliary": auxiliary,
            }
        )
        print(
            json.dumps({"prepared": scene["name"][:3], "areas": len(areas)}), flush=True
        )
    save(
        output / "prepared.json",
        {
            "policy": ref(POLICY),
            "products": products,
            "network_bytes": client.bytes,
            **GUARDS,
        },
    )


def analysis(
    product: dict[str, Any], area: dict[str, Any], directory: Path
) -> dict[str, Any]:
    expected = np.load(checked(area["chip"]), allow_pickle=False)
    dn = np.stack(
        [np.load(checked(b["dn"]), allow_pickle=False) for b in area["bands"]]
    )
    raw, nominal = bands(directory / "raw.dim", True), bands(
        directory / "calibration.dim"
    )
    result = compare(raw, nominal, dn, expected, 0.001)
    result["noise"] = []
    corrected = bands(directory / "noise.dim")
    for i, pol in enumerate(("HH", "HV")):
        band = product["bands"][i]
        noise = next(
            n for n in product["metadata"]["noise"] if n["polarization"] == pol
        )
        power = noise_power(
            checked(noise["source"]), area["window"], pol, snap_precision=True
        )
        surviving = np.maximum(dn[i].astype("float64") ** 2 - power, 1e-5)
        reference = reference_db(
            checked(band["calibration"]), np.sqrt(surviving), area["window"]
        )
        valid = np.isfinite(corrected[i]) & np.isfinite(reference)
        delta = np.abs(corrected[i][valid] - reference[valid])
        result["noise"].append(
            {
                "polarization": pol,
                "finite_pixels": int(valid.sum()),
                "total_pixels": int(valid.size),
                "max_absolute_db": float(delta.max()) if delta.size else None,
                "within_tolerance_pixels": int((delta <= 0.001).sum()),
                "passed": bool(valid.all() and delta.size and delta.max() <= 0.001),
                "below_estimated_noise_pixels": int(
                    (dn[i].astype("float64") ** 2 <= power).sum()
                ),
                "median_change_db": (
                    float(np.median(corrected[i][valid] - nominal[i][valid]))
                    if valid.any()
                    else None
                ),
            }
        )
    return result


def geometry_diagnostic(
    product: dict[str, Any], area: dict[str, Any], directory: Path
) -> dict[str, Any]:
    """Compare metadata interpolation at nine positions, not absolute accuracy."""
    dim = ET.parse(directory / "raw.dim")
    rr, cc = np.meshgrid([0.5, 192.5, 383.5], [0.5, 192.5, 383.5], indexing="ij")
    grids = {}
    for name in ("latitude", "longitude", "incident_angle"):
        info = next(
            g
            for g in dim.findall(".//Tie_Point_Grid_Info")
            if g.findtext("TIE_POINT_GRID_NAME") == name
        )
        with rasterio.open(directory / f"raw.data/tie_point_grids/{name}.img") as ds:
            values = ds.read(1)
        yy = float(info.findtext("OFFSET_Y", "nan")) + np.arange(
            values.shape[0]
        ) * float(info.findtext("STEP_Y", "nan"))
        xx = float(info.findtext("OFFSET_X", "nan")) + np.arange(
            values.shape[1]
        ) * float(info.findtext("STEP_X", "nan"))
        grids[name] = RegularGridInterpolator((yy, xx), values)(
            np.stack([rr, cc], axis=-1)
        )
    pol_grids = []
    for band in product["bands"]:
        _, gcps = annotation_metadata(
            checked(band["annotation"]).read_bytes(),
            product,
            band["polarization"].lower(),
        )
        with GCPTransformer(gcps, tps=True) as transform:
            xs, ys = transform.xy(
                (rr + area["window"][1]).ravel().tolist(),
                (cc + area["window"][0]).ravel().tolist(),
                offset="ul",
            )
        pol_grids.append((np.asarray(xs), np.asarray(ys)))
    delta = GEOD.inv(
        pol_grids[0][0],
        pol_grids[0][1],
        grids["longitude"].ravel(),
        grids["latitude"].ravel(),
    )[2]
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite geolocation comparison")
    return {
        "sample_positions": 9,
        "max_separation_m": float(np.max(delta)),
        "hh_hv_metadata_positions_equal": all(
            np.array_equal(x, y)
            for x, y in zip(pol_grids[0], pol_grids[1], strict=True)
        ),
        "snap_center_incidence_deg": float(grids["incident_angle"][1, 1]),
        "method": "SNAP bilinear tie-point interpolation versus existing TPS "
        "at pixel centres; different interpolators; no absolute location reference",
        "absolute_geolocation_validated": False,
    }


def run(prepared: Path, output: Path) -> None:
    policy = json.loads(POLICY.read_text())
    payload = json.loads(prepared.read_text())
    verify_prepared(payload)
    if payload["policy"] != ref(POLICY):
        raise ValueError("Frozen pilot required")
    inputs()
    if (
        available_commit() < 3 * 1024**3
        or psutil.virtual_memory().available < 3 * 1024**3
        or shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3
    ):
        raise ValueError("Insufficient bounded comparison headroom")
    receipt_path = REPO_ROOT / policy["runtime_receipt"]
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["image_id"] != policy["image_id"]
        or ref(checked(receipt["installer"]))["sha256"] != policy["installer_sha256"]
    ):
        raise ValueError("Pinned reference runtime required")
    docker = Path(shutil.which("docker") or "docker")
    found = subprocess.run(
        [str(docker), "image", "inspect", policy["image_id"], "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if found.stdout.strip() != policy["image_id"]:
        raise ValueError("Wrong runtime")
    fresh(output)
    report: dict[str, Any] = {
        "policy": ref(POLICY),
        "prepared": ref(prepared),
        "runtime": ref(receipt_path),
        "areas": [],
        **GUARDS,
    }
    for product in payload["products"]:
        for area in product["areas"]:
            directory = output / area["id"]
            directory.mkdir()
            source = wrapper(product, area, product["metadata"], directory)
            result: dict[str, Any] = {
                "id": area["id"],
                "product_id": product["product_id"],
                "mission": product["name"][:3],
                "regime": area["regime"],
                "variants": {},
            }
            for variant in policy["variants"]:
                target = directory / f"{variant}.dim"
                path = directory / f"{variant}.xml"
                path.write_text(
                    graph(
                        Path("/work") / source.relative_to(output),
                        area["window"],
                        Path("/work") / target.relative_to(output),
                        variant,
                    ),
                    encoding="utf-8",
                )
                name = "tritoneye-cd-" + uuid4().hex
                execution = execute(
                    docker,
                    [
                        *docker_arguments(output, name, policy["image_id"]),
                        (Path("/work") / path.relative_to(output)).as_posix(),
                        "-q",
                        "1",
                        "-c",
                        "128M",
                    ],
                    directory / f"{variant}.log",
                    policy["process_timeout_s"],
                )
                if execution["timed_out"]:
                    subprocess.run(
                        [str(docker), "container", "rm", "--force", name],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                files = [target, *sorted(target.with_suffix(".data").rglob("*"))]
                result["variants"][variant] = {
                    **execution,
                    "graph": ref(path),
                    "files": [ref(p) for p in files if p.is_file()],
                }
                print(
                    json.dumps(
                        {
                            "area": area["id"],
                            "variant": variant,
                            "exit_code": execution["exit_code"],
                        }
                    ),
                    flush=True,
                )
            try:
                result["comparison"] = analysis(product, area, directory)
            except (ValueError, OSError) as error:
                result["failure"] = str(error)
            report["areas"].append(result)
            save(output / f"progress_{len(report['areas'])}.json", report)
    save(output / "report.json", report)


def verify_prepared(payload: dict[str, Any]) -> None:
    if payload["policy"] != ref(POLICY) or any(
        payload.get(k) is not False for k in GUARDS
    ):
        raise ValueError("Prepared provenance/guards differ")
    original = inputs()
    if len(payload["products"]) != 2 or {
        p["product_id"] for p in payload["products"]
    } != {r["scene"]["product_id"] for r in original}:
        raise ValueError("Exact C/D product inventory required")
    for product in payload["products"]:
        record = next(
            r for r in original if r["scene"]["product_id"] == product["product_id"]
        )
        if product["name"] != record["scene"]["name"] or len(product["bands"]) != 2:
            raise ValueError("Prepared native identity differs")
        headers = []
        for band, source in zip(product["bands"], record["native_bands"], strict=True):
            if any(band[k] != v for k, v in source.items()):
                raise ValueError("Original band provenance changed")
            noise = next(
                n
                for n in product["metadata"]["noise"]
                if n["polarization"] == band["polarization"]
            )
            annotation = checked(band["annotation"]).read_bytes()
            headers.append(
                header(
                    annotation,
                    checked(band["calibration"]).read_bytes(),
                    checked(noise["source"]).read_bytes(),
                    product["name"][:3],
                    band["polarization"],
                )
            )
            root = ET.fromstring(annotation)
            if band["width"] != int(root.findtext(".//numberOfSamples", "0")) or band[
                "height"
            ] != int(root.findtext(".//numberOfLines", "0")):
                raise ValueError("Native raster dimensions changed")
        if headers != product["headers"]:
            raise ValueError("Prepared header summary differs")
        root = ET.parse(checked(product["metadata"]["manifest"])).getroot()
        for tag, key in (("software", "software"), ("resource", "auxiliary")):
            if [
                dict(e.attrib) for e in root.iter() if e.tag.split("}")[-1] == tag
            ] != product[key]:
                raise ValueError("Processing provenance summary differs")


def verify(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    if report["policy"] != ref(POLICY) or any(
        report.get(k) is not v for k, v in GUARDS.items()
    ):
        raise ValueError("Pilot provenance/guards differ")
    payload = json.loads(checked(report["prepared"]).read_text())
    verify_prepared(payload)
    checked(report["runtime"])
    if report.get("retry_of"):
        prior = verify(checked(report["retry_of"]))
        if prior.get("retry_of") or prior["prepared"] != report["prepared"]:
            raise ValueError("One retry of identical prepared inputs required")
    original = inputs()
    expected = [(p, a) for p in payload["products"] for a in p["areas"]]
    if (
        len(expected) != 4
        or len(report["areas"]) != 4
        or {a["id"] for _, a in expected} != {a["id"] for a in report["areas"]}
    ):
        raise ValueError("Complete four-area comparison required")
    for product, area in expected:
        record = next(
            r for r in original if r["scene"]["product_id"] == product["product_id"]
        )
        roi = next(r for r in record["rois"] if r["id"] == area["id"])
        if area["split"] != "train" or area["window"] != [
            roi["window"][0] + 64,
            roi["window"][1] + 64,
            384,
            384,
        ]:
            raise ValueError("Pilot window or role moved")
        with rasterio.open(checked(roi["chip"])) as ds:
            if not np.array_equal(
                ds.read()[:, 64:448, 64:448],
                np.load(checked(area["chip"]), allow_pickle=False),
            ):
                raise ValueError("Pilot pixels changed")
        for i, b in enumerate(area["bands"]):
            if not np.array_equal(
                np.load(checked(b["dn"]), allow_pickle=False),
                np.load(checked(roi["bands"][i]["dn"]), allow_pickle=False)[
                    64:448, 64:448
                ],
            ):
                raise ValueError("Original DN changed")
        item = next(r for r in report["areas"] if r["id"] == area["id"])
        if (
            item["product_id"] != product["product_id"]
            or item["mission"] != product["name"][:3]
            or item["regime"] != area["regime"]
        ):
            raise ValueError("Reference output identity differs")
        for v in item["variants"].values():
            checked(v["graph"])
            checked(v["log"])
            for f in v["files"]:
                checked(f)
        directory = checked(item["variants"]["raw"]["graph"]).parent
        if (
            "comparison" in item
            and analysis(product, area, directory) != item["comparison"]
        ):
            raise ValueError("Comparison replay differs")
    return report


def retry(path: Path, output: Path) -> None:
    """One exact-settings retry; retain the first attempt and copied good outputs."""
    original = verify(path)
    if original.get("retry_of"):
        raise ValueError("Only one automatic retry permitted")
    policy = json.loads(POLICY.read_text())
    payload = json.loads(checked(original["prepared"]).read_text())
    if (
        available_commit() < 3 * 1024**3
        or psutil.virtual_memory().available < 3 * 1024**3
        or shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3
    ):
        raise ValueError("Insufficient retry headroom")
    fresh(output)
    result = json.loads(json.dumps(original))
    result["retry_of"] = ref(path)
    docker = Path(shutil.which("docker") or "docker")
    for row in result["areas"]:
        failed = [
            k
            for k, v in row["variants"].items()
            if v["exit_code"] != 0 or v["timed_out"]
        ]
        if not failed:
            continue
        product = next(
            p for p in payload["products"] if p["product_id"] == row["product_id"]
        )
        area = next(a for a in product["areas"] if a["id"] == row["id"])
        directory = output / row["id"]
        directory.mkdir()
        source = wrapper(product, area, product["metadata"], directory)
        for variant, receipt in row["variants"].items():
            if variant in failed:
                continue
            old_directory = checked(receipt["graph"]).parent
            shutil.copyfile(
                old_directory / f"{variant}.dim", directory / f"{variant}.dim"
            )
            shutil.copytree(
                old_directory / f"{variant}.data", directory / f"{variant}.data"
            )
            graph_path = directory / f"{variant}.xml"
            shutil.copyfile(checked(receipt["graph"]), graph_path)
            receipt["adopted_from"] = {
                "report": ref(path),
                "area": row["id"],
                "variant": variant,
            }
            receipt["graph"] = ref(graph_path)
            receipt["files"] = [
                ref(directory / f"{variant}.dim"),
                *[
                    ref(p)
                    for p in sorted((directory / f"{variant}.data").rglob("*"))
                    if p.is_file()
                ],
            ]
        for variant in failed:
            graph_path = directory / f"{variant}.xml"
            graph_path.write_text(
                graph(
                    Path("/work") / source.relative_to(output),
                    area["window"],
                    Path("/work") / (directory / f"{variant}.dim").relative_to(output),
                    variant,
                ),
                encoding="utf-8",
            )
            name = "tritoneye-cd-retry-" + uuid4().hex
            receipt = execute(
                docker,
                [
                    *docker_arguments(output, name, policy["image_id"]),
                    (Path("/work") / graph_path.relative_to(output)).as_posix(),
                    "-q",
                    "1",
                    "-c",
                    "128M",
                ],
                directory / f"{variant}.log",
                policy["process_timeout_s"],
            )
            if receipt["timed_out"]:
                subprocess.run(
                    [str(docker), "container", "rm", "--force", name],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            files = [
                directory / f"{variant}.dim",
                *sorted((directory / f"{variant}.data").rglob("*")),
            ]
            row["variants"][variant] = {
                **receipt,
                "graph": ref(graph_path),
                "files": [ref(p) for p in files if p.is_file()],
            }
            print(
                json.dumps(
                    {
                        "retry": row["id"],
                        "variant": variant,
                        "exit_code": receipt["exit_code"],
                    }
                ),
                flush=True,
            )
        row.pop("comparison", None)
        row.pop("failure", None)
        try:
            row["comparison"] = analysis(product, area, directory)
        except (ValueError, OSError) as error:
            row["failure"] = str(error)
    save(output / "report.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--retry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare and args.output:
        prepare(args.output)
    elif args.run and args.output:
        run(args.run, args.output)
    elif args.verify:
        print(
            json.dumps({"verified": True, "areas": len(verify(args.verify)["areas"])})
        )
    elif args.retry and args.output:
        retry(args.retry, args.output)
    else:
        parser.error("Choose prepare, run or verify")
