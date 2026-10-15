"""Additional C/D processing checks on preselected training acquisitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil
import rasterio

from agents.artifacts import REPO_ROOT
from agents.cd_noise_diagnosis import timed_noise
from agents.cd_reference import geometry_diagnostic, header, retry_directory
from agents.hh_hv_reservation import choose, query, save
from agents.native_cd_input import SPEC, specification
from agents.native_cd_input import prepare as tensor
from agents.nl_benchmark import require_development_scene
from agents.provisional_training import training_role
from agents.reserved_sar import DOWNLOAD, Cdse, prepare_scene
from agents.snap_reference import (
    bands,
    compare,
    docker_arguments,
    execute,
    graph,
    wrapper,
)
from agents.source_followup import fresh
from agents.source_reservation import checked, ref
from agents.source_windows import local_context
from agents.statoil_baseline import available_commit

POLICY = REPO_ROOT / "configs/reservations/cd_acceptance_v020.json"
REPORT = REPO_ROOT / "docs/evidence/cd_acceptance_v020.json"
LOCK = REPO_ROOT / "configs/reservations/cd_acceptance_v020.lock.json"
GUARDS = {
    "training_approved": False,
    "independent_review": False,
    "promotion_eligible": False,
    "labels_admitted": 0,
    "classifier_scored": False,
}


def inventory() -> (
    tuple[list[dict[str, Any]], set[str], list[dict[str, Any]], dict[str, str]]
):
    from agents.provisional_training import inventory as original_inventory

    priors, exposed, old, hashes = original_inventory()
    for path in sorted(
        (REPO_ROOT / "reservations/native_training").glob("*/manifest.json")
    ):
        manifest = json.loads(path.read_text())
        private_path = checked(manifest["private_selection"])
        hashes[ref(path)["path"]] = ref(path)["sha256"]
        hashes[ref(private_path)["path"]] = ref(private_path)["sha256"]
        for area in json.loads(private_path.read_text())["areas"]:
            old.append(area)
            exposed.add(area["native_acquisition_group"])
            priors.append(
                {
                    "geometry": area["geometry"],
                    "group": area["geographic_group"],
                    "split": area["split"],
                    "acquisition_group": area["native_acquisition_group"],
                    "quarantine": False,
                }
            )
    return priors, exposed, old, hashes


def select_additional(
    products: list[dict[str, Any]],
    mission: str,
    group: dict[str, Any],
    old_policy: dict[str, Any],
    priors: list[dict[str, Any]],
    exposed: set[str],
) -> dict[str, Any]:
    rules = json.loads(POLICY.read_text())
    scene = choose(
        [
            p
            for p in products
            if p["Name"].startswith(mission + "_") and "_COG" not in p["Name"]
        ],
        group,
        {"id": "cd_acceptance_v020", "start": rules["start"], "end": rules["end"]},
        old_policy,
        exposed,
    )
    for roi in scene["rois"]:
        assigned = training_role(roi["geometry"], priors, scene["acquisition_group"])
        if assigned != group["id"]:
            raise ValueError(
                "Additional check must retain the original training geography"
            )
    return scene


def discover(output: Path) -> None:
    fresh(output)
    rules = json.loads(POLICY.read_text())
    old_policy = json.loads(
        (REPO_ROOT / "configs/reservations/hh_hv_development_v010.json").read_text()
    )
    priors, exposed, _, hashes = inventory()
    scenes, searches = [], []
    for mission, group_id in rules["mission_groups"].items():
        group = next(g for g in old_policy["groups"] if g["id"] == group_id)
        products, receipt = query(group, {"start": rules["start"], "end": rules["end"]})
        scenes.append(
            select_additional(products, mission, group, old_policy, priors, exposed)
        )
        searches.append({"mission": mission, **receipt})
        exposed.add(scenes[-1]["acquisition_group"])
    save(
        output / "selection.json",
        {
            "policy": ref(POLICY),
            "specification": ref(SPEC),
            "scenes": scenes,
            "searches": searches,
            "priors": priors,
            "prior_metadata_hashes": hashes,
            "pixels_accessed": False,
            **GUARDS,
        },
    )
    print(json.dumps({"selected": [s["name"] for s in scenes]}), flush=True)


def selection(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    rules = json.loads(POLICY.read_text())
    if value["policy"] != ref(POLICY) or value["specification"] != ref(SPEC):
        raise ValueError("Frozen selection/specification differs")
    for name, digest in value["prior_metadata_hashes"].items():
        checked({"path": name, "sha256": digest})
    old_policy = json.loads(
        (REPO_ROOT / "configs/reservations/hh_hv_development_v010.json").read_text()
    )
    # Reconstruct the preselection exposure set, not later runtime discoveries.
    _, exposed, _, _ = inventory()
    if len(value["scenes"]) != 2:
        raise ValueError("One additional acquisition per mission required")
    for scene, search in zip(value["scenes"], value["searches"], strict=True):
        page = json.loads(checked(search["cached_response"]).read_text())["page"]
        if page.get("@odata.nextLink"):
            raise ValueError("Incomplete catalogue")
        group = next(
            g
            for g in old_policy["groups"]
            if g["id"] == rules["mission_groups"][search["mission"]]
        )
        replay = select_additional(
            page["value"],
            search["mission"],
            group,
            old_policy,
            value["priors"],
            exposed,
        )
        if json.loads(json.dumps(replay)) != scene:
            raise ValueError("Preselected acquisition replay differs")
        exposed.add(scene["acquisition_group"])
    return value


def software(manifest: Path) -> dict[str, str]:
    entries = [
        {"name": e.get("name", ""), "version": e.get("version", "")}
        for e in ET.parse(manifest).iter()
        if e.tag.split("}")[-1] == "software" and e.get("name") == "Sentinel-1 IPF"
    ]
    if (
        not entries
        or len({tuple(sorted(e.items())) for e in entries}) != 1
        or entries[0].get("version") not in specification()["processing_versions"]
    ):
        raise ValueError("Unsupported or ambiguous processor version")
    return entries[0]


def prepare(selection_path: Path, output: Path) -> None:
    selected = selection(selection_path)
    headroom()
    fresh(output)
    client, products = Cdse(), []
    for scene in selected["scenes"]:
        native = prepare_scene(client, selected, scene, output)
        directory = output / scene["product_id"]
        base = f"{DOWNLOAD}({scene['product_id']})/Nodes({scene['name']})"
        manifest = directory / "manifest.safe"
        manifest.write_bytes(client.fetch(base + "/Nodes(manifest.safe)/$value"))
        processor = software(manifest)
        noises, native_bands = [], []
        for band in native["native_bands"]:
            name = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
            noise = directory / ("noise-" + name[:-5] + ".xml")
            noise.write_bytes(
                client.fetch(
                    base
                    + "/Nodes(annotation)/Nodes(calibration)"
                    + f"/Nodes({noise.name})/$value"
                )
            )
            identity = header(
                checked(band["annotation"]).read_bytes(),
                checked(band["calibration"]).read_bytes(),
                noise.read_bytes(),
                scene["name"][:3],
                band["polarization"],
            )
            root = ET.parse(checked(band["annotation"]))
            native_bands.append(
                band
                | {
                    "width": int(root.findtext(".//numberOfSamples", "0")),
                    "height": int(root.findtext(".//numberOfLines", "0")),
                    "identity": identity,
                }
            )
            noises.append({"polarization": band["polarization"], "source": ref(noise)})
        areas = []
        for n, roi in enumerate(native["rois"]):
            x, y, _, _ = roi["window"]
            with rasterio.open(checked(roi["chip"])) as ds:
                values = ds.read()[:, 64:448, 64:448]
            chip = directory / f"nominal_{n}.npy"
            np.save(chip, values, allow_pickle=False)
            band_refs = []
            for i, b in enumerate(roi["bands"]):
                path = directory / f"dn_{n}_{i}.npy"
                np.save(
                    path,
                    np.load(checked(b["dn"]), allow_pickle=False)[64:448, 64:448],
                    allow_pickle=False,
                )
                band_refs.append({"dn": ref(path)})
            areas.append(
                {
                    "id": f"{scene['name'][:3]}_{n}",
                    "split": "train",
                    "regime": roi["regime"],
                    "window": [x + 64, y + 64, 384, 384],
                    "original": roi,
                    "chip": ref(chip),
                    "bands": band_refs,
                }
            )
        products.append(
            {
                "product_id": scene["product_id"],
                "name": scene["name"],
                "scene": scene,
                "bands": native_bands,
                "areas": areas,
                "software": processor,
                "metadata": {"manifest": ref(manifest), "noise": noises},
            }
        )
        save(output / f"progress_{len(products)}.json", {"products": products})
        print(
            json.dumps({"prepared": scene["name"][:3], "network_bytes": client.bytes}),
            flush=True,
        )
    save(
        output / "prepared.json",
        {
            "selection": ref(selection_path),
            "products": products,
            "network_bytes": client.bytes,
            **GUARDS,
        },
    )


def metadata(product: dict[str, Any], area: dict[str, Any]) -> dict[str, Any]:
    band = product["bands"][0]
    context = local_context(
        checked(band["annotation"]).read_bytes(), *area["window"][:2]
    )
    identity = band["identity"]
    return {
        "name": product["name"],
        "mission": identity["mission"],
        "mode": "IW",
        "product_type": "GRD",
        "band_order": ["HH", "HV"],
        "pixel_spacing_m": [identity["range_spacing_m"], identity["azimuth_spacing_m"]],
        "thermal_noise_corrected": False,
        "unit": "sigma0_db",
        "processing_software": product["software"]["name"],
        "processing_version": product["software"]["version"],
        "native_start_utc": annotation_utc(identity["start"]),
        "native_stop_utc": annotation_utc(identity["stop"]),
        "native_center_utc": context["center_native_time_utc"],
        "incidence_center_deg": context["center_incidence_deg"],
    }


def annotation_utc(value: str) -> str:
    """Sentinel annotation startTime/stopTime fields are UTC without a suffix."""
    instant = datetime.fromisoformat(value.removesuffix("Z"))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).isoformat()


def analyze(
    product: dict[str, Any], area: dict[str, Any], directory: Path
) -> dict[str, Any]:
    from agents.calibration import Calibrator

    values = np.load(checked(area["chip"]), allow_pickle=False)
    dns = np.stack(
        [np.load(checked(b["dn"]), allow_pickle=False) for b in area["bands"]]
    )
    noise_summary = []
    original = area["original"]
    with rasterio.open(checked(original["chip"])) as ds:
        if not np.array_equal(values, ds.read()[:, 64:448, 64:448]):
            raise ValueError("Original nominal crop changed")
    for i, band in enumerate(product["bands"]):
        if not np.array_equal(
            dns[i],
            np.load(checked(original["bands"][i]["dn"]), allow_pickle=False)[
                64:448, 64:448
            ],
        ):
            raise ValueError("Original DN crop changed")
        noise = next(
            n
            for n in product["metadata"]["noise"]
            if n["polarization"] == band["polarization"]
        )
        actual = header(
            checked(band["annotation"]).read_bytes(),
            checked(band["calibration"]).read_bytes(),
            checked(noise["source"]).read_bytes(),
            product["name"][:3],
            band["polarization"],
        )
        if actual != band["identity"]:
            raise ValueError("Native header changed")
        db = (
            Calibrator(str(checked(band["calibration"])))
            .to_sigma0_db(dns[i], area["window"][1], area["window"][0])
            .astype("float32")
        )
        if not np.array_equal(db, values[i]):
            raise ValueError("Nominal calibration replay differs")
        power = timed_noise(
            checked(band["annotation"]),
            checked(noise["source"]),
            area["window"],
            band["polarization"],
        )
        noise_summary.append(
            {
                "polarization": band["polarization"],
                "fraction_at_or_below_estimated_noise": float(
                    (dns[i].astype("float64") ** 2 <= power).mean()
                ),
                "diagnostic_only": True,
            }
        )
    if software(checked(product["metadata"]["manifest"])) != product["software"]:
        raise ValueError("Processor provenance differs")
    meta = metadata(product, area)
    result = tensor(values, [191, 191, 193, 193], meta, "train")
    if result["status"] != "research_tensor_only":
        raise ValueError("Native research tensor refused")
    return {
        "nominal": compare(
            bands(directory / "raw.dim", True),
            bands(directory / "calibration.dim"),
            dns,
            values,
            0.001,
        ),
        "geometry": geometry_diagnostic(product, area, directory),
        "metadata": meta,
        "tensor_sha256": hashlib.sha256(result["tensor"].tobytes()).hexdigest(),
        "noise_diagnostics": noise_summary,
        "tensor_shape": list(result["tensor"].shape),
        "tensor_box_is_class_label": False,
    }


def verify_prepared(payload: dict[str, Any]) -> dict[str, Any]:
    from shapely.geometry import shape

    selected = selection(checked(payload["selection"]))
    if (
        any(payload.get(k) != v for k, v in GUARDS.items())
        or [p["scene"] for p in payload["products"]] != selected["scenes"]
    ):
        raise ValueError("Prepared selection or guards differ")
    for product, scene in zip(payload["products"], selected["scenes"], strict=True):
        if (
            product["product_id"] != scene["product_id"]
            or product["name"] != scene["name"]
            or [b["polarization"] for b in product["bands"]] != ["HH", "HV"]
            or len(product["areas"]) != 2
            or {a["regime"] for a in product["areas"]} != {"coastal", "offshore"}
        ):
            raise ValueError("Prepared native identity/inventory differs")
        for area in product["areas"]:
            original = area["original"]
            roi = next(r for r in scene["rois"] if r["regime"] == area["regime"])
            x, y, width, height = original["window"]
            if (
                original["regime"] != area["regime"]
                or width != 512
                or height != 512
                or area["split"] != "train"
                or area["window"] != [x + 64, y + 64, 384, 384]
                or not shape(roi["geometry"]).covers(shape(original["geometry"]))
            ):
                raise ValueError("Prepared crop escapes frozen training envelope")
            training_role(
                original["geometry"], selected["priors"], scene["acquisition_group"]
            )
            require_development_scene(
                scene["product_id"],
                {
                    "name": scene["name"],
                    "split": "train",
                    "study_roi": original["geometry"],
                },
            )
    return selected


def decision(rows: list[dict[str, Any]]) -> bool:
    rules = json.loads(POLICY.read_text())
    expected = {(m, r) for m in rules["mission_groups"] for r in rules["regimes"]}
    return (
        len(rows) == len(expected)
        and {(r["mission"], r["regime"]) for r in rows} == expected
        and len({r["product_id"] for r in rows}) == 2
        and all(
            r.get("analysis", {}).get("nominal", {}).get("passed") is True
            and r["analysis"]["geometry"]["hh_hv_metadata_positions_equal"] is True
            and r["analysis"]["geometry"]["max_separation_m"]
            <= rules["maximum_metadata_interpolator_separation_m"]
            and set(r["variants"]) == set(rules["variants"])
            and all(
                v["exit_code"] == 0
                and v["timed_out"] is False
                and v["elapsed_limit_exceeded"] is False
                for v in r["variants"].values()
            )
            for r in rows
        )
    )


def headroom() -> None:
    if (
        available_commit() < 3 * 1024**3
        or psutil.virtual_memory().available < 3 * 1024**3
        or shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3
    ):
        raise ValueError("Insufficient bounded comparison headroom")


def run(prepared: Path, output: Path) -> None:
    payload = json.loads(prepared.read_text())
    verify_prepared(payload)
    rules = json.loads(POLICY.read_text())
    retry_directory(output, payload)
    headroom()
    docker = Path(shutil.which("docker") or "docker")
    found = subprocess.run(
        [str(docker), "image", "inspect", rules["image_id"], "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if found.stdout.strip() != rules["image_id"]:
        raise ValueError("Pinned reference runtime required")
    fresh(output)
    report: dict[str, Any] = {
        "policy": ref(POLICY),
        "specification": ref(SPEC),
        "prepared": ref(prepared),
        "areas": [],
        **GUARDS,
    }
    for product in payload["products"]:
        for area in product["areas"]:
            headroom()
            directory = output / area["id"]
            directory.mkdir()
            source = wrapper(product, area, product["metadata"], directory)
            row: dict[str, Any] = {
                "id": area["id"],
                "product_id": product["product_id"],
                "mission": product["name"][:3],
                "regime": area["regime"],
                "variants": {},
            }
            for variant in rules["variants"]:
                path = directory / f"{variant}.xml"
                path.write_text(
                    graph(
                        Path("/work") / source.relative_to(output),
                        area["window"],
                        Path("/work")
                        / (directory / f"{variant}.dim").relative_to(output),
                        variant,
                    ),
                    encoding="utf-8",
                )
                name = "tritoneye-acceptance-" + uuid4().hex
                receipt = execute(
                    docker,
                    [
                        *docker_arguments(output, name, rules["image_id"]),
                        (Path("/work") / path.relative_to(output)).as_posix(),
                        "-q",
                        "1",
                        "-c",
                        "128M",
                    ],
                    directory / f"{variant}.log",
                    rules["process_timeout_s"],
                )
                if receipt["timed_out"]:
                    subprocess.run(
                        [str(docker), "container", "rm", "--force", name],
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                row["variants"][variant] = {
                    **receipt,
                    "graph": ref(path),
                    "files": [
                        ref(p)
                        for p in [
                            directory / f"{variant}.dim",
                            *sorted((directory / f"{variant}.data").rglob("*")),
                        ]
                        if p.is_file()
                    ],
                }
                print(
                    json.dumps(
                        {
                            "area": area["id"],
                            "variant": variant,
                            "exit": receipt["exit_code"],
                        }
                    ),
                    flush=True,
                )
            try:
                row["analysis"] = analyze(product, area, directory)
            except (ValueError, OSError) as error:
                row["failure"] = str(error)
            report["areas"].append(row)
            save(output / f"progress_{len(report['areas'])}.json", report)
    report["additional_processing_checks_passed"] = decision(report["areas"])
    save(output / "report.json", report)


def verify_run(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    if (
        report["policy"] != ref(POLICY)
        or report["specification"] != ref(SPEC)
        or any(report.get(k) != v for k, v in GUARDS.items())
    ):
        raise ValueError("Acceptance provenance/guard differs")
    payload = json.loads(checked(report["prepared"]).read_text())
    selected = verify_prepared(payload)
    if report.get("reanalysis_of"):
        original = json.loads(checked(report["reanalysis_of"]).read_text())
        if (
            original.get("reanalysis_of")
            or original["prepared"] != report["prepared"]
            or original["policy"] != report["policy"]
            or original["specification"] != report["specification"]
            or any(original.get(k) != v for k, v in GUARDS.items())
            or [
                {k: v for k, v in r.items() if k not in {"analysis", "failure"}}
                for r in original["areas"]
            ]
            != [
                {k: v for k, v in r.items() if k not in {"analysis", "failure"}}
                for r in report["areas"]
            ]
        ):
            raise ValueError("Analysis repair changed original processing evidence")
    if [p["scene"] for p in payload["products"]] != selected["scenes"]:
        raise ValueError("Prepared products differ from selection")
    expected = [(p, a) for p in payload["products"] for a in p["areas"]]
    if (
        len(report["areas"]) != 4
        or len(expected) != 4
        or len({r["id"] for r in report["areas"]}) != 4
    ):
        raise ValueError("Complete additional inventory required")
    for product, area in expected:
        roi = next(r for r in product["scene"]["rois"] if r["regime"] == area["regime"])
        training_role(
            area["original"]["geometry"],
            selected["priors"],
            product["scene"]["acquisition_group"],
        )
        from shapely.geometry import shape

        if not shape(roi["geometry"]).covers(shape(area["original"]["geometry"])):
            raise ValueError("Native context outside frozen envelope")
        require_development_scene(
            product["product_id"],
            {
                "name": product["name"],
                "split": "train",
                "study_roi": area["original"]["geometry"],
            },
        )
        row = next(r for r in report["areas"] if r["id"] == area["id"])
        if (
            row["product_id"] != product["product_id"]
            or row["regime"] != area["regime"]
            or row["mission"] != product["name"][:3]
        ):
            raise ValueError("Acceptance scene identity differs")
        for variant in row["variants"].values():
            checked(variant["graph"])
            checked(variant["log"])
            for source in variant["files"]:
                checked(source)
        if (
            "analysis" in row
            and analyze(product, area, checked(row["variants"]["raw"]["graph"]).parent)
            != row["analysis"]
        ):
            raise ValueError("Numerical acceptance replay differs")
    if report["additional_processing_checks_passed"] is not decision(report["areas"]):
        raise ValueError("Acceptance decision differs")
    return report


def publish(path: Path) -> None:
    if REPORT.exists() or LOCK.exists():
        raise ValueError("Fresh evidence version required")
    report = verify_run(path)
    save(
        REPORT,
        {
            **report,
            "private_run": ref(path),
            "input_contract_implemented": True,
            "absolute_geolocation_validated": False,
            "model_transfer_validated": False,
            "native_cd_consumer": ref(REPO_ROOT / "agents/native_cd_input.py"),
        },
    )
    save(
        LOCK, {"report": ref(REPORT), "policy": ref(POLICY), "specification": ref(SPEC)}
    )


def reanalyze(path: Path, output: Path) -> None:
    """Keep the first attempt; recompute checks on identical saved SNAP outputs."""
    report = json.loads(path.read_text())
    if (
        report.get("reanalysis_of")
        or report["policy"] != ref(POLICY)
        or report["specification"] != ref(SPEC)
    ):
        raise ValueError("One exact-input analysis repair permitted")
    payload = json.loads(checked(report["prepared"]).read_text())
    verify_prepared(payload)
    for product in payload["products"]:
        for area in product["areas"]:
            row = next(r for r in report["areas"] if r["id"] == area["id"])
            for variant in row["variants"].values():
                checked(variant["graph"])
                checked(variant["log"])
                for source in variant["files"]:
                    checked(source)
            directory = checked(row["variants"]["raw"]["graph"]).parent
            row.pop("failure", None)
            row["analysis"] = analyze(product, area, directory)
    report["reanalysis_of"] = ref(path)
    report["analysis_repair"] = (
        "Explicit UTC offsets for native XML UTC fields; no pixel, processor, "
        "window or tolerance change"
    )
    report["additional_processing_checks_passed"] = decision(report["areas"])
    fresh(output)
    save(output / "report.json", report)


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    checked(lock["policy"])
    checked(lock["specification"])
    report = json.loads(checked(lock["report"]).read_text())
    checked(report["native_cd_consumer"])
    if (
        any(report.get(k) != v for k, v in GUARDS.items())
        or report["additional_processing_checks_passed"]
        is not decision(report["areas"])
        or report["absolute_geolocation_validated"] is not False
        or report["model_transfer_validated"] is not False
    ):
        raise ValueError("Public acceptance decision differs")
    if local:
        original = verify_run(checked(report["private_run"]))
        if any(report[k] != v for k, v in original.items()):
            raise ValueError("Published acceptance replay differs")
    return {
        "verified": True,
        "local": local,
        "additional_processing_checks_passed": report[
            "additional_processing_checks_passed"
        ],
        **GUARDS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--reanalyze", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.discover and args.output:
        discover(args.output)
    elif args.prepare and args.output:
        prepare(args.prepare, args.output)
    elif args.run and args.output:
        run(args.run, args.output)
    elif args.publish:
        publish(args.publish)
    elif args.reanalyze and args.output:
        reanalyze(args.reanalyze, args.output)
    elif args.verify:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose discovery, prepare, run, publish or verify")
