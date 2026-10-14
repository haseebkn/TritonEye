"""Persist a classifier-blind NL candidate set for both later comparison arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import rasterio
import scipy

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.geo import Georeferencer
from agents.hh_hv_candidates import CONFIG_SHA256, frozen_settings, generate
from agents.nl_benchmark import bounded_path, load_dataset, require_development_scene
from agents.statoil_baseline import available_commit, resource_check
from agents.statoil_shadow import DEFAULT_DATASET, native_audit


def identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def run(output: Path, dataset: Path = DEFAULT_DATASET) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() or not output.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Fresh ignored candidate output required")
    resource_check()
    settings, config = frozen_settings()
    manifest, _, _ = load_dataset(dataset)
    # Write the definition BEFORE accessing SAR arrays or generating any candidates.
    definition = {
        "configuration": config,
        "config_sha256": CONFIG_SHA256,
        "code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "hh_hv_candidates.py",
                "hh_hv_candidate_run.py",
                "calibration.py",
                "statoil_shadow.py",
                "geo.py",
            )
        },
        "dataset_release_sha256": sha256_file(dataset / "release.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "selection": "Previously exposed development ROIs; no fresh holdout claim",
        "comparison": "Frozen before filter comparison; no classifier invoked",
        "attribution": manifest["licence"]["attribution"],
    }
    output.mkdir(parents=True)
    write_json(output / "definition.json", definition)
    started = time.perf_counter()
    cache: dict[str, str] = {}
    areas: list[dict[str, Any]] = []
    excluded_test: list[str] = []
    excluded_other: list[str] = []
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            excluded_test.extend(r["id"] for r in scene["rois"])
            continue
        if scene["polarizations"] != ["hh", "hv"]:
            excluded_other.extend(r["id"] for r in scene["rois"])
            continue
        if scene["split"] not in ("train", "validation") or scene["region"] not in (
            "Newfoundland",
            "Labrador",
        ):
            raise ValueError("NL development HH/HV acquisitions required")
        require_development_scene(scene["product_id"], scene)
        for roi in scene["rois"]:
            if (
                available_commit() < 1024**3
                or psutil.Process().memory_info().rss > 2 * 1024**3
            ):
                raise RuntimeError("Candidate run exceeded memory budget")
            entry: dict[str, Any] = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "acquisition_time": scene["acquisition_time"],
                "split": scene["split"],
                "region": scene["region"],
                "regime": roi["regime"],
                "input_chip": roi["chip"],
            }
            try:
                images, audit = native_audit(scene, roi, cache)
                candidates, coverage = generate(
                    images, polarizations=("HH", "HV"), settings=settings
                )
                features = []
                with rasterio.open(bounded_path(REPO_ROOT, roi["chip"]["path"])) as src:
                    with Georeferencer.from_dataset(src) as geo:
                        if candidates:
                            peaks = np.asarray([c["peak_pixel_xy"] for c in candidates])
                            # Peak integers are sample indices, so locate pixel centres.
                            lon, lat = geo.xy(peaks[:, 1] + 0.5, peaks[:, 0] + 0.5)
                        for i, candidate in enumerate(candidates):
                            features.append(
                                {
                                    "type": "Feature",
                                    "id": (
                                        f"{scene['product_id']}/{roi['id']}/"
                                        f"{candidate['component']:06d}"
                                    ),
                                    "geometry": {
                                        "type": "Point",
                                        "coordinates": [float(lon[i]), float(lat[i])],
                                    },
                                    "properties": {
                                        **candidate,
                                        "class": "unassociated_candidate",
                                        "physical_surface": "unknown",
                                        "product_id": scene["product_id"],
                                        "roi_id": roi["id"],
                                    },
                                }
                            )
                collection = {
                    "type": "FeatureCollection",
                    "features": features,
                    "generator_version": settings.version,
                    "config_sha256": CONFIG_SHA256,
                    "product_id": scene["product_id"],
                    "roi_id": roi["id"],
                    "input_sha256": roi["chip"]["sha256"],
                    "coverage": coverage,
                    "scope": "Experimental HH/HV candidates; no class/accuracy truth",
                }
                filename = f"area_{len(areas):03d}.geojson"
                write_json(output / filename, collection)
                entry.update(
                    status="processed",
                    coverage=coverage,
                    candidate_count=len(features),
                    candidates_path=filename,
                    candidates_sha256=sha256_file(output / filename),
                    candidate_identity=identity(collection),
                    calibration_replay_max_abs_db=audit[
                        "calibration_replay_max_abs_db"
                    ],
                )
            except (ValueError, OSError, ET.ParseError) as error:
                entry.update(status="failed", error=str(error))
            areas.append(entry)
    bundle = {
        "scope": "Frozen experimental NL HH/HV candidate bundle",
        "definition_sha256": sha256_file(output / "definition.json"),
        "config_sha256": CONFIG_SHA256,
        "generator_version": settings.version,
        "areas": areas,
        "excluded_locked_test_rois": excluded_test,
        "excluded_non_hh_hv_rois": excluded_other,
        "comparison_inputs": {
            "unfiltered": "this_exact_bundle",
            "classifier_filter": "this_exact_bundle",
        },
        "summary": {
            "selected_areas": len(areas),
            "processed_areas": sum(a["status"] == "processed" for a in areas),
            "failed_areas": sum(a["status"] == "failed" for a in areas),
            "candidate_count": sum(a.get("candidate_count", 0) for a in areas),
            "evaluated_pixels": sum(
                a.get("coverage", {}).get("evaluated_pixels", 0) for a in areas
            ),
            "roi_pixels": sum(
                a.get("coverage", {}).get("roi_pixels", 0) for a in areas
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "accuracy": None,
        "false_alarms_per_km2": None,
        "promotion_eligible": False,
        "limits": [
            "No labelled accuracy or measured false-alarm rate",
            "Ring edges and invalid pixels are unevaluated",
            "Connected targets can merge; boxes are not hull dimensions",
            "No shoreline/clutter/ice exclusion and no classifier inference",
            "A fixed 0.1.0 heuristic, not optimized against labels or filter results",
            "Existing development imagery was previously viewed",
        ],
    }
    write_json(output / "bundle.json", bundle)
    return bundle


def paired_inputs(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Both arms load one sealed candidate population; never regenerate per arm."""
    path = path.resolve()
    if (
        not path.is_relative_to((REPO_ROOT / "data").resolve())
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError("Bound, hash-matching candidate bundle required")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    settings, config = frozen_settings()
    if (
        bundle["scope"] != "Frozen experimental NL HH/HV candidate bundle"
        or bundle["config_sha256"] != CONFIG_SHA256
        or bundle["summary"]["failed_areas"]
        or not bundle["areas"]
    ):
        raise ValueError("Complete frozen candidate bundle required")
    definition = path.parent / "definition.json"
    if sha256_file(definition) != bundle["definition_sha256"]:
        raise ValueError("Frozen definition hash mismatch")
    frozen = json.loads(definition.read_text(encoding="utf-8"))
    if (
        frozen["configuration"] != config
        or frozen["config_sha256"] != CONFIG_SHA256
        or bundle["generator_version"] != settings.version
    ):
        raise ValueError("Frozen generation settings disagree")
    collections = []
    for area in bundle["areas"]:
        if area["status"] != "processed":
            raise ValueError("Partial candidate populations cannot be compared")
        candidate_path = (path.parent / area["candidates_path"]).resolve()
        if (
            candidate_path.parent != path.parent
            or sha256_file(candidate_path) != area["candidates_sha256"]
        ):
            raise ValueError("Candidate bytes changed")
        value = json.loads(candidate_path.read_text(encoding="utf-8"))
        if (
            identity(value) != area["candidate_identity"]
            or len(value["features"]) != area["candidate_count"]
            or value["product_id"] != area["product_id"]
            or value["roi_id"] != area["roi_id"]
            or value["config_sha256"] != CONFIG_SHA256
        ):
            raise ValueError("Candidate identity changed")
        collections.append(value)
    ids = [f["id"] for c in collections for f in c["features"]]
    if len(ids) != len(set(ids)) or len(ids) != bundle["summary"]["candidate_count"]:
        raise ValueError("Candidate identities duplicated or omitted")
    return {
        "bundle_sha256": expected_sha256,
        "unfiltered": collections,
        "classifier_filter": copy.deepcopy(collections),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    bundle = run(args.output, args.dataset)
    print(
        json.dumps(
            {
                "bundle_sha256": sha256_file(args.output / "bundle.json"),
                **bundle["summary"],
            }
        )
    )
    if bundle["summary"]["failed_areas"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
