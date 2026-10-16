"""Paired validation shadow inference and guarded lifecycle evaluation.

Runs use the existing calibrated xView3 adapter. All selected validation areas
stay in coverage, including failed and unsupported HH/HV areas. Only the local
research alias can be promoted; predictions do not affect production missions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.baseline import Resources, annotate, infer_roi
from agents.baseline_context import area_km2
from agents.baseline_metrics import score_roi
from agents.baseline_report import DEFAULT_TARGETS, DEFAULT_THRESHOLDS, make_report
from agents.landmask import LandMask
from agents.model_lifecycle import DEFAULT_STORE, Lifecycle, now
from agents.nl_benchmark import (
    load_dataset,
    require_development_scene,
    validation_labels,
)
from agents.run_versions import hash_paths
from agents.xview3_detector import XView3Detector

SCORE_FLOOR = 0.05
MAX_RESOURCE_RATIO = 2.0  # provisional engineering budget, not operational requirements


def code_identity() -> dict[str, str]:
    return hash_paths(list((REPO_ROOT / "agents").rglob("*.py")), REPO_ROOT)


def configuration_identity() -> dict[str, str]:
    return hash_paths(
        list((REPO_ROOT / "configs").rglob("*"))
        + [REPO_ROOT / "requirements.txt", REPO_ROOT / "pyproject.toml"],
        REPO_ROOT,
    )


def validation_scope(dataset: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _, status = load_dataset(dataset)
    for scene in manifest["scenes"]:
        if scene["split"] != "validation":
            continue
        for roi in scene["rois"]:
            require_development_scene(
                scene["product_id"],
                {**scene, "study_roi": roi["geometry"], "roi_id": roi["id"]},
            )
    return manifest, {r["roi_id"]: r for r in status["review_status"]}


def evaluate_shadow(lifecycle: Lifecycle, shadow: dict[str, Any]) -> dict[str, Any]:
    """Recompute from sealed predictions and current hash-bound review history."""
    if shadow["code_sha256"] != code_identity():
        raise ValueError("Shadow code changed; run a new comparison")
    if shadow["configuration_sha256"] != configuration_identity():
        raise ValueError("Shadow configuration changed; run a new comparison")
    for path, checksum in shadow["reference_assets_sha256"].items():
        if sha256_file(path) != checksum:
            raise ValueError("Shadow shoreline reference changed")
    dataset = Path(shadow["dataset"])
    if sha256_file(dataset / "release.json") != shadow["dataset_release_sha256"]:
        raise ValueError("Shadow dataset release changed")
    manifest, statuses = validation_scope(dataset)
    expected = {
        r["id"]: (s, r)
        for s in manifest["scenes"]
        if s["split"] == "validation"
        for r in s["rois"]
    }
    reports: dict[str, Any] = {}
    for role in ("reference", "candidate"):
        lifecycle.package(shadow[f"{role}_id"])
        cases = shadow["cases"][role]
        if len(cases) != len(expected) or {c["roi_id"] for c in cases} != set(expected):
            raise ValueError(
                "Shadow must account for every validation area exactly once"
            )
        evaluated = []
        for original in cases:
            scene, roi = expected[original["roi_id"]]
            case = dict(original)
            case.update(
                split="validation",
                product_id=scene["product_id"],
                acquisition_group=scene["acquisition_group"],
                geographic_group=roi["group"],
                region=scene["region"],
                regime=roi["regime"],
                season=scene["season"],
                polarization="/".join(scene["polarizations"]),
                metric_ready=statuses[roi["id"]]["metric_ready"],
            )
            detections = {
                "type": "FeatureCollection",
                "features": [],
                "sar_product_id": scene["product_id"],
                "sar_product": scene["name"],
                "acquisition_time": scene["acquisition_time"],
                "study_roi": roi["geometry"],
                "split": "validation",
            }
            labels = None
            if case["processing_complete"]:
                prediction = Path(case["prediction_path"])
                if sha256_file(prediction) != case["prediction_sha256"]:
                    raise ValueError("Shadow prediction changed")
                detections = json.loads(prediction.read_text(encoding="utf-8"))
                for key in (
                    "sar_product_id",
                    "sar_product",
                    "acquisition_time",
                    "study_roi",
                    "split",
                ):
                    if (
                        detections.get(key)
                        != {
                            "sar_product_id": scene["product_id"],
                            "sar_product": scene["name"],
                            "acquisition_time": scene["acquisition_time"],
                            "study_roi": roi["geometry"],
                            "split": "validation",
                        }[key]
                    ):
                        raise ValueError(
                            "Shadow predictions refer to a different scene or area"
                        )
                require_development_scene(scene["product_id"], detections)
                if any(
                    not shape(roi["geometry"]).covers(shape(f["geometry"]).centroid)
                    for f in detections["features"]
                ):
                    raise ValueError("Shadow returns outside selected validation area")
                if case["metric_ready"]:
                    labels = validation_labels(
                        dataset, roi["id"], artifact_root=Path(shadow["artifact_root"])
                    )
                    mask = LandMask.for_footprint(
                        shape(roi["geometry"]).bounds,
                        cache_dir=shadow["shoreline_cache"],
                    )
                    annotate(detections, mask, infrastructure=True)
                    annotate(labels, mask, infrastructure=False)
                    area = area_km2(mask.water_geometry(shape(roi["geometry"])))
                    if not math.isclose(area, case["water_area_km2"], rel_tol=1e-9):
                        raise ValueError("Physical water exposure changed")
            case["scores"] = [
                score_roi(
                    detections,
                    labels,
                    threshold=t,
                    score_floor=SCORE_FLOOR,
                    water_area_km2=case["water_area_km2"],
                    processing_complete=case["processing_complete"],
                )
                for t in DEFAULT_THRESHOLDS
            ]
            evaluated.append(case)
        reports[role] = make_report(evaluated, list(DEFAULT_THRESHOLDS))
    selection = reports["candidate"]["threshold_selection"]
    blockers = []
    threshold = selection["selected_threshold"]
    if threshold is None:
        blockers.append(
            "Reviewed validation evidence and operating constraints not satisfied"
        )
    state = lifecycle.status()
    if (
        state["revision"] != shadow["reference_revision"]
        or (state["champion_id"] or state["reference_id"]) != shadow["reference_id"]
    ):
        blockers.append("Reference changed after shadow run")
    reference_threshold = shadow["reference_threshold"] or 0.15
    reference_trial = next(
        t
        for t in reports["reference"]["threshold_selection"]["trials"]
        if t["threshold"] == reference_threshold
    )
    if threshold is not None:
        candidate_trial = next(
            t for t in selection["trials"] if t["threshold"] == threshold
        )
        control, candidate = (
            reference_trial["summary"]["post_policy"],
            candidate_trial["summary"]["post_policy"],
        )
        if control is None:
            blockers.append("Reference accuracy unmeasured")
        else:
            for metric, sign in (
                ("recall", 1),
                ("precision", 1),
                ("false_alarms_per_km2", -1),
            ):
                if (
                    control[metric] is None
                    or candidate[metric] is None
                    or sign * candidate[metric] < sign * control[metric]
                ):
                    blockers.append(f"Candidate {metric} regressed or unavailable")
    for control, candidate in zip(
        shadow["cases"]["reference"], shadow["cases"]["candidate"]
    ):
        if control["roi_id"] != candidate["roi_id"]:
            raise ValueError("Paired shadow areas differ in order")
        if control["processing_complete"] and not candidate["processing_complete"]:
            blockers.append(f"Candidate processing failed: {control['roi_id']}")
        if control["processing_complete"] and candidate["processing_complete"]:
            for metric in ("wall_seconds", "rss_sampled_peak_bytes"):
                a, b = control["resources"][metric], candidate["resources"][metric]
                if any(
                    isinstance(v, bool)
                    or not isinstance(v, (float, int))
                    or not math.isfinite(v)
                    or v <= 0
                    for v in (a, b)
                ):
                    raise ValueError("Positive measured runtime and memory required")
                if b > MAX_RESOURCE_RATIO * a:
                    blockers.append(
                        f"Candidate {metric} exceeds provisional 2x budget: "
                        f"{control['roi_id']}"
                    )
    return {
        "status": (
            "eligible_for_manual_research_promotion" if not blockers else "blocked"
        ),
        "blockers": blockers,
        "selected_threshold": threshold,
        "reports": reports,
        "targets": DEFAULT_TARGETS,
        "max_resource_ratio": MAX_RESOURCE_RATIO,
        "operational_alerts_enabled": False,
    }


def run_shadow(
    lifecycle: Lifecycle,
    candidate_id: str,
    dataset: Path,
    artifact_root: Path,
    output: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute two isolated, paired inference branches, with no pointer mutation."""
    manifest, _ = validation_scope(dataset)  # guard before loading any weights/pixels
    if output.exists() or any(
        output.resolve().is_relative_to(p.resolve())
        for p in (REPO_ROOT / "datasets", dataset, artifact_root / "datasets")
    ):
        raise ValueError("Fresh output outside datasets required")
    resolved = lifecycle.resolve()
    candidate = lifecycle.package(candidate_id)
    load_dataset(
        dataset,
        verify_files=True,
        artifact_root=artifact_root,
        verify_development_only=True,
    )
    output.mkdir(parents=True)
    cache = artifact_root / "data/reference"
    shadow: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now(),
        "reference_id": resolved["model_id"],
        "reference_revision": resolved["revision"],
        "reference_threshold": resolved["threshold"],
        "candidate_id": candidate_id,
        "dataset": str(dataset.resolve()),
        "dataset_release_sha256": sha256_file(dataset / "release.json"),
        "artifact_root": str(artifact_root.resolve()),
        "shoreline_cache": str(cache.resolve()),
        "code_sha256": code_identity(),
        "configuration_sha256": configuration_identity(),
        "reference_assets_sha256": {},
        "device": device,
        "platform": platform.platform(),
        "scope": "all validation areas; no train/test inference",
        "cases": {},
        "operational_alerts_enabled": False,
    }
    for role, package in (("reference", resolved["package"]), ("candidate", candidate)):
        cases = []
        detector = None
        for scene in manifest["scenes"]:
            if scene["split"] != "validation":
                continue
            for roi in scene["rois"]:
                case: dict[str, Any] = {
                    "roi_id": roi["id"],
                    "processing_complete": False,
                    "water_area_km2": None,
                    "resources": None,
                }
                try:
                    mask = LandMask.for_footprint(
                        shape(roi["geometry"]).bounds, cache_dir=str(cache)
                    )
                    if mask.shapefile:
                        for suffix in (".shp", ".shx", ".dbf", ".prj"):
                            reference = Path(mask.shapefile).with_suffix(suffix)
                            checksum = sha256_file(reference)
                            previous = shadow["reference_assets_sha256"].setdefault(
                                str(reference.resolve()), checksum
                            )
                            if previous != checksum:
                                raise ValueError(
                                    "Shoreline changed during shadow inference"
                                )
                    case["water_area_km2"] = area_km2(
                        mask.water_geometry(shape(roi["geometry"]))
                    )
                    if scene["polarizations"] != package["metadata"]["polarizations"]:
                        case.update(
                            state="unsupported_polarization",
                            reason="HH/HV adapter unavailable",
                        )
                    else:
                        if detector is None:
                            start = time.perf_counter()
                            detector = XView3Detector(
                                package["weights"], device=device, threshold=SCORE_FLOOR
                            )
                            case["model_load_seconds"] = time.perf_counter() - start
                        cuda = detector.device.startswith("cuda")
                        if cuda:
                            detector._torch.cuda.synchronize()
                        with Resources() as resources:
                            predictions, contexts = infer_roi(
                                scene, roi, manifest, artifact_root, detector
                            )
                            if cuda:
                                detector._torch.cuda.synchronize()
                        annotate(predictions, mask, infrastructure=True)
                        predictions["shoreline_status"] = "ok"
                        path = output / f"{role}_{roi['id']}.geojson"
                        write_json(path, predictions)
                        case.update(
                            processing_complete=True,
                            state="processed",
                            resources=resources.summary(),
                            contexts=contexts,
                            prediction_path=str(path.resolve()),
                            prediction_sha256=sha256_file(path),
                        )
                except Exception as exc:
                    case.update(state="failed", reason=f"{type(exc).__name__}: {exc}")
                cases.append(case)
                print(f"{role}/{roi['id']}: {case['state']}", flush=True)
        shadow["cases"][role] = cases
        del detector
        gc.collect()
    # Dataset and implementation changes during inference invalidate the run.
    evaluation = evaluate_shadow(lifecycle, shadow)
    identifier = lifecycle.save("shadows", shadow)
    result = {"shadow_id": identifier, "shadow": shadow, "evaluation": evaluation}
    write_json(output / "shadow.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = run_shadow(
        Lifecycle(args.store),
        args.candidate,
        args.dataset,
        args.artifact_root,
        args.output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "shadow_id": result["shadow_id"],
                "status": result["evaluation"]["status"],
                "blockers": result["evaluation"]["blockers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
