"""NL model research readiness, public label audit and frozen CROMA probe.

No training or performance-comparison claim is emitted by this interface.
Supervised development is blocked until usable training labels exist, and
validation metrics retain the independent full-area review requirement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from shapely.geometry import Point

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.baseline import Resources
from agents.baseline_report import DEFAULT_TARGETS, DEFAULT_THRESHOLDS
from agents.croma_features import (
    SOURCE_REVISION,
    WEIGHTS_REVISION,
    WEIGHTS_SHA256,
    WINDOW_SIZE,
    load_encoder,
    normalize_window,
    parameter_digest,
)
from agents.nl_benchmark import (
    DEFAULT_DATASET,
    bounded_path,
    load_dataset,
    require_development_scene,
    verify_chip,
)
from agents.region import REGION_PATH, load_region

SEED = 20261006


def readiness(
    manifest: dict[str, Any], annotation: dict[str, Any], status: dict[str, Any]
) -> dict[str, Any]:
    """Keep polarization branches and train/validation/test roles distinct."""
    review = {r["roi_id"]: r for r in status["review_status"]}
    branches: dict[str, Any] = {}
    for pol in (["vv", "vh"], ["hh", "hv"]):
        cases = []
        for scene in manifest["scenes"]:
            if scene["polarizations"] != pol or scene["split"] == "test":
                continue
            for roi in scene["rois"]:
                labels = [o for o in annotation["objects"] if o["roi_id"] == roi["id"]]
                record = review[roi["id"]]
                cases.append(
                    {
                        "roi_id": roi["id"],
                        "product_id": scene["product_id"],
                        "split": scene["split"],
                        "region": scene["region"],
                        "regime": roi["regime"],
                        "confirmed_vessels": sum(
                            o["class"] == "vessel" for o in labels
                        ),
                        "unresolved": record["unresolved"],
                        "resolved_first_pass": bool(
                            record["first_pass_complete"] and not record["unresolved"]
                        ),
                        "metric_ready": record["metric_ready"],
                    }
                )
        positives = sum(c["confirmed_vessels"] for c in cases if c["split"] == "train")
        usable = sum(
            c["confirmed_vessels"]
            for c in cases
            if c["split"] == "train" and c["resolved_first_pass"]
        )
        valid = [c for c in cases if c["split"] == "validation"]
        branches["/".join(p.upper() for p in pol)] = {
            "cases": cases,
            "training_vessels": positives,
            "resolved_training_vessels": usable,
            "supervised_fit_status": "labels_available" if usable else "blocked_labels",
            "validation_status": (
                "labels_available"
                if valid and all(c["metric_ready"] for c in valid)
                else "blocked_independent_labels"
            ),
            "foundation_input_status": (
                "VV/VH radar encoder; native-grid domain shift remains"
                if pol == ["vv", "vh"]
                else "unsupported HH/HV; no renaming or channel substitution"
            ),
        }
    return {
        "dataset_version": manifest["dataset_version"],
        "step_5_complete": False,
        "trainer_implemented": False,
        "supervised_data_exported": False,
        "branches": branches,
        "test_locked": True,
        "operating_constraints": {
            "targets": DEFAULT_TARGETS,
            "threshold_grid": DEFAULT_THRESHOLDS,
            "match_radius_m": 100,
            "coastal_buffer_m": 300,
            "alerts_enabled": False,
        },
        "limits": [
            "Labels_available is readiness, not an implemented trainer or fit result.",
            "Resolved provisional training labels still require disclosed provenance.",
            "Context clutter is not an exhaustive negative annotation.",
            "Unresolved targets and AIS silence cannot become training negatives.",
        ],
    }


def audit_labels(path: Path) -> dict[str, Any]:
    """Geographic screening only, not adoption or endorsement of public labels."""
    region = load_region()
    total = inside = 0
    scenes: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"detect_lon", "detect_lat", "GRD_product_identifier", "is_vessel"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("SARFish geographic label fields are missing")
        for row in reader:
            lon, lat = float(row["detect_lon"]), float(row["detect_lat"])
            if not math.isfinite(lon + lat) or not (
                -180 <= lon <= 180 and -90 <= lat <= 90
            ):
                raise ValueError("Invalid label coordinates")
            total += 1
            if region.covers(Point(lon, lat)):
                inside += 1
                scenes.add(row["GRD_product_identifier"])
                counts[row["is_vessel"] or "unknown"] += 1
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "labels": total,
        "inside_nl_study_area": inside,
        "regional_products": sorted(scenes),
        "regional_is_vessel_counts": dict(counts),
        "boundary_sha256": sha256_file(REGION_PATH),
        "adopted_for_training": False,
        "limitation": "No claim about other datasets or province-wide availability.",
    }


def chip_windows(values: np.ndarray[Any, Any]) -> list[tuple[int, int, Any]]:
    """Complete disjoint native windows; no resize, padding or leftover pixels."""
    if (
        values.ndim != 3
        or values.shape[0] != 2
        or values.shape[1] % WINDOW_SIZE
        or values.shape[2] % WINDOW_SIZE
    ):
        raise ValueError("Complete two-band chip divisible by 128 required")
    return [
        (row, col, normalize_window(values[:, row : row + 128, col : col + 128]))
        for row in range(0, values.shape[1], 128)
        for col in range(0, values.shape[2], 128)
    ]


def extract_features(
    dataset: Path, root: Path, weights: Path, output: Path, device: str
) -> dict[str, Any]:
    # This must be set before the first CUDA context/BLAS handle is initialized.
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA workspace must be :4096:8 or :16:8")
    import torch

    manifest, annotation, status = load_dataset(dataset)
    if output.exists():
        raise ValueError(
            "Use a fresh experiment directory; previous evidence is immutable"
        )
    if device not in {"cpu", "cuda"} or (
        device == "cuda" and not torch.cuda.is_available()
    ):
        raise ValueError("Requested compute device is unavailable")
    # Preflight all selected sources before loading the encoder or writing evidence.
    selected = []
    cases = []
    for scene in manifest["scenes"]:
        for roi in scene["rois"]:
            case = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "acquisition_group": scene["acquisition_group"],
                "acquisition_time": scene["acquisition_time"],
                "split": scene["split"],
                "region": scene["region"],
                "regime": roi["regime"],
                "season": scene["season"],
                "polarizations": scene["polarizations"],
            }
            if scene["split"] == "test":
                cases.append({**case, "status": "locked_test_not_encoded"})
                continue
            if scene["polarizations"] != ["vv", "vh"]:
                cases.append({**case, "status": "unsupported_HH_HV_not_encoded"})
                continue
            require_development_scene(
                scene["product_id"], {**scene, "study_roi": roi["geometry"]}
            )
            path = bounded_path(root, roi["chip"]["path"])
            if sha256_file(path) != roi["chip"]["sha256"]:
                raise ValueError("Released chip checksum mismatch")
            verify_chip(path, roi, [])
            with rasterio.open(path) as src:
                if src.descriptions != ("sigma0_db_vv", "sigma0_db_vh"):
                    raise ValueError(
                        "Physical VV/VH band order differs from checkpoint"
                    )
                chip_windows(
                    src.read()
                )  # validate complete normalization before output
            selected.append((path, roi, case))
    if not selected:
        raise ValueError("No eligible non-test VV/VH chips; encoder was not loaded")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with Resources() as resources:
        start = time.perf_counter()
        model = load_encoder(weights, device)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        load_seconds = time.perf_counter() - start
        before = parameter_digest(model)
        output.mkdir(parents=True, exist_ok=False)
        for path, roi, case in selected:
            started = time.perf_counter()
            with rasterio.open(path) as src:
                windows = chip_windows(src.read())
            tokens, pooled, origins = [], [], []
            repeat_error = None
            with torch.inference_mode():
                for row, col, values in windows:
                    tensor = torch.from_numpy(values[None]).to(device)
                    result = model(tensor)
                    features = result["SAR_encodings"].cpu().numpy()
                    gap = result["SAR_GAP"].cpu().numpy()
                    if not np.isfinite(features).all() or not np.isfinite(gap).all():
                        raise ValueError("Encoder produced nonfinite features")
                    if repeat_error is None:
                        repeated = model(tensor)["SAR_encodings"].cpu().numpy()
                        repeat_error = float(np.abs(features - repeated).max())
                        if repeat_error > 1e-5:
                            raise ValueError("Same-input repeatability check failed")
                    tokens.append(features[0])
                    pooled.append(gap[0])
                    origins.append((row, col))
            target = output / f"{roi['id']}.npz"
            np.savez_compressed(
                target,
                tokens=np.stack(tokens),
                pooled=np.stack(pooled),
                origins=origins,
            )
            cases.append(
                {
                    **case,
                    "status": "frozen_features_only",
                    "chip_sha256": roi["chip"]["sha256"],
                    "windows": len(windows),
                    "token_shape": list(np.stack(tokens).shape),
                    "pooled_shape": list(np.stack(pooled).shape),
                    "pixels_encoded": len(windows) * WINDOW_SIZE**2,
                    "pixels_selected": roi["pixels"],
                    "repeat_max_absolute_error": repeat_error,
                    "seconds_including_serialization": time.perf_counter() - started,
                    "artifact": target.name,
                    "artifact_sha256": sha256_file(target),
                }
            )
        after = parameter_digest(model)
        if before != after or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Frozen encoder parameters changed")
        compute = resources.summary()
        compute.update(
            {
                "device": device,
                "device_name": (
                    torch.cuda.get_device_name()
                    if device == "cuda"
                    else platform.processor()
                ),
                "model_load_seconds": load_seconds,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if device == "cuda" else None
                ),
            }
        )
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        commit = git.stdout.strip() if git.returncode == 0 else None
        dirty = bool(changes.stdout) if changes.returncode == 0 else None
    except OSError:
        commit, dirty = None, None
    report = {
        "experiment_type": "frozen_radar_feature_compatibility",
        "status": "features_extracted_not_a_detector_comparison",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness(manifest, annotation, status),
        "dataset_release_sha256": sha256_file(dataset / "release.json"),
        "split_lock_sha256": sha256_file(dataset / "split_lock.json"),
        "checkpoint_sha256": WEIGHTS_SHA256,
        "checkpoint_revision": WEIGHTS_REVISION,
        "upstream_source_revision": SOURCE_REVISION,
        "parameter_digest_before": before,
        "parameter_digest_after": after,
        "executing_code": {
            "base_commit": commit,
            "worktree_dirty": dirty,
            "file_sha256": {
                name: sha256_file(REPO_ROOT / name)
                for name in (
                    "agents/model_research.py",
                    "agents/croma_features.py",
                    "agents/nl_benchmark.py",
                    "agents/region.py",
                    "agents/baseline_report.py",
                    "agents/baseline.py",
                )
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "numpy": version("numpy"),
            "rasterio": version("rasterio"),
        },
        "settings": {
            "seed": SEED,
            "window_size": WINDOW_SIZE,
            "patch_size": 8,
            "band_order": ["sigma0_db_vv", "sigma0_db_vh"],
            "normalization": "per-window per-band mean +/- 2 sample SD clipped [0,1]",
            "dtype": "float32",
            "batch_size": 1,
            "encoder_frozen": True,
            "deterministic_algorithms": True,
            "tf32": False,
            "cublas_workspace_config": workspace,
            "optimizer": None,
            "training_steps": 0,
            "losses": [],
        },
        "compute": compute,
        "cases": cases,
        "vessel_metrics": None,
        "improvement": None,
        "unsuccessful_experiments": [],
        "blocked_experiments": [
            {
                "name": "detector_fine_tuning",
                "reason": "No usable training vessel labels",
            },
            {
                "name": "frozen_feature_vessel_head",
                "reason": "No usable training vessel labels",
            },
            {
                "name": "foundation_adaptation",
                "reason": "No validated frozen-head evidence",
            },
            {
                "name": "HH_HV_model_comparison",
                "reason": "No compatible model or resolved labels",
            },
        ],
        "limits": [
            "Frozen features are not detection scores, recall or precision.",
            "Native radar grid differs from SSL4EO geocoded land imagery pretraining.",
            "128px windows extrapolate the pretrained 120px field of view.",
            "An 8px feature token can obscure small vessels; usefulness is unmeasured.",
            "Blocked fits were not run; they are not failed improvement experiments.",
        ],
    }
    write_json(output / "experiment.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("readiness")
    check.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    check.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit-labels")
    audit.add_argument("--train", type=Path, required=True)
    audit.add_argument("--validation", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    probe = sub.add_parser("features")
    probe.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    probe.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    probe.add_argument("--weights", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Research outputs must stay inside the project")
    if args.output.exists():
        raise ValueError("Use a new output path to preserve previous evidence")
    if args.command == "readiness":
        report = readiness(*load_dataset(args.dataset))
    elif args.command == "audit-labels":
        report = {
            "geographic_audit_only": True,
            "files": [audit_labels(args.train), audit_labels(args.validation)],
        }
    else:
        started = time.perf_counter()
        try:
            report = extract_features(
                args.dataset, args.artifact_root, args.weights, args.output, args.device
            )
        except Exception as error:
            write_json(
                args.output / "experiment.json",
                {
                    "experiment_type": "frozen_radar_feature_compatibility",
                    "status": "failed_execution",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "exception_type": type(error).__name__,
                    "reason": str(error),
                    "wall_seconds_until_failure": time.perf_counter() - started,
                    "seed": SEED,
                    "device": args.device,
                    "checkpoint_sha256": (
                        sha256_file(args.weights) if args.weights.is_file() else None
                    ),
                    "executing_file_sha256": {
                        name: sha256_file(REPO_ROOT / name)
                        for name in (
                            "agents/model_research.py",
                            "agents/croma_features.py",
                        )
                    },
                    "training_steps": 0,
                    "vessel_metrics": None,
                    "improvement": None,
                    "limitation": "Execution failure is not an improvement result.",
                },
            )
            raise
        print(json.dumps({"status": report["status"], "compute": report["compute"]}))
        return
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
