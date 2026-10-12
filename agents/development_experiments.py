"""Exploratory positive-only NL fits; never an independent accuracy evaluation.

Fit an existing xView3 member's final objectness bias and a CROMA token head.
Unknown pixels receive no loss. Sparse provisional positives cannot calibrate
background rejection; these deliberately bounded fits test that limitation.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from shapely.geometry import Point, mapping, shape

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.baseline import MODEL_SHA256, Resources, annotate, infer_roi
from agents.baseline_context import area_km2
from agents.baseline_report import DEFAULT_TARGETS, DEFAULT_THRESHOLDS
from agents.coastal_policy import eligible
from agents.croma_features import parameter_digest
from agents.geo import Georeferencer
from agents.landmask import LandMask
from agents.model_research import extract_features
from agents.nl_benchmark import bounded_path, load_dataset, validation_labels
from agents.run_versions import processing_versions
from agents.xview3_detector import (
    KEY_FISHING,
    KEY_OBJECTNESS,
    KEY_OFFSET,
    KEY_SIZE,
    KEY_VESSEL,
    XView3Detector,
    _disable_jit_fusion,
)

SEEDS = (20261012, 20261013, 20261014)
STEPS = 60
LEARNING_RATE = 0.001
BIAS_NAME = "model.head.objectness.final.bias"


def freeze_member(member: Any, device: str) -> Any:
    """ScriptModules do not support nn.Module.requires_grad_; tensors do."""
    member.eval().to(device)
    for parameter in member.parameters():
        parameter.requires_grad_(False)
    return member


def training_examples(
    manifest: dict[str, Any], annotation: dict[str, Any]
) -> list[dict[str, Any]]:
    """Only explicit provisional train VV/VH vessels, never inferred negatives."""
    result = []
    for scene in manifest["scenes"]:
        if scene["split"] != "train" or scene["polarizations"] != ["vv", "vh"]:
            continue
        for roi in scene["rois"]:
            for obj in annotation["objects"]:
                if obj["roi_id"] == roi["id"] and obj["class"] == "vessel":
                    x0, y0, x1, y1 = obj["bbox_px"]
                    result.append(
                        {
                            "object_id": obj["id"],
                            "roi_id": roi["id"],
                            "col": (x0 + x1) / 2,
                            "row": (y0 + y1) / 2,
                        }
                    )
    if not result:
        raise ValueError("No explicit provisional training VV/VH vessels")
    return result


def fit_positive_head(
    head: Any, vectors: Any, *, steps: int = STEPS, learning_rate: float = LEARNING_RATE
) -> list[float]:
    """Positive BCE only: no background, unknown or validation supervision."""
    import torch
    import torch.nn.functional as functional

    if vectors.ndim != 2 or len(vectors) == 0 or not torch.isfinite(vectors).all():
        raise ValueError("Finite positive training vectors required")
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=0)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = functional.softplus(-head(vectors)).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def fit_member_bias(bias: Any, logits: Any) -> list[float]:
    """Exact bias-only objective of the existing head, with frozen features.

    Final convolution adds this scalar before the pinned member's unit-
    temperature sigmoid. Cached logits retain its original bias. No exported
    graph backward, full-backbone optimization or pseudo-labeling is implied.
    """
    import torch
    import torch.nn.functional as functional

    initial = bias.detach().clone()
    optimizer = torch.optim.Adam([bias], lr=LEARNING_RATE, weight_decay=0)
    losses = []
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = functional.softplus(-(logits + bias - initial)).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


class MemberDetector(XView3Detector):
    """Identical production decoding, one member, no ensemble/flip TTA."""

    def __init__(self, member: Any, device: str) -> None:
        import torch

        self.model = member
        self.device = device
        self.threshold = 0.05
        self._torch = torch
        self.maps: list[Any] = []
        self.retain_maps = False

    def _forward(self, tile_vh_vv: np.ndarray[Any, Any]) -> dict[str, Any]:
        torch = self._torch
        x = torch.from_numpy(tile_vh_vv[None].astype("float32")).to(self.device)
        # Float32 for the control and fit; differs from production ensemble AMP.
        with torch.no_grad():
            outputs = [value.detach().cpu() for value in self.model(x)]
        # Member tuple differs from the enclosing ensemble tuple: objectness
        # first, then vessel and fishing (verified pinned ScriptModule contract).
        objectness, vessel, fishing, size, offset = outputs
        if self.retain_maps:
            self.maps.append(objectness[0, 0].clone())
        return dict(
            zip(
                (KEY_VESSEL, KEY_FISHING, KEY_OBJECTNESS, KEY_SIZE, KEY_OFFSET),
                (vessel, fishing, objectness, size, offset),
            )
        )


def prediction_summary(collection: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive return counts, with every accuracy field explicitly null."""
    result = []
    for threshold in DEFAULT_THRESHOLDS:
        selected = [
            f
            for f in collection["features"]
            if f["properties"]["confidence"] > threshold
        ]
        result.append(
            {
                "threshold": threshold,
                "returns": len(selected),
                "eligible_returns": sum(eligible(f["properties"]) for f in selected),
                "coastal_withheld": sum(
                    f["properties"].get("alert_eligibility_reason") == "coastal_buffer"
                    for f in selected
                ),
                "land_returns": sum(
                    f["properties"]["physical_surface"] == "land" for f in selected
                ),
                "precision": None,
                "recall": None,
                "false_alarms_per_km2": None,
                "measured": False,
            }
        )
    return result


def feature_vector(archive: Any, col: float, row: float) -> Any:
    """Select the 8px token containing the provisional image-box centre."""
    origin = (int(row) // 128 * 128, int(col) // 128 * 128)
    origins = [tuple(map(int, pair)) for pair in archive["origins"]]
    index = origins.index(origin)
    token = (int(row) % 128 // 8) * 16 + int(col) % 128 // 8
    return archive["tokens"][index, token]


def foundation_predictions(
    head: Any, archive: Any, scene: dict[str, Any], roi: dict[str, Any], root: Path
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    scores = np.empty((roi["size"] // 8, roi["size"] // 8), dtype="float32")
    with torch.no_grad():
        probabilities = head(torch.from_numpy(archive["tokens"])).sigmoid().numpy()
    for index, (row, col) in enumerate(archive["origins"]):
        scores[row // 8 : row // 8 + 16, col // 8 : col // 8 + 16] = probabilities[
            index
        ].reshape(16, 16)
    tensor = torch.from_numpy(scores)
    peaks = (tensor == functional.max_pool2d(tensor[None, None], 3, 1, 1)[0, 0]) & (
        tensor > 0.05
    )
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    rows, cols = (ys.numpy() * 8 + 4).tolist(), (xs.numpy() * 8 + 4).tolist()
    with (
        rasterio.open(bounded_path(root, roi["chip"]["path"])) as source,
        Georeferencer.from_dataset(source) as geo,
    ):
        lons, lats = geo.xy(rows, cols)
    return {
        "type": "FeatureCollection",
        "sar_product_id": scene["product_id"],
        "study_roi": roi["geometry"],
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(float(lon), float(lat))),
                "properties": {
                    "detection_id": f"{roi['id']}_token_{int(y)}_{int(x)}",
                    "confidence": float(scores[y, x]),
                },
            }
            for lon, lat, y, x in zip(lons, lats, ys.tolist(), xs.tolist())
        ],
    }


def run(dataset: Path, root: Path, output: Path, device: str) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if output.exists() or not output.resolve().is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Fresh project-local output directory required")
    manifest, annotation, _ = load_dataset(
        dataset, verify_files=True, artifact_root=root, verify_development_only=True
    )
    examples = training_examples(manifest, annotation)
    weights = root / "models/xview3/traced_ensemble.jit"
    if sha256_file(weights) != MODEL_SHA256:
        raise ValueError("Pinned xView3 checkpoint mismatch")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEEDS[0])
    _disable_jit_fusion(torch)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    selected = []
    guard = []
    for scene in manifest["scenes"]:
        for roi in scene["rois"]:
            case = {
                "roi_id": roi["id"],
                "split": scene["split"],
                "product_id": scene["product_id"],
                "acquisition_group": scene["acquisition_group"],
                "region": scene["region"],
                "regime": roi["regime"],
                "season": scene["season"],
                "polarizations": scene["polarizations"],
            }
            if scene["split"] == "test":
                cases.append({**case, "status": "locked_test_excluded"})
                continue
            if scene["split"] == "validation":
                try:
                    validation_labels(dataset, roi["id"], artifact_root=root)
                except ValueError as error:
                    guard.append(
                        {"roi_id": roi["id"], "status": "blocked", "reason": str(error)}
                    )
                else:
                    raise ValueError(
                        "Reviewed validation labels require a new study protocol"
                    )
            if scene["polarizations"] != ["vv", "vh"]:
                cases.append({**case, "status": "unsupported_HH_HV"})
                continue
            selected.append((scene, roi, case))
    versions = processing_versions(
        REPO_ROOT,
        {},
        artifact_root=root,
        shoreline_cache=root / "data/reference",
        shoreline_source="osm",
    )
    masks = {
        roi["id"]: LandMask.for_footprint(
            shape(roi["geometry"]).bounds, cache_dir=str(root / "data/reference")
        )
        for _, roi, _ in selected
    }
    for _, roi, case in selected:
        case["water_area_km2"] = area_km2(
            masks[roi["id"]].water_geometry(shape(roi["geometry"]))
        )
    with Resources() as resources:
        started = time.perf_counter()
        ensemble = torch.jit.load(str(weights), map_location="cpu")
        member = ensemble.model.models._modules["0"]
        temperature = float(member.code_with_constants[1].c0)
        if temperature != 1:
            raise ValueError(
                "Bias objective requires the pinned unit-temperature member"
            )
        del ensemble
        freeze_member(member, device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        before = parameter_digest(member)
        detector = MemberDetector(member, device)
        bias = dict(member.named_parameters())[BIAS_NAME]
        original_bias = bias.detach().clone()
        base_logits = []
        for scene, roi, case in selected:
            detector.maps.clear()
            positive = [e for e in examples if e["roi_id"] == roi["id"]]
            detector.retain_maps = bool(positive)
            collection, contexts = infer_roi(scene, roi, manifest, root, detector)
            for example in positive:
                row = roi["window"][1] + example["row"]
                col = roi["window"][0] + example["col"]
                for context, scores in zip(contexts, detector.maps):
                    y, x = int((row - context["row"]) // 2), int(
                        (col - context["col"]) // 2
                    )
                    if 0 <= y < 1024 and 0 <= x < 1024:
                        probability = float(scores[y, x])
                        if not 0 < probability < 1:
                            raise ValueError(
                                "Positive score cannot be converted to a finite logit"
                            )
                        base_logits.append(
                            float(np.log(probability / (1 - probability)))
                        )
                        break
                else:
                    raise ValueError("Training centre not covered by native context")
            annotate(collection, masks[roi["id"]], infrastructure=False)
            cases.append(
                {
                    **case,
                    "status": "processed",
                    "model": "pretrained_member_0",
                    "seed": None,
                    "trials": prediction_summary(collection),
                }
            )
            write_json(output / f"pretrained_{roi['id']}.json", collection)
            print(f"pretrained {roi['id']}", flush=True)
        detector.maps.clear()
        detector.retain_maps = False
        fits = []
        logits = torch.tensor(base_logits, device=device)
        for seed in SEEDS:
            torch.manual_seed(seed)
            with torch.no_grad():
                bias.copy_(original_bias)
            bias.requires_grad_(True)
            losses = fit_member_bias(bias, logits)
            bias.requires_grad_(False)
            after = parameter_digest(member)
            if before == after:
                raise ValueError("Fine-tune did not change model parameters")
            patch = output / f"member_bias_{seed}.pt"
            torch.save({BIAS_NAME: bias.detach().cpu()}, patch)
            fits.append(
                {
                    "model": "bias_finetuned_member_0",
                    "seed": seed,
                    "trainable_parameters": 1,
                    "training_examples": len(examples),
                    "losses": losses,
                    "parameter_digest_before": before,
                    "parameter_digest_after": after,
                    "bias_delta": float((bias - original_bias).item()),
                    "checkpoint": patch.name,
                    "checkpoint_sha256": sha256_file(patch),
                }
            )
            for scene, roi, case in selected:
                collection, _ = infer_roi(scene, roi, manifest, root, detector)
                annotate(collection, masks[roi["id"]], infrastructure=False)
                cases.append(
                    {
                        **case,
                        "status": "processed",
                        "model": "bias_finetuned_member_0",
                        "seed": seed,
                        "trials": prediction_summary(collection),
                    }
                )
                write_json(output / f"fine_{seed}_{roi['id']}.json", collection)
            print(f"fine-tuned member seed {seed}", flush=True)
        del detector, member, bias
        member_compute = {
            "seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
        }
        if device == "cuda":
            torch.cuda.empty_cache()
        features = extract_features(
            dataset,
            root,
            root / "models/croma/CROMA_base.pt",
            output / "croma_features",
            device,
        )
        by_roi = {
            c["roi_id"]: c
            for c in features["cases"]
            if c["status"] == "frozen_features_only"
        }
        vectors = []
        for example in examples:
            with np.load(
                output / "croma_features" / by_roi[example["roi_id"]]["artifact"]
            ) as archive:
                vectors.append(
                    feature_vector(archive, example["col"], example["row"]).copy()
                )
        training = torch.from_numpy(np.stack(vectors))
        for seed in SEEDS:
            torch.manual_seed(seed)
            head: Any = torch.nn.Linear(training.shape[1], 1)
            initial = parameter_digest(head)
            losses = fit_positive_head(head, training)
            patch = output / f"croma_head_{seed}.pt"
            torch.save(head.state_dict(), patch)
            fits.append(
                {
                    "model": "frozen_croma_token_head",
                    "seed": seed,
                    "trainable_parameters": sum(p.numel() for p in head.parameters()),
                    "training_examples": len(examples),
                    "losses": losses,
                    "parameter_digest_before": initial,
                    "parameter_digest_after": parameter_digest(head),
                    "checkpoint": patch.name,
                    "checkpoint_sha256": sha256_file(patch),
                }
            )
            for scene, roi, case in selected:
                with np.load(
                    output / "croma_features" / by_roi[roi["id"]]["artifact"]
                ) as archive:
                    collection = foundation_predictions(head, archive, scene, roi, root)
                annotate(collection, masks[roi["id"]], infrastructure=False)
                cases.append(
                    {
                        **case,
                        "status": "processed",
                        "model": "frozen_croma_token_head",
                        "seed": seed,
                        "trials": prediction_summary(collection),
                    }
                )
                write_json(output / f"croma_{seed}_{roi['id']}.json", collection)
            print(f"CROMA head seed {seed}", flush=True)
        compute = resources.summary()
        compute.update(
            {
                "device": device,
                "device_name": (
                    torch.cuda.get_device_name()
                    if device == "cuda"
                    else platform.processor()
                ),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "wall_seconds": time.perf_counter() - started,
                "member_phase": member_compute,
                "peak_scope": "final CUDA peak is CROMA; member phase separate",
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                ),
            }
        )
    report = {
        "protocol_version": 1,
        "status": "exploratory_positive_only_fits",
        "step_5_complete": False,
        "independent_review_available": False,
        "test_imagery_opened": False,
        "selected_threshold": None,
        "improvement": None,
        "alerts_enabled": False,
        "dataset_version": manifest["dataset_version"],
        "release_sha256": sha256_file(dataset / "release.json"),
        "split_lock_sha256": sha256_file(dataset / "split_lock.json"),
        "annotations_sha256": sha256_file(dataset / "annotations.json"),
        "upstream_checkpoint_sha256": MODEL_SHA256,
        "settings": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0,
            "augmentation": "none",
            "objective": "positive-only BCE at provisional image-box centres",
            "negative_examples": 0,
            "precision": "float32",
            "deterministic_algorithms": True,
            "targets": DEFAULT_TARGETS,
            "thresholds": DEFAULT_THRESHOLDS,
            "buffer_m": 300,
            "match_radius_m": 100,
        },
        "training_examples": examples,
        "validation_guard": guard,
        "fits": fits,
        "cases": cases,
        "compute": compute,
        "foundation_features": {
            "report": "croma_features/experiment.json",
            "sha256": sha256_file(output / "croma_features/experiment.json"),
            "checkpoint_sha256": features["checkpoint_sha256"],
            "parameter_digest_before": features["parameter_digest_before"],
            "parameter_digest_after": features["parameter_digest_after"],
        },
        "provenance": {
            "processing_versions": versions,
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "source_hashes": {
                name: sha256_file(REPO_ROOT / name)
                for name in (
                    "agents/development_experiments.py",
                    "agents/model_research.py",
                    "agents/croma_features.py",
                    "agents/baseline.py",
                    "agents/baseline_context.py",
                    "agents/xview3_detector.py",
                    "agents/nl_benchmark.py",
                )
            },
        },
        "limits": [
            "Provisional positives may be mislabelled; no independent truth or "
            "identity claims.",
            "Positive-only loss rewards score inflation and cannot learn "
            "background rejection; reduced loss is not measured improvement.",
            "Fine-tuning changes one existing objectness bias; backbone, vessel "
            "and offset heads remain frozen. This is not full fine-tuning.",
            "Member zero before/after is the matched architecture control; "
            "the production 12-member flip-TTA ensemble is different.",
            "CROMA has an 8px token stride without offset refinement; raw scores "
            "are not calibrated across architectures.",
            "Approaches use identical supported development areas and fixed "
            "policy/grid, but only xView3 uses native 2048 contexts.",
            "Unresolved objects, clutter, empty areas and AIS silence are never "
            "negatives; validation annotations never enter fitting.",
            "HH/HV is excluded. Labrador model coverage remains absent.",
            "Accuracy intervals need reviewed truth; optimizer seeds are "
            "not independent data samples.",
            "Pretraining overlap and prior development exposure are limitations.",
            "Encoder adaptation deferred pending useful frozen-head evidence.",
        ],
    }
    for category in ("code", "configuration"):
        for name, checksum in versions[category].items():
            if sha256_file(REPO_ROOT / name) != checksum:
                raise ValueError("Executing code/configuration changed during study")
    write_json(output / "experiment.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "datasets/nl_benchmark/v0.2.0"
    )
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a fresh output")
    if args.publish and (
        args.publish.exists()
        or not args.publish.resolve().is_relative_to(REPO_ROOT.resolve())
    ):
        raise ValueError("Fresh project-local public evidence path required")
    started = time.perf_counter()
    try:
        report = run(args.dataset, args.artifact_root, args.output, args.device)
    except Exception as error:
        if args.output.resolve().is_relative_to(REPO_ROOT.resolve()):
            write_json(
                args.output / "failure.json",
                {
                    "status": "failed_execution",
                    "reason": str(error),
                    "exception_type": type(error).__name__,
                    "seconds": time.perf_counter() - started,
                    "improvement": None,
                    "alerts_enabled": False,
                    "source_sha256": sha256_file(Path(__file__)),
                },
            )
        raise
    if args.publish:
        public = {
            **report,
            "training_examples": [
                {"object_id": e["object_id"], "roi_id": e["roi_id"]}
                for e in report["training_examples"]
            ],
            "local_report_sha256": sha256_file(args.output / "experiment.json"),
        }
        write_json(args.publish, public)


if __name__ == "__main__":
    main()
