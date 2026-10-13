"""Provisional frozen HH/HV classifier transfer, development imagery only."""

from __future__ import annotations

import argparse
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
from rasterio.windows import Window

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.calibration import Calibrator
from agents.nl_benchmark import bounded_path, load_dataset
from agents.statoil_baseline import SEEDS, available_commit, resource_check

LEDGER = "docs/evidence/statoil_baseline_20261013.json"
DEFAULT_DATASET = REPO_ROOT / "datasets/nl_benchmark/v0.2.0"


def checked(path: str, checksum: str, cache: dict[str, str]) -> Path:
    """Bound every input to the project and verify its recorded original bytes."""
    result = bounded_path(REPO_ROOT, path)
    if path not in cache:
        cache[path] = sha256_file(result)
    if cache[path] != checksum:
        raise ValueError(f"Input hash mismatch: {path}")
    return result


def frozen_model() -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Load only the ensemble bound to the published original research ledger."""
    import torch
    from torch import nn

    ledger = json.loads((REPO_ROOT / LEDGER).read_text(encoding="utf-8"))
    cache: dict[str, str] = {}
    path = checked(ledger["full_report_path"], ledger["full_report_sha256"], cache)
    if not path.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Ignored local model report required")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["selected_by_validation"] != "cnn_ensemble":
        raise ValueError("Original validation did not select this ensemble")
    for name, checksum in report["artifacts"].items():
        artifact = (path.parent / name).resolve()
        if artifact.parent != path.parent or sha256_file(artifact) != checksum:
            raise ValueError("Frozen model artifact hash mismatch")
    selection = json.loads((path.parent / "selection.json").read_text())
    models = []
    for seed in SEEDS:
        model = nn.Sequential(
            nn.Conv2d(2, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(24 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        model.load_state_dict(
            torch.load(
                path.parent / f"cnn_{seed}.pth", map_location="cpu", weights_only=True
            )
        )
        model.eval().requires_grad_(False)
        models.append(model)
    return models, selection, report


def parameter_hash(models: list[Any]) -> str:
    checksum = hashlib.sha256()
    for model in models:
        for name, tensor in sorted(model.state_dict().items()):
            checksum.update(name.encode())
            checksum.update(tensor.detach().cpu().numpy().tobytes())
    return checksum.hexdigest()


def crop(images: Any, bbox: list[float]) -> tuple[Any | None, dict[str, Any]]:
    """Native 75px context; never resize, pad, stretch, or silently recenter."""
    x0, y0, x1, y1 = bbox
    if not all(np.isfinite(bbox)) or x1 <= x0 or y1 <= y0:
        raise ValueError("Finite positive target box required")
    col, row = int(np.floor((x0 + x1) / 2)), int(np.floor((y0 + y1) / 2))
    left, top = col - 37, row - 37
    info = {"window_px": [left, top, 75, 75], "target_bbox_px": bbox}
    if x1 - x0 > 75 or y1 - y0 > 75:
        return None, {**info, "status": "abstained_target_larger_than_chip"}
    if top < 0 or left < 0 or top + 75 > images.shape[1] or left + 75 > images.shape[2]:
        return None, {**info, "status": "abstained_roi_boundary"}
    result = images[:, top : top + 75, left : left + 75]
    if not np.isfinite(result).all() or (result <= -60).any():
        return None, {**info, "status": "abstained_invalid_pixels"}
    return result, {**info, "status": "shadow_scored"}


def distribution(images: Any, reference: list[list[float]]) -> dict[str, Any]:
    return {
        "quantiles_db_min_p01_median_p99_max": [
            np.quantile(images[b], [0, 0.01, 0.5, 0.99, 1]).tolist() for b in (0, 1)
        ],
        "fraction_outside_kaggle_p01_p99": [
            float(
                np.mean((images[b] < reference[b][1]) | (images[b] > reference[b][3]))
            )
            for b in (0, 1)
        ],
        "scope": (
            "Descriptive domain comparison; percentile overlap "
            "is not compatibility proof"
        ),
    }


def predict(models: list[Any], image: Any, selection: dict[str, Any]) -> dict[str, Any]:
    import torch

    mean = np.asarray(selection["band_normalization_mean"], dtype="float32").reshape(
        2, 1, 1
    )
    std = np.asarray(selection["band_normalization_std"], dtype="float32").reshape(
        2, 1, 1
    )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid frozen training normalization")
    tensor = torch.from_numpy(((image - mean) / std)[None].astype("float32"))
    with torch.inference_mode():
        scores = [float(model(tensor).flatten().sigmoid()[0]) for model in models]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite frozen classifier output")
    score = float(np.mean(scores))
    return {
        "iceberg_sigmoid_score": score,
        "seed_scores": scores,
        "seed_score_std": float(np.std(scores)),
        "seed_class_disagreement_at_0_5": len({p >= 0.5 for p in scores}) > 1,
        "hypothetical_class_at_0_5": "iceberg" if score >= 0.5 else "ship",
        "score_scope": "Uncalibrated transfer score, not verified class probability",
        "return_preserved": True,
    }


def native_audit(
    scene: dict[str, Any], roi: dict[str, Any], cache: dict[str, str]
) -> tuple[Any, dict[str, Any]]:
    path = checked(roi["chip"]["path"], roi["chip"]["sha256"], cache)
    with rasterio.open(path) as ds:
        if ds.count != 2 or ds.descriptions != ("sigma0_db_hh", "sigma0_db_hv"):
            raise ValueError(
                "Explicit HH then HV calibrated raster descriptions required"
            )
        images = ds.read()
    if not np.isfinite(images).all() or (images <= -60).any():
        raise ValueError("Complete finite native imagery required")
    col, row, width, height = roi["window"]
    if images.shape != (2, height, width):
        raise ValueError("Native ROI dimensions changed")
    metadata, replay_errors = [], []
    for band, pol in enumerate(("hh", "hv")):
        sources = [
            s
            for s in scene["sources"]
            if f"-{pol}-" in s["path"] and s["path"].endswith(".tiff")
        ]
        calibrations = [s for s in roi["calibration"] if f"-{pol}-" in s["path"]]
        if len(sources) != 1 or len(calibrations) != 1:
            raise ValueError(
                "One exact native source and LUT required per polarization"
            )
        source = checked(sources[0]["path"], sources[0]["sha256"], cache)
        calibration = checked(calibrations[0]["path"], calibrations[0]["sha256"], cache)
        annotation = source.parent.parent / "annotation" / (source.stem + ".xml")
        tree = ET.parse(annotation)
        if tree.findtext(".//adsHeader/polarisation", "").lower() != pol:
            raise ValueError("Native annotation polarization mismatch")
        with rasterio.open(source) as ds:
            dn = ds.read(1, window=Window(col, row, width, height))
        calibrated = (
            Calibrator(str(calibration)).to_sigma0_db(dn, row, col).astype("float32")
        )
        difference = float(np.max(np.abs(calibrated - images[band])))
        if difference > 1e-4:
            raise ValueError("Native calibration replay differs from frozen ROI")
        replay_errors.append(difference)
        metadata.append(
            {
                "polarization": pol.upper(),
                "acquisition_mode": tree.findtext(".//adsHeader/mode"),
                "measurement": sources[0],
                "calibration": calibrations[0],
                "annotation_path": str(annotation.relative_to(REPO_ROOT)),
                "annotation_sha256": sha256_file(annotation),
                "range_pixel_spacing_m": float(
                    tree.findtext(".//rangePixelSpacing", "nan")
                ),
                "azimuth_pixel_spacing_m": float(
                    tree.findtext(".//azimuthPixelSpacing", "nan")
                ),
                "incidence_angle_mid_swath_deg": float(
                    tree.findtext(".//incidenceAngleMidSwath", "nan")
                ),
                "thermal_noise_correction_performed": tree.findtext(
                    ".//thermalNoiseCorrectionPerformed"
                ),
            }
        )
    if any(
        not np.isfinite(
            [
                b["range_pixel_spacing_m"],
                b["azimuth_pixel_spacing_m"],
                b["incidence_angle_mid_swath_deg"],
            ]
        ).all()
        for b in metadata
    ):
        raise ValueError("Native spacing and angle metadata required")
    return images, {
        "chip": roi["chip"],
        "native_bands": metadata,
        "calibration_replay_max_abs_db": replay_errors,
        "input_shape": list(images.shape),
        "bands": ["HH", "HV"],
        "compatibility": "provisional_transfer_only",
        "resizing": False,
        "padding": False,
        "chip_size_m_from_native_spacing": [
            75 * metadata[0]["range_pixel_spacing_m"],
            75 * metadata[0]["azimuth_pixel_spacing_m"],
        ],
        "unresolved_compatibility": [
            "Kaggle pixel spacing and preprocessing unavailable",
            "No terrain/noise correction in project calibration",
            "Domain and label transfer unverified",
        ],
    }


def run(output: Path, dataset: Path = DEFAULT_DATASET) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() or not output.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Fresh ignored project output required")
    resource_check()
    import torch

    torch.set_num_threads(4)
    start = time.perf_counter()
    manifest, annotations, status = load_dataset(dataset)
    models, selection, original = frozen_model()
    before = parameter_hash(models)
    cache: dict[str, str] = {}
    results = []
    excluded_test: list[str] = []
    excluded_polarization: list[str] = []
    for scene in manifest["scenes"]:
        # Exclude locked acquisitions before raster, XML or image hash access.
        if scene["split"] == "test":
            excluded_test.extend(r["id"] for r in scene["rois"])
            continue
        if scene["polarizations"] != ["hh", "hv"]:
            excluded_polarization.extend(r["id"] for r in scene["rois"])
            continue
        if scene["split"] not in ("train", "validation") or scene["region"] not in (
            "Newfoundland",
            "Labrador",
        ):
            raise ValueError("Only NL development acquisitions permitted")
        for roi in scene["rois"]:
            if (
                available_commit() < 1024**3
                or psutil.Process().memory_info().rss > 2 * 1024**3
            ):
                raise RuntimeError("Shadow resource budget exhausted")
            objects = [o for o in annotations["objects"] if o["roi_id"] == roi["id"]]
            entry = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "acquisition_time": scene["acquisition_time"],
                "split": scene["split"],
                "region": scene["region"],
                "regime": roi["regime"],
                "objects": [],
                "ignored_context_areas": sum(o["kind"] != "object" for o in objects),
            }
            try:
                images, audit = native_audit(scene, roi, cache)
                entry["audit"] = audit
                entry["distribution"] = distribution(
                    images, original["preprocessing"]["band_quantiles_db"]
                )
                entry["status"] = "processed"
                for obj in objects:
                    if obj["kind"] != "object":
                        continue
                    image, info = crop(images, obj["bbox_px"])
                    record = {
                        "object_id": obj["id"],
                        "provisional_label": obj["class"],
                        "label_scope": (
                            "Existing machine-assisted interpretation; "
                            "no independent review"
                        ),
                        "return_preserved": True,
                        **info,
                    }
                    if image is not None:
                        record.update(predict(models, image, selection))
                        record["distribution"] = distribution(
                            image, original["preprocessing"]["band_quantiles_db"]
                        )
                    entry["objects"].append(record)
            except (ValueError, OSError) as error:
                entry["status"], entry["error"] = "failed", str(error)
            results.append(entry)
    after = parameter_hash(models)
    if before != after:
        raise ValueError("Frozen ensemble parameters changed")
    records = [o for r in results for o in r["objects"]]
    report = {
        "scope": "Provisional real NL HH/HV frozen-classifier shadow study",
        "dataset_release_sha256": sha256_file(dataset / "release.json"),
        "model_report_sha256": sha256_file(
            bounded_path(
                REPO_ROOT,
                json.loads((REPO_ROOT / LEDGER).read_text())["full_report_path"],
            )
        ),
        "model_artifacts": original["artifacts"],
        "model_training_provenance": original["provenance"],
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "code_sha256": sha256_file(__file__),
        "calibrator_sha256": sha256_file(Path(__file__).with_name("calibration.py")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "seeds": list(SEEDS),
        "frozen_normalization": {
            k: selection[k]
            for k in ("band_normalization_mean", "band_normalization_std")
        },
        "attribution": manifest["licence"]["attribution"],
        "independent_review_available": status["independent_review_available"],
        "areas": results,
        "excluded_locked_test_rois": excluded_test,
        "excluded_non_hh_hv_rois": excluded_polarization,
        "summary": {
            "selected_areas": len(results),
            "processed_areas": sum(r["status"] == "processed" for r in results),
            "failed_areas": sum(r["status"] == "failed" for r in results),
            "object_centres": len(records),
            "scored_objects": sum(o["status"] == "shadow_scored" for o in records),
            "abstained_objects": sum(o["status"] != "shadow_scored" for o in records),
            "independently_reviewed_objects": 0,
            "returns_suppressed": 0,
        },
        "accuracy": None,
        "false_alarms_per_km2": None,
        "vessel_recall_change": None,
        "promotion_eligible": False,
        "limitations": [
            "Object-centred selection; no target discovery or detector benchmark",
            "No reviewed truth or labelled icebergs; unresolved labels unchanged",
            "Kaggle spacing/processing unknown; same polarization is insufficient",
            "0.5 predictions/disagreement are descriptive, not operating thresholds",
            "HH/HV scoring does not improve or extend the current VV/VH detector",
            "All returns retained; no model promotion or threshold tuning",
        ],
        "wall_seconds": time.perf_counter() - start,
    }
    output.mkdir(parents=True)
    write_json(output / "shadow.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    report = run(args.output, args.dataset)
    if args.evidence is not None:
        publish(args.output / "shadow.json", args.evidence)
    print(json.dumps(report["summary"]))
    if report["summary"]["failed_areas"]:
        raise SystemExit(1)


def publish(source: Path, output: Path) -> None:
    """Publish native-scene scores and provenance; never Kaggle chips or weights."""
    source, output = source.resolve(), output.resolve()
    if not source.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Ignored local shadow report required")
    if output.exists() or not output.is_relative_to(
        (REPO_ROOT / "docs/evidence").resolve()
    ):
        raise ValueError("Fresh public evidence path required")
    report = json.loads(source.read_text(encoding="utf-8"))
    if (
        report["scope"] != "Provisional real NL HH/HV frozen-classifier shadow study"
        or report["promotion_eligible"] is not False
        or report["accuracy"] is not None
        or report["summary"]["returns_suppressed"] != 0
        or report["parameter_sha256_before"] != report["parameter_sha256_after"]
    ):
        raise ValueError("Provisional frozen shadow contract required")
    write_json(
        output,
        {
            **report,
            "full_report_path": str(source.relative_to(REPO_ROOT)),
            "full_report_sha256": sha256_file(source),
        },
    )


if __name__ == "__main__":
    main()
