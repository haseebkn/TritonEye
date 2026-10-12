"""Verify saved exploratory outputs; no SAR imagery or test pixels are opened."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.development_experiments import prediction_summary
from agents.nl_benchmark import bounded_path, load_dataset


def verify_bias_predictions(
    before: dict[str, Any], after: dict[str, Any], delta: float
) -> dict[str, Any]:
    """Actual replay probabilities must reflect the existing bias update."""

    def indexed(collection: dict[str, Any]) -> dict[tuple[float, float], float]:
        return {
            (f["properties"]["source_row"], f["properties"]["source_col"]): f[
                "properties"
            ]["confidence"]
            for f in collection["features"]
        }

    old, new = indexed(before), indexed(after)
    errors = []
    for position, probability in old.items():
        if position not in new:
            raise ValueError("Positive bias update lost an existing peak")
        expected = 1 / (1 + (1 / probability - 1) * math.exp(-delta))
        error = abs(new[position] - expected)
        if error > 1e-5:
            raise ValueError("Replay probabilities do not reflect trained bias")
        errors.append(error)
    return {
        "shared_peaks": len(errors),
        "max_probability_error": max(errors, default=0),
    }


def verify(directory: Path, dataset: Path) -> dict[str, Any]:
    def read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    report = read(directory / "experiment.json")
    manifest, _, _ = load_dataset(dataset)
    if report["release_sha256"] != sha256_file(dataset / "release.json"):
        raise ValueError("Dataset release differs from the experiment")
    if report["step_5_complete"] or report["improvement"] is not None:
        raise ValueError("Exploratory report cannot claim completed improvement")
    if report["test_imagery_opened"] or report["selected_threshold"] is not None:
        raise ValueError("Test lock or threshold boundary violated")
    if not all(g["status"] == "blocked" for g in report["validation_guard"]):
        raise ValueError("Independent-review boundary violated")
    identities = {
        roi["id"]: (scene["split"], scene["product_id"])
        for scene in manifest["scenes"]
        for roi in scene["rois"]
    }
    files, controls, shifts = {}, {}, []
    bias = {f["seed"]: f["bias_delta"] for f in report["fits"] if "bias_delta" in f}
    for case in report["cases"]:
        if (case["split"], case["product_id"]) != identities[case["roi_id"]]:
            raise ValueError("Prediction product/split identity mismatch")
        if case["status"] != "processed":
            continue
        if case["split"] == "test":
            raise ValueError("Held-out predictions cannot be verified or published")
        name = case["model"]
        prefix = {
            "pretrained_member_0": "pretrained",
            "bias_finetuned_member_0": "fine",
            "frozen_croma_token_head": "croma",
        }[name]
        seed = "" if case["seed"] is None else f"_{case['seed']}"
        path = bounded_path(directory, f"{prefix}{seed}_{case['roi_id']}.json")
        collection = read(path)
        if collection["sar_product_id"] != case["product_id"]:
            raise ValueError("Saved prediction product differs from selected scene")
        if prediction_summary(collection) != case["trials"]:
            raise ValueError("Saved predictions differ from published counts")
        files[path.name] = sha256_file(path)
        if name == "pretrained_member_0":
            controls[case["roi_id"]] = collection
        elif name == "bias_finetuned_member_0":
            shifts.append(
                {
                    "roi_id": case["roi_id"],
                    "seed": case["seed"],
                    **verify_bias_predictions(
                        controls[case["roi_id"]], collection, bias[case["seed"]]
                    ),
                }
            )
    for fit in report["fits"]:
        path = bounded_path(directory, fit["checkpoint"])
        if sha256_file(path) != fit["checkpoint_sha256"]:
            raise ValueError("Fitted parameter artifact hash mismatch")
        files[path.name] = fit["checkpoint_sha256"]
    feature_report = directory / "croma_features/experiment.json"
    if sha256_file(feature_report) != report["foundation_features"]["sha256"]:
        raise ValueError("Frozen feature report hash mismatch")
    features = read(feature_report)
    for case in features["cases"]:
        if case["status"] != "frozen_features_only":
            continue
        path = bounded_path(feature_report.parent, case["artifact"])
        if sha256_file(path) != case["artifact_sha256"]:
            raise ValueError("Frozen feature archive hash mismatch")
        files[f"croma_features/{path.name}"] = case["artifact_sha256"]
    return {
        "status": "saved_artifacts_verified_accuracy_unmeasured",
        "experiment_sha256": sha256_file(directory / "experiment.json"),
        "dataset_release_sha256": report["release_sha256"],
        "file_sha256": files,
        "member_probability_checks": shifts,
        "validation_exports_blocked": len(report["validation_guard"]),
        "sar_imagery_opened_by_verifier": False,
        "improvement": None,
        "verifier_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "datasets/nl_benchmark/v0.2.0"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        REPO_ROOT.resolve()
    ):
        raise ValueError("Fresh project-local evidence path required")
    write_json(args.output, verify(args.experiment, args.dataset))


if __name__ == "__main__":
    main()
