"""Export aggregate classification results, never restricted chips or IDs."""

import argparse
import json
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT, sha256_file, write_json


def export(report: Path, output: Path) -> dict[str, Any]:
    report, output = report.resolve(), output.resolve()
    if not report.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Local research report required")
    if output.exists() or not output.is_relative_to(
        (REPO_ROOT / "docs/evidence").resolve()
    ):
        raise ValueError("Fresh public aggregate evidence path required")
    value = json.loads(report.read_text(encoding="utf-8"))
    for name, digest in value["artifacts"].items():
        artifact = (report.parent / name).resolve()
        if artifact.parent != report.parent or sha256_file(artifact) != digest:
            raise ValueError("Local baseline artifact hash mismatch")
    audit = value["audit"]
    result = {
        "scope": value["scope"],
        "full_report_path": str(report.relative_to(REPO_ROOT)),
        "full_report_sha256": sha256_file(report),
        "provenance": value["provenance"],
        "audit": {
            k: audit[k]
            for k in (
                "near_definition",
                "angle_groups",
                "missing_angles",
                "missing_angle_labels",
                "rows_in_repeated_angle_groups",
                "rows_in_pure_class_repeated_angle_groups",
                "split_proxy_groups",
                "largest_proxy_group",
                "acquisition_ids_available",
                "coordinates_available",
                "acquisition_disjointness_verified",
            )
        },
        "exact_duplicate_pairs": len(audit["exact_duplicate_pairs"]),
        "near_duplicate_pairs": len(audit["near_duplicate_pairs"]),
        "preprocessing": value["preprocessing"],
        "splits": value["splits"],
        "validation": value["validation"],
        "holdout": value["holdout"],
        "selected_by_validation": value["selected_by_validation"],
        "paired_holdout_comparison": value["paired_holdout_comparison"],
        "training_settings": {
            "seeds": [h["seed"] for h in value["training_histories"]],
            "selected_epochs": [
                h["selected_epoch"] for h in value["training_histories"]
            ],
            "max_epochs": max(len(h["history"]) for h in value["training_histories"]),
            "linear_grid_validation_log_loss": value["linear_grid_validation_log_loss"],
        },
        "wall_seconds": value["wall_seconds"],
        "artifacts": value["artifacts"],
        "limitations": value["limitations"],
        "publication": "Aggregate evidence; chip IDs/pixels/predictions stay local",
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.report, args.output)
    print(json.dumps({"evidence": str(args.output)}))


if __name__ == "__main__":
    main()
