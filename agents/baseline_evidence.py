"""Publish baseline accounting and hashes without raw predictions or AIS identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT, sha256_file, write_json

CASE_FIELDS = (
    "roi_id",
    "product_id",
    "native_product",
    "acquisition_time",
    "acquisition_group",
    "geographic_group",
    "split",
    "region",
    "regime",
    "polarization",
    "season",
    "metric_ready",
    "processing_complete",
    "water_area_km2",
    "open_water_area_km2",
    "resources",
    "state",
    "reason",
    "prior_development_exposure",
    "detections_sha256",
    "scores",
    "coastal_buffer_trials",
    "association",
    "ais_snapshot_sha256",
    "ais_alignment_counts",
    "ais_supporting_files_sha256",
)


def publish(report_file: Path, output: Path) -> dict[str, Any]:
    """Export the controlled baseline report contract; refuse evidence replacement."""
    if output.exists():
        raise ValueError("Fresh evidence path required")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report["provenance"]["baseline_protocol_version"] != 3:
        raise ValueError("Protocol 3 baseline required")
    if any(c["split"] == "test" for c in report["cases"]):
        raise ValueError("Test cases cannot enter development evidence")
    evidence = {
        "schema_version": 2,
        "full_report_path": str(report_file.resolve().relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
        "full_report_sha256": sha256_file(report_file),
        **{
            k: report[k]
            for k in (
                "purpose",
                "evidence_status",
                "scope",
                "baseline_threshold",
                "validation_baseline",
                "stratified_validation",
                "threshold_selection",
                "provenance",
                "limits",
            )
        },
        "cases": [{k: c[k] for k in CASE_FIELDS if k in c} for c in report["cases"]],
    }
    write_json(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.report, args.output)


if __name__ == "__main__":
    main()
