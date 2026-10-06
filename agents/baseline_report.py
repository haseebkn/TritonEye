"""Validation operating points with explicit evidence and coverage requirements."""

from __future__ import annotations

import math
from typing import Any

from agents.baseline_statistics import (
    binomial_interval,
    block_interval,
    dependence_blocks,
    poisson_interval,
)

DEFAULT_TARGETS: dict[str, Any] = {
    "status": "provisional research targets, not C-CORE requirements",
    "max_false_alarms_per_km2": 0.1,
    "min_recall": 0.7,
    "min_coverage": 0.9,
    "min_vessels": 30,
    "min_independent_blocks": 5,
    "family_alpha": 0.05,
}
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90, 0.99)


def coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases or any(c["water_area_km2"] is None for c in cases):
        return {"processing": None, "measured": None, "water_km2": None}
    total = sum(c["water_area_km2"] for c in cases)
    return {
        "water_km2": total,
        "processing": (
            sum(c["water_area_km2"] for c in cases if c["processing_complete"]) / total
            if total
            else None
        ),
        "measured": (
            sum(
                c["water_area_km2"]
                for c in cases
                if c["processing_complete"]
                and c["metric_ready"]
                and all(s["measured"] for s in c["scores"])
            )
            / total
            if total
            else None
        ),
    }


def summarize(cases: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Conditional measured subset only; unavailable ROIs remain in coverage."""
    available = [
        (c, next(s for s in c["scores"] if s["threshold"] == threshold)) for c in cases
    ]
    measured = [(c, s) for c, s in available if s["measured"]]
    result: dict[str, Any] = {
        "threshold": threshold,
        "areas": len(cases),
        "measured_areas": len(measured),
        "coverage": coverage(cases),
        "raw": None,
        "post_policy": None,
        "scope": "reviewed, complete validation ROIs only; not province-wide",
        "returns": sum(s["returns"] for c, s in available if c["processing_complete"]),
        "coastal_exclusions": sum(
            s["coastal_exclusions"] for c, s in available if c["processing_complete"]
        ),
        "abstentions": sum(
            s["abstentions"] for c, s in available if c["processing_complete"]
        ),
        "policy_added_misses": (
            sum(s["policy_added_misses"] for _, s in measured) if measured else None
        ),
        "detector_missed_vessels": (
            sum(s["detector_missed_vessels"] for _, s in measured) if measured else None
        ),
        "independent_blocks": len(dependence_blocks([c for c, _ in measured])),
    }
    if not measured:
        result["reason"] = (
            "No complete independently reviewed areas; "
            "no accuracy or false-alarm measurement"
        )
        return result
    for key in ("raw", "post_policy"):
        counts = {
            field: sum(s[key][field] for _, s in measured)
            for field in ("tp", "fp", "fn", "vessels", "false_alarms_on_water")
        }
        exposure = (
            sum(c["water_area_km2"] for c, _ in measured)
            if all(c["water_area_km2"] is not None for c, _ in measured)
            else None
        )
        tp, fp, vessels = counts["tp"], counts["fp"], counts["vessels"]
        rows = [
            {**c, "num": s[key]["tp"], "den": s[key]["vessels"]} for c, s in measured
        ]
        result[key] = {
            **counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / vessels if vessels else None,
            "precision_ci95_conditional": binomial_interval(tp, tp + fp),
            "recall_ci95_conditional": binomial_interval(tp, vessels),
            "water_km2": exposure,
            "false_alarms_per_km2": (
                counts["false_alarms_on_water"] / exposure if exposure else None
            ),
            "false_alarms_per_km2_ci95_conditional": (
                poisson_interval(counts["false_alarms_on_water"], exposure)
                if exposure
                else None
            ),
            "recall_block_uncertainty": block_interval(rows, "num", "den"),
            "precision_block_uncertainty": block_interval(
                [
                    {**c, "num": s[key]["tp"], "den": s[key]["tp"] + s[key]["fp"]}
                    for c, s in measured
                ],
                "num",
                "den",
            ),
            "false_alarm_block_uncertainty": (
                block_interval(
                    [
                        {
                            **c,
                            "num": s[key]["false_alarms_on_water"],
                            "den": c["water_area_km2"],
                        }
                        for c, s in measured
                    ],
                    "num",
                    "den",
                )
                if exposure
                else None
            ),
        }
    return result


def choose_threshold(
    cases: list[dict[str, Any]], thresholds: list[float], targets: dict[str, Any]
) -> dict[str, Any]:
    """No test/train tuning and no winning by withholding every vessel."""
    if any(c["split"] != "validation" for c in cases):
        raise ValueError(
            "Operating point selection is validation only; test remains locked"
        )
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(
            isinstance(t, bool) or not math.isfinite(t) or not 0 < t < 1
            for t in thresholds
        )
    ):
        raise ValueError("Unique nonempty threshold grid required")
    if any(
        isinstance(targets[k], bool) or not math.isfinite(targets[k])
        for k in (
            "min_recall",
            "min_coverage",
            "max_false_alarms_per_km2",
            "min_vessels",
            "min_independent_blocks",
            "family_alpha",
        )
    ):
        raise ValueError("Finite numeric research targets required")
    if any(
        not isinstance(targets[k], int)
        for k in ("min_vessels", "min_independent_blocks")
    ):
        raise ValueError("Integer minimum evidence counts required")
    if not (
        0 < targets["min_recall"] <= 1
        and 0 < targets["min_coverage"] <= 1
        and targets["max_false_alarms_per_km2"] > 0
        and targets["min_vessels"] > 0
        and targets["min_independent_blocks"] >= 2
        and 0 < targets["family_alpha"] < 1
    ):
        raise ValueError(
            "Useful positive recall/coverage and valid evidence targets required"
        )
    cov = coverage(cases)
    blockers = []
    for key in ("processing", "measured"):
        if cov[key] is None or cov[key] < targets["min_coverage"]:
            blockers.append(f"{key} water coverage below research target")
    # Missing/unsupported regions or sensor/setting strata cannot be hidden
    # by averaging them into a dominant successfully processed region.
    for field in ("region", "regime", "polarization"):
        for value in sorted({c[field] for c in cases}):
            group = coverage([c for c in cases if c[field] == value])
            if any(
                group[key] is None or group[key] < targets["min_coverage"]
                for key in ("processing", "measured")
            ):
                blockers.append(f"Insufficient {field} coverage: {value}")
    trials: list[dict[str, Any]] = []
    alpha = targets["family_alpha"] / (2 * len(thresholds))
    for threshold in thresholds:
        summary = summarize(cases, threshold)
        metric = summary["post_policy"]
        failures = list(blockers)
        recall_bounds = fa_bounds = None
        if metric is None:
            failures.append("Independent resolved complete-area labels unavailable")
        else:
            if metric["vessels"] < targets["min_vessels"]:
                failures.append("Too few independently labelled vessels")
            if summary["independent_blocks"] < targets["min_independent_blocks"]:
                failures.append("Too few independent acquisition/geography blocks")
            recall_bounds = binomial_interval(metric["tp"], metric["vessels"], alpha)
            fa_bounds = (
                poisson_interval(
                    metric["false_alarms_on_water"], metric["water_km2"], alpha
                )
                if metric["water_km2"]
                else None
            )
            if recall_bounds is None or recall_bounds[0] < targets["min_recall"]:
                failures.append("Recall lower bound below research target")
            if fa_bounds is None or fa_bounds[1] > targets["max_false_alarms_per_km2"]:
                failures.append("False-alarm upper bound above research target")
        trials.append(
            {
                "threshold": threshold,
                "admissible": not failures,
                "failures": failures,
                "recall_selection_interval": recall_bounds,
                "false_alarm_selection_interval": fa_bounds,
                "summary": summary,
            }
        )
    admissible = [t for t in trials if t["admissible"]]
    winner = (
        max(
            admissible,
            key=lambda t: (
                t["summary"]["post_policy"]["recall"],
                -t["summary"]["post_policy"]["false_alarms_per_km2"],
                -t["threshold"],
            ),
        )
        if admissible
        else None
    )
    return {
        "status": "research_candidate_requires_review" if winner else "blocked",
        "selected_threshold": winner["threshold"] if winner else None,
        "targets": targets,
        "trials": trials,
        "selection_uncertainty": (
            "Conditional binomial/Poisson bounds, Bonferroni adjusted over fixed "
            "grid and two constraints; purposive sampling and clustered targets "
            "limit interpretation"
        ),
        "configuration_changed": False,
        "operational_alerts_enabled": False,
    }


def make_report(
    cases: list[dict[str, Any]],
    thresholds: list[float],
    targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = [c for c in cases if c["split"] == "validation"]
    if 0.15 not in thresholds:
        raise ValueError("Grid must include the fixed 0.15 baseline")
    if any(c["split"] == "test" for c in cases):
        raise ValueError("Test data cannot enter baseline threshold reports")
    strata: dict[str, Any] = {}
    for field in ("region", "regime", "polarization", "season"):
        strata[field] = {
            value: summarize([c for c in validation if c[field] == value], 0.15)
            for value in sorted({c[field] for c in validation})
        }
    return {
        "purpose": (
            "NL research baseline, not operational certification "
            "or C-CORE requirements"
        ),
        "baseline_threshold": 0.15,
        "validation_baseline": summarize(validation, 0.15),
        "stratified_validation": strata,
        "threshold_selection": choose_threshold(
            validation, thresholds, targets or DEFAULT_TARGETS
        ),
        "cases": [
            {k: v for k, v in c.items() if k not in {"labels", "detections"}}
            for c in cases
        ],
        "limits": [
            "No independent SAR reviewer available; released labels remain provisional",
            "AIS subset proximity recall is supporting evidence, never detector "
            "recall or association correctness",
            "Geometric assignment correctness requires independent vessel "
            "identity adjudication",
            "Conditional intervals do not account for shoreline error, label "
            "error or purposive sample selection",
            "Winter, ice, sea state and incidence-angle generalization are "
            "unverified; season is only an acquisition-time descriptor",
            "Sparse shoreline controls are not province-wide or full-area "
            "shoreline validation",
            "Test acquisitions remain locked; processing coverage refers to "
            "selected ROIs, not entire satellite swaths",
        ],
    }
