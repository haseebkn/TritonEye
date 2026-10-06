"""Fixed-area vessel metrics, distinct from AIS supporting evidence."""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import shape

from agents.association import one_to_one_matches
from agents.baseline_statistics import binomial_interval, poisson_interval
from agents.coastal_benchmark import compare_buffers
from agents.coastal_policy import annotations as policy_annotations
from agents.coastal_policy import eligible


def score_roi(
    detections: dict[str, Any],
    labels: dict[str, Any] | None,
    *,
    threshold: float,
    score_floor: float,
    water_area_km2: float | None,
    processing_complete: bool,
    buffer_m: float = 300.0,
    match_radius_m: float = 100.0,
) -> dict[str, Any]:
    """Score reviewed complete imagery only; coastal vessels stay in recall.

    The caller must bind cached detections and exhaustive reviewed labels to
    the immutable dataset release before calling this interface. No missing
    labels or AIS silence is interpreted as a negative example.
    """
    if not (
        math.isfinite(threshold)
        and math.isfinite(score_floor)
        and 0 < score_floor <= threshold < 1
    ):
        raise ValueError("Threshold must be in [cached floor, 1)")
    if water_area_km2 is not None and (
        not math.isfinite(water_area_km2) or water_area_km2 < 0
    ):
        raise ValueError("Finite nonnegative physical water area required")
    features = []
    identifiers = set()
    for feature in detections["features"]:
        props = feature["properties"]
        identifier = props["detection_id"]
        score = props["confidence"]
        if not identifier or identifier in identifiers:
            raise ValueError("Unique prediction IDs required")
        identifiers.add(identifier)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not score_floor < score <= 1
        ):
            raise ValueError("Prediction violates cached score-floor contract")
        if score > threshold:  # production detector uses strict >
            distance = (
                math.inf
                if props.get("shore_distance_unbounded")
                else props.get("distance_to_shore_m")
            )
            features.append(
                {
                    **feature,
                    "properties": {
                        **props,
                        **policy_annotations(
                            props.get("physical_surface", "unknown"),
                            distance,
                            buffer_m,
                            props.get("infrastructure_proximity") is True,
                        ),
                    },
                }
            )
    selected = {**detections, "features": features}
    effective_labels = labels if processing_complete else None
    trial = compare_buffers(
        selected,
        effective_labels,
        selected["sar_product_id"],
        buffers_m=[buffer_m],
        match_radius_m=match_radius_m,
    )["trials"][0]
    result: dict[str, Any] = {
        "threshold": threshold,
        "processing_complete": processing_complete,
        "returns": len(features),
        "coastal_exclusions": trial["harbour_coastal_returns"],
        "eligible_returns": trial["open_water_eligible_returns"],
        "abstentions": sum(
            f["properties"].get("physical_surface") not in {"water", "land"}
            or f["properties"].get("alert_eligibility_reason")
            == "shore_distance_unknown"
            for f in features
        ),
        "physical_land_returns": sum(
            f["properties"].get("physical_surface") == "land" for f in features
        ),
        "measured": trial["measured"],
        "reason": trial["reason"],
        "water_area_km2": water_area_km2,
        "raw": None,
        "post_policy": None,
        "by_shore_regime": trial["by_regime"],
        "operational_alerts_enabled": False,
    }
    if not trial["measured"]:
        return result
    # compare_buffers has already checked independent exhaustive provenance,
    # exact product/time, valid ROI, resolved labels and physical policy state.
    assert effective_labels is not None
    roi = shape(effective_labels["benchmark"]["valid_imagery_roi"])
    active = [f for f in features if roi.covers(shape(f["geometry"]).centroid)]
    truth = [
        shape(f["geometry"])
        for f in effective_labels["features"]
        if f["properties"]["class"] == "vessel"
    ]
    eligible_features = [f for f in active if eligible(f["properties"])]
    for key, subset in [("raw", active), ("post_policy", eligible_features)]:
        pairs, _, ambiguous = one_to_one_matches(
            [shape(f["geometry"]).centroid for f in subset], truth, match_radius_m
        )
        tp, fp, fn = len(pairs), len(subset) - len(pairs), len(truth) - len(pairs)
        density_fp = sum(
            i not in pairs and f["properties"]["physical_surface"] == "water"
            for i, f in enumerate(subset)
        )
        result[key] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "vessels": len(truth),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / len(truth) if truth else None,
            "precision_ci95_conditional": binomial_interval(tp, tp + fp),
            "recall_ci95_conditional": binomial_interval(tp, len(truth)),
            "false_alarms_on_water": density_fp,
            "false_alarms_per_km2": (
                density_fp / water_area_km2 if water_area_km2 else None
            ),
            "false_alarms_per_km2_ci95_conditional": (
                poisson_interval(density_fp, water_area_km2) if water_area_km2 else None
            ),
            "geometrically_ambiguous_targets": len(ambiguous),
        }
    result["policy_added_misses"] = trial["policy_added_misses"]
    result["detector_missed_vessels"] = trial["detector_missed_vessels"]
    return result
