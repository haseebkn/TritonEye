"""AIS supporting metrics, deliberately separate from vessel detection truth."""

from typing import Any

from shapely.geometry import shape

from agents.association import one_to_one_matches
from agents.coastal_policy import eligible


def association_summary(
    detections: dict[str, Any],
    positions: Any,
    *,
    threshold: float,
    processing_complete: bool,
    radius_m: float = 100.0,
) -> dict[str, Any]:
    """Proximity recall and geometric ambiguity cannot prove identity correctness."""
    roi = shape(detections["study_roi"])
    positions = positions.loc[positions.geometry.map(roi.covers)]
    n = len(positions)
    result: dict[str, Any] = {
        "aligned_ais_in_roi": n,
        "ais_subset_proximity_recall_raw": None,
        "ais_subset_proximity_recall_post_policy": None,
        "association_correctness": None,
        "ambiguous_assignments": None,
        "correctness_reason": "Independent vessel-identity adjudication unavailable",
        "supporting_only": True,
    }
    if not processing_complete or not n:
        result["reason"] = "Processing incomplete or no co-temporal AIS in selected ROI"
        return result
    returns = [
        f for f in detections["features"] if f["properties"]["confidence"] > threshold
    ]
    for name, subset in (
        ("raw", returns),
        ("post_policy", [f for f in returns if eligible(f["properties"])]),
    ):
        points = [shape(f["geometry"]).centroid for f in subset]
        pairs, _, _ = one_to_one_matches(points, list(positions.geometry), radius_m)
        result[f"ais_subset_proximity_recall_{name}"] = len(pairs) / n
    points = [
        shape(f["geometry"]).centroid for f in returns if eligible(f["properties"])
    ]
    pairs, _, ambiguous = one_to_one_matches(
        points,
        list(positions.geometry),
        [radius_m + float(u) for u in positions.uncertainty_m],
    )
    result["geometric_assignments"] = len(pairs)
    result["ambiguous_assignments"] = len(ambiguous)
    result["ambiguity_scope"] = (
        "Competing geometric candidates, including unmatched returns; "
        "not identity errors"
    )
    return result
