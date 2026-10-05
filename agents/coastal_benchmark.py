"""Validation-only coastal-buffer comparison using independent SAR labels.

AIS absence is never a negative label. Without exhaustive independent vessel
annotations, false alarms and missed vessels stay null, not zero.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import Point, shape

from agents.acquisition import product_id
from agents.association import one_to_one_matches
from agents.coastal_policy import annotations
from agents.region import load_region
from agents.run_versions import digest

DEFAULT_BUFFERS_M = (0.0, 100.0, 300.0, 500.0, 1000.0)


def compare_buffers(
    detections: dict[str, Any],
    labels: dict[str, Any] | None,
    selected_product_id: str,
    *,
    buffers_m: Sequence[float] = DEFAULT_BUFFERS_M,
    match_radius_m: float = 100.0,
) -> dict[str, Any]:
    """Compare fixed detections; never tune the detector or choose a winner.

    Inputs must already carry physical shoreline annotations from the selected
    reference. Labels define an exhaustively reviewed *valid-imagery* ROI.
    Results hold a fixed vessel denominator, so withholding a coastal vessel
    is reported as a policy miss instead of removing it from the denominator.
    """
    selected = product_id(selected_product_id)
    if not buffers_m or any(not math.isfinite(b) or b < 0 for b in buffers_m):
        raise ValueError("Buffers must be finite, nonnegative and nonempty")
    if not math.isfinite(match_radius_m) or match_radius_m <= 0:
        raise ValueError("Match radius must be finite and positive")
    if detections.get("type") != "FeatureCollection":
        raise ValueError("Detections must be a FeatureCollection")
    if product_id(detections.get("sar_product_id", "")) != selected:
        raise ValueError("Detection product ID does not match selected scene")
    region = load_region()
    features = detections.get("features")
    if not isinstance(features, list):
        raise ValueError("Detections require a feature list")
    points, properties = [], []
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid or not region.covers(geometry):
            raise ValueError("Benchmark detections must be valid and inside NL")
        points.append(geometry.centroid)
        properties.append(feature.get("properties") or {})

    roi = None
    truth: list[Point] = []
    truth_properties: list[dict[str, Any]] = []
    label_metadata: dict[str, Any] = {}
    if labels is not None:
        label_metadata = labels.get("benchmark") or {}
        if label_metadata.get("split") != "validation":
            raise ValueError("Only validation labels may be used for buffer trials")
        if product_id(label_metadata.get("product_id", "")) != selected:
            raise ValueError("Label product ID does not match selected scene")
        if label_metadata.get("acquisition_time") != detections.get("acquisition_time"):
            raise ValueError("Label acquisition time does not match detections")
        if (
            labels.get("type") != "FeatureCollection"
            or label_metadata.get("annotation_complete") is not True
            or label_metadata.get("annotation_scope")
            != "exhaustive_vessels_on_valid_imagery"
            or label_metadata.get("source_kind") != "independent_sar_review"
            or not all(
                label_metadata.get(k)
                for k in ("source", "reviewer", "licence", "acquisition_time")
            )
        ):
            raise ValueError(
                "Exhaustive independent SAR labels and provenance required; "
                "AIS is insufficient"
            )
        roi = shape(label_metadata["valid_imagery_roi"])
        if (
            roi.geom_type not in {"Polygon", "MultiPolygon"}
            or roi.is_empty
            or not roi.is_valid
            or not region.covers(roi)
        ):
            raise ValueError("Label ROI must be valid polygonal NL imagery")
        ids = set()
        for feature in labels["features"]:
            props = feature["properties"]
            identifier = props["label_id"]
            if not identifier or identifier in ids:
                raise ValueError("Unique label IDs required")
            ids.add(identifier)
            point = shape(feature["geometry"])
            if (
                point.geom_type != "Point"
                or point.is_empty
                or not point.is_valid
                or not roi.covers(point)
            ):
                raise ValueError("Labels must be points in the reviewed ROI")
            if props.get("class") not in {"vessel", "non_vessel"}:
                raise ValueError(
                    "Resolve uncertain labels before declaring exhaustive review"
                )
            if props["class"] == "vessel":
                truth.append(point)
                truth_properties.append(props)

    active = [i for i, point in enumerate(points) if roi is None or roi.covers(point)]
    rows = []
    raw_pairs, _, _ = one_to_one_matches(
        [points[i] for i in active], truth, match_radius_m
    )
    for buffer in dict.fromkeys(buffers_m):
        policy = []
        for i in active:
            props = properties[i]
            distance = props.get("distance_to_shore_m")
            if props.get("shore_distance_unbounded") is True:
                distance = math.inf
            physical = props.get("physical_surface", "unknown")
            if physical == "water" and distance is not None and float(distance) < 0:
                raise ValueError("Contradictory physical water / inland shore distance")
            policy.append(
                annotations(
                    physical,
                    distance,
                    buffer,
                    props.get("infrastructure_proximity") is True,
                )
            )
        eligible_indices = [i for i, p in zip(active, policy) if p["alert_eligible"]]
        coastal_count = sum(p["coastal_zone"] for p in policy)
        pairs, _, ambiguous = one_to_one_matches(
            [points[i] for i in eligible_indices], truth, match_radius_m
        )
        truth_policy = [
            annotations(
                p.get("physical_surface", "unknown"),
                (
                    math.inf
                    if p.get("shore_distance_unbounded") is True
                    else p.get("distance_to_shore_m")
                ),
                buffer,
            )
            for p in truth_properties
        ]
        measured = (
            labels is not None
            and detections.get("shoreline_status") == "ok"
            and all(
                p["physical_surface"] in {"land", "water"}
                and p["alert_eligibility_reason"] != "shore_distance_unknown"
                for p in policy + truth_policy
            )
        )
        research_indices = [i for i, p in zip(active, policy) if p["coastal_zone"]]
        research_pairs, _, _ = one_to_one_matches(
            [points[i] for i in research_indices], truth, match_radius_m
        )
        found = {j for j, _ in pairs.values()}
        raw_found = {j for j, _ in raw_pairs.values()}
        strata = {}
        for name in ("harbour_coastal", "open_water", "land_or_shoreline_mismatch"):
            members = {
                i
                for i, p in enumerate(truth_policy)
                if (
                    "land_or_shoreline_mismatch"
                    if p["physical_surface"] == "land"
                    else "harbour_coastal" if p["coastal_zone"] else "open_water"
                )
                == name
            }
            strata[name] = {
                "vessels": len(members) if measured else None,
                "matched_raw": len(members & raw_found) if measured else None,
                "matched_eligible": len(members & found) if measured else None,
                "missed_eligible": len(members - found) if measured else None,
            }
        rows.append(
            {
                "buffer_m": buffer,
                "open_water_eligible_returns": len(eligible_indices),
                "harbour_coastal_returns": coastal_count,
                "retained_returns": len(active),
                "measured": measured,
                "vessels": len(truth) if measured else None,
                "true_positives": len(pairs) if measured else None,
                "false_alarms": (
                    len(eligible_indices) - len(pairs) if measured else None
                ),
                "missed_vessels": len(truth) - len(pairs) if measured else None,
                "detector_missed_vessels": (
                    len(truth) - len(raw_pairs) if measured else None
                ),
                "policy_added_misses": (
                    len(raw_pairs) - len(pairs) if measured else None
                ),
                "recall": len(pairs) / len(truth) if measured and truth else None,
                "precision": (
                    len(pairs) / len(eligible_indices)
                    if measured and eligible_indices
                    else None
                ),
                "ambiguous_targets": len(ambiguous) if measured else None,
                "false_alarms_per_km2": None,
                "by_regime": strata,
                "coastal_research_true_positives": (
                    len(research_pairs) if measured else None
                ),
                "coastal_research_false_returns": (
                    len(research_indices) - len(research_pairs) if measured else None
                ),
                "reason": (
                    None
                    if measured
                    else "Independent exhaustive labels or physical shoreline "
                    "classification unavailable"
                ),
            }
        )
    return {
        "product_id": selected,
        "purpose": (
            "validation buffer comparison; no automatic operating-point selection"
        ),
        "detection_sha256": digest(detections),
        "labels_sha256": digest(labels) if labels is not None else None,
        "labels_provenance": label_metadata,
        "match_radius_m": match_radius_m,
        "denominator": (
            "All labelled vessels in the fixed reviewed ROI, "
            "including harbour/coastal vessels"
        ),
        "trials": rows,
        "limitations": (
            "Counts are proximity-based and conditional on annotation completeness "
            "and geolocation. AIS-only data cannot establish false alarms. "
            "No raster-valid water-area denominator supplied; no density or "
            "province-wide performance claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--buffers", nargs="+", type=float, default=list(DEFAULT_BUFFERS_M)
    )
    parser.add_argument("--match-radius-m", type=float, default=100.0)
    args = parser.parse_args()
    result = compare_buffers(
        json.loads(args.detections.read_text(encoding="utf-8")),
        json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None,
        args.product_id,
        buffers_m=args.buffers,
        match_radius_m=args.match_radius_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
