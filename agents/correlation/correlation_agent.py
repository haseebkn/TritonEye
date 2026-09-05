#!/usr/bin/env python3
"""Annotate SAR targets with tentative AIS associations and review eligibility."""

import argparse
import json
import os
import sys
from typing import Any

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.association import (  # noqa: E402
    aligned_ais,
    one_to_one_matches,
    real_ais_source,
)
from agents.region import contains_points, load_region  # noqa: E402
from agents.tracking import RunTracker  # noqa: E402

CORRELATION_RADIUS_M = 500.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=str)
    parser.add_argument("--payload-file", type=str)
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload:
        return dict(json.loads(args.payload))
    if args.payload_file:
        with open(args.payload_file, encoding="utf-8") as source:
            return dict(json.load(source))
    if not sys.stdin.isatty():
        return dict(json.loads(sys.stdin.read()))
    raise ValueError("Provide --payload, --payload-file, or a JSON payload on stdin")


def correlate_targets(
    detections_path: str,
    ais_path: str,
    acquisition_time: str = "",
    metric_crs: str = "",
    *,
    ais_coverage: str = "unknown",
    analysis_region: Any = None,
) -> gpd.GeoDataFrame:
    """Return ALL annotated targets, including exclusions and uncertainty.

    ``metric_crs`` remains accepted for old callers; distances are geodesic.
    Unknown surface, missing timestamp, and missing telemetry never yield an
    automatic alert. A proximity association is tentative; an unmatched return
    is a review candidate, not a confirmed vessel or intentional AIS silence.
    """
    del metric_crs
    targets = gpd.read_file(detections_path).reset_index(drop=True)
    if targets.crs is None:
        raise ValueError("Detections require an explicit geographic CRS")
    targets = targets.to_crs("EPSG:4326")
    region = load_region(analysis_region)
    defaults: dict[str, Any] = {
        "correlation_status": "unassessable",
        "correlation_reason": "AIS coverage or timing is unavailable",
        "review_required": False,
        "operational_alert": False,
        "association_mmsi": None,
        "association_distance_m": None,
        "association_uncertainty_m": None,
        "association_method": None,
        "candidate_mmsis": "[]",
        "ais_coverage": ais_coverage,
        "ais_coverage_completeness": "unverified",
        "association_eligible": False,
    }
    for key, value in defaults.items():
        targets[key] = pd.Series([value] * len(targets), dtype="object")
    if "surface" not in targets:
        targets["surface"] = "unknown"
    targets["surface"] = targets["surface"].fillna("unknown")
    eligible: list[int] = []
    for index, row in targets.iterrows():
        index = int(index)
        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            state, reason = "excluded_invalid_geometry", "Invalid or absent geometry"
        elif not region.covers(geometry):
            state, reason = "excluded_outside_region", "Outside the NL study area"
        elif row["surface"] != "water":
            surface = row["surface"]
            if surface in {"land", "coastal"}:
                state = f"excluded_{surface}"
            elif surface in {"infrastructure", "infrastructure_proximity"}:
                state = "excluded_infrastructure"
            else:
                state = "excluded_unknown_surface"
            reason = f"Surface={surface}; open-water eligibility not established"
        else:
            eligible.append(index)
            targets.at[index, "association_eligible"] = True
            continue
        targets.at[index, "correlation_status"] = state
        targets.at[index, "correlation_reason"] = reason

    try:
        records = pd.read_csv(ais_path, dtype={"mmsi": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        records = pd.DataFrame()
        targets.attrs["ais_read_error"] = str(error)
    positions, diagnostics = aligned_ais(records, acquisition_time)
    if not positions.empty:
        positions = positions.loc[
            contains_points(list(positions.lon), list(positions.lat), region=region)
        ].reset_index(drop=True)
    targets.attrs["ais_diagnostics"] = diagnostics
    targets.attrs["ais_positions"] = len(positions)
    if not real_ais_source(ais_coverage) or positions.empty:
        reason = diagnostics.get("reason", "Real AIS scene coverage is not established")
        for index in eligible:
            targets.at[index, "correlation_reason"] = reason
        return targets
    points = [targets.loc[index].geometry.centroid for index in eligible]
    gates = [CORRELATION_RADIUS_M + float(u) for u in positions.uncertainty_m]
    selected, alternatives, ambiguous = one_to_one_matches(
        points, list(positions.geometry), gates
    )
    for local, index in enumerate(eligible):
        targets.at[index, "candidate_mmsis"] = json.dumps(
            [str(positions.iloc[j].mmsi) for j in alternatives.get(local, [])]
        )
        if local in selected:
            position_index, distance = selected[local]
            position = positions.iloc[position_index]
            targets.at[index, "association_mmsi"] = str(position.mmsi)
            targets.at[index, "association_distance_m"] = distance
            targets.at[index, "association_uncertainty_m"] = float(
                position.uncertainty_m
            )
            targets.at[index, "association_method"] = position.alignment_method
        if local in ambiguous:
            state = "ambiguous_association"
            reason = "Competing gated associations; selected identity is tentative"
        elif local in selected:
            state = "ais_associated"
            reason = "Unique gated proximity association; identity is not verified"
        else:
            state = "uncorrelated_candidate"
            reason = (
                "No gated AIS association; vessel identity and reception coverage "
                "are unverified. Iceberg, clutter, or missing AIS remain possible"
            )
        targets.at[index, "correlation_status"] = state
        targets.at[index, "correlation_reason"] = reason
        targets.at[index, "review_required"] = state != "ais_associated"
    return targets


def summarize_correlation(targets: gpd.GeoDataFrame) -> dict[str, Any]:
    states = {
        str(k): int(v) for k, v in targets.correlation_status.value_counts().items()
    }
    return {
        "states": states,
        "raw_detections": len(targets),
        "eligible_detections": int(targets.association_eligible.sum()),
        "review_candidates": int(targets.review_required.sum()),
        "ais_positions": targets.attrs.get("ais_positions", 0),
        "ais_diagnostics": targets.attrs.get("ais_diagnostics", {}),
        "operational_alerts": 0,
        "precision": None,
        "note": (
            "Analyst-review research outputs; no confirmed dark vessels. "
            "No validated precision or false-alarm rate. Automatic alerts disabled."
        ),
    }


def main() -> None:
    payload = get_payload(parse_arguments())
    detections = payload.get("detections_geojson")
    ais = payload.get("ais_telemetry")
    if not detections or not ais:
        raise ValueError("Payload requires detections_geojson and ais_telemetry")
    bounds = payload.get("spatial_bounds") or {}
    targets = correlate_targets(
        detections,
        ais,
        payload.get("acquisition_time", ""),
        ais_coverage=bounds.get("ais_coverage", "unknown"),
        analysis_region=bounds.get("analysis_region"),
    )
    directory = os.path.dirname(os.path.abspath(detections))
    all_path = os.path.join(directory, "correlation.geojson")
    candidates_path = os.path.join(directory, "review_candidates.geojson")
    targets.to_file(all_path, driver="GeoJSON")
    targets.loc[targets.review_required.astype(bool)].to_file(
        candidates_path, driver="GeoJSON"
    )
    payload["correlation_geojson"] = all_path
    payload["candidates_geojson"] = candidates_path
    # Compatibility pointer only: no artifact or classification claims darkness.
    payload["dark_vessels_geojson"] = candidates_path
    payload["deprecated_fields"] = list(
        dict.fromkeys(payload.get("deprecated_fields", []) + ["dark_vessels_geojson"])
    )
    summary = summarize_correlation(targets)
    payload["correlation_summary"] = summary
    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        tracker.set_tags({"stage": "correlation", "automatic_alerts": "disabled"})
        tracker.log_params({"correlation_base_radius_m": CORRELATION_RADIUS_M})
        metrics = {f"correlation.{k}": float(v) for k, v in summary["states"].items()}
        metrics["correlation.review_candidates"] = float(summary["review_candidates"])
        metrics["correlation.operational_alerts"] = 0.0
        tracker.log_metrics(metrics)
        tracker.log_artifact(all_path, "correlation")
        tracker.log_artifact(candidates_path, "review_candidates")
    finally:
        tracker.end()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
