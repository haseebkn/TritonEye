#!/usr/bin/env python3
"""
TritonEye Correlation Agent
Correlates detected vessel footprints with AIS telemetry.
Isolates "dark" vessels (vessel detections without corresponding active AIS signals).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Agents run as standalone scripts; put the workspace root on the path first.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Third-party imports (handled gracefully if missing during bootstrap checks)
try:
    import geopandas as gpd
    import pandas as pd
    import shapely.geometry

    from agents.tracking import RunTracker
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    # We will let the script fail gracefully on execution rather than crash on import


# Radius around an AIS position within which a detection counts as correlated,
# in metres. Roughly 1 nautical mile, absorbing transponder timing offset and
# vessel drift across the +/-5 minute matching window.
CORRELATION_RADIUS_M = 2000.0


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="TritonEye Correlation Agent")
    parser.add_argument(
        "--payload", type=str, help="JSON payload string passed directly"
    )
    parser.add_argument(
        "--payload-file", type=str, help="Path to JSON file containing payload"
    )
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Retrieves the payload from CLI args or stdin."""
    if args.payload:
        return json.loads(args.payload)  # type: ignore[no-any-return]
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # Stdin fallback
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())  # type: ignore[no-any-return]

    raise ValueError(
        "No input payload provided via --payload, --payload-file, or stdin."
    )


def correlate_targets(
    detections_path: str,
    ais_path: str,
    acquisition_time: str = "",
    metric_crs: str = "",
) -> gpd.GeoDataFrame:
    """
    Loads detections and AIS records, projects them to a metric CRS, buffers AIS
    points by CORRELATION_RADIUS_M, and filters for detections with NO active AIS.
    Enforces a ±5-minute temporal window on AIS records if acquisition_time is provided.

    When metric_crs is empty the appropriate UTM zone is derived from the
    detections themselves. The buffer is a distance in metres, so it is only
    meaningful in a projected CRS covering the data — a zone inherited from
    elsewhere in the payload would silently distort it.
    """
    # 1. Load detections GeoJSON
    detections_gdf = gpd.read_file(detections_path)

    if detections_gdf.empty:
        if detections_gdf.crs:
            return detections_gdf.to_crs("EPSG:4326")
        return detections_gdf

    # Only open water is alert-eligible:
    #   land            terrain -- a rock outcrop has no AIS transmitter and
    #                   would otherwise satisfy the dark-vessel definition
    #   coastal         the ambiguous nearshore band
    #   infrastructure  a known production platform, which would otherwise be
    #                   reported dark on every single pass over its field
    #
    # These stay in detections.geojson -- annotated, not deleted -- so the
    # rejection remains auditable. Detections predating the mask carry no
    # `surface` property and are treated as water, so older missions correlate
    # exactly as they did before.
    if "surface" in detections_gdf.columns:
        before = len(detections_gdf)
        detections_gdf = detections_gdf[
            detections_gdf["surface"].isna() | (detections_gdf["surface"] == "water")
        ].copy()
        excluded = before - len(detections_gdf)
        if excluded:
            print(
                f"Land mask: {excluded} of {before} detections excluded from "
                f"dark-vessel candidacy (not classified as water).",
                file=sys.stderr,
            )
        if detections_gdf.empty:
            return detections_gdf.to_crs("EPSG:4326")

    if not metric_crs:
        metric_crs = detections_gdf.estimate_utm_crs()

    # 2. Load and parse AIS telemetry CSV
    ais_df = pd.read_csv(ais_path)

    # Apply temporal filtering if acquisition_time is supplied
    if acquisition_time:
        ais_df["timestamp_dt"] = pd.to_datetime(ais_df["timestamp"])
        acq_dt = pd.to_datetime(acquisition_time)
        time_diff = abs(ais_df["timestamp_dt"] - acq_dt)
        ais_df = ais_df[time_diff <= pd.Timedelta(minutes=5)].copy()

    # Convert lat/lon fields to shapely Points
    geometry = [shapely.geometry.Point(xy) for xy in zip(ais_df["lon"], ais_df["lat"])]

    # Cast to GeoDataFrame setting WGS84 CRS
    ais_gdf = gpd.GeoDataFrame(ais_df, crs="EPSG:4326", geometry=geometry)

    # 3. Reproject to metric projected coordinate reference system (UTM Zone 22N)
    detections_projected = detections_gdf.to_crs(metric_crs)
    ais_projected = ais_gdf.to_crs(metric_crs)

    # 4. Apply the correlation buffer to the AIS point shapes
    if not ais_projected.empty:
        ais_projected["geometry"] = ais_projected.geometry.buffer(CORRELATION_RADIUS_M)

    # 5. Spatial Join - Correlate detections with active AIS footprints
    # detections_projected is left df, ais_projected is right df
    if not ais_projected.empty:
        joined_gdf = gpd.sjoin(
            detections_projected, ais_projected, how="left", predicate="intersects"
        )
    else:
        # If no AIS points are in range, all detections are dark vessels
        joined_gdf = detections_projected.copy()
        joined_gdf["mmsi"] = None

    # Dark vessels are detections with no matching AIS records (NaN in joined columns)
    dark_mask = joined_gdf["mmsi"].isna()
    dark_vessels_projected = joined_gdf[dark_mask].copy()

    # Clean up joined index and telemetry columns to restore original schema
    original_cols = list(detections_gdf.columns)
    # Ensure 'geometry' remains in original columns list
    if "geometry" not in original_cols:
        original_cols.append("geometry")

    # Keep only the original columns to clean up the output schema
    dark_vessels_projected = dark_vessels_projected[original_cols]

    # 6. Reproject isolated Dark Vessels back to EPSG:4326 for standard GIS viewing
    dark_vessels_wgs84 = dark_vessels_projected.to_crs("EPSG:4326")

    return dark_vessels_wgs84


def main() -> None:
    # 1. Parse arguments and configuration
    args = parse_arguments()
    payload = get_payload(args)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 2. Retrieve file path pointers from Phase 3 payload
    detections_geojson = payload.get("detections_geojson")
    ais_telemetry = payload.get("ais_telemetry")
    acquisition_time = payload.get("acquisition_time", "")
    mission_id = payload.get("mission_id", "mission_default")

    if not detections_geojson or not ais_telemetry:
        raise ValueError(
            "Invalid payload: Missing detections_geojson "
            "or ais_telemetry path references."
        )

    # 3. Perform correlation and isolate Dark Vessels. The metric CRS is derived
    #    from the detections rather than taken from the payload, so the buffer
    #    distance is always applied in a zone that actually covers them.
    dark_vessels_gdf = correlate_targets(
        detections_geojson, ais_telemetry, acquisition_time
    )

    # 4. Save results to GeoJSON
    mission_dir = os.path.join(base_dir, "missions", mission_id)
    os.makedirs(mission_dir, exist_ok=True)

    output_geojson_path = os.path.join(mission_dir, "dark_vessels.geojson")

    # Geopandas handles file writing conversion
    dark_vessels_gdf.to_file(output_geojson_path, driver="GeoJSON")

    # 5. Update payload and print to stdout
    payload["dark_vessels_geojson"] = output_geojson_path

    # The dark ratio is the headline operational number, and it is only
    # meaningful alongside the AIS coverage that produced it — a 100% dark
    # mission with no telemetry is a coverage gap, not a finding.
    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        detections_count = len(gpd.read_file(detections_geojson))
        dark_count = len(dark_vessels_gdf)
        tracker.set_tags({"stage": "correlation"})
        tracker.log_params({"correlation_radius_m": CORRELATION_RADIUS_M})
        metrics: Dict[str, float] = {
            "dark_vessels": dark_count,
            "correlated_vessels": max(0, detections_count - dark_count),
        }
        if detections_count:
            metrics["dark_ratio"] = dark_count / detections_count
        tracker.log_metrics(metrics)
        tracker.log_artifact(output_geojson_path, "dark_vessels")
    finally:
        tracker.end()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
