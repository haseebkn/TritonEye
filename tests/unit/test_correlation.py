import json
import os
import sys
import tempfile
from typing import Iterator

import pytest

# Align python path to workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.correlation.correlation_agent import correlate_targets


@pytest.fixture
def temp_files() -> Iterator[tuple[str, str]]:
    # 1. Create a mock detections GeoJSON containing 2 features (in EPSG:4326)
    # Target 1: Near center_lon/center_lat
    # Target 2: Far away (will be dark)
    detections = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": 0,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-52.0001, 47.5001],
                            [-51.9999, 47.5001],
                            [-51.9999, 47.5003],
                            [-52.0001, 47.5003],
                            [-52.0001, 47.5001],
                        ]
                    ],
                },
                "properties": {
                    "target_id": "TRITON-001",
                    "class_name": "cargo",
                    "confidence": 0.95,
                },
            },
            {
                "type": "Feature",
                "id": 1,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-52.5000, 47.8000],
                            [-52.4990, 47.8000],
                            [-52.4990, 47.8010],
                            [-52.5000, 47.8010],
                            [-52.5000, 47.8000],
                        ]
                    ],
                },
                "properties": {
                    "target_id": "TRITON-002",
                    "class_name": "cargo",
                    "confidence": 0.88,
                },
            },
        ],
    }

    # 2. Create a mock AIS CSV
    # One active transponder coordinate matches Target 1 exactly at t_img
    ais_data = (
        "mmsi,lat,lon,timestamp,speed_knots,course_deg\n"
        "316000001,47.5002,-52.0000,2026-07-08 05:00:00,10.0,90.0\n"
    )

    fd_det, path_det = tempfile.mkstemp(suffix=".geojson")
    with os.fdopen(fd_det, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    fd_ais, path_ais = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd_ais, "w", encoding="utf-8") as f:
        f.write(ais_data)

    yield path_det, path_ais

    os.remove(path_det)
    os.remove(path_ais)


def test_correlation_isolates_dark_vessel(temp_files: tuple[str, str]) -> None:
    path_det, path_ais = temp_files

    # Execute correlation agent logic
    # Acquisition time matches the AIS timestamp exactly
    dark_vessels_gdf = correlate_targets(
        path_det, path_ais, acquisition_time="2026-07-08 05:00:00"
    )

    # Assert TRITON-001 matches cooperative AIS and is excluded
    # Assert TRITON-002 has no AIS match and is isolated as a Dark Vessel
    assert len(dark_vessels_gdf) == 1
    assert dark_vessels_gdf.iloc[0]["target_id"] == "TRITON-002"


def _write_detections(path, entries):
    """entries: list of (lon, lat, surface_or_None)."""
    feats = []
    for i, (lon, lat, surface) in enumerate(entries):
        d = 0.001
        ring = [
            [lon - d, lat - d],
            [lon + d, lat - d],
            [lon + d, lat + d],
            [lon - d, lat + d],
            [lon - d, lat - d],
        ]
        props = {"target_id": f"TRITON-{i:03d}", "confidence": 0.5}
        if surface is not None:
            props["surface"] = surface
        feats.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def _write_empty_ais(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("mmsi,lat,lon,timestamp,speed_knots,course_deg\n")


def test_land_detections_do_not_become_dark_vessels(tmp_path):
    """
    A land return has no AIS transmitter and would otherwise satisfy the
    dark-vessel definition perfectly. Only water may raise an alert.
    """
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    _write_detections(
        det,
        [
            (-52.70, 48.00, "water"),
            (-52.71, 48.01, "land"),
            (-52.72, 48.02, "land"),
            (-52.73, 48.03, "coastal"),
        ],
    )
    _write_empty_ais(ais)

    dark = correlate_targets(str(det), str(ais))
    assert len(dark) == 1
    assert dark.iloc[0]["surface"] == "water"


def test_detections_without_surface_property_still_correlate(tmp_path):
    """
    Missions predating the land mask carry no `surface` field. They must keep
    correlating exactly as before rather than being silently dropped.
    """
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    _write_detections(det, [(-52.70, 48.00, None), (-52.71, 48.01, None)])
    _write_empty_ais(ais)

    dark = correlate_targets(str(det), str(ais))
    assert len(dark) == 2


def test_all_land_scene_yields_no_alerts(tmp_path):
    """An entirely land-classified scene must return empty, not crash."""
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    _write_detections(det, [(-52.70, 48.00, "land"), (-52.71, 48.01, "land")])
    _write_empty_ais(ais)

    dark = correlate_targets(str(det), str(ais))
    assert len(dark) == 0
