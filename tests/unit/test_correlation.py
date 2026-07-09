import sys
import os
import json
import tempfile
import pytest
import geopandas as gpd

# Align python path to workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.correlation.correlation_agent import correlate_targets


@pytest.fixture
def temp_files() -> tuple[str, str]:
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
                    "coordinates": [[
                        [-52.0001, 47.5001],
                        [-51.9999, 47.5001],
                        [-51.9999, 47.5003],
                        [-52.0001, 47.5003],
                        [-52.0001, 47.5001]
                    ]]
                },
                "properties": {
                    "target_id": "TRITON-001",
                    "class_name": "cargo",
                    "confidence": 0.95
                }
            },
            {
                "type": "Feature",
                "id": 1,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-52.5000, 47.8000],
                        [-52.4990, 47.8000],
                        [-52.4990, 47.8010],
                        [-52.5000, 47.8010],
                        [-52.5000, 47.8000]
                    ]]
                },
                "properties": {
                    "target_id": "TRITON-002",
                    "class_name": "cargo",
                    "confidence": 0.88
                }
            }
        ]
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
