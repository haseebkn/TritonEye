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
def temp_files_temporal() -> Iterator[tuple[str, str]]:
    # Detections GeoJSON (Target 1 at center_lon/center_lat)
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
                    "surface": "water",
                },
            }
        ],
    }

    # AIS track contains coordinate matches but only at out-of-bounds timestamps:
    # 2026-07-08 04:30:00 (30 minutes prior to acquisition time)
    ais_data = (
        "mmsi,lat,lon,timestamp,speed_knots,course_deg\n"
        "316000001,47.5002,-52.0000,2026-07-08 04:30:00,10.0,90.0\n"
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


def test_correlation_filters_out_of_time_telemetry(
    temp_files_temporal: tuple[str, str],
) -> None:
    path_det, path_ais = temp_files_temporal

    # Execute correlation agent logic
    # Acquisition time is 05:00:00. The AIS data is at 04:30:00.
    # The ±5m window should drop the out-of-time telemetry record.
    # Stale AIS is missing evidence, never proof of a dark vessel.
    targets = correlate_targets(
        path_det,
        path_ais,
        acquisition_time="2026-07-08 05:00:00",
        ais_coverage="partial",
    )

    assert len(targets) == 1
    assert targets.iloc[0]["correlation_status"] == "unassessable"
    assert not targets.iloc[0]["review_required"]
    assert not targets.iloc[0]["operational_alert"]
