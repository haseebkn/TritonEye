import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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
                    "surface": "water",
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
                    "surface": "water",
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


def test_correlation_retains_association_and_review_candidate(
    temp_files: tuple[str, str],
) -> None:
    path_det, path_ais = temp_files

    # Execute correlation agent logic
    # Acquisition time matches the AIS timestamp exactly
    targets = correlate_targets(
        path_det,
        path_ais,
        acquisition_time="2026-07-08 05:00:00",
        ais_coverage="partial",
    )

    assert len(targets) == 2
    assert list(targets.correlation_status) == [
        "ais_associated",
        "uncorrelated_candidate",
    ]
    assert targets.iloc[1].review_required
    assert not targets.operational_alert.any()


def _write_detections(
    path: Path, entries: List[Tuple[float, float, Optional[str]]]
) -> None:
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


def _write_empty_ais(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("mmsi,lat,lon,timestamp,speed_knots,course_deg\n")


def test_land_and_coastal_exclusions_are_retained_and_counted(tmp_path: Path) -> None:
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

    targets = correlate_targets(str(det), str(ais))
    assert list(targets.correlation_status) == [
        "unassessable",
        "excluded_land",
        "excluded_land",
        "excluded_coastal",
    ]
    assert not targets.review_required.any()
    from agents.correlation.correlation_agent import summarize_correlation

    summary = summarize_correlation(targets)
    assert summary["eligible_detections"] == 1
    assert summary["states"].get("ais_associated", 0) == 0


def test_missing_surface_is_unknown_and_ineligible(tmp_path: Path) -> None:
    """
    Missions lacking surface evidence remain auditable but cannot create alerts.
    """
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    _write_detections(det, [(-52.70, 48.00, None), (-52.71, 48.01, None)])
    _write_empty_ais(ais)

    targets = correlate_targets(str(det), str(ais))
    assert len(targets) == 2
    assert set(targets.correlation_status) == {"excluded_unknown_surface"}
    assert not targets.review_required.any()


def test_all_land_scene_yields_no_alerts(tmp_path: Path) -> None:
    """An entirely land-classified scene retains exclusions without alerts."""
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    _write_detections(det, [(-52.70, 48.00, "land"), (-52.71, 48.01, "land")])
    _write_empty_ais(ais)

    targets = correlate_targets(str(det), str(ais))
    assert len(targets) == 2
    assert not targets.operational_alert.any()
    assert not targets.review_required.any()


def test_out_of_region_return_cannot_be_review_candidate(tmp_path: Path) -> None:
    det, ais = tmp_path / "detections.geojson", tmp_path / "ais.csv"
    _write_detections(det, [(-70.0, 42.0, "water")])
    _write_empty_ais(ais)
    targets = correlate_targets(str(det), str(ais))
    assert targets.iloc[0].correlation_status == "excluded_outside_region"
    assert not targets.iloc[0].review_required


def test_competing_returns_remain_ambiguous(tmp_path: Path) -> None:
    det, ais = tmp_path / "detections.geojson", tmp_path / "ais.csv"
    _write_detections(det, [(-52.0, 47.5, "water"), (-52.001, 47.5, "water")])
    ais.write_text(
        "mmsi,lat,lon,timestamp\n316000001,47.5,-52.0,2026-07-08T05:00:00Z\n"
    )
    targets = correlate_targets(
        str(det), str(ais), "2026-07-08T05:00:00Z", ais_coverage="partial"
    )
    assert set(targets.correlation_status) == {"ambiguous_association"}
    assert targets.association_mmsi.notna().sum() == 1
    assert targets.review_required.all()
    assert not targets.operational_alert.any()
