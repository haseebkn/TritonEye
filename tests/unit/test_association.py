"""Behavioral evidence for bounded AIS alignment and non-inflating assignment."""

import os
import sys

import pandas as pd
import pytest
from shapely.geometry import Point

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.association import GEOD, aligned_ais, one_to_one_matches  # noqa: E402

ACQ = "2026-08-18T09:15:00Z"


def test_timezone_offsets_represent_the_same_instant() -> None:
    records = pd.DataFrame(
        [
            {
                "mmsi": 316000001,
                "lon": -52.0,
                "lat": 48.0,
                "timestamp": "2026-08-18T06:45:00-02:30",
            },
        ]
    )
    positions, _ = aligned_ais(records, ACQ)
    assert positions.iloc[0].alignment_method == "observed"
    assert positions.iloc[0].report_age_s == 0


def test_interpolation_and_propagation_align_motion() -> None:
    east, north, _ = GEOD.fwd(-52.0, 48.0, 90, 1200)
    records = pd.DataFrame(
        [
            {
                "mmsi": 316000001,
                "lon": -52.0,
                "lat": 48.0,
                "timestamp": "2026-08-18T09:13:00Z",
            },
            {
                "mmsi": 316000001,
                "lon": east,
                "lat": north,
                "timestamp": "2026-08-18T09:17:00Z",
            },
            {
                "mmsi": 316000002,
                "lon": -52.0,
                "lat": 48.0,
                "timestamp": "2026-08-18T09:13:00Z",
                "speed_knots": 5 / (1852 / 3600),
                "course_deg": 90,
            },
        ]
    )
    positions, _ = aligned_ais(records, ACQ)
    assert list(positions.alignment_method) == ["interpolated", "velocity_propagated"]
    for point in positions.geometry:
        _, _, distance = GEOD.inv(-52.0, 48.0, point.x, point.y)
        assert distance == pytest.approx(600, abs=0.1)


def test_stale_invalid_sentinel_and_conflicting_fixes_are_rejected() -> None:
    records = pd.DataFrame(
        [
            {
                "mmsi": 316000001,
                "lon": -52,
                "lat": 48,
                "timestamp": "2026-08-18T08:00:00Z",
            },
            {"mmsi": 316000002, "lon": 181, "lat": 91, "timestamp": ACQ},
            {"mmsi": 0, "lon": -52, "lat": 48, "timestamp": ACQ},
            {"mmsi": 316000003, "lon": -52, "lat": 48, "timestamp": ACQ},
            {"mmsi": 316000003, "lon": -53, "lat": 48, "timestamp": ACQ},
            {
                "mmsi": 316000004,
                "lon": -52,
                "lat": 48,
                "timestamp": "2026-08-18T09:13:00Z",
                "speed_knots": 102.3,
                "course_deg": 360,
            },
        ]
    )
    positions, stats = aligned_ais(records, ACQ)
    assert positions.empty
    assert stats["invalid_records"] == 2
    assert stats["stale_records"] == 1
    assert stats["rejected_tracks"] == 2


def test_missing_acquisition_time_never_assumes_now() -> None:
    positions, stats = aligned_ais(pd.DataFrame(), "")
    assert positions.empty
    assert "acquisition" in stats["reason"]


def test_one_return_cannot_match_two_vessels() -> None:
    selected, alternatives, ambiguous = one_to_one_matches(
        [Point(-52, 48)], [Point(-52, 48), Point(-52.001, 48)], 500
    )
    assert len(selected) == 1
    assert len(alternatives[0]) == 2
    assert ambiguous == {0}


def test_maximum_cardinality_assignment_beats_greedy_nearest() -> None:
    # A sees both vessels; B sees only vessel 1. Nearest-first would waste A on 1.
    def east(metres: float) -> Point:
        lon, lat, _ = GEOD.fwd(-52, 48, 90, metres)
        return Point(lon, lat)

    selected, _, _ = one_to_one_matches(
        [east(100), east(-200)], [east(0), east(400)], 350
    )
    assert selected[0][0] == 1
    assert selected[1][0] == 0
