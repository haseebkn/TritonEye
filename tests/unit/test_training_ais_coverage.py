"""Observed reception, boundary motion, acquisition identity and missing support."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.artifacts import REPO_ROOT
from agents.training_ais_coverage import analyse, reception, verify

BASE = datetime(2026, 10, 1, 12, tzinfo=timezone.utc).timestamp()


def report(
    offset: float, lon: float = -55.5, mmsi: str = "123456789"
) -> dict[str, Any]:
    stamp = datetime.fromtimestamp(BASE + offset, timezone.utc).isoformat()
    values = {
        "mmsi": mmsi,
        "lon": lon,
        "lat": 50.2,
        "timestamp": stamp,
        "speed_knots": 5,
        "course_deg": 90,
        "length": 100,
    }
    return {
        "time": BASE + offset,
        "timestamp": stamp,
        "mmsi": mmsi,
        "lon": lon,
        "lat": 50.2,
        "values": values,
    }


def product() -> dict[str, Any]:
    return {
        "Id": "e4f3f146-305f-4dcf-bd9e-000000000001",
        "Name": "S1C_IW_GRDH_1SDH_20261001T115950_20261001T120010_"
        "009500_012E71_1234.SAFE",
        "ContentDate": {"Start": "2026-10-01T11:59:50Z", "End": "2026-10-01T12:00:10Z"},
        "GeoFootprint": mapping(box(-56, 50, -55, 51)),
    }


def run(
    rows: list[dict[str, Any]],
    products: list[dict[str, Any]],
    priors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    regions = [
        {
            "id": "sample",
            "group": "new_cell_-56_50",
            "geometry": mapping(box(-55.51, 50.19, -55.49, 50.21)),
        }
    ]
    return dict(
        analyse(
            rows,
            products,
            regions,
            {"priors": priors or [], "exposed": []},
            "2026-10-01T13:00:00Z",
        )["regions"][0]
    )


def test_silence_includes_boundaries_and_is_not_complete_reception() -> None:
    result = reception(
        [report(-100), report(100)], BASE - 1000, BASE + 2000, BASE + 3000
    )
    assert result["reports"] == 2
    assert result["unique_identities"] == 1
    assert result["longest_silent_interval_s_including_edges"] == 1900
    assert result["last_report_age_s"] == 2900
    assert result["coverage_complete"] is False
    empty = reception([], BASE, BASE + 1000, BASE + 2000)
    assert empty["longest_silent_interval_s_including_edges"] == 1000
    assert empty["last_report_age_s"] is None


def test_interpolate_before_spatial_filtering_and_remove_cog_alias() -> None:
    rows = [report(-60, -55.515), report(60, -55.485)]
    alias = product()
    alias["Name"] = alias["Name"].replace(".SAFE", "_COG.SAFE")
    alias["Id"] = "e4f3f146-305f-4dcf-bd9e-000000000002"
    result = run(rows, [product(), alias])
    assert result["reception"]["reports"] == 0
    assert result["cd_hh_hv_physical_slices"] == 1
    assert result["slices_with_tight_track_support"] == 1
    assert result["new_eligible_slices_with_tight_support"] == 1
    assert result["passes"][0]["label_admitted"] is False


def test_missing_swath_out_of_swath_stale_and_conflicting_tracks() -> None:
    missing = product()
    del missing["GeoFootprint"]
    assert run([report(-60), report(60)], [missing])["cd_hh_hv_physical_slices"] == 0
    elsewhere = product() | {"GeoFootprint": mapping(box(-54, 50, -53, 51))}
    assert (
        run([report(-60), report(60)], [elsewhere])["slices_with_tight_track_support"]
        == 0
    )
    stale = run([report(-1000), report(1000)], [product()])
    assert stale["slices_with_tight_track_support"] == 0
    conflict = run([report(-60), report(-60, -55.52), report(60)], [product()])
    assert conflict["slices_with_tight_track_support"] == 0


def test_validation_acquisition_cannot_become_training_opportunity() -> None:
    prior = {
        "geometry": mapping(box(-53, 48, -52, 49)),
        "group": "other",
        "split": "validation",
        "quarantine": False,
        "acquisition_group": "S1C_009500_012E71",
    }
    result = run([report(-60), report(60)], [product()], [prior])
    assert result["slices_with_tight_track_support"] == 1
    assert result["new_eligible_slices_with_tight_support"] == 0


def test_two_acquisitions_same_day_remain_distinct() -> None:
    second = deepcopy(product())
    second["Id"] = "e4f3f146-305f-4dcf-bd9e-000000000003"
    second["Name"] = second["Name"].replace("009500", "009501")
    result = run([report(-60), report(60)], [product(), second])
    assert result["cd_hh_hv_physical_slices"] == 2
    assert result["cd_hh_hv_native_acquisition_groups"] == 2


def test_adjacent_slices_are_one_native_group() -> None:
    adjacent = deepcopy(product())
    adjacent["Id"] = "e4f3f146-305f-4dcf-bd9e-000000000004"
    adjacent["Name"] = adjacent["Name"].replace(
        "115950_20261001T120010", "120010_20261001T120030"
    )
    adjacent["ContentDate"] = {
        "Start": "2026-10-01T12:00:10Z",
        "End": "2026-10-01T12:00:30Z",
    }
    result = run([report(-60), report(60)], [product(), adjacent])
    assert result["cd_hh_hv_physical_slices"] == 2
    assert result["cd_hh_hv_native_acquisition_groups"] == 1


def test_committed_coverage_evidence_is_replayable_publicly() -> None:
    assert verify() == {"verified": True, "local": False, "labels_admitted": 0}


def test_frozen_real_coverage_replay_when_available() -> None:
    if not (REPO_ROOT / "data/research/training_ais_coverage_v010_01").exists():
        pytest.skip("Requires retained AIS snapshot and catalogue metadata")
    assert verify(local=True)["verified"] is True
