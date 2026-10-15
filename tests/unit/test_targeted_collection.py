"""Future pass selection is metadata-only and retains scientific guardrails."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.targeted_collection import plans, publish, reception, select, verify

NOW = datetime(2026, 10, 15, 15, tzinfo=timezone.utc)
CELL = {
    "id": "cell_-556_501",
    "grid": [-556, 501],
    "rank": 1,
    "geometry": mapping(box(-55.6, 50.1, -55.5, 50.2)),
    "recent_moving_reports": 2,
}
FROZEN: dict[str, Any] = {"priors": [], "exposed": []}


def segment(day: int = 17) -> dict[str, Any]:
    return {
        "planned_native_group": f"S1C_009923_{day:06X}",
        "mission": "S1C",
        "mode": "IW",
        "polarization": "HH/HV",
        "start_utc": f"2026-10-{day:02d}T09:00:00+00:00",
        "end_utc": f"2026-10-{day:02d}T09:10:00+00:00",
        "relative_orbit": 25,
        "planned_footprint": mapping(box(-56, 50, -55, 51)),
    }


def test_next_three_are_distinct_future_passes_with_no_labels_or_product_ids() -> None:
    result = select(
        [segment(20), segment(18), segment(17), segment(17), segment(19)],
        [CELL],
        FROZEN,
        NOW,
    )
    assert result["reserved"] == 3
    assert [p["start_utc"][8:10] for p in result["passes"]] == ["17", "18", "19"]
    for p in result["passes"]:
        assert p["label"] is None and p["actual_product_id"] is None
        assert p["split"] == "train" and p["provisional"] is True
        assert p["promotion_eligible"] is False
        assert p["classifier_scored"] is False
        assert p["collection_start_utc"][11:16] == "08:30"
        assert p["collection_end_utc"][11:16] == "09:40"


@pytest.mark.parametrize(
    "split,quarantine", [("validation", False), ("test", False), ("train", True)]
)
def test_protected_geography_cannot_be_selected(split: str, quarantine: bool) -> None:
    prior = {
        "geometry": CELL["geometry"],
        "group": "protected",
        "split": split,
        "quarantine": quarantine,
    }
    assert (
        select([segment()], [CELL], {"priors": [prior], "exposed": []}, NOW)["reserved"]
        == 0
    )


def test_acquisition_role_exposure_and_partial_footprint_are_rejected() -> None:
    s = segment()
    prior = {
        "geometry": mapping(box(-60, 54, -59.9, 54.1)),
        "group": "elsewhere",
        "split": "validation",
        "quarantine": False,
        "acquisition_group": s["planned_native_group"],
    }
    assert select([s], [CELL], {"priors": [prior], "exposed": []}, NOW)["reserved"] == 0
    assert (
        select(
            [s], [CELL], {"priors": [], "exposed": [s["planned_native_group"]]}, NOW
        )["reserved"]
        == 0
    )
    s["planned_footprint"] = mapping(box(-55.55, 50.1, -55.5, 50.2))
    assert select([s], [CELL], FROZEN, NOW)["reserved"] == 0


def test_past_pass_silent_cell_and_conflicting_plans_do_not_fill_quota() -> None:
    assert select([segment(14)], [CELL], FROZEN, NOW)["reserved"] == 0
    assert (
        select([segment()], [CELL | {"recent_moving_reports": 0}], FROZEN, NOW)[
            "reserved"
        ]
        == 0
    )
    conflict = segment() | {"relative_orbit": 99}
    with pytest.raises(ValueError, match="Conflicting"):
        select([segment(), conflict], [CELL], FROZEN, NOW)


def test_reception_requires_recent_finite_motion_and_preserves_privacy() -> None:
    row = {
        "time": NOW.timestamp() - 60,
        "timestamp": "2026-10-15T14:59:00+00:00",
        "mmsi": "123456789",
        "lon": -55.55,
        "lat": 50.15,
        "values": {"speed_knots": "8"},
    }
    rows = [row]
    for speed in ["nan", "0", "102.3", None]:
        rows.append(row | {"values": {"speed_knots": speed}})
    rows.extend(
        [
            row | {"time": NOW.timestamp() + 60},
            row | {"time": NOW.timestamp() - 8 * 86400},
        ]
    )
    result = reception(rows, [CELL], NOW)[0]
    assert result["recent_moving_reports"] == 1
    assert result["recent_moving_identities"] == 1
    assert "mmsi" not in result


def test_real_kml_structure_timezone_and_hex_identity(tmp_path: Path) -> None:
    fields = {
        "SatelliteId": "S1D",
        "DatatakeId": "9965",
        "Mode": "IW",
        "Polarisation": "DH",
        "ObservationTimeStart": "2026-10-18T09:50:59",
        "ObservationTimeStop": "2026-10-18T09:58:48",
        "OrbitAbsolute": "5068",
        "OrbitRelative": "127",
    }
    data = "".join(
        f'<Data name="{k}"><value>{v}</value></Data>' for k, v in fields.items()
    )
    xml = (
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        f"<ExtendedData>{data}</ExtendedData><LinearRing><coordinates>"
        "-56,50,0 -55,50,0 -55,51,0 -56,51,0 -56,50,0"
        "</coordinates></LinearRing></Placemark></Document></kml>"
    )
    path = tmp_path / "plan.kml"
    path.write_text(xml)
    p = plans(path)[0]
    assert p["planned_native_group"] == "S1D_005068_009965"
    assert p["start_utc"] == "2026-10-18T09:50:59+00:00"
    assert p["polarization"] == "HH/HV"
    path.write_text(xml.replace("<value>DH</value>", "<value>DV</value>"))
    assert plans(path) == []
    path.write_text(xml.replace("09:58:48", "09:00:00"))
    with pytest.raises(ValueError, match="Invalid plan"):
        plans(path)


def test_selection_does_not_mutate_frozen_priors() -> None:
    frozen = deepcopy(FROZEN)
    select([segment()], [CELL], frozen, NOW)
    assert frozen == FROZEN


def test_published_collection_contract_and_overwrite_guard(tmp_path: Path) -> None:
    result = verify()
    assert result == {
        "verified": True,
        "local": False,
        "reserved": 3,
        "labels_admitted": 0,
    }
    with pytest.raises(ValueError, match="Fresh version"):
        publish(tmp_path)
