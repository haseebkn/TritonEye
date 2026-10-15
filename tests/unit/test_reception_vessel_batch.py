"""Exercise usable reservations, source failures, split safety and empty batches."""

import json
from copy import deepcopy
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.cd_training_reservations import rules
from agents.reception_vessel_batch import (
    decide,
    public,
    shortlist,
    source_reasons,
    verify,
)
from agents.source_reservation import checked
from tests.unit.test_provisional_training import candidate


def source_policy() -> dict[str, Any]:
    return dict(json.loads(checked(rules()["source_policy"]).read_text()))


CELLS = [{"id": "west", "rank": 1, "geometry": mapping(box(-55.6, 50.1, -55.4, 50.3))}]
FROZEN: dict[str, Any] = {"priors": [], "exposed": [], "old": []}


def scene(number: int = 1, identity: str = "123456789") -> dict[str, Any]:
    product = deepcopy(candidate(number)["product"])
    product["Name"] = product["Name"].replace("S1A_", "S1C_")
    return {
        "product": product,
        "reference_midpoint_utc": "2022-05-01T10:00:12+00:00",
        "polarization": "HH/HV",
        "support": [
            {
                "mmsi": identity,
                "lon": -55.5,
                "lat": 50.2,
                "nearest_offset_s": 20,
                "bracket_gap_s": 40,
                "uncertainty_m": 150,
                "physical_surface": "water",
                "signed_shore_distance_m": 1500,
                "reports": [{"time": 1651399232, "values": {"speed_knots": 8}}],
            }
        ],
    }


def run(
    scenes: list[dict[str, Any]],
    frozen: dict[str, Any] | None = None,
    cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return decide(
        scenes,
        CELLS if cells is None else cells,
        frozen or FROZEN,
        source_policy(),
        rules(),
    )


def test_valid_source_reserves_before_pixels_and_keeps_full_competing_pool() -> None:
    s = scene()
    competitor = deepcopy(s["support"][0])
    competitor.update(mmsi="987654321", lon=-57)
    s["support"].append(competitor)
    result = run([s])
    area = result["areas"][0]
    assert area["split"] == "train" and area["label"] is None
    assert area["imagery_inspected"] is False and area["classifier_scored"] is False
    assert len(area["competing_identity_pool"]) == 2
    view = public(result, CELLS, FROZEN)
    assert view["reserved"] == 1 and view["labels_admitted"] == 0
    assert view["areas"][0]["native_input_verified"] is False
    assert view["promotion_eligible"] is False and view["training_approved"] is False
    # Intentional public output privacy contract, not source inspection.
    encoded = json.dumps(view)
    assert "123456789" not in encoded and "987654321" not in encoded


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"nearest_offset_s": 121}, "nearest_report_over_120s_or_missing"),
        ({"bracket_gap_s": None}, "bracket_over_240s_or_missing"),
        ({"bracket_gap_s": 241}, "bracket_over_240s_or_missing"),
        ({"bracket_gap_s": -1}, "bracket_over_240s_or_missing"),
        ({"uncertainty_m": float("nan")}, "uncertainty_over_500m_or_invalid"),
        ({"uncertainty_m": 501}, "uncertainty_over_500m_or_invalid"),
        (
            {"signed_shore_distance_m": 499},
            "reference_water_and_500m_shore_separation_unmet",
        ),
        (
            {"physical_surface": "land"},
            "reference_water_and_500m_shore_separation_unmet",
        ),
    ],
)
def test_failed_source_cannot_fill_batch(change: dict[str, Any], reason: str) -> None:
    s = scene()
    s["support"][0].update(change)
    result = run([s])
    assert result["areas"] == []
    assert result["source_rejections"][reason] == 1
    assert (
        public(result, CELLS, FROZEN)["status"]
        == "blocked_no_qualifying_source_overlap"
    )


@pytest.mark.parametrize("speed", [0, 0.8, float("nan"), 102.3, "unavailable"])
def test_slow_or_invalid_motion_is_not_relaxed(speed: Any) -> None:
    s = scene()
    s["support"][0]["reports"][0]["values"]["speed_knots"] = speed
    result = run([s])
    assert not result["areas"]
    assert result["source_rejections"]["motion_below_1kt_or_invalid"] == 1


@pytest.mark.parametrize("split", ["test", "validation", "quarantine"])
def test_protected_geography_cannot_become_training(split: str) -> None:
    prior = {
        "geometry": CELLS[0]["geometry"],
        "group": "held",
        "split": split,
        "quarantine": split == "quarantine",
    }
    assert not run([scene()], FROZEN | {"priors": [prior]})["areas"]


def test_remote_acquisition_role_exposure_and_missing_footprint_block() -> None:
    s = scene()
    group = "S1C_000001_01ABCD"
    remote = {
        "geometry": mapping(box(-53, 48, -52, 49)),
        "group": "held",
        "split": "validation",
        "quarantine": False,
        "acquisition_group": group,
    }
    assert not run([s], FROZEN | {"priors": [remote]})["areas"]
    assert not run([s], FROZEN | {"exposed": [group]})["areas"]
    s["product"]["GeoFootprint"] = mapping(box(-55.49, 50.2, -55.4, 50.3))
    assert not run([s])["areas"]


def test_same_day_acquisitions_remain_distinct_but_slices_and_identity_dedupe() -> None:
    a, b = scene(), scene(2, "987654321")
    alias = deepcopy(a)
    alias["product"]["Id"] = "e4f3f146-305f-4dcf-bd9e-000000000003"
    alias["product"]["Name"] = alias["product"]["Name"].replace(
        "1234.SAFE", "5678.SAFE"
    )
    result = run([a, b, alias])
    assert len(result["areas"]) == 2
    assert len({a["native_acquisition_group"] for a in result["areas"]}) == 2
    assert len({a["id"] for a in result["areas"]}) == 2
    assert len(run([a, scene(3)])["areas"]) == 1


def test_priority_is_frozen_and_whole_cell_and_envelope_roles_are_separate() -> None:
    cells = CELLS + [
        {"id": "second", "rank": 2, "geometry": mapping(box(-57, 50.1, -56.8, 50.3))}
    ]
    second = scene(2, "987654321")
    second["support"][0]["lon"] = -56.9
    result = run([second, scene()], cells=cells)
    assert [a["source_cell_id"] for a in result["areas"]] == ["west", "second"]
    assert run([scene()], cells=[])["areas"] == []
    assert source_reasons(scene()["support"][0], 1651399212, source_policy()) == []
    report = {
        "cells": [{"id": "a", "geometry": CELLS[0]["geometry"]}],
        "shortlist": [{"rank": 1, "cell_id": "a"}],
    }
    assert shortlist(report)[0]["id"] == "a"


def test_public_release_replays() -> None:
    result = verify()
    assert result["verified"] is True and result["labels_admitted"] == 0
