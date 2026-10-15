"""Reception is evidence of a feed, never truth or permission to reuse splits."""

from copy import deepcopy
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.nl_iceberg_search import anchors, requests_for
from agents.nl_iceberg_search import verify as verify_icebergs
from agents.nl_reception import analyse, geojson, render, role
from agents.nl_reception import verify as verify_reception
from tests.unit.test_training_ais_coverage import product, report

RULES = {
    "grid_degrees": 0.1,
    "minimum_days": 2,
    "minimum_identities": 2,
    "maximum_shortlist": 8,
    "half_width_m": 3000,
    "minimum_speed_knots": 1,
    "maximum_uncertainty_m": 500,
    "limits": [],
}
FROZEN: dict[str, Any] = {"priors": [], "exposed": [], "old": []}


def rows() -> list[dict[str, Any]]:
    return [report(-86400, -55.55), report(0, -55.55, "987654321")]


def scene() -> dict[str, Any]:
    return {
        "product": product() | {"Online": True},
        "polarization": "HH/HV",
        "reference_midpoint_utc": "2026-10-01T12:00:00+00:00",
        "support": [
            {"lon": -55.55, "lat": 50.25, "close_120s": True, "bracket_240s": True}
        ],
    }


def run(
    scenes: list[dict[str, Any]], frozen: dict[str, Any] | None = None
) -> dict[str, Any]:
    return analyse(rows(), scenes, frozen or FROZEN, RULES, "2026-10-02T12:00:00+00:00")


def prior(split: str, geometry: Any = None) -> dict[str, Any]:
    return {
        "geometry": mapping(
            geometry if geometry is not None else box(-55.6, 50.2, -55.5, 50.3)
        ),
        "group": "held_area",
        "split": split,
        "quarantine": split == "quarantine",
        "acquisition_group": "S1C_009500_012E71",
    }


def test_reception_shortlist_is_not_label_admission() -> None:
    result = run([scene()])
    assert result["nl_reports"] == 2
    assert result["shortlist"][0]["eligible_native_groups"] == 1
    assert result["shortlist"][0]["reserved"] is False
    assert result["labels_admitted"] == 0
    assert result["coverage_complete"] is False
    feature = geojson(result)["features"][0]
    assert feature["properties"]["reception"]["unique_identities"] == 2
    assert "mmsi" not in feature["properties"]
    assert "tight_hh_hv_opportunities" not in feature["properties"]
    assert "NL observed AIS reception" in render(result)


@pytest.mark.parametrize("split", ["test", "validation", "quarantine"])
def test_geography_cannot_become_training(split: str) -> None:
    assert run([scene()], FROZEN | {"priors": [prior(split)]})["shortlist"] == []


def test_remote_validation_acquisition_blocks_support_but_not_reception() -> None:
    frozen = FROZEN | {"priors": [prior("validation", box(-53, 48, -52, 49))]}
    result = run([scene()], frozen)
    assert result["shortlist"][0]["status"] == "reception_only_no_eligible_sar_overlap"
    assert result["shortlist"][0]["eligible_native_groups"] == 0


def test_exposed_and_offline_acquisitions_cannot_raise_rank() -> None:
    assert (
        run([scene()], FROZEN | {"exposed": ["S1C_009500_012E71"]})["shortlist"][0][
            "eligible_native_groups"
        ]
        == 0
    )
    s = scene()
    s["product"]["Online"] = False
    assert run([s])["shortlist"][0]["eligible_native_groups"] == 0


def test_stationary_timing_lead_is_not_motion_qualified() -> None:
    s = scene()
    r = report(0, -55.55)
    r["values"]["speed_knots"] = 0.8
    s["support"][0].update(reports=[r], uncertainty_m=200)
    result = run([s])
    assert result["shortlist"][0]["eligible_native_groups"] == 1
    assert result["motion_qualified_training_groups"] == 0
    r["values"]["speed_knots"] = 5
    assert run([s])["motion_qualified_training_groups"] == 1
    s["support"][0]["uncertainty_m"] = 501
    assert run([s])["motion_qualified_training_groups"] == 0


def test_stale_or_out_of_swath_support_does_not_make_opportunity() -> None:
    s = scene()
    s["support"][0]["bracket_240s"] = False
    assert run([s])["shortlist"][0]["eligible_native_groups"] == 0
    s = scene()
    s["product"]["GeoFootprint"] = mapping(box(-55.54, 50.2, -55.5, 50.3))
    assert run([s])["shortlist"][0]["eligible_native_groups"] == 0


def test_interpolation_only_cell_and_adjacent_slice_grouping() -> None:
    s = scene()
    s["support"][0]["lon"] = -55.75
    a = deepcopy(s)
    a["product"]["Id"] = "e4f3f146-305f-4dcf-bd9e-000000000002"
    result = run([s, a])
    assert result["interpolation_only_cells"] == 1
    assert result["split_safe_observed_cells"] == 1
    aligned = next(c for c in result["cells"] if c["reception"]["reports"] == 0)
    assert aligned["eligible_native_groups"] == 1
    assert aligned["reception_screen_passed"] is False


def test_nl_boundary_and_ten_kilometre_protected_margin() -> None:
    assert role(mapping(box(-75, 40, -74, 41)), FROZEN, "new")["split_safe"] is False
    frozen = FROZEN | {"priors": [prior("test", box(-55.6, 50.3, -55.5, 50.4))]}
    assert (
        role(mapping(box(-55.6, 50.25, -55.5, 50.26)), frozen, "new")["split_safe"]
        is False
    )


def test_public_reception_evidence_replays() -> None:
    assert verify_reception()["labels_admitted"] == 0


def test_public_iceberg_evidence_replays() -> None:
    assert verify_icebergs() == {
        "verified": True,
        "local": False,
        "anchors": 6,
        "labels_admitted": 0,
    }


ICE_RULES: dict[str, Any] = {
    "minimum_year": 2018,
    "sizes": ["MED", "LG", "VLG"],
    "maximum_sar_offset_s": 900,
    "half_width_m": 3000,
    "maximum_per_geographic_group": 2,
    "maximum_anchors": 6,
    "optical_half_window_s": 3600,
    "position_allowance_m": 300,
    "drift_discovery_m_per_s": 1,
}


def observation() -> dict[str, Any]:
    return {
        "iceberg_id": "2020:1",
        "timestamp": "2020-05-01T12:00:00+00:00",
        "method": "VIS",
        "source": "AIRCRAFT",
        "size": "LG",
        "lon": -55.55,
        "lat": 50.25,
    }


def test_source_first_optical_search_does_not_infer_sar_overlap() -> None:
    selected = anchors(
        {"ranked": []}, FROZEN, ICE_RULES, {"observations": [observation()]}
    )
    a = selected["anchors"][0]
    assert a["product_id"] is None
    assert a["label"] is None
    assert a["training_approved"] is False
    params, stac, target = requests_for(a, ICE_RULES)
    assert stac["datetime"] == "2020-05-01T11:00:00Z/2020-05-01T13:00:00Z"
    assert "SENTINEL-2" in params["$filter"]
    assert target["lon"] == observation()["lon"]


@pytest.mark.parametrize(
    "change", [{"source": "S1D"}, {"method": "RAD"}, {"size": "SM"}]
)
def test_satellite_derived_or_unsupported_sources_are_excluded(
    change: dict[str, str],
) -> None:
    assert (
        anchors(
            {"ranked": []},
            FROZEN,
            ICE_RULES,
            {"observations": [observation() | change]},
        )["anchors"]
        == []
    )


def test_repeated_iceberg_and_protected_observation_are_not_new_examples() -> None:
    obs = observation()
    selected = anchors({"ranked": []}, FROZEN, ICE_RULES, {"observations": [obs, obs]})
    assert len(selected["anchors"]) == 1
    frozen = FROZEN | {"priors": [prior("test")]}
    assert (
        anchors({"ranked": []}, frozen, ICE_RULES, {"observations": [obs]})["anchors"]
        == []
    )
    frozen = FROZEN | {
        "old": [{"stream": "iceberg_support", "source_identity": obs["iceberg_id"]}]
    }
    assert (
        anchors({"ranked": []}, frozen, ICE_RULES, {"observations": [obs]})["anchors"]
        == []
    )
