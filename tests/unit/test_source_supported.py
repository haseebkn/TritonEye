"""Exercise source parsing, exact footprints, evidence dependence and selection."""

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.artifacts import REPO_ROOT
from agents.source_reservation import (
    assign,
    envelope,
    reserve,
    validate_manifest,
    verify,
    vessel_opportunities,
)
from agents.source_supported import (
    CATALOGUE,
    POLICY,
    iip_overlaps,
    query_year,
    read_iip,
)


def policy() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(POLICY.read_text())
    return value


def product() -> dict[str, Any]:
    return {
        "Id": "e4f3f146-305f-4dcf-bd9e-98d6d3569be8",
        "Name": "S1A_IW_GRDH_1SDH_20170601T100000_20170601T100025_"
        "016000_01ABCD_1234.SAFE",
        "ContentDate": {"Start": "2017-06-01T10:00:00Z", "End": "2017-06-01T10:00:25Z"},
        "GeoFootprint": mapping(box(-56, 46, -52, 50)),
        "Online": True,
    }


def observation(**overrides: Any) -> dict[str, Any]:
    from datetime import datetime

    row = {
        "iceberg_id": "2017:1",
        "timestamp": "2017-06-01T10:01:00+00:00",
        "time": datetime.fromisoformat("2017-06-01T10:01:00+00:00").timestamp(),
        "lon": -53.2,
        "lat": 48.3,
        "method": "VIS",
        "source": "SHIP",
    }
    return {**row, **overrides}


def test_iip_header_variants_duplicates_and_bad_times(tmp_path: Path) -> None:
    path = tmp_path / "IIP_2018IcebergSeason.csv"
    fields = [
        "ICEBERG YEAR",
        "ICEBERG NUMBER",
        "SIGHTING\nDATE",
        "SIGHTING\nTIME",
        "SIGHTING\nLATITUDE",
        "SIGHTING\nLONGITUDE",
        "SIGHTING\nMETHOD",
        "SIZE",
        "SHAPE",
        "SOURCE",
    ]
    row = [2018, 1, "6/1/2018", "940", 48.3, -53.2, "VIS", "SM", "GEN", "SHIP"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(
            [row, row, [*row[:3], "2460", *row[4:]], [*row[:5], 53.2, *row[6:]]]
        )
    rows, counts = read_iip(tmp_path)
    assert len(rows) == 1 and counts["duplicate_sightings"] == 1
    assert counts["invalid_rows"] == 2
    assert rows[0]["timestamp"] == "2018-06-01T09:40:00+00:00"
    assert rows[0]["label"] is None
    assert rows[0]["source_line"] > 2


def test_iip_excludes_circular_source_and_out_of_swath() -> None:
    rows = [
        observation(),
        observation(iceberg_id="2017:2", method="R/V", source="SN1A"),
        observation(iceberg_id="2017:3", lon=-58),
    ]
    ranked, counts = iip_overlaps([product()], rows, policy())
    assert len(ranked) == 1
    assert counts["visual_method_with_satellite_source_excluded"] == 1
    assert ranked[0]["absolute_offset_s"] == 47.5
    assert ranked[0]["temporal_eligible"] is True
    assert ranked[0]["label"] is None
    broken = deepcopy(product())
    broken.pop("GeoFootprint")
    assert iip_overlaps([broken], rows, policy())[0] == []


def test_two_hour_screen_is_not_thirty_minute_reservation() -> None:
    row = observation(time=observation()["time"] + 3600)
    ranked, _ = iip_overlaps([product()], [row], policy())
    assert len(ranked) == 1 and not ranked[0]["temporal_eligible"]
    selected, _ = reserve(ranked, [], [], set(), policy())
    assert not selected


def test_assignment_retains_validation_and_refuses_locked_quarantine() -> None:
    geometry = envelope(-53.2, 48.3, 3000)
    prior = {
        "geometry": geometry,
        "group": "prior",
        "split": "validation",
        "quarantine": False,
    }
    assert assign(geometry, [prior]) == ("validation", "prior")
    for changed in ({**prior, "split": "test"}, {**prior, "quarantine": True}):
        with pytest.raises(ValueError, match="Locked/quarantine"):
            assign(geometry, [changed])
    with pytest.raises(ValueError, match="Conflicting"):
        assign(geometry, [prior, {**prior, "split": "train"}])


def test_reservation_deduplicates_identity_and_keeps_null_labels() -> None:
    ranked, _ = iip_overlaps([product()], [observation(), observation()], policy())
    selected, excluded = reserve(ranked, [], [], set(), policy())
    assert len(selected) == 1
    assert any(r["reason"] == "Repeated source identity" for r in excluded)
    assert selected[0]["label"] is None and selected[0]["ignore_for_accuracy"]
    assert not selected[0]["classifier_scored"]
    assert not selected[0]["promotion_eligible"]


def test_native_validation_role_cannot_be_reassigned_to_training() -> None:
    geometry = envelope(-53.2, 48.3, 3000)
    prior = {
        "geometry": envelope(-56, 50, 3000),
        "group": "distant",
        "split": "validation",
        "quarantine": False,
        "acquisition_group": "known",
    }
    assert assign(geometry, [prior], "known")[0] == "validation"
    with pytest.raises(ValueError, match="Conflicting"):
        assign(
            geometry,
            [
                prior,
                {
                    **prior,
                    "geometry": geometry,
                    "split": "train",
                    "acquisition_group": "other",
                },
            ],
            "known",
        )


def test_original_test_datatake_is_rejected_before_any_pixels() -> None:
    manifest = json.loads(
        (REPO_ROOT / "datasets/nl_benchmark/v0.2.0/manifest.json").read_text()
    )
    held = next(s for s in manifest["scenes"] if s["split"] == "test")
    row = {
        "stream": "vessel_support",
        "product": {
            **product(),
            "Id": held["product_id"],
            "Name": held["name"].replace("1SDV", "1SDH"),
        },
        "lon": -53.2,
        "lat": 48.3,
        "absolute_offset_s": 0,
        "source_identity": "private-vessel",
        "reference_midpoint_utc": "2026-01-01T00:00:00+00:00",
    }
    selected, excluded = reserve([], [row], [], set(), policy())
    assert not selected
    assert any("Held-out" in r["reason"] for r in excluded)


def test_catalogue_full_page_without_next_link_continues_by_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    first = [
        {"Id": str(i), "ContentDate": {"Start": "2017-06-01T10:00:00Z"}}
        for i in range(1000)
    ]

    class Response:
        url = CATALOGUE

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "value": (
                    first
                    if len(calls) == 1
                    else [
                        first[-1],
                        {
                            "Id": "later",
                            "ContentDate": {"Start": "2017-12-01T10:00:00Z"},
                        },
                    ]
                )
            }

    def get(url: str, **kwargs: Any) -> Response:
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr("agents.source_supported.requests.get", get)
    query_year(2017, tmp_path)
    result = json.loads((tmp_path / "catalogue_hh_hv_2017.json").read_text())
    assert len(calls) == 2
    assert "ContentDate/Start ge 2017-06-01" in calls[1]["params"]["$filter"]
    assert result["complete"] is True and len(result["products"]) == 1001


def test_catalogue_no_progress_fails_instead_of_marking_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        url = CATALOGUE

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "value": [
                    {"Id": str(i), "ContentDate": {"Start": "2017-06-01T10:00:00Z"}}
                    for i in range(1000)
                ]
            }

    monkeypatch.setattr(
        "agents.source_supported.requests.get", lambda *a, **kw: Response()
    )
    with pytest.raises(ValueError, match="did not advance"):
        query_year(2017, tmp_path)
    assert not (tmp_path / "catalogue_hh_hv_2017.json").exists()


def test_ais_missing_shoreline_and_stale_brackets_are_not_eligible() -> None:
    support = {
        "mmsi": "123456789",
        "lon": -53.2,
        "lat": 48.3,
        "physical_surface": "water",
        "signed_shore_distance_m": 2000,
        "nearest_offset_s": 1,
        "bracket_gap_s": 60,
        "uncertainty_m": 150,
        "reports": [{"time": observation()["time"], "values": {"speed_knots": "10"}}],
    }
    scene = {
        "polarization": "HH/HV",
        "reference_midpoint_utc": observation()["timestamp"],
        "product": product(),
        "support": [support],
    }
    assert len(vessel_opportunities({"scenes": [scene]}, policy())) == 1
    for changed in (
        {**support, "signed_shore_distance_m": None},
        {**support, "bracket_gap_s": 300},
        {**support, "nearest_offset_s": 121},
    ):
        assert not vessel_opportunities(
            {"scenes": [{**scene, "support": [changed]}]}, policy()
        )


def test_published_reservation_integrity_and_provisional_contract() -> None:
    directory = REPO_ROOT / "reservations/source_supported/v0.1.0"
    lock = json.loads(
        (REPO_ROOT / "configs/reservations/source_supported_v010.lock.json").read_text()
    )
    verify(directory, lock["seal_sha256"])
    original = json.loads((directory / "manifest.json").read_text())
    for field in (
        "classifier_scored",
        "imagery_inspected",
        "independent_review",
        "promotion_eligible",
    ):
        manifest = deepcopy(original)
        manifest["areas"][0][field] = True
        with pytest.raises(ValueError, match="guard"):
            validate_manifest(manifest)
    manifest = deepcopy(original)
    manifest["areas"][0]["label"] = "iceberg"
    with pytest.raises(ValueError, match="guard"):
        validate_manifest(manifest)
    manifest = deepcopy(original)
    manifest["areas"].pop()
    with pytest.raises(ValueError, match="inventory"):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match="seal"):
        verify(directory, "0" * 64)
