"""Exercise native metadata, source ambiguity, provisional guards and replay."""

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from shapely import wkt
from shapely.geometry import box, mapping

from agents.artifacts import REPO_ROOT
from agents.native_nl_input import prepare
from agents.source_reservation import checked
from agents.training_annotation import (
    DIRECTORY,
    FLAGS,
    input_metadata,
    object_position,
    source_matches,
    validate,
    verify,
    window_query,
)


def product() -> dict[str, Any]:
    common = {
        "mission": "S1A",
        "mode": "IW",
        "range_spacing_m": 10.0,
        "azimuth_spacing_m": 10.0,
        "thermal_noise_corrected": "false",
    }
    return {"bands": [{**common, "polarization": pol} for pol in ("HH", "HV")]}


def row() -> dict[str, Any]:
    return {
        "native_context": {
            "center_incidence_deg": 37.0,
            "center_native_time_utc": "2020-01-01T10:00:00+00:00",
        },
        "geometry": mapping(box(-55.1, 50.0, -55.0, 50.1)),
        "window": [100, 200, 384, 384],
    }


def test_native_metadata_yields_exact_original_crop() -> None:
    values = np.arange(2 * 384 * 384, dtype="float32").reshape(2, 384, 384)
    result = prepare(
        values, [108, 115, 130, 170], input_metadata(product(), row()), "train"
    )
    assert result["status"] == "research_tensor_only"
    assert np.array_equal(result["tensor"], values[:, 105:180, 82:157])
    assert result["training_approved"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("mission", "S1B"),
        ("mode", "EW"),
        ("range_spacing_m", 20.0),
        ("azimuth_spacing_m", 20.0),
        ("thermal_noise_corrected", "true"),
        ("polarization", "VH"),
    ],
)
def test_hh_hv_metadata_disagreement_is_refused(field: str, value: Any) -> None:
    changed = product()
    changed["bands"][1][field] = value
    with pytest.raises(ValueError, match="metadata disagree"):
        input_metadata(changed, row())


def test_missing_noise_flag_cannot_be_treated_as_uncorrected() -> None:
    changed = product()
    for band in changed["bands"]:
        band["thermal_noise_corrected"] = None
    with pytest.raises(ValueError, match="Explicit"):
        input_metadata(changed, row())


def test_corrected_input_abstains_under_nominal_recipe() -> None:
    changed = product()
    for band in changed["bands"]:
        band["thermal_noise_corrected"] = "true"
    result = prepare(
        np.ones((2, 384, 384)),
        [108, 115, 130, 170],
        input_metadata(changed, row()),
        "train",
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "metadata_missing_or_incompatible"


def test_native_target_time_uses_object_row_not_catalogue_midpoint() -> None:
    points = [
        GroundControlPoint(row=r, col=c, x=-55 + c * 0.0001, y=50 + r * 0.0001)
        for r in (0, 500, 1000)
        for c in (0, 500, 1000)
    ]
    lon, lat, instant = object_position(
        {"bbox_px": [10, 12, 20, 22]},
        row(),
        {"native_start_time": "2020-01-01T10:00:00Z", "azimuth_interval_s": 0.5},
        points,
    )
    assert lon == pytest.approx(-54.9885, abs=1e-8)
    assert lat == pytest.approx(50.0217, abs=1e-8)
    assert instant.isoformat() == "2020-01-01T10:01:48.500000+00:00"


def test_window_query_includes_complete_polygon_and_frozen_time_interval() -> None:
    query = window_query(row(), {"lon": -55.05, "lat": 50.05}, 3)
    geometry = wkt.loads(query["$filter"].split("SRID=4326;", 1)[1].split("'", 1)[0])
    assert geometry.equals(box(-55.1, 50.0, -55.0, 50.1))
    assert "2020-01-01T07:00:00Z" in query["$filter"]
    assert "2020-01-01T13:00:00Z" in query["$filter"]


def observation(identity: str, **changes: Any) -> dict[str, Any]:
    return {
        "iceberg_id": identity,
        "time": 0,
        "lon": -55,
        "lat": 50,
        "method": "R/V",
        "source": "SHIP",
        **changes,
    }


def test_matching_excludes_circular_out_of_time_and_distant_observations() -> None:
    source = observation("selected")
    records = [
        source,
        observation("satellite", source="S1D"),
        observation("late", time=1900),
        observation("far", lon=-57),
    ]
    result = source_matches(
        -55, 50, datetime.fromtimestamp(0, timezone.utc), source, records
    )
    assert result["compatible_iceberg_identities"] == ["selected"]
    assert result["inside_selected_allowance"] is True


def test_competing_identities_and_repeated_sightings_are_counted_separately() -> None:
    source = observation("selected")
    result = source_matches(
        -55,
        50,
        datetime.fromtimestamp(0, timezone.utc),
        source,
        [source, observation("selected", time=10), observation("other")],
    )
    assert len(result["matches"]) == 3
    assert result["compatible_iceberg_identities"] == ["other", "selected"]


def collection() -> dict[str, Any]:
    value: dict[str, Any] = json.loads((DIRECTORY / "collection.json").read_text())
    return value


@pytest.mark.parametrize(
    "flag", [*FLAGS, "exploratory_training_admitted", "accuracy_truth_admitted"]
)
def test_collection_cannot_claim_scoring_review_or_admission(flag: str) -> None:
    value = collection()
    value[flag] = True
    with pytest.raises(ValueError, match="guard|admit"):
        validate(value)


def test_annotation_box_and_training_role_are_bound_to_first_pass() -> None:
    for key, change in (("split", "validation"), ("chip_sha256", "0" * 64)):
        value = collection()
        value["areas"][0][key] = change
        with pytest.raises(ValueError, match="identity or role"):
            validate(value)
    value = collection()
    value["areas"][0]["objects"][0]["bbox_px"][0] += 1
    with pytest.raises(ValueError, match="identity or role"):
        validate(value)


@pytest.mark.parametrize(
    "field", ["product_id", "geographic_group", "native_acquisition_group"]
)
def test_public_collection_must_keep_reserved_acquisition_and_groups(
    field: str,
) -> None:
    value = collection()
    value["areas"][0][field] = "changed"
    with pytest.raises(ValueError, match="reserved acquisition or group"):
        validate(value)


def test_incomplete_coverage_cannot_support_positive_class() -> None:
    value = collection()
    value["areas"][0]["source_coverage"]["source_association_abstain"] = True
    with pytest.raises(ValueError, match="Incomplete coverage"):
        validate(value)


def test_public_collection_cannot_leak_private_source_coordinates() -> None:
    value = collection()
    value["areas"][0]["source_identity"] = "not-public"
    with pytest.raises(ValueError, match="Private"):
        validate(value)


def test_empty_visual_scan_is_still_ignored_not_a_verified_negative() -> None:
    value = collection()
    empty = next(a for a in value["areas"] if not a["objects"])
    assert empty["whole_area_ignore_for_accuracy"] is True
    assert empty["source_coverage"]["source_association_abstain"] is True
    assert value["accuracy_truth_admitted"] is False
    assert verify()["training_admitted"] is False


def test_private_pixels_calibration_sources_and_masks_replay_when_available() -> None:
    value = collection()
    if not (REPO_ROOT / value["prepared"]["path"]).exists():
        pytest.skip("Requires retained native SAR and source files")
    assert verify(local=True)["verified"] is True
    report = json.loads(checked(value["local_report"]).read_text())
    prepared = json.loads(checked(value["prepared"]).read_text())
    target = next(
        a
        for p in prepared["products"]
        for a in p["areas"]
        if a["id"] == "training_iceberg_support_01"
    )
    with rasterio.open(checked(target["chip"])) as ds:
        values = ds.read()
    tensor = np.load(checked(report["samples"][0]["chip"]), allow_pickle=False)
    assert np.array_equal(tensor, values[:, 105:180, 82:157])
