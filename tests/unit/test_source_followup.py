"""Native uncertainty coverage uses geometry and both acquisition row times."""

from copy import deepcopy
from typing import Any

import pytest
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from agents.source_followup import coverage
from agents.source_followup_annotations import validate_collection, validate_support


def geometry(half_width: float) -> dict[str, Any]:
    crs = CRS.from_proj4("+proj=aeqd +lat_0=48 +lon_0=-53 +datum=WGS84 +units=m")
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    return dict(
        mapping(
            transform(
                inverse.transform, box(-half_width, -half_width, half_width, half_width)
            )
        )
    )


def test_coverage_complete_at_fine_precision() -> None:
    result = coverage(
        geometry(2000),
        {"lon": -53, "lat": 48, "time": 0},
        {"SIGHTING_LATITUDE": "48.0000", "SIGHTING_LONGITUDE": "-53.0000"},
        "1970-01-01T00:01:00+00:00",
        "1970-01-01T00:01:01+00:00",
    )
    assert result["all_scenarios_complete"]
    assert not result["source_association_abstain"]
    assert result["maximum_native_row_offset_s"] == 61
    assert not result["accuracy_bounds_established"]


def test_nominal_coverage_does_not_hide_coordinate_sensitivity() -> None:
    result = coverage(
        geometry(800),
        {"lon": -53, "lat": 48, "time": 0},
        {"SIGHTING_LATITUDE": "48.00", "SIGHTING_LONGITUDE": "-53.00"},
        "1970-01-01T00:01:00+00:00",
        "1970-01-01T00:01:01+00:00",
    )
    assert result["scenarios"]["original"]["complete"]
    assert not result["scenarios"]["nearest_rounding"]["complete"]
    assert result["source_association_abstain"]
    assert (
        result["scenarios"]["nearest_rounding"]["approximate_covered_area_fraction"] < 1
    )


def test_coverage_uses_worst_row_time_on_either_side_of_observation() -> None:
    source = {"lon": -53, "lat": 48, "time": 0}
    literal = {"SIGHTING_LATITUDE": "48.0000", "SIGHTING_LONGITUDE": "-53.0000"}
    result = coverage(
        geometry(800),
        source,
        literal,
        "1969-12-31T23:49:10+00:00",
        "1970-01-01T00:00:10+00:00",
    )
    assert result["maximum_native_row_offset_s"] == 650
    assert not result["scenarios"]["original"]["complete"]


def evidence() -> dict[str, Any]:
    return {
        "areas": [
            {
                "id": "area",
                "source_association_abstain": False,
                "objects": [
                    {
                        "object_id": "u1",
                        "inside_selected_allowance": True,
                        "compatible_iceberg_identities": ["ice"],
                        "matches": [{"selected_identity": True}],
                    }
                ],
            }
        ]
    }


def positive() -> list[dict[str, Any]]:
    return [{"id": "area", "objects": [{"id": "u1", "class": "plausible_iceberg"}]}]


def test_incomplete_source_coverage_cannot_support_iceberg_class() -> None:
    source = evidence()
    validate_support(positive(), source)
    source["areas"][0]["source_association_abstain"] = True
    with pytest.raises(ValueError, match="complete unique"):
        validate_support(positive(), source)


def test_competing_returns_or_identities_prevent_positive_support() -> None:
    source = evidence()
    for case in ("objects", "identities", "wrong_source"):
        changed = deepcopy(source)
        objects = changed["areas"][0]["objects"]
        if case == "objects":
            objects.append({**objects[0], "object_id": "u2"})
        elif case == "identities":
            objects[0]["compatible_iceberg_identities"].append("other")
        else:
            objects[0]["matches"][0]["selected_identity"] = False
        with pytest.raises(ValueError, match="complete unique"):
            validate_support(positive(), changed)
    vessel = positive()
    vessel[0]["objects"][0]["class"] = "provisional_vessel"
    with pytest.raises(ValueError, match="no co-temporal vessel"):
        validate_support(vessel, source)


@pytest.mark.parametrize(
    "flag",
    [
        "classifier_scored",
        "detector_scored",
        "independent_review",
        "training_approved",
        "promotion_eligible",
        "compatibility_established",
        "annotation_completeness_verified",
    ],
)
def test_provisional_collection_refuses_approval_or_score_flags(flag: str) -> None:
    value = {
        key: key == flag
        for key in (
            "classifier_scored",
            "detector_scored",
            "independent_review",
            "training_approved",
            "promotion_eligible",
            "compatibility_established",
            "annotation_completeness_verified",
        )
    }
    with pytest.raises(ValueError, match="guard changed"):
        validate_collection(value)
