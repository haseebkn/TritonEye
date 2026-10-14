"""Native uncertainty coverage uses geometry and both acquisition row times."""

from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from agents.source_followup import coverage


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
