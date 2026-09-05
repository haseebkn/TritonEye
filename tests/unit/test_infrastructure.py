"""
Fixed offshore infrastructure masking tests.

Positions are public record, so these assert against the published coordinates
directly rather than against a fixture -- if a coordinate is ever edited, that
is a factual change and a test should notice.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.infrastructure import (  # noqa: E402
    EXCLUSION_RADIUS_M,
    INSTALLATIONS,
    classify_infrastructure,
    installations_in_bounds,
    summarize,
)

GRAND_BANKS = [-51.0, 45.5, -47.5, 47.5]
HIBERNIA_LON, HIBERNIA_LAT = -48.782933, 46.750433


def east_of(lon: float, lat: float, metres: float) -> float:
    """Longitude shifted east by an approximate ground distance."""
    return lon + metres / (111_320.0 * math.cos(math.radians(lat)))


def test_all_four_installations_lie_in_the_grand_banks_aoi() -> None:
    """
    The AOI exists partly to contain these; if it is ever resized so that an
    installation falls outside, that installation silently stops being masked.
    """
    found = {i["name"] for i in installations_in_bounds(GRAND_BANKS)}
    assert found == {i["name"] for i in INSTALLATIONS}
    assert len(found) == 4


def test_detection_on_a_platform_is_attributed_to_it() -> None:
    hit, names = classify_infrastructure([HIBERNIA_LON], [HIBERNIA_LAT], GRAND_BANKS)
    assert hit == [True]
    assert names == ["Hibernia"]


def test_detection_inside_provisional_proximity_radius_is_flagged() -> None:
    """A nearby target is flagged for review, not identified as a structure."""
    lon = east_of(HIBERNIA_LON, HIBERNIA_LAT, 400.0)
    hit, names = classify_infrastructure([lon], [HIBERNIA_LAT], GRAND_BANKS)
    assert hit == [True]
    assert names == ["Hibernia"]


def test_supply_vessel_working_the_field_is_not_masked() -> None:
    """
    5 km from the structure is a vessel, not the structure.

    This is the trade the radius encodes: an installation is served by supply
    and standby vessels, and masking the field rather than the structure would
    hide exactly the traffic an MDA system exists to show.
    """
    lon = east_of(HIBERNIA_LON, HIBERNIA_LAT, 5_000.0)
    hit, names = classify_infrastructure([lon], [HIBERNIA_LAT], GRAND_BANKS)
    assert hit == [False]
    assert names == [""]


def test_radius_boundary_is_respected() -> None:
    """A detection beyond a tight radius but inside a wide one flips."""
    lon = east_of(HIBERNIA_LON, HIBERNIA_LAT, 1_500.0)
    tight, _ = classify_infrastructure(
        [lon], [HIBERNIA_LAT], GRAND_BANKS, radius_m=1_000.0
    )
    wide, _ = classify_infrastructure(
        [lon], [HIBERNIA_LAT], GRAND_BANKS, radius_m=2_000.0
    )
    assert tight == [False]
    assert wide == [True]


def test_each_detection_is_attributed_to_its_nearest_installation() -> None:
    """Hebron and Terra Nova are only ~8 km apart; attribution must not blur."""
    hit, names = classify_infrastructure(
        [-48.498056, -48.479440], [46.543889, 46.475000], GRAND_BANKS
    )
    assert hit == [True, False]
    assert names == ["Hebron", ""]


def test_aoi_without_installations_masks_nothing() -> None:
    """An AOI containing no known structures must classify everything as clear."""
    coastal = [-53.5, 47.3, -52.0, 48.8]  # eastern_newfoundland
    assert installations_in_bounds(coastal) == []
    hit, names = classify_infrastructure([-52.5, -53.0], [48.0, 47.5], coastal)
    assert hit == [False, False]
    assert names == ["", ""]


def test_empty_input_is_handled() -> None:
    hit, names = classify_infrastructure([], [], GRAND_BANKS)
    assert hit == []
    assert names == []


def test_mismatched_input_lengths_raise() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        classify_infrastructure([-48.0, -48.1], [46.5], GRAND_BANKS)


def test_single_detection_does_not_warn() -> None:
    """
    Same pyproj scalar fast-path trap the land mask hit: a size-1 ndarray is
    converted to a scalar, which NumPy deprecated and will make an error.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        hit, _ = classify_infrastructure([HIBERNIA_LON], [HIBERNIA_LAT], GRAND_BANKS)
    assert hit == [True]


def test_published_positions_are_plausible_for_the_jeanne_darc_basin() -> None:
    """
    Guards against a transposed sign or a swapped lat/lon in the registry --
    the failure mode that would silently disable masking while looking fine.
    """
    for inst in INSTALLATIONS:
        assert 46.0 < inst["lat"] < 47.5, inst["name"]
        assert -49.5 < inst["lon"] < -47.5, inst["name"]
        assert inst["source"].startswith("https://"), inst["name"]
        assert inst["kind"] in {"gravity base structure", "FPSO"}, inst["name"]


def test_gravity_base_structures_are_marked_fixed() -> None:
    """
    A GBS sits on the seabed and its position is good indefinitely; an FPSO
    holds station in a watch circle and can disconnect. Flattening that
    distinction would make an FPSO position look more authoritative than it is.
    """
    by_name = {i["name"]: i for i in INSTALLATIONS}
    assert by_name["Hibernia"]["fixed"] is True
    assert by_name["Hebron"]["fixed"] is True
    assert by_name["Terra Nova"]["fixed"] is False
    assert by_name["White Rose (SeaRose)"]["fixed"] is False


def test_summarize_counts_per_installation() -> None:
    assert summarize(["Hibernia", "Hibernia", "", "Hebron"]) == {
        "Hibernia": 2,
        "Hebron": 1,
    }
    assert summarize([]) == {}
    assert summarize(["", ""]) == {}


def test_default_radius_is_documented_as_a_trade() -> None:
    """The value is a judgement, not a constant of nature; keep it visible."""
    assert EXCLUSION_RADIUS_M == 500.0
