"""
Land mask classification tests.

These build a synthetic coastline rather than depending on a several-hundred-MB
network download, so the suite stays hermetic and fast. The geometry is a square
"island" in the North Atlantic near the real AOI, which keeps the projection
maths representative of production use.
"""

import math
import os
import sys
from typing import Sequence

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.landmask import (  # noqa: E402
    DEFAULT_COASTAL_BUFFER_M,
    SURFACE_COASTAL,
    SURFACE_LAND,
    SURFACE_WATER,
    LandMask,
    LandMaskUnavailable,
    local_aeqd_crs,
    summarize,
)

# A footprint over eastern Newfoundland, matching the real AOI.
BOUNDS = [-53.5, 47.3, -52.0, 48.8]

# Island centre, and a half-width of ~0.2 deg lat (~22 km).
ISLAND_LON, ISLAND_LAT = -52.75, 48.05
ISLAND_HALF_DEG = 0.2


def build_mask(
    bounds: Sequence[float] = BOUNDS, half_deg: float = ISLAND_HALF_DEG
) -> LandMask:
    """A LandMask holding one square island, projected to the local AEQD CRS."""
    import geopandas as gpd
    from shapely.geometry import box

    crs = local_aeqd_crs(bounds)
    island = box(
        ISLAND_LON - half_deg,
        ISLAND_LAT - half_deg,
        ISLAND_LON + half_deg,
        ISLAND_LAT + half_deg,
    )
    gdf = gpd.GeoDataFrame(geometry=[island], crs="EPSG:4326").to_crs(crs)
    return LandMask(list(gdf.geometry), crs, "synthetic", "test fixture")


def offset_lon(lon: float, lat: float, metres: float) -> float:
    """Shifts a longitude east by an approximate ground distance."""
    return lon + metres / (111_320.0 * math.cos(math.radians(lat)))


def test_point_inside_island_is_land() -> None:
    mask = build_mask()
    surfaces, dist = mask.classify([ISLAND_LON], [ISLAND_LAT])
    assert surfaces == [SURFACE_LAND]
    assert dist[0] < 0

    # Nearest shore from the centre is the E/W edge, not the N/S one: at
    # latitude 48 a degree of longitude is only cos(48) as long as a degree of
    # latitude, so 0.2 deg east is ~14.9 km while 0.2 deg north is ~22.3 km.
    expected = ISLAND_HALF_DEG * 111_320.0 * math.cos(math.radians(ISLAND_LAT))
    assert abs(dist[0]) == pytest.approx(expected, rel=0.02)


def test_point_well_offshore_is_water() -> None:
    mask = build_mask()
    # ~50 km east of the island's eastern edge.
    lon = offset_lon(ISLAND_LON + ISLAND_HALF_DEG, ISLAND_LAT, 50_000)
    surfaces, dist = mask.classify([lon], [ISLAND_LAT])
    assert surfaces == [SURFACE_WATER]
    assert dist[0] > DEFAULT_COASTAL_BUFFER_M


def test_point_just_offshore_is_coastal() -> None:
    """100 m offshore with a 300 m buffer must classify as coastal, not water."""
    mask = build_mask()
    lon = offset_lon(ISLAND_LON + ISLAND_HALF_DEG, ISLAND_LAT, 100.0)
    surfaces, dist = mask.classify(
        [lon], [ISLAND_LAT], coastal_buffer_m=DEFAULT_COASTAL_BUFFER_M
    )
    assert surfaces == [SURFACE_COASTAL]
    assert 0 < dist[0] <= DEFAULT_COASTAL_BUFFER_M


def test_distance_is_metric_not_degrees() -> None:
    """
    Regression guard for the most likely bug in this module.

    Classifying in EPSG:4326 without reprojecting returns distances in degrees,
    which are ~1e5 times too small and silently make every detection 'coastal'.
    A point 1 km offshore must report ~1000 m, not ~0.01.
    """
    mask = build_mask()
    lon = offset_lon(ISLAND_LON + ISLAND_HALF_DEG, ISLAND_LAT, 1_000.0)
    _, dist = mask.classify([lon], [ISLAND_LAT])
    assert dist[0] == pytest.approx(1_000.0, rel=0.05)


def test_buffer_width_is_respected() -> None:
    """A point beyond a narrow buffer but inside a wide one flips class."""
    mask = build_mask()
    lon = offset_lon(ISLAND_LON + ISLAND_HALF_DEG, ISLAND_LAT, 500.0)
    narrow, _ = mask.classify([lon], [ISLAND_LAT], coastal_buffer_m=300.0)
    wide, _ = mask.classify([lon], [ISLAND_LAT], coastal_buffer_m=1_000.0)
    assert narrow == [SURFACE_WATER]
    assert wide == [SURFACE_COASTAL]


def test_classify_is_vectorised_over_many_points() -> None:
    """All three classes resolve correctly in a single batched call."""
    mask = build_mask()
    east = ISLAND_LON + ISLAND_HALF_DEG
    lons = [
        ISLAND_LON,  # land
        offset_lon(east, ISLAND_LAT, 100.0),  # coastal
        offset_lon(east, ISLAND_LAT, 50_000.0),  # water
    ]
    lats = [ISLAND_LAT] * 3
    surfaces, dist = mask.classify(lons, lats)
    assert surfaces == [SURFACE_LAND, SURFACE_COASTAL, SURFACE_WATER]
    assert len(dist) == 3
    assert dist[0] < 0 < dist[1] < dist[2]


def test_empty_input_returns_empty_arrays() -> None:
    mask = build_mask()
    surfaces, dist = mask.classify([], [])
    assert surfaces == []
    assert len(dist) == 0


def test_mismatched_input_lengths_raise() -> None:
    mask = build_mask()
    with pytest.raises(ValueError, match="length mismatch"):
        mask.classify([-52.0, -52.1], [48.0])


def test_all_ocean_footprint_classifies_everything_as_water() -> None:
    """
    A footprint with no land must not crash and must not reject detections.

    Open-ocean scenes are the case the mask is least needed for and the one an
    empty-geometry bug would silently break.
    """
    crs = local_aeqd_crs(BOUNDS)
    mask = LandMask([], crs, "synthetic", "test fixture")
    assert mask.is_empty
    surfaces, dist = mask.classify([-52.5, -52.6], [48.0, 48.1])
    assert surfaces == [SURFACE_WATER, SURFACE_WATER]
    assert np.isinf(dist).all()


def test_aeqd_distance_matches_geodesic_across_scene() -> None:
    """
    The projection must be accurate across a footprint spanning two UTM zones.

    The 2026-08-17 scene spans lon -55.00..-51.17, straddling the 21N/22N
    boundary at -54W, which is why a fixed UTM zone was rejected in favour of
    AEQD centred on the footprint.
    """
    from pyproj import Geod, Transformer

    scene = [-55.00, 47.88, -51.17, 49.77]
    crs = local_aeqd_crs(scene)
    lon0 = (scene[0] + scene[2]) / 2.0
    lat0 = (scene[1] + scene[3]) / 2.0

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geod = Geod(ellps="WGS84")

    for lon, lat in [(-55.00, 47.88), (-51.17, 49.77), (-53.00, 48.50)]:
        x, y = transformer.transform(lon, lat)
        planar = math.hypot(x, y)
        _, _, geodesic = geod.inv(lon0, lat0, lon, lat)
        assert planar == pytest.approx(geodesic, abs=1.0)


def test_summarize_counts_each_class() -> None:
    counts = summarize(
        [SURFACE_WATER, SURFACE_LAND, SURFACE_LAND, SURFACE_COASTAL, SURFACE_WATER]
    )
    assert counts == {SURFACE_WATER: 2, SURFACE_COASTAL: 1, SURFACE_LAND: 2}


def test_summarize_on_empty_input() -> None:
    assert summarize([]) == {
        SURFACE_WATER: 0,
        SURFACE_COASTAL: 0,
        SURFACE_LAND: 0,
    }


def test_missing_coastline_data_raises_with_actionable_message() -> None:
    """The error must name the command that fixes it."""
    with pytest.raises(LandMaskUnavailable, match="--fetch"):
        LandMask.for_footprint(BOUNDS, source="osm", cache_dir="/nonexistent-path")


def test_unknown_source_is_rejected() -> None:
    from agents.landmask import fetch

    with pytest.raises(LandMaskUnavailable, match="Unknown coastline source"):
        fetch("not-a-real-source", cache_dir=".")


def test_single_detection_scene_does_not_warn() -> None:
    """
    Regression: pyproj takes a scalar fast-path for size-1 ndarrays, which
    converts a 1-element array to a scalar. NumPy 1.25 deprecated that and it is
    slated to raise, so a scene with exactly one detection would have started
    failing while every multi-detection scene kept passing.
    """
    import warnings

    mask = build_mask()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        surfaces, dist = mask.classify([ISLAND_LON], [ISLAND_LAT])
    assert surfaces == [SURFACE_LAND]
    assert len(dist) == 1
