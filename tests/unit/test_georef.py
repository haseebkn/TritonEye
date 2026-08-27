import glob
import math
import os
import sys
from typing import Any, Iterator

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

# Align python path to workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.geo import Georeferencer, GeoreferencingError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def metres_apart(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    """Approximate great-circle separation, adequate at these scales."""
    dx = (lon_a - lon_b) * 111320.0 * math.cos(math.radians(lat_b))
    dy = (lat_a - lat_b) * 110570.0
    return math.hypot(dx, dy)


def gcp_grid() -> list[GroundControlPoint]:
    """A regular 4x4 GCP grid spanning a 1000x1000 raster, in EPSG:4326."""
    return [
        GroundControlPoint(
            row=float(r), col=float(c), x=-52.0 + c / 1000.0, y=47.5 - r / 1000.0
        )
        for r in (0, 333, 666, 1000)
        for c in (0, 333, 666, 1000)
    ]


@pytest.fixture
def gcp_raster() -> Iterator[Any]:
    """An in-memory raster carrying GCPs but no CRS, like a Sentinel-1 GRD."""
    profile = dict(driver="GTiff", dtype="uint8", width=1001, height=1001, count=1)
    with MemoryFile() as mem:
        with mem.open(gcps=gcp_grid(), crs="EPSG:4326", **profile) as dst:
            dst.write(np.zeros((1001, 1001), "uint8"), 1)
        with mem.open() as src:
            yield src


@pytest.fixture
def affine_raster() -> Iterator[Any]:
    """An in-memory raster carrying a projected CRS and affine transform."""
    profile = dict(
        driver="GTiff",
        dtype="uint8",
        width=1000,
        height=1000,
        count=1,
        crs="EPSG:32622",
        transform=from_bounds(350000.0, 5250000.0, 360000.0, 5260000.0, 1000, 1000),
    )
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.zeros((1000, 1000), "uint8"), 1)
        with mem.open() as src:
            yield src


@pytest.fixture
def bare_raster() -> Iterator[Any]:
    """An in-memory raster with neither a CRS nor GCPs."""
    profile = dict(driver="GTiff", dtype="uint8", width=100, height=100, count=1)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.zeros((100, 100), "uint8"), 1)
        with mem.open() as src:
            yield src


# --------------------------------------------------------------------------
# Strategy selection
# --------------------------------------------------------------------------


def test_gcp_raster_selects_tps(gcp_raster: Any) -> None:
    # A raster with GCPs and no usable affine transform must not be treated as
    # projected. This is the case that produced 34-277 km geolocation errors.
    assert gcp_raster.crs is None
    assert gcp_raster.transform.is_identity
    with Georeferencer.from_dataset(gcp_raster) as geo:
        assert geo.method == "gcp_tps"


def test_projected_raster_selects_affine(affine_raster: Any) -> None:
    with Georeferencer.from_dataset(affine_raster) as geo:
        assert geo.method == "affine"


def test_raster_without_georeferencing_raises(bare_raster: Any) -> None:
    # Refusing is the point: a detection at invented coordinates is worse than
    # no detection at all.
    with pytest.raises(GeoreferencingError):
        Georeferencer.from_dataset(bare_raster)


# --------------------------------------------------------------------------
# Coordinate accuracy
# --------------------------------------------------------------------------


def test_tps_reproduces_its_control_points(gcp_raster: Any) -> None:
    # Thin plate spline interpolation passes exactly through every control
    # point, so each GCP must map back to its own ground coordinate.
    gcps, _ = gcp_raster.gcps
    with Georeferencer.from_dataset(gcp_raster) as geo:
        rows = [g.row for g in gcps]
        cols = [g.col for g in gcps]
        lons, lats = geo.xy(rows, cols)

    for gcp, lon, lat in zip(gcps, lons, lats):
        assert metres_apart(lon, lat, gcp.x, gcp.y) < 0.5


def test_tps_interpolates_between_control_points(gcp_raster: Any) -> None:
    # The synthetic grid is exactly linear, so the midpoint between two GCPs has
    # a known answer the spline should recover.
    with Georeferencer.from_dataset(gcp_raster) as geo:
        lons, lats = geo.xy([500.0], [500.0])
    assert metres_apart(lons[0], lats[0], -51.5, 47.0) < 1.0


def test_affine_path_reprojects_to_wgs84(affine_raster: Any) -> None:
    # UTM 22N easting 350000 / northing 5260000 is the raster's upper-left.
    with Georeferencer.from_dataset(affine_raster) as geo:
        lons, lats = geo.xy([0.0], [0.0])
    assert -53.0 < lons[0] < -51.0
    assert 47.0 < lats[0] < 48.0


def test_xy_accepts_scalars_and_sequences(gcp_raster: Any) -> None:
    with Georeferencer.from_dataset(gcp_raster) as geo:
        one_lon, one_lat = geo.xy(0.0, 0.0)
        many_lon, many_lat = geo.xy([0.0, 500.0], [0.0, 500.0])
    assert len(one_lon) == 1 and len(one_lat) == 1
    assert len(many_lon) == 2 and len(many_lat) == 2
    assert one_lon[0] == pytest.approx(many_lon[0])


# --------------------------------------------------------------------------
# Footprint
# --------------------------------------------------------------------------


def test_footprint_covers_all_control_points(gcp_raster: Any) -> None:
    gcps, _ = gcp_raster.gcps
    with Georeferencer.from_dataset(gcp_raster) as geo:
        min_lon, min_lat, max_lon, max_lat = geo.footprint_bounds()

    for gcp in gcps:
        assert min_lon - 1e-6 <= gcp.x <= max_lon + 1e-6
        assert min_lat - 1e-6 <= gcp.y <= max_lat + 1e-6


# --------------------------------------------------------------------------
# Real Sentinel-1 products, when the (gitignored) archive is present
# --------------------------------------------------------------------------


def real_grd_products() -> list[str]:
    pattern = os.path.join(
        REPO_ROOT, "data", "raw", "S1*.SAFE", "measurement", "*.tif*"
    )
    found = []
    for path in sorted(glob.glob(pattern)):
        try:
            with rasterio.open(path) as src:
                if src.crs is None and src.gcps[0]:
                    found.append(path)
        except Exception:
            continue
    return found


@pytest.mark.skipif(
    not real_grd_products(), reason="no real Sentinel-1 GRD products in data/raw"
)
def test_real_grd_products_georeference_from_gcps() -> None:
    for path in real_grd_products():
        with rasterio.open(path) as src:
            gcps, _ = src.gcps
            with Georeferencer.from_dataset(src) as geo:
                assert geo.method == "gcp_tps", path

                # Every control point must come back to itself.
                lons, lats = geo.xy([g.row for g in gcps], [g.col for g in gcps])
                worst = max(
                    metres_apart(lon, lat, g.x, g.y)
                    for g, lon, lat in zip(gcps, lons, lats)
                )
                assert worst < 1.0, f"{path}: worst GCP residual {worst:.2f} m"

                # The footprint must span the real swath rather than collapsing
                # onto whichever AOI polygon the product was searched with.
                min_lon, min_lat, max_lon, max_lat = geo.footprint_bounds()
                assert max_lon - min_lon > 1.5, path
                assert max_lat - min_lat > 1.0, path
