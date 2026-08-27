#!/usr/bin/env python3
"""
TritonEye Georeferencing

Maps raster pixel coordinates to WGS84 longitude/latitude.

Sentinel-1 Level-1 GRD measurement rasters are stored in radar geometry: they
carry no CRS and no affine geotransform, only a grid of ground control points
(GCPs) tying pixel positions to the ground. Products that have been terrain
corrected (and TritonEye's synthetic mock products) carry a real CRS and affine
transform instead.

This module picks the right strategy for whichever it is handed, and raises
rather than guessing when a raster carries neither. A detection reported at the
wrong coordinates is worse than a detection not reported at all.
"""

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import rasterio
import rasterio.transform
from pyproj import Transformer
from rasterio.transform import GCPTransformer

# numpy generics need explicit parameters under mypy --strict
FloatArray = np.ndarray[Any, Any]

WGS84 = "EPSG:4326"

# GCP row/col reference the upper-left corner of their pixel, so pixel
# coordinates are mapped with the same convention throughout. Bounding box
# corners are grid positions rather than sample centres, which makes "ul" the
# correct offset for both georeferencing strategies.
_PIXEL_OFFSET = "ul"

# Thin plate spline interpolation passes exactly through every GCP and follows
# the curvature of the swath between them. A first-order affine fit to the same
# control points (rasterio.transform.from_gcps) is off by a mean of ~700 m on a
# full Sentinel-1 IW scene, which is too coarse to correlate against AIS.
_USE_TPS = True

# Samples taken along each raster edge when computing the ground footprint. The
# edges of a radar swath are curved, so corners alone understate the extent.
_FOOTPRINT_SAMPLES = 64


class GeoreferencingError(RuntimeError):
    """Raised when a raster carries neither a usable CRS nor GCPs."""


class Georeferencer:
    """
    Converts raster pixel coordinates to WGS84 longitude/latitude.

    Build one with :meth:`from_dataset` and reuse it for every conversion on
    that raster; constructing the GCP transformer solves a dense system and is
    far more expensive than the conversions themselves.
    """

    def __init__(
        self,
        method: str,
        width: int,
        height: int,
        affine: Any = None,
        src_crs: Any = None,
        gcp_transformer: Any = None,
        gcp_crs: Any = None,
    ) -> None:
        self.method = method
        self.width = width
        self.height = height
        self._affine = affine
        self._gcp_transformer = gcp_transformer

        source_crs = src_crs if method == "affine" else gcp_crs
        self._to_wgs84 = None
        if source_crs is not None and str(source_crs).upper() != WGS84:
            self._to_wgs84 = Transformer.from_crs(source_crs, WGS84, always_xy=True)

    @classmethod
    def from_dataset(cls, src: Any) -> "Georeferencer":
        """
        Chooses a strategy for an open rasterio dataset.

        A CRS paired with a real affine transform wins; otherwise GCPs are used.
        A dataset carrying a CRS but an identity transform is not georeferenced
        by the CRS alone, so the identity case falls through to the GCPs.
        """
        if src.crs is not None and not src.transform.is_identity:
            return cls(
                method="affine",
                width=src.width,
                height=src.height,
                affine=src.transform,
                src_crs=src.crs,
            )

        gcps, gcp_crs = src.gcps
        if gcps:
            return cls(
                method="gcp_tps" if _USE_TPS else "gcp_polynomial",
                width=src.width,
                height=src.height,
                gcp_transformer=GCPTransformer(gcps, tps=_USE_TPS),
                gcp_crs=gcp_crs or WGS84,
            )

        raise GeoreferencingError(
            f"Raster {getattr(src, 'name', '<unknown>')} has no CRS with an affine "
            "transform and no ground control points; its detections cannot be "
            "placed on the earth."
        )

    def xy(
        self,
        rows: Union[float, Sequence[float], FloatArray],
        cols: Union[float, Sequence[float], FloatArray],
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Maps pixel (row, col) positions to (longitude, latitude) arrays.

        Accepts scalars or sequences and always returns arrays. Rows and cols may
        be fractional; detection box corners rarely land on whole pixels.
        """
        rows_arr = np.atleast_1d(np.asarray(rows, dtype="float64"))
        cols_arr = np.atleast_1d(np.asarray(cols, dtype="float64"))

        if self.method == "affine":
            xs, ys = rasterio.transform.xy(
                self._affine,
                rows_arr.tolist(),
                cols_arr.tolist(),
                offset=_PIXEL_OFFSET,
            )
        else:
            xs, ys = self._gcp_transformer.xy(
                rows_arr.tolist(), cols_arr.tolist(), offset=_PIXEL_OFFSET
            )

        xs_arr = np.atleast_1d(np.asarray(xs, dtype="float64"))
        ys_arr = np.atleast_1d(np.asarray(ys, dtype="float64"))

        if self._to_wgs84 is not None:
            lon, lat = self._to_wgs84.transform(xs_arr, ys_arr)
            return np.atleast_1d(lon), np.atleast_1d(lat)
        return xs_arr, ys_arr

    def footprint_bounds(self) -> List[float]:
        """
        Returns the raster's ground footprint as [min_lon, min_lat, max_lon, max_lat].

        Sampled along all four edges rather than at the corners, because a radar
        swath's edges bow outward and corner-only bounds clip the scene.
        """
        n = _FOOTPRINT_SAMPLES
        last_row = float(self.height)
        last_col = float(self.width)
        along_rows = np.linspace(0.0, last_row, n)
        along_cols = np.linspace(0.0, last_col, n)

        rows = np.concatenate(
            [np.zeros(n), np.full(n, last_row), along_rows, along_rows]
        )
        cols = np.concatenate(
            [along_cols, along_cols, np.zeros(n), np.full(n, last_col)]
        )

        lon, lat = self.xy(rows, cols)
        return [
            float(np.min(lon)),
            float(np.min(lat)),
            float(np.max(lon)),
            float(np.max(lat)),
        ]

    def close(self) -> None:
        if self._gcp_transformer is not None:
            self._gcp_transformer.close()
            self._gcp_transformer = None

    def __enter__(self) -> "Georeferencer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def describe_raster_georeferencing(raster_path: str) -> Tuple[str, List[float]]:
    """
    Opens a raster and reports (method, footprint_bounds) without keeping it open.

    Used by the ingest agent to record honest spatial metadata for a product.
    """
    with rasterio.open(raster_path) as src:
        with Georeferencer.from_dataset(src) as geo:
            return geo.method, geo.footprint_bounds()
