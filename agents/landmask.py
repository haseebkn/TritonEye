#!/usr/bin/env python3
"""
TritonEye Land Mask

Classifies detections as water, coastal, or land against an open coastline
dataset, so that terrain returns do not enter the dark-vessel alert list.

WHY THIS EXISTS: the 2026-08-17 eastern Newfoundland scene produced 298
detections, the majority clustered over the Newfoundland landmass rather than
open water, while the open sea to the east was nearly empty. A SAR vessel
detector responds to bright compact targets against a dark background, and rock
outcrops, buildings, and radar-facing terrain slopes satisfy that description as
readily as a hull does.

TWO DESIGN DECISIONS ARE LOAD-BEARING:

1. CLASSIFY POINTS, NOT PIXELS. The obvious implementation rasterises a
   coastline to the scene grid and masks the imagery. A full IW scene is roughly
   25000x16000, so that costs hundreds of MB -- and this pipeline has already
   been bitten once by full-raster allocations (see interpolate_lut_grid in
   agents/calibration.py). Classifying ~300 detection centroids after
   georeferencing is milliseconds and allocates nothing.

2. FLAG, DO NOT DROP. Detections are annotated with `surface` and
   `distance_to_shore_m`, never deleted. Only water-classified detections become
   alerts. This keeps the rejection count auditable -- "298 raw -> N marine" is
   itself the evidence that a false-alarm mode was found and fixed -- and avoids
   silently discarding genuine harbour traffic on the strength of georeferencing
   that carries known residual error.

Coastline sources, all open, none paid:

  osm     OSM land polygons, ODbL. Derived from natural=coastline ways, so
          inland lakes are PART of the land polygons rather than holes in them.
          Newfoundland's interior is full of ponds and this is the behaviour we
          want. https://osmdata.openstreetmap.de/data/land-polygons.html
  gshhg   GSHHG, public domain, long-established in the SAR community. Fallback
          for when ODbL share-alike is awkward.
          https://www.soest.hawaii.edu/pwessel/gshhg/

CanVec (Open Government Licence - Canada) is the intended refinement for
Canadian waters but is packaged per NTS tile and needs tile-selection logic; it
is not wired up here.
"""

import os
import sys
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FloatArray = np.ndarray[Any, Any]

# Surface classes. `coastal` is deliberately a third class rather than a
# water/land binary: real nearshore vessels and land bleed both occupy that
# band, and forcing a binary would misreport one of them.
SURFACE_WATER = "water"
SURFACE_COASTAL = "coastal"
SURFACE_LAND = "land"

# Detections within this distance of shore are `coastal`. ~30 px at 10 m ground
# spacing, comfortably above the measured gcp_tps georeferencing residual.
DEFAULT_COASTAL_BUFFER_M = 300.0

SOURCES: Dict[str, Dict[str, str]] = {
    "osm": {
        "url": (
            "https://osmdata.openstreetmap.de/download/" "land-polygons-split-4326.zip"
        ),
        "archive": "land-polygons-split-4326.zip",
        "shapefile": "land-polygons-split-4326/land_polygons.shp",
        "licence": "ODbL - (c) OpenStreetMap contributors",
    },
    "gshhg": {
        "url": "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip",
        "archive": "gshhg-shp-2.3.7.zip",
        # L1 = continental land. Lakes (L2) are deliberately NOT subtracted:
        # an inland pond is not navigable water for our purposes, and a
        # detection on one is a false positive either way.
        "shapefile": "GSHHS_shp/f/GSHHS_f_L1.shp",
        "licence": "public domain",
    },
}


class LandMaskUnavailable(RuntimeError):  # noqa: N818 - reads better than ...Error
    """Raised when coastline data or its runtime dependencies are missing."""


def local_aeqd_crs(bounds: Sequence[float]) -> Any:
    """
    Builds an azimuthal equidistant CRS centred on a footprint.

    Distances must be metric, and a fixed UTM zone will not do: the 2026-08-17
    footprint spans lon -55.00..-51.17, straddling the zone 21N/22N boundary at
    -54W, so either zone distorts the far edge of the scene. AEQD centred on the
    footprint is exact for distance-from-centre by construction -- which is the
    quantity being measured -- and needs no zone-selection logic for any AOI.
    """
    try:
        from pyproj import CRS
    except ImportError as e:  # pragma: no cover
        raise LandMaskUnavailable(f"pyproj is required: {e}") from e

    min_lon, min_lat, max_lon, max_lat = bounds
    lon0 = (min_lon + max_lon) / 2.0
    lat0 = (min_lat + max_lat) / 2.0
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )


def resolve_cache_dir(base_dir: str, cache_dir: Optional[str] = None) -> str:
    """Returns the coastline cache directory, creating it if absent."""
    path = cache_dir or os.path.join(base_dir, "data", "reference")
    os.makedirs(path, exist_ok=True)
    return path


def fetch(source: str = "osm", cache_dir: str = ".", quiet: bool = False) -> str:
    """
    Downloads and extracts a coastline dataset, returning the shapefile path.

    Idempotent: a no-op when the shapefile is already extracted. The download is
    large (hundreds of MB), so this is deliberately never called implicitly from
    a mission run -- see `python -m agents.landmask --fetch`.
    """
    if source not in SOURCES:
        raise LandMaskUnavailable(
            f"Unknown coastline source {source!r}; expected one of "
            f"{sorted(SOURCES)}"
        )
    spec = SOURCES[source]
    shp_path = os.path.join(cache_dir, spec["shapefile"])
    if os.path.exists(shp_path):
        return shp_path

    os.makedirs(cache_dir, exist_ok=True)
    archive_path = os.path.join(cache_dir, spec["archive"])

    if not os.path.exists(archive_path):
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise LandMaskUnavailable(f"requests is required to fetch: {e}") from e
        if not quiet:
            print(f"Downloading {source} coastline: {spec['url']}", file=sys.stderr)
        with requests.get(spec["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            with open(archive_path + ".part", "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if not quiet and total:
                        pct = 100.0 * done / total
                        print(
                            f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)",
                            end="",
                            file=sys.stderr,
                        )
            if not quiet:
                print(file=sys.stderr)
        os.replace(archive_path + ".part", archive_path)

    if not quiet:
        print(f"Extracting {spec['archive']}", file=sys.stderr)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(cache_dir)

    if not os.path.exists(shp_path):
        raise LandMaskUnavailable(
            f"Extracted {spec['archive']} but {spec['shapefile']} is not present"
        )
    return shp_path


class LandMask:
    """
    Coastline geometry for one scene footprint, in a local metric projection.

    Construct via `for_footprint`, which clips the global coastline to the scene
    before projecting -- loading world land polygons per mission would be
    needlessly slow, and the projection is only valid near its centre anyway.
    """

    def __init__(
        self,
        land_geoms: Any,
        crs: Any,
        source: str,
        licence: str,
        shapefile: Optional[str] = None,
    ) -> None:
        from shapely import STRtree
        from shapely.ops import unary_union

        self.source = source
        self.licence = licence
        self.shapefile = shapefile
        self.crs = crs
        self._geoms = list(land_geoms)
        # A prepared union answers contains() far faster than iterating parts,
        # and the boundary is what distance-to-shore is measured against.
        self._union = unary_union(self._geoms) if self._geoms else None
        self._tree = STRtree(self._geoms) if self._geoms else None
        self._boundary = self._union.boundary if self._union is not None else None

    @property
    def is_empty(self) -> bool:
        """True when no land intersects the footprint - an all-ocean scene."""
        return self._union is None

    @classmethod
    def for_footprint(
        cls,
        bounds: Sequence[float],
        source: str = "osm",
        cache_dir: str = ".",
        shapefile: Optional[str] = None,
    ) -> "LandMask":
        """
        Loads coastline polygons intersecting `bounds` (min_lon, min_lat,
        max_lon, max_lat), reprojected to a local AEQD CRS.
        """
        try:
            import geopandas as gpd
        except ImportError as e:  # pragma: no cover
            raise LandMaskUnavailable(f"geopandas is required: {e}") from e

        spec = SOURCES.get(source, {})
        path = shapefile or os.path.join(cache_dir, spec.get("shapefile", ""))
        if not path or not os.path.exists(path):
            raise LandMaskUnavailable(
                f"Coastline data for source {source!r} not found at {path!r}. "
                f"Run: python -m agents.landmask --fetch --source {source}"
            )

        # Reading with a bbox filter pushes the spatial query into the driver,
        # so world-scale files never fully enter memory.
        gdf = gpd.read_file(path, bbox=tuple(bounds))
        crs = local_aeqd_crs(bounds)
        if len(gdf):
            gdf = gdf.to_crs(crs)
        return cls(
            list(gdf.geometry),
            crs,
            source,
            spec.get("licence", "unknown"),
            shapefile=path,
        )

    def classify(
        self,
        lons: Sequence[float],
        lats: Sequence[float],
        coastal_buffer_m: float = DEFAULT_COASTAL_BUFFER_M,
    ) -> Tuple[List[str], FloatArray]:
        """
        Classifies detection centroids as water, coastal, or land.

        Returns (surfaces, distance_to_shore_m). Distance is signed: positive
        offshore, negative inland, and measured to the nearest coastline, so a
        detection 1 km out to sea returns ~1000.0 and one 1 km inland ~-1000.0.
        """
        from pyproj import Transformer
        from shapely import points as shapely_points

        n = len(lons)
        if n != len(lats):
            raise ValueError(f"lons/lats length mismatch: {n} != {len(lats)}")
        if n == 0:
            return [], np.zeros(0, dtype="float64")

        # An all-ocean footprint has no coastline to measure against; every
        # detection is water at effectively unbounded distance from shore.
        if self.is_empty:
            return [SURFACE_WATER] * n, np.full(n, np.inf)

        transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

        # Coordinates go in as plain lists, not ndarrays. Given a SIZE-1 ndarray
        # pyproj takes its scalar fast-path, which converts a 1-element array to
        # a scalar -- deprecated in NumPy 1.25 and slated to raise. A scene with
        # exactly one detection would then fail on a future NumPy while every
        # multi-detection scene passed. Lists take the sequence path at any
        # length, and the conversion cost is nil at detection-count scale.
        xs, ys = transformer.transform(list(lons), list(lats))
        pts = shapely_points(
            np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")
        )

        # shapely 2.x vectorises both of these over arrays; do not loop.
        inland = self._union.contains(pts)
        dist = self._boundary.distance(pts)

        signed = np.where(inland, -dist, dist).astype("float64")
        surfaces = [
            (
                SURFACE_LAND
                if signed[i] < 0.0
                else SURFACE_COASTAL if signed[i] <= coastal_buffer_m else SURFACE_WATER
            )
            for i in range(n)
        ]
        return surfaces, signed


def summarize(surfaces: Sequence[str]) -> Dict[str, int]:
    """Counts detections per surface class, for the mission payload and report."""
    return {
        SURFACE_WATER: sum(1 for s in surfaces if s == SURFACE_WATER),
        SURFACE_COASTAL: sum(1 for s in surfaces if s == SURFACE_COASTAL),
        SURFACE_LAND: sum(1 for s in surfaces if s == SURFACE_LAND),
    }


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="TritonEye land mask utility")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download and extract the coastline dataset into the cache.",
    )
    parser.add_argument("--source", default="osm", choices=sorted(SOURCES))
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cache_dir = resolve_cache_dir(base_dir, args.cache_dir)
    if args.fetch:
        path = fetch(args.source, cache_dir)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Coastline ready: {path} ({size_mb:.1f} MB shapefile)")
        print(f"Licence: {SOURCES[args.source]['licence']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
