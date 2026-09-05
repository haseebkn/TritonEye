"""Newfoundland and Labrador research area, in RFC 7946 longitude/latitude.

This is a study boundary, not a claim about provincial maritime jurisdiction.
External payload boundaries may narrow, but never expand, this scope.
"""

import json
import math
from pathlib import Path
from typing import Any, Sequence

from shapely.geometry import Point, shape
from shapely.ops import unary_union

REGION_NAME = "Newfoundland and Labrador maritime study area"
REGION_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "aois"
    / ("newfoundland_labrador.geojson")
)


def load_region(path_or_geojson: Any = None) -> Any:
    """Load polygon/GeoJSON; reject malformed or out-of-scope study regions."""
    value = path_or_geojson
    if value is None or isinstance(value, (str, Path)):
        with open(value or REGION_PATH, encoding="utf-8") as stream:
            value = json.load(stream)
    if hasattr(value, "geom_type"):
        geom = value
    elif value.get("type") == "FeatureCollection":
        geom = unary_union([shape(f["geometry"]) for f in value["features"]])
    elif value.get("type") == "Feature":
        geom = shape(value["geometry"])
    else:
        geom = shape(value)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty:
        raise ValueError("Study area must be a nonempty Polygon or MultiPolygon")
    if not geom.is_valid:
        raise ValueError("Study area geometry is invalid")
    if path_or_geojson is not None and not load_region().covers(geom):
        raise ValueError("Analysis area must stay within Newfoundland and Labrador")
    return geom


def contains_points(
    lons: Sequence[float], lats: Sequence[float], region: Any = None
) -> list[bool]:
    """Boundary-inclusive membership; invalid coordinates are outside scope."""
    if len(lons) != len(lats):
        raise ValueError("Longitude and latitude lengths differ")
    area = load_region(region)
    return [
        math.isfinite(float(lon))
        and math.isfinite(float(lat))
        and bool(area.covers(Point(float(lon), float(lat))))
        for lon, lat in zip(lons, lats)
    ]


def validate_aoi(geometry: dict[str, Any]) -> None:
    """Reject imaging requests outside the Newfoundland/Labrador study area."""
    load_region(geometry)
