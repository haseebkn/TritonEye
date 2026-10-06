"""Production-aligned context windows and cross-split imagery guards."""

from typing import Any

import numpy as np
from pyproj import Geod, Transformer
from shapely.geometry import Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import transform

from agents.xview3_detector import TILE_SIZE, tile_origins


def selected_origins(
    width: int, height: int, window: list[int]
) -> list[tuple[int, int]]:
    col, row, w, h = window
    # Six-pixel dedup margin includes neighboring seam detections.
    return [
        (r, c)
        for r, c in tile_origins(width, height)
        if c < col + w + 6
        and c + TILE_SIZE > col - 6
        and r < row + h + 6
        and r + TILE_SIZE > row - 6
    ]


def context_geometry(geo: Any, r: int, c: int) -> Any:
    vertices: list[tuple[float, float]] = []
    for r0, c0, r1, c1 in (
        (r, c, r, c + TILE_SIZE),
        (r, c + TILE_SIZE, r + TILE_SIZE, c + TILE_SIZE),
        (r + TILE_SIZE, c + TILE_SIZE, r + TILE_SIZE, c),
        (r + TILE_SIZE, c, r, c),
    ):
        xs, ys = geo.xy(
            np.linspace(r0, r1, 17).tolist(), np.linspace(c0, c1, 17).tolist()
        )
        vertices.extend(zip(xs, ys))
    geometry = Polygon(vertices)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Invalid context georeferencing")
    return geometry


def require_context_split(geometry: Any, split: str, manifest: dict[str, Any]) -> None:
    project = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
    context = transform(project.transform, geometry)
    for scene in manifest["scenes"]:
        if scene["split"] == split:
            continue
        for roi in scene["rois"]:
            forbidden = transform(project.transform, shape(roi["geometry"])).buffer(
                manifest["separation_m"]
            )
            if context.intersects(forbidden):
                raise ValueError(f"Context imagery overlaps another split: {roi['id']}")


def area_km2(geometry: Any) -> float:
    """Geodesic polygon area, with correctly oriented interior rings."""
    if geometry.is_empty:
        return 0.0
    if geometry.geom_type == "Polygon":
        return (
            abs(float(Geod(ellps="WGS84").geometry_area_perimeter(orient(geometry))[0]))
            / 1e6
        )
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        return sum(area_km2(g) for g in geometry.geoms)
    return 0.0
