"""Versioned, quarantined AllenAI alignment evidence; not benchmark truth.

The CC BY rslearn release and the older SIV database are distinct assets.
This module never makes the older labels training/evaluation-ready by inference.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from agents.artifacts import REPO_ROOT, sha256_file
from agents.public_vessel_labels import pixel_lonlat

WORLD_M = 2 * math.pi * 6378137
RESOLUTION = WORLD_M / (512 * 2**13)


def warp_publisher_window(
    source: Any, affine: Affine, window: Window
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Explicit GCP reprojection: do not silently warp an identity affine.

    Caller must enforce geographic/holdout scope before pixel access.
    Sentinel-1 positive DN is retained; zero denotes invalid output context.
    """
    gcps, crs = source.gcps
    if not gcps or str(crs) != "EPSG:4326":
        raise ValueError("Expected explicit WGS84 Sentinel-1 GCPs")
    width, height = int(window.width), int(window.height)
    if width <= 0 or height <= 0 or (width, height) != (window.width, window.height):
        raise ValueError("Positive integer publisher window dimensions required")
    values = np.zeros((height, width), dtype="uint16")
    reproject(
        source=rasterio.band(source, 1),
        destination=values,
        gcps=gcps,
        src_crs=crs,
        src_nodata=0,
        dst_transform=window_transform(window, affine),
        dst_crs="EPSG:3857",
        dst_nodata=0,
        resampling=Resampling.bilinear,
        num_threads=1,
        SRC_METHOD="GCP_POLYNOMIAL",
    )
    return values, values > 0


def bounded_patch_center(
    column: float, row: float, width: int, height: int
) -> tuple[float, float]:
    """Keep a 256-pixel QA context in-bounds; never move the label itself.

    One-pixel margin accommodates the GCP forward/inverse round-trip flooring.
    This is a crop-placement rule, not an annotation-position correction.
    """
    if not all(math.isfinite(v) for v in (column, row)):
        raise ValueError("Finite native point required")
    if width < 258 or height < 258 or not (0 <= column < width and 0 <= row < height):
        raise ValueError("Native point outside usable source raster")
    return min(max(column, 129), width - 129), min(max(row, 129), height - 129)


def publisher_xy(
    transform: Affine, image_origin: tuple[int, int], column: float, row: float
) -> tuple[float, float]:
    """Undo integer image-origin truncation, without snapping to bright pixels.

    Label coordinates are global integer *publisher* pixels. Reconstructing a
    GDAL suggested affine is not proof of byte-identical upstream imagery.
    """
    if not all(math.isfinite(v) for v in (*transform[:6], column, row)):
        raise ValueError("Finite publisher grid required")
    if (
        abs(transform.a - RESOLUTION) > 1e-7
        or abs(transform.e + RESOLUTION) > 1e-7
        or transform.b != 0
        or transform.d != 0
    ):
        raise ValueError("Expected north-up zoom-13 publisher grid")
    actual_origin = (
        int((transform.c + WORLD_M / 2) / RESOLUTION),
        int((WORLD_M / 2 - transform.f) / RESOLUTION),
    )
    if image_origin != actual_origin:
        raise ValueError("Publisher image origin does not match source metadata")
    x, y = transform * (column - image_origin[0], row - image_origin[1])
    return float(x), float(y)


def archive_crosswalk(
    index: dict[str, Any], database: Path, region: Any
) -> dict[str, Any]:
    """Crosswalk TAR window IDs AND origins, not merely a coincident numeric ID."""
    windows: set[str] = set()
    for member in index["members"]:
        path = PurePosixPath(member["name"])
        if len(path.parts) >= 3 and path.parts[0] == "windows":
            windows.add(path.parts[2])
    if len(windows) != index["windows"]:
        raise ValueError("Repeated archive window across groups or wrong count")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        records = {
            r["id"]: r
            for r in db.execute(
                "SELECT w.*,i.zoom,i.projection,i.name FROM windows w "
                "JOIN images i ON i.id=w.image_id WHERE w.dataset_id=1"
            )
        }
    splits: Counter[str] = Counter()
    intersecting, contained = [], []
    for name in sorted(windows):
        parts = name.split("_")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Unknown publisher archive window naming")
        col, row, identifier = map(int, parts)
        record = records.get(identifier)
        if (
            record is None
            or (record["column"], record["row"]) != (col, row)
            or str(record["projection"]).lower() != "epsg:3857"
        ):
            raise ValueError("Archive/source identity crosswalk unresolved")
        left, top = pixel_lonlat(col, row, record["zoom"])
        right, bottom = pixel_lonlat(
            col + record["width"], row + record["height"], record["zoom"]
        )
        geom = box(left, bottom, right, top)
        splits[record["split"]] += 1
        if region.intersects(geom):
            intersecting.append(identifier)
        if region.covers(geom):
            contained.append(identifier)
    return {
        "identity_check": "window_id_and_global_column_row",
        "crosswalked_windows": len(windows),
        "original_database_splits": dict(sorted(splits.items())),
        "nl_intersecting_windows": intersecting,
        "nl_wholly_contained_windows": contained,
        "imagery_extracted": False,
        "annotation_payloads_checked": False,
        "scope_limit": (
            "Coverage inferred from exact crosswalk to pinned original window "
            "extents. No claim of equal label content between distributions."
        ),
    }


def load_alignment_subset(release: Path, *, purpose: str = "alignment_review") -> Any:
    """Verify public/private manifests and artifacts before returning QA inputs.

    This release has no training or evaluation export. Neither a boolean edit
    nor a change of purpose can bypass the original benchmark review guard.
    """
    if purpose != "alignment_review":
        raise ValueError("Quarantined release: training and evaluation are blocked")
    descriptor = json.loads(release.read_text(encoding="utf-8"))
    if descriptor["version"] != "0.1.0" or descriptor["status"] != "quarantined_qa":
        raise ValueError("Unsupported alignment release")
    public = release.with_name("manifest.json")
    if sha256_file(public) != descriptor["manifest_sha256"]:
        raise ValueError("Public release integrity mismatch")
    if (
        sha256_file(release.with_name("source_evidence.json"))
        != descriptor["source_evidence_sha256"]
    ):
        raise ValueError("Source evidence integrity mismatch")
    reference = descriptor["local_manifest"]
    local = (REPO_ROOT / reference["path"]).resolve()
    data_root = (REPO_ROOT / "data/model_research/allenai_nl_v010").resolve()
    if not local.is_relative_to(data_root):
        raise ValueError("Local manifest must stay inside its release data root")
    if sha256_file(local) != reference["sha256"]:
        raise ValueError("Local release integrity mismatch")
    manifest = json.loads(local.read_text(encoding="utf-8"))
    if manifest["training_allowed"] or manifest["evaluation_allowed"]:
        raise ValueError("Quarantined manifest cannot enable scientific adoption")
    for artifact in manifest["artifacts"]:
        path = (REPO_ROOT / artifact["path"]).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Artifact outside release data root")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError("Alignment artifact integrity mismatch")
    return manifest
