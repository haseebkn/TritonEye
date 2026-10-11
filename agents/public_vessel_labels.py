"""Screen published vessel points for NL; never approve labels or upstream splits."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from shapely.geometry import Point, box, mapping

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.region import REGION_PATH, load_region

REVISION = "60b003d059937a9fcddbf6e1c9b64c00c33c0279"
METADATA_SHA256 = "d3a8bbd6c6685a6f8bb685f122b37fda613649d6da4c450e3240cace353ca2ac"
SOURCE = f"https://github.com/allenai/vessel-detection-sentinels/tree/{REVISION}"


def pixel_lonlat(column: float, row: float, zoom: int) -> tuple[float, float]:
    """Upstream global pixel grid, not window-local or raw SAR pixel offsets."""
    if zoom != 13 or not math.isfinite(column) or not math.isfinite(row):
        raise ValueError("Finite zoom-13 Web Mercator pixel coordinates required")
    world = 512 * 2**zoom
    if not 0 <= column <= world or not 0 <= row <= world:
        raise ValueError("Point outside global Web Mercator grid")
    return column / world * 360 - 180, math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * row / world)))
    )


def screen(database: Path, region: Any) -> dict[str, Any]:
    """Read-only candidate inventory, including empty windows and split leaks.

    This low-level function accepts test fixtures. The public CLI additionally
    requires the pinned source checksum. Every selected area must be wholly NL;
    no label, including a published vessel point, becomes metric-ready here.
    """
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tasks = connection.execute(
            "SELECT d.id FROM datasets d JOIN collections c ON c.id=d.collection_id "
            "WHERE c.name='sentinel1' AND d.name='vessels' AND d.task='point'"
        ).fetchall()
        if len(tasks) != 1:
            raise ValueError("Expected one Sentinel-1 vessel-point dataset")
        dataset_id = tasks[0]["id"]
        rows = connection.execute(
            "SELECT w.id,w.column,w.row,w.width,w.height,w.hidden,w.split,"
            "i.name,i.uuid,i.zoom,i.projection,i.hidden AS image_hidden "
            "FROM windows w JOIN images i ON i.id=w.image_id "
            "WHERE w.dataset_id=? ORDER BY w.id",
            (dataset_id,),
        ).fetchall()
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        # One linear label scan: a correlated count per window is quadratic.
        labels = connection.execute(
            "SELECT l.id,l.window_id,l.column,l.row,l.properties FROM labels l "
            "JOIN windows w ON w.id=l.window_id WHERE w.dataset_id=? ORDER BY l.id",
            (dataset_id,),
        ).fetchall()
        for label in labels:
            grouped[label["window_id"]].append(dict(label))

    cases = []
    excluded: Counter[str] = Counter()
    acquisitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["hidden"] or row["image_hidden"]:
            excluded["hidden"] += 1
            continue
        if str(row["projection"]).lower() != "epsg:3857":
            raise ValueError("Unsupported published point projection")
        if row["width"] <= 0 or row["height"] <= 0:
            raise ValueError("Positive complete window dimensions required")
        left, top = pixel_lonlat(row["column"], row["row"], row["zoom"])
        right, bottom = pixel_lonlat(
            row["column"] + row["width"], row["row"] + row["height"], row["zoom"]
        )
        geometry = box(left, bottom, right, top)
        if not region.covers(geometry):
            excluded["not_wholly_inside_nl"] += 1
            continue
        group = native_acquisition_group(row["name"])
        if group is None or "_1SDV_" not in row["name"]:
            raise ValueError("Expected resolved Sentinel-1 VV/VH acquisition")
        points = []
        for label in grouped[row["id"]]:
            properties = json.loads(label["properties"] or "{}")
            if not isinstance(properties, dict):
                raise ValueError("Invalid point properties")
            if "OnKey" in properties:
                excluded["annotation_helper_point"] += 1
                continue
            lon, lat = pixel_lonlat(label["column"], label["row"], row["zoom"])
            if not (
                row["column"] <= label["column"] < row["column"] + row["width"]
                and row["row"] <= label["row"] < row["row"] + row["height"]
                and geometry.covers(Point(lon, lat))
            ):
                raise ValueError("Published point outside its selected window")
            points.append(
                {
                    "source_label_id": label["id"],
                    "lon": lon,
                    "lat": lat,
                    "class": "vessel",
                    "class_authority": "published_source_not_local_verification",
                    "metric_ready": False,
                }
            )
        acquisitions[group].add(row["split"])
        cases.append(
            {
                "source_window_id": row["id"],
                "source_name": row["name"],
                "source_uuid": row["uuid"],
                "acquisition_group": group,
                "polarizations": ["vv", "vh"],
                "upstream_split": row["split"],
                "geometry": mapping(geometry),
                "global_pixel_window": [
                    row["column"],
                    row["row"],
                    row["width"],
                    row["height"],
                ],
                "points": points,
                "annotation_completeness": "unverified_locally",
                "empty_window_is_confirmed_negative": False,
                "metric_ready": False,
            }
        )
    return {
        "status": "candidate_screen_only",
        "adopted_as_training_data": False,
        "independent_local_review": False,
        "threshold_selection_allowed": False,
        "upstream_splits_adopted": False,
        "windows_wholly_inside_nl": len(cases),
        "published_vessel_points": sum(len(c["points"]) for c in cases),
        "empty_windows_not_confirmed_negatives": sum(not c["points"] for c in cases),
        "acquisitions": len(acquisitions),
        "acquisitions_with_multiple_upstream_splits": sorted(
            group for group, splits in acquisitions.items() if len(splits) > 1
        ),
        "excluded": dict(excluded),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=REPO_ROOT / "data/model_research/allenai_audit/metadata.sqlite3",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("New project-contained output path required")
    if sha256_file(args.database) != METADATA_SHA256:
        raise ValueError("Metadata does not match the pinned published database")
    result = screen(args.database, load_region())
    result["provenance"] = {
        "source": SOURCE,
        "source_revision": REVISION,
        "metadata_sha256": METADATA_SHA256,
        "region_sha256": sha256_file(REGION_PATH),
        "screen_code_sha256": sha256_file(__file__),
        "repository_licence": "Apache-2.0; not a separately verified data grant",
        "label_kind": "vessel centres, not hull outlines or exhaustive local truth",
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "windows_wholly_inside_nl",
                    "published_vessel_points",
                    "empty_windows_not_confirmed_negatives",
                    "acquisitions",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
