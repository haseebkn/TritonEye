"""Build fresh NL-only alignment QA evidence from pinned public metadata.

No detector inference, AIS identities, annotation adoption or test pixels.
Cached exact SAFE inputs are hash-checked; no downloads or billing occur here.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Geod, Transformer
from rasterio.transform import GCPTransformer
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping

from agents.allenai_archive import ETAG, GENERATION, SIZE, URL
from agents.allenai_subset import (
    RESOLUTION,
    archive_crosswalk,
    bounded_patch_center,
    publisher_xy,
    warp_publisher_window,
)
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.geo import Georeferencer
from agents.nl_benchmark import require_development_scene
from agents.public_vessel_labels import METADATA_SHA256, REVISION, screen
from agents.region import REGION_PATH, load_region
from scripts.build_nl_benchmark import chip

ROOT = REPO_ROOT
AUDIT = ROOT / "data/model_research/allenai_audit"
WORK = ROOT / "data/model_research/allenai_nl_v010"
INVENTORY_SHA256 = "46a85a869854baa68105b357abd0c0c8a0cfa18a0eb35f438cdd6f01358d8a52"


def fresh_output(output: Path) -> Path:
    """Normalize relative CLI paths before constructing artifact references."""
    output = output.resolve()
    if output.exists() or not output.is_relative_to(WORK.resolve()):
        raise ValueError("Fresh output inside the versioned work folder required")
    if (ROOT / "datasets/allenai_nl/v0.1.0/release.json").exists():
        raise ValueError("Published version is immutable; a new version is required")
    return output


def build(output: Path) -> None:
    output = fresh_output(output)
    database = AUDIT / "metadata.sqlite3"
    if sha256_file(database) != METADATA_SHA256:
        raise ValueError("Pinned database checksum mismatch")
    index_path = WORK / "cc_archive_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (index["url"], index["generation"], index["bytes"], index["etag"]) != (
        URL,
        GENERATION,
        SIZE,
        ETAG,
    ):
        raise ValueError("Archive does not match pinned publisher object")
    coverage = archive_crosswalk(index, database, load_region())
    selection = screen(database, load_region())
    inventory_path = AUDIT / "sar_download_inventory_20261006.json"
    if sha256_file(inventory_path) != INVENTORY_SHA256:
        raise ValueError("Pinned source inventory mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        images = {r["name"]: dict(r) for r in db.execute("SELECT * FROM images")}
        labels = {r["id"]: dict(r) for r in db.execute("SELECT * FROM labels")}
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)
    geod = Geod(ellps="WGS84")
    cases, point_checks, sources, blocked = [], [], [], []
    for product in inventory["products"]:
        selected = [
            c for c in selection["cases"] if c["source_name"] == product["name"]
        ]
        # Guard all proposed source extents BEFORE source file/pixel access.
        for case in selected:
            require_development_scene(
                product["product_id"],
                {"name": product["name"], "study_roi": case["geometry"]},
            )
        for item in product["files"]:
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Cached SAFE checksum mismatch")
        paths = [
            ROOT
            / next(
                f["path"]
                for f in product["files"]
                if "/measurement/" in f["path"] and f"-{band}-" in f["path"]
            )
            for band in ("vv", "vh")
        ]
        original = images[product["name"]]
        with rasterio.open(paths[0]) as src:
            gcps, crs = src.gcps
            affine, width, height = calculate_default_transform(
                crs,
                "EPSG:3857",
                src.width,
                src.height,
                gcps=gcps,
                resolution=RESOLUTION,
            )
            if (
                abs(width - original["width"]) > 1
                or abs(height - original["height"]) > 1
            ):
                raise ValueError("Reconstructed publisher grid dimensions disagree")
            origin = (original["column"], original["row"])
            # Origin agreement is required, not inferred from a nearby target.
            publisher_xy(affine, origin, origin[0], origin[1])
            with (
                GCPTransformer(gcps, tps=False) as inverse,
                Georeferencer.from_dataset(src) as geo,
            ):
                for case in selected:
                    col, row, w, h = case["global_pixel_window"]
                    window = Window(col - origin[0], row - origin[1], w, h)
                    if (
                        window.col_off < 0
                        or window.row_off < 0
                        or window.col_off + w > original["width"]
                        or window.row_off + h > original["height"]
                    ):
                        raise ValueError("Published window outside source grid")
                    ring = []
                    for a, b, c, d in [
                        (col, row, col + w, row),
                        (col + w, row, col + w, row + h),
                        (col + w, row + h, col, row + h),
                        (col, row + h, col, row),
                    ]:
                        for x, y in zip(np.linspace(a, c, 17), np.linspace(b, d, 17)):
                            lon, lat = to_wgs.transform(
                                *publisher_xy(affine, origin, float(x), float(y))
                            )
                            rr, cc = inverse.rowcol([lon], [lat], op=lambda v: v)
                            xs, ys = geo.xy(rr, cc)
                            ring.append([float(xs[0]), float(ys[0])])
                    physical = Polygon(ring)
                    if not physical.is_valid or not load_region().covers(physical):
                        blocked.append(
                            {
                                "window_id": case["source_window_id"],
                                "reason": "reconstructed_extent_not_wholly_nl",
                            }
                        )
                        continue
                    require_development_scene(
                        product["product_id"],
                        {"name": product["name"], "study_roi": mapping(physical)},
                    )
                    directory = output / "windows" / str(case["source_window_id"])
                    directory.mkdir(parents=True, exist_ok=False)
                    previews = []
                    for band, path in zip(("vv", "vh"), paths):
                        with rasterio.open(path) as band_src:
                            raw, valid = warp_publisher_window(band_src, affine, window)
                        if not valid.any():
                            raise ValueError("Empty publisher warp; not a usable area")
                        image = Image.fromarray(np.clip(raw, 0, 255).astype("uint8"))
                        target = directory / f"publisher_dn_{band}.png"
                        image.save(target)
                        mask_path = directory / f"valid_{band}.png"
                        Image.fromarray(valid.astype("uint8") * 255).save(mask_path)
                        previews.append(
                            {
                                "path": target.relative_to(ROOT).as_posix(),
                                "sha256": sha256_file(target),
                                "band": band,
                                "valid_pixels": int(valid.sum()),
                                "pixels": int(w * h),
                                "valid_mask": {
                                    "path": mask_path.relative_to(ROOT).as_posix(),
                                    "sha256": sha256_file(mask_path),
                                },
                            }
                        )
                    checks = []
                    for point in case["points"]:
                        published = labels[point["source_label_id"]]
                        lon, lat = to_wgs.transform(
                            *publisher_xy(
                                affine, origin, published["column"], published["row"]
                            )
                        )
                        rr, cc = inverse.rowcol([lon], [lat], op=lambda v: v)
                        xs, ys = geo.xy(rr, cc)
                        native_lon, native_lat = float(xs[0]), float(ys[0])
                        centre_col, centre_row = bounded_patch_center(
                            float(cc[0]), float(rr[0]), src.width, src.height
                        )
                        cx, cy = geo.xy([centre_row], [centre_col])
                        crop_center = [float(cx[0]), float(cy[0])]
                        # The native patch can extend beyond the publisher area.
                        # Check its exact proposed extent before reading pixels.
                        with GCPTransformer(gcps, tps=True) as tps:
                            pr, pc = tps.rowcol([crop_center[0]], [crop_center[1]])
                        r0, c0 = int(pr[0]) - 128, int(pc[0]) - 128
                        patch_ring = []
                        for ra, ca, rb, cb in [
                            (r0, c0, r0, c0 + 256),
                            (r0, c0 + 256, r0 + 256, c0 + 256),
                            (r0 + 256, c0 + 256, r0 + 256, c0),
                            (r0 + 256, c0, r0, c0),
                        ]:
                            xx, yy = geo.xy(
                                np.linspace(ra, rb, 17), np.linspace(ca, cb, 17)
                            )
                            patch_ring.extend(
                                [[float(a), float(b)] for a, b in zip(xx, yy)]
                            )
                        require_development_scene(
                            product["product_id"],
                            {
                                "name": product["name"],
                                "study_roi": mapping(Polygon(patch_ring)),
                            },
                        )
                        diagnostic = geod.inv(
                            point["lon"], point["lat"], native_lon, native_lat
                        )[2]
                        patch = chip(
                            {
                                "id": str(point["source_label_id"]),
                                "size": 256,
                                "center": crop_center,
                            },
                            paths,
                            output / "points" / str(point["source_label_id"]),
                        )
                        px = float(cc[0]) - patch["window"][0]
                        py = float(rr[0]) - patch["window"][1]
                        check = {
                            "source_label_id": point["source_label_id"],
                            "native_pixel": [float(cc[0]), float(rr[0])],
                            "point_px": [px, py],
                            "crop_center_edge_clamped": (
                                centre_col != float(cc[0]) or centre_row != float(rr[0])
                            ),
                            "grid_vs_native_tps_displacement_m": diagnostic,
                            "visual_status": "unreviewed_not_confirmed_vessel",
                            "patch": patch,
                        }
                        checks.append(check)
                        point_checks.append(check)
                    cases.append(
                        {
                            **case,
                            "physical_geometry": mapping(physical),
                            "publisher_previews": previews,
                            "alignment_checks": checks,
                        }
                    )
            sources.append(
                {
                    **product,
                    "original_image_uuid": original["uuid"],
                    "publisher_grid_affine": list(affine)[:6],
                    "stored_dimensions": [original["width"], original["height"]],
                    "suggested_dimensions": [width, height],
                    "source_origin_match": True,
                }
            )
        print(f"Prepared {product['name']}: {len(selected)} windows", flush=True)
    # Point-indexed sheets are second-pass alignment QA, not blind annotation.
    sheet = Image.new(
        "RGB", (4 * 280, math.ceil(len(point_checks) / 4) * 310), "#15202b"
    )
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(point_checks):
        image = Image.open(ROOT / record["patch"]["previews"][0]["path"]).convert("RGB")
        x, y = (i % 4) * 280, (i // 4) * 310
        sheet.paste(image, (x + 12, y + 30))
        px, py = record["point_px"]
        draw.ellipse(
            (x + 12 + px - 8, y + 30 + py - 8, x + 12 + px + 8, y + 30 + py + 8),
            outline="#ffb000",
            width=1,
        )
        draw.text(
            (x + 12, y + 8),
            f"source {record['source_label_id']} | VV sigma0",
            fill="white",
        )
    sheet.save(output / "point_alignment_sheet.png")
    artifacts = [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "version": "0.1.0",
        "status": "quarantined_qa",
        "training_allowed": False,
        "evaluation_allowed": False,
        "independent_local_review": False,
        "annotation_completeness": "unverified",
        "upstream_splits_adopted": False,
        "metadata_sha256": METADATA_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "original_source_revision": REVISION,
        "archive_index_sha256": sha256_file(index_path),
        "cc_archive_crosswalk": coverage,
        "region_sha256": sha256_file(REGION_PATH),
        "python": platform.python_version(),
        "rasterio": rasterio.__version__,
        "gdal": rasterio.__gdal_version__,
        "code_sha256": sha256_file(__file__),
        "geometry_code_sha256": sha256_file(ROOT / "agents/allenai_subset.py"),
        "sources": sources,
        "cases": cases,
        "blocked": blocked,
        "artifacts": artifacts,
        "whole_nl_candidate_windows": selection["windows_wholly_inside_nl"],
        "whole_nl_candidate_points": selection["published_vessel_points"],
        "prepared_windows": len(cases),
        "prepared_points": len(point_checks),
        "empty_windows_not_confirmed_negative": sum(not c["points"] for c in cases),
        "alignment_limits": [
            "GDAL reconstruction, not byte-identical original publisher tiles",
            "Source-edge nodata is explicitly masked; it is not negative supervision",
            "Suggested dimensions differ by at most one edge pixel "
            "from stored publisher dimensions",
            "GDAL default polynomial GCP inverse is compared to TritonEye TPS, "
            "not surveyed absolute accuracy",
            "Publisher DN clipping is not calibrated sigma0; "
            "native QA patches are separately calibrated",
            "Integer click convention has unresolved subpixel uncertainty; "
            "no snapping or fitted label shifts",
            "No exhaustive independent annotation review; "
            "empty windows are not ground-truth negatives",
        ],
        "licence": {
            "cc_by_4_archive": "explicit publisher CC BY 4.0 grant for this archive",
            "older_database": "Repository Apache-2.0; annotation-specific "
            "scope not separately verified",
            "grant_transferred": False,
            "imagery": "Copernicus Sentinel data terms; not the annotation grant",
        },
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {k: manifest[k] for k in ["prepared_windows", "prepared_points", "status"]}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
