"""Materialize a detector-blind, calibrated SAR annotation pilot.

Public catalogue metadata are frozen locally; authentication is needed only
for missing SAFE downloads. No detector, model weights or AIS truth are used.
Existing source products are never overwritten. The locked test receives QA
chips but no model evaluation. Derived imagery is stored under ignored data/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from PIL import __version__ as pillow_version
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer, rowcol
from rasterio.windows import Window
from shapely.geometry import Point, Polygon, mapping, shape

from agents.acquisition import product_id
from agents.ais_validation import parse_utc
from agents.artifacts import sha256_file, write_json
from agents.calibration import NODATA_DB, Calibrator
from agents.geo import Georeferencer
from agents.region import load_region
from agents.scene_watch import ais_rows_near

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
LICENCE = "https://cds.climate.copernicus.eu/licences/ec-sentinel"


def supporting_ais(
    product: dict[str, Any], rois: list[dict[str, Any]], data: Path
) -> dict[str, Any]:
    """Freeze available observations; never interpret their absence as a label."""
    target = data / "ais" / f"{product['Id']}.json"
    if target.exists():
        snapshot = json.loads(target.read_text(encoding="utf-8"))
    else:
        archive = ROOT / "data/raw/ais_stream"
        acquired = parse_utc(product["ContentDate"]["Start"])
        if acquired is None:
            raise ValueError("Invalid acquisition time")
        paths = list(archive.glob(f"ais_stream_{acquired.date().isoformat()}*.csv"))
        # Fail on unreadable archives rather than calling them empty coverage.
        for path in paths:
            with path.open("rb") as stream:
                stream.read(1)
        rows = ais_rows_near(str(archive), product["ContentDate"]["Start"])
        snapshot = {
            "product_id": product["Id"],
            "window_minutes": 5,
            "archive_files_at_snapshot": [p.name for p in sorted(paths)],
            "rows": rows,
        }
        write_json(target, snapshot)
    footprint = shape(product["GeoFootprint"])
    points = [Point(float(r["lon"]), float(r["lat"])) for r in snapshot["rows"]]
    in_swath = [p for p in points if footprint.covers(p) and load_region().covers(p)]
    counts = {
        r["id"]: sum(shape(r["geometry"]).covers(p) for p in in_swath) for r in rois
    }
    state = (
        "missing_archive"
        if not snapshot["archive_files_at_snapshot"]
        else (
            "no_co_temporal_rows"
            if not points
            else (
                "outside_swath"
                if not in_swath
                else (
                    "support_available"
                    if any(counts.values())
                    else "outside_selected_areas"
                )
            )
        )
    )
    return {
        "status": state,
        "truth_role": "supporting_only",
        "window_minutes": 5,
        "observations_in_window": len(points),
        "observations_in_swath": len(in_swath),
        "observations_by_roi": counts,
        "snapshot": {
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(target),
        },
        "licence": (
            "AISstream provider terms; private supporting evidence, not "
            "redistributed or treated as open-labelled truth."
        ),
    }


def metadata(scene: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Cache exact-name catalogue entries; reject substitutes and invalid paths."""
    name = scene["name"]
    if not re.fullmatch(r"S1[ACD]_IW_GRDH_1SD[VH]_[A-Za-z0-9_]+\.SAFE", name):
        raise ValueError("Expected an exact Sentinel-1 SAFE name")
    target = directory / f"{name}.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
    else:
        response = requests.get(
            CATALOGUE, params={"$filter": f"Name eq '{name}'", "$top": 2}, timeout=60
        )
        response.raise_for_status()
        values = response.json()["value"]
        if len(values) != 1:
            raise ValueError(f"Exact catalogue identity missing/ambiguous: {name}")
        result = values[0]
        write_json(target, result)
    if result["Name"] != name or (
        scene.get("product_id")
        and product_id(result["Id"]) != product_id(scene["product_id"])
    ):
        raise ValueError("Catalogue product substitution")
    product_id(result["Id"])
    return result


def download_product(product: dict[str, Any], raw: Path) -> Path:
    """Download and CRC-check a missing exact product without exposing secrets."""
    target = raw / product["Name"]
    if target.exists():
        if not (target / "manifest.safe").is_file():
            raise ValueError(f"Incomplete cache; preserve and inspect: {target}")
        return target
    load_dotenv(ROOT / ".env")
    user, password = os.getenv("COPERNICUS_USER"), os.getenv("COPERNICUS_PASS")
    if not user or not password:
        raise ValueError("Missing COPERNICUS_USER/PASS for new public SAR downloads")
    response = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/"
        "openid-connect/token",
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user,
            "password": password,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"CDSE authentication failed (HTTP {response.status_code})")
    token = response.json()["access_token"]
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark_download_", dir=raw) as tmp:
        archive = Path(tmp) / "product.zip"
        with requests.get(
            f"{DOWNLOAD}({product_id(product['Id'])})/$value",
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=120,
        ) as source:
            source.raise_for_status()
            with archive.open("wb") as stream:
                for chunk in source.iter_content(1 << 20):
                    stream.write(chunk)
            if source.headers.get("Content-Length") and archive.stat().st_size != int(
                source.headers["Content-Length"]
            ):
                raise ValueError("Incomplete product download")
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                resolved = (Path(tmp) / member.filename).resolve()
                if (
                    not resolved.is_relative_to(Path(tmp).resolve())
                    or "\\" in member.filename
                ):
                    raise ValueError("Unsafe archive path")
            zipped.extractall(tmp)
        extracted = Path(tmp) / product["Name"]
        if not (extracted / "manifest.safe").is_file():
            raise ValueError("Downloaded SAFE manifest missing")
        extracted.rename(target)
    return target


def preview(db: np.ndarray[Any, Any], valid: np.ndarray[Any, Any]) -> Image.Image:
    """Fixed [-30,+5] dB stretch, not per-chip contrast that hides conditions."""
    pixels = np.clip((db + 30) / 35 * 255, 0, 255).astype("uint8")
    pixels[~valid] = 0
    return Image.fromarray(pixels).convert("RGB")


def chip(roi: dict[str, Any], paths: list[Path], directory: Path) -> dict[str, Any]:
    """Read a fixed native-radar window, preserving offsets and GCP geometry."""
    size = roi["size"]
    if not isinstance(size, int) or size < 256 or size % 256:
        raise ValueError("ROI size must be a positive multiple of 256")
    with rasterio.open(paths[0]) as source:
        gcps, gcp_crs = source.gcps
        lon, lat = roi["center"]
        if gcps:
            if str(gcp_crs) != "EPSG:4326":
                raise ValueError("Expected WGS84 Sentinel-1 GCPs")
            with GCPTransformer(gcps, tps=True) as converter:
                rows, cols = converter.rowcol([lon], [lat])
                row, col = int(rows[0]) - size // 2, int(cols[0]) - size // 2
        else:
            if str(source.crs) != "EPSG:4326":
                raise ValueError("Unsupported affine source CRS")
            r, c = rowcol(source.transform, lon, lat)
            row, col = int(r) - size // 2, int(c) - size // 2
        if (
            row < 0
            or col < 0
            or row + size > source.height
            or col + size > source.width
        ):
            raise ValueError(f"ROI outside raster: {roi['id']}")
        window = Window(col, row, size, size)
        with Georeferencer.from_dataset(source) as geo:
            ring: list[list[float]] = []
            for r0, c0, r1, c1 in [
                (row, col, row, col + size),
                (row, col + size, row + size, col + size),
                (row + size, col + size, row + size, col),
                (row + size, col, row, col),
            ]:
                xs, ys = geo.xy(np.linspace(r0, r1, 17), np.linspace(c0, c1, 17))
                ring.extend([[float(x), float(y)] for x, y in zip(xs, ys)])
        polygon = Polygon(ring)
        if not polygon.is_valid or not load_region().covers(polygon):
            raise ValueError(f"ROI outside NL study boundary: {roi['id']}")
        shifted = [
            GroundControlPoint(row=g.row - row, col=g.col - col, x=g.x, y=g.y, z=g.z)
            for g in gcps
        ]
        affine = source.window_transform(window)
        dimensions = (source.width, source.height)
    arrays, valid_arrays, calibration = [], [], []
    for path in paths:
        cal = Calibrator.for_measurement(str(path))
        if cal is None:
            raise ValueError(f"Calibration LUT missing: {path.name}")
        with rasterio.open(path) as source:
            band_gcps, band_crs = source.gcps
            if (
                (source.width, source.height) != dimensions
                or (
                    [(g.row, g.col, g.x, g.y) for g in band_gcps]
                    != [(g.row, g.col, g.x, g.y) for g in gcps]
                )
                or band_crs != gcp_crs
                or source.window_transform(window) != affine
            ):
                raise ValueError("Band geometry/alignment mismatch")
            values = source.read(1, window=window, masked=True)
            valid = (
                ~np.ma.getmaskarray(values)
                & np.isfinite(values.data)
                & (values.data > 0)
            )
            db = cal.to_sigma0_db(values.filled(0), row, col).astype("float32")
            db[~valid] = NODATA_DB
            arrays.append(db)
            valid_arrays.append(valid)
            calibration.append(
                {
                    "path": Path(cal.xml_path).relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(cal.xml_path),
                }
            )
    valid = np.logical_and.reduce(valid_arrays)
    if not valid.all():
        # Pilot ROIs have no nodata holes, making the reviewed-area denominator
        # exact. Do not turn missing pixels into purported negative examples.
        raise ValueError(
            f"ROI has nodata; select a different complete area: {roi['id']}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    tiff = directory / "sigma0.tif"
    with rasterio.open(
        tiff,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=2,
        dtype="float32",
        nodata=NODATA_DB,
        compress="deflate",
    ) as out:
        if shifted:
            out.gcps = (shifted, gcp_crs)
        else:
            out.crs, out.transform = "EPSG:4326", affine
        for i, array in enumerate(arrays, 1):
            out.write(array, i)
            out.set_band_description(i, "sigma0_db_" + paths[i - 1].name.split("-")[3])
    sheets = []
    for band, array in enumerate(arrays):
        image = preview(array, valid)
        path = directory / f"band_{band+1}.png"
        image.save(path)
        sheets.append(
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        )
    # Full-resolution, unannotated cells prevent overview-only review of tiny
    # objects. No predictions/AIS seeds are shown in the first-pass images.
    cells = []
    canvas = Image.new("RGB", (size + 100, size + 50), "#182128")
    canvas.paste(preview(arrays[0], valid), (100, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), roi["id"] + " | co-pol | native radar pixels", fill="white")
    for cy in range(0, size, 256):
        for cx in range(0, size, 256):
            cell_id = f"{cx//256}_{cy//256}"
            cells.append({"id": cell_id, "bbox_px": [cx, cy, cx + 256, cy + 256]})
            draw.rectangle((cx + 100, cy + 50, cx + 355, cy + 305), outline="#388d83")
            draw.text((cx + 103, cy + 53), cell_id, fill="#00ffff")
    overview = directory / "review_grid.png"
    canvas.save(overview)
    return {
        **roi,
        "window": [col, row, size, size],
        "geometry": mapping(polygon),
        "valid_pixels": int(valid.sum()),
        "pixels": size * size,
        "cells": cells,
        "calibration": calibration,
        "chip": {
            "path": tiff.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(tiff),
        },
        "previews": sheets,
        "review_grid": {
            "path": overview.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(overview),
        },
        "db_percentiles": [np.percentile(a, [5, 50, 95]).tolist() for a in arrays],
    }


def build(selection: Path, output: Path) -> None:
    if (output.parent / "release.json").exists():
        raise ValueError(
            "Released dataset is immutable; materialize a copy in a "
            "new directory/version"
        )
    chosen = json.loads(selection.read_text(encoding="utf-8"))
    data = ROOT / "data" / "benchmarks" / "nl" / chosen["dataset_version"]
    data.mkdir(parents=True, exist_ok=True)
    frozen = data / "selection.json"
    selection_hash = sha256_file(selection)
    if frozen.exists() and sha256_file(frozen) != selection_hash:
        raise ValueError(
            "Selection changed: create a new dataset version, never rewrite a lock"
        )
    # Byte-identical copy is the immutable selection contract.
    if not frozen.exists():
        frozen.write_bytes(selection.read_bytes())
    licence = data / "sentinel_licence.html"
    if not licence.exists():
        response = requests.get(LICENCE, timeout=60)
        response.raise_for_status()
        licence.write_bytes(response.content)
    scenes = []
    for scene in chosen["scenes"]:
        product = metadata(scene, data / "catalogue")
        print("Materializing", product["Name"], flush=True)
        safe = download_product(product, ROOT / "data" / "raw")
        pair = ("hh", "hv") if "_1SDH_" in product["Name"] else ("vv", "vh")
        paths = []
        for pol in pair:
            matches = [
                p
                for p in (safe / "measurement").glob("*.tif*")
                if f"-{pol}-" in p.name.lower()
            ]
            if len(matches) != 1:
                raise ValueError("Missing/ambiguous polarization measurement")
            paths.append(matches[0])
        sources = [
            {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256_file(p)}
            for p in [safe / "manifest.safe", *paths]
        ]
        rois = [chip(r, paths, data / r["id"]) for r in scene["rois"]]
        tokens = product["Name"].split("_")
        # Absolute orbit + datatake group also catches reprocessing, COG copies
        # and adjacent frames from the same pass, not just one catalogue UUID.
        group = "_".join([tokens[0], tokens[6], tokens[7]])
        cat_path = data / "catalogue" / f"{product['Name']}.json"
        scenes.append(
            {
                **scene,
                "rois": rois,
                "product_id": product["Id"],
                "acquisition_group": group,
                "acquisition_time": product["ContentDate"]["Start"],
                "acquisition_end": product["ContentDate"]["End"],
                "footprint": product["GeoFootprint"],
                "polarizations": list(pair),
                "sources": sources,
                "catalogue": {
                    "path": cat_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(cat_path),
                },
                "ais": supporting_ais(product, rois, data),
            }
        )
    manifest = {
        "dataset_version": chosen["dataset_version"],
        "status": chosen["status"],
        "selection_sha256": selection_hash,
        "separation_m": chosen["separation_m"],
        "licence": {
            "url": LICENCE,
            "snapshot": licence.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(licence),
            "attribution": "Contains modified Copernicus Sentinel data (2025, 2026).",
        },
        "processing": {
            "calibration": (
                "sigma0 = DN^2 / LUT^2; 10log10; no terrain/noise correction"
            ),
            "geolocation": "native radar grid, WGS84 GCP thin plate spline",
            "preview_stretch_db": [-30, 5],
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "gdal": rasterio.__gdal_version__,
                "pillow": pillow_version,
            },
            "regime_labels": (
                "Selection intent, not independently validated physical "
                "land/water/ice truth."
            ),
            "builder_sha256": sha256_file(Path(__file__)),
            "calibrator_sha256": sha256_file(ROOT / "agents/calibration.py"),
        },
        "scenes": scenes,
    }
    write_json(output, manifest)
    print("Saved", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "datasets/nl_benchmark/v0.1.0/selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets/nl_benchmark/v0.1.0/manifest.json",
    )
    args = parser.parse_args()
    build(args.selection, args.output)
