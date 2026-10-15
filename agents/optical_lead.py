"""Download and render a fixed optical lead without model scores or class admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping
from shapely.geometry import shape as geometry_shape
from shapely.ops import transform as geometry_transform

from agents.artifacts import REPO_ROOT
from agents.cd_reference import GUARDS
from agents.hh_hv_reservation import save
from agents.nl_benchmark import require_development_scene
from agents.reserved_sar import DOWNLOAD, Cdse
from agents.source_followup import fresh
from agents.source_reservation import checked, prior_assignments, projected, ref

POLICY = REPO_ROOT / "configs/reservations/cd_followup_v010.json"
SCOPE = REPO_ROOT / "configs/reservations/optical_inspection_v010.json"


def require_permitted_geometry(
    geometry: dict[str, Any], priors: list[dict[str, Any]], separation: float
) -> float:
    distances = [
        projected(geometry).distance(projected(p["geometry"]))
        for p in priors
        if p["quarantine"] or p["split"] == "test"
    ]
    if not distances or min(distances) < separation:
        raise ValueError("Optical scope overlaps protected separation")
    return float(min(distances))


def window_geometry(window: Window, transform: Any, crs: str) -> dict[str, Any]:
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    geometry: dict[str, Any] = mapping(
        geometry_transform(
            inverse.transform, box(*rasterio.windows.bounds(window, transform))
        )
    )
    return geometry


def window_contains(outer: Window, inner: Window) -> bool:
    return bool(
        inner.col_off >= outer.col_off
        and inner.row_off >= outer.row_off
        and inner.col_off + inner.width <= outer.col_off + outer.width
        and inner.row_off + inner.height <= outer.row_off + outer.height
    )


def classification_patch(
    classification_path: Path, mask_window: Window, centre: tuple[float, float]
) -> np.ndarray[Any, Any]:
    central_window = Window(int(centre[0]) - 2, int(centre[1]) - 2, 5, 5)
    if not window_contains(mask_window, central_window):
        raise ValueError("Classification target patch outside permitted window")
    with rasterio.open(classification_path) as ds:
        patch: np.ndarray[Any, Any] = ds.read(window=central_window)
        return patch


def inspection_window(
    transform: Any, crs: str, shape: tuple[int, int]
) -> tuple[Any, float, list[dict[str, Any]], float]:
    rules = json.loads(SCOPE.read_text())
    prepared = json.loads(checked(rules["prepared"]).read_text())
    product, area = next(
        (p, a)
        for p in prepared["products"]
        for a in p["areas"]
        if a["id"] == rules["area_id"]
    )
    if area["split"] != "train":
        raise ValueError("Original training assignment required")
    project = Transformer.from_crs(4326, crs, always_xy=True)
    extent = geometry_transform(project.transform, geometry_shape(area["geometry"]))
    requested = from_bounds(*extent.bounds, transform)
    col, row = int(np.floor(requested.col_off)), int(np.floor(requested.row_off))
    window = Window(
        col,
        row,
        int(np.ceil(requested.col_off + requested.width)) - col,
        int(np.ceil(requested.row_off + requested.height)) - row,
    ).intersection(Window(0, 0, shape[1], shape[0]))
    read_geometry = window_geometry(window, transform, crs)
    require_development_scene(
        product["product_id"],
        {"name": product["name"], "split": "train", "study_roi": read_geometry},
    )
    priors, _, _ = prior_assignments()
    minimum_separation = rules["minimum_protected_separation_m"]
    distance = require_permitted_geometry(read_geometry, priors, minimum_separation)
    return window, distance, priors, minimum_separation


def source() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rules = json.loads(POLICY.read_text())
    search = json.loads(checked(rules["optical_search"]).read_text())
    target = next(t for t in search["targets"] if t["id"] == rules["target_id"])
    product = next(
        p
        for p in target["sentinel2"]["products"]
        if p["id"] == rules["optical_product_id"]
    )
    if (
        product["name"] != rules["optical_product_name"]
        or not product["covers_original_position"]
    ):
        raise ValueError("Exact previously discovered covering lead required")
    return rules, target, product


def download(output: Path) -> None:
    rules, target, product = source()
    fresh(output)
    client = Cdse()
    base = f"{DOWNLOAD}({product['id']})/Nodes({product['name']})"
    requests: list[dict[str, Any]] = []

    def fetch(url: str, name: str, size: int | None = None) -> dict[str, str]:
        if size is not None and not 0 < size <= rules["maximum_asset_bytes"]:
            raise ValueError("Optical asset exceeds frozen budget")
        path = output / name
        if size is None or size <= 8 * 1024**2:
            path.write_bytes(client.fetch(url))
        else:
            with path.open("xb") as stream:
                for start in range(0, size, 8 * 1024**2):
                    stream.write(
                        client.fetch(url, start, min(8 * 1024**2, size - start), size)
                    )
        if size is not None and path.stat().st_size != size:
            raise ValueError("Optical asset length differs")
        if client.bytes > rules["maximum_download_bytes"]:
            raise ValueError("Optical transfer budget exceeded")
        receipt = ref(path)
        requests.append({"url": url, "bytes": path.stat().st_size, "artifact": receipt})
        return receipt

    metadata = fetch(base + "/Nodes(MTD_MSIL1C.xml)/$value", "product.xml")
    manifest = fetch(base + "/Nodes(manifest.safe)/$value", "manifest.safe")
    listing = fetch(base + "/Nodes(GRANULE)/Nodes", "granules.json")
    granules = json.loads(checked(listing).read_text())["result"]
    if len(granules) != 1 or "T22TCT" not in granules[0]["Name"]:
        raise ValueError("Single exact tile required")
    tile = base + f"/Nodes(GRANULE)/Nodes({granules[0]['Name']})"
    tile_metadata = fetch(tile + "/Nodes(MTD_TL.xml)/$value", "tile.xml")
    image_listing = fetch(tile + "/Nodes(IMG_DATA)/Nodes", "images.json")
    quality_listing = fetch(tile + "/Nodes(QI_DATA)/Nodes", "quality.json")
    images = json.loads(checked(image_listing).read_text())["result"]
    quality = json.loads(checked(quality_listing).read_text())["result"]
    assets = {}
    for band in rules["optical_bands"]:
        item = next(i for i in images if i["Name"].endswith(f"_{band}.jp2"))
        assets[band] = fetch(
            tile + f"/Nodes(IMG_DATA)/Nodes({item['Name']})/$value",
            band + ".jp2",
            item["ContentLength"],
        )
        print(
            json.dumps({"downloaded": band, "bytes": item["ContentLength"]}), flush=True
        )
    item = next(i for i in quality if i["Name"] == "MSK_CLASSI_B00.jp2")
    assets["classification"] = fetch(
        tile + f"/Nodes(QI_DATA)/Nodes({item['Name']})/$value",
        "classification.jp2",
        item["ContentLength"],
    )
    save(
        output / "download.json",
        {
            "policy": ref(POLICY),
            "source_search": rules["optical_search"],
            "product": product,
            "target": {k: target[k] for k in ("id", "lon", "lat", "native_time_utc")},
            "metadata": metadata,
            "manifest": manifest,
            "tile_metadata": tile_metadata,
            "assets": assets,
            "requests": requests,
            "network_bytes": client.bytes,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            **GUARDS,
        },
    )


def geocoding(xml: Path, resolution: int) -> tuple[Any, str, tuple[int, int]]:
    root = ET.parse(xml)
    geo = root.find(f".//Geoposition[@resolution='{resolution}']")
    size = root.find(f".//Size[@resolution='{resolution}']")
    if geo is None or size is None:
        raise ValueError("Native tile geocoding required")
    crs = root.findtext(".//HORIZONTAL_CS_CODE")
    transform = rasterio.Affine(
        float(geo.findtext("XDIM", "nan")),
        0,
        float(geo.findtext("ULX", "nan")),
        0,
        float(geo.findtext("YDIM", "nan")),
        float(geo.findtext("ULY", "nan")),
    )
    if not crs or transform.a != resolution or transform.e != -resolution:
        raise ValueError("Expected native north-up tile required")
    return (
        transform,
        crs,
        (int(size.findtext("NROWS", "0")), int(size.findtext("NCOLS", "0"))),
    )


def radiometry(xml: Path) -> tuple[float, dict[str, float]]:
    root = ET.parse(xml)
    q = float(root.findtext(".//QUANTIFICATION_VALUE", "nan"))
    offsets = {
        int(e.attrib["band_id"]): float(e.text or "nan")
        for e in root.findall(".//RADIO_ADD_OFFSET")
    }
    if not np.isfinite(q) or q <= 0 or not offsets:
        raise ValueError("Explicit Collection-1 radiometry required")
    return q, {
        b: offsets[i] for b, i in (("B02", 1), ("B03", 2), ("B04", 3), ("B08", 7))
    }


def render(path: Path, output: Path) -> None:
    rules, target, product = source()
    download = json.loads(path.read_text())
    if (
        download["policy"] != ref(POLICY)
        or download["product"] != product
        or any(download.get(k) is not False for k in GUARDS)
    ):
        raise ValueError("Optical source provenance/guards differ")
    for request in download["requests"]:
        checked(request["artifact"])
    fresh(output)
    result = render_data(path, output)
    save(output / "render.json", result)


def render_data(path: Path, output: Path) -> dict[str, Any]:
    """Replay native pixels and display products into a fresh review directory."""
    rules, target, product = source()
    download = json.loads(path.read_text())
    validate_download(path)
    transform, crs, shape = geocoding(checked(download["tile_metadata"]), 10)
    quantification, offsets = radiometry(checked(download["metadata"]))
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        target["lon"], target["lat"]
    )
    radius = rules["motion_search_base_m"] + rules["motion_search_speed_m_s"] * abs(
        product["offset_s"]
    )
    requested = from_bounds(x - radius, y - radius, x + radius, y + radius, transform)
    col, row = int(np.floor(requested.col_off)), int(np.floor(requested.row_off))
    endcol, endrow = int(np.ceil(requested.col_off + requested.width)), int(
        np.ceil(requested.row_off + requested.height)
    )
    requested = Window(col, row, endcol - col, endrow - row)
    window, protected_distance, priors, minimum_separation = inspection_window(
        transform, crs, shape
    )
    arrays = {}
    for band in rules["optical_bands"]:
        with rasterio.open(checked(download["assets"][band])) as ds:
            if (ds.height, ds.width) != shape or ds.count != 1:
                raise ValueError("Native optical dimensions differ")
            arrays[band] = ds.read(1, window=window)
    values = np.stack([arrays[b] for b in ("B04", "B03", "B02")]).astype("float32")
    valid = np.logical_and.reduce([v > 0 for v in arrays.values()])
    for i, band in enumerate(("B04", "B03", "B02")):
        values[i] = (values[i] + offsets[band]) / quantification
    display = np.clip(values / rules["display_reflectance_max"], 0, 1) ** (
        1 / rules["display_gamma"]
    )
    rgb = (np.moveaxis(display, 0, -1) * 255).astype("uint8")
    rgb[~valid] = [80, 0, 80]
    native = output / "rgb_native.png"
    Image.fromarray(rgb).save(native)
    centre = (~transform) * (x, y)
    cx, cy = centre[0] - window.col_off, centre[1] - window.row_off
    overview = Image.fromarray(rgb)
    overview.thumbnail((1400, 1400))
    scale = overview.width / rgb.shape[1]
    draw = ImageDraw.Draw(overview)
    draw.ellipse(
        [
            (cx - radius / 10) * scale,
            (cy - radius / 10) * scale,
            (cx + radius / 10) * scale,
            (cy + radius / 10) * scale,
        ],
        outline="yellow",
        width=2,
    )
    draw.line(
        [(cx * scale - 12, cy * scale), (cx * scale + 12, cy * scale)],
        fill="red",
        width=2,
    )
    draw.line(
        [(cx * scale, cy * scale - 12), (cx * scale, cy * scale + 12)],
        fill="red",
        width=2,
    )
    draw.text(
        (8, 8),
        "S2 RGB: SAR position red; heuristic motion circle yellow; missing data purple",
        fill="yellow",
    )
    overview.save(output / "overview.png")
    tiles = []
    for yy in range(0, rgb.shape[0], 1024):
        for xx in range(0, rgb.shape[1], 1024):
            tile = output / f"review_{yy}_{xx}.png"
            Image.fromarray(rgb[yy : yy + 1024, xx : xx + 1024]).save(tile)
            tiles.append({"offset": [xx, yy], "image": ref(tile)})
    mask_transform, mask_crs, mask_shape = geocoding(
        checked(download["tile_metadata"]), 60
    )
    mask_window = (
        from_bounds(*rasterio.windows.bounds(window, transform), mask_transform)
        .round_offsets()
        .round_lengths()
    )
    require_permitted_geometry(
        window_geometry(mask_window, mask_transform, mask_crs),
        priors,
        minimum_separation,
    )
    with rasterio.open(checked(download["assets"]["classification"])) as ds:
        if ds.count != 3 or (ds.height, ds.width) != mask_shape:
            raise ValueError("Three native classification planes required")
        masks = ds.read(window=mask_window)
    mask_centre = (~mask_transform) * (x, y)
    central = classification_patch(
        checked(download["assets"]["classification"]), mask_window, mask_centre
    )
    tile_root = ET.parse(checked(download["tile_metadata"]))
    return {
        "policy": ref(POLICY),
        "download": ref(path),
        "target_id": target["id"],
        "inspection_policy": ref(SCOPE),
        "minimum_protected_distance_m": protected_distance,
        "tile_sensing_time": next(
            (
                e.text
                for e in tile_root.iter()
                if e.tag.split("}")[-1] == "SENSING_TIME"
            ),
            None,
        ),
        "catalogue_offset_s": product["offset_s"],
        "radius_m": radius,
        "requested_window": list(requested.flatten()),
        "read_window": list(window.flatten()),
        "search_square_inspected_fraction": window.width
        * window.height
        / (requested.width * requested.height),
        "valid_fraction_of_read_window": float(valid.mean()),
        "quantification": quantification,
        "offsets": offsets,
        "mask_planes": ["opaque_cloud", "cirrus", "snow_ice"],
        "mask_nonzero_fraction_of_read_window": [float((m > 0).mean()) for m in masks],
        "central_5x5_mask_counts": [int((m > 0).sum()) for m in central],
        "mask_zero_interpretation": "Unflagged is not verified clear sky; "
        "fractions include unavailable areas",
        "rgb_native": ref(native),
        "overview": ref(output / "overview.png"),
        "review_tiles": tiles,
        "pixels_inspected": False,
        "same_object_confirmed": False,
        "labels_admitted": 0,
        **GUARDS,
    }


def validate_download(path: Path) -> dict[str, Any]:
    rules, target, product = source()
    record: dict[str, Any] = json.loads(path.read_text())
    if (
        record["policy"] != ref(POLICY)
        or record["product"] != product
        or any(record.get(k) is not False for k in GUARDS)
    ):
        raise ValueError("Optical source provenance/guards differ")
    if record["target"] != {
        k: target[k] for k in ("id", "lon", "lat", "native_time_utc")
    }:
        raise ValueError("Optical target changed")
    manifest = ET.parse(checked(record["manifest"]))
    objects = manifest.findall(".//dataObject/byteStream")
    canonical = f"{DOWNLOAD}({product['id']})/Nodes({product['name']})/"
    verified = []
    for request in record["requests"]:
        asset = checked(request["artifact"])
        if (
            not request["url"].startswith(canonical)
            or asset.stat().st_size != request["bytes"]
        ):
            raise ValueError("Optical download identity/size differs")
        if request["url"].endswith("/$value") and asset != checked(record["manifest"]):
            filename = request["url"].split("/Nodes(")[-1].split(")")[0]
            entries = [
                s
                for s in objects
                if (location := s.find("fileLocation")) is not None
                and location.attrib["href"].rsplit("/", 1)[-1] == filename
            ]
            if len(entries) != 1:
                raise ValueError("Publisher manifest entry required")
            checksum = entries[0].find("checksum")
            if (
                checksum is None
                or checksum.attrib.get("checksumName") != "SHA3-256"
                or int(entries[0].attrib["size"]) != request["bytes"]
            ):
                raise ValueError("Publisher SHA3-256 and exact size required")
            if (
                hashlib.sha3_256(asset.read_bytes()).hexdigest()
                != (checksum.text or "").lower()
            ):
                raise ValueError("Publisher optical checksum mismatch")
            verified.append(asset)
    for asset in record["assets"].values():
        if checked(asset) not in verified:
            raise ValueError("Optical pixels lack publisher verification")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--render", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.download:
        download(args.output)
    elif args.render:
        render(args.render, args.output)
    else:
        parser.error("Choose download or render")
