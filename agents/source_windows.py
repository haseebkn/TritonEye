"""Prepare the exact source-supported reservation without inference or resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.control import GroundControlPoint
from shapely.geometry import shape

from agents.artifacts import REPO_ROOT
from agents.calibration import Calibrator
from agents.hh_hv_reservation import save
from agents.nl_benchmark import require_development_scene
from agents.native_cog import read_cog
from agents.region import load_region
from agents.reserved_sar import DOWNLOAD, Cdse, NativeTiff, annotation_metadata, locate
from agents.source_reservation import checked, ref, verify

PROTOCOL = REPO_ROOT / "configs/reservations/source_windows_v011.json"
SIZE = 384
Array = np.ndarray[Any, Any]


def scope(area: dict[str, Any], geometry: dict[str, Any]) -> None:
    """Run before any image read; a source envelope is not a permission bypass."""
    polygon = shape(geometry)
    if (
        not polygon.is_valid
        or polygon.is_empty
        or not shape(area["geometry"]).covers(polygon)
        or not shape(area["product"]["GeoFootprint"]).covers(polygon)
        or not load_region().covers(polygon)
        or area["split"] not in {"train", "validation"}
    ):
        raise ValueError("Native window outside reserved development scope")
    require_development_scene(
        area["product_id"],
        {"name": area["name"], "split": area["split"], "study_roi": geometry},
    )


def metadata(
    xml: bytes, area: dict[str, Any], pol: str
) -> tuple[dict[str, Any], list[GroundControlPoint]]:
    info, gcps = annotation_metadata(xml, area, pol)
    tree = ET.fromstring(xml)
    actual = datetime.fromisoformat(
        str(info["native_start_time"]).replace("Z", "+00:00")
    )
    if actual.tzinfo is None:
        actual = actual.replace(tzinfo=timezone.utc)
    expected = datetime.fromisoformat(
        area["product"]["ContentDate"]["Start"].replace("Z", "+00:00")
    )
    mission = tree.findtext(".//adsHeader/missionId", "").upper()
    if mission != area["name"][:3] or abs((actual - expected).total_seconds()) > 1:
        raise ValueError("Native acquisition identity/time mismatch")
    info.update(
        width=int(tree.findtext(".//numberOfSamples", "0")),
        height=int(tree.findtext(".//numberOfLines", "0")),
        azimuth_interval_s=float(tree.findtext(".//azimuthTimeInterval", "nan")),
        mission=mission,
    )
    if (
        min(info["width"], info["height"]) < 512
        or not np.isfinite(info["azimuth_interval_s"])
        or info["azimuth_interval_s"] <= 0
    ):
        raise ValueError("Native dimensions/timing unavailable")
    return info, gcps


def local_context(xml: bytes, col: int, row: int) -> dict[str, Any]:
    """Bilinear native grid incidence; native azimuth time, not scene midpoint."""
    from scipy.interpolate import LinearNDInterpolator

    tree = ET.fromstring(xml)
    points = tree.findall(".//geolocationGridPoint")
    coordinates = [
        (float(p.findtext("line", "nan")), float(p.findtext("pixel", "nan")))
        for p in points
    ]
    incidences = [float(p.findtext("incidenceAngle", "nan")) for p in points]
    incidence = float(
        LinearNDInterpolator(coordinates, incidences)(row + SIZE // 2, col + SIZE // 2)
    )
    start = datetime.fromisoformat(
        tree.findtext(".//adsHeader/startTime", "").replace("Z", "+00:00")
    )
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    interval = float(tree.findtext(".//azimuthTimeInterval", "nan"))
    if not np.isfinite(incidence):
        raise ValueError("Native window incidence unavailable")
    return {
        "center_incidence_deg": incidence,
        "incidence_method": "linear interpolation of native geolocation incidence grid",
        "center_native_time_utc": datetime.fromtimestamp(
            start.timestamp() + (row + SIZE // 2) * interval, timezone.utc
        ).isoformat(),
        "first_row_native_time_utc": datetime.fromtimestamp(
            start.timestamp() + row * interval, timezone.utc
        ).isoformat(),
        "last_row_native_time_utc": datetime.fromtimestamp(
            start.timestamp() + (row + SIZE - 1) * interval, timezone.utc
        ).isoformat(),
        "azimuth_interval_s": interval,
    }


def full_measurement(
    client: Cdse, url: str, size: int, path: Path, budget: dict[str, int]
) -> dict[str, Any]:
    """Bounded streaming fallback for the exact reserved compressed COG only."""
    if size > 1536 * 1024**2 or size <= 0 or path.exists():
        raise ValueError("Full measurement size/path refused")
    if client.bytes + budget["full_bytes"] + size > 4 * 1024**3:
        raise ValueError("Combined transfer budget exhausted")
    client.authenticate()
    digest, count = hashlib.sha256(), 0
    with client.session.get(url, stream=True, timeout=(30, 120)) as response:
        if response.status_code != 200:
            raise ValueError(f"Full native download failed HTTP {response.status_code}")
        with path.open("xb") as stream:
            for block in response.iter_content(4 * 1024**2):
                count += len(block)
                if count > size:
                    raise ValueError("Oversized full native response")
                digest.update(block)
                stream.write(block)
    budget["full_bytes"] += count
    if count != size:
        raise ValueError("Truncated full native response")
    return {
        "bytes": count,
        "sha256": digest.hexdigest(),
        "path": ref(path)["path"],
        "publisher_checksum_verified": False,
    }


def write_window(
    directory: Path,
    record: dict[str, Any],
    arrays: list[Array],
    gcps: list[GroundControlPoint],
) -> None:
    col, row, _, _ = record["window"]
    path = directory / f"{record['id']}_sigma0.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=2,
        dtype="float32",
        compress="deflate",
        nodata=-60,
    ) as dst:
        dst.gcps = (
            [
                GroundControlPoint(row=g.row - row, col=g.col - col, x=g.x, y=g.y)
                for g in gcps
            ],
            "EPSG:4326",
        )
        dst.write(np.stack(arrays))
        dst.descriptions = ("sigma0_db_hh", "sigma0_db_hv")
    record["chip"] = ref(path)
    stride = SIZE + 10
    canvas = Image.new("RGB", (stride * 2, SIZE + 40), "#182128")
    draw = ImageDraw.Draw(canvas)
    for b, pol in enumerate(("HH", "HV")):
        pixels = np.clip((arrays[b] + 30) / 35 * 255, 0, 255).astype("uint8")
        canvas.paste(Image.fromarray(pixels).convert("RGB"), (b * stride, 40))
        draw.text(
            (b * stride + 5, 5), f"{record['id']} {pol} | native {SIZE}px", fill="white"
        )
        for tick in (0, 96, 192, 288):
            draw.text((b * stride + tick + 2, 26), str(tick), fill="#00ffff")
            draw.text((b * stride, 40 + tick), str(tick), fill="#00ffff")
    preview = directory / f"{record['id']}_review.png"
    canvas.save(preview)
    record["preview"] = ref(preview)


def prepare_product(
    client: Cdse, areas: list[dict[str, Any]], output: Path, budget: dict[str, int]
) -> dict[str, Any]:
    area = areas[0]
    directory = output / area["product_id"]
    directory.mkdir()
    base = f"{DOWNLOAD}({area['product_id']})/Nodes({area['name']})"
    listing = json.loads(client.fetch(base + "/Nodes(measurement)/Nodes"))["result"]
    save(directory / "measurement_listing.json", listing)
    records: dict[str, Any] = {}
    pixels: dict[str, list[Array]] = {a["id"]: [] for a in areas}
    bands = []
    original_gcps: list[GroundControlPoint] = []
    for pol in ("hh", "hv"):
        matches = [
            n
            for n in listing
            if f"-{pol}-" in n["Name"] and n["Name"].endswith(".tiff")
        ]
        if len(matches) != 1:
            raise ValueError("Exact HH/HV native inventory required")
        item = matches[0]
        name = item["Name"]
        if Path(name).name != name or any(c in name for c in "()\\/"):
            raise ValueError("Unsafe native source name")
        annotation = client.fetch(
            base + f"/Nodes(annotation)/Nodes({name[:-5]}.xml)/$value"
        )
        info, gcps = metadata(annotation, area, pol)
        if original_gcps and [(g.row, g.col, g.x, g.y) for g in original_gcps] != [
            (g.row, g.col, g.x, g.y) for g in gcps
        ]:
            raise ValueError("Native HH/HV geolocation mismatch")
        original_gcps = gcps
        ap, cp = (
            directory / f"{pol}_annotation.xml",
            directory / f"{pol}_calibration.xml",
        )
        ap.write_bytes(annotation)
        calibration = client.fetch(
            base
            + "/Nodes(annotation)/Nodes(calibration)"
            + f"/Nodes(calibration-{name[:-5]}.xml)/$value"
        )
        cal_header = ET.fromstring(calibration)
        if (
            cal_header.findtext(".//adsHeader/polarisation", "").lower() != pol
            or cal_header.findtext(".//adsHeader/missionId", "").upper()
            != area["name"][:3]
        ):
            raise ValueError("Calibration identity mismatch")
        cp.write_bytes(calibration)
        info.update(annotation=ref(ap), calibration=ref(cp))
        windows = []
        for a in areas:
            col, row, geometry = locate(gcps, (a["lon"], a["lat"]), SIZE)
            scope(a, geometry)
            if (
                min(col, row) < 0
                or col + SIZE > info["width"]
                or row + SIZE > info["height"]
            ):
                raise ValueError("Reserved crop outside native raster")
            window = [col, row, SIZE, SIZE]
            if a["id"] in records and records[a["id"]]["window"] != window:
                raise ValueError("Band window mismatch")
            records.setdefault(
                a["id"],
                {
                    "id": a["id"],
                    "product_id": a["product_id"],
                    "split": a["split"],
                    "geographic_group": a["geographic_group"],
                    "native_acquisition_group": a["native_acquisition_group"],
                    "prior_datatake_exposure": a["prior_datatake_exposure"],
                    "window": window,
                    "geometry": geometry,
                    "native_context": local_context(annotation, col, row),
                    "bands": [],
                },
            )
            windows.append((a, col, row))
        url = base + f"/Nodes(measurement)/Nodes({name})/$value"
        receipts: list[dict[str, Any]] = []

        def read(offset: int, length: int) -> bytes:
            raw = client.fetch(url, offset, length, item["ContentLength"])
            receipts.append(
                {
                    "offset": offset,
                    "length": length,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            return raw

        # Exact COG UUID is retained; full-resolution raster read, no overviews.
        native = None
        compressed = None
        if "_COG.SAFE" in area["name"]:
            compressed, receipts = read_cog(
                client,
                url,
                item["ContentLength"],
                (info["width"], info["height"]),
                [(c, r, SIZE) for _, c, r in windows],
            )
        else:
            native = NativeTiff(read, item["ContentLength"])
            if native.width != info["width"] or native.height != info["height"]:
                raise ValueError("Native TIFF/annotation dimensions differ")
        if True:
            for index, (a, col, row) in enumerate(windows):
                if compressed is not None:
                    dn = compressed[index]
                else:
                    assert native is not None
                    dn = native.window(col, row, SIZE)
                if dn.shape != (SIZE, SIZE) or not (dn > 0).all():
                    raise ValueError(
                        "Invalid reserved pixels; no padding or relocation permitted"
                    )
                dp = directory / f"{a['id']}_{pol}_dn.npy"
                np.save(dp, dn, allow_pickle=False)
                db = Calibrator(str(cp)).to_sigma0_db(dn, row, col).astype("float32")
                if not np.isfinite(db).all():
                    raise ValueError("Nonfinite calibration")
                pixels[a["id"]].append(db)
                records[a["id"]]["bands"].append(
                    {
                        "polarization": pol.upper(),
                        "dn": ref(dp),
                        "valid_pixels": int(dn.size),
                    }
                )
        info.update(
            measurement_url=url,
            measurement_bytes=item["ContentLength"],
            range_receipts=receipts,
            transport=(
                "bounded_native_compressed_ranges"
                if compressed is not None
                else "bounded_native_strips"
            ),
            full_source_hash_verified=False,
        )
        bands.append(info)
    for record in records.values():
        write_window(directory, record, pixels[record["id"]], original_gcps)
    result = {
        "product_id": area["product_id"],
        "name": area["name"],
        "bands": bands,
        "areas": list(records.values()),
        "scored": False,
        "resampled": False,
    }
    save(directory / "prepared.json", result)
    return result


def prepare(output: Path) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Fresh project data directory required")
    if shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3:
        raise ValueError("At least 5 GiB free space required")
    protocol = json.loads(PROTOCOL.read_text())
    lock = json.loads((REPO_ROOT / protocol["reservation_lock"]).read_text())
    verify(REPO_ROOT / lock["directory"], lock["seal_sha256"], True)
    manifest = json.loads((REPO_ROOT / lock["directory"] / "manifest.json").read_text())
    private = json.loads(checked(manifest["private_selection"]).read_text())
    output.mkdir(parents=True)
    save(
        output / "selection.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": ref(PROTOCOL),
            "reservation_seal_sha256": lock["seal_sha256"],
            "private_selection": manifest["private_selection"],
            "scored": False,
        },
    )
    client, budget = Cdse(), {"full_bytes": 0}
    results, failures = [], []
    products = list(dict.fromkeys(a["product_id"] for a in private["areas"]))
    for product in products:
        try:
            results.append(
                prepare_product(
                    client,
                    [a for a in private["areas"] if a["product_id"] == product],
                    output,
                    budget,
                )
            )
            print(
                json.dumps(
                    {
                        "prepared": product,
                        "network_bytes": client.bytes + budget["full_bytes"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            failures.append({"product_id": product, "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
        save(
            output / f"progress_{products.index(product)+1}.json",
            {
                "version": "0.1.0",
                "selection": ref(output / "selection.json"),
                "products": results,
                "failures": failures,
                "scored": False,
                "network_bytes": client.bytes + budget["full_bytes"],
            },
        )
    save(
        output / "manifest.json",
        {
            "version": "0.1.1",
            "selection": ref(output / "selection.json"),
            "products": results,
            "failures": failures,
            "scored": False,
            "network_bytes": client.bytes + budget["full_bytes"],
        },
    )
    if failures:
        raise ValueError(
            "Preparation incomplete; failed reservations retained without substitutions"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
