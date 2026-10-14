"""Bounded, unresampled native HH/HV acquisition for the frozen reservation.

Only uncompressed single-band uint16 classic TIFF strips are supported. Range
responses must be exact; unsupported layouts fail before any pixel inference.
Full source-file checksums are not verified by partial downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from shapely.geometry import Polygon, mapping, shape

from agents.artifacts import REPO_ROOT, sha256_file
from agents.calibration import Calibrator
from agents.hh_hv_reservation import load_reservation, require_scope, save
from agents.nl_benchmark import require_development_scene

PROTOCOL = REPO_ROOT / "configs/reservations/hh_hv_transfer_v010.json"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
Array = np.ndarray[Any, Any]


def reference(path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


class NativeTiff:
    """Read verified classic TIFF metadata, then only strips intersecting a crop."""

    def __init__(self, read: Callable[[int, int], bytes], size: int) -> None:
        self.read, self.size = read, size
        header = read(0, 8)
        if header[:4] != b"II\x2a\x00":
            raise ValueError("Only little-endian classic TIFF supported")
        offset = struct.unpack("<I", header[4:])[0]
        count = struct.unpack("<H", read(offset, 2))[0]
        if not 1 <= count <= 128:
            raise ValueError("Unbounded TIFF directory")
        directory = read(offset + 2, count * 12 + 4)
        if struct.unpack("<I", directory[-4:])[0]:
            raise ValueError("Multiple TIFF directories unsupported")
        tags: dict[int, list[int]] = {}
        for i in range(count):
            tag, kind, n, value = struct.unpack(
                "<HHII", directory[i * 12 : i * 12 + 12]
            )
            if tag not in {256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 339}:
                continue
            if kind not in {3, 4} or not 1 <= n <= 100000:
                raise ValueError("Unsupported TIFF numeric tag")
            length = n * (2 if kind == 3 else 4)
            raw = directory[i * 12 + 8 : i * 12 + 12][:length]
            if length > 4:
                raw = read(value, length)
            tags[tag] = list(struct.unpack("<" + ("H" if kind == 3 else "I") * n, raw))
        for tag, expected in {258: 16, 259: 1, 262: 1, 277: 1, 284: 1, 339: 1}.items():
            if tags.get(tag, [1]) != [expected]:
                raise ValueError(
                    "Only uncompressed unsigned 16-bit single-band TIFF supported"
                )
        self.width, self.height = tags[256][0], tags[257][0]
        self.rows = tags[278][0]
        self.offsets, self.counts = tags[273], tags[279]
        if (
            not 1 <= self.width <= 100000
            or not 1 <= self.height <= 100000
            or self.rows < 1
        ):
            raise ValueError("Invalid TIFF dimensions")
        if len(self.offsets) != (self.height + self.rows - 1) // self.rows or len(
            self.counts
        ) != len(self.offsets):
            raise ValueError("Invalid strip inventory")
        for i, (start, length) in enumerate(
            zip(self.offsets, self.counts, strict=True)
        ):
            expected_length = (
                min(self.rows, self.height - i * self.rows) * self.width * 2
            )
            if length != expected_length or start < 8 or start + length > size:
                raise ValueError("Invalid native strip extent")

    def window(self, col: int, row: int, size: int) -> Array:
        if (
            size != 512
            or min(col, row) < 0
            or col + size > self.width
            or row + size > self.height
        ):
            raise ValueError("Only bounded 512-pixel windows permitted")
        first, last = row // self.rows, (row + size - 1) // self.rows
        starts = self.offsets[first : last + 1]
        lengths = self.counts[first : last + 1]
        if any(a + n != b for a, n, b in zip(starts, lengths, starts[1:])):
            raise ValueError("Noncontiguous native strips unsupported")
        if sum(lengths) > 128 * 1024**2:
            raise ValueError("Native window transfer budget exceeded")
        raw = self.read(starts[0], sum(lengths))
        values = np.frombuffer(raw, dtype="<u2").reshape(-1, self.width)
        within = row - first * self.rows
        return values[within : within + size, col : col + size].copy()


class Cdse:
    def __init__(self) -> None:
        load_dotenv(REPO_ROOT / ".env")
        self.session = requests.Session()
        self.bytes = 0
        self.authenticated = 0.0

    def authenticate(self) -> None:
        user, password = os.getenv("COPERNICUS_USER"), os.getenv("COPERNICUS_PASS")
        if not user or not password:
            raise ValueError("COPERNICUS_USER/PASS required")
        result = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data={
                "client_id": "cdse-public",
                "grant_type": "password",
                "username": user,
                "password": password,
            },
            timeout=30,
        )
        if result.status_code != 200:
            raise ValueError(f"CDSE authentication failed HTTP {result.status_code}")
        self.session.headers["Authorization"] = (
            "Bearer " + result.json()["access_token"]
        )
        self.authenticated = time.monotonic()

    def fetch(
        self,
        url: str,
        start: int | None = None,
        count: int | None = None,
        total: int | None = None,
    ) -> bytes:
        if not url.startswith(DOWNLOAD + "("):
            raise ValueError("Only canonical CDSE product URLs permitted")
        if time.monotonic() - self.authenticated > 450:
            self.authenticate()
        headers = {}
        limit = 8 * 1024**2
        if start is not None:
            if (
                count is None
                or total is None
                or start < 0
                or not 0 < count <= 128 * 1024**2
                or start + count > total
            ):
                raise ValueError("Invalid bounded byte request")
            headers["Range"] = f"bytes={start}-{start + count - 1}"
            limit = count
        with self.session.get(
            url, headers=headers, stream=True, timeout=(30, 120)
        ) as result:
            if result.status_code != (206 if start is not None else 200):
                raise ValueError(f"CDSE request failed HTTP {result.status_code}")
            if (
                start is not None
                and result.headers.get("Content-Range")
                != f"bytes {start}-{start + limit - 1}/{total}"
            ):
                raise ValueError("CDSE ignored or changed byte range")
            body = result.raw.read(limit + 1)
        if len(body) > limit or (start is not None and len(body) != limit):
            raise ValueError("Truncated or oversized CDSE response")
        self.bytes += len(body)
        if self.bytes > 2 * 1024**3:
            raise ValueError("Acquisition transfer budget exhausted")
        return body


def annotation_metadata(
    xml: bytes, scene: dict[str, Any], pol: str
) -> tuple[dict[str, Any], list[GroundControlPoint]]:
    tree = ET.fromstring(xml)
    if (
        tree.findtext(".//adsHeader/polarisation", "").lower() != pol
        or tree.findtext(".//adsHeader/mode") != "IW"
    ):
        raise ValueError("Native polarization/mode mismatch")
    gcps = [
        GroundControlPoint(
            row=float(p.findtext("line", "nan")),
            col=float(p.findtext("pixel", "nan")),
            x=float(p.findtext("longitude", "nan")),
            y=float(p.findtext("latitude", "nan")),
        )
        for p in tree.findall(".//geolocationGridPoint")
    ]
    if len(gcps) < 9 or not np.isfinite([(g.row, g.col, g.x, g.y) for g in gcps]).all():
        raise ValueError("Complete native geolocation required")
    metadata = {
        "polarization": pol.upper(),
        "mode": "IW",
        "range_spacing_m": float(tree.findtext(".//rangePixelSpacing", "nan")),
        "azimuth_spacing_m": float(tree.findtext(".//azimuthPixelSpacing", "nan")),
        "incidence_mid_swath_deg": float(
            tree.findtext(".//incidenceAngleMidSwath", "nan")
        ),
        "thermal_noise_corrected": tree.findtext(".//thermalNoiseCorrectionPerformed"),
        "native_start_time": tree.findtext(".//adsHeader/startTime"),
        "product_id": scene["product_id"],
    }
    if not np.isfinite(
        [
            metadata[k]
            for k in ("range_spacing_m", "azimuth_spacing_m", "incidence_mid_swath_deg")
        ]
    ).all():
        raise ValueError("Finite spacing and incidence metadata required")
    return metadata, gcps


def locate(
    gcps: list[GroundControlPoint], center: tuple[float, float]
) -> tuple[int, int, dict[str, Any]]:
    with GCPTransformer(gcps, tps=True) as transform:
        rr, cc = transform.rowcol([center[0]], [center[1]])
        row, col = int(rr[0]) - 256, int(cc[0]) - 256
        rows = np.concatenate(
            [
                np.full(17, row),
                np.linspace(row, row + 512, 17),
                np.full(17, row + 512),
                np.linspace(row + 512, row, 17),
            ]
        )
        cols = np.concatenate(
            [
                np.linspace(col, col + 512, 17),
                np.full(17, col + 512),
                np.linspace(col + 512, col, 17),
                np.full(17, col),
            ]
        )
        xs, ys = transform.xy(rows.tolist(), cols.tolist(), offset="ul")
    polygon = Polygon(zip(xs, ys))
    if not polygon.is_valid:
        raise ValueError("Invalid native crop footprint")
    return col, row, mapping(polygon)


def prepare_scene(
    client: Cdse, reservation: dict[str, Any], scene: dict[str, Any], output: Path
) -> dict[str, Any]:
    directory = output / scene["product_id"]
    directory.mkdir()
    base = f"{DOWNLOAD}({scene['product_id']})/Nodes({scene['name']})"
    listing = json.loads(client.fetch(base + "/Nodes(measurement)/Nodes"))["result"]
    save(directory / "measurement_listing.json", listing)
    bands: list[dict[str, Any]] = []
    window_records: dict[str, Any] = {}
    for pol in ("hh", "hv"):
        items = [
            n
            for n in listing
            if f"-{pol}-" in n["Name"] and n["Name"].endswith(".tiff")
        ]
        if len(items) != 1:
            raise ValueError("One exact native measurement per polarization required")
        item = items[0]
        name = item["Name"]
        if Path(name).name != name or any(c in name for c in "()\\/"):
            raise ValueError("Unsafe source name")
        annotation_url = base + f"/Nodes(annotation)/Nodes({name[:-5]}.xml)/$value"
        calibration_url = (
            base
            + "/Nodes(annotation)/Nodes(calibration)"
            + f"/Nodes(calibration-{name[:-5]}.xml)/$value"
        )
        annotation = client.fetch(annotation_url)
        metadata, gcps = annotation_metadata(annotation, scene, pol)
        annotation_path, calibration_path = (
            directory / f"{pol}_annotation.xml",
            directory / f"{pol}_calibration.xml",
        )
        annotation_path.write_bytes(annotation)
        calibration_path.write_bytes(client.fetch(calibration_url))
        metadata.update(
            annotation=reference(annotation_path),
            calibration=reference(calibration_path),
        )
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

        source = NativeTiff(read, item["ContentLength"])
        for roi in scene["rois"]:
            key = roi["regime"]
            centroid = shape(roi["geometry"]).centroid
            col, row, geometry = locate(gcps, (centroid.x, centroid.y))
            context = {
                "group": scene["geographic_group"],
                "regime": key,
                "geometry": geometry,
            }
            require_scope(
                reservation, {**scene, "polarizations": ["hh", "hv"]}, context
            )
            require_development_scene(
                scene["product_id"], {**scene, "study_roi": geometry}
            )
            previous = window_records.get(key)
            if previous and previous["window"] != [col, row, 512, 512]:
                raise ValueError("Band native window mismatch")
            if (
                pol == "hv"
                and [(g.row, g.col, g.x, g.y) for g in gcps] != bands[0]["gcp_values"]
            ):
                raise ValueError("Band geolocation mismatch")
            dn = source.window(col, row, 512)
            if not (dn > 0).all():
                raise ValueError(
                    "Nodata in preselected crop; "
                    "preserve exclusion instead of moving it"
                )
            dn_path = directory / f"{key}_{pol}_dn.npy"
            np.save(dn_path, dn, allow_pickle=False)
            db = (
                Calibrator(str(calibration_path))
                .to_sigma0_db(dn, row, col)
                .astype("float32")
            )
            window_records.setdefault(
                key,
                {
                    "id": scene["geographic_group"] + "_" + scene["epoch"] + "_" + key,
                    "window": [col, row, 512, 512],
                    "geometry": geometry,
                    "regime": key,
                    "bands": [],
                },
            )
            window_records[key]["bands"].append({"dn": reference(dn_path), "db": db})
        metadata.update(
            measurement_url=url,
            measurement_bytes=item["ContentLength"],
            range_receipts=receipts,
            full_source_hash_verified=False,
            gcp_values=[(g.row, g.col, g.x, g.y) for g in gcps],
        )
        bands.append(metadata)
    for roi in window_records.values():
        col, row, _, _ = roi["window"]
        tiff = directory / f"{roi['regime']}_sigma0.tif"
        arrays = np.stack([b.pop("db") for b in roi["bands"]])
        with rasterio.open(
            tiff,
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=2,
            dtype="float32",
            compress="deflate",
            nodata=-60.0,
        ) as destination:
            destination.gcps = (
                [
                    GroundControlPoint(row=g[0] - row, col=g[1] - col, x=g[2], y=g[3])
                    for g in bands[0]["gcp_values"]
                ],
                "EPSG:4326",
            )
            destination.write(arrays)
            destination.set_band_description(1, "sigma0_db_hh")
            destination.set_band_description(2, "sigma0_db_hv")
        roi["chip"] = reference(tiff)
        canvas = Image.new("RGB", (1044, 552), "#182128")
        draw = ImageDraw.Draw(canvas)
        for b, pol in enumerate(("HH", "HV")):
            pixels = np.clip((arrays[b] + 30) / 35 * 255, 0, 255).astype("uint8")
            canvas.paste(Image.fromarray(pixels).convert("RGB"), (b * 522, 40))
            draw.text(
                (b * 522 + 5, 5), f"{roi['id']} {pol} | native 512px", fill="white"
            )
            for tick in (0, 128, 256, 384):
                draw.text((b * 522 + tick + 2, 26), str(tick), fill="#00ffff")
                draw.text((b * 522, 40 + tick), str(tick), fill="#00ffff")
        preview = directory / f"{roi['regime']}_review.png"
        canvas.save(preview)
        roi["preview"] = reference(preview)
    for band in bands:
        band.pop("gcp_values")
    result = {
        "scene": scene,
        "native_bands": bands,
        "rois": list(window_records.values()),
        "scored": False,
        "resampled": False,
        "status": "prepared_before_annotation",
    }
    save(directory / "prepared.json", result)
    return result


def prepare(output: Path) -> None:
    output = output.resolve()
    if output.exists() or not output.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Fresh project data directory required")
    if shutil.disk_usage(REPO_ROOT).free < 1024**3:
        raise ValueError("At least 1 GiB free project space required")
    protocol = json.loads(PROTOCOL.read_text())
    lock = json.loads((REPO_ROOT / protocol["reservation_lock"]).read_text())
    reservation = load_reservation(REPO_ROOT / lock["directory"], lock["seal_sha256"])
    output.mkdir(parents=True)
    save(
        output / "selection.json",
        {
            "protocol": protocol,
            "protocol_sha256": sha256_file(PROTOCOL),
            "reservation_seal_sha256": lock["seal_sha256"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scored": False,
        },
    )
    client = Cdse()
    results, failures = [], []
    for scene in reservation["scenes"]:
        try:
            results.append(prepare_scene(client, reservation, scene, output))
            print(
                json.dumps({"prepared": scene["name"], "network_bytes": client.bytes}),
                flush=True,
            )
        except (ValueError, OSError, requests.RequestException) as error:
            failures.append({"product_id": scene["product_id"], "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
    save(
        output / "manifest.json",
        {
            "version": protocol["version"],
            "selection": reference(output / "selection.json"),
            "scenes": results,
            "failures": failures,
            "scored": False,
            "network_bytes": client.bytes,
        },
    )
    if failures:
        raise ValueError("Some reserved acquisitions failed; no substitutions made")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
