#!/usr/bin/env python3
"""
TritonEye Ingestion Agent
Retrieves Sentinel-1 SAR imagery and processes AIS telemetry.
Supports a deterministic mock mode for offline testing and pipeline validation.
"""

import csv
import glob
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Agents are executed as standalone scripts, so the workspace root has to be on
# the path before the shared modules under agents/ can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.ais_recorder import CSV_HEADER
from agents.ais_validation import course_deg, parse_utc, speed_knots, utc_string
from agents.artifacts import mission_directory, write_json
from agents.region import REGION_NAME, contains_points, load_region, validate_aoi

# Third-party imports (handled gracefully if missing during bootstrap checks)
try:
    import numpy as np
    import rasterio
    import rasterio.transform
    import requests
    import shapely.geometry
    from pyproj import Transformer

    from agents.geo import describe_raster_georeferencing
    from agents.tracking import RunTracker
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    # We will let the script fail gracefully on execution rather than crash on import


# Degrees of slack added around the scene footprint when selecting AIS records.
# A vessel just outside the swath edge can still fall inside the correlation
# radius, and ~0.05 deg is comfortably wider than the 2 km buffer used there.
AIS_MARGIN_DEG = 0.05


def load_aoi(aoi_path: str) -> Dict[str, Any]:
    """Loads and returns the geojson AOI configuration."""
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI configuration not found at {aoi_path}")
    with open(aoi_path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def pad_bbox(bbox: List[float], margin_deg: float) -> List[float]:
    """Expands [min_lon, min_lat, max_lon, max_lat] by a margin in degrees."""
    return [
        bbox[0] - margin_deg,
        bbox[1] - margin_deg,
        bbox[2] + margin_deg,
        bbox[3] + margin_deg,
    ]


def aoi_bbox(geom: Dict[str, Any]) -> List[float]:
    """Returns the [min_lon, min_lat, max_lon, max_lat] envelope of an AOI polygon."""
    return list(shapely.geometry.shape(geom).bounds)


def generate_synthetic_data(
    aoi_path: str, output_dir: str, seed: int = 42
) -> Dict[str, Any]:
    """
    Generates deterministic mock Sentinel-1 GRD GeoTIFF files and matching AIS tracks.
    Coordinates are centered around St. John's, NL offshore area.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Establish file structures and naming conventions
    override_timestamp = os.getenv("MOCK_INGEST_TIMESTAMP", "")
    if override_timestamp:
        timestamp_str = override_timestamp
    else:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mission_id = f"mission_mock_{timestamp_str}_{Path(aoi_path).stem}"
    synthetic_dir = mission_directory(mission_id, root=Path(output_dir) / "synthetic")
    safe_name = f"SYNTHETIC_VV_VH_{timestamp_str}.SAFE"
    safe_dir = str(synthetic_dir / safe_name)
    measurement_dir = os.path.join(safe_dir, "measurement")
    os.makedirs(measurement_dir, exist_ok=True)

    aoi = load_aoi(aoi_path)["features"][0]["geometry"]
    validate_aoi(aoi)
    center = shapely.geometry.shape(aoi).representative_point()
    center_lon, center_lat = center.x, center.y

    # 2. Setup Spatial Projection (EPSG:32622 - UTM Zone 22N)
    proj_wgs84 = "EPSG:4326"
    proj_utm = "EPSG:32622"

    transformer = Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True)
    rev_transformer = Transformer.from_crs(proj_utm, proj_wgs84, always_xy=True)

    center_x, center_y = transformer.transform(center_lon, center_lat)

    # Create a 100 km x 100 km bounding box (10000 x 10000 pixels at 10m spacing)
    half_size = 50000.0  # 50 km in each direction
    x_min = center_x - half_size
    x_max = center_x + half_size
    y_min = center_y - half_size
    y_max = center_y + half_size

    width = 10000
    height = 10000
    resolution = 10.0  # 10 meters per pixel

    # Rasterio affine transform
    transform = rasterio.transform.from_bounds(
        x_min, y_min, x_max, y_max, width, height
    )

    # 3. Generate 50 deterministic vessel tracks
    vessels: List[Dict[str, Any]] = []
    base_mmsi = 316000000
    image_time = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )

    for i in range(50):
        mmsi = base_mmsi + i
        # Keep vessels away from the extreme edges of our 100km grid
        vessel_x = random.uniform(x_min + 5000.0, x_max - 5000.0)
        vessel_y = random.uniform(y_min + 5000.0, y_max - 5000.0)

        speed_knots = random.uniform(5.0, 25.0)
        speed_mps = speed_knots * 0.514444
        heading = random.uniform(0.0, 360.0)

        # Determine if this vessel is active or dark (80% active, 20% dark)
        is_active = i < 40

        # Track parameters (5 points: -10 min, -5 min, 0, 5 min, 10 min)
        track_points = []
        for dt_seconds in [-600, -300, 0, 300, 600]:
            pt_time = image_time + timedelta(seconds=dt_seconds)

            # Position offset
            pt_x = vessel_x + speed_mps * math.sin(math.radians(heading)) * dt_seconds
            pt_y = vessel_y + speed_mps * math.cos(math.radians(heading)) * dt_seconds

            pt_lon, pt_lat = rev_transformer.transform(pt_x, pt_y)
            track_points.append({"time": pt_time, "lon": pt_lon, "lat": pt_lat})

        vessels.append(
            {
                "mmsi": mmsi,
                "x": vessel_x,
                "y": vessel_y,
                "speed": speed_knots,
                "heading": heading,
                "is_active": is_active,
                "track": track_points,
            }
        )

    # 4. Generate mock GeoTIFF bands (VV and VH polarizations)
    # Background representing low-intensity ocean clutter (values 5 to 20)
    vv_data = np.random.randint(5, 20, size=(height, width), dtype=np.uint8)
    vh_data = np.random.randint(2, 10, size=(height, width), dtype=np.uint8)

    # Embed high-intensity target signatures for all 50 vessels
    for v in vessels:
        col = int((v["x"] - x_min) / resolution)
        row = int((y_max - v["y"]) / resolution)

        # Clamp coordinates to safely stay within bounds
        row = max(1, min(height - 2, row))
        col = max(1, min(width - 2, col))

        # High intensity peak at center
        vv_data[row, col] = 255
        vh_data[row, col] = 230

        # High backscatter spillover (cross pattern)
        for r_offset, c_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            vv_data[row + r_offset, col + c_offset] = 180
            vh_data[row + r_offset, col + c_offset] = 150

    # Write GeoTIFFs
    vv_path = os.path.join(measurement_dir, "s1a-iw-grd-vv.tiff")
    vh_path = os.path.join(measurement_dir, "s1a-iw-grd-vh.tiff")

    tiff_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": proj_utm,
        "transform": transform,
    }

    with rasterio.open(vv_path, "w", **tiff_profile) as dst:
        dst.write(vv_data, 1)

    with rasterio.open(vh_path, "w", **tiff_profile) as dst:
        dst.write(vh_data, 1)

    # Write a basic manifest file
    manifest_path = os.path.join(safe_dir, "manifest.safe")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(
            f"<xml><metadata><product>{safe_name}</product>"
            f"<crs>{proj_utm}</crs></metadata></xml>"
        )

    # 5. Write AIS track records to CSV (Active vessels only)
    ais_path = str(synthetic_dir / "ais_mock.csv")
    with open(ais_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
        )

        for v in vessels:
            if not v["is_active"]:
                continue
            for pt in v["track"]:
                writer.writerow(
                    [
                        v["mmsi"],
                        round(pt["lat"], 6),
                        round(pt["lon"], 6),
                        utc_string(pt["time"]),
                        round(v["speed"], 1),
                        round(v["heading"], 1),
                    ]
                )

    # 6. Map the image corners back to lat/lon for the response metadata
    top_left_lon, top_left_lat = rev_transformer.transform(x_min, y_max)
    bottom_right_lon, bottom_right_lat = rev_transformer.transform(x_max, y_min)

    return {
        "mission_id": mission_id,
        "status": "success",
        "mode": "mock",
        "sar_product": safe_name,
        "sar_bands": {"VV": vv_path, "VH": vh_path},
        "ais_telemetry": ais_path,
        "acquisition_time": utc_string(image_time),
        "region_name": REGION_NAME,
        "analysis_region": shapely.geometry.mapping(load_region()),
        "data_provenance": (
            "synthetic demonstration; not evidence of detection performance"
        ),
        "spatial_bounds": {
            # Mock products are written with a real CRS and affine transform, so
            # they take the affine georeferencing path downstream.
            "georeferencing": "affine",
            "crs": proj_wgs84,
            "bbox": [top_left_lon, bottom_right_lat, bottom_right_lon, top_left_lat],
            "ais_coverage": "mock",
        },
    }


# Operating area. This project covers Newfoundland and Labrador only:
# the Grand Banks carry the region's commercial fishing, the Jeanne d'Arc
# Basin production installations, and the approaches to St. John's.
DEFAULT_AOI = "grand_banks.geojson"


# The aisstream.io recorder already writes the target schema, so this map is
# the identity. It exists so _filter_ais_file stays source-agnostic: a second
# AIS source with different column names is added by supplying another map, not
# by branching inside the filter.
_NORMALIZED_COLUMNS = {
    "timestamp": "timestamp",
    "lon": "lon",
    "lat": "lat",
    "speed_knots": "speed_knots",
    "course_deg": "course_deg",
}


def _filter_ais_file(
    ais_file_path: str,
    acquisition_time_str: str,
    bbox: List[float],
    output_path: str,
    columns: Dict[str, str],
) -> bool:
    """
    Filters one AIS CSV to the +/-5 minute acquisition window and bbox, and
    writes the result in the target schema (mmsi, lat, lon, timestamp,
    speed_knots, course_deg). Source-agnostic: a source with different column
    archive paths, which differ only in their source column names.
    """
    import pandas as pd

    acq_dt = parse_utc(acquisition_time_str, allow_naive=True)
    if acq_dt is None:
        raise ValueError("A valid acquisition timestamp is required for AIS filtering")
    time_window_start = acq_dt - timedelta(minutes=5)
    time_window_end = acq_dt + timedelta(minutes=5)
    lon_min, lat_min, lon_max, lat_max = bbox

    print(f"Filtering AIS data from {ais_file_path}...", file=sys.stderr)
    print(f"Time window: {time_window_start} to {time_window_end}", file=sys.stderr)
    print(
        f"BBox bounds: Lon [{lon_min}, {lon_max}], Lat [{lat_min}, {lat_max}]",
        file=sys.stderr,
    )

    ts_col, lon_col, lat_col = columns["timestamp"], columns["lon"], columns["lat"]
    filtered_rows = []
    chunk_size = 100000

    try:
        for chunk in pd.read_csv(ais_file_path, chunksize=chunk_size):
            if not {"mmsi", ts_col, lon_col, lat_col}.issubset(chunk.columns):
                raise ValueError(
                    "AIS file is missing identity, time or position fields"
                )
            chunk["timestamp_dt"] = pd.to_datetime(
                chunk[ts_col], errors="coerce", utc=True, format="mixed"
            )
            for name in ("mmsi", lon_col, lat_col):
                chunk[name] = pd.to_numeric(chunk[name], errors="coerce")

            mask_temp = (chunk["timestamp_dt"] >= time_window_start) & (
                chunk["timestamp_dt"] <= time_window_end
            )
            mask_spatial = (
                (chunk[lon_col] >= lon_min)
                & (chunk[lon_col] <= lon_max)
                & (chunk[lat_col] >= lat_min)
                & (chunk[lat_col] <= lat_max)
                & chunk[lon_col].between(-180, 180)
                & chunk[lat_col].between(-90, 90)
            )
            mask_identity = chunk["mmsi"].between(100000000, 999999999) & (
                chunk["mmsi"] % 1 == 0
            )
            filtered_chunk = chunk[mask_temp & mask_spatial & mask_identity].copy()
            if not filtered_chunk.empty:
                filtered_chunk = filtered_chunk.loc[
                    contains_points(
                        filtered_chunk[lon_col].tolist(),
                        filtered_chunk[lat_col].tolist(),
                    )
                ]
            if not filtered_chunk.empty:
                filtered_rows.append(filtered_chunk)

        if not filtered_rows:
            print(
                f"No AIS records in {ais_file_path} matched the window/bbox.",
                file=sys.stderr,
            )
            return False

        full_filtered = pd.concat(filtered_rows)
        renamed_df = full_filtered.rename(columns={v: k for k, v in columns.items()})

        renamed_df["mmsi"] = renamed_df["mmsi"].astype("int64")
        renamed_df["timestamp"] = renamed_df["timestamp_dt"].map(
            lambda value: utc_string(value.to_pydatetime())
        )
        for column, validator in (
            ("speed_knots", speed_knots),
            ("course_deg", course_deg),
        ):
            renamed_df[column] = (
                renamed_df[column].map(validator) if column in renamed_df else None
            )
        defaults = {
            "source": "legacy_local_archive",
            "timestamp_basis": "legacy_archive_utc_assumed",
        }
        for column in CSV_HEADER:
            if column not in renamed_df:
                renamed_df[column] = defaults.get(column)
        renamed_df = renamed_df[CSV_HEADER].drop_duplicates()

        # Multiple source files (e.g. an acquisition window spanning UTC
        # midnight) may already have been filtered into output_path by an
        # earlier call; append rather than overwrite in that case.
        write_header = not os.path.exists(output_path)
        renamed_df.to_csv(output_path, mode="a", header=write_header, index=False)
        print(f"Wrote {len(renamed_df)} AIS records to {output_path}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Error filtering {ais_file_path}: {e}", file=sys.stderr)
        return False


def filter_ais_stream_archive(
    archive_dir: str,
    acquisition_time_str: str,
    bbox: List[float],
    output_path: str,
) -> bool:
    """
    Filters the locally recorded aisstream.io archive (see agents/ais_recorder.py)
    to the acquisition window, bbox and approved NL study area. The existence
    of matching rows establishes observations only, never complete coverage.

    An acquisition window can straddle a UTC day boundary, so every daily file
    the window overlaps is checked.
    """
    acq_dt = parse_utc(acquisition_time_str, allow_naive=True)
    if acq_dt is None:
        raise ValueError("A valid acquisition timestamp is required for AIS filtering")
    window_start = acq_dt - timedelta(minutes=5)
    window_end = acq_dt + timedelta(minutes=5)
    candidate_dates = {
        window_start.strftime("%Y-%m-%d"),
        window_end.strftime("%Y-%m-%d"),
    }

    import tempfile

    sources = [
        path
        for date_str in sorted(candidate_dates)
        for path in sorted(
            glob.glob(os.path.join(archive_dir, f"ais_stream_{date_str}*.csv"))
        )
    ]
    if Path(output_path).resolve() in {Path(source).resolve() for source in sources}:
        raise ValueError("Filtered output must not overwrite a source AIS archive")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=Path(output_path).parent, suffix=".csv.tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(CSV_HEADER)
    matched_any = False
    try:
        for day_file in sources:
            if _filter_ais_file(
                day_file, acquisition_time_str, bbox, temporary, _NORMALIZED_COLUMNS
            ):
                matched_any = True
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    if not matched_any:
        print(
            f"No recorded aisstream.io coverage for {acquisition_time_str} "
            f"in {archive_dir}.",
            file=sys.stderr,
        )
    return matched_any


def archive_coverage_details(
    archive_dir: str, acquisition_time: str, bbox: List[float], filtered_path: str
) -> Dict[str, Any]:
    """Describe observed data and recorded sessions without inventing coverage."""
    dt = parse_utc(acquisition_time, allow_naive=True)
    if dt is None:
        raise ValueError("Coverage requires a valid acquisition timestamp")
    start, end = dt - timedelta(minutes=5), dt + timedelta(minutes=5)
    sessions = []
    for path in sorted(Path(archive_dir).glob("coverage_*.jsonl")):
        events = []
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                    event_time = parse_utc(value.get("time"))
                    if event_time is not None:
                        events.append(value)
                except (ValueError, TypeError, AttributeError):
                    continue  # partial final journal line after a crash
        if not events:
            continue
        first = parse_utc(events[0]["time"])
        last = parse_utc(events[-1]["time"])
        if first is None or last is None or first > end or last < start:
            continue
        sessions.append({"journal": str(path), "events": events})
    observations = 0
    if os.path.exists(filtered_path):
        with open(filtered_path, newline="", encoding="utf-8") as stream:
            observations = sum(1 for _ in csv.DictReader(stream))
    return {
        "status": "partial" if observations else "none",
        "coverage_known": False,
        "source": "aisstream.io local archive",
        "observations": observations,
        "query_bbox": bbox,
        "window_start": utc_string(start),
        "window_end": utc_string(end),
        "recording_sessions": sessions,
        "limitation": (
            "Observed messages and subscriptions do not establish receiver coverage "
            "or vessel silence; legacy records lack verified provenance."
        ),
    }


def select_polarization_bands(
    tiffs: List[str],
) -> "tuple[Optional[str], Optional[str], str]":
    """
    Picks the co-pol and cross-pol measurement bands from a product's TIFFs.

    Sentinel-1 IW GRD ships either VV+VH (product type 1SDV) or HH+HV (1SDH).
    Returns (co_pol_file, cross_pol_file, label).

    The pairing matters because everything downstream assumes the first band is
    co-pol and the second cross-pol. An earlier version fell back to "take the
    first two TIFFs in directory order" whenever it could not find VV/VH, which
    silently relabelled HH as VV on 1SDH products and fed the detectors data
    from a different scattering regime without any indication.
    """
    lower = {f: f.lower() for f in tiffs}

    def find(tag: str) -> Optional[str]:
        return next(
            (
                f
                for f in tiffs
                if f"-{tag}-" in lower[f] or lower[f].endswith(f"{tag}.tiff")
            ),
            None,
        )

    vv, vh = find("vv"), find("vh")
    if vv and vh:
        return vv, vh, "VV/VH"

    hh, hv = find("hh"), find("hv")
    if hh and hv:
        return hh, hv, "HH/HV"

    # Single-pol or an unrecognised naming scheme: report what was found rather
    # than guessing a pair.
    return None, None, "unknown"


def query_copernicus_data(
    aoi_data: Dict[str, Any],
    output_dir: str,
    user: str,
    password: str,
    target_date: str = "",
) -> Dict[str, Any]:
    """Queries and downloads actual Sentinel-1 SAR datasets using CDSE OData API."""
    import tempfile
    import zipfile

    # Convert GeoJSON AOI to WKT
    geom = aoi_data["features"][0]["geometry"]
    validate_aoi(geom)
    poly = shapely.geometry.shape(geom)
    footprint = poly.wkt

    print("Authenticating with Copernicus CDSE Keycloak OAuth2...", file=sys.stderr)
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    token_data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": user,
        "password": password,
    }

    try:
        response = requests.post(token_url, data=token_data, timeout=15)
        response.raise_for_status()
        token = response.json()["access_token"]
    except Exception as e:
        raise RuntimeError(f"Authentication with Copernicus CDSE failed: {e}")

    # Query matching Sentinel-1 product
    query_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    # OData filter assembled from named parts: as one literal it is unreadable
    # and well past any sane line length.
    base_filter = (
        "Collection/Name eq 'SENTINEL-1' "
        "and contains(Name, 'IW_GRDH') "
        "and contains(Name, '_1SDV_') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
    )
    if target_date:
        date = datetime.strptime(target_date, "%Y-%m-%d").date()
        following_date = (date + timedelta(days=1)).isoformat()
        window = (
            f" and ContentDate/Start ge {date.isoformat()}T00:00:00Z"
            f" and ContentDate/Start lt {following_date}T00:00:00Z"
        )
    else:
        start_date = (datetime.now(timezone.utc) - timedelta(days=15)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        window = f" and ContentDate/Start ge {start_date}"

    params: Dict[str, Any] = {
        "$filter": base_filter + window,
        "$orderby": "ContentDate/Start desc",
        "$top": 1,
    }
    headers = {"Authorization": f"Bearer {token}"}

    print("Searching for matching Sentinel-1 products...", file=sys.stderr)
    try:
        res = requests.get(query_url, params=params, headers=headers, timeout=20)
        res.raise_for_status()
        products = res.json().get("value", [])
    except Exception as e:
        raise RuntimeError(f"CDSE catalogue query failed: {e}")

    if not products:
        msg = "in the last 15 days" if not target_date else f"on date {target_date}"
        raise RuntimeError(
            f"No matching Sentinel-1 scenes found intersecting the AOI {msg}."
        )

    product = products[0]
    product_uuid = product["Id"]
    product_name = product["Name"]
    start_time_str = product["ContentDate"]["Start"]
    dt = parse_utc(start_time_str)
    if dt is None:
        raise ValueError("Catalogue product has no valid UTC acquisition timestamp")
    if target_date and dt.date().isoformat() != target_date:
        raise ValueError("Catalogue returned a product outside TARGET_DATE")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", product_name) or not re.fullmatch(
        r"[A-Fa-f0-9-]{36}", product_uuid
    ):
        raise ValueError("Invalid catalogue product identifier")

    print(f"Found product: {product_name} (UUID: {product_uuid})", file=sys.stderr)
    print(f"Acquisition time: {dt.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    os.makedirs(output_dir, exist_ok=True)

    # Check local cache first
    safe_name = (
        product_name if product_name.endswith(".SAFE") else f"{product_name}.SAFE"
    )
    product_path = os.path.join(output_dir, safe_name)

    local_cache_valid = False
    if os.path.exists(product_path):
        measurement_dir = os.path.join(product_path, "measurement")
        if os.path.exists(measurement_dir):
            tiffs = [
                f
                for f in os.listdir(measurement_dir)
                if f.endswith(".tiff") or f.endswith(".tif")
            ]
            co, cross, pol = select_polarization_bands(tiffs)
            if (
                co
                and cross
                and pol == "VV/VH"
                and os.path.isfile(os.path.join(product_path, "manifest.safe"))
            ):
                for band in (co, cross):
                    with rasterio.open(os.path.join(measurement_dir, band)) as raster:
                        raster.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
                local_cache_valid = True
                print(
                    f"Product {product_name} already exists locally; "
                    "skipping download.",
                    file=sys.stderr,
                )

    if not local_cache_valid:
        if os.path.exists(product_path):
            raise RuntimeError(
                f"Incomplete product cache at {product_path}; "
                "preserve or move it before retrying"
            )
        download_url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_uuid})/$value"

        print(f"Downloading product {product_name}...", file=sys.stderr)
        fd, temp_zip_path = tempfile.mkstemp(suffix=".zip.part", dir=output_dir)
        os.close(fd)

        try:
            with requests.get(
                download_url, headers=headers, stream=True, timeout=120
            ) as r:
                r.raise_for_status()
                bytes_received = 0
                with open(temp_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bytes_received += len(chunk)
                declared_length = r.headers.get("Content-Length")
                if declared_length and bytes_received != int(declared_length):
                    raise RuntimeError("Incomplete satellite product download")

            print(f"Extracting product to {output_dir}...", file=sys.stderr)
            with tempfile.TemporaryDirectory(
                prefix="safe_extract_", dir=output_dir
            ) as staging:
                with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                    for member in zip_ref.infolist():
                        target = (Path(staging) / member.filename).resolve()
                        if (
                            not target.is_relative_to(Path(staging).resolve())
                            or "\\" in member.filename
                        ):
                            raise RuntimeError(
                                "Unsafe path in satellite product archive"
                            )
                    # extractall verifies member CRC while writing. A partial
                    # extraction never becomes the accepted cache directory.
                    zip_ref.extractall(staging)
                extracted = Path(staging) / safe_name
                if not (extracted / "manifest.safe").is_file():
                    raise RuntimeError(
                        "Downloaded product is missing its SAFE manifest"
                    )
                extracted_bands = [
                    p.name for p in (extracted / "measurement").glob("*.tif*")
                ]
                co, cross, pol = select_polarization_bands(extracted_bands)
                if not co or not cross or pol != "VV/VH":
                    raise RuntimeError(
                        "Downloaded product lacks required VV/VH measurements"
                    )
                for band in (co, cross):
                    with rasterio.open(extracted / "measurement" / band) as raster:
                        raster.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
                os.replace(extracted, product_path)

        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

    measurement_dir = os.path.join(product_path, "measurement")
    if not os.path.exists(measurement_dir):
        raise RuntimeError(
            f"Measurement directory missing in extracted product: {measurement_dir}"
        )

    tiffs = [
        f
        for f in os.listdir(measurement_dir)
        if f.endswith(".tiff") or f.endswith(".tif")
    ]
    vv_file, vh_file, polarization = select_polarization_bands(tiffs)
    if vv_file is None or vh_file is None:
        raise RuntimeError(
            f"Could not locate a co-pol/cross-pol band pair in {measurement_dir}. "
            f"Found: {sorted(tiffs)}"
        )
    if polarization != "VV/VH":
        # The detectors are trained on VV/VH. An HH/HV product is a different
        # scattering regime, so results from it are not comparable — say so
        # rather than silently relabelling HH as VV, which is what this code
        # used to do via a "just take the first two files" fallback.
        raise ValueError(
            f"Unsupported polarization {polarization}: detector requires VV/VH"
        )

    # Measure the product's true ground footprint. Level-1 GRD rasters are in
    # radar geometry and carry GCPs rather than a CRS, so the footprint has to
    # be derived from whichever the product actually provides. This raises if it
    # provides neither, rather than assuming a projection the scene is not in.
    vv_full_path = os.path.join(measurement_dir, vv_file)
    georeferencing, scene_bbox = describe_raster_georeferencing(vv_full_path)
    print(
        f"Product georeferencing: {georeferencing}; ground footprint "
        f"lon {scene_bbox[0]:.4f}..{scene_bbox[2]:.4f}, "
        f"lat {scene_bbox[1]:.4f}..{scene_bbox[3]:.4f}",
        file=sys.stderr,
    )

    # AIS is filtered to the imaged area, not to the AOI that was searched with.
    # A Sentinel-1 swath is ~250 km across and merely intersects the AOI; any
    # vessel detected outside the AOI would otherwise have no telemetry to match
    # against and be misreported as a dark vessel.
    ais_bbox = pad_bbox(scene_bbox, AIS_MARGIN_DEG)
    acq_time_str_full = utc_string(dt)
    mission_id = f"mission_{dt.strftime('%Y%m%d_%H%M%S')}_{product_uuid[:8]}"
    mission_dir = mission_directory(mission_id)

    # This implementation integrates the local aisstream archive only. Other
    # public/licensed sources may exist but have not been integrated. Matching
    # messages are cooperative observations, not independent vessel labels.
    ais_telemetry_path = None
    ais_coverage = "none"

    ais_stream_dir = os.path.join(output_dir, "ais_stream")
    output_stream_path = str(mission_dir / "ais_filtered.csv")
    if filter_ais_stream_archive(
        ais_stream_dir, acq_time_str_full, ais_bbox, output_stream_path
    ):
        ais_telemetry_path = output_stream_path
        ais_coverage = "partial"

    if ais_telemetry_path is None:
        # No matching observations were available from the configured archive.
        # An empty, explicitly marked file allows downstream stages to report
        # cooperation as unassessable, without inventing synthetic AIS matches.
        ais_telemetry_path = output_stream_path
        if not os.path.exists(ais_telemetry_path):
            with open(ais_telemetry_path, "w", newline="", encoding="utf-8") as empty_f:
                csv.writer(empty_f).writerow(CSV_HEADER)
        print(
            "No matching AIS observations: vessel cooperation is unassessable. "
            "An absent AIS record must not be called a dark vessel.",
            file=sys.stderr,
        )

    return {
        "mission_id": mission_id,
        "status": "success",
        "mode": "production",
        "sar_product": product_name,
        "sar_product_id": product_uuid,
        "source_catalogue": "Copernicus Data Space Ecosystem",
        "catalogue_content_date": product["ContentDate"],
        "catalogue_checksum": product.get("Checksum", []),
        "sar_bands": {"VV": vv_full_path, "VH": os.path.join(measurement_dir, vh_file)},
        "ais_telemetry": ais_telemetry_path,
        "acquisition_time": utc_string(dt),
        "analysis_region": shapely.geometry.mapping(load_region()),
        "region_name": REGION_NAME,
        "ais_coverage_details": archive_coverage_details(
            ais_stream_dir, acq_time_str_full, ais_bbox, ais_telemetry_path
        ),
        "spatial_bounds": {
            "georeferencing": georeferencing,
            "crs": "EPSG:4326",
            "bbox": scene_bbox,
            "aoi_bbox": aoi_bbox(geom),
            "ais_coverage": ais_coverage,
        },
    }


def main() -> None:
    # Load .env file
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # 1. Establish workspace paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(base_dir, "data", "raw")

    target_date_override = os.getenv("TARGET_DATE", "").strip()

    # Check for AOI override via env var
    aoi_override = os.getenv("AOI_NAME")
    if aoi_override:
        if not aoi_override.endswith(".geojson"):
            aoi_override += ".geojson"
        aoi_path = os.path.join(base_dir, "configs", "aois", aoi_override)
    else:
        aoi_path = os.path.join(base_dir, "configs", "aois", DEFAULT_AOI)

    # Ensure raw output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Check credentials & mock flag
    mock_flag = os.getenv("MOCK_INGEST", "false").lower() == "true"
    user = os.getenv("COPERNICUS_USER", "")
    password = os.getenv("COPERNICUS_PASS", "")

    try:
        if aoi_override and os.path.basename(aoi_override) != aoi_override:
            raise ValueError("AOI_NAME must be a filename under configs/aois")
        aoi_data = load_aoi(aoi_path)
        validate_aoi(aoi_data["features"][0]["geometry"])
        if target_date_override:
            datetime.strptime(target_date_override, "%Y-%m-%d")

        if mock_flag:
            print("Explicit synthetic ingestion via MOCK_INGEST=true.", file=sys.stderr)
            result: Dict[str, Any] = generate_synthetic_data(aoi_path, output_dir)
        else:
            if not user or not password:
                raise RuntimeError(
                    "Production ingestion requires COPERNICUS_USER and "
                    "COPERNICUS_PASS. Set them in .env; use MOCK_INGEST=true "
                    "only for a synthetic demonstration."
                )
            print("Initiating Copernicus Sentinel-1 API Ingest...", file=sys.stderr)
            # An explicit date is a reproducibility constraint. No silent
            # substitution with recent imagery or unrelated archived AIS dates.
            result = query_copernicus_data(
                aoi_data, output_dir, user, password, target_date=target_date_override
            )

        # Open the mission's tracking run here, at the head of the pipeline, and
        # pass its id downstream so every later stage records into the same run.
        tracker = RunTracker.start(result["mission_id"])
        if tracker.run_id:
            result["mlflow_run_id"] = tracker.run_id
        sb = result.get("spatial_bounds", {})
        tracker.set_tags(
            {
                "mission_id": result["mission_id"],
                "mode": result.get("mode"),
                "stage": "ingest",
                "aoi": os.path.splitext(os.path.basename(aoi_path))[0],
                "georeferencing": sb.get("georeferencing"),
                "ais_coverage": sb.get("ais_coverage"),
            }
        )
        tracker.log_params(
            {
                "sar_product": result.get("sar_product"),
                "acquisition_time": result.get("acquisition_time"),
                "mode": result.get("mode"),
                "aoi": os.path.splitext(os.path.basename(aoi_path))[0],
                "georeferencing": sb.get("georeferencing"),
                "ais_coverage": sb.get("ais_coverage"),
            }
        )
        bbox = sb.get("bbox") or []
        if len(bbox) == 4:
            tracker.log_metrics(
                {
                    "scene_lon_span_deg": bbox[2] - bbox[0],
                    "scene_lat_span_deg": bbox[3] - bbox[1],
                }
            )
        ais_path_out = result.get("ais_telemetry")
        if ais_path_out and os.path.exists(ais_path_out):
            try:
                with open(ais_path_out, encoding="utf-8") as fh:
                    tracker.log_metrics({"ais_records": max(0, sum(1 for _ in fh) - 1)})
            except Exception:
                pass
        tracker.end()
        write_json(mission_directory(result["mission_id"]) / "ingest.json", result)

        # Write clean task-axi payload string to stdout
        print(json.dumps(result, indent=2))

    except Exception as err:
        error_payload = {"status": "failed", "error": str(err)}
        print(json.dumps(error_payload), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
