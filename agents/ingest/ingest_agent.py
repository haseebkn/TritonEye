#!/usr/bin/env python3
"""
TritonEye Ingestion Agent
Retrieves Sentinel-1 SAR imagery and processes AIS telemetry.
Supports a deterministic mock mode for offline testing and pipeline validation.
"""

import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Agents are executed as standalone scripts, so the workspace root has to be on
# the path before the shared modules under agents/ can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
    coords = geom["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


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
    safe_name = (
        f"S1A_IW_GRDH_1SDV_{timestamp_str}_" "20260708T050025_061234_07ABCD_1234.SAFE"
    )
    safe_dir = os.path.join(output_dir, safe_name)
    measurement_dir = os.path.join(safe_dir, "measurement")
    os.makedirs(measurement_dir, exist_ok=True)

    # Coordinates of St. John's offshore center
    center_lon = -52.0
    center_lat = 47.5

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
    image_time = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")

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
            f"<crs>{proj_utm}</crs></xml>"
        )

    # 5. Write AIS track records to CSV (Active vessels only)
    ais_path = os.path.join(output_dir, "ais_mock.csv")
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
                        pt["time"].strftime("%Y-%m-%d %H:%M:%S"),
                        round(v["speed"], 1),
                        round(v["heading"], 1),
                    ]
                )

    # 6. Map the image corners back to lat/lon for the response metadata
    top_left_lon, top_left_lat = rev_transformer.transform(x_min, y_max)
    bottom_right_lon, bottom_right_lat = rev_transformer.transform(x_max, y_min)

    return {
        "mission_id": f"mission_{image_time.strftime('%Y%m%d_%H%M%S')}",
        "status": "success",
        "mode": "mock",
        "sar_product": safe_name,
        "sar_bands": {"VV": vv_path, "VH": vh_path},
        "ais_telemetry": ais_path,
        "acquisition_time": image_time.strftime("%Y-%m-%d %H:%M:%S"),
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

    acq_dt = datetime.fromisoformat(acquisition_time_str.replace(" ", "T"))
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
            chunk["timestamp_dt"] = pd.to_datetime(chunk[ts_col])

            mask_temp = (chunk["timestamp_dt"] >= time_window_start) & (
                chunk["timestamp_dt"] <= time_window_end
            )
            mask_spatial = (
                (chunk[lon_col] >= lon_min)
                & (chunk[lon_col] <= lon_max)
                & (chunk[lat_col] >= lat_min)
                & (chunk[lat_col] <= lat_max)
            )

            filtered_chunk = chunk[mask_temp & mask_spatial]
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

        columns_to_keep = [
            "mmsi",
            "lat",
            "lon",
            "timestamp",
            "speed_knots",
            "course_deg",
        ]
        renamed_df = renamed_df[
            [col for col in columns_to_keep if col in renamed_df.columns]
        ]

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
    to the acquisition window and bbox. This archive is
    global, so it is the fallback for acquisitions outside US waters — but it
    only has coverage from whenever the recorder was actually running.

    An acquisition window can straddle a UTC day boundary, so every daily file
    the window overlaps is checked.
    """
    if not os.path.isdir(archive_dir):
        print(f"AIS stream archive not found: {archive_dir}", file=sys.stderr)
        return False

    acq_dt = datetime.fromisoformat(acquisition_time_str.replace(" ", "T"))
    window_start = acq_dt - timedelta(minutes=5)
    window_end = acq_dt + timedelta(minutes=5)
    candidate_dates = {
        window_start.strftime("%Y-%m-%d"),
        window_end.strftime("%Y-%m-%d"),
    }

    if os.path.exists(output_path):
        os.remove(output_path)  # _filter_ais_file appends across daily files

    matched_any = False
    for date_str in sorted(candidate_dates):
        day_file = os.path.join(archive_dir, f"ais_stream_{date_str}.csv")
        if not os.path.exists(day_file):
            continue
        if _filter_ais_file(
            day_file, acquisition_time_str, bbox, output_path, _NORMALIZED_COLUMNS
        ):
            matched_any = True

    if not matched_any:
        print(
            f"No recorded aisstream.io coverage for {acquisition_time_str} "
            f"in {archive_dir}.",
            file=sys.stderr,
        )
    return matched_any


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
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
    )
    if target_date:
        window = (
            f" and ContentDate/Start ge {target_date}T00:00:00Z"
            f" and ContentDate/Start le {target_date}T23:59:59Z"
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
    try:
        dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)

    print(f"Found product: {product_name} (UUID: {product_uuid})", file=sys.stderr)
    print(f"Acquisition time: {dt.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    os.makedirs(output_dir, exist_ok=True)

    # Check local cache first
    product_path = os.path.join(output_dir, f"{product_name}.SAFE")
    if not os.path.exists(product_path):
        potential_paths = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.startswith(product_name)
        ]
        if potential_paths:
            product_path = potential_paths[0]

    local_cache_valid = False
    if os.path.exists(product_path):
        measurement_dir = os.path.join(product_path, "measurement")
        if os.path.exists(measurement_dir):
            tiffs = [
                f
                for f in os.listdir(measurement_dir)
                if f.endswith(".tiff") or f.endswith(".tif")
            ]
            if len(tiffs) >= 2:
                local_cache_valid = True
                print(
                    f"Product {product_name} already exists locally; "
                    "skipping download.",
                    file=sys.stderr,
                )

    if not local_cache_valid:
        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_uuid})/$value"

        # Resolve redirects manually so the Authorization header survives a
        # change of domain.
        try:
            r_head = requests.get(
                download_url, headers=headers, allow_redirects=False, timeout=15
            )
            if r_head.status_code in (301, 302, 303, 307, 308):
                download_url = r_head.headers["Location"]
        except Exception as e:
            print(
                f"Warning: Failed to pre-resolve download redirect: {e}",
                file=sys.stderr,
            )

        print(f"Downloading product {product_name}...", file=sys.stderr)
        fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)

        try:
            with requests.get(
                download_url, headers=headers, stream=True, timeout=120
            ) as r:
                r.raise_for_status()
                with open(temp_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            print(f"Extracting product to {output_dir}...", file=sys.stderr)
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)

        finally:
            if os.path.exists(temp_zip_path):
                os.remove(temp_zip_path)

    product_path = os.path.join(output_dir, f"{product_name}.SAFE")
    if not os.path.exists(product_path):
        potential_paths = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.startswith(product_name)
        ]
        if potential_paths:
            product_path = potential_paths[0]
        else:
            raise RuntimeError(
                f"Could not locate extracted product directory at {product_path}"
            )

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
        print(
            f"WARNING: product polarization is {polarization}, not VV/VH. "
            "Detectors are trained on VV/VH; detections from this scene are "
            "outside their trained domain and should not be trusted.",
            file=sys.stderr,
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
    acq_time_str_full = dt.strftime("%Y-%m-%d %H:%M:%S")

    # The locally recorded aisstream.io archive is the ONLY AIS source for this
    # operating area. MarineCadastre — the usual free bulk archive — is US Coast
    # Guard data and holds zero records east of -67.4W, so it does not reach
    # Newfoundland and Labrador at all; it is not consulted. That leaves the
    # recorder (agents/ais_recorder.py) as the sole ground-truth source, and it
    # only covers time during which it was actually running.
    ais_telemetry_path = None
    ais_coverage = "none"

    ais_stream_dir = os.path.join(output_dir, "ais_stream")
    output_stream_path = os.path.join(
        output_dir, f"ais_stream_{dt.strftime('%Y%m%d_%H%M%S')}_filtered.csv"
    )
    if filter_ais_stream_archive(
        ais_stream_dir, acq_time_str_full, ais_bbox, output_stream_path
    ):
        ais_telemetry_path = output_stream_path
        ais_coverage = "aisstream"

    if ais_telemetry_path is None:
        # No real telemetry exists for this acquisition. Previously this fell
        # back to ais_mock.csv — the synthetic generator's fabricated vessels —
        # which meant a production correlation run could silently match real
        # detections against invented MMSI numbers, or hide genuine dark
        # vessels behind mock "cooperative" tracks. An empty, correctly-schemed
        # file makes the absence of coverage explicit and auditable instead:
        # correlation still runs, and every detection is reported dark, but
        # spatial_bounds.ais_coverage records that this reflects missing
        # telemetry, not confirmed silence from real vessels.
        ais_telemetry_path = os.path.join(output_dir, "ais_empty.csv")
        if not os.path.exists(ais_telemetry_path):
            with open(ais_telemetry_path, "w", newline="", encoding="utf-8") as empty_f:
                csv.writer(empty_f).writerow(
                    ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
                )
        print(
            "WARNING: no recorded AIS coverage for this acquisition. All "
            "detections will be reported dark, reflecting missing telemetry "
            "coverage rather than confirmed silence. Newfoundland and Labrador "
            "has no historical AIS archive, so coverage exists only for periods "
            "when agents/ais_recorder.py was running.",
            file=sys.stderr,
        )

    acq_time_str = dt.strftime("%Y%m%d_%H%M%S")
    return {
        "mission_id": f"mission_{acq_time_str}",
        "status": "success",
        "mode": "production",
        "sar_product": product_name,
        "sar_bands": {"VV": vv_full_path, "VH": os.path.join(measurement_dir, vh_file)},
        "ais_telemetry": ais_telemetry_path,
        "acquisition_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
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

    # Check for TARGET_DATE override via env var
    target_date_override = os.getenv("TARGET_DATE")
    if target_date_override:
        target_dates = [target_date_override]
    else:
        target_dates = []
        if os.path.exists(output_dir):
            for name in sorted(os.listdir(output_dir)):
                if name.startswith("ais-") and os.path.isdir(
                    os.path.join(output_dir, name)
                ):
                    parts = name.split("-")
                    if (
                        len(parts) == 4
                        and len(parts[1]) == 4
                        and len(parts[2]) == 2
                        and len(parts[3]) == 2
                    ):
                        target_dates.append(f"{parts[1]}-{parts[2]}-{parts[3]}")

    # Check for AOI override via env var
    aoi_override = os.getenv("AOI_NAME")
    if aoi_override:
        if not aoi_override.endswith(".geojson"):
            aoi_override += ".geojson"
        aoi_path = os.path.join(base_dir, "configs", "aois", aoi_override)
    elif target_dates:
        aoi_path = os.path.join(base_dir, "configs", "aois", DEFAULT_AOI)
    else:
        aoi_path = os.path.join(
            base_dir, "configs", "aois", "st_johns_offshore.geojson"
        )

    # Ensure raw output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Check credentials & mock flag
    mock_flag = os.getenv("MOCK_INGEST", "false").lower() == "true"
    user = os.getenv("COPERNICUS_USER", "")
    password = os.getenv("COPERNICUS_PASS", "")

    # Fallback to mock mode if credentials are missing
    is_mock = mock_flag or not user or not password

    try:
        aoi_data = load_aoi(aoi_path)

        if is_mock:
            if mock_flag:
                print(
                    "Mock Ingestion flagged manually via MOCK_INGEST.",
                    file=sys.stderr,
                )
            else:
                print(
                    "Copernicus credentials missing. Falling back to mock "
                    "ingestion mode.",
                    file=sys.stderr,
                )

            result: Dict[str, Any] = generate_synthetic_data(aoi_path, output_dir)
        else:
            print("Initiating Copernicus Sentinel-1 API Ingest...", file=sys.stderr)

            result = {}
            if target_dates:
                for target_date in target_dates:
                    print(
                        f"Checking for matching Sentinel-1 imagery on {target_date}...",
                        file=sys.stderr,
                    )
                    try:
                        result = query_copernicus_data(
                            aoi_data,
                            output_dir,
                            user,
                            password,
                            target_date=target_date,
                        )
                        print(
                            f"Successfully aligned with date: {target_date}",
                            file=sys.stderr,
                        )
                        break
                    except Exception as e:
                        print(
                            f"No match or download failed for date {target_date}: {e}",
                            file=sys.stderr,
                        )

                if not result:
                    print("Falling back to query latest 15 days...", file=sys.stderr)
                    result = query_copernicus_data(aoi_data, output_dir, user, password)
            else:
                result = query_copernicus_data(aoi_data, output_dir, user, password)

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

        # Write clean task-axi payload string to stdout
        print(json.dumps(result, indent=2))

    except Exception as err:
        error_payload = {"status": "failed", "error": str(err)}
        print(json.dumps(error_payload), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
