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
from typing import Any, Dict, List

# Third-party imports (handled gracefully if missing during bootstrap checks)
try:
    import numpy as np
    import rasterio
    import rasterio.transform
    import shapely.geometry
    from pyproj import Transformer
    import requests
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    # We will let the script fail gracefully on execution rather than crash on import


def load_aoi(aoi_path: str) -> Dict[str, Any]:
    """Loads and returns the geojson AOI configuration."""
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI configuration not found at {aoi_path}")
    with open(aoi_path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


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
        f"S1A_IW_GRDH_1SDV_{timestamp_str}_"
        "20260708T050025_061234_07ABCD_1234.SAFE"
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
            "projection": proj_utm,
            "crs": proj_wgs84,
            "bbox": [top_left_lon, bottom_right_lat, bottom_right_lon, top_left_lat],
        },
    }


def filter_real_ais_data(
    ais_dir: str,
    acquisition_time_str: str,
    bbox: List[float],
    output_path: str
) -> bool:
    """
    Parses the massive daily AIS telemetry file, filters it temporally and spatially,
    renames columns, and writes the filtered subset to a small CSV.
    """
    import pandas as pd

    # Locate the AIS CSV file in the directory
    if not os.path.exists(ais_dir):
        print(f"AIS directory not found: {ais_dir}", file=sys.stderr)
        return False
    
    files = [f for f in os.listdir(ais_dir) if os.path.isfile(os.path.join(ais_dir, f))]
    if not files:
        print(f"No files found in AIS directory: {ais_dir}", file=sys.stderr)
        return False
    
    ais_file_path = os.path.join(ais_dir, files[0])
    
    # Parse acquisition time
    acq_dt = datetime.fromisoformat(acquisition_time_str.replace(" ", "T"))
    time_window_start = acq_dt - timedelta(minutes=5)
    time_window_end = acq_dt + timedelta(minutes=5)
    
    # Bounding box bounds (min_lon, min_lat, max_lon, max_lat)
    lon_min, lat_min, lon_max, lat_max = bbox
    
    print(f"Filtering real AIS data from {ais_file_path}...", file=sys.stderr)
    print(f"Time window: {time_window_start} to {time_window_end}", file=sys.stderr)
    print(f"BBox bounds: Lon [{lon_min}, {lon_max}], Lat [{lat_min}, {lat_max}]", file=sys.stderr)
    
    filtered_rows = []
    chunk_size = 100000
    
    try:
        for chunk in pd.read_csv(ais_file_path, chunksize=chunk_size):
            # Convert timestamp column
            chunk["timestamp_dt"] = pd.to_datetime(chunk["base_date_time"])
            
            # Temporal filter
            mask_temp = (chunk["timestamp_dt"] >= time_window_start) & (chunk["timestamp_dt"] <= time_window_end)
            
            # Spatial filter
            mask_spatial = (
                (chunk["longitude"] >= lon_min) & 
                (chunk["longitude"] <= lon_max) & 
                (chunk["latitude"] >= lat_min) & 
                (chunk["latitude"] <= lat_max)
            )
            
            filtered_chunk = chunk[mask_temp & mask_spatial]
            if not filtered_chunk.empty:
                filtered_rows.append(filtered_chunk)
                
        if not filtered_rows:
            print("No real AIS records found matching the acquisition time and spatial bounds.", file=sys.stderr)
            return False
            
        full_filtered = pd.concat(filtered_rows)
        # Rename columns to match expected output schema
        renamed_df = full_filtered.rename(columns={
            "base_date_time": "timestamp",
            "longitude": "lon",
            "latitude": "lat",
            "sog": "speed_knots",
            "cog": "course_deg"
        })
        
        # Keep only required columns
        columns_to_keep = ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
        renamed_df = renamed_df[[col for col in columns_to_keep if col in renamed_df.columns]]
        
        # Save to output path
        renamed_df.to_csv(output_path, index=False)
        print(f"Successfully wrote {len(renamed_df)} real filtered AIS records to {output_path}", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"Error filtering real AIS data: {e}", file=sys.stderr)
        return False


def query_copernicus_data(
    aoi_data: Dict[str, Any], output_dir: str, user: str, password: str, target_date: str = ""
) -> Dict[str, Any]:
    """Queries and downloads actual Sentinel-1 SAR datasets using CDSE OData API."""
    import zipfile
    import tempfile

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
        "password": password
    }
    
    try:
        response = requests.post(token_url, data=token_data, timeout=15)
        response.raise_for_status()
        token = response.json()["access_token"]
    except Exception as e:
        raise RuntimeError(f"Authentication with Copernicus CDSE failed: {e}")

    # Query matching Sentinel-1 product
    query_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    if target_date:
        params = {
            "$filter": f"Collection/Name eq 'SENTINEL-1' and contains(Name, 'IW_GRDH') and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and ContentDate/Start ge {target_date}T00:00:00Z and ContentDate/Start le {target_date}T23:59:59Z",
            "$orderby": "ContentDate/Start desc",
            "$top": 1
        }
    else:
        start_date = (datetime.now(timezone.utc) - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "$filter": f"Collection/Name eq 'SENTINEL-1' and contains(Name, 'IW_GRDH') and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and ContentDate/Start ge {start_date}",
            "$orderby": "ContentDate/Start desc",
            "$top": 1
        }
    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("Searching for matching Sentinel-1 products...", file=sys.stderr)
    try:
        res = requests.get(query_url, params=params, headers=headers, timeout=20)
        res.raise_for_status()
        products = res.json().get("value", [])
    except Exception as e:
        raise RuntimeError(f"CDSE catalogue query failed: {e}")

    if not products:
        msg = f"in the last 15 days" if not target_date else f"on date {target_date}"
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
        potential_paths = [os.path.join(output_dir, name) for name in os.listdir(output_dir) if name.startswith(product_name)]
        if potential_paths:
            product_path = potential_paths[0]

    local_cache_valid = False
    if os.path.exists(product_path):
        measurement_dir = os.path.join(product_path, "measurement")
        if os.path.exists(measurement_dir):
            tiffs = [f for f in os.listdir(measurement_dir) if f.endswith(".tiff") or f.endswith(".tif")]
            if len(tiffs) >= 2:
                local_cache_valid = True
                print(f"Product {product_name} already exists locally. Skipping download.", file=sys.stderr)

    if not local_cache_valid:
        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_uuid})/$value"

        # Resolve redirects manually to avoid losing the Authorization header on domain change
        try:
            r_head = requests.get(download_url, headers=headers, allow_redirects=False, timeout=15)
            if r_head.status_code in (301, 302, 303, 307, 308):
                download_url = r_head.headers["Location"]
        except Exception as e:
            print(f"Warning: Failed to pre-resolve download redirect: {e}", file=sys.stderr)

        print(f"Downloading product {product_name}...", file=sys.stderr)
        fd, temp_zip_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)

        try:
            with requests.get(download_url, headers=headers, stream=True, timeout=120) as r:
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
        potential_paths = [os.path.join(output_dir, name) for name in os.listdir(output_dir) if name.startswith(product_name)]
        if potential_paths:
            product_path = potential_paths[0]
        else:
            raise RuntimeError(f"Could not locate extracted product directory at {product_path}")

    measurement_dir = os.path.join(product_path, "measurement")
    if not os.path.exists(measurement_dir):
        raise RuntimeError(f"Measurement directory missing in extracted product: {measurement_dir}")
    
    tiffs = [f for f in os.listdir(measurement_dir) if f.endswith(".tiff") or f.endswith(".tif")]
    vv_file = next((f for f in tiffs if "-vv-" in f.lower() or f.endswith("vv.tiff")), None)
    vh_file = next((f for f in tiffs if "-vh-" in f.lower() or f.endswith("vh.tiff")), None)
    
    if not vv_file or not vh_file:
        if len(tiffs) >= 2:
            vv_file = tiffs[0]
            vh_file = tiffs[1]
        else:
            raise RuntimeError(f"Could not locate VV and VH polarization bands in {measurement_dir}")

    # Detect the actual projection of the Sentinel-1 product
    try:
        with rasterio.open(os.path.join(measurement_dir, vv_file)) as src:
            projection_crs = src.crs.to_string() if src.crs else "EPSG:32622"
    except Exception:
        projection_crs = "EPSG:32622"

    # Filter the real local AIS data if available
    ais_telemetry_path = os.path.join(output_dir, "ais_mock.csv")
    try:
        coords = geom["coordinates"][0]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        bbox = [-52.678045, 47.053959, -51.333290, 47.941981]

    if target_date:
        ais_dir = os.path.join(output_dir, f"ais-{target_date}")
        output_ais_path = os.path.join(output_dir, f"ais_{target_date}_filtered.csv")
        success = filter_real_ais_data(ais_dir, dt.strftime("%Y-%m-%d %H:%M:%S"), bbox, output_ais_path)
        if success:
            ais_telemetry_path = output_ais_path

    acq_time_str = dt.strftime("%Y%m%d_%H%M%S")
    return {
        "mission_id": f"mission_{acq_time_str}",
        "status": "success",
        "mode": "production",
        "sar_product": product_name,
        "sar_bands": {
            "VV": os.path.join(measurement_dir, vv_file),
            "VH": os.path.join(measurement_dir, vh_file)
        },
        "ais_telemetry": ais_telemetry_path,
        "acquisition_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "spatial_bounds": {
            "projection": projection_crs,
            "crs": "EPSG:4326",
            "bbox": bbox
        }
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
                if name.startswith("ais-") and os.path.isdir(os.path.join(output_dir, name)):
                    parts = name.split("-")
                    if len(parts) == 4 and len(parts[1]) == 4 and len(parts[2]) == 2 and len(parts[3]) == 2:
                        target_dates.append(f"{parts[1]}-{parts[2]}-{parts[3]}")

    # Check for AOI override via env var
    aoi_override = os.getenv("AOI_NAME")
    if aoi_override:
        if not aoi_override.endswith(".geojson"):
            aoi_override += ".geojson"
        aoi_path = os.path.join(base_dir, "configs", "aois", aoi_override)
    elif target_dates:
        aoi_path = os.path.join(
            base_dir, "configs", "aois", "boston_offshore.geojson"
        )
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

            result = generate_synthetic_data(aoi_path, output_dir)
        else:
            print("Initiating Copernicus Sentinel-1 API Ingest...", file=sys.stderr)
            
            result = None
            if target_dates:
                for target_date in target_dates:
                    print(f"Checking for matching Sentinel-1 imagery on {target_date}...", file=sys.stderr)
                    try:
                        result = query_copernicus_data(aoi_data, output_dir, user, password, target_date=target_date)
                        print(f"Successfully aligned with date: {target_date}", file=sys.stderr)
                        break
                    except Exception as e:
                        print(f"No match or download failed for date {target_date}: {e}", file=sys.stderr)
                
                if not result:
                    print("Falling back to query latest 15 days...", file=sys.stderr)
                    result = query_copernicus_data(aoi_data, output_dir, user, password)
            else:
                result = query_copernicus_data(aoi_data, output_dir, user, password)

        # Write clean task-axi payload string to stdout
        print(json.dumps(result, indent=2))

    except Exception as err:
        error_payload = {"status": "failed", "error": str(err)}
        print(json.dumps(error_payload), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
