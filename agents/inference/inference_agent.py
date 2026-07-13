#!/usr/bin/env python3
"""
TritonEye Inference Agent
Runs computer vision target detection on Sentinel-1 SAR imagery.
Supports windowed raster processing, NMS stitching, and mock target detection fallback.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Third-party imports (handled gracefully if missing during bootstrap checks)
try:
    import numpy as np
    import rasterio
    import rasterio.windows
    import yaml
    from pyproj import Transformer
    # Added for production YOLOv8 and Hugging Face Hub integration
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    # We will let the script fail gracefully on execution rather than crash on import


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads operational configuration from model.yaml."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="TritonEye Inference Agent")
    parser.add_argument(
        "--payload", type=str, help="JSON payload string passed directly"
    )
    parser.add_argument(
        "--payload-file", type=str, help="Path to JSON file containing payload"
    )
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Retrieves the payload from CLI args or stdin."""
    if args.payload:
        return json.loads(args.payload)  # type: ignore[no-any-return]
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # Stdin fallback
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())  # type: ignore[no-any-return]

    raise ValueError(
        "No input payload provided via --payload, --payload-file, or stdin."
    )


def generate_raster_windows(
    width: int, height: int, tile_size: int, overlap: int
) -> List[rasterio.windows.Window]:
    """Generates overlapping sliding windows over the raster dimensions."""
    step_size = tile_size - overlap
    windows = []

    for r in range(0, height, step_size):
        for c in range(0, width, step_size):
            # Clamp the window dimensions to raster edges
            w_width = min(tile_size, width - c)
            w_height = min(tile_size, height - r)
            windows.append(rasterio.windows.Window(c, r, w_width, w_height))

    return windows


def non_max_suppression(
    boxes: List[Tuple[float, float, float, float, float, int]],
    iou_threshold: float,
) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes.
    Boxes coordinate schema: (x_min, y_min, x_max, y_max, score, class_id)
    Coordinates are in UTM projection spatial values.
    """
    if not boxes:
        return []

    # Sort boxes by score descending
    sorted_boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while sorted_boxes:
        current = sorted_boxes.pop(0)
        keep.append(current)
        remaining = []

        for box in sorted_boxes:
            # Skip checking IOU if they are different class targets
            if current[5] != box[5]:
                remaining.append(box)
                continue

            # Calculate intersection
            ix_min = max(current[0], box[0])
            iy_min = max(current[1], box[1])
            ix_max = min(current[2], box[2])
            iy_max = min(current[3], box[3])

            iw = max(0.0, ix_max - ix_min)
            ih = max(0.0, iy_max - iy_min)
            intersection = iw * ih

            # Calculate union
            area_current = (current[2] - current[0]) * (current[3] - current[1])
            area_box = (box[2] - box[0]) * (box[3] - box[1])
            union = area_current + area_box - intersection

            iou = intersection / union if union > 0.0 else 0.0

            if iou < iou_threshold:
                remaining.append(box)
        sorted_boxes = remaining

    return keep


def map_class_id(pred_class_id: int, is_custom_model: bool) -> int:
    """
    Maps the model's predicted class ID to TritonEye's class schema (0: cargo, 1: tanker, 2: fishing, 3: military, 4: unknown).
    If it's the fallback COCO model, we map COCO class 8 ('boat') to 2 ('fishing') as a reasonable guess.
    If it's a custom model, we assume it already matches the target classes.
    """
    if not is_custom_model:
        if pred_class_id == 8:
            return 2  # Map to fishing
        return 4  # Unknown
    else:
        return max(0, min(4, pred_class_id))


def load_yolo_model(base_dir: str, config: Dict[str, Any]) -> Any:
    """
    Downloads custom model weights from Hugging Face Hub if credentials/repo are provided.
    Reads HUGGINGFACE_MODEL_FILE from env for the path inside the repo (may include subdirs).
    Falls back to local file, or standard yolov8m.pt if retrieval fails.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    repo = os.getenv("HUGGINGFACE_MODEL_REPO")
    hf_filename = os.getenv("HUGGINGFACE_MODEL_FILE", "unquantized/best.pt")
    model_name = config.get("model", {}).get("name", "yolov8m_sar_vessel")

    local_weights_dir = os.path.join(base_dir, "models")
    os.makedirs(local_weights_dir, exist_ok=True)
    local_weights_path = os.path.join(local_weights_dir, os.path.basename(hf_filename))

    if repo and token:
        print(f"Attempting to download '{hf_filename}' from HF repo '{repo}'...", file=sys.stderr)
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=hf_filename,
                token=token,
                local_dir=local_weights_dir,
            )
            print(f"Successfully downloaded model to {downloaded_path}", file=sys.stderr)
            return YOLO(downloaded_path)
        except Exception as e:
            print(f"Hugging Face download failed: {e}. Falling back to local/default weights...", file=sys.stderr)

    if os.path.exists(local_weights_path):
        print(f"Loading local model weights from {local_weights_path}...", file=sys.stderr)
        return YOLO(local_weights_path)

    fallback_model = "yolov8m.pt"
    print(f"Weights for '{model_name}' not found. Falling back to '{fallback_model}'...", file=sys.stderr)
    return YOLO(fallback_model)

def run_inference_on_tile(
    vv_path: str, vh_path: str, config: Dict[str, Any], model: Any, is_mock_mode: bool = False
) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Runs YOLOv8 target detection on the Sentinel-1 VV and VH bands using sliding windows.
    If is_mock_mode is True, falls back to the simple pixel-intensity threshold detector.
    """
    tiling_cfg = config.get("tiling", {})
    tile_size = tiling_cfg.get("tile_size", 640)
    overlap = tiling_cfg.get("overlap", 128)

    inf_cfg = config.get("inference", {})
    conf_threshold = inf_cfg.get("conf_threshold", 0.35)

    raw_detections: List[Tuple[float, float, float, float, float, int]] = []
    use_windowed_read = tiling_cfg.get("use_windowed_read", True)

    # Determine if it's the custom model or COCO fallback
    is_custom_model = False
    if not is_mock_mode and model is not None:
        try:
            is_custom_model = model.names.get(0) != "person"
        except Exception:
            is_custom_model = True

    # Open both VV and VH polarizations
    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
        width = src_vv.width
        height = src_vv.height
        transform = src_vv.transform

        has_georeferencing = src_vv.crs is not None

        # Determine scaling factors based on datatype
        is_production = src_vv.dtypes[0] in ("uint16", "float32")

        if is_mock_mode:
            # Setup threshold fallback parameters
            if is_production:
                vv_thresh = 10000
                vh_thresh = 5000
            else:
                vv_thresh = 255
                vh_thresh = 200
            box_half_size = 25.0

        # Slide windows across the raster
        windows = generate_raster_windows(width, height, tile_size, overlap)

        if not use_windowed_read:
            # Read entire bands into memory once for high performance
            vv_data = src_vv.read(1)
            vh_data = src_vh.read(1)

        for win in windows:
            if use_windowed_read:
                vv_win = src_vv.read(1, window=win)
                vh_win = src_vh.read(1, window=win)
            else:
                vv_win = vv_data[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
                vh_win = vh_data[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]

            # Skip empty windows
            if vv_win.size == 0 or vh_win.size == 0:
                continue

            if is_mock_mode:
                # 1. Simple Threshold Mock Fallback
                rows, cols = np.where((vv_win >= vv_thresh) & (vh_win >= vh_thresh))
                for r, c in zip(rows, cols):
                    global_row = win.row_off + r
                    global_col = win.col_off + c

                    if has_georeferencing:
                        x, y = rasterio.transform.xy(transform, global_row, global_col)
                    else:
                        x = global_col * 10.0
                        y = (height - global_row) * 10.0

                    x_min_spatial = x - box_half_size
                    x_max_spatial = x + box_half_size
                    y_min_spatial = y - box_half_size
                    y_max_spatial = y + box_half_size

                    offset = float((global_row + global_col) % 100) / 285.0
                    score = 0.65 + offset

                    if score >= conf_threshold:
                        class_id = int((global_row + global_col) % 5)
                        raw_detections.append(
                            (x_min_spatial, y_min_spatial, x_max_spatial, y_max_spatial, score, class_id)
                        )
            else:
                # 2. Real YOLOv8 Model Inference
                if np.max(vv_win) == 0 and np.max(vh_win) == 0:
                    continue

                vv_f = vv_win.astype(float)
                vh_f = vh_win.astype(float)

                if is_production:
                    vv_norm = np.clip((vv_f / 10000.0) * 255.0, 0, 255).astype(np.uint8)
                    vh_norm = np.clip((vh_f / 5000.0) * 255.0, 0, 255).astype(np.uint8)
                else:
                    vv_norm = np.clip(vv_f, 0, 255).astype(np.uint8)
                    vh_norm = np.clip(vh_f, 0, 255).astype(np.uint8)

                tile_h, tile_w = vv_win.shape
                tile_rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                tile_rgb[..., 0] = vv_norm
                tile_rgb[..., 1] = vh_norm
                tile_rgb[..., 2] = vv_norm

                results = model.predict(tile_rgb, conf=conf_threshold, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel = xyxy
                        score = float(box.conf[0])
                        pred_class_id = int(box.cls[0])

                        # Filter non-boat classes if using the generic COCO model
                        if not is_custom_model and pred_class_id != 8:
                            continue

                        class_id = map_class_id(pred_class_id, is_custom_model)

                        if has_georeferencing:
                            x_min_coord, y_max_coord = rasterio.transform.xy(transform, win.row_off + y_min_pixel, win.col_off + x_min_pixel)
                            x_max_coord, y_min_coord = rasterio.transform.xy(transform, win.row_off + y_max_pixel, win.col_off + x_max_pixel)
                            
                            x_min_spatial = min(x_min_coord, x_max_coord)
                            x_max_spatial = max(x_min_coord, x_max_coord)
                            y_min_spatial = min(y_min_coord, y_max_coord)
                            y_max_spatial = max(y_min_coord, y_max_coord)
                        else:
                            x_min_spatial = (win.col_off + x_min_pixel) * 10.0
                            x_max_spatial = (win.col_off + x_max_pixel) * 10.0
                            y_min_spatial = (height - (win.row_off + y_max_pixel)) * 10.0
                            y_max_spatial = (height - (win.row_off + y_min_pixel)) * 10.0

                        raw_detections.append(
                            (x_min_spatial, y_min_spatial, x_max_spatial, y_max_spatial, score, class_id)
                        )
    return raw_detections


def main() -> None:
    # Load dotenv to configure environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 1. Parse arguments and configuration
    args = parse_arguments()
    payload = get_payload(args)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(base_dir, "configs", "model.yaml")
    config = load_yaml_config(config_path)

    # 2. Extract paths from the Phase 2 payload
    sar_bands = payload.get("sar_bands", {})
    vv_path = sar_bands.get("VV")
    vh_path = sar_bands.get("VH")
    mission_id = payload.get("mission_id", "mission_default")

    if not vv_path or not vh_path:
        raise ValueError(
            "Invalid payload: Missing VV or VH band raster path references."
        )

    # 3. Load YOLOv8 model (unless running on simulated mock data)
    is_mock_mode = payload.get("mode", "production") == "mock"
    
    # Check override to force real inference even on mock data
    force_real = os.getenv("FORCE_REAL_INFERENCE", "false").lower() == "true"
    if force_real:
        is_mock_mode = False
        
    model = None
    if not is_mock_mode:
        model = load_yolo_model(base_dir, config)

    # 4. Execute sliding-window inference and NMS deduplication
    raw_detections = run_inference_on_tile(vv_path, vh_path, config, model, is_mock_mode)

    inf_cfg = config.get("inference", {})
    iou_threshold = inf_cfg.get("iou_threshold", 0.45)

    final_detections = non_max_suppression(raw_detections, iou_threshold)

    # 4. Reproject UTM/pixel coordinates back to standard WGS84 GeoJSON
    has_georeferencing = False
    with rasterio.open(vv_path) as src_vv:
        if src_vv.crs is not None:
            has_georeferencing = True
        width = src_vv.width
        height = src_vv.height

    spatial_bounds = payload.get("spatial_bounds", {})
    projection_crs = spatial_bounds.get("projection", "EPSG:32622")
    rev_transformer = Transformer.from_crs(projection_crs, "EPSG:4326", always_xy=True)

    class_map = config.get("model", {}).get("classes", {
        0: "cargo",
        1: "tanker",
        2: "fishing",
        3: "military",
        4: "unknown",
    })

    features = []
    for idx, (x_min, y_min, x_max, y_max, score, class_id) in enumerate(
        final_detections
    ):
        if has_georeferencing:
            # Reproject corners back to degrees lat/lon
            lon_min, lat_min = rev_transformer.transform(x_min, y_min)
            lon_max, lat_max = rev_transformer.transform(x_max, y_max)
        else:
            # Linear interpolation mapping to WGS84 using spatial_bounds bbox
            bbox = spatial_bounds.get("bbox", [-52.678045, 47.053959, -51.333290, 47.941981])
            
            col_min = x_min / 10.0
            col_max = x_max / 10.0
            row_min = height - y_max / 10.0
            row_max = height - y_min / 10.0
            
            lon_min = bbox[0] + (col_min / width) * (bbox[2] - bbox[0])
            lon_max = bbox[0] + (col_max / width) * (bbox[2] - bbox[0])
            lat_min = bbox[3] - (row_max / height) * (bbox[3] - bbox[1])
            lat_max = bbox[3] - (row_min / height) * (bbox[3] - bbox[1])

        # Bounding box polygon footprint
        coordinates = [
            [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min],
            ]
        ]

        features.append(
            {
                "type": "Feature",
                "id": idx,
                "geometry": {"type": "Polygon", "coordinates": coordinates},
                "properties": {
                    "target_id": f"TRITON-{idx:03d}",
                    "class_id": class_id,
                    "class_name": class_map.get(class_id, "unknown"),
                    "confidence": round(score, 3),
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    # 5. Save the resulting GeoJSON
    mission_dir = os.path.join(base_dir, "missions", mission_id)
    os.makedirs(mission_dir, exist_ok=True)

    geojson_path = os.path.join(mission_dir, "detections.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    # 6. Update payload and print to stdout
    payload["detections_geojson"] = geojson_path
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
