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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Agents are executed as standalone scripts, so the workspace root has to be on
# the path before the shared modules under agents/ can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Third-party imports (handled gracefully if missing during bootstrap checks)
try:
    import numpy as np
    import rasterio
    import rasterio.windows
    import yaml
    from huggingface_hub import hf_hub_download

    # Added for production YOLOv8 and Hugging Face Hub integration
    # ultralytics does not declare an explicit __all__ re-export for YOLO
    from ultralytics import YOLO  # type: ignore[attr-defined]

    from agents.geo import Georeferencer
    from agents.tracking import RunTracker
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    # We will let the script fail gracefully on execution rather than crash on import


# TritonEye class schema index for an unclassified target. The xView3
# ensemble detects vessels without typing them, so it reports this rather
# than inventing a class.
UNKNOWN_CLASS_ID = 4


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
    Applies Non-Maximum Suppression to eliminate duplicate bounding boxes.
    Box schema: (x_min, y_min, x_max, y_max, score, class_id).
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
    Maps the model's predicted class ID to TritonEye's class schema
    (0: cargo, 1: tanker, 2: fishing, 3: military, 4: unknown).

    NOTE: the deployed detector has a single class ('ship'), so this clamps
    every real detection to 0/'cargo'. The class labels in the output are
    therefore not meaningful — see readme 'Not implemented'.

    For the fallback COCO model, class 8 ('boat') maps to 2 ('fishing').
    """
    if not is_custom_model:
        if pred_class_id == 8:
            return 2  # Map to fishing
        return 4  # Unknown
    else:
        return max(0, min(4, pred_class_id))


def load_yolo_model(base_dir: str, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Downloads custom model weights from Hugging Face Hub when a repo and token
    are configured. HUGGINGFACE_MODEL_FILE gives the path inside the repo and
    may include subdirectories.
    Falls back to local file, or standard yolov8m.pt if retrieval fails.

    Returns (model, weights_path) so callers can record exactly which weights
    produced a mission's detections.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    repo = os.getenv("HUGGINGFACE_MODEL_REPO")
    hf_filename = os.getenv("HUGGINGFACE_MODEL_FILE", "unquantized/best.pt")
    model_name = config.get("model", {}).get("name", "yolov8m_sar_vessel")

    local_weights_dir = os.path.join(base_dir, "models")
    os.makedirs(local_weights_dir, exist_ok=True)
    local_weights_path = os.path.join(local_weights_dir, os.path.basename(hf_filename))

    if repo and token:
        print(
            f"Attempting to download '{hf_filename}' from HF repo '{repo}'...",
            file=sys.stderr,
        )
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=hf_filename,
                token=token,
                local_dir=local_weights_dir,
            )
            print(
                f"Successfully downloaded model to {downloaded_path}", file=sys.stderr
            )
            return YOLO(downloaded_path), downloaded_path
        except Exception as e:
            print(
                f"Hugging Face download failed: {e}. "
                "Falling back to local/default weights...",
                file=sys.stderr,
            )

    if os.path.exists(local_weights_path):
        print(
            f"Loading local model weights from {local_weights_path}...", file=sys.stderr
        )
        return YOLO(local_weights_path), local_weights_path

    fallback_model = "yolov8m.pt"
    print(
        f"Weights for '{model_name}' not found. Falling back to '{fallback_model}'...",
        file=sys.stderr,
    )
    return YOLO(fallback_model), fallback_model


def run_inference_on_tile(
    vv_path: str,
    vh_path: str,
    config: Dict[str, Any],
    model: Any,
    is_mock_mode: bool = False,
) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Runs YOLOv8 target detection over the VV and VH bands in sliding windows.
    With is_mock_mode, falls back to a pixel-intensity threshold detector.

    Boxes are returned in whole-raster pixel coordinates
    (col_min, row_min, col_max, row_max, score, class_id). Detection and NMS both
    stay in pixel space; georeferencing happens once, on the surviving boxes, in
    main(). Pixel spacing is uniform across the raster, so IoU is unaffected by
    the choice, and it keeps the expensive coordinate transform off the hot path.
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
            # Half a vessel footprint, in pixels. At Sentinel-1 GRD IW's 10 m
            # ground spacing this is the 25 m half-extent used previously.
            box_half_size_px = 2.5

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
                vv_win = vv_data[
                    win.row_off : win.row_off + win.height,
                    win.col_off : win.col_off + win.width,
                ]
                vh_win = vh_data[
                    win.row_off : win.row_off + win.height,
                    win.col_off : win.col_off + win.width,
                ]

            # Skip empty windows
            if vv_win.size == 0 or vh_win.size == 0:
                continue

            if is_mock_mode:
                # 1. Simple Threshold Mock Fallback
                rows, cols = np.where((vv_win >= vv_thresh) & (vh_win >= vh_thresh))
                for r, c in zip(rows, cols):
                    global_row = float(win.row_off + r)
                    global_col = float(win.col_off + c)

                    offset = float((int(global_row) + int(global_col)) % 100) / 285.0
                    score = 0.65 + offset

                    if score >= conf_threshold:
                        class_id = int((int(global_row) + int(global_col)) % 5)
                        raw_detections.append(
                            (
                                global_col - box_half_size_px,
                                global_row - box_half_size_px,
                                global_col + box_half_size_px,
                                global_row + box_half_size_px,
                                score,
                                class_id,
                            )
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

                        # Tile-local pixels to whole-raster pixels
                        raw_detections.append(
                            (
                                win.col_off + x_min_pixel,
                                win.row_off + y_min_pixel,
                                win.col_off + x_max_pixel,
                                win.row_off + y_max_pixel,
                                score,
                                class_id,
                            )
                        )
    return raw_detections


def select_detector(config: Dict[str, Any]) -> str:
    """
    Chooses the detection backend: "yolov8" (default) or "xview3".

    Env var TRITONEYE_DETECTOR overrides configs/model.yaml, so a scene can be
    re-run against the other backend without editing config.
    """
    configured = str(config.get("inference", {}).get("detector", "yolov8")).lower()
    return os.getenv("TRITONEYE_DETECTOR", configured).lower()


def run_xview3_inference(
    vv_path: str, vh_path: str, base_dir: str, config: Dict[str, Any]
) -> Tuple[List[Tuple[float, float, float, float, float, int]], int, str]:
    """
    Runs the xView3 challenge-winning ensemble over a whole product.

    Returns boxes in the same whole-raster pixel schema as the YOLO path, so
    georeferencing and GeoJSON assembly downstream are unchanged. The ensemble
    predicts points rather than extents, so each detection becomes a nominal
    square; the class is reported as "unknown" because this model does not
    classify vessel type, unlike the incumbent which silently labels everything
    cargo.

    This backend needs calibrated sigma-nought and ~28 min per scene, which is
    why it is opt-in. See EVALUATION.md section 5a.
    """
    from agents.xview3_detector import (
        NOMINAL_HALF_EXTENT_PX,
        XView3Detector,
        XView3Unavailable,
        dedupe_detections,
        resolve_weights,
        tile_origins,
    )

    weights = resolve_weights(base_dir)
    if not weights:
        raise XView3Unavailable(
            "xView3 detector selected but weights were not found. Download "
            "traced_ensemble.jit into models/xview3/ or set XVIEW3_WEIGHTS."
        )

    inf_cfg = config.get("inference", {})
    # Env override mirrors TRITONEYE_DETECTOR, so an operating-point sweep can
    # vary the threshold across runs without mutating tracked config.
    threshold = os.getenv("TRITONEYE_XVIEW3_THRESHOLD") or inf_cfg.get(
        "xview3_threshold"
    )
    overlap = int(config.get("tiling", {}).get("xview3_overlap", 128))

    detector = (
        XView3Detector(weights, threshold=float(threshold))
        if threshold is not None
        else XView3Detector(weights)
    )
    print(
        f"xView3 ensemble on {detector.device}, threshold {detector.threshold}",
        file=sys.stderr,
    )

    raw = list(detector.detect_scene(vv_path, vh_path, overlap=overlap, progress=False))
    kept = dedupe_detections(raw)
    print(
        f"xView3: {len(raw)} raw detections, {len(kept)} after cross-tile dedup",
        file=sys.stderr,
    )

    half = NOMINAL_HALF_EXTENT_PX
    boxes = [
        (
            d["col"] - half,
            d["row"] - half,
            d["col"] + half,
            d["row"] + half,
            d["score"],
            UNKNOWN_CLASS_ID,
        )
        for d in kept
    ]

    with rasterio.open(vv_path) as src:
        tiles = len(tile_origins(src.width, src.height, overlap))
    return boxes, tiles, weights


def classify_surfaces(
    lons: List[float],
    lats: List[float],
    scene_bounds: Optional[Sequence[float]],
    base_dir: str,
    config: Dict[str, Any],
) -> Tuple[Optional[List[str]], Any, Dict[str, Any]]:
    """
    Labels detection centroids water / coastal / land against a coastline.

    Returns (surfaces, distance_to_shore_m, metadata). `surfaces` is None when
    masking is disabled or unavailable, which leaves feature properties exactly
    as they were before this existed -- so `landmask.enabled: false` reproduces
    the previous output rather than emitting empty fields.

    A missing coastline download is a WARNING, not a failure. The alternative is
    aborting a 90-minute inference run at the final step over a reference file
    that can be fetched in a minute; the mission still yields detections, and
    the payload records that they were not masked.
    """
    from agents import landmask as lm

    cfg = config.get("landmask", {}) or {}
    if not cfg.get("enabled", False):
        return None, None, {"enabled": False}
    if not scene_bounds:
        print(
            "WARNING: no scene footprint; skipping land mask.",
            file=sys.stderr,
        )
        return None, None, {"enabled": True, "status": "no_footprint"}

    source = cfg.get("source", "osm")
    cache_dir = lm.resolve_cache_dir(base_dir, cfg.get("cache_dir"))
    buffer_m = float(cfg.get("coastal_buffer_m", lm.DEFAULT_COASTAL_BUFFER_M))

    try:
        mask = lm.LandMask.for_footprint(
            scene_bounds, source=source, cache_dir=cache_dir
        )
        surfaces, dist = mask.classify(lons, lats, coastal_buffer_m=buffer_m)
    except lm.LandMaskUnavailable as e:
        print(
            f"WARNING: land mask unavailable, detections unmasked: {e}", file=sys.stderr
        )
        return None, None, {"enabled": True, "status": "unavailable", "error": str(e)}

    counts = lm.summarize(surfaces)
    print(
        f"Land mask ({source}): {counts['water']} water, "
        f"{counts['coastal']} coastal, {counts['land']} land-rejected "
        f"of {len(surfaces)}",
        file=sys.stderr,
    )
    meta = {
        "enabled": True,
        "status": "ok",
        "source": source,
        "licence": mask.licence,
        "coastal_buffer_m": buffer_m,
        "counts": counts,
    }
    return surfaces, dist, meta


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

    inf_cfg = config.get("inference", {})
    iou_threshold = inf_cfg.get("iou_threshold", 0.45)
    conf_threshold = inf_cfg.get("conf_threshold", 0.35)
    tiling_cfg = config.get("tiling", {})
    tile_size = tiling_cfg.get("tile_size", 640)
    overlap = tiling_cfg.get("overlap", 128)

    # The xView3 backend is opt-in: it is far more accurate but needs ~28 min
    # per scene against ~47 s, and cannot run on mock data because it requires
    # a calibration LUT that synthetic products do not carry.
    detector = select_detector(config)
    use_xview3 = detector == "xview3" and not is_mock_mode

    model = None
    resolved_weights = ""

    if use_xview3:
        inference_started = time.monotonic()
        final_detections, tiles_processed, resolved_weights = run_xview3_inference(
            vv_path, vh_path, base_dir, config
        )
        inference_seconds = time.monotonic() - inference_started
        # Detections are already deduplicated across tiles by the detector.
        raw_detections = final_detections
    else:
        if detector == "xview3" and is_mock_mode:
            print(
                "xView3 requested but mock data has no calibration LUT; "
                "falling back to the mock detector.",
                file=sys.stderr,
            )
            detector = "yolov8"
        if not is_mock_mode:
            model, resolved_weights = load_yolo_model(base_dir, config)

        # 4. Execute sliding-window inference and NMS deduplication
        inference_started = time.monotonic()
        raw_detections = run_inference_on_tile(
            vv_path, vh_path, config, model, is_mock_mode
        )
        inference_seconds = time.monotonic() - inference_started

        with rasterio.open(vv_path) as _src:
            tiles_processed = len(
                generate_raster_windows(_src.width, _src.height, tile_size, overlap)
            )

        final_detections = non_max_suppression(raw_detections, iou_threshold)

    # 5. Georeference the surviving boxes into WGS84 GeoJSON.
    #    Sentinel-1 GRD rasters carry GCPs rather than a CRS; Georeferencer picks
    #    the right strategy and raises if the raster carries neither.
    with rasterio.open(vv_path) as src_vv:
        with Georeferencer.from_dataset(src_vv) as geo:
            print(f"Georeferencing detections via '{geo.method}'...", file=sys.stderr)
            scene_bounds = geo.footprint_bounds()

            # Map all four corners of every box in a single vectorised call. An
            # axis-aligned box in pixel space is a rotated quadrilateral on the
            # ground, so the corners are carried through individually rather
            # than reduced to a lon/lat min-max rectangle.
            corner_rows: List[float] = []
            corner_cols: List[float] = []
            for col_min, row_min, col_max, row_max, _, _ in final_detections:
                corner_rows.extend([row_min, row_min, row_max, row_max])
                corner_cols.extend([col_min, col_max, col_max, col_min])

            lons, lats = (
                geo.xy(corner_rows, corner_cols)
                if corner_rows
                else (np.empty(0), np.empty(0))
            )

    class_map = config.get("model", {}).get(
        "classes",
        {
            0: "cargo",
            1: "tanker",
            2: "fishing",
            3: "military",
            4: "unknown",
        },
    )

    # Classify each detection against the coastline before features are built,
    # so `surface` is available as a property. This runs on detection centroids
    # rather than the raster: point-in-polygon over a few hundred points costs
    # milliseconds, where rasterising a coastline to a 25000x16000 scene grid
    # would cost hundreds of MB. See agents/landmask.py.
    centroid_lons = [
        float(np.mean([lons[idx * 4 + k] for k in range(4)]))
        for idx in range(len(final_detections))
    ]
    centroid_lats = [
        float(np.mean([lats[idx * 4 + k] for k in range(4)]))
        for idx in range(len(final_detections))
    ]
    surfaces, shore_dist, landmask_meta = classify_surfaces(
        centroid_lons, centroid_lats, scene_bounds, base_dir, config
    )

    features = []
    for idx, (_, _, _, _, score, class_id) in enumerate(final_detections):
        base = idx * 4
        ring = [[float(lons[base + k]), float(lats[base + k])] for k in range(4)]
        ring.append(ring[0])  # GeoJSON rings must close

        properties = {
            "target_id": f"TRITON-{idx:03d}",
            "class_id": class_id,
            "class_name": class_map.get(class_id, "unknown"),
            "confidence": round(score, 3),
        }
        if surfaces is not None:
            properties["surface"] = surfaces[idx]
            d = float(shore_dist[idx])
            properties["distance_to_shore_m"] = (
                None if not np.isfinite(d) else round(d, 1)
            )

        features.append(
            {
                "type": "Feature",
                "id": idx,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties,
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}

    # 6. Save the resulting GeoJSON
    mission_dir = os.path.join(base_dir, "missions", mission_id)
    os.makedirs(mission_dir, exist_ok=True)

    geojson_path = os.path.join(mission_dir, "detections.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    # 7. Update payload and print to stdout. The scene footprint measured off the
    #    raster itself supersedes whatever bounds the AOI search produced.
    payload["detections_geojson"] = geojson_path
    spatial_bounds = payload.setdefault("spatial_bounds", {})
    spatial_bounds["georeferencing"] = geo.method
    spatial_bounds["crs"] = "EPSG:4326"
    spatial_bounds["scene_bbox"] = scene_bounds
    payload["detector"] = detector
    # Which coastline produced the mask, under what licence, and how many
    # detections it rejected. The rejection tally is evidence in its own right,
    # so it belongs in the mission record rather than only in stderr.
    payload["landmask"] = landmask_meta

    # Record what produced these detections. The detector is the variable most
    # likely to change between missions, so its identity and thresholds are the
    # ones worth pinning to the run.
    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        tracker.set_tags({"stage": "inference", "detector": detector})
        tracker.log_params(
            {
                "detector": detector,
                "model_repo": os.getenv("HUGGINGFACE_MODEL_REPO"),
                "model_file": os.getenv(
                    "HUGGINGFACE_MODEL_FILE", "unquantized/best.pt"
                ),
                "model_classes": getattr(model, "names", None) if model else "mock",
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "tile_size": tile_size,
                "tile_overlap": overlap,
                "mock_mode": is_mock_mode,
                "georeferencing": geo.method,
            }
        )
        tracker.log_metrics(
            {
                "raw_detections": len(raw_detections),
                "detections": len(final_detections),
                "nms_suppressed": len(raw_detections) - len(final_detections),
                "inference_seconds": round(inference_seconds, 2),
                "tiles_processed": tiles_processed,
            }
        )
        tracker.log_artifact(geojson_path, "detections")
        if not is_mock_mode:
            tracker.register_detector(
                resolved_weights,
                name="tritoneye-sar-detector",
                metadata={
                    "repo": os.getenv("HUGGINGFACE_MODEL_REPO", "unknown"),
                    "classes": getattr(model, "names", {}),
                    "conf_threshold": conf_threshold,
                },
            )
    finally:
        tracker.end()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
