#!/usr/bin/env python3
"""
TritonEye Inference Agent
Runs computer vision target detection on Sentinel-1 SAR imagery.
Supports windowed processing and an explicitly synthetic demonstration detector.
"""

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
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
    from ultralytics import YOLO

    from agents.artifacts import mission_directory, sha256_file, write_json
    from agents.geo import Georeferencer
    from agents.region import REGION_NAME, contains_points, load_region
    from agents.tracking import RunTracker
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )
    raise


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
    if width <= 0 or height <= 0 or tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError(
            "Positive raster/tile dimensions and 0 <= overlap < tile_size required"
        )
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
    Coordinates are in whole-raster pixels.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be in (0, 1]")
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
    """Neither deployed detector provides validated vessel-type classifications."""
    return UNKNOWN_CLASS_ID


def load_yolo_model(base_dir: str, config: Dict[str, Any]) -> Tuple[Any, str]:
    """
    Loads explicitly configured SAR weights. Never substitutes a COCO model.

    Returns (model, weights_path) so callers can record exactly which weights
    produced a mission's detections.
    """
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    repo = os.getenv("HUGGINGFACE_MODEL_REPO")
    hf_filename = os.getenv("HUGGINGFACE_MODEL_FILE", "unquantized/best.pt")
    local_weights_dir = os.path.join(base_dir, "models")
    os.makedirs(local_weights_dir, exist_ok=True)
    local_weights_path = os.getenv("YOLO_WEIGHTS") or os.path.join(
        local_weights_dir, hf_filename
    )
    expected = os.getenv("YOLO_WEIGHTS_SHA256") or config.get("model", {}).get(
        "yolo_sha256"
    )
    if not expected:
        raise ValueError(
            "YOLO_WEIGHTS_SHA256 is required to pin the experimental SAR model"
        )
    if not os.path.isfile(local_weights_path) and repo:
        print(
            f"Attempting to download '{hf_filename}' from HF repo '{repo}'...",
            file=sys.stderr,
        )
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo,
                filename=hf_filename,
                token=token or None,
                revision=os.getenv("HUGGINGFACE_MODEL_REVISION"),
                local_dir=local_weights_dir,
            )
            print(
                f"Successfully downloaded model to {downloaded_path}", file=sys.stderr
            )
            local_weights_path = downloaded_path
        except Exception as e:
            raise RuntimeError("Configured SAR weights could not be retrieved") from e
    if not os.path.isfile(local_weights_path):
        raise FileNotFoundError(
            "SAR weights missing; configure YOLO_WEIGHTS or the Hugging Face model"
        )
    if sha256_file(local_weights_path) != str(expected).lower():
        raise ValueError("SAR weights SHA-256 does not match configured model identity")
    model = YOLO(local_weights_path)
    if set(str(v).lower() for v in model.names.values()) - {"ship", "vessel", "boat"}:
        raise ValueError("Configured detector is not a ship/vessel-only SAR model")
    return model, local_weights_path


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

    # Validate the configured SAR model upstream; no generic fallback is loaded.
    is_custom_model = False
    if not is_mock_mode and model is not None:
        try:
            is_custom_model = model.names.get(0) != "person"
        except Exception:
            is_custom_model = True

    # Open both VV and VH polarizations
    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
        if (
            src_vv.shape != src_vh.shape
            or src_vv.transform != src_vh.transform
            or src_vv.crs != src_vh.crs
        ):
            raise ValueError("VV/VH rasters are not aligned")
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
    Chooses xView3 by default; YOLO is an explicitly selected experiment.

    Env var TRITONEYE_DETECTOR overrides configs/model.yaml, so a scene can be
    re-run against the other backend without editing config.
    """
    configured = str(config.get("inference", {}).get("detector", "xview3")).lower()
    detector = os.getenv("TRITONEYE_DETECTOR", configured).strip().lower()
    if detector not in {"yolov8", "xview3"}:
        raise ValueError(f"Unknown detector: {detector!r}")
    return detector


def detector_threshold(config: Dict[str, Any], detector: str) -> float:
    cfg = config.get("inference", {})
    value = (
        (os.getenv("TRITONEYE_XVIEW3_THRESHOLD") or cfg.get("xview3_threshold", 0.15))
        if detector == "xview3"
        else cfg.get("conf_threshold", 0.35)
    )
    result = float(value)
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError(
            "Detector threshold must be finite and strictly between zero and one"
        )
    return result


def run_xview3_inference(
    vv_path: str,
    vh_path: str,
    base_dir: str,
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[float, float, float, float, float, int]], int, str]:
    """
    Runs the xView3 challenge-winning ensemble over a whole product.

    Returns boxes in the same whole-raster pixel schema as the YOLO path, so
    georeferencing and GeoJSON assembly downstream are unchanged. The ensemble
    predicts points rather than extents, so each detection becomes a nominal
    square; the class is reported as "unknown" because this model does not
    classify vessel type, unlike the incumbent which silently labels everything
    cargo.

    This default backend needs calibrated sigma-nought and substantial compute.
    See EVALUATION.md for the current validation and resource limitations.
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
    threshold = detector_threshold(config, "xview3")
    overlap = int(config.get("tiling", {}).get("xview3_overlap", 128))

    expected = config.get("model", {}).get("xview3_sha256")
    if not expected or sha256_file(weights) != expected:
        raise ValueError(
            "xView3 weights SHA-256 mismatch or absent; "
            "use the pinned reference artifact"
        )
    detector = XView3Detector(
        weights, threshold=threshold, device=inf_cfg.get("device", "cuda")
    )
    print(
        f"xView3 ensemble on {detector.device}, threshold {detector.threshold}",
        file=sys.stderr,
    )

    raw = list(detector.detect_scene(vv_path, vh_path, overlap=overlap, progress=True))
    kept = dedupe_detections(raw)
    if stats is not None:
        stats.update(
            {
                "compute_device": detector.device,
                "compute_precision": (
                    "cuda_autocast_float16"
                    if detector.device.startswith("cuda")
                    else "float32"
                ),
                "raw_detections": len(raw),
                "deduplicated_detections": len(kept),
                **getattr(detector, "scene_stats", {}),
            }
        )
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

    # Proximity to a provisional fixed reference is an uncertain review flag,
    # not proof that the detection is the installation rather than a vessel.
    infra_meta: Dict[str, Any] = {"enabled": False}
    infra_cfg = config.get("infrastructure", {}) or {}
    if infra_cfg.get("enabled", False):
        from agents import infrastructure as infra

        radius_m = float(infra_cfg.get("exclusion_radius_m", infra.EXCLUSION_RADIUS_M))
        is_infra, attributed = infra.classify_infrastructure(
            lons, lats, scene_bounds, radius_m=radius_m
        )
        for i, flagged in enumerate(is_infra):
            if flagged:
                surfaces[i] = infra.SURFACE_INFRASTRUCTURE
        per_installation = infra.summarize(attributed)
        infra_meta = {
            "enabled": True,
            "exclusion_radius_m": radius_m,
            "total": sum(per_installation.values()),
            "per_installation": per_installation,
        }
        if per_installation:
            named = ", ".join(f"{k}: {v}" for k, v in sorted(per_installation.items()))
            print(f"Infrastructure mask: {named}", file=sys.stderr)

    counts = lm.summarize(surfaces)
    infra_n = sum(1 for x in surfaces if x == "infrastructure_proximity")
    print(
        f"Land mask ({source}): {counts['water']} water, "
        f"{counts['coastal']} coastal, {counts['land']} land-rejected, "
        f"{infra_n} infrastructure of {len(surfaces)}",
        file=sys.stderr,
    )
    meta = {
        "enabled": True,
        "status": "ok",
        "source": source,
        "licence": mask.licence,
        "coastal_buffer_m": buffer_m,
        "counts": {**counts, "infrastructure": infra_n},
        "infrastructure": infra_meta,
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
    mission_dir = mission_directory(mission_id)
    region = load_region((payload.get("spatial_bounds") or {}).get("analysis_region"))

    if not vv_path or not vh_path:
        raise ValueError(
            "Invalid payload: Missing VV or VH band raster path references."
        )

    from shapely.geometry import box

    with rasterio.open(vv_path) as preflight_src:
        with Georeferencer.from_dataset(preflight_src) as preflight_geo:
            if not box(*preflight_geo.footprint_bounds()).intersects(region):
                raise ValueError("SAR scene does not intersect the NL study area")

    # 3. Load YOLOv8 model (unless running on simulated mock data)
    is_mock_mode = payload.get("mode", "production") == "mock"

    # Check override to force real inference even on mock data
    force_real = os.getenv("FORCE_REAL_INFERENCE", "false").lower() == "true"
    if force_real:
        is_mock_mode = False

    inf_cfg = config.get("inference", {})
    iou_threshold = inf_cfg.get("iou_threshold", 0.45)
    conf_threshold = detector_threshold(config, select_detector(config))
    tiling_cfg = config.get("tiling", {})
    tile_size = tiling_cfg.get("tile_size", 640)
    overlap = tiling_cfg.get("overlap", 128)

    # Real xView3 inference requires calibration LUTs absent from mock products.
    detector = select_detector(config)
    use_xview3 = detector == "xview3" and not is_mock_mode

    model = None
    resolved_weights = ""
    scene_stats: Dict[str, Any] = {}

    if use_xview3:
        inference_started = time.monotonic()
        final_detections, tiles_processed, resolved_weights = run_xview3_inference(
            vv_path, vh_path, base_dir, config, scene_stats
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
            detector = "mock"
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
        scene_stats.update(
            {"raw_detections": len(raw_detections), "tiles_processed": tiles_processed}
        )
    if is_mock_mode:
        detector = "mock"

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
    within_region = contains_points(centroid_lons, centroid_lats, region)

    features: List[Dict[str, Any]] = []
    outside_features: List[Dict[str, Any]] = []
    for idx, (_, _, _, _, score, class_id) in enumerate(final_detections):
        base = idx * 4
        ring = [[float(lons[base + k]), float(lats[base + k])] for k in range(4)]
        # RFC 7946 exterior rings follow the right-hand rule.
        area2 = sum(
            ring[k][0] * ring[(k + 1) % 4][1] - ring[(k + 1) % 4][0] * ring[k][1]
            for k in range(4)
        )
        if area2 < 0:
            ring.reverse()
        ring.append(ring[0])  # GeoJSON rings must close

        properties = {
            "target_id": f"TRITON-{idx:03d}",
            "class_id": UNKNOWN_CLASS_ID,
            "class_name": "unknown",
            "confidence": round(score, 3),
            "score_kind": "uncalibrated_objectness",
            "in_study_area": within_region[idx],
            "surface": "unknown",
            "ice_discrimination": "unvalidated",
            "geometry_kind": "nominal_display_marker" if use_xview3 else "detector_box",
        }
        if surfaces is not None:
            properties["surface"] = surfaces[idx]
            d = float(shore_dist[idx])
            properties["distance_to_shore_m"] = (
                None if not np.isfinite(d) else round(d, 1)
            )

        feature = {
            "type": "Feature",
            "id": idx,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": properties,
        }
        (features if within_region[idx] else outside_features).append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    # 6. Save the resulting GeoJSON
    geojson_path = str(mission_dir / "detections.geojson")
    write_json(geojson_path, geojson)
    excluded_path = str(mission_dir / "outside_study_area.geojson")
    write_json(
        excluded_path, {"type": "FeatureCollection", "features": outside_features}
    )

    # 7. Update payload and print to stdout. The scene footprint measured off the
    #    raster itself supersedes whatever bounds the AOI search produced.
    payload["detections_geojson"] = geojson_path
    spatial_bounds = payload.setdefault("spatial_bounds", {})
    spatial_bounds["georeferencing"] = geo.method
    spatial_bounds["crs"] = "EPSG:4326"
    spatial_bounds["scene_bbox"] = scene_bounds
    from shapely.geometry import mapping

    spatial_bounds["analysis_region"] = mapping(region)
    spatial_bounds["region_name"] = REGION_NAME
    payload["outside_study_area_geojson"] = excluded_path
    payload["outside_study_area_count"] = len(outside_features)
    payload["detector"] = detector
    # Which coastline produced the mask, under what licence, and how many
    # detections it rejected. The rejection tally is evidence in its own right,
    # so it belongs in the mission record rather than only in stderr.
    payload["landmask"] = landmask_meta
    processing = {
        "software_versions": {
            name: version(name)
            for name in ("torch", "rasterio", "numpy", "geopandas", "scipy")
        },
        "complete": True,
        "detector": detector,
        "weights_sha256": sha256_file(resolved_weights) if resolved_weights else None,
        "threshold": conf_threshold,
        "tile_size": 2048 if use_xview3 else tile_size,
        "tile_overlap": (
            tiling_cfg.get("xview3_overlap", 128) if use_xview3 else overlap
        ),
        "normalization": (
            "sigma0_db_sigmoid_vh_vv"
            if use_xview3
            else "mock_threshold" if is_mock_mode else "experimental_dn_vv_vh_vv"
        ),
        "georeferencing": geo.method,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            str(path.relative_to(base_dir)).replace("\\", "/"): sha256_file(path)
            for path in sorted((Path(base_dir) / "agents").rglob("*.py"))
        },
        "input_paths": {"VV": vv_path, "VH": vh_path},
        "input_sizes_bytes": {
            "VV": os.path.getsize(vv_path),
            "VH": os.path.getsize(vh_path),
        },
        "input_sha256": {"VV": sha256_file(vv_path), "VH": sha256_file(vh_path)},
        "inference_seconds": round(inference_seconds, 2),
        "in_region_detections": len(features),
        "outside_region_detections": len(outside_features),
        "precision_validated": False,
        "compute_precision": (
            "cuda_autocast_float16_or_cpu_float32" if use_xview3 else "float32"
        ),
        **scene_stats,
    }
    payload["processing"] = processing
    write_json(mission_dir / "processing.json", processing)
    write_json(mission_dir / "inference_payload.json", payload)

    # Record what produced these detections. The detector is the variable most
    # likely to change between missions, so its identity and thresholds are the
    # ones worth pinning to the run.
    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        tracker.set_tags({"stage": "inference", "detector": detector})
        tracker.log_params(
            {
                "detector": detector,
                "model_file": (
                    Path(resolved_weights).name if resolved_weights else "synthetic"
                ),
                "weights_sha256": processing["weights_sha256"],
                "model_classes": "unknown",
                "detector_threshold": conf_threshold,
                "config_sha256": processing["config_sha256"],
                "iou_threshold": iou_threshold,
                "tile_size": processing["tile_size"],
                "tile_overlap": processing["tile_overlap"],
                "mock_mode": is_mock_mode,
                "georeferencing": geo.method,
            }
        )
        tracker.log_metrics(
            {
                "raw_detections": scene_stats.get(
                    "raw_detections", len(raw_detections)
                ),
                "detections": len(features),
                "outside_region_detections": len(outside_features),
                "nms_suppressed": scene_stats.get("raw_detections", len(raw_detections))
                - len(final_detections),
                "inference_seconds": round(inference_seconds, 2),
                "tiles_processed": tiles_processed,
            }
        )
        tracker.log_artifact(geojson_path, "detections")
        tracker.log_artifact(str(mission_dir / "processing.json"), "provenance")
        if not is_mock_mode:
            tracker.register_detector(
                resolved_weights,
                name="tritoneye-sar-detector",
                metadata={
                    "detector": detector,
                    "weights_sha256": processing["weights_sha256"],
                    "classes": "unknown",
                },
            )
    finally:
        tracker.end("FAILED" if sys.exc_info()[0] else "FINISHED")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
