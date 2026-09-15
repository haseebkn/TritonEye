#!/usr/bin/env python3
"""Calibrated Sentinel-1 VV/VH inference using the pinned xView3 ensemble.

This is an adaptation of BloodAxe's MIT-licensed challenge solution, not a
reproduction of its competition score. Inputs are sigma-zero dB, normalized as
sigmoid((dB + 20) * 0.18), in VH/VV order, with fixed 2048-pixel tiles.
The exported model already applies sigmoid to objectness. Output stride is two.

Per-tile peak decoding and cross-tile deduplication differ from upstream's
weighted heatmap stitching. Objectness is not a calibrated vessel probability.
The vessel head has not been validated here for iceberg discrimination; no
automatic dark-vessel conclusion is supported. NL precision and recall require
independent, co-temporal labels. AIS observations alone are incomplete references.
"""

import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.calibration import Calibrator

FloatArray = np.ndarray[Any, Any]

# The traced ensemble accepts only the size it was traced at.
TILE_SIZE = 2048

# Dense predictions come out at half the input resolution.
OUTPUT_STRIDE = 2

# xView3 SigmoidNormalization, from xview3/dataset/normalization.py.
NORM_MIDPOINT_DB = -20.0
NORM_TEMPERATURE = 0.18

# Provisional, not optimized on a labelled NL holdout.
DEFAULT_THRESHOLD = 0.15

# Half-width of the square emitted around each detection, in pixels (~150 m at
# 10 m spacing). The ensemble predicts a point, not an extent; its SIZE head is
# not yet decoded into metres, so this is a nominal marker size for display and
# correlation rather than a measured vessel footprint.
NOMINAL_HALF_EXTENT_PX = 15.0

# Detections closer than this in scene pixels are treated as the same vessel
# seen from two overlapping tiles (~60 m at 10 m spacing; provisional).
DEDUP_RADIUS_PX = 6.0

# Output keys, from xview3/centernet/constants.py.
KEY_OBJECTNESS = "CENTERNET_OUTPUT_OBJECTNESS_MAP"
KEY_OFFSET = "CENTERNET_OUTPUT_OFFSET"
KEY_SIZE = "CENTERNET_OUTPUT_SIZE"
KEY_VESSEL = "CENTERNET_OUTPUT_VESSEL_MAP"
KEY_FISHING = "CENTERNET_OUTPUT_FISHING_MAP"


class XView3Unavailable(RuntimeError):  # noqa: N818 - reads better than ...Error
    """Raised when the weights or their runtime dependencies are missing."""


def normalize_db(db: FloatArray) -> FloatArray:
    """Applies the ensemble's sigmoid normalisation to sigma-nought in dB."""
    return np.asarray(
        1.0 / (1.0 + np.exp(-((db - NORM_MIDPOINT_DB) * NORM_TEMPERATURE)))
    )


def tile_origins(width: int, height: int, overlap: int = 128) -> List[Tuple[int, int]]:
    """
    Top-left corners of the 2048x2048 tiles covering a raster.

    Tiles are clamped to the raster edge rather than padded, so the model always
    receives a full-size input; edge tiles overlap their neighbours more than
    `overlap` instead.
    """
    if width <= 0 or height <= 0 or not 0 <= overlap < TILE_SIZE:
        raise ValueError("Positive raster dimensions and 0 <= overlap < 2048 required")
    step = TILE_SIZE - overlap
    origins = []
    rows = list(range(0, max(1, height - TILE_SIZE + step), step))
    cols = list(range(0, max(1, width - TILE_SIZE + step), step))
    for r in rows:
        for c in cols:
            origins.append(
                (min(r, max(0, height - TILE_SIZE)), min(c, max(0, width - TILE_SIZE)))
            )
    return sorted(set(origins))


def _disable_jit_fusion(torch: Any) -> None:
    """
    Turns off TorchScript's CUDA kernel fusion.

    Each toggle is attempted independently and failures are ignored: these are
    private APIs whose availability varies across torch releases, and the point
    is best-effort hardening, not a guarantee. Missing one of them is not worth
    aborting a run over.
    """
    toggles: Tuple[Callable[[], Any], ...] = (
        lambda: torch._C._jit_set_texpr_fuser_enabled(False),
        lambda: torch._C._jit_set_nvfuser_enabled(False),
        lambda: torch._C._jit_override_can_fuse_on_gpu(False),
        lambda: torch._C._jit_override_can_fuse_on_cpu(False),
        lambda: torch._C._jit_set_profiling_executor(False),
        lambda: torch._C._jit_set_profiling_mode(False),
    )
    for apply in toggles:
        try:
            apply()
        except Exception:
            continue


class XView3Detector:
    """
    Wraps the traced ensemble with the preprocessing and decoding it expects.

    Construct once per run — loading 1.3 GB of weights dominates the cost of any
    single tile.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise XView3Unavailable(f"PyTorch is required: {e}") from e
        if not os.path.exists(weights_path):
            raise XView3Unavailable(f"Weights not found at {weights_path}")

        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.threshold = threshold
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Objectness threshold must be in (0, 1)")
        self.scene_stats: Dict[str, Any] = {}

        # The ensemble was traced under torch 1.10 and runs here under 2.x, so
        # TorchScript's JIT fuser tries to build fused CUDA kernels at runtime
        # via nvrtc/ptxas. That compilation needs host memory, and under memory
        # pressure it dies with "ptxas fatal: Memory allocation failure" —
        # partway through a scene, after the tiles already processed are lost.
        # Running the graph unfused costs some throughput but removes an entire
        # class of mid-run failure, which matters more for a ~30 min job.
        _disable_jit_fusion(torch)

        # Explicitly Any: torch ships partial stubs, so `jit.load` is an
        # untyped call where torch IS installed and resolves to Any where it is
        # NOT (CI type-checks without the ML stack). A `type: ignore` would be
        # correct in one environment and flagged unused in the other; widening
        # the attribute is correct in both.
        self._torch: Any = torch
        model = self._torch.jit.load(weights_path, map_location=self.device)
        self.model = model.eval()

    def _forward(self, tile_vh_vv: FloatArray) -> Dict[str, Any]:
        torch = self._torch
        x = torch.from_numpy(tile_vh_vv[None].astype("float32")).to(self.device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.device.startswith("cuda"),
            ),
        ):
            # Decode the small output maps on CPU, leaving GPU memory for the
            # ensemble. This does not change scores or input precision.
            return {key: value.detach().cpu() for key, value in self.model(x).items()}

    def detect_tile(self, db_vv: FloatArray, db_vh: FloatArray) -> List[Dict[str, Any]]:
        """
        Detects vessels in one 2048x2048 tile of calibrated sigma-nought (dB).

        Returns detections in tile pixel coordinates, sub-pixel refined via the
        offset head.
        """
        if db_vv.shape != (TILE_SIZE, TILE_SIZE) or db_vh.shape != db_vv.shape:
            raise ValueError(
                f"xView3 ensemble requires {TILE_SIZE}x{TILE_SIZE} tiles, "
                f"got VV={db_vv.shape}, VH={db_vh.shape}"
            )

        torch = self._torch
        import torch.nn.functional as F  # noqa: N812 - conventional alias

        # Channel order is (VH, VV). See module docstring.
        stacked = np.stack([normalize_db(db_vh), normalize_db(db_vv)])
        out = self._forward(stacked)

        objectness = out[KEY_OBJECTNESS][0, 0]
        # The pinned release applies its sigmoid inside the ensemble. A different
        # output contract is an error, not permission to silently rescale scores.
        if not bool(torch.isfinite(objectness).all()) or (
            float(objectness.min()) < 0.0 or float(objectness.max()) > 1.0
        ):
            raise ValueError("Pinned ensemble objectness must be finite in [0, 1]")

        offset = out[KEY_OFFSET][0]
        size_map = out[KEY_SIZE][0, 0]
        vessel_map = out[KEY_VESSEL][0, 0]

        # 3x3 max-pool NMS: a pixel survives if it is its own neighbourhood max.
        pooled = F.max_pool2d(objectness[None, None], 3, 1, 1)[0, 0]
        peaks = (objectness == pooled) & (objectness > self.threshold)
        ys, xs = torch.nonzero(peaks, as_tuple=True)

        detections: List[Dict[str, Any]] = []
        for i in range(len(ys)):
            hy, hx = int(ys[i]), int(xs[i])
            px = (hx + float(offset[0, hy, hx])) * OUTPUT_STRIDE
            py = (hy + float(offset[1, hy, hx])) * OUTPUT_STRIDE
            detections.append(
                {
                    "col": px,
                    "row": py,
                    "score": float(objectness[hy, hx]),
                    # Raw head outputs. SIZE is not in metres and VESSEL is not
                    # yet a calibrated probability; both need decoding against
                    # labelled data before they are reported to a user.
                    #
                    # VESSEL in particular is not an ice or land filter -- its
                    # trained negative class is fixed marine infrastructure. See
                    # the module docstring and EVALUATION.md section 8.2.
                    "size_raw": float(size_map[hy, hx]),
                    "vessel_raw": float(vessel_map[hy, hx]),
                }
            )
        return detections

    def detect_scene(
        self,
        vv_path: str,
        vh_path: str,
        overlap: int = 128,
        progress: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        Streams detections across a whole product, in raster pixel coordinates.

        Yields per tile so a caller can accumulate or checkpoint; a full IW scene
        is ~126 tiles and takes on the order of an hour on a single mid-range GPU.
        """
        import rasterio
        from rasterio.windows import Window

        cal_vv = Calibrator.for_measurement(vv_path)
        cal_vh = Calibrator.for_measurement(vh_path)
        if cal_vv is None or cal_vh is None:
            raise XView3Unavailable(
                "No calibration LUT for this product; the ensemble needs "
                "sigma-nought in dB and cannot run on raw digital numbers."
            )

        with (
            rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024),
            rasterio.open(vv_path) as src_vv,
            rasterio.open(vh_path) as src_vh,
        ):
            if (
                src_vv.shape != src_vh.shape
                or src_vv.transform != src_vh.transform
                or src_vv.crs != src_vh.crs
            ):
                raise ValueError("VV/VH raster grids must agree")
            vv_gcps, vv_crs = src_vv.gcps
            vh_gcps, vh_crs = src_vh.gcps
            vv_grid = [(p.row, p.col, p.x, p.y, p.z) for p in vv_gcps]
            vh_grid = [(p.row, p.col, p.x, p.y, p.z) for p in vh_gcps]
            if (
                vv_crs != vh_crs
                or len(vv_grid) != len(vh_grid)
                or (vv_grid and not np.allclose(vv_grid, vh_grid, rtol=0, atol=1e-8))
            ):
                raise ValueError("VV/VH ground control points must agree")
            origins = tile_origins(src_vv.width, src_vv.height, overlap)
            self.scene_stats = {
                "tiles_total": len(origins),
                "tiles_processed": 0,
                "tiles_nodata": 0,
                "tiles_failed": 0,
                "nodata_detections_rejected": 0,
            }
            for n, (r0, c0) in enumerate(origins, 1):
                win = Window(c0, r0, TILE_SIZE, TILE_SIZE)
                padded = src_vv.width < TILE_SIZE or src_vv.height < TILE_SIZE
                dn_vv = src_vv.read(1, window=win, boundless=padded, fill_value=0)
                dn_vh = src_vh.read(1, window=win, boundless=padded, fill_value=0)
                if dn_vv.max() == 0 and dn_vh.max() == 0:
                    self.scene_stats["tiles_nodata"] += 1
                    continue  # entirely in the product's zero-fill border

                db_vv = cal_vv.to_sigma0_db(dn_vv, r0, c0)
                db_vh = cal_vh.to_sigma0_db(dn_vh, r0, c0)

                # Never report a partially processed scene as a completed scan.
                try:
                    detections = self.detect_tile(db_vv, db_vh)
                except Exception as e:
                    self.scene_stats["tiles_failed"] += 1
                    raise RuntimeError(
                        f"Tile {n}/{len(origins)} failed; scene is incomplete"
                    ) from e
                self.scene_stats["tiles_processed"] += 1

                for det in detections:
                    row, col = int(round(det["row"])), int(round(det["col"]))
                    if (
                        not (
                            0 <= row < min(TILE_SIZE, src_vv.height - r0)
                            and 0 <= col < min(TILE_SIZE, src_vv.width - c0)
                        )
                        or dn_vv[row, col] <= 0
                        or dn_vh[row, col] <= 0
                    ):
                        self.scene_stats["nodata_detections_rejected"] += 1
                        continue
                    det["tile_id"] = (r0, c0)
                    det["col"] += c0
                    det["row"] += r0
                    yield det

                if progress:
                    print(f"  tile {n}/{len(origins)}", file=sys.stderr)


def resolve_weights(base_dir: str) -> Optional[str]:
    """Returns the traced ensemble path if present, honouring an env override."""
    override = os.getenv("XVIEW3_WEIGHTS")
    if override:
        return override if os.path.exists(override) else None
    default = os.path.join(base_dir, "models", "xview3", "traced_ensemble.jit")
    return default if os.path.exists(default) else None


def dedupe_detections(
    detections: List[Dict[str, Any]], radius_px: float = DEDUP_RADIUS_PX
) -> List[Dict[str, Any]]:
    """
    Removes duplicates arising from tile overlap, keeping the highest score.

    Tiles deliberately overlap so vessels near a seam are not clipped, which
    means those vessels are detected twice. Greedy highest-score-first
    suppression in scene pixel space; without it the detection count inflates
    and precision looks better than it is.
    """
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Deduplication radius must be finite and positive")
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: -d["score"])
    kept: List[Dict[str, Any]] = []
    cols: FloatArray = np.empty(0, dtype="float64")
    rows: FloatArray = np.empty(0, dtype="float64")

    for det in ordered:
        distances = np.hypot(cols - det["col"], rows - det["row"])
        if any(
            float(distance) < radius_px
            and (
                "tile_id" not in det
                or "tile_id" not in other
                or det["tile_id"] != other["tile_id"]
            )
            for other, distance in zip(kept, distances)
        ):
            continue
        kept.append(det)
        cols = np.asarray(np.append(cols, det["col"]), dtype="float64")
        rows = np.asarray(np.append(rows, det["row"]), dtype="float64")
    return kept
