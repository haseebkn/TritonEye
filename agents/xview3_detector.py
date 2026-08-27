#!/usr/bin/env python3
"""
TritonEye xView3 Detector

Runs the xView3-SAR challenge winning ensemble (Eugene Khvedchenya, MIT licence)
over Sentinel-1 GRD imagery.

The model is a CircleNet-style encoder-decoder ensemble — EfficientNet B4/B5/V2S
backbones with U-Net decoders — producing dense predictions at stride 2 rather
than the stride-8 grid a YOLO detector uses. That matters at 10 m ground
spacing, where a 70 m vessel spans 7 pixels and occupies less than one YOLOv8
output cell.

Weights are distributed as a frozen TorchScript trace:
    https://github.com/BloodAxe/xView3-The-First-Place-Solution/releases

THREE INPUT DETAILS ARE LOAD-BEARING, and each one, when wrong, looks exactly
like "the model is broken" rather than "the input is wrong":

1. CALIBRATED sigma-nought in dB, not raw digital numbers. Their normalisation
   is sigmoid((x + 20) * 0.18), which saturates above roughly +10 dB. Raw GRD DN
   (median ~165) drives every pixel to 1.0 — a uniformly white image.
2. EXACTLY 2048x2048 input. Concrete sizes were baked in when the ensemble was
   traced, so its twelve sub-models only agree at that size; anything else fails
   inside the ensembling stack with a shape mismatch.
3. CHANNEL ORDER IS (VH, VV) — the reverse of what the rest of this pipeline
   uses. Measured on a known vessel: (VH, VV) peaked at 0.33 objectness 56 px
   from the target, (VV, VH) at 0.11 and 918 px away.

Measured against AIS ground truth on the full 2025-01-08 Boston scene (126
tiles, 141 AIS vessels in swath): 59% recall on vessels >=50 m at threshold
0.05, against 4% for the deployed yolov8n. See EVALUATION.md for the operating
point sweep and the precision caveat.

WHAT THIS MODEL DOES NOT DO, both of which are easy to misread from its outputs:

- It does not discriminate icebergs from vessels. An iceberg is a bright compact
  target against a dark sea -- the signature this model is trained to find -- and
  ice is nowhere in its training negatives. Over the Newfoundland AOI this is not
  a tuning problem: the pipeline defines a dark vessel as a detection with no AIS
  correlation, and an iceberg carries no transmitter, so every detected iceberg
  becomes a confident false alert by construction. See EVALUATION.md section 8.1.
- Its VESSEL_MAP head separates vessels from FIXED MARINE INFRASTRUCTURE, the
  negative class the xView3 challenge defined. A low score means "not a vessel
  under that training distribution", which lumps together infrastructure, ice,
  and terrain clutter. It is not a classifier, an ice filter, or a land filter.
  See EVALUATION.md section 8.2.
"""

import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

# Objectness peaks well below 1.0 — a CenterNet heatmap is not calibrated like a
# classifier, so this is nowhere near a 0.5 default.
#
# Swept against AIS ground truth over a full IW scene (141 AIS vessels in swath):
#
#   thresh   dets   recall(all)   recall(>=50m)   detections per AIS vessel
#     0.05   1419         51%            59%            10.1x
#     0.10    313         33%            52%             2.2x
#     0.15    160         21%            48%             1.1x
#     0.25     97         13%            44%             0.7x
#     0.40     40          8%            30%             0.3x
#
# Recall on resolvable vessels decays far more slowly than the detection count,
# so 0.15 is the better operating point: 8.9x fewer detections than 0.05 for an
# 11-point drop in >=50 m recall. Lower it toward 0.05 once a land mask exists
# to absorb the resulting coastal false positives.
DEFAULT_THRESHOLD = 0.15

# Half-width of the square emitted around each detection, in pixels (~150 m at
# 10 m spacing). The ensemble predicts a point, not an extent; its SIZE head is
# not yet decoded into metres, so this is a nominal marker size for display and
# correlation rather than a measured vessel footprint.
NOMINAL_HALF_EXTENT_PX = 15.0

# Detections closer than this in scene pixels are treated as the same vessel
# seen from two overlapping tiles (~200 m at 10 m spacing).
DEDUP_RADIUS_PX = 20.0

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
    for apply in (
        lambda: torch._C._jit_set_texpr_fuser_enabled(False),
        lambda: torch._C._jit_set_nvfuser_enabled(False),
        lambda: torch._C._jit_override_can_fuse_on_gpu(False),
        lambda: torch._C._jit_override_can_fuse_on_cpu(False),
        lambda: torch._C._jit_set_profiling_executor(False),
        lambda: torch._C._jit_set_profiling_mode(False),
    ):
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

        self._torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.threshold = threshold

        # The ensemble was traced under torch 1.10 and runs here under 2.x, so
        # TorchScript's JIT fuser tries to build fused CUDA kernels at runtime
        # via nvrtc/ptxas. That compilation needs host memory, and under memory
        # pressure it dies with "ptxas fatal: Memory allocation failure" —
        # partway through a scene, after the tiles already processed are lost.
        # Running the graph unfused costs some throughput but removes an entire
        # class of mid-run failure, which matters more for a ~30 min job.
        _disable_jit_fusion(torch)

        self.model = torch.jit.load(weights_path, map_location=self.device).eval()

    def _forward(self, tile_vh_vv: FloatArray) -> Dict[str, Any]:
        torch = self._torch
        x = torch.from_numpy(tile_vh_vv[None].astype("float32")).to(self.device)
        with torch.no_grad():
            return dict(self.model(x))

    def detect_tile(self, db_vv: FloatArray, db_vh: FloatArray) -> List[Dict[str, Any]]:
        """
        Detects vessels in one 2048x2048 tile of calibrated sigma-nought (dB).

        Returns detections in tile pixel coordinates, sub-pixel refined via the
        offset head.
        """
        if db_vv.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(
                f"xView3 ensemble requires {TILE_SIZE}x{TILE_SIZE} tiles, "
                f"got {db_vv.shape}"
            )

        torch = self._torch
        import torch.nn.functional as F  # noqa: N812 - conventional alias

        # Channel order is (VH, VV). See module docstring.
        stacked = np.stack([normalize_db(db_vh), normalize_db(db_vv)])
        out = self._forward(stacked)

        objectness = out[KEY_OBJECTNESS][0, 0]
        # The head already emits probabilities; only apply a sigmoid if the
        # values are outside [0, 1].
        if float(objectness.min()) < 0.0 or float(objectness.max()) > 1.0:
            objectness = torch.sigmoid(objectness)

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

        failed = 0
        with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
            origins = tile_origins(src_vv.width, src_vv.height, overlap)
            for n, (r0, c0) in enumerate(origins, 1):
                win = Window(c0, r0, TILE_SIZE, TILE_SIZE)
                dn_vv = src_vv.read(1, window=win).astype("float64")
                dn_vh = src_vh.read(1, window=win).astype("float64")
                if dn_vv.shape != (TILE_SIZE, TILE_SIZE):
                    continue
                if dn_vv.max() == 0 and dn_vh.max() == 0:
                    continue  # entirely in the product's zero-fill border

                db_vv = cal_vv.to_sigma0_db(dn_vv, r0, c0)
                db_vh = cal_vh.to_sigma0_db(dn_vh, r0, c0)

                # A scene is ~126 tiles and tens of minutes of work. A transient
                # per-tile failure — a CUDA OOM, a JIT compile failure under
                # memory pressure — should cost that tile, not the whole run, so
                # it is reported and skipped. A run that fails everywhere still
                # surfaces clearly: every tile logs its own reason.
                try:
                    detections = self.detect_tile(db_vv, db_vh)
                except Exception as e:
                    failed += 1
                    print(
                        f"  tile {n}/{len(origins)} at (row={r0}, col={c0}) "
                        f"failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}",
                        file=sys.stderr,
                    )
                    continue

                for det in detections:
                    det["col"] += c0
                    det["row"] += r0
                    yield det

                if progress:
                    print(f"  tile {n}/{len(origins)}", file=sys.stderr)

            if failed:
                print(
                    f"WARNING: {failed} of {len(origins)} tiles failed; "
                    "coverage of this scene is incomplete.",
                    file=sys.stderr,
                )


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
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: -d["score"])
    kept: List[Dict[str, Any]] = []
    cols = np.empty(0)
    rows = np.empty(0)

    for det in ordered:
        if (
            len(kept)
            and float(np.min(np.hypot(cols - det["col"], rows - det["row"])))
            < radius_px
        ):
            continue
        kept.append(det)
        cols = np.append(cols, det["col"])
        rows = np.append(rows, det["row"])
    return kept
