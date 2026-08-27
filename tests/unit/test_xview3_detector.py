import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.xview3_detector import (
    NORM_MIDPOINT_DB,
    OUTPUT_STRIDE,
    TILE_SIZE,
    XView3Detector,
    XView3Unavailable,
    normalize_db,
    resolve_weights,
    tile_origins,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ------------------------------------------------------------- normalisation


def test_normalization_midpoint_is_one_half() -> None:
    assert normalize_db(np.array([NORM_MIDPOINT_DB]))[0] == pytest.approx(0.5)


def test_normalization_spans_the_sar_range() -> None:
    # Ocean sigma-nought lives around -25..-5 dB; the transform must keep that
    # band well inside (0, 1) rather than saturating it.
    vals = normalize_db(np.array([-25.0, -15.0, -5.0]))
    assert 0.05 < vals[0] < vals[1] < vals[2] < 0.95


def test_raw_digital_numbers_would_saturate() -> None:
    # The failure mode this module exists to prevent: uncalibrated GRD DN drives
    # every pixel to 1.0, and the model sees a white image.
    raw_dn = np.array([165.0, 281.0, 8523.0])  # measured VV median / p99 / max
    assert np.all(normalize_db(raw_dn) > 0.999)


# -------------------------------------------------------------------- tiling


def test_tiles_cover_the_raster() -> None:
    w, h = 25846, 16687
    origins = tile_origins(w, h, overlap=128)
    assert origins, "no tiles generated"
    for r, c in origins:
        assert 0 <= r <= h - TILE_SIZE
        assert 0 <= c <= w - TILE_SIZE
    # every raster pixel must fall inside at least one tile
    assert min(r for r, _ in origins) == 0
    assert min(c for _, c in origins) == 0
    assert max(r for r, _ in origins) == h - TILE_SIZE
    assert max(c for _, c in origins) == w - TILE_SIZE


def test_tiles_overlap_by_the_requested_amount() -> None:
    origins = tile_origins(TILE_SIZE * 3, TILE_SIZE, overlap=128)
    cols = sorted({c for _, c in origins})
    assert cols[1] - cols[0] == TILE_SIZE - 128


def test_raster_smaller_than_one_tile_still_yields_an_origin() -> None:
    assert tile_origins(1000, 1000) == [(0, 0)]


def test_tile_count_is_tractable_for_a_full_scene() -> None:
    # A full IW GRD scene should be ~O(100) tiles; a regression that made this
    # O(1000) would push inference past a working day.
    n = len(tile_origins(25846, 16687, overlap=128))
    assert 50 < n < 400, n


# -------------------------------------------------------------- availability


def test_missing_weights_raise_clearly(tmp_path) -> None:
    with pytest.raises(XView3Unavailable):
        XView3Detector(str(tmp_path / "absent.jit"))


def test_resolve_weights_returns_none_when_absent(monkeypatch) -> None:
    monkeypatch.delenv("XVIEW3_WEIGHTS", raising=False)
    got = resolve_weights(str(REPO_ROOT))
    # Present only if the 1.3 GB artifact has been downloaded; either answer is
    # valid, but a returned path must exist.
    assert got is None or os.path.exists(got)


def test_resolve_weights_env_override_must_exist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XVIEW3_WEIGHTS", str(tmp_path / "nope.jit"))
    assert resolve_weights(str(REPO_ROOT)) is None
    real = tmp_path / "yes.jit"
    real.write_bytes(b"")
    monkeypatch.setenv("XVIEW3_WEIGHTS", str(real))
    assert resolve_weights(str(REPO_ROOT)) == str(real)


# ---------------------------------------------- contract, with weights present


@pytest.fixture(scope="module")
def loaded_detector() -> XView3Detector:
    """
    One shared detector for the contract tests.

    The traced ensemble is 1.3 GB; loading it per test dominated the whole
    suite's runtime.
    """
    return XView3Detector(resolve_weights(REPO_ROOT))


@pytest.mark.slow
@pytest.mark.skipif(
    resolve_weights(REPO_ROOT) is None, reason="xView3 weights not downloaded"
)
def test_wrong_tile_size_is_rejected(loaded_detector: XView3Detector) -> None:
    # The traced ensemble only agrees with itself at 2048x2048; a clear error
    # beats a shape mismatch thrown from inside the ensembling stack.
    small = np.full((512, 512), -12.0)
    with pytest.raises(ValueError, match="2048"):
        loaded_detector.detect_tile(small, small)


@pytest.mark.slow
@pytest.mark.skipif(
    resolve_weights(REPO_ROOT) is None, reason="xView3 weights not downloaded"
)
def test_output_stride_matches_head_resolution(
    loaded_detector: XView3Detector,
) -> None:
    import torch

    x = torch.zeros(1, 2, TILE_SIZE, TILE_SIZE, device=loaded_detector.device)
    with torch.no_grad():
        out = dict(loaded_detector.model(x))
    obj = out["CENTERNET_OUTPUT_OBJECTNESS_MAP"]
    assert obj.shape[-1] == TILE_SIZE // OUTPUT_STRIDE
    assert obj.shape[-2] == TILE_SIZE // OUTPUT_STRIDE


# -------------------------------------------------------------- deduplication


def test_dedup_collapses_overlapping_tile_duplicates() -> None:
    from agents.xview3_detector import dedupe_detections

    # same vessel seen from two overlapping tiles, plus a distinct one
    dets = [
        {"col": 100.0, "row": 100.0, "score": 0.4},
        {"col": 105.0, "row": 103.0, "score": 0.9},
        {"col": 900.0, "row": 900.0, "score": 0.3},
    ]
    kept = dedupe_detections(dets, radius_px=20.0)
    assert len(kept) == 2
    # the higher-scoring member of the pair survives
    assert kept[0]["score"] == 0.9


def test_dedup_keeps_genuinely_distinct_detections() -> None:
    from agents.xview3_detector import dedupe_detections

    dets = [
        {"col": 0.0, "row": 0.0, "score": 0.5},
        {"col": 50.0, "row": 0.0, "score": 0.5},
    ]
    assert len(dedupe_detections(dets, radius_px=20.0)) == 2


def test_dedup_handles_empty_input() -> None:
    from agents.xview3_detector import dedupe_detections

    assert dedupe_detections([]) == []
