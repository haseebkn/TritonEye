import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box

from agents.artifacts import mission_directory, write_json
from agents.inference.inference_agent import (
    detector_threshold,
    generate_raster_windows,
    load_yolo_model,
    map_class_id,
    select_detector,
)
from agents.region import contains_points, load_region
from agents.xview3_detector import dedupe_detections, tile_origins


def test_scope_includes_newfoundland_and_labrador_not_boston() -> None:
    assert contains_points([-52.7, -57, -71.06], [47.5, 56, 42.36]) == [
        True,
        True,
        False,
    ]
    assert contains_points([float("nan")], [48]) == [False]
    with pytest.raises(ValueError, match="within"):
        load_region(box(-72, 41, -70, 43))
    assert load_region(box(-58, 55, -57, 56)).area == 1


@pytest.mark.parametrize("mission_id", ["../escape", "/absolute", "..", "a/b", "a\\b"])
def test_mission_path_cannot_escape(tmp_path: Path, mission_id: str) -> None:
    with pytest.raises(ValueError):
        mission_directory(mission_id, tmp_path)
    assert not list(tmp_path.iterdir())


def test_atomic_json_preserves_previous_valid_artifact(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"status": "ok"})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"status": "ok"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("overlap", [-1, 2048, 2049])
def test_invalid_tiling_fails_before_loop(overlap: int) -> None:
    with pytest.raises(ValueError):
        tile_origins(100, 100, overlap)
    with pytest.raises(ValueError):
        generate_raster_windows(100, 100, 2048, overlap)


def test_nearby_distinct_peaks_in_same_tile_are_not_suppressed() -> None:
    targets = [
        {"col": 10, "row": 10, "score": 0.9, "tile_id": (0, 0)},
        {"col": 14, "row": 10, "score": 0.8, "tile_id": (0, 0)},
        {"col": 10, "row": 10, "score": 0.7, "tile_id": (0, 100)},
    ]
    assert dedupe_detections(targets) == targets[:2]


@pytest.mark.parametrize("threshold", ["nan", "inf", "-0.1", "0", "1"])
def test_invalid_threshold_rejected(monkeypatch: Any, threshold: str) -> None:
    monkeypatch.setenv("TRITONEYE_XVIEW3_THRESHOLD", threshold)
    with pytest.raises(ValueError):
        detector_threshold({}, "xview3")


def test_unknown_detector_is_not_silent_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("TRITONEYE_DETECTOR", "typo")
    with pytest.raises(ValueError):
        select_detector({})


def test_missing_sar_weights_never_load_generic_yolo(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("YOLO_WEIGHTS", str(tmp_path / "absent.pt"))
    monkeypatch.setenv("YOLO_WEIGHTS_SHA256", "a" * 64)
    monkeypatch.delenv("HUGGINGFACE_MODEL_REPO", raising=False)
    with pytest.raises(FileNotFoundError):
        load_yolo_model(str(tmp_path), {})
    assert map_class_id(0, True) == 4
    assert map_class_id(8, False) == 4


def test_pinned_output_contract_rejects_logits(monkeypatch: Any) -> None:
    import torch

    from agents.xview3_detector import KEY_OBJECTNESS, TILE_SIZE, XView3Detector

    detector = XView3Detector.__new__(XView3Detector)
    detector._torch = torch
    detector.threshold = 0.15
    monkeypatch.setattr(
        detector, "_forward", lambda _: {KEY_OBJECTNESS: torch.full((1, 1, 2, 2), -2.0)}
    )
    tile = np.full((TILE_SIZE, TILE_SIZE), -20.0, dtype="float32")
    with pytest.raises(ValueError, match="finite in"):
        detector.detect_tile(tile, tile)
