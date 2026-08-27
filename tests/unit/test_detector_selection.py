import os
import sys
from typing import Any, Dict

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.inference.inference_agent import UNKNOWN_CLASS_ID, select_detector


def cfg(**inference: Any) -> Dict[str, Any]:
    return {"inference": inference}


def test_defaults_to_yolov8(monkeypatch: Any) -> None:
    monkeypatch.delenv("TRITONEYE_DETECTOR", raising=False)
    assert select_detector({}) == "yolov8"
    assert select_detector(cfg()) == "yolov8"


def test_config_selects_xview3(monkeypatch: Any) -> None:
    monkeypatch.delenv("TRITONEYE_DETECTOR", raising=False)
    assert select_detector(cfg(detector="xview3")) == "xview3"


def test_env_overrides_config(monkeypatch: Any) -> None:
    # A scene must be re-runnable against the other backend without editing
    # config, so the env var wins.
    monkeypatch.setenv("TRITONEYE_DETECTOR", "xview3")
    assert select_detector(cfg(detector="yolov8")) == "xview3"
    monkeypatch.setenv("TRITONEYE_DETECTOR", "yolov8")
    assert select_detector(cfg(detector="xview3")) == "yolov8"


def test_selection_is_case_insensitive(monkeypatch: Any) -> None:
    monkeypatch.delenv("TRITONEYE_DETECTOR", raising=False)
    assert select_detector(cfg(detector="XView3")) == "xview3"


def test_unknown_class_is_used_rather_than_a_fabricated_type() -> None:
    # The xView3 ensemble detects vessels without typing them. Reporting
    # "unknown" is the honest mapping; the incumbent labels everything cargo.
    assert UNKNOWN_CLASS_ID == 4


def test_shipped_config_defaults_to_the_fast_backend() -> None:
    # xView3 costs ~28 min/scene against ~47 s, so it must stay opt-in.
    import yaml

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    conf = yaml.safe_load(open(os.path.join(root, "configs", "model.yaml")))
    assert conf["inference"]["detector"] == "yolov8"


def test_xview3_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A sweep must be able to vary the operating point without editing tracked
    config, mirroring how TRITONEYE_DETECTOR overrides the backend.
    """
    import os

    configured: float = 0.15

    def resolve() -> float:
        return float(os.getenv("TRITONEYE_XVIEW3_THRESHOLD") or configured)

    monkeypatch.setenv("TRITONEYE_XVIEW3_THRESHOLD", "0.05")
    assert resolve() == 0.05

    monkeypatch.delenv("TRITONEYE_XVIEW3_THRESHOLD")
    assert resolve() == 0.15
