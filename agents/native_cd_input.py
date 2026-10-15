"""Versioned C/D research tensors; no label admission or model selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_transfer import native_chip

SPEC = REPO_ROOT / "configs/native_nl/input_v020.json"
SPEC_SHA256 = "296840436cd533e4fd89c02fd4a32a0c67c6d4ae3c22de68dd92af181fcbec77"


def specification() -> dict[str, Any]:
    if sha256_file(SPEC) != SPEC_SHA256:
        raise ValueError("Frozen C/D specification changed; release a new version")
    return dict(json.loads(SPEC.read_text()))


def prepare(
    values: Any, bbox: list[int], metadata: dict[str, Any], split: str
) -> dict[str, Any]:
    spec = specification()
    if split not in {"train", "validation"}:
        raise ValueError("Only development roles permitted")
    result: dict[str, Any] = {
        "status": "abstain",
        "tensor": None,
        "reason": None,
        "specification_version": spec["version"],
        "training_approved": False,
        "independent_review": False,
        "promotion_eligible": False,
    }
    name = metadata.get("name", "")
    mission = metadata.get("mission")
    expected = {
        k: spec[k]
        for k in (
            "mode",
            "product_type",
            "band_order",
            "pixel_spacing_m",
            "thermal_noise_corrected",
            "unit",
            "processing_software",
        )
    }
    start, stop, instant = [
        parse_utc(metadata.get(k))
        for k in ("native_start_utc", "native_stop_utc", "native_center_utc")
    ]
    if (
        mission not in spec["mission"]
        or not isinstance(name, str)
        or not name.startswith(str(mission) + spec["product_name_marker"])
        or metadata.get("processing_version") not in spec["processing_versions"]
        or any(metadata.get(k) != v for k, v in expected.items())
        or start is None
        or stop is None
        or instant is None
        or not start <= instant <= stop
        or start >= stop
    ):
        result["reason"] = "metadata_missing_or_incompatible"
        return result
    try:
        named_start, named_stop = [
            datetime.strptime(part, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            for part in name.split("_")[4:6]
        ]
    except ValueError:
        result["reason"] = "native_product_time_mismatch"
        return result
    if (
        start.replace(microsecond=0) != named_start
        or stop.replace(microsecond=0) != named_stop
    ):
        result["reason"] = "native_product_time_mismatch"
        return result
    angle = metadata.get("incidence_center_deg")
    if (
        isinstance(angle, bool)
        or not isinstance(angle, (int, float))
        or not np.isfinite(angle)
        or not 0 < angle < 90
    ):
        result["reason"] = "incidence_missing_or_invalid"
        return result
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 3
        or values.shape[0] != 2
        or not np.isfinite(values).all()
        or len(bbox) != 4
        or any(type(v) is not int for v in bbox)
    ):
        result["reason"] = "invalid_array_or_bbox"
        return result
    try:
        chip = native_chip(values, bbox)
    except ValueError:
        chip = None
    if chip is None:
        result["reason"] = "incomplete_context_or_target_not_fitting"
        return result
    with np.errstate(over="ignore"):
        tensor = chip.astype("float32")
    if not np.isfinite(tensor).all():
        result["reason"] = "float32_overflow"
        return result
    result.update(status="research_tensor_only", tensor=tensor)
    return result


def load_specification(path: Path) -> dict[str, Any]:
    proposed = json.loads(path.read_text())
    if proposed != specification():
        raise ValueError("Unsupported C/D recipe")
    return dict(proposed)
