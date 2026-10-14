"""Consume the frozen native NL contract; no fitting, scoring or label admission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_transfer import native_chip

SPEC = REPO_ROOT / "configs/native_nl/input_v010.json"
SPEC_SHA256 = "1ac869d85ddcaf3b89b4c1a2f9c5c391ae7214e5e7be00fff8462c9be2529e3a"


def specification() -> dict[str, Any]:
    if sha256_file(SPEC) != SPEC_SHA256:
        raise ValueError("Frozen native specification changed; release a new version")
    return dict(json.loads(SPEC.read_text()))


def prepare(
    values: Any, bbox: list[int], metadata: dict[str, Any], split: str
) -> dict[str, Any]:
    """Return a research tensor or an explicit abstention, never training approval.

    Callers supply hash-verified calibrated sigma0 arrays and original metadata.
    This consumer checks the contract, not physical class or geographic truth.
    """
    spec = specification()
    result: dict[str, Any] = {
        "status": "abstain",
        "tensor": None,
        "reason": None,
        "specification_version": spec["version"],
        "training_approved": False,
        "promotion_eligible": False,
    }
    if split not in {"train", "validation"}:
        raise ValueError("Only development roles permitted")
    required = {
        "mode": spec["mode"],
        "product_type": spec["product_type"],
        "band_order": spec["band_order"],
        "pixel_spacing_m": spec["pixel_spacing_m"],
        "thermal_noise_corrected": spec["thermal_noise_corrected"],
        "unit": spec["unit"],
    }
    if metadata.get("mission") not in spec["mission"] or any(
        metadata.get(k) != v for k, v in required.items()
    ):
        result["reason"] = "metadata_missing_or_incompatible"
        return result
    angle = metadata.get("incidence_center_deg")
    if (
        not isinstance(angle, (int, float))
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
    tensor = chip.astype("float32")
    if not np.isfinite(tensor).all():
        result["reason"] = "float32_overflow"
        return result
    result.update(status="research_tensor_only", tensor=tensor)
    return result


def load_specification(path: Path) -> dict[str, Any]:
    """Refuse silently edited or unsupported recipe versions."""
    proposed = json.loads(path.read_text())
    if proposed != specification():
        raise ValueError("Unsupported native recipe; use a new implemented version")
    return dict(proposed)
