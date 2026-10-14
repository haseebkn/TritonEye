"""Frozen local-contrast HH/HV research candidates, never vessel probabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import find_objects, label, uniform_filter

from agents.artifacts import REPO_ROOT, sha256_file

CONFIG_PATH = "configs/candidates/hh_hv_local_contrast_v010.json"
CONFIG_SHA256 = "59360b88d1f07056d54e82e3fd6dbc33f6923f31eb6aa9797a1d36b161b5245d"


@dataclass(frozen=True)
class Settings:
    version: str
    outer: int
    guard: int
    thresholds: tuple[float, float]
    maximum_pixels: int


def frozen_settings() -> tuple[Settings, dict[str, Any]]:
    path = REPO_ROOT / CONFIG_PATH
    if sha256_file(path) != CONFIG_SHA256:
        raise ValueError("Frozen candidate configuration changed; use a new version")
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        Settings(
            value["version"],
            value["outer_window_px"],
            value["guard_window_px"],
            (
                float(value["contrast_threshold_db"][0]),
                float(value["contrast_threshold_db"][1]),
            ),
            value["maximum_roi_pixels"],
        ),
        value,
    )


def generate(
    images: Any, *, polarizations: tuple[str, str], settings: Settings
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use calibrated HH/HV only, with full ring support and no classifier/labels.

    Annular background averages LINEAR power. Bright connected pixels produce one
    candidate each; this can merge vessels and cannot identify clutter or ice.
    No candidate is removed by size, shoreline, score ranking or classifier output.
    """
    if polarizations != ("HH", "HV"):
        raise ValueError("Native HH then HV required; no polarization substitution")
    if (
        images.ndim != 3
        or images.shape[0] != 2
        or np.prod(images.shape[1:]) > settings.maximum_pixels
        or min(images.shape[1:]) < settings.outer
        or settings.outer % 2 != 1
        or settings.guard % 2 != 1
        or not 1 <= settings.guard < settings.outer
        or not np.isfinite(settings.thresholds).all()
    ):
        raise ValueError("Bounded two-band ROI and valid odd ring windows required")
    valid = np.isfinite(images).all(axis=0) & (images > -60).all(axis=0)
    power = np.zeros(images.shape, dtype="float64")
    with np.errstate(over="ignore", invalid="ignore"):
        power[:, valid] = np.power(10.0, images[:, valid].astype("float64") / 10.0)
    if not np.isfinite(power).all():
        raise ValueError("Nonfinite linear sigma0; no guessed clipping")
    outer, guard = settings.outer, settings.guard
    count = outer**2 - guard**2
    ring_count = np.rint(
        uniform_filter(valid.astype("float64"), outer, mode="constant") * outer**2
        - uniform_filter(valid.astype("float64"), guard, mode="constant") * guard**2
    )
    supported = valid & (ring_count == count)
    # Counts alone cannot justify padding at any ROI boundary.
    radius = outer // 2
    supported[:radius, :] = supported[-radius:, :] = False
    supported[:, :radius] = supported[:, -radius:] = False
    contrast = np.full(images.shape, -np.inf, dtype="float64")
    for band in (0, 1):
        background = (
            uniform_filter(power[band], outer, mode="constant") * outer**2
            - uniform_filter(power[band], guard, mode="constant") * guard**2
        ) / count
        usable = supported & (background > 0) & (power[band] > 0)
        contrast[band, usable] = 10 * np.log10(power[band, usable] / background[usable])
        supported &= usable
    margins = contrast - np.asarray(settings.thresholds)[:, None, None]
    peak_margin = margins.max(axis=0)
    bright = supported & (peak_margin >= 0)
    components, total = label(bright, structure=np.ones((3, 3), dtype="uint8"))
    candidates = []
    for index, box in enumerate(find_objects(components), start=1):
        if box is None:
            continue
        ys, xs = box
        member = components[box] == index
        scores = np.where(member, peak_margin[box], -np.inf)
        dy, dx = np.unravel_index(np.argmax(scores), scores.shape)
        row, col = ys.start + int(dy), xs.start + int(dx)
        bbox = [xs.start, ys.start, xs.stop, ys.stop]
        context = (
            "component_exceeds_classifier_chip"
            if xs.stop - xs.start > 75 or ys.stop - ys.start > 75
            else (
                "roi_boundary"
                if row < 37
                or col < 37
                or row + 38 > images.shape[1]
                or col + 38 > images.shape[2]
                else "available"
            )
        )
        if (
            context == "available"
            and not valid[row - 37 : row + 38, col - 37 : col + 38].all()
        ):
            context = "invalid_pixels"
        candidates.append(
            {
                "component": index,
                "peak_pixel_xy": [col, row],
                "bbox_px": bbox,
                "component_pixels": int(member.sum()),
                "contrast_db_hh_hv": contrast[:, row, col].tolist(),
                "threshold_margin_db": float(peak_margin[row, col]),
                "triggered_polarizations": [
                    p for b, p in enumerate(polarizations) if margins[b, row, col] >= 0
                ],
                "classifier_window_px": [col - 37, row - 37, 75, 75],
                "classifier_context_status": context,
                "research_retained": True,
                "operational_alert": False,
            }
        )
    return candidates, {
        "roi_pixels": int(valid.size),
        "valid_pixels": int(valid.sum()),
        "evaluated_pixels": int(supported.sum()),
        "unevaluated_pixels": int(valid.size - supported.sum()),
        "coverage_fraction": float(supported.mean()),
        "bright_pixels": int(bright.sum()),
        "components": int(total),
        "candidate_count": len(candidates),
        "scope": "Pixel processing support; not evaluated water area or vessel recall",
    }
