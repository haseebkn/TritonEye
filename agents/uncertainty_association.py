"""Experimental SAR/AIS assignment with explicit, uncalibrated error assumptions.

This is not a replacement for the production geometric matcher. Gaussian error
terms describe a research score, not calibrated identity probabilities.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

from agents.ais_validation import parse_utc
from agents.association import (
    GEOD,
    KNOT_M_S,
    MAX_PLAUSIBLE_SPEED_KNOTS,
    aligned_ais,
)


@dataclass(frozen=True)
class UncertaintyConfig:
    """Provisional research defaults; all distances metres and times seconds."""

    sar_sigma_m: float = 100.0
    ais_sigma_m: float = 30.0
    sar_time_sigma_s: float = 15.0
    ais_time_sigma_s: float = 5.0
    speed_sigma_m_s: float = 1.0
    course_sigma_deg: float = 10.0
    acceleration_sigma_m_s2: float = 0.01
    gate_squared: float = 9.21
    max_distance_m: float = 2000.0
    abstention_cost: float = 20.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Positive finite uncertainty setting required: {name}"
                )


def covariance(
    position: dict[str, Any], config: UncertaintyConfig
) -> np.ndarray[Any, Any]:
    """Independent error terms in local east/north coordinates.

    Clock error acts along track; speed/course and acceleration errors grow with
    the alignment interval. Unknown motion uses a conservative isotropic speed
    bound, not a zero-speed assumption. No measured covariance is claimed.
    """
    config.validate()
    age = float(position["report_age_s"])
    if not math.isfinite(age) or age < 0:
        raise ValueError("Finite nonnegative report age required")
    time_variance = config.sar_time_sigma_s**2 + config.ais_time_sigma_s**2
    base = config.sar_sigma_m**2 + config.ais_sigma_m**2
    acceleration = (0.5 * config.acceleration_sigma_m_s2 * age**2) ** 2
    if not position.get("motion_known"):
        bound = MAX_PLAUSIBLE_SPEED_KNOTS * KNOT_M_S
        return np.eye(2) * (base + acceleration + bound**2 * (time_variance + age**2))
    speed = float(position["speed_m_s"])
    course = float(position["course_deg"])
    if not math.isfinite(speed + course) or not (
        0 <= speed <= MAX_PLAUSIBLE_SPEED_KNOTS * KNOT_M_S and 0 <= course < 360
    ):
        raise ValueError("Invalid aligned motion metadata")
    angle = math.radians(course)
    along = np.array([math.sin(angle), math.cos(angle)])
    across = np.array([math.cos(angle), -math.sin(angle)])
    return (
        np.eye(2) * (base + acceleration)
        + np.outer(along, along)
        * (speed**2 * time_variance + (age * config.speed_sigma_m_s) ** 2)
        + np.outer(across, across)
        * (age * speed * math.radians(config.course_sigma_deg)) ** 2
    )


def uncertainty_matches(
    targets: list[dict[str, Any]],
    records: pd.DataFrame,
    acquisition_time: str,
    config: UncertaintyConfig = UncertaintyConfig(),
) -> dict[str, Any]:
    """One AIS identity per target, optional abstention, per-target time alignment.

    Returns all gated alternatives and flags *all* competitors as ambiguous,
    including targets that lose a contested identity. Scores are not confidence.
    """
    config.validate()
    start = parse_utc(acquisition_time)
    if start is None:
        raise ValueError("Explicit UTC acquisition time required")
    aligned: list[dict[str, dict[str, Any]]] = []
    for target in targets:
        instant = target.get("timestamp", acquisition_time)
        time = parse_utc(instant)
        if time is None or abs((time - start).total_seconds()) > 300:
            raise ValueError(
                "Target timestamp must be explicit UTC within 300s of scene"
            )
        positions, _ = aligned_ais(records, instant)
        aligned.append({str(row["mmsi"]): row for row in positions.to_dict("records")})
    identities = sorted({mmsi for row in aligned for mmsi in row})
    n, m = len(targets), len(identities)
    if n * (m + n) > 20_000_000:
        raise ValueError("Association matrix too large; split the mission into tiles")
    edges: list[dict[str, Any]] = []
    cost = np.full((n, m + n), config.abstention_cost, dtype=float)
    cost[:, :m] = config.abstention_cost * 3
    allowed = np.zeros((n, m), dtype=bool)
    for i, target in enumerate(targets):
        point = Point(target["lon"], target["lat"])
        if not math.isfinite(point.x + point.y) or not (
            -180 <= point.x <= 180 and -90 <= point.y <= 90
        ):
            raise ValueError("Invalid SAR target coordinates")
        for j, mmsi in enumerate(identities):
            position = aligned[i].get(mmsi)
            if position is None:
                continue
            azimuth, _, distance = GEOD.inv(
                position["lon"], position["lat"], point.x, point.y
            )
            if distance > config.max_distance_m:
                continue
            angle = math.radians(azimuth)
            delta = distance * np.array([math.sin(angle), math.cos(angle)])
            cov = covariance(position, config)
            squared = float(delta @ np.linalg.solve(cov, delta))
            if squared > config.gate_squared:
                continue
            # log determinant penalizes diffuse positions: dividing by a large
            # covariance alone would reward very uncertain tracks.
            score = 0.5 * (squared + float(np.linalg.slogdet(cov / 10000)[1]))
            allowed[i, j] = True
            cost[i, j] = score
            edges.append(
                {
                    "target_index": i,
                    "mmsi": mmsi,
                    "distance_m": float(distance),
                    "squared_mahalanobis": squared,
                    "cost": score,
                    "covariance_en_m2": cov.tolist(),
                    "alignment_method": position["alignment_method"],
                    "report_age_s": position["report_age_s"],
                }
            )
    rows, cols = linear_sum_assignment(cost)
    selected = {
        int(row): identities[col]
        for row, col in zip(rows, cols)
        if col < m and allowed[row, col] and cost[row, col] < config.abstention_cost
    }
    alternatives = {
        i: [identities[j] for j in np.flatnonzero(allowed[i])]
        for i in range(n)
        if allowed[i].any()
    }
    contested = allowed.sum(axis=0) > 1
    ambiguous = [
        i
        for i in alternatives
        if allowed[i].sum() > 1 or bool((allowed[i] & contested).any())
    ]
    return {
        "selected": selected,
        "alternatives": alternatives,
        "ambiguous": ambiguous,
        "edges": edges,
        "configuration": asdict(config),
        "probabilities_calibrated": False,
    }
