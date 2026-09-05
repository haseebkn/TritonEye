"""Auditable, bounded AIS alignment and one-to-one proximity association.

Uncertainty values are engineering allowances, not calibrated probabilities.
AIS reception is incomplete; absence of an association proves no vessel intent.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyproj import Geod
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

GEOD = Geod(ellps="WGS84")
MAX_REPORT_AGE_S = 300.0
MAX_PROPAGATION_S = 120.0
MAX_UNPROPAGATED_AGE_S = 60.0
MAX_PLAUSIBLE_SPEED_KNOTS = 60.0
BASE_UNCERTAINTY_M = 100.0
UNCERTAINTY_GROWTH_M_S = 2.0
KNOT_M_S = 1852.0 / 3600.0


def real_ais_source(coverage: str) -> bool:
    """True means real observations may exist, never verified scene coverage."""
    return coverage.lower() not in {"", "none", "mock", "unknown", "n/a", "synthetic"}


def aligned_ais(
    records: pd.DataFrame, acquisition_time: str
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Return at most one time-aligned WGS84 position per valid MMSI.

    Naive legacy timestamps are explicitly interpreted as UTC. Interpolation
    requires bracketing reports within five minutes and plausible displacement.
    Otherwise velocity extrapolation is limited to two minutes; reports without
    valid kinematics may be used only within one minute, with added uncertainty.
    """
    empty = gpd.GeoDataFrame(
        columns=["mmsi", "lon", "lat", "uncertainty_m", "alignment_method"],
        geometry=[],
        crs="EPSG:4326",
    )
    stats: dict[str, Any] = {
        "input_records": len(records),
        "aligned_vessels": 0,
        "timestamp_policy": "timezone-aware UTC; naive legacy values assumed UTC",
        "uncertainty_model": "heuristic allowance; not a calibrated probability",
    }
    acq = pd.to_datetime(acquisition_time, utc=True, errors="coerce")
    if not acquisition_time or pd.isna(acq):
        stats["reason"] = "missing or invalid acquisition timestamp"
        return empty, stats
    required = {"mmsi", "lon", "lat", "timestamp"}
    if not required.issubset(records.columns):
        stats["reason"] = "AIS schema lacks MMSI, position, or timestamp"
        return empty, stats
    data = records.copy()
    for key in ("mmsi", "lon", "lat", "speed_knots", "course_deg", "length"):
        data[key] = pd.to_numeric(data.get(key, np.nan), errors="coerce")
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    valid = (
        data["mmsi"].between(100_000_000, 999_999_999)
        & (data["mmsi"] % 1 == 0)
        & data["lon"].between(-180, 180)
        & data["lat"].between(-90, 90)
        & data["timestamp"].notna()
    )
    stats["invalid_records"] = int((~valid).sum())
    data = data.loc[valid].copy()
    data["dt_s"] = (data["timestamp"] - acq).dt.total_seconds()
    fresh = data["dt_s"].abs() <= MAX_REPORT_AGE_S
    stats["stale_records"] = int((~fresh).sum())
    data = data.loc[fresh].copy()
    data["mmsi"] = data["mmsi"].astype("int64")
    outputs: list[dict[str, Any]] = []
    rejected_tracks = 0
    for mmsi, group in data.groupby("mmsi", sort=True):
        group = group.sort_values("timestamp", kind="stable")
        # Conflicting reports at the same instant cannot identify a reliable fix.
        conflicts = group.groupby("timestamp")[["lon", "lat"]].nunique()
        if (conflicts > 1).any().any():
            rejected_tracks += 1
            continue
        group = group.drop_duplicates("timestamp")
        nearest = group.loc[group["dt_s"].abs().idxmin()]
        lon, lat = float(nearest.lon), float(nearest.lat)
        age = abs(float(nearest.dt_s))
        method = "observed" if age == 0 else "nearest"
        before = group[group["dt_s"] < 0]
        after = group[group["dt_s"] > 0]
        if age and not before.empty and not after.empty:
            left, right = before.iloc[-1], after.iloc[0]
            azimuth, _, distance = GEOD.inv(left.lon, left.lat, right.lon, right.lat)
            interval = float(right.dt_s - left.dt_s)
            if distance / interval > MAX_PLAUSIBLE_SPEED_KNOTS * KNOT_M_S:
                rejected_tracks += 1
                continue
            fraction = -float(left.dt_s) / interval
            lon, lat, _ = GEOD.fwd(left.lon, left.lat, azimuth, distance * fraction)
            method = "interpolated"
            age = max(abs(float(left.dt_s)), abs(float(right.dt_s)))
        elif age:
            speed, course = float(nearest.speed_knots), float(nearest.course_deg)
            kinematics = (
                np.isfinite(speed)
                and 0 <= speed <= MAX_PLAUSIBLE_SPEED_KNOTS
                and np.isfinite(course)
                and 0 <= course < 360
            )
            if age <= MAX_PROPAGATION_S and kinematics:
                lon, lat, _ = GEOD.fwd(
                    lon, lat, course, -float(nearest.dt_s) * speed * KNOT_M_S
                )
                method = "velocity_propagated"
            elif age > MAX_UNPROPAGATED_AGE_S:
                rejected_tracks += 1
                continue
        outputs.append(
            {
                "mmsi": str(mmsi),
                "lon": lon,
                "lat": lat,
                "length": float(nearest.length),
                "report_age_s": age,
                "alignment_method": method,
                "uncertainty_m": BASE_UNCERTAINTY_M + age * UNCERTAINTY_GROWTH_M_S,
                "geometry": Point(lon, lat),
            }
        )
    stats["rejected_tracks"] = rejected_tracks
    stats["aligned_vessels"] = len(outputs)
    if not outputs:
        stats["reason"] = "no valid, sufficiently recent AIS positions"
        return empty, stats
    return gpd.GeoDataFrame(outputs, geometry="geometry", crs="EPSG:4326"), stats


def one_to_one_matches(
    targets: list[Point], positions: list[Point], radius_m: float | list[float]
) -> tuple[dict[int, tuple[int, float]], dict[int, list[int]], set[int]]:
    """Maximum-cardinality, minimum-distance geodesic bipartite assignment.

    Return selected pairs, all gated alternatives per target, and targets with
    competing alternatives. Unmatched competitors remain ambiguous, preventing
    duplicate detections from becoming spurious unmatched-vessel claims.
    """
    if not targets or not positions:
        return {}, {}, set()
    n, m = len(targets), len(positions)
    if n * (m + n) > 20_000_000:
        raise ValueError("Association matrix too large; split the mission into tiles")
    radii = np.broadcast_to(np.asarray(radius_m, dtype=float), (m,))
    if not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("Association radii must be positive finite metres")
    distances: NDArray[np.float64] = np.empty((n, m), dtype=float)
    lons = [p.x for p in positions]
    lats = [p.y for p in positions]
    for i, target in enumerate(targets):
        _, _, values = GEOD.inv([target.x] * m, [target.y] * m, lons, lats)
        distances[i] = values
    allowed = distances <= radii[None, :]
    # A missed pair costs more than all feasible distances combined; therefore
    # assignment maximizes count first, then minimizes total distance.
    penalty = (n + m + 1) * (float(radii.max()) + 1)
    cost: NDArray[np.float64] = np.full((n, m + n), penalty, dtype=float)
    cost[:, :m] = np.where(allowed, distances, penalty * 3)
    rows, cols = linear_sum_assignment(cost)
    selected = {
        int(row): (int(col), float(distances[row, col]))
        for row, col in zip(rows, cols)
        if col < m and allowed[row, col]
    }
    alternatives = {
        i: [int(j) for j in np.flatnonzero(allowed[i])]
        for i in range(n)
        if allowed[i].any()
    }
    contested = allowed.sum(axis=0) > 1
    ambiguous = {
        i
        for i, choices in alternatives.items()
        if len(choices) > 1 or any(contested[j] for j in choices)
    }
    return selected, alternatives, ambiguous
