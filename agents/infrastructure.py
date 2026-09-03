#!/usr/bin/env python3
"""
TritonEye Fixed Offshore Infrastructure

Known permanent installations in the Newfoundland and Labrador offshore, so a
production platform is not reported as a dark vessel every time it is imaged.

WHY THIS MATTERS MORE THAN IT LOOKS: a dark vessel is defined here as a
detection with no correlating AIS transmission. A gravity base structure is a
large, bright, radar-hard target that appears in every single acquisition over
its field, and it does not move. That makes it a FALSE POSITIVE THAT REPEATS --
not a random one -- and repeatable false alerts erode trust in an alert list
faster than sporadic ones. Four installations, imaged on a 6-day repeat cycle,
generate a steady stream of confident nonsense unless they are known about.

The xView3 ensemble's VESSEL head is trained to separate vessels from exactly
this class of target (see EVALUATION.md 8.2), but that head is not yet decoded
into a calibrated probability, and a published position is more reliable than
an uncalibrated model output for something whose location is a matter of public
record.

POSITIONS ARE PUBLIC RECORD, not estimates. Each carries its source below.

WHAT THIS DELIBERATELY DOES NOT DO: it does not mask the surrounding field. A
production installation is served by supply vessels, standby vessels, and
shuttle tankers, and those are real traffic that an MDA system exists to see.
The default radius covers the structure and its immediate safety zone, not the
field around it -- see EXCLUSION_RADIUS_M.
"""

import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FloatArray = np.ndarray[Any, Any]

SURFACE_INFRASTRUCTURE = "infrastructure"

# Radius around a published position within which a detection is attributed to
# the installation rather than to a vessel.
#
# This is a trade, and it is worth stating which way it errs. The standard
# offshore safety zone is 500 m, and gcp_tps georeferencing carries residual
# error of its own, so anything tighter risks missing the structure it exists to
# catch. Anything much wider starts swallowing the supply and standby vessels
# working the field -- which are real traffic, and exactly what an operator
# needs to see. 1 km covers the structure plus georeferencing slop while leaving
# the field itself visible.
EXCLUSION_RADIUS_M = 1000.0

# Fixed production installations, Jeanne d'Arc Basin, Grand Banks of
# Newfoundland. All four lie within configs/aois/grand_banks.geojson.
#
# `fixed` distinguishes a gravity base structure -- concrete, sitting on the
# seabed, positionally immovable -- from a moored FPSO, which holds station
# within a watch circle and can disconnect and leave. A GBS position is good
# indefinitely; an FPSO position should be treated as nominal, which is why the
# distinction is recorded rather than flattened away.
INSTALLATIONS: List[Dict[str, Any]] = [
    {
        "name": "Hibernia",
        "kind": "gravity base structure",
        "lat": 46.750433,
        "lon": -48.782933,
        "fixed": True,
        "source": "https://en.wikipedia.org/wiki/Hibernia_oil_field",
    },
    {
        "name": "Hebron",
        "kind": "gravity base structure",
        "lat": 46.543889,
        "lon": -48.498056,
        "fixed": True,
        "source": "https://en.wikipedia.org/wiki/Hebron-Ben_Nevis_oil_field",
    },
    {
        "name": "Terra Nova",
        "kind": "FPSO",
        "lat": 46.475000,
        "lon": -48.479440,
        "fixed": False,
        "source": "https://en.wikipedia.org/wiki/Terra_Nova_oil_field",
    },
    {
        "name": "White Rose (SeaRose)",
        "kind": "FPSO",
        "lat": 46.788610,
        "lon": -48.015000,
        "fixed": False,
        "source": "https://en.wikipedia.org/wiki/White_Rose_oil_field",
    },
]


def installations_in_bounds(bounds: Sequence[float]) -> List[Dict[str, Any]]:
    """Returns the installations inside (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return [
        i
        for i in INSTALLATIONS
        if min_lon <= i["lon"] <= max_lon and min_lat <= i["lat"] <= max_lat
    ]


def classify_infrastructure(
    lons: Sequence[float],
    lats: Sequence[float],
    bounds: Sequence[float],
    radius_m: float = EXCLUSION_RADIUS_M,
) -> Tuple[List[bool], List[str]]:
    """
    Flags detections falling within `radius_m` of a known installation.

    Returns (is_infrastructure, attributed_name) per detection, where the name
    is "" for detections attributed to nothing. Naming the installation matters:
    an operator seeing a suppressed detection needs to know WHICH structure
    absorbed it, otherwise the mask is unauditable.

    Distances are metric via the same azimuthal equidistant projection the land
    mask uses, so a footprint spanning a UTM zone boundary is handled correctly.
    """
    from pyproj import Transformer

    from agents.landmask import local_aeqd_crs

    n = len(lons)
    if n != len(lats):
        raise ValueError(f"lons/lats length mismatch: {n} != {len(lats)}")
    if n == 0:
        return [], []

    present = installations_in_bounds(bounds)
    if not present:
        return [False] * n, [""] * n

    crs = local_aeqd_crs(bounds)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Lists, not ndarrays: pyproj takes a scalar fast-path for size-1 arrays
    # that NumPy has deprecated. Same reasoning as agents/landmask.py.
    xs, ys = transformer.transform(list(lons), list(lats))
    px = np.asarray(xs, dtype="float64")
    py = np.asarray(ys, dtype="float64")

    ix, iy = transformer.transform(
        [i["lon"] for i in present], [i["lat"] for i in present]
    )
    ix = np.asarray(ix, dtype="float64")
    iy = np.asarray(iy, dtype="float64")

    # (n detections, m installations)
    d = np.hypot(px[:, None] - ix[None, :], py[:, None] - iy[None, :])
    nearest = np.argmin(d, axis=1)
    hit = d[np.arange(n), nearest] <= radius_m

    names = [present[nearest[k]]["name"] if hit[k] else "" for k in range(n)]
    return [bool(h) for h in hit], names


def summarize(names: Sequence[str]) -> Dict[str, int]:
    """Counts detections attributed to each installation, for the report."""
    counts: Dict[str, int] = {}
    for n in names:
        if n:
            counts[n] = counts.get(n, 0) + 1
    return counts
