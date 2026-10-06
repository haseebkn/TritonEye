"""Reference-relative shoreline classification and a separate research policy.

`surface` remains a legacy operating-zone field, NOT physical land/water.
Automatic operational alerts stay disabled regardless of eligibility.
"""

import math
from typing import Any, Mapping

POLICY_VERSION = "open_water_v2"


def annotations(
    physical: str,
    distance_m: float | None,
    buffer_m: float = 300.0,
    infrastructure: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Coastal buffer must be finite and nonnegative")
    known = physical in {"water", "land"}
    # Positive infinity means no coastline intersects the padded footprint.
    # None is unknown, not an invitation to promote an unclassified target.
    distance_known = distance_m is not None and not math.isnan(distance_m)
    coastal = bool(
        physical == "water"
        and distance_known
        and distance_m is not None
        and distance_m <= buffer_m
    )
    eligible = bool(
        physical == "water"
        and distance_known
        and distance_m is not None
        and distance_m > buffer_m
        and not infrastructure
    )
    reason = (
        "physical_surface_unknown"
        if not known
        else (
            "physical_land"
            if physical == "land"
            else (
                "shore_distance_unknown"
                if not distance_known
                else (
                    "infrastructure_proximity"
                    if infrastructure
                    else "coastal_buffer" if coastal else "open_water"
                )
            )
        )
    )
    zone = (
        "infrastructure_proximity"
        if infrastructure and known
        else (
            "land"
            if physical == "land"
            else "coastal" if coastal else "water" if eligible else "unknown"
        )
    )
    return {
        "physical_surface": physical if known else "unknown",
        "surface": zone,
        "coastal_zone": coastal,
        "coastal_buffer_m": buffer_m,
        "infrastructure_proximity": bool(infrastructure),
        "shore_distance_unbounded": distance_m == math.inf,
        "distance_to_shore_m": (
            distance_m if distance_m is not None and math.isfinite(distance_m) else None
        ),
        "alert_eligible": eligible,
        "alert_eligibility_reason": reason,
        "coastal_policy_version": POLICY_VERSION,
        "research_retained": True,
        "operational_alert": False,
    }


def eligible(properties: Mapping[str, Any]) -> bool:
    """Recompute eligibility, failing closed on contradictory/new metadata.

    Legacy outputs remain readable, but their operating zone is not converted
    into independently verified physical shoreline truth.
    """
    if "physical_surface" not in properties:
        return properties.get("surface") == "water"
    if properties.get("physical_surface") != "water":
        return False
    if properties.get("surface") != "water":
        return False
    if properties.get("infrastructure_proximity") is True:
        return False
    if properties.get("alert_eligible") is not True:
        return False
    try:
        buffer = float(properties["coastal_buffer_m"])
        distance = (
            math.inf
            if properties.get("shore_distance_unbounded") is True
            else float(properties["distance_to_shore_m"])
        )
        return math.isfinite(buffer) and buffer >= 0 and distance > buffer
    except (KeyError, TypeError, ValueError):
        return False
