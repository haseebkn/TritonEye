"""AIS field checks shared by the live recorder and archive ingestion.

Position timestamps from aisstream metadata are provider timestamps, not proof
of the vessel's GNSS fix time. AIS SOG/COG unavailable codes remain unknown.
See https://www.navcen.uscg.gov/ais-class-a-reports .
"""

import math
from datetime import datetime, timezone
from typing import Any


def parse_utc(value: Any, *, allow_naive: bool = False) -> datetime | None:
    """Parse an actual timestamp; never substitute the local receipt time."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().removesuffix(" UTC").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        if not allow_naive:
            return None
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_string(value: datetime) -> str:
    """Serialize UTC with an explicit offset and preserve subsecond precision."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def valid_mmsi(value: Any) -> int | None:
    number = finite_number(value)
    if number is None or number != int(number):
        return None
    return int(number) if 100000000 <= number <= 999999999 else None


def speed_knots(value: Any) -> float | None:
    number = finite_number(value)
    # 102.2 represents "102.2 or higher", not an exact usable velocity.
    return number if number is not None and 0 <= number < 102.2 else None


def course_deg(value: Any) -> float | None:
    number = finite_number(value)
    return number if number is not None and 0 <= number < 360 else None
