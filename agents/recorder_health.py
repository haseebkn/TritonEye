"""Recorder liveness and data freshness; neither establishes complete AIS coverage."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.ais_validation import parse_utc, utc_string

HEARTBEAT_INTERVAL_S = 60.0
HEARTBEAT_MAX_AGE_S = 180.0
OBSERVATION_MAX_AGE_S = 600.0


def recorder_health(
    archive_dir: str | Path,
    *,
    now: datetime | None = None,
    heartbeat_max_age_s: float = HEARTBEAT_MAX_AGE_S,
    observation_max_age_s: float = OBSERVATION_MAX_AGE_S,
) -> dict[str, Any]:
    instant = now or datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "checked_at": utc_string(instant),
        "status": "unknown",
        "healthy": False,
        "coverage_complete": False,
    }
    try:
        status = json.loads(
            (Path(archive_dir) / "recorder_status.json").read_text(encoding="utf-8")
        )
        if not isinstance(status, dict):
            raise ValueError("Recorder status is not an object")
    except (OSError, ValueError):
        result["reason"] = "recorder heartbeat missing or malformed"
        return result
    heartbeat = parse_utc(status.get("heartbeat_at"))
    observation = parse_utc(status.get("last_observed_at"))
    receipt = parse_utc(status.get("last_received_at"))
    age = (instant - heartbeat).total_seconds() if heartbeat else None
    observation_age = (instant - observation).total_seconds() if observation else None
    receipt_age = (instant - receipt).total_seconds() if receipt else None
    result.update(
        heartbeat_age_s=age,
        observation_age_s=observation_age,
        receipt_age_s=receipt_age,
        connected=status.get("connected") is True,
        gaps=status.get("gaps", []),
        session_id=status.get("session_id"),
    )
    if age is None or age < -60 or age > heartbeat_max_age_s:
        result["status"] = "heartbeat_stale"
    elif status.get("connected") is not True:
        result["status"] = "disconnected"
    elif (
        observation_age is None
        or receipt_age is None
        or not -60 <= observation_age <= observation_max_age_s
        or not -60 <= receipt_age <= observation_max_age_s
    ):
        result["status"] = "observations_stale"
    else:
        result.update(status="fresh", healthy=True)
    return result
