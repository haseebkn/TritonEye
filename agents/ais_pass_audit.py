"""Pass-specific recording diagnostics; feed activity never establishes coverage."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, shape

from agents.ais_validation import parse_utc
from agents.nl_reception import grid_key


def journals(directory: Path) -> dict[str, Any]:
    sessions = []
    malformed = 0
    for path in sorted(directory.glob("coverage_*.jsonl")):
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                instant = parse_utc(row.get("time")) if isinstance(row, dict) else None
                if instant is None:
                    raise ValueError("Invalid event time")
                events.append({**row, "seconds": instant.timestamp()})
            except (ValueError, TypeError):
                malformed += 1
        events.sort(key=lambda e: e["seconds"])
        intervals = []
        beats = [e for e in events if e.get("event") == "heartbeat"]
        breaks = [
            e["seconds"]
            for e in events
            if e.get("event") in {"disconnected", "session_stop", "transport_open"}
        ]
        for left, right in zip(beats, beats[1:]):
            a, b = left["seconds"], right["seconds"]
            if (
                left.get("connected") is True
                and right.get("connected") is True
                and 0 < b - a <= 180
                and bisect.bisect_right(breaks, a) == bisect.bisect_right(breaks, b)
            ):
                intervals.append((a, b))
        changes = []
        state: dict[str, Any] = {}
        for event in events:
            kind = event.get("event")
            if kind in {"transport_open", "disconnected", "session_stop"}:
                state = {}
            if kind == "transport_open" and isinstance(
                event.get("permessage_deflate"), bool
            ):
                state["compression"] = event["permessage_deflate"]
            if kind == "subscription_confirmation" and event.get("accepted") is True:
                state["ack"] = True
                if isinstance(event.get("compression_enabled"), bool):
                    state["compression"] = event["compression_enabled"]
            if kind in {
                "transport_open",
                "disconnected",
                "session_stop",
                "subscription_confirmation",
            }:
                changes.append({"seconds": event["seconds"], **state})
        sessions.append({"intervals": intervals, "changes": changes})
    return {"sessions": sessions, "malformed_events": malformed}


def transport_window(data: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    if end <= start:
        raise ValueError("Positive pass context interval required")
    intervals = []
    acknowledgements = set()
    compression = set()
    for session in data["sessions"]:
        active = [
            (max(a, start), min(b, end))
            for a, b in session["intervals"]
            if a < end and b > start
        ]
        intervals.extend(active)
        # Report state attached to actual covered connection intervals. A new
        # transport or disconnect invalidates a prior confirmation in that file.
        changes = session["changes"]
        times = [c["seconds"] for c in changes]
        for a, b in active:
            left, right = bisect.bisect_right(times, a), bisect.bisect_left(times, b)
            states = [changes[left - 1] if left else {}] + changes[left:right]
            for state in states:
                acknowledgements.add("confirmed" if state.get("ack") else "unknown")
                compression.add(
                    str(state["compression"]).lower()
                    if "compression" in state
                    else "unknown"
                )
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    covered = sum(b - a for a, b in merged)
    return {
        "window_seconds": end - start,
        "connected_heartbeat_seconds": covered,
        "connected_heartbeat_fraction": covered / (end - start),
        "uncovered_or_unknown_seconds": end - start - covered,
        "subscription_acknowledgement_states": sorted(acknowledgements or {"unknown"}),
        "compression_states": sorted(compression or {"unknown"}),
        "packet_loss_measured": False,
        "area_wide_reception_proven": False,
    }


def collection(
    scenes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    journal: dict[str, Any],
    half_window_s: float,
    grid_step: float,
) -> list[dict[str, Any]]:
    times = [r["time"] for r in rows]
    indexed = {tuple(c["grid"]): c for c in cells}
    results = []
    for scene in scenes:
        product = scene["product"]
        start, end = parse_utc(product["ContentDate"]["Start"]), parse_utc(
            product["ContentDate"]["End"]
        )
        assert start is not None and end is not None
        lower, upper = (
            start.timestamp() - half_window_s,
            end.timestamp() + half_window_s,
        )
        swath = shape(product["GeoFootprint"])
        intersecting = [c for c in cells if swath.intersects(shape(c["geometry"]))]
        if not intersecting:
            continue
        observed = []
        counts: dict[str, int] = {}
        for row in rows[
            bisect.bisect_left(times, lower) : bisect.bisect_right(times, upper)
        ]:
            cell = indexed.get(grid_key(row["lon"], row["lat"], grid_step))
            if cell and swath.covers(Point(row["lon"], row["lat"])):
                observed.append(row)
                counts[cell["id"]] = counts.get(cell["id"], 0) + 1
        latencies = []
        missing = 0
        for row in observed:
            receipt = parse_utc(row["values"].get("received_at"))
            if receipt is None:
                missing += 1
            else:
                latency = receipt.timestamp() - row["time"]
                if math.isfinite(latency):
                    latencies.append(latency)
        results.append(
            {
                "product_id": product["Id"],
                "name": product["Name"],
                "native_acquisition_group": scene["native_acquisition_group"],
                "midpoint_utc": scene["reference_midpoint_utc"],
                "eligible_cells_intersecting_swath": len(intersecting),
                "cells_with_reports": counts,
                "reports_in_context": len(observed),
                "identities_in_context": len({r["mmsi"] for r in observed}),
                "reports_during_acquisition": sum(
                    start.timestamp() <= r["time"] <= end.timestamp() for r in observed
                ),
                "receipt_timestamp_missing": missing,
                "receipt_minus_observation_seconds": {
                    "count": len(latencies),
                    "negative": sum(v < 0 for v in latencies),
                    "median": float(np.median(latencies)) if latencies else None,
                    "p95": float(np.percentile(latencies, 95)) if latencies else None,
                    "maximum": max(latencies) if latencies else None,
                },
                "transport": transport_window(journal, lower, upper),
                "aligned_supports_in_search_cells": sum(
                    grid_key(s["lon"], s["lat"], grid_step) in indexed
                    for s in scene["support"]
                ),
                "coverage_complete": False,
            }
        )
    return results
