#!/usr/bin/env python3
"""
TritonEye AIS Stream Recorder

Subscribes to aisstream.io's live WebSocket feed and appends normalized
position reports to daily CSV files under data/raw/ais_stream/.

aisstream.io has no historical archive — it only streams AIS as vessels
transmit it. This recorder exists to build that archive ourselves, so that
future Sentinel-1 acquisitions over Newfoundland and Labrador can be correlated
against real AIS. This project currently integrates this source only. Recording
does not establish complete receiver coverage or independent ground truth.

It cannot retroactively supply AIS for acquisitions already on disk; it only
covers time recorded from the moment it is started onward. Run it
continuously, well ahead of the Sentinel-1 acquisitions you want to validate.

SUBSCRIBE TO AN ENVELOPE, NOT TO ONE IMAGING AOI. The bounding box here decides
what is recorded, and an acquisition outside it is unscorable no matter how long
the recorder ran. This is a silent failure: the archive fills up, the row counts
look healthy, and every scene in the area you actually care about still scores
zero. It happened here -- a week was recorded against `eastern_newfoundland`
(lon -53.5..-52.0), which has NO longitude overlap with `grand_banks`
(-51.0..-47.5), so the Grand Banks had exactly zero rows despite ~15k recorded.

Use `configs/aois/nl_shelf.geojson`, which is sized to contain every imaging AOI
in the project rather than to match any one of them:

    python agents/ais_recorder.py --aoi nl_shelf

Output schema is the one ingest_agent.py filters against directly:
mmsi, lat, lon, timestamp, speed_knots, course_deg.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from types import FrameType
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ais_validation import (
    course_deg,
    finite_number,
    parse_utc,
    speed_knots,
    utc_string,
    valid_mmsi,
)
from agents.region import contains_points, load_region, validate_aoi

try:
    import websockets
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )

STREAM_URL = "wss://stream.aisstream.io/v0/stream"

# The recording envelope: sized to contain every imaging AOI, not to match any
# one of them. See the module docstring for why this is not an imaging AOI.
DEFAULT_RECORDING_AOI = "nl_shelf"

# The service closes the connection if a subscription isn't sent within 3
# seconds of connecting, and rate-limits subscription updates to 1/second.
# We only ever send one, immediately after connecting, so neither limit binds.
SUBSCRIBE_TIMEOUT_S = 3.0

# Reconnect backoff. aisstream.io is explicitly beta with "no SLA," so drops
# are expected; retry with jitter-free exponential backoff up to a ceiling.
RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 300.0

CORE_CSV_HEADER = ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
CSV_HEADER = CORE_CSV_HEADER + [
    "source",
    "message_type",
    "timestamp_basis",
    "received_at",
    "position_accuracy",
    "ais_second",
]
POSITION_TYPES = (
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
)


def load_bbox_from_aoi(aoi_path: str) -> List[float]:
    """Returns [min_lon, min_lat, max_lon, max_lat] for a GeoJSON AOI polygon."""
    with open(aoi_path, "r", encoding="utf-8") as f:
        geom = json.load(f)["features"][0]["geometry"]
    from shapely.geometry import shape

    return list(shape(geom).bounds)


def bbox_to_subscription_area(bbox: Sequence[float]) -> List[List[List[float]]]:
    """
    Converts [min_lon, min_lat, max_lon, max_lat] to aisstream.io's
    BoundingBoxes format: a list of [[lat, lon], [lat, lon]] corner pairs.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return [[[min_lat, min_lon], [max_lat, max_lon]]]


def parse_position_report(
    envelope: Dict[str, Any], received_at: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """
    Extracts a normalized AIS row from one aisstream.io message envelope.

    Returns None for envelopes that aren't a usable PositionReport — a
    different message type, or one missing the fields we need — rather than
    raising, since a live feed will send message types we don't subscribe to
    filter out entirely and shouldn't crash the recorder over.
    """
    message_type = envelope.get("MessageType")
    if message_type not in POSITION_TYPES:
        return None

    message = envelope.get("Message", {})
    if not isinstance(message, dict):
        return None
    report = message.get(message_type, {})
    # The provider's documented spelling is MetaData. Metadata is accepted
    # only for compatibility with older saved envelopes.
    metadata = envelope.get("MetaData", envelope.get("Metadata", {}))
    if not isinstance(report, dict) or not isinstance(metadata, dict):
        return None
    if report.get("Valid") is False:
        return None

    mmsi = valid_mmsi(report.get("UserID", metadata.get("MMSI")))
    lat = finite_number(
        report.get("Latitude", metadata.get("latitude", metadata.get("Latitude")))
    )
    lon = finite_number(
        report.get("Longitude", metadata.get("longitude", metadata.get("Longitude")))
    )
    if mmsi is None or lat is None or lon is None:
        return None
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None

    # aisstream.io reports timestamps as "YYYY-MM-DD HH:MM:SS.ffffff +0000 UTC"
    # in MetaData.time_utc. Missing/invalid timestamps must not turn stale fixes
    # into apparent observations at the satellite acquisition time.
    time_utc = metadata.get("time_utc", "")
    timestamp = _parse_aisstream_time(time_utc)
    if timestamp is None:
        return None
    receipt = received_at or datetime.now(timezone.utc)
    if (timestamp - receipt).total_seconds() > 60:
        return None
    accuracy = report.get("PositionAccuracy")
    second = finite_number(report.get("Timestamp"))

    return {
        "mmsi": mmsi,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "timestamp": utc_string(timestamp),
        "speed_knots": speed_knots(report.get("Sog")),
        "course_deg": course_deg(report.get("Cog")),
        "source": "aisstream.io",
        "message_type": message_type,
        "timestamp_basis": "provider_metadata_utc",
        "received_at": utc_string(receipt),
        "position_accuracy": accuracy if isinstance(accuracy, bool) else None,
        "ais_second": (
            int(second)
            if second is not None and second.is_integer() and 0 <= second <= 63
            else None
        ),
    }


def _parse_aisstream_time(time_utc: str) -> Optional[datetime]:
    return parse_utc(time_utc)


class DailyCsvWriter:
    """
    Appends rows to a CSV file named for the message UTC date, opening a new
    file automatically when the date rolls over. Each row is flushed to disk
    immediately — this is a recorder for a live feed with no replay, so a row
    lost to a crash before the next flush is a row lost for good.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._date = ""
        self._file: Any = None
        self._writer: Any = None

    def write(self, row: Dict[str, Any]) -> None:
        timestamp = parse_utc(row.get("timestamp"), allow_naive=True)
        if timestamp is None:
            raise ValueError("AIS archive rows require an actual UTC timestamp")
        today = timestamp.strftime("%Y-%m-%d")
        if today != self._date:
            self._rotate(today)
        self._writer.writerow(row)
        self._file.flush()

    def _rotate(self, date_str: str) -> None:
        self.close()
        path = os.path.join(self.output_dir, f"ais_stream_{date_str}.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as existing:
                header = next(csv.reader(existing), [])
            if header != CSV_HEADER:
                # Never append an expanded schema beneath a legacy header or
                # rewrite old raw observations to fabricate provenance.
                path = os.path.join(self.output_dir, f"ais_stream_{date_str}_v2.csv")
        is_new = not os.path.exists(path)
        if not is_new:
            with open(path, newline="", encoding="utf-8") as existing:
                if next(csv.reader(existing), []) != CSV_HEADER:
                    raise ValueError(f"Unexpected AIS archive schema: {path}")
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_HEADER)
        if is_new:
            self._writer.writeheader()
            self._file.flush()
        self._date = date_str
        print(f"Recording to {path}", file=sys.stderr)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


async def record(
    api_key: str,
    bbox: Sequence[float],
    output_dir: str,
    duration_s: Optional[float] = None,
) -> None:
    """
    Connects to aisstream.io, subscribes to PositionReports in bbox, and
    writes every report to a daily CSV until stopped.

    Runs until cancelled (SIGINT/SIGTERM) or, if duration_s is given, until
    that many seconds have elapsed — useful for scheduled/bounded recording
    windows rather than a permanently running process.
    """
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("--duration must be a positive finite number of seconds")
    if len(bbox) != 4 or not all(finite_number(v) is not None for v in bbox):
        raise ValueError("Subscription bbox requires four finite coordinates")
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("Subscription bbox coordinates are invalid")
    region = load_region()
    west, south, east, north = region.bounds
    if not (west <= bbox[0] < bbox[2] <= east and south <= bbox[1] < bbox[3] <= north):
        raise ValueError("Subscription must stay within the NL recording envelope")
    writer = DailyCsvWriter(output_dir)
    subscription = {
        "APIKey": api_key,
        "BoundingBoxes": bbox_to_subscription_area(bbox),
        "FilterMessageTypes": list(POSITION_TYPES),
    }
    row_count = 0
    journal_name = (
        f"coverage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}.jsonl"
    )
    journal = open(os.path.join(output_dir, journal_name), "a", encoding="utf-8")

    def event(kind: str, **details: Any) -> None:
        journal.write(
            json.dumps(
                {
                    "event": kind,
                    "time": utc_string(datetime.now(timezone.utc)),
                    **details,
                }
            )
            + "\n"
        )
        journal.flush()

    event(
        "session_start",
        source="aisstream.io",
        subscription_bbox=list(bbox),
        message_types=list(POSITION_TYPES),
        coverage_known=False,
        limitation="Subscription and connection do not establish receiver coverage",
    )

    async def consume() -> None:
        nonlocal row_count
        backoff = RECONNECT_MIN_S
        while True:
            # Backoff resets only once data actually arrives on a connection,
            # not on a bare handshake success. During an upstream outage the
            # server can accept a connection and then go silent, or close it
            # again immediately, with zero messages delivered; if backoff
            # reset on handshake alone, that pattern degenerates into a
            # reconnect storm, which is what got this client rate-limited
            # (HTTP 429) by aisstream.io on 2026-08-20.
            received_any = False
            disconnect_reason: Optional[str] = None

            try:
                async with websockets.connect(
                    STREAM_URL, open_timeout=10, close_timeout=0.25
                ) as ws:
                    await asyncio.wait_for(
                        ws.send(json.dumps(subscription)), SUBSCRIBE_TIMEOUT_S
                    )
                    print(
                        f"Subscribed: bbox={list(bbox)}",
                        file=sys.stderr,
                    )
                    event("subscribed")

                    async for raw in ws:
                        try:
                            envelope = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(envelope, dict):
                            continue
                        row = parse_position_report(envelope)
                        if row is not None and (
                            bbox[0] <= row["lon"] <= bbox[2]
                            and bbox[1] <= row["lat"] <= bbox[3]
                            and contains_points([row["lon"]], [row["lat"]], region)[0]
                        ):
                            if not received_any:
                                received_any = True
                                event("first_observation", timestamp=row["timestamp"])
                            writer.write(row)
                            row_count += 1
                            if row_count % 500 == 0:
                                print(
                                    f"{row_count} position reports recorded",
                                    file=sys.stderr,
                                )
                # `async for` also ends without raising on a clean server-side
                # close (e.g. close code 1000). Any way the connection ends —
                # error or clean close — is treated the same below: never
                # reconnect with zero delay, and only count it towards the
                # backoff ceiling if it never delivered any data.
                disconnect_reason = (
                    "server closed the connection cleanly"
                    if received_any
                    else "server closed the connection without sending any data"
                )
            except asyncio.CancelledError:
                event("disconnected", reason="duration_or_cancelled")
                raise
            except Exception as e:
                # Exceptions must never echo the API key or request contents.
                disconnect_reason = type(e).__name__

            if received_any:
                backoff = RECONNECT_MIN_S
            event("disconnected", reason=disconnect_reason, retry_delay_s=backoff)
            print(
                f"Connection lost ({disconnect_reason}); retrying in {backoff:.0f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)

    try:
        if duration_s is None:
            await consume()
        else:
            try:
                # Covers handshake, a silent recv(), and retry sleeps alike.
                await asyncio.wait_for(consume(), timeout=duration_s)
            except asyncio.TimeoutError:
                pass
    finally:
        writer.close()
        event("session_stop", observations=row_count)
        journal.close()
        print(
            f"Stopped. {row_count} position reports recorded this run.", file=sys.stderr
        )


def parse_arguments() -> argparse.Namespace:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="TritonEye AIS Stream Recorder")
    parser.add_argument(
        "--aoi",
        type=str,
        default=None,
        help="AOI name under configs/aois/ (without .geojson) to derive the "
        "bounding box from. Defaults to nl_shelf, the envelope containing "
        "every imaging AOI -- subscribing to a single narrow imaging AOI "
        "silently makes scenes outside it unscorable.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(base_dir, "data", "raw", "ais_stream"),
        help="Directory to write daily ais_stream_YYYY-MM-DD.csv files to.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds. Omit to run until interrupted.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = parse_arguments()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    api_key = os.getenv("AISSTREAM_API_KEY", "")
    if not api_key:
        print(
            "AISSTREAM_API_KEY is not set. Generate a key at aisstream.io "
            "(GitHub sign-in, free) and add it to .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Deliberately NOT AOI_NAME: that variable selects which area to IMAGE, and
    # the recorder must cover every such area at once. Honouring it here is how
    # the recorder ends up subscribed to one narrow box while scenes are
    # processed somewhere else entirely.
    aoi_name = args.aoi or DEFAULT_RECORDING_AOI
    if not aoi_name.endswith(".geojson"):
        aoi_name += ".geojson"
    aoi_path = os.path.join(base_dir, "configs", "aois", aoi_name)
    if os.path.basename(aoi_name) != aoi_name:
        raise ValueError("AOI must be a name under configs/aois")
    if aoi_name != f"{DEFAULT_RECORDING_AOI}.geojson":
        with open(aoi_path, encoding="utf-8") as aoi_file:
            validate_aoi(json.load(aoi_file)["features"][0]["geometry"])
    bbox = load_bbox_from_aoi(aoi_path)

    print(f"AOI: {aoi_name}  bbox={bbox}", file=sys.stderr)

    def handle_signal(signum: int, frame: Optional[FrameType]) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(record(api_key, bbox, args.output_dir, args.duration))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
