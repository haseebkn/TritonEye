#!/usr/bin/env python3
"""
TritonEye AIS Stream Recorder

Subscribes to aisstream.io's live WebSocket feed and appends normalized
position reports to daily CSV files under data/raw/ais_stream/.

aisstream.io has no historical archive — it only streams AIS as vessels
transmit it. This recorder exists to build that archive ourselves, so that
future Sentinel-1 acquisitions in areas MarineCadastre does not cover (it is
US Coast Guard data, so anything outside US waters — including Newfoundland —
has zero coverage there) can still be correlated against real AIS.

It cannot retroactively supply AIS for acquisitions already on disk; it only
covers time recorded from the moment it is started onward. Run it
continuously, well ahead of the Sentinel-1 acquisitions you want to validate.

Output schema matches the one ingest_agent.py's MarineCadastre filtering
already produces, so ingest_agent can read either source with the same
filtering code: mmsi, lat, lon, timestamp, speed_knots, course_deg.
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
from datetime import datetime, timezone
from types import FrameType
from typing import Any, Dict, List, Optional, Sequence

try:
    import websockets
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )

STREAM_URL = "wss://stream.aisstream.io/v0/stream"

# The service closes the connection if a subscription isn't sent within 3
# seconds of connecting, and rate-limits subscription updates to 1/second.
# We only ever send one, immediately after connecting, so neither limit binds.
SUBSCRIBE_TIMEOUT_S = 3.0

# Reconnect backoff. aisstream.io is explicitly beta with "no SLA," so drops
# are expected; retry with jitter-free exponential backoff up to a ceiling.
RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 300.0

CSV_HEADER = ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]


def load_bbox_from_aoi(aoi_path: str) -> List[float]:
    """Returns [min_lon, min_lat, max_lon, max_lat] for a GeoJSON AOI polygon."""
    with open(aoi_path, "r", encoding="utf-8") as f:
        geom = json.load(f)["features"][0]["geometry"]
    coords = geom["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def bbox_to_subscription_area(bbox: Sequence[float]) -> List[List[List[float]]]:
    """
    Converts [min_lon, min_lat, max_lon, max_lat] to aisstream.io's
    BoundingBoxes format: a list of [[lat, lon], [lat, lon]] corner pairs.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return [[[min_lat, min_lon], [max_lat, max_lon]]]


def parse_position_report(envelope: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts a normalized AIS row from one aisstream.io message envelope.

    Returns None for envelopes that aren't a usable PositionReport — a
    different message type, or one missing the fields we need — rather than
    raising, since a live feed will send message types we don't subscribe to
    filter out entirely and shouldn't crash the recorder over.
    """
    if envelope.get("MessageType") != "PositionReport":
        return None

    report = envelope.get("Message", {}).get("PositionReport", {})
    metadata = envelope.get("Metadata", {})

    mmsi = report.get("UserID") or metadata.get("MMSI")
    lat = report.get("Latitude", metadata.get("Latitude"))
    lon = report.get("Longitude", metadata.get("Longitude"))
    if mmsi is None or lat is None or lon is None:
        return None

    # aisstream.io reports timestamps as "YYYY-MM-DD HH:MM:SS.ffffff +0000 UTC"
    # in Metadata.time_utc. Fall back to receipt time if that's ever absent,
    # since a missing AIS timestamp shouldn't drop an otherwise-usable fix.
    time_utc = metadata.get("time_utc", "")
    timestamp = _parse_aisstream_time(time_utc) or datetime.now(timezone.utc)

    return {
        "mmsi": int(mmsi),
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "speed_knots": round(float(report.get("Sog", 0.0)), 1),
        "course_deg": round(float(report.get("Cog", 0.0)), 1),
    }


def _parse_aisstream_time(time_utc: str) -> Optional[datetime]:
    if not time_utc:
        return None
    # Strip the trailing " +0000 UTC" suffix; the leading portion is a
    # standard "YYYY-MM-DD HH:MM:SS.ffffff" timestamp, already UTC.
    head = time_utc.split(" +")[0].split(" UTC")[0].strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(head, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class DailyCsvWriter:
    """
    Appends rows to a CSV file named for the current UTC date, opening a new
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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._date:
            self._rotate(today)
        self._writer.writerow(row)
        self._file.flush()

    def _rotate(self, date_str: str) -> None:
        self.close()
        path = os.path.join(self.output_dir, f"ais_stream_{date_str}.csv")
        is_new = not os.path.exists(path)
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
    writer = DailyCsvWriter(output_dir)
    subscription = {
        "APIKey": api_key,
        "BoundingBoxes": bbox_to_subscription_area(bbox),
        "FilterMessageTypes": ["PositionReport"],
    }

    loop = asyncio.get_event_loop()
    deadline = loop.time() + duration_s if duration_s else None
    backoff = RECONNECT_MIN_S
    row_count = 0

    try:
        while deadline is None or loop.time() < deadline:
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
                async with websockets.connect(STREAM_URL, open_timeout=10) as ws:
                    await asyncio.wait_for(
                        ws.send(json.dumps(subscription)), SUBSCRIBE_TIMEOUT_S
                    )
                    print(
                        f"Subscribed: bbox={list(bbox)}",
                        file=sys.stderr,
                    )

                    async for raw in ws:
                        if deadline is not None and loop.time() >= deadline:
                            break
                        if not received_any:
                            received_any = True
                            backoff = RECONNECT_MIN_S
                        try:
                            envelope = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        row = parse_position_report(envelope)
                        if row is not None:
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
                raise
            except Exception as e:
                disconnect_reason = str(e)

            if received_any:
                backoff = RECONNECT_MIN_S
            print(
                f"Connection lost ({disconnect_reason}); retrying in {backoff:.0f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)
    finally:
        writer.close()
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
        "bounding box from. Defaults to AOI_NAME env var, then "
        "st_johns_offshore.",
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

    aoi_name = args.aoi or os.getenv("AOI_NAME", "st_johns_offshore")
    if not aoi_name.endswith(".geojson"):
        aoi_name += ".geojson"
    aoi_path = os.path.join(base_dir, "configs", "aois", aoi_name)
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
