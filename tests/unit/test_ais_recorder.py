import csv
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.ais_recorder import (
    DailyCsvWriter,
    bbox_to_subscription_area,
    load_bbox_from_aoi,
    parse_position_report,
)


def position_report_envelope(**overrides: Any) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "MessageType": "PositionReport",
        "Metadata": {
            "MMSI": 316000123,
            "Latitude": 47.5,
            "Longitude": -52.7,
            "time_utc": "2026-08-18 09:15:30.123456 +0000 UTC",
        },
        "Message": {
            "PositionReport": {
                "UserID": 316000123,
                "Latitude": 47.5,
                "Longitude": -52.7,
                "Sog": 12.3,
                "Cog": 88.5,
            }
        },
    }
    envelope.update(overrides)
    return envelope


def test_parse_position_report_extracts_normalized_row() -> None:
    row = parse_position_report(position_report_envelope())
    assert row == {
        "mmsi": 316000123,
        "lat": 47.5,
        "lon": -52.7,
        "timestamp": "2026-08-18 09:15:30",
        "speed_knots": 12.3,
        "course_deg": 88.5,
    }


def test_parse_position_report_ignores_other_message_types() -> None:
    envelope = position_report_envelope(MessageType="ShipStaticData")
    assert parse_position_report(envelope) is None


def test_parse_position_report_falls_back_to_metadata_position() -> None:
    # Some envelopes carry position only in Metadata, not in the nested
    # PositionReport payload.
    envelope = position_report_envelope()
    del envelope["Message"]["PositionReport"]["Latitude"]
    del envelope["Message"]["PositionReport"]["Longitude"]
    row = parse_position_report(envelope)
    assert row is not None
    assert row["lat"] == 47.5
    assert row["lon"] == -52.7


def test_parse_position_report_missing_identity_returns_none() -> None:
    envelope = position_report_envelope()
    del envelope["Metadata"]["MMSI"]
    del envelope["Message"]["PositionReport"]["UserID"]
    assert parse_position_report(envelope) is None


def test_parse_position_report_missing_timestamp_falls_back_to_now() -> None:
    envelope = position_report_envelope()
    envelope["Metadata"]["time_utc"] = ""
    row = parse_position_report(envelope)
    assert row is not None
    assert row["timestamp"]  # some timestamp was still produced


def test_bbox_to_subscription_area_matches_aisstream_corner_format() -> None:
    # aisstream.io expects [[lat, lon], [lat, lon]] pairs, not the
    # [min_lon, min_lat, max_lon, max_lat] convention used everywhere else in
    # this codebase — an easy place to transpose lat/lon by accident.
    area = bbox_to_subscription_area([-52.6, 47.3, -51.5, 47.8])
    assert area == [[[47.3, -52.6], [47.8, -51.5]]]


def test_daily_csv_writer_creates_header_once(tmp_path: Any) -> None:
    writer = DailyCsvWriter(str(tmp_path))
    row = {
        "mmsi": 1,
        "lat": 1.0,
        "lon": 2.0,
        "timestamp": "2026-08-18 00:00:00",
        "speed_knots": 0.0,
        "course_deg": 0.0,
    }
    writer.write(row)
    writer.write(row)
    writer.close()

    files = os.listdir(tmp_path)
    assert len(files) == 1
    with open(tmp_path / files[0], newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
    assert len(rows) == 3  # header + 2 data rows


# --------------------------------------------------------------- reconnection


class _FlakyServer:
    """
    A local websockets server that accepts a connection and subscription, then
    immediately closes without sending anything -- reproducing the failure
    mode aisstream.io exhibited during its 2026-08-20 outage: handshake
    succeeds, but zero data is ever delivered before the connection dies.
    """

    def __init__(self) -> None:
        self.accept_count = 0

    async def handler(self, ws: Any) -> None:
        self.accept_count += 1
        await ws.recv()  # the subscription message
        await ws.close()


def test_backoff_does_not_reset_without_receiving_data(monkeypatch: Any) -> None:
    """
    Regression test for the 2026-08-20 incident: our client's backoff used to
    reset to RECONNECT_MIN_S on any successful handshake, even one that
    delivered zero data before the connection ended (whether by error or by
    the server closing cleanly). Against a server stuck accepting connections
    but never sending anything, that degenerates into a reconnect storm --
    which is what got this client rate-limited (HTTP 429) by aisstream.io.

    Runs record() against a local server that always closes immediately after
    the handshake, with the backoff constants shrunk so several reconnects fit
    into a fast test, and observes the actual delay passed to asyncio.sleep on
    each retry (rather than the rounded text it logs) to confirm it grows.
    """
    import asyncio

    import websockets

    import agents.ais_recorder as m

    monkeypatch.setattr(m, "RECONNECT_MIN_S", 0.05)
    monkeypatch.setattr(m, "RECONNECT_MAX_S", 0.4)

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def spy_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(delay)

    async def run() -> int:
        server = _FlakyServer()
        async with websockets.serve(server.handler, "127.0.0.1", 0) as ws_server:
            port = list(ws_server.sockets)[0].getsockname()[1]
            monkeypatch.setattr(m, "STREAM_URL", f"ws://127.0.0.1:{port}")
            monkeypatch.setattr(asyncio, "sleep", spy_sleep)
            task = asyncio.create_task(
                m.record("key", [-1, -1, 1, 1], "/tmp/unused", duration_s=None)
            )
            await real_sleep(1.0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return server.accept_count

    accept_count = asyncio.run(run())
    assert accept_count >= 3, "server should have been hit multiple times"

    # Patching asyncio.sleep patches the one global asyncio module, so this
    # also captures unrelated internal sleeps from asyncio/websockets plumbing
    # -- both zero-delay scheduling yields and the library's own much larger
    # keepalive-ping waits. Keep only values in our own backoff's range.
    delays = [
        d for d in delays if m.RECONNECT_MIN_S - 1e-9 <= d <= m.RECONNECT_MAX_S + 1e-9
    ]
    assert len(delays) >= 3, f"expected multiple reconnect delays, got {delays!r}"

    # A reset-on-handshake (or reset-on-clean-close) bug would show the
    # minimum repeated forever; the fix must show growth up to the ceiling.
    assert max(delays) > delays[0], f"backoff never grew: {delays}"
    for prev, cur in zip(delays, delays[1:]):
        assert cur >= prev - 1e-9, f"backoff did not grow monotonically: {delays}"


def test_recording_envelope_contains_every_imaging_aoi() -> None:
    """
    Regression: the recorder was subscribed to `eastern_newfoundland`, whose
    bbox has NO longitude overlap with `grand_banks`. A week of recording
    produced ~15k rows and exactly zero inside the Grand Banks, so every scene
    there was unscorable. The archive looked healthy the whole time, which is
    what makes this worth a test rather than a comment.

    Any imaging AOI added later must also fall inside the envelope.
    """
    import glob

    base = os.path.join(os.path.dirname(__file__), "..", "..")
    aoi_dir = os.path.abspath(os.path.join(base, "configs", "aois"))
    envelope = load_bbox_from_aoi(os.path.join(aoi_dir, "nl_shelf.geojson"))

    imaging = [
        p
        for p in glob.glob(os.path.join(aoi_dir, "*.geojson"))
        if not p.endswith("nl_shelf.geojson")
    ]
    assert imaging, "expected at least one imaging AOI"

    for path in imaging:
        b = load_bbox_from_aoi(path)
        name = os.path.basename(path)
        assert (
            envelope[0] <= b[0] and b[2] <= envelope[2]
        ), f"{name} lon outside envelope"
        assert (
            envelope[1] <= b[1] and b[3] <= envelope[3]
        ), f"{name} lat outside envelope"


def test_recorder_default_aoi_is_the_envelope_not_an_imaging_aoi() -> None:
    """
    The default must be the recording envelope.

    Defaulting to an imaging AOI is the same failure the envelope exists to
    prevent: a narrow subscription that looks healthy while scenes processed
    elsewhere score zero. It reached the shipped default once already.
    """
    from agents.ais_recorder import DEFAULT_RECORDING_AOI

    assert DEFAULT_RECORDING_AOI == "nl_shelf"

    base = os.path.join(os.path.dirname(__file__), "..", "..")
    aoi_dir = os.path.abspath(os.path.join(base, "configs", "aois"))
    envelope = load_bbox_from_aoi(
        os.path.join(aoi_dir, f"{DEFAULT_RECORDING_AOI}.geojson")
    )
    grand_banks = load_bbox_from_aoi(os.path.join(aoi_dir, "grand_banks.geojson"))
    assert envelope[0] <= grand_banks[0] and grand_banks[2] <= envelope[2]
    assert envelope[1] <= grand_banks[1] and grand_banks[3] <= envelope[3]
