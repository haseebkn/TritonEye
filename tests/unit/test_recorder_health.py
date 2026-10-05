"""Heartbeat freshness and AIS observation freshness are separate evidence."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agents.artifacts import write_json
from agents.recorder_health import recorder_health

NOW = datetime(2026, 10, 5, 22, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("heartbeat_age", "observation_age", "connected", "expected"),
    [
        (10, 20, True, "fresh"),
        (181, 20, True, "heartbeat_stale"),
        (10, 601, True, "observations_stale"),
        (10, None, True, "observations_stale"),
        (10, 20, False, "disconnected"),
        (-120, 20, True, "heartbeat_stale"),
    ],
)
def test_freshness_is_not_container_running_state(
    tmp_path: Path,
    heartbeat_age: int,
    observation_age: int | None,
    connected: bool,
    expected: str,
) -> None:
    observation = (
        (NOW - timedelta(seconds=observation_age)).isoformat()
        if observation_age is not None
        else None
    )
    write_json(
        tmp_path / "recorder_status.json",
        {
            "heartbeat_at": (NOW - timedelta(seconds=heartbeat_age)).isoformat(),
            "connected": connected,
            "last_observed_at": observation,
            "last_received_at": observation,
            "gaps": [{"gap_type": "heartbeat_gap", "seconds": 3600}],
        },
    )
    health = recorder_health(tmp_path, now=NOW)
    assert health["status"] == expected
    assert health["healthy"] is (expected == "fresh")
    assert health["gaps"][0]["seconds"] == 3600
    assert health["coverage_complete"] is False


def test_fresh_receipt_of_old_observation_remains_stale(tmp_path: Path) -> None:
    write_json(
        tmp_path / "recorder_status.json",
        {
            "heartbeat_at": NOW.isoformat(),
            "connected": True,
            "last_received_at": NOW.isoformat(),
            "last_observed_at": (NOW - timedelta(hours=1)).isoformat(),
        },
    )
    assert recorder_health(tmp_path, now=NOW)["status"] == "observations_stale"


@pytest.mark.parametrize("content", [None, "bad-json", "[]"])
def test_missing_or_malformed_heartbeat_is_unknown(
    tmp_path: Path, content: str | None
) -> None:
    if content is not None:
        (tmp_path / "recorder_status.json").write_text(content)
    health = recorder_health(tmp_path, now=NOW)
    assert health["status"] == "unknown"
    assert health["healthy"] is False


def test_running_silent_feed_writes_live_heartbeat(
    monkeypatch: Any, tmp_path: Path
) -> None:
    import asyncio
    import json

    import websockets

    from agents import ais_recorder

    async def silent(ws: Any) -> None:
        await ws.recv()
        await ws.wait_closed()

    async def run() -> None:
        async with websockets.serve(silent, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            monkeypatch.setattr(ais_recorder, "STREAM_URL", f"ws://127.0.0.1:{port}")
            monkeypatch.setattr(ais_recorder, "HEARTBEAT_INTERVAL_S", 0.02)
            task = asyncio.create_task(
                ais_recorder.record(
                    "test-secret", [-53, 47, -52, 48], str(tmp_path), 0.3
                )
            )
            for _ in range(20):
                await asyncio.sleep(0.01)
                if recorder_health(tmp_path)["status"] == "observations_stale":
                    break
            health = recorder_health(tmp_path)
            assert health["status"] == "observations_stale"
            assert health["heartbeat_age_s"] < 1
            await task

    asyncio.run(run())
    status = json.loads((tmp_path / "recorder_status.json").read_text())
    assert status["connected"] is False
    assert status["observations"] == 0
    journal = next(tmp_path.glob("coverage_*.jsonl")).read_text()
    assert "heartbeat" in [json.loads(line)["event"] for line in journal.splitlines()]
    assert "test-secret" not in journal


def test_missed_heartbeat_interval_is_persisted_as_gap(
    monkeypatch: Any, tmp_path: Path
) -> None:
    import asyncio
    import json

    import websockets

    from agents import ais_recorder

    async def silent(ws: Any) -> None:
        await ws.recv()
        await ws.wait_closed()

    async def run() -> None:
        async with websockets.serve(silent, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            monkeypatch.setattr(ais_recorder, "STREAM_URL", f"ws://127.0.0.1:{port}")
            # A short maximum permitted gap reproduces missed-heartbeat
            # recording without suspending the host or waiting three minutes.
            monkeypatch.setattr(ais_recorder, "HEARTBEAT_INTERVAL_S", 0.04)
            monkeypatch.setattr(ais_recorder, "HEARTBEAT_MAX_AGE_S", 0.02)
            await ais_recorder.record(
                "test-key", [-53, 47, -52, 48], str(tmp_path), 0.2
            )

    asyncio.run(run())
    status = json.loads((tmp_path / "recorder_status.json").read_text())
    assert status["gaps"]
    assert all(gap["seconds"] > 0.02 for gap in status["gaps"])
    events = [
        json.loads(line)
        for line in next(tmp_path.glob("coverage_*.jsonl")).read_text().splitlines()
    ]
    assert any(event["event"] == "heartbeat_gap" for event in events)
    assert events[-1]["event"] == "session_stop"
