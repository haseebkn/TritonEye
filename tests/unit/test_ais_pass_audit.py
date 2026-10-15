"""Verify pass diagnostics, neighbour scope and real WebSocket telemetry."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import websockets
from shapely.geometry import box, mapping

from agents.ais_pass_audit import collection, journals, transport_window
from agents.artifacts import REPO_ROOT
from agents.cd_training_reservations import rules
from agents.neighbour_vessel_batch import expand, verify
from agents.reception_vessel_batch import decide
from agents.source_reservation import checked
from tests.unit.test_reception_vessel_batch import FROZEN, scene


def event(second: float, kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "time": datetime.fromtimestamp(second, timezone.utc).isoformat(),
        "event": kind,
        **fields,
    }


def journal(
    tmp_path: Path, events: list[dict[str, Any]], second_copy: bool = False
) -> dict[str, Any]:
    payload = "\n".join(json.dumps(e) for e in events) + "\n"
    (tmp_path / "coverage_one.jsonl").write_text(payload)
    if second_copy:
        (tmp_path / "coverage_two.jsonl").write_text(payload)
    return journals(tmp_path)


def test_legacy_subscribed_is_not_ack_or_compression_evidence(tmp_path: Path) -> None:
    data = journal(
        tmp_path,
        [
            event(0, "subscribed"),
            event(0, "heartbeat", connected=True),
            event(60, "heartbeat", connected=True),
        ],
    )
    result = transport_window(data, 0, 60)
    assert result["connected_heartbeat_seconds"] == 60
    assert result["subscription_acknowledgement_states"] == ["unknown"]
    assert result["compression_states"] == ["unknown"]
    assert result["packet_loss_measured"] is False
    assert result["area_wide_reception_proven"] is False


def test_confirmation_state_resets_on_reconnection_and_intervals_do_not_double_count(
    tmp_path: Path,
) -> None:
    events = [
        event(-10, "transport_open", permessage_deflate=True),
        event(-5, "subscription_confirmation", accepted=True, compression_enabled=True),
        event(0, "heartbeat", connected=True),
        event(60, "heartbeat", connected=True),
        event(70, "disconnected"),
        event(80, "transport_open", permessage_deflate=False),
        event(120, "heartbeat", connected=True),
        event(180, "heartbeat", connected=True),
    ]
    result = transport_window(journal(tmp_path, events, True), 0, 180)
    assert result["connected_heartbeat_seconds"] == 120
    assert result["uncovered_or_unknown_seconds"] == 60
    assert result["subscription_acknowledgement_states"] == ["confirmed", "unknown"]
    assert result["compression_states"] == ["false", "true"]


@pytest.mark.parametrize(
    "events",
    [
        [
            event(0, "heartbeat", connected=True),
            event(300, "heartbeat", connected=True),
        ],
        [
            event(0, "heartbeat", connected=False),
            event(60, "heartbeat", connected=True),
        ],
        [
            event(0, "heartbeat", connected=True),
            event(20, "disconnected"),
            event(60, "heartbeat", connected=True),
        ],
    ],
)
def test_sparse_disconnected_or_suspended_journal_cannot_prove_continuity(
    tmp_path: Path, events: list[dict[str, Any]]
) -> None:
    result = transport_window(journal(tmp_path, events), 0, 60)
    assert result["connected_heartbeat_fraction"] == 0


def test_malformed_events_and_uncovered_windows_are_explicit(tmp_path: Path) -> None:
    (tmp_path / "coverage_bad.jsonl").write_text("bad\n{}\n[]\n")
    data = journals(tmp_path)
    assert data["malformed_events"] == 3
    assert transport_window(data, 0, 60)["compression_states"] == ["unknown"]
    with pytest.raises(ValueError):
        transport_window(data, 60, 0)


def report() -> dict[str, Any]:
    return {
        "cells": [
            {
                "id": "cell_-556_502",
                "grid": [-556, 502],
                "geometry": mapping(box(-55.6, 50.2, -55.5, 50.3)),
            }
        ],
        "shortlist": [{"rank": 1, "cell_id": "cell_-556_502"}],
    }


def policy() -> dict[str, Any]:
    return {"grid_degrees": 0.1, "neighbour_rings": 2}


def test_neighbours_are_bounded_deduped_and_frozen_without_reception_requirement() -> (
    None
):
    cells = expand(report(), FROZEN, policy())
    assert len(cells) == 25
    assert cells[0]["id"] == "cell_-556_502" and cells[0]["ring"] == 0
    assert len({c["id"] for c in cells}) == 25
    assert sum(c["ring"] == 2 for c in cells) == 16
    assert cells == expand(report(), FROZEN, policy())
    with pytest.raises(ValueError):
        expand(report(), FROZEN, policy() | {"neighbour_rings": 3})


@pytest.mark.parametrize("split", ["validation", "test", "quarantine"])
def test_neighbour_geography_cannot_override_existing_roles(split: str) -> None:
    prior = {
        "geometry": report()["cells"][0]["geometry"],
        "group": "held",
        "split": split,
        "quarantine": split == "quarantine",
    }
    cells = expand(report(), FROZEN | {"priors": [prior]}, policy())
    assert cells[0]["split_safe"] is False
    assert cells[0]["exclusion"]


def test_qualifying_neighbour_reservation_retains_competitors_and_null_labels() -> None:
    cells = expand(report(), FROZEN, policy())
    s = scene()
    s["support"][0].update(lon=-55.45, lat=50.25)
    result = decide(
        [s],
        [c for c in cells if c["split_safe"]],
        FROZEN,
        json.loads(checked(rules()["source_policy"]).read_text()),
        rules(),
    )
    assert len(result["areas"]) == 1
    assert result["areas"][0]["source_cell_id"] != cells[0]["id"]
    assert result["areas"][0]["label"] is None
    assert result["areas"][0]["classifier_scored"] is False


def test_pass_windows_filter_swath_and_keep_clock_and_missing_receipts_explicit() -> (
    None
):
    s = scene()
    s["native_acquisition_group"] = "S1C_000001_01ABCD"
    s["product"]["ContentDate"] = {
        "Start": "2022-05-01T10:00:00Z",
        "End": "2022-05-01T10:00:25Z",
    }
    centre = datetime(2022, 5, 1, 10, tzinfo=timezone.utc).timestamp()
    cells = expand(report(), FROZEN, policy())
    rows: list[dict[str, Any]] = []
    for offset, lon, receipt in [
        (10, -55.55, 12),
        (20, -55.55, 19),
        (30, -55.55, None),
        (4000, -55.55, 4001),
        (10, -52, 12),
    ]:
        rows.append(
            {
                "time": centre + offset,
                "lon": lon,
                "lat": 50.25,
                "mmsi": "123456789",
                "values": {
                    "received_at": (
                        datetime.fromtimestamp(
                            centre + receipt, timezone.utc
                        ).isoformat()
                        if receipt is not None
                        else None
                    )
                },
            }
        )
    rows.sort(key=lambda r: r["time"])
    result = collection(
        [s], rows, cells, {"sessions": [], "malformed_events": 0}, 900, 0.1
    )[0]
    assert result["reports_in_context"] == 3
    assert result["reports_during_acquisition"] == 2
    assert result["receipt_timestamp_missing"] == 1
    assert result["receipt_minus_observation_seconds"]["negative"] == 1
    assert result["transport"]["window_seconds"] == 1825
    assert result["coverage_complete"] is False
    assert "123456789" not in json.dumps(result)


@pytest.mark.parametrize("compressed", [True, False])
def test_recorder_records_actual_binary_confirmation_and_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    import agents.ais_recorder as recorder

    async def run() -> None:
        async def handler(ws: Any) -> None:
            subscription = json.loads(await ws.recv())
            assert "StandardClassBPositionReport" in subscription["FilterMessageTypes"]
            await ws.send(
                json.dumps(
                    {
                        "MessageType": "SubscriptionConfirmation",
                        "Message": {"CompressionEnabled": compressed},
                    }
                ).encode()
            )
            await ws.wait_closed()

        async with websockets.serve(
            handler, "127.0.0.1", 0, compression="deflate" if compressed else None
        ) as server:
            port = list(server.sockets)[0].getsockname()[1]
            monkeypatch.setattr(recorder, "STREAM_URL", f"ws://127.0.0.1:{port}")
            await recorder.record(
                "fixture", [-53, 47, -52, 48], str(tmp_path), duration_s=0.3
            )

    asyncio.run(run())
    events = [
        json.loads(line)
        for path in tmp_path.glob("coverage_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    transport = next(e for e in events if e["event"] == "transport_open")
    confirmation = next(e for e in events if e["event"] == "subscription_confirmation")
    assert transport["permessage_deflate"] is compressed
    assert confirmation["accepted"] is True
    assert confirmation["compression_enabled"] is compressed
    assert "fixture" not in json.dumps(events)
    assert not list(tmp_path.glob("ais_stream_*.csv"))


def test_public_neighbour_release_replays() -> None:
    assert verify()["labels_admitted"] == 0


def test_nonempty_reservation_publishes_and_replays_without_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Source references intentionally reject paths outside the repository.
    # Keep this generated-interface fixture portable to Linux /tmp checkouts.
    parent = REPO_ROOT / "data/validation"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pass_replay_", dir=parent) as directory:
        _nonempty_reservation_roundtrip(Path(directory), monkeypatch)


def _nonempty_reservation_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agents.neighbour_vessel_batch as batch
    from agents.hh_hv_reservation import save

    private = tmp_path / "private"
    private.mkdir()
    (private / "archive_snapshot").mkdir()
    reception = tmp_path / "reception.json"
    save(reception, report())
    monkeypatch.setattr(batch, "RECEPTION", reception)
    monkeypatch.setattr(batch, "DIRECTORY", tmp_path / "published")
    monkeypatch.setattr(batch, "LOCK", tmp_path / "lock.json")
    frozen = FROZEN | {"prior_metadata_hashes": {}}
    cells = expand(report(), frozen, policy())
    s = scene()
    s["native_acquisition_group"] = "S1C_000001_01ABCD"
    s["product"]["ContentDate"] = {
        "Start": "2022-05-01T10:00:00Z",
        "End": "2022-05-01T10:00:25Z",
    }
    s["support"][0].update(lon=-55.45, lat=50.25)
    raw = {"scenes": [s], "rejected": []}
    save(private / "priors.json", frozen)
    save(private / "search_cells.json", cells)
    save(private / "catalogue.json", [s["product"]])
    save(private / "overlaps_private.json", raw)
    save(private / "shoreline_private.json", {"scenes": [s]})
    save(private / "snapshot_manifest.json", {"files": []})
    save(private / "source_inputs.json", {"coastline": [], "controls": []})
    row = {
        "time": 1651399212.0,
        "lon": -55.45,
        "lat": 50.25,
        "mmsi": "123456789",
        "values": {},
    }
    monkeypatch.setattr(
        batch, "observations", lambda path: ([row], {"valid_unique_reports": 1})
    )
    monkeypatch.setattr(batch, "audit_products", lambda *args: ([s], []))
    monkeypatch.setattr(batch, "enrich_sources", lambda *args: None)
    batch.publish(private)
    assert batch.verify(True)["reserved"] == 1
    published = json.loads((batch.DIRECTORY / "manifest.json").read_text())
    assert published["areas"][0]["id"].startswith("neighbour_v040_")
    assert published["areas"][0]["label"] is None
    with pytest.raises(ValueError, match="Fresh reservation"):
        batch.publish(private)
