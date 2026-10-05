"""Executable watcher regressions: identity, state transitions and versioned replay."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from agents import scene_watch, watch
from agents.run_versions import processing_versions

FIRST = "42286d3d-0000-4000-8000-000000000001"
SECOND = "66d3167d-0000-4000-8000-000000000002"
NOW = datetime(2026, 9, 27, 22, tzinfo=timezone.utc)


def product(identity: str) -> dict[str, Any]:
    return {
        "Id": identity,
        "Name": "S1D_IW_GRDH_1SDV_20260927T213023_20260927T213048_004420_0082A1_1234",
        "ContentDate": {"Start": "2026-09-27T21:30:23Z"},
        "GeoFootprint": {
            "type": "Polygon",
            "coordinates": [[[-52, 46], [-49, 46], [-49, 49], [-52, 49], [-52, 46]]],
        },
    }


def archive(path: Path, *, extra: bool = False, outside: bool = False) -> str:
    path.mkdir(exist_ok=True)
    with open(
        path / "ais_stream_2026-09-27_v2.csv", "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, ["mmsi", "lon", "lat", "timestamp"])
        writer.writeheader()
        writer.writerow(
            {
                "mmsi": "316000001",
                "lon": -60 if outside else -50,
                "lat": 47,
                "timestamp": "2026-09-27T21:30:00Z",
            }
        )
        if extra:
            writer.writerow(
                {
                    "mmsi": "316000002",
                    "lon": -50.1,
                    "lat": 47,
                    "timestamp": "2026-09-27T21:31:00Z",
                }
            )
    return str(path)


def runner(**kwargs: Any) -> dict[str, Any]:
    identity = kwargs["env"]["TARGET_PRODUCT_ID"]
    return {
        "sar_product_id": identity,
        "pipeline_status": "completed",
        "processing": {"sar_product_id": identity, "complete": True},
        "evaluation": {
            "sar_product_id": identity,
            "scored": False,
            "reason": "no AIS on valid SAR pixels",
        },
    }


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        scene_watch, "water_eligible", lambda positions, _: len(positions)
    )
    return {
        "aoi": "eastern_newfoundland",
        "archive_dir": archive(tmp_path / "archive"),
        "cache_dir": str(tmp_path),
        "record_dir": tmp_path / "watch",
        "versions": {
            "code": {"agents/pipeline.py": "version-one"},
            "model": {"expected_sha256": "pinned"},
            "configuration": {},
        },
        "env": {},
        "now": NOW,
        "runner": runner,
    }


def record(setup: dict[str, Any], identity: str = FIRST) -> dict[str, Any]:
    return dict(
        json.loads(
            (
                setup["record_dir"]
                / "records/eastern_newfoundland"
                / f"{identity}.json"
            ).read_text()
        )
    )


def test_two_products_same_day_are_processed_independently(
    setup: dict[str, Any],
) -> None:
    first = watch.run_check([product(FIRST), product(SECOND)], **setup)
    second = watch.run_check([product(FIRST), product(SECOND)], **setup)
    third = watch.run_check([product(FIRST), product(SECOND)], **setup)
    assert first["product_id"] == FIRST
    assert second["product_id"] == SECOND
    assert third["eligible_pending"] == 0
    assert record(setup)["attempts"][0]["processed_product_id"] == FIRST
    assert record(setup, SECOND)["attempts"][0]["processed_product_id"] == SECOND


def test_completed_without_measurement_is_only_processed(setup: dict[str, Any]) -> None:
    outcome = watch.run_check([product(FIRST)], **setup)
    assert outcome["outcome"] == "processed"
    saved = record(setup)
    assert [event["state"] for event in saved["history"]] == [
        "discovered",
        "eligible",
        "processed",
    ]
    assert saved["attempts"][0]["evaluation"]["scored"] is False
    assert not list(setup["record_dir"].glob("*.done"))


def test_reason_named_scored_cannot_fabricate_measurement(
    setup: dict[str, Any],
) -> None:
    def reason_is_scored(**kwargs: Any) -> dict[str, Any]:
        payload = runner(**kwargs)
        payload["evaluation"]["reason"] = "scored"
        return payload

    setup["runner"] = reason_is_scored
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "processed"
    assert record(setup)["state"] == "processed"


def test_measurement_requires_matching_identity_and_actual_metrics(
    setup: dict[str, Any],
) -> None:
    def measured(**kwargs: Any) -> dict[str, Any]:
        payload = runner(**kwargs)
        payload["evaluation"].update(
            scored=True, recall_all=0.5, ais_vessels_in_swath=2
        )
        return payload

    setup["runner"] = measured
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "measured"
    assert record(setup)["state"] == "measured"


@pytest.mark.parametrize("wrong", ["product", "evaluation", "metrics", "incomplete"])
def test_invalid_success_cannot_be_measured(setup: dict[str, Any], wrong: str) -> None:
    def broken(**kwargs: Any) -> dict[str, Any]:
        payload = runner(**kwargs)
        payload["evaluation"].update(
            scored=True, recall_all=1.0, ais_vessels_in_swath=1
        )
        if wrong == "product":
            payload["sar_product_id"] = SECOND
        elif wrong == "evaluation":
            payload["evaluation"]["sar_product_id"] = SECOND
        elif wrong == "metrics":
            del payload["evaluation"]["recall_all"]
        else:
            payload["processing"]["complete"] = False
        return payload

    setup["runner"] = broken
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "failed"
    assert record(setup)["state"] == "failed"


@pytest.mark.parametrize("change", ["code", "model", "configuration", "ais"])
def test_changed_versions_allow_re_evaluation(
    setup: dict[str, Any], change: str
) -> None:
    watch.run_check([product(FIRST)], **setup)
    if change == "ais":
        archive(Path(setup["archive_dir"]), extra=True)
    else:
        setup["versions"][change]["revision"] = "two"
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "processed"
    attempts = record(setup)["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["version_key"] != attempts[1]["version_key"]


def test_future_ais_append_and_commit_only_change_do_not_reprocess(
    setup: dict[str, Any],
) -> None:
    watch.run_check([product(FIRST)], **setup)
    setup["versions"]["git_commit"] = "new-commit-same-code"
    with open(
        Path(setup["archive_dir"]) / "ais_stream_2026-09-27_v2.csv",
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write("316000003,-50,47,2026-09-27T23:00:00Z\n")
    assert watch.run_check([product(FIRST)], **setup)["eligible_pending"] == 0


@pytest.mark.parametrize("missing", [True, False])
def test_missing_footprint_and_out_of_swath_never_run(
    setup: dict[str, Any], missing: bool
) -> None:
    selected = product(FIRST)
    if missing:
        del selected["GeoFootprint"]
    else:
        archive(Path(setup["archive_dir"]), outside=True)
    assert watch.run_check([selected], **setup)["eligible_pending"] == 0
    assert record(setup)["state"] == "discovered"
    assert record(setup)["attempts"] == []


def test_failed_attempt_retries_after_backoff_and_does_not_block_other_product(
    setup: dict[str, Any],
) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("inference tile failed")

    setup["runner"] = fail
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "failed"
    setup["runner"] = runner
    assert (
        watch.run_check([product(FIRST), product(SECOND)], **setup)["product_id"]
        == SECOND
    )
    setup["now"] += timedelta(hours=1)
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "processed"


def test_attempt_uses_frozen_ais_window(setup: dict[str, Any]) -> None:
    def inspect(**kwargs: Any) -> dict[str, Any]:
        snapshot = Path(kwargs["env"]["AIS_ARCHIVE_DIR"])
        rows = scene_watch.ais_rows_near(
            str(snapshot), product(FIRST)["ContentDate"]["Start"]
        )
        assert len(rows) == 1
        assert kwargs["provenance"]["dataset"]["ais_window_rows"] == 1
        archive(Path(setup["archive_dir"]), extra=True)
        assert (
            len(
                scene_watch.ais_rows_near(
                    str(snapshot), product(FIRST)["ContentDate"]["Start"]
                )
            )
            == 1
        )
        return runner(**kwargs)

    setup["runner"] = inspect
    assert watch.run_check([product(FIRST)], **setup)["outcome"] == "processed"


def test_check_only_records_eligibility_without_processing(
    setup: dict[str, Any],
) -> None:
    watch.run_check([product(FIRST)], check_only=True, **setup)
    assert record(setup)["state"] == "eligible"
    assert record(setup)["attempts"] == []


def test_os_lock_refuses_overlapping_watchers_and_releases(tmp_path: Path) -> None:
    with watch.watch_lock(tmp_path):
        with pytest.raises(OSError):
            with watch.watch_lock(tmp_path):
                pytest.fail("second watcher acquired the lock")
    with watch.watch_lock(tmp_path):
        pass


def test_versions_hash_actual_model_config_code_and_reference(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "agents").mkdir()
    (tmp_path / "models/xview3").mkdir(parents=True)
    (tmp_path / "configs/model.yaml").write_text(
        "inference: {detector: xview3}\nmodel: {xview3_sha256: pinned}\n"
    )
    weights = tmp_path / "models/xview3/traced_ensemble.jit"
    weights.write_bytes(b"first weights")
    before = processing_versions(tmp_path, {"COPERNICUS_PASS": "must-not-leak"})
    weights.write_bytes(b"second weights")
    after = processing_versions(tmp_path, {})
    assert before["model"]["actual_sha256"] != after["model"]["actual_sha256"]
    assert "must-not-leak" not in json.dumps(before)
    assert "configs/model.yaml" in before["configuration"]
