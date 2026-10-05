import json
import os
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from agents import pipeline


def test_stages_evaluate_before_report_and_keep_outputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[str] = []

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        stage = args[-1].split(".")[-2]
        calls.append(stage)
        payload = json.loads(kwargs["input"])
        if stage == "evaluate":
            payload["evaluation"] = {"precision": None}
        if stage == "report":
            assert "evaluation" in payload
        return CompletedProcess(args, 0, json.dumps(payload))

    monkeypatch.setattr(pipeline, "mission_directory", lambda _: tmp_path)
    monkeypatch.setattr(subprocess, "run", execute)
    result = pipeline.run_pipeline(env={**os.environ, "TRITONEYE_TRACKING": "off"})
    assert calls == ["ingest", "inference", "correlation", "evaluate", "report"]
    assert result["pipeline_status"] == "completed"
    assert json.loads((tmp_path / "report.json").read_text())["evaluation"] == {
        "precision": None
    }


def test_stage_failure_stops_without_success_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[str] = []

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        stage = args[-1].split(".")[-2]
        calls.append(stage)
        return CompletedProcess(args, 1, "")

    monkeypatch.setattr(pipeline, "mission_directory", lambda _: tmp_path)
    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    with pytest.raises(RuntimeError, match="inference failed"):
        pipeline.run_pipeline({"mission_id": "replay"})
    assert calls == ["inference"]
    assert not (tmp_path / "mission.json").exists()
    assert json.loads((tmp_path / "failure.json").read_text())["status"] == "failed"


def test_exact_product_survives_each_stage_and_replay_is_preserved(
    tmp_path: Path, monkeypatch: Any
) -> None:
    identity = "42286d3d-cd80-4dc7-9aec-1613992b8256"
    original_ais = tmp_path / "original.csv"
    original_ais.write_text("mmsi,lon,lat,timestamp\n")
    calls: list[str] = []

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        stage = args[-1].split(".")[-2]
        calls.append(stage)
        assert kwargs["env"]["TARGET_PRODUCT_ID"] == identity
        payload = json.loads(kwargs["input"])
        if stage == "ingest":
            payload.update(
                sar_product_id=identity,
                mission_id="original_mission",
                ais_telemetry=str(original_ais),
            )
        else:
            assert payload["sar_product_id"] == identity
            assert payload["selected_product_id"] == identity
            assert payload["run_provenance"]["version_key"] == "replay-version"
            assert payload["ais_telemetry"] != str(original_ais)
        if stage == "inference":
            payload["processing"] = {"sar_product_id": identity, "complete": True}
        if stage == "evaluate":
            payload["evaluation"] = {"sar_product_id": identity, "scored": False}
        return CompletedProcess(args, 0, json.dumps(payload))

    monkeypatch.setattr(pipeline, "mission_directory", lambda _: tmp_path)
    monkeypatch.setattr(subprocess, "run", execute)
    result = pipeline.run_pipeline(
        env={"TARGET_PRODUCT_ID": identity},
        provenance={"version_key": "replay-version"},
    )
    assert calls == list(pipeline.STAGES)
    assert (
        result["selected_product_id"]
        == result["processing"]["sar_product_id"]
        == result["evaluation"]["sar_product_id"]
    )
    assert result["evaluation"]["scored"] is False
    original_ais.write_text("changed later\n")
    assert Path(result["ais_telemetry"]).read_text() == "mmsi,lon,lat,timestamp\n"
    assert (
        json.loads((tmp_path / "mission.json").read_text())["selected_product_id"]
        == identity
    )


def test_ingestion_returning_different_product_aborts_before_inference(
    tmp_path: Path, monkeypatch: Any
) -> None:
    selected = "42286d3d-cd80-4dc7-9aec-1613992b8256"
    calls: list[str] = []

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        calls.append(args[-1])
        return CompletedProcess(
            args,
            0,
            json.dumps({"sar_product_id": "66d3167d-240a-459a-8066-75b9d2a458f2"}),
        )

    monkeypatch.setattr(pipeline, "mission_directory", lambda _: tmp_path)
    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    with pytest.raises(ValueError, match="differs from selected"):
        pipeline.run_pipeline(env={"TARGET_PRODUCT_ID": selected})
    assert calls == ["agents.ingest.ingest_agent"]
    assert (
        json.loads((tmp_path / "failure.json").read_text())["selected_product_id"]
        == selected
    )
