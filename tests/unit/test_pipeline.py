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
