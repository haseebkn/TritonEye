import json
import os
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from agents import pipeline
from agents.tracking import RunTracker


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
            destination = tmp_path / "ais_filtered.csv"
            destination.write_bytes(original_ais.read_bytes())
            payload.update(
                sar_product_id=identity,
                mission_id=kwargs["env"]["TRITONEYE_MISSION_ID"],
                source_mission_id="original_mission",
                ais_telemetry=str(destination),
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


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("versioned", [False, True])
def test_saved_payload_replay_preserves_original_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool, versioned: bool
) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    prior_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    env = {"TRITONEYE_TRACKING": "on", "MLFLOW_TRACKING_URI": uri}
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "tritoneye", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    original = tmp_path / "original"
    original.mkdir()
    files = {
        "detections.geojson": "detections",
        "processing.json": "provenance",
        "report.html": "",
    }
    try:
        with RunTracker.start("original_mission", env=env) as tracker:
            original_id = tracker.run_id
            assert original_id is not None
            tracker.set_tags({"stage": "report", "mission_id": "original_mission"})
            tracker.log_metrics({"detections": 1})
            for filename, artifact_path in files.items():
                path = original / filename
                path.write_text("original evidence for " + filename)
                tracker.log_artifact(str(path), artifact_path)
        before = client.get_run(original_id).to_dictionary()
        original_ais = original / "ais_filtered.csv"
        original_ais.write_text("mmsi,lon,lat,timestamp\n")
        saved = {
            "mission_id": "original_mission",
            "mlflow_run_id": original_id,
            "ais_telemetry": str(original_ais),
        }
        seen: list[dict[str, Any]] = []

        def directory(identifier: str) -> Path:
            destination = tmp_path / "missions" / identifier
            destination.mkdir(parents=True, exist_ok=True)
            return destination

        def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
            current = json.loads(kwargs["input"])
            seen.append(current)
            stage = args[-1].split(".")[-2]
            with RunTracker.resume(
                current.get("mlflow_run_id"), env=kwargs["env"]
            ) as active:
                active.set_tags({"stage": stage})
                active.log_metrics({"detections": 99})
                for filename, artifact_path in files.items():
                    path = directory(current["mission_id"]) / filename
                    path.write_text("replayed evidence for " + filename)
                    active.log_artifact(str(path), artifact_path)
            return CompletedProcess(args, 1 if fail else 0, json.dumps(current))

        monkeypatch.setattr(pipeline, "mission_directory", directory)
        monkeypatch.setattr(pipeline.subprocess, "run", execute)
        provenance = {"version_key": "new-version"} if versioned else None
        if fail:
            with pytest.raises(RuntimeError, match="inference failed"):
                pipeline.run_pipeline(saved, env=env, provenance=provenance)
        else:
            result = pipeline.run_pipeline(saved, env=env, provenance=provenance)
            assert result["source_mlflow_run_id"] == original_id
        assert client.get_run(original_id).to_dictionary() == before
        downloads = tmp_path / "downloaded_original"
        downloads.mkdir()
        for filename, artifact_path in files.items():
            remote = f"{artifact_path}/{filename}" if artifact_path else filename
            downloaded = client.download_artifacts(original_id, remote, str(downloads))
            assert Path(downloaded).read_bytes() == (original / filename).read_bytes()
        replays = [
            r for r in client.search_runs([experiment]) if r.info.run_id != original_id
        ]
        assert len(replays) == 1
        replay_run = replays[0]
        assert replay_run.info.status == ("FAILED" if fail else "FINISHED")
        assert replay_run.data.tags["source_mlflow_run_id"] == original_id
        assert replay_run.data.tags["source_mission_id"] == "original_mission"
        assert all(p["mlflow_run_id"] == replay_run.info.run_id for p in seen)
        assert all(p["mission_id"] != saved["mission_id"] for p in seen)
        assert all(Path(p["ais_telemetry"]) != original_ais for p in seen)
        assert saved["mlflow_run_id"] == original_id
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(prior_uri)


@pytest.mark.parametrize("tracking", ["off", "unavailable"])
def test_untracked_replay_drops_inherited_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: str
) -> None:
    monkeypatch.setattr(pipeline, "mission_directory", lambda _: tmp_path)
    monkeypatch.setenv("TRITONEYE_TRACKING", "on")
    if tracking == "unavailable":
        monkeypatch.setattr(
            RunTracker, "_connect", classmethod(lambda cls, env=None: None)
        )
    seen: list[dict[str, Any]] = []

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        current = json.loads(kwargs["input"])
        seen.append(current)
        return CompletedProcess(args, 0, json.dumps(current))

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    result = pipeline.run_pipeline(
        {"mission_id": "original", "mlflow_run_id": "original-run"},
        env={"TRITONEYE_TRACKING": "off" if tracking == "off" else "on"},
        provenance={"version_key": "new-version"},
    )
    assert result["source_mlflow_run_id"] == "original-run"
    assert "mlflow_run_id" not in result
    assert all("mlflow_run_id" not in payload for payload in seen)
