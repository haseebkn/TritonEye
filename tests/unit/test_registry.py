from pathlib import Path
from typing import Any

from agents.tracking import RunTracker


def test_registry_reuses_same_content_in_real_sqlite_backend(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    monkeypatch.chdir(tmp_path)
    uri = "sqlite:///" + str(tmp_path / "tracking.db").replace("\\", "/")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("TRITONEYE_TRACKING", "on")
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"test model identity; no inference performed")
    original = mlflow.get_tracking_uri()
    try:
        with RunTracker.start("registry_test_one") as first:
            assert first.active
            v1 = first.register_detector(str(weights), "test-detector", {})
        with RunTracker.start("registry_test_two") as second:
            v2 = second.register_detector(str(weights), "test-detector", {})
        assert v1 == v2 == "1"
        assert len(MlflowClient().search_model_versions("name='test-detector'")) == 1
    finally:
        mlflow.set_tracking_uri(original)
