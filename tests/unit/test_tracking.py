import os
import sys
from typing import Any, Dict, Iterator

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.tracking import RunTracker, resolve_tracking_uri, tracking_enabled


@pytest.fixture
def no_tracking_env(monkeypatch: Any) -> Iterator[None]:
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    yield


def test_tracking_can_be_disabled(monkeypatch: Any) -> None:
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    assert not tracking_enabled()
    monkeypatch.setenv("TRITONEYE_TRACKING", "on")
    assert tracking_enabled()


def test_tracking_uri_respects_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    assert resolve_tracking_uri() == "http://tracking.example:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    # Defaults to a repo-local SQLite database: MLflow 3.x refuses the old file
    # store, and the model registry only exists on a database backend.
    assert resolve_tracking_uri().startswith("sqlite:///")
    assert resolve_tracking_uri().endswith("mlflow.db")


def test_tracking_uri_is_cwd_independent(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    before = resolve_tracking_uri()
    monkeypatch.chdir(tmp_path)
    assert resolve_tracking_uri() == before


def test_disabled_tracker_is_inert(no_tracking_env: None) -> None:
    # The pipeline must run identically with tracking off, so every method has
    # to be safe to call on an inactive tracker.
    t = RunTracker.start("mission_test")
    assert not t.active
    assert t.run_id is None
    t.log_params({"a": 1})
    t.log_metrics({"b": 2.0})
    t.set_tags({"c": "d"})
    t.log_artifact("nonexistent.txt")
    t.register_detector(None, "x", {})
    t.end()


def test_resume_without_run_id_is_inert(no_tracking_env: None) -> None:
    assert not RunTracker.resume(None).active
    assert not RunTracker.resume("").active


def test_inactive_tracker_context_manager_swallows_nothing(
    no_tracking_env: None,
) -> None:
    # The context manager marks the run FAILED on error but must not suppress
    # the exception itself.
    with pytest.raises(ValueError):
        with RunTracker.start("mission_test"):
            raise ValueError("boom")


def test_log_metrics_skips_non_numeric(monkeypatch: Any) -> None:
    captured: Dict[str, float] = {}

    class FakeMlflow:
        def log_metrics(self, m: Dict[str, float]) -> None:
            captured.update(m)

    class FakeRun:
        class info:  # noqa: N801 - mirrors mlflow's Run.info attribute
            run_id = "abc123"

    t = RunTracker(run=FakeRun(), mlflow_mod=FakeMlflow())
    t.log_metrics(
        {
            "good_int": 3,
            "good_float": 1.5,
            "good_str_number": "2.5",
            "bad_text": "not a number",
            "bad_none": None,
            "bad_bool": True,
        }
    )
    assert captured == {"good_int": 3.0, "good_float": 1.5, "good_str_number": 2.5}


def test_log_params_stringifies_and_drops_none() -> None:
    captured: Dict[str, str] = {}

    class FakeMlflow:
        def log_params(self, p: Dict[str, str]) -> None:
            captured.update(p)

    class FakeRun:
        class info:  # noqa: N801 - mirrors mlflow's Run.info attribute
            run_id = "abc123"

    t = RunTracker(run=FakeRun(), mlflow_mod=FakeMlflow())
    t.log_params({"conf": 0.35, "repo": "a/b", "missing": None})
    assert captured == {"conf": "0.35", "repo": "a/b"}


def test_backend_failure_does_not_propagate() -> None:
    # A tracking backend going down must never take the pipeline with it.
    class ExplodingMlflow:
        def log_metrics(self, m: Dict[str, float]) -> None:
            raise RuntimeError("tracking server unreachable")

        def log_params(self, p: Dict[str, str]) -> None:
            raise RuntimeError("tracking server unreachable")

        def end_run(self, status: str = "") -> None:
            raise RuntimeError("tracking server unreachable")

    class FakeRun:
        class info:  # noqa: N801 - mirrors mlflow's Run.info attribute
            run_id = "abc123"

    t = RunTracker(run=FakeRun(), mlflow_mod=ExplodingMlflow())
    t.log_metrics({"a": 1.0})
    t.log_params({"b": "c"})
    t.end()
