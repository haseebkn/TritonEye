#!/usr/bin/env python3
"""
TritonEye Experiment Tracking

Thin wrapper over MLflow that records one run per mission, spanning all four
pipeline stages.

The pipeline runs as separate processes piped together, so the run is created
by the ingest agent and its id travels downstream in the JSON payload
(`mlflow_run_id`). Later stages resume that run rather than opening their own,
so a mission's parameters, operational metrics and output artifacts all land in
one place.

Tracking is deliberately non-fatal. A missions pipeline that cannot reach its
tracking backend should still produce detections, so every call here degrades to
a no-op and reports the reason on stderr rather than raising. Set
TRITONEYE_TRACKING=off to disable entirely.
"""

import math
import os
import sys
from types import TracebackType
from typing import Any, Dict, Optional, Type

DEFAULT_EXPERIMENT = "tritoneye"

# Repo root, so the tracking store lands in the same place no matter which
# directory an agent is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def tracking_enabled() -> bool:
    return os.getenv("TRITONEYE_TRACKING", "on").lower() not in ("off", "0", "false")


def resolve_tracking_uri() -> str:
    """
    Tracking backend, defaulting to a repo-local SQLite database.

    SQLite rather than the older `file:./mlruns` store for two reasons: MLflow
    3.x has put the filesystem backend into maintenance mode and refuses it
    without an opt-out flag, and the **model registry requires a database
    backend** — a file store silently has no registry at all. Still no server
    to run; inspect with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
    Override with MLFLOW_TRACKING_URI to point at a shared tracking server.
    """
    override = os.getenv("MLFLOW_TRACKING_URI")
    if override:
        return override
    db_path = os.path.join(_REPO_ROOT, "mlflow.db").replace("\\", "/")
    return f"sqlite:///{db_path}"


class RunTracker:
    """
    A single mission's MLflow run.

    Use :meth:`start` in the first stage of the pipeline and :meth:`resume` in
    later stages, passing the run id along in the payload. When tracking is
    disabled or unavailable, every method silently does nothing and
    :attr:`run_id` is None, so callers need no conditional logic.
    """

    def __init__(self, run: Any = None, client: Any = None, mlflow_mod: Any = None):
        self._run = run
        self._client = client
        self._mlflow = mlflow_mod

    # ---------------------------------------------------------------- lifecycle

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def run_id(self) -> Optional[str]:
        if self._run is None:
            return None
        return str(self._run.info.run_id)

    @classmethod
    def _connect(cls) -> Any:
        if not tracking_enabled():
            return None
        try:
            import mlflow

            mlflow.set_tracking_uri(resolve_tracking_uri())
            return mlflow
        except Exception as e:  # pragma: no cover - depends on local install
            print(f"Tracking disabled ({e}).", file=sys.stderr)
            return None

    @classmethod
    def start(
        cls, mission_id: str, experiment: str = DEFAULT_EXPERIMENT
    ) -> "RunTracker":
        """Opens a new run named for the mission."""
        mlflow = cls._connect()
        if mlflow is None:
            return cls()
        try:
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(run_name=mission_id)
            print(
                f"Tracking run {run.info.run_id} in {resolve_tracking_uri()}",
                file=sys.stderr,
            )
            return cls(run=run, mlflow_mod=mlflow)
        except Exception as e:
            print(f"Tracking unavailable ({e}); continuing untracked.", file=sys.stderr)
            return cls()

    @classmethod
    def resume(cls, run_id: Optional[str]) -> "RunTracker":
        """Re-attaches to a run opened by an upstream stage."""
        if not run_id:
            return cls()
        mlflow = cls._connect()
        if mlflow is None:
            return cls()
        try:
            run = mlflow.start_run(run_id=run_id)
            return cls(run=run, mlflow_mod=mlflow)
        except Exception as e:
            print(f"Could not resume run {run_id} ({e}).", file=sys.stderr)
            return cls()

    def end(self, status: str = "FINISHED") -> None:
        if not self.active:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as e:
            print(f"Failed to close tracking run ({e}).", file=sys.stderr)

    def __enter__(self) -> "RunTracker":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.end("FAILED" if exc_type is not None else "FINISHED")

    # ---------------------------------------------------------------- recording

    def log_params(self, params: Dict[str, Any]) -> None:
        """Records run inputs. Values are stringified; None entries are skipped."""
        if not self.active:
            return
        clean = {k: str(v) for k, v in params.items() if v is not None}
        if not clean:
            return
        try:
            self._mlflow.log_params(clean)
        except Exception as e:
            print(f"Failed to log params ({e}).", file=sys.stderr)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Records numeric outcomes. Non-numeric or None entries are skipped."""
        if not self.active:
            return
        clean = {}
        for k, v in metrics.items():
            if v is None or isinstance(v, bool):
                continue
            try:
                number = float(v)
                if math.isfinite(number):
                    clean[k] = number
            except (TypeError, ValueError):
                continue
        if not clean:
            return
        try:
            self._mlflow.log_metrics(clean)
        except Exception as e:
            print(f"Failed to log metrics ({e}).", file=sys.stderr)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        if not self.active:
            return
        clean = {k: str(v) for k, v in tags.items() if v is not None}
        if not clean:
            return
        try:
            self._mlflow.set_tags(clean)
        except Exception as e:
            print(f"Failed to set tags ({e}).", file=sys.stderr)

    def log_artifact(self, path: Optional[str], artifact_path: str = "") -> None:
        """Attaches an output file (GeoJSON, HTML report) to the run."""
        if not self.active or not path or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path, artifact_path or None)
        except Exception as e:
            print(f"Failed to log artifact {path} ({e}).", file=sys.stderr)

    # ----------------------------------------------------------------- registry

    def register_detector(
        self, weights_path: Optional[str], name: str, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Records which detector produced this mission, in the model registry.

        The weights are a third-party artifact rather than something trained
        here, so they are registered for provenance: the registry answers "which
        model produced this mission's detections", which is what production
        monitoring needs when results shift.

        Registration is done with `create_model_version` against the logged
        artifact rather than `register_model`, because MLflow 3 resolves
        `runs:/<id>/model` through its LoggedModel abstraction and rejects a
        plain artifact directory. Returns the version string, or None if the
        backend has no registry (a file store does not).
        """
        if not self.active or not weights_path or not os.path.exists(weights_path):
            return None
        try:
            from mlflow.tracking import MlflowClient

            from agents.artifacts import sha256_file

            client = self._client or MlflowClient()
            digest = sha256_file(weights_path)
            # Reuse the immutable weights identity; thresholds belong to the
            # mission, not a fresh 1.3 GB model version per inference run.
            metadata = {**metadata, "weights_sha256": digest}
            self.set_tags({f"model.{k}": v for k, v in metadata.items()})
            if not name or any(
                c
                not in (
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                )
                for c in name
            ):
                raise ValueError("Invalid registered model name")
            matches = client.search_model_versions(
                f"name = '{name}' and tags.weights_sha256 = '{digest}'"
            )
            if matches:
                existing = matches[0]
                self.set_tags({"model.registry_version": str(existing.version)})
                return str(existing.version)
            try:
                client.create_registered_model(name)
            except Exception:
                client.get_registered_model(
                    name
                )  # distinguish existing from backend failure
            self._mlflow.log_artifact(weights_path, "model")
            version = client.create_model_version(
                name=name, source=f"runs:/{self.run_id}/model", run_id=self.run_id
            )
            for k, v in metadata.items():
                try:
                    client.set_model_version_tag(name, version.version, k, str(v))
                except Exception:
                    pass
            print(
                f"Registered detector '{name}' version {version.version}.",
                file=sys.stderr,
            )
            self.set_tags({"model.registry_version": str(version.version)})
            return str(version.version)
        except Exception as e:
            # A file-store backend has no registry; the weights and provenance
            # tags on the run still record what ran.
            print(
                f"Model registry unavailable ({e}); registration not confirmed.",
                file=sys.stderr,
            )
            return None
