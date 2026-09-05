"""One reproducible, sequential mission with stage logs and a durable status.

Usage: python -m agents.pipeline --mock
       python -m agents.pipeline --date YYYY-MM-DD --aoi labrador_shelf
       python -m agents.pipeline --input existing_ingest_payload.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT, mission_directory, write_json
from agents.tracking import RunTracker

STAGES = ("ingest", "inference", "correlation", "evaluate", "report")


def run_pipeline(
    payload: dict[str, Any] | None = None, *, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """A failure stops the mission; later stages never consume partial outputs."""
    process_env = dict(os.environ) if env is None else dict(env)
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONUNBUFFERED"] = "1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    execution_dir = mission_directory(f"execution_{stamp}")
    current = dict(payload or {})
    stages = STAGES[1:] if payload is not None else STAGES
    history: list[dict[str, Any]] = []
    try:
        for stage in stages:
            print(f"Running {stage}...", file=sys.stderr, flush=True)
            with open(execution_dir / f"{stage}.log", "w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-m", f"agents.{stage}.{stage}_agent"],
                    input=json.dumps(current, allow_nan=False),
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    encoding="utf-8",
                    cwd=REPO_ROOT,
                    env=process_env,
                    check=False,
                )
            history.append({"stage": stage, "exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError(
                    f"{stage} failed; details: {execution_dir / (stage + '.log')}"
                )
            current = json.loads(result.stdout)
            if not isinstance(current, dict):
                raise ValueError(f"{stage} did not return a JSON object")
            write_json(execution_dir / f"{stage}.json", current)
        current["pipeline_status"] = "completed"
        current["execution_dir"] = str(execution_dir)
        write_json(execution_dir / "mission.json", current)
        return current
    except Exception as error:
        RunTracker.resume(current.get("mlflow_run_id")).end("FAILED")
        write_json(
            execution_dir / "failure.json",
            {"status": "failed", "stages": history, "error": str(error)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--aoi", default="grand_banks")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    env = dict(os.environ)
    if args.mock:
        env["MOCK_INGEST"] = "true"
        env["MOCK_INGEST_TIMESTAMP"] = "20260801T120000"
    else:
        env["MOCK_INGEST"] = "false"
    if args.date:
        env["TARGET_DATE"] = args.date
    env["AOI_NAME"] = args.aoi
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    print(json.dumps(run_pipeline(payload, env=env), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
