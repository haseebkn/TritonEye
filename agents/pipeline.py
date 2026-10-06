"""One reproducible, sequential mission with stage logs and a durable status.

Usage: python -m agents.pipeline --mock
       python -m agents.pipeline --date YYYY-MM-DD --aoi labrador_shelf
       python -m agents.pipeline --input existing_ingest_payload.json
"""

import argparse
import json
import os
import shutil
import subprocess as subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.acquisition import product_id, require_product
from agents.artifacts import REPO_ROOT, mission_directory, sha256_file, write_json
from agents.tracking import RunTracker

STAGES = ("ingest", "inference", "correlation", "evaluate", "report")


def run_pipeline(
    payload: dict[str, Any] | None = None,
    *,
    env: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A failure stops the mission; later stages never consume partial outputs."""
    process_env = dict(os.environ) if env is None else dict(env)
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONUNBUFFERED"] = "1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    execution_dir = mission_directory(f"execution_{stamp}")
    current = dict(payload or {})
    selected_id = process_env.get("TARGET_PRODUCT_ID", "")
    if selected_id:
        selected_id = product_id(selected_id)
        process_env["TARGET_PRODUCT_ID"] = selected_id
        if payload is not None:
            require_product(current, selected_id)
    if provenance is not None:
        current["run_provenance"] = provenance
        process_env["TRITONEYE_MISSION_ID"] = f"execution_{stamp}"
    stages = STAGES[1:] if payload is not None else STAGES
    history: list[dict[str, Any]] = []
    replay_run_id = None
    try:
        if payload is not None:
            current["source_mission_id"] = current.get("mission_id")
            current["source_mlflow_run_id"] = current.pop("mlflow_run_id", None)
            current["mission_id"] = f"execution_{stamp}"
            with RunTracker.start(current["mission_id"], env=process_env) as tracker:
                replay_run_id = tracker.run_id
                if replay_run_id:
                    current["mlflow_run_id"] = replay_run_id
                tracker.set_tags(
                    {
                        key: current.get(key)
                        for key in (
                            "mission_id",
                            "source_mission_id",
                            "source_mlflow_run_id",
                        )
                    }
                )
            ais_path = current.get("ais_telemetry")
            if ais_path:
                snapshot = execution_dir / "ais_filtered.csv"
                shutil.copyfile(ais_path, snapshot)
                current["ais_telemetry"] = str(snapshot)
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
            if selected_id:
                require_product(current, selected_id)
                current["selected_product_id"] = selected_id
            if provenance is not None:
                current["run_provenance"] = provenance
            write_json(execution_dir / f"{stage}.json", current)
        current["pipeline_status"] = "completed"
        current["execution_dir"] = str(execution_dir)
        write_json(execution_dir / "mission.json", current)
        return current
    except Exception as error:
        RunTracker.resume(
            replay_run_id if payload is not None else current.get("mlflow_run_id"),
            env=process_env,
        ).end("FAILED")
        write_json(
            execution_dir / "failure.json",
            {
                "status": "failed",
                "selected_product_id": selected_id or None,
                "provenance": provenance,
                "stages": history,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--product-id", help="Exact Copernicus product UUID")
    parser.add_argument("--aoi", default="grand_banks")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, help="Save the complete mission payload")
    args = parser.parse_args()
    if args.product_id and args.mock:
        parser.error("--product-id cannot be combined with --mock")
    env = dict(os.environ)
    if args.mock:
        env["MOCK_INGEST"] = "true"
        env["MOCK_INGEST_TIMESTAMP"] = "20260801T120000"
    else:
        env["MOCK_INGEST"] = "false"
    if args.date:
        env["TARGET_DATE"] = args.date
    if args.product_id:
        env["TARGET_PRODUCT_ID"] = product_id(args.product_id)
    env["AOI_NAME"] = args.aoi
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    provenance = None
    if args.product_id:
        from dotenv import load_dotenv

        from agents.run_versions import digest, processing_versions

        load_dotenv()
        env = {**os.environ, **env}
        processing = processing_versions(REPO_ROOT, env)
        dataset = {
            "selected_product_id": env["TARGET_PRODUCT_ID"],
            "input_manifest_sha256": sha256_file(args.input) if args.input else None,
            "ais_sha256": (
                sha256_file(payload["ais_telemetry"])
                if payload and payload.get("ais_telemetry")
                else None
            ),
        }
        provenance = {
            **processing,
            "dataset": dataset,
            "version_key": digest(
                {
                    "processing": {
                        key: value
                        for key, value in processing.items()
                        if key != "git_commit"
                    },
                    "dataset": dataset,
                }
            ),
        }
    result = run_pipeline(payload, env=env, provenance=provenance)
    if args.output:
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "pipeline_status": result["pipeline_status"],
                    "selected_product_id": result.get("selected_product_id"),
                    "processed_product_id": result.get("sar_product_id"),
                    "evaluation": result.get("evaluation"),
                    "execution_dir": result["execution_dir"],
                    "output": str(args.output),
                },
                indent=2,
                allow_nan=False,
            )
        )
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
