"""Persist acquisition outcomes and run one eligible, versioned product per check."""

import argparse
import csv
import io
import json
import os as os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator

from agents.acquisition import product_id, require_product
from agents.ais_validation import finite_number, parse_utc, utc_string
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.pipeline import run_pipeline
from agents.recorder_health import recorder_health
from agents.region import validate_aoi
from agents.run_versions import acquisition_versions as acquisition_versions
from agents.run_versions import digest, processing_versions
from agents.scene_watch import ais_rows_near, assess, search_acquisitions

# Failed attempts allowed per version key before the scene stops retrying.
MAX_FAILED_ATTEMPTS = 3

PipelineRunner = Callable[..., dict[str, Any]]


@contextmanager
def watch_lock(directory: Path) -> Iterator[None]:
    """Release locks on exit/crash and prevent overlapping scheduled processing."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "watch.lock", "a+b") as stream:
        if os.name == "nt":
            msvcrt = import_module("msvcrt")

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl = import_module("fcntl")

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def snapshot_ais(
    directory: Path, archive: str, row: dict[str, Any], revision: dict[str, Any]
) -> Path:
    rows = ais_rows_near(archive, row["acquired"])
    ordered = sorted(rows, key=lambda value: json.dumps(value, sort_keys=True))
    if digest(ordered) != revision["dataset"]["ais_window_sha256"]:
        raise ValueError(
            "AIS window changed during selection; retry with its new version"
        )
    target = (
        directory / "inputs" / str(row["product_id"]) / str(revision["version_key"])
    )
    target.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list[dict[str, str]]] = {}
    for observation in ordered:
        timestamp = parse_utc(observation["timestamp"], allow_naive=True)
        assert timestamp is not None
        by_day.setdefault(timestamp.date().isoformat(), []).append(observation)
    for day, observations in by_day.items():
        path = target / f"ais_stream_{day}.csv"
        content = io.StringIO(newline="")
        writer = csv.DictWriter(
            content,
            sorted({key for observation in observations for key in observation}),
        )
        writer.writeheader()
        writer.writerows(observations)
        expected = content.getvalue().encode("utf-8")
        if path.exists():
            if path.read_bytes() != expected:
                raise ValueError("Existing AIS snapshot differs from its version")
            continue
        fd, temporary = tempfile.mkstemp(dir=target, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(expected)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if set(target.glob("ais_stream_*.csv")) != {
        target / f"ais_stream_{day}.csv" for day in by_day
    }:
        raise ValueError("AIS snapshot contains files outside its version")
    return target


def transition(
    record: dict[str, Any], state: str, now: datetime, **details: Any
) -> None:
    record["state"] = state
    record.setdefault("history", []).append(
        {"state": state, "at": utc_string(now), **details}
    )


def run_check(
    products: list[dict[str, Any]],
    *,
    aoi: str,
    archive_dir: str,
    cache_dir: str,
    record_dir: Path,
    versions: dict[str, Any],
    env: dict[str, str],
    check_only: bool = False,
    now: datetime | None = None,
    runner: PipelineRunner = run_pipeline,
) -> dict[str, Any]:
    """Only evaluator output establishes measurement; eligibility is provisional."""
    instant = now or datetime.now(timezone.utc)
    if not aoi or Path(aoi).name != aoi or aoi in (".", ".."):
        raise ValueError("AOI must be a filename under configs/aois")
    rows = assess(products, archive_dir, cache_dir)
    # Read health at the moment of reading, not at `instant`: assessment above
    # can take minutes, during which the recorder keeps writing heartbeats, so
    # a pre-assessment timestamp made a healthy recorder look -60 s+ "from the
    # future" and report heartbeat_stale. Tests still inject a fixed clock.
    health = recorder_health(archive_dir, now=None if now is None else instant)
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for product, row in zip(products, rows):
        if row["product_id"] is None:
            continue
        selected = product_id(row["product_id"])
        path = record_dir / "records" / aoi / f"{selected}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("product_id") != selected:
                raise ValueError("Acquisition record has the wrong product ID")
        else:
            record = {
                "schema_version": 1,
                "product_id": selected,
                "aoi": aoi,
                "attempts": [],
            }
            transition(record, "discovered", instant)
        record.update(
            acquisition=row,
            catalogue=product,
            last_checked_at=utc_string(instant),
            recorder_health=health,
        )
        revision = acquisition_versions(versions, product, archive_dir)
        record["current_version_key"] = revision["version_key"]
        previous = [
            attempt
            for attempt in record["attempts"]
            if attempt["version_key"] == revision["version_key"]
        ]
        latest = previous[-1] if previous else None
        failures = sum(1 for attempt in previous if attempt["state"] == "failed")
        if row["eligible"]:
            if latest and latest["state"] in ("processed", "measured"):
                record["state"] = latest["state"]
            elif failures >= MAX_FAILED_ATTEMPTS:
                # A deterministic failure after inference would otherwise burn
                # ~16 min of GPU every hour indefinitely. Changed code, model,
                # config or data yields a new version key and reopens the scene.
                record["state"] = "failed"
                record["retry_exhausted"] = True
            else:
                retry_at = parse_utc(latest.get("finished_at")) if latest else None
                retry_due = (
                    retry_at is None or (instant - retry_at).total_seconds() >= 3600
                )
                if retry_due:
                    if record.get("state") != "eligible":
                        transition(
                            record,
                            "eligible",
                            instant,
                            version_key=revision["version_key"],
                        )
                    candidates.append((path, record, row, revision))
        else:
            record["state"] = "discovered"
        write_json(path, record)
    result: dict[str, Any] = {
        "acquisitions": rows,
        "recorder_health": health,
        "outcome": "checked",
        "eligible_pending": len(candidates),
    }
    if check_only or not candidates:
        return result
    path, record, row, revision = min(
        candidates,
        key=lambda candidate: (
            parse_utc(candidate[1]["attempts"][-1].get("started_at"))
            if candidate[1]["attempts"]
            else None
        )
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    attempt: dict[str, Any] = {
        "version_key": revision["version_key"],
        "provenance": revision,
        "selected_product_id": row["product_id"],
        "started_at": utc_string(instant),
        "state": "eligible",
    }
    record["attempts"].append(attempt)
    write_json(path, record)
    run_env = {
        **env,
        "TARGET_PRODUCT_ID": row["product_id"],
        "TARGET_DATE": row["acquired"][:10],
        "AOI_NAME": aoi,
        "MOCK_INGEST": "false",
    }
    try:
        run_env["AIS_ARCHIVE_DIR"] = str(
            snapshot_ais(record_dir, archive_dir, row, revision)
        )
        run_env["AIS_COVERAGE_ARCHIVE_DIR"] = archive_dir
        payload = runner(env=run_env, provenance=revision)
        require_product(payload, row["product_id"])
        if payload.get("pipeline_status") != "completed":
            raise ValueError("Pipeline returned without completing all stages")
        processing = payload.get("processing") or {}
        if (
            processing.get("complete") is not True
            or processing.get("sar_product_id") != row["product_id"]
        ):
            raise ValueError("Inference did not complete for the selected product")
        expected_weights = revision.get("model", {}).get("actual_sha256")
        if expected_weights and processing.get("weights_sha256") != expected_weights:
            raise ValueError("Inference weights differ from selected model version")
        expected_config = revision.get("configuration", {}).get("configs/model.yaml")
        if expected_config and processing.get("config_sha256") != expected_config:
            raise ValueError("Inference configuration changed after selection")
        finished = datetime.now(timezone.utc) if now is None else instant
        attempt.update(
            state="processed",
            finished_at=utc_string(finished),
            processed_product_id=payload["sar_product_id"],
            execution_dir=payload.get("execution_dir"),
            actual_processing=processing,
        )
        ais_path = payload.get("ais_telemetry")
        attempt["filtered_ais_sha256"] = (
            sha256_file(ais_path) if ais_path and Path(ais_path).is_file() else None
        )
        transition(record, "processed", finished, version_key=revision["version_key"])
        evaluation = payload.get("evaluation") or {}
        attempt["evaluation"] = evaluation
        attempt["ais_coverage_details"] = payload.get("ais_coverage_details")
        if evaluation.get("scored") is True:
            recall = finite_number(evaluation.get("recall_all"))
            denominator = finite_number(evaluation.get("ais_vessels_in_swath"))
            if (
                evaluation.get("sar_product_id") != row["product_id"]
                or recall is None
                or not 0 <= recall <= 1
                or denominator is None
                or denominator <= 0
            ):
                raise ValueError(
                    "Evaluator claimed a measurement without matching product "
                    "and usable metrics"
                )
            attempt["state"] = "measured"
            transition(
                record, "measured", finished, version_key=revision["version_key"]
            )
        else:
            attempt["reason"] = evaluation.get(
                "reason", "evaluation missing or no measurement"
            )
        write_json(path, record)
        return {
            **result,
            "outcome": attempt["state"],
            "product_id": row["product_id"],
            "record": str(path),
            "reason": attempt.get("reason"),
        }
    except Exception as error:
        finished = datetime.now(timezone.utc) if now is None else instant
        attempt.update(
            state="failed", finished_at=utc_string(finished), error=str(error)
        )
        transition(
            record,
            "failed",
            finished,
            version_key=revision["version_key"],
            error=str(error),
        )
        write_json(path, record)
        return {
            **result,
            "outcome": "failed",
            "product_id": row["product_id"],
            "record": str(path),
            "error": str(error),
        }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    # The whole study polygon: every eligible scene in the first week of
    # October fell outside the 1.5-degree eastern_newfoundland box.
    parser.add_argument("--aoi", default="newfoundland_labrador")
    parser.add_argument("--days", type=int, default=12)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--archive-dir", type=Path, default=REPO_ROOT / "data/raw/ais_stream"
    )
    args = parser.parse_args()
    if Path(args.aoi).name != args.aoi or args.days <= 0:
        parser.error("AOI must be a filename and days must be positive")
    env = dict(os.environ)
    aoi_path = (
        REPO_ROOT
        / "configs/aois"
        / (args.aoi if args.aoi.endswith(".geojson") else args.aoi + ".geojson")
    )
    try:
        # Same guard ingest applies. Without it the watcher discovered
        # "eligible" products under nl_shelf (a recording envelope, wider than
        # the study polygon) that ingest rejects every time -- each attempt a
        # guaranteed failure.
        aoi_feature = json.loads(aoi_path.read_text(encoding="utf-8"))["features"][0]
        validate_aoi(aoi_feature["geometry"])
        with watch_lock(REPO_ROOT / "data/watch"):
            products = search_acquisitions(str(aoi_path), days=args.days)
            result = run_check(
                products,
                aoi=args.aoi,
                archive_dir=str(args.archive_dir),
                cache_dir=str(REPO_ROOT / "data/reference"),
                record_dir=REPO_ROOT / "data/watch",
                versions=processing_versions(REPO_ROOT, env),
                env=env,
                check_only=args.check_only,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        sys.exit(1 if result["outcome"] == "failed" else 0)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Watch failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
