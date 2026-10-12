"""Frozen supporting telemetry is reproducible and cannot substitute vessel truth."""

import json
from pathlib import Path
from typing import Any

import pytest

from agents import baseline_evidence
from agents.artifacts import sha256_file, write_json
from agents.baseline_support import load_support


def test_frozen_archive_alignment_inputs_and_integrity(tmp_path: Path) -> None:
    archive = tmp_path / "archive_snapshot"
    archive.mkdir()
    csv = archive / "ais_stream_2026-09-22.csv"
    csv.write_text(
        "mmsi,lon,lat,timestamp,speed_knots,course_deg\n"
        "123456789,-52.7,47.5,2026-09-22T21:00:00Z,2,90\n"
        "987654321,-60,55,2026-09-22T21:02:00Z,3,180\n"
        "111111111,-52.7,47.5,2026-09-22T20:00:00Z,0,0\n"
    )
    manifest = tmp_path / "snapshot_manifest.json"
    write_json(manifest, {"files": [{"file": csv.name, "sha256": sha256_file(csv)}]})
    scene = {
        "acquisition_time": "2026-09-22T21:00:00Z",
        "ais": {
            "snapshot": {
                "path": manifest.name,
                "sha256": sha256_file(manifest),
            }
        },
    }
    rows, hashes = load_support(scene, tmp_path)
    assert len(rows) == 2 and {r["mmsi"] for r in rows} == {"123456789", "987654321"}
    assert len(hashes) == 2  # Outside-area competitors retained before alignment.
    csv.write_text("modified archive")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        load_support(scene, tmp_path)
    unexpected = archive / "ais_stream_2026-09-23.csv"
    unexpected.write_text("extra unversioned input")
    with pytest.raises(ValueError, match="Unlisted"):
        load_support(scene, tmp_path)


def test_legacy_frozen_rows_verify_before_reading(tmp_path: Path) -> None:
    snapshot = tmp_path / "rows.json"
    write_json(snapshot, {"rows": []})
    scene = {
        "ais": {"snapshot": {"path": snapshot.name, "sha256": sha256_file(snapshot)}}
    }
    assert load_support(scene, tmp_path)[0] == []
    snapshot.write_text('{"rows":[{}]}')
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        load_support(scene, tmp_path)


def test_public_evidence_preserves_null_metrics_and_refuses_test_and_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_evidence, "REPO_ROOT", tmp_path)
    report: dict[str, Any] = {
        k: None
        for k in (
            "purpose",
            "evidence_status",
            "scope",
            "baseline_threshold",
            "validation_baseline",
            "stratified_validation",
            "limits",
        )
    }
    report.update(
        {
            "provenance": {"baseline_protocol_version": 3},
            "threshold_selection": {"status": "blocked", "selected_threshold": None},
            "cases": [
                {
                    "split": "validation",
                    "roi_id": "synthetic",
                    "metric_ready": False,
                    "scores": [{"raw": None}],
                    "private_ais_rows": [{"mmsi": "123456789"}],
                }
            ],
        }
    )
    path, output = tmp_path / "baseline.json", tmp_path / "public.json"
    write_json(path, report)
    result = baseline_evidence.publish(path, output)
    assert json.loads(output.read_text()) == result
    assert result["cases"][0]["scores"][0]["raw"] is None
    assert "private_ais_rows" not in result["cases"][0]
    assert result["full_report_sha256"] == sha256_file(path)
    with pytest.raises(ValueError, match="Fresh"):
        baseline_evidence.publish(path, output)
    report["cases"][0]["split"] = "test"
    write_json(path, report)
    with pytest.raises(ValueError, match="Test cases"):
        baseline_evidence.publish(path, tmp_path / "test.json")
