"""Read hash-bound AIS rows or a frozen archive for baseline supporting metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.ais_validation import parse_utc
from agents.artifacts import sha256_file
from agents.nl_benchmark import bounded_path
from agents.overlap_audit import observations


def load_support(scene: dict[str, Any], root: Path) -> tuple[list[Any], dict[str, str]]:
    snapshot = scene["ais"]["snapshot"]
    path = bounded_path(root, snapshot["path"])
    if sha256_file(path) != snapshot["sha256"]:
        raise ValueError("AIS snapshot hash mismatch")
    content = json.loads(path.read_text(encoding="utf-8"))
    if "rows" in content:
        if not isinstance(content["rows"], list):
            raise ValueError("AIS snapshot rows must be a list")
        return content["rows"], {snapshot["path"]: snapshot["sha256"]}
    archive = path.parent / "archive_snapshot"
    files = content["files"]
    listed = {f["file"] for f in files}
    if len(listed) != len(files):
        raise ValueError("Duplicate frozen archive entries")
    consumed = {
        p.name
        for pattern in ("ais_stream_*.csv", "coverage_*.jsonl")
        for p in archive.glob(pattern)
    }
    if consumed - listed:
        raise ValueError("Unlisted frozen archive input")
    checksums = {snapshot["path"]: snapshot["sha256"]}
    for item in files:
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Plain archive filename required")
        relative = str((archive / item["file"]).relative_to(root)).replace("\\", "/")
        source = bounded_path(root, relative)
        if sha256_file(source) != item["sha256"]:
            raise ValueError("Frozen AIS archive hash mismatch")
        checksums[relative] = item["sha256"]
    acq = parse_utc(scene["acquisition_time"])
    if acq is None:
        raise ValueError("Valid scene-start timestamp required")
    reports, _ = observations(archive)
    # Retain all geography and competing identities before motion alignment.
    rows = [r["values"] for r in reports if abs(r["time"] - acq.timestamp()) <= 300]
    return rows, checksums
