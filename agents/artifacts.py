"""Atomic mission artifacts and content-addressed provenance."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mission_directory(mission_id: str, root: str | Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", mission_id):
        raise ValueError("Invalid mission_id; expected a plain identifier")
    directory = Path(root or REPO_ROOT / "missions") / mission_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, value: Any) -> None:
    """Never expose half-written JSON or emit nonstandard NaN values."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content + "\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
