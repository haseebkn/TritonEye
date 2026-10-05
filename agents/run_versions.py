"""Content identities used to decide whether an acquisition needs re-evaluation."""

import hashlib
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from agents.artifacts import sha256_file
from agents.landmask import SOURCES
from agents.scene_watch import ais_rows_near


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def hash_paths(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def processing_versions(root: Path, env: dict[str, str]) -> dict[str, Any]:
    """Hash actual local weights/reference data and only non-secret overrides."""
    config = yaml.safe_load((root / "configs/model.yaml").read_text(encoding="utf-8"))
    detector = env.get("TRITONEYE_DETECTOR") or config["inference"]["detector"]
    if detector == "xview3":
        weights = Path(
            env.get("XVIEW3_WEIGHTS", str(root / "models/xview3/traced_ensemble.jit"))
        )
        expected = config["model"].get("xview3_sha256")
    else:
        weights = Path(
            env.get(
                "YOLO_WEIGHTS",
                str(
                    root
                    / "models"
                    / env.get("HUGGINGFACE_MODEL_FILE", "unquantized/best.pt")
                ),
            )
        )
        expected = env.get("YOLO_WEIGHTS_SHA256") or config["model"].get("yolo_sha256")
    if not weights.is_absolute():
        weights = root / weights
    cache = Path(config.get("landmask", {}).get("cache_dir") or root / "data/reference")
    if not cache.is_absolute():
        cache = root / cache
    source = config.get("landmask", {}).get("source", "osm")
    shapefile = cache / SOURCES[source]["shapefile"]
    reference = {
        path.name: sha256_file(path)
        for extension in (".shp", ".shx", ".dbf", ".prj")
        if (path := shapefile.with_suffix(extension)).is_file()
    }
    software: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "rasterio", "geopandas", "shapely", "numpy", "scipy"):
        try:
            software[name] = version(name)
        except PackageNotFoundError:
            software[name] = "unavailable"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit,
        "code": hash_paths(
            list((root / "agents").rglob("*.py"))
            + list((root / "scripts").glob("watch*.sh")),
            root,
        ),
        "configuration": hash_paths(
            list((root / "configs").rglob("*"))
            + [root / "requirements.txt", root / "pyproject.toml"],
            root,
        ),
        "model": {
            "detector": detector,
            "expected_sha256": expected,
            "actual_sha256": sha256_file(weights) if weights.is_file() else None,
        },
        "overrides": {
            key: env[key]
            for key in (
                "TRITONEYE_DETECTOR",
                "TRITONEYE_XVIEW3_THRESHOLD",
                "FORCE_REAL_INFERENCE",
                "YOLO_WEIGHTS_SHA256",
                "HUGGINGFACE_MODEL_REPO",
                "HUGGINGFACE_MODEL_FILE",
                "HUGGINGFACE_MODEL_REVISION",
            )
            if key in env
        },
        "software": software,
        "shoreline": {"source": source, "files_sha256": reference},
    }


def acquisition_versions(
    processing: dict[str, Any], product: dict[str, Any], archive_dir: str
) -> dict[str, Any]:
    rows = ais_rows_near(archive_dir, product["ContentDate"]["Start"])
    dataset = {
        "catalogue_sha256": digest(
            {
                key: product.get(key)
                for key in ("Id", "Name", "ContentDate", "GeoFootprint", "Checksum")
            }
        ),
        "ais_window_sha256": digest(
            sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
        ),
        "ais_window_rows": len(rows),
    }
    # The commit is recorded for navigation; actual source hashes determine
    # identity, so committing unchanged content does not invalidate a result.
    identity = {key: value for key, value in processing.items() if key != "git_commit"}
    return {
        **processing,
        "dataset": dataset,
        "version_key": digest({"processing": identity, "dataset": dataset}),
    }
