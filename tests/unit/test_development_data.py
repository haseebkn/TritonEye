"""Executable discovery checks protect held-out data before any native raster open."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import rasterio

from agents.artifacts import sha256_file
from agents.nl_benchmark import DEFAULT_DATASET, load_dataset
from tests.unit import test_calibration, test_georef
from tests.unit.development_data import measurements


def _shorten_measurement_paths(directory: Path) -> None:
    """Shrink copied SAFE/measurement paths to fit Windows MAX_PATH under tmp_path."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, scene in enumerate(manifest["scenes"]):
        for source in scene["sources"]:
            if "/measurement/" not in source["path"]:
                continue
            pol = next(
                p for p in ("vv", "vh", "hh", "hv") if f"-grd-{p}-" in source["path"]
            )
            source["path"] = (
                f"data/raw/s{index}.SAFE/measurement/s-grd-{pol}-{index}.tiff"
            )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    release_path = directory / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["files"]["manifest.json"] = sha256_file(manifest_path)
    release_path.write_text(json.dumps(release), encoding="utf-8")


def test_optional_real_checks_exclude_held_out_and_unregistered_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "datasets/nl_benchmark/v0.1.0"
    shutil.copytree(DEFAULT_DATASET, directory)  # Public metadata only.
    _shorten_measurement_paths(directory)  # Windows MAX_PATH guard for this copy.
    manifest, _, _ = load_dataset(directory)
    development = next(
        s
        for s in manifest["scenes"]
        if s["split"] == "train" and s["polarizations"] == ["vv", "vh"]
    )
    held = next(s for s in manifest["scenes"] if s["split"] == "test")
    allowed_path = tmp_path / next(
        s["path"]
        for s in development["sources"]
        if "/measurement/" in s["path"] and "-grd-vv-" in s["path"]
    )
    held_path = tmp_path / next(
        s["path"]
        for s in held["sources"]
        if "/measurement/" in s["path"] and "-grd-vv-" in s["path"]
    )
    unknown = tmp_path / "data/raw/unregistered.SAFE/measurement/unregistered-vv.tiff"
    for path in (allowed_path, held_path, unknown):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic native raster placeholder")
    assert measurements(tmp_path) == [str(allowed_path)]
    monkeypatch.setattr(test_calibration, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(test_georef, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(test_calibration, "find_calibration_xml", lambda p: "synthetic")
    opened = []

    @contextmanager
    def guarded_open(path: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        assert Path(path) == allowed_path, "Held-out or unregistered raster opened"
        opened.append(path)
        yield SimpleNamespace(crs=None, gcps=([object()], None))

    monkeypatch.setattr(rasterio, "open", guarded_open)
    assert test_calibration.real_products() == [str(allowed_path)]
    assert test_georef.real_grd_products() == [str(allowed_path)]
    assert opened == [str(allowed_path)]
