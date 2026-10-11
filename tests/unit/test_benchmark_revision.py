"""Executable safeguards for immutable pilot enrichment and image-first review."""

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from agents.artifacts import sha256_file, write_json
from agents.benchmark_revision import draft, finalize, inherit_splits
from agents.nl_benchmark import DEFAULT_DATASET, load_dataset, validation_labels


def test_inherited_scenes_cannot_change() -> None:
    parent, _, _ = load_dataset()
    revised = copy.deepcopy(parent)
    revised["scenes"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="Inherited scene changed"):
        inherit_splits(parent, revised)


@pytest.mark.parametrize("change", ["datatake", "geography", "test"])
def test_new_scene_cannot_cross_parent_boundaries(change: str) -> None:
    parent, _, _ = load_dataset()
    revised = copy.deepcopy(parent)
    scene = copy.deepcopy(parent["scenes"][1])
    scene["product_id"] = "00000000-0000-4000-8000-000000000001"
    if change == "datatake":
        scene["split"] = "validation"
    elif change == "geography":
        scene["acquisition_group"] = "S1D_999999_FFFFFF"
        scene["split"] = "validation"
    else:
        scene["split"] = "test"
    revised["scenes"].append(scene)
    with pytest.raises(ValueError, match="crosses splits|split overlap|new test data"):
        inherit_splits(parent, revised)


def fixture(root: Path) -> tuple[Path, Path]:
    parent_dir = root / "datasets/nl_benchmark/v0.1.0"
    shutil.copytree(DEFAULT_DATASET, parent_dir)
    bundle = root / "bundle"
    images = bundle / "image_first"
    area_dir = images / "new_area"
    area_dir.mkdir(parents=True)
    chip = area_dir / "sigma0.tif"
    affine = from_origin(-52.46, 47.56, 0.00005, 0.00005)
    with rasterio.open(
        chip,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=affine,
        nodata=-99,
    ) as out:
        out.write(np.ones((2, 256, 256), dtype="float32"))
    previews = []
    for band in (1, 2):
        path = area_dir / f"band_{band}.png"
        Image.new("RGB", (256, 256), "gray").save(path)
        previews.append(
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        )
    grid = area_dir / "review_grid.png"
    Image.new("RGB", (256, 256), "gray").save(grid)
    pid = "00000000-0000-4000-8000-000000000001"
    product = {
        "Id": pid,
        "Name": (
            "S1D_IW_GRDH_1SDV_20261010T210000_20261010T210025_"
            "999999_FFFFFF_ABCD.SAFE"
        ),
        "ContentDate": {"Start": "2026-10-10T21:00:00Z", "End": "2026-10-10T21:00:25Z"},
        "GeoFootprint": mapping(box(-52.6, 47.4, -52.3, 47.7)),
    }
    roi = {
        "id": "new_area",
        "size": 256,
        "center": [-52.4536, 47.5536],
        "product_id": pid,
        "window": [0, 0, 256, 256],
        "geometry": mapping(box(-52.46, 47.5472, -52.4472, 47.56)),
        "pixels": 65536,
        "valid_pixels": 65536,
        "cells": [{"id": "0_0", "bbox_px": [0, 0, 256, 256]}],
        "calibration": [],
        "chip": {
            "path": chip.relative_to(root).as_posix(),
            "sha256": sha256_file(chip),
        },
        "previews": previews,
        "review_grid": {
            "path": grid.relative_to(root).as_posix(),
            "sha256": sha256_file(grid),
        },
        "polarizations": ["VV", "VH"],
    }
    write_json(
        images / "manifest.json", {"areas": [roi], "source_bands": [], "failures": []}
    )
    for path in (
        images / "selection_lock.json",
        images / "bundle_hashes.json",
        bundle / "snapshot_manifest.json",
    ):
        write_json(path, {})
    write_json(bundle / "catalogue_products.json", [product])
    plan = root / "plan.json"
    write_json(
        plan,
        {
            "dataset_version": "0.2.0",
            "parent": "datasets/nl_benchmark/v0.1.0",
            "bundle": "bundle",
            "assignments": {
                "new_area": {
                    "split": "train",
                    "group": "eastern_avalon",
                    "regime": "offshore",
                    "region": "Newfoundland",
                }
            },
        },
    )
    shutil.copyfile(
        Path(__file__).parents[2] / "agents/benchmark_revision.py",
        root / "producer.py",
    )
    return plan, root / "datasets/nl_benchmark/v0.2.0"


def assemble(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    plan, directory = fixture(root)
    monkeypatch.setattr("agents.benchmark_revision.__file__", str(root / "producer.py"))
    draft(root, plan, directory)
    return directory


def decision(directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        (directory / "annotation_decisions.json").read_text()
    )
    return result


def test_incomplete_decision_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = assemble(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="complete image-first pass"):
        finalize(tmp_path, directory)
    assert not (directory / "release.json").exists()


@pytest.mark.parametrize("change", ["duplicate_cells", "changed_chip"])
def test_invalid_review_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    directory = assemble(tmp_path, monkeypatch)
    data = decision(directory)
    data["areas"][0].update(
        complete=True, cells_reviewed=["0_0"], notes="Complete synthetic image pass"
    )
    if change == "duplicate_cells":
        data["areas"][0]["cells_reviewed"].append("0_0")
    else:
        (tmp_path / "bundle/image_first/new_area/sigma0.tif").write_bytes(b"changed")
    write_json(directory / "annotation_decisions.json", data)
    with pytest.raises(ValueError, match="complete image-first pass|chip changed"):
        finalize(tmp_path, directory)
    assert not (directory / "release.json").exists()


def test_empty_full_pass_preserves_parent_history_and_blocks_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = assemble(tmp_path, monkeypatch)
    data = decision(directory)
    data["areas"][0].update(
        complete=True,
        cells_reviewed=["0_0"],
        notes="No distinct object in complete synthetic pass; not exhaustive truth.",
    )
    write_json(directory / "annotation_decisions.json", data)
    for name in ("ANNOTATION_GUIDE.md", "DATASET_CARD.md"):
        (directory / name).write_text(
            "Provisional synthetic fixture; no independent review.\n"
        )
    parent_history = json.loads(
        (tmp_path / "datasets/nl_benchmark/v0.1.0/review_history.json").read_text()
    )["events"]
    status = finalize(tmp_path, directory)
    assert status["areas"] == 13
    assert status["independent_review_available"] is False
    history = json.loads((directory / "review_history.json").read_text())["events"]
    assert history[: len(parent_history)] == parent_history
    manifest, annotations, recomputed = load_dataset(directory, artifact_root=tmp_path)
    assert manifest["scenes"][-1]["polarizations"] == ["vv", "vh"]
    assert recomputed["review_status"][-1]["first_pass_complete"] is True
    assert recomputed["review_status"][-1]["objects"] == 0
    with pytest.raises(ValueError, match="Independent complete-area review"):
        validation_labels(directory, "bonavista_coast", artifact_root=tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        finalize(tmp_path, directory)
