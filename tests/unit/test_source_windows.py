"""Executable native transport and provisional annotation safeguards."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from shapely.geometry import box, mapping, shape

from agents.artifacts import REPO_ROOT
from agents.native_cog import read_cog
from agents.reserved_sar import Cdse, NativeTiff, locate
from agents.source_annotations import (
    check_collection,
    public_metadata,
    validate_first_pass,
    validate_reconciliation,
)
from agents.source_reservation import ref
from agents.source_windows import scope


def test_compressed_native_ranges_reproduce_base_pixels(tmp_path: Path) -> None:
    path = tmp_path / "compressed.tif"
    values = np.random.default_rng(21).integers(1, 5000, (2048, 2048), dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=2048,
        height=2048,
        count=1,
        dtype="uint16",
        blocksize=256,
        compress="deflate",
    ) as dst:
        dst.write(values, 1)
    raw = path.read_bytes()

    class Client(Cdse):
        def __init__(self) -> None:
            self.bytes = 0

        def fetch(
            self,
            url: str,
            start: int | None = None,
            count: int | None = None,
            total: int | None = None,
        ) -> bytes:
            assert start is not None and count is not None and total == len(raw)
            assert 0 <= start < start + count <= total
            self.bytes += count
            return raw[start : start + count]

    client = Client()
    arrays, receipts = read_cog(
        client, "fixture", len(raw), (2048, 2048), [(301, 287, 384), (1100, 1200, 384)]
    )
    np.testing.assert_array_equal(arrays[0], values[287:671, 301:685])
    np.testing.assert_array_equal(arrays[1], values[1200:1584, 1100:1484])
    assert receipts and client.bytes < len(raw)
    with pytest.raises(ValueError, match="incompatible"):
        read_cog(client, "fixture", len(raw), (512, 512), [(0, 0, 384)])


def test_small_strip_window_exact_and_outside_refused(tmp_path: Path) -> None:
    path = tmp_path / "strip.tif"
    values = np.arange(800 * 800, dtype="uint16").reshape(800, 800)
    with rasterio.open(
        path, "w", driver="GTiff", width=800, height=800, count=1, dtype="uint16"
    ) as dst:
        dst.write(values, 1)
    raw = path.read_bytes()
    native = NativeTiff(lambda offset, count: raw[offset : offset + count], len(raw))
    np.testing.assert_array_equal(native.window(15, 33, 384), values[33:417, 15:399])
    with pytest.raises(ValueError, match="bounded"):
        native.window(500, 500, 384)


def test_rotated_window_must_fit_frozen_envelope() -> None:
    gcps = [
        GroundControlPoint(
            row=r, col=c, x=-54 + (c + r) * 0.00008, y=50 + (c - r) * 0.00008
        )
        for r in (0, 500, 1000)
        for c in (0, 500, 1000)
    ]
    center = (-53.92, 50)
    _, _, large = locate(gcps, center)
    _, _, small = locate(gcps, center, 384)
    assert shape(large).area > shape(small).area
    assert shape(large).covers(shape(small))
    area: dict[str, Any] = {
        "geometry": small,
        "product": {"GeoFootprint": mapping(box(-55, 49, -52, 52))},
        "split": "train",
    }
    # Fails on envelope before attempting to resolve a native scene identity.
    with pytest.raises(ValueError, match="reserved development scope"):
        scope(area, large)


def decision() -> dict[str, Any]:
    return {
        "classifier_scores_viewed": False,
        "detector_outputs_viewed": False,
        "source_overlays_viewed": False,
        "independent_review": False,
        "prior_source_exposure": True,
        "annotator": "AI, not independent",
        "areas": [
            {
                "id": "a",
                "complete_visual_scan": True,
                "notes": "HH and HV inspected; speckle and one unresolved return",
                "chip_sha256": "a" * 64,
                "bands_reviewed": ["HH", "HV"],
                "cells_reviewed": [
                    "top_left",
                    "top_right",
                    "bottom_left",
                    "bottom_right",
                ],
                "objects": [
                    {
                        "id": "u1",
                        "class": "unresolved",
                        "bbox_px": [180, 180, 195, 195],
                        "reason": "compact return, class uncertain",
                    }
                ],
            }
        ],
    }


def test_first_pass_requires_full_coverage_and_no_scoring() -> None:
    original = decision()
    validate_first_pass(original, {"a"})
    for flag in (
        "classifier_scores_viewed",
        "detector_outputs_viewed",
        "source_overlays_viewed",
        "independent_review",
    ):
        changed = deepcopy(original)
        changed[flag] = True
        with pytest.raises(ValueError, match="image-first"):
            validate_first_pass(changed, {"a"})
    for field, value in (
        ("bands_reviewed", ["HH"]),
        ("cells_reviewed", ["top_left"]),
        ("complete_visual_scan", False),
        ("chip_sha256", None),
    ):
        changed = deepcopy(original)
        changed["areas"][0][field] = value
        with pytest.raises(ValueError):
            validate_first_pass(changed, {"a"})
    with pytest.raises(ValueError, match="inventory"):
        validate_first_pass(original, {"a", "b"})
    changed = deepcopy(original)
    changed["areas"][0]["objects"][0]["class"] = "vessel"
    with pytest.raises(ValueError, match="first-pass"):
        validate_first_pass(changed, {"a"})


def test_public_metadata_rejects_nested_private_ais_fields() -> None:
    public_metadata(
        {
            "areas": [
                {
                    "product_id": "public-uuid",
                    "bbox_px": [1, 2, 3, 4],
                    "native_context": {"center_incidence_deg": 34},
                }
            ]
        }
    )
    for key in (
        "mmsi",
        "lon",
        "lat",
        "source_identity",
        "geometry",
        "window",
        "reports",
    ):
        with pytest.raises(ValueError, match="cannot be published"):
            public_metadata({"areas": [{"private": {key: "fixture"}}]})


def test_published_annotations_keep_guards_and_reservation() -> None:
    report = json.loads(
        (REPO_ROOT / "datasets/source_supported/v0.1.0/collection.json").read_text()
    )
    check_collection(report)
    assert report["class_counts"] == {
        "unresolved": 7,
        "clutter": 2,
        "provisional_vessel": 2,
    }
    for flag in (
        "classifier_scored",
        "independent_review",
        "promotion_eligible",
        "training_approved",
        "annotation_completeness_verified",
        "compatibility_established",
    ):
        changed = deepcopy(report)
        changed[flag] = True
        with pytest.raises(ValueError, match="guard"):
            check_collection(changed)
    changed = deepcopy(report)
    changed["areas"][0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        check_collection(changed)
    changed = deepcopy(report)
    changed["areas"][0]["chip_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="guard"):
        check_collection(changed)


def test_local_replay_rejects_iceberg_label_on_identified_vessel() -> None:
    # ref()/checked() require artifact paths to live inside REPO_ROOT, so the
    # tampered copies must be written under the repo (in the gitignored data/
    # tree) rather than pytest's tmp_path.
    tmp_dir = REPO_ROOT / "data" / "_test_local_replay_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        collection = json.loads(
            (REPO_ROOT / "datasets/source_supported/v0.1.0/collection.json").read_text()
        )
        check_collection(collection, local=True)

        reconciliation = json.loads(
            (
                REPO_ROOT / "configs/reservations/source_reconciliation_v010.json"
            ).read_text()
        )
        evidence = json.loads(
            (
                REPO_ROOT
                / "data/source_supported/annotation_v010_01/source_support.json"
            ).read_text()
        )
        support = next(a for a in evidence["areas"] if a["id"] == "vessel_support_01")
        matches = next(o for o in support["objects"] if o["object_id"] == "u1")[
            "matches"
        ]
        compatible = [m for m in matches if m["inside_source_allowance"]]
        assert len(compatible) == 1 and compatible[0]["selected_source"] is True

        for area in reconciliation["areas"]:
            if area["id"] == "vessel_support_01":
                for obj in area["objects"]:
                    if obj["id"] == "u1":
                        obj["class"] = "plausible_iceberg"
        recon_path = tmp_dir / "reconciliation.json"
        recon_path.write_text(json.dumps(reconciliation))

        report = json.loads(
            (
                REPO_ROOT / "data/source_supported/collection_v010_01/report.json"
            ).read_text()
        )
        report["reconciliation"] = ref(recon_path)
        report_path = tmp_dir / "report.json"
        report_path.write_text(json.dumps(report))

        for area in collection["areas"]:
            if area["id"] == "vessel_support_01":
                for obj in area["objects"]:
                    if obj["id"] == "u1":
                        obj["class"] = "plausible_iceberg"
        collection["reconciliation"] = ref(recon_path)
        collection["local_report"] = ref(report_path)
        collection["class_counts"] = {
            "unresolved": 7,
            "clutter": 2,
            "provisional_vessel": 1,
            "plausible_iceberg": 1,
        }

        with pytest.raises(ValueError, match="contradicts saved source support"):
            check_collection(collection, local=True)
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()


def test_reconciliation_cannot_move_drop_or_approve_objects() -> None:
    first = ref(REPO_ROOT / "configs/reservations/source_first_pass_v010.json")
    original = json.loads(
        (REPO_ROOT / "configs/reservations/source_reconciliation_v010.json").read_text()
    )
    validate_reconciliation(first, original)
    changed = deepcopy(original)
    changed["areas"][0]["objects"][0]["bbox_px"][0] += 1
    with pytest.raises(ValueError, match="move"):
        validate_reconciliation(first, changed)
    changed = deepcopy(original)
    changed["areas"][0]["objects"].pop()
    with pytest.raises(ValueError, match="add or remove"):
        validate_reconciliation(first, changed)
    changed = deepcopy(original)
    changed["areas"][0]["objects"][0]["class"] = "plausible_iceberg"
    with pytest.raises(ValueError, match="source-consistent"):
        validate_reconciliation(first, changed)
    changed = deepcopy(original)
    changed["areas"][0]["objects"][0]["ignore_for_accuracy"] = False
    with pytest.raises(ValueError, match="guard"):
        validate_reconciliation(first, changed)
