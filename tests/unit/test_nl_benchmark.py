"""Behavioral contracts for complete-area annotation and no-leak review gates."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import Point, box, mapping

from agents.artifacts import sha256_file, write_json
from agents.nl_benchmark import (
    DEFAULT_DATASET,
    load_dataset,
    objects_digest,
    selection_lock,
    validate_contract,
    validation_labels,
)


def fixture() -> tuple[dict[str, Any], ...]:
    selection: dict[str, Any] = {
        "dataset_version": "test",
        "separation_m": 1000,
        "scenes": [],
    }
    manifest: dict[str, Any] = {
        "dataset_version": "test",
        "separation_m": 1000,
        "scenes": [],
    }
    objects = []
    events = []
    for i, (split, lon, lat) in enumerate(
        [("train", -53.1, 48.6), ("validation", -55.1, 49.2), ("test", -59.1, 55.1)]
    ):
        name = (
            f"S1A_IW_GRDH_1SDV_2026090{i+1}T210000_2026090{i+1}T210025_"
            f"00000{i+1}_00000{i+1}_ABCD.SAFE"
        )
        pid = f"00000000-0000-4000-8000-{i+1:012d}"
        rid = f"roi_{i}"
        roi = {
            "id": rid,
            "group": f"group_{i}",
            "regime": "coastal",
            "center": [lon, lat],
            "size": 256,
        }
        scene: dict[str, Any] = {
            "name": name,
            "product_id": pid,
            "split": split,
            "region": "Labrador" if i == 2 else "Newfoundland",
            "season": "autumn",
            "prior_development_exposure": False,
            "rois": [roi],
        }
        selection["scenes"].append(copy.deepcopy(scene))
        roi.update(
            {
                "geometry": mapping(
                    box(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
                ),
                "pixels": 65536,
                "valid_pixels": 65536,
                "cells": [{"id": "0_0", "bbox_px": [0, 0, 256, 256]}],
                "chip": {"path": "chip.tif", "sha256": "1" * 64},
            }
        )
        tokens = name.split("_")
        scene.update(
            {
                "acquisition_group": "_".join([tokens[0], tokens[6], tokens[7]]),
                "acquisition_time": f"2026-09-0{i+1}T21:00:00Z",
                "acquisition_end": f"2026-09-0{i+1}T21:00:25Z",
                "polarizations": ["vv", "vh"],
                "footprint": mapping(box(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)),
                "ais": {"truth_role": "supporting_only"},
            }
        )
        manifest["scenes"].append(scene)
        obj = {
            "id": f"obj_{i}",
            "roi_id": rid,
            "product_id": pid,
            "class": "unresolved",
            "kind": "object",
            "bbox_px": [100, 100, 110, 110],
            "geometry": mapping(Point(lon, lat)),
            "decision": "Cannot distinguish vessel from clutter",
            "evidence": ["Native SAR"],
            "annotator": "first",
            "ais_role": "supporting_only",
        }
        objects.append(obj)
        events.append(
            {
                "id": f"pass_{i}",
                "kind": "first_pass",
                "roi_id": rid,
                "timestamp": "2026-10-05T23:00:00Z",
                "reviewer": "first",
                "source_chip_sha256": "1" * 64,
                "annotation_digest": objects_digest([obj]),
                "cells_reviewed": ["0_0"],
                "decision": "complete_first_pass",
                "notes": "Reviewed full co/cross bands without detector outputs",
            }
        )
    return (
        manifest,
        selection,
        selection_lock(selection),
        {"dataset_version": "test", "objects": objects},
        {"dataset_version": "test", "events": events},
    )


def test_first_pass_is_not_measurement() -> None:
    result = validate_contract(*fixture())
    assert result["areas"] == 3 and result["pixels"] == 3 * 65536
    assert all(r["first_pass_complete"] for r in result["review_status"])
    assert (
        not result["accuracy_measured"] and not result["independent_review_available"]
    )
    assert all(not r["metric_ready"] for r in result["review_status"])


def test_locked_split_mutation_rejected() -> None:
    args = fixture()
    args[1]["scenes"][0]["split"] = "test"
    with pytest.raises(ValueError, match="Split lock"):
        validate_contract(*args)


def test_reprocessed_or_adjacent_pass_cannot_cross_splits() -> None:
    args = fixture()
    original = args[1]["scenes"][0]["name"]
    replacement = original.replace("ABCD.SAFE", "FFFF_COG.SAFE")
    args[0]["scenes"][1]["name"] = args[1]["scenes"][1]["name"] = replacement
    args[0]["scenes"][1]["acquisition_group"] = args[0]["scenes"][0][
        "acquisition_group"
    ]
    args = (*args[:2], selection_lock(args[1]), *args[3:])
    with pytest.raises(ValueError, match="datatake crosses splits"):
        validate_contract(*args)


def test_geographic_groups_and_spatial_windows_do_not_cross_splits() -> None:
    args = fixture()
    args[0]["scenes"][1]["rois"][0]["group"] = "group_0"
    args[1]["scenes"][1]["rois"][0]["group"] = "group_0"
    args = (*args[:2], selection_lock(args[1]), *args[3:])
    with pytest.raises(ValueError, match="Geographic group"):
        validate_contract(*args)
    args = fixture()
    first = args[0]["scenes"][0]
    second = args[0]["scenes"][1]
    second["rois"][0]["geometry"] = first["rois"][0]["geometry"]
    second["footprint"] = first["footprint"]
    with pytest.raises(ValueError, match="Spatial leakage"):
        validate_contract(*args)


@pytest.mark.parametrize(
    "change",
    [
        "nodata",
        "missing_cell",
        "duplicate_cell",
        "bad_bbox",
        "bad_class",
        "synthetic_pol",
        "ais_truth",
    ],
)
def test_incomplete_or_misleading_content_is_rejected(change: str) -> None:
    args = fixture()
    roi = args[0]["scenes"][0]["rois"][0]
    if change == "nodata":
        roi["valid_pixels"] -= 1
    elif change == "missing_cell":
        args[4]["events"][0]["cells_reviewed"] = []
    elif change == "duplicate_cell":
        args[4]["events"][0]["cells_reviewed"] = ["0_0", "0_0"]
    elif change == "bad_bbox":
        args[3]["objects"][0]["bbox_px"] = [0, 0, 300, 20]
    elif change == "bad_class":
        args[3]["objects"][0]["class"] = "dark_vessel"
    elif change == "synthetic_pol":
        args[0]["scenes"][0]["polarizations"] = ["hh", "hv"]
    elif change == "ais_truth":
        args[3]["objects"][0]["ais_role"] = "exhaustive_truth"
    with pytest.raises(ValueError):
        validate_contract(*args)


def test_independent_review_is_bound_to_image_and_current_labels() -> None:
    args = fixture()
    obj = args[3]["objects"][1]
    obj["class"] = "vessel"
    first = args[4]["events"][1]
    first["annotation_digest"] = objects_digest([obj])
    review = {
        **first,
        "id": "second",
        "kind": "independent_review",
        "reviewer": "second",
        "independent": True,
        "decision": "accepted",
        "qualifications": "SAR analyst, self-declared",
    }
    args[4]["events"].append(review)
    assert validate_contract(*args)["review_status"][1]["metric_ready"]
    review["reviewer"] = "first"
    assert not validate_contract(*args)["review_status"][1]["metric_ready"]
    review["reviewer"] = "second"
    obj["decision"] = "Changed after review"
    assert not validate_contract(*args)["review_status"][1]["metric_ready"]
    review["source_chip_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="different image"):
        validate_contract(*args)


def test_unresolved_objects_block_metrics_even_after_review() -> None:
    args = fixture()
    first = args[4]["events"][1]
    args[4]["events"].append(
        {
            **first,
            "id": "second",
            "kind": "independent_review",
            "reviewer": "second",
            "independent": True,
            "decision": "accepted",
            "qualifications": "Declared SAR expertise",
        }
    )
    result = validate_contract(*args)["review_status"][1]
    assert result["independently_reviewed"] and not result["metric_ready"]


def test_actual_release_contract_and_annotation_mutation(tmp_path: Path) -> None:
    _, _, status = load_dataset()
    assert status["acquisitions"] == 6 and status["areas"] == 12
    assert not status["independent_review_available"]
    assert all(not r["metric_ready"] for r in status["review_status"])
    assert sum(r["test_locked"] for r in status["review_status"]) == 2
    for file in DEFAULT_DATASET.iterdir():
        if file.is_file():
            (tmp_path / file.name).write_bytes(file.read_bytes())
    annotation = json.loads((tmp_path / "annotations.json").read_text())
    annotation["objects"][0]["class"] = "vessel"
    write_json(tmp_path / "annotations.json", annotation)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_dataset(tmp_path)
    assert sha256_file(tmp_path / "annotations.json") != sha256_file(
        DEFAULT_DATASET / "annotations.json"
    )


def test_real_test_export_and_unreviewed_validation_export_blocked() -> None:
    with pytest.raises(ValueError, match="test stays locked"):
        validation_labels(DEFAULT_DATASET, "hopedale_coast")
    with pytest.raises(ValueError, match="accuracy remains unmeasured"):
        validation_labels(DEFAULT_DATASET, "bonavista_coast")


def test_previously_exposed_test_is_rejected() -> None:
    args = fixture()
    args[0]["scenes"][2]["prior_development_exposure"] = True
    args[1]["scenes"][2]["prior_development_exposure"] = True
    args = (*args[:2], selection_lock(args[1]), *args[3:])
    with pytest.raises(ValueError, match="previously exposed"):
        validate_contract(*args)
