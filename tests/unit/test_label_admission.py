"""Behavioral checks for the read-only, current-evidence admission consumer."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from agents import label_admission as admission
from agents.artifacts import REPO_ROOT
from agents.source_reservation import checked


def example(label: str = "plausible_iceberg") -> dict[str, Any]:
    return {
        "id": "u1",
        "class": label,
        "provisional": True,
        "ignore_for_accuracy": True,
        "independent_review": False,
        "source_consistent": True,
        "association_unique": True,
        "bbox_px": [10, 10, 20, 20],
        "reason": "Synthetic source context",
    }


def test_train_role_and_unique_source_do_not_admit_label() -> None:
    result = admission.decision({"split": "train"}, example())
    assert result["disposition"] == "source_supported_hold"
    assert not result["training_approved"]
    assert not result["exploratory_training_admitted"]
    assert not result["accuracy_truth_admitted"]
    assert result["ignore_for_accuracy"]


def test_validation_vessel_stays_reserved() -> None:
    result = admission.decision({"split": "validation"}, example("provisional_vessel"))
    assert "validation_role_cannot_train" in result["blockers"]
    assert not result["exploratory_training_admitted"]


@pytest.mark.parametrize("label", ["unresolved", "clutter", "coastal_structure"])
def test_contexts_and_unknowns_are_not_binary_negatives(label: str) -> None:
    result = admission.decision({"split": "train"}, example(label))
    assert result["disposition"] == "ignored"
    assert not result["verified_negative"]


@pytest.mark.parametrize(
    "patch",
    [
        {"split": "test"},
        {"split": "train", "source_coverage": {"source_association_abstain": True}},
    ],
)
def test_locked_or_incomplete_positive_is_refused(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        admission.decision(patch, example())


@pytest.mark.parametrize(
    "patch",
    [
        {"provisional": False},
        {"ignore_for_accuracy": False},
        {"independent_review": True},
        {"association_unique": False},
        {"source_consistent": False},
        {"class": "confirmed_iceberg"},
    ],
)
def test_annotation_guard_cannot_be_upgraded(patch: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        admission.decision({"split": "train"}, {**example(), **patch})


def test_incomplete_unresolved_return_keeps_abstention() -> None:
    result = admission.decision(
        {
            "split": "validation",
            "source_coverage": {"source_association_abstain": True},
        },
        {**example("unresolved"), "status": "boundary_or_size_abstention"},
    )
    assert "source_uncertainty_coverage_incomplete" in result["blockers"]
    assert "native75_extraction_abstained" in result["blockers"]


def test_duplicate_annotation_cannot_inflate_register() -> None:
    obj = example("unresolved")
    collection = {
        "areas": [
            {"id": "a", "product_id": "p", "split": "train", "objects": [obj, obj]}
        ],
        "classifier_scored": False,
        "independent_review": False,
        "promotion_eligible": False,
        "compatibility_established": False,
    }
    with pytest.raises(ValueError, match="Duplicate annotation"):
        admission.records(collection, {"path": "fixture.json", "sha256": "fixture"})


def test_release_consumer_replays_complete_inventory_without_private_pixels() -> None:
    result = admission.verify()
    assert result["verified"]
    assert result["summary"]["areas"] == 31
    assert result["summary"]["annotations_and_contexts"] == 58
    assert result["summary"]["interpretations"]["plausible_iceberg"] == 4
    assert result["summary"]["interpretations"]["provisional_vessel"] == 2
    assert result["summary"]["training_admitted"] == 0
    report = admission.build()
    checks = {r["dimension"]: r["decision"] for r in report["compatibility"]["checks"]}
    assert checks["nominal_calibration"] == "passes_audited_scope"
    assert checks["runtime_limits"] == "overrun_recorded"
    assert checks["thermal_noise"] == "unresolved"
    assert report["compatibility"]["approved_preprocessing_recipe"] is None
    assert report["summary"]["source_supported_roles"] == {
        "provisional_vessel:validation": 2,
        "plausible_iceberg:train": 1,
        "plausible_iceberg:validation": 3,
    }


def test_rehashed_tampering_still_fails_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = copy.deepcopy(admission.build())
    value["records"][0]["training_approved"] = True
    artifact = tmp_path / "evidence.json"
    artifact.write_text(json.dumps(value))
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"evidence": {"path": "fixture", "sha256": "fixture"}}))
    monkeypatch.setattr(admission, "LOCK", lock)
    monkeypatch.setattr(admission, "OUTPUT", REPO_ROOT / "fixture")
    original = checked
    monkeypatch.setattr(
        admission,
        "checked",
        lambda r: artifact if r["path"] == "fixture" else original(r),
    )
    with pytest.raises(ValueError, match="decisions differ"):
        admission.verify()
