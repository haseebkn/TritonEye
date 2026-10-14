"""Observable safeguards and morphology for provisional transfer preparation."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from agents.artifacts import REPO_ROOT
from agents.hh_hv_transfer import (
    check_report,
    extent,
    native_chip,
    validate_decisions,
    verified,
    verify_release,
)


def test_extent_measures_bright_component_not_class() -> None:
    values = np.full((2, 75, 75), -25.0)
    assert extent(values) is None
    values[0, 32:43, 34:41] = -10
    result = extent(values)
    assert result is not None
    assert result["width_px"] == 7 and result["height_px"] == 11
    assert not result["touches_edge"]
    assert "class" not in result


def test_far_bright_component_is_not_centered_target() -> None:
    values = np.full((2, 75, 75), -25.0)
    values[:, 12:18, 12:18] = -5
    assert extent(values) is None


def test_chip_is_native_and_boundary_abstains() -> None:
    values = np.arange(2 * 512 * 512).reshape(2, 512, 512)
    chip = native_chip(values, [240, 240, 272, 272])
    np.testing.assert_array_equal(chip, values[:, 219:294, 219:294])
    assert native_chip(values, [0, 0, 20, 20]) is None
    assert native_chip(values, [100, 100, 200, 200]) is None
    assert native_chip(values, [0, 0, 512, 512], context=True) is not None
    with pytest.raises(ValueError, match="bounds"):
        native_chip(values, [-1, 0, 20, 20])


def test_decisions_reject_exposure_missing_area_and_invalid_boxes() -> None:
    decision = {
        "classifier_scores_viewed": False,
        "detector_outputs_viewed": False,
        "independent_review": False,
        "regions": [
            {
                "id": "a",
                "notes": "both bands inspected",
                "objects": [
                    {
                        "class": "unresolved",
                        "bbox_px": [20, 20, 30, 30],
                        "reason": "ambiguous",
                    }
                ],
                "contexts": [],
            }
        ],
    }
    validate_decisions(decision, {"a"})
    for flag in (
        "classifier_scores_viewed",
        "detector_outputs_viewed",
        "independent_review",
    ):
        changed = deepcopy(decision)
        changed[flag] = True
        with pytest.raises(ValueError, match="Unscored"):
            validate_decisions(changed, {"a"})
    with pytest.raises(ValueError, match="Complete"):
        validate_decisions(decision, {"a", "b"})


def test_artifact_hash_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("tamper")
    with pytest.raises(ValueError, match="integrity"):
        verified({"path": str(path), "sha256": "0" * 64})


def test_published_collection_guards_and_decision_integrity() -> None:
    directory = REPO_ROOT / "datasets/hh_hv_transfer/v0.1.0"
    lock = json.loads(
        (REPO_ROOT / "configs/reservations/hh_hv_collection_v010.lock.json").read_text()
    )
    verify_release(directory, lock["seal_sha256"])
    original = json.loads((directory / "collection.json").read_text())
    for field in ("classifier_scored", "promotion_eligible", "independent_review"):
        report = deepcopy(original)
        report[field] = True
        with pytest.raises(ValueError, match="guard"):
            check_report(report)
    report = deepcopy(original)
    report["samples"][0]["class"] = "vessel"
    with pytest.raises(ValueError, match="decision"):
        check_report(report)
    report = deepcopy(original)
    report["samples"].pop()
    with pytest.raises(ValueError, match="inventory"):
        check_report(report)
    report = deepcopy(original)
    report["areas"][0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        check_report(report)
