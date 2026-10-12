"""Training executes on sparse positives without converting unknowns to truth."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from agents.development_evidence import verify_bias_predictions
from agents.development_experiments import (
    MemberDetector,
    feature_vector,
    fit_member_bias,
    fit_positive_head,
    freeze_member,
    prediction_summary,
    training_examples,
)
from agents.nl_benchmark import load_dataset


def test_examples_exclude_validation_test_hh_and_unresolved() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest, annotation, _ = load_dataset(root / "datasets/nl_benchmark/v0.2.0")
    examples = training_examples(manifest, annotation)
    assert [e["roi_id"] for e in examples] == ["scene_01_area_02"]
    changed = copy.deepcopy(annotation)
    for obj in changed["objects"]:
        if obj["roi_id"] == "scene_01_area_02":
            obj["class"] = "unresolved"
    with pytest.raises(ValueError, match="training"):
        training_examples(manifest, changed)


def test_positive_fit_updates_head_and_reports_real_loss() -> None:
    torch.manual_seed(3)
    head: Any = torch.nn.Linear(4, 1)
    vectors = torch.tensor([[1.0, 2, 3, 4]])
    before = head(vectors).detach().clone()
    losses = fit_positive_head(head, vectors, steps=20, learning_rate=0.01)
    assert losses[-1] < losses[0]
    assert torch.all(head(vectors) > before)
    with pytest.raises(ValueError, match="Finite"):
        fit_positive_head(head, torch.empty(0, 4))


def test_bias_training_is_actual_existing_layer_update() -> None:
    torch.manual_seed(4)
    conv = torch.nn.Conv2d(2, 1, 3, padding=1).requires_grad_(False)
    tensor = torch.randn(1, 2, 16, 16)
    original = conv(tensor).detach()
    weight_before = conv.weight.detach().clone()
    bias_before = conv.bias.detach().clone()
    conv.bias.requires_grad_(True)
    losses = fit_member_bias(conv.bias, original[0, 0, 6, 8].reshape(1))
    actual = conv(tensor).detach()
    expected = original + conv.bias.detach() - bias_before
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(conv.weight, weight_before, atol=0, rtol=0)
    assert not torch.equal(conv.bias, bias_before)
    assert losses[-1] < losses[0]


def test_freeze_real_scriptmodule_preserves_inference() -> None:
    model = torch.jit.trace(
        torch.nn.Conv2d(2, 1, 3, padding=1), torch.zeros(1, 2, 8, 8)
    )
    x = torch.randn(1, 2, 8, 8)
    before = model(x).detach()
    freeze_member(model, "cpu")
    assert not any(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(model(x), before)


def test_member_output_heads_are_mapped_by_member_contract() -> None:
    class Fixture(torch.nn.Module):  # type: ignore[misc]
        def forward(self, x: Any) -> Any:
            base = x[:, :1] * 0
            return base + 0.25, base + 0.50, base + 0.75, base + 9, x * 0

    model = torch.jit.trace(Fixture(), torch.zeros(1, 2, 2, 2))
    detector = MemberDetector(model, "cpu")
    actual = detector._forward(np.zeros((2, 2, 2), dtype="float32"))
    from agents.xview3_detector import KEY_FISHING, KEY_OBJECTNESS, KEY_VESSEL

    assert torch.all(actual[KEY_OBJECTNESS] == 0.25)
    assert torch.all(actual[KEY_VESSEL] == 0.5)
    assert torch.all(actual[KEY_FISHING] == 0.75)


def test_token_mapping_preserves_window_and_pixel_coordinates() -> None:
    tokens = np.arange(4 * 256 * 3).reshape(4, 256, 3)
    archive = {"origins": [[0, 0], [0, 128], [128, 0], [128, 128]], "tokens": tokens}
    np.testing.assert_array_equal(feature_vector(archive, 145, 154), tokens[3, 50])


def test_counts_never_emit_accuracy_and_threshold_is_strict() -> None:
    from agents.coastal_policy import annotations

    collection = {
        "features": [
            {"properties": {"confidence": 0.15, **annotations("water", 500)}},
            {"properties": {"confidence": 0.2, **annotations("water", 100)}},
        ]
    }
    trial = next(t for t in prediction_summary(collection) if t["threshold"] == 0.15)
    assert trial["returns"] == 1
    assert trial["coastal_withheld"] == 1
    assert trial["eligible_returns"] == 0
    assert trial["precision"] is trial["recall"] is None
    assert trial["measured"] is False


def test_verifier_detects_unchanged_inference_after_claimed_bias_fit() -> None:
    old = {
        "features": [
            {
                "properties": {
                    "source_row": 10.0,
                    "source_col": 20.0,
                    "confidence": 0.5,
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="trained bias"):
        verify_bias_predictions(old, old, 0.1)
    new = copy.deepcopy(old)
    new["features"][0]["properties"]["confidence"] = 1 / (1 + np.exp(-0.1))
    result = verify_bias_predictions(old, new, 0.1)
    assert result["shared_peaks"] == 1
    assert result["max_probability_error"] < 1e-10
    with pytest.raises(ValueError, match="lost"):
        verify_bias_predictions(old, {"features": []}, 0.1)
