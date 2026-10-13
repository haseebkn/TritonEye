"""Executable classification contracts with generated, unrestricted chips."""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import statoil_baseline as study
from agents.artifacts import REPO_ROOT


def rows() -> list[dict[str, Any]]:
    return [
        {
            "id": str(i),
            "band_1": [float(i)] * 5625,
            "band_2": [float(i + 1)] * 5625,
            "is_iceberg": i % 2,
            "inc_angle": "na" if i == 0 else 30.0 + i,
        }
        for i in range(12)
    ]


def test_training_schema_and_polarization_order(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    source.write_text(json.dumps(rows()))
    ids, images, labels, angles = study.load_training(source)
    assert len(ids) == 12
    assert images.shape == (12, 2, 75, 75)
    assert images[3, 0, 0, 0] == 3
    assert images[3, 1, 0, 0] == 4
    assert labels.tolist() == [i % 2 for i in range(12)]
    assert angles[0] is None


@pytest.mark.parametrize(
    "fault",
    ["missing_label", "duplicate_id", "band_shape", "nan", "angle", "boolean_label"],
)
def test_invalid_training_data_rejected(tmp_path: Path, fault: str) -> None:
    data = rows()
    if fault == "missing_label":
        del data[0]["is_iceberg"]
    elif fault == "duplicate_id":
        data[1]["id"] = data[0]["id"]
    elif fault == "band_shape":
        data[0]["band_1"] = [1.0]
    elif fault == "nan":
        data[0]["band_1"][0] = float("nan")
    elif fault == "angle":
        data[0]["inc_angle"] = -10
    else:
        data[0]["is_iceberg"] = True
    source = tmp_path / "train.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        study.load_training(source)


def test_competition_test_refused_before_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only labelled"):
        study.load_training(tmp_path / "test.json")


def test_duplicates_transform_and_angle_groups_stay_together() -> None:
    rng = np.random.default_rng(7)
    images = rng.normal(size=(20, 2, 75, 75)).astype("float32")
    labels = np.asarray([i % 2 for i in range(20)])
    images[2] = images[0]
    images[4] = np.rot90(images[0], axes=(1, 2))
    angles: list[float | None] = [30.0 + i for i in range(20)]
    angles[6] = angles[8]
    angles[10] = angles[12] = None
    audit = study.audit_groups(images, labels, angles)
    groups = audit["groups"]
    assert groups[0] == groups[2] == groups[4]
    assert groups[6] == groups[8]
    assert groups[10] == groups[12]
    assert audit["exact_duplicate_pairs"] == [[0, 2]]
    assert len(audit["near_duplicate_pairs"]) >= 2
    split = study.grouped_split(labels, groups)
    assert split == study.grouped_split(labels, groups)
    for g in set(groups):
        assert len({s for s, group in zip(split, groups) if group == g}) == 1
    assert set(split) == {"train", "validation", "holdout"}


def test_conflicting_duplicate_labels_refused() -> None:
    image = np.arange(2 * 75 * 75, dtype="float32").reshape(2, 75, 75)
    with pytest.raises(ValueError, match="Conflicting labels"):
        study.audit_groups(np.stack([image, image]), np.asarray([0, 1]), [30, 31])


def test_normalization_uses_training_only() -> None:
    rng = np.random.default_rng(9)
    images = rng.normal(size=(8, 2, 75, 75)).astype("float32")
    before = study.train_normalization(images, np.asarray([0, 1, 2, 3]))
    images[4:] += 10000
    after = study.train_normalization(images, np.asarray([0, 1, 2, 3]))
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_linear_fit_and_observable_confusion() -> None:
    x = np.asarray([[-2], [-1], [1], [2]])
    y = np.asarray([0, 0, 1, 1])
    weights = study.fit_linear(x, y, 0.01)
    p = study.linear_predict(x, weights)
    assert study.log_loss(y, p) < study.log_loss(y, np.full(4, 0.5))
    result = study.metrics(y, p)
    assert result["confusion_rows_true_columns_predicted_ship_iceberg"] == [
        [2, 0],
        [0, 2],
    ]
    assert result["ship_recall"]["value"] == 1
    assert result["ship_recall"]["conditional_95_ci"][0] < 1
    wrong = study.metrics(y, np.asarray([0.1, 0.9, 0.1, 0.9]))
    assert wrong["icebergs_called_ships"] == 1
    assert wrong["ships_called_icebergs"] == 1


def test_proxy_group_bootstrap_is_paired_and_deterministic() -> None:
    y = np.asarray([0, 1, 0, 1])
    base = np.full(4, 0.5)
    better = np.asarray([0.1, 0.9, 0.2, 0.8])
    groups = np.asarray([0, 0, 1, 1])
    result = study.paired_bootstrap(y, base, better, groups)
    assert result == study.paired_bootstrap(y, base, better, groups)
    assert result["log_loss_difference_95_interval"][1] < 0


def test_cnn_checkpoint_uses_validation_and_updates_weights() -> None:
    import torch

    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 2, 75, 75)).astype("float32")
    y = np.asarray([i % 2 for i in range(10)])
    model, history, state = study.fit_cnn(
        x, y, np.arange(8), np.asarray([8, 9]), seed=1, epochs=2, device="cpu"
    )
    best = min(history["history"], key=lambda e: e["validation_log_loss"])
    assert history["selected_epoch"] == best["epoch"]
    with torch.no_grad():
        p = model(torch.from_numpy(x[8:])).flatten().sigmoid().numpy()
    assert study.log_loss(y[8:], p) == pytest.approx(best["validation_log_loss"])
    assert all(torch.isfinite(v).all() for v in state.values())


def test_output_and_input_guard_fail_before_loading(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="training archive"):
        study.run(tmp_path / "test.json", tmp_path / "output")
    with pytest.raises(ValueError, match="Fresh ignored"):
        study.run(
            tmp_path / "statoil-iceberg-classifier-challenge/train.json",
            tmp_path / "output",
        )


def test_synthetic_run_and_public_export(tmp_path: Path, monkeypatch: Any) -> None:
    from agents import statoil_evidence

    monkeypatch.setattr(study, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(study, "resource_check", lambda: None)
    monkeypatch.setattr(statoil_evidence, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "fixture")
    root = tmp_path / "statoil-iceberg-classifier-challenge"
    root.mkdir()
    source = root / "train.json"
    rng = np.random.default_rng(10)
    data = rows()
    for row in data:
        for band in ("band_1", "band_2"):
            row[band] = rng.normal(size=5625).tolist()
        row["id"] = "RESTRICTED_ID_" + row["id"]
    source.write_text(json.dumps(data))
    output = tmp_path / "data/run"
    result = study.run(source, output, epochs=1)
    assert sum(s["rows"] for s in result["splits"].values()) == len(data)
    assert (output / "selection.json").is_file()
    assert result["holdout"]["cnn_ensemble"]["rows"] > 0
    public = tmp_path / "docs/evidence/aggregate.json"
    evidence = statoil_evidence.export(output / "report.json", public)
    assert "RESTRICTED_ID_" not in public.read_text()
    assert "groups" not in evidence["audit"]
    assert evidence["holdout"] == result["holdout"]
    with pytest.raises(ValueError, match="Fresh public"):
        statoil_evidence.export(output / "report.json", public)
    (output / "linear.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        statoil_evidence.export(output / "report.json", public.with_name("other.json"))


def test_memory_pressure_stops_before_dataset_or_model_loading(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(study, "available_commit", lambda: 100)
    with pytest.raises(RuntimeError, match="Insufficient memory"):
        study.run(study.DEFAULT_INPUT, REPO_ROOT / "data/nonexistent_fixture_run")
