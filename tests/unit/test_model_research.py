"""Research interfaces must not turn integration success into measured skill."""

import copy
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agents.croma_features import (
    build_encoder,
    load_encoder,
    normalize_window,
    parameter_digest,
)
from agents.model_research import (
    audit_labels,
    chip_windows,
    extract_features,
    readiness,
)
from agents.nl_benchmark import load_dataset


def test_normalization_preserves_band_order_and_uses_local_statistics() -> None:
    x = np.stack(
        [
            np.arange(128**2).reshape(128, 128),
            np.arange(128**2, 0, -1).reshape(128, 128),
        ]
    ).astype(np.float32)
    actual = normalize_window(x)
    expected = np.clip(
        (x - (x.mean((1, 2), keepdims=True) - 2 * x.std((1, 2), ddof=1, keepdims=True)))
        / (4 * x.std((1, 2), ddof=1, keepdims=True)),
        0,
        1,
    )
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 0, 0] < actual[0, -1, -1]
    assert actual[1, 0, 0] > actual[1, -1, -1]
    np.testing.assert_allclose(normalize_window(x * 2 + 100), actual, atol=1e-6)


@pytest.mark.parametrize(
    "value",
    [np.ones((2, 128, 128)), np.full((2, 128, 128), np.nan), np.ones((3, 128, 128))],
)
def test_bad_windows_fail_closed(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize_window(value)


def test_complete_native_window_coverage() -> None:
    values = np.random.default_rng(9).normal(size=(2, 512, 512))
    windows = chip_windows(values)
    coverage = np.zeros((512, 512), dtype=int)
    for row, col, _ in windows:
        coverage[row : row + 128, col : col + 128] += 1
    assert len(windows) == 16
    assert np.all(coverage == 1)
    with pytest.raises(ValueError):
        chip_windows(values[:, :511])


def test_current_labels_block_fit_and_keep_polarizations_separate() -> None:
    report = readiness(*load_dataset())
    assert report["step_5_complete"] is False
    for branch in report["branches"].values():
        assert branch["resolved_training_vessels"] == 0
        assert branch["supervised_fit_status"] == "blocked_labels"
        assert branch["validation_status"] == "blocked_independent_labels"
        assert all(case["split"] != "test" for case in branch["cases"])
    assert len(report["branches"]["VV/VH"]["cases"]) == 6
    assert len(report["branches"]["HH/HV"]["cases"]) == 4


def test_train_positive_does_not_unlock_validation_or_hh() -> None:
    manifest, annotation, status = copy.deepcopy(load_dataset())
    annotation["objects"].append({"class": "vessel", "roi_id": "st_johns_harbour"})
    record = next(
        r for r in status["review_status"] if r["roi_id"] == "st_johns_harbour"
    )
    record["unresolved"] = 0
    report = readiness(manifest, annotation, status)
    assert report["branches"]["VV/VH"]["supervised_fit_status"] == "labels_available"
    assert (
        report["branches"]["VV/VH"]["validation_status"] == "blocked_independent_labels"
    )
    assert report["branches"]["HH/HV"]["supervised_fit_status"] == "blocked_labels"
    assert report["step_5_complete"] is False


def test_label_geographic_audit_does_not_adopt_data(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    source.write_text(
        "detect_lon,detect_lat,GRD_product_identifier,is_vessel\n"
        "-52.7,47.56,nl,true\n-71,42,boston,true\n",
        encoding="utf-8",
    )
    result = audit_labels(source)
    assert result["labels"] == 2
    assert result["inside_nl_study_area"] == 1
    assert result["regional_products"] == ["nl"]
    assert result["adopted_for_training"] is False
    source.write_text(
        "detect_lon,detect_lat,GRD_product_identifier,is_vessel\n" "nan,47,nl,true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coordinates"):
        audit_labels(source)


def test_locked_and_hh_chips_are_not_opened_or_encoded(
    tmp_path: Path, monkeypatch
) -> None:
    import agents.model_research as research

    manifest, annotation, status = copy.deepcopy(load_dataset())
    manifest["scenes"] = [
        s
        for s in manifest["scenes"]
        if s["split"] == "test" or s["polarizations"] == ["hh", "hv"]
    ]
    monkeypatch.setattr(
        research, "load_dataset", lambda _: (manifest, annotation, status)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No unsupported or locked imagery may be read/encoded")

    monkeypatch.setattr(research.rasterio, "open", forbidden)
    monkeypatch.setattr(research, "load_encoder", forbidden)
    output = tmp_path / "never_created"
    with pytest.raises(ValueError, match="No eligible"):
        extract_features(tmp_path, tmp_path, tmp_path / "unused", output, "cpu")
    assert not output.exists()


def test_extract_features_runs_real_pipeline_on_eligible_roi(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the actual success path: windowing, encoding, freeze and output."""
    import agents.model_research as research

    manifest, annotation, status = copy.deepcopy(load_dataset())
    scene = next(
        s
        for s in manifest["scenes"]
        if s["polarizations"] == ["vv", "vh"] and s["split"] != "test"
    )
    roi = copy.deepcopy(scene["rois"][0])
    roi["size"] = 128
    roi["chip"]["path"] = "chip.tif"
    chip_path = tmp_path / "chip.tif"
    values = np.random.default_rng(3).normal(size=(2, 128, 128)).astype("float32")
    with rasterio.open(
        chip_path,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-55.2, 49.3, 0.0001, 0.0001),
    ) as out:
        out.write(values)
        out.set_band_description(1, "sigma0_db_vv")
        out.set_band_description(2, "sigma0_db_vh")
    from agents.artifacts import sha256_file

    roi["chip"]["sha256"] = sha256_file(chip_path)
    scene["rois"] = [roi]
    manifest["scenes"] = [scene]

    monkeypatch.setattr(
        research, "load_dataset", lambda _: (manifest, annotation, status)
    )
    monkeypatch.setattr(
        research,
        "load_encoder",
        lambda path, device: build_encoder(dim=32, depth=1)
        .eval()
        .requires_grad_(False),
    )

    from agents.nl_benchmark import DEFAULT_DATASET

    output = tmp_path / "experiment"
    report = extract_features(
        DEFAULT_DATASET, tmp_path, tmp_path / "unused.pt", output, "cpu"
    )

    assert report["status"] == "features_extracted_not_a_detector_comparison"
    assert report["vessel_metrics"] is None
    assert report["improvement"] is None
    [case] = report["cases"]
    assert case["status"] == "frozen_features_only"
    assert case["windows"] == 1
    assert case["repeat_max_absolute_error"] < 1e-5
    assert report["parameter_digest_before"] == report["parameter_digest_after"]

    artifact = output / f"{roi['id']}.npz"
    assert artifact.exists()
    saved = np.load(artifact)
    assert saved["tokens"].shape[0] == 1
    assert np.isfinite(saved["tokens"]).all()
    assert np.isfinite(saved["pooled"]).all()
    assert (output / "experiment.json").exists()


def test_checkpoint_checksum_rejects_wrong_weights(tmp_path: Path) -> None:
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not the public checkpoint")
    with pytest.raises(ValueError, match="pinned"):
        load_encoder(path, "cpu")


def test_radar_architecture_executes_and_freeze_is_observable() -> None:
    import torch

    model = build_encoder(dim=32, depth=1).eval().requires_grad_(False)
    values = torch.from_numpy(
        np.random.default_rng(8).random((1, 2, 128, 128)).astype("float32")
    )
    before = parameter_digest(model)
    with torch.inference_mode():
        first, second = model(values), model(values)
    assert tuple(first["SAR_encodings"].shape) == (1, 256, 32)
    assert tuple(first["SAR_GAP"].shape) == (1, 32)
    torch.testing.assert_close(
        first["SAR_encodings"], second["SAR_encodings"], rtol=0, atol=0
    )
    assert not any(p.requires_grad for p in model.parameters())
    assert parameter_digest(model) == before
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert parameter_digest(model) != before


def test_cli_persists_execution_failure_without_claiming_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    import agents.model_research as research

    monkeypatch.setattr(research, "REPO_ROOT", tmp_path)
    for filename in ("model_research.py", "croma_features.py"):
        target = tmp_path / "agents" / filename
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"fixture byte identity for failure provenance")

    def fail(*args, **kwargs):
        raise RuntimeError("fixture execution failure")

    monkeypatch.setattr(research, "extract_features", fail)
    output = tmp_path / "experiment"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_research",
            "features",
            "--weights",
            str(tmp_path / "missing.pt"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="fixture execution"):
        research.main()
    import json

    report = json.loads((output / "experiment.json").read_text())
    assert report["status"] == "failed_execution"
    assert report["reason"] == "fixture execution failure"
    assert report["training_steps"] == 0
    assert report["vessel_metrics"] is None
    assert report["improvement"] is None
    original = (output / "experiment.json").read_bytes()
    with pytest.raises(ValueError, match="preserve"):
        research.main()
    assert (output / "experiment.json").read_bytes() == original
