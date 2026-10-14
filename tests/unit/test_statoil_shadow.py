"""Shadow inference behavior on generated unrestricted HH/HV imagery."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agents import statoil_shadow as shadow
from agents.artifacts import sha256_file
from agents.statoil_baseline import SEEDS


def test_native_crop_retains_channels_values_and_abstains() -> None:
    data = np.stack([np.full((100, 100), -20), np.full((100, 100), -30)])
    chip, info = shadow.crop(data, [45, 45, 55, 55])
    assert chip is not None
    assert chip.shape == (2, 75, 75)
    assert chip[0].mean() == -20 and chip[1].mean() == -30
    assert info["window_px"] == [13, 13, 75, 75]
    assert shadow.crop(data, [0, 0, 10, 10])[1]["status"] == "abstained_roi_boundary"
    assert (
        shadow.crop(data, [0, 0, 90, 90])[1]["status"]
        == "abstained_target_larger_than_chip"
    )
    data[0, 50, 50] = -60
    assert (
        shadow.crop(data, [45, 45, 55, 55])[1]["status"] == "abstained_invalid_pixels"
    )
    with pytest.raises(ValueError):
        shadow.crop(data, [50, 50, 40, 40])


def models() -> list[Any]:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(2 * 75 * 75, 1))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.fill_(2)
    return [model.eval().requires_grad_(False)]


def selection() -> dict[str, Any]:
    return {"band_normalization_mean": [-20, -30], "band_normalization_std": [2, 3]}


def test_shadow_preserves_return_and_weights() -> None:
    ensemble = models()
    before = shadow.parameter_hash(ensemble)
    result = shadow.predict(ensemble, np.full((2, 75, 75), -25), selection())
    assert result["hypothetical_class_at_0_5"] == "iceberg"
    assert result["return_preserved"] is True
    assert result["iceberg_sigmoid_score"] == pytest.approx(0.880797, abs=1e-6)
    assert shadow.parameter_hash(ensemble) == before
    with pytest.raises(ValueError):
        shadow.predict(
            ensemble,
            np.full((2, 75, 75), -25),
            {**selection(), "band_normalization_std": [0, 1]},
        )


def native_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scene: dict[str, Any] = {"sources": []}
    roi: dict[str, Any] = {"window": [0, 0, 100, 100], "calibration": []}
    bands = []
    for pol, dn_value in [("hh", 10), ("hv", 20)]:
        path = root / "data/raw/example.SAFE/measurement" / f"image-{pol}-001.tiff"
        path.parent.mkdir(parents=True, exist_ok=True)
        dn = np.full((100, 100), dn_value, dtype="uint16")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=1,
            dtype="uint16",
            transform=from_origin(1, 1, 0.01, 0.01),
        ) as ds:
            ds.write(dn, 1)
        scene["sources"].append(
            {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        )
        cal = (
            path.parent.parent
            / "annotation/calibration"
            / f"calibration-image-{pol}-001.xml"
        )
        cal.parent.mkdir(parents=True, exist_ok=True)
        cal.write_text(
            "<calibration><calibrationVector><line>0</line><pixel>0 100</pixel>"
            "<sigmaNought>10 10</sigmaNought></calibrationVector>"
            "<calibrationVector><line>100</line><pixel>0 100</pixel>"
            "<sigmaNought>10 10</sigmaNought></calibrationVector></calibration>"
        )
        roi["calibration"].append(
            {"path": str(cal.relative_to(root)), "sha256": sha256_file(cal)}
        )
        annotation = path.parent.parent / "annotation" / (path.stem + ".xml")
        annotation.write_text(
            f"<product><adsHeader><polarisation>{pol.upper()}</polarisation></adsHeader><rangePixelSpacing>10</rangePixelSpacing><azimuthPixelSpacing>10</azimuthPixelSpacing><incidenceAngleMidSwath>35</incidenceAngleMidSwath><thermalNoiseCorrectionPerformed>false</thermalNoiseCorrectionPerformed></product>"
        )
        bands.append(
            np.full((100, 100), 10 * np.log10(dn_value**2 / 100), dtype="float32")
        )
    chip = root / "data/chip.tif"
    with rasterio.open(
        chip,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=2,
        dtype="float32",
        transform=from_origin(1, 1, 0.01, 0.01),
    ) as ds:
        ds.write(np.stack(bands))
        ds.descriptions = ("sigma0_db_hh", "sigma0_db_hv")
    roi["chip"] = {"path": str(chip.relative_to(root)), "sha256": sha256_file(chip)}
    return scene, roi


def test_native_calibration_replay_and_input_tampering(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)
    scene, roi = native_fixture(tmp_path)
    images, audit = shadow.native_audit(scene, roi, {})
    assert images.shape == (2, 100, 100)
    assert audit["calibration_replay_max_abs_db"] == [0, 0]
    assert audit["chip_size_m_from_native_spacing"] == [750, 750]
    assert audit["compatibility"] == "provisional_transfer_only"
    (tmp_path / roi["chip"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        shadow.native_audit(scene, roi, {})


def test_vv_vh_raster_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)
    scene, roi = native_fixture(tmp_path)
    path = tmp_path / roi["chip"]["path"]
    with rasterio.open(path, "r+") as ds:
        ds.descriptions = ("sigma0_db_vv", "sigma0_db_vh")
    roi["chip"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="HH then HV"):
        shadow.native_audit(scene, roi, {})


def test_run_excludes_locked_and_unsupported_before_image_access(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(shadow, "resource_check", lambda: None)
    monkeypatch.setattr(shadow, "available_commit", lambda: 8 * 1024**3)
    ensemble = models()
    original = {
        "artifacts": {},
        "provenance": {"scope": "fixture"},
        "preprocessing": {"band_quantiles_db": [[-40, -30, -20, -10, 0]] * 2},
    }
    monkeypatch.setattr(
        shadow, "frozen_model", lambda: (ensemble, selection(), original)
    )
    scene, roi = native_fixture(tmp_path)
    scene.update(
        split="validation",
        region="Labrador",
        polarizations=["hh", "hv"],
        product_id="fixture",
        acquisition_time="2026-07-01T12:00:00Z",
    )
    roi.update(id="dev", regime="offshore")
    scene["rois"] = [roi]
    manifest = {
        "scenes": [
            {
                "split": "test",
                "rois": [{"id": "locked", "chip": {"path": "never-open-test"}}],
            },
            {
                "split": "train",
                "polarizations": ["vv", "vh"],
                "rois": [{"id": "unsupported"}],
            },
            scene,
        ],
        "licence": {"attribution": "fixture"},
    }
    annotation = {
        "objects": [
            {
                "roi_id": "dev",
                "id": "uncertain",
                "kind": "object",
                "class": "unresolved",
                "bbox_px": [45, 45, 55, 55],
            },
            {"roi_id": "dev", "kind": "area", "class": "clutter"},
        ]
    }
    monkeypatch.setattr(
        shadow,
        "load_dataset",
        lambda directory: (
            manifest,
            annotation,
            {"independent_review_available": False},
        ),
    )
    ledger = tmp_path / shadow.LEDGER
    ledger.parent.mkdir(parents=True)
    source = tmp_path / "data/model_report.json"
    source.write_text("{}")
    ledger.write_text(
        json.dumps({"full_report_path": str(source.relative_to(tmp_path))})
    )
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    (dataset / "release.json").write_text("{}")
    result = shadow.run(tmp_path / "data/shadow", dataset)
    assert result["excluded_locked_test_rois"] == ["locked"]
    assert result["excluded_non_hh_hv_rois"] == ["unsupported"]
    assert result["summary"]["scored_objects"] == 1
    assert result["summary"]["returns_suppressed"] == 0
    assert result["areas"][0]["objects"][0]["provisional_label"] == "unresolved"
    assert result["accuracy"] is None and result["promotion_eligible"] is False
    public = tmp_path / "docs/evidence/shadow.json"
    shadow.publish(tmp_path / "data/shadow/shadow.json", public)
    saved = json.loads(public.read_text())
    assert saved["full_report_sha256"] == sha256_file(
        tmp_path / "data/shadow/shadow.json"
    )
    assert saved["summary"] == result["summary"]
    with pytest.raises(ValueError, match="Fresh public"):
        shadow.publish(tmp_path / "data/shadow/shadow.json", public)
    with pytest.raises(ValueError, match="Fresh ignored"):
        shadow.run(tmp_path / "data/shadow", dataset)


def _base_provisional_report() -> dict[str, Any]:
    return {
        "scope": "Provisional real NL HH/HV frozen-classifier shadow study",
        "parameter_sha256_before": "same",
        "parameter_sha256_after": "same",
        "summary": {"returns_suppressed": 0, "independently_reviewed_objects": 0},
        "accuracy": None,
        "false_alarms_per_km2": None,
        "vessel_recall_change": None,
        "promotion_eligible": False,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(vessel_recall_change=0.1),
        lambda report: report.update(false_alarms_per_km2=1.0),
        lambda report: report["summary"].update(independently_reviewed_objects=1),
    ],
)
def test_publish_rejects_reports_that_violate_provisional_contract(
    tmp_path: Path, monkeypatch: Any, mutation: Any
) -> None:
    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)
    report = _base_provisional_report()
    mutation(report)
    source = tmp_path / "data/shadow/shadow.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Provisional frozen shadow contract"):
        shadow.publish(source, tmp_path / "docs/evidence/shadow.json")


def test_memory_guard_prevents_model_loading(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)

    def refuse() -> None:
        raise RuntimeError("memory")

    monkeypatch.setattr(shadow, "resource_check", refuse)
    with pytest.raises(RuntimeError, match="memory"):
        shadow.run(tmp_path / "data/run")


def test_frozen_loader_matches_training_architecture_and_refuses_tamper(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import torch

    from agents.statoil_baseline import fit_cnn

    monkeypatch.setattr(shadow, "REPO_ROOT", tmp_path)
    x = np.random.default_rng(7).normal(size=(10, 2, 75, 75)).astype("float32")
    y = np.asarray([i % 2 for i in range(10)])
    reference, _, state = fit_cnn(
        x, y, np.arange(8), np.asarray([8, 9]), seed=7, epochs=1, device="cpu"
    )
    root = tmp_path / "data/model"
    root.mkdir(parents=True)
    for seed in SEEDS:
        torch.save(state, root / f"cnn_{seed}.pth")
    (root / "selection.json").write_text(json.dumps(selection()))
    report = {
        "selected_by_validation": "cnn_ensemble",
        "artifacts": {p.name: sha256_file(p) for p in root.iterdir()},
    }
    path = root / "report.json"
    path.write_text(json.dumps(report))
    ledger = tmp_path / shadow.LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "full_report_path": str(path.relative_to(tmp_path)),
                "full_report_sha256": sha256_file(path),
            }
        )
    )
    ensemble, _, _ = shadow.frozen_model()
    with torch.no_grad():
        tensor = torch.from_numpy(x[8:])
        for model in ensemble:
            assert torch.equal(reference(tensor), model(tensor))
            assert not model.training
            assert all(not p.requires_grad for p in model.parameters())
    (root / f"cnn_{SEEDS[0]}.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        shadow.frozen_model()
