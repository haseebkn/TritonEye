"""Behavioral candidate generation and sealed comparison-population contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agents import hh_hv_candidate_run as runner
from agents import hh_hv_candidates as detector
from agents.artifacts import sha256_file, write_json


def make_images() -> Any:
    return np.zeros((2, 101, 101), dtype="float32")


def detect(images: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return detector.generate(
        images, polarizations=("HH", "HV"), settings=detector.frozen_settings()[0]
    )


def test_uniform_water_has_no_candidates_and_edges_are_unevaluated() -> None:
    candidates, coverage = detect(make_images())
    assert candidates == []
    assert coverage["evaluated_pixels"] == 41**2
    assert coverage["unevaluated_pixels"] == 101**2 - 41**2
    assert coverage["coverage_fraction"] < 1


@pytest.mark.parametrize("band,contrast", [(0, 10), (1, 7)])
def test_each_native_band_can_trigger_against_exact_ring_background(
    band: int, contrast: float
) -> None:
    images = make_images()
    images[band, 50, 50] = contrast
    candidates, _ = detect(images)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["peak_pixel_xy"] == [50, 50]
    assert candidate["contrast_db_hh_hv"][band] == pytest.approx(contrast)
    assert candidate["triggered_polarizations"] == [["HH"], ["HV"]][band]
    assert candidate["operational_alert"] is False
    assert candidate["research_retained"] is True


def test_guard_excludes_neighbor_target_and_components_merge_diagonals() -> None:
    images = make_images()
    images[0, 50, 50] = 12
    images[0, 51, 51] = 10
    candidates, _ = detect(images)
    assert len(candidates) == 1
    assert candidates[0]["component_pixels"] == 2
    assert candidates[0]["bbox_px"] == [50, 50, 52, 52]
    assert candidates[0]["contrast_db_hh_hv"][0] == pytest.approx(12)


def test_db_offsets_preserve_contrast_and_inputs_remain_unchanged() -> None:
    images = make_images()
    images[0, 50, 50] = 10
    original = images.copy()
    before, _ = detect(images)
    np.testing.assert_array_equal(images, original)
    shifted, _ = detect(images + np.asarray([[-20], [-30]])[:, :, None])
    assert before[0]["contrast_db_hh_hv"] == pytest.approx(
        shifted[0]["contrast_db_hh_hv"]
    )


def test_boundaries_nodata_and_missing_classifier_context_are_explicit() -> None:
    images = make_images()
    images[0, 10, 10] = 20
    images[0, 31, 50] = 12
    candidates, _ = detect(images)
    assert len(candidates) == 1
    assert candidates[0]["classifier_context_status"] == "roi_boundary"
    images = make_images()
    images[0, 50, 50] = 12
    images[:, 50, 85] = -60
    candidates, coverage = detect(images)
    assert candidates[0]["classifier_context_status"] == "invalid_pixels"
    assert coverage["evaluated_pixels"] < 41**2
    _, missing = detect(np.full((2, 101, 101), -60))
    assert missing["evaluated_pixels"] == 0


def test_native_polarization_and_bounded_inputs_enforced() -> None:
    settings, _ = detector.frozen_settings()
    with pytest.raises(ValueError, match="HH then HV"):
        detector.generate(make_images(), polarizations=("VV", "VH"), settings=settings)
    with pytest.raises(ValueError, match="Bounded"):
        detect(np.zeros((2, 2000, 2000)))
    with pytest.raises(FrozenInstanceError):
        setattr(settings, "outer", 31)


def test_config_tampering_requires_new_version(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(detector, "REPO_ROOT", tmp_path)
    path = tmp_path / detector.CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        detector.frozen_settings()


def test_candidate_memory_preflight_refuses_pressure(monkeypatch: Any) -> None:
    monkeypatch.setattr(runner, "available_commit", lambda: 100)
    with pytest.raises(RuntimeError, match="Insufficient headroom"):
        runner.resource_check()
    with pytest.raises(RuntimeError, match="memory budget"):
        runner.runtime_check()


def test_real_runner_contract_on_synthetic_data_and_identical_arms(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(runner, "available_commit", lambda: 8 * 1024**3)
    checked_scopes = []

    def check_scope(product: str, metadata: dict[str, Any]) -> None:
        assert product == "fixture"
        assert metadata["study_roi"] == roi["geometry"]
        assert metadata["geographic_group"] == "fixture_group"
        checked_scopes.append(metadata["roi_id"])

    monkeypatch.setattr(runner, "require_development_scene", check_scope)
    images = make_images()
    images[0, 50, 50] = 12
    path = tmp_path / "data/source.tif"
    path.parent.mkdir()
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=101,
        height=101,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-53, 48, 0.0001, 0.0001),
    ) as src:
        src.write(images)
    roi = {
        "id": "dev",
        "group": "fixture_group",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[-53, 48], [-52.99, 48], [-52.99, 47.99], [-53, 47.99], [-53, 48]]
            ],
        },
        "regime": "offshore",
        "chip": {"path": "data/source.tif", "sha256": sha256_file(path)},
    }
    scene = {
        "split": "validation",
        "polarizations": ["hh", "hv"],
        "region": "Newfoundland",
        "product_id": "fixture",
        "acquisition_time": "2026-10-01T00:00:00Z",
        "rois": [roi],
    }
    manifest = {
        "scenes": [
            {
                "split": "test",
                "rois": [{"id": "locked", "chip": {"path": "NEVER_OPEN_LOCKED"}}],
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
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda directory: (
            manifest,
            {"objects": [{"bbox_px": "labels must not be used"}]},
            {},
        ),
    )

    def audit(*args: Any) -> tuple[Any, dict[str, Any]]:
        assert (tmp_path / "data/run/definition.json").is_file()
        return images, {"calibration_replay_max_abs_db": [0, 0]}

    monkeypatch.setattr(runner, "native_audit", audit)
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    (dataset / "release.json").write_text("{}")
    result = runner.run(tmp_path / "data/run", dataset)
    assert result["summary"]["candidate_count"] == 1
    assert checked_scopes == ["dev"]
    assert result["excluded_locked_test_rois"] == ["locked"]
    assert result["excluded_non_hh_hv_rois"] == ["unsupported"]
    bundle_path = tmp_path / "data/run/bundle.json"
    checksum = sha256_file(bundle_path)
    pair = runner.paired_inputs(bundle_path, checksum)
    assert pair["unfiltered"] == pair["classifier_filter"]
    feature = pair["unfiltered"][0]["features"][0]
    assert feature["geometry"]["coordinates"] == pytest.approx([-52.99495, 47.99495])
    pair["classifier_filter"][0]["features"].clear()
    assert len(pair["unfiltered"][0]["features"]) == 1
    candidate_path = tmp_path / "data/run/area_000.geojson"
    candidate_path.write_text("{}")
    with pytest.raises(ValueError, match="Candidate bytes"):
        runner.paired_inputs(bundle_path, checksum)
    result["summary"]["failed_areas"] = 1
    write_json(bundle_path, result)
    with pytest.raises(ValueError, match="Complete frozen"):
        runner.paired_inputs(bundle_path, sha256_file(bundle_path))
    with pytest.raises(ValueError, match="Fresh ignored"):
        runner.run(tmp_path / "data/run", dataset)
