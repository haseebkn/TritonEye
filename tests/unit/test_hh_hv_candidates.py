"""Behavioral candidate generation and sealed comparison-population contracts."""

import json
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
from agents.calibration import CalibrationError
from agents.geo import GeoreferencingError


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


def test_large_connected_return_is_retained_with_unusable_classifier_context() -> None:
    images = np.zeros((2, 201, 201), dtype="float32")
    images[0, 100, 60:141] = 30
    candidates, _ = detect(images)
    assert len(candidates) == 1
    assert candidates[0]["bbox_px"] == [60, 100, 141, 101]
    assert (
        candidates[0]["classifier_context_status"]
        == "component_exceeds_classifier_chip"
    )
    assert candidates[0]["research_retained"] is True


def test_peak_centred_classifier_window_must_contain_entire_component() -> None:
    images = np.zeros((2, 201, 201), dtype="float32")
    images[0, 100, 60:130] = 30
    images[0, 100, 60] = 31
    candidates, _ = detect(images)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["bbox_px"] == [60, 100, 130, 101]
    assert candidate["peak_pixel_xy"] == [60, 100]
    assert candidate["classifier_window_px"] == [23, 63, 75, 75]
    assert (
        candidate["classifier_context_status"] == "component_clipped_by_classifier_chip"
    )
    assert candidate["research_retained"] is True


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


@pytest.mark.parametrize(
    "failure",
    [None, "window", "raster", "calibration", "georeferencing", "runtime", "memory"],
)
def test_real_runner_contract_on_synthetic_data_and_identical_arms(
    tmp_path: Path, monkeypatch: Any, failure: str | None
) -> None:
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(runner, "available_commit", lambda: 8 * 1024**3)
    checked_scopes: list[str] = []

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
        width=4096 if failure == "raster" else 101,
        height=4096 if failure == "raster" else 101,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-53, 48, 0.0001, 0.0001),
    ) as src:
        if failure != "raster":
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
        "window": (
            [0, 0, 4096, 4096] if failure == "window" else [0, 0, 101, 101]
        ),
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

    audit_calls: list[str] = []
    fatal_types = {
        "calibration": CalibrationError,
        "georeferencing": GeoreferencingError,
        "runtime": RuntimeError,
        "memory": MemoryError,
    }
    if failure in fatal_types:
        scene["rois"].extend([{**roi, "id": "failed"}, {**roi, "id": "unreached"}])

    def audit(scene: Any, roi: Any, cache: Any) -> tuple[Any, dict[str, Any]]:
        assert (tmp_path / "data/run/definition.json").is_file()
        audit_calls.append(roi["id"])
        assert failure not in ("window", "raster")
        if roi["id"] == "failed":
            raise fatal_types[str(failure)]("fixture failure")
        return images, {"calibration_replay_max_abs_db": [0, 0]}

    monkeypatch.setattr(runner, "native_audit", audit)
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    (dataset / "release.json").write_text("{}")
    if failure == "runtime":
        checks = 0

        def runtime_check() -> None:
            nonlocal checks
            checks += 1
            if checks == 4:
                raise RuntimeError("fixture failure")

        monkeypatch.setattr(runner, "runtime_check", runtime_check)
    if failure in fatal_types:
        with pytest.raises(fatal_types[str(failure)], match="fixture failure"):
            runner.run(tmp_path / "data/run", dataset)
        bundle_path = tmp_path / "data/run/bundle.json"
        result = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert result["status"] == "incomplete"
        assert result["summary"]["processed_areas"] == 1
        assert result["summary"]["failed_areas"] == 1
        assert [area["roi_id"] for area in result["areas"]] == ["dev", "failed"]
        assert result["areas"][1]["error"] == "fixture failure"
        assert "unreached" not in audit_calls
        with pytest.raises(ValueError, match="Complete frozen"):
            runner.paired_inputs(bundle_path, sha256_file(bundle_path))
        return
    result = runner.run(tmp_path / "data/run", dataset)
    if failure in ("window", "raster"):
        assert audit_calls == []
        assert result["summary"]["failed_areas"] == 1
        assert result["status"] == "incomplete"
        assert "Bounded" in result["areas"][0]["error"]
        return
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
