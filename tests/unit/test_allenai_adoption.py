"""Behavioral adoption QA contracts using only synthetic images and labels."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from agents import allenai_adoption as adoption
from agents import allenai_curation as cur
from agents.artifacts import sha256_file
from tests.unit.test_allenai_curation import frozen_pass as frozen_pass
from tests.unit.test_allenai_curation import prepared as prepared


def observations(path: Path, *, usable: bool = False) -> Path:
    cur.save(
        path,
        {
            "actor": "synthetic annotator",
            "method": "ai_assisted",
            "prior_label_exposure": True,
            "independent_review": False,
            "complete_point_display_inspection": True,
            "points": [
                {
                    "source_label_id": 155812,
                    "state": "usable_provisional" if usable else "unresolved_class",
                    "reason": "Synthetic observation, not real vessel truth",
                    "first_pass_region_index": 1 if usable else None,
                    "alignment_evidence": (
                        "Synthetic prior vessel region" if usable else None
                    ),
                }
            ],
        },
    )
    return path


def second_pass(frozen_pass: tuple[Path, Path], *, usable: bool = False) -> Path:
    workspace, first = frozen_pass
    packet = workspace.parent / "alignment"
    cur.reveal(workspace, first, packet)
    cur.save(
        packet / "alignment_index.json",
        {
            "first_pass_seal_sha256": sha256_file(first / "seal.json"),
            "sheets": [],
        },
    )
    authored = observations(workspace.parent / "observations.json", usable=usable)
    output = workspace.parent / "second"
    adoption.seal_review(workspace, first, packet, authored, output)
    return output


def test_uncertain_points_expand_ignores_without_new_truth(
    frozen_pass: tuple[Path, Path],
) -> None:
    workspace, first = frozen_pass
    before = sha256_file(first / "seal.json")
    second = second_pass(frozen_pass)
    result = adoption.completeness(workspace, first, second)
    assert result["point_states"]["unresolved_class"] == 1
    assert result["pixel_counts"]["ignore"] == 64
    assert result["pixel_counts"]["vessel"] == 0
    assert result["pixel_counts"]["background"] == 0
    assert not result["training_allowed"] and not result["evaluation_allowed"]
    assert result["upstream_exhaustive_annotation"] == "unverified"
    assert sha256_file(first / "seal.json") == before


def test_provisional_consistency_is_not_adoption(
    frozen_pass: tuple[Path, Path],
) -> None:
    workspace, first = frozen_pass
    result = adoption.completeness(
        workspace, first, second_pass(frozen_pass, usable=True)
    )
    assert result["point_states"]["usable_provisional"] == 1
    assert result["pixel_counts"]["vessel"] == 9
    assert result["prior_label_exposure"] is True
    assert result["independent_review"] is False
    assert result["training_allowed"] is False
    with pytest.raises(ValueError, match="quarantine"):
        cur.load_workspace(workspace, purpose="training")


@pytest.mark.parametrize("change", ["point", "state", "review", "duplicate", "mask"])
def test_rehashed_second_pass_corruption_fails_semantic_replay(
    frozen_pass: tuple[Path, Path],
    change: str,
) -> None:
    workspace, first = frozen_pass
    second = second_pass(frozen_pass, usable=True)
    path = second / "point_decisions.json"
    data = json.loads(path.read_text())
    point = data["cases"][0]["points"][0]
    if change == "point":
        point["published_lon"] += 0.01
    elif change == "state":
        point["state"] = "confirmed"
    elif change == "review":
        data["independent_review"] = True
    elif change == "duplicate":
        data["cases"][0]["points"].append(copy.deepcopy(point))
    else:
        Image.fromarray(np.zeros((8, 8), dtype="uint8")).save(
            second / "area_001_classes.png"
        )
    cur.save(path, data)
    seal = json.loads((second / "seal.json").read_text())
    seal["point_decisions"]["sha256"] = sha256_file(path)
    for item in seal["artifacts"]:
        item["sha256"] = sha256_file(second / item["path"])
    cur.save(second / "seal.json", seal)
    with pytest.raises(ValueError):
        adoption.completeness(workspace, first, second)


@pytest.mark.parametrize(
    "field,value",
    [
        ("independent_review", True),
        ("prior_label_exposure", False),
        ("actor", ""),
        ("complete_point_display_inspection", False),
    ],
)
def test_invented_review_or_missing_exposure_is_refused(
    frozen_pass: tuple[Path, Path],
    field: str,
    value: Any,
) -> None:
    workspace, first = frozen_pass
    packet = workspace.parent / "alignment"
    cur.reveal(workspace, first, packet)
    cur.save(
        packet / "alignment_index.json",
        {
            "first_pass_seal_sha256": sha256_file(first / "seal.json"),
            "sheets": [],
        },
    )
    authored = observations(workspace.parent / "observations.json")
    data = json.loads(authored.read_text())
    data[field] = value
    cur.save(authored, data)
    with pytest.raises(ValueError, match="exposure"):
        adoption.seal_review(
            workspace, first, packet, authored, workspace.parent / "reject"
        )
    assert not (workspace.parent / "reject").exists()


def test_sigmoid_input_order_and_patch_shape() -> None:
    values = np.stack([np.full((4, 4), -20.0), np.full((4, 4), -30.0)])
    result = adoption.input_diagnostic(values)
    assert result["channel_order"] == ["vh", "vv"]
    np.testing.assert_allclose(
        result["normalized_percentiles_vh_vv"][0], np.full(3, 1 / (1 + np.exp(1.8)))
    )
    assert result["normalized_percentiles_vh_vv"][1] == [0.5, 0.5, 0.5]
    assert not result["native_tile_shape_compatible"]
    assert not result["training_allowed"]
    # Clipping maps different original amplitudes to the same display value.
    raw = np.array([300.0, 900.0])
    assert np.array_equal(np.clip(raw, 0, 255), [255, 255])
    assert 20 * np.log10(raw[0] / 1000) != 20 * np.log10(raw[1] / 1000)


@pytest.mark.parametrize("values", [np.zeros((1, 4, 4)), np.full((2, 4, 4), np.nan)])
def test_invalid_radar_arrays_refused(values: Any) -> None:
    with pytest.raises(ValueError, match="two-band"):
        adoption.input_diagnostic(values)


def native_inputs(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    workspace, first = frozen_pass
    root = workspace.parent.parent
    monkeypatch.setattr(adoption, "REPO_ROOT", root)
    monkeypatch.setattr(adoption, "require_development_scene", lambda *args: None)
    descriptor, source, index = cur.load_workspace(workspace)
    case = source["cases"][0]["source_case"]
    files = []
    calibrations = []
    for pol in ("vv", "vh"):
        raster = root / f"data/measurement/test-{pol}-source.tiff"
        raster.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            raster,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(-53, 48, 0.01, 0.01),
        ) as dst:
            dst.write(np.full((4, 4), 500, dtype="uint16"), 1)
        calibration = root / f"data/calibration/test-{pol}-source.xml"
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text(
            "<root>"
            + "".join(
                f"<calibrationVector><line>{line}</line><pixel>0 3</pixel>"
                "<sigmaNought>1000 1000</sigmaNought></calibrationVector>"
                for line in (0, 3)
            )
            + "</root>"
        )
        files.extend([cur.record(root, raster), cur.record(root, calibration)])
        calibrations.append(cur.record(root, calibration))
    chip = root / "data/point/sigma0.tif"
    chip.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        chip,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-53, 48, 0.01, 0.01),
    ) as dst:
        dst.write(np.full((2, 4, 4), 20 * np.log10(0.5), dtype="float32"))
        dst.set_band_description(1, "sigma0_db_vv")
        dst.set_band_description(2, "sigma0_db_vh")
    source["sources"][0]["files"] = files
    case["alignment_checks"] = [
        {
            "source_label_id": 155812,
            "patch": {
                "geometry": case["physical_geometry"],
                "window": [0, 0, 4, 4],
                "chip": cur.record(root, chip),
                "calibration": calibrations,
            },
        }
    ]
    original_loader = cur.load_workspace

    def loader(*args: Any, **kwargs: Any) -> Any:
        original_loader(*args, **kwargs)
        return descriptor, source, index

    monkeypatch.setattr(cur, "load_workspace", loader)
    return workspace, first, chip


def test_native_calibration_replay_and_display_rejection(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)
    report = adoption.compatibility(workspace, first)
    assert report["source_files_hash_checked"] == 4
    assert report["native_patch_radiometry_replayed"] == 1
    assert report["native_calibration_patches"][0]["calibration_max_abs_error_db"] == 0
    assert report["publisher_pngs_directly_compatible"] is False
    assert report["training_allowed"] is False


@pytest.mark.parametrize("point_count", [1, 23])
def test_source_hashing_scales_with_unique_files_per_call(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    point_count: int,
) -> None:
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)
    _, source, _ = cur.load_workspace(workspace)
    case = source["cases"][0]["source_case"]
    template = case["alignment_checks"][0]
    case["alignment_checks"] = [
        {**copy.deepcopy(template), "source_label_id": i} for i in range(point_count)
    ]
    references = source["sources"][0]["files"]
    paths = {adoption.REPO_ROOT / item["path"] for item in references}
    source["sources"][0]["files"].append(copy.deepcopy(references[0]))
    hashes: dict[Path, int] = dict.fromkeys(paths, 0)

    def counted_hash(path: Any) -> str:
        resolved = Path(path)
        if resolved in hashes:
            hashes[resolved] += 1
        return sha256_file(path)

    monkeypatch.setattr(adoption, "sha256_file", counted_hash)
    for calls in (1, 2):
        report = adoption.compatibility(workspace, first)
        assert report["native_patch_radiometry_replayed"] == point_count
        assert all(
            patch["calibration_max_abs_error_db"] == 0
            for patch in report["native_calibration_patches"]
        )
        assert hashes == dict.fromkeys(paths, calls)


@pytest.mark.parametrize("change", ["source_bytes", "calibration_hash", "path"])
def test_source_cache_preserves_integrity_and_path_refusal(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)
    _, source, _ = cur.load_workspace(workspace)
    patch = source["cases"][0]["source_case"]["alignment_checks"][0]["patch"]
    if change == "source_bytes":
        item = source["sources"][0]["files"][0]
        path = adoption.REPO_ROOT / item["path"]
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "calibration_hash":
        patch["calibration"][0]["sha256"] = "0" * 64
    else:
        item = patch["calibration"][0]
        outside = adoption.REPO_ROOT / "outside-vv-calibration.xml"
        outside.write_bytes((adoption.REPO_ROOT / item["path"]).read_bytes())
        item["path"] = outside.relative_to(adoption.REPO_ROOT).as_posix()
    with pytest.raises(ValueError, match="integrity mismatch|project data artifact"):
        adoption.compatibility(workspace, first)


def test_rehashed_wrong_radiometry_rejected(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, first, chip = native_inputs(frozen_pass, monkeypatch)
    with rasterio.open(chip, "r+") as dst:
        dst.write(np.zeros((2, 4, 4), dtype="float32"))
    _, source, _ = cur.load_workspace(workspace)
    source["cases"][0]["source_case"]["alignment_checks"][0]["patch"]["chip"][
        "sha256"
    ] = sha256_file(chip)
    with pytest.raises(ValueError, match="calibration replay"):
        adoption.compatibility(workspace, first)


def test_held_out_native_context_rejected_before_source_access(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)

    def blocked(*args: Any) -> None:
        raise ValueError("Held-out native QA context")

    def forbidden(*args: Any) -> None:
        pytest.fail("source accessed before geographic guard")

    monkeypatch.setattr(adoption, "require_development_scene", blocked)
    monkeypatch.setattr(adoption, "checked_project", forbidden)
    with pytest.raises(ValueError, match="Held-out"):
        adoption.compatibility(workspace, first)


def test_aggregate_evidence_retains_quarantine_and_has_no_point_coordinates(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)
    second = second_pass(frozen_pass, usable=True)
    inputs = workspace.parent / "input_audit.json"
    cur.save(inputs, adoption.compatibility(workspace, first))
    result = adoption.evidence(workspace, first, second, inputs)
    assert result["areas"] == 1
    assert result["native_calibrated_patches_replayed"] == 1
    assert result["point_states"]["usable_provisional"] == 1
    assert result["native_model_contexts_prepared"] == 0
    assert not result["training_allowed"] and not result["promotion_allowed"]
    assert "native_calibration_patches" not in result
    assert "publisher_previews" not in result
    assert result["input_audit_sha256"] == sha256_file(inputs)
    data = json.loads(inputs.read_text())
    data["training_allowed"] = True
    cur.save(inputs, data)
    with pytest.raises(ValueError, match="quarantined input"):
        adoption.evidence(workspace, first, second, inputs)


def test_cli_input_check_and_publish_preserve_quarantine(
    frozen_pass: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the operator CLI and its persisted aggregate-output contract."""
    workspace, first, _ = native_inputs(frozen_pass, monkeypatch)
    second = second_pass(frozen_pass, usable=True)
    frozen_hash = sha256_file(first / "seal.json")
    release = workspace.parent.parent / "release.json"
    release_hash = sha256_file(release)

    def invoke(action: str, output: Path, *extra: str) -> dict[str, Any]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "allenai_adoption",
                action,
                "--workspace",
                str(workspace),
                "--first-pass",
                str(first),
                "--output",
                str(output),
                *extra,
            ],
        )
        adoption.main()
        result: dict[str, Any] = json.loads(capsys.readouterr().out)
        return result

    inputs = workspace.parent / "cli_inputs"
    diagnostic = invoke("inputs", inputs)
    assert diagnostic["native_patch_radiometry_replayed"] == 1
    assert not diagnostic["publisher_pngs_directly_compatible"]
    assert json.loads((inputs / "audit.json").read_text())["status"] == (
        "quarantined_input_diagnostic"
    )
    checked = workspace.parent / "cli_check"
    review = invoke("check", checked, "--second-pass", str(second))
    assert review["mask_pairs_replayed"] == 1
    assert review == json.loads((checked / "audit.json").read_text())
    output = adoption.REPO_ROOT / "docs/evidence/synthetic_adoption.json"
    extra = ("--second-pass", str(second), "--input-audit", str(inputs / "audit.json"))
    published = invoke("publish", output, *extra)
    assert published == json.loads(output.read_text())
    assert published["status"] == "adoption_blocked_after_provisional_qa"
    assert published["calibration_max_abs_error_db"] == 0
    assert published["native_model_contexts_prepared"] == 0
    assert all(
        published[key] is False
        for key in (
            "training_allowed",
            "evaluation_allowed",
            "threshold_selection_allowed",
            "promotion_allowed",
            "independent_review",
            "locked_test_pixels_read",
        )
    )
    assert "native_calibration_patches" not in published
    assert "publisher_previews" not in published
    published_hash = sha256_file(output)
    with pytest.raises(SystemExit) as error:
        invoke("publish", output, *extra)
    assert error.value.code == 2
    assert sha256_file(output) == published_hash
    assert sha256_file(first / "seal.json") == frozen_hash
    assert sha256_file(release) == release_hash
