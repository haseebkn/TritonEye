"""Executable numerical and failure contracts, independent of private imagery."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.native_nl_input import load_specification, prepare, specification
from agents.target_noise_audit import (
    LOCK,
    interpolate,
    masks,
    noise_power,
    region_stats,
    require_uncorrected,
    verify,
)


def write_noise(path: Path, overlap: bool = False) -> Path:
    root = ET.Element("noise")
    ET.SubElement(ET.SubElement(root, "adsHeader"), "polarisation").text = "HV"
    ranges = ET.SubElement(root, "noiseRangeVectorList")
    for line in (100, 110):
        vector = ET.SubElement(ranges, "noiseRangeVector")
        ET.SubElement(vector, "line").text = str(line)
        ET.SubElement(vector, "pixel").text = "200 210"
        ET.SubElement(vector, "noiseRangeLut").text = "2 4" if line == 100 else "4 6"
    azimuth = ET.SubElement(root, "noiseAzimuthVectorList")
    for _ in range(2 if overlap else 1):
        vector = ET.SubElement(azimuth, "noiseAzimuthVector")
        for name, value in {
            "firstRangeSample": "200",
            "lastRangeSample": "210",
            "firstAzimuthLine": "100",
            "lastAzimuthLine": "110",
            "line": "100 110",
            "noiseAzimuthLut": "1 2",
        }.items():
            ET.SubElement(vector, name).text = value
    ET.ElementTree(root).write(path)
    return path


def metadata() -> dict[str, Any]:
    return {
        "mission": "S1B",
        "mode": "IW",
        "product_type": "GRD",
        "band_order": ["HH", "HV"],
        "pixel_spacing_m": [10.0, 10.0],
        "thermal_noise_corrected": False,
        "unit": "sigma0_db",
        "incidence_center_deg": 34.2,
    }


def test_native_offsets_and_two_axis_noise_product(tmp_path: Path) -> None:
    source = write_noise(tmp_path / "noise.xml")
    actual = noise_power(source, [204, 104, 3, 3], "HV")
    expected = np.array(
        [
            [3.6 * 1.4, 3.8 * 1.4, 4.0 * 1.4],
            [3.8 * 1.5, 4.0 * 1.5, 4.2 * 1.5],
            [4.0 * 1.6, 4.2 * 1.6, 4.4 * 1.6],
        ]
    )
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="coverage"):
        noise_power(source, [0, 0, 3, 3], "HV")
    with pytest.raises(ValueError, match="polarization"):
        noise_power(source, [204, 104, 3, 3], "HH")
    with pytest.raises(ValueError, match="coverage"):
        noise_power(write_noise(tmp_path / "overlap.xml", True), [204, 104, 3, 3], "HV")


def test_malformed_and_legacy_lut_abstain(tmp_path: Path) -> None:
    source = write_noise(tmp_path / "noise.xml")
    root = ET.parse(source)
    root.find(".//noiseRangeLut").text = "1"  # type: ignore[union-attr]
    root.write(source)
    with pytest.raises(ValueError, match="vectors"):
        noise_power(source, [204, 104, 3, 3], "HV")
    source.write_text(
        "<noise><adsHeader><polarisation>HV</polarisation></adsHeader></noise>"
    )
    with pytest.raises(ValueError, match="schema"):
        noise_power(source, [204, 104, 3, 3], "HV")


@pytest.mark.parametrize("flag", ["true", "", "unknown"])
def test_double_subtraction_and_missing_flag_refused(flag: str) -> None:
    root = ET.ElementTree(
        ET.fromstring(
            f"<root><thermalNoiseCorrectionPerformed>{flag}</thermalNoiseCorrectionPerformed></root>"
        )
    )
    with pytest.raises(ValueError, match="double subtraction"):
        require_uncorrected(root)


def test_interpolation_precision_and_invalid_axes() -> None:
    np.testing.assert_allclose(interpolate([2], [3], [1, 2, 3]), [3, 3, 3])
    np.testing.assert_allclose(interpolate([2, 4], [3, 7], [1, 3, 5]), [1, 5, 9])
    with pytest.raises(ValueError, match="vectors"):
        interpolate([2, 2], [3, 7], [2])


def test_masks_exclude_competing_boxes_and_guard() -> None:
    target, background = masks((50, 50), [[15, 15, 20, 20], [25, 25, 28, 28]], 0, 2, 12)
    assert target.sum() == 25
    assert not np.any(target & background)
    assert not background[23:30, 23:30].any()
    assert not background[13:22, 13:22].any()
    with pytest.raises(ValueError, match="margin"):
        masks((50, 50), [[15, 15, 20, 20]], 0, 5, 5)


def test_power_statistics_distinguish_target_from_context() -> None:
    dn = np.array([[2, 4], [2, 4]], dtype="uint16")
    noise = np.full((2, 2), 5.0)
    nominal = 10 * np.log10(dn.astype("float64") ** 2)
    corrected = 10 * np.log10(np.maximum(dn.astype("float64") ** 2 - noise, 1e-5))
    row = region_stats(dn, noise, nominal, corrected, np.ones((2, 2), dtype=bool))
    assert row["pixels"] == 4
    assert row["nonpositive_residual_fraction"] == 0.5
    assert row["residual_power_q05_50_95"] == [-1.0, 5.0, 11.0]


def test_native_contract_preserves_pixels_and_returns_copy() -> None:
    values = np.arange(2 * 100 * 100, dtype="float64").reshape(2, 100, 100) / 100 - 50
    original = values.copy()
    result = prepare(values, [45, 45, 55, 55], metadata(), "train")
    assert result["status"] == "research_tensor_only"
    np.testing.assert_array_equal(
        result["tensor"], values[:, 13:88, 13:88].astype("float32")
    )
    result["tensor"][0, 0, 0] = 999
    np.testing.assert_array_equal(values, original)
    assert not result["training_approved"] and not result["promotion_eligible"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("band_order", ["HV", "HH"]),
        ("unit", "intensity"),
        ("pixel_spacing_m", [40, 40]),
        ("thermal_noise_corrected", True),
        ("mode", "EW"),
        ("mission", "unknown"),
        ("incidence_center_deg", None),
    ],
)
def test_incompatible_metadata_abstains(key: str, value: Any) -> None:
    meta = metadata()
    meta[key] = value
    result = prepare(np.zeros((2, 100, 100)), [40, 40, 50, 50], meta, "validation")
    assert result["status"] == "abstain" and result["tensor"] is None


@pytest.mark.parametrize("bbox", [[0, 0, 5, 5], [0, 0, 80, 80], [50, 50, 40, 40]])
def test_partial_context_or_oversize_abstains(bbox: list[int]) -> None:
    result = prepare(np.zeros((2, 100, 100)), bbox, metadata(), "train")
    assert result["status"] == "abstain"


def test_test_role_rejected_and_nonfinite_abstains() -> None:
    values = np.zeros((2, 100, 100))
    with pytest.raises(ValueError, match="development"):
        prepare(values, [40, 40, 50, 50], metadata(), "test")
    values[0, 0, 0] = np.nan
    assert prepare(values, [40, 40, 50, 50], metadata(), "train")["status"] == "abstain"


def test_altered_recipe_refused(tmp_path: Path) -> None:
    spec = deepcopy(specification())
    spec["normalization"] = "Kaggle means"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Unsupported"):
        load_specification(path)


def test_canonical_recipe_edits_require_new_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agents.native_nl_input as module

    spec = specification()
    spec["thermal_noise_corrected"] = True
    path = tmp_path / "mutated_canonical.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(module, "SPEC", path)
    with pytest.raises(ValueError, match="Frozen"):
        module.prepare(np.zeros((2, 100, 100)), [40, 40, 50, 50], metadata(), "train")


def test_public_release_integrity_and_provisional_scope() -> None:
    result = verify()
    assert result["verified"] and not result["training_approved"]
    from agents.source_reservation import checked

    report = json.loads(checked(json.loads(LOCK.read_text())["evidence"]).read_text())
    assert len(report["areas"]) == 4
    assert {a["split"] for a in report["areas"]} == {"train", "validation"}
    for key in (
        "training_approved",
        "promotion_eligible",
        "independent_review",
        "classifier_scored",
        "labels_modified",
    ):
        assert report[key] is False


def test_retained_data_replay_when_available() -> None:
    from agents.artifacts import REPO_ROOT

    if not (
        REPO_ROOT / "data/source_supported/snap_interpreted_v020_01/report.json"
    ).exists():
        pytest.skip("Retained native/SNAP inputs required")
    assert verify(local=True)["verified"]
