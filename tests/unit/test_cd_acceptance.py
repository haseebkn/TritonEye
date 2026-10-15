"""C/D tensor and fail-closed processing acceptance contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents import cd_acceptance as acceptance
from agents import native_cd_input as cd
from agents.artifacts import REPO_ROOT
from agents.native_nl_input import prepare as old_prepare


def metadata() -> dict[str, Any]:
    return {
        "mission": "S1C",
        "name": "S1C_IW_GRDH_1SDH_20260715T094924_20260715T094949_"
        "008552_010EE3_C13E.SAFE",
        "mode": "IW",
        "product_type": "GRD",
        "band_order": ["HH", "HV"],
        "pixel_spacing_m": [10.0, 10.0],
        "thermal_noise_corrected": False,
        "unit": "sigma0_db",
        "processing_software": "Sentinel-1 IPF",
        "processing_version": "004.03",
        "native_start_utc": "2026-07-15T09:49:24Z",
        "native_stop_utc": "2026-07-15T09:49:49Z",
        "native_center_utc": "2026-07-15T09:49:36Z",
        "incidence_center_deg": 35.0,
    }


@pytest.mark.parametrize("mission", ["S1C", "S1D"])
def test_exact_native_crop_and_no_approval(mission: str) -> None:
    values = np.arange(2 * 100 * 100, dtype="float32").reshape(2, 100, 100)
    meta = metadata()
    meta.update(mission=mission, name=mission + meta["name"][3:])
    result = cd.prepare(values, [49, 49, 51, 51], meta, "train")
    assert result["status"] == "research_tensor_only"
    np.testing.assert_array_equal(result["tensor"], values[:, 13:88, 13:88])
    assert not np.shares_memory(values, result["tensor"])
    assert result["tensor"].dtype == np.float32
    assert all(
        result[k] is False
        for k in ("training_approved", "independent_review", "promotion_eligible")
    )
    assert old_prepare(values, [49, 49, 51, 51], meta, "train")["status"] == "abstain"


@pytest.mark.parametrize(
    "key,value",
    [
        ("mission", "S1B"),
        ("name", "S1D_IW_GRDH_1SDH_wrong.SAFE"),
        ("processing_version", "004.04"),
        ("processing_software", "unknown"),
        ("mode", "EW"),
        ("product_type", "SLC"),
        ("band_order", ["HV", "HH"]),
        ("pixel_spacing_m", [20, 20]),
        ("thermal_noise_corrected", True),
        ("unit", "linear"),
        ("native_center_utc", "2026-07-15T09:49:50Z"),
        ("native_start_utc", None),
        ("incidence_center_deg", float("nan")),
    ],
)
def test_incompatible_metadata_abstains(key: str, value: Any) -> None:
    meta = metadata() | {key: value}
    result = cd.prepare(np.zeros((2, 100, 100)), [49, 49, 51, 51], meta, "train")
    assert result["status"] == "abstain" and result["tensor"] is None


@pytest.mark.parametrize(
    "bbox", [[0, 0, 2, 2], [0, 0, 90, 90], [50, 50, 49, 49], [True, 49, 51, 51]]
)
def test_incomplete_or_invalid_crops_abstain(bbox: list[int]) -> None:
    assert (
        cd.prepare(np.zeros((2, 100, 100)), bbox, metadata(), "train")["status"]
        == "abstain"
    )


def test_locked_role_and_silent_recipe_edit_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development"):
        cd.prepare(np.zeros((2, 100, 100)), [49, 49, 51, 51], metadata(), "test")
    changed = cd.specification() | {"thermal_noise_corrected": True}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Unsupported"):
        cd.load_specification(path)
    values = np.full((2, 100, 100), np.inf)
    assert (
        cd.prepare(values, [49, 49, 51, 51], metadata(), "train")["status"] == "abstain"
    )


def test_duplicate_processor_receipts_and_conflicting_versions(tmp_path: Path) -> None:
    path = tmp_path / "manifest.safe"
    path.write_text(
        '<root><software name="Sentinel-1 IPF" version="004.03"/>'
        '<software name="Sentinel-1 IPF" version="004.03"/></root>'
    )
    assert acceptance.software(path)["version"] == "004.03"
    path.write_text(
        '<root><software name="Sentinel-1 IPF" version="004.03"/>'
        '<software name="Sentinel-1 IPF" version="004.04"/></root>'
    )
    with pytest.raises(ValueError, match="processor"):
        acceptance.software(path)


def test_native_utc_annotation_conversion_and_product_binding() -> None:
    assert (
        acceptance.annotation_utc("2026-07-15T09:49:24.876496")
        == "2026-07-15T09:49:24.876496+00:00"
    )
    meta = metadata() | {"native_start_utc": "2026-07-15T09:49:25Z"}
    result = cd.prepare(np.zeros((2, 100, 100)), [49, 49, 51, 51], meta, "train")
    assert result["reason"] == "native_product_time_mismatch"


def passing_rows() -> list[dict[str, Any]]:
    return [
        {
            "mission": m,
            "product_id": m + "_product",
            "regime": r,
            "analysis": {
                "nominal": {"passed": True},
                "geometry": {
                    "hh_hv_metadata_positions_equal": True,
                    "max_separation_m": 0.5,
                },
            },
            "variants": {
                v: {"exit_code": 0, "timed_out": False, "elapsed_limit_exceeded": False}
                for v in ("raw", "calibration")
            },
        }
        for m in ("S1C", "S1D")
        for r in ("coastal", "offshore")
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "calibration",
        "geometry",
        "timeout",
        "deadline",
        "exit",
        "band_geometry",
    ],
)
def test_acceptance_refuses_incomplete_or_failed_evidence(failure: str) -> None:
    rows = passing_rows()
    assert acceptance.decision(rows)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif failure == "calibration":
        rows[0]["analysis"]["nominal"]["passed"] = False
    elif failure == "geometry":
        rows[0]["analysis"]["geometry"]["max_separation_m"] = 25.001
    elif failure == "band_geometry":
        rows[0]["analysis"]["geometry"]["hh_hv_metadata_positions_equal"] = False
    else:
        key, value = {
            "timeout": ("timed_out", True),
            "deadline": ("elapsed_limit_exceeded", True),
            "exit": ("exit_code", 1),
        }[failure]
        rows[0]["variants"]["calibration"][key] = value
    assert not acceptance.decision(rows)


def test_released_acceptance_and_reservation_guards() -> None:
    from agents.cd_training_reservations import verify

    assert acceptance.verify()["verified"]
    result = verify()
    assert result["labels_admitted"] == 0
    assert result["training_approved"] is False


def test_local_acceptance_and_reservation_replay() -> None:
    from agents.cd_training_reservations import verify

    if not (REPO_ROOT / "data/ca_r/report.json").exists():
        pytest.skip("Local native C/D reference inputs required")
    assert verify(True)["verified"]
