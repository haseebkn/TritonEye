"""Behavioral checks for association diagnostics and fresh provisional selection."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping

from agents.calibration import Calibrator
from agents.source_error_audit import literal_record, precision_sensitivity
from agents.source_next_batch import POLICY, iceberg_tier, next_role, select, validate
from agents.source_preprocessing import reference_db, sparse_measurement
from agents.source_reservation import envelope


def test_reference_calibration_uses_actual_native_offsets(tmp_path: Path) -> None:
    # Nonconstant LUT: wrong origin or a swapped axis produces a measurable error.
    path = tmp_path / "calibration.xml"
    path.write_text(
        "<calibration><calibrationVectorList>"
        "<calibrationVector><line>0</line><pixel>0 200</pixel>"
        "<sigmaNought>2 4</sigmaNought></calibrationVector>"
        "<calibrationVector><line>100</line><pixel>0 200</pixel>"
        "<sigmaNought>3 5</sigmaNought></calibrationVector>"
        "</calibrationVectorList></calibration>"
    )
    dn = np.full((5, 7), 200, dtype="uint16")
    expected = 20 * np.log10(
        200 / (2 + (np.arange(40, 45)[:, None] + np.arange(80, 87)[None, :]) / 100)
    )
    np.testing.assert_allclose(
        reference_db(path, dn, [80, 40, 7, 5]), expected, atol=1e-10
    )
    np.testing.assert_allclose(
        Calibrator(str(path)).to_sigma0_db(dn, 40, 80), expected, atol=3e-6
    )
    assert np.max(np.abs(reference_db(path, dn, [0, 0, 7, 5]) - expected)) > 3


def test_reference_refuses_ragged_pixel_grids(tmp_path: Path) -> None:
    path = tmp_path / "calibration.xml"
    path.write_text(
        "<calibration><calibrationVector><line>0</line><pixel>0 10</pixel>"
        "<sigmaNought>1 2</sigmaNought></calibrationVector>"
        "<calibrationVector><line>10</line><pixel>0 11</pixel>"
        "<sigmaNought>1 2</sigmaNought></calibrationVector></calibration>"
    )
    with pytest.raises(ValueError, match="regular"):
        reference_db(path, np.ones((2, 2)), [1, 1, 2, 2])


def test_sparse_reference_preserves_exact_offsets_without_filling_outside(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurement.tiff"
    native = np.arange(100, 180, dtype="uint16").reshape(8, 10)
    sparse_measurement(path, 600, 500, [([123, 231, 10, 8], native)])
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(
            src.read(1, window=Window(123, 231, 10, 8)), native
        )
        assert not src.read(1, window=Window(0, 0, 10, 8)).any()
        assert src.block_shapes == [(1, 600)]


def test_precision_is_literal_representation_not_a_changed_gate() -> None:
    coarse = precision_sensitivity("48.30", "-53.20", (-53.21, 48.30), 50)
    fine = precision_sensitivity("48.3000", "-53.2000", (-53.21, 48.30), 50)
    assert coarse["nearest_rounding"]["center_to_corner_m"] > 600
    assert fine["nearest_rounding"]["center_to_corner_m"] < 7
    assert (
        coarse["minute_time_sensitivity_s"][
            "unknown_round_or_truncate_max_absolute_offset"
        ]
        == 110
    )
    assert "inside_source_allowance" not in coarse


def test_raw_csv_precision_and_lineage_are_checked(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text("SIGHTING_LATITUDE,SIGHTING_LONGITUDE\n48.3000,-53.20\n")
    source = {"source_file": path.name, "source_line": 2, "lat": 48.3, "lon": -53.2}
    assert literal_record(source, tmp_path)["SIGHTING_LATITUDE"] == "48.3000"
    with pytest.raises(ValueError, match="differs"):
        literal_record({**source, "lat": 49}, tmp_path)


def candidate() -> dict[str, Any]:
    return {
        "stream": "iceberg_support",
        "source_identity": "2019:1",
        "lon": -53.2,
        "lat": 48.3,
        "absolute_offset_s": 600,
        "product": {
            "Id": "e4f3f146-305f-4dcf-bd9e-98d6d3569be8",
            "Name": "S1A_IW_GRDH_1SDH_20190601T100000_20190601T100025_"
            "016000_01ABCD_1234.SAFE",
            "GeoFootprint": mapping(box(-56, 46, -52, 50)),
            "Online": True,
        },
        "source": {"size": "MED"},
    }


def test_ranking_excludes_old_tiny_satellite_and_distant_sources() -> None:
    policy = json.loads(POLICY.read_text())
    row: dict[str, Any] = {
        "absolute_offset_s": 600,
        "observation": {
            "timestamp": "2019-06-01",
            "method": "VIS",
            "source": "SHIP",
            "size": "MED",
        },
    }
    assert iceberg_tier(row, policy) == 0
    for changed in (
        {"timestamp": "2017-06-01"},
        {"size": "GR"},
        {"source": "S1A"},
        {"method": "SAT-HIGH"},
    ):
        assert (
            iceberg_tier(
                {**row, "observation": {**row["observation"], **changed}}, policy
            )
            is None
        )
    assert iceberg_tier({**row, "absolute_offset_s": 1801}, policy) is None
    assert (
        iceberg_tier(
            {
                **row,
                "absolute_offset_s": 901,
                "observation": {**row["observation"], "size": "SM"},
            },
            policy,
        )
        is None
    )


def test_selection_excludes_previous_datatakes_and_protects_locked_geography() -> None:
    policy = json.loads(POLICY.read_text())
    row = candidate()
    selected, _ = select([row], [], set(), [], policy)
    assert len(selected) == 1 and selected[0]["label"] is None
    assert selected[0]["ignore_for_accuracy"] and not selected[0]["training_approved"]
    assert (
        select([row], [], {selected[0]["native_acquisition_group"]}, [], policy)[0]
        == []
    )
    assert select([row], [], set(), selected, policy)[0] == []
    for split, quarantine in (("test", False), ("quarantine", True)):
        prior = {
            "geometry": envelope(-53.2, 48.3, 3000),
            "group": "protected",
            "split": split,
            "quarantine": quarantine,
        }
        assert select([row], [prior], set(), [], policy)[0] == []


def test_new_roles_are_deterministic_and_existing_role_is_preserved() -> None:
    geometry = envelope(-53.2, 48.3, 3000)
    assert next_role(geometry, [], "new") == next_role(geometry, [], "new")
    prior = {
        "geometry": geometry,
        "group": "fixed",
        "split": "validation",
        "quarantine": False,
        "acquisition_group": "old",
    }
    assert next_role(geometry, [prior], "new") == ("validation", "fixed")


def test_public_reservation_cannot_claim_annotations_or_training() -> None:
    row = {
        "id": "x",
        "stream": "iceberg_support",
        "split": "train",
        "native_acquisition_group": "g",
        "label": None,
        "provisional": True,
        "ignore_for_accuracy": True,
        **{
            k: False
            for k in (
                "imagery_inspected",
                "classifier_scored",
                "independent_review",
                "promotion_eligible",
                "training_approved",
            )
        },
    }
    manifest: dict[str, Any] = {
        "areas": [row],
        "annotated_objects": 0,
        "class_counts": {"iceberg_support": 1},
        "split_source_counts": {"train:iceberg_support": 1},
        "native_acquisitions": 1,
        **{
            k: False
            for k in (
                "scored",
                "independent_review",
                "promotion_eligible",
                "training_approved",
                "compatibility_established",
            )
        },
    }
    validate(manifest)
    for level, key, value in (
        ("root", "training_approved", True),
        ("root", "annotated_objects", 1),
        ("area", "classifier_scored", True),
        ("area", "label", "iceberg"),
    ):
        changed = deepcopy(manifest)
        if level == "root":
            changed[key] = value
        else:
            changed["areas"][0][key] = value
        with pytest.raises(ValueError):
            validate(changed)
