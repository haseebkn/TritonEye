"""Synthetic behavioral tests, not evidence of real-world identity accuracy."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agents.artifacts import sha256_file
from agents.association import GEOD, aligned_ais
from agents.association_study import (
    case_digest,
    compare_bundle,
    identity_metrics,
    paired_error_reduction,
    reviewed_pair_counts,
    validate_bundle,
)
from agents.nl_benchmark import load_dataset
from agents.run_versions import digest
from agents.uncertainty_association import (
    UncertaintyConfig,
    covariance,
    uncertainty_matches,
)

ACQ = "2026-08-17T21:22:09.815408Z"


def track(mmsi: str = "316000001", **fields: Any) -> dict[str, Any]:
    return {
        "mmsi": mmsi,
        "lon": -53.114,
        "lat": 48.651,
        "timestamp": ACQ,
        "speed_knots": 10.0,
        "course_deg": 90.0,
        **fields,
    }


def target(distance: float = 0, azimuth: float = 90) -> dict[str, Any]:
    lon, lat, _ = GEOD.fwd(-53.114, 48.651, azimuth, distance)
    return {"id": "t1", "lon": lon, "lat": lat}


def bundle(tmp_path: Path, *, reviewed: bool = False) -> dict[str, Any]:
    manifest, _, _ = load_dataset()
    scene = next(s for s in manifest["scenes"] if s["acquisition_time"] == ACQ)
    roi = scene["rois"][0]
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    case = {
        "id": "synthetic_case",
        "product_id": scene["product_id"],
        "name": scene["name"],
        "acquisition_group": "S1D_004171_007A29",
        "acquisition_time": ACQ,
        "split": "validation",
        "geographic_group": roi["group"],
        "region": "Newfoundland",
        "regime": "coastal",
        "polarizations": ["vv", "vh"],
        "study_roi": roi["geometry"],
        "targets": [target()],
        "ais_records": [track()],
        "sources": [
            {"kind": kind, "path": "source.json", "sha256": sha256_file(source)}
            for kind in ("sar", "ais")
        ],
        "labels": [
            {
                "target_id": "t1",
                "object_class": "vessel" if reviewed else "unresolved",
                "state": "matched" if reviewed else "unresolved",
                "mmsis": ["316000001"] if reviewed else [],
                "decision": "Synthetic fixture",
                "evidence": ["Synthetic fixture only"],
            }
        ],
    }
    case["review"] = {
        "case_sha256": case_digest(case),
        "initial_annotator": "fixture_author",
        "reviewer": "fixture_reviewer" if reviewed else None,
        "independent": reviewed,
        "complete_target_inventory": reviewed,
        "reviewed_at": ACQ if reviewed else None,
        "evidence": ["Synthetic fixture only"] if reviewed else [],
    }
    history = (
        [
            {
                "case_id": case["id"],
                "case_sha256": case_digest(case),
                "labels_sha256": digest(case["labels"]),
                "reviewer": "fixture_reviewer",
                "reviewed_at": ACQ,
            }
        ]
        if reviewed
        else []
    )
    return {
        "schema_version": "1.0.0",
        "dataset_version": "synthetic-unit-test",
        "cases": [case],
        "review_history": history,
    }


def test_aligned_motion_metadata_uses_interpolated_track() -> None:
    lon, lat, _ = GEOD.fwd(-53.114, 48.651, 90, 200)
    positions, _ = aligned_ais(
        pd.DataFrame(
            [
                track(
                    timestamp="2026-08-17T21:21:59.815408Z",
                    speed_knots=None,
                    course_deg=None,
                ),
                track(
                    timestamp="2026-08-17T21:22:19.815408Z",
                    lon=lon,
                    lat=lat,
                    speed_knots=None,
                    course_deg=None,
                ),
            ]
        ),
        ACQ,
    )
    row = positions.iloc[0]
    assert row.motion_known
    assert row.speed_m_s == pytest.approx(10)
    assert row.course_deg == pytest.approx(90, abs=0.01)


def test_covariance_accounts_for_clock_motion_and_sar_location() -> None:
    config = UncertaintyConfig()
    position = {
        "report_age_s": 0,
        "motion_known": True,
        "speed_m_s": 10,
        "course_deg": 90,
    }
    cov = covariance(position, config)
    assert cov[0, 0] == pytest.approx(100**2 + 30**2 + 10**2 * (15**2 + 5**2))
    assert cov[1, 1] == pytest.approx(100**2 + 30**2)
    older = covariance({**position, "report_age_s": 120}, config)
    assert np.linalg.eigvalsh(older - cov).min() > 0
    unknown = covariance({"report_age_s": 30, "motion_known": False}, config)
    assert unknown[0, 0] == unknown[1, 1] > cov[0, 0]


def test_directional_gate_rejects_cross_track_but_allows_along_track() -> None:
    records = pd.DataFrame([track()])
    along = uncertainty_matches([target(350)], records, ACQ)
    across = uncertainty_matches([target(350, 0)], records, ACQ)
    assert along["selected"] == {0: "316000001"}
    assert across["selected"] == {}
    assert not along["probabilities_calibrated"]


def test_unknown_motion_does_not_force_a_match() -> None:
    output = uncertainty_matches(
        [target(100)],
        pd.DataFrame([track(speed_knots=None, course_deg=None)]),
        ACQ,
        UncertaintyConfig(abstention_cost=0.01),
    )
    assert output["alternatives"] == {0: ["316000001"]}
    assert output["selected"] == {}


def test_contested_unmatched_return_is_ambiguous() -> None:
    output = uncertainty_matches(
        [target(), {**target(20), "id": "t2"}], pd.DataFrame([track()]), ACQ
    )
    assert len(output["selected"]) == 1
    assert output["ambiguous"] == [0, 1]


def test_per_target_timestamp_propagates_motion() -> None:
    later = "2026-08-17T21:23:09.815408Z"
    point = target(10 * 1852 / 3600 * 60)
    output = uncertainty_matches(
        [{**point, "timestamp": later}], pd.DataFrame([track()]), ACQ
    )
    assert output["edges"][0]["distance_m"] < 0.01
    assert output["edges"][0]["alignment_method"] == "velocity_propagated"
    with pytest.raises(ValueError, match="timestamp"):
        uncertainty_matches(
            [{**point, "timestamp": "2026-08-18T00:00:00Z"}],
            pd.DataFrame([track()]),
            ACQ,
        )


@pytest.mark.parametrize("setting", [float("nan"), -1, 0])
def test_invalid_configuration_fails(setting: float) -> None:
    with pytest.raises(ValueError):
        uncertainty_matches(
            [], pd.DataFrame(), ACQ, UncertaintyConfig(sar_sigma_m=setting)
        )


def test_stale_and_conflicting_tracks_cannot_match() -> None:
    records = pd.DataFrame(
        [
            track(timestamp="2026-08-17T20:00:00Z"),
            track("316000002"),
            track("316000002", lon=-53.115),
        ]
    )
    assert uncertainty_matches([target()], records, ACQ)["selected"] == {}


def test_provisional_bundle_emits_no_accuracy(tmp_path: Path) -> None:
    result = compare_bundle(bundle(tmp_path), artifact_root=tmp_path)
    assert result["reviewed_validation_cases"] == 0
    assert result["validation_cases"][0]["geometric"] is None
    assert result["validation_cases"][0]["descriptive_assignments"]["geometric"] == 1
    assert not result["step_6_complete"]
    assert not result["learned_ranker"]["training_readiness"]


def test_empty_telemetry_has_zero_image_coverage_not_a_measurement(tmp_path: Path) -> None:
    dataset = bundle(tmp_path)
    case = dataset["cases"][0]
    case["ais_records"] = []
    case["review"]["case_sha256"] = case_digest(case)
    result = compare_bundle(dataset, artifact_root=tmp_path)
    row = result["validation_cases"][0]
    assert row["aligned_ais_in_roi"] == 0
    assert row["geometric"] is None
    assert row["assignments"][0]["uncertainty_mmsi"] is None


def test_reviewed_fixture_measures_both_methods(tmp_path: Path) -> None:
    result = compare_bundle(bundle(tmp_path, reviewed=True), artifact_root=tmp_path)
    assert result["reviewed_validation_cases"] == 1
    for method in ("geometric", "uncertainty"):
        metrics = result["validation_cases"][0][method]
        assert metrics["association_recall"] == 1
        assert metrics["identity_errors"] == 0
    assert result["paired_error_reduction"]["interval"] is None
    assert not result["improvement_demonstrated"]


@pytest.mark.parametrize(
    "mutation",
    ["targets", "ais", "source", "reviewer", "history", "test", "metadata", "identity"],
)
def test_invalid_review_and_provenance_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    dataset = bundle(tmp_path, reviewed=True)
    case = dataset["cases"][0]
    if mutation == "targets":
        case["targets"][0]["lon"] += 0.0001
    elif mutation == "ais":
        case["ais_records"][0]["speed_knots"] = 2
    elif mutation == "source":
        (tmp_path / "source.json").write_text("changed", encoding="utf-8")
    elif mutation == "reviewer":
        case["review"]["reviewer"] = "fixture_author"
    elif mutation == "history":
        dataset["review_history"] = []
    elif mutation == "test":
        case["split"] = "test"
    elif mutation == "metadata":
        case["name"] = case["name"].replace("004171", "004172")
    else:
        case["labels"][0]["mmsis"] = ["316999999"]
    with pytest.raises(ValueError):
        validate_bundle(dataset, artifact_root=tmp_path)


def test_wrong_identity_misses_and_ambiguity_are_not_successes(tmp_path: Path) -> None:
    case = bundle(tmp_path, reviewed=True)["cases"][0]
    wrong = identity_metrics(case, {0: "316000002"}, set())
    assert wrong["wrong_tentative_identities"] == 1
    assert wrong["missed_associations"] == 1
    assert wrong["accepted_identity_precision"] == 0
    abstained = identity_metrics(case, {}, set())
    assert abstained["missed_associations"] == 1
    assert abstained["accepted_identity_precision"] is None
    ambiguous = identity_metrics(case, {0: "316000001"}, {0})
    assert ambiguous["correct_tentative_identities"] == 1
    assert ambiguous["association_recall"] == 0
    case["labels"][0]["state"] = "ambiguous"
    unsafe = identity_metrics(case, {0: "316000001"}, set())
    assert unsafe["unsafe_ambiguity_resolutions"] == 1


def test_duplicate_labels_and_partial_inventory_fail_closed(tmp_path: Path) -> None:
    dataset = bundle(tmp_path, reviewed=True)
    case = dataset["cases"][0]
    case["review"]["complete_target_inventory"] = False
    assert (
        compare_bundle(dataset, artifact_root=tmp_path)["reviewed_validation_cases"]
        == 0
    )
    case["labels"].append(deepcopy(case["labels"][0]))
    with pytest.raises(ValueError, match="every selected target"):
        validate_bundle(dataset, artifact_root=tmp_path)


def test_paired_intervals_keep_shared_geography_together() -> None:
    rows = [
        {
            "acquisition_group": str(i),
            "geographic_group": "same",
            "targets": 10,
            "geometric_errors": 3,
            "uncertainty_errors": i,
        }
        for i in range(5)
    ]
    assert paired_error_reduction(rows)["blocks"] == 1
    for i, row in enumerate(rows):
        row["geographic_group"] = str(i)
    result = paired_error_reduction(rows)
    assert result["blocks"] == 5
    assert result["interval"] is not None
    assert result == paired_error_reduction(rows)


def test_pair_readiness_counts_actual_aligned_pairs_not_silence(tmp_path: Path) -> None:
    case = bundle(tmp_path, reviewed=True)["cases"][0]
    case["ais_records"].append(track("316000002"))
    assert reviewed_pair_counts([case]) == {"positive": 1, "negative": 1, "total": 2}
    case["labels"][0]["state"] = "ambiguous"
    assert reviewed_pair_counts([case])["total"] == 0
    case["labels"][0]["state"] = "no_association"
    case["ais_records"] = []
    assert reviewed_pair_counts([case])["negative"] == 0
