"""Synthetic behavioral tests, not evidence of real-world identity accuracy."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box, mapping

from agents.artifacts import REPO_ROOT, sha256_file
from agents.association import GEOD, aligned_ais
from agents.association_study import (
    case_digest,
    compare_bundle,
    compare_matchers,
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


def test_empty_telemetry_has_zero_image_coverage_not_a_measurement(
    tmp_path: Path,
) -> None:
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


@pytest.mark.parametrize("identity", ["uppercase", "compact", "alternate_catalogue"])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_released_acquisition_aliases_preserve_split(
    tmp_path: Path, identity: str, split: str
) -> None:
    dataset = bundle(tmp_path)
    case = dataset["cases"][0]
    if identity == "uppercase":
        case["product_id"] = case["product_id"].upper()
    elif identity == "compact":
        case["product_id"] = case["product_id"].replace("-", "")
    else:
        case["product_id"] = "42286d3d-0000-4000-8000-000000000001"
        case["name"] = case["name"].replace("97C4.SAFE", "1234_COG.SAFE")
    case["split"] = split
    case["geographic_group"] = "synthetic_offshore_area"
    case["study_roi"] = mapping(box(-48.01, 45.99, -47.99, 46.01))
    case["targets"][0].update(lon=-48, lat=46)
    case["review"]["case_sha256"] = case_digest(case)
    if split == "train":
        with pytest.raises(ValueError, match="split differs from the released"):
            validate_bundle(dataset, artifact_root=tmp_path)
    else:
        assert validate_bundle(dataset, artifact_root=tmp_path) == [case]


@pytest.mark.parametrize("identity", ["uppercase", "compact"])
def test_canonical_product_identity_still_requires_released_metadata(
    tmp_path: Path, identity: str
) -> None:
    dataset = bundle(tmp_path)
    case = dataset["cases"][0]
    case["product_id"] = (
        case["product_id"].upper()
        if identity == "uppercase"
        else case["product_id"].replace("-", "")
    )
    case["name"] = case["name"].replace("004171", "004172")
    case["acquisition_group"] = "S1D_004172_007A29"
    case["review"]["case_sha256"] = case_digest(case)
    with pytest.raises(ValueError, match="metadata differs from the released"):
        validate_bundle(dataset, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "native_code,declared",
    [
        ("1SDV", ["hh", "hv"]),
        ("1SDH", ["vv", "vh"]),
        ("1SSV", ["vv", "vh"]),
        ("1SSH", ["hh", "hv"]),
    ],
)
def test_catalogue_alias_rejects_false_polarization(
    tmp_path: Path, native_code: str, declared: list[str]
) -> None:
    dataset = bundle(tmp_path)
    case = dataset["cases"][0]
    case["product_id"] = "42286d3d-0000-4000-8000-000000000001"
    case["name"] = case["name"].replace("_1SDV_", f"_{native_code}_")
    case["polarizations"] = declared
    case["review"]["case_sha256"] = case_digest(case)
    with pytest.raises(ValueError, match="Native polarization"):
        compare_bundle(dataset, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "native_code,declared", [("1SDV", ["vv", "vh"]), ("1SDH", ["hh", "hv"])]
)
def test_catalogue_alias_preserves_native_polarization_stratum(
    tmp_path: Path, native_code: str, declared: list[str]
) -> None:
    dataset = bundle(tmp_path)
    case = dataset["cases"][0]
    case["product_id"] = "42286d3d-0000-4000-8000-000000000001"
    case["name"] = case["name"].replace("_1SDV_", f"_{native_code}_")
    case["polarizations"] = declared
    case["review"]["case_sha256"] = case_digest(case)
    result = compare_bundle(dataset, artifact_root=tmp_path)
    assert result["polarization_case_ids"]["/".join(declared).upper()] == [case["id"]]
    assert result["reviewed_validation_cases"] == 0
    assert result["validation_cases"][0]["geometric"] is None


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


def test_report_outside_region_can_move_into_shared_scene_pool() -> None:
    point = target()
    region = box(
        point["lon"] - 0.001,
        point["lat"] - 0.001,
        point["lon"] + 0.001,
        point["lat"] + 0.001,
    )
    outside_lon, outside_lat, _ = GEOD.fwd(point["lon"], point["lat"], 270, 400)
    assert not region.covers(Point(outside_lon, outside_lat))
    record = track(
        lon=outside_lon,
        lat=outside_lat,
        timestamp="2026-08-17T21:21:09.815408Z",
        speed_knots=400 / 60 / (1852 / 3600),
    )
    result = compare_matchers(
        [point], pd.DataFrame([record]), ACQ, analysis_region=region
    )
    assert result["candidate_pool"]["mmsis"] == ["316000001"]
    assert result["geometric_selected"] == {0: "316000001"}
    assert result["experimental"]["selected"] == {0: "316000001"}
    assert result["experimental"]["edges"][0]["distance_m"] < 0.1


def test_per_target_alignment_cannot_introduce_extra_identities() -> None:
    point = target()
    region = box(
        point["lon"] - 0.001,
        point["lat"] - 0.001,
        point["lon"] + 0.001,
        point["lat"] + 0.001,
    )
    outside_lon, outside_lat, _ = GEOD.fwd(point["lon"], point["lat"], 270, 400)
    records = pd.DataFrame(
        [
            track(speed_knots=0),
            track(
                "316000002",
                lon=outside_lon,
                lat=outside_lat,
                speed_knots=400 / 60 / (1852 / 3600),
            ),
        ]
    )
    later = {**point, "timestamp": "2026-08-17T21:23:09.815408Z"}
    unrestricted = uncertainty_matches([later], records, ACQ)
    assert unrestricted["alternatives"][0] == ["316000001", "316000002"]
    result = compare_matchers([later], records, ACQ, analysis_region=region)
    assert result["candidate_pool"]["mmsis"] == ["316000001"]
    assert result["experimental"]["alternatives"] == {0: ["316000001"]}
    assert result["experimental"]["selected"] == {0: "316000001"}


def test_shared_pool_retains_per_target_motion_refinement() -> None:
    point = {**target(300), "timestamp": "2026-08-17T21:23:09.815408Z"}
    records = pd.DataFrame([track(speed_knots=5 / (1852 / 3600))])
    result = compare_matchers([point], records, ACQ)
    assert result["geometric_selected"] == {0: "316000001"}
    assert result["experimental"]["selected"] == {0: "316000001"}
    edge = result["experimental"]["edges"][0]
    assert edge["alignment_method"] == "velocity_propagated"
    assert edge["distance_m"] < 0.1


def test_explicit_empty_pool_cannot_produce_candidates() -> None:
    result = uncertainty_matches(
        [target()], pd.DataFrame([track()]), ACQ, admissible_mmsis=set()
    )
    assert result["selected"] == {}
    assert result["alternatives"] == {}
    assert result["edges"] == []
    assert result["admissible_mmsis"] == []


@pytest.mark.parametrize(
    "reviewed", [False, True], ids=["unreviewed", "synthetic-review"]
)
def test_comparison_cli_persists_report_and_rejects_stale_review(
    tmp_path: Path, reviewed: bool
) -> None:
    """Exercise the user CLI; fixture review is never real identity evidence."""
    dataset = bundle(tmp_path, reviewed=reviewed)
    case = dataset["cases"][0]
    source = "tests/fixtures/association_synthetic_source.json"
    for item in case["sources"]:
        item.update(path=source, sha256=sha256_file(REPO_ROOT / source))
    case["review"]["case_sha256"] = case_digest(case)
    if reviewed:
        dataset["review_history"][0]["case_sha256"] = case_digest(case)
    input_path = tmp_path / "synthetic-bundle.json"
    input_path.write_text(json.dumps(dataset), encoding="utf-8")
    output_path = tmp_path / "synthetic-comparison.json"
    command = [
        sys.executable,
        "-m",
        "agents.association_study",
        "compare",
        "--bundle",
        str(input_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == {
        "output": str(output_path),
        "step_6_complete": False,
    }
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dataset_version"] == "synthetic-unit-test"
    assert report["bundle_file_sha256"] == sha256_file(input_path)
    assert report["dataset_sha256"] == digest(dataset)
    assert report["runtime"]["python"]
    for name, recorded in report["code_sha256"].items():
        assert recorded == sha256_file(REPO_ROOT / "agents" / name)
    for flag in (
        "step_6_complete",
        "improvement_demonstrated",
        "production_method_changed",
        "automatic_alerts",
    ):
        assert report[flag] is False
    assert report["learned_ranker"]["implemented"] is False
    assert report["learned_ranker"]["training_readiness"] is False
    assert report["reviewed_validation_cases"] == int(reviewed)
    row = report["validation_cases"][0]
    assert row["candidate_pool"]["mmsis"] == ["316000001"]
    assert row["candidate_pool"]["count"] == 1
    assert row["descriptive_assignments"] == {"geometric": 1, "uncertainty": 1}
    for method in ("geometric", "uncertainty"):
        if reviewed:
            assert row[method]["association_recall"] == 1
            assert row[method]["identity_errors"] == 0
        else:
            assert row[method] is None
    assert report["paired_error_reduction"]["interval"] is None

    original = output_path.read_bytes()
    repeated = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "previous review/evaluation artifacts are immutable" in repeated.stderr
    assert output_path.read_bytes() == original

    case["ais_records"][0]["speed_knots"] = 2
    stale_path = tmp_path / "synthetic-stale-bundle.json"
    stale_path.write_text(json.dumps(dataset), encoding="utf-8")
    rejected_path = tmp_path / "must-not-exist.json"
    command[-3] = str(stale_path)
    command[-1] = str(rejected_path)
    rejected = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert (
        "Review refers to stale targets, telemetry or source bytes" in rejected.stderr
    )
    assert not rejected_path.exists()
