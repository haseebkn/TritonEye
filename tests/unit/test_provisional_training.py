"""Exercise provisional decisions, source ranking, split protection and replay."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box, mapping

import agents.provisional_training as training
import agents.source_reservation as sources
from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save


def candidate(number: int = 1, **changes: Any) -> dict[str, Any]:
    row = {
        "stream": "iceberg_support",
        "product": {
            "Id": f"e4f3f146-305f-4dcf-bd9e-{number:012d}",
            "Name": f"S1A_IW_GRDH_1SDH_20220501T100000_20220501T100025_"
            f"{number:06d}_01ABCD_1234.SAFE",
            "Online": True,
            "GeoFootprint": mapping(box(-58, 46, -50, 54)),
        },
        "source_identity": f"2022:{number}",
        "lon": -55.5,
        "lat": 50.2,
        "reference_midpoint_utc": "2022-05-01T10:00:12Z",
        "absolute_offset_s": 40,
        "search_uncertainty_m": 340,
        "source": {"method": "VIS", "source": "SHIP"},
    }
    return {**row, **changes}


def test_training_reservation_is_unscored_and_not_an_admission() -> None:
    selected, ranked = training.select([candidate()], [], set(), [], training.policy())
    assert len(selected) == 1 and ranked[0]["decision"] == "reserved_train"
    assert selected[0]["split"] == "train"
    assert selected[0]["label"] is None
    assert selected[0]["training_approved"] is False
    assert selected[0]["ignore_for_accuracy"] is True
    assert selected[0]["imagery_inspected"] is False


@pytest.mark.parametrize("role", ["validation", "test", "quarantine"])
def test_prior_roles_cannot_be_reassigned(role: str) -> None:
    row = candidate()
    prior = {
        "geometry": sources.envelope(row["lon"], row["lat"], 3000),
        "group": "protected",
        "split": role,
        "quarantine": role == "quarantine",
    }
    selected, ranked = training.select([row], [prior], set(), [], training.policy())
    assert selected == [] and ranked[0]["decision"] == "excluded"


def test_named_geographic_group_role_preserved_even_beyond_distance_gate() -> None:
    row = candidate()
    prior = {
        "geometry": sources.envelope(-55.1, 50.8, 100),
        "group": "new_cell_-56_50",
        "split": "validation",
        "quarantine": False,
    }
    selected, ranked = training.select([row], [prior], set(), [], training.policy())
    assert selected == []
    assert "role" in ranked[0]["reason"]


def test_repeated_identity_and_native_acquisition_cannot_inflate_batch() -> None:
    row = candidate()
    other = candidate(2, source_identity=row["source_identity"])
    adjacent = candidate(3)
    adjacent["product"]["Name"] = row["product"]["Name"].replace("1234", "5678")
    selected, ranked = training.select(
        [row, other, adjacent], [], set(), [], training.policy()
    )
    assert len(selected) == 1
    assert [r["reason"] for r in ranked[1:]] == [
        "Previously reserved source identity",
        "Native-acquisition batch cap",
    ]


def test_exposure_old_reservations_missing_swath_and_online_fail_closed() -> None:
    rules = training.policy()
    row = candidate()
    group = native_acquisition_group(row["product"]["Name"])
    assert group
    for exposed, old in (
        ({group}, []),
        (set(), [{**row, "native_acquisition_group": group}]),
    ):
        selected, _ = training.select([row], [], exposed, old, rules)
        assert not selected
    row["product"]["GeoFootprint"] = mapping(box(-54, 48, -53, 49))
    assert not training.select([row], [], set(), [], rules)[0]
    row = candidate()
    row["product"]["Online"] = False
    assert not training.select([row], [], set(), [], rules)[0]


@pytest.mark.parametrize("mission", ["S1C", "S1D"])
def test_reservation_refuses_missions_outside_frozen_native_recipe(
    mission: str,
) -> None:
    row = candidate()
    row["product"]["Name"] = row["product"]["Name"].replace("S1A_", f"{mission}_")
    selected, ranked = training.select([row], [], set(), [], training.policy())
    assert selected == []
    assert ranked[0]["reason"] == "Mission outside frozen native input contract"


def test_source_ranking_prefers_tight_offsets_and_excludes_circular_sources() -> None:
    rules = training.policy()
    rows = []
    for number, offset, source in ((1, 100, "SHIP"), (2, 20, "SHIP"), (3, 1, "S1A")):
        item = candidate(number)
        rows.append(
            {
                "product": item["product"],
                "absolute_offset_s": offset,
                "search_uncertainty_m": 300 + offset,
                "reference_midpoint_utc": item["reference_midpoint_utc"],
                "observation": {
                    "timestamp": "2022-05-01T10:00:00Z",
                    "size": "MED",
                    "method": "R/V",
                    "source": source,
                    "iceberg_id": item["source_identity"],
                    "lon": -55.5,
                    "lat": 50.2,
                },
            }
        )
    pool = training.source_pool({"ranked": rows}, {"scenes": []}, rules)
    assert [p["source_identity"] for p in pool] == ["2022:2", "2022:1"]
    assert (
        training.source_pool({"ranked": list(reversed(rows))}, {"scenes": []}, rules)
        == pool
    )


@pytest.fixture
def dossier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    rules = training.policy()
    row = candidate()
    value: dict[str, Any] = {
        "policy": sources.ref(training.POLICY),
        "product_id": row["product"]["Id"],
        "name": row["product"]["Name"],
        "object_id": "u1",
        "split": "train",
        "class": "vessel",
        "provisional": True,
        "geographic_group": "new_cell_-56_50",
        "geometry": sources.envelope(row["lon"], row["lat"], 3000),
        "prior_exposure_disclosed": True,
        "native_specification": rules["native_specification"],
        "annotation_completed_utc": "2026-10-14T12:00:00Z",
        "first_score_utc": None,
        "scores_viewed": False,
    }
    monkeypatch.setattr(sources, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(training, "inventory", lambda: ([], set(), [], {}))
    save(tmp_path / "annotation.json", value)
    value = json.loads((tmp_path / "annotation.json").read_text())
    value["annotation"] = sources.ref(tmp_path / "annotation.json")
    names = rules["admission"]["required_checks"] + rules["admission"]["vessel_checks"]
    names += rules["admission"]["iceberg_checks"]
    save(
        tmp_path / "evidence.json",
        {
            "product_id": value["product_id"],
            "object_id": "u1",
            "checks": {name: True for name in names},
        },
    )
    value["checks"] = {
        name: {"passed": True, "evidence": [sources.ref(tmp_path / "evidence.json")]}
        for name in names
    }
    value["_rules"] = rules
    return value


def test_eligible_dossier_only_admits_exploratory_training(
    dossier: dict[str, Any],
) -> None:
    result = training.assess(dossier, dossier["_rules"])
    assert result["exploratory_training_admitted"] is True, result["blockers"]
    assert result["disposition"] == "provisional_exploratory_training"
    assert result["accuracy_truth_admitted"] is False
    assert result["promotion_eligible"] is False
    assert result["training_approved"] is False
    assert result["independent_review"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "validation"),
        ("split", "test"),
        ("class", "unresolved"),
        ("scores_viewed", True),
        ("first_score_utc", "2026-10-14T11:00:00Z"),
        ("first_score_utc", "malformed"),
        ("prior_exposure_disclosed", False),
        ("native_specification", {}),
        ("geographic_group", "different"),
    ],
)
def test_incomplete_or_leaking_dossier_stays_on_hold(
    dossier: dict[str, Any], field: str, value: Any
) -> None:
    dossier[field] = value
    assert (
        training.assess(dossier, dossier["_rules"])["exploratory_training_admitted"]
        is False
    )


def test_each_required_check_is_effective(dossier: dict[str, Any]) -> None:
    for name in dossier["checks"]:
        if name in dossier["_rules"]["admission"]["iceberg_checks"]:
            continue
        changed = deepcopy(dossier)
        changed["checks"][name]["passed"] = False
        assert name in training.assess(changed, changed["_rules"])["blockers"]


def test_claimed_training_role_cannot_override_validation_inventory(
    dossier: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = {
        "geometry": dossier["geometry"],
        "group": dossier["geographic_group"],
        "split": "validation",
        "quarantine": False,
        "acquisition_group": native_acquisition_group(dossier["name"]),
    }
    monkeypatch.setattr(training, "inventory", lambda: ([prior], set(), [], {}))
    result = training.assess(dossier, dossier["_rules"])
    assert result["exploratory_training_admitted"] is False


def test_release_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Fresh scoped"):
        training.publish(tmp_path / "unused_iip", tmp_path / "unused_ais", tmp_path)


def test_hash_tampering_and_wrong_subject_cannot_supply_evidence(
    dossier: dict[str, Any], tmp_path: Path
) -> None:
    name = "native_input_verified"
    save(
        tmp_path / "wrong_subject.json",
        {
            "product_id": dossier["product_id"],
            "object_id": "other",
            "checks": {name: True},
        },
    )
    dossier["checks"][name]["evidence"] = [sources.ref(tmp_path / "wrong_subject.json")]
    assert (
        f"{name}:evidence_subject_or_result_mismatch"
        in training.assess(dossier, dossier["_rules"])["blockers"]
    )
    dossier["checks"][name]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/path"):
        training.assess(dossier, dossier["_rules"])


def test_iceberg_requires_physical_corroboration(
    dossier: dict[str, Any], tmp_path: Path
) -> None:
    dossier["class"] = "iceberg"
    annotation = json.loads((tmp_path / "annotation.json").read_text())
    annotation["class"] = "iceberg"
    save(tmp_path / "iceberg_annotation.json", annotation)
    dossier["annotation"] = sources.ref(tmp_path / "iceberg_annotation.json")
    assert (
        training.assess(dossier, dossier["_rules"])["exploratory_training_admitted"]
        is True
    )
    dossier["checks"]["additional_physical_corroboration"]["passed"] = False
    result = training.assess(dossier, dossier["_rules"])
    assert result["exploratory_training_admitted"] is False
    assert "additional_physical_corroboration" in result["blockers"]


def test_public_release_and_guard_tampering() -> None:
    assert training.verify()["labels_admitted"] == 0
    manifest = json.loads((training.DIRECTORY / "manifest.json").read_text())
    rankings = json.loads((training.DIRECTORY / "rankings.json").read_text())
    manifest["areas"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        training.validate(manifest, rankings)


def test_private_replay_when_sources_available() -> None:
    if not (
        REPO_ROOT / "data/source_supported/v010_01/corrected_audit/iip_ranked.json"
    ).exists():
        pytest.skip("Requires retained private source snapshots")
    assert training.verify(local=True)["labels_admitted"] == 0
