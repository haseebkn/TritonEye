"""Refresh C/D vessel opportunities without images, annotation or admission."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT
from agents.cd_acceptance import LOCK as ACCEPTANCE_LOCK
from agents.cd_acceptance import inventory, selection
from agents.cd_acceptance import verify as verify_acceptance
from agents.hh_hv_reservation import save
from agents.native_cd_input import SPEC, specification
from agents.provisional_training import (
    GUARDS,
    policy,
    public_area,
    select,
    source_checks,
)
from agents.source_followup import fresh
from agents.source_reservation import checked, ref, vessel_opportunities

DIRECTORY = REPO_ROOT / "reservations/cd_vessels/v0.2.0"
LOCK = REPO_ROOT / "configs/reservations/cd_vessel_training_v020.lock.json"


def rules() -> dict[str, Any]:
    specification()
    return policy() | {"native_specification": ref(SPEC)}


def candidates(ais: dict[str, Any]) -> list[dict[str, Any]]:
    value = rules()
    return vessel_opportunities(
        ais, json.loads(checked(value["source_policy"]).read_text())
    )


def freeze_priors(selected: Path) -> dict[str, Any]:
    priors, exposed, old, hashes = inventory()
    acquisition = selection(selected)
    hashes[ref(selected)["path"]] = ref(selected)["sha256"]
    for scene in acquisition["scenes"]:
        exposed.add(scene["acquisition_group"])
        for roi in scene["rois"]:
            priors.append(
                {
                    "geometry": roi["geometry"],
                    "group": scene["geographic_group"],
                    "split": "train",
                    "acquisition_group": scene["acquisition_group"],
                    "quarantine": False,
                }
            )
    for path in sorted((REPO_ROOT / "reservations/cd_vessels").glob("*/manifest.json")):
        original = json.loads(path.read_text())
        private_path = checked(original["private_selection"])
        hashes[ref(path)["path"]] = ref(path)["sha256"]
        hashes[ref(private_path)["path"]] = ref(private_path)["sha256"]
        for area in json.loads(private_path.read_text())["areas"]:
            exposed.add(area["native_acquisition_group"])
            old.append(area)
            priors.append(
                {
                    "geometry": area["geometry"],
                    "group": area["geographic_group"],
                    "split": area["split"],
                    "acquisition_group": area["native_acquisition_group"],
                    "quarantine": False,
                }
            )
    return {
        "priors": priors,
        "exposed": sorted(exposed),
        "old": old,
        "prior_metadata_hashes": hashes,
    }


def select_frozen(
    ais: dict[str, Any], frozen: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return select(
        candidates(ais),
        frozen["priors"],
        set(frozen["exposed"]),
        frozen["old"],
        rules(),
    )


def public(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        public_area(a)
        | {
            "status": "reserved_training_pending_native_and_object_checks",
            "processing_version_verified": False,
            "native_input_verified": False,
            "exploratory_training_admitted": False,
        }
        for a in selected
    ]


def refresh(ais_path: Path, selection_path: Path, output: Path) -> None:
    if DIRECTORY.exists() or LOCK.exists():
        raise ValueError("Fresh vessel reservation version required")
    acceptance = verify_acceptance(True)
    if acceptance["additional_processing_checks_passed"] is not True:
        raise ValueError(
            "Additional C/D checks required before actual reservation refresh"
        )
    iip_path = (
        REPO_ROOT / "data/source_supported/v010_01/corrected_audit/iip_ranked.json"
    )
    sources = source_checks(iip_path, ais_path)
    frozen = freeze_priors(selection_path)
    selected, ranked = select_frozen(sources["ais"], frozen)
    fresh(output)
    private = {
        **frozen,
        "ais": ref(ais_path),
        "iip": ref(iip_path),
        "selection": ref(selection_path),
        "native_specification": ref(SPEC),
        "areas": selected,
        "rankings": ranked,
        **GUARDS,
    }
    save(output / "selection_private.json", private)
    save(
        output / "annotation_template.json",
        {
            "provisional": True,
            "scores_viewed": False,
            "prior_project_exposure_disclosed": True,
            "native_specification": ref(SPEC),
            "areas": [
                {
                    "id": a["id"],
                    "product_id": a["product_id"],
                    "split": "train",
                    "geometry": a["geometry"],
                    "objects": [],
                    "annotation_completed_utc": None,
                    "all_visible_returns_recorded": False,
                    "source_uncertainty_covered": False,
                    "native_input_verified": False,
                }
                for a in selected
            ],
        },
    )
    manifest = {
        "version": "0.2.0",
        "status": "metadata_only_training_reservation",
        "native_specification": ref(SPEC),
        "acceptance_lock": ref(ACCEPTANCE_LOCK),
        "admission_policy": ref(
            REPO_ROOT / "configs/reservations/provisional_training_v010.json"
        ),
        "admission_policy_supports_cd": False,
        "private_selection": ref(output / "selection_private.json"),
        "annotation_template": ref(output / "annotation_template.json"),
        "source_audit": ref(ais_path),
        "archive": sources["ais"]["summary"]["archive"],
        "audit_utc": sources["ais"]["summary"]["audited_at"],
        "areas": public(selected),
        "rankings": ranked,
        "opportunities": len(ranked),
        "reserved": len(selected),
        "exclusion_counts": dict(
            Counter(r["reason"] for r in ranked if r["decision"] == "excluded")
        ),
        "annotated_objects": 0,
        "labels_admitted": 0,
        "accuracy_truth_admitted": False,
        "exploratory_training_admitted": False,
        **GUARDS,
        "limits": [
            "AIS reception is incomplete; this audit cannot establish absence "
            "of vessels",
            "C/D processing scope is IPF 004.03; each reserved product still "
            "needs native metadata/pixel checks",
            "Source timing is screened at scene midpoint; verify bracketing "
            "at native target time before admission",
            "Full competing AIS pools retained privately; missing competitors "
            "remain possible",
            "Reservation creates no training labels or export; C/D admission "
            "policy extension still required",
            "Independent review, accuracy truth, Kaggle transfer and model "
            "promotion remain unavailable",
        ],
    }
    save(DIRECTORY / "manifest.json", manifest)
    save(
        LOCK,
        {
            "manifest": ref(DIRECTORY / "manifest.json"),
            "specification": ref(SPEC),
            "implementation": ref(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {
                "opportunities": len(ranked),
                "reserved": len(selected),
                "excluded": manifest["exclusion_counts"],
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    checked(lock["implementation"])
    checked(lock["specification"])
    manifest = json.loads(checked(lock["manifest"]).read_text())
    checked(manifest["acceptance_lock"])
    checked(manifest["admission_policy"])
    verify_acceptance(local)
    if (
        manifest["native_specification"] != ref(SPEC)
        or any(manifest.get(k) != v for k, v in GUARDS.items())
        or manifest["annotated_objects"] != 0
        or manifest["labels_admitted"] != 0
        or manifest["accuracy_truth_admitted"] is not False
        or manifest["exploratory_training_admitted"] is not False
        or manifest["admission_policy_supports_cd"] is not False
        or manifest["reserved"] != len(manifest["areas"])
        or manifest["opportunities"] != len(manifest["rankings"])
    ):
        raise ValueError("Provisional reservation decision differs")
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        for path, sha in private["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        checked(manifest["annotation_template"])
        sources = source_checks(checked(private["iip"]), checked(private["ais"]))
        selected, ranked = select_frozen(sources["ais"], private)
        if (
            json.loads(json.dumps(selected)) != private["areas"]
            or ranked != private["rankings"]
            or public(selected) != manifest["areas"]
            or ranked != manifest["rankings"]
        ):
            raise ValueError("Vessel reservation source replay differs")
    return {
        "verified": True,
        "local": local,
        "opportunities": manifest["opportunities"],
        "reserved": manifest["reserved"],
        "labels_admitted": 0,
        **GUARDS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.refresh and args.selection and args.output:
        refresh(args.refresh, args.selection, args.output)
    elif args.verify:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose refresh with selection/output, or verify")
