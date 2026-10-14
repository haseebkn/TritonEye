"""Replay a read-only admission register and compatibility decisions from evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.source_reservation import checked, ref

POLICY = REPO_ROOT / "configs/reservations/label_admission_v010.json"
OUTPUT = REPO_ROOT / "docs/evidence/label_admission_v010.json"
LOCK = REPO_ROOT / "configs/reservations/label_admission_v010.lock.json"
CLASSES = {
    "unresolved",
    "clutter",
    "coastal_structure",
    "provisional_vessel",
    "plausible_iceberg",
}
GUARDS = {
    "provisional": True,
    "ignore_for_accuracy": True,
    "independent_review": False,
    "training_approved": False,
    "promotion_eligible": False,
    "compatibility_established": False,
}


def decision(area: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    """Current-policy hold; source consistency and train split confer no approval."""
    if area["split"] not in {"train", "validation"}:
        raise ValueError("Only development roles are permitted")
    if (
        obj["class"] not in CLASSES
        or obj.get("provisional") is not True
        or obj.get("ignore_for_accuracy") is not True
        or obj.get("independent_review") is not False
    ):
        raise ValueError("Original provisional annotation guard required")
    positive = obj["class"] in {"provisional_vessel", "plausible_iceberg"}
    coverage = area.get("source_coverage", {})
    if positive and (
        obj.get("source_consistent") is not True
        or obj.get("association_unique") is not True
        or coverage.get("source_association_abstain") is True
    ):
        raise ValueError("Source-supported interpretation contradicts evidence")
    reasons = ["no_independent_review", "input_compatibility_unresolved"]
    if area["split"] != "train":
        reasons.append("validation_role_cannot_train")
    if positive:
        reasons.append("physical_identity_unconfirmed")
    else:
        reasons.append("class_unresolved_or_context_not_verified_negative")
    if coverage.get("source_association_abstain") is True:
        reasons.append("source_uncertainty_coverage_incomplete")
    if obj.get("status") == "boundary_or_size_abstention":
        reasons.append("native75_extraction_abstained")
    return {
        "disposition": "source_supported_hold" if positive else "ignored",
        "blockers": reasons,
        "exploratory_training_admitted": False,
        "accuracy_truth_admitted": False,
        "verified_negative": False,
        **GUARDS,
    }


def records(collection: dict[str, Any], source: dict[str, str]) -> list[dict[str, Any]]:
    for key in (
        "classifier_scored",
        "independent_review",
        "promotion_eligible",
        "compatibility_established",
    ):
        if collection.get(key) is not False:
            raise ValueError("Unscored provisional source collection required")
    areas = collection["areas"]
    area_map = {a["id"]: a for a in areas}
    if len(area_map) != len(areas):
        raise ValueError("Duplicate area identity")
    pairs = (
        [(area_map[o["area_id"]], o) for o in collection["samples"]]
        if "samples" in collection
        else [(a, o) for a in areas for o in a["objects"]]
    )
    rows = []
    seen = set()
    for area, obj in pairs:
        identity = f"{source['path']}::{area['id']}::{obj['id']}"
        if identity in seen:
            raise ValueError("Duplicate annotation identity")
        seen.add(identity)
        rows.append(
            {
                "id": identity,
                "collection": source,
                "area_id": area["id"],
                "object_id": obj["id"],
                "product_id": area["product_id"],
                "split": area["split"],
                "geographic_group": area.get("geographic_group", area.get("group")),
                "native_acquisition_group": area.get("native_acquisition_group"),
                "interpretation": obj["class"],
                "kind": obj.get("kind", "annotation"),
                "bbox_px": obj["bbox_px"],
                "annotation_reason": obj["reason"],
                "source_consistent": obj.get("source_consistent"),
                "association_unique_in_available_pool": obj.get("association_unique"),
                "source_coverage": area.get("source_coverage"),
                "extraction_status": obj.get("status", "see_hash_bound_local_report"),
                **decision(area, obj),
            }
        )
    return rows


def compatibility(
    snap: dict[str, Any], noise: dict[str, Any], reserved: dict[str, Any]
) -> dict[str, Any]:
    if (
        snap.get("compatibility_established") is not False
        or noise.get("compatibility_established") is not False
        or len(snap["areas"]) != 4
        or {a["id"] for a in snap["areas"]} != {a["id"] for a in noise["areas"]}
    ):
        raise ValueError("Complete provisional SNAP/noise evidence required")
    checks = [
        {
            "dimension": "polarization_and_layout",
            "decision": "documented",
            "finding": "Native calibrated HH then HV; 75x75 context extraction "
            "does not prove physical equivalence.",
        },
        {
            "dimension": "nominal_calibration",
            "decision": (
                "passes_audited_scope"
                if snap["calibration_all_passed"]
                else "unresolved"
            ),
            "finding": "SNAP numerical comparison covers four acquisitions/eight "
            "bands only; other areas have pipeline replays.",
        },
        {
            "dimension": "physical_scale_and_resolution",
            "decision": "unresolved",
            "finding": "Native IW GRDH 10m sampling is documented; Kaggle physical "
            "spacing, resolution and target scale equivalence are unknown.",
        },
        {
            "dimension": "thermal_noise",
            "decision": "unresolved",
            "finding": "Substantial HV changes and inferred power pile-up. Keep "
            "the original uncorrected inputs; select no correction recipe from "
            "this diagnostic.",
        },
        {
            "dimension": "incidence_mode_and_conditions",
            "decision": "unresolved",
            "finding": "Native mode/incidence metadata exists; incidence overlap "
            "alone cannot match sensor geometry, sea state or effective resolution.",
        },
        {
            "dimension": "values_and_apparent_extent",
            "decision": "descriptive_only",
            "finding": "Training-only Kaggle summaries show distribution/extent "
            "differences; no histogram matching, resampling or compatibility "
            "threshold selected.",
        },
        {
            "dimension": "normalization",
            "decision": "frozen_shadow_only",
            "finding": "Retain original Kaggle training normalization for "
            "reproducible shadow inference. It is not proof of native compatibility.",
        },
        {
            "dimension": "runtime_limits",
            "decision": (
                "passes" if snap["bounded_execution_all_passed"] else "overrun_recorded"
            ),
            "finding": "Execution-limit compliance is separate from numerical "
            "calibration agreement.",
        },
    ]
    return {
        "checks": checks,
        "kaggle_training_reference": reserved["kaggle_training_reference"],
        "noise_diagnostics": noise["areas"],
        "snap_calibration_passed": snap["calibration_all_passed"],
        "snap_bounded_execution_passed": snap["bounded_execution_all_passed"],
        "decision": "hold_native_transfer_training_and_filter_selection",
        "approved_preprocessing_recipe": None,
        "compatibility_established": False,
        "next_evidence": [
            "Establish publisher preprocessing and physical scale, or design a "
            "separate native-domain experiment with its own assumptions",
            "Investigate low surviving HV power with source noise metadata; "
            "finite values alone cannot validate the subtraction",
            "Use only training acquisitions to select adaptations, then freeze "
            "before validation comparison",
        ],
    }


def build() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text())
    collections = [json.loads(checked(r).read_text()) for r in policy["collections"]]
    snap = json.loads(checked(policy["snap"]).read_text())
    noise = json.loads(checked(policy["noise"]).read_text())
    checked(policy["kaggle"])
    rows = [
        r
        for c, src in zip(collections, policy["collections"], strict=True)
        for r in records(c, src)
    ]
    roles: dict[str, str] = {}
    for row in rows:
        key = row["native_acquisition_group"] or row["product_id"]
        if key in roles and roles[key] != row["split"]:
            raise ValueError("Acquisition crosses development roles")
        roles[key] = row["split"]
    inventory = [
        {
            "collection": src,
            "area_id": a["id"],
            "product_id": a["product_id"],
            "split": a["split"],
            "prepared": c["prepared"],
            "whole_area_negative_admitted": False,
            "annotation_completeness_verified": False,
        }
        for c, src in zip(collections, policy["collections"], strict=True)
        for a in c["areas"]
    ]
    return {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "scope": policy["scope"],
        "prior_exposure": policy["prior_exposure"],
        "inventory": inventory,
        "records": rows,
        "summary": {
            "areas": len(inventory),
            "annotations_and_contexts": len(rows),
            "interpretations": dict(Counter(r["interpretation"] for r in rows)),
            "source_supported_roles": dict(
                Counter(
                    f"{r['interpretation']}:{r['split']}"
                    for r in rows
                    if r["disposition"] == "source_supported_hold"
                )
            ),
            "training_admitted": 0,
            "accuracy_truth_admitted": 0,
            "verified_negatives": 0,
        },
        "compatibility": compatibility(snap, noise, collections[0]),
        "future_admission_requirements": policy["future_admission_requirements"],
        "new_scoring_performed": False,
        "labels_modified": False,
        **GUARDS,
    }


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    report = json.loads(checked(lock["evidence"]).read_text())
    if (
        lock["evidence"]["path"] != OUTPUT.relative_to(REPO_ROOT).as_posix()
        or report != build()
    ):
        raise ValueError("Admission decisions differ from pinned evidence")
    if local:
        from agents.hh_hv_transfer import check_report
        from agents.source_annotations import check_collection
        from agents.source_followup_annotations import verify as followup_verify
        from agents.source_reference_report import verify as reference_verify

        policy = json.loads(POLICY.read_text())
        collections = [
            json.loads(checked(r).read_text()) for r in policy["collections"]
        ]
        check_report(collections[0], local=True)
        check_collection(collections[1], local=True)
        followup_lock = json.loads(
            (
                REPO_ROOT
                / "configs/reservations/source_followup_collection_v020.lock.json"
            ).read_text()
        )
        followup_verify(
            REPO_ROOT / followup_lock["directory"],
            followup_lock["seal_sha256"],
            local=True,
        )
        reference_verify(local=True)
    return {
        "verified": True,
        "local": local,
        "summary": report["summary"],
        "evidence": ref(OUTPUT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.publish and not args.verify and not args.local:
        if OUTPUT.exists() or LOCK.exists():
            raise ValueError(
                "Use a new release version; existing evidence is immutable"
            )
        save(OUTPUT, build())
        save(LOCK, {"evidence": ref(OUTPUT)})
    elif args.verify and not args.publish:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose --publish or --verify [--local]")


if __name__ == "__main__":
    main()
