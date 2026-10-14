"""Provisional admission decisions and source-ranked training-only reservations."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from agents.acquisition import native_acquisition_group
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_reservation import save
from agents.nl_benchmark import require_development_scene
from agents.region import load_region
from agents.source_next_batch import iceberg_tier
from agents.source_reservation import (
    assign,
    checked,
    envelope,
    prior_assignments,
    public_entry,
    ref,
    vessel_opportunities,
)

POLICY = REPO_ROOT / "configs/reservations/provisional_training_v010.json"
DIRECTORY = REPO_ROOT / "reservations/native_training/v0.1.0"
LOCK = REPO_ROOT / "configs/reservations/provisional_training_v010.lock.json"
GUARDS = {
    "provisional": True,
    "ignore_for_accuracy": True,
    "independent_review": False,
    "promotion_eligible": False,
    "training_approved": False,
}


def policy() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(POLICY.read_text())
    for key in ("native_specification", "historical_admission", "source_policy"):
        checked(value[key])
    return value


def assess(dossier: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Decide exploratory eligibility from explicit, hash-bound declarations.

    This is an audit interface, not a physical-truth oracle or training loader.
    No existing annotation, mask, benchmark review state or model is modified.
    """
    reasons = []
    if dossier.get("policy") != {
        "path": POLICY.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(POLICY),
    }:
        reasons.append("frozen_admission_policy_required")
    if (
        not isinstance(dossier.get("object_id"), str)
        or not dossier["object_id"].strip()
    ):
        reasons.append("object_identity_required")
    if dossier.get("split") != "train":
        reasons.append("training_role_required")
    if dossier.get("class") not in rules["admission"]["classes"]:
        reasons.append("unresolved_or_unsupported_class")
    if dossier.get("provisional") is not True:
        reasons.append("provisional_status_required")
    if dossier.get("prior_exposure_disclosed") is not True:
        reasons.append("prior_exposure_disclosure_required")
    if dossier.get("native_specification") != rules["native_specification"]:
        reasons.append("frozen_native_recipe_required")
    annotated = parse_utc(dossier.get("annotation_completed_utc"))
    scored = parse_utc(dossier.get("first_score_utc"))
    if annotated is None or (
        dossier.get("scores_viewed") is not False
        or (dossier.get("first_score_utc") is not None and scored is None)
        or (scored is not None and annotated >= scored)
    ):
        reasons.append("annotation_before_scores_required")
    checks = rules["admission"]["required_checks"] + rules["admission"].get(
        f"{dossier.get('class')}_checks", []
    )
    evidence = dossier.get("checks", {})
    for name in checks:
        item = evidence.get(name, {})
        if item.get("passed") is not True or not item.get("evidence"):
            reasons.append(name)
            continue
        for reference in item["evidence"]:
            record = json.loads(checked(reference).read_text())
            if (
                record.get("product_id") != dossier.get("product_id")
                or record.get("object_id") != dossier.get("object_id")
                or record.get("checks", {}).get(name) is not True
            ):
                reasons.append(f"{name}:evidence_subject_or_result_mismatch")
    try:
        annotation = json.loads(checked(dossier["annotation"]).read_text())
        if (
            annotation["product_id"] != dossier["product_id"]
            or annotation["name"] != dossier["name"]
            or annotation["object_id"] != dossier["object_id"]
            or annotation["split"] != "train"
            or annotation["class"] != dossier["class"]
            or annotation.get("provisional") is not True
            or annotation["geometry"] != dossier["geometry"]
            or annotation["geographic_group"] != dossier["geographic_group"]
            or annotation["annotation_completed_utc"]
            != dossier["annotation_completed_utc"]
        ):
            reasons.append("annotation_identity_or_role_mismatch")
        geometry = dossier["geometry"]
        load_region(geometry)
        require_development_scene(
            dossier["product_id"],
            {"name": dossier["name"], "split": dossier["split"], "study_roi": geometry},
        )
        group = native_acquisition_group(dossier["name"])
        priors, _, _, _ = inventory()
        if group is None or training_role(geometry, priors, group) != dossier.get(
            "geographic_group"
        ):
            reasons.append("prior_group_or_split_assignment_mismatch")
    except (KeyError, ValueError, TypeError, FileNotFoundError):
        reasons.append("annotation_or_development_identity_unverified")
    return {
        "disposition": "hold" if reasons else "provisional_exploratory_training",
        "exploratory_training_admitted": not reasons,
        "accuracy_truth_admitted": False,
        "verified_negative": False,
        "blockers": reasons,
        **GUARDS,
    }


def source_pool(
    iip: dict[str, Any], ais: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = vessel_opportunities(
        ais, json.loads(checked(rules["source_policy"]).read_text())
    )
    for row in iip["ranked"]:
        if iceberg_tier(row, rules) is None or re.match(
            r"^(S1[AB]|SN|RS|RCM|SENT|TSX|ENVI)", row["observation"]["source"]
        ):
            continue
        obs = row["observation"]
        rows.append(
            {
                **row,
                "stream": "iceberg_support",
                "source": obs,
                "source_identity": obs["iceberg_id"],
                "lon": obs["lon"],
                "lat": obs["lat"],
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            r["stream"] != "vessel_support",
            r["absolute_offset_s"],
            r["search_uncertainty_m"],
            r.get("bracket_gap_s", 0),
            r["product"]["Id"],
            r["source_identity"],
        ),
    )


def training_role(
    geometry: dict[str, Any], priors: list[dict[str, Any]], group: str
) -> str:
    split, geographic = assign(geometry, priors, group)
    roles = {
        p["split"]
        for p in priors
        if p["group"] == geographic or p.get("acquisition_group") == group
    }
    if split != "train" or roles - {"train"}:
        raise ValueError("Existing validation or protected role cannot become training")
    return geographic


def select(
    pool: list[dict[str, Any]],
    priors: list[dict[str, Any]],
    exposed: set[str],
    old: list[dict[str, Any]],
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    ranked = []
    old_groups = {r["native_acquisition_group"] for r in old}
    missions = json.loads(checked(rules["native_specification"]).read_text())["mission"]
    identities = {(r["stream"], str(r["source_identity"])) for r in old}
    working = list(priors)
    for rank, candidate in enumerate(pool, 1):
        row = dict(candidate)
        product = row["product"]
        group = native_acquisition_group(product["Name"])
        reason = None
        try:
            if product["Name"].split("_", 1)[0] not in missions:
                raise ValueError("Mission outside frozen native input contract")
            if not group or group in exposed or group in old_groups:
                raise ValueError("Previously exposed or reserved native acquisition")
            if (row["stream"], str(row["source_identity"])) in identities:
                raise ValueError("Previously reserved source identity")
            if (
                product.get("Online") is not True
                or "_IW_GRDH_1SDH_" not in product["Name"]
            ):
                raise ValueError("Online IW GRDH HH/HV required")
            if any(
                not math.isfinite(float(row[k])) or float(row[k]) < 0
                for k in ("absolute_offset_s", "search_uncertainty_m")
            ):
                raise ValueError("Finite source timing and uncertainty required")
            geometry = envelope(row["lon"], row["lat"], rules["envelope_half_width_m"])
            load_region(geometry)
            if not shape(product["GeoFootprint"]).covers(shape(geometry)):
                raise ValueError("Complete envelope outside catalogue footprint")
            geographic = training_role(geometry, working, group)
            require_development_scene(
                product["Id"],
                {"name": product["Name"], "split": "train", "study_roi": geometry},
            )
            if (
                sum(s["stream"] == row["stream"] for s in selected)
                >= rules["maximum_per_stream"]
            ):
                raise ValueError("Source-stream batch cap")
            if (
                sum(s["native_acquisition_group"] == group for s in selected)
                >= rules["maximum_per_native_acquisition"]
            ):
                raise ValueError("Native-acquisition batch cap")
            ordinal = sum(s["stream"] == row["stream"] for s in selected) + 1
            row.update(
                id=f"training_{row['stream']}_{ordinal:02d}",
                product_id=product["Id"],
                name=product["Name"],
                native_acquisition_group=group,
                split="train",
                geographic_group=geographic,
                geometry=geometry,
                prior_datatake_exposure=False,
                imagery_inspected=False,
                classifier_scored=False,
                label=None,
                **GUARDS,
            )
            selected.append(row)
            identities.add((row["stream"], str(row["source_identity"])))
            working.append(
                {
                    "geometry": geometry,
                    "group": geographic,
                    "split": "train",
                    "acquisition_group": group,
                    "quarantine": False,
                }
            )
        except ValueError as error:
            reason = str(error)
        ranked.append(
            {
                "rank": rank,
                "stream": row["stream"],
                "product_id": product["Id"],
                "name": product["Name"],
                "absolute_offset_s": row["absolute_offset_s"],
                "search_uncertainty_m": row["search_uncertainty_m"],
                "bracket_gap_s": row.get("bracket_gap_s"),
                "decision": "excluded" if reason else "reserved_train",
                "reason": reason,
                "reservation_id": None if reason else row["id"],
            }
        )
    return selected, ranked


def inventory() -> (
    tuple[list[dict[str, Any]], set[str], list[dict[str, Any]], dict[str, str]]
):
    priors, exposed, hashes = prior_assignments()
    old = []
    for path in sorted(
        (REPO_ROOT / "reservations/source_supported").glob("*/manifest.json")
    ):
        manifest = json.loads(path.read_text())
        hashes[path.relative_to(REPO_ROOT).as_posix()] = sha256_file(path)
        private_path = checked(manifest["private_selection"])
        hashes[private_path.relative_to(REPO_ROOT).as_posix()] = sha256_file(
            private_path
        )
        private = json.loads(private_path.read_text())
        for area in private["areas"]:
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
    return priors, exposed, old, hashes


def public_area(row: dict[str, Any]) -> dict[str, Any]:
    return public_entry(row) | {"training_approved": False}


def validate(manifest: dict[str, Any], rankings: list[dict[str, Any]]) -> None:
    if (
        any(manifest.get(k) is not v for k, v in GUARDS.items())
        or manifest["annotated_objects"] != 0
    ):
        raise ValueError("Provisional reservation guards changed")
    selected = [r for r in rankings if r["decision"] == "reserved_train"]
    areas = manifest["areas"]
    if (
        len({a["id"] for a in areas}) != len(areas)
        or len(selected) != len(areas)
        or {r["reservation_id"] for r in selected} != {a["id"] for a in areas}
        or len({a["native_acquisition_group"] for a in areas}) != len(areas)
        or dict(Counter(a["stream"] for a in areas)) != manifest["class_counts"]
    ):
        raise ValueError("Training reservation inventory mismatch")
    for area in areas:
        if (
            area["split"] != "train"
            or area["label"] is not None
            or any(area.get(k) is not v for k, v in GUARDS.items())
            or any(
                area.get(k) is not False
                for k in (
                    "imagery_inspected",
                    "classifier_scored",
                    "prior_datatake_exposure",
                )
            )
        ):
            raise ValueError("Uninspected training-only reservation required")
        selected_row = next(r for r in selected if r["reservation_id"] == area["id"])
        if any(selected_row[k] != area[k] for k in ("product_id", "name", "stream")):
            raise ValueError("Ranked source and reserved product differ")
    rules = policy()
    expected = {
        stream: rules["maximum_per_stream"] - manifest["class_counts"].get(stream, 0)
        for stream in ("vessel_support", "iceberg_support")
    }
    if (
        any(v < 0 for v in expected.values())
        or expected != manifest["shortfalls_to_batch_cap"]
    ):
        raise ValueError("Source coverage shortfall mismatch")
    if [r["rank"] for r in rankings] != list(range(1, len(rankings) + 1)):
        raise ValueError("Complete sequential source ranking required")


def source_checks(iip_path: Path, ais_path: Path) -> dict[str, Any]:
    iip_path, ais_path = iip_path.resolve(), ais_path.resolve()
    iip, ais = [json.loads(p.read_text()) for p in (iip_path, ais_path)]
    normalized = json.loads(checked(iip["provenance"]["normalized"]).read_text())
    rules = policy()
    if iip["provenance"]["policy_sha256"] != rules["source_policy"]["sha256"]:
        raise ValueError("IIP source ranking policy mismatch")
    for reference in normalized["sources"] + iip["provenance"]["catalogues"]:
        checked(reference)
    if any(
        json.loads(checked(r).read_text()).get("complete") is not True
        for r in iip["provenance"]["catalogues"]
    ):
        raise ValueError("Complete IIP catalogue snapshots required")
    for name, key in (
        ("snapshot_manifest.json", "snapshot_manifest_sha256"),
        ("catalogue_products.json", "catalogue_sha256"),
    ):
        checked(
            {
                "path": (ais_path.parent / name).relative_to(REPO_ROOT).as_posix(),
                "sha256": ais["summary"][key],
            }
        )
    snapshot = json.loads((ais_path.parent / "snapshot_manifest.json").read_text())
    for entry in snapshot["files"]:
        path = (ais_path.parent / "archive_snapshot" / entry["file"]).resolve()
        if path.parent != (ais_path.parent / "archive_snapshot").resolve():
            raise ValueError("AIS snapshot path escape")
        checked(
            {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": entry["sha256"]}
        )
    return {"iip": iip, "ais": ais}


def publish(iip_path: Path, ais_path: Path, output: Path) -> None:
    if (
        DIRECTORY.exists()
        or LOCK.exists()
        or output.exists()
        or not output.resolve().is_relative_to(
            (REPO_ROOT / "data/source_supported").resolve()
        )
    ):
        raise ValueError("Fresh scoped release and private directory required")
    rules = policy()
    sources = source_checks(iip_path, ais_path)
    priors, exposed, old, hashes = inventory()
    selected, ranked = select(
        source_pool(**sources, rules=rules), priors, exposed, old, rules
    )
    inputs = {
        "policy": ref(POLICY),
        "iip": ref(iip_path),
        "ais": ref(ais_path),
        "implementation": ref(Path(__file__)),
    }
    private = {
        "inputs": inputs,
        "priors": priors,
        "exposed": sorted(exposed),
        "old": old,
        "prior_metadata_hashes": hashes,
        "areas": selected,
        "rankings": ranked,
    }
    save(output / "selection_private.json", private)
    dossier_templates = []
    check_names = (
        rules["admission"]["required_checks"]
        + rules["admission"]["vessel_checks"]
        + rules["admission"]["iceberg_checks"]
    )
    for area in selected:
        path = output / f"admission_{area['id']}.json"
        save(
            path,
            {
                "policy": ref(POLICY),
                "reservation_id": area["id"],
                "product_id": area["product_id"],
                "name": area["name"],
                "geographic_group": area["geographic_group"],
                "geometry": area["geometry"],
                "split": "train",
                "object_id": None,
                "class": None,
                "annotation": None,
                "provisional": True,
                "prior_exposure_disclosed": True,
                "native_specification": rules["native_specification"],
                "annotation_completed_utc": None,
                "first_score_utc": None,
                "scores_viewed": False,
                "checks": {
                    name: {"passed": False, "evidence": []} for name in check_names
                },
            },
        )
        dossier_templates.append(ref(path))
    save(
        output / "annotation_template.json",
        {
            "provisional": True,
            "prior_project_exposure": True,
            "classifier_scores_viewed": False,
            "areas": [
                {
                    "id": s["id"],
                    "product_id": s["product_id"],
                    "split": "train",
                    "geometry": s["geometry"],
                    "annotation_completed_utc": None,
                    "all_visible_returns_recorded": False,
                    "cells_reviewed": [],
                    "objects": [],
                    "complete": False,
                }
                for s in selected
            ],
        },
    )
    counts = dict(Counter(s["stream"] for s in selected))
    manifest = {
        "version": rules["version"],
        "inputs": inputs,
        "private_selection": ref(output / "selection_private.json"),
        "annotation_template": ref(output / "annotation_template.json"),
        "dossier_templates": dossier_templates,
        "areas": [public_area(s) for s in selected],
        "class_counts": counts,
        "shortfalls_to_batch_cap": {
            s: rules["maximum_per_stream"] - counts.get(s, 0)
            for s in ("vessel_support", "iceberg_support")
        },
        "source_snapshot": {
            "ais_archive": sources["ais"]["summary"]["archive"],
            "ais_audited_at": sources["ais"]["summary"]["audited_at"],
            "iip_years": [2014, 2021],
            "fresh_ais_catalogue_query": sources["ais"]["summary"].get("replay_of")
            is None,
            "iip_catalogue_query": "Retained historical snapshots; not refreshed",
        },
        "annotated_objects": 0,
        "historical_labels_admitted": 0,
        "accuracy_truth_admitted": False,
        **GUARDS,
        "limitations": rules["limitations"],
    }
    validate(manifest, ranked)
    save(DIRECTORY / "manifest.json", manifest)
    save(DIRECTORY / "rankings.json", ranked)
    save(
        LOCK,
        {
            "manifest": ref(DIRECTORY / "manifest.json"),
            "rankings": ref(DIRECTORY / "rankings.json"),
        },
    )
    print(
        json.dumps(
            {
                "reserved": counts,
                "shortfalls": manifest["shortfalls_to_batch_cap"],
                "annotated_objects": 0,
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    manifest = json.loads(checked(lock["manifest"]).read_text())
    rankings = json.loads(checked(lock["rankings"]).read_text())
    rules = policy()
    checked(manifest["inputs"]["policy"])
    checked(manifest["inputs"]["implementation"])
    validate(manifest, rankings)
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        checked(manifest["annotation_template"])
        for reference in manifest["dossier_templates"]:
            checked(reference)
        for path, sha in private["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        sources = source_checks(
            checked(private["inputs"]["iip"]), checked(private["inputs"]["ais"])
        )
        selected, replay = select(
            source_pool(**sources, rules=rules),
            private["priors"],
            set(private["exposed"]),
            private["old"],
            rules,
        )
        if (
            json.loads(json.dumps(selected)) != private["areas"]
            or replay != rankings
            or [public_area(s) for s in selected] != manifest["areas"]
        ):
            raise ValueError("Source ranking replay differs")
    return {
        "verified": True,
        "local": local,
        "reserved": manifest["class_counts"],
        "annotated_objects": 0,
        "labels_admitted": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--iip", type=Path)
    parser.add_argument("--ais", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assess", type=Path, help="Read-only provisional dossier decision"
    )
    args = parser.parse_args()
    if sum((args.publish, args.verify, args.assess is not None)) != 1:
        parser.error("Choose publish, verify or assess")
    if args.publish:
        if not all((args.iip, args.ais, args.output)):
            parser.error("Publish needs IIP, AIS and fresh private output")
        publish(args.iip, args.ais, args.output)
    elif args.assess:
        print(json.dumps(assess(json.loads(args.assess.read_text()), policy())))
    else:
        print(json.dumps(verify(args.local)))


if __name__ == "__main__":
    main()
