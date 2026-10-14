"""Reserve a fresh source-supported batch after error and preprocessing audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_reservation import save
from agents.nl_benchmark import require_development_scene
from agents.region import load_region
from agents.source_annotations import reservation
from agents.source_reservation import (
    assign,
    checked,
    envelope,
    prior_assignments,
    projected,
    public_entry,
    ref,
    vessel_opportunities,
)
from agents.source_supported import POLICY as SOURCE_POLICY

POLICY = REPO_ROOT / "configs/reservations/source_next_v020.json"


def iceberg_tier(row: dict[str, Any], policy: dict[str, Any]) -> int | None:
    obs = row["observation"]
    if (
        int(obs["timestamp"][:4]) < policy["minimum_iceberg_year"]
        or obs["method"] not in {"VIS", "R/V"}
        or re.match(r"^(S1[AB]|SN|RS|RCM|SENT|TSX|ENVI)", obs["source"])
        or row["absolute_offset_s"] > policy["maximum_iceberg_offset_seconds"]
    ):
        return None
    if obs["size"] in policy["preferred_sizes"]:
        return 0 if row["absolute_offset_s"] <= 900 else 1
    if (
        obs["size"] in policy["exploratory_sizes"]
        and row["absolute_offset_s"] <= policy["small_iceberg_maximum_offset_seconds"]
    ):
        return 2
    return None


def next_role(
    geometry: dict[str, Any], priors: list[dict[str, Any]], group: str
) -> tuple[str, str]:
    split, geographic = assign(geometry, priors, group)
    previous = [
        p
        for p in priors
        if p.get("group") == geographic or p.get("acquisition_group") == group
    ]
    roles = {p["split"] for p in previous}
    if len(roles) > 1:
        raise ValueError("Existing geographic-group role conflict")
    if previous:
        role = next(iter(roles))
        if role not in {"train", "validation"}:
            raise ValueError("Protected geographic-group assignment")
        return role, geographic
    if geographic.startswith("new_cell_"):
        split = (
            "train"
            if hashlib.sha256(geographic.encode()).digest()[0] % 2 == 0
            else "validation"
        )
    return split, geographic


def select(
    candidates: list[dict[str, Any]],
    priors: list[dict[str, Any]],
    exposed: set[str],
    old: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    identities = {(r["stream"], r["source_identity"]) for r in old}
    old_groups = {r["native_acquisition_group"] for r in old}
    working = list(priors)
    for candidate in candidates:
        row = dict(candidate)
        product = row["product"]
        group = native_acquisition_group(product["Name"])
        try:
            if not group or group in exposed or group in old_groups:
                raise ValueError("Prior datatake exposure or reservation")
            if (
                product.get("Online") is not True
                or "_IW_GRDH_1SDH_" not in product["Name"]
            ):
                raise ValueError("Online native IW HH/HV required")
            if (row["stream"], row["source_identity"]) in identities:
                raise ValueError("Previously reserved source identity")
            cap = (
                policy["maximum_iceberg_areas"]
                if row["stream"] == "iceberg_support"
                else policy["maximum_vessel_areas"]
            )
            if sum(s["stream"] == row["stream"] for s in selected) >= cap:
                raise ValueError("Source-class batch cap")
            if (
                sum(s["product_id"] == product["Id"] for s in selected)
                >= policy["maximum_per_product"]
            ):
                raise ValueError("Per-product batch cap")
            geometry = envelope(row["lon"], row["lat"], policy["envelope_half_width_m"])
            load_region(geometry)
            if not shape(product["GeoFootprint"]).covers(shape(geometry)):
                raise ValueError("Complete envelope outside swath")
            split, geographic = next_role(geometry, working, group)
            require_development_scene(
                product["Id"],
                {"name": product["Name"], "split": split, "study_roi": geometry},
            )
            for prior in selected:
                distance = projected(geometry).distance(projected(prior["geometry"]))
                if (
                    prior["native_acquisition_group"] == group
                    and prior["product_id"] != product["Id"]
                ):
                    raise ValueError("Adjacent slice of selected datatake")
                if prior["product_id"] == product["Id"] and distance < 1000:
                    raise ValueError("Overlapping same-scene envelopes")
                if prior["split"] != split and (
                    distance < 10000 or prior["geographic_group"] == geographic
                ):
                    raise ValueError("Cross-split geographic overlap")
            ordinal = sum(s["stream"] == row["stream"] for s in selected) + 1
            row.update(
                id=f"next_{row['stream']}_{ordinal:02d}",
                product_id=product["Id"],
                name=product["Name"],
                native_acquisition_group=group,
                split=split,
                geographic_group=geographic,
                geometry=geometry,
                prior_datatake_exposure=False,
                imagery_inspected=False,
                classifier_scored=False,
                label=None,
                ignore_for_accuracy=True,
                provisional=True,
                independent_review=False,
                promotion_eligible=False,
                training_approved=False,
            )
            selected.append(row)
            identities.add((row["stream"], row["source_identity"]))
            working.append(
                {
                    "geometry": geometry,
                    "group": geographic,
                    "split": split,
                    "acquisition_group": group,
                    "quarantine": False,
                }
            )
        except ValueError as error:
            excluded.append(
                {
                    "product_id": product["Id"],
                    "stream": row["stream"],
                    "reason": str(error),
                }
            )
    return selected, excluded


def candidates(
    iip: dict[str, Any], ais: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    result = []
    for row in iip["ranked"]:
        tier = iceberg_tier(row, policy)
        if tier is None:
            continue
        obs = row["observation"]
        result.append(
            {
                **row,
                "tier": tier,
                "stream": "iceberg_support",
                "source": obs,
                "source_identity": obs["iceberg_id"],
                "lon": obs["lon"],
                "lat": obs["lat"],
            }
        )
    result.sort(
        key=lambda r: (
            r["tier"],
            r["absolute_offset_s"],
            r["source"]["method"] != "R/V",
            r["product"]["Id"],
            r["source_identity"],
        )
    )
    return result + vessel_opportunities(ais, json.loads(SOURCE_POLICY.read_text()))


def build(
    iip_path: Path,
    ais_path: Path,
    error_path: Path,
    preprocessing_path: Path,
    output: Path,
    publish: Path,
) -> None:
    if (
        output.exists()
        or publish.exists()
        or not output.resolve().is_relative_to((REPO_ROOT / "data").resolve())
        or publish.resolve().parent
        != (REPO_ROOT / "reservations/source_supported").resolve()
    ):
        raise ValueError("Fresh scoped reservation directories required")
    error, preprocessing = [
        json.loads(p.read_text()) for p in (error_path, preprocessing_path)
    ]
    if (
        error["labels_changed"] is not False
        or not preprocessing["numerical_reference_all_passed"]
    ):
        raise ValueError("Complete association audit and passing calibration required")
    iip, ais = [json.loads(p.read_text()) for p in (iip_path, ais_path)]
    for r in iip["provenance"]["catalogues"]:
        checked(r)
    normalized = json.loads(checked(iip["provenance"]["normalized"]).read_text())
    for r in normalized["sources"]:
        checked(r)
    for name, expected in (
        ("snapshot_manifest.json", ais["summary"]["snapshot_manifest_sha256"]),
        ("catalogue_products.json", ais["summary"]["catalogue_sha256"]),
    ):
        if sha256_file(ais_path.parent / name) != expected:
            raise ValueError("AIS input provenance mismatch")
    prior_public, _ = reservation()
    private = json.loads(checked(prior_public["private_selection"]).read_text())
    priors, exposed, hashes = prior_assignments()
    for area in private["areas"]:
        priors.append(
            {
                "geometry": area["geometry"],
                "group": area["geographic_group"],
                "split": area["split"],
                "acquisition_group": area["native_acquisition_group"],
                "quarantine": False,
            }
        )
    policy = json.loads(POLICY.read_text())
    pool = candidates(iip, ais, policy)
    selected, excluded = select(pool, priors, exposed, private["areas"], policy)
    output.mkdir(parents=True)
    refs = {
        "iip": ref(iip_path),
        "ais": ref(ais_path),
        "association_audit": ref(error_path),
        "preprocessing": ref(preprocessing_path),
        "policy": ref(POLICY),
        "original_selection": ref(checked(prior_public["private_selection"])),
    }
    save(
        output / "selection_private.json",
        {
            "inputs": refs,
            "priors": priors,
            "exposed": sorted(exposed),
            "prior_metadata_hashes": hashes,
            "areas": selected,
            "exclusions": excluded,
        },
    )
    save(
        output / "first_pass_template.json",
        {
            "independent_review": False,
            "classifier_scores_viewed": False,
            "prior_source_exposure": True,
            "areas": [
                {
                    "id": s["id"],
                    "product_id": s["product_id"],
                    "objects": [],
                    "complete": False,
                    "cells_reviewed": [],
                    "source_chip_sha256": None,
                    "notes": "Await scoped native preparation, then both-band "
                    "inspection before scores",
                }
                for s in selected
            ],
        },
    )
    publish.mkdir(parents=True)
    areas = [
        public_entry(row)
        | {
            "training_approved": False,
            "reported_size": row["source"].get("size"),
            "priority_tier": row.get("tier"),
        }
        for row in selected
    ]
    result = {
        "version": policy["version"],
        "inputs": refs,
        "private_selection": ref(output / "selection_private.json"),
        "first_pass_template": ref(output / "first_pass_template.json"),
        "areas": areas,
        "exclusions": excluded,
        "screened_opportunities": len(pool),
        "class_counts": dict(Counter(s["stream"] for s in selected)),
        "split_source_counts": dict(
            Counter(s["split"] + ":" + s["stream"] for s in selected)
        ),
        "native_acquisitions": len({s["native_acquisition_group"] for s in selected}),
        "annotated_objects": 0,
        "scored": False,
        "independent_review": False,
        "promotion_eligible": False,
        "training_approved": False,
        "compatibility_established": False,
        "limitations": [
            "Source opportunities are not image labels",
            "Both-class split coverage is a reported outcome, not assumed",
            "No new image pixels, classifier scores or source contact used",
        ],
    }
    save(publish / "manifest.json", result)
    save(
        publish / "audit_summary.json",
        {
            "association": error,
            "preprocessing": {
                key: preprocessing[key]
                for key in (
                    "version",
                    "policy",
                    "prepared",
                    "metadata",
                    "collection",
                    "gdal_version",
                    "rasterio_version",
                    "calibration",
                    "numerical_reference_all_passed",
                    "gdal_calibration_all_passed",
                    "noise_inventory",
                    "kaggle_training",
                    "distribution",
                    "compatibility_established",
                    "training_approved",
                    "independent_review",
                    "classifier_scored",
                    "reference_scope",
                    "limitations",
                )
            },
            "full_private_reports": {
                "association": ref(error_path),
                "preprocessing": ref(preprocessing_path),
            },
        },
    )
    save(
        publish / "seal.json",
        {
            "manifest": ref(publish / "manifest.json"),
            "audit_summary": ref(publish / "audit_summary.json"),
        },
    )
    print(
        json.dumps(
            {
                "seal": ref(publish / "seal.json"),
                "counts": result["split_source_counts"],
            }
        )
    )


def validate(manifest: dict[str, Any]) -> None:
    for key in (
        "scored",
        "independent_review",
        "promotion_eligible",
        "training_approved",
        "compatibility_established",
    ):
        if manifest[key] is not False:
            raise ValueError("Provisional reservation guard changed")
    if manifest["annotated_objects"] != 0:
        raise ValueError("Reservation contains annotations")
    areas = manifest["areas"]
    if (
        len({a["id"] for a in areas}) != len(areas)
        or dict(Counter(a["stream"] for a in areas)) != manifest["class_counts"]
        or dict(Counter(a["split"] + ":" + a["stream"] for a in areas))
        != manifest["split_source_counts"]
        or len({a["native_acquisition_group"] for a in areas})
        != manifest["native_acquisitions"]
    ):
        raise ValueError("Reservation inventory mismatch")
    for area in manifest["areas"]:
        if (
            area["label"] is not None
            or area["provisional"] is not True
            or area["ignore_for_accuracy"] is not True
            or any(
                area[k] is not False
                for k in (
                    "imagery_inspected",
                    "classifier_scored",
                    "independent_review",
                    "promotion_eligible",
                    "training_approved",
                )
            )
        ):
            raise ValueError("Uninspected provisional area required")
        if area["split"] not in {"train", "validation"}:
            raise ValueError("Development split required")


def verify(directory: Path, digest: str, local: bool = False) -> None:
    if sha256_file(directory / "seal.json") != digest:
        raise ValueError("Trusted next-batch seal changed")
    seal = json.loads((directory / "seal.json").read_text())
    manifest = json.loads(checked(seal["manifest"]).read_text())
    checked(seal["audit_summary"])
    checked(manifest["inputs"]["policy"])
    validate(manifest)
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        checked(manifest["first_pass_template"])
        for r in private["inputs"].values():
            checked(r)
        for path, expected in private["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": expected})
        policy = json.loads(checked(private["inputs"]["policy"]).read_text())
        iip = json.loads(checked(private["inputs"]["iip"]).read_text())
        ais = json.loads(checked(private["inputs"]["ais"]).read_text())
        original = json.loads(
            checked(private["inputs"]["original_selection"]).read_text()
        )
        replay, excluded = select(
            candidates(iip, ais, policy),
            private["priors"],
            set(private["exposed"]),
            original["areas"],
            policy,
        )
        if (
            json.loads(json.dumps(replay)) != private["areas"]
            or excluded != private["exclusions"]
        ):
            raise ValueError("Next-batch replay differs")
        public = [
            public_entry(row)
            | {
                "training_approved": False,
                "reported_size": row["source"].get("size"),
                "priority_tier": row.get("tier"),
            }
            for row in replay
        ]
        if public != manifest["areas"]:
            raise ValueError("Public/private reservation mismatch")
    print(
        json.dumps(
            {
                "verified": True,
                "local": local,
                "reserved": len(manifest["areas"]),
                "scored": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "iip",
        "ais",
        "error-audit",
        "preprocessing",
        "output",
        "publish",
        "verify",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.verify and args.seal_sha256:
        verify(args.verify, args.seal_sha256, args.local)
    elif all(
        (
            args.iip,
            args.ais,
            args.error_audit,
            args.preprocessing,
            args.output,
            args.publish,
        )
    ):
        build(
            args.iip,
            args.ais,
            args.error_audit,
            args.preprocessing,
            args.output,
            args.publish,
        )
    else:
        parser.error("Choose build or verify with required paths")
