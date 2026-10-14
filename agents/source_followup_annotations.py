"""Seal provisional follow-up annotations and source evidence before scoring."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer

from agents.artifacts import REPO_ROOT, sha256_file
from agents.association import GEOD
from agents.hh_hv_reservation import save
from agents.hh_hv_transfer import native_chip
from agents.source_annotations import (
    public_metadata,
    validate_first_pass,
    validate_reconciliation,
)
from agents.source_followup import PROTOCOL, fresh, replay, selection
from agents.source_reservation import checked, ref
from agents.source_windows import SIZE, metadata


def source_evidence(prepared: Path, first_path: Path) -> dict[str, Any]:
    rows = replay(prepared)
    first = json.loads(first_path.read_text())
    validate_first_pass(first, {a["id"] for a in rows})
    _, private = selection()
    manifest = json.loads(prepared.read_text())
    ranking = json.loads(checked(private["inputs"]["iip"]).read_text())
    normalized = json.loads(checked(ranking["provenance"]["normalized"]).read_text())
    observations = [
        o
        for o in normalized["observations"]
        if o["method"] in {"VIS", "R/V"}
        and not re.match(r"^(S1[ABC]|SN|RS|RCM|SENT|TSX|ENVI)", o["source"])
    ]
    areas = []
    for row in rows:
        original = next(a for a in private["areas"] if a["id"] == row["id"])
        product = next(
            p for p in manifest["products"] if p["product_id"] == row["product_id"]
        )
        xml = checked(product["bands"][0]["annotation"]).read_bytes()
        info, gcps = metadata(xml, original, "hh")
        decisions = next(a for a in first["areas"] if a["id"] == row["id"])
        if decisions["chip_sha256"] != row["chip"]["sha256"]:
            raise ValueError("First-pass chip binding changed")
        objects = []
        for obj in decisions["objects"]:
            x0, y0, x1, y1 = obj["bbox_px"]
            col, line = (
                row["window"][0] + (x0 + x1) / 2,
                row["window"][1] + (y0 + y1) / 2,
            )
            with GCPTransformer(gcps, tps=True) as tx:
                xs, ys = tx.xy([line], [col], offset="ul")
            start = datetime.fromisoformat(
                str(info["native_start_time"]).replace("Z", "+00:00")
            )
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            instant = start.timestamp() + line * info["azimuth_interval_s"]
            selected = original["source"]
            offset = abs(instant - selected["time"])
            distance = GEOD.inv(xs[0], ys[0], selected["lon"], selected["lat"])[2]
            matches = []
            for obs in observations:
                delta = abs(instant - obs["time"])
                if delta > 1800:
                    continue
                dist = GEOD.inv(xs[0], ys[0], obs["lon"], obs["lat"])[2]
                if dist <= 300 + delta:
                    matches.append(
                        {
                            "source_identity": obs["iceberg_id"],
                            "distance_m": dist,
                            "absolute_offset_s": delta,
                            "selected_identity": obs["iceberg_id"]
                            == original["source_identity"],
                        }
                    )
            objects.append(
                {
                    "object_id": obj["id"],
                    "selected_source_distance_m": distance,
                    "frozen_allowance_m": 300 + offset,
                    "inside_selected_allowance": distance <= 300 + offset,
                    "native_time_utc": datetime.fromtimestamp(
                        instant, timezone.utc
                    ).isoformat(),
                    "compatible_iceberg_identities": sorted(
                        {m["source_identity"] for m in matches}
                    ),
                    "matches": matches,
                }
            )
        areas.append(
            {
                "id": row["id"],
                "objects": objects,
                "source_association_abstain": row["source_coverage"][
                    "source_association_abstain"
                ],
            }
        )
    return {
        "prepared": ref(prepared),
        "first_pass": ref(first_path),
        "areas": areas,
        "scored": False,
        "labels_assigned_automatically": False,
        "limitations": [
            "All nearby visual/RV reports in frozen IIP data considered; not exhaustive truth",
            "No co-temporal area-wide AIS or independent imagery confirmation",
            "Frozen allowance is heuristic; sensitivity cells do not change matching gates",
        ],
    }


def validate_support(areas: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    for area in areas:
        support = next(a for a in evidence["areas"] if a["id"] == area["id"])
        for obj in area["objects"]:
            if obj["class"] == "provisional_vessel":
                raise ValueError(
                    "This batch has no co-temporal vessel identity evidence"
                )
            if obj["class"] == "plausible_iceberg":
                record = next(
                    o for o in support["objects"] if o["object_id"] == obj["id"]
                )
                eligible = [
                    o for o in support["objects"] if o["inside_selected_allowance"]
                ]
                if (
                    support["source_association_abstain"]
                    or not record["inside_selected_allowance"]
                    or len(eligible) != 1
                    or len(record["compatible_iceberg_identities"]) != 1
                    or not any(m["selected_identity"] for m in record["matches"])
                ):
                    raise ValueError(
                        "Iceberg interpretation lacks complete unique provisional source support"
                    )


def build(
    prepared: Path,
    first_path: Path,
    reconciliation_path: Path,
    output: Path,
    publish: Path,
) -> None:
    if (
        publish.exists()
        or publish.resolve().parent
        != (REPO_ROOT / "datasets/source_supported").resolve()
    ):
        raise ValueError("Fresh separate source collection required")
    rows = replay(prepared)
    first = json.loads(first_path.read_text())
    validate_first_pass(first, {a["id"] for a in rows})
    reconciliation = json.loads(reconciliation_path.read_text())
    validate_reconciliation(ref(first_path), reconciliation)
    evidence = json.loads(
        checked(reconciliation["private_source_evidence"]).read_text()
    )
    if evidence != source_evidence(prepared, first_path):
        raise ValueError("Source reconciliation evidence replay differs")
    validate_support(reconciliation["areas"], evidence)
    fresh(output)
    areas, samples = [], []
    for row in rows:
        objects = next(a for a in reconciliation["areas"] if a["id"] == row["id"])[
            "objects"
        ]
        with rasterio.open(checked(row["chip"])) as ds:
            values = ds.read()
        mask_path = output / f"{row['id']}_ignore.npy"
        np.save(mask_path, np.ones((SIZE, SIZE), dtype="uint8"), allow_pickle=False)
        for obj in objects:
            chip = native_chip(values, obj["bbox_px"], obj["class"] == "clutter")
            reference = None
            if chip is not None:
                path = output / f"{row['id']}_{obj['id']}_native75.npy"
                np.save(path, chip, allow_pickle=False)
                reference = ref(path)
            samples.append(
                {
                    "area_id": row["id"],
                    "object_id": obj["id"],
                    "chip": reference,
                    "abstention": (
                        None
                        if reference
                        else "Native75 extent/edge: no padding, stretching or shifting"
                    ),
                }
            )
        areas.append(
            {
                "id": row["id"],
                "product_id": row["product_id"],
                "split": row["split"],
                "geographic_group": row["geographic_group"],
                "native_acquisition_group": row["native_acquisition_group"],
                "source_coverage": row["source_coverage"],
                "objects": objects,
                "chip_sha256": row["chip"]["sha256"],
                "ignore_mask": ref(mask_path),
                "whole_area_ignore_for_accuracy": True,
                "independent_review": False,
            }
        )
    save(
        output / "report.json",
        {
            "prepared": ref(prepared),
            "first_pass": ref(first_path),
            "reconciliation": ref(reconciliation_path),
            "areas": areas,
            "samples": samples,
            "classifier_scored": False,
            "training_approved": False,
        },
    )
    collection = {
        "version": "0.2.0",
        "protocol": ref(PROTOCOL),
        "prepared": ref(prepared),
        "first_pass": ref(first_path),
        "reconciliation": ref(reconciliation_path),
        "local_report": ref(output / "report.json"),
        "areas": areas,
        "class_counts": dict(Counter(o["class"] for a in areas for o in a["objects"])),
        "native_calibration_bands_replayed": len(rows) * 2,
        "extracted_native75_contexts": sum(s["chip"] is not None for s in samples),
        "classifier_scored": False,
        "detector_scored": False,
        "independent_review": False,
        "training_approved": False,
        "promotion_eligible": False,
        "compatibility_established": False,
        "annotation_completeness_verified": False,
    }
    public_metadata(collection)
    validate_collection(collection)
    publish.mkdir(parents=True)
    save(publish / "collection.json", collection)
    save(
        publish / "seal.json",
        {
            "collection": ref(publish / "collection.json"),
            "status": "provisional_before_scoring",
        },
    )
    print(
        json.dumps(
            {
                "areas": len(areas),
                "class_counts": collection["class_counts"],
                "seal": ref(publish / "seal.json"),
            }
        )
    )


def validate_collection(value: dict[str, Any]) -> None:
    for key in (
        "classifier_scored",
        "detector_scored",
        "independent_review",
        "training_approved",
        "promotion_eligible",
        "compatibility_established",
        "annotation_completeness_verified",
    ):
        if value[key] is not False:
            raise ValueError("Provisional collection guard changed")
    public_metadata(value)
    checked(value["protocol"])
    first = json.loads(checked(value["first_pass"]).read_text())
    reconciliation = json.loads(checked(value["reconciliation"]).read_text())
    validate_first_pass(first, {a["id"] for a in value["areas"]})
    validate_reconciliation(value["first_pass"], reconciliation)
    if len(value["areas"]) != 6 or len({a["id"] for a in value["areas"]}) != 6:
        raise ValueError("Complete six-area collection required")
    for area in value["areas"]:
        decision = next(a for a in reconciliation["areas"] if a["id"] == area["id"])
        original = next(a for a in first["areas"] if a["id"] == area["id"])
        if (
            area["whole_area_ignore_for_accuracy"] is not True
            or area["independent_review"] is not False
            or area["objects"] != decision["objects"]
            or area["chip_sha256"] != original["chip_sha256"]
            or area["split"] not in {"train", "validation"}
        ):
            raise ValueError("Provisional annotation lineage changed")
        if area["source_coverage"]["source_association_abstain"] and any(
            o["class"] in {"plausible_iceberg", "provisional_vessel"}
            for o in area["objects"]
        ):
            raise ValueError(
                "Incomplete coverage must abstain from source class assignment"
            )
    if value["class_counts"] != dict(
        Counter(o["class"] for a in value["areas"] for o in a["objects"])
    ):
        raise ValueError("Class counts differ")


def verify(directory: Path, digest: str, local: bool = False) -> None:
    if sha256_file(directory / "seal.json") != digest:
        raise ValueError("Trusted follow-up collection seal changed")
    seal = json.loads((directory / "seal.json").read_text())
    value = json.loads(checked(seal["collection"]).read_text())
    validate_collection(value)
    if local:
        rows = replay(checked(value["prepared"]))
        reconciliation = json.loads(checked(value["reconciliation"]).read_text())
        evidence = json.loads(
            checked(reconciliation["private_source_evidence"]).read_text()
        )
        if evidence != source_evidence(
            checked(value["prepared"]), checked(value["first_pass"])
        ):
            raise ValueError("Source evidence replay differs")
        validate_support(reconciliation["areas"], evidence)
        report = json.loads(checked(value["local_report"]).read_text())
        if report["areas"] != value["areas"] or report["prepared"] != value["prepared"]:
            raise ValueError("Local/public annotation reports differ")
        for area in value["areas"]:
            row = next(r for r in rows if r["id"] == area["id"])
            if (
                area["source_coverage"] != row["source_coverage"]
                or area["chip_sha256"] != row["chip"]["sha256"]
                or any(
                    area[k] != row[k]
                    for k in (
                        "product_id",
                        "split",
                        "geographic_group",
                        "native_acquisition_group",
                    )
                )
            ):
                raise ValueError("Annotation scope or coverage changed")
            mask = np.load(checked(area["ignore_mask"]), allow_pickle=False)
            if (
                mask.shape != (SIZE, SIZE)
                or mask.dtype != np.uint8
                or not (mask == 1).all()
            ):
                raise ValueError("Whole-area ignore mask changed")
            with rasterio.open(checked(row["chip"])) as ds:
                values = ds.read()
            for obj in area["objects"]:
                sample = next(
                    s
                    for s in report["samples"]
                    if s["area_id"] == area["id"] and s["object_id"] == obj["id"]
                )
                expected = native_chip(
                    values, obj["bbox_px"], obj["class"] == "clutter"
                )
                if expected is None:
                    if sample["chip"] is not None:
                        raise ValueError("Native edge abstention changed")
                elif sample["chip"] is None or not np.array_equal(
                    expected, np.load(checked(sample["chip"]), allow_pickle=False)
                ):
                    raise ValueError("Native75 context differs from source window")
    print(
        json.dumps(
            {
                "verified": True,
                "local": local,
                "areas": len(value["areas"]),
                "scored": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared",
        "first-pass",
        "reconciliation",
        "output",
        "publish",
        "verify",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--support", action="store_true")
    parser.add_argument("--seal-sha256")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.verify and args.seal_sha256:
        verify(args.verify, args.seal_sha256, args.local)
    elif args.support and args.prepared and args.first_pass and args.output:
        if args.output.exists() or not args.output.resolve().is_relative_to(
            (REPO_ROOT / "data").resolve()
        ):
            raise ValueError("Fresh private evidence file required")
        evidence = source_evidence(args.prepared, args.first_pass)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, evidence)
        print(json.dumps({"evidence": ref(args.output), "scored": False}))
    elif all(
        (args.prepared, args.first_pass, args.reconciliation, args.output, args.publish)
    ):
        build(
            args.prepared,
            args.first_pass,
            args.reconciliation,
            args.output,
            args.publish,
        )
    else:
        parser.error("Choose support, build or verify")
