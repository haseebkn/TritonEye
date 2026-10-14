"""Seal provisional image-first annotations; never train, score or approve models."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from agents.artifacts import REPO_ROOT, sha256_file
from agents.calibration import Calibrator
from agents.hh_hv_reservation import save
from agents.hh_hv_transfer import native_chip
from agents.source_reservation import checked, ref, verify
from agents.source_windows import PROTOCOL, SIZE, scope

CELLS = {"top_left", "top_right", "bottom_left", "bottom_right"}
FIRST_CLASSES = {"unresolved", "clutter", "coastal_structure"}
FINAL_CLASSES = FIRST_CLASSES | {"provisional_vessel", "plausible_iceberg"}


def reservation() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text())
    if (
        protocol["native_window_pixels"] != SIZE
        or protocol["scores_viewed_for_this_pass"] is not False
    ):
        raise ValueError("Unscored native preparation protocol changed")
    lock = json.loads((REPO_ROOT / protocol["reservation_lock"]).read_text())
    verify(REPO_ROOT / lock["directory"], lock["seal_sha256"])
    manifest = json.loads((REPO_ROOT / lock["directory"] / "manifest.json").read_text())
    return manifest, lock


def validate_first_pass(decisions: dict[str, Any], expected: set[str]) -> None:
    for flag in (
        "classifier_scores_viewed",
        "detector_outputs_viewed",
        "source_overlays_viewed",
        "independent_review",
    ):
        if decisions.get(flag) is not False:
            raise ValueError("Unscored image-first provisional pass required")
    if decisions.get("prior_source_exposure") is not True or not decisions.get(
        "annotator"
    ):
        raise ValueError("Source exposure and AI annotator disclosure required")
    areas = decisions["areas"]
    if len(areas) != len(expected) or {a["id"] for a in areas} != expected:
        raise ValueError("Complete unique reserved area inventory required")
    for area in areas:
        if (
            area.get("complete_visual_scan") is not True
            or not area.get("notes")
            or not area.get("chip_sha256")
        ):
            raise ValueError("Complete inspection, chip binding and notes required")
        if (
            set(area["bands_reviewed"]) != {"HH", "HV"}
            or len(area["bands_reviewed"]) != 2
            or set(area["cells_reviewed"]) != CELLS
            or len(area["cells_reviewed"]) != 4
        ):
            raise ValueError("Both bands and all four cells must be reviewed")
        ids = [o["id"] for o in area["objects"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate object identity")
        for obj in area["objects"]:
            if obj["class"] not in FIRST_CLASSES or not obj["reason"]:
                raise ValueError("Unsupported first-pass decision")
            native_chip(np.empty((2, SIZE, SIZE)), obj["bbox_px"], True)


def validate_reconciliation(
    first: dict[str, Any], reconciliation: dict[str, Any]
) -> None:
    if (
        reconciliation["first_pass_sha256"] != first["sha256"]
        or reconciliation.get("classifier_scored") is not False
        or reconciliation.get("independent_review") is not False
    ):
        raise ValueError("Reconciliation must bind the sealed unscored first pass")
    decisions = json.loads(checked(first).read_text())
    expected = {a["id"]: a for a in decisions["areas"]}
    rows = reconciliation["areas"]
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Reconciliation area inventory mismatch")
    for row in rows:
        if not row.get("source_association_notes"):
            raise ValueError("Source uncertainty and alternative explanation required")
        originals = {o["id"]: o for o in expected[row["id"]]["objects"]}
        if len(row["objects"]) != len(originals) or {
            o["id"] for o in row["objects"]
        } != set(originals):
            raise ValueError("Reconciliation cannot add or remove first-pass objects")
        for obj in row["objects"]:
            if (
                obj["class"] not in FINAL_CLASSES
                or obj["bbox_px"] != originals[obj["id"]]["bbox_px"]
                or not obj["reason"]
            ):
                raise ValueError("Reconciliation cannot move image annotations")
            if obj["class"] in {"provisional_vessel", "plausible_iceberg"} and (
                obj.get("source_consistent") is not True
                or obj.get("association_unique") is not True
            ):
                raise ValueError(
                    "Positive provisional label needs a unique "
                    "source-consistent interpretation"
                )


def check_collection(collection: dict[str, Any], local: bool = False) -> None:
    for flag in (
        "classifier_scored",
        "independent_review",
        "promotion_eligible",
        "training_approved",
        "compatibility_established",
        "annotation_completeness_verified",
    ):
        if collection.get(flag) is not False:
            raise ValueError("Provisional collection guard changed")
    expected, lock = reservation()
    if collection["reservation_seal_sha256"] != lock["seal_sha256"]:
        raise ValueError("Reservation binding changed")
    first_ref = collection["first_pass"]
    first = json.loads(checked(first_ref).read_text())
    expected_ids = {a["id"] for a in expected["areas"]}
    validate_first_pass(first, expected_ids)
    reconciliation = json.loads(checked(collection["reconciliation"]).read_text())
    validate_reconciliation(first_ref, reconciliation)
    rows = collection["areas"]
    originals = {a["id"]: a for a in expected["areas"]}
    if len(rows) != len(originals) or {a["id"] for a in rows} != set(originals):
        raise ValueError("Collection area inventory mismatch")
    objects = []
    for area in rows:
        reserved = originals[area["id"]]
        if any(
            area[k] != reserved[k]
            for k in (
                "product_id",
                "split",
                "native_acquisition_group",
                "prior_datatake_exposure",
            )
        ):
            raise ValueError("Reserved acquisition/split changed")
        if (
            area["whole_area_ignore_for_accuracy"] is not True
            or area["independent_review"] is not False
        ):
            raise ValueError("Provisional area guard changed")
        row = next(r for r in reconciliation["areas"] if r["id"] == area["id"])
        if area["objects"] != row["objects"]:
            raise ValueError("Published annotations differ from sealed decisions")
        objects.extend(area["objects"])
    if dict(Counter(o["class"] for o in objects)) != collection["class_counts"]:
        raise ValueError("Annotation counts differ")
    if local:
        replay_prepared(checked(collection["prepared"]))
        report = json.loads(checked(collection["local_report"]).read_text())
        if (
            report["first_pass"] != first_ref
            or report["reconciliation"] != collection["reconciliation"]
        ):
            raise ValueError("Local annotation lineage changed")
        for sample in report["samples"]:
            if sample.get("chip"):
                chip = np.load(checked(sample["chip"]), allow_pickle=False)
                if chip.shape != (2, 75, 75) or not np.isfinite(chip).all():
                    raise ValueError("Native sample corrupted")
        for area in report["areas"]:
            mask = np.load(checked(area["ignore_mask"]), allow_pickle=False)
            if mask.shape != (SIZE, SIZE) or not (mask == 1).all():
                raise ValueError("Provisional ignore mask changed")


def replay_prepared(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    expected, lock = reservation()
    selection = json.loads(checked(manifest["selection"]).read_text())
    if (
        selection["protocol"] != ref(PROTOCOL)
        or selection["reservation_seal_sha256"] != lock["seal_sha256"]
        or manifest["scored"] is not False
        or manifest["failures"]
    ):
        raise ValueError("Complete unscored frozen preparation required")
    private = json.loads(checked(expected["private_selection"]).read_text())
    originals = {a["id"]: a for a in private["areas"]}
    rows = [a for p in manifest["products"] for a in p["areas"]]
    if len(rows) != len(originals) or {a["id"] for a in rows} != set(originals):
        raise ValueError("Prepared inventory differs from reservation")
    for product in manifest["products"]:
        if (
            product["scored"] is not False
            or product["resampled"] is not False
            or [b["polarization"] for b in product["bands"]] != ["HH", "HV"]
        ):
            raise ValueError("Unscored native band order required")
        for area in product["areas"]:
            original = originals[area["id"]]
            if (
                any(
                    area[k] != original[k]
                    for k in (
                        "product_id",
                        "split",
                        "geographic_group",
                        "native_acquisition_group",
                        "prior_datatake_exposure",
                    )
                )
                or area["product_id"] != product["product_id"]
                or product["name"] != original["name"]
            ):
                raise ValueError("Prepared acquisition/split identity mismatch")
            scope(original, area["geometry"])
            col, row, width, height = area["window"]
            if (width, height) != (SIZE, SIZE):
                raise ValueError("Native window dimensions changed")
            with rasterio.open(checked(area["chip"])) as dataset:
                values = dataset.read()
                if values.shape != (2, SIZE, SIZE) or dataset.descriptions != (
                    "sigma0_db_hh",
                    "sigma0_db_hv",
                ):
                    raise ValueError("Calibrated band layout mismatch")
            checked(area["preview"])
            for index, band in enumerate(product["bands"]):
                checked(band["annotation"])
                dn = np.load(checked(area["bands"][index]["dn"]), allow_pickle=False)
                calibrated = (
                    Calibrator(str(checked(band["calibration"])))
                    .to_sigma0_db(dn, row, col)
                    .astype("float32")
                )
                if (
                    dn.shape != (SIZE, SIZE)
                    or not (dn > 0).all()
                    or not np.array_equal(values[index], calibrated)
                    or not np.isfinite(calibrated).all()
                ):
                    raise ValueError("Native calibration replay failed")
    return rows


def build(
    prepared: Path,
    first_path: Path,
    reconciliation_path: Path,
    output: Path,
    publish: Path,
) -> None:
    if (
        output.exists()
        or publish.exists()
        or not output.resolve().is_relative_to((REPO_ROOT / "data").resolve())
        or publish.resolve().parent
        != (REPO_ROOT / "datasets/source_supported").resolve()
    ):
        raise ValueError("Fresh project-local data and separate release required")
    rows = replay_prepared(prepared)
    first_ref = ref(first_path)
    first = json.loads(first_path.read_text())
    validate_first_pass(first, {a["id"] for a in rows})
    reconciliation = json.loads(reconciliation_path.read_text())
    validate_reconciliation(first_ref, reconciliation)
    final = {a["id"]: a for a in reconciliation["areas"]}
    first_rows = {a["id"]: a for a in first["areas"]}
    output.mkdir(parents=True)
    public, local, samples = [], [], []
    for area in rows:
        if first_rows[area["id"]]["chip_sha256"] != area["chip"]["sha256"]:
            raise ValueError("Image-first annotation chip binding mismatch")
        with rasterio.open(checked(area["chip"])) as dataset:
            values = dataset.read()
        mask_path = output / f"{area['id']}_ignore.npy"
        np.save(mask_path, np.ones((SIZE, SIZE), dtype="uint8"), allow_pickle=False)
        local.append({**area, "ignore_mask": ref(mask_path)})
        for obj in final[area["id"]]["objects"]:
            chip = native_chip(
                values, obj["bbox_px"], obj["class"] in {"clutter", "coastal_structure"}
            )
            sample = {
                "area_id": area["id"],
                **obj,
                "provisional": True,
                "ignore_for_accuracy": True,
                "independent_review": False,
                "chip": None,
                "status": "edge_or_size_abstention",
            }
            if chip is not None:
                cp = output / f"{area['id']}_{obj['id']}_75.npy"
                np.save(cp, chip, allow_pickle=False)
                sample.update(chip=ref(cp), status="native_context_extracted")
            samples.append(sample)
        public.append(
            {
                k: area[k]
                for k in (
                    "id",
                    "product_id",
                    "split",
                    "native_acquisition_group",
                    "prior_datatake_exposure",
                    "native_context",
                )
            }
        )
        public[-1].update(
            native_pixels=SIZE,
            chip_sha256=area["chip"]["sha256"],
            objects=final[area["id"]]["objects"],
            whole_area_ignore_for_accuracy=True,
            independent_review=False,
        )
    report = {
        "prepared": ref(prepared),
        "first_pass": first_ref,
        "reconciliation": ref(reconciliation_path),
        "areas": local,
        "samples": samples,
        "scored": False,
        "calibration_bands_replayed_exactly": len(rows) * 2,
    }
    save(output / "report.json", report)
    _, lock = reservation()
    collection = {
        "version": "0.1.0",
        "protocol": ref(PROTOCOL),
        "prepared": ref(prepared),
        "first_pass": first_ref,
        "reconciliation": ref(reconciliation_path),
        "local_report": ref(output / "report.json"),
        "reservation_seal_sha256": lock["seal_sha256"],
        "areas": public,
        "class_counts": dict(
            Counter(o["class"] for a in final.values() for o in a["objects"])
        ),
        "classifier_scored": False,
        "independent_review": False,
        "promotion_eligible": False,
        "training_approved": False,
        "compatibility_established": False,
        "annotation_completeness_verified": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    check_collection(collection, True)
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
                "areas": len(rows),
                "class_counts": collection["class_counts"],
                "calibration_bands_replayed_exactly": len(rows) * 2,
                "samples_extracted": sum(s["chip"] is not None for s in samples),
                "seal": ref(publish / "seal.json"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--first-pass", type=Path)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.verify and args.seal_sha256:
        if sha256_file(args.verify / "seal.json") != args.seal_sha256:
            raise ValueError("Trusted annotation seal changed")
        seal = json.loads((args.verify / "seal.json").read_text())
        check_collection(
            json.loads(checked(seal["collection"]).read_text()), args.local
        )
        print(json.dumps({"verified": True, "local": args.local, "scored": False}))
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
        parser.error("Use complete build arguments or --verify and --seal-sha256")
