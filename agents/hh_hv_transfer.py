"""Unscored, provisional transfer collection and descriptive compatibility audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import label, median_filter

from agents.artifacts import REPO_ROOT, sha256_file
from agents.calibration import Calibrator
from agents.hh_hv_reservation import load_reservation, require_scope, save
from agents.nl_benchmark import require_development_scene
from agents.reserved_sar import PROTOCOL, reference

Array = np.ndarray[Any, Any]
DECISIONS = REPO_ROOT / "configs/reservations/hh_hv_first_pass_v010.json"


def verified(ref: dict[str, Any]) -> Path:
    path = (REPO_ROOT / str(ref["path"])).resolve()
    if (
        not path.is_relative_to(REPO_ROOT.resolve())
        or sha256_file(path) != ref["sha256"]
    ):
        raise ValueError("Artifact integrity mismatch")
    return path


def native_chip(values: Array, bbox: list[int], context: bool = False) -> Array | None:
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= values.shape[2] and 0 <= y0 < y1 <= values.shape[1]):
        raise ValueError("Invalid annotation bounds")
    x, y = (x0 + x1) // 2 - 37, (y0 + y1) // 2 - 37
    if (
        (not context and max(x1 - x0, y1 - y0) > 75)
        or min(x, y) < 0
        or x + 75 > values.shape[2]
        or y + 75 > values.shape[1]
    ):
        return None
    return values[:, y : y + 75, x : x + 75].copy()


def extent(values: Array) -> dict[str, Any] | None:
    """Frozen descriptive proxy; never a class label or hull measurement."""
    if values.shape != (2, 75, 75) or not np.isfinite(values).all():
        raise ValueError("Finite native two-band 75px chip required")
    border = np.ones((75, 75), dtype=bool)
    border[10:-10, 10:-10] = False
    background = np.median(values[:, border], axis=1)
    smooth = median_filter(values, size=(1, 3, 3))
    components, count = label(
        np.any(smooth > background[:, None, None] + 6, axis=0),
        structure=np.ones((3, 3)),
    )
    choices = []
    for index in range(1, count + 1):
        yy, xx = np.where(components == index)
        distance = float(np.sqrt((xx - 37) ** 2 + (yy - 37) ** 2).min())
        if distance <= 12:
            choices.append((distance, index, xx, yy))
    if not choices:
        return None
    distance, _, xx, yy = min(choices, key=lambda item: (item[0], item[1]))
    return {
        "width_px": int(xx.max() - xx.min() + 1),
        "height_px": int(yy.max() - yy.min() + 1),
        "area_px": len(xx),
        "center_distance_px": distance,
        "touches_edge": bool(
            xx.min() == 0 or yy.min() == 0 or xx.max() == 74 or yy.max() == 74
        ),
    }


def summary(chips: list[Array]) -> dict[str, Any]:
    proxies = [extent(chip) for chip in chips]
    valid = [p for p in proxies if p is not None]
    return {
        "chips": len(chips),
        "value_quantiles_01_50_99_db": (
            np.quantile(np.stack(chips), [0.01, 0.5, 0.99], axis=(0, 2, 3)).tolist()
            if chips
            else []
        ),
        "proxy_found": len(valid),
        "proxy_edge_count": sum(p["touches_edge"] for p in valid),
        "proxy_extent_quantiles_10_50_90_px": (
            {
                key: np.quantile([p[key] for p in valid], [0.1, 0.5, 0.9]).tolist()
                for key in ("width_px", "height_px", "area_px")
            }
            if valid
            else {}
        ),
    }


def training_reference() -> dict[str, Any]:
    ledger = json.loads(
        (REPO_ROOT / "docs/evidence/statoil_baseline_20261013.json").read_text()
    )
    report_path = verified(
        {"path": ledger["full_report_path"], "sha256": ledger["full_report_sha256"]}
    )
    split_path = report_path.parent / "split.json"
    if (
        sha256_file(split_path)
        != "1f04bc422e61e1980d39ee38d26d89694aa40edf0660515547743ca2b1fdea8d"
    ):
        raise ValueError("Frozen Kaggle split changed")
    split = json.loads(split_path.read_text())
    source = verified(
        {"path": ledger["provenance"]["source_path"], "sha256": split["source_sha256"]}
    )
    ids = {row["id"] for row in split["rows"] if row["split"] == "train"}
    chips: dict[str, list[Array]] = {"ship": [], "iceberg": []}
    seen = set()
    for row in json.loads(source.read_text()):
        if row["id"] not in ids:
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate training ID")
        seen.add(row["id"])
        values = np.asarray([row["band_1"], row["band_2"]], dtype="float32").reshape(
            2, 75, 75
        )
        chips["iceberg" if row["is_iceberg"] else "ship"].append(values)
    if seen != ids or len(ids) != 1122:
        raise ValueError("Incomplete frozen training reference")
    return {
        "source": reference(source),
        "split": reference(split_path),
        "partition": "train_only",
        "classes": {key: summary(value) for key, value in chips.items()},
        "physical_spacing_known": False,
        "source_preprocessing_equivalence_established": False,
    }


def validate_decisions(decisions: dict[str, Any], ids: set[str]) -> None:
    if (
        decisions["classifier_scores_viewed"] is not False
        or decisions["detector_outputs_viewed"] is not False
        or decisions["independent_review"] is not False
    ):
        raise ValueError("Unscored provisional first pass required")
    rows = decisions["regions"]
    if len(rows) != len(ids) or {r["id"] for r in rows} != ids:
        raise ValueError("Complete unique area decisions required")
    for row in rows:
        if not row["notes"]:
            raise ValueError("Inspection notes required")
        for item in row["objects"] + row["contexts"]:
            if (
                item["class"] not in {"unresolved", "coastal_structure", "clutter"}
                or not item["reason"]
            ):
                raise ValueError("Unsupported first-pass class")
            native_chip(np.empty((2, 512, 512)), item["bbox_px"], True)


def build(prepared: Path, output: Path) -> None:
    if output.exists() or not output.resolve().is_relative_to(
        (REPO_ROOT / "data").resolve()
    ):
        raise ValueError("Fresh local data output required")
    manifest = json.loads(prepared.read_text())
    selection = json.loads(verified(manifest["selection"]).read_text())
    if selection["protocol_sha256"] != sha256_file(PROTOCOL) or selection[
        "protocol"
    ] != json.loads(PROTOCOL.read_text()):
        raise ValueError("Frozen protocol mismatch")
    lock = json.loads(
        (REPO_ROOT / selection["protocol"]["reservation_lock"]).read_text()
    )
    reservation = load_reservation(REPO_ROOT / lock["directory"], lock["seal_sha256"])
    expected = {s["product_id"]: s for s in reservation["scenes"]}
    if (
        manifest["scored"] is not False
        or manifest["failures"]
        or len(manifest["scenes"]) != len(expected)
        or {r["scene"]["product_id"] for r in manifest["scenes"]} != set(expected)
    ):
        raise ValueError("Complete unscored reservation required")
    decisions = json.loads(DECISIONS.read_text())
    ids = {r["id"] for s in manifest["scenes"] for r in s["rois"]}
    validate_decisions(decisions, ids)
    output.mkdir(parents=True)
    save(
        output / "annotation_seal.json",
        {
            "decisions": reference(DECISIONS),
            "prepared": reference(prepared),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "classifier_scored": False,
            "prior_exposure": decisions["prior_exposure"],
        },
    )
    decisions_by_id = {r["id"]: r for r in decisions["regions"]}
    rows: list[dict[str, Any]] = []
    areas: list[dict[str, Any]] = []
    object_chips: list[Array] = []
    context_chips: list[Array] = []
    replayed = 0
    for record in manifest["scenes"]:
        scene = record["scene"]
        if (
            scene != expected[scene["product_id"]]
            or record["scored"] is not False
            or record["resampled"] is not False
        ):
            raise ValueError("Reserved identity or unscored state changed")
        for roi in record["rois"]:
            require_scope(
                reservation,
                {**scene, "polarizations": ["hh", "hv"]},
                {**roi, "group": scene["geographic_group"]},
            )
            require_development_scene(
                scene["product_id"], {**scene, "study_roi": roi["geometry"]}
            )
            with rasterio.open(verified(roi["chip"])) as dataset:
                values = dataset.read()
            col, row, _, _ = roi["window"]
            for index, band in enumerate(record["native_bands"]):
                verified(band["annotation"])
                dn = np.load(verified(roi["bands"][index]["dn"]), allow_pickle=False)
                calibrated = (
                    Calibrator(str(verified(band["calibration"])))
                    .to_sigma0_db(dn, row, col)
                    .astype("float32")
                )
                if (
                    not np.array_equal(values[index], calibrated)
                    or not np.isfinite(values[index]).all()
                ):
                    raise ValueError("Native calibration replay mismatch")
                replayed += 1
            verified(roi["preview"])
            areas.append(
                {
                    "id": roi["id"],
                    "product_id": scene["product_id"],
                    "split": scene["split"],
                    "group": scene["geographic_group"],
                    "acquisition_time": scene["acquisition_time"],
                    "geometry": roi["geometry"],
                    "window": roi["window"],
                    "chip": roi["chip"],
                    "preview": roi["preview"],
                    "native_bands": record["native_bands"],
                    "notes": decisions_by_id[roi["id"]]["notes"],
                }
            )
            for kind in ("objects", "contexts"):
                for index, item in enumerate(decisions_by_id[roi["id"]][kind]):
                    name = f"{roi['id']}_{kind}_{index}"
                    chip = native_chip(values, item["bbox_px"], kind == "contexts")
                    result = {
                        "id": name,
                        "area_id": roi["id"],
                        "kind": kind,
                        **item,
                        "provisional": True,
                        "ignore_for_accuracy": True,
                        "independent_review": False,
                        "status": (
                            "boundary_or_size_abstention"
                            if chip is None
                            else "extracted"
                        ),
                    }
                    if chip is not None:
                        path = output / f"{name}.npy"
                        np.save(path, chip, allow_pickle=False)
                        result.update(
                            chip=reference(path), apparent_extent=extent(chip)
                        )
                        (object_chips if kind == "objects" else context_chips).append(
                            chip
                        )
                    rows.append(result)
    report = {
        "version": "0.1.0",
        "annotation_seal": reference(output / "annotation_seal.json"),
        "protocol": reference(PROTOCOL),
        "decisions": reference(DECISIONS),
        "prepared": reference(prepared),
        "reservation_seal_sha256": lock["seal_sha256"],
        "areas": areas,
        "samples": rows,
        "calibration_bands_replayed_exactly": replayed,
        "object_chips": summary(object_chips),
        "context_chips": summary(context_chips),
        "kaggle_training_reference": training_reference(),
        "classifier_scored": False,
        "promotion_eligible": False,
        "independent_review": False,
        "confirmed_vessels": 0,
        "plausible_icebergs": 0,
        "compatibility_established": False,
        "limitations": [
            "AI first pass, prior project label exposure, no independent reviewer",
            "Unresolved objects ignored; contexts are not verified negatives",
            "No defensible vessel or iceberg labels in the reserved crops",
            "Pixel extent is not physical target size; "
            "Kaggle spacing/preprocessing unknown",
            "Partial source reads do not verify complete source TIFF checksums",
            "Winter/year/platform confounded; "
            "512px crops do not cover full reserved envelopes",
            "No classifier performance, false-positive reduction "
            "or vessel recall measured",
        ],
    }
    save(output / "report.json", report)
    print(
        json.dumps(
            {
                "areas": len(areas),
                "samples": len(rows),
                "calibration_replays": replayed,
                "object_chips": len(object_chips),
                "context_chips": len(context_chips),
                "report": str(output / "report.json"),
            }
        )
    )


def publish(report_path: Path, destination: Path) -> None:
    """Publish metadata only; raw chips stay in ignored project data."""
    destination = destination.resolve()
    if (
        destination.exists()
        or destination.parent != (REPO_ROOT / "datasets/hh_hv_transfer").resolve()
    ):
        raise ValueError("Fresh separate transfer release required")
    report = json.loads(report_path.read_text())
    check_report(report, local=True)
    destination.mkdir(parents=True)
    save(destination / "collection.json", report)
    save(
        destination / "seal.json",
        {
            "collection": reference(destination / "collection.json"),
            "source_report": reference(report_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "provisional_unscored_not_benchmark",
        },
    )
    print(json.dumps(reference(destination / "seal.json")))


def check_report(report: dict[str, Any], local: bool = False) -> None:
    """Check metadata everywhere; optionally verify local native/chip bytes."""
    for flag in (
        "classifier_scored",
        "promotion_eligible",
        "independent_review",
        "compatibility_established",
    ):
        if report[flag] is not False:
            raise ValueError("Provisional unscored guard changed")
    protocol = json.loads(verified(report["protocol"]).read_text())
    decisions = json.loads(verified(report["decisions"]).read_text())
    lock = json.loads((REPO_ROOT / protocol["reservation_lock"]).read_text())
    reservation = load_reservation(REPO_ROOT / lock["directory"], lock["seal_sha256"])
    scenes = {s["product_id"]: s for s in reservation["scenes"]}
    areas = report["areas"]
    if (
        len(areas) != 16
        or len({a["id"] for a in areas}) != 16
        or {a["product_id"] for a in areas} != set(scenes)
    ):
        raise ValueError("Incomplete area inventory")
    validate_decisions(decisions, {a["id"] for a in areas})
    for area in areas:
        scene = scenes[area["product_id"]]
        if (
            area["split"] != scene["split"]
            or area["group"] != scene["geographic_group"]
        ):
            raise ValueError("Reserved split changed")
        regime = area["id"].rsplit("_", 1)[1]
        require_scope(
            reservation,
            {**scene, "polarizations": ["hh", "hv"]},
            {"group": area["group"], "regime": regime, "geometry": area["geometry"]},
        )
        require_development_scene(
            scene["product_id"], {**scene, "study_roi": area["geometry"]}
        )
    for sample in report["samples"]:
        if (
            sample["ignore_for_accuracy"] is not True
            or sample["provisional"] is not True
            or sample["independent_review"] is not False
        ):
            raise ValueError("Sample review guard changed")
    expected_samples = {
        f"{area['id']}_{kind}_{index}": (area["id"], kind, item)
        for area in decisions["regions"]
        for kind in ("objects", "contexts")
        for index, item in enumerate(area[kind])
    }
    if len(report["samples"]) != len(expected_samples) or {
        sample["id"] for sample in report["samples"]
    } != set(expected_samples):
        raise ValueError("Incomplete first-pass sample inventory")
    for sample in report["samples"]:
        area_id, kind, item = expected_samples[sample["id"]]
        if (
            sample["area_id"] != area_id
            or sample["kind"] != kind
            or any(sample[key] != value for key, value in item.items())
        ):
            raise ValueError("First-pass decision changed")
    if local:

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    verified(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(report)
        visit(json.loads(verified(report["prepared"]).read_text()))
        visit(json.loads(verified(report["annotation_seal"]).read_text()))


def verify_release(directory: Path, seal_sha256: str, local: bool = False) -> None:
    if sha256_file(directory / "seal.json") != seal_sha256:
        raise ValueError("Release seal mismatch")
    seal = json.loads((directory / "seal.json").read_text())
    report = json.loads(verified(seal["collection"]).read_text())
    check_report(report, local)
    print(json.dumps({"verified": True, "local_artifacts": local, "scored": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.verify and args.seal_sha256:
        verify_release(args.verify, args.seal_sha256, args.local)
    elif args.publish and args.report:
        publish(args.report, args.publish)
    elif args.prepared and args.output:
        build(args.prepared, args.output)
    else:
        parser.error("Choose build, publish or verify with required paths")


if __name__ == "__main__":
    main()
