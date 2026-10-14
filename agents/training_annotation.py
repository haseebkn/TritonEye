"""Image-first provisional annotations and source evidence for training windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw
from rasterio.transform import GCPTransformer
from shapely.geometry import shape

from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT
from agents.association import GEOD
from agents.hh_hv_reservation import save
from agents.iceberg_corroboration import (
    optical_query,
    optical_summary,
    query_optical,
    repeat_context,
)
from agents.native_nl_input import prepare as native_input
from agents.provisional_training import verify as verify_reservation
from agents.source_annotations import (
    public_metadata,
    validate_first_pass,
    validate_reconciliation,
)
from agents.source_followup import fresh
from agents.source_followup_annotations import validate_support
from agents.source_reservation import checked, ref
from agents.source_supported import CATALOGUE
from agents.source_windows import SIZE, metadata
from agents.training_windows import PROTOCOL, replay, selection

FIRST = REPO_ROOT / "configs/reservations/training_first_pass_v010.json"
RECONCILIATION = REPO_ROOT / "configs/reservations/training_reconciliation_v010.json"
DIRECTORY = REPO_ROOT / "datasets/native_training/v0.1.0"
LOCK = REPO_ROOT / "configs/reservations/training_collection_v010.lock.json"
FLAGS = {
    "classifier_scored": False,
    "detector_scored": False,
    "independent_review": False,
    "training_approved": False,
    "promotion_eligible": False,
    "annotation_completeness_verified": False,
}


def review(prepared: Path, output: Path) -> None:
    """Display-only nearest-neighbour enlargements; input values stay unchanged."""
    rows = replay(prepared)
    fresh(output)
    receipts = []
    for row in rows:
        with rasterio.open(checked(row["chip"])) as ds:
            values = ds.read()
        for band, pol in enumerate(("HH", "HV")):
            low, high = np.percentile(values[band], [0.5, 99.5])
            grey = np.clip((values[band] - low) / (high - low) * 255, 0, 255).astype(
                "uint8"
            )
            canvas = Image.new("RGB", (800, 830), "#182128")
            draw = ImageDraw.Draw(canvas)
            for index, (x, y) in enumerate(((0, 0), (192, 0), (0, 192), (192, 192))):
                tile = Image.fromarray(grey[y : y + 192, x : x + 192]).resize(
                    (384, 384), Image.Resampling.NEAREST
                )
                left, top = (index % 2) * 400, (index // 2) * 415
                canvas.paste(tile.convert("RGB"), (left, top + 30))
                draw.text(
                    (left + 5, top + 5),
                    f"{pol}: x={x}..{x+191}, y={y}..{y+191}",
                    fill="white",
                )
            path = output / f"{row['id']}_{pol}_quadrants.png"
            canvas.save(path)
            receipts.append(
                {
                    "area_id": row["id"],
                    "polarization": pol,
                    "image": ref(path),
                    "display_percentiles": [0.5, 99.5],
                    "display_bounds_db": [float(low), float(high)],
                    "resampling": "nearest_display_only",
                    "source_chip": row["chip"],
                }
            )
    save(
        output / "review.json",
        {
            "prepared": ref(prepared),
            "views": receipts,
            "classifier_scored": False,
            "detector_scored": False,
        },
    )


def first_pass(prepared: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = replay(prepared)
    first = json.loads(FIRST.read_text())
    validate_first_pass(first, {a["id"] for a in rows})
    if parse_utc(first.get("completed_at_utc")) is None:
        raise ValueError("Sealed annotation time required")
    for row in rows:
        decision = next(a for a in first["areas"] if a["id"] == row["id"])
        if decision["chip_sha256"] != row["chip"]["sha256"]:
            raise ValueError("First-pass pixel binding changed")
    return rows, first


def object_position(
    obj: dict[str, Any], row: dict[str, Any], info: dict[str, Any], gcps: list[Any]
) -> tuple[float, float, datetime]:
    x0, y0, x1, y1 = obj["bbox_px"]
    col, line = row["window"][0] + (x0 + x1) / 2, row["window"][1] + (y0 + y1) / 2
    with GCPTransformer(gcps, tps=True) as tx:
        xs, ys = tx.xy([line], [col], offset="ul")
    start = datetime.fromisoformat(info["native_start_time"].replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return (
        float(xs[0]),
        float(ys[0]),
        start + timedelta(seconds=line * info["azimuth_interval_s"]),
    )


def source_matches(
    lon: float,
    lat: float,
    instant: datetime,
    source: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = []
    for obs in observations:
        delta = abs(instant.timestamp() - obs["time"])
        if (
            delta > 1800
            or obs["method"] not in {"VIS", "R/V"}
            or re.match(r"^(S1[ABCD]|SN|RS|RCM|SENT|TSX|ENVI)", obs["source"])
        ):
            continue
        distance = GEOD.inv(lon, lat, obs["lon"], obs["lat"])[2]
        if distance <= 300 + delta:
            matches.append(
                {
                    "source_identity": obs["iceberg_id"],
                    "distance_m": distance,
                    "absolute_offset_s": delta,
                    "selected_identity": obs["iceberg_id"] == source["iceberg_id"],
                }
            )
    offset = abs(instant.timestamp() - source["time"])
    distance = GEOD.inv(lon, lat, source["lon"], source["lat"])[2]
    return {
        "selected_source_distance_m": distance,
        "absolute_offset_s": offset,
        "frozen_allowance_m": 300 + offset,
        "inside_selected_allowance": distance <= 300 + offset,
        "native_time_utc": instant.isoformat(),
        "compatible_iceberg_identities": sorted(
            {m["source_identity"] for m in matches}
        ),
        "matches": matches,
    }


def window_query(
    row: dict[str, Any], original: dict[str, Any], hours: float
) -> dict[str, str]:
    instant = datetime.fromisoformat(row["native_context"]["center_native_time_utc"])
    query = optical_query(original["lon"], original["lat"], instant, hours)
    query["$filter"] = (
        query["$filter"].rsplit("and OData.CSC.Intersects", 1)[0]
        + "and OData.CSC.Intersects(area=geography'SRID=4326;"
        + f"{shape(row['geometry']).wkt}')"
    )
    return query


def evidence(prepared: Path, output: Path) -> None:
    rows, first = first_pass(prepared)
    _, private = selection()
    preparation = json.loads(prepared.read_text())
    ranking = json.loads(checked(private["inputs"]["iip"]).read_text())
    normalized_path = checked(ranking["provenance"]["normalized"])
    normalized = json.loads(normalized_path.read_text())
    for source in normalized["sources"]:
        checked(source)
    rules = json.loads(PROTOCOL.read_text())
    fresh(output)
    results = []
    for row in rows:
        original = next(a for a in private["areas"] if a["id"] == row["id"])
        product = next(
            p for p in preparation["products"] if p["product_id"] == row["product_id"]
        )
        info, gcps = metadata(
            checked(product["bands"][0]["annotation"]).read_bytes(), original, "hh"
        )
        # Query the complete frozen window, not only the source centre.
        query = window_query(row, original, rules["optical_window_hours"])
        receipt = query_optical(query, output / f"{row['id']}_optical.json")
        decision = next(a for a in first["areas"] if a["id"] == row["id"])
        objects = []
        for obj in decision["objects"]:
            lon, lat, instant = object_position(obj, row, info, gcps)
            objects.append(
                {
                    "object_id": obj["id"],
                    "lon": lon,
                    "lat": lat,
                    **source_matches(
                        lon,
                        lat,
                        instant,
                        original["source"],
                        normalized["observations"],
                    ),
                    "repeat_reports": repeat_context(
                        original["source"],
                        normalized["observations"],
                        instant.timestamp(),
                        lon,
                        lat,
                        rules["repeat_iip_window_hours"] * 3600,
                    ),
                    "optical": optical_summary(receipt, lon, lat),
                    "additional_physical_corroboration": False,
                }
            )
        results.append(
            {
                "id": row["id"],
                "objects": objects,
                "source_association_abstain": row["source_coverage"][
                    "source_association_abstain"
                ],
                "optical_receipt": receipt,
                "source_position_optical": optical_summary(
                    receipt, original["lon"], original["lat"]
                ),
                "independent_confirmation": False,
            }
        )
        print(
            json.dumps(
                {
                    "evidence": row["id"],
                    "objects": len(objects),
                    "optical_status": receipt["status"],
                }
            ),
            flush=True,
        )
    archive_dates = sorted(
        p.stem.removeprefix("ais_stream_")
        for p in (REPO_ROOT / "data/raw/ais_stream").glob("ais_stream_*.csv")
    )
    historical = sorted(
        p.name.removeprefix("ais-") for p in (REPO_ROOT / "data/raw").glob("ais-20*")
    )
    save(
        output / "report.json",
        {
            "prepared": ref(prepared),
            "first_pass": ref(FIRST),
            "normalized_iip": ref(normalized_path),
            "areas": results,
            "ais_date_inventory": {
                "archive": archive_dates,
                "historical_directories": historical,
                "inspection": "Names only; receiver coverage not established",
                "potential_cotemporal_archive": any(
                    d.startswith(("2019", "2020")) for d in archive_dates + historical
                ),
            },
            "labels_assigned_automatically": False,
            **FLAGS,
        },
    )


def validate(value: dict[str, Any]) -> None:
    checked(value["protocol"])
    verify_reservation()
    reservation = json.loads(
        (REPO_ROOT / "reservations/native_training/v0.1.0/manifest.json").read_text()
    )
    for key, expected in FLAGS.items():
        if value.get(key) is not expected:
            raise ValueError("Provisional collection guard changed")
    if (
        value.get("exploratory_training_admitted") is not False
        or value.get("accuracy_truth_admitted") is not False
    ):
        raise ValueError("Collection cannot admit training or accuracy truth")
    public_metadata(value)
    first = json.loads(checked(value["first_pass"]).read_text())
    reconciliation = json.loads(checked(value["reconciliation"]).read_text())
    validate_first_pass(first, {a["id"] for a in value["areas"]})
    validate_reconciliation(value["first_pass"], reconciliation)
    if len(value["areas"]) != 2 or len({a["id"] for a in value["areas"]}) != 2:
        raise ValueError("Exact two-area collection required")
    for area in value["areas"]:
        prior = next(a for a in first["areas"] if a["id"] == area["id"])
        decision = next(a for a in reconciliation["areas"] if a["id"] == area["id"])
        if (
            area["split"] != "train"
            or area["independent_review"] is not False
            or area["whole_area_ignore_for_accuracy"] is not True
            or area["objects"] != decision["objects"]
            or area["chip_sha256"] != prior["chip_sha256"]
        ):
            raise ValueError("Provisional annotation identity or role changed")
        original = next(
            (a for a in reservation["areas"] if a["id"] == area["id"]), None
        )
        if original is None or any(
            area[k] != original[k]
            for k in (
                "product_id",
                "split",
                "geographic_group",
                "native_acquisition_group",
            )
        ):
            raise ValueError("Collection differs from reserved acquisition or group")
        if any(
            o.get("exploratory_training_admitted") is not False
            or o.get("additional_physical_corroboration") is not False
            for o in area["objects"]
        ):
            raise ValueError("Uncorroborated objects cannot be training admissions")
        if area["source_coverage"]["source_association_abstain"] and any(
            o["class"] in {"plausible_iceberg", "provisional_vessel"}
            for o in area["objects"]
        ):
            raise ValueError("Incomplete coverage cannot assign source class")
    if value["class_counts"] != dict(
        Counter(o["class"] for a in value["areas"] for o in a["objects"])
    ):
        raise ValueError("Annotation counts changed")


def input_metadata(product: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    bands = product["bands"]
    keys = (
        "mission",
        "mode",
        "range_spacing_m",
        "azimuth_spacing_m",
        "thermal_noise_corrected",
    )
    if [b["polarization"] for b in bands] != ["HH", "HV"] or any(
        bands[0][k] != bands[1][k] for k in keys
    ):
        raise ValueError("Native HH/HV input metadata disagree")
    if bands[0]["thermal_noise_corrected"] not in {"false", "true"}:
        raise ValueError("Explicit native thermal-noise flag required")
    return {
        "mission": bands[0]["mission"],
        "mode": bands[0]["mode"],
        "product_type": "GRD",
        "band_order": ["HH", "HV"],
        "unit": "sigma0_db",
        "pixel_spacing_m": [bands[0]["range_spacing_m"], bands[0]["azimuth_spacing_m"]],
        "thermal_noise_corrected": bands[0]["thermal_noise_corrected"] != "false",
        "incidence_center_deg": row["native_context"]["center_incidence_deg"],
    }


def verify_source(prepared: Path, source: dict[str, Any]) -> None:
    if (
        any(source.get(k) is not v for k, v in FLAGS.items())
        or source.get("labels_assigned_automatically") is not False
    ):
        raise ValueError("Unscored provisional source evidence required")
    rows, first = first_pass(prepared)
    _, private = selection()
    preparation = json.loads(prepared.read_text())
    ranking = json.loads(checked(private["inputs"]["iip"]).read_text())
    if (
        source["normalized_iip"] != ranking["provenance"]["normalized"]
        or source["first_pass"] != ref(FIRST)
        or source["prepared"] != ref(prepared)
    ):
        raise ValueError("Source evidence lineage changed")
    normalized = json.loads(checked(source["normalized_iip"]).read_text())
    for reference in normalized["sources"]:
        checked(reference)
    if len(source["areas"]) != 2 or {a["id"] for a in source["areas"]} != {
        r["id"] for r in rows
    }:
        raise ValueError("Source evidence area inventory changed")
    rules = json.loads(PROTOCOL.read_text())
    for row in rows:
        original = next(a for a in private["areas"] if a["id"] == row["id"])
        product = next(
            p for p in preparation["products"] if p["product_id"] == row["product_id"]
        )
        info, gcps = metadata(
            checked(product["bands"][0]["annotation"]).read_bytes(), original, "hh"
        )
        support = next(a for a in source["areas"] if a["id"] == row["id"])
        receipt = support["optical_receipt"]
        page = json.loads(checked(receipt["response"]).read_text())
        status = (
            "failed"
            if page.get("failed") is True
            else ("truncated" if page.get("@odata.nextLink") else "complete")
        )
        if (
            status != receipt["status"]
            or support["independent_confirmation"] is not False
            or support["source_association_abstain"]
            != row["source_coverage"]["source_association_abstain"]
        ):
            raise ValueError("Source uncertainty or optical query status changed")
        if status != "failed":
            expected_url = (
                requests.Request(
                    "GET",
                    CATALOGUE,
                    params=window_query(row, original, rules["optical_window_hours"]),
                )
                .prepare()
                .url
            )
            completed = parse_utc(first["completed_at_utc"])
            retrieved = parse_utc(receipt.get("retrieved_at"))
            if (
                receipt["url"] != expected_url
                or completed is None
                or retrieved is None
                or retrieved <= completed
            ):
                raise ValueError("Optical query scope or prescore chronology changed")
        decision = next(a for a in first["areas"] if a["id"] == row["id"])
        expected = []
        for obj in decision["objects"]:
            lon, lat, instant = object_position(obj, row, info, gcps)
            expected.append(
                {
                    "object_id": obj["id"],
                    "lon": lon,
                    "lat": lat,
                    **source_matches(
                        lon,
                        lat,
                        instant,
                        original["source"],
                        normalized["observations"],
                    ),
                    "repeat_reports": repeat_context(
                        original["source"],
                        normalized["observations"],
                        instant.timestamp(),
                        lon,
                        lat,
                        rules["repeat_iip_window_hours"] * 3600,
                    ),
                    "optical": optical_summary(receipt, lon, lat),
                    "additional_physical_corroboration": False,
                }
            )
        if expected != support["objects"] or support[
            "source_position_optical"
        ] != optical_summary(receipt, original["lon"], original["lat"]):
            raise ValueError("Object association or corroboration replay differs")


def build(prepared: Path, source_path: Path, output: Path) -> None:
    rows, first = first_pass(prepared)
    source = json.loads(source_path.read_text())
    verify_source(prepared, source)
    if (
        source["prepared"] != ref(prepared)
        or source["first_pass"] != ref(FIRST)
        or any(source.get(k) is not v for k, v in FLAGS.items())
    ):
        raise ValueError("Unscored source report binding changed")
    reconciliation = json.loads(RECONCILIATION.read_text())
    validate_reconciliation(ref(FIRST), reconciliation)
    if reconciliation["private_source_evidence"] != ref(source_path):
        raise ValueError("Frozen source evidence required")
    validate_support(reconciliation["areas"], source)
    if DIRECTORY.exists() or LOCK.exists():
        raise ValueError("Fresh versioned training collection required")
    fresh(output)
    preparation = json.loads(prepared.read_text())
    areas, samples = [], []
    for row in rows:
        objects = next(
            a["objects"] for a in reconciliation["areas"] if a["id"] == row["id"]
        )
        product = next(
            p for p in preparation["products"] if p["product_id"] == row["product_id"]
        )
        info = input_metadata(product, row)
        with rasterio.open(checked(row["chip"])) as ds:
            values = ds.read()
        mask_path = output / f"{row['id']}_ignore.npy"
        np.save(mask_path, np.ones((SIZE, SIZE), dtype="uint8"), allow_pickle=False)
        for obj in objects:
            result = native_input(values, obj["bbox_px"], info, "train")
            tensor = result.pop("tensor")
            path = output / f"{row['id']}_{obj['id']}_native75.npy"
            if tensor is not None:
                np.save(path, tensor, allow_pickle=False)
            samples.append(
                {
                    "area_id": row["id"],
                    "object_id": obj["id"],
                    "input": result,
                    "chip": ref(path) if tensor is not None else None,
                    "tensor_sha256_float32_c_order": (
                        hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
                        if tensor is not None
                        else None
                    ),
                }
            )
        support = next(a for a in source["areas"] if a["id"] == row["id"])
        summaries = [
            {
                k: o[k]
                for k in (
                    "object_id",
                    "selected_source_distance_m",
                    "absolute_offset_s",
                    "frozen_allowance_m",
                    "inside_selected_allowance",
                    "native_time_utc",
                    "additional_physical_corroboration",
                )
            }
            | {
                "compatible_identity_count": len(o["compatible_iceberg_identities"]),
                "repeat_report_count": len(o["repeat_reports"]),
                "optical": o["optical"],
            }
            for o in support["objects"]
        ]
        areas.append(
            {
                k: row[k]
                for k in (
                    "id",
                    "product_id",
                    "split",
                    "geographic_group",
                    "native_acquisition_group",
                    "source_coverage",
                )
            }
            | {
                "objects": objects,
                "source_evidence": summaries,
                "source_position_optical": support["source_position_optical"],
                "chip_sha256": row["chip"]["sha256"],
                "ignore_mask": ref(mask_path),
                "whole_area_ignore_for_accuracy": True,
                "independent_review": False,
                "native_input_metadata": info,
            }
        )
    save(
        output / "report.json",
        {
            "prepared": ref(prepared),
            "source": ref(source_path),
            "areas": areas,
            "samples": samples,
            **FLAGS,
        },
    )
    collection = {
        "version": "0.1.0",
        "protocol": ref(PROTOCOL),
        "prepared": ref(prepared),
        "first_pass": ref(FIRST),
        "reconciliation": ref(RECONCILIATION),
        "source_evidence": ref(source_path),
        "local_report": ref(output / "report.json"),
        "areas": areas,
        "class_counts": dict(Counter(o["class"] for a in areas for o in a["objects"])),
        "native_calibration_bands_replayed": 4,
        "native75_contexts": sum(s["chip"] is not None for s in samples),
        "exploratory_training_admitted": False,
        "accuracy_truth_admitted": False,
        **FLAGS,
    }
    validate(collection)
    save(DIRECTORY / "collection.json", collection)
    save(LOCK, {"collection": ref(DIRECTORY / "collection.json")})
    print(
        json.dumps(
            {
                "areas": 2,
                "classes": collection["class_counts"],
                "native75": collection["native75_contexts"],
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    collection = json.loads(checked(lock["collection"]).read_text())
    validate(collection)
    if local:
        rows = replay(checked(collection["prepared"]))
        report = json.loads(checked(collection["local_report"]).read_text())
        source = json.loads(checked(collection["source_evidence"]).read_text())
        verify_source(checked(collection["prepared"]), source)
        validate_support(
            json.loads(checked(collection["reconciliation"]).read_text())["areas"],
            source,
        )
        if report["areas"] != collection["areas"]:
            raise ValueError("Local/public collection differs")
        for area in collection["areas"]:
            row = next(r for r in rows if r["id"] == area["id"])
            if any(
                area[k] != row[k]
                for k in (
                    "product_id",
                    "split",
                    "geographic_group",
                    "native_acquisition_group",
                )
            ):
                raise ValueError("Collection acquisition or group changed")
            preparation = json.loads(checked(collection["prepared"]).read_text())
            product = next(
                p
                for p in preparation["products"]
                if p["product_id"] == row["product_id"]
            )
            if area["native_input_metadata"] != input_metadata(product, row):
                raise ValueError("Collection native input metadata changed")
            if (
                area["chip_sha256"] != row["chip"]["sha256"]
                or area["source_coverage"] != row["source_coverage"]
            ):
                raise ValueError("Prepared pixel or coverage binding changed")
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
                expected = native_input(
                    values, obj["bbox_px"], area["native_input_metadata"], "train"
                )
                sample = next(
                    s
                    for s in report["samples"]
                    if s["area_id"] == area["id"] and s["object_id"] == obj["id"]
                )
                tensor = expected.pop("tensor")
                if sample["input"] != expected or (tensor is None) != (
                    sample["chip"] is None
                ):
                    raise ValueError("Native input abstention changed")
                if tensor is not None and (
                    not np.array_equal(
                        tensor, np.load(checked(sample["chip"]), allow_pickle=False)
                    )
                    or hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
                    != sample["tensor_sha256_float32_c_order"]
                ):
                    raise ValueError("Native75 pixels differ")
    return {
        "verified": True,
        "local": local,
        "areas": 2,
        "classes": collection["class_counts"],
        "scored": False,
        "training_admitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "evidence", "build", "source", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.review and args.output:
        review(args.review, args.output)
    elif args.evidence and args.output:
        evidence(args.evidence, args.output)
    elif args.build and args.source and args.output:
        build(args.build, args.source, args.output)
    elif args.verify:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose review, evidence, build or verify")


if __name__ == "__main__":
    main()
