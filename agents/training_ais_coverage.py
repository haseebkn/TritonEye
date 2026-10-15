"""Audit observed AIS reception in training envelopes; never infer completeness."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import Point, shape

from agents.acquisition import native_acquisition_group, unique_acquisitions
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT
from agents.association import aligned_ais
from agents.cd_training_reservations import freeze_priors
from agents.hh_hv_reservation import POLICY as REGIONS
from agents.hh_hv_reservation import envelopes, save
from agents.overlap_audit import fetch_catalogue, observations, snapshot
from agents.provisional_training import training_role
from agents.region import load_region
from agents.source_followup import fresh
from agents.source_reservation import checked, ref

REPORT = REPO_ROOT / "docs/evidence/training_ais_coverage_v010.json"
LOCK = REPO_ROOT / "configs/reservations/training_ais_coverage_v010.lock.json"
SELECTION = REPO_ROOT / "data/ca_sel/selection.json"
LIMITS = [
    "Observed reception only; no denominator for vessels not received.",
    "Silent intervals include vessel absence, feed limits and recorder outages; "
    "these causes cannot be distinguished here.",
    "SAR support uses catalogue midpoint, not native target time; no label admission.",
    "Training envelopes are sparse sampling areas, not province-wide coverage.",
    "Adjacent slices, repeated identities and report counts are not "
    "independent trials.",
    "No model scores, protected SAR pixels or quarantined annotations are read.",
]


def areas(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    rules = json.loads(REGIONS.read_text())
    result = []
    for group in rules["groups"]:
        if group["split"] != "train":
            continue
        for area in envelopes(group, rules):
            geometry = area["geometry"]
            load_region(geometry)
            assigned = training_role(geometry, frozen["priors"], "coverage_metadata")
            if assigned != group["id"]:
                raise ValueError("Training envelope role changed")
            result.append(
                area | {"id": group["id"] + "_" + area["regime"], "group": assigned}
            )
    return result


def reception(
    rows: list[dict[str, Any]], start: float, stop: float, instant: float
) -> dict[str, Any]:
    selected = [r for r in rows if start <= r["time"] <= stop]
    stamps = sorted({r["time"] for r in selected})
    gaps = [b - a for a, b in zip([start, *stamps], [*stamps, stop])]
    hours: dict[str, int] = defaultdict(int)
    for row in selected:
        hour = datetime.fromtimestamp(row["time"], timezone.utc).strftime("%Y-%m-%dT%H")
        hours[hour] += 1
    last = stamps[-1] if stamps else None
    return {
        "reports": len(selected),
        "unique_identities": len({r["mmsi"] for r in selected}),
        "hours_with_reports": len(hours),
        "hourly_reports_utc": dict(sorted(hours.items())),
        "first_utc": selected[0]["timestamp"] if selected else None,
        "last_utc": selected[-1]["timestamp"] if selected else None,
        "last_report_age_s": instant - last if last is not None else None,
        "longest_silent_interval_s_including_edges": max(gaps),
        "coverage_complete": False,
    }


def analyse(
    rows: list[dict[str, Any]],
    products: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    frozen: dict[str, Any],
    audited_at: str,
) -> dict[str, Any]:
    now = parse_utc(audited_at)
    if not rows or now is None:
        raise ValueError("Valid archive and audit UTC required")
    rows = sorted(rows, key=lambda r: r["time"])
    times = [r["time"] for r in rows]
    start, stop = times[0], times[-1]
    result = []
    # Unique physical slices remove native/COG aliases before pass counting.
    physical = unique_acquisitions(products)
    for area in regions:
        scope = shape(area["geometry"])
        local = [r for r in rows if scope.covers(Point(r["lon"], r["lat"]))]
        passes, invalid = [], 0
        for product in physical:
            name = product.get("Name", "")
            if name[:3] not in {"S1C", "S1D"} or "_IW_GRDH_1SDH_" not in name:
                continue
            first = parse_utc(product.get("ContentDate", {}).get("Start"))
            last = parse_utc(product.get("ContentDate", {}).get("End"))
            try:
                footprint = shape(product["GeoFootprint"])
                if footprint.is_empty or not footprint.is_valid:
                    raise ValueError("Invalid swath")
            except (KeyError, TypeError, ValueError):
                invalid += 1
                continue
            if first is None or last is None or first >= last:
                invalid += 1
                continue
            midpoint = (first + (last - first) / 2).timestamp()
            if not start <= midpoint <= stop or not scope.intersects(footprint):
                continue
            group = native_acquisition_group(name)
            if group is None:
                invalid += 1
                continue
            reason = None
            try:
                if (
                    training_role(area["geometry"], frozen["priors"], group)
                    != area["group"]
                ):
                    raise ValueError("Conflicting training geography")
                if group in frozen["exposed"]:
                    reason = "previously_exposed_or_reserved_acquisition"
            except ValueError as error:
                reason = str(error)
            # Retain full nearby histories BEFORE spatial filtering, including
            # vessels that cross the region boundary during interpolation.
            nearby = rows[
                bisect.bisect_left(times, midpoint - 1800) : bisect.bisect_right(
                    times, midpoint + 1800
                )
            ]
            aligned, _ = aligned_ais(
                pd.DataFrame([r["values"] for r in nearby]),
                datetime.fromtimestamp(midpoint, timezone.utc).isoformat(),
            )
            support, tight = 0, 0
            for _, vessel in aligned.iterrows():
                if not (
                    scope.covers(vessel.geometry) and footprint.covers(vessel.geometry)
                ):
                    continue
                support += 1
                history = [r["time"] for r in nearby if r["mmsi"] == vessel.mmsi]
                before = [t for t in history if t <= midpoint]
                after = [t for t in history if t >= midpoint]
                gap = min(after) - max(before) if before and after else None
                tight += int(
                    gap is not None
                    and gap <= 240
                    and min(abs(t - midpoint) for t in history) <= 120
                )
            passes.append(
                {
                    "product_id": product["Id"],
                    "name": name,
                    "native_acquisition_group": group,
                    "midpoint_utc": datetime.fromtimestamp(
                        midpoint, timezone.utc
                    ).isoformat(),
                    "whole_envelope_in_swath": footprint.covers(scope),
                    "aligned_tracks_in_envelope_and_swath": support,
                    "tracks_with_120s_report_and_240s_bracket": tight,
                    "new_training_acquisition_eligible": reason is None,
                    "exclusion_reason": reason,
                    "native_input_verified": False,
                    "label_admitted": False,
                }
            )
        result.append(
            area
            | {
                "reception": reception(local, start, stop, now.timestamp()),
                "cd_hh_hv_physical_slices": len(passes),
                "cd_hh_hv_native_acquisition_groups": len(
                    {p["native_acquisition_group"] for p in passes}
                ),
                "slices_with_tight_track_support": sum(
                    p["tracks_with_120s_report_and_240s_bracket"] > 0 for p in passes
                ),
                "new_eligible_slices_with_tight_support": sum(
                    p["new_training_acquisition_eligible"]
                    and p["tracks_with_120s_report_and_240s_bracket"] > 0
                    for p in passes
                ),
                "invalid_catalogue_records_skipped": invalid,
                "passes": sorted(
                    passes, key=lambda p: (p["midpoint_utc"], p["product_id"])
                ),
            }
        )
    return {
        "audited_at": audited_at,
        "archive_start_utc": rows[0]["timestamp"],
        "archive_end_utc": rows[-1]["timestamp"],
        "observation_span_s": stop - start,
        "regions": result,
        "coverage_complete": False,
        "labels_admitted": 0,
        "limits": LIMITS,
    }


def build(output: Path) -> None:
    if REPORT.exists() or LOCK.exists():
        raise ValueError("Fresh public evidence version required")
    fresh(output)
    snapshot(REPO_ROOT / "data/raw/ais_stream", output / "archive_snapshot")
    rows, stats = observations(output / "archive_snapshot")
    if not rows:
        raise ValueError("No observations")
    frozen = freeze_priors(SELECTION)
    regions = areas(frozen)
    products = fetch_catalogue(stats["first_utc"], stats["last_utc"], load_region())
    save(output / "catalogue.json", products)
    save(output / "priors.json", frozen)
    value = analyse(
        rows, products, regions, frozen, datetime.now(timezone.utc).isoformat()
    )
    value.update(
        archive=stats,
        catalogue_records=len(products),
        inputs={
            "snapshot": ref(output / "snapshot_manifest.json"),
            "catalogue": ref(output / "catalogue.json"),
            "priors": ref(output / "priors.json"),
            "regions": ref(REGIONS),
            "implementation": ref(Path(__file__)),
        },
    )
    save(REPORT, value)
    save(LOCK, {"report": ref(REPORT)})
    print(json.dumps({"verified": True, "archive": stats, "regions": len(regions)}))


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    value = json.loads(checked(lock["report"]).read_text())
    if value["coverage_complete"] is not False or value["labels_admitted"] != 0:
        raise ValueError("Coverage or label claims changed")
    checked(value["inputs"]["implementation"])
    checked(value["inputs"]["regions"])
    if local:
        inputs = value["inputs"]
        manifest_path = checked(inputs["snapshot"])
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            checked(
                {
                    "path": (manifest_path.parent / "archive_snapshot" / entry["file"])
                    .relative_to(REPO_ROOT)
                    .as_posix(),
                    "sha256": entry["sha256"],
                }
            )
        frozen = json.loads(checked(inputs["priors"]).read_text())
        for path, sha in frozen["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        rows, stats = observations(manifest_path.parent / "archive_snapshot")
        replay = analyse(
            rows,
            json.loads(checked(inputs["catalogue"]).read_text()),
            areas(frozen),
            frozen,
            value["audited_at"],
        )
        replay = json.loads(json.dumps(replay))
        if any(replay[k] != value[k] for k in replay) or stats != value["archive"]:
            raise ValueError("Coverage replay differs")
    return {"verified": True, "local": local, "labels_admitted": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.build and not args.verify:
        build(args.build)
    elif args.verify and not args.build:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose build or verify")
