"""Freeze and rank private NL SAR–AIS support; never manufacture vessel truth.

Run ``python -m agents.overlap_audit --help``. Outputs belong under ignored
data/research/. The image-only bundle has no predictions, identities or AIS
positions. Selection is AIS-informed, so it is not an unbiased recall benchmark.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import html
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from agents.acquisition import (
    _complete_footprint,
    native_acquisition_group,
    product_id,
    unique_acquisitions,
)
from agents.ais_validation import parse_utc, valid_mmsi
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.association import GEOD, aligned_ais
from agents.landmask import LandMask
from agents.recorder_health import recorder_health
from agents.region import REGION_PATH, load_region

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
LIMITS = [
    "AIS is partial supporting evidence, not exhaustive truth or reviewed identity.",
    "Independent review unavailable; annotations and improvement remain blocked.",
    "Catalogue midpoint is a screening time, not per-pixel radar azimuth time.",
    "Source-relative shoreline is not surveyed; controls do not cover the province.",
    "AIS-informed development selection is not an unbiased detector benchmark.",
    "Connected heartbeat is not proof of area-wide vessel reception.",
    "HH/HV is experimental and incompatible with the production VV/VH detector.",
    "Adjacent slices and repeated vessels do not create independent trials.",
]


def snapshot(archive: Path, output: Path) -> dict[str, Any]:
    """Freeze bounded complete-line prefixes without stopping the live recorder.

    Files are captured sequentially, not transactionally. Never include a
    partially appended CSV/journal line. Replay uses these immutable bytes.
    """
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for source in sorted(archive.iterdir()):
        if not (
            source.name.startswith("ais_stream_")
            and source.suffix == ".csv"
            or source.name.startswith("coverage_")
            and source.suffix == ".jsonl"
        ):
            continue
        start = datetime.now(timezone.utc).isoformat()
        size = source.stat().st_size
        with source.open("rb") as stream:
            payload = stream.read(size)
        end = payload.rfind(b"\n") + 1
        target = output / source.name
        target.write_bytes(payload[:end])
        files.append(
            {
                "file": source.name,
                "sha256": sha256_file(target),
                "captured_at": start,
                "source_bytes_at_start": size,
                "snapshot_bytes": end,
                "incomplete_tail_bytes_excluded": len(payload) - end,
            }
        )
    result = {
        "files": files,
        "transactional": False,
        "live_recorder": recorder_health(archive),
    }
    write_json(output.parent / "snapshot_manifest.json", result)
    return result


def observations(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize private reports, preserving motion fields and source references."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    counts: Counter[str] = Counter()
    for path in sorted(directory.glob("ais_stream_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                counts["input_rows"] += 1
                raw_time = row.get("timestamp", "")
                # Match the production aligner's explicit legacy UTC policy.
                try:
                    instant = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    naive = instant.tzinfo is None
                    if naive:
                        instant = instant.replace(tzinfo=timezone.utc)
                    instant = instant.astimezone(timezone.utc)
                    lon, lat = float(row["lon"]), float(row["lat"])
                    identity = row["mmsi"].strip()
                    if not (
                        valid_mmsi(identity)
                        and math.isfinite(lon)
                        and math.isfinite(lat)
                        and -180 <= lon <= 180
                        and -90 <= lat <= 90
                    ):
                        raise ValueError("Invalid AIS report")
                except (ValueError, KeyError, TypeError):
                    counts["invalid_rows"] += 1
                    continue
                # Deduplicate identical reports, not conflicting motion values.
                key = (
                    identity,
                    instant,
                    lon,
                    lat,
                    row.get("speed_knots"),
                    row.get("course_deg"),
                    row.get("length"),
                )
                if key in seen:
                    counts["duplicate_reports_removed"] += 1
                    continue
                seen.add(key)
                counts["legacy_naive_utc_reports"] += int(naive)
                rows.append(
                    {
                        "time": instant.timestamp(),
                        "timestamp": instant.isoformat(),
                        "mmsi": identity,
                        "lon": lon,
                        "lat": lat,
                        "values": {**row, "timestamp": instant.isoformat()},
                        "source": {"file": path.name, "line": line},
                        "legacy_naive_utc": naive,
                    }
                )
    rows.sort(key=lambda r: r["time"])
    return rows, {
        **dict(counts),
        "valid_unique_reports": len(rows),
        "first_utc": rows[0]["timestamp"] if rows else None,
        "last_utc": rows[-1]["timestamp"] if rows else None,
        "dates_present": sorted({r["timestamp"][:10] for r in rows}),
    }


def fetch_catalogue(start: str, end: str, region: Any) -> list[dict[str, Any]]:
    """Public metadata only; fail on unsafe/cyclic pagination, never use tokens."""
    query = (
        "Collection/Name eq 'SENTINEL-1' and contains(Name, 'IW_GRDH') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{region.wkt}') "
        f"and ContentDate/Start ge {start} and ContentDate/Start le {end}"
    )
    url = (
        CATALOGUE
        + "?"
        + urlencode(
            {"$filter": query, "$orderby": "ContentDate/Start asc", "$top": "1000"}
        )
    )
    products = []
    visited: set[str] = set()
    while url:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "catalogue.dataspace.copernicus.eu"
            or url in visited
            or len(visited) >= 100
        ):
            raise ValueError("Invalid catalogue pagination")
        visited.add(url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        page = response.json()
        products.extend(page["value"])
        url = page.get("@odata.nextLink")
    return products


def test_exclusion(manifest: Path) -> tuple[set[str], Any]:
    """Conservative buffered test geography; never read test image pixels."""
    value = json.loads(manifest.read_text(encoding="utf-8"))
    tests = [s for s in value["scenes"] if s["split"] == "test"]
    groups = {native_acquisition_group(s["name"]) for s in tests}
    if not tests or None in groups:
        raise ValueError("Missing recognized locked test acquisition group")
    geographic = unary_union(
        [shape(r["geometry"]) for s in tests for r in s["rois"]]
    ).buffer(0.03)
    if geographic.is_empty or not geographic.is_valid:
        raise ValueError("Invalid locked test geography")
    return {str(g) for g in groups}, geographic


def journal_support(directory: Path, centre: float) -> dict[str, Any]:
    """Bounded connected heartbeat bracketing, including disconnects and holes."""
    sessions = []
    for path in sorted(directory.glob("coverage_*.jsonl")):
        beats = []
        breaks = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            instant = parse_utc(event.get("time"))
            # Like the pass-specific transport audit, explicit connection
            # transitions invalidate a heartbeat span. Include both endpoints:
            # equal timestamps cannot establish event ordering safely.
            if instant and event.get("event") in {
                "disconnected",
                "session_stop",
                "transport_open",
            }:
                breaks.append(instant.timestamp())
            if event.get("event") == "heartbeat" and instant:
                beats.append((instant.timestamp(), event.get("connected") is True))
        beats.sort()
        left = [b for b in beats if b[0] <= centre]
        right = [b for b in beats if b[0] >= centre]
        if left and right:
            a, b = left[-1], right[0]
            gap = b[0] - a[0]
            sessions.append(
                {
                    "journal": path.name,
                    "bracket_gap_s": gap,
                    "connected_bracket_within_180s": bool(
                        a[1]
                        and b[1]
                        and gap <= 180
                        and centre - a[0] <= 180
                        and b[0] - centre <= 180
                        and not any(a[0] <= t <= b[0] for t in breaks)
                    ),
                }
            )
    return {
        "status": (
            "connected_bracket"
            if any(s["connected_bracket_within_180s"] for s in sessions)
            else "gap_or_unknown"
        ),
        "sessions": sessions,
        "area_wide_reception_proven": False,
    }


def audit_products(
    products: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    region: Any,
    locked_groups: set[str],
    locked_geo: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Time-align full histories BEFORE NL/swath filtering (boundary motion)."""
    times = [r["time"] for r in rows]
    results, rejected = [], []
    for p in unique_acquisitions(products):
        pid = product_id(p["Id"])
        group = native_acquisition_group(p.get("Name"))
        start = parse_utc(p.get("ContentDate", {}).get("Start"))
        end = parse_utc(p.get("ContentDate", {}).get("End"))
        reason = None
        if (
            not group
            or not start
            or not end
            or start >= end
            or not re.fullmatch(
                r"S1[A-D]_IW_GRDH_1SD[VH]_[A-Za-z0-9_]+\.SAFE", p.get("Name", "")
            )
        ):
            reason = "invalid_identity_or_time"
        elif not _complete_footprint(p.get("GeoFootprint")):
            reason = "missing_or_invalid_footprint"
        else:
            swath = shape(p["GeoFootprint"])
            if group in locked_groups or swath.intersects(locked_geo):
                reason = "locked_test_group_or_geography"
            elif not swath.intersects(region):
                reason = "outside_nl"
        if reason:
            rejected.append({"product_id": pid, "reason": reason})
            continue
        assert start is not None and end is not None
        centre = (start + (end - start) / 2).timestamp()
        window = rows[
            bisect.bisect_left(times, centre - 1800) : bisect.bisect_right(
                times, centre + 1800
            )
        ]
        aligned, stats = aligned_ais(
            pd.DataFrame([r["values"] for r in window]),
            datetime.fromtimestamp(centre, timezone.utc).isoformat(),
        )
        support = []
        for _, position in aligned.iterrows():
            if not (
                swath.covers(position.geometry) and region.covers(position.geometry)
            ):
                continue
            history = [r for r in window if r["mmsi"] == position.mmsi]
            nearest = min(history, key=lambda r: abs(r["time"] - centre))
            left = [r for r in history if r["time"] <= centre]
            right = [r for r in history if r["time"] >= centre]
            gap = right[0]["time"] - left[-1]["time"] if left and right else None
            support.append(
                {
                    "mmsi": position.mmsi,
                    "lon": position.lon,
                    "lat": position.lat,
                    "alignment_method": position.alignment_method,
                    "uncertainty_m": position.uncertainty_m,
                    "nearest_offset_s": nearest["time"] - centre,
                    "close_120s": abs(nearest["time"] - centre) <= 120,
                    "bracket_gap_s": gap,
                    "bracket_240s": gap is not None and gap <= 240,
                    "legacy_naive_utc": any(r["legacy_naive_utc"] for r in history),
                    "reports": history,
                }
            )
        raw_in = sum(
            swath.covers(Point(r["lon"], r["lat"]))
            and region.covers(Point(r["lon"], r["lat"]))
            for r in window
        )
        results.append(
            {
                "product": p,
                "native_acquisition_group": group,
                "reference_midpoint_utc": datetime.fromtimestamp(
                    centre, timezone.utc
                ).isoformat(),
                "polarization": "HH/HV" if "_1SDH_" in p["Name"] else "VV/VH",
                "raw_reports_in_swath": raw_in,
                "aligned_in_swath": len(support),
                "close_120s": sum(s["close_120s"] for s in support),
                "bracket_240s": sum(s["bracket_240s"] for s in support),
                "alignment_statistics": stats,
                "support": support,
                "status": (
                    "temporal_candidate"
                    if any(s["close_120s"] for s in support)
                    else "sparse_support_only" if raw_in else "no_observed_overlap"
                ),
                "ground_truth_eligible": False,
                "valid_pixels": "not_checked",
            }
        )
    return results, rejected


def enrich(scene: dict[str, Any], archive: Path, raw: Path, cache: Path) -> None:
    """Shoreline is physical evidence, not alert suppression or vessel labels."""
    support = scene["support"]
    bounds = (
        unary_union([Point(s["lon"], s["lat"]) for s in support]).buffer(0.05).bounds
    )
    mask = LandMask.for_footprint(bounds, cache_dir=str(cache))
    scene["shoreline_validation"] = mask.validation
    physical, distances_to_shore = mask.classify_physical(
        [s["lon"] for s in support], [s["lat"] for s in support]
    )
    policy, _ = mask.classify(
        [s["lon"] for s in support], [s["lat"] for s in support], coastal_buffer_m=300
    )
    for s, surface, zone, distance in zip(
        support, physical, policy, distances_to_shore
    ):
        s["physical_surface"] = surface
        s["policy_300m"] = zone
        s["signed_shore_distance_m"] = (
            float(distance) if math.isfinite(distance) else None
        )
    scene["physical_counts"] = dict(Counter(s["physical_surface"] for s in support))
    scene["policy_counts"] = dict(Counter(s["policy_300m"] for s in support))
    scene["close_water"] = sum(
        s["close_120s"] and s["physical_surface"] == "water" for s in support
    )
    scene["close_open_water"] = sum(
        s["close_120s"] and s["policy_300m"] == "water" for s in support
    )
    # A scene-level competing-vessel opportunity, not a SAR match or identity.
    distances = [
        GEOD.inv(a["lon"], a["lat"], b["lon"], b["lat"])[2]
        for i, a in enumerate(support)
        for b in support[i + 1 :]
    ]
    scene["nearby_track_pairs_5km"] = sum(d <= 5000 for d in distances)
    scene["recorder"] = journal_support(
        archive, datetime.fromisoformat(scene["reference_midpoint_utc"]).timestamp()
    )
    # Keep canonical UUID; a cached alias is not silently substituted.
    scene["local_exact_product"] = (
        raw / scene["product"]["Name"] / "manifest.safe"
    ).is_file()


def rank_key(scene: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -scene.get("close_water", 0),
        -scene["bracket_240s"],
        -scene.get("nearby_track_pairs_5km", 0),
        -scene["aligned_in_swath"],
        scene["product"]["ContentDate"]["Start"],
        scene["product"]["Id"],
    )


def public_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Explicit aggregate allowlist; never copy the private support payload."""
    keys = (
        "native_acquisition_group",
        "reference_midpoint_utc",
        "polarization",
        "raw_reports_in_swath",
        "aligned_in_swath",
        "close_120s",
        "bracket_240s",
        "status",
        "ground_truth_eligible",
        "valid_pixels",
        "physical_counts",
        "policy_counts",
        "close_water",
        "close_open_water",
        "nearby_track_pairs_5km",
        "local_exact_product",
        "shoreline_validation",
        "review_areas_count",
    )
    p = scene["product"]
    return {
        "product_id": p["Id"],
        "name": p["Name"],
        "start_utc": p["ContentDate"]["Start"],
        "end_utc": p["ContentDate"]["End"],
        **{k: scene[k] for k in keys if k in scene},
        "recorder_status": scene.get("recorder", {}).get("status", "unknown"),
        "aliases": p.get("catalogue_aliases", [{"Id": p["Id"], "Name": p["Name"]}]),
    }


def blind_html(areas: list[dict[str, Any]]) -> str:
    """Portable image-only view: no telemetry, labels or detector dependencies."""
    sections = []
    for area in areas:
        images = []
        for pol, preview in zip(area["polarizations"], area["previews"]):
            path = REPO_ROOT / preview["path"]
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            images.append(
                f"<figure><figcaption>{html.escape(pol)}</figcaption>"
                f'<img width="512" height="512" alt="Native {html.escape(pol)} SAR" '
                f'src="data:image/png;base64,{payload}"></figure>'
            )
        sections.append(
            f'<section><h2>{html.escape(area["id"])}</h2>'
            f'<p>{html.escape(area["product_id"])} · '
            f'{html.escape(area["acquisition_time"])} · '
            f'chip SHA-256 {area["chip"]["sha256"]}</p>'
            "<p>Inspect every 256×256 cell in both bands; annotate vessels, "
            "fixed structures, clutter, identifiable ice and unresolved objects. "
            "Record an empty decision explicitly. No model or AIS overlay.</p>"
            f'<div class="images">{"".join(images)}</div></section>'
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>NL detector-blind development review</title><style>"
        "body{font:16px system-ui;background:#edf2f4;color:#172830;margin:24px}"
        "section{background:white;padding:20px;margin:24px 0}"
        "p{overflow-wrap:anywhere}.images{display:flex;flex-wrap:wrap;gap:16px}"
        "figure{margin:0}img{max-width:100%;height:auto}</style>"
        "<h1>NL detector-blind development review</h1>"
        "<p>Contains modified Copernicus Sentinel data (2026). "
        "Independent review unavailable. AIS-informed area selection; preparer "
        "has seen telemetry. Not an independent or unbiased detector benchmark. "
        "Keep first-pass decisions frozen before opening the separate telemetry "
        "directory. No labels have been supplied or presumed.</p>"
        + "".join(sections)
        + "</html>"
    )


def bundle(
    scenes: list[dict[str, Any]],
    output: Path,
    raw: Path,
    locked_geo: Any,
    count: int,
    download: bool,
) -> dict[str, Any]:
    """Freeze AIS-informed selection, then materialize complete dual-pol areas."""
    from scripts.build_nl_benchmark import chip, download_product

    output.mkdir(parents=True, exist_ok=False)
    telemetry = output.parent / "telemetry_second_pass"
    telemetry.mkdir(exist_ok=False)
    selected, groups = [], set()
    for scene in sorted(scenes, key=rank_key):
        if (
            scene["status"] != "temporal_candidate"
            or scene["native_acquisition_group"] in groups
        ):
            continue
        if not download and not scene.get("local_exact_product"):
            continue
        selected.append(scene)
        groups.add(scene["native_acquisition_group"])
        if len(selected) == count:
            break
    proposals: list[dict[str, Any]] = []
    for i, scene in enumerate(selected, 1):
        fixes = sorted(
            scene["support"],
            key=lambda s: (
                not s["close_120s"],
                s.get("physical_surface") != "water",
                abs(s["nearest_offset_s"]),
                s["mmsi"],
            ),
        )
        centres: list[dict[str, Any]] = []
        for fix in fixes:
            if not fix["close_120s"]:
                continue
            if any(
                GEOD.inv(fix["lon"], fix["lat"], x["lon"], x["lat"])[2] < 6500
                for x in centres
            ):
                continue
            centres.append(fix)
            if len(centres) == 2:
                break
        proposals.extend(
            {
                "id": f"scene_{i:02d}_area_{j:02d}",
                "size": 512,
                "center": [fix["lon"], fix["lat"]],
                "product_id": scene["product"]["Id"],
            }
            for j, fix in enumerate(centres, 1)
        )
    write_json(
        output / "selection_lock.json",
        {
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "areas": proposals,
            "split": "new_development_only_not_frozen_benchmark",
            "detector_exposure": False,
            "prior_ais_exposure": True,
            "independent_review_available": False,
            "selection_bias": "AIS-informed complete-area correlation pilot",
        },
    )
    areas: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for scene in selected:
        p = scene["product"]
        try:
            if not (raw / p["Name"] / "manifest.safe").is_file():
                size = p.get("ContentLength")
                if not isinstance(size, int) or size <= 0:
                    raise ValueError(
                        "Missing product size; download space cannot be checked"
                    )
                # Archive + extraction can coexist. Preserve 1 GiB for other
                # project activity; never erase cached data to fit a download.
                needed = 2 * size + (1 << 30)
                if shutil.disk_usage(raw).free < needed:
                    raise ValueError("Insufficient free space for safe SAR download")
            product = download_product(p, raw)
            pols = ["hh", "hv"] if scene["polarization"] == "HH/HV" else ["vv", "vh"]
            paths = []
            for pol in pols:
                candidates = list((product / "measurement").glob(f"*-{pol}-*.tiff"))
                if len(candidates) != 1:
                    raise ValueError("Missing or ambiguous polarization measurement")
                paths.append(candidates[0])
            sources.extend(
                {
                    "product_id": p["Id"],
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in paths
            )
            for proposal in [a for a in proposals if a["product_id"] == p["Id"]]:
                try:
                    result = chip(proposal, paths, output / proposal["id"])
                    footprint = shape(result["geometry"])
                    if footprint.intersects(locked_geo) or not shape(
                        p["GeoFootprint"]
                    ).covers(footprint):
                        raise ValueError(
                            "Chip exceeds exact swath or locked test boundary"
                        )
                    if any(
                        a["product_id"] == p["Id"]
                        and shape(a["geometry"]).intersection(footprint).area > 0
                        for a in areas
                    ):
                        raise ValueError(
                            "Selected review areas overlap; not a new area"
                        )
                    result.update(
                        polarizations=[pol.upper() for pol in pols],
                        acquisition_time=scene["reference_midpoint_utc"],
                    )
                    mask = LandMask.for_footprint(
                        footprint.bounds, cache_dir=str(REPO_ROOT / "data/reference")
                    )
                    area_m2 = abs(GEOD.geometry_area_perimeter(footprint)[0])
                    physical_water = mask.water_geometry(footprint, buffer_m=0)
                    open_water = mask.water_geometry(footprint, buffer_m=300)
                    result.update(
                        area_km2=area_m2 / 1e6,
                        reference_water_km2=abs(
                            GEOD.geometry_area_perimeter(physical_water)[0]
                        )
                        / 1e6,
                        policy_300m_water_km2=abs(
                            GEOD.geometry_area_perimeter(open_water)[0]
                        )
                        / 1e6,
                        shoreline_validation=mask.validation,
                    )
                    areas.append(result)
                    scene["review_areas_count"] = scene.get("review_areas_count", 0) + 1
                    scene["valid_pixels"] = "selected_areas_verified_not_full_scene"
                    # Keep the whole aligned scene pool, including competing
                    # identities just outside a chip. ROI membership is a flag,
                    # not a prefilter that can manufacture an easy match.
                    supporting = [
                        {
                            **s,
                            "inside_review_area": footprint.covers(
                                Point(s["lon"], s["lat"])
                            ),
                        }
                        for s in scene["support"]
                    ]
                    write_json(
                        telemetry / (proposal["id"] + ".json"),
                        {
                            "product_id": p["Id"],
                            "support": supporting,
                            "ground_truth_eligible": False,
                            "correct_identities_reviewed": False,
                        },
                    )
                except (ValueError, OSError) as error:
                    failures.append({"area_id": proposal["id"], "reason": str(error)})
        except (ValueError, OSError, RuntimeError) as error:
            failures.append({"product_id": p["Id"], "reason": str(error)})
        print("Bundle scene", p["Id"], "complete areas", len(areas), flush=True)
    manifest = {
        "areas": areas,
        "failures": failures,
        "source_bands": sources,
        "annotation_complete": False,
        "independently_reviewed": False,
        "ground_truth_eligible": False,
        "limits": LIMITS,
        "calibration": (
            "Sigma0 LUT only; no thermal-noise or terrain correction; "
            "fixed -30 to +5 dB previews"
        ),
        "licence": "https://cds.climate.copernicus.eu/licences/ec-sentinel",
        "chip_builder_sha256": sha256_file(REPO_ROOT / "scripts/build_nl_benchmark.py"),
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "first_pass_template.json",
        {
            "reviewer": None,
            "independent": False,
            "completed_at": None,
            "areas": [
                {
                    "roi_id": a["id"],
                    "chip_sha256": a["chip"]["sha256"],
                    "cells_reviewed": [],
                    "objects": [],
                    "complete": False,
                    "notes": "",
                }
                for a in areas
            ],
        },
    )
    (output / "review.html").write_text(blind_html(areas), encoding="utf-8")
    write_json(
        output / "bundle_hashes.json",
        {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return {
        "selected_groups": len(selected),
        "complete_areas": len(areas),
        "complete_products": len({a["product_id"] for a in areas}),
        "failures": failures,
        "html_sha256": sha256_file(output / "review.html"),
    }


def run(
    output: Path, archive: Path, replay: Path | None, count: int, download: bool
) -> None:
    output = output.resolve()
    if not output.is_relative_to((REPO_ROOT / "data/research").resolve()):
        raise ValueError("Private outputs must stay under ignored data/research")
    output.mkdir(parents=True, exist_ok=False)
    manifest = REPO_ROOT / "datasets/nl_benchmark/v0.1.0/manifest.json"
    groups, lockgeo = test_exclusion(manifest)
    if replay:
        frozen = json.loads((replay / "snapshot_manifest.json").read_text())
        (output / "archive_snapshot").mkdir()
        for entry in frozen["files"]:
            path = replay / "archive_snapshot" / entry["file"]
            if (
                path.parent.resolve() != (replay / "archive_snapshot").resolve()
                or sha256_file(path) != entry["sha256"]
            ):
                raise ValueError("Snapshot hash/path mismatch")
            shutil.copyfile(path, output / "archive_snapshot" / entry["file"])
        previous = json.loads((replay / "public_summary.json").read_text())
        if (
            sha256_file(replay / "catalogue_products.json")
            != previous["summary"]["catalogue_sha256"]
        ):
            raise ValueError("Catalogue hash mismatch")
        archive = output / "archive_snapshot"
        write_json(output / "snapshot_manifest.json", frozen)
        products = json.loads((replay / "catalogue_products.json").read_text())
    else:
        snapshot(archive, output / "archive_snapshot")
        archive = output / "archive_snapshot"
        products = None
    rows, stats = observations(archive)
    if not rows:
        raise ValueError("No valid archived observations; no ranking manufactured")
    region = load_region()
    if products is None:
        start = datetime.fromtimestamp(rows[0]["time"] - 1800, timezone.utc).isoformat()
        end = datetime.fromtimestamp(rows[-1]["time"] + 1800, timezone.utc).isoformat()
        products = fetch_catalogue(start, end, region)
    write_json(output / "catalogue_products.json", products)
    scenes, rejected = audit_products(products, rows, region, groups, lockgeo)
    raw, cache = REPO_ROOT / "data/raw", REPO_ROOT / "data/reference"
    for scene in scenes:
        if scene["status"] == "temporal_candidate":
            enrich(scene, archive, raw, cache)
            print("Candidate", scene["product"]["Id"], scene["close_120s"], flush=True)
    candidates = sorted(
        [s for s in scenes if s["status"] == "temporal_candidate"], key=rank_key
    )
    summary = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "archive": stats,
        "catalogue_records": len(products),
        "physical_slices": len(scenes) + len(rejected),
        "audited_slices": len(scenes),
        "excluded": dict(Counter(r["reason"] for r in rejected)),
        "statuses": dict(Counter(s["status"] for s in scenes)),
        "candidate_groups": len({s["native_acquisition_group"] for s in candidates}),
        "ranking": (
            "close physical-water tracks, short brackets, nearby competing tracks, "
            "aligned count, UTC, UUID"
        ),
        "thresholds": {
            "retrieval_window_s": 1800,
            "close_report_s": 120,
            "bracket_s": 240,
            "nearby_pair_m": 5000,
            "role": "provisional research screening; not operational requirements",
        },
        "snapshot_manifest_sha256": sha256_file(output / "snapshot_manifest.json"),
        "catalogue_sha256": sha256_file(output / "catalogue_products.json"),
        "catalogue_endpoint": CATALOGUE,
        "catalogue_authentication": "public metadata; no credentials transmitted",
        "code_sha256": sha256_file(Path(__file__)),
        "alignment_code_sha256": sha256_file(REPO_ROOT / "agents/association.py"),
        "region_sha256": sha256_file(REGION_PATH),
        "test_manifest_sha256": sha256_file(manifest),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "replay_of": str(replay) if replay else None,
        "limits": LIMITS,
    }
    write_json(
        output / "audit_private.json",
        {"summary": summary, "scenes": scenes, "excluded": rejected},
    )
    if count:
        summary["bundle"] = bundle(
            candidates, output / "image_first", raw, lockgeo, count, download
        )
    public = {
        "summary": summary,
        "ranked_candidates": [public_scene(s) for s in candidates],
        "excluded": rejected,
    }
    write_json(output / "public_summary.json", public)
    print(
        json.dumps(
            {
                "archive": stats,
                "statuses": summary["statuses"],
                "bundle": summary.get("bundle"),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--archive", type=Path, default=REPO_ROOT / "data/raw/ais_stream"
    )
    parser.add_argument(
        "--replay", type=Path, help="Frozen prior audit; no catalogue refresh"
    )
    parser.add_argument("--bundle-count", type=int, default=0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing selected exact SAFE products",
    )
    args = parser.parse_args()
    if not 0 <= args.bundle_count <= 12:
        parser.error("bundle count must be 0..12")
    if not args.replay:
        health = recorder_health(args.archive)
        print(
            "Live recorder:", health["status"], "complete coverage:", False, flush=True
        )
    run(args.output, args.archive, args.replay, args.bundle_count, args.download)


if __name__ == "__main__":
    main()
