"""Map observed NL AIS reception and shortlist metadata-only training areas."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, Point, box, mapping, shape

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT
from agents.cd_training_reservations import freeze_priors
from agents.hh_hv_reservation import save
from agents.overlap_audit import audit_products, fetch_catalogue, observations, snapshot
from agents.provisional_training import GUARDS, training_role
from agents.region import REGION_PATH, load_region
from agents.source_followup import fresh
from agents.source_reservation import checked, envelope, ref
from agents.training_ais_coverage import SELECTION, reception

POLICY = REPO_ROOT / "configs/reservations/nl_reception_v010.json"
REPORT = REPO_ROOT / "docs/evidence/nl_reception_v010.json"
LOCK = REPO_ROOT / "configs/reservations/nl_reception_v010.lock.json"
MAP = REPO_ROOT / "docs/evidence/nl_reception_v010.geojson"
VIEW = REPO_ROOT / "docs/evidence/nl_reception_v010.html"


def grid_key(lon: float, lat: float, step: float) -> tuple[int, int]:
    return math.floor(lon / step), math.floor(lat / step)


def role(
    geometry: dict[str, Any], frozen: dict[str, Any], group: str
) -> dict[str, Any]:
    try:
        load_region(geometry)
        geographic = training_role(geometry, frozen["priors"], group)
        if group in frozen["exposed"]:
            raise ValueError("Previously exposed or reserved acquisition")
        return {"split_safe": True, "geographic_group": geographic, "exclusion": None}
    except ValueError as error:
        return {"split_safe": False, "geographic_group": None, "exclusion": str(error)}


def analyse(
    rows: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    frozen: dict[str, Any],
    rules: dict[str, Any],
    audited_at: str,
) -> dict[str, Any]:
    """No pixel reads or identities in public outputs; preserve full private pools."""
    if not rows:
        raise ValueError("Nonempty normalized archive required")
    region = load_region()
    step = rules["grid_degrees"]
    if not isinstance(step, (float, int)) or not 0 < step <= 1:
        raise ValueError("Invalid grid step")
    local: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["time"]):
        if region.covers(Point(row["lon"], row["lat"])):
            local[grid_key(row["lon"], row["lat"], step)].append(row)
    now = datetime.fromisoformat(audited_at).timestamp()
    start, stop = min(r["time"] for r in rows), max(r["time"] for r in rows)
    cells = []
    for (x, y), reports in sorted(local.items()):
        geometry = mapping(box(x * step, y * step, (x + 1) * step, (y + 1) * step))
        cells.append(
            {
                "id": f"cell_{x}_{y}",
                "grid": [x, y],
                "geometry": geometry,
                **role(dict(geometry), frozen, "reception_metadata_only"),
                "reception": reception(reports, start, stop, now),
                "days_with_reports": len({r["timestamp"][:10] for r in reports}),
                "tight_hh_hv_opportunities": [],
            }
        )
    indexed = {tuple(c["grid"]): c for c in cells}
    exclusions: Counter[str] = Counter()
    aligned_only_cells: dict[tuple[int, int], dict[str, Any]] = {}
    for scene in scenes:
        p = scene["product"]
        group = native_acquisition_group(p["Name"])
        if scene["polarization"] != "HH/HV" or group is None:
            continue
        for support in scene["support"]:
            if not (support["close_120s"] and support["bracket_240s"]):
                continue
            geometry = envelope(support["lon"], support["lat"], rules["half_width_m"])
            decision = role(geometry, frozen, group)
            if not shape(p["GeoFootprint"]).covers(shape(geometry)):
                decision = {
                    "split_safe": False,
                    "geographic_group": None,
                    "exclusion": "Incomplete candidate envelope in swath",
                }
            if not decision["split_safe"]:
                exclusions[decision["exclusion"]] += 1
            key = grid_key(support["lon"], support["lat"], step)
            # Boundary-crossing interpolation can land in a cell with no raw reports.
            if key not in indexed:
                x, y = key
                cell_geometry = mapping(
                    box(x * step, y * step, (x + 1) * step, (y + 1) * step)
                )
                indexed[key] = {
                    "id": f"cell_{x}_{y}",
                    "grid": [x, y],
                    "geometry": cell_geometry,
                    **role(dict(cell_geometry), frozen, "reception_metadata_only"),
                    "reception": reception([], start, stop, now),
                    "days_with_reports": 0,
                    "tight_hh_hv_opportunities": [],
                }
                aligned_only_cells[key] = indexed[key]
            opportunity = {
                "product_id": p["Id"],
                "name": p["Name"],
                "native_acquisition_group": group,
                "midpoint_utc": scene["reference_midpoint_utc"],
                **decision,
                "online_at_catalogue_query": p.get("Online") is True,
                "native_input_verified": False,
                "physical_water_verified": False,
                "source_identity_private": True,
                "label_admitted": False,
            }
            midpoint = datetime.fromisoformat(
                scene["reference_midpoint_utc"]
            ).timestamp()
            history = support.get("reports", [])
            nearest = (
                min(history, key=lambda r: abs(r["time"] - midpoint))
                if history
                else None
            )
            try:
                speed = float(nearest["values"]["speed_knots"]) if nearest else math.nan
                uncertainty = float(support.get("uncertainty_m", math.nan))
            except (KeyError, TypeError, ValueError):
                speed, uncertainty = math.nan, math.nan
            moving = bool(
                math.isfinite(speed)
                and speed >= rules["minimum_speed_knots"]
                and math.isfinite(uncertainty)
                and 0 <= uncertainty <= rules["maximum_uncertainty_m"]
            )
            opportunity["motion_uncertainty_screen_passed"] = moving
            opportunity["maximum_recorded_speed_knots"] = (
                speed if math.isfinite(speed) else None
            )
            # Aggregate tracks without publishing individual identities or positions.
            existing = next(
                (
                    a
                    for a in indexed[key]["tight_hh_hv_opportunities"]
                    if a["product_id"] == p["Id"]
                    and a["split_safe"] == decision["split_safe"]
                    and a["geographic_group"] == decision["geographic_group"]
                    and a["exclusion"] == decision["exclusion"]
                    and a["motion_uncertainty_screen_passed"] == moving
                ),
                None,
            )
            if existing is None:
                opportunity["aligned_identities"] = 1
                indexed[key]["tight_hh_hv_opportunities"].append(opportunity)
            else:
                existing["aligned_identities"] += 1
                finite_speeds = [
                    v
                    for v in (
                        existing["maximum_recorded_speed_knots"],
                        opportunity["maximum_recorded_speed_knots"],
                    )
                    if v is not None
                ]
                existing["maximum_recorded_speed_knots"] = (
                    max(finite_speeds) if finite_speeds else None
                )
    cells.extend(aligned_only_cells.values())
    for cell in cells:
        opportunities = cell["tight_hh_hv_opportunities"]
        cell["eligible_native_groups"] = len(
            {
                a["native_acquisition_group"]
                for a in opportunities
                if a["split_safe"] and a["online_at_catalogue_query"]
            }
        )
        cell["reception_screen_passed"] = bool(
            cell["split_safe"]
            and cell["days_with_reports"] >= rules["minimum_days"]
            and cell["reception"]["unique_identities"] >= rules["minimum_identities"]
        )
        cell["motion_qualified_native_groups"] = len(
            {
                a["native_acquisition_group"]
                for a in opportunities
                if a["split_safe"]
                and a["online_at_catalogue_query"]
                and a["motion_uncertainty_screen_passed"]
            }
        )
    ranked = sorted(
        [c for c in cells if c["reception_screen_passed"]],
        key=lambda c: (
            -c["motion_qualified_native_groups"],
            -c["eligible_native_groups"],
            -c["days_with_reports"],
            -c["reception"]["hours_with_reports"],
            -c["reception"]["unique_identities"],
            c["id"],
        ),
    )
    shortlist = [
        {
            "rank": i,
            "cell_id": c["id"],
            "geographic_group": c["geographic_group"],
            "eligible_native_groups": c["eligible_native_groups"],
            "motion_qualified_native_groups": c["motion_qualified_native_groups"],
            "status": (
                "sar_timing_candidate"
                if c["eligible_native_groups"]
                else "reception_only_no_eligible_sar_overlap"
            ),
            "reserved": False,
        }
        for i, c in enumerate(ranked[: rules["maximum_shortlist"]], 1)
    ]
    return {
        "version": "0.1.0",
        "audited_at": audited_at,
        "archive_start_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "archive_end_utc": datetime.fromtimestamp(stop, timezone.utc).isoformat(),
        "nl_reports": sum(len(v) for v in local.values()),
        "nl_identities": len({r["mmsi"] for v in local.values() for r in v}),
        "observed_cells": len(local),
        "interpolation_only_cells": len(aligned_only_cells),
        "split_safe_observed_cells": sum(
            c["split_safe"] for c in cells if tuple(c["grid"]) not in aligned_only_cells
        ),
        "tight_opportunity_exclusions": dict(exclusions),
        "motion_qualified_training_groups": len(
            {
                a["native_acquisition_group"]
                for c in cells
                for a in c["tight_hh_hv_opportunities"]
                if a["split_safe"]
                and a["online_at_catalogue_query"]
                and a["motion_uncertainty_screen_passed"]
            }
        ),
        "cells": sorted(cells, key=lambda c: c["id"]),
        "shortlist": shortlist,
        "labels_admitted": 0,
        "accuracy_truth_admitted": False,
        "coverage_complete": False,
        **GUARDS,
        "limits": rules["limits"],
    }


def geojson(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": c["geometry"],
                "properties": {
                    k: v
                    for k, v in c.items()
                    if k not in {"geometry", "tight_hh_hv_opportunities"}
                },
            }
            for c in report["cells"]
        ],
    }


def render(report: dict[str, Any]) -> str:
    """Self-contained coordinate map; blank cells mean no received reports."""

    def polygon(geometry: dict[str, Any], fill: str, title: str) -> str:
        points = " ".join(
            f"{(x + 62) * 45:.2f},{(62 - y) * 45:.2f}"
            for x, y in geometry["coordinates"][0]
        )
        return (
            f'<polygon points="{points}" fill="{fill}" stroke="#8494a7" '
            f'stroke-width="0.3"><title>{html.escape(title)}</title></polygon>'
        )

    content = [
        polygon(
            dict(mapping(load_region())),
            "#e9eef4",
            "NL study boundary; no jurisdiction claim",
        )
    ]
    for cell in report["cells"]:
        color = (
            "#b86666"
            if not cell["split_safe"]
            else "#63a786" if cell["reception_screen_passed"] else "#dfc782"
        )
        title = (
            f'{cell["id"]}: {cell["reception"]["reports"]} reports; '
            f'{cell["days_with_reports"]} days; '
            f'{cell["eligible_native_groups"]} eligible SAR groups; '
            f'{cell["exclusion"] or "geography screen passed"}'
        )
        content.append(polygon(cell["geometry"], color, title))
    for lon in range(-61, -44, 2):
        x = (lon + 62) * 45
        content.append(f'<text x="{x}" y="778" font-size="11">{lon}°</text>')
    for lat in range(46, 62, 2):
        y = (62 - lat) * 45
        content.append(f'<text x="2" y="{y}" font-size="11">{lat}°N</text>')
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>NL observed AIS reception</title>"
        "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto}"
        "svg{max-width:100%;height:750px}</style>"
        "<h1>NL observed AIS reception</h1><p>"
        + html.escape(report["archive_start_utc"] + " to " + report["archive_end_utc"])
        + "</p><p>Green: reception screen and training geography pass. "
        "Yellow: sparse reception. Red: protected or conflicting geography. "
        "Hover for counts. Grey: no reports in this snapshot. "
        "Missing reports do not establish vessel absence or an outage.</p>"
        '<svg viewBox="0 0 800 790" role="img" '
        'aria-label="Longitude latitude reception grid">'
        + "".join(content)
        + "</svg><p>Coordinate grid in longitude/latitude; "
        "no shoreline or physical-water validation. "
        "No identities or tracks are published. Candidate selection is "
        "AIS-informed; all labels remain provisional.</p></html>"
    )


def build(output: Path) -> None:
    if any(p.exists() for p in (REPORT, LOCK, MAP, VIEW)):
        raise ValueError("Fresh public version required")
    fresh(output)
    snapshot(REPO_ROOT / "data/raw/ais_stream", output / "archive_snapshot")
    rows, stats = observations(output / "archive_snapshot")
    if not rows:
        raise ValueError("No archive observations")
    frozen = freeze_priors(SELECTION)
    save(output / "priors.json", frozen)
    products = fetch_catalogue(stats["first_utc"], stats["last_utc"], load_region())
    save(output / "catalogue.json", products)
    hh = [p for p in products if "_IW_GRDH_1SDH_" in p.get("Name", "")]
    # Metadata audit only. ROI split checks below avoid excluding an entire
    # swath merely because another distant part intersects protected geography.
    scenes, rejected = audit_products(
        hh, rows, load_region(), set(), GeometryCollection()
    )
    save(output / "overlaps_private.json", {"scenes": scenes, "rejected": rejected})
    publish(output)


def publish(output: Path) -> None:
    """Seal a retained snapshot once, without resnapshotting the live recorder."""
    if any(p.exists() for p in (REPORT, LOCK, MAP, VIEW)):
        raise ValueError("Fresh public version required")
    rows, stats = observations(output / "archive_snapshot")
    frozen = json.loads((output / "priors.json").read_text())
    products = json.loads((output / "catalogue.json").read_text())
    overlaps = json.loads((output / "overlaps_private.json").read_text())
    scenes, rejected = overlaps["scenes"], overlaps["rejected"]
    rules = json.loads(POLICY.read_text())
    report = analyse(
        rows, scenes, frozen, rules, datetime.now(timezone.utc).isoformat()
    )
    report.update(
        archive=stats,
        catalogue_records=len(products),
        hh_hv_scenes=len(scenes),
        rejected_catalogue_records=len(rejected),
        inputs={
            "policy": ref(POLICY),
            "region": ref(REGION_PATH),
            "snapshot": ref(output / "snapshot_manifest.json"),
            "priors": ref(output / "priors.json"),
            "catalogue": ref(output / "catalogue.json"),
            "overlaps": ref(output / "overlaps_private.json"),
            "implementation": ref(Path(__file__)),
        },
    )
    save(REPORT, report)
    save(MAP, geojson(report))
    VIEW.write_text(render(report), encoding="utf-8")
    save(LOCK, {"report": ref(REPORT), "geojson": ref(MAP), "view": ref(VIEW)})
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "nl_reports",
                    "observed_cells",
                    "split_safe_observed_cells",
                    "shortlist",
                )
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    report = json.loads(checked(lock["report"]).read_text())
    rules = json.loads(checked(report["inputs"]["policy"]).read_text())
    checked(report["inputs"]["implementation"])
    checked(report["inputs"]["region"])
    if (
        any(report.get(k) is not v for k, v in GUARDS.items())
        or report["labels_admitted"] != 0
        or report["coverage_complete"] is not False
    ):
        raise ValueError("Provisional reception guard changed")
    if json.loads(checked(lock["geojson"]).read_text()) != geojson(report) or checked(
        lock["view"]
    ).read_text(encoding="utf-8") != render(report):
        raise ValueError("Map and report differ")
    if local:
        inputs = report["inputs"]
        manifest_path = checked(inputs["snapshot"])
        for entry in json.loads(manifest_path.read_text())["files"]:
            checked(
                {
                    "path": (manifest_path.parent / "archive_snapshot" / entry["file"])
                    .relative_to(REPO_ROOT)
                    .as_posix(),
                    "sha256": entry["sha256"],
                }
            )
        frozen = json.loads(checked(inputs["priors"]).read_text())
        for path, digest in frozen["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": digest})
        rows, stats = observations(manifest_path.parent / "archive_snapshot")
        products = json.loads(checked(inputs["catalogue"]).read_text())
        scenes, rejected = audit_products(
            [p for p in products if "_IW_GRDH_1SDH_" in p.get("Name", "")],
            rows,
            load_region(),
            set(),
            GeometryCollection(),
        )
        if json.loads(
            json.dumps({"scenes": scenes, "rejected": rejected})
        ) != json.loads(checked(inputs["overlaps"]).read_text()):
            raise ValueError("Private overlap replay differs")
        replay = json.loads(
            json.dumps(analyse(rows, scenes, frozen, rules, report["audited_at"]))
        )
        if stats != report["archive"] or any(report[k] != v for k, v in replay.items()):
            raise ValueError("Reception/shortlist replay differs")
    return {
        "verified": True,
        "local": local,
        "labels_admitted": 0,
        "shortlist": len(report["shortlist"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.build, args.publish, args.verify)) != 1:
        parser.error("Choose build, publish or verify")
    if args.build:
        build(args.build)
    elif args.publish:
        publish(args.publish)
    elif args.verify:
        print(json.dumps(verify(args.local)))
