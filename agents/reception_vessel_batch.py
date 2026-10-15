"""Reserve source-supported training windows from the frozen reception shortlist."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, Point, shape

from agents.artifacts import REPO_ROOT
from agents.cd_training_reservations import freeze_priors, rules
from agents.hh_hv_reservation import save
from agents.landmask import LandMask
from agents.nl_reception import REPORT as RECEPTION
from agents.nl_reception import role
from agents.nl_reception import verify as verify_reception
from agents.overlap_audit import audit_products, fetch_catalogue, observations, snapshot
from agents.provisional_training import GUARDS, public_area, select
from agents.region import REGION_PATH, load_region
from agents.source_followup import fresh
from agents.source_reservation import checked, ref, vessel_opportunities
from agents.training_ais_coverage import SELECTION

POLICY = REPO_ROOT / "configs/reservations/reception_vessels_v030.json"
DIRECTORY = REPO_ROOT / "reservations/cd_vessels/v0.3.0"
LOCK = REPO_ROOT / "configs/reservations/reception_vessels_v030.lock.json"
COASTLINE = REPO_ROOT / "data/reference/land-polygons-split-4326/land_polygons.shp"


def shortlist(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the existing ordered eight cells; do not rerank after seeing images."""
    cells = {c["id"]: c for c in report["cells"]}
    return [
        {
            "rank": a["rank"],
            "id": a["cell_id"],
            "geometry": cells[a["cell_id"]]["geometry"],
        }
        for a in report["shortlist"]
    ]


def locate(
    support: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any] | None:
    point = Point(support["lon"], support["lat"])
    return next((c for c in cells if shape(c["geometry"]).covers(point)), None)


def source_reasons(
    support: dict[str, Any], centre: float, policy: dict[str, Any]
) -> list[str]:
    reasons = []
    reports = support.get("reports", [])
    nearest = min(reports, key=lambda r: abs(r["time"] - centre)) if reports else None
    try:
        speed = float(nearest["values"]["speed_knots"]) if nearest else math.nan
    except (KeyError, ValueError, TypeError):
        speed = math.nan
    checks = {
        "nearest_report_over_120s_or_missing": support.get("nearest_offset_s"),
        "bracket_over_240s_or_missing": support.get("bracket_gap_s"),
        "uncertainty_over_500m_or_invalid": support.get("uncertainty_m"),
    }
    thresholds = [
        policy["ais_nearest_max_seconds"],
        policy["ais_bracket_max_seconds"],
        policy["ais_max_uncertainty_m"],
    ]
    for (name, raw), maximum in zip(checks.items(), thresholds):
        if (
            not isinstance(raw, (int, float))
            or not math.isfinite(raw)
            or abs(raw) > maximum
            or (name != "nearest_report_over_120s_or_missing" and raw < 0)
        ):
            reasons.append(name)
    if not math.isfinite(speed) or not policy["ais_min_speed_knots"] <= speed < 102.3:
        reasons.append("motion_below_1kt_or_invalid")
    distance = support.get("signed_shore_distance_m")
    if (
        support.get("physical_surface") != "water"
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
        or distance < policy["ais_min_shore_distance_m"]
    ):
        reasons.append("reference_water_and_500m_shore_separation_unmet")
    return reasons


def decide(
    scenes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    frozen: dict[str, Any],
    source_policy: dict[str, Any],
    selection_rules: dict[str, Any],
) -> dict[str, Any]:
    """Metadata/source decisions only; preserve competing histories privately."""
    pool = []
    rejection_counts: Counter[str] = Counter()
    source_rows = []
    cell_roles = {
        c["id"]: role(c["geometry"], frozen, "reception_metadata_only") for c in cells
    }
    for scene in scenes:
        centre = datetime.fromisoformat(scene["reference_midpoint_utc"]).timestamp()
        for support in scene["support"]:
            cell = locate(support, cells)
            if cell is None:
                continue
            reasons = source_reasons(support, centre, source_policy)
            if not cell_roles[cell["id"]]["split_safe"]:
                reasons.append("shortlist_cell_training_role_unmet")
            rejection_counts.update(reasons)
            source_rows.append(
                {
                    "cell_id": cell["id"],
                    "product_id": scene["product"]["Id"],
                    "midpoint_utc": scene["reference_midpoint_utc"],
                    "reasons": reasons,
                    "source_screen_passed": not reasons,
                }
            )
            if reasons:
                continue
            # The established source consumer repeats the screen and retains the
            # complete swath-wide aligned competing pool, not just this cell.
            candidates = vessel_opportunities({"scenes": [scene]}, source_policy)
            for candidate in candidates:
                if candidate["source_identity"] == support["mmsi"]:
                    pool.append(
                        candidate
                        | {"source_cell_id": cell["id"], "priority": cell["rank"]}
                    )
    pool.sort(
        key=lambda r: (
            r["priority"],
            r["absolute_offset_s"],
            r["bracket_gap_s"],
            r["search_uncertainty_m"],
            r["product"]["Id"],
            r["source_identity"],
        )
    )
    selected, ranked = select(
        pool, frozen["priors"], set(frozen["exposed"]), frozen["old"], selection_rules
    )
    # Stable unique IDs across separately versioned reservations.
    for area in selected:
        old_id = area["id"]
        area["id"] = "reception_v030_" + old_id
        for row in ranked:
            if row["reservation_id"] == old_id:
                row["reservation_id"] = area["id"]
    return {
        "areas": selected,
        "rankings": ranked,
        "source_screens": source_rows,
        "source_rejections": dict(rejection_counts),
    }


def public(
    result: dict[str, Any], cells: list[dict[str, Any]], frozen: dict[str, Any]
) -> dict[str, Any]:
    return {
        "areas": [
            public_area(a)
            | {
                "source_cell_id": a["source_cell_id"],
                "status": "reserved_pending_native_object_and_permission_checks",
                "native_input_verified": False,
                "processing_version_verified": False,
                "exploratory_training_admitted": False,
            }
            for a in result["areas"]
        ],
        "rankings": result["rankings"],
        "source_screens": result["source_screens"],
        "source_rejections": result["source_rejections"],
        "prospective_priorities": [
            c
            | role(c["geometry"], frozen, "reception_metadata_only")
            | {"acquisition_reserved": False}
            for c in cells
        ],
        "reserved": len(result["areas"]),
        "status": (
            "metadata_only_training_reservation"
            if result["areas"]
            else "blocked_no_qualifying_source_overlap"
        ),
        "labels_admitted": 0,
        "annotated_objects": 0,
        "classifier_scored": False,
        "imagery_inspected": False,
        "accuracy_truth_admitted": False,
        "exploratory_training_admitted": False,
        **GUARDS,
    }


def enrich_sources(scenes: list[dict[str, Any]], cells: list[dict[str, Any]]) -> None:
    """Apply the same retained physical reference on capture and local replay."""
    for scene in scenes:
        local = [s for s in scene["support"] if locate(s, cells)]
        if not local:
            continue
        points = [Point(s["lon"], s["lat"]) for s in local]
        bounds = GeometryCollection(points).bounds
        mask = LandMask.for_footprint(bounds, shapefile=str(COASTLINE))
        surfaces, distances = mask.classify_physical(
            [s["lon"] for s in local], [s["lat"] for s in local]
        )
        for support, surface, distance in zip(local, surfaces, distances):
            support.update(
                physical_surface=surface,
                signed_shore_distance_m=(
                    float(distance) if math.isfinite(distance) else None
                ),
            )
        scene["shoreline_validation"] = mask.validation


def capture(output: Path) -> None:
    fresh(output)
    snapshot(REPO_ROOT / "data/raw/ais_stream", output / "archive_snapshot")
    rows, stats = observations(output / "archive_snapshot")
    if not rows:
        raise ValueError("Nonempty AIS snapshot required")
    verify_reception()
    save(output / "priors.json", freeze_priors(SELECTION))
    products = fetch_catalogue(stats["first_utc"], stats["last_utc"], load_region())
    save(output / "catalogue.json", products)
    scenes, rejected = audit_products(
        [p for p in products if "_IW_GRDH_1SDH_" in p.get("Name", "")],
        rows,
        load_region(),
        set(),
        GeometryCollection(),
    )
    save(output / "overlaps_private.json", {"scenes": scenes, "rejected": rejected})
    cells = shortlist(json.loads(RECEPTION.read_text()))
    # Classify source positions only. Never read SAR, optical or locked-test pixels.
    enrich_sources(scenes, cells)
    save(output / "shoreline_private.json", {"scenes": scenes})
    coast_files = sorted(COASTLINE.parent.glob("land_polygons.*"))
    controls = sorted((REPO_ROOT / "configs/coastline").glob("*.json")) + sorted(
        (REPO_ROOT / "configs/coastline").glob("*.geojson")
    )
    save(
        output / "source_inputs.json",
        {
            "coastline": [ref(p) for p in coast_files],
            "controls": [ref(p) for p in controls],
        },
    )
    print(
        json.dumps(
            {
                "captured": str(output),
                "archive": stats,
                "catalogue_products": len(products),
                "hh_hv_scenes": len(scenes),
            }
        )
    )


def publish(output: Path) -> None:
    if DIRECTORY.exists() or LOCK.exists():
        raise ValueError("Fresh public reservation version required")
    verify_reception()
    policy = json.loads(POLICY.read_text())
    selection_rules = rules()
    if (
        policy["batch_cap"] != selection_rules["maximum_per_stream"]
        or policy["maximum_per_native_acquisition"]
        != selection_rules["maximum_per_native_acquisition"]
    ):
        raise ValueError("Reservation cap differs from frozen admission policy")
    source_policy = json.loads(checked(selection_rules["source_policy"]).read_text())
    frozen = json.loads((output / "priors.json").read_text())
    scenes = json.loads((output / "shoreline_private.json").read_text())["scenes"]
    cells = shortlist(json.loads(RECEPTION.read_text()))
    result = decide(scenes, cells, frozen, source_policy, selection_rules)
    save(output / "selection_private.json", result | frozen)
    save(
        output / "annotation_template.json",
        {
            **GUARDS,
            "scores_viewed": False,
            "prior_project_exposure_disclosed": True,
            "areas": [
                {
                    "id": a["id"],
                    "product_id": a["product_id"],
                    "split": "train",
                    "geometry": a["geometry"],
                    "objects": [],
                    "annotation_completed_utc": None,
                    "native_input_verified": False,
                    "source_uncertainty_covered": False,
                    "all_visible_returns_recorded": False,
                }
                for a in result["areas"]
            ],
        },
    )
    _, stats = observations(output / "archive_snapshot")
    manifest = public(result, cells, frozen) | {
        "version": "0.3.0",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "archive": stats,
        "private_selection": ref(output / "selection_private.json"),
        "annotation_template": ref(output / "annotation_template.json"),
        "implementation_dependencies": [
            ref(REPO_ROOT / "agents" / name)
            for name in (
                "nl_reception.py",
                "cd_training_reservations.py",
                "provisional_training.py",
                "source_reservation.py",
                "overlap_audit.py",
                "association.py",
                "landmask.py",
                "acquisition.py",
                "nl_benchmark.py",
                "region.py",
            )
        ],
        "inputs": {
            "policy": ref(POLICY),
            "reception": ref(RECEPTION),
            "region": ref(REGION_PATH),
            "native_specification": selection_rules["native_specification"],
            "source_policy": selection_rules["source_policy"],
            "selection_policy": ref(
                REPO_ROOT / "configs/reservations/provisional_training_v010.json"
            ),
            "implementation": ref(Path(__file__)),
            "snapshot": ref(output / "snapshot_manifest.json"),
            "catalogue": ref(output / "catalogue.json"),
            "overlaps": ref(output / "overlaps_private.json"),
            "shoreline": ref(output / "shoreline_private.json"),
            "source_inputs": ref(output / "source_inputs.json"),
        },
        "limits": policy["limits"],
    }
    save(DIRECTORY / "manifest.json", manifest)
    save(LOCK, {"manifest": ref(DIRECTORY / "manifest.json")})
    print(
        json.dumps(
            {
                "reserved": manifest["reserved"],
                "source_screens": len(result["source_screens"]),
                "source_rejections": result["source_rejections"],
                "status": manifest["status"],
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    manifest = json.loads(checked(json.loads(LOCK.read_text())["manifest"]).read_text())
    inputs = manifest["inputs"]
    for name in (
        "policy",
        "reception",
        "region",
        "native_specification",
        "source_policy",
        "selection_policy",
        "implementation",
    ):
        checked(inputs[name])
    for dependency in manifest["implementation_dependencies"]:
        checked(dependency)
    if (
        any(manifest.get(k) is not v for k, v in GUARDS.items())
        or any(
            manifest.get(k) != v
            for k, v in {
                "labels_admitted": 0,
                "annotated_objects": 0,
                "classifier_scored": False,
                "imagery_inspected": False,
                "accuracy_truth_admitted": False,
                "exploratory_training_admitted": False,
            }.items()
        )
        or manifest["reserved"] != len(manifest["areas"])
    ):
        raise ValueError("Provisional reservation guards changed")
    for area in manifest["areas"]:
        if (
            area["split"] != "train"
            or area["label"] is not None
            or any(area.get(k) is not v for k, v in GUARDS.items())
        ):
            raise ValueError("Training-only unlabelled reservation required")
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        for path, sha in private["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        annotation = json.loads(checked(manifest["annotation_template"]).read_text())
        if (
            len(annotation["areas"]) != manifest["reserved"]
            or annotation["scores_viewed"] is not False
        ):
            raise ValueError("Annotation handoff differs")
        snapshot_path = checked(inputs["snapshot"])
        for entry in json.loads(snapshot_path.read_text())["files"]:
            checked(
                {
                    "path": (snapshot_path.parent / "archive_snapshot" / entry["file"])
                    .relative_to(REPO_ROOT)
                    .as_posix(),
                    "sha256": entry["sha256"],
                }
            )
        rows, stats = observations(snapshot_path.parent / "archive_snapshot")
        scenes, rejected = audit_products(
            [
                p
                for p in json.loads(checked(inputs["catalogue"]).read_text())
                if "_IW_GRDH_1SDH_" in p.get("Name", "")
            ],
            rows,
            load_region(),
            set(),
            GeometryCollection(),
        )
        if json.loads(
            json.dumps({"scenes": scenes, "rejected": rejected})
        ) != json.loads(checked(inputs["overlaps"]).read_text()):
            raise ValueError("Private overlap replay differs")
        for entries in json.loads(
            checked(inputs["source_inputs"]).read_text()
        ).values():
            for entry in entries:
                checked(entry)
        enriched = json.loads(checked(inputs["shoreline"]).read_text())["scenes"]
        cells = shortlist(json.loads(checked(inputs["reception"]).read_text()))
        enrich_sources(scenes, cells)
        if json.loads(json.dumps(scenes)) != enriched:
            raise ValueError("Physical reference replay differs")
        result = decide(
            enriched,
            cells,
            private,
            json.loads(checked(inputs["source_policy"]).read_text()),
            rules(),
        )
        if (
            stats != manifest["archive"]
            or result["areas"] != private["areas"]
            or any(manifest[k] != v for k, v in public(result, cells, private).items())
        ):
            raise ValueError("Source selection replay differs")
    return {
        "verified": True,
        "local": local,
        "reserved": manifest["reserved"],
        "labels_admitted": 0,
        "status": manifest["status"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", type=Path)
    action.add_argument("--publish", type=Path)
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.capture:
        capture(args.capture)
    elif args.publish:
        publish(args.publish)
    else:
        print(json.dumps(verify(args.local)))
