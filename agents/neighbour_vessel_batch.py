"""Audit AIS collection per pass and reserve qualifying neighbouring windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, box, mapping

from agents.acquisition import native_acquisition_group
from agents.ais_pass_audit import collection, journals
from agents.artifacts import REPO_ROOT
from agents.cd_training_reservations import freeze_priors, rules
from agents.hh_hv_reservation import save
from agents.nl_reception import REPORT as RECEPTION
from agents.nl_reception import role
from agents.nl_reception import verify as verify_reception
from agents.overlap_audit import audit_products, fetch_catalogue, observations, snapshot
from agents.provisional_training import GUARDS
from agents.reception_vessel_batch import (
    COASTLINE,
    decide,
    enrich_sources,
    public,
    shortlist,
)
from agents.region import REGION_PATH, load_region
from agents.source_followup import fresh
from agents.source_reservation import checked, ref
from agents.training_ais_coverage import SELECTION

POLICY = REPO_ROOT / "configs/reservations/neighbour_vessels_v040.json"
DIRECTORY = REPO_ROOT / "reservations/cd_vessels/v0.4.0"
LOCK = REPO_ROOT / "configs/reservations/neighbour_vessels_v040.lock.json"


def expand(
    report: dict[str, Any], frozen: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    originals = shortlist(report)
    by_id = {c["id"]: c for c in report["cells"]}
    step, rings = policy["grid_degrees"], policy["neighbour_rings"]
    if step != 0.1 or not isinstance(rings, int) or not 0 <= rings <= 2:
        raise ValueError("Frozen bounded reception grid required")
    proposed: dict[tuple[int, int], dict[str, Any]] = {}
    for anchor in originals:
        x, y = by_id[anchor["id"]]["grid"]
        for dx in range(-rings, rings + 1):
            for dy in range(-rings, rings + 1):
                key = x + dx, y + dy
                candidate = {
                    "id": f"cell_{key[0]}_{key[1]}",
                    "grid": list(key),
                    "ring": max(abs(dx), abs(dy)),
                    "anchor_rank": anchor["rank"],
                    "geometry": mapping(
                        box(
                            key[0] * step,
                            key[1] * step,
                            (key[0] + 1) * step,
                            (key[1] + 1) * step,
                        )
                    ),
                }
                old = proposed.get(key)
                if old is None or (candidate["ring"], candidate["anchor_rank"]) < (
                    old["ring"],
                    old["anchor_rank"],
                ):
                    proposed[key] = candidate
    ordered = sorted(
        proposed.values(), key=lambda c: (c["ring"], c["anchor_rank"], c["id"])
    )
    return [
        c | {"rank": i} | role(c["geometry"], frozen, "reception_metadata_only")
        for i, c in enumerate(ordered, 1)
    ]


def analyse(output: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    policy = json.loads(POLICY.read_text())
    cells = json.loads((output / "search_cells.json").read_text())
    eligible = [c for c in cells if c["split_safe"]]
    scenes = json.loads((output / "shoreline_private.json").read_text())["scenes"]
    selection_rules = rules()
    result = decide(
        scenes,
        eligible,
        frozen,
        json.loads(checked(selection_rules["source_policy"]).read_text()),
        selection_rules,
    )
    for ranking in result["rankings"]:
        group = native_acquisition_group(ranking["name"])
        ranking["prior_acquisition_roles"] = sorted(
            {
                p["split"]
                for p in frozen["priors"]
                if p.get("acquisition_group") == group
            }
        )
    for area in result["areas"]:
        old = area["id"]
        area["id"] = old.replace("reception_v030_", "neighbour_v040_")
        for ranking in result["rankings"]:
            if ranking["reservation_id"] == old:
                ranking["reservation_id"] = area["id"]
    rows, _ = observations(output / "archive_snapshot")
    journal = journals(output / "archive_snapshot")
    passes = collection(
        scenes,
        rows,
        eligible,
        journal,
        policy["pass_context_half_window_seconds"],
        policy["grid_degrees"],
    )
    return {
        "result": result,
        "public": public(result, eligible, frozen),
        "pass_collection": passes,
        "malformed_journal_events": journal["malformed_events"],
    }


def capture(output: Path) -> None:
    fresh(output)
    verify_reception()
    snapshot(REPO_ROOT / "data/raw/ais_stream", output / "archive_snapshot")
    rows, stats = observations(output / "archive_snapshot")
    if not rows:
        raise ValueError("Nonempty AIS archive required")
    frozen = freeze_priors(SELECTION)
    save(output / "priors.json", frozen)
    cells = expand(
        json.loads(RECEPTION.read_text()), frozen, json.loads(POLICY.read_text())
    )
    save(output / "search_cells.json", cells)
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
    enrich_sources(scenes, [c for c in cells if c["split_safe"]])
    save(output / "shoreline_private.json", {"scenes": scenes})
    save(
        output / "source_inputs.json",
        {
            "coastline": [
                ref(p) for p in sorted(COASTLINE.parent.glob("land_polygons.*"))
            ],
            "controls": [
                ref(p)
                for p in sorted((REPO_ROOT / "configs/coastline").glob("*"))
                if p.is_file() and p.suffix in {".json", ".geojson"}
            ],
        },
    )
    print(
        json.dumps(
            {
                "captured": str(output),
                "reports": len(rows),
                "cells": len(cells),
                "eligible_cells": sum(c["split_safe"] for c in cells),
                "hh_hv_scenes": len(scenes),
            }
        )
    )


def publish(output: Path) -> None:
    if DIRECTORY.exists() or LOCK.exists():
        raise ValueError("Fresh reservation release required")
    frozen = json.loads((output / "priors.json").read_text())
    analysis = analyse(output, frozen)
    result = analysis["result"]
    save(output / "selection_private.json", frozen | result)
    save(
        output / "annotation_template.json",
        {
            **GUARDS,
            "scores_viewed": False,
            "prior_project_exposure_disclosed": True,
            "native_specification": rules()["native_specification"],
            "areas": [
                {
                    "id": a["id"],
                    "product_id": a["product_id"],
                    "name": a["name"],
                    "native_acquisition_group": a["native_acquisition_group"],
                    "geographic_group": a["geographic_group"],
                    "geometry": a["geometry"],
                    "split": "train",
                    "objects": [],
                    "annotation_completed_utc": None,
                    "all_visible_returns_recorded": False,
                    "source_uncertainty_covered": False,
                    "native_input_verified": False,
                }
                for a in result["areas"]
            ],
        },
    )
    _, stats = observations(output / "archive_snapshot")
    cells = json.loads((output / "search_cells.json").read_text())
    policy = json.loads(POLICY.read_text())
    manifest = analysis["public"] | {
        "version": "0.4.0",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "archive": stats,
        "search_cells": cells,
        "search_exclusion_counts": dict(
            Counter(c["exclusion"] for c in cells if not c["split_safe"])
        ),
        "pass_collection": analysis["pass_collection"],
        "malformed_journal_events": analysis["malformed_journal_events"],
        "private_selection": ref(output / "selection_private.json"),
        "annotation_template": ref(output / "annotation_template.json"),
        "inputs": {
            "policy": ref(POLICY),
            "reception": ref(RECEPTION),
            "region": ref(REGION_PATH),
            "source_policy": rules()["source_policy"],
            "selection_policy": ref(
                REPO_ROOT / "configs/reservations/provisional_training_v010.json"
            ),
            "native_specification": rules()["native_specification"],
        },
        "private_inputs": {
            name: ref(output / filename)
            for name, filename in {
                "snapshot": "snapshot_manifest.json",
                "catalogue": "catalogue.json",
                "overlaps": "overlaps_private.json",
                "shoreline": "shoreline_private.json",
                "source_inputs": "source_inputs.json",
                "cells": "search_cells.json",
            }.items()
        },
        "implementation_dependencies": [
            ref(REPO_ROOT / "agents" / name)
            for name in (
                "neighbour_vessel_batch.py",
                "ais_pass_audit.py",
                "reception_vessel_batch.py",
                "ais_recorder.py",
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
        "limits": policy["limits"],
    }
    save(DIRECTORY / "manifest.json", manifest)
    save(LOCK, {"manifest": ref(DIRECTORY / "manifest.json")})
    print(
        json.dumps(
            {
                "reserved": manifest["reserved"],
                "supports_screened": len(result["source_screens"]),
                "passes_audited": len(manifest["pass_collection"]),
                "source_rejections": result["source_rejections"],
                "rankings": result["rankings"],
            }
        )
    )


def verify(local: bool = False) -> dict[str, Any]:
    manifest = json.loads(checked(json.loads(LOCK.read_text())["manifest"]).read_text())
    for reference in (
        list(manifest["inputs"].values()) + manifest["implementation_dependencies"]
    ):
        checked(reference)
    if (
        any(manifest.get(k) is not v for k, v in GUARDS.items())
        or manifest["labels_admitted"] != 0
        or manifest["reserved"] != len(manifest["areas"])
        or any(
            manifest.get(k) is not False
            for k in (
                "classifier_scored",
                "imagery_inspected",
                "accuracy_truth_admitted",
                "exploratory_training_admitted",
            )
        )
    ):
        raise ValueError("Provisional reservation guard changed")
    for area in manifest["areas"]:
        if (
            area["split"] != "train"
            or area["label"] is not None
            or any(area.get(k) is not v for k, v in GUARDS.items())
        ):
            raise ValueError("Training-only unlabelled reservation required")
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        output = checked(manifest["private_inputs"]["snapshot"]).parent
        for reference in manifest["private_inputs"].values():
            checked(reference)
        for path, sha in private["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        for entry in json.loads((output / "snapshot_manifest.json").read_text())[
            "files"
        ]:
            checked(
                {
                    "path": (output / "archive_snapshot" / entry["file"])
                    .relative_to(REPO_ROOT)
                    .as_posix(),
                    "sha256": entry["sha256"],
                }
            )
        for entries in json.loads((output / "source_inputs.json").read_text()).values():
            for entry in entries:
                checked(entry)
        cells = expand(
            json.loads(RECEPTION.read_text()), private, json.loads(POLICY.read_text())
        )
        if json.loads(json.dumps(cells)) != manifest["search_cells"]:
            raise ValueError("Neighbour geometry replay differs")
        rows, stats = observations(output / "archive_snapshot")
        scenes, rejected = audit_products(
            [
                p
                for p in json.loads((output / "catalogue.json").read_text())
                if "_IW_GRDH_1SDH_" in p.get("Name", "")
            ],
            rows,
            load_region(),
            set(),
            GeometryCollection(),
        )
        if json.loads(
            json.dumps({"scenes": scenes, "rejected": rejected})
        ) != json.loads((output / "overlaps_private.json").read_text()):
            raise ValueError("Overlap replay differs")
        enrich_sources(scenes, [c for c in cells if c["split_safe"]])
        if json.loads(json.dumps({"scenes": scenes})) != json.loads(
            (output / "shoreline_private.json").read_text()
        ):
            raise ValueError("Shoreline replay differs")
        # JSON persists coordinate tuples as lists. Compare the serialized
        # contract for nonempty reservations as well as the real empty release.
        replay = json.loads(json.dumps(analyse(output, private)))
        if (
            stats != manifest["archive"]
            or replay["result"]["areas"] != private["areas"]
            or any(manifest[k] != v for k, v in replay["public"].items())
            or replay["pass_collection"] != manifest["pass_collection"]
        ):
            raise ValueError("Pass collection/reservation replay differs")
        template = json.loads(checked(manifest["annotation_template"]).read_text())
        if (
            template["scores_viewed"] is not False
            or len(template["areas"]) != manifest["reserved"]
        ):
            raise ValueError("Annotation handoff differs")
    return {
        "verified": True,
        "local": local,
        "reserved": manifest["reserved"],
        "labels_admitted": 0,
        "passes_audited": len(manifest["pass_collection"]),
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
