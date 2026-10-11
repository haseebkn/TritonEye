"""Extend frozen NL review pilots without inventing labels or review history.

Draft assembly consumes an existing image-only bundle and public catalogue.
Annotation decisions are supplied after native image inspection. Historical
objects, reviews and test assignments are inherited unchanged; AIS histories
stay outside the portable image review document.
"""

from __future__ import annotations

import argparse
import base64
import copy
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
from pyproj import Geod, Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.geo import Georeferencer
from agents.nl_benchmark import (
    bounded_path,
    load_dataset,
    objects_digest,
    selection_lock,
    validate_contract,
    verify_chip,
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def entry(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def inherit_splits(parent: dict[str, Any], manifest: dict[str, Any]) -> None:
    """An extension cannot reassign old acquisitions, geography or selected areas."""
    old = {s["product_id"]: s for s in parent["scenes"]}
    new = {s["product_id"]: s for s in manifest["scenes"]}
    if any(pid not in new or new[pid] != scene for pid, scene in old.items()):
        raise ValueError("Inherited scene changed or missing")
    acquisition_splits = {s["acquisition_group"]: s["split"] for s in old.values()}
    projection = Transformer.from_crs(4326, 3347, always_xy=True)
    geography = [
        (s["split"], transform(projection.transform, shape(r["geometry"])))
        for s in old.values()
        for r in s["rois"]
    ]
    for pid, scene in new.items():
        if pid in old:
            continue
        if scene["split"] == "test":
            raise ValueError("Exposed review-bundle scenes cannot become new test data")
        prior = acquisition_splits.get(scene["acquisition_group"])
        if prior and prior != scene["split"]:
            raise ValueError("Inherited datatake crosses splits")
        for roi in scene["rois"]:
            projected = transform(projection.transform, shape(roi["geometry"]))
            if any(
                split != scene["split"]
                and projected.distance(geom) < parent["separation_m"]
                for split, geom in geography
            ):
                raise ValueError("Inherited geographic split overlap")


def draft(root: Path, plan_path: Path, directory: Path) -> None:
    """Freeze extension inputs and split selection before new annotation."""
    plan = read(plan_path)
    version = plan["dataset_version"]
    if not isinstance(version, str) or not version or Path(version).name != version:
        raise ValueError("Simple dataset version required")
    if (
        directory.exists()
        or (root / "datasets/nl_benchmark" / ("v" + version) / "release.json").exists()
    ):
        raise ValueError("Fresh dataset version and directory required")
    parent_dir = bounded_path(root, plan["parent"])
    parent, _, _ = load_dataset(parent_dir, artifact_root=root)
    bundle = bounded_path(root, plan["bundle"])
    image_manifest_path = bundle / "image_first/manifest.json"
    image_manifest = read(image_manifest_path)
    products = {p["Id"]: p for p in read(bundle / "catalogue_products.json")}
    assignments = plan["assignments"]
    areas = image_manifest["areas"]
    if set(assignments) != {a["id"] for a in areas}:
        raise ValueError("Every prepared area requires exactly one assignment")
    selected_at = datetime.now(timezone.utc).isoformat()
    manifest = copy.deepcopy(parent)
    manifest.update(
        dataset_version=version,
        status="provisional_extended_review_pilot",
        parent_release=entry(root, parent_dir / "release.json"),
        extension={
            "plan": entry(root, plan_path),
            "image_manifest": entry(root, image_manifest_path),
            "selection_lock": entry(root, bundle / "image_first/selection_lock.json"),
            "image_inventory": entry(root, bundle / "image_first/bundle_hashes.json"),
            "snapshot_manifest": entry(root, bundle / "snapshot_manifest.json"),
            "catalogue": entry(root, bundle / "catalogue_products.json"),
            "assembly_method": entry(root, Path(__file__)),
            "selected_at": selected_at,
            "failed_areas": image_manifest["failures"],
            "sampling": (
                "Inherited geographic pilot plus AIS-informed development enrichment."
            ),
            "pretraining_overlap": "not_verified; no independent generalization claim",
        },
    )
    new_scenes: dict[str, dict[str, Any]] = {}
    template = []
    for area in areas:
        assignment = assignments[area["id"]]
        product = products[area["product_id"]]
        roi = copy.deepcopy(area)
        roi.update(group=assignment["group"], regime=assignment["regime"])
        for field in ("chip", "review_grid"):
            roi[field]["path"] = roi[field]["path"].replace("\\", "/")
        for field in ("previews", "calibration"):
            for item in roi[field]:
                item["path"] = item["path"].replace("\\", "/")
        for item in roi["previews"] + [roi["chip"], roi["review_grid"]]:
            if sha256_file(bounded_path(root, item["path"])) != item["sha256"]:
                raise ValueError("Prepared image hash mismatch")
        verify_chip(bounded_path(root, roi["chip"]["path"]), roi, [])
        scene = new_scenes.setdefault(
            product["Id"],
            {
                "name": product["Name"],
                "product_id": product["Id"],
                "acquisition_group": native_acquisition_group(product["Name"]),
                "split": assignment["split"],
                "region": assignment["region"],
                "season": "autumn",
                "prior_development_exposure": True,
                "acquisition_time": product["ContentDate"]["Start"],
                "acquisition_end": product["ContentDate"]["End"],
                "footprint": product["GeoFootprint"],
                "polarizations": [p.lower() for p in area["polarizations"]],
                "sources": [
                    {"path": b["path"].replace("\\", "/"), "sha256": b["sha256"]}
                    for b in image_manifest["source_bands"]
                    if b["product_id"] == product["Id"]
                ],
                "catalogue": entry(root, bundle / "catalogue_products.json"),
                "ais": {
                    "truth_role": "supporting_only",
                    "status": "frozen_support_not_yet_adjudicated",
                    "snapshot": entry(root, bundle / "snapshot_manifest.json"),
                    "licence": (
                        "Private AISstream supporting archive; do not redistribute "
                        "identities or positions."
                    ),
                    "reception": "historical_gaps_or_unknown",
                },
                "rois": [],
            },
        )
        if (scene["split"], scene["region"]) != (
            assignment["split"],
            assignment["region"],
        ):
            raise ValueError("One scene has conflicting assignments")
        scene["rois"].append(roi)
        template.append(
            {
                "roi_id": roi["id"],
                "source_chip_sha256": roi["chip"]["sha256"],
                "cells_reviewed": [],
                "complete": False,
                "notes": "",
                "objects": [],
            }
        )
    manifest["scenes"].extend(new_scenes.values())
    inherit_splits(parent, manifest)
    selection = {
        "dataset_version": version,
        "status": manifest["status"],
        "separation_m": parent["separation_m"],
        "selection_basis": manifest["extension"]["sampling"],
        "scenes": [
            {
                **{
                    k: s[k]
                    for k in (
                        "name",
                        "product_id",
                        "split",
                        "region",
                        "season",
                        "prior_development_exposure",
                    )
                },
                "rois": [
                    {k: r[k] for k in ("id", "group", "regime", "center", "size")}
                    for r in s["rois"]
                ],
            }
            for s in manifest["scenes"]
        ],
    }
    directory.mkdir(parents=True)
    write_json(directory / "selection.json", selection)
    manifest["selection_sha256"] = sha256_file(directory / "selection.json")
    write_json(directory / "split_lock.json", selection_lock(selection))
    write_json(directory / "manifest.json", manifest)
    write_json(
        directory / "annotation_decisions.json",
        {
            "dataset_version": version,
            "annotator": "codex_primary_machine_assisted",
            "qualifications": (
                "AI image inspection; not a human SAR expert or independent reviewer."
            ),
            "areas": template,
        },
    )


def finalize(root: Path, directory: Path) -> dict[str, Any]:
    """Bind explicit new decisions to images; preserve old reviews verbatim."""
    manifest = read(directory / "manifest.json")
    version = manifest["dataset_version"]
    output = root / "data/benchmarks/nl" / version / "review_bundle.html"
    if (
        (directory / "release.json").exists()
        or (root / "datasets/nl_benchmark" / ("v" + version) / "release.json").exists()
        or (output.parent / "release_record.json").exists()
    ):
        raise ValueError("Released version is immutable")
    parent_entry = manifest["parent_release"]
    parent_dir = bounded_path(root, parent_entry["path"]).parent
    if sha256_file(parent_dir / "release.json") != parent_entry["sha256"]:
        raise ValueError("Parent release changed")
    parent, inherited, _ = load_dataset(parent_dir, artifact_root=root)
    inherit_splits(parent, manifest)
    history = read(parent_dir / "review_history.json")
    history["dataset_version"] = version
    objects = inherited["objects"]
    now = datetime.now(timezone.utc).isoformat()
    history["events"].append(
        {
            "id": "extension_selection_locked",
            "kind": "selection_locked",
            "timestamp": manifest["extension"]["selected_at"],
            "notes": (
                "New assignments frozen before new image inspection; earlier "
                "exposure and inherited selection remain disclosed."
            ),
        }
    )
    decisions = read(directory / "annotation_decisions.json")
    drafts = {a["roi_id"]: a for a in decisions["areas"]}
    parent_ids = {s["product_id"] for s in parent["scenes"]}
    new_scenes = [s for s in manifest["scenes"] if s["product_id"] not in parent_ids]
    if (
        decisions["dataset_version"] != version
        or set(drafts) != {r["id"] for s in new_scenes for r in s["rois"]}
        or len(drafts) != len(decisions["areas"])
    ):
        raise ValueError("Exactly one decision per new area required")
    sections = []
    for scene in new_scenes:
        for roi in scene["rois"]:
            rid = roi["id"]
            decision = drafts[rid]
            if (
                decision["source_chip_sha256"] != roi["chip"]["sha256"]
                or not decision["complete"]
                or not decision["notes"]
                or sorted(decision["cells_reviewed"])
                != sorted(c["id"] for c in roi["cells"])
            ):
                raise ValueError("Explicit complete image-first pass required")
            chip = bounded_path(root, roi["chip"]["path"])
            if sha256_file(chip) != roi["chip"]["sha256"]:
                raise ValueError("Reviewed chip changed")
            labels = []
            with (
                rasterio.open(chip) as source,
                Georeferencer.from_dataset(source) as geo,
            ):
                for i, obj in enumerate(decision["objects"], 1):
                    x0, y0, x1, y1 = obj["bbox_px"]
                    lon, lat = geo.xy((y0 + y1) / 2, (x0 + x1) / 2)
                    labels.append(
                        {
                            **obj,
                            "id": f"{rid}_{i:03d}",
                            "roi_id": rid,
                            "product_id": scene["product_id"],
                            "annotator": decisions["annotator"],
                            "ais_role": "supporting_only",
                            "evidence": [
                                "Native image-first inspection of both bands; "
                                "detector output hidden."
                            ],
                            "geometry": {
                                "type": "Point",
                                "coordinates": [float(lon[0]), float(lat[0])],
                            },
                        }
                    )
            objects.extend(labels)
            history["events"].append(
                {
                    "id": f"{rid}_first_pass",
                    "kind": "first_pass",
                    "timestamp": now,
                    "roi_id": rid,
                    "reviewer": decisions["annotator"],
                    "independent": False,
                    "qualifications": decisions["qualifications"],
                    "source_chip_sha256": roi["chip"]["sha256"],
                    "annotation_digest": objects_digest(labels),
                    "cells_reviewed": decision["cells_reviewed"],
                    "decision": "complete_first_pass",
                    "notes": decision["notes"]
                    + " Timestamp records entry, not inspection start.",
                }
            )
            figures = []
            for pol, preview in zip(scene["polarizations"], roi["previews"]):
                path = bounded_path(root, preview["path"])
                if sha256_file(path) != preview["sha256"]:
                    raise ValueError("Reviewed preview changed")
                encoded = base64.b64encode(path.read_bytes()).decode()
                figures.append(
                    f"<figure><figcaption>{pol.upper()}</figcaption>"
                    f'<img width="{roi["size"]}" height="{roi["size"]}" '
                    f'src="data:image/png;base64,{encoded}"></figure>'
                )
            sections.append(
                f"<section><h2>{html.escape(rid)}</h2>"
                f'<p>{scene["acquisition_time"]} · {scene["split"]} · '
                f'{scene["product_id"]}</p>'
                f'<div class="images">{"".join(figures)}</div><details><summary>'
                "First-pass decisions: open after your own full-area inspection"
                "</summary>"
                f"<pre>{html.escape(json.dumps(decision, indent=2))}</pre>"
                "</details></section>"
            )
    annotation = {
        "dataset_version": version,
        "source_kind": "machine_assisted_sar_first_pass",
        "objects": objects,
    }
    status = validate_contract(
        manifest,
        read(directory / "selection.json"),
        read(directory / "split_lock.json"),
        annotation,
        history,
    )
    status["step_3_complete"] = False
    status["parent_version"] = parent["dataset_version"]
    for name, content in (
        ("annotations.json", annotation),
        ("review_history.json", history),
        ("status.json", status),
    ):
        write_json(directory / name, content)
    # Persist the image-first annotations before consulting private telemetry.
    # Supporting proximity cannot revise these labels or certify identity.
    context = supporting_context(root, manifest, objects)
    context["first_pass_annotations_sha256"] = sha256_file(
        directory / "annotations.json"
    )
    write_json(directory / "ais_support.json", context)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>NL benchmark {html.escape(version)} review</title><style>"
        "body{font:16px system-ui;margin:24px;background:#edf2f4}"
        "section{background:white;padding:20px;margin:20px 0}"
        ".images{display:flex;flex-wrap:wrap;gap:16px}"
        "img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>"
        "<h1>NL benchmark extension: image-first review</h1><p>Contains modified "
        "Copernicus Sentinel data (2025, 2026). AIS-informed selection; provisional "
        "AI first pass; independent review unavailable. Review all four cells "
        "in both bands, including empty-looking areas.</p>"
        "<p>Inherited areas and original review histories remain in the parent "
        "bundle. Test areas stay locked. No detector predictions or AIS markers "
        "appear here.</p>" + "".join(sections) + "</html>",
        encoding="utf-8",
    )
    names = [
        "selection.json",
        "split_lock.json",
        "manifest.json",
        "annotation_decisions.json",
        "annotations.json",
        "review_history.json",
        "status.json",
        "ais_support.json",
        "ANNOTATION_GUIDE.md",
        "DATASET_CARD.md",
    ]
    write_json(
        directory / "release.json",
        {
            "dataset_version": version,
            "status": "provisional_not_independently_reviewed",
            "created_at": now,
            "finalization_method": entry(root, Path(__file__)),
            "files": {n: sha256_file(directory / n) for n in names},
            "review_bundle": entry(root, output),
        },
    )
    write_json(
        output.parent / "release_record.json",
        {
            "dataset_version": version,
            "release_sha256": sha256_file(directory / "release.json"),
        },
    )
    return status


def supporting_context(
    root: Path, manifest: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate frozen midpoint AIS context after the image-first pass.

    Every scene competitor is retained, including positions outside a chip.
    These counts are corroborating proximity, not one-to-one matching accuracy.
    """
    image_manifest = bounded_path(root, manifest["extension"]["image_manifest"]["path"])
    bundle = image_manifest.parent.parent
    geod = Geod(ellps="WGS84")
    results = []
    for scene in manifest["scenes"]:
        for roi in scene["rois"]:
            path = bundle / "telemetry_second_pass" / (roi["id"] + ".json")
            if not path.exists():
                continue  # Inherited areas keep their historical supporting snapshot.
            telemetry = read(path)
            if telemetry["product_id"] != scene["product_id"]:
                raise ValueError("Supporting telemetry product substitution")
            tracks = telemetry["support"]
            if len({t["mmsi"] for t in tracks}) != len(tracks):
                raise ValueError("Duplicate aligned identity in supporting pool")
            row: dict[str, Any] = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "telemetry": entry(root, path),
                "scene_competitors": len(tracks),
                "tracks_in_selected_area": sum(
                    shape(roi["geometry"]).covers(Point(t["lon"], t["lat"]))
                    for t in tracks
                ),
                "vessel_interpretations": [],
                "reviewed_correct_identities": None,
            }
            for obj in objects:
                if obj["roi_id"] != roi["id"] or obj["class"] != "vessel":
                    continue
                lon, lat = obj["geometry"]["coordinates"]
                distances = [
                    abs(geod.inv(lon, lat, t["lon"], t["lat"])[2]) for t in tracks
                ]
                row["vessel_interpretations"].append(
                    {
                        "annotation_id": obj["id"],
                        "supporting_tracks_within_500m": sum(
                            d <= 500 for d in distances
                        ),
                        "nearest_track_distance_m": (
                            round(min(distances), 1) if distances else None
                        ),
                        "identity_decision": "unreviewed",
                    }
                )
            results.append(row)
    return {
        "dataset_version": manifest["dataset_version"],
        "time_reference": "Frozen overlap-bundle catalogue midpoint alignment",
        "truth_role": "supporting_only",
        "independent_review_available": False,
        "proximity_gate_m": 500,
        "accuracy_measured": False,
        "areas": results,
        "limits": [
            "AIS-informed sampling; incomplete historical reception.",
            "A nearby track does not establish correct object identity.",
            "These counts are not detector recall, exhaustive truth or improvement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("draft", "finalize"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.action == "draft":
        if args.plan is None:
            parser.error("draft requires --plan")
        draft(REPO_ROOT, args.plan, args.dataset)
        print("Draft selection frozen; image annotation is still pending.")
    else:
        print(json.dumps(finalize(REPO_ROOT, args.dataset), indent=2))


if __name__ == "__main__":
    main()
