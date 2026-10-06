"""Versioned NL SAR annotations and fail-closed review/split contracts.

This dataset is independent of the live scene watcher. A machine-assisted first
pass is useful annotation work, not independent truth or a performance score.
The lock guards the exposed interface; it is not a security boundary against a
repository owner deliberately editing data or reading local test imagery.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

from agents.acquisition import product_id
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.geo import Georeferencer
from agents.region import load_region
from agents.run_versions import digest

CLASSES = {"vessel", "fixed_structure", "clutter", "ice", "unresolved"}
SPLITS = {"train", "validation", "test"}
DEFAULT_DATASET = REPO_ROOT / "datasets/nl_benchmark/v0.1.0"


def require_development_scene(
    selected_product_id: str, *metadata: dict[str, Any]
) -> None:
    selected = product_id(selected_product_id)
    groups: set[str] = set()
    records = [
        item for record in metadata for item in (record, record.get("benchmark") or {})
    ]
    for record in records:
        if record.get("split") == "test":
            raise ValueError("Held-out test scenes cannot be used for buffer trials")
        if record.get("acquisition_group"):
            groups.add(record["acquisition_group"])
        for field in ("name", "sar_product"):
            name = record.get(field)
            if name:
                tokens = name.split("_")
                if len(tokens) >= 8:
                    groups.add("_".join([tokens[0], tokens[6], tokens[7]]))
    held_out: set[str] = set()
    for path in (REPO_ROOT / "datasets/nl_benchmark").glob("*/split_lock.json"):
        lock = json.loads(path.read_text(encoding="utf-8"))
        held_out.update(
            assignment["acquisition_group"]
            for assignment in lock["assignments"]
            if assignment["split"] == "test"
        )
        manifest = json.loads(
            path.with_name("manifest.json").read_text(encoding="utf-8")
        )
        for scene in manifest["scenes"]:
            if product_id(scene["product_id"]) == selected:
                groups.add(scene["acquisition_group"])
                if scene["split"] == "test":
                    raise ValueError(
                        "Held-out test product cannot be used for buffer trials"
                    )
    if groups & held_out:
        raise ValueError("Held-out test datatake cannot be used for buffer trials")


def bounded_path(root: Path, relative: str) -> Path:
    """Manifests cannot read outside their declared local root."""
    path = (root / relative).resolve()
    if (
        not relative
        or Path(relative).is_absolute()
        or not path.is_relative_to(root.resolve())
    ):
        raise ValueError("Artifact path must remain inside its root")
    return path


def objects_digest(objects: list[dict[str, Any]]) -> str:
    return digest(sorted(objects, key=lambda o: o["id"]))


def verify_chip(path: Path, roi: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    """Verify actual pixels and box-centre geolocation, not manifest assertions."""
    with rasterio.open(path) as src:
        if src.width != roi["size"] or src.height != roi["size"] or src.count != 2:
            raise ValueError("Actual chip shape/band count mismatch")
        values = src.read(masked=True)
        if not np.all(src.read_masks()) or not np.isfinite(values.data).all():
            raise ValueError("Actual chip has invalid/nodata pixels")
        with Georeferencer.from_dataset(src) as geo:
            for obj in objects:
                x0, y0, x1, y1 = obj["bbox_px"]
                lon, lat = geo.xy((y0 + y1) / 2, (x0 + x1) / 2)
                expected = shape(obj["geometry"])
                if (
                    abs(float(lon[0]) - expected.x) > 1e-7
                    or abs(float(lat[0]) - expected.y) > 1e-7
                ):
                    raise ValueError(
                        "Annotation geography differs from its native pixel box"
                    )


def selection_lock(selection: dict[str, Any]) -> dict[str, Any]:
    """Immutable assignment made before annotation, including COG aliases."""
    assignments = []
    for scene in selection["scenes"]:
        tokens = scene["name"].split("_")
        assignments.append(
            {
                "name": scene["name"],
                "split": scene["split"],
                "acquisition_group": "_".join([tokens[0], tokens[6], tokens[7]]),
                "rois": scene["rois"],
            }
        )
    return {
        "dataset_version": selection["dataset_version"],
        "selection_digest": digest(selection),
        "assignments": assignments,
        "test_policy": (
            "No tuning, buffer comparison or training export; final evaluation "
            "requires a separate versioned release protocol."
        ),
    }


def validate_contract(
    manifest: dict[str, Any],
    selection: dict[str, Any],
    lock: dict[str, Any],
    annotations: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, Any]:
    """Validate content, leakage, complete-area accounting and review lineage.

    Review status is never inferred from AIS, detector detections, or a clean
    process exit. An explicit full-grid pass is required even for empty ROIs.
    Review claims remain assertions by the recorded reviewers, not proof of
    expertise. No score is calculated by this function.
    """
    if lock != selection_lock(selection):
        raise ValueError("Split lock differs from the original selection")
    version = manifest["dataset_version"]
    if any(
        d.get("dataset_version") != version
        for d in (selection, lock, annotations, history)
    ):
        raise ValueError("Dataset versions disagree")
    selected = {s["name"]: s for s in selection["scenes"]}
    scenes = manifest["scenes"]
    if len(selected) != len(selection["scenes"]) or len(scenes) != len(selected):
        raise ValueError("Duplicate/missing acquisition")
    if len({s["name"] for s in scenes}) != len(scenes):
        raise ValueError("Duplicate catalogue name")
    region = load_region()
    groups: dict[str, str] = {}
    acquisitions: dict[str, str] = {}
    roi_map: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    geometries: list[tuple[str, str, Any]] = []
    projected = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
    separation = manifest["separation_m"]
    if (
        not isinstance(separation, (int, float))
        or not math.isfinite(separation)
        or separation < 1000
    ):
        raise ValueError("At least 1000 m cross-split spatial separation required")
    if separation != selection["separation_m"]:
        raise ValueError("Separation changed after selection")
    for scene in scenes:
        original = selected.get(scene["name"])
        if original is None or any(
            scene[k] != original[k]
            for k in ("split", "prior_development_exposure", "region", "season")
        ):
            raise ValueError("Scene assignment/provenance differs from selection")
        identity = product_id(scene["product_id"])
        start, end = parse_utc(scene["acquisition_time"]), parse_utc(
            scene["acquisition_end"]
        )
        if start is None or end is None or start >= end:
            raise ValueError("Valid ordered UTC acquisition times required")
        expected_pol = ["hh", "hv"] if "_1SDH_" in scene["name"] else ["vv", "vh"]
        if scene["polarizations"] != expected_pol:
            raise ValueError("Native polarization cannot be renamed")
        if scene["ais"].get("truth_role") != "supporting_only":
            raise ValueError("AIS is supporting evidence, not truth")
        if (
            original.get("product_id")
            and product_id(original["product_id"]) != identity
        ):
            raise ValueError("Selected product was substituted")
        if scene["split"] not in SPLITS or (
            scene["split"] == "test"
            and scene["prior_development_exposure"] is not False
        ):
            raise ValueError("Unknown split or previously exposed test acquisition")
        tokens = scene["name"].split("_")
        acquisition = "_".join([tokens[0], tokens[6], tokens[7]])
        if scene["acquisition_group"] != acquisition:
            raise ValueError("Acquisition group not derived from native identity")
        for key in (identity, acquisition):
            if key in acquisitions and acquisitions[key] != scene["split"]:
                raise ValueError("One acquisition/datatake crosses splits")
            acquisitions[key] = scene["split"]
        if len(scene["rois"]) != len(original["rois"]):
            raise ValueError("Missing selected area")
        footprint = shape(scene["footprint"])
        for roi, spec in zip(scene["rois"], original["rois"]):
            if any(roi[k] != spec[k] for k in spec):
                raise ValueError("ROI selection changed")
            rid = roi["id"]
            if not rid or rid in roi_map:
                raise ValueError("Unique ROI IDs required")
            group = roi["group"]
            if not group or group in groups and groups[group] != scene["split"]:
                raise ValueError("Geographic group crosses splits")
            groups[group] = scene["split"]
            geom = shape(roi["geometry"])
            if (
                geom.geom_type != "Polygon"
                or geom.is_empty
                or not geom.is_valid
                or not region.covers(geom)
            ):
                raise ValueError("ROI must be a valid polygon inside NL")
            # Catalogue polygons are approximate; source GCP-derived corners
            # may differ by metres. A 100 m tolerance isn't valid-pixel proof.
            pf = transform(projected.transform, footprint)
            pg = transform(projected.transform, geom)
            if not pf.buffer(100).covers(pg):
                raise ValueError("ROI outside selected catalogue swath")
            if (
                roi["valid_pixels"] != roi["pixels"]
                or roi["pixels"] != roi["size"] ** 2
            ):
                raise ValueError(
                    "This pilot requires fully valid imagery, not nodata negatives"
                )
            size = roi["size"]
            expected = [
                {"id": f"{x//256}_{y//256}", "bbox_px": [x, y, x + 256, y + 256]}
                for y in range(0, size, 256)
                for x in range(0, size, 256)
            ]
            if size < 256 or size % 256 or roi["cells"] != expected:
                raise ValueError("Review grid must cover the complete selected area")
            geometries.append((rid, scene["split"], pg))
            roi_map[rid] = (scene, roi)
    for i, (rid, split, geom) in enumerate(geometries):
        for other, other_split, other_geom in geometries[i + 1 :]:
            if other_split != split and geom.distance(other_geom) < separation:
                raise ValueError(f"Spatial leakage between {rid} and {other}")
    ids: set[str] = set()
    by_roi: dict[str, list[dict[str, Any]]] = {rid: [] for rid in roi_map}
    for obj in annotations["objects"]:
        if not obj["id"] or obj["id"] in ids or obj["roi_id"] not in roi_map:
            raise ValueError("Invalid/duplicate annotation identity")
        ids.add(obj["id"])
        scene, roi = roi_map[obj["roi_id"]]
        if (
            product_id(obj["product_id"]) != scene["product_id"]
            or obj["class"] not in CLASSES
        ):
            raise ValueError("Annotation scene or class mismatch")
        if obj.get("kind") not in {"object", "area"} or (
            obj["class"] in {"vessel", "fixed_structure"} and obj["kind"] != "object"
        ):
            raise ValueError(
                "Context areas must not masquerade as individual vessels/structures"
            )
        bounds = obj["bbox_px"]
        if len(bounds) != 4 or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds
        ):
            raise ValueError("Finite native pixel bounds required")
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= roi["size"] and 0 <= y0 < y1 <= roi["size"]):
            raise ValueError("Annotation outside selected pixel area")
        geom = shape(obj["geometry"])
        if geom.geom_type != "Point" or not shape(roi["geometry"]).covers(geom):
            raise ValueError("Object centre outside selected area")
        if (
            not obj.get("decision")
            or not obj.get("annotator")
            or not obj.get("evidence")
        ):
            raise ValueError("Annotation decisions and authorship required")
        if obj.get("ais_role", "supporting_only") != "supporting_only":
            raise ValueError("AIS cannot be exhaustive truth")
        by_roi[obj["roi_id"]].append(obj)
    events = history["events"]
    event_ids: set[str] = set()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    last_time = None
    for event in events:
        event_time = parse_utc(event.get("timestamp", ""))
        if not event["id"] or event["id"] in event_ids or event_time is None:
            raise ValueError("Unique timestamped review events required")
        if last_time is not None and event_time < last_time:
            raise ValueError("Review history must stay chronological and append-only")
        last_time = event_time
        event_ids.add(event["id"])
        if event["kind"] not in {"first_pass", "independent_review"}:
            continue
        rid = event["roi_id"]
        if rid not in roi_map or not event.get("reviewer") or not event.get("notes"):
            raise ValueError("Review scope, reviewer and decision notes required")
        if event["kind"] == "independent_review" and not event.get("qualifications"):
            raise ValueError(
                "Independent reviewer qualifications/limitations must be disclosed"
            )
        _, roi = roi_map[rid]
        if event["source_chip_sha256"] != roi["chip"]["sha256"]:
            raise ValueError("Review references a different image")
        expected_cells = [c["id"] for c in roi["cells"]]
        if sorted(event["cells_reviewed"]) != sorted(expected_cells):
            raise ValueError("Incomplete/duplicated full-area review cells")
        latest[(rid, event["kind"])] = event
    statuses = []
    for rid, (scene, roi) in roi_map.items():
        objs = by_roi[rid]
        current = objects_digest(objs)
        first = latest.get((rid, "first_pass"))
        second = latest.get((rid, "independent_review"))
        first_ok = bool(
            first
            and first.get("annotation_digest") == current
            and first.get("decision") == "complete_first_pass"
        )
        independent_ok = bool(
            first_ok
            and first is not None
            and second
            and second.get("decision") == "accepted"
            and second.get("annotation_digest") == current
            and second.get("independent") is True
            and second.get("reviewer") != first["reviewer"]
            and second.get("reviewer") not in {o["annotator"] for o in objs}
        )
        unresolved = sum(o["class"] == "unresolved" for o in objs)
        statuses.append(
            {
                "roi_id": rid,
                "split": scene["split"],
                "first_pass_complete": first_ok,
                "independently_reviewed": independent_ok,
                "unresolved": unresolved,
                "objects": len(objs),
                "metric_ready": independent_ok and unresolved == 0,
                "test_locked": scene["split"] == "test",
            }
        )
    counts = {
        c: sum(o["class"] == c for o in annotations["objects"]) for c in sorted(CLASSES)
    }
    return {
        "dataset_version": version,
        "acquisitions": len(scenes),
        "areas": len(roi_map),
        "pixels": sum(r["pixels"] for _, r in roi_map.values()),
        "class_counts": counts,
        "review_status": statuses,
        "independent_review_available": any(
            s["independently_reviewed"] for s in statuses
        ),
        "accuracy_measured": False,
        "limits": [
            "Purposive pilot, not province-wide or statistically "
            "representative validation.",
            "First-pass machine-assisted labels are not independently "
            "reviewed ground truth.",
            "AIS silence is neither evidence of non-vessel nor "
            "deliberate dark activity.",
        ],
    }


def load_dataset(
    directory: Path = DEFAULT_DATASET, *, verify_files: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify versioned small artifacts, optionally rehash all local SAR bytes."""

    def read(name: str) -> dict[str, Any]:
        return json.loads((directory / name).read_text(encoding="utf-8"))  # type: ignore[no-any-return]

    release = read("release.json")
    for name, checksum in release["files"].items():
        if sha256_file(bounded_path(directory, name)) != checksum:
            raise ValueError(f"Versioned artifact hash mismatch: {name}")
    for required in (
        "selection.json",
        "manifest.json",
        "split_lock.json",
        "annotations.json",
        "review_history.json",
        "ANNOTATION_GUIDE.md",
        "DATASET_CARD.md",
    ):
        if required not in release["files"]:
            raise ValueError(f"Release omits required artifact: {required}")
    manifest, selection, lock = (
        read("manifest.json"),
        read("selection.json"),
        read("split_lock.json"),
    )
    annotations, history = read("annotations.json"), read("review_history.json")
    if manifest["selection_sha256"] != sha256_file(directory / "selection.json"):
        raise ValueError("Selection hash mismatch")
    status = validate_contract(manifest, selection, lock, annotations, history)
    if release["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("Release version mismatch")
    if verify_files:
        entries = [
            {
                "path": manifest["licence"]["snapshot"],
                "sha256": manifest["licence"]["sha256"],
            }
        ]
        for scene in manifest["scenes"]:
            entries.extend(
                scene["sources"] + [scene["catalogue"], scene["ais"]["snapshot"]]
            )
            for roi in scene["rois"]:
                entries.extend(
                    roi["calibration"]
                    + roi["previews"]
                    + [roi["chip"], roi["review_grid"]]
                )
        for item in entries:
            if sha256_file(bounded_path(REPO_ROOT, item["path"])) != item["sha256"]:
                raise ValueError(
                    f"Source/derived artifact hash mismatch: {item['path']}"
                )
        for scene in manifest["scenes"]:
            for roi in scene["rois"]:
                verify_chip(
                    bounded_path(REPO_ROOT, roi["chip"]["path"]),
                    roi,
                    [o for o in annotations["objects"] if o["roi_id"] == roi["id"]],
                )
        bundle = release["review_bundle"]
        if sha256_file(bounded_path(REPO_ROOT, bundle["path"])) != bundle["sha256"]:
            raise ValueError("Review bundle hash mismatch")
    return manifest, annotations, status


def validation_labels(directory: Path, roi_id: str) -> dict[str, Any]:
    """Bridge approved validation labels to coastal trials; no test bypass."""
    manifest, annotation, status = load_dataset(directory)
    selected = next((r for r in status["review_status"] if r["roi_id"] == roi_id), None)
    if selected is None or selected["split"] != "validation":
        raise ValueError(
            "Only the validation split may be exported for buffer trials; "
            "test stays locked"
        )
    if not selected["metric_ready"]:
        raise ValueError(
            "Independent complete-area review and resolved annotations required; "
            "accuracy remains unmeasured"
        )
    load_dataset(directory, verify_files=True)
    scene = next(
        s for s in manifest["scenes"] if any(r["id"] == roi_id for r in s["rois"])
    )
    roi = next(r for r in scene["rois"] if r["id"] == roi_id)
    history = json.loads(
        (directory / "review_history.json").read_text(encoding="utf-8")
    )
    reviewer = [
        e["reviewer"]
        for e in history["events"]
        if e.get("roi_id") == roi_id and e["kind"] == "independent_review"
    ][-1]
    return {
        "type": "FeatureCollection",
        "benchmark": {
            "split": "validation",
            "product_id": scene["product_id"],
            "acquisition_time": scene["acquisition_time"],
            "annotation_complete": True,
            "annotation_scope": "exhaustive_vessels_on_valid_imagery",
            "source_kind": "independent_sar_review",
            "source": f"NL benchmark {manifest['dataset_version']} / {roi_id}",
            "reviewer": reviewer,
            "licence": manifest["licence"]["url"],
            "valid_imagery_roi": roi["geometry"],
        },
        "features": [
            {
                "type": "Feature",
                "geometry": obj["geometry"],
                "properties": {
                    "label_id": obj["id"],
                    "class": "vessel" if obj["class"] == "vessel" else "non_vessel",
                    "original_class": obj["class"],
                    "annotation_decision": obj["decision"],
                },
            }
            for obj in annotation["objects"]
            if obj["roi_id"] == roi_id and obj["kind"] == "object"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--export-validation", metavar="ROI_ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.export_validation:
        if args.output is None:
            parser.error("--export-validation requires --output")
        write_json(args.output, validation_labels(args.dataset, args.export_validation))
    else:
        _, _, status = load_dataset(args.dataset, verify_files=args.verify_files)
        if args.output:
            write_json(args.output, status)
        print(json.dumps(status, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
