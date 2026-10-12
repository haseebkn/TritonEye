"""Image-first, provisional curation sidecars; never a training-data exporter.

The first-pass packet has only neutral IDs and whole-area images/masks. Source
points are revealed only after a complete, hash-bound first pass. These local
workflow guards are not access control against an owner reading source files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import Point, box, shape

from agents.allenai_subset import RESOLUTION, WORLD_M, load_alignment_subset
from agents.artifacts import REPO_ROOT, sha256_file
from agents.nl_benchmark import require_development_scene

CLASSES = {
    "ignore": 0,
    "vessel": 1,
    "background": 2,
    "fixed_structure": 3,
    "clutter": 4,
    "ice": 5,
}
POINT_STATES = {
    "usable_provisional",
    "unresolved_alignment",
    "unresolved_class",
    "unusable",
}
ROOT = REPO_ROOT / "data/model_research/allenai_curation"


def point_pixel(point: dict[str, Any], case: dict[str, Any]) -> list[float]:
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(
        point["lon"], point["lat"]
    )
    col, row, _, _ = case["global_pixel_window"]
    return [(x + WORLD_M / 2) / RESOLUTION - col, (WORLD_M / 2 - y) / RESOLUTION - row]


def save(path: Path, value: Any) -> None:
    """New sidecar files use deterministic LF bytes, not frozen-release writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fresh(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT.resolve() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Fresh output must be beneath the separate curation root")
    if path.exists():
        raise ValueError("Existing outputs cannot be overwritten")
    for parent in path.parents:
        if (parent / "workspace.json").exists() or (parent / "seal.json").exists():
            raise ValueError("Outputs cannot be nested inside a review packet or seal")
        if parent == ROOT.resolve():
            break
    return path


def member(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its bundle")
    return path


def record(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def checked(root: Path, item: dict[str, str]) -> Path:
    path = member(root, item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Curation artifact integrity mismatch")
    return path


def scope(cases: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    lookup = {source["name"]: source for source in sources}
    for case in cases:
        require_development_scene(
            lookup[case["source_name"]]["product_id"],
            {"name": case["source_name"], "study_roi": case["physical_geometry"]},
        )


def prepare(release: Path, output: Path) -> dict[str, Any]:
    """Copy only verified whole-window previews into a neutral first-pass packet."""
    output = fresh(output)
    source = load_alignment_subset(release)
    scope(source["cases"], source["sources"])  # Guard before any image read/copy.
    cases = list(source["cases"])
    random.Random(sha256_file(release)).shuffle(cases)
    packet = output / "first_pass"
    index: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    sections = []
    for number, case in enumerate(cases, 1):
        identifier = f"area_{number:03d}"
        previews = {p["band"]: p for p in case["publisher_previews"]}
        if set(previews) != {"vv", "vh"}:
            raise ValueError("Only correctly identified VV/VH inputs are supported")
        valid = None
        images = []
        dims = None
        for band in ("vv", "vh"):
            preview = previews[band]
            image_path = REPO_ROOT / preview["path"]
            mask_path = REPO_ROOT / preview["valid_mask"]["path"]
            if (
                sha256_file(image_path) != preview["sha256"]
                or sha256_file(mask_path) != preview["valid_mask"]["sha256"]
            ):
                raise ValueError("Preview or validity hash mismatch")
            with Image.open(image_path) as image:
                array = np.asarray(image)
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)
            if (
                array.ndim != 2
                or mask.shape != array.shape
                or not np.isin(mask, [0, 255]).all()
            ):
                raise ValueError("Expected grayscale imagery and binary validity")
            if dims is not None and dims != array.shape:
                raise ValueError("Band dimensions disagree")
            dims = array.shape
            valid = mask > 0 if valid is None else valid & (mask > 0)
            destination = packet / identifier / f"{band}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image_path, destination)
            images.append({"band": band, **record(packet, destination)})
        assert valid is not None and dims is not None
        mask_destination = packet / identifier / "valid.png"
        Image.fromarray(valid.astype("uint8") * 255).save(mask_destination)
        ignore_destination = packet / identifier / "unreviewed_ignore.png"
        Image.fromarray(np.full(valid.shape, 255, dtype="uint8")).save(
            ignore_destination
        )
        index.append(
            {
                "id": identifier,
                "width": dims[1],
                "height": dims[0],
                "images": images,
                "valid_mask": record(packet, mask_destination),
                "initial_ignore_mask": record(packet, ignore_destination),
            }
        )
        mapping.append({"id": identifier, "source_case": case})
        decisions.append(
            {"id": identifier, "full_valid_area_reviewed": False, "regions": []}
        )
        sections.append(
            f"<h2>{identifier}</h2><p>VV, VH; pixel coordinates x right, y down, "
            "origin top left.</p>"
            f'<img width="512" src="{identifier}/vv.png" alt="VV">'
            f'<img width="512" src="{identifier}/vh.png" alt="VH">'
            "<p>Validity: white valid, black ignored.</p>"
            f'<img width="512" src="{identifier}/valid.png" alt="Valid pixels">'
        )
    save(packet / "index.json", {"version": "curation-0.1.0", "cases": index})
    save(
        packet / "decisions.json",
        {
            "actor": "",
            "method": "self",
            "prior_label_exposure": None,
            "independent_review": False,
            "cases": decisions,
        },
    )
    (packet / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Provisional image-first curation</title><h1>Whole-area first pass</h1>"
        "<p>Publisher labels, detector predictions and AIS are withheld. "
        "Inspect every valid pixel. This is provisional self/AI curation, "
        "not independent truth. This raw-DN display is not calibrated sigma0.</p>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    save(output / "source_map.json", {"sources": source["sources"], "cases": mapping})
    descriptor = {
        "version": "curation-0.1.0",
        "status": "provisional_quarantined",
        "created_at": now(),
        "source_release_sha256": sha256_file(release),
        "code_sha256": sha256_file(__file__),
        "independent_review": False,
        "training_allowed": False,
        "evaluation_allowed": False,
        "threshold_selection_allowed": False,
        "source_map": record(output, output / "source_map.json"),
        "packet_index": record(output, packet / "index.json"),
        "packet_html": record(output, packet / "index.html"),
    }
    save(output / "workspace.json", descriptor)
    return {
        "status": "prepared_unreviewed",
        "areas": len(index),
        "published_points_revealed": 0,
        "locked_test_pixels_read": False,
        "workspace_sha256": sha256_file(output / "workspace.json"),
    }


def load_workspace(
    workspace: Path, *, purpose: str = "curation_qa"
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if purpose != "curation_qa":
        raise ValueError(
            "Provisional quarantine: training/evaluation/threshold export blocked"
        )
    descriptor = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    if descriptor["status"] != "provisional_quarantined" or any(
        descriptor[key] is not False
        for key in (
            "independent_review",
            "training_allowed",
            "evaluation_allowed",
            "threshold_selection_allowed",
        )
    ):
        raise ValueError("Curation cannot enable adoption or independent review")
    source = json.loads(checked(workspace, descriptor["source_map"]).read_text())
    scope([item["source_case"] for item in source["cases"]], source["sources"])
    index = json.loads(checked(workspace, descriptor["packet_index"]).read_text())
    checked(workspace, descriptor["packet_html"])
    packet = workspace / "first_pass"
    for case in index["cases"]:
        for item in [*case["images"], case["valid_mask"], case["initial_ignore_mask"]]:
            checked(packet, item)
    return descriptor, source, index


def masks(valid: Any, regions: list[dict[str, Any]]) -> Any:
    """Rasterize explicit pixel-edge polygons; ignore/nodata always win.

    Background is an explicit reviewed region, not the absence of vessel points.
    Object regions override background; incompatible object overlaps fail closed.
    Raster cells are selected by their centres, not all_touched area inflation.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("Validity must have two dimensions")
    height, width = valid.shape
    result = np.zeros(valid.shape, dtype="uint8")
    painted: dict[str, Any] = {
        label: np.zeros(valid.shape, dtype=bool) for label in CLASSES
    }
    for region in regions:
        label = region.get("class")
        reason = region.get("reason")
        if label not in CLASSES or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Class and nonempty annotation reason required")
        geom = shape(region["geometry"])
        if (
            geom.geom_type not in {"Polygon", "MultiPolygon"}
            or geom.is_empty
            or not geom.is_valid
            or not all(math.isfinite(x) for x in geom.bounds)
            or not box(0, 0, width, height).covers(geom)
        ):
            raise ValueError(
                "Regions must be valid finite polygons inside pixel bounds"
            )
        painted[label] |= rasterize(
            [(geom, 1)],
            out_shape=valid.shape,
            transform=Affine.identity(),
            fill=0,
            dtype="uint8",
        ).astype(bool)
    result[painted["background"]] = CLASSES["background"]
    occupied = np.zeros(valid.shape, dtype=bool)
    for label in ("vessel", "fixed_structure", "clutter", "ice"):
        selected = painted[label] & valid & ~painted["ignore"]
        if (selected & occupied).any():
            raise ValueError("Conflicting object classes; resolve with explicit ignore")
        result[selected] = CLASSES[label]
        occupied |= selected
    result[~valid | painted["ignore"]] = 0
    return result


def freeze_first_pass(workspace: Path, decisions: Path, output: Path) -> dict[str, Any]:
    """Freeze all-area curation and coverage before source points can be revealed."""
    output = fresh(output)
    _, _, index = load_workspace(workspace)
    data = json.loads(decisions.read_text(encoding="utf-8"))
    if (
        not isinstance(data.get("actor"), str)
        or not data["actor"].strip()
        or data.get("method") not in {"self", "ai_assisted"}
        or type(data.get("prior_label_exposure")) is not bool
        or data.get("independent_review") is not False
    ):
        raise ValueError(
            "Explicit provisional actor, method and prior exposure required"
        )
    expected = {case["id"] for case in index["cases"]}
    actual = [case["id"] for case in data["cases"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Exactly one first-pass decision per whole area required")
    inputs = {case["id"]: case for case in data["cases"]}
    artifacts = []
    totals = {label: 0 for label in CLASSES}
    for case in index["cases"]:
        decision = inputs[case["id"]]
        if decision.get("full_valid_area_reviewed") is not True:
            raise ValueError(
                "All valid areas must be explicitly reviewed before reveal"
            )
        with Image.open(checked(workspace / "first_pass", case["valid_mask"])) as image:
            valid = np.asarray(image) > 0
        result = masks(valid, decision["regions"])
        # Explicit ignore polygons count as reviewed but contribute no negatives.
        coverage = (
            masks(valid, [{**r, "class": "background"} for r in decision["regions"]])
            > 0
        )
        if not coverage[valid].all():
            raise ValueError("Unreviewed valid pixels remain; full-area claim refused")
        path = output / f"{case['id']}_classes.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(path)
        ignore = output / f"{case['id']}_ignore.png"
        Image.fromarray((result == 0).astype("uint8") * 255).save(ignore)
        artifacts.extend([record(output, path), record(output, ignore)])
        for label, code in CLASSES.items():
            totals[label] += int((result == code).sum())
    save(output / "decisions.json", data)
    seal = {
        "status": "provisional_first_pass_frozen",
        "workspace_sha256": sha256_file(workspace / "workspace.json"),
        "frozen_at": now(),
        "decisions": record(output, output / "decisions.json"),
        "artifacts": artifacts,
        "class_codes": CLASSES,
        "pixel_counts": totals,
        "independent_review": False,
        "training_allowed": False,
        "evaluation_allowed": False,
    }
    save(output / "seal.json", seal)
    return seal


def reveal(workspace: Path, frozen: Path, output: Path) -> dict[str, Any]:
    output = fresh(output)
    _, source, _ = load_workspace(workspace)
    seal = json.loads((frozen / "seal.json").read_text())
    if seal["status"] != "provisional_first_pass_frozen" or seal[
        "workspace_sha256"
    ] != sha256_file(workspace / "workspace.json"):
        raise ValueError("A matching frozen first pass is required")
    checked(frozen, seal["decisions"])
    for item in seal["artifacts"]:
        checked(frozen, item)
    cases = []
    for entry in source["cases"]:
        case = entry["source_case"]
        origin_x, origin_y, _, _ = case["global_pixel_window"]
        # Publisher grid positions, not native/TPS positions and not crop centres.
        points = [
            {
                "source_label_id": point["source_label_id"],
                "published_lon": point["lon"],
                "published_lat": point["lat"],
                "publisher_pixel": point_pixel(point, case),
                "state": "unresolved_class",
                "reason": "",
                "first_pass_region_index": None,
            }
            for point in case["points"]
        ]
        cases.append(
            {
                "id": entry["id"],
                "source_window_id": case["source_window_id"],
                "source_name": case["source_name"],
                "publisher_global_pixel_origin": [origin_x, origin_y],
                "points": points,
                "ignore_regions": [],
            }
        )
    save(
        output / "point_decisions.json",
        {
            "status": "provisional_second_pass",
            "first_pass_seal_sha256": sha256_file(frozen / "seal.json"),
            "cases": cases,
        },
    )
    return {
        "status": "revealed_for_alignment_only",
        "points": sum(len(case["points"]) for case in cases),
        "training_allowed": False,
    }


def seal_second_pass(
    workspace: Path, frozen: Path, decisions: Path, output: Path
) -> dict[str, Any]:
    """Record point decisions without changing first-pass masks or source labels."""
    output = fresh(output)
    _, source, index = load_workspace(workspace)
    seal = json.loads((frozen / "seal.json").read_text())
    if seal["status"] != "provisional_first_pass_frozen" or seal[
        "workspace_sha256"
    ] != sha256_file(workspace / "workspace.json"):
        raise ValueError("First-pass workspace mismatch")
    first = json.loads(checked(frozen, seal["decisions"]).read_text())
    for item in seal["artifacts"]:
        checked(frozen, item)
    data = json.loads(decisions.read_text())
    if data["first_pass_seal_sha256"] != sha256_file(frozen / "seal.json"):
        raise ValueError("First-pass seal mismatch")
    expected = {
        (entry["id"], point["source_label_id"])
        for entry in source["cases"]
        for point in entry["source_case"]["points"]
    }
    seen = []
    regions = {case["id"]: case["regions"] for case in first["cases"]}
    counts = dict.fromkeys(sorted(POINT_STATES), 0)
    source_cases = {entry["id"]: entry["source_case"] for entry in source["cases"]}
    packet_cases = {case["id"]: case for case in index["cases"]}
    case_ids = [case["id"] for case in data["cases"]]
    if set(case_ids) != set(source_cases) or len(case_ids) != len(source_cases):
        raise ValueError("Exactly one second-pass case per whole area required")
    artifacts = []
    for case in data["cases"]:
        original = source_cases[case["id"]]
        if (
            case["source_name"] != original["source_name"]
            or case["source_window_id"] != original["source_window_id"]
        ):
            raise ValueError("Second-pass source identity changed")
        originals = {point["source_label_id"]: point for point in original["points"]}
        if any(region["class"] != "ignore" for region in case["ignore_regions"]):
            raise ValueError("Second pass can add only ignore regions, not new truth")
        input_case = packet_cases[case["id"]]
        with Image.open(
            checked(workspace / "first_pass", input_case["valid_mask"])
        ) as image:
            valid = np.asarray(image) > 0
        result = masks(valid, regions[case["id"]] + case["ignore_regions"])
        for point in case["points"]:
            seen.append((case["id"], point["source_label_id"]))
            state = point["state"]
            original_point = originals.get(point["source_label_id"])
            if (
                original_point is None
                or point["published_lon"] != original_point["lon"]
                or point["published_lat"] != original_point["lat"]
                or point["publisher_pixel"] != point_pixel(original_point, original)
            ):
                raise ValueError("Published point coordinates changed")
            x, y = point["publisher_pixel"]
            if not 0 <= x < valid.shape[1] or not 0 <= y < valid.shape[0]:
                raise ValueError("Published point outside source window")
            if (
                state not in POINT_STATES
                or not isinstance(point.get("reason"), str)
                or not point["reason"].strip()
            ):
                raise ValueError(
                    "Each point needs a supported state and decision reason"
                )
            if state == "usable_provisional":
                idx = point["first_pass_region_index"]
                if (
                    type(idx) is not int
                    or not 0 <= idx < len(regions[case["id"]])
                    or regions[case["id"]][idx]["class"] != "vessel"
                ):
                    raise ValueError(
                        "Usable point must reference a first-pass vessel region"
                    )
                if (
                    not isinstance(point.get("alignment_evidence"), str)
                    or not point["alignment_evidence"].strip()
                ):
                    raise ValueError(
                        "Usable point requires explicit alignment evidence"
                    )
                if (
                    not shape(regions[case["id"]][idx]["geometry"]).covers(Point(x, y))
                    or result[int(y), int(x)] != CLASSES["vessel"]
                ):
                    raise ValueError(
                        "Usable point must align to valid first-pass vessel pixels"
                    )
            elif result[int(y), int(x)] != 0:
                raise ValueError(
                    "Unresolved/unusable point context must be explicitly ignored"
                )
            counts[state] += 1
        path = output / f"{case['id']}_classes.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(path)
        ignore = output / f"{case['id']}_ignore.png"
        Image.fromarray((result == 0).astype("uint8") * 255).save(ignore)
        artifacts.extend([record(output, path), record(output, ignore)])
    if set(seen) != expected or len(seen) != len(expected):
        raise ValueError("Exactly one decision per published point required")
    save(output / "point_decisions.json", data)
    result = {
        "status": "provisional_quarantined",
        "first_pass_seal_sha256": sha256_file(frozen / "seal.json"),
        "point_decisions": record(output, output / "point_decisions.json"),
        "artifacts": artifacts,
        "counts": counts,
        "independent_review": False,
        "training_allowed": False,
        "evaluation_allowed": False,
        "licence_clarification": "unresolved",
    }
    save(output / "seal.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "freeze", "reveal", "seal-points", "check"]
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=REPO_ROOT / "datasets/allenai_nl/v0.1.0/release.json",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--first-pass", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action != "prepare" and args.workspace is None:
        parser.error("--workspace required")
    if args.action != "check" and args.output is None:
        parser.error("--output required")
    if args.action in {"freeze", "seal-points"} and args.decisions is None:
        parser.error("--decisions required")
    if args.action in {"reveal", "seal-points"} and args.first_pass is None:
        parser.error("--first-pass required")
    if args.action == "prepare":
        result = prepare(args.release, args.output)
    elif args.action == "freeze":
        result = freeze_first_pass(args.workspace, args.decisions, args.output)
    elif args.action == "reveal":
        result = reveal(args.workspace, args.first_pass, args.output)
    elif args.action == "seal-points":
        result = seal_second_pass(
            args.workspace, args.first_pass, args.decisions, args.output
        )
    else:
        descriptor, _, index = load_workspace(args.workspace)
        result = {
            "status": descriptor["status"],
            "areas": len(index["cases"]),
            "training_allowed": False,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
