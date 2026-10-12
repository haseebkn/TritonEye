"""Quarantined AllenAI alignment and input QA; never an adoption exporter."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window
from shapely.geometry import Point, box, mapping, shape

from agents import allenai_curation as cur
from agents.allenai_first_pass import audit, read_image
from agents.artifacts import REPO_ROOT, sha256_file
from agents.calibration import Calibrator
from agents.nl_benchmark import require_development_scene
from agents.xview3_detector import TILE_SIZE, normalize_db


def input_diagnostic(db_vv_vh: Any) -> dict[str, Any]:
    """Inspect calibrated arrays without running a model or approving labels."""
    values = np.asarray(db_vv_vh)
    if values.ndim != 3 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError("Finite two-band sigma0 dB arrays required")
    normalized = np.stack([normalize_db(values[1]), normalize_db(values[0])])
    if (
        not np.isfinite(normalized).all()
        or not ((normalized >= 0) & (normalized <= 1)).all()
    ):
        raise ValueError("Invalid xView3 normalization")
    return {
        "shape": list(values.shape),
        "sigma0_db_percentiles_vv_vh": np.percentile(
            values, [1, 50, 99], axis=(1, 2)
        ).T.tolist(),
        "normalized_percentiles_vh_vv": np.percentile(
            normalized, [1, 50, 99], axis=(1, 2)
        ).T.tolist(),
        "channel_order": ["vh", "vv"],
        "radiometry_arithmetic_checked": True,
        "native_tile_shape_compatible": values.shape[1:] == (TILE_SIZE, TILE_SIZE),
        "training_allowed": False,
    }


def first_decisions(frozen: Path) -> dict[str, Any]:
    seal = json.loads((frozen / "seal.json").read_text())
    data = json.loads(cur.checked(frozen, seal["decisions"]).read_text())
    return {case["id"]: case for case in data["cases"]}


def alignment_packet(workspace: Path, frozen: Path, output: Path) -> dict[str, Any]:
    """Reveal original points only after replaying the sealed whole-area pass."""
    first = audit(workspace, frozen)
    _, source, index = cur.load_workspace(workspace)
    cur.reveal(workspace, frozen, output)
    authored = first_decisions(frozen)
    packet = {case["id"]: case for case in index["cases"]}
    sources = {item["name"]: item for item in source["sources"]}
    rows = []
    panels = []
    for entry in source["cases"]:
        case = entry["source_case"]
        identifier = entry["id"]
        classes = read_image(frozen / f"{identifier}_classes.png")
        images = [
            read_image(cur.checked(workspace / "first_pass", item))
            for item in packet[identifier]["images"]
        ]
        alignments = {p["source_label_id"]: p for p in case["alignment_checks"]}
        for point in case["points"]:
            x, y = cur.point_pixel(point, case)
            if not 0 <= x < classes.shape[1] or not 0 <= y < classes.shape[0]:
                raise ValueError("Publisher point outside whole-area image")
            matching = [
                i
                for i, region in enumerate(authored[identifier]["regions"])
                if region["class"] == "vessel"
                and shape(region["geometry"]).covers(Point(x, y))
            ]
            native = alignments[point["source_label_id"]]
            require_development_scene(
                sources[case["source_name"]]["product_id"],
                {"name": case["source_name"], "study_roi": native["patch"]["geometry"]},
            )
            panel = Image.new("RGB", (1024, 290), "#181818")
            draw = ImageDraw.Draw(panel)
            draw.text(
                (8, 2),
                f"{identifier} / {point['source_label_id']} / "
                f"first class {int(classes[int(y), int(x)])}",
                fill="white",
            )
            for band, array in enumerate(images):
                left = min(max(int(x) - 128, 0), array.shape[1] - 256)
                top = min(max(int(y) - 128, 0), array.shape[0] - 256)
                crop = (
                    Image.fromarray(array)
                    .crop((left, top, left + 256, top + 256))
                    .convert("RGB")
                )
                mark(crop, x - left, y - top)
                panel.paste(crop, (band * 256, 30))
                item = native["patch"]["previews"][band]
                path = checked_project(item)
                with Image.open(path) as image:
                    calibrated = image.convert("RGB")
                mark(calibrated, *native["point_px"])
                panel.paste(calibrated, ((band + 2) * 256, 30))
            draw.text(
                (8, 17),
                "Publisher DN VV / VH | Native calibrated VV / VH; "
                "red = unchanged source point",
                fill="white",
            )
            panels.append(panel)
            rows.append(
                {
                    "area_id": identifier,
                    "source_label_id": point["source_label_id"],
                    "first_pass_class_at_point": int(classes[int(y), int(x)]),
                    "first_pass_vessel_region_indices": matching,
                    "publisher_pixel": [x, y],
                    "native_point_px": native["point_px"],
                    "transform_displacement_m": native[
                        "grid_vs_native_tps_displacement_m"
                    ],
                }
            )
    sheets = []
    for start in range(0, len(panels), 6):
        group = panels[start : start + 6]
        sheet = Image.new("RGB", (1024, 290 * len(group)), "black")
        for row, panel in enumerate(group):
            sheet.paste(panel, (0, row * 290))
        path = output / f"alignment_{start // 6 + 1:02d}.png"
        sheet.save(path)
        sheets.append(cur.record(output, path))
    result = {
        "status": "provisional_alignment_packet",
        "first_pass_seal_sha256": first["first_pass_seal_sha256"],
        "points": len(rows),
        "rows": rows,
        "sheets": sheets,
        "training_allowed": False,
        "independent_review": False,
    }
    cur.save(output / "alignment_index.json", result)
    return {key: value for key, value in result.items() if key != "rows"}


def mark(image: Image.Image, x: float, y: float) -> None:
    draw = ImageDraw.Draw(image)
    for offset in (-9, 9):
        draw.line((x + offset, y - 3, x + offset, y + 3), fill="red", width=1)
        draw.line((x - 3, y + offset, x + 3, y + offset), fill="red", width=1)


def checked_project(item: dict[str, Any]) -> Path:
    path = (REPO_ROOT / item["path"]).resolve()
    if not path.is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("QA input must be a project data artifact")
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Source/QA input integrity mismatch")
    return Path(path)


def compatibility(workspace: Path, frozen: Path) -> dict[str, Any]:
    """Replay point-patch calibration and quantify lossy display saturation."""
    first = audit(workspace, frozen)
    _, source, index = cur.load_workspace(workspace)
    sources = {item["name"]: item for item in source["sources"]}
    for entry in source["cases"]:
        case = entry["source_case"]
        for point in case["alignment_checks"]:
            require_development_scene(
                sources[case["source_name"]]["product_id"],
                {"name": case["source_name"], "study_roi": point["patch"]["geometry"]},
            )
    for item in source["sources"]:
        for reference in item["files"]:
            checked_project(reference)
    previews = []
    for case in index["cases"]:
        valid = (
            read_image(cur.checked(workspace / "first_pass", case["valid_mask"])) > 0
        )
        for item in case["images"]:
            image = read_image(cur.checked(workspace / "first_pass", item))
            if image.dtype != np.uint8 or image.shape != valid.shape:
                raise ValueError("Expected clipped uint8 publisher preview")
            previews.append(
                {
                    "area_id": case["id"],
                    "band": Path(item["path"]).stem,
                    "valid_pixels": int(valid.sum()),
                    "pixels_at_255": int((image[valid] == 255).sum()),
                    "direct_xview3_input_compatible": False,
                }
            )
    patches = []
    for entry in source["cases"]:
        case = entry["source_case"]
        src = sources[case["source_name"]]
        for point in case["alignment_checks"]:
            patch = point["patch"]
            require_development_scene(
                src["product_id"], {"name": src["name"], "study_roi": patch["geometry"]}
            )
            with rasterio.open(checked_project(patch["chip"])) as image:
                db = image.read()
                if (
                    image.descriptions != ("sigma0_db_vv", "sigma0_db_vh")
                    or not (image.read_masks() > 0).all()
                ):
                    raise ValueError("Native patch metadata/validity mismatch")
            col, row, width, height = patch["window"]
            maximum_error = 0.0
            for band, pol in enumerate(("vv", "vh")):
                measurement = next(
                    f
                    for f in src["files"]
                    if "/measurement/" in f["path"] and f"-{pol}-" in f["path"]
                )
                calibration = patch["calibration"][band]
                if f"-{pol}-" not in calibration["path"]:
                    raise ValueError("Calibration polarization mismatch")
                with rasterio.open(checked_project(measurement)) as raster:
                    dn = raster.read(1, window=Window(col, row, width, height))
                replay = (
                    Calibrator(str(checked_project(calibration)))
                    .to_sigma0_db(dn, row, col)
                    .astype("float32")
                )
                error = float(np.max(np.abs(db[band] - replay)))
                if error > 1e-5:
                    raise ValueError("Native sigma0 calibration replay mismatch")
                maximum_error = max(maximum_error, error)
            patches.append(
                {
                    "source_label_id": point["source_label_id"],
                    "calibration_max_abs_error_db": maximum_error,
                    **input_diagnostic(db),
                }
            )
    return {
        "status": "quarantined_input_diagnostic",
        "source_release_sha256": first["source_release_sha256"],
        "first_pass_seal_sha256": first["first_pass_seal_sha256"],
        "source_files_hash_checked": sum(len(s["files"]) for s in source["sources"]),
        "publisher_previews": previews,
        "native_calibration_patches": patches,
        "publisher_pngs_directly_compatible": False,
        "native_patch_radiometry_replayed": len(patches),
        "native_context_and_label_alignment_ready": False,
        "scope": (
            "Arithmetic/metadata QA only; 256px patches are not native 2048px "
            "model contexts. PNG clipping is irreversible; calibrate original "
            "DN at native LUT coordinates. Publisher geometry cannot be "
            "substituted for native TPS alignment."
        ),
        "training_allowed": False,
        "evaluation_allowed": False,
        "independent_review": False,
        "licence_clarification": "unresolved",
        "completeness": "unverified; no unmarked pixel is accepted as negative truth",
        "verifier_code_sha256": sha256_file(__file__),
    }


def seal_review(
    workspace: Path, frozen: Path, packet: Path, observations: Path, output: Path
) -> dict[str, Any]:
    """Seal authored second-pass decisions and expand uncertain ignore context."""
    audit(workspace, frozen)
    _, _, index = cur.load_workspace(workspace)
    packet_index = json.loads((packet / "alignment_index.json").read_text())
    if packet_index["first_pass_seal_sha256"] != sha256_file(frozen / "seal.json"):
        raise ValueError("Alignment packet first-pass mismatch")
    for item in packet_index["sheets"]:
        cur.checked(packet, item)
    authored = json.loads(observations.read_text())
    if (
        not isinstance(authored.get("actor"), str)
        or not authored["actor"].strip()
        or authored.get("method") not in {"self", "ai_assisted"}
        or authored.get("prior_label_exposure") is not True
        or authored.get("independent_review") is not False
        or authored.get("complete_point_display_inspection") is not True
    ):
        raise ValueError("Explicit provisional actor and label exposure required")
    decisions = json.loads((packet / "point_decisions.json").read_text())
    expected = {p["source_label_id"] for c in decisions["cases"] for p in c["points"]}
    actual = [p["source_label_id"] for p in authored["points"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Exactly one authored observation per publisher point required"
        )
    lookup = {p["source_label_id"]: p for p in authored["points"]}
    dims = {c["id"]: (c["width"], c["height"]) for c in index["cases"]}
    decisions.update({k: v for k, v in authored.items() if k != "points"})
    decisions["observations_sha256"] = sha256_file(observations)
    decisions["alignment_index_sha256"] = sha256_file(packet / "alignment_index.json")
    for case in decisions["cases"]:
        for point in case["points"]:
            observation = lookup[point["source_label_id"]]
            point.update(
                state=observation["state"],
                reason=observation["reason"],
                first_pass_region_index=observation.get("first_pass_region_index"),
                alignment_evidence=observation.get("alignment_evidence"),
            )
            if point["state"] != "usable_provisional":
                x, y = point["publisher_pixel"]
                width, height = dims[case["id"]]
                case["ignore_regions"].append(
                    {
                        "class": "ignore",
                        "reason": point["reason"],
                        "geometry": mapping(
                            box(
                                max(0, x - 64),
                                max(0, y - 64),
                                min(width, x + 64),
                                min(height, y + 64),
                            )
                        ),
                    }
                )
    staged = cur.fresh(output)
    # Stage only decisions, outside sealed bundles; keep the original draft intact.
    input_path = staged.with_name(staged.name + "_decisions.json")
    if input_path.exists():
        raise ValueError("Existing second-pass input cannot be overwritten")
    cur.save(input_path, decisions)
    seal = cur.seal_second_pass(workspace, frozen, input_path, staged)
    return seal


def completeness(workspace: Path, frozen: Path, second: Path) -> dict[str, Any]:
    """Replay every second-pass mask and report unresolved supervision limits."""
    first = audit(workspace, frozen)
    _, source, index = cur.load_workspace(workspace)
    seal = json.loads((second / "seal.json").read_text())
    if (
        seal["first_pass_seal_sha256"] != first["first_pass_seal_sha256"]
        or seal["status"] != "provisional_quarantined"
        or any(
            seal[k] is not False
            for k in ("training_allowed", "evaluation_allowed", "independent_review")
        )
    ):
        raise ValueError("Matching quarantined second-pass seal required")
    data = json.loads(cur.checked(second, seal["point_decisions"]).read_text())
    if (
        data.get("independent_review") is not False
        or data.get("prior_label_exposure") is not True
    ):
        raise ValueError("Second-pass review/exposure declaration changed")
    artifacts = {item["path"]: item for item in seal["artifacts"]}
    expected = {
        f"{c['id']}_{kind}.png"
        for c in index["cases"]
        for kind in ("classes", "ignore")
    }
    if set(artifacts) != expected or len(artifacts) != len(seal["artifacts"]):
        raise ValueError("Incomplete second-pass mask inventory")
    original = first_decisions(frozen)
    source_cases = {entry["id"]: entry["source_case"] for entry in source["cases"]}
    decisions = {case["id"]: case for case in data["cases"]}
    if len(decisions) != len(data["cases"]) or set(decisions) != {
        c["id"] for c in index["cases"]
    }:
        raise ValueError("Second-pass whole-area inventory mismatch")
    states: Counter[str] = Counter()
    counts: Counter[str] = Counter(dict.fromkeys(cur.CLASSES, 0))
    for case in index["cases"]:
        decision = decisions[case["id"]]
        if any(r["class"] != "ignore" for r in decision["ignore_regions"]):
            raise ValueError("Second-pass added non-ignore supervision")
        valid = (
            read_image(cur.checked(workspace / "first_pass", case["valid_mask"])) > 0
        )
        replay = cur.masks(
            valid, original[case["id"]]["regions"] + decision["ignore_regions"]
        )
        actual = read_image(cur.checked(second, artifacts[f"{case['id']}_classes.png"]))
        ignored = read_image(cur.checked(second, artifacts[f"{case['id']}_ignore.png"]))
        if not np.array_equal(actual, replay) or not np.array_equal(
            ignored, (replay == 0).astype("uint8") * 255
        ):
            raise ValueError("Second-pass mask replay mismatch")
        original_case = source_cases[case["id"]]
        if (
            decision["source_name"] != original_case["source_name"]
            or decision["source_window_id"] != original_case["source_window_id"]
        ):
            raise ValueError("Second-pass source identity mismatch")
        points = {p["source_label_id"]: p for p in decision["points"]}
        source_points = {p["source_label_id"]: p for p in original_case["points"]}
        if set(points) != set(source_points) or len(points) != len(decision["points"]):
            raise ValueError("Second-pass point identity mismatch")
        for identifier, point in points.items():
            source_point = source_points[identifier]
            if (
                point["published_lon"] != source_point["lon"]
                or point["published_lat"] != source_point["lat"]
                or point["publisher_pixel"]
                != cur.point_pixel(source_point, original_case)
                or point["state"] not in cur.POINT_STATES
                or not isinstance(point.get("reason"), str)
                or not point["reason"].strip()
            ):
                raise ValueError("Second-pass point position/state changed")
            x, y = point["publisher_pixel"]
            if point["state"] == "usable_provisional":
                idx = point["first_pass_region_index"]
                regions = original[case["id"]]["regions"]
                if (
                    type(idx) is not int
                    or not 0 <= idx < len(regions)
                    or regions[idx]["class"] != "vessel"
                    or not shape(regions[idx]["geometry"]).covers(Point(x, y))
                    or replay[int(y), int(x)] != cur.CLASSES["vessel"]
                    or not isinstance(point.get("alignment_evidence"), str)
                    or not point["alignment_evidence"].strip()
                ):
                    raise ValueError("Provisional point lost first-pass alignment")
            elif replay[int(y), int(x)] != 0:
                raise ValueError("Unresolved point is not ignored")
        for key, code in cur.CLASSES.items():
            counts[key] += int((replay == code).sum())
        states.update(p["state"] for p in decision["points"])
    if {k: states[k] for k in cur.POINT_STATES} != seal["counts"]:
        raise ValueError("Second-pass point accounting mismatch")
    return {
        "status": "provisional_second_pass_integrity_verified",
        "source_release_sha256": first["source_release_sha256"],
        "first_pass_seal_sha256": first["first_pass_seal_sha256"],
        "second_pass_seal_sha256": sha256_file(second / "seal.json"),
        "decision_sha256": seal["point_decisions"]["sha256"],
        "actor": data["actor"],
        "method": data["method"],
        "prior_label_exposure": True,
        "independent_review": False,
        "areas": len(index["cases"]),
        "point_states": seal["counts"],
        "mask_pairs_replayed": len(index["cases"]),
        "pixel_counts": dict(counts),
        "unmarked_windows": sum(
            not c["source_case"]["points"] for c in source["cases"]
        ),
        "upstream_exhaustive_annotation": "unverified",
        "local_independent_completeness": "unverified",
        "missing_label_policy": (
            "Unresolved points and nodata remain ignored. Unmarked windows and "
            "AI background are not independently established negatives. No "
            "negative or positive supervision is admitted from this "
            "quarantined release."
        ),
        "training_allowed": False,
        "evaluation_allowed": False,
        "locked_test_pixels_read": False,
        "verifier_code_sha256": sha256_file(__file__),
    }


def evidence(
    workspace: Path, frozen: Path, second: Path, inputs: Path
) -> dict[str, Any]:
    """Publish aggregate engineering evidence without coordinates or imagery."""
    review = completeness(workspace, frozen, second)
    radiometry = json.loads(inputs.read_text())
    if (
        radiometry["verifier_code_sha256"] != sha256_file(__file__)
        or radiometry["first_pass_seal_sha256"] != review["first_pass_seal_sha256"]
        or radiometry["source_release_sha256"] != review["source_release_sha256"]
        or radiometry["status"] != "quarantined_input_diagnostic"
        or any(
            radiometry[k] is not False
            for k in (
                "training_allowed",
                "evaluation_allowed",
                "independent_review",
                "publisher_pngs_directly_compatible",
                "native_context_and_label_alignment_ready",
            )
        )
    ):
        raise ValueError("Fresh matching quarantined input audit required")
    _, source, _ = cur.load_workspace(workspace)
    expected = {
        p["source_label_id"]
        for c in source["cases"]
        for p in c["source_case"]["points"]
    }
    patches = radiometry["native_calibration_patches"]
    if (
        {p["source_label_id"] for p in patches} != expected
        or len(patches) != len(expected)
        or radiometry["native_patch_radiometry_replayed"] != len(expected)
        or len(radiometry["publisher_previews"]) != 2 * review["areas"]
    ):
        raise ValueError("Input audit has incomplete point/window accounting")
    previews = radiometry["publisher_previews"]
    valid_band_pixels = sum(p["valid_pixels"] for p in previews)
    saturated = sum(p["pixels_at_255"] for p in previews)
    unresolved = sum(
        review["point_states"][state]
        for state in ("unresolved_class", "unresolved_alignment")
    )
    return {
        "schema_version": 1,
        "created_at": cur.now(),
        "status": "adoption_blocked_after_provisional_qa",
        "source_release_sha256": review["source_release_sha256"],
        "first_pass_seal_sha256": review["first_pass_seal_sha256"],
        "second_pass_seal_sha256": review["second_pass_seal_sha256"],
        "second_pass_decisions_sha256": review["decision_sha256"],
        "input_audit_sha256": sha256_file(inputs),
        "verifier_code_sha256": sha256_file(__file__),
        "areas": review["areas"],
        "acquisitions": len(source["sources"]),
        "point_states": review["point_states"],
        "mask_pairs_replayed": review["mask_pairs_replayed"],
        "pixel_counts": review["pixel_counts"],
        "unmarked_windows": review["unmarked_windows"],
        "actor": review["actor"],
        "method": review["method"],
        "prior_label_exposure": True,
        "independent_review": False,
        "source_files_hash_checked": radiometry["source_files_hash_checked"],
        "publisher_band_images_checked": len(previews),
        "joint_valid_band_pixel_exposures": valid_band_pixels,
        "band_pixels_at_display_ceiling": saturated,
        "display_ceiling_fraction": (
            saturated / valid_band_pixels if valid_band_pixels else None
        ),
        "native_calibrated_patches_replayed": len(patches),
        "calibration_max_abs_error_db": max(
            p["calibration_max_abs_error_db"] for p in patches
        ),
        "publisher_pngs_directly_compatible": False,
        "native_model_contexts_prepared": 0,
        "blockers": [
            (
                "Annotation-specific permission for the 2022 database unresolved; "
                "clarification request not sent."
            ),
            (
                f"{unresolved} "
                "point interpretations unresolved; "
                f"{review['point_states']['usable_provisional']} source/first-pass "
                "consistencies are provisional self/AI decisions."
            ),
            (
                "Upstream and independently reviewed completeness unverified; "
                f"{review['unmarked_windows']} unmarked windows are not "
                "confirmed negatives."
            ),
            (
                "Clipped PNGs cannot supply native calibrated sigma0; QA patches "
                "do not satisfy the 2048px model-context contract."
            ),
            (
                "Annotation-time grid and intended objects need clarification; "
                "upstream pretraining/split leakage unaudited."
            ),
        ],
        "training_allowed": False,
        "evaluation_allowed": False,
        "threshold_selection_allowed": False,
        "promotion_allowed": False,
        "locked_test_pixels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "inputs", "seal", "check", "publish"]
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--first-pass", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--second-pass", type=Path)
    parser.add_argument("--input-audit", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = alignment_packet(args.workspace, args.first_pass, args.output)
    elif args.action == "seal":
        if args.packet is None or args.observations is None:
            parser.error("seal requires --packet and --observations")
        result = seal_review(
            args.workspace, args.first_pass, args.packet, args.observations, args.output
        )
    elif args.action == "publish":
        if args.second_pass is None or args.input_audit is None:
            parser.error("publish requires --second-pass and --input-audit")
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to(
            (REPO_ROOT / "docs/evidence").resolve()
        ):
            parser.error("Fresh docs/evidence output required")
        result = evidence(
            args.workspace, args.first_pass, args.second_pass, args.input_audit
        )
        cur.save(output, result)
    else:
        output = cur.fresh(args.output)
        if args.action == "check":
            if args.second_pass is None:
                parser.error("check requires --second-pass")
            result = completeness(args.workspace, args.first_pass, args.second_pass)
        else:
            result = compatibility(args.workspace, args.first_pass)
        cur.save(output / "audit.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {"publisher_previews", "native_calibration_patches", "artifacts"}
            }
        )
    )


if __name__ == "__main__":
    main()
