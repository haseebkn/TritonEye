"""Replay provisional first-pass masks and publish aggregate QA, never labels.

The audit checks software integrity/coverage, not visual-review quality or vessel
truth. Original point coordinates and private image/annotation pixels are not
redistributed. No publisher-point reveal, training or inference is performed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agents.allenai_curation import CLASSES, checked, load_workspace, masks, save
from agents.artifacts import REPO_ROOT, sha256_file


def read_image(path: Path) -> Any:
    with Image.open(path) as image:
        return np.asarray(image)


def audit(
    workspace: Path, frozen: Path, *, purpose: str = "curation_qa"
) -> dict[str, Any]:
    if purpose != "curation_qa":
        raise ValueError("Provisional first-pass quarantine: adoption exports blocked")
    descriptor, source, index = load_workspace(workspace)
    seal_path = frozen / "seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal["status"] != "provisional_first_pass_frozen"
        or seal["workspace_sha256"] != sha256_file(workspace / "workspace.json")
        or seal["class_codes"] != CLASSES
        or any(
            seal[key] is not False
            for key in ("independent_review", "training_allowed", "evaluation_allowed")
        )
    ):
        raise ValueError("Matching provisional, quarantined first pass required")
    decisions = json.loads(
        checked(frozen, seal["decisions"]).read_text(encoding="utf-8")
    )
    if (
        not isinstance(decisions.get("actor"), str)
        or not decisions["actor"].strip()
        or decisions["method"] not in {"self", "ai_assisted"}
        or type(decisions["prior_label_exposure"]) is not bool
        or decisions["independent_review"] is not False
    ):
        raise ValueError("Provisional actor and honest exposure declaration required")
    inputs = {case["id"]: case for case in decisions["cases"]}
    expected = {case["id"] for case in index["cases"]}
    if (
        len(expected) != len(index["cases"])
        or len(inputs) != len(decisions["cases"])
        or set(inputs) != expected
    ):
        raise ValueError("Exactly one authored decision per whole area required")
    expected_artifacts = {
        f"{identifier}_{kind}.png"
        for identifier in expected
        for kind in ("classes", "ignore")
    }
    artifacts = {item["path"]: item for item in seal["artifacts"]}
    if set(artifacts) != expected_artifacts or len(artifacts) != len(seal["artifacts"]):
        raise ValueError("Exactly one class/ignore mask pair per area required")
    totals: Counter[str] = Counter(dict.fromkeys(CLASSES, 0))
    region_counts: Counter[str] = Counter(dict.fromkeys(CLASSES, 0))
    valid_pixels, invalid_pixels = 0, 0
    cases = []
    for case in index["cases"]:
        decision = inputs[case["id"]]
        review = decision["inspection"]
        if (
            decision["full_valid_area_reviewed"] is not True
            or review["bands"] != ["vv", "vh"]
            or review["resolution"] != [case["width"], case["height"]]
            or not isinstance(review["observations"], str)
            or not review["observations"].strip()
            or any(
                review[key] is not False
                for key in (
                    "source_labels_displayed",
                    "detector_outputs_displayed",
                    "ais_displayed",
                    "independent",
                )
            )
        ):
            raise ValueError("Complete provisional current-display inspection required")
        valid = read_image(checked(workspace / "first_pass", case["valid_mask"]))
        if (
            valid.shape != (case["height"], case["width"])
            or not np.isin(valid, [0, 255]).all()
        ):
            raise ValueError("Malformed packet validity mask")
        valid = valid > 0
        authored = masks(valid, decision["regions"])
        coverage = (
            masks(
                valid,
                [{**region, "class": "background"} for region in decision["regions"]],
            )
            > 0
        )
        if not coverage[valid].all():
            raise ValueError("Unreviewed valid pixels remain")
        class_ref = artifacts[f"{case['id']}_classes.png"]
        ignore_ref = artifacts[f"{case['id']}_ignore.png"]
        actual = read_image(checked(frozen, class_ref))
        ignore = read_image(checked(frozen, ignore_ref))
        if not np.array_equal(actual, authored) or not np.array_equal(
            ignore, (authored == 0).astype("uint8") * 255
        ):
            raise ValueError("Masks disagree with authored decisions/ignore policy")
        counts = {label: int((actual == code).sum()) for label, code in CLASSES.items()}
        authored_counts = dict(
            Counter(region["class"] for region in decision["regions"])
        )
        totals.update(counts)
        region_counts.update(authored_counts)
        valid_pixels += int(valid.sum())
        invalid_pixels += int((~valid).sum())
        cases.append(
            {
                "id": case["id"],
                "valid_pixels": int(valid.sum()),
                "nodata_pixels": int((~valid).sum()),
                "pixel_counts": counts,
                "authored_region_counts": authored_counts,
                "images": case["images"],
                "valid_mask": case["valid_mask"],
                "class_mask": class_ref,
                "ignore_mask": ignore_ref,
            }
        )
    if dict(totals) != seal["pixel_counts"]:
        raise ValueError("Frozen aggregate pixel counts disagree")
    return {
        "status": "provisional_first_pass_integrity_verified",
        "actor": decisions["actor"],
        "method": decisions["method"],
        "prior_label_exposure": decisions["prior_label_exposure"],
        "exposure_details": decisions.get(
            "prior_label_exposure_details", "Not recorded; consult actor"
        ),
        "independent_review": False,
        "workspace_sha256": sha256_file(workspace / "workspace.json"),
        "source_release_sha256": descriptor["source_release_sha256"],
        "first_pass_seal_sha256": sha256_file(seal_path),
        "decisions_sha256": seal["decisions"]["sha256"],
        "frozen_at": seal["frozen_at"],
        "verifier_code_sha256": sha256_file(__file__),
        "first_pass_completed_areas": len(cases),
        "acquisitions": len(
            {case["source_case"]["source_name"] for case in source["cases"]}
        ),
        "current_pass_whole_band_images": 2 * len(cases),
        "class_ignore_masks_replayed": 2 * len(cases),
        "valid_pixels": valid_pixels,
        "nodata_pixels": invalid_pixels,
        "ignored_valid_pixels": totals["ignore"] - invalid_pixels,
        "pixel_counts": dict(totals),
        "authored_region_counts": dict(region_counts),
        "training_allowed": False,
        "evaluation_allowed": False,
        "threshold_selection_allowed": False,
        "publisher_points_revealed_in_current_pass": 0,
        "locked_test_pixels_read": False,
        "licence_clarification": "unresolved",
        "scope": (
            "Mask/coverage/integrity replay and actor-declared visual inspection only. "
            "No independent vessel truth, hull segmentation, exhaustive negatives, "
            "training, performance metrics or measured improvement."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--first-pass", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        target = args.output.resolve()
        if (
            target.exists()
            or not target.is_relative_to(REPO_ROOT.resolve())
            or target.is_relative_to((REPO_ROOT / "datasets").resolve())
            or target.is_relative_to(args.workspace.resolve())
            or target.is_relative_to(args.first_pass.resolve())
        ):
            raise ValueError(
                "Fresh project-local audit output outside frozen inputs required"
            )
    result = audit(args.workspace, args.first_pass)
    if args.output is not None:
        save(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
