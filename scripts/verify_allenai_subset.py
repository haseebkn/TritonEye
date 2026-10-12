"""Replay all published-window pixels against exact cached NL source products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.windows import Window

from agents.allenai_subset import WORLD_M, load_alignment_subset, warp_publisher_window
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.nl_benchmark import require_development_scene


def verify(release: Path) -> dict[str, Any]:
    data = load_alignment_subset(release)
    sources = {s["name"]: s for s in data["sources"]}
    source_hashes = 0
    for source in sources.values():
        for case in (c for c in data["cases"] if c["source_name"] == source["name"]):
            require_development_scene(
                source["product_id"],
                {"name": source["name"], "study_roi": case["physical_geometry"]},
            )
        for item in source["files"]:
            if sha256_file(REPO_ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Source product hash mismatch")
            source_hashes += 1
    windows, patches = 0, 0
    for case in data["cases"]:
        source = sources[case["source_name"]]
        affine = Affine(*source["publisher_grid_affine"])
        origin = (
            int((affine.c + WORLD_M / 2) / affine.a),
            int((WORLD_M / 2 - affine.f) / affine.a),
        )
        col, row, w, h = case["global_pixel_window"]
        window = Window(col - origin[0], row - origin[1], w, h)
        for preview in case["publisher_previews"]:
            path = REPO_ROOT / next(
                f["path"]
                for f in source["files"]
                if "/measurement/" in f["path"] and f"-{preview['band']}-" in f["path"]
            )
            with rasterio.open(path) as src:
                values, valid = warp_publisher_window(src, affine, window)
            actual = np.asarray(Image.open(REPO_ROOT / preview["path"]))
            mask = np.asarray(Image.open(REPO_ROOT / preview["valid_mask"]["path"]))
            if not np.array_equal(actual, np.clip(values, 0, 255).astype("uint8")):
                raise ValueError("Publisher window replay pixels disagree")
            if not np.array_equal(mask, valid.astype("uint8") * 255):
                raise ValueError("Publisher validity mask replay disagrees")
            if int(valid.sum()) != preview["valid_pixels"] or not valid.any():
                raise ValueError("Valid-pixel count mismatch or blank window")
            windows += 1
        for point in case["alignment_checks"]:
            px, py = point["point_px"]
            if not 0 <= px < 256 or not 0 <= py < 256:
                raise ValueError("Label not within its native QA patch")
            patch = point["patch"]
            require_development_scene(
                source["product_id"],
                {"name": source["name"], "study_roi": patch["geometry"]},
            )
            with rasterio.open(REPO_ROOT / patch["chip"]["path"]) as src:
                array = src.read()
                if array.shape != (2, 256, 256) or not np.isfinite(array).all():
                    raise ValueError("Invalid calibrated native patch")
                if not (src.read_masks() > 0).all():
                    raise ValueError("Native QA patch has missing imagery")
                if src.descriptions != ("sigma0_db_vv", "sigma0_db_vh"):
                    raise ValueError(
                        "Incorrect patch polarization/calibration metadata"
                    )
            patches += 1
    blocked = []
    for purpose in ("training", "evaluation", "threshold_selection"):
        try:
            load_alignment_subset(release, purpose=purpose)
        except ValueError:
            blocked.append(purpose)
        else:
            raise ValueError("Scientific-adoption quarantine was bypassed")
    return {
        "status": "passed",
        "release_sha256": sha256_file(release),
        "local_artifacts_hash_verified": len(data["artifacts"]),
        "source_files_hash_verified": source_hashes,
        "publisher_band_windows_replayed": windows,
        "native_patches_checked": patches,
        "adoption_requests_blocked": blocked,
        "locked_test_pixels_read": False,
        "scope": "Software, geometry and pixel replay only; "
        "no vessel class or completeness certification",
        "verifier_code_sha256": sha256_file(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=REPO_ROOT / "datasets/allenai_nl/v0.1.0/release.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Fresh project-local verification output required")
    result = verify(args.release)
    write_json(args.output, result)
    print(json.dumps(result))
