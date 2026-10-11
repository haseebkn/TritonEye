"""Audit sparse shoreline controls on native NL benchmark pixels, not vessels.

Never open locked test rasters. Optical controls are development checks, not
independent SAR labels, tide surveys or certification of intervening shoreline.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import GCPTransformer, rowcol
from rasterio.windows import Window
from shapely.geometry import Point, shape

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.landmask import LandMask, LandMaskUnavailable
from agents.nl_benchmark import DEFAULT_DATASET, bounded_path, require_development_scene
from agents.region import load_region
from agents.run_versions import digest


def verify_reference(root: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Verify recorded export/pixel provenance when present; never invent it."""
    reference = document.get("reference", {})
    if not reference.get("image_sha256"):
        return {"status": "image_hash_not_recorded"}
    image = bounded_path(root, reference["local_provenance"]).with_name("imagery.png")
    if sha256_file(image) != reference["image_sha256"]:
        raise ValueError("Reference imagery hash mismatch")
    xmin, ymin, xmax, ymax = reference["image_extent_wgs84"]
    width, height = reference["image_size"]
    if not (width > 0 and height > 0 and xmin < xmax and ymin < ymax):
        raise ValueError("Valid image dimensions and extent required")
    for feature in document["features"]:
        if "image_pixel_xy" not in feature["properties"]:
            continue
        x, y = feature["properties"]["image_pixel_xy"]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Control outside reference image")
        expected = [xmin + x / width * (xmax - xmin), ymax - y / height * (ymax - ymin)]
        coordinates = feature["geometry"]["coordinates"]
        if not np.allclose(expected, coordinates, rtol=0, atol=1e-7):
            raise ValueError("Control geography differs from reference image pixel")
    return {"status": "image_hash_and_recorded_pixels_verified"}


def valid_chip_points(path: Path, points: list[Point]) -> list[bool]:
    """Require finite, unmasked pixels in both calibrated bands; 0 dB is valid."""
    if not points:
        return []
    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError("Two calibrated polarization bands required")
        gcps, gcp_crs = src.gcps
        crs = gcp_crs if gcps else src.crs
        if crs is None:
            raise ValueError("Chip georeferencing CRS required")
        project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = project.transform([p.x for p in points], [p.y for p in points])
        if gcps:
            with GCPTransformer(gcps, tps=True) as converter:
                rows, cols = converter.rowcol(xs, ys)
        else:
            rows, cols = rowcol(src.transform, xs, ys)
        valid = []
        for row, col in zip(rows, cols):
            if not (0 <= row < src.height and 0 <= col < src.width):
                valid.append(False)
                continue
            pixel = src.read(window=Window(int(col), int(row), 1, 1), masked=True)
            valid.append(
                bool(not np.asarray(pixel.mask).any() and np.isfinite(pixel.data).all())
            )
        return valid


def area_coverage(
    roi: dict[str, Any], chip: Path, mask: LandMask, collections: list[dict[str, Any]]
) -> dict[str, Any]:
    """A bounding-box check is not proof that a control is inside valid imagery."""
    geometry = shape(roi["geometry"])
    controls = []
    for collection in collections:
        document = collection["document"]
        for feature in document["features"]:
            point = shape(feature["geometry"])
            if geometry.covers(point):
                controls.append((collection["location"], feature, point))
    valid = valid_chip_points(chip, [c[2] for c in controls])
    checks = []
    for (location, feature, point), keep in zip(controls, valid):
        if not keep:
            continue
        surfaces, distances = mask.classify_physical([point.x], [point.y])
        expected = feature["properties"]["expected"]
        checks.append(
            {
                "location": location,
                "id": feature["properties"]["id"],
                "expected": expected,
                "actual": surfaces[0],
                "passed": surfaces[0] == expected,
                "distance_to_shore_m": (
                    float(distances[0]) if np.isfinite(distances[0]) else None
                ),
            }
        )
    has_both = {c["expected"] for c in checks} == {"land", "water"}
    return {
        "status": (
            "failed"
            if any(not c["passed"] for c in checks)
            else "sparse_controls_passed" if has_both else "insufficient_controls"
        ),
        "controls_in_roi": len(controls),
        "controls_on_joint_valid_pixels": len(checks),
        "outside_joint_valid_pixels": len(controls) - len(checks),
        "both_land_and_water_checked": has_both,
        "checks": checks,
    }


def audit_coverage(
    root: Path, manifest_path: Path, registry_path: Path
) -> dict[str, Any]:
    """Verify chip hashes and disclose each area's scope and skipped test split."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("type") != "ShorelineControlRegistry":
        raise ValueError("ShorelineControlRegistry required")
    collections = []
    for entry in registry["collections"]:
        path = bounded_path(registry_path.parent, entry["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        collections.append(
            {
                "location": entry["location"],
                "sha256": sha256_file(path),
                "document": document,
                "reference_integrity": verify_reference(root, document),
            }
        )
    areas = []
    skipped: list[str] = []
    region = load_region()
    seen_rois: set[str] = set()
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            skipped.extend(r["id"] for r in scene["rois"])
            continue  # Before path resolution, hashing, opening or mask queries.
        if scene["split"] not in {"train", "validation"}:
            raise ValueError("Known development split required")
        for roi in scene["rois"]:
            geometry = shape(roi["geometry"])
            if (
                geometry.geom_type not in {"Polygon", "MultiPolygon"}
                or geometry.is_empty
                or not geometry.is_valid
                or not region.covers(geometry)
            ):
                raise ValueError("Valid polygonal NL review area required")
            if roi["id"] in seen_rois:
                raise ValueError("Duplicate review area cannot inflate coverage")
            seen_rois.add(roi["id"])
            require_development_scene(
                scene["product_id"], {**scene, "study_roi": roi["geometry"]}
            )
            path = bounded_path(root, roi["chip"]["path"])
            result = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "acquisition_time": scene["acquisition_time"],
                "split": scene["split"],
                "region": scene["region"],
                "regime": roi["regime"],
                "polarizations": scene["polarizations"],
                "chip_sha256": roi["chip"]["sha256"],
            }
            try:
                if sha256_file(path) != roi["chip"]["sha256"]:
                    raise ValueError("Chip hash mismatch")
                mask = LandMask.for_footprint(
                    geometry.bounds,
                    cache_dir=str(root / "data/reference"),
                    controls_path=str(registry_path),
                )
                result.update(area_coverage(roi, path, mask, collections))
                result["shoreline_source"] = mask.source
                result["shoreline_licence"] = mask.licence
                result["runtime_control_validation"] = {
                    key: mask.validation.get(key)
                    for key in (
                        "status",
                        "registry_sha256",
                        "controls_sha256",
                        "checked",
                        "failed",
                    )
                }
                result["physical_reference_land_within_roi_sha256"] = digest(
                    {
                        "crs": "EPSG:4326",
                        "wkb": geometry.difference(
                            mask.water_geometry(geometry)
                        ).wkb_hex,
                    }
                )
            except (OSError, ValueError, LandMaskUnavailable) as exc:
                result.update(status="unavailable", reason=str(exc))
            areas.append(result)
    coastal = [a for a in areas if a["regime"] == "coastal"]
    return {
        "purpose": "Sparse imagery-backed development shoreline coverage only",
        "manifest_sha256": sha256_file(manifest_path),
        "registry_sha256": sha256_file(registry_path),
        "method_sha256": sha256_file(Path(__file__)),
        "collections": [
            {
                "location": c["location"],
                "sha256": c["sha256"],
                "reference": c["document"].get("reference", {}),
                "reference_integrity": c["reference_integrity"],
            }
            for c in collections
        ],
        "areas": areas,
        "locked_test_areas_not_opened": skipped,
        "coastal_areas": len(coastal),
        "coastal_areas_with_passing_land_and_water_controls": sum(
            a["status"] == "sparse_controls_passed" for a in coastal
        ),
        "province_wide_validation": False,
        "independent_vessel_labels_added": 0,
        "buffer_selected": None,
        "step_2_complete": False,
        "remaining": "Independent exhaustive validation labels are needed to "
        "measure false alarms and missed vessels across coastal buffers. "
        "Uncontrolled shoreline segments and locked test areas remain unvalidated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_DATASET / "manifest.json"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "configs/coastline/regional_controls.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Fresh output required; evidence cannot be overwritten")
    output = audit_coverage(args.artifact_root, args.manifest, args.registry)
    write_json(args.output, output)
    print(
        json.dumps(
            {
                "coastal_areas": output["coastal_areas"],
                "passing": output["coastal_areas_with_passing_land_and_water_controls"],
                "step_2_complete": False,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
