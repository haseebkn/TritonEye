"""Publish aggregate provenance only; keep unresolved source annotations local."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents.allenai_archive import ETAG, GENERATION, SIZE, URL
from agents.artifacts import REPO_ROOT, sha256_file, write_json

BASE = REPO_ROOT / "data/model_research/allenai_nl_v010"
PUBLIC = REPO_ROOT / "datasets/allenai_nl/v0.1.0"


def publish(local: Path) -> None:
    local = local.resolve()
    if not local.is_relative_to(BASE.resolve()):
        raise ValueError("Expected a local alignment release")
    if (PUBLIC / "manifest.json").exists() or (PUBLIC / "release.json").exists():
        raise ValueError("Public release is immutable")
    data = json.loads(local.read_text(encoding="utf-8"))
    if data["version"] != "0.1.0" or data["status"] != "quarantined_qa":
        raise ValueError("Unexpected private release version")
    if data["training_allowed"] or data["evaluation_allowed"]:
        raise ValueError("Quarantined data cannot be scientifically adopted")
    for artifact in data["artifacts"]:
        path = (REPO_ROOT / artifact["path"]).resolve()
        if not path.is_relative_to(BASE.resolve()):
            raise ValueError("Artifact escaped versioned data root")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError("Local artifact integrity mismatch")
    points = [p for c in data["cases"] for p in c["alignment_checks"]]
    shifts = [p["grid_vs_native_tps_displacement_m"] for p in points]
    index = json.loads((BASE / "cc_archive_index.json").read_text(encoding="utf-8"))
    public = {
        "version": data["version"],
        "status": data["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_allowed": False,
        "evaluation_allowed": False,
        "threshold_selection_allowed": False,
        "independent_local_review": False,
        "annotation_completeness": data["annotation_completeness"],
        "upstream_splits_adopted": False,
        "candidate_screen": {
            "whole_nl_windows": data["whole_nl_candidate_windows"],
            "published_points": data["whole_nl_candidate_points"],
        },
        "prepared_windows": data["prepared_windows"],
        "prepared_points": data["prepared_points"],
        "empty_windows_not_confirmed_negative": data[
            "empty_windows_not_confirmed_negative"
        ],
        "fully_valid_publisher_windows": sum(
            all(p["valid_pixels"] == p["pixels"] for p in c["publisher_previews"])
            for c in data["cases"]
        ),
        "partial_publisher_windows": [
            {
                "source_window_id": c["source_window_id"],
                "bands": [
                    {k: p[k] for k in ("band", "valid_pixels", "pixels")}
                    for p in c["publisher_previews"]
                ],
            }
            for c in data["cases"]
            if any(p["valid_pixels"] != p["pixels"] for p in c["publisher_previews"])
        ],
        "blocked": data["blocked"],
        "local_artifact_count": len(data["artifacts"]),
        "source_metadata_sha256": data["metadata_sha256"],
        "source_inventory_sha256": data["inventory_sha256"],
        "code_sha256": data["code_sha256"],
        "geometry_code_sha256": data["geometry_code_sha256"],
        "region_sha256": data["region_sha256"],
        "software": {k: data[k] for k in ("python", "rasterio", "gdal")},
        "licence": data["licence"],
        "alignment_limits": data["alignment_limits"],
        "grid_vs_native_tps_displacement_m": {
            "n": len(shifts),
            "minimum": float(min(shifts)),
            "median": float(np.median(shifts)),
            "maximum": float(max(shifts)),
            "meaning": "Transform-convention diagnostic, "
            "not measured vessel position error",
        },
        "cc_archive": {
            "url": URL,
            "generation": GENERATION,
            "bytes": SIZE,
            "etag": ETAG,
            "index_sha256": data["archive_index_sha256"],
            "header_sequence_sha256": index["header_sequence_sha256"],
            "windows": index["windows"],
            "members": len(index["members"]),
            "range_requests": index["range_requests"],
            "bytes_transferred": index["bytes_transferred"],
            "raster_members_extracted": 0,
            "limits": index["limits"],
            "crosswalk": data["cc_archive_crosswalk"],
        },
        "sources": [
            {
                k: source[k]
                for k in (
                    "name",
                    "product_id",
                    "original_image_uuid",
                    "acquisition_time",
                    "polarizations",
                    "catalogue_sha256",
                    "stored_dimensions",
                    "suggested_dimensions",
                    "source_origin_match",
                    "files",
                )
            }
            for source in data["sources"]
        ],
        "point_qa": [
            {
                "source_label_id": p["source_label_id"],
                "visual_status": p["visual_status"],
                "grid_vs_native_tps_displacement_m": p[
                    "grid_vs_native_tps_displacement_m"
                ],
                "point_px": p["point_px"],
                "crop_center_edge_clamped": p["crop_center_edge_clamped"],
                "patch_sha256": p["patch"]["chip"]["sha256"],
            }
            for p in points
        ],
        "lock_metadata_sha256": {
            p.relative_to(REPO_ROOT).as_posix(): sha256_file(p)
            for p in sorted((REPO_ROOT / "datasets/nl_benchmark").glob("*/*.json"))
        },
    }
    source_assets = json.loads(
        (BASE / "source_downloads.json").read_text(encoding="utf-8")
    )
    original_source = (
        "https://raw.githubusercontent.com/allenai/vessel-detection-sentinels/"
        + data["original_source_revision"]
        + "/"
    )
    paths = {
        "siv_README.md": "README.md",
        "siv_warp.py": "src/data/warp.py",
        "siv_preprocess.py": "src/data/preprocess.py",
        "siv_dataset.py": "src/data/dataset.py",
        "siv_download_imagery.py": "download_imagery.py",
        "siv_sentinel1.py": "src/utils/download/sentinel1.py",
    }
    for filename, source_path in paths.items():
        path = BASE / "sources" / filename
        source_assets["assets"].append(
            {
                "url": original_source + source_path,
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    # Snapshot origins must be documented accurately; no source payloads published.
    for asset in source_assets["assets"]:
        path = REPO_ROOT / asset["path"]
        if sha256_file(path) != asset["sha256"]:
            raise ValueError("Publisher source snapshot integrity mismatch")
        asset["path"] = path.relative_to(REPO_ROOT).as_posix()
    write_json(PUBLIC / "manifest.json", public)
    write_json(PUBLIC / "source_evidence.json", source_assets)
    release = {
        "version": "0.1.0",
        "status": "quarantined_qa",
        "manifest_sha256": sha256_file(PUBLIC / "manifest.json"),
        "source_evidence_sha256": sha256_file(PUBLIC / "source_evidence.json"),
        "local_manifest": {
            "path": local.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(local),
        },
    }
    write_json(PUBLIC / "release.json", release)
    print(json.dumps(release))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    publish(parser.parse_args().manifest)
