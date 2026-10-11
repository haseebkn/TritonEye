"""Record explicitly authored first-pass decisions and a portable blind bundle.

This does not generate SAR labels or invent independent review. It converts
checked-in, image-inspected pixel decisions to geographic centres and binds
full-area review records to the actual images and exact annotations.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.geo import Georeferencer
from agents.nl_benchmark import (
    DEFAULT_DATASET,
    bounded_path,
    objects_digest,
    validate_contract,
)


def prepare(directory: Path) -> None:
    if (directory / "release.json").exists():
        raise ValueError("Immutable release; preserve it and prepare a new version")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    canonical_release = (
        REPO_ROOT
        / "datasets/nl_benchmark"
        / ("v" + manifest["dataset_version"])
        / "release.json"
    )
    output = (
        REPO_ROOT
        / "data/benchmarks/nl"
        / manifest["dataset_version"]
        / "review_bundle.html"
    )
    if canonical_release.exists() or (output.parent / "release_record.json").exists():
        raise ValueError(
            "Released data version cannot be overwritten through another output path"
        )
    decisions = json.loads(
        (directory / "annotation_decisions.json").read_text(encoding="utf-8")
    )
    area_map = {r["roi_id"]: r for r in decisions["areas"]}
    expected = {r["id"] for s in manifest["scenes"] for r in s["rois"]}
    if set(area_map) != expected or len(decisions["areas"]) != len(expected):
        raise ValueError("Every selected area needs an explicit image-first decision")
    now = datetime.now(timezone.utc).isoformat()
    locked = datetime.fromtimestamp(
        (directory / "split_lock.json").stat().st_mtime, timezone.utc
    ).isoformat()
    events: list[dict[str, Any]] = [
        {
            "id": "selection_locked",
            "kind": "selection_locked",
            "timestamp": locked,
            "notes": "Recorded before first image inspection; assignment is immutable.",
        }
    ]
    objects = []
    sections = []
    for scene in manifest["scenes"]:
        for roi in scene["rois"]:
            rid = roi["id"]
            draft = area_map[rid]
            labels = []
            with rasterio.open(bounded_path(REPO_ROOT, roi["chip"]["path"])) as src:
                with Georeferencer.from_dataset(src) as geo:
                    for i, decision in enumerate(draft["objects"], 1):
                        x0, y0, x1, y1 = decision["bbox_px"]
                        lon, lat = geo.xy((y0 + y1) / 2, (x0 + x1) / 2)
                        labels.append(
                            {
                                **decision,
                                "id": f"{rid}_{i:03d}",
                                "roi_id": rid,
                                "product_id": scene["product_id"],
                                "annotator": decisions["annotator"],
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [float(lon[0]), float(lat[0])],
                                },
                                "ais_role": "supporting_only",
                                "evidence": [
                                    "Image-first native co-pol and cross-pol "
                                    "inspection; no detector overlay."
                                ],
                            }
                        )
            objects.extend(labels)
            events.append(
                {
                    "id": f"{rid}_first_pass",
                    "kind": "first_pass",
                    "timestamp": now,
                    "roi_id": rid,
                    "reviewer": decisions["annotator"],
                    "independent": False,
                    "qualifications": decisions["qualifications"],
                    "cells_reviewed": [c["id"] for c in roi["cells"]],
                    "source_chip_sha256": roi["chip"]["sha256"],
                    "annotation_digest": objects_digest(labels),
                    "decision": "complete_first_pass",
                    "notes": draft["notes"]
                    + " Timestamp records entry, not an exact inspection start.",
                }
            )
            figures = []
            for band, entry in zip(scene["polarizations"], roi["previews"]):
                payload = base64.b64encode(
                    bounded_path(REPO_ROOT, entry["path"]).read_bytes()
                ).decode()
                figures.append(
                    f"<figure><figcaption>{html.escape(band.upper())}, "
                    "native 512 × 512</figcaption>"
                    f'<img src="data:image/png;base64,{payload}" width="512" '
                    f'height="512" alt="{html.escape(rid)} {band}"></figure>'
                )
            sections.append(
                f"<section><h2>{html.escape(rid)} — "
                f'{html.escape(scene["split"])}</h2>'
                f'<p>{html.escape(scene["acquisition_time"])} · '
                f'{scene["product_id"]} · chip SHA-256 {roi["chip"]["sha256"]}</p>'
                "<p>Review cells 0_0, 1_0, 0_1, 1_1 in both images, including "
                "empty-looking areas. Test imagery is annotation/QA only: "
                "no detector tuning or scoring.</p>"
                f'<div class="images">{"".join(figures)}</div><details><summary>'
                "Reveal provisional first-pass decisions only after your own "
                "complete-area pass</summary>"
                f"<pre>{html.escape(json.dumps(draft,indent=2))}</pre>"
                "</details></section>"
            )
    events.append(
        {
            "id": "independent_review_unavailable",
            "kind": "review_limitation",
            "timestamp": now,
            "decision": "unavailable",
            "notes": (
                "Owner explicitly confirmed no independent reviewer available "
                "on 2026-10-05. No human expert or independent AI review is claimed."
            ),
        }
    )
    annotation = {
        "dataset_version": manifest["dataset_version"],
        "source_kind": "machine_assisted_sar_first_pass",
        "objects": objects,
    }
    history = {"dataset_version": manifest["dataset_version"], "events": events}
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    lock = json.loads((directory / "split_lock.json").read_text(encoding="utf-8"))
    status = validate_contract(manifest, selection, lock, annotation, history)
    write_json(directory / "annotations.json", annotation)
    write_json(directory / "review_history.json", history)
    write_json(directory / "status.json", status)
    output.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>NL SAR provisional review bundle</title><style>"
        "body{font:16px system-ui;background:#edf2f4;color:#172830;margin:24px}"
        "section{background:white;padding:20px;margin:24px 0}"
        "p{overflow-wrap:anywhere}.images{display:flex;flex-wrap:wrap;gap:16px}"
        "figure{margin:0}img{max-width:100%;height:auto}"
        "pre{white-space:pre-wrap}summary{cursor:pointer}</style>"
        "<h1>NL SAR detector-blind review bundle</h1>"
        "<p>Contains modified Copernicus Sentinel data (2025, 2026). "
        "Provisional AI first pass; independent review unavailable. "
        "No vessel accuracy claims. Read the checked-in annotation guide. "
        "Predictions and AIS identities are not embedded.</p>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    files = [
        "selection.json",
        "split_lock.json",
        "manifest.json",
        "annotation_decisions.json",
        "annotations.json",
        "review_history.json",
        "status.json",
        "ANNOTATION_GUIDE.md",
        "DATASET_CARD.md",
    ]
    write_json(
        directory / "release.json",
        {
            "dataset_version": manifest["dataset_version"],
            "status": "provisional_not_independently_reviewed",
            "created_at": now,
            "files": {name: sha256_file(directory / name) for name in files},
            "review_bundle": {
                "path": output.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(output),
            },
        },
    )
    write_json(
        output.parent / "release_record.json",
        {
            "dataset_version": manifest["dataset_version"],
            "release_sha256": sha256_file(directory / "release.json"),
            "notes": (
                "Do not overwrite data for a released version via another "
                "output directory."
            ),
        },
    )
    print(json.dumps(status, indent=2))
    print("Review bundle:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    prepare(args.dataset)
