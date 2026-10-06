"""Fresh bounded NL baseline replay; locked test imagery is never inferred."""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point, mapping, shape

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.association import aligned_ais
from agents.baseline_association import AIS_PROXIMITY_RADIUS_M, association_summary
from agents.baseline_context import (
    area_km2,
    context_geometry,
    require_context_split,
    selected_origins,
)
from agents.baseline_metrics import score_roi
from agents.baseline_report import DEFAULT_TARGETS, DEFAULT_THRESHOLDS, make_report
from agents.calibration import NODATA_DB, Calibrator
from agents.coastal_benchmark import compare_buffers
from agents.coastal_policy import annotations as policy_annotations
from agents.correlation.correlation_agent import CORRELATION_RADIUS_M
from agents.geo import Georeferencer
from agents.infrastructure import classify_infrastructure
from agents.landmask import LandMask, resolve_cache_dir
from agents.nl_benchmark import (
    DEFAULT_DATASET,
    bounded_path,
    load_dataset,
    require_development_scene,
    validation_labels,
)
from agents.run_versions import digest, processing_versions
from agents.xview3_detector import TILE_SIZE, XView3Detector, dedupe_detections

MODEL_SHA256 = "c21d6b4204e2803b56014c9780d8d62190ed81d9c2ec56d0cbf437b439307d38"
SCORE_FLOOR = 0.05


class Resources:
    """Sample process RSS at 100 ms; absolute memory, not an OS peak guarantee."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> Resources:
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

    def summary(self) -> dict[str, Any]:
        return {
            "wall_seconds": time.perf_counter() - self.started,
            "rss_sampled_peak_bytes": self.peak,
            "rss_sampling_seconds": 0.1,
        }


def annotate(
    collection: dict[str, Any], mask: LandMask, *, infrastructure: bool
) -> None:
    points = [shape(f["geometry"]).centroid for f in collection["features"]]
    lons, lats = [p.x for p in points], [p.y for p in points]
    physical, distances = mask.classify_physical(lons, lats)
    flags, names = (
        classify_infrastructure(
            lons,
            lats,
            shape(
                collection.get("study_roi")
                or collection["benchmark"]["valid_imagery_roi"]
            ).bounds,
        )
        if infrastructure
        else ([False] * len(points), [""] * len(points))
    )
    for f, surface, distance, flag, name in zip(
        collection["features"], physical, distances, flags, names
    ):
        f["properties"].update(
            policy_annotations(surface, float(distance), 300.0, flag)
        )
        f["properties"]["infrastructure_name"] = name


def infer_roi(
    scene: dict[str, Any],
    roi: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    detector: XView3Detector,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = {
        "type": "FeatureCollection",
        "sar_product_id": scene["product_id"],
        "sar_product": scene["name"],
        "acquisition_time": scene["acquisition_time"],
        "acquisition_group": scene["acquisition_group"],
        "split": scene["split"],
        "study_roi": roi["geometry"],
        "features": [],
    }
    require_development_scene(scene["product_id"], metadata)
    bands = [
        bounded_path(
            root,
            next(s["path"] for s in scene["sources"] if f"-grd-{pol}-" in s["path"]),
        )
        for pol in ("vv", "vh")
    ]
    calibrators = [Calibrator.for_measurement(str(p)) for p in bands]
    if any(cal is None for cal in calibrators):
        raise ValueError("Native radiometric calibration unavailable")
    contexts = []
    returns = []
    with (
        rasterio.open(bands[0]) as vv,
        rasterio.open(bands[1]) as vh,
        Georeferencer.from_dataset(vv) as geo,
    ):
        if (vv.shape, vv.transform, vv.crs) != (vh.shape, vh.transform, vh.crs):
            raise ValueError("VV/VH raster grids differ")
        gcps1, crs1 = vv.gcps
        gcps2, crs2 = vh.gcps
        g1 = np.array([(g.row, g.col, g.x, g.y, g.z) for g in gcps1])
        g2 = np.array([(g.row, g.col, g.x, g.y, g.z) for g in gcps2])
        if (
            crs1 != crs2
            or g1.shape != g2.shape
            or not np.allclose(g1, g2, rtol=0, atol=1e-8)
        ):
            raise ValueError("VV/VH GCP grids differ")
        origins = selected_origins(vv.width, vv.height, roi["window"])
        if not origins:
            raise ValueError("No native contexts cover ROI")
        # Validate every context BEFORE any inference, including held-out geography.
        for r, c in origins:
            geometry = context_geometry(geo, r, c)
            require_context_split(geometry, scene["split"], manifest)
            contexts.append({"row": r, "col": c, "geometry": mapping(geometry)})
        for r, c in origins:
            window = Window(c, r, TILE_SIZE, TILE_SIZE)
            dn = [s.read(1, window=window) for s in (vv, vh)]
            if any(a.shape != (TILE_SIZE, TILE_SIZE) for a in dn):
                raise ValueError("Incomplete native context; padding prohibited")
            valid = (dn[0] > 0) & (dn[1] > 0)
            db = []
            for cal, a in zip(calibrators, dn):
                assert cal is not None
                db.append(cal.to_sigma0_db(a, row_offset=r, col_offset=c))
            for a in db:
                a[~valid] = NODATA_DB
            for det in detector.detect_tile(db[0], db[1]):
                rr, cc = int(round(det["row"])), int(round(det["col"]))
                if 0 <= rr < TILE_SIZE and 0 <= cc < TILE_SIZE and valid[rr, cc]:
                    returns.append(
                        {
                            **det,
                            "row": det["row"] + r,
                            "col": det["col"] + c,
                            "tile_id": (r, c),
                        }
                    )
        col, row, w, h = roi["window"]
        region = shape(roi["geometry"])
        features: list[dict[str, Any]] = []
        for det in dedupe_detections(returns):
            if not (col <= det["col"] < col + w and row <= det["row"] < row + h):
                continue
            xs, ys = geo.xy([det["row"]], [det["col"]])
            point = Point(float(xs[0]), float(ys[0]))
            if not region.covers(point):
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(point),
                    "properties": {
                        "detection_id": f"{roi['id']}_{len(features)}",
                        "confidence": float(det["score"]),
                        "source_row": float(det["row"]),
                        "source_col": float(det["col"]),
                    },
                }
            )
        metadata["features"] = features
    return metadata, contexts


def run_baseline(
    dataset: Path, artifact_root: Path, output: Path, *, device: str = "cuda"
) -> dict[str, Any]:
    manifest, _, status = load_dataset(
        dataset, verify_files=True, artifact_root=artifact_root
    )
    weights = artifact_root / "models/xview3/traced_ensemble.jit"
    if sha256_file(weights) != MODEL_SHA256:
        raise ValueError("Baseline model hash differs from pinned research model")
    artifact_root = artifact_root.resolve()
    shoreline_cache = artifact_root / "data/reference"
    versions = processing_versions(
        REPO_ROOT,
        {},
        artifact_root=artifact_root,
        shoreline_cache=shoreline_cache,
        shoreline_source="osm",
    )
    if versions["model"]["actual_sha256"] != MODEL_SHA256:
        raise ValueError("Operational model selection differs from pinned baseline")
    assets = {
        "model": versions.pop("model"),
        "shoreline": versions.pop("shoreline"),
        "verified_scene_sources": {
            s["product_id"]: s["sources"]
            for s in manifest["scenes"]
            if s["split"] != "test"
        },
        "ais_snapshots": {
            s["product_id"]: s["ais"]["snapshot"]
            for s in manifest["scenes"]
            if s["split"] != "test"
        },
    }
    if output.exists():
        raise ValueError(
            "Fresh output directory required; previous evidence cannot be overwritten"
        )
    output.mkdir(parents=True)
    cases = []
    detector = None
    model_load = None
    start = time.perf_counter()
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            continue
        snapshot = json.loads(
            bounded_path(artifact_root, scene["ais"]["snapshot"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        positions, alignment = aligned_ais(
            pd.DataFrame(snapshot["rows"]), scene["acquisition_time"]
        )
        for roi in scene["rois"]:
            reviewed = next(
                r for r in status["review_status"] if r["roi_id"] == roi["id"]
            )
            case = {
                "roi_id": roi["id"],
                "product_id": scene["product_id"],
                "native_product": scene["name"],
                "acquisition_time": scene["acquisition_time"],
                "acquisition_group": scene["acquisition_group"],
                "geographic_group": roi["group"],
                "split": scene["split"],
                "region": scene["region"],
                "regime": roi["regime"],
                "polarization": "/".join(scene["polarizations"]),
                "season": scene["season"],
                "metric_ready": reviewed["metric_ready"],
                "processing_complete": False,
                "water_area_km2": None,
                "resources": None,
                "prior_development_exposure": scene["prior_development_exposure"],
            }
            detections = {
                "type": "FeatureCollection",
                "features": [],
                "sar_product_id": scene["product_id"],
                "sar_product": scene["name"],
                "study_roi": roi["geometry"],
                "split": scene["split"],
                "acquisition_time": scene["acquisition_time"],
            }
            labels = None
            try:
                mask = LandMask.for_footprint(
                    shape(roi["geometry"]).bounds,
                    cache_dir=resolve_cache_dir(
                        str(artifact_root), str(shoreline_cache)
                    ),
                )
                case["water_area_km2"] = area_km2(
                    mask.water_geometry(shape(roi["geometry"]))
                )
                case["open_water_area_km2"] = area_km2(
                    mask.water_geometry(shape(roi["geometry"]), 300.0)
                )
                case["shoreline"] = {
                    "source": mask.source,
                    "licence": mask.licence,
                    "controls": mask.validation,
                }
                if scene["polarizations"] != ["vv", "vh"]:
                    case["state"] = "unsupported_polarization"
                    case["reason"] = (
                        "Pinned detector requires VV/VH; HH/HV is not relabelled "
                        "or omitted from coverage"
                    )
                else:
                    if detector is None:
                        load_start = time.perf_counter()
                        detector = XView3Detector(
                            str(weights), device=device, threshold=SCORE_FLOOR
                        )
                        model_load = time.perf_counter() - load_start
                    torch = detector._torch
                    cuda = detector.device.startswith("cuda")
                    if cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    with Resources() as resources:
                        detections, contexts = infer_roi(
                            scene, roi, manifest, artifact_root, detector
                        )
                        if cuda:
                            torch.cuda.synchronize()
                    case["resources"] = {
                        **resources.summary(),
                        "native_contexts": len(contexts),
                        "cuda_peak_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated()) if cuda else None
                        ),
                        "cuda_peak_reserved_bytes": (
                            int(torch.cuda.max_memory_reserved()) if cuda else None
                        ),
                    }
                    annotate(detections, mask, infrastructure=True)
                    detections["shoreline_status"] = "ok"
                    if reviewed["metric_ready"] and scene["split"] == "validation":
                        labels = validation_labels(
                            dataset, roi["id"], artifact_root=artifact_root
                        )
                        annotate(labels, mask, infrastructure=False)
                    write_json(output / f"{roi['id']}.geojson", detections)
                    case["detections_sha256"] = sha256_file(
                        output / f"{roi['id']}.geojson"
                    )
                    case["contexts"] = contexts
                    case["processing_complete"] = True
                    case["state"] = (
                        "processed_unmeasured" if labels is None else "measured"
                    )
            except Exception as exc:
                case["state"] = "failed"
                case["reason"] = f"{type(exc).__name__}: {exc}"
                labels = None
            case["scores"] = [
                score_roi(
                    detections,
                    labels,
                    threshold=t,
                    score_floor=SCORE_FLOOR,
                    water_area_km2=case["water_area_km2"],
                    processing_complete=case["processing_complete"],
                )
                for t in DEFAULT_THRESHOLDS
            ]
            case["coastal_buffer_trials"] = compare_buffers(
                detections, labels, scene["product_id"]
            )
            case["association"] = association_summary(
                detections,
                positions,
                threshold=0.15,
                processing_complete=case["processing_complete"],
            )
            case["ais_snapshot_sha256"] = scene["ais"]["snapshot"]["sha256"]
            case["ais_alignment_counts"] = alignment
            cases.append(case)
            print(f"{roi['id']}: {case['state']}", flush=True)
    report = make_report(cases, list(DEFAULT_THRESHOLDS), DEFAULT_TARGETS)
    report["provenance"] = {
        "baseline_protocol_version": 2,
        "processing_versions": versions,
        "assets": assets,
        "baseline_configuration_sha256": digest(
            {
                "score_floor": SCORE_FLOOR,
                "thresholds": DEFAULT_THRESHOLDS,
                "targets": DEFAULT_TARGETS,
                "buffer_m": 300.0,
                "match_radius_m": 100.0,
                "ais_proximity_radius_m": AIS_PROXIMITY_RADIUS_M,
                "association_base_radius_m": CORRELATION_RADIUS_M,
                "tile_size": TILE_SIZE,
            }
        ),
        "dataset_version": manifest["dataset_version"],
        "release_sha256": sha256_file(dataset / "release.json"),
        "model_sha256": MODEL_SHA256,
        "model_load_seconds": model_load,
        "device": detector.device if detector else None,
        "precision": "CUDA autocast float16; CPU float32",
        "python": platform.python_version(),
        "torch": detector._torch.__version__ if detector else None,
        "source_hashes": {
            s["product_id"]: s["sources"]
            for s in manifest["scenes"]
            if s["split"] != "test"
        },
        "git_head": versions["git_commit"],
        "wall_seconds_excluding_integrity_check": time.perf_counter() - start,
        "threshold_score_semantics": (
            "Uncalibrated objectness; strict > threshold, cached floor 0.05"
        ),
    }
    for category in ("code", "configuration"):
        for name, checksum in versions[category].items():
            if sha256_file(REPO_ROOT / name) != checksum:
                raise ValueError(
                    "Code or configuration changed during replay; "
                    "report is not reproducible"
                )
    report["limits"].extend(
        [
            "Native 2048-pixel production-aligned contexts, bounded selected-ROI "
            "inference, not full-scene runtime",
            "Context-only deduplication may differ at edges from a full-swath run; "
            "cross-split contexts are refused",
            "Processing is conditional on frozen valid-pixel checks and "
            "OSM physical-water polygons",
        ]
    )
    write_json(output / "baseline.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    run_baseline(args.dataset, args.artifact_root, args.output, device=args.device)


if __name__ == "__main__":
    main()
