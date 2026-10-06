"""Replay physical shoreline and buffer policy on cached NL detections.

This never re-runs inference, overwrites old missions or invents vessel labels.
Source and derived outputs remain distinct. Use --manifest for explicit scope.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from agents.acquisition import native_acquisition_group, require_product
from agents.artifacts import sha256_file
from agents.coastal_benchmark import compare_buffers
from agents.coastal_policy import annotations
from agents.correlation.correlation_agent import (
    correlate_targets,
    summarize_correlation,
)
from agents.evaluate.evaluate_agent import evaluate, swath_hull, valid_sar_points
from agents.inference.inference_agent import classify_surfaces, load_yaml_config
from agents.nl_benchmark import require_development_scene
from agents.report.report_agent import build_html_report
from agents.run_versions import digest


def replay(root: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = []
    for scene in manifest["scenes"]:
        source_payload = root / scene["payload"]
        payload = json.loads(source_payload.read_text(encoding="utf-8"))
        require_product(payload, scene["product_id"])
        detection_path = Path(payload["detections_geojson"])
        detections = json.loads(detection_path.read_text(encoding="utf-8"))
        require_development_scene(scene["product_id"], scene, payload, detections)
        if scene.get("independent_labels") and scene["split"] != "validation":
            raise ValueError("Labelled buffer trials require validation scene split")
        inputs.append((scene, source_payload, payload, detection_path, detections))
    config = load_yaml_config(str(root / "configs/model.yaml"))
    registry_path = root / manifest["controls_registry"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    results = []
    for scene, source_payload, payload, detection_path, detections in inputs:
        detections["sar_product_id"] = scene["product_id"]
        detections["acquisition_time"] = payload["acquisition_time"]
        native_name = (
            payload.get("sar_product")
            or detections.get("sar_product")
            or scene.get("name")
        )
        if native_name:
            detections["sar_product"] = native_name
            detections["acquisition_group"] = native_acquisition_group(native_name)
        from shapely.geometry import shape

        centres = [shape(f["geometry"]).centroid for f in detections["features"]]
        vv = payload["sar_bands"]["VV"]
        footprint = swath_hull(vv)
        surfaces, distances, metadata = classify_surfaces(
            [p.x for p in centres],
            [p.y for p in centres],
            footprint.bounds,
            str(root),
            config,
        )
        physical = metadata.pop("physical_surfaces", [])
        detections["shoreline_status"] = metadata.get("status", "unavailable")
        infrastructure = metadata.pop("infrastructure_flags", [])
        for i, feature in enumerate(detections["features"]):
            # Drop stale policy fields before constructing the new annotations.
            feature["properties"].update(
                annotations(
                    physical[i] if surfaces is not None else "unknown",
                    float(distances[i]) if surfaces is not None else None,
                    float(config["landmask"]["coastal_buffer_m"]),
                    bool(infrastructure[i]) if surfaces is not None else False,
                )
            )
        valid = valid_sar_points(vv, centres)
        detections["features"] = [
            f for f, keep in zip(detections["features"], valid) if keep
        ]
        directory = output_dir / scene["product_id"]
        directory.mkdir(parents=True, exist_ok=True)
        derived_detections = directory / "detections.geojson"
        derived_detections.write_text(
            json.dumps(detections, indent=2, allow_nan=False), encoding="utf-8"
        )
        coverage = []
        for collection in registry["collections"]:
            controls_path = registry_path.parent / collection["path"]
            controls = json.loads(controls_path.read_text(encoding="utf-8"))
            checks = controls["features"]
            imaged = valid_sar_points(vv, [shape(f["geometry"]) for f in checks])
            coverage.append(
                {
                    "location": collection["location"],
                    "controls_total": len(checks),
                    "controls_on_valid_scene_pixels": sum(imaged),
                    "controls_sha256": sha256_file(controls_path),
                    "reference": controls["reference"],
                    "scope": (
                        "Sparse physical shoreline controls, "
                        "not accuracy over the whole swath"
                    ),
                }
            )
        required = set(scene["locations_to_check"])
        checked_locations = {
            c["location"] for c in coverage if c["controls_on_valid_scene_pixels"] > 0
        }
        if not required <= checked_locations:
            raise ValueError(
                "Inventory claims locations with no controls on valid scene pixels"
            )
        labels_path = (
            root / scene["independent_labels"]
            if scene.get("independent_labels")
            else None
        )
        labels = (
            json.loads(labels_path.read_text(encoding="utf-8")) if labels_path else None
        )
        if labels is not None:
            label_centres = [shape(f["geometry"]).centroid for f in labels["features"]]
            if not all(valid_sar_points(vv, label_centres)):
                raise ValueError("Benchmark labels outside valid SAR pixels")
            label_surfaces, label_distances, label_metadata = classify_surfaces(
                [p.x for p in label_centres],
                [p.y for p in label_centres],
                footprint.bounds,
                str(root),
                config,
            )
            for i, feature in enumerate(labels["features"]):
                feature["properties"].update(
                    annotations(
                        (
                            label_metadata["physical_surfaces"][i]
                            if label_surfaces is not None
                            else "unknown"
                        ),
                        (
                            float(label_distances[i])
                            if label_surfaces is not None
                            else None
                        ),
                        float(config["landmask"]["coastal_buffer_m"]),
                    )
                )
        benchmark = compare_buffers(
            detections,
            labels,
            scene["product_id"],
            buffers_m=manifest["buffers_m"],
        )
        payload.update(
            {"detections_geojson": str(derived_detections), "landmask": metadata}
        )
        evaluation = evaluate(payload)
        targets = correlate_targets(
            str(derived_detections),
            payload["ais_telemetry"],
            payload["acquisition_time"],
            ais_coverage=payload["spatial_bounds"].get("ais_coverage", "unknown"),
            analysis_region=payload["spatial_bounds"].get("analysis_region"),
        )
        targets.to_file(directory / "correlation.geojson", driver="GeoJSON")
        targets.loc[targets.coastal_review_required.astype(bool)].to_file(
            directory / "coastal_research.geojson", driver="GeoJSON"
        )
        targets.loc[targets.association_eligible.astype(bool)].to_file(
            directory / "open_water_research.geojson", driver="GeoJSON"
        )
        payload["evaluation"] = evaluation
        payload["correlation_summary"] = summarize_correlation(targets)
        correlations = json.loads((directory / "correlation.geojson").read_text())
        (directory / "report.html").write_text(
            build_html_report(payload, detections, correlations), encoding="utf-8"
        )
        result = {
            "product_id": scene["product_id"],
            "acquisition_time": payload["acquisition_time"],
            "split": scene["split"],
            "source_payload_sha256": sha256_file(source_payload),
            "source_detections_sha256": sha256_file(detection_path),
            "derived_detections_sha256": sha256_file(derived_detections),
            "source_labels_sha256": sha256_file(labels_path) if labels_path else None,
            "valid_pixel_returns": len(detections["features"]),
            "outside_valid_pixels": sum(not keep for keep in valid),
            "landmask": metadata,
            "regional_coverage": coverage,
            "buffer_trials": benchmark,
            "ais_evaluation": evaluation,
            "correlation_summary": payload["correlation_summary"],
        }
        (directory / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        results.append(result)
    output = {
        "inventory_sha256": sha256_file(manifest_path),
        "configuration_sha256": digest(config),
        "code_sha256": {
            p.name: sha256_file(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("coastal_policy.py"),
                Path(__file__).with_name("coastal_benchmark.py"),
                Path(__file__).with_name("landmask.py"),
            )
        },
        "status": (
            "measured_validation"
            if all(r["buffer_trials"]["trials"][0]["measured"] for r in results)
            and results
            else "prebenchmark_unmeasured"
        ),
        "scenes": results,
        "not_covered": manifest["not_covered"],
        "limitation": (
            "No province-wide validation; no buffer selected "
            "or promoted from unlabelled returns"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "configs/benchmarks/nl_coastal_inventory.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = replay(root, args.manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": output["status"],
                "products": [s["product_id"] for s in output["scenes"]],
                "summary": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
