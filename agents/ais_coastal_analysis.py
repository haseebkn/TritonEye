"""Replay coastal buffers on the observed AIS subset and cached SAR returns.

This measures policy exclusion sensitivity and proximity matches. AIS reports
are supporting evidence; they cannot establish image truth or false alarms.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from shapely.geometry import Point, shape

from agents.acquisition import native_acquisition_group, require_product
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.association import aligned_ais, one_to_one_matches
from agents.coastal_policy import annotations
from agents.evaluate.evaluate_agent import valid_sar_points
from agents.inference.inference_agent import classify_surfaces, load_yaml_config
from agents.nl_benchmark import require_development_scene
from agents.overlap_audit import journal_support, observations
from agents.region import load_region
from agents.run_versions import digest


def compare(
    targets: list[Point],
    target_properties: list[dict[str, Any]],
    positions: list[Point],
    position_properties: list[dict[str, Any]],
    buffers_m: Sequence[float],
    radius_m: float = 500.0,
    detections_available: bool = True,
) -> dict[str, Any]:
    """Keep all observed positions as competitors and a fixed denominator.

    Inputs are already aligned once per MMSI, inside NL, the exact footprint,
    and valid pixels in both native DN bands. Recompute assignment after each
    target-policy change; filtering AIS competitors would inflate matches.
    """
    if len(targets) != len(target_properties) or len(positions) != len(
        position_properties
    ):
        raise ValueError("Points and physical annotations must have equal lengths")
    if not buffers_m or any(not math.isfinite(b) or b < 0 for b in buffers_m):
        raise ValueError("Finite nonnegative buffers required")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Positive finite match radius required")

    def policy(properties: list[dict[str, Any]], buffer: float) -> list[dict[str, Any]]:
        result = []
        for p in properties:
            distance = p.get("distance_to_shore_m")
            if p.get("shore_distance_unbounded") is True:
                distance = math.inf
            if p.get("physical_surface") == "water" and (
                distance is not None and (math.isnan(distance) or distance < 0)
            ):
                raise ValueError("Contradictory physical water / shore distance")
            result.append(
                annotations(
                    p.get("physical_surface", "unknown"),
                    distance,
                    buffer,
                    p.get("infrastructure_proximity") is True,
                )
            )
        return result

    raw, _, ambiguous = one_to_one_matches(targets, positions, radius_m)
    raw_found = {j for j, _ in raw.values()}
    trials = []
    for buffer in dict.fromkeys(buffers_m):
        tp, ap = policy(target_properties, buffer), policy(position_properties, buffer)
        active = [i for i, p in enumerate(tp) if p["alert_eligible"]]
        pairs, _, contested = one_to_one_matches(
            [targets[i] for i in active], positions, radius_m
        )
        found = {j for j, _ in pairs.values()}
        coastal = {i for i, p in enumerate(ap) if p["coastal_zone"]}
        strata = {}
        for regime in ("coastal_water", "open_water", "land_mismatch", "unknown"):
            members = {
                i
                for i, p in enumerate(ap)
                if (
                    "land_mismatch"
                    if p["physical_surface"] == "land"
                    else (
                        "unknown"
                        if p["physical_surface"] == "unknown"
                        or p["alert_eligibility_reason"] == "shore_distance_unknown"
                        else "coastal_water" if p["coastal_zone"] else "open_water"
                    )
                )
                == regime
            }
            strata[regime] = {
                "observed_ais": len(members),
                "matched_raw": len(members & raw_found),
                "matched_eligible": len(members & found),
                "unmatched_raw": len(members - raw_found),
                "unmatched_eligible": len(members - found),
            }
        trials.append(
            {
                "buffer_m": buffer,
                "observed_ais_denominator": len(positions),
                "coastal_ais_withheld": len(coastal),
                "coastal_ais_with_raw_match": len(coastal & raw_found),
                "ais_policy_eligible": sum(p["alert_eligible"] for p in ap),
                "eligible_returns": len(active),
                "coastal_returns_retained": sum(p["coastal_zone"] for p in tp),
                "matched_eligible": len(found),
                "raw_matched_without_eligible_match": len(raw_found - found),
                "eligible_matched_without_raw_match": len(found - raw_found),
                "eligible_ambiguous_returns": len(contested),
                "ais_subset_proximity_recall_eligible": (
                    len(found) / len(positions) if positions else None
                ),
                "strata": strata,
                "false_alarms": None,
                "total_missed_vessels": None,
            }
        )
    result = {
        "status": "observed_ais_subset_only" if positions else "no_observed_overlap",
        "observed_ais_denominator": len(positions),
        "retained_returns": len(targets),
        "matched_raw": len(raw_found),
        "raw_ambiguous_returns": len(ambiguous),
        "ais_subset_proximity_recall_raw": (
            len(raw_found) / len(positions) if positions else None
        ),
        "match_radius_m": radius_m,
        "trials": trials,
    }
    if not detections_available:
        result["status"] = "ais_policy_only_no_detector_output"
        for key in (
            "retained_returns",
            "matched_raw",
            "raw_ambiguous_returns",
            "ais_subset_proximity_recall_raw",
        ):
            result[key] = None
        for row in trials:
            for key in (
                "coastal_ais_with_raw_match",
                "eligible_returns",
                "coastal_returns_retained",
                "matched_eligible",
                "raw_matched_without_eligible_match",
                "eligible_matched_without_raw_match",
                "eligible_ambiguous_returns",
                "ais_subset_proximity_recall_eligible",
            ):
                row[key] = None
            for stratum in row["strata"].values():
                for key in (
                    "matched_raw",
                    "matched_eligible",
                    "unmatched_raw",
                    "unmatched_eligible",
                ):
                    stratum[key] = None
    return result


def run(root: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh output required; preserve previous evidence")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = root / manifest["archive_bundle"]
    snapshot_path = bundle / "snapshot_manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    catalogue_path = bundle / "catalogue_products.json"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    config = load_yaml_config(str(root / "configs/model.yaml"))
    region = load_region()
    selected = []
    slices = set()
    # Guard every product and its geographic footprint before opening SAR data.
    for scene in manifest["scenes"]:
        products = [p for p in catalogue if p["Id"] == scene["product_id"]]
        if len(products) != 1:
            raise ValueError("Exact catalogue product required")
        product = products[0]
        payload_path = root / scene["payload"] if scene.get("payload") else None
        if payload_path is None:
            if scene.get("ais_only") is not True:
                raise ValueError("Payload or explicit AIS-only scope required")
            payload = {
                "sar_product_id": scene["product_id"],
                "sar_product": product["Name"],
                "acquisition_time": product["ContentDate"]["Start"],
                "mode": "ais_geometry_only",
                "processing": {},
            }
        else:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        require_product(payload, scene["product_id"])
        if product["Name"] != payload["sar_product"]:
            raise ValueError("Exact catalogue product/name required")
        footprint = shape(product["GeoFootprint"])
        if (
            footprint.is_empty
            or not footprint.is_valid
            or footprint.geom_type not in {"Polygon", "MultiPolygon"}
        ):
            raise ValueError("Valid exact catalogue footprint required")
        require_development_scene(
            scene["product_id"],
            scene,
            payload,
            {"name": product["Name"], "scene_footprint": product["GeoFootprint"]},
        )
        require_development_scene(
            scene["product_id"],
            {"name": product["Name"], "scene_footprint": product["GeoFootprint"]},
        )
        acq, start = parse_utc(payload["acquisition_time"]), parse_utc(
            product["ContentDate"]["Start"]
        )
        if (
            acq is None
            or start is None
            or acq.replace(microsecond=0) != start.replace(microsecond=0)
        ):
            raise ValueError("Payload must use the catalogue scene start second")
        slice_key = product["Name"].split("_")[:8]
        slice_id = "_".join(slice_key)
        if slice_id in slices:
            raise ValueError("Duplicate physical slice, including COG aliases")
        slices.add(slice_id)
        processing = payload.get("processing") or {}
        if payload_path is None:
            bands = {}
            for pol in ("VV", "VH"):
                paths = list(
                    (root / "data/raw" / product["Name"] / "measurement").glob(
                        f"*-{pol.lower()}-*.tiff"
                    )
                )
                if len(paths) != 1:
                    raise ValueError("Exact cached VV/VH measurements required")
                bands[pol] = str(paths[0])
            payload["sar_bands"] = bands
        elif (
            payload.get("mode") != "production"
            or processing.get("complete") is not True
        ):
            raise ValueError("Complete production inference required")
        if payload_path is not None and (
            processing.get("detector") != "xview3"
            or set(payload["sar_bands"])
            != {
                "VV",
                "VH",
            }
        ):
            raise ValueError("This study requires compatible xView3 VV/VH output")
        selected.append((scene, payload_path, payload, product, footprint, acq))

    archive = bundle / "archive_snapshot"
    consumed = {
        path.name
        for pattern in ("ais_stream_*.csv", "coverage_*.jsonl")
        for path in archive.glob(pattern)
    }
    if consumed - {record["file"] for record in snapshot["files"]}:
        raise ValueError("Unlisted frozen archive input")
    for record in snapshot["files"]:
        path = archive / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Frozen archive hash mismatch")
    reports, archive_stats = observations(archive)
    results = []
    shoreline_hashes: dict[str, str] = {}
    for scene, payload_path, payload, product, footprint, acq in selected:
        # Full histories first; select geography only after bounded motion alignment.
        records = [
            r["values"] for r in reports if abs(r["time"] - acq.timestamp()) <= 300
        ]
        ais, alignment = aligned_ais(pd.DataFrame(records), payload["acquisition_time"])
        points = [p for p in ais.geometry if region.covers(p) and footprint.covers(p)]
        detection_path = Path(payload["detections_geojson"]) if payload_path else None
        detections = (
            json.loads(detection_path.read_text(encoding="utf-8"))
            if detection_path
            else {"features": []}
        )
        for record in (detections,):
            if "sar_product_id" in record:
                require_product(record, scene["product_id"])
            if record.get("sar_product") not in (None, payload["sar_product"]):
                raise ValueError("Detection native identity mismatch")
            if (
                "acquisition_time" in record
                and parse_utc(record["acquisition_time"]) != acq
            ):
                raise ValueError("Detection acquisition time mismatch")
        targets = [
            shape(f["geometry"]).centroid
            for f in detections["features"]
            if region.covers(shape(f["geometry"]))
            and footprint.covers(shape(f["geometry"]).centroid)
        ]
        all_points = points + targets
        valid = [True] * len(all_points)
        band_hashes = {}
        for pol, band in payload["sar_bands"].items():
            band_hashes[pol] = sha256_file(band)
            if (
                payload_path
                and band_hashes[pol] != payload["processing"]["input_sha256"][pol]
            ):
                raise ValueError("Inference input band hash mismatch")
            valid = [a and b for a, b in zip(valid, valid_sar_points(band, all_points))]
        kept_ais = [p for p, keep in zip(points, valid[: len(points)]) if keep]
        kept_targets = [p for p, keep in zip(targets, valid[len(points) :]) if keep]
        combined = kept_ais + kept_targets
        surfaces, distances, metadata = classify_surfaces(
            [p.x for p in combined],
            [p.y for p in combined],
            footprint.bounds,
            str(root),
            config,
        )
        properties = [
            annotations(
                metadata["physical_surfaces"][i] if surfaces is not None else "unknown",
                float(distances[i]) if surfaces is not None else None,
                300,
                (
                    bool(metadata["infrastructure_flags"][i])
                    if surfaces is not None
                    else False
                ),
            )
            for i in range(len(combined))
        ]
        physical = metadata.pop("physical_surfaces", None)
        metadata.pop("infrastructure_flags", None)
        # Bind the physical reference, not just the URL or candidate buffer.
        reference_hash = None
        if physical is not None:
            from agents.landmask import LandMask, resolve_cache_dir

            mask = LandMask.for_footprint(
                footprint.bounds,
                source=config["landmask"].get("source", "osm"),
                cache_dir=resolve_cache_dir(
                    str(root), config["landmask"].get("cache_dir")
                ),
            )
            if mask.shapefile:
                source_path = Path(mask.shapefile)
                for suffix in (".shp", ".shx", ".dbf", ".prj"):
                    component = source_path.with_suffix(suffix)
                    if component.is_file() and component.name not in shoreline_hashes:
                        shoreline_hashes[component.name] = sha256_file(component)
                reference_hash = digest(shoreline_hashes)
        results.append(
            {
                "product_id": scene["product_id"],
                "product_name": product["Name"],
                "acquisition_group": native_acquisition_group(product["Name"]),
                "acquisition_time": payload["acquisition_time"],
                "catalogue_interval": product["ContentDate"],
                "setting": scene["setting"],
                "polarization": "VV/VH",
                "payload_sha256": sha256_file(payload_path) if payload_path else None,
                "detections_sha256": (
                    sha256_file(detection_path) if detection_path else None
                ),
                "detections_available": detection_path is not None,
                "input_band_sha256": band_hashes,
                "detector": {
                    k: payload["processing"].get(k)
                    for k in (
                        "detector",
                        "weights_sha256",
                        "threshold",
                        "config_sha256",
                        "complete",
                    )
                },
                "alignment": alignment,
                "archive_recording": journal_support(
                    bundle / "archive_snapshot", acq.timestamp()
                ),
                "ais_inside_nl_and_catalogue": len(points),
                "ais_outside_joint_valid_pixels": len(points) - len(kept_ais),
                "returns_outside_scope_or_valid_pixels": (
                    (len(detections["features"]) - len(kept_targets))
                    if detection_path
                    else None
                ),
                "landmask": metadata,
                "physical_reference_sha256": reference_hash,
                "comparison": compare(
                    kept_targets,
                    properties[len(kept_ais) :],
                    kept_ais,
                    properties[: len(kept_ais)],
                    manifest["buffers_m"],
                    detections_available=detection_path is not None,
                ),
            }
        )
    report = {
        "status": "supporting_analysis_only",
        "selected_buffer_m": None,
        "step_2_complete": False,
        "manifest_sha256": sha256_file(manifest_path),
        "snapshot_manifest_sha256": sha256_file(snapshot_path),
        "catalogue_sha256": sha256_file(catalogue_path),
        "archive": archive_stats,
        "configuration_sha256": digest(config),
        "shoreline_source_files_sha256": shoreline_hashes,
        "method_sha256": sha256_file(Path(__file__)),
        "dependencies_sha256": {
            name: sha256_file(REPO_ROOT / name)
            for name in (
                "agents/association.py",
                "agents/coastal_policy.py",
                "agents/evaluate/evaluate_agent.py",
                "agents/landmask.py",
                "agents/inference/inference_agent.py",
                "agents/overlap_audit.py",
            )
        },
        "scenes": results,
        "limitations": [
            "AIS-selected convenience sample; incomplete self-reported reception.",
            "Scene start to whole seconds; per-pixel radar times are unavailable.",
            "500 m one-to-one proximity is not reviewed identity correctness.",
            "A withheld AIS position is not an established SAR-visible missed vessel.",
            "Reference land may indicate berths, shoreline or alignment error.",
            "Sparse shoreline controls do not validate the complete imaged coastline.",
            "Adjacent slices share acquisitions; rows are not independent trials.",
            "No independent labels, false-alarm measurement or operating-point choice.",
        ],
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "configs/benchmarks/ais_coastal_analysis.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(REPO_ROOT, args.manifest, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "products": len(result["scenes"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
