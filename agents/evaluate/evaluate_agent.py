#!/usr/bin/env python3
"""
TritonEye Evaluation Agent

Scores proximity recall on the observed AIS subset, not complete ground truth.
AIS reception, position accuracy, and self-reported identity are imperfect.
This selected subset is neither a lower nor an upper bound on overall recall.
Unmatched SAR returns cannot establish precision or a false-positive rate.

Recall is stratified by vessel length where the raw AIS archive provides it,
because 10 m imagery cannot resolve small craft and a single blended number
hides which sizes the model is actually failing on.

Reads a pipeline payload on stdin (or --payload / --payload-file) and writes it
back with an `evaluation` block added.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import GCPTransformer, rowcol
    from rasterio.windows import Window
    from shapely.geometry import MultiPoint, Point, Polygon
    from shapely.ops import transform

    from agents.association import aligned_ais, one_to_one_matches, real_ais_source
    from agents.region import contains_points, load_region
    from agents.tracking import RunTracker
except ImportError as e:
    print(f"Dependency missing during startup: {e}", file=sys.stderr)
    print(
        "Please ensure your Python environment matches pyproject.toml.",
        file=sys.stderr,
    )

# A detection this close to an AIS position counts as that vessel. Wide enough
# to absorb AIS reporting slop and the vessel's motion across the matching
# window; tight enough that coincidental pairings stay unlikely outside dense
# port traffic.
MATCH_RADIUS_M = 500.0

# Sentinel-1 IW GRD has ~10 m ground spacing, so anything under about 20 m is
# a handful of pixels at best. Vessels at or above 50 m are the ones the sensor
# should comfortably resolve, and headline recall is reported on that subset.
LENGTH_BANDS = [(0.0, 20.0), (20.0, 50.0), (50.0, 100.0), (100.0, 100000.0)]
RESOLVABLE_M = 50.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TritonEye Evaluation Agent")
    parser.add_argument("--payload", type=str, help="JSON payload string")
    parser.add_argument("--payload-file", type=str, help="Path to JSON payload file")
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.payload:
        return json.loads(args.payload)  # type: ignore[no-any-return]
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())  # type: ignore[no-any-return]
    raise ValueError(
        "No input payload provided via --payload, --payload-file, or stdin."
    )


def swath_hull(vv_path: str) -> Any:
    """
    Returns the imaged footprint as a polygon, from the product's GCPs.

    The padded rectangle used to query AIS is deliberately larger than the
    scene; scoring against it would count vessels that were never imaged as
    misses.
    """
    with rasterio.open(vv_path) as src:
        gcps, gcp_crs = src.gcps
        if gcps:
            hull = MultiPoint([Point(g.x, g.y) for g in gcps]).convex_hull
            crs = gcp_crs
        else:
            hull = Polygon(
                [
                    src.transform * xy
                    for xy in [
                        (0, 0),
                        (src.width, 0),
                        (src.width, src.height),
                        (0, src.height),
                    ]
                ]
            )
            crs = src.crs
        if crs is None:
            raise ValueError("SAR raster has no georeferencing CRS")
        project = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        return transform(project.transform, hull)


def valid_sar_points(vv_path: str, points: list[Point]) -> list[bool]:
    """Test actual SAR pixels, excluding mask/nodata, NaN, and zero DN padding.

    Georeferencing uses the GCP CRS and TPS inverse for unwarped Sentinel GRD,
    or the affine transform for projected products. Read one pixel at a time so
    full scenes and their validity masks never need to fit in memory.
    """
    if not points:
        return []
    with rasterio.open(vv_path) as src:
        gcps, gcp_crs = src.gcps
        crs = gcp_crs if gcps else src.crs
        if crs is None:
            raise ValueError("SAR raster has no georeferencing CRS")
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
            pixel = src.read(1, window=Window(int(col), int(row), 1, 1), masked=True)
            valid.append(
                bool(
                    not bool(np.asarray(pixel.mask).any())
                    and np.isfinite(pixel[0, 0])
                    and pixel[0, 0] != 0
                )
            )
        return valid


def vessel_lengths(ais_dir_hint: str, mmsis: List[int]) -> Dict[int, float]:
    """
    Pulls self-reported vessel length from a raw AIS daily archive.

    The filtered per-mission CSV keeps only kinematics, so length has to come
    from the raw file when it is still on disk. Absent that, recall is reported
    unstratified rather than not at all.
    """
    if not ais_dir_hint or not os.path.isdir(ais_dir_hint):
        return {}
    files = [
        os.path.join(ais_dir_hint, f)
        for f in os.listdir(ais_dir_hint)
        if os.path.isfile(os.path.join(ais_dir_hint, f))
    ]
    if not files:
        return {}
    wanted = set(int(m) for m in mmsis)
    lengths: Dict[int, float] = {}
    try:
        for chunk in pd.read_csv(
            files[0], usecols=["mmsi", "length"], chunksize=500_000
        ):
            hit = chunk[chunk.mmsi.isin(wanted) & chunk.length.notna()]
            for m, ln in zip(hit.mmsi, hit.length):
                if ln and ln > 0:
                    lengths[int(m)] = max(lengths.get(int(m), 0.0), float(ln))
    except Exception as e:
        print(f"Could not read vessel lengths ({e}).", file=sys.stderr)
    return lengths


def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Score one-to-one proximity recall on the observed, time-aligned AIS subset."""
    detections_path = payload.get("detections_geojson")
    ais_path = payload.get("ais_telemetry")
    vv_path = (payload.get("sar_bands") or {}).get("VV")
    acq = payload.get("acquisition_time", "")
    coverage = (payload.get("spatial_bounds") or {}).get("ais_coverage", "unknown")

    result: Dict[str, Any] = {
        "ais_coverage": coverage,
        "match_radius_m": MATCH_RADIUS_M,
        "scored": False,
        "metric_scope": "observed AIS subset proximity recall",
        "precision": None,
        "false_positive_rate": None,
        "limitations": (
            "AIS is incomplete and self-reported; subset recall is neither a lower "
            "nor upper bound on overall recall. Precision and false-alarm rate "
            "require independent labelled SAR targets, including ice and clutter."
        ),
    }

    if not real_ais_source(coverage):
        result["reason"] = f"no real AIS observations (coverage={coverage})"
        return result

    processing = payload.get("processing") or payload.get("processing_manifest") or {}
    if processing.get("complete") is False or processing.get("detector") == "mock":
        result["reason"] = "incomplete processing or synthetic detector output"
        return result

    if not detections_path or not ais_path or not vv_path:
        result["reason"] = "payload missing detections, AIS, or VV band"
        return result

    detections = gpd.read_file(detections_path)
    try:
        ais = pd.read_csv(ais_path, dtype={"mmsi": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        result["reason"] = f"AIS unavailable: {error}"
        return result
    if ais.empty:
        result["reason"] = "AIS file is empty"
        return result

    truth, diagnostics = aligned_ais(ais, acq)
    result["ais_alignment"] = diagnostics
    if truth.empty:
        result["reason"] = diagnostics.get("reason", "no usable AIS positions")
        return result
    region = load_region((payload.get("spatial_bounds") or {}).get("analysis_region"))
    truth = truth.loc[
        contains_points(list(truth.lon), list(truth.lat), region=region)
    ].copy()
    try:
        truth = truth.loc[valid_sar_points(vv_path, list(truth.geometry))].copy()
    except (ValueError, rasterio.errors.RasterioError) as error:
        result["reason"] = f"SAR validity/georeferencing unavailable: {error}"
        return result
    truth = truth.reset_index(drop=True)
    result["ais_vessels_in_swath"] = int(len(truth))
    result["detections"] = int(len(detections))

    if truth.empty:
        result["reason"] = "no AIS vessels on valid SAR pixels inside the NL study area"
        return result

    if detections.crs is None:
        result["reason"] = "detections have no CRS"
        return result
    detections = detections.to_crs("EPSG:4326")
    detections = detections.loc[
        detections.geometry.notna()
        & ~detections.geometry.is_empty
        & detections.geometry.is_valid
    ].copy()
    detections = detections.loc[
        [region.covers(geometry) for geometry in detections.geometry]
    ].copy()
    detections = detections.loc[
        valid_sar_points(
            vv_path, [geometry.centroid for geometry in detections.geometry]
        )
    ].reset_index(drop=True)

    # attach length where the raw archive is still available
    base = os.path.dirname(os.path.abspath(ais_path))
    date_tag = acq[:10] if len(acq) >= 10 else ""
    lengths = vessel_lengths(
        os.path.join(base, f"ais-{date_tag}"), [int(m) for m in truth.mmsi]
    )
    truth["vessel_len"] = truth["length"].where(truth["length"] > 0)
    truth["vessel_len"] = truth["vessel_len"].fillna(
        truth.mmsi.astype(int).map(lengths)
    )
    raw_pairs, _, raw_ambiguous = one_to_one_matches(
        [geometry.centroid for geometry in detections.geometry],
        list(truth.geometry),
        MATCH_RADIUS_M,
    )
    surfaces = detections.get("surface", pd.Series("unknown", index=detections.index))
    eligible = detections.loc[surfaces == "water"]
    eligible_pairs, _, eligible_ambiguous = one_to_one_matches(
        [geometry.centroid for geometry in eligible.geometry],
        list(truth.geometry),
        MATCH_RADIUS_M,
    )
    found = {position for position, _ in raw_pairs.values()}
    eligible_found = {position for position, _ in eligible_pairs.values()}
    truth["found"] = [index in found for index in range(len(truth))]
    result["raw_detections_in_valid_region"] = len(detections)
    result["eligible_detections"] = len(eligible)
    result["matched_eligible"] = len(eligible_found)
    result["recall_eligible"] = round(len(eligible_found) / len(truth), 4)
    result["raw_ambiguous_targets"] = len(raw_ambiguous)
    result["eligible_ambiguous_targets"] = len(eligible_ambiguous)
    result["eligibility_scope"] = "known open-water detections; same AIS denominator"

    result["scored"] = True
    result["matched"] = int(truth.found.sum())
    result["recall_all"] = round(float(truth.found.mean()), 4)
    result["recall_raw"] = result["recall_all"]

    known = truth[truth["vessel_len"].notna()]
    if not known.empty:
        bands = {}
        for lo, hi in LENGTH_BANDS:
            band = known[(known["vessel_len"] >= lo) & (known["vessel_len"] < hi)]
            if len(band):
                label = f"{int(lo)}_{int(hi)}m" if hi < 100000 else f"{int(lo)}m_plus"
                bands[label] = {
                    "in_swath": int(len(band)),
                    "matched": int(band.found.sum()),
                    "recall": round(float(band.found.mean()), 4),
                }
        result["by_length"] = bands

        resolvable = known[known["vessel_len"] >= RESOLVABLE_M]
        if len(resolvable):
            result["resolvable_in_swath"] = int(len(resolvable))
            result["resolvable_matched"] = int(resolvable.found.sum())
            result["recall_resolvable"] = round(float(resolvable.found.mean()), 4)
    else:
        result["by_length"] = {}
        result["note"] = "vessel lengths unavailable; recall not stratified"

    return result


def flatten_metrics(ev: Dict[str, Any]) -> Dict[str, float]:
    """Turns the evaluation block into flat numeric metrics for tracking."""
    out: Dict[str, float] = {}
    for key in (
        "ais_vessels_in_swath",
        "detections",
        "matched",
        "recall_all",
        "resolvable_in_swath",
        "resolvable_matched",
        "recall_resolvable",
        "recall_raw",
        "recall_eligible",
        "matched_eligible",
        "raw_detections_in_valid_region",
        "eligible_detections",
        "raw_ambiguous_targets",
        "eligible_ambiguous_targets",
    ):
        if isinstance(ev.get(key), (int, float)):
            out[f"eval.{key}"] = float(ev[key])
    for band, stats in (ev.get("by_length") or {}).items():
        out[f"eval.recall_{band}"] = float(stats["recall"])
        out[f"eval.count_{band}"] = float(stats["in_swath"])
    return out


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    args = parse_arguments()
    payload = get_payload(args)

    ev = evaluate(payload)
    payload["evaluation"] = ev

    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        tracker.set_tags({"eval.scored": ev.get("scored", False)})
        tracker.log_metrics(flatten_metrics(ev))
    finally:
        tracker.end()

    if ev.get("scored"):
        print(
            f"Observed AIS-subset recall: {ev['matched']}/{ev['ais_vessels_in_swath']}"
            f" = {ev['recall_all']:.1%}",
            file=sys.stderr,
        )
        if "recall_resolvable" in ev:
            print(
                f"  on vessels >= {RESOLVABLE_M:.0f} m: "
                f"{ev['resolvable_matched']}/{ev['resolvable_in_swath']}"
                f" = {ev['recall_resolvable']:.1%}",
                file=sys.stderr,
            )
    else:
        print(f"Not scored: {ev.get('reason', 'unknown')}", file=sys.stderr)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
