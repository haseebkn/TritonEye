#!/usr/bin/env python3
"""
TritonEye Evaluation Agent

Scores a mission's detections against ground truth derived from AIS, and logs
the result as operational metrics.

Every AIS-broadcasting vessel inside the imaged swath at acquisition time was
definitely there, so the fraction the detector found is a *lower bound* on
recall — dark vessels are absent from AIS by construction, so true recall can be
higher but never lower. That makes this a conservative, always-available
regression metric: it needs no hand labelling and it recomputes on every
mission that has AIS coverage.

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
    import pandas as pd
    import rasterio
    from shapely.geometry import MultiPoint, Point

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
        gcps, _ = src.gcps
        if gcps:
            return MultiPoint([Point(g.x, g.y) for g in gcps]).convex_hull
        bounds = src.bounds
        return (
            Point(bounds.left, bounds.bottom)
            .buffer(0)
            .envelope.union(Point(bounds.right, bounds.top).buffer(0).envelope)
        )


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
    """Scores detections against AIS ground truth for one mission."""
    detections_path = payload.get("detections_geojson")
    ais_path = payload.get("ais_telemetry")
    vv_path = (payload.get("sar_bands") or {}).get("VV")
    acq = payload.get("acquisition_time", "")
    coverage = (payload.get("spatial_bounds") or {}).get("ais_coverage", "unknown")

    result: Dict[str, Any] = {
        "ais_coverage": coverage,
        "match_radius_m": MATCH_RADIUS_M,
        "scored": False,
    }

    if coverage in ("none", "mock"):
        # No real telemetry means no ground truth. Reporting 0% recall here
        # would be indistinguishable from a detector failure, so the mission is
        # explicitly marked unscored instead.
        result["reason"] = f"no real AIS ground truth (coverage={coverage})"
        return result

    if not detections_path or not ais_path or not vv_path:
        result["reason"] = "payload missing detections, AIS, or VV band"
        return result

    detections = gpd.read_file(detections_path)
    ais = pd.read_csv(ais_path)
    if ais.empty:
        result["reason"] = "AIS file is empty"
        return result

    # one position per vessel: the report closest to acquisition
    ais["timestamp"] = pd.to_datetime(ais["timestamp"])
    acq_ts = pd.Timestamp(acq)
    ais["dt"] = (ais["timestamp"] - acq_ts).abs()
    nearest = ais.sort_values("dt").groupby("mmsi", as_index=False).first()

    truth = gpd.GeoDataFrame(
        nearest,
        geometry=[Point(xy) for xy in zip(nearest.lon, nearest.lat)],
        crs="EPSG:4326",
    )

    hull = swath_hull(vv_path)
    truth = truth[truth.geometry.within(hull)].copy()
    result["ais_vessels_in_swath"] = int(len(truth))
    result["detections"] = int(len(detections))

    if truth.empty:
        result["reason"] = "no AIS vessels inside the imaged swath"
        return result

    # attach length where the raw archive is still available
    base = os.path.dirname(os.path.abspath(ais_path))
    date_tag = acq[:10] if len(acq) >= 10 else ""
    lengths = vessel_lengths(os.path.join(base, f"ais-{date_tag}"), list(truth.mmsi))
    truth["vessel_len"] = truth.mmsi.astype(int).map(lengths)

    if detections.empty:
        truth["found"] = False
    else:
        metric_crs = detections.estimate_utm_crs()
        cent = detections.to_crs(metric_crs).geometry.centroid
        truth_m = truth.to_crs(metric_crs)
        truth["found"] = [
            bool(cent.distance(g).min() <= MATCH_RADIUS_M) for g in truth_m.geometry
        ]

    result["scored"] = True
    result["matched"] = int(truth.found.sum())
    result["recall_all"] = round(float(truth.found.mean()), 4)

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
            f"Recall vs AIS ground truth: {ev['matched']}/{ev['ais_vessels_in_swath']}"
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
