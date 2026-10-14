"""Audit sealed source associations without changing their labels or allowances."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Geod
from rasterio.transform import GCPTransformer
from scipy.interpolate import LinearNDInterpolator

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.source_annotations import replay_prepared, reservation
from agents.source_reservation import checked, ref
from agents.source_windows import metadata

GEOD = Geod(ellps="WGS84")


def literal_record(source: dict[str, Any], directory: Path) -> dict[str, str]:
    """Recover supplied CSV text, including zeros lost by float normalization."""
    path = directory / source["source_file"]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for original in reader:
            if reader.line_num == source["source_line"]:
                row = {"_".join(k.split()).upper(): v for k, v in original.items()}
                if (
                    float(row["SIGHTING_LATITUDE"]) != source["lat"]
                    or float(row["SIGHTING_LONGITUDE"]) != source["lon"]
                ):
                    raise ValueError("CSV sighting differs from sealed source")
                return row
    raise ValueError("Original CSV record unavailable")


def precision_sensitivity(
    latitude: str, longitude: str, target: tuple[float, float], offset_s: float
) -> dict[str, Any]:
    """Decimal representation scenarios; these are NOT source accuracy bounds."""
    lat, lon = float(latitude), float(longitude)
    steps = [
        float(Decimal(10) ** Decimal(v.as_tuple().exponent))
        for v in (Decimal(latitude), Decimal(longitude))
    ]
    if not all(np.isfinite(steps)) or max(steps) > 1 or min(steps) <= 0:
        raise ValueError("Unsupported coordinate representation")
    results: dict[str, Any] = {
        "latitude_step_deg": steps[0],
        "longitude_step_deg": steps[1],
    }
    for name, factor in (("nearest_rounding", 0.5), ("symmetric_one_step", 1.0)):
        dy, dx = steps[0] * factor, steps[1] * factor
        corner = max(
            GEOD.inv(lon, lat, lon + x, lat + y)[2]
            for x in (-dx, dx)
            for y in (-dy, dy)
        )
        # A clamped lon/lat point is a diagnostic approximation to nearest cell
        # distance, not a geodesic optimization or association probability.
        near_lon = float(np.clip(target[0], lon - dx, lon + dx))
        near_lat = float(np.clip(target[1], lat - dy, lat + dy))
        results[name] = {
            "center_to_corner_m": corner,
            "distance_to_clamped_cell_point_m": GEOD.inv(
                target[0], target[1], near_lon, near_lat
            )[2],
        }
    results["minute_time_sensitivity_s"] = {
        "nearest_rounding_max_absolute_offset": abs(offset_s) + 30,
        "unknown_round_or_truncate_max_absolute_offset": abs(offset_s) + 60,
    }
    return results


def grid_consistency(gcps: list[Any]) -> dict[str, Any]:
    """Withheld annotation-grid points test interpolation, not absolute accuracy."""
    rows = sorted({g.row for g in gcps})
    cols = sorted({g.col for g in gcps})
    interior = [g for g in gcps if g.row in rows[1:-1] and g.col in cols[1:-1]]
    selected = interior[:: max(1, len(interior) // 9)][:9]
    residuals = []
    for hold in selected:
        with GCPTransformer([g for g in gcps if g is not hold], tps=True) as tx:
            x, y = tx.xy([hold.row], [hold.col], offset="ul")
        residuals.append(GEOD.inv(hold.x, hold.y, x[0], y[0])[2])
    with GCPTransformer(gcps, tps=True) as tx:
        rr, cc = tx.rowcol(
            [g.x for g in selected], [g.y for g in selected], op=lambda v: v
        )
    inverse = [
        float(np.hypot(r - g.row, c - g.col))
        for r, c, g in zip(rr, cc, selected, strict=True)
    ]
    return {
        "withheld_points": len(selected),
        "withheld_grid_max_m": max(residuals, default=None),
        "inverse_max_pixels": max(inverse, default=None),
        "independent_ground_control": False,
    }


def audit(prepared: Path, evidence_path: Path, iip: Path, output: Path) -> None:
    if output.exists() or not output.resolve().is_relative_to(REPO_ROOT / "data"):
        raise ValueError("Fresh private audit directory required")
    rows = replay_prepared(prepared)
    evidence = json.loads(evidence_path.read_text())
    if evidence["prepared"] != ref(prepared) or evidence["scored"] is not False:
        raise ValueError("Source evidence preparation mismatch")
    manifest = json.loads(prepared.read_text())
    public, _ = reservation()
    private = json.loads(checked(public["private_selection"]).read_text())
    originals = {a["id"]: a for a in private["areas"]}
    source_refs = {
        r["path"]: {"path": r["path"], "sha256": r["sha256"]}
        for r in private["iip_sources"]
    }
    for r in source_refs.values():
        checked(r)
    results, grids = [], []
    for product in manifest["products"]:
        source = originals[product["areas"][0]["id"]]
        _, gcps = metadata(
            checked(product["bands"][0]["annotation"]).read_bytes(), source, "hh"
        )
        grids.append({"product_id": product["product_id"], **grid_consistency(gcps)})
    for area in rows:
        source = originals[area["id"]]
        support = next(a for a in evidence["areas"] if a["id"] == area["id"])
        product = next(
            p for p in manifest["products"] if p["product_id"] == area["product_id"]
        )
        _, gcps = metadata(
            checked(product["bands"][0]["annotation"]).read_bytes(), source, "hh"
        )
        linear_geo = LinearNDInterpolator(
            [(g.row, g.col) for g in gcps], [(g.x, g.y) for g in gcps]
        )
        first = json.loads(checked(evidence["first_pass"]).read_text())
        decisions = next(a for a in first["areas"] if a["id"] == area["id"])
        original_csv = None
        if source["stream"] == "iceberg_support":
            expected = ref(iip / source["source"]["source_file"])
            if source_refs.get(expected["path"]) != expected:
                raise ValueError("Unexpected IIP source directory")
            original_csv = literal_record(source["source"], iip)
        for obj in support["objects"]:
            match = next(
                (
                    m
                    for m in obj["matches"]
                    if source["stream"] == "iceberg_support" or m.get("selected_source")
                ),
                None,
            )
            if match is None:
                raise ValueError("Selected source unavailable at target time")
            result = {
                "area_id": area["id"],
                "object_id": obj["object_id"],
                "stream": source["stream"],
                "distance_m": match["distance_m"],
                "frozen_allowance_m": match["uncertainty_m"],
                "inside_frozen_allowance": match["inside_source_allowance"],
                "compatible_sources": obj["inside_source_allowance_count"],
                "label_changed": False,
            }
            decision = next(
                o for o in decisions["objects"] if o["id"] == obj["object_id"]
            )
            x0, y0, x1, y1 = decision["bbox_px"]
            row = area["window"][1] + (y0 + y1) / 2
            col = area["window"][0] + (x0 + x1) / 2
            linear_lon, linear_lat = linear_geo(row, col)
            result["tps_vs_linear_grid_m"] = GEOD.inv(
                obj["lon"], obj["lat"], float(linear_lon), float(linear_lat)
            )[2]
            with GCPTransformer(gcps, tps=True) as tx:
                xs, ys = tx.xy([row], [col], offset="center")
            result["half_pixel_convention_sensitivity_m"] = GEOD.inv(
                obj["lon"], obj["lat"], xs[0], ys[0]
            )[2]
            if original_csv is not None:
                result["reported_size"] = source["source"]["size"]
                result["precision_sensitivity"] = precision_sensitivity(
                    original_csv["SIGHTING_LATITUDE"],
                    original_csv["SIGHTING_LONGITUDE"],
                    (obj["lon"], obj["lat"]),
                    match["observation_offset_s"],
                )
                result["native_minus_catalogue_midpoint_s"] = (
                    datetime.fromisoformat(obj["native_time_utc"]).timestamp()
                    - datetime.fromisoformat(
                        source["reference_midpoint_utc"]
                    ).timestamp()
                )
            results.append(result)
    output.mkdir(parents=True)
    save(
        output / "report.json",
        {
            "version": "0.1.0",
            "prepared": ref(prepared),
            "source_evidence": ref(evidence_path),
            "iip_sources": list(source_refs.values()),
            "objects": results,
            "grid_consistency": grids,
            "labels_changed": False,
            "allowances_changed": False,
            "scored": False,
            "independent_review": False,
            "compatibility_established": False,
            "limitations": [
                "Decimal precision is not positional accuracy; rounding rule unknown",
                "Grid consistency is not independent shoreline or ground validation",
                "No motion correction or calibrated uncertainty model",
                "Diffuse contexts do not establish iceberg detection",
            ],
        },
    )
    print(json.dumps({"report": ref(output / "report.json"), "objects": len(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--iip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.evidence, args.iip, args.output)
