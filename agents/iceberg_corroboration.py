"""Audit supporting source evidence without changing any sealed SAR labels."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from rasterio.transform import GCPTransformer
from shapely.geometry import Point, shape

from agents.artifacts import REPO_ROOT
from agents.association import GEOD
from agents.hh_hv_reservation import save
from agents.snap_reference import POLICY, inputs
from agents.source_followup import fresh, selection
from agents.source_reservation import checked, ref
from agents.source_supported import CATALOGUE
from agents.source_windows import metadata

SATELLITE = re.compile(r"^(S1[ABC]|SN|RS|RCM|SENT|TSX|ENVI)")


def repeat_context(
    source: dict[str, Any],
    observations: list[dict[str, Any]],
    instant: float,
    lon: float,
    lat: float,
    span: float,
) -> list[dict[str, Any]]:
    """Repeated sightings remain correlated publisher context; no new class."""
    result, seen = [], set()
    for obs in observations:
        key = (
            obs["iceberg_id"],
            obs["time"],
            obs["lon"],
            obs["lat"],
            obs["method"],
            obs["source"],
        )
        if (
            obs["iceberg_id"] != source["iceberg_id"]
            or key in seen
            or abs(obs["time"] - instant) > span
            or (obs["source_file"], obs["source_line"])
            == (source["source_file"], source["source_line"])
        ):
            continue
        seen.add(key)
        delta = obs["time"] - instant
        distance = GEOD.inv(lon, lat, obs["lon"], obs["lat"])[2]
        primary = obs["method"] in {"VIS", "R/V"} and not SATELLITE.match(obs["source"])
        result.append(
            {
                "source": obs,
                "signed_offset_s": delta,
                "distance_m": distance,
                "primary_method": bool(primary),
                "inside_original_gate": bool(
                    abs(delta) <= 1800 and distance <= 300 + abs(delta)
                ),
                "independent_confirmation": False,
            }
        )
    return sorted(result, key=lambda r: abs(r["signed_offset_s"]))


def optical_query(
    lon: float, lat: float, instant: datetime, hours: float
) -> dict[str, str]:
    start = (instant - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    end = (instant + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    return {
        "$filter": "Collection/Name eq 'SENTINEL-2' "
        + "and contains(Name,'MSIL1C') "
        + f"and ContentDate/Start ge {start} and ContentDate/Start le {end} "
        + "and OData.CSC.Intersects(area=geography'SRID=4326;"
        + f"POINT({lon:.10f} {lat:.10f})')",
        "$orderby": "ContentDate/Start asc",
        "$top": "100",
        "$expand": "Attributes",
    }


def query_optical(params: dict[str, str], output: Path) -> dict[str, Any]:
    """No credentials; freeze response, pagination state and retrieval failures."""
    try:
        response = requests.get(CATALOGUE, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload.get("value"), list):
            raise ValueError("Invalid catalogue schema")
        save(output, payload)
        return {
            "status": "complete" if not payload.get("@odata.nextLink") else "truncated",
            "url": response.url,
            "response": ref(output),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    except (requests.RequestException, ValueError) as error:
        save(output, {"failed": True, "error_type": type(error).__name__})
        return {
            "status": "failed",
            "response": ref(output),
            "error_type": type(error).__name__,
        }


def optical_summary(receipt: dict[str, Any], lon: float, lat: float) -> dict[str, Any]:
    if receipt["status"] not in {"complete", "truncated"}:
        return {
            "query_status": receipt["status"],
            "covering_products": None,
            "object_confirmation": False,
        }
    page = json.loads(checked(receipt["response"]).read_text())
    products = []
    for product in page["value"]:
        footprint = product.get("GeoFootprint")
        if footprint is None or not shape(footprint).covers(Point(lon, lat)):
            continue
        clouds = [
            a.get("Value")
            for a in product.get("Attributes", [])
            if a.get("Name") == "cloudCover"
        ]
        products.append(
            {
                "product_id": product["Id"],
                "name": product["Name"],
                "acquisition": product["ContentDate"],
                "scene_cloud_cover": clouds[0] if clouds else None,
                "pixels_inspected": False,
                "local_clear_sky_verified": False,
            }
        )
    return {
        "query_status": receipt["status"],
        "covering_products": len(products),
        "products": products,
        "object_confirmation": False,
        "limitation": "Footprint/cloud metadata does not establish local "
        "visibility or target identity; no motion search or optical pixels evaluated",
    }


def audit(output: Path) -> None:
    policy, manifest, rows = inputs()
    _, private = selection()
    ranking = json.loads(checked(private["inputs"]["iip"]).read_text())
    normalized_path = checked(ranking["provenance"]["normalized"])
    normalized = json.loads(normalized_path.read_text())
    # Hash check original CSVs as well as their normalized record artifact.
    for source in normalized["sources"]:
        checked(source)
    first_path = REPO_ROOT / "configs/reservations/source_followup_first_pass_v010.json"
    first = json.loads(first_path.read_text())
    fresh(output)
    results = []
    for row in rows:
        original = next(a for a in private["areas"] if a["id"] == row["id"])
        product = next(
            p for p in manifest["products"] if p["product_id"] == row["product_id"]
        )
        info, gcps = metadata(
            checked(product["bands"][0]["annotation"]).read_bytes(), original, "hh"
        )
        decision = next(a for a in first["areas"] if a["id"] == row["id"])
        if (
            len(decision["objects"]) != 1
            or decision["chip_sha256"] != row["chip"]["sha256"]
        ):
            raise ValueError("Sealed first-pass object binding differs")
        x0, y0, x1, y1 = decision["objects"][0]["bbox_px"]
        col, line = row["window"][0] + (x0 + x1) / 2, row["window"][1] + (y0 + y1) / 2
        with GCPTransformer(gcps, tps=True) as tx:
            xs, ys = tx.xy([line], [col], offset="ul")
        instant = datetime.fromisoformat(info["native_start_time"]).replace(
            tzinfo=timezone.utc
        ) + timedelta(seconds=line * info["azimuth_interval_s"])
        contexts = repeat_context(
            original["source"],
            normalized["observations"],
            instant.timestamp(),
            xs[0],
            ys[0],
            policy["repeat_iip_window_hours"] * 3600,
        )
        receipt = query_optical(
            optical_query(xs[0], ys[0], instant, policy["optical_window_hours"]),
            output / f"{row['id']}_optical.json",
        )
        results.append(
            {
                "id": row["id"],
                "split": row["split"],
                "source": original["source"],
                "lon": xs[0],
                "lat": ys[0],
                "native_time_utc": instant.isoformat(),
                "repeat_reports": contexts,
                "optical_receipt": receipt,
                "optical": optical_summary(receipt, xs[0], ys[0]),
                "independent_confirmation": False,
            }
        )
        save(output / f"progress_{len(results)}.json", {"areas": results})
    # Inventory only known local archives; date names are discovery evidence,
    # not a replayed AIS reception/coverage guarantee. Never read live CSV bodies.
    archive_files = sorted((REPO_ROOT / "data/raw/ais_stream").glob("ais_stream_*.csv"))
    archive_dates = sorted(p.stem.removeprefix("ais_stream_") for p in archive_files)
    historical_paths = sorted((REPO_ROOT / "data/raw").glob("ais-20*"))
    historical_dates = sorted(p.name.removeprefix("ais-") for p in historical_paths)
    inventory = {
        "live_archive_date_names": archive_dates,
        "historical_directory_date_names": historical_dates,
        "inspection": "Date-name inventory only; CSV contents and receiver "
        "completeness not evaluated",
        "cotemporal_support_available": False,
    }
    for result in results:
        year = result["native_time_utc"][:4]
        if any(d.startswith(year) for d in archive_dates + historical_dates):
            raise ValueError(
                "Potential historical AIS requires content audit before absence claim"
            )
    report = {
        "policy": ref(POLICY),
        "normalized_iip": ref(normalized_path),
        "first_pass": ref(first_path),
        "areas": results,
        "ais_inventory": inventory,
        "independent_review": False,
        "training_approved": False,
        "labels_modified": False,
        "classifier_scored": False,
        "detector_scored": False,
        "limitations": [
            "Repeated publisher identity is correlated supporting context, "
            "not independent verification",
            "No original observer logs, photographs or calibrated position "
            "errors in the available IIP CSV",
            "Catalogue search is bounded to Sentinel-2 L1C within three hours "
            "at the SAR location; no exhaustive multisensor search",
            "No qualified independent reviewer; existing labels and "
            "association allowances retained",
        ],
    }
    save(output / "report.json", report)
    print(
        json.dumps(
            {
                "report": ref(output / "report.json"),
                "candidates": len(results),
                "independently_confirmed": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.output)
