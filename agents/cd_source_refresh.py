"""Refresh conditional vessel reservations and bounded iceberg evidence searches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import Point, shape

from agents.artifacts import REPO_ROOT
from agents.cd_reference import GUARDS, POLICY
from agents.hh_hv_reservation import save
from agents.iceberg_corroboration import query_optical
from agents.provisional_training import (
    inventory,
    policy,
    public_area,
    select,
    source_checks,
    source_pool,
)
from agents.source_followup import fresh
from agents.source_reservation import checked, envelope, ref

SEARCH = REPO_ROOT / "configs/reservations/cd_evidence_search_v010.json"


def reserve(ais: Path, output: Path) -> None:
    rules = policy()
    iip = REPO_ROOT / "data/source_supported/v010_01/corrected_audit/iip_ranked.json"
    sources = source_checks(iip, ais)
    priors, exposed, old, hashes = inventory()
    # The original inventory predates native_training. Include those sealed
    # acquisitions and identities without rewriting any original release.
    for path in sorted(
        (REPO_ROOT / "reservations/native_training").glob("*/manifest.json")
    ):
        manifest = json.loads(path.read_text())
        private = json.loads(checked(manifest["private_selection"]).read_text())
        hashes[ref(path)["path"]] = ref(path)["sha256"]
        for area in private["areas"]:
            old.append(area)
            priors.append(
                {
                    "geometry": area["geometry"],
                    "group": area["geographic_group"],
                    "split": area["split"],
                    "acquisition_group": area["native_acquisition_group"],
                    "quarantine": False,
                }
            )
    pool = [
        r
        for r in source_pool(**sources, rules=rules)
        if r["stream"] == "vessel_support"
    ]
    # Hypothetical C/D eligibility isolates non-input blockers. It does not
    # modify the A/B consumer, admission policy or actual training permission.
    screening = rules | {"native_specification": ref(POLICY)}
    selected, ranked = select(pool, priors, exposed, old, screening)
    fresh(output)
    private = {
        "ais": ref(ais),
        "iip": ref(iip),
        "policy": ref(POLICY),
        "prior_metadata_hashes": hashes,
        "priors": priors,
        "exposed": sorted(exposed),
        "old": old,
        "areas": selected,
        "rankings": ranked,
        **GUARDS,
    }
    save(output / "selection_private.json", private)
    save(
        output / "reservation.json",
        {
            "private_selection": ref(output / "selection_private.json"),
            "areas": [
                public_area(a)
                | {"status": "conditional_pending_mission_and_object_admission"}
                for a in selected
            ],
            "rankings": ranked,
            "archive": sources["ais"]["summary"]["archive"],
            "annotated_objects": 0,
            "labels_admitted": 0,
            "scope": "C/D compatibility-conditional training reservations; "
            "no input-contract extension",
            **GUARDS,
        },
    )
    print(json.dumps({"opportunities": len(pool), "reserved": len(selected)}))


def verify_reservation(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    private = json.loads(checked(value["private_selection"]).read_text())
    for key in GUARDS:
        if value.get(key) is not False or private.get(key) is not False:
            raise ValueError("Reservation guards changed")
    for name, digest in private["prior_metadata_hashes"].items():
        checked({"path": name, "sha256": digest})
    sources = source_checks(checked(private["iip"]), checked(private["ais"]))
    rules = policy()
    pool = [
        r
        for r in source_pool(**sources, rules=rules)
        if r["stream"] == "vessel_support"
    ]
    selected, ranked = select(
        pool,
        private["priors"],
        set(private["exposed"]),
        private["old"],
        rules | {"native_specification": ref(POLICY)},
    )
    if (
        json.loads(json.dumps(selected)) != private["areas"]
        or ranked != private["rankings"]
        or ranked != value["rankings"]
    ):
        raise ValueError("Vessel reservation replay differs")
    public = [
        public_area(a) | {"status": "conditional_pending_mission_and_object_admission"}
        for a in selected
    ]
    if (
        value["areas"] != public
        or value["annotated_objects"] != 0
        or value["labels_admitted"] != 0
    ):
        raise ValueError("Reservation is not a label collection")
    return value


def targets() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    evidence = json.loads(
        (REPO_ROOT / "docs/evidence/snap_corroboration_20261014.json").read_text()
    )
    old = checked(evidence["private_inputs"]["corroboration"])
    new = REPO_ROOT / "data/source_supported/training_evidence_v010_01/report.json"
    rows = []
    for row in json.loads(old.read_text())["areas"]:
        rows.append(
            {
                "id": row["id"],
                "lon": row["lon"],
                "lat": row["lat"],
                "native_time_utc": row["native_time_utc"],
            }
        )
    for row in json.loads(new.read_text())["areas"]:
        for obj in row["objects"]:
            rows.append(
                {
                    "id": row["id"],
                    **{k: obj[k] for k in ("lon", "lat", "native_time_utc")},
                }
            )
    if len(rows) != 5:
        raise ValueError("Exact five existing provisional hypotheses required")
    return rows, [ref(old), ref(new)]


def search_params(target: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    rules = json.loads(SEARCH.read_text())
    hours = rules["hours_each_side"]
    instant = datetime.fromisoformat(target["native_time_utc"])
    start = (instant - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    end = (instant + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    # Search allowance only, not a physical drift model or identity gate.
    geometry = envelope(target["lon"], target["lat"], rules["envelope_half_width_m"])
    params = {
        "$filter": "Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') "
        + f"and ContentDate/Start ge {start} and ContentDate/Start le {end} "
        + f"and OData.CSC.Intersects(area=geography'SRID=4326;{shape(geometry).wkt}')",
        "$orderby": "ContentDate/Start asc",
        "$top": str(rules["maximum_results_per_request"]),
        "$expand": "Attributes",
    }
    stac = {
        "collections": ["landsat-c2-l2"],
        "bbox": list(shape(geometry).bounds),
        "datetime": f"{start}/{end}",
        "limit": rules["maximum_results_per_request"],
    }
    return params, stac


def response_status(payload: dict[str, Any], stac: bool) -> str:
    if "error_type" in payload and not payload.get("features" if stac else "value"):
        return "failed"
    if not isinstance(payload.get("features" if stac else "value"), list):
        raise ValueError("Invalid optical response schema")
    more = (
        any(x.get("rel") == "next" for x in payload.get("links", []))
        if stac
        else bool(payload.get("@odata.nextLink"))
    )
    return "truncated" if more else "complete"


def summarize_optical(
    payload: dict[str, Any], target: dict[str, Any], stac: bool
) -> list[dict[str, Any]]:
    results = []
    time = datetime.fromisoformat(target["native_time_utc"])
    for product in payload.get("features" if stac else "value", []):
        if stac:
            identifier, name = product["id"], product["id"]
            instant = product["properties"]["datetime"]
            geometry = product.get("geometry")
            cloud = product["properties"].get("eo:cloud_cover")
        else:
            identifier, name = product["Id"], product["Name"]
            instant, geometry = product["ContentDate"]["Start"], product.get(
                "GeoFootprint"
            )
            cloud = next(
                (
                    a.get("Value")
                    for a in product.get("Attributes", [])
                    if a.get("Name") == "cloudCover"
                ),
                None,
            )
        timestamp = datetime.fromisoformat(instant.replace("Z", "+00:00"))
        results.append(
            {
                "id": identifier,
                "name": name,
                "acquisition": instant,
                "offset_s": (timestamp - time).total_seconds(),
                "cloud_cover_scene": cloud,
                "covers_original_position": bool(
                    geometry
                    and shape(geometry).covers(Point(target["lon"], target["lat"]))
                ),
                "pixels_inspected": False,
                "same_object_confirmed": False,
            }
        )
    return results


def search(output: Path) -> None:
    rows, references = targets()
    fresh(output)
    report: dict[str, Any] = {
        "policy": ref(SEARCH),
        "inputs": references,
        "targets": [],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        **GUARDS,
    }
    for row in rows:
        params, stac = search_params(row)
        receipt = query_optical(params, output / f"{row['id']}_s2.json")
        page = json.loads(checked(receipt["response"]).read_text())
        s2 = (
            summarize_optical(page, row, False) if receipt["status"] != "failed" else []
        )
        path = output / f"{row['id']}_landsat.json"
        request: dict[str, Any] = {
            "url": "https://planetarycomputer.microsoft.com/api/stac/v1/search",
            "body": stac,
        }
        try:
            response = requests.post(request["url"], json=stac, timeout=60)
            response.raise_for_status()
            page = response.json()
            if not isinstance(page.get("features"), list):
                raise ValueError("Invalid STAC response")
            save(path, page)
            status = (
                "truncated"
                if any(x.get("rel") == "next" for x in page.get("links", []))
                else "complete"
            )
            landsat = summarize_optical(page, row, True)
        except (requests.RequestException, ValueError) as error:
            save(path, {"error_type": type(error).__name__})
            status, landsat = "failed", []
        report["targets"].append(
            {
                **row,
                "sentinel2": {"request": params, "receipt": receipt, "products": s2},
                "landsat": {
                    "request": request,
                    "status": status,
                    "response": ref(path),
                    "products": landsat,
                },
                "physical_corroboration": False,
            }
        )
        save(output / f"progress_{len(report['targets'])}.json", report)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "s2": len(s2),
                    "landsat": len(landsat),
                    "statuses": [receipt["status"], status],
                }
            ),
            flush=True,
        )
    save(output / "report.json", report)


def verify_search(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    rows, references = targets()
    if any(report.get(k) is not False for k in GUARDS):
        raise ValueError("Evidence search guards changed")
    if (
        report["policy"] != ref(SEARCH)
        or report["inputs"] != references
        or len(report["targets"]) != len(rows)
    ):
        raise ValueError("Evidence source inventory differs")
    for row, original in zip(report["targets"], rows, strict=True):
        if (
            any(row[k] != original[k] for k in original)
            or row["physical_corroboration"] is not False
        ):
            raise ValueError("No new physical confirmation from metadata")
        s2, ls = search_params(original)
        if row["sentinel2"]["request"] != s2 or row["landsat"]["request"]["body"] != ls:
            raise ValueError("Frozen search differs")
        for key, is_stac in (("sentinel2", False), ("landsat", True)):
            result = row[key]
            receipt = result if is_stac else result["receipt"]
            page = json.loads(checked(receipt["response"]).read_text())
            if receipt["status"] != response_status(page, is_stac):
                raise ValueError("Failed or truncated optical query misreported")
            expected = (
                summarize_optical(page, original, is_stac)
                if receipt["status"] != "failed"
                else []
            )
            if expected != result["products"]:
                raise ValueError("Optical summary replay differs")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--verify-reservation", type=Path)
    parser.add_argument("--verify-search", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reserve and args.output:
        reserve(args.reserve, args.output)
    elif args.search and args.output:
        search(args.output)
    elif args.verify_reservation:
        print(
            json.dumps(
                {
                    "verified": True,
                    "reserved": len(
                        verify_reservation(args.verify_reservation)["areas"]
                    ),
                }
            )
        )
    elif args.verify_search:
        print(
            json.dumps(
                {
                    "verified": True,
                    "hypotheses": len(verify_search(args.verify_search)["targets"]),
                }
            )
        )
    else:
        parser.error("Choose reserve, search or a verification")
