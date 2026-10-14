"""Audit source evidence and reserve NL annotation opportunities before scoring."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests
from shapely.geometry import Point, shape

from agents.acquisition import (
    _complete_footprint,
    native_acquisition_group,
    unique_acquisitions,
)
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_reservation import save
from agents.region import load_region

POLICY = REPO_ROOT / "configs/reservations/source_supported_v010.json"
IIP_BASE = "https://noaadata.apps.nsidc.org/NOAA/G00807/"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def read_iip(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use sighting coordinates/time only; never modeled analysis positions."""
    region = load_region()
    counts: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    result = []
    seen: set[tuple[Any, ...]] = set()
    for path in sorted(directory.glob("IIP_*IcebergSeason.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "ICEBERG_YEAR",
                "ICEBERG_NUMBER",
                "SIGHTING_DATE",
                "SIGHTING_TIME",
                "SIGHTING_LATITUDE",
                "SIGHTING_LONGITUDE",
                "SIGHTING_METHOD",
                "SIZE",
                "SHAPE",
                "SOURCE",
            }
            actual = {
                "_".join(key.split()).upper() for key in (reader.fieldnames or [])
            }
            if not required.issubset(actual):
                raise ValueError(f"Unsupported IIP schema: {path.name}")
            for row in reader:
                line = reader.line_num
                row = {
                    "_".join(key.split()).upper(): value
                    for key, value in row.items()
                    if key is not None
                }
                counts["input_rows"] += 1
                try:
                    clock = row["SIGHTING_TIME"].strip().zfill(4)
                    if len(clock) != 4 or not clock.isdigit():
                        raise ValueError("Invalid sighting time")
                    instant = datetime.strptime(
                        row["SIGHTING_DATE"].strip(), "%m/%d/%Y"
                    ).replace(
                        hour=int(clock[:2]), minute=int(clock[2:]), tzinfo=timezone.utc
                    )
                    lon, lat = float(row["SIGHTING_LONGITUDE"]), float(
                        row["SIGHTING_LATITUDE"]
                    )
                    # Format-A downloads use signed west longitude. Positive-west
                    # legacy conventions are not guessed for these CSV inputs.
                    if (
                        not math.isfinite(lon)
                        or not math.isfinite(lat)
                        or not -180 <= lon <= 0
                        or not -90 <= lat <= 90
                    ):
                        raise ValueError("Invalid signed coordinates")
                    if instant < datetime(2014, 10, 1, tzinfo=timezone.utc):
                        counts["before_study_sentinel_period"] += 1
                        continue
                    if not region.covers(Point(lon, lat)):
                        counts["outside_nl"] += 1
                        continue
                    method, source = (
                        row["SIGHTING_METHOD"].strip().upper(),
                        row["SOURCE"].strip().upper(),
                    )
                    identity = (
                        f"{row['ICEBERG_YEAR'].strip()}:{row['ICEBERG_NUMBER'].strip()}"
                    )
                    key = (identity, instant.isoformat(), lon, lat, method, source)
                    if key in seen:
                        counts["duplicate_sightings"] += 1
                        continue
                    seen.add(key)
                    result.append(
                        {
                            "iceberg_id": identity,
                            "timestamp": instant.isoformat(),
                            "time": instant.timestamp(),
                            "lon": lon,
                            "lat": lat,
                            "method": method,
                            "source": source,
                            "size": row["SIZE"].strip(),
                            "shape": row["SHAPE"].strip(),
                            "source_file": path.name,
                            "source_line": line,
                            "label": None,
                            "provisional": True,
                        }
                    )
                    methods[method] += 1
                    sources[source] += 1
                except (ValueError, KeyError, TypeError):
                    counts["invalid_rows"] += 1
    result.sort(
        key=lambda row: (
            row["time"],
            row["iceberg_id"],
            row["source_file"],
            row["source_line"],
        )
    )
    return result, {
        **dict(counts),
        "valid_unique_nl_sightings": len(result),
        "methods": dict(methods),
        "sources": dict(sources),
        "first_utc": result[0]["timestamp"] if result else None,
        "last_utc": result[-1]["timestamp"] if result else None,
    }


def query_year(year: int, output: Path, seed: Path | None = None) -> None:
    """Bounded public catalogue pages; no source or satellite credentials."""
    filename = output / f"catalogue_hh_hv_{year}.json"
    if filename.exists():
        cached = json.loads(filename.read_text())
        if cached.get("complete") is not True:
            raise ValueError("Unverified catalogue cache; use fresh output and --seed")
        return
    region = load_region()
    params = {
        "$filter": "Collection/Name eq 'SENTINEL-1' and contains(Name,'IW_GRDH_1SDH') "
        + f"and OData.CSC.Intersects(area=geography'SRID=4326;{region.wkt}') "
        + f"and ContentDate/Start ge {year}-01-01T00:00:00Z "
        + f"and ContentDate/Start lt {year+1}-01-01T00:00:00Z",
        "$orderby": "ContentDate/Start asc",
        "$top": "1000",
        "$expand": "Attributes",
    }
    url = CATALOGUE
    products: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    anchor: str | None = None
    seed_ref = None
    if seed is not None and (seed / filename.name).exists():
        source = seed / filename.name
        cached = json.loads(source.read_text())
        products = cached["products"]
        receipts = cached["queries"]
        seed_ref = {
            "path": source.resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(source),
        }
        if receipts[-1]["count"] < 1000:
            url = ""
        elif products:
            anchor = products[-1]["ContentDate"]["Start"]
            instant = parse_utc(anchor)
            if instant is None:
                raise ValueError("Invalid catalogue continuation timestamp")
            params = {
                **params,
                "$filter": params["$filter"]
                + f" and ContentDate/Start ge {instant.isoformat()}",
            }
    seen: set[str] = set()
    current_params: dict[str, str] | None = params
    while url:
        parsed = urlparse(url)
        key = url + ("?" + urlencode(current_params) if current_params else "")
        if (
            parsed.scheme != "https"
            or parsed.netloc != "catalogue.dataspace.copernicus.eu"
            or key in seen
            or len(seen) >= 100
        ):
            raise ValueError("Invalid catalogue pagination")
        seen.add(key)
        response = requests.get(url, params=current_params, timeout=60)
        response.raise_for_status()
        page = response.json()
        receipts.append(
            {
                "url": response.url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "count": len(page["value"]),
            }
        )
        products.extend(page["value"])
        url = page.get("@odata.nextLink")
        current_params = None
        if not url and len(page["value"]) == 1000:
            latest = page["value"][-1]["ContentDate"]["Start"]
            instant = parse_utc(latest)
            if instant is None or (anchor is not None and latest <= anchor):
                raise ValueError("Catalogue timestamp continuation did not advance")
            anchor = latest
            url = CATALOGUE
            current_params = {
                **params,
                "$filter": params["$filter"]
                + f" and ContentDate/Start ge {instant.isoformat()}",
            }
        print(
            json.dumps(
                {"year": year, "page": len(receipts), "rows": len(page["value"])}
            ),
            flush=True,
        )
    products = list({p["Id"]: p for p in products}.values())
    save(
        filename,
        {"queries": receipts, "products": products, "complete": True, "seed": seed_ref},
    )
    print(json.dumps({"year": year, "products": len(products)}), flush=True)


def iip_overlaps(
    products: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary = [
        r
        for r in observations
        if r["method"] in policy["iip_primary_methods"]
        and not re.match(r"^(SN|RS|RCM|SENT|TSX|ENVI)", r["source"])
    ]
    primary.sort(key=lambda r: r["time"])
    times = [r["time"] for r in primary]
    results = []
    counts: Counter[str] = Counter()
    for product in unique_acquisitions(products):
        counts["physical_slices"] += 1
        start = parse_utc(product.get("ContentDate", {}).get("Start"))
        end = parse_utc(product.get("ContentDate", {}).get("End"))
        if (
            not start
            or not end
            or start >= end
            or not _complete_footprint(product.get("GeoFootprint"))
            or native_acquisition_group(product.get("Name")) is None
            or "_IW_GRDH_1SDH_" not in product.get("Name", "")
        ):
            counts["invalid_products"] += 1
            continue
        center = (start + (end - start) / 2).timestamp()
        span = policy["iip_catalogue_screen_seconds"]
        footprint = shape(product["GeoFootprint"])
        for row in primary[
            bisect.bisect_left(times, center - span) : bisect.bisect_right(
                times, center + span
            )
        ]:
            if not footprint.covers(Point(row["lon"], row["lat"])):
                continue
            delta = abs(center - row["time"])
            results.append(
                {
                    "product": product,
                    "observation": row,
                    "absolute_offset_s": delta,
                    "reference_midpoint_utc": datetime.fromtimestamp(
                        center, timezone.utc
                    ).isoformat(),
                    "search_uncertainty_m": policy["iip_uncertainty_base_m"]
                    + policy["iip_drift_allowance_m_per_second"] * delta,
                    "temporal_eligible": delta
                    <= policy["iip_reservation_max_offset_seconds"],
                    "evidence": "visual_or_radar_visual_sighting",
                    "label": None,
                    "provisional": True,
                }
            )
    results.sort(
        key=lambda r: (
            not r["temporal_eligible"],
            r["absolute_offset_s"],
            r["observation"]["method"] != "R/V",
            r["product"].get("Online") is not True,
            r["product"]["Id"],
            r["observation"]["iceberg_id"],
        )
    )
    return results, {
        **dict(counts),
        "primary_sightings": len(primary),
        "visual_method_with_satellite_source_excluded": sum(
            r["method"] in policy["iip_primary_methods"] for r in observations
        )
        - len(primary),
        "overlap_pairs_within_2h": len(results),
        "pairs_within_30min": sum(r["temporal_eligible"] for r in results),
        "unique_icebergs_within_30min": len(
            {r["observation"]["iceberg_id"] for r in results if r["temporal_eligible"]}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--seed", type=Path)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((REPO_ROOT / "data").resolve()):
        raise ValueError("Audit outputs must stay in project data")
    args.output.mkdir(parents=True, exist_ok=True)
    expected_sources = {f"IIP_{year}IcebergSeason.csv" for year in range(2014, 2022)}
    if {p.name for p in args.iip.glob("IIP_*IcebergSeason.csv")} != expected_sources:
        raise ValueError("All eight original IIP annual CSV files required")
    observations, summary = read_iip(args.iip)
    save(
        args.output / "iip_normalized.json",
        {
            "summary": summary,
            "observations": observations,
            "sources": [
                {
                    "file": p.name,
                    "path": p.resolve().relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(p),
                    "url": IIP_BASE + p.name,
                }
                for p in sorted(args.iip.glob("*.csv"))
            ],
        },
    )
    print(json.dumps(summary), flush=True)
    if args.fetch:
        for year in range(2014, 2022):
            query_year(year, args.output, args.seed)
    if args.fetch or args.catalogue:
        expected = {f"catalogue_hh_hv_{year}.json" for year in range(2014, 2022)}
        actual = {
            p.name
            for p in (args.catalogue or args.output).glob("catalogue_hh_hv_*.json")
        }
        if actual != expected:
            raise ValueError("All eight yearly catalogue snapshots required")
        products = [
            p
            for path in sorted(
                (args.catalogue or args.output).glob("catalogue_hh_hv_*.json")
            )
            for p in json.loads(path.read_text())["products"]
        ]
        if any(
            json.loads(p.read_text()).get("complete") is not True
            for p in (args.catalogue or args.output).glob("catalogue_hh_hv_*.json")
        ):
            raise ValueError("Incomplete catalogue coverage cannot be ranked")
        ranked, counts = iip_overlaps(
            products, observations, json.loads(POLICY.read_text())
        )
        catalogue_files = sorted(
            (args.catalogue or args.output).glob("catalogue_hh_hv_*.json")
        )
        save(
            args.output / "iip_ranked.json",
            {
                "summary": counts,
                "ranked": ranked,
                "provenance": {
                    "normalized": {
                        "path": (args.output / "iip_normalized.json")
                        .resolve()
                        .relative_to(REPO_ROOT)
                        .as_posix(),
                        "sha256": sha256_file(args.output / "iip_normalized.json"),
                    },
                    "catalogues": [
                        {
                            "path": p.resolve().relative_to(REPO_ROOT).as_posix(),
                            "sha256": sha256_file(p),
                        }
                        for p in catalogue_files
                    ],
                    "policy_sha256": sha256_file(POLICY),
                    "code_sha256": sha256_file(Path(__file__)),
                },
            },
        )
        print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
