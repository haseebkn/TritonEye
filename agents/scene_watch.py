#!/usr/bin/env python3
"""
Finds Sentinel-1 products eligible for an NL open-water AIS-subset experiment.
These metadata checks prioritize acquisitions; valid pixels and aligned AIS
are checked after processing. Eligibility does not establish a measurement,
vessel precision, overall recall or complete receiver coverage.

  1. VV/VH polarization (1SDV). The detector cannot use HH/HV, and over parts
     of this region HH/HV is all that is acquired -- see docs/DATA_SOURCES.md.
  2. Inside the study area.
  3. Locally recorded AIS in the acquisition window and footprint. The
     integrated live source cannot backfill gaps; other archives may exist.

CHEAP CHECKS FIRST. A source product is ~1.7 GB, so this orders the tests by
cost: catalogue metadata answers polarization and geometry for free, and the
local AIS archive answers coverage for free. Only a scene that passes all three
is worth downloading. Running the pipeline first and discovering afterwards that
no telemetry existed is how the earlier attempts were wasted.

This reports; agents.watch handles versioned processing and outcome records.

    python -m agents.scene_watch --days 12
    python -m agents.scene_watch --days 30 --json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from agents.acquisition import product_id
from agents.ais_validation import finite_number, parse_utc, valid_mmsi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token"
)

# The detector is trained on VV/VH; HH/HV products carry a different scattering
# regime and are refused at ingest rather than relabelled.
REQUIRED_POLARIZATION = "1SDV"

# Matching window used throughout the pipeline for AIS/detection association.
AIS_WINDOW_MINUTES = 5


class SceneWatchUnavailable(RuntimeError):  # noqa: N818 - reads better
    """Raised when the catalogue cannot be queried."""


def _token() -> str:
    import requests

    user = os.getenv("COPERNICUS_USER")
    password = os.getenv("COPERNICUS_PASS")
    if not user or not password:
        raise SceneWatchUnavailable(
            "COPERNICUS_USER / COPERNICUS_PASS are not set; see .env.example"
        )
    r = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user,
            "password": password,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise SceneWatchUnavailable(f"CDSE authentication failed: HTTP {r.status_code}")
    return str(r.json()["access_token"])


def _footprint_wkt(aoi_path: str) -> str:
    with open(aoi_path, "r", encoding="utf-8") as f:
        geom = json.load(f)["features"][0]["geometry"]
    ring = geom["coordinates"][0]
    return "POLYGON((" + ",".join(f"{x} {y}" for x, y in ring) + "))"


def search_acquisitions(
    aoi_path: str, days: int = 12, token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns recent IW GRDH products intersecting the AOI, newest first.

    Both polarizations are returned. HH/HV scenes are kept deliberately: the
    ratio of usable to unusable coverage is itself the finding recorded in
    docs/DATA_SOURCES.md, and hiding them here would make the constraint
    invisible at exactly the point someone is looking for a scene.
    """
    import requests

    token = token or _token()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    area = _footprint_wkt(aoi_path)
    flt = (
        "Collection/Name eq 'SENTINEL-1' and contains(Name, 'IW_GRDH') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{area}') "
        f"and ContentDate/Start ge {since}"
    )
    params: Dict[str, str] = {
        "$filter": flt,
        "$orderby": "ContentDate/Start desc",
        "$top": "200",
    }
    r = requests.get(
        CATALOGUE,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=90,
    )
    if r.status_code != 200:
        raise SceneWatchUnavailable(f"Catalogue query failed: HTTP {r.status_code}")

    # Each slice has its own footprint and identity. Never collapse products
    # merely because their names or acquisition dates share a prefix.
    page = r.json()
    products = list(page.get("value", []))
    visited: set[str] = set()
    while page.get("@odata.nextLink"):
        next_link = page["@odata.nextLink"]
        parsed = urlparse(next_link)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "catalogue.dataspace.copernicus.eu"
            or next_link in visited
            or len(visited) >= 100
        ):
            raise SceneWatchUnavailable("Invalid or cyclic catalogue pagination")
        visited.add(next_link)
        response = requests.get(
            next_link, headers={"Authorization": f"Bearer {token}"}, timeout=90
        )
        if response.status_code != 200:
            raise SceneWatchUnavailable(
                f"Catalogue page failed: HTTP {response.status_code}"
            )
        page = response.json()
        products.extend(page.get("value", []))
    seen: Dict[str, Dict[str, Any]] = {}
    for p in products:
        seen.setdefault(product_id(p["Id"]), p)
    return sorted(seen.values(), key=lambda p: p["ContentDate"]["Start"], reverse=True)


def ais_rows_near(
    archive_dir: str, acquisition_iso: str, window_minutes: int = AIS_WINDOW_MINUTES
) -> List[Dict[str, str]]:
    """
    Returns valid archived rows, including legacy and expanded-schema files.

    Reads the daily archive directly rather than the pipeline's filter, because
    this must answer "is there anything to score against" before any scene is
    downloaded or any bbox is known.
    """
    import csv
    import glob

    acq = parse_utc(acquisition_iso, allow_naive=True)
    if acq is None:
        return []
    lo, hi = (
        acq - timedelta(minutes=window_minutes),
        acq + timedelta(minutes=window_minutes),
    )

    # Day boundaries: a window can straddle midnight, so check both files.
    days = {lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")}
    found: List[Dict[str, str]] = []
    for day in sorted(days):
        for path in sorted(
            glob.glob(os.path.join(archive_dir, f"ais_stream_{day}*.csv"))
        ):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        t = parse_utc(row.get("timestamp", ""), allow_naive=True)
                        if t is None or not (lo <= t <= hi):
                            continue
                        lon, lat = finite_number(row.get("lon")), finite_number(
                            row.get("lat")
                        )
                        if (
                            valid_mmsi(row.get("mmsi")) is None
                            or lon is None
                            or lat is None
                            or not -180 <= lon <= 180
                            or not -90 <= lat <= 90
                        ):
                            continue
                        found.append(
                            {
                                key: value
                                for key, value in row.items()
                                if isinstance(key, str) and isinstance(value, str)
                            }
                        )
            except OSError:
                continue
    return found


def ais_observations_near(
    archive_dir: str, acquisition_iso: str, window_minutes: int = AIS_WINDOW_MINUTES
) -> List[Tuple[float, float]]:
    return [
        (float(row["lon"]), float(row["lat"]))
        for row in ais_rows_near(archive_dir, acquisition_iso, window_minutes)
    ]


def within_footprint(
    positions: Sequence[Tuple[float, float]], footprint: Optional[Dict[str, Any]]
) -> List[Tuple[float, float]]:
    """
    Keeps only positions inside the product's imaged swath.

    `footprint` is the catalogue's GeoFootprint (GeoJSON). A missing or
    malformed footprint returns NOTHING rather than everything: claiming a
    scene is scorable when its swath is unknown is the failure this guards
    against, and it is far more expensive than missing an opportunity.
    """
    if not positions or not footprint:
        return []
    try:
        from shapely.geometry import Point, shape

        swath = shape(footprint)
    except Exception:
        return []
    if (
        swath.is_empty
        or not swath.is_valid
        or swath.geom_type not in ("Polygon", "MultiPolygon")
    ):
        return []
    return [p for p in positions if swath.covers(Point(p[0], p[1]))]


def water_eligible(positions: Sequence[Tuple[float, float]], cache_dir: str) -> int:
    """
    Counts AIS positions the pipeline would treat as alert-eligible water.

    This watcher prioritizes open-water reference observations. The evaluator
    can also measure raw AIS-subset recall in coastal settings; this selection
    policy does not make harbour AIS intrinsically unsuitable for evaluation.

    Harbour water can be physically water and still fall in the 300 m coastal
    exclusion band. Municipal imagery controls confirm that the cached OSM
    polygon leaves St. John's basin and The Narrows open; the earlier diagnosis
    of an enclosed harbour was incorrect. Eligibility remains policy-dependent.

    Returns 0 rather than raising when the coastline is not downloaded; the
    caller reports coverage as unknown instead of overstating it.
    """
    if not positions:
        return 0
    try:
        from agents.landmask import LandMask
    except ImportError:
        return 0

    lons = [p[0] for p in positions]
    lats = [p[1] for p in positions]
    pad = 0.25
    bounds = [min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad]
    try:
        mask = LandMask.for_footprint(bounds, source="osm", cache_dir=cache_dir)
        surfaces, _ = mask.classify(lons, lats)
    except Exception:
        return 0
    return sum(1 for s in surfaces if s == "water")


def assess(
    products: Sequence[Dict[str, Any]], archive_dir: str, cache_dir: str
) -> List[Dict[str, Any]]:
    """
    Annotates each acquisition with why it is, or is not, scorable.

    Eligibility is a metadata precheck for an open-water experiment. Actual
    measurement additionally requires usable aligned AIS on valid SAR pixels.
    A precheck cannot certify that an evaluation will produce a measurement.

    The swath check is not optional. The recording envelope spans most of
    Atlantic Canada, so "AIS in the time window" almost always finds vessels
    somewhere. Without it this function reported 2026-09-22 and 2026-09-27 as
    scorable on the strength of 85 observations that were all OUTSIDE the
    footprint -- two downloads and two full pipeline runs that could only ever
    come back `scored: False`.
    """
    out = []
    from shapely.geometry import Point

    from agents.region import load_region

    region = load_region()
    for p in products:
        name = p["Name"]
        start = p["ContentDate"]["Start"]
        usable_pol = f"_{REQUIRED_POLARIZATION}_" in name
        in_window = ais_observations_near(archive_dir, start) if usable_pol else []
        positions = within_footprint(in_window, p.get("GeoFootprint"))
        positions = [
            position for position in positions if region.covers(Point(*position))
        ]
        eligible = water_eligible(positions, cache_dir) if positions else 0

        try:
            selected_id = product_id(p.get("Id", ""))
        except ValueError:
            selected_id = None
        if selected_id is None:
            reason = "missing or invalid satellite product ID"
        elif not usable_pol:
            reason = "HH/HV - detector requires VV/VH"
        elif not in_window:
            reason = "no recorded AIS in the +/-5 min window"
        elif not p.get("GeoFootprint"):
            reason = "missing satellite footprint; imaged swath is unknown"
        elif not positions:
            reason = (
                f"{len(in_window)} AIS observations in the window, but none "
                "inside the imaged swath and NL study area"
            )
        elif eligible == 0:
            reason = (
                f"{len(positions)} AIS observations, but none in alert-eligible "
                "water, or shoreline eligibility is unavailable"
            )
        else:
            reason = f"SCORABLE - {eligible} of {len(positions)} AIS obs in open water"

        out.append(
            {
                "product_id": selected_id,
                "acquired": start,
                "name": name,
                "polarization": "VV/VH" if usable_pol else "HH/HV",
                "ais_observations": len(in_window),
                "ais_in_swath": len(positions),
                "ais_in_water": eligible,
                "eligible": bool(selected_id and usable_pol and eligible),
                "scorable": bool(selected_id and usable_pol and eligible),
                "reason": reason,
            }
        )
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(rows),
        "vv_vh": sum(1 for r in rows if r["polarization"] == "VV/VH"),
        "hh_hv": sum(1 for r in rows if r["polarization"] == "HH/HV"),
        "scorable": sum(1 for r in rows if r["scorable"]),
    }


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description="Find scorable Sentinel-1 acquisitions")
    ap.add_argument("--days", type=int, default=12, help="Look-back window.")
    ap.add_argument(
        "--aoi",
        default="eastern_newfoundland",
        help="AOI under configs/aois/ to search. Coastal AOIs carry most of "
        "this region's VV/VH coverage.",
    )
    ap.add_argument(
        "--archive-dir", default=os.path.join(base, "data", "raw", "ais_stream")
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON for automation.")
    args = ap.parse_args()

    aoi = args.aoi if args.aoi.endswith(".geojson") else args.aoi + ".geojson"
    aoi_path = os.path.join(base, "configs", "aois", aoi)
    if not os.path.exists(aoi_path):
        print(f"AOI not found: {aoi_path}", file=sys.stderr)
        sys.exit(2)

    try:
        rows = assess(
            search_acquisitions(aoi_path, days=args.days),
            args.archive_dir,
            os.path.join(base, "data", "reference"),
        )
    except SceneWatchUnavailable as e:
        print(f"Cannot query the catalogue: {e}", file=sys.stderr)
        sys.exit(1)

    counts = summarize(rows)
    if args.json:
        print(json.dumps({"summary": counts, "acquisitions": rows}, indent=2))
    else:
        print(f"  {args.aoi}, last {args.days} days")
        print(
            f"    {counts['total']} acquisitions  "
            f"VV/VH {counts['vv_vh']}  HH/HV {counts['hh_hv']}  "
            f"SCORABLE {counts['scorable']}"
        )
        print()
        for r in rows:
            flag = "  >>" if r["scorable"] else "    "
            print(f"{flag} {r['acquired']}  {r['polarization']}  {r['reason']}")
        if counts["scorable"]:
            print("\n  Score one with:")
            for r in rows:
                if r["scorable"]:
                    print(
                        f"    python -m agents.pipeline --product-id {r['product_id']} "
                        f"--aoi {args.aoi}"
                    )
                    break
        else:
            print(
                "\n  Nothing scorable. The recorder must be running BEFORE an "
                "acquisition;\n  coverage cannot be backfilled."
            )

    # Exit 0 when something is scorable, 3 when not, so a scheduled job can
    # branch on it without parsing text.
    sys.exit(0 if counts["scorable"] else 3)


if __name__ == "__main__":
    main()
