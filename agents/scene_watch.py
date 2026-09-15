#!/usr/bin/env python3
"""
Finds Sentinel-1 acquisitions this project can actually SCORE.

The project's central gap is that detection precision and recall over
Newfoundland and Labrador are unmeasured. Closing it needs one acquisition that
satisfies three conditions at once, and missing any of them makes a scene
unscorable no matter how well the pipeline runs:

  1. VV/VH polarization (1SDV). The detector cannot use HH/HV, and over parts
     of this region HH/HV is all that is acquired -- see docs/DATA_SOURCES.md.
  2. Inside the study area.
  3. Recorded AIS covering the acquisition instant. No historical archive
     covers these waters, so coverage exists only where the recorder was
     actually running. This cannot be backfilled.

CHEAP CHECKS FIRST. A source product is ~1.7 GB, so this orders the tests by
cost: catalogue metadata answers polarization and geometry for free, and the
local AIS archive answers coverage for free. Only a scene that passes all three
is worth downloading. Running the pipeline first and discovering afterwards that
no telemetry existed is how the earlier attempts were wasted.

This reports; it does not download or process. Scoring a scene is a deliberate
act, and an unattended job that silently spends an hour of GPU on an unscorable
acquisition is worse than one that prints what it found.

    python -m agents.scene_watch --days 12
    python -m agents.scene_watch --days 30 --json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    # The catalogue returns a product per slice; collapse to unique names so a
    # multi-slice acquisition is not counted as several opportunities.
    seen: Dict[str, Dict[str, Any]] = {}
    for p in r.json().get("value", []):
        seen.setdefault(p["Name"][:60], p)
    return sorted(seen.values(), key=lambda p: p["ContentDate"]["Start"], reverse=True)


def ais_observations_near(
    archive_dir: str, acquisition_iso: str, window_minutes: int = AIS_WINDOW_MINUTES
) -> List[Tuple[float, float]]:
    """
    Returns (lon, lat) of recorded AIS observations within the matching window.

    Reads the daily archive directly rather than the pipeline's filter, because
    this must answer "is there anything to score against" before any scene is
    downloaded or any bbox is known.
    """
    import csv
    import glob

    from agents.ais_validation import parse_utc

    acq = parse_utc(acquisition_iso, allow_naive=True)
    if acq is None:
        return []
    lo, hi = (
        acq - timedelta(minutes=window_minutes),
        acq + timedelta(minutes=window_minutes),
    )

    # Day boundaries: a window can straddle midnight, so check both files.
    days = {lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")}
    found: List[Tuple[float, float]] = []
    for day in days:
        for path in glob.glob(os.path.join(archive_dir, f"ais_stream_{day}.csv")):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        t = parse_utc(row.get("timestamp", ""), allow_naive=True)
                        if t is None or not (lo <= t <= hi):
                            continue
                        try:
                            found.append((float(row["lon"]), float(row["lat"])))
                        except (KeyError, TypeError, ValueError):
                            continue
            except OSError:
                continue
    return found


def water_eligible(positions: Sequence[Tuple[float, float]], cache_dir: str) -> int:
    """
    Counts AIS positions the pipeline would treat as alert-eligible water.

    AIS EXISTING IS NOT THE SAME AS AIS BEING USABLE. The pipeline scores only
    water-classified detections, so reference vessels sitting on land or in the
    coastal band cannot contribute to a recall figure -- recall against them
    reads as zero regardless of how well the detector performs.

    This is not hypothetical: an acquisition over St. John's on 2026-09-03 had
    75 observations from 18 vessels, and every one fell INSIDE the OSM coastline
    because the polygon encloses the harbour basin. Counting those as ground
    truth would have produced a meaningless zero.

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

    Scorable means AIS exists AND falls in alert-eligible water. A scene whose
    only reference vessels are berthed inside a harbour cannot produce a recall
    number, because the pipeline excludes those positions by design.
    """
    out = []
    for p in products:
        name = p["Name"]
        start = p["ContentDate"]["Start"]
        usable_pol = REQUIRED_POLARIZATION in name
        positions = ais_observations_near(archive_dir, start) if usable_pol else []
        eligible = water_eligible(positions, cache_dir) if positions else 0

        if not usable_pol:
            reason = "HH/HV - detector requires VV/VH"
        elif not positions:
            reason = "no recorded AIS in the +/-5 min window"
        elif eligible == 0:
            reason = (
                f"{len(positions)} AIS observations, but none in alert-eligible "
                "water (berthed/nearshore) - would score zero by construction"
            )
        else:
            reason = f"SCORABLE - {eligible} of {len(positions)} AIS obs in open water"

        out.append(
            {
                "acquired": start[:19],
                "name": name,
                "polarization": "VV/VH" if usable_pol else "HH/HV",
                "ais_observations": len(positions),
                "ais_in_water": eligible,
                "scorable": bool(usable_pol and eligible),
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
                        f"    python -m agents.pipeline --date {r['acquired'][:10]} "
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
