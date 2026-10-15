"""Reserve future AIS collection opportunities from official KML plans, not labels."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon, mapping, shape

from agents.artifacts import REPO_ROOT
from agents.cd_training_reservations import freeze_priors
from agents.hh_hv_reservation import save
from agents.neighbour_vessel_batch import POLICY as NEIGHBOUR_POLICY
from agents.neighbour_vessel_batch import expand
from agents.nl_reception import REPORT as RECEPTION
from agents.nl_reception import grid_key, role
from agents.overlap_audit import observations
from agents.provisional_training import GUARDS
from agents.source_reservation import checked, ref
from agents.training_ais_coverage import SELECTION

NS = {"k": "http://www.opengis.net/kml/2.2"}
REPORT = REPO_ROOT / "reservations/targeted_collection/v0.1.0/manifest.json"
LOCK = REPO_ROOT / "configs/reservations/targeted_collection_v010.lock.json"
POLICY = REPO_ROOT / "configs/reservations/targeted_collection_v010.json"
SOURCES = {
    "s1c.kml": "https://sentinels.copernicus.eu/documents/d/sentinel/"
    "s1c_mp_user_20261014t183647_20261105t201100",
    "s1d.kml": "https://sentinels.copernicus.eu/documents/d/sentinel/"
    "s1d_mp_user_20261013t173113_20261102t193900",
}


def instant(value: str) -> datetime:
    """ESA plan times without offsets are UTC; never apply the host timezone."""
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def plans(path: Path) -> list[dict[str, Any]]:
    """Parse only compatible planned modes; invalid relevant metadata fails closed."""
    result = []
    for mark in ET.parse(path).findall(".//k:Placemark", NS):
        fields = {
            node.attrib["name"]: node.findtext("k:value", namespaces=NS)
            for node in mark.findall("k:ExtendedData/k:Data", NS)
        }
        if fields.get("Mode") != "IW" or fields.get("Polarisation") != "DH":
            continue
        if fields.get("SatelliteId") not in {"S1C", "S1D"}:
            continue
        start = instant(str(fields["ObservationTimeStart"]))
        stop = instant(str(fields["ObservationTimeStop"]))
        orbit = int(str(fields["OrbitAbsolute"]))
        datatake = int(str(fields["DatatakeId"]), 16)
        coords = mark.findtext(".//k:coordinates", namespaces=NS)
        if coords is None:
            raise ValueError("Missing plan footprint")
        points = [tuple(map(float, p.split(",")[:2])) for p in coords.split()]
        if not all(
            len(p) == 2 and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in points
        ):
            raise ValueError("Invalid plan coordinates")
        footprint = Polygon(points)
        if (
            not footprint.is_valid
            or footprint.is_empty
            or stop <= start
            or orbit <= 0
            or datatake < 0
        ):
            raise ValueError("Invalid plan timing, identity or geometry")
        result.append(
            {
                "planned_native_group": (
                    f"{fields['SatelliteId']}_{orbit:06d}_{datatake:06X}"
                ),
                "mission": fields["SatelliteId"],
                "mode": "IW",
                "polarization": "HH/HV",
                "start_utc": start.isoformat(),
                "end_utc": stop.isoformat(),
                "relative_orbit": int(str(fields["OrbitRelative"])),
                "planned_footprint": mapping(footprint),
            }
        )
    return result


def reception(
    rows: list[dict[str, Any]], cells: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Recent moving reports guide collection only, not vessel classification."""
    by_grid = {tuple(c["grid"]): c for c in cells}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = by_grid.get(grid_key(row["lon"], row["lat"], 0.1))
        if cell is None or not 0 <= now.timestamp() - row["time"] <= 7 * 86400:
            continue
        try:
            speed = float(row["values"]["speed_knots"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(speed) and 1 <= speed < 102.3:
            grouped[cell["id"]].append(row)
    return [
        c
        | {
            "recent_moving_reports": len(grouped[c["id"]]),
            "recent_moving_identities": len({r["mmsi"] for r in grouped[c["id"]]}),
            "last_moving_report_utc": max(
                (r["timestamp"] for r in grouped[c["id"]]), default=None
            ),
        }
        for c in cells
    ]


def select(
    segments: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    frozen: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for segment in segments:
        group = segment["planned_native_group"]
        if group in unique and unique[group] != segment:
            raise ValueError("Conflicting plan entries for one native group")
        unique[group] = segment
    rankings: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    working = dict(frozen) | {"priors": list(frozen["priors"])}
    for s in sorted(
        unique.values(), key=lambda p: (p["start_utc"], p["planned_native_group"])
    ):
        start, stop = instant(s["start_utc"]), instant(s["end_utc"])
        if start - timedelta(minutes=30) <= now:
            continue
        footprint = shape(s["planned_footprint"])
        eligible = []
        for cell in cells:
            if not footprint.covers(shape(cell["geometry"])):
                continue
            decision = role(cell["geometry"], working, s["planned_native_group"])
            if decision["split_safe"] and cell["recent_moving_reports"] > 0:
                eligible.append(cell | decision)
        if not eligible:
            continue
        eligible.sort(key=lambda c: (c["rank"], c["id"]))
        row = s | {"eligible_cells": eligible, "selected": len(selected) < 3}
        rankings.append(row)
        if len(selected) >= 3:
            continue
        chosen = eligible[0]
        selected.append(
            s
            | GUARDS
            | {
                "id": f"targeted_v010_{len(selected) + 1:02d}",
                "status": "planned_collection_pending_actual_product_and_AIS",
                "split": "train",
                "target_cell": chosen,
                "collection_start_utc": (start - timedelta(minutes=30)).isoformat(),
                "collection_end_utc": (stop + timedelta(minutes=30)).isoformat(),
                "actual_product_id": None,
                "actual_product_name": None,
                "label": None,
                "native_input_verified": False,
                "source_support_verified": False,
                "classifier_scored": False,
                "exploratory_training_admitted": False,
            }
        )
        working["priors"].append(
            {
                "geometry": chosen["geometry"],
                "group": chosen["geographic_group"],
                "split": "train",
                "acquisition_group": s["planned_native_group"],
                "quarantine": False,
            }
        )
    return {"passes": selected, "rankings": rankings, "reserved": len(selected)}


def publish(directory: Path) -> None:
    if REPORT.exists() or LOCK.exists() or (directory / "priors.json").exists():
        raise ValueError("Fresh version required")
    now = datetime.now(timezone.utc)
    frozen = freeze_priors(SELECTION)
    save(directory / "priors.json", frozen)
    base = json.loads(
        (REPO_ROOT / "reservations/cd_vessels/v0.4.0/manifest.json").read_text()
    )
    checked(base["private_inputs"]["snapshot"])
    archive = checked(base["private_inputs"]["snapshot"]).parent / "archive_snapshot"
    rows, stats = observations(archive)
    cells = reception(
        rows,
        expand(
            json.loads(RECEPTION.read_text()),
            frozen,
            json.loads(NEIGHBOUR_POLICY.read_text()),
        ),
        now,
    )
    segments = [s for filename in SOURCES for s in plans(directory / filename)]
    result = select(segments, cells, frozen, now)
    manifest = (
        result
        | GUARDS
        | {
            "version": "0.1.0",
            "reserved_at_utc": now.isoformat(),
            "labels_admitted": 0,
            "imagery_inspected": False,
            "classifier_scored": False,
            "archive": stats,
            "inputs": {
                "policy": ref(POLICY),
                "neighbour_policy": ref(NEIGHBOUR_POLICY),
                "reception": ref(RECEPTION),
                "previous_audit": ref(
                    REPO_ROOT / "reservations/cd_vessels/v0.4.0/manifest.json"
                ),
            },
            "implementation": ref(Path(__file__)),
            "private_priors": ref(directory / "priors.json"),
            "plans": [
                {"url": url, **ref(directory / name)} for name, url in SOURCES.items()
            ],
            "collection_scope": (
                "Existing continuous NL envelope; no MMSI filter or extra recorder"
            ),
            "limits": json.loads(POLICY.read_text())["limits"],
        }
    )
    save(REPORT, manifest)
    save(LOCK, {"manifest": ref(REPORT)})
    print(json.dumps({"reserved": result["reserved"], "labels_admitted": 0}))


def verify(local: bool = False) -> dict[str, Any]:
    manifest = json.loads(checked(json.loads(LOCK.read_text())["manifest"]).read_text())
    checked(manifest["implementation"])
    for reference in manifest["inputs"].values():
        checked(reference)
    if any(manifest.get(k) is not v for k, v in GUARDS.items()):
        raise ValueError("Provisional guard changed")
    if manifest["labels_admitted"] != 0 or manifest["reserved"] != len(
        manifest["passes"]
    ):
        raise ValueError("Invalid reservation count")
    for item in manifest["passes"]:
        if (
            any(item.get(k) is not v for k, v in GUARDS.items())
            or item["label"] is not None
            or item["split"] != "train"
            or item["actual_product_id"] is not None
            or item["classifier_scored"] is not False
        ):
            raise ValueError("Unlabelled training collection required")
    if local:
        frozen = json.loads(checked(manifest["private_priors"]).read_text())
        for path, sha in frozen["prior_metadata_hashes"].items():
            checked({"path": path, "sha256": sha})
        base = json.loads(checked(manifest["inputs"]["previous_audit"]).read_text())
        snapshot_path = checked(base["private_inputs"]["snapshot"])
        archive = snapshot_path.parent / "archive_snapshot"
        for entry in json.loads(snapshot_path.read_text())["files"]:
            checked(
                {
                    "path": (archive / entry["file"]).relative_to(REPO_ROOT).as_posix(),
                    "sha256": entry["sha256"],
                }
            )
        rows, stats = observations(archive)
        now = instant(manifest["reserved_at_utc"])
        cells = reception(
            rows,
            expand(
                json.loads(RECEPTION.read_text()),
                frozen,
                json.loads(NEIGHBOUR_POLICY.read_text()),
            ),
            now,
        )
        replay = select(
            [s for source in manifest["plans"] for s in plans(checked(source))],
            cells,
            frozen,
            now,
        )
        if stats != manifest["archive"] or any(
            manifest[k] != v for k, v in json.loads(json.dumps(replay)).items()
        ):
            raise ValueError("Collection reservation replay differs")
    return {
        "verified": True,
        "local": local,
        "reserved": manifest["reserved"],
        "labels_admitted": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", type=Path)
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish(args.publish)
    else:
        print(json.dumps(verify(args.local)))
