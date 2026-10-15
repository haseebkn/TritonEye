"""Publish replayable C/D processing, vessel screening and optical search evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT
from agents.cd_reference import (
    GUARDS,
    POLICY,
    geometry_diagnostic,
)
from agents.cd_reference import (
    verify as verify_snap,
)
from agents.cd_source_refresh import SEARCH, verify_reservation, verify_search
from agents.hh_hv_reservation import save
from agents.source_reservation import checked, ref

OUTPUT = REPO_ROOT / "docs/evidence/cd_reference_v010.json"
LOCK = REPO_ROOT / "configs/reservations/cd_reference_v010.lock.json"
RESERVATION = REPO_ROOT / "reservations/cd_vessels/v0.1.0/manifest.json"


def mission_decisions(areas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for mission in ("S1C", "S1D"):
        rows = [a for a in areas if a["mission"] == mission]
        complete = len(rows) == 2 and {r["regime"] for r in rows} == {
            "coastal",
            "offshore",
        }
        nominal = complete and all(
            r.get("comparison", {}).get("passed") is True
            and all(
                r["variants"].get(k, {}).get("exit_code") == 0
                and r["variants"][k]["timed_out"] is False
                for k in ("raw", "calibration")
            )
            for r in rows
        )
        noise = complete and all(
            len(r.get("comparison", {}).get("noise", [])) == 2
            and all(b["passed"] is True for b in r["comparison"]["noise"])
            and r["variants"].get("noise", {}).get("exit_code") == 0
            and r["variants"]["noise"]["timed_out"] is False
            for r in rows
        )
        runtime = complete and all(
            set(r["variants"]) == {"raw", "calibration", "noise"}
            and all(
                v["exit_code"] == 0
                and v["timed_out"] is False
                and v["elapsed_limit_exceeded"] is False
                for v in r["variants"].values()
            )
            for r in rows
        )
        results.append(
            {
                "mission": mission,
                "areas": len(rows),
                "nominal_reference_passed": nominal,
                "noise_reference_passed": noise,
                "bounded_execution_passed": runtime,
                "absolute_geolocation_validated": False,
                "model_transfer_validated": False,
                "input_contract_extended": False,
                "training_admitted": False,
            }
        )
    return results


def summarize(
    snap: dict[str, Any], reservation: dict[str, Any], search: dict[str, Any]
) -> dict[str, Any]:
    areas = [
        {
            k: a[k]
            for k in (
                "id",
                "product_id",
                "mission",
                "regime",
                "variants",
                "comparison",
                "failure",
            )
            if k in a
        }
        for a in snap["areas"]
    ]
    targets = []
    for row in search["targets"]:
        targets.append(
            {
                "id": row["id"],
                "sentinel2_status": row["sentinel2"]["receipt"]["status"],
                "landsat_status": row["landsat"]["status"],
                "sentinel2_products": row["sentinel2"]["products"],
                "landsat_products": row["landsat"]["products"],
                "physical_corroboration": False,
            }
        )
    return {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "search_policy": ref(SEARCH),
        "areas": areas,
        "missions": mission_decisions(areas),
        "vessel_reservation": reservation,
        "iceberg_search": targets,
        "labels_admitted": 0,
        "accuracy_truth_admitted": False,
        "prior_development_exposure": True,
        **GUARDS,
    }


def derived_metadata(snap: dict[str, Any]) -> dict[str, Any]:
    prepared = json.loads(checked(snap["prepared"]).read_text())
    metadata = [
        {
            "product_id": p["product_id"],
            "name": p["name"],
            "headers": p["headers"],
            "software": p["software"],
            "auxiliary": [
                {"role": r.get("role"), "name": r.get("name", "").rsplit("/", 1)[-1]}
                for r in p["auxiliary"]
            ],
            "native_contexts": [
                {"id": a["id"], **a["native_context"]} for a in p["areas"]
            ],
        }
        for p in prepared["products"]
    ]
    diagnostics = []
    for product in prepared["products"]:
        for area in product["areas"]:
            row = next(a for a in snap["areas"] if a["id"] == area["id"])
            directory = checked(row["variants"]["raw"]["graph"]).parent
            try:
                diagnostic = geometry_diagnostic(product, area, directory)
            except (ValueError, OSError, StopIteration) as error:
                diagnostic = {
                    "failure": str(error),
                    "absolute_geolocation_validated": False,
                }
            diagnostics.append({"id": area["id"], **diagnostic})
    result: dict[str, Any] = {"metadata": metadata, "geometry_diagnostics": diagnostics}
    if snap.get("retry_of"):
        original = verify_snap(checked(snap["retry_of"]))
        result["first_attempt"] = {
            "source": snap["retry_of"],
            "missions": mission_decisions(original["areas"]),
            "executions": [
                {
                    "id": a["id"],
                    "variants": {
                        k: {
                            f: v[f]
                            for f in (
                                "exit_code",
                                "timed_out",
                                "elapsed_s",
                                "elapsed_limit_exceeded",
                                "log",
                            )
                        }
                        for k, v in a["variants"].items()
                    },
                }
                for a in original["areas"]
            ],
        }
    return result


def publish(snap_path: Path, reservation_path: Path, search_path: Path) -> None:
    if OUTPUT.exists() or LOCK.exists() or RESERVATION.exists():
        raise ValueError("Fresh additive release required")
    snap, reservation, search = (
        verify_snap(snap_path),
        verify_reservation(reservation_path),
        verify_search(search_path),
    )
    result = summarize(snap, reservation, search)
    result.update(derived_metadata(snap))
    result["local_inputs"] = {
        "snap": ref(snap_path),
        "reservation": ref(reservation_path),
        "search": ref(search_path),
    }
    result["limitations"] = [
        "Two acquisitions, four previously inspected training contexts; "
        "not mission-wide validation",
        "Sparse wrappers support pointwise calibration/noise only; "
        "absent pixels never evaluated",
        "Reference agreement is not absolute geolocation "
        "or learned model compatibility",
        "C/D screening is conditional; original A/B input contract "
        "and label admission remain unchanged",
        "Optical footprints/cloud estimates and motion search envelopes "
        "do not confirm same-object identity",
        "No independent review, verified negatives, classifier scores "
        "or model promotion",
    ]
    save(RESERVATION, reservation)
    save(OUTPUT, result)
    save(LOCK, {"report": ref(OUTPUT), "reservation": ref(RESERVATION)})


def validate(report: dict[str, Any]) -> None:
    if (
        any(report.get(k) is not False for k in GUARDS)
        or report["labels_admitted"] != 0
        or report["accuracy_truth_admitted"] is not False
    ):
        raise ValueError("Research guards changed")
    if (
        report["policy"] != ref(POLICY)
        or report["search_policy"] != ref(SEARCH)
        or report["missions"] != mission_decisions(report["areas"])
    ):
        raise ValueError("Policy or mission decision differs")
    if len(report["areas"]) != 4 or len({a["id"] for a in report["areas"]}) != 4:
        raise ValueError("Four distinct comparison contexts required")
    for row in report["iceberg_search"]:
        if row["physical_corroboration"] is not False or any(
            p["same_object_confirmed"] is not False
            or p["pixels_inspected"] is not False
            for k in ("sentinel2_products", "landsat_products")
            for p in row[k]
        ):
            raise ValueError("Metadata cannot confirm an iceberg")


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    report = json.loads(checked(lock["report"]).read_text())
    reservation = json.loads(checked(lock["reservation"]).read_text())
    validate(report)
    if reservation != report["vessel_reservation"]:
        raise ValueError("Reservation publication differs")
    if local:
        refs = report["local_inputs"]
        snap, reservation, search = (
            verify_snap(checked(refs["snap"])),
            verify_reservation(checked(refs["reservation"])),
            verify_search(checked(refs["search"])),
        )
        if any(report[k] != v for k, v in summarize(snap, reservation, search).items()):
            raise ValueError("Public evidence replay differs")
        if any(report[k] != v for k, v in derived_metadata(snap).items()):
            raise ValueError("Metadata or attempt-history replay differs")
    return {
        "verified": True,
        "local": local,
        "missions": report["missions"],
        "reserved_vessel_areas": len(reservation["areas"]),
        "labels_admitted": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--snap", type=Path)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--search", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.publish and args.snap and args.reservation and args.search:
        publish(args.snap, args.reservation, args.search)
    elif args.verify:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose publish with inputs or verify")
