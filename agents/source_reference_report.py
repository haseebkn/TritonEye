"""Publish and replay the provisional SNAP/source audit without changing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_reservation import save
from agents.iceberg_corroboration import optical_summary, repeat_context
from agents.snap_reference import POLICY, bands, compare, inputs
from agents.source_followup import selection
from agents.source_reservation import checked, ref

LOCK = REPO_ROOT / "configs/reservations/snap_corroboration_evidence_v010.lock.json"
NOISE_LOCK = REPO_ROOT / "configs/reservations/snap_noise_floor_v010.lock.json"
FLAGS = (
    "training_approved",
    "independent_review",
    "compatibility_established",
    "classifier_scored",
    "detector_scored",
)


def noise_floor_stats(values: Any, nominal: Any, dn: Any) -> dict[str, Any]:
    if values.shape != nominal.shape or values.shape != dn.shape:
        raise ValueError("Aligned native noise, nominal and DN arrays required")
    finite = np.isfinite(values) & np.isfinite(nominal) & (dn > 0)
    # Both variants use the same calibration LUT, which cancels in the ratio.
    # This is a post-run diagnostic of the observed power pile-up, not a filter.
    power = dn.astype("float64")[finite] ** 2 * 10 ** (
        (values[finite] - nominal[finite]) / 10
    )
    at_floor = np.isclose(power, 0.00001, rtol=0.00001, atol=0)
    return {
        "total_pixels": int(values.size),
        "finite_pixels": int(finite.sum()),
        "pixels_near_reconstructed_power_1e_minus5": int(at_floor.sum()),
        "fraction_of_all_pixels": float(at_floor.sum() / values.size),
        "power_floor_relative_tolerance": 0.00001,
        "noise_db_quantiles_05_50_95": (
            np.quantile(values[finite], [0.05, 0.5, 0.95]).tolist()
            if finite.any()
            else None
        ),
    }


def noise_records(snap: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    prepared = json.loads(checked(snap["prepared"]).read_text())
    native = {a["id"]: a for p in prepared["products"] for a in p["areas"]}
    for area in snap["areas"]:
        variant = area["variants"]["noise"]
        directory = checked(variant["graph"]).parent
        if variant["exit_code"] != 0 or not area.get("comparison", {}).get("passed"):
            records.append({"id": area["id"], "measured": False})
            continue
        values = bands(directory / "noise.dim")
        nominal = bands(directory / "calibration.dim")
        records.append(
            {
                "id": area["id"],
                "measured": True,
                "bands": [
                    {
                        "polarization": pol,
                        **noise_floor_stats(
                            values[i],
                            nominal[i],
                            np.load(
                                checked(native[area["id"]]["bands"][i]["dn"]),
                                allow_pickle=False,
                            ),
                        ),
                    }
                    for i, pol in enumerate(("HH", "HV"))
                ],
            }
        )
    return records


def publish_noise(snap_path: Path, output: Path) -> None:
    if output.resolve().parent != (REPO_ROOT / "docs/evidence").resolve():
        raise ValueError("Public audit evidence directory required")
    snap = json.loads(snap_path.read_text())
    if snap["policy"] != ref(POLICY) or any(snap.get(k) is not False for k in FLAGS):
        raise ValueError("Provisional noise audit required")
    report = {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "execution": ref(snap_path),
        "areas": noise_records(snap),
        **dict.fromkeys(FLAGS, False),
        "interpretation": "Post-run inferred power = DN^2 * 10^((noise_db "
        "- nominal_db)/10); same calibration LUT cancels. Descriptive pile-up "
        "near 1e-5 is not proof of correct physical noise subtraction. "
        "No mask, classifier input recipe or operating threshold selected.",
    }
    save(output, report)
    save(NOISE_LOCK, {"evidence": ref(output)})


def summarize(snap: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    policy = json.loads(POLICY.read_text())
    for report in (snap, evidence):
        if report["policy"] != ref(POLICY) or any(
            report.get(k) is not False
            for k in FLAGS
            if k != "compatibility_established"
        ):
            raise ValueError("Provisional evidence flags or policy changed")
    if (
        snap["compatibility_established"] is not False
        or evidence["labels_modified"] is not False
    ):
        raise ValueError("Compatibility and sealed labels must remain unchanged")
    expected = set(policy["area_ids"])
    for report in (snap, evidence):
        if len(report["areas"]) != 4 or {a["id"] for a in report["areas"]} != expected:
            raise ValueError("Complete four-candidate audit required")
    rows = []
    for area in snap["areas"]:
        support = next(a for a in evidence["areas"] if a["id"] == area["id"])
        variants = {
            k: {
                field: v.get(field)
                for field in (
                    "exit_code",
                    "timed_out",
                    "elapsed_s",
                    "finite",
                    "failure",
                )
            }
            for k, v in area["variants"].items()
        }
        comparison = area.get("comparison")
        for variant in variants.values():
            elapsed = variant["elapsed_s"]
            variant["elapsed_limit_exceeded"] = (
                elapsed > policy["process_timeout_s"] if elapsed is not None else None
            )
        if (
            comparison
            and comparison["passed"]
            and any(
                variants[k]["exit_code"] != 0 or variants[k]["timed_out"]
                for k in ("raw", "calibration")
            )
        ):
            raise ValueError(
                "Failed SNAP execution cannot produce a passing comparison"
            )
        rows.append(
            {
                "id": area["id"],
                "split": area["split"],
                "product_id": area["product_id"],
                "snap_variants": variants,
                "calibration_comparison": comparison,
                "failure": area.get("failure"),
                "noise_sensitivity": area.get("noise_sensitivity"),
                "additional_same_identity_iip_reports_24h": len(
                    support["repeat_reports"]
                ),
                "additional_primary_iip_reports_24h": sum(
                    r["primary_method"] for r in support["repeat_reports"]
                ),
                "repeat_offsets_s": [
                    r["signed_offset_s"] for r in support["repeat_reports"]
                ],
                "optical": support["optical"],
                "independent_confirmation": False,
                "class": "plausible_iceberg",
                "ignore_for_accuracy": True,
            }
        )
    return {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "areas": rows,
        "snap_reference": snap["reference_scope"],
        "snap_unit_interpretation": snap.get("interpretation"),
        "calibration_all_passed": all(
            a["calibration_comparison"] and a["calibration_comparison"]["passed"]
            for a in rows
        ),
        "bounded_execution_all_passed": all(
            v["exit_code"] == 0
            and not v["timed_out"]
            and v["elapsed_limit_exceeded"] is False
            for a in rows
            for v in a["snap_variants"].values()
        ),
        "independently_confirmed": 0,
        "labels_modified": False,
        **dict.fromkeys(FLAGS, False),
        "ais_inventory": evidence["ais_inventory"],
        "limitations": evidence["limitations"]
        + [
            "Native numerical agreement does not establish Kaggle physical "
            "scale or preprocessing equivalence",
            "Sparse source wrappers contain absent pixels outside the exact "
            "frozen subsets; no contextual filters or full-image analysis allowed",
            "Noise variant describes processing sensitivity; no adaptation or "
            "classifier filtering was selected",
        ],
    }


def publish(snap_path: Path, evidence_path: Path, output: Path) -> None:
    snap = json.loads(snap_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    summary = summarize(snap, evidence)
    summary["private_inputs"] = {
        "snap": ref(snap_path),
        "corroboration": ref(evidence_path),
    }
    # SNAP's .dim and ENVI headers control band type/nodata interpretation too.
    # Bind every generated output, including these sidecars, without rewriting
    # the original execution report or any native input.
    artifacts: dict[str, dict[str, str]] = {}
    for area in snap["areas"]:
        for variant in area["variants"].values():
            directory = checked(variant["graph"]).parent
            for dim in directory.glob("*.dim"):
                for path in [dim, *dim.with_suffix(".data").rglob("*")]:
                    if path.is_file():
                        record = ref(path)
                        artifacts[record["path"]] = record
    summary["snap_outputs"] = sorted(artifacts.values(), key=lambda r: r["path"])
    if output.resolve().parent != (REPO_ROOT / "docs/evidence").resolve():
        raise ValueError("Public audit evidence directory required")
    save(output, summary)
    save(LOCK, {"evidence": ref(output)})


def verify(local: bool) -> None:
    public_path = checked(json.loads(LOCK.read_text())["evidence"])
    report = json.loads(public_path.read_text())
    if any(report.get(k) is not False for k in FLAGS):
        raise ValueError("Provisional audit approval flags changed")
    noise = json.loads(
        checked(json.loads(NOISE_LOCK.read_text())["evidence"]).read_text()
    )
    if any(noise.get(k) is not False for k in FLAGS) or noise["policy"] != ref(POLICY):
        raise ValueError("Provisional noise audit changed")
    checked(report["policy"])
    if local:
        _, _, rows = inputs()
        snap = json.loads(checked(report["private_inputs"]["snap"]).read_text())
        if (
            noise["execution"] != report["private_inputs"]["snap"]
            or noise_records(snap) != noise["areas"]
        ):
            raise ValueError("Noise floor replay differs")
        evidence = json.loads(
            checked(report["private_inputs"]["corroboration"]).read_text()
        )
        expected = summarize(snap, evidence)
        expected["private_inputs"] = report["private_inputs"]
        expected["snap_outputs"] = report["snap_outputs"]
        if expected != report:
            raise ValueError("Public evidence summary differs from private receipts")
        for artifact in report["snap_outputs"]:
            checked(artifact)
        for area in snap["areas"]:
            row = next(r for r in rows if r["id"] == area["id"])
            if area["source_chip"] != row["chip"] or area["window"] != row["window"]:
                raise ValueError("SNAP source identity/window changed")
            for variant in area["variants"].values():
                checked(variant["log"])
                checked(variant["graph"])
                for file in variant.get("files", []):
                    checked(file)
            if area.get("comparison"):
                directory = checked(area["variants"]["raw"]["graph"]).parent
                dn = np.stack(
                    [
                        np.load(checked(b["dn"]), allow_pickle=False)
                        for b in row["bands"]
                    ]
                )
                with rasterio.open(checked(row["chip"])) as ds:
                    values = ds.read()
                replayed = compare(
                    bands(directory / "raw.dim", True),
                    bands(directory / "calibration.dim"),
                    dn,
                    values,
                    json.loads(POLICY.read_text())["calibration_tolerance_db"],
                )
                if replayed != area["comparison"]:
                    raise ValueError("SNAP pixel comparison replay differs")
        for area in evidence["areas"]:
            optical = optical_summary(area["optical_receipt"], area["lon"], area["lat"])
            if optical != area["optical"]:
                raise ValueError("Optical catalogue replay differs")
        normalized = json.loads(checked(evidence["normalized_iip"]).read_text())
        _, private = selection()
        for source in normalized["sources"]:
            checked(source)
        for area in evidence["areas"]:
            original = next(a for a in private["areas"] if a["id"] == area["id"])
            if area["source"] != original["source"]:
                raise ValueError("Original IIP source differs")
            from datetime import datetime

            instant = datetime.fromisoformat(area["native_time_utc"]).timestamp()
            repeats = repeat_context(
                area["source"],
                normalized["observations"],
                instant,
                area["lon"],
                area["lat"],
                json.loads(POLICY.read_text())["repeat_iip_window_hours"] * 3600,
            )
            if repeats != area["repeat_reports"]:
                raise ValueError("IIP repeat-source replay differs")
        checked(snap["metadata"])
    print(
        json.dumps(
            {
                "verified": True,
                "local": local,
                "evidence_sha256": sha256_file(public_path),
                "training_approved": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap", type=Path)
    parser.add_argument("--corroboration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--noise-floor", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.local)
    elif args.noise_floor and args.snap and args.output:
        publish_noise(args.snap, args.output)
    elif args.snap and args.corroboration and args.output:
        publish(args.snap, args.corroboration, args.output)
    else:
        parser.error("Choose publish or verify")
