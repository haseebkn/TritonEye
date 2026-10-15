"""Additive SNAP time-domain noise replay; preserve the historical line replay."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

from agents.cd_reference import GUARDS, verify
from agents.hh_hv_reservation import save
from agents.optical_lead import POLICY
from agents.snap_reference import bands
from agents.source_followup import fresh
from agents.source_preprocessing import reference_db
from agents.source_reservation import checked, ref
from agents.target_noise_audit import noise_power

SOURCE = "https://raw.githubusercontent.com/senbox-org/microwave-toolbox/38998c9baecfc1d568647abbd6424b054f423fdb/sar-op-calibration/src/main/java/eu/esa/sar/calibration/gpf/Sentinel1RemoveThermalNoiseOp.java"
UTC_SOURCE = "https://raw.githubusercontent.com/senbox-org/snap-engine/14.0.0/snap-core/src/main/java/org/esa/snap/core/datamodel/ProductData.java"


def mjd2000(value: str) -> float:
    """Match ProductData.UTC days + (seconds + micros/1e6)/86400."""
    instant = datetime.fromisoformat(value.removesuffix("Z"))
    delta = instant - datetime(2000, 1, 1)
    return float(delta.days) + (1.0 / 86400.0) * (
        delta.seconds + (1.0 / 1000000.0) * delta.microseconds
    )


def linear(nodes: Any, values: Any, query: Any) -> Any:
    """Ordered piecewise linear interpolation with SNAP's arithmetic order."""
    nodes, values, query = np.asarray(nodes), np.asarray(values), np.asarray(query)
    if (
        nodes.ndim != 1
        or values.shape != nodes.shape
        or not nodes.size
        or not np.isfinite(nodes).all()
        or not np.isfinite(values).all()
        or not np.isfinite(query).all()
        or (np.diff(nodes) <= 0).any()
    ):
        raise ValueError("Finite strictly ordered vectors required")
    if nodes.size == 1:
        return np.full(query.shape, values[0], dtype="float64")
    index = np.clip(np.searchsorted(nodes, query, side="left") - 1, 0, len(nodes) - 2)
    fraction = (query - nodes[index]) / (nodes[index + 1] - nodes[index])
    return values[index] + fraction * (values[index + 1] - values[index])


def timed_noise(annotation: Path, noise: Path, bounds: list[int], pol: str) -> Any:
    """Native IW GRD pilot only; fail if block/vector coverage is unsupported."""
    anno, root = ET.parse(annotation), ET.parse(noise)
    if (
        root.findtext(".//adsHeader/polarisation") != pol
        or anno.findtext(".//thermalNoiseCorrectionPerformed") != "false"
    ):
        raise ValueError("Matching uncorrected polarization required")
    x, y, w, h = bounds
    if min(x, y) < 0 or min(w, h) <= 0:
        raise ValueError("Positive native window required")
    t0 = mjd2000(anno.findtext(".//imageInformation/productFirstLineUtcTime", ""))
    dt = float(anno.findtext(".//azimuthTimeInterval", "nan")) / 86400.0
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive native time interval required")
    rows, cols = np.arange(y, y + h), np.arange(x, x + w)
    query_time = t0 + dt * rows
    vectors = root.findall(".//noiseRangeVector")
    times = np.array([mjd2000(v.findtext("azimuthTime", "")) for v in vectors])
    output = np.full((h, w), np.nan)
    coverage = np.zeros((h, w), dtype="int32")
    for az in root.findall(".//noiseAzimuthVector"):
        ax0, ax1, ay0, ay1 = [
            int(az.findtext(k, "-1"))
            for k in (
                "firstRangeSample",
                "lastRangeSample",
                "firstAzimuthLine",
                "lastAzimuthLine",
            )
        ]
        inside = (
            (rows[:, None] >= ay0)
            & (rows[:, None] <= ay1)
            & (cols[None, :] >= ax0)
            & (cols[None, :] <= ax1)
        )
        if not inside.any():
            continue
        selected = np.flatnonzero((times >= t0 + ay0 * dt) & (times <= t0 + ay1 * dt))
        if not len(selected):
            raise ValueError(
                "Pilot time block has no range vector; unsupported fallback"
            )
        profiles = []
        for index in selected:
            vector = vectors[index]
            nodes = np.fromstring(vector.findtext("pixel", ""), sep=" ")
            lut = (
                np.fromstring(vector.findtext("noiseRangeLut", ""), sep=" ")
                .astype("float32")
                .astype("float64")
            )
            profiles.append(linear(nodes, lut, cols))
        profiles_array = np.stack(profiles)
        ranges = np.stack(
            [
                linear(times[selected], profiles_array[:, c], query_time)
                for c in range(w)
            ],
            axis=1,
        )
        factor = linear(
            np.fromstring(az.findtext("line", ""), sep=" "),
            np.fromstring(az.findtext("noiseAzimuthLut", ""), sep=" ")
            .astype("float32")
            .astype("float64"),
            rows,
        )
        combined = factor[:, None] * ranges
        output[inside] = combined[inside]
        coverage[inside] += 1
    if (coverage != 1).any() or not np.isfinite(output).all() or (output <= 0).any():
        raise ValueError("Complete positive noise coverage required")
    return output


def corrected_power(dn: Any, noise: Any) -> Any:
    """SNAP clips negative residuals only; small positive residuals survive."""
    residual = dn.astype("float64") ** 2 - noise
    return np.where(residual < 0, 1e-5, residual).astype("float32").astype("float64")


def diagnose(report: dict[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(checked(report["prepared"]).read_text())
    results = []
    for product in payload["products"]:
        for area in product["areas"]:
            row = next(a for a in report["areas"] if a["id"] == area["id"])
            directory = checked(row["variants"]["raw"]["graph"]).parent
            corrected = bands(directory / "noise.dim")
            for i, pol in enumerate(("HH", "HV")):
                band = product["bands"][i]
                xml = next(
                    n["source"]
                    for n in product["metadata"]["noise"]
                    if n["polarization"] == pol
                )
                dn = np.load(checked(area["bands"][i]["dn"]), allow_pickle=False)
                old = noise_power(
                    checked(xml), area["window"], pol, snap_precision=True
                )
                timed = timed_noise(
                    checked(band["annotation"]), checked(xml), area["window"], pol
                )
                old_db = reference_db(
                    checked(band["calibration"]),
                    np.sqrt(np.maximum(dn.astype("float64") ** 2 - old, 1e-5)),
                    area["window"],
                )
                line_storage = reference_db(
                    checked(band["calibration"]),
                    np.sqrt(corrected_power(dn, old)),
                    area["window"],
                )
                time_only = reference_db(
                    checked(band["calibration"]),
                    np.sqrt(np.maximum(dn.astype("float64") ** 2 - timed, 1e-5)),
                    area["window"],
                )
                new_power = corrected_power(dn, timed)
                if (new_power <= 0).any():
                    raise ValueError("Zero residual prevents finite dB comparison")
                new_db = reference_db(
                    checked(band["calibration"]), np.sqrt(new_power), area["window"]
                )
                old_delta, delta = np.abs(corrected[i] - old_db), np.abs(
                    corrected[i] - new_db
                )
                worst = np.unravel_index(np.argmax(old_delta), old_delta.shape)
                result = {
                    "id": area["id"],
                    "mission": product["name"][:3],
                    "polarization": pol,
                    "pixels": int(delta.size),
                    "old_max_absolute_db": float(old_delta.max()),
                    "time_replay_max_absolute_db": float(delta.max()),
                    "line_with_snap_storage_max_absolute_db": float(
                        np.abs(corrected[i] - line_storage).max()
                    ),
                    "time_without_snap_storage_max_absolute_db": float(
                        np.abs(corrected[i] - time_only).max()
                    ),
                    "time_replay_within_tolerance": int((delta <= 0.001).sum()),
                    "time_replay_passed": bool(
                        np.isfinite(delta).all() and delta.max() <= 0.001
                    ),
                    "maximum_noise_power_change": float(np.max(np.abs(timed - old))),
                    "old_worst_pixel": {
                        "row": int(worst[0]),
                        "col": int(worst[1]),
                        "dn_power": float(dn[worst]) ** 2,
                        "line_noise_power": float(old[worst]),
                        "time_noise_power": float(timed[worst]),
                        "time_residual_power_float32": float(new_power[worst]),
                        "snap_corrected_db": float(corrected[i][worst]),
                        "line_replay_db": float(old_db[worst]),
                        "time_replay_db": float(new_db[worst]),
                    },
                }
                results.append(result)
    return results


def run(path: Path, output: Path) -> None:
    report = verify(path)
    fresh(output)
    sources = []
    for name, url in (
        ("Sentinel1RemoveThermalNoiseOp.java", SOURCE),
        ("ProductData.java", UTC_SOURCE),
    ):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        destination = output / name
        destination.write_bytes(response.content)
        sources.append({"url": url, "artifact": ref(destination)})
    save(
        output / "report.json",
        {
            "policy": ref(POLICY),
            "snap": ref(path),
            "sources": sources,
            "runtime_calibration_jar": ref(Path("data/cd_calibration.jar")),
            "algorithm": "SNAP14 time interpolation in MJD2000, float32 LUT "
            "values and residual storage, negative-only clipping",
            "bands": diagnose(report),
            "labels_admitted": 0,
            **GUARDS,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snap, args.output)
