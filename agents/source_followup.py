"""Prepare the six sealed follow-up windows with metadata-first coverage checks."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point, box, shape
from shapely.ops import transform

from agents.artifacts import REPO_ROOT
from agents.calibration import Calibrator
from agents.hh_hv_reservation import save
from agents.reserved_sar import DOWNLOAD, Cdse, locate
from agents.source_error_audit import literal_record
from agents.source_next_batch import verify
from agents.source_reservation import checked, ref
from agents.source_windows import SIZE, local_context, metadata, prepare_product, scope

PROTOCOL = REPO_ROOT / "configs/reservations/source_followup_v010.json"


def selection(local: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["native_window_pixels"] != SIZE:
        raise ValueError("Frozen native size required")
    lock = json.loads(checked_ref_path(protocol["reservation_lock"]).read_text())
    directory = REPO_ROOT / lock["directory"]
    verify(directory, lock["seal_sha256"], local)
    manifest = json.loads((directory / "manifest.json").read_text())
    private = json.loads(checked(manifest["private_selection"]).read_text())
    if len(private["areas"]) != 6 or any(
        a["stream"] != "iceberg_support" for a in private["areas"]
    ):
        raise ValueError("The exact six-area follow-up batch is required")
    return manifest, private


def checked_ref_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Project path required")
    return path


def fresh(output: Path) -> None:
    if output.exists() or not output.resolve().is_relative_to(
        (REPO_ROOT / "data").resolve()
    ):
        raise ValueError("Fresh project data directory required")
    output.mkdir(parents=True)


def original_csv(
    area: dict[str, Any], private: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    ranking = json.loads(checked(private["inputs"]["iip"]).read_text())
    normalized = json.loads(checked(ranking["provenance"]["normalized"]).read_text())
    record = next(
        r for r in normalized["sources"] if r["file"] == area["source"]["source_file"]
    )
    path = checked(record)
    return literal_record(area["source"], path.parent), ref(path)


def coverage(
    geometry: dict[str, Any],
    source: dict[str, Any],
    literal: dict[str, str],
    first_time: str,
    last_time: str,
) -> dict[str, Any]:
    """Coverage under disclosed scenarios; not statistical identity confidence."""
    lon, lat = source["lon"], source["lat"]
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    project = Transformer.from_crs(4326, crs, always_xy=True)
    footprint = transform(project.transform, shape(geometry))
    if (
        not footprint.is_valid
        or footprint.is_empty
        or not footprint.covers(Point(0, 0))
    ):
        raise ValueError("Source centre must lie inside native footprint")
    times = [datetime.fromisoformat(t).timestamp() for t in (first_time, last_time)]
    maximum = max(abs(t - source["time"]) for t in times)
    dy, dx = [
        float(Decimal(10) ** Decimal(Decimal(literal[k]).as_tuple().exponent))
        for k in ("SIGHTING_LATITUDE", "SIGHTING_LONGITUDE")
    ]
    if not 0 < min(dx, dy) <= max(dx, dy) <= 1:
        raise ValueError("Unsupported decimal coordinate representation")
    results = {}
    for name, factor, seconds in (
        ("original", 0, 0),
        ("nearest_rounding", 0.5, 30),
        ("symmetric_one_step", 1, 60),
    ):
        if factor == 0:
            cell = Point(0, 0)
        else:
            # Densification retains the curved WGS84 cell edges after projection.
            cell = transform(
                project.transform,
                box(
                    lon - dx * factor,
                    lat - dy * factor,
                    lon + dx * factor,
                    lat + dy * factor,
                ).segmentize(min(dx, dy) / 16),
            )
        radius = 300 + maximum + seconds
        cell_inside = footprint.covers(cell)
        clearance = float(footprint.boundary.distance(cell)) if cell_inside else 0.0
        region = cell.buffer(radius, quad_segs=256)
        complete = cell_inside and clearance >= radius + 0.01
        results[name] = {
            "motion_and_base_radius_m": radius,
            "cell_clearance_m": clearance,
            "complete": complete,
            "approximate_covered_area_fraction": float(
                footprint.intersection(region).area / region.area
            ),
        }
    return {
        "scenarios": results,
        "all_scenarios_complete": all(r["complete"] for r in results.values()),
        "maximum_native_row_offset_s": maximum,
        "source_association_abstain": not all(r["complete"] for r in results.values()),
        "accuracy_bounds_established": False,
    }


def plan(output: Path) -> None:
    manifest, private = selection(True)
    fresh(output)
    client = Cdse()
    products, failures = [], []
    for pid in dict.fromkeys(a["product_id"] for a in private["areas"]):
        areas = [a for a in private["areas"] if a["product_id"] == pid]
        directory = output / pid
        directory.mkdir()
        area = areas[0]
        base = f"{DOWNLOAD}({pid})/Nodes({area['name']})"
        try:
            listing = json.loads(client.fetch(base + "/Nodes(measurement)/Nodes"))[
                "result"
            ]
            bands: list[dict[str, Any]] = []
            rows: list[dict[str, Any]] = []
            for pol in ("hh", "hv"):
                names = [
                    n["Name"]
                    for n in listing
                    if f"-{pol}-" in n["Name"] and n["Name"].endswith(".tiff")
                ]
                if (
                    len(names) != 1
                    or Path(names[0]).name != names[0]
                    or any(c in names[0] for c in "()\\/")
                ):
                    raise ValueError("Exact safe native band name required")
                xml = client.fetch(
                    base + f"/Nodes(annotation)/Nodes({names[0][:-5]}.xml)/$value"
                )
                path = directory / f"{pol}_annotation.xml"
                path.write_bytes(xml)
                info, gcps = metadata(xml, area, pol)
                bands.append({"polarization": pol.upper(), "annotation": ref(path)})
                pol_rows = []
                for a in areas:
                    col, row, geometry = locate(gcps, (a["lon"], a["lat"]), SIZE)
                    scope(a, geometry)
                    if (
                        min(col, row) < 0
                        or col + SIZE > info["width"]
                        or row + SIZE > info["height"]
                    ):
                        raise ValueError("Native crop outside raster")
                    context = local_context(xml, col, row)
                    literal, source_ref = original_csv(a, private)
                    pol_rows.append(
                        {
                            "id": a["id"],
                            "window": [col, row, SIZE, SIZE],
                            "geometry": geometry,
                            "native_context": context,
                            "source_csv": source_ref,
                            "source_coverage": coverage(
                                geometry,
                                a["source"],
                                literal,
                                context["first_row_native_time_utc"],
                                context["last_row_native_time_utc"],
                            ),
                        }
                    )
                if rows and json.loads(json.dumps(rows)) != json.loads(
                    json.dumps(pol_rows)
                ):
                    raise ValueError("Band footprint, timing or coverage disagreement")
                rows = pol_rows
            products.append(
                {"product_id": pid, "name": area["name"], "bands": bands, "areas": rows}
            )
            print(json.dumps({"planned": pid}), flush=True)
        except Exception as error:
            failures.append({"product_id": pid, "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
    save(
        output / "plan.json",
        {
            "protocol": ref(PROTOCOL),
            "reservation": ref(
                REPO_ROOT / "reservations/source_supported/v0.2.0/manifest.json"
            ),
            "private_selection": manifest["private_selection"],
            "products": products,
            "failures": failures,
            "pixels_read": False,
            "scored": False,
            "network_bytes": client.bytes,
        },
    )
    if failures:
        raise ValueError(
            "Metadata planning incomplete; no pixels read or substitutions"
        )


def validate_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(
        (
            REPO_ROOT / "configs/reservations/source_followup_plan_v010.lock.json"
        ).read_text()
    )
    if ref(path) != lock["plan"]:
        raise ValueError("Trusted pre-pixel plan changed")
    value = json.loads(path.read_text())
    manifest, private = selection()
    if (
        value["protocol"] != ref(PROTOCOL)
        or value["failures"]
        or value["pixels_read"] is not False
        or value["scored"] is not False
    ):
        raise ValueError("Frozen metadata-only plan required")
    if value["private_selection"] != manifest["private_selection"] or value[
        "reservation"
    ] != ref(REPO_ROOT / "reservations/source_supported/v0.2.0/manifest.json"):
        raise ValueError("Planned reservation changed")
    originals = {a["id"]: a for a in private["areas"]}
    rows = [a for p in value["products"] for a in p["areas"]]
    if len(rows) != 6 or {a["id"] for a in rows} != set(originals):
        raise ValueError("Complete six-area inventory required")
    for product in value["products"]:
        if [b["polarization"] for b in product["bands"]] != ["HH", "HV"]:
            raise ValueError("Both native bands required")
        for area in product["areas"]:
            original = originals[area["id"]]
            if (
                product["product_id"] != original["product_id"]
                or product["name"] != original["name"]
            ):
                raise ValueError("Exact reserved product required")
            for band in product["bands"]:
                xml = checked(band["annotation"]).read_bytes()
                info, gcps = metadata(xml, original, band["polarization"].lower())
                col, row, geometry = locate(
                    gcps, (original["lon"], original["lat"]), SIZE
                )
                scope(original, geometry)
                context = local_context(xml, col, row)
                literal, source_ref = original_csv(original, private)
                expected = coverage(
                    geometry,
                    original["source"],
                    literal,
                    context["first_row_native_time_utc"],
                    context["last_row_native_time_utc"],
                )
                if (
                    area["window"] != [col, row, SIZE, SIZE]
                    or min(col, row) < 0
                    or col + SIZE > info["width"]
                    or row + SIZE > info["height"]
                    or json.loads(json.dumps(geometry)) != area["geometry"]
                    or area["source_csv"] != source_ref
                    or area["native_context"] != context
                    or area["source_coverage"] != expected
                ):
                    raise ValueError("Native plan or coverage replay differs")
    return value, private


def prepare(plan_path: Path, output: Path) -> None:
    value, private = validate_plan(plan_path)
    if shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3:
        raise ValueError("Five GiB free space required")
    fresh(output)
    client = Cdse()
    products, failures = [], []
    for planned in value["products"]:

        def check(
            area: dict[str, Any], record: dict[str, Any], band: dict[str, Any]
        ) -> None:
            expected = next(a for a in planned["areas"] if a["id"] == area["id"])
            annotation = next(
                b["annotation"]
                for b in planned["bands"]
                if b["polarization"] == band["polarization"]
            )
            if annotation["sha256"] != band["annotation"]["sha256"] or any(
                json.loads(json.dumps(record[k])) != expected[k]
                for k in ("window", "geometry", "native_context")
            ):
                raise ValueError("Metadata changed after pre-pixel plan")
            record["source_coverage"] = expected["source_coverage"]

        try:
            products.append(
                prepare_product(
                    client,
                    [
                        a
                        for a in private["areas"]
                        if a["product_id"] == planned["product_id"]
                    ],
                    output,
                    check,
                )
            )
            print(
                json.dumps(
                    {"prepared": planned["product_id"], "network_bytes": client.bytes}
                ),
                flush=True,
            )
        except Exception as error:
            failures.append({"product_id": planned["product_id"], "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
        save(
            output / f"progress_{len(products)+len(failures)}.json",
            {
                "plan": ref(plan_path),
                "products": products,
                "failures": failures,
                "scored": False,
            },
        )
    save(
        output / "manifest.json",
        {
            "plan": ref(plan_path),
            "protocol": ref(PROTOCOL),
            "products": products,
            "failures": failures,
            "network_bytes": client.bytes,
            "scored": False,
        },
    )
    if failures:
        raise ValueError("Preparation incomplete; failed reservations retained")


def replay(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    value, private = validate_plan(checked(manifest["plan"]))
    if (
        manifest["scored"] is not False
        or manifest["failures"]
        or manifest["protocol"] != ref(PROTOCOL)
    ):
        raise ValueError("Complete unscored preparation required")
    expected = {a["id"]: a for p in value["products"] for a in p["areas"]}
    originals = {a["id"]: a for a in private["areas"]}
    rows = [a for p in manifest["products"] for a in p["areas"]]
    if len(rows) != 6 or {a["id"] for a in rows} != set(expected):
        raise ValueError("Prepared inventory differs")
    for product in manifest["products"]:
        if (
            product["scored"] is not False
            or product["resampled"] is not False
            or [b["polarization"] for b in product["bands"]] != ["HH", "HV"]
        ):
            raise ValueError("Unscored native two-band inputs required")
        planned = next(
            p for p in value["products"] if p["product_id"] == product["product_id"]
        )
        for area in product["areas"]:
            original = originals[area["id"]]
            if (
                product["name"] != original["name"]
                or product["product_id"] != original["product_id"]
                or any(
                    area[k] != original[k]
                    for k in (
                        "product_id",
                        "split",
                        "geographic_group",
                        "native_acquisition_group",
                        "prior_datatake_exposure",
                    )
                )
                or any(
                    area[k] != expected[area["id"]][k]
                    for k in ("window", "geometry", "native_context", "source_coverage")
                )
            ):
                raise ValueError("Prepared scope or coverage differs")
            with rasterio.open(checked(area["chip"])) as ds:
                values = ds.read()
                if ds.descriptions != (
                    "sigma0_db_hh",
                    "sigma0_db_hv",
                ) or values.shape != (2, SIZE, SIZE):
                    raise ValueError("Native calibrated layout differs")
            checked(area["preview"])
            for index, band in enumerate(product["bands"]):
                if (
                    band["annotation"]["sha256"]
                    != planned["bands"][index]["annotation"]["sha256"]
                ):
                    raise ValueError("Prepared annotation differs")
                dn = np.load(checked(area["bands"][index]["dn"]), allow_pickle=False)
                col, row, _, _ = area["window"]
                calibrated = (
                    Calibrator(str(checked(band["calibration"])))
                    .to_sigma0_db(dn, row, col)
                    .astype("float32")
                )
                if (
                    dn.shape != (SIZE, SIZE)
                    or not (dn > 0).all()
                    or not np.isfinite(calibrated).all()
                    or not np.array_equal(calibrated, values[index])
                ):
                    raise ValueError("Exact native calibration replay failed")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan and args.output:
        plan(args.output)
    elif args.prepare and args.output:
        prepare(args.prepare, args.output)
    elif args.verify:
        print(
            json.dumps(
                {"verified": True, "areas": len(replay(args.verify)), "scored": False}
            )
        )
    else:
        parser.error("Choose plan, prepare or verify")
