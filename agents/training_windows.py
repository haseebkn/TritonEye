"""Prepare the two training reservations with frozen native uncertainty coverage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from agents.artifacts import REPO_ROOT
from agents.calibration import Calibrator
from agents.hh_hv_reservation import save
from agents.provisional_training import DIRECTORY, verify
from agents.reserved_sar import DOWNLOAD, Cdse, locate
from agents.source_followup import coverage, fresh, original_csv
from agents.source_reservation import checked, ref
from agents.source_windows import SIZE, local_context, metadata, prepare_product, scope

PROTOCOL = REPO_ROOT / "configs/reservations/training_windows_v010.json"


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def selection() -> tuple[dict[str, Any], dict[str, Any]]:
    verify(local=True)
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    private = json.loads(checked(manifest["private_selection"]).read_text())
    if (
        protocol["native_window_pixels"] != SIZE
        or len(private["areas"]) != 2
        or any(
            a["split"] != "train" or a["stream"] != "iceberg_support"
            for a in private["areas"]
        )
    ):
        raise ValueError("Exact two training-only iceberg opportunities required")
    return manifest, private


def planned_area(
    xml: bytes, area: dict[str, Any], private: dict[str, Any], pol: str
) -> dict[str, Any]:
    info, gcps = metadata(xml, area, pol)
    col, row, geometry = locate(gcps, (area["lon"], area["lat"]), SIZE)
    scope(area, geometry)
    if min(col, row) < 0 or col + SIZE > info["width"] or row + SIZE > info["height"]:
        raise ValueError("Reserved crop outside native raster")
    context = local_context(xml, col, row)
    literal, reference = original_csv(area, private)
    result: dict[str, Any] = normalized(
        {
            "id": area["id"],
            "window": [col, row, SIZE, SIZE],
            "geometry": geometry,
            "native_context": context,
            "source_csv": reference,
            "source_coverage": coverage(
                geometry,
                area["source"],
                literal,
                context["first_row_native_time_utc"],
                context["last_row_native_time_utc"],
            ),
        }
    )
    return result


def plan(output: Path) -> None:
    manifest, private = selection()
    fresh(output)
    client = Cdse()
    products, failures = [], []
    for area in private["areas"]:
        directory = output / area["product_id"]
        directory.mkdir()
        base = f"{DOWNLOAD}({area['product_id']})/Nodes({area['name']})"
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
                    raise ValueError("Exact safe native band inventory required")
                xml = client.fetch(
                    base + f"/Nodes(annotation)/Nodes({names[0][:-5]}.xml)/$value"
                )
                path = directory / f"{pol}_annotation.xml"
                path.write_bytes(xml)
                bands.append({"polarization": pol.upper(), "annotation": ref(path)})
                row = planned_area(xml, area, private, pol)
                if rows and row != rows[0]:
                    raise ValueError("HH/HV native window or coverage disagreement")
                rows = [row]
            products.append(
                {
                    "product_id": area["product_id"],
                    "name": area["name"],
                    "bands": bands,
                    "areas": rows,
                }
            )
            print(
                json.dumps({"planned": area["id"], "coverage": row["source_coverage"]}),
                flush=True,
            )
        except Exception as error:
            failures.append({"product_id": area["product_id"], "error": str(error)})
            print(json.dumps(failures[-1]), flush=True)
    save(
        output / "plan.json",
        {
            "protocol": ref(PROTOCOL),
            "reservation": ref(DIRECTORY / "manifest.json"),
            "private_selection": manifest["private_selection"],
            "products": products,
            "failures": failures,
            "pixels_read": False,
            "scored": False,
            "network_bytes": client.bytes,
        },
    )
    if failures:
        raise ValueError("Metadata preparation failed; no pixels read or substitutions")


def validate_plan(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, private = selection()
    value = json.loads(path.read_text())
    if (
        value["protocol"] != ref(PROTOCOL)
        or value["reservation"] != ref(DIRECTORY / "manifest.json")
        or value["private_selection"] != manifest["private_selection"]
        or value["failures"]
        or value["pixels_read"] is not False
        or value["scored"] is not False
    ):
        raise ValueError("Frozen complete metadata-only plan required")
    expected = {a["id"]: a for a in private["areas"]}
    rows = [a for p in value["products"] for a in p["areas"]]
    if len(rows) != 2 or {a["id"] for a in rows} != set(expected):
        raise ValueError("Complete two-area plan required")
    for product in value["products"]:
        if [b["polarization"] for b in product["bands"]] != ["HH", "HV"]:
            raise ValueError("Both native bands required")
        for area in product["areas"]:
            original = expected[area["id"]]
            if (
                product["product_id"] != original["product_id"]
                or product["name"] != original["name"]
            ):
                raise ValueError("Exact reserved product required")
            for band in product["bands"]:
                if (
                    planned_area(
                        checked(band["annotation"]).read_bytes(),
                        original,
                        private,
                        band["polarization"].lower(),
                    )
                    != area
                ):
                    raise ValueError("Native plan or source coverage replay differs")
    return value, private


def prepare(plan_path: Path, output: Path) -> None:
    value, private = validate_plan(plan_path)
    if shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3:
        raise ValueError("Five GiB workspace headroom required")
    fresh(output)
    client = Cdse()
    products, failures = [], []
    for planned in value["products"]:

        def check(
            area: dict[str, Any], record: dict[str, Any], band: dict[str, Any]
        ) -> None:
            expected = next(a for a in planned["areas"] if a["id"] == area["id"])
            reference = next(
                b["annotation"]
                for b in planned["bands"]
                if b["polarization"] == band["polarization"]
            )
            if reference["sha256"] != band["annotation"]["sha256"]:
                raise ValueError("Metadata changed after pre-pixel plan")
            if any(
                normalized(record[k]) != expected[k]
                for k in ("window", "geometry", "native_context")
            ):
                raise ValueError("Native window moved after planning")
            record["source_coverage"] = expected["source_coverage"]

        try:
            areas = [
                a for a in private["areas"] if a["product_id"] == planned["product_id"]
            ]
            products.append(prepare_product(client, areas, output, check))
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
            {"products": products, "failures": failures, "scored": False},
        )
    save(
        output / "manifest.json",
        {
            "protocol": ref(PROTOCOL),
            "plan": ref(plan_path),
            "products": products,
            "failures": failures,
            "network_bytes": client.bytes,
            "scored": False,
        },
    )
    if failures:
        raise ValueError("Preparation incomplete; failures retained")


def replay(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    value, private = validate_plan(checked(manifest["plan"]))
    if (
        manifest["protocol"] != ref(PROTOCOL)
        or manifest["failures"]
        or manifest["scored"] is not False
    ):
        raise ValueError("Complete unscored native preparation required")
    originals = {a["id"]: a for a in private["areas"]}
    plans = {a["id"]: a for p in value["products"] for a in p["areas"]}
    rows = [a for p in manifest["products"] for a in p["areas"]]
    if len(rows) != 2 or {a["id"] for a in rows} != set(originals):
        raise ValueError("Complete reserved preparation required")
    for product in manifest["products"]:
        if (
            product["scored"] is not False
            or product["resampled"] is not False
            or [b["polarization"] for b in product["bands"]] != ["HH", "HV"]
        ):
            raise ValueError("Unscored native HH/HV required")
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
            ):
                raise ValueError("Prepared identity or role changed")
            if any(
                area[k] != plans[area["id"]][k]
                for k in ("window", "geometry", "native_context", "source_coverage")
            ):
                raise ValueError("Prepared window or coverage changed")
            scope(original, area["geometry"])
            with rasterio.open(checked(area["chip"])) as ds:
                values = ds.read()
                if ds.descriptions != (
                    "sigma0_db_hh",
                    "sigma0_db_hv",
                ) or values.shape != (2, SIZE, SIZE):
                    raise ValueError("Calibrated native layout changed")
            checked(area["preview"])
            for index, band in enumerate(product["bands"]):
                xml = checked(band["annotation"]).read_bytes()
                if (
                    planned_area(xml, original, private, band["polarization"].lower())
                    != plans[area["id"]]
                ):
                    raise ValueError("Prepared metadata plan differs")
                dn = np.load(checked(area["bands"][index]["dn"]), allow_pickle=False)
                col, row, _, _ = area["window"]
                result = (
                    Calibrator(str(checked(band["calibration"])))
                    .to_sigma0_db(dn, row, col)
                    .astype("float32")
                )
                if (
                    dn.shape != (SIZE, SIZE)
                    or not (dn > 0).all()
                    or not np.array_equal(result, values[index])
                    or not np.isfinite(result).all()
                ):
                    raise ValueError("Exact native calibration replay failed")
    return rows


def main() -> None:
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


if __name__ == "__main__":
    main()
