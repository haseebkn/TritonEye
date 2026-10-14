"""Run bounded SNAP pixel comparisons on four sealed development windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psutil
import rasterio

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.reserved_sar import DOWNLOAD, Cdse
from agents.source_followup import fresh, replay
from agents.source_followup_annotations import verify
from agents.source_preprocessing import sparse_measurement
from agents.source_reservation import checked, ref
from agents.statoil_baseline import available_commit

POLICY = REPO_ROOT / "configs/reservations/snap_corroboration_v020.json"


def inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    policy = json.loads(POLICY.read_text())
    lock_path = (REPO_ROOT / policy["collection_lock"]).resolve()
    if not lock_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Project collection lock required")
    lock = json.loads(lock_path.read_text())
    directory = (REPO_ROOT / lock["directory"]).resolve()
    verify(directory, lock["seal_sha256"], False)
    collection = json.loads((directory / "collection.json").read_text())
    prepared = checked(policy["prepared"])
    rows = replay(prepared)
    selected = [r for r in rows if r["id"] in policy["area_ids"]]
    if len(selected) != 4 or {r["split"] for r in selected} != {"train", "validation"}:
        raise ValueError("Exact four development candidates required")
    for row in selected:
        labels = next(a for a in collection["areas"] if a["id"] == row["id"])
        if (
            len(labels["objects"]) != 1
            or labels["objects"][0]["class"] != "plausible_iceberg"
        ):
            raise ValueError("Exact provisional iceberg interpretations required")
    return policy, json.loads(prepared.read_text()), selected


def graph(manifest: Path, bounds: list[int], destination: Path, variant: str) -> str:
    """Only pointwise operations; subset output excludes all absent pixels."""
    if variant not in {"raw", "calibration", "noise"}:
        raise ValueError("Unknown SNAP variant")
    if len(bounds) != 4 or min(bounds[:2]) < 0 or bounds[2:] != [384, 384]:
        raise ValueError("Exact native 384 window required")
    root = ET.Element("graph", id="native_reference")
    ET.SubElement(root, "version").text = "1.0"

    def node(name: str, operator: str, source: str | None, **params: Any) -> None:
        element = ET.SubElement(root, "node", id=name)
        ET.SubElement(element, "operator").text = operator
        sources = ET.SubElement(element, "sources")
        if source:
            ET.SubElement(sources, "sourceProduct", refid=source)
        parameters = ET.SubElement(element, "parameters")
        for key, value in params.items():
            ET.SubElement(parameters, key).text = str(value)

    node("read", "Read", None, file=manifest.as_posix())
    source = "read"
    if variant == "noise":
        node(
            "noise",
            "ThermalNoiseRemoval",
            source,
            selectedPolarisations="HH,HV",
            removeThermalNoise="true",
        )
        source = "noise"
    if variant != "raw":
        node(
            "calibrate",
            "Calibration",
            source,
            selectedPolarisations="HH,HV",
            outputSigmaBand="true",
            outputImageScaleInDb="true",
        )
        source = "calibrate"
    node(
        "subset",
        "Subset",
        source,
        region=",".join(map(str, bounds)),
        copyMetadata="true",
    )
    node(
        "write", "Write", "subset", file=destination.as_posix(), formatName="BEAM-DIMAP"
    )
    return ET.tostring(root, encoding="unicode")


def compare(
    raw: Any, calibrated: Any, expected_dn: Any, expected_db: Any, tolerance: float
) -> dict[str, Any]:
    if (
        raw.shape != expected_dn.shape
        or calibrated.shape != expected_db.shape
        or raw.shape != (2, 384, 384)
    ):
        raise ValueError("SNAP native shape/band inventory differs")
    if not np.array_equal(raw, expected_dn):
        raise ValueError("SNAP raw pixels differ from exact original DN")
    if not np.isfinite(calibrated).all() or not np.isfinite(expected_db).all():
        raise ValueError("SNAP nonfinite calibration output")
    result = []
    for i, pol in enumerate(("HH", "HV")):
        delta = calibrated[i].astype("float64") - expected_db[i]
        result.append(
            {
                "polarization": pol,
                "pixels": int(delta.size),
                "max_absolute_db": float(np.abs(delta).max()),
                "mean_signed_db": float(delta.mean()),
                "rmse_db": float(np.sqrt(np.square(delta).mean())),
                "passed": bool(np.abs(delta).max() <= tolerance),
            }
        )
    return {
        "raw_pixels_equal": True,
        "bands": result,
        "passed": all(b["passed"] for b in result),
    }


def fetch(output: Path) -> None:
    policy, manifest, rows = inputs()
    fresh(output)
    client, products = Cdse(), []
    ids = {r["product_id"] for r in rows}
    for product in manifest["products"]:
        if product["product_id"] not in ids:
            continue
        directory = output / product["product_id"]
        directory.mkdir()
        base = f"{DOWNLOAD}({product['product_id']})/Nodes({product['name']})"
        path = directory / "manifest.safe"
        path.write_bytes(client.fetch(base + "/Nodes(manifest.safe)/$value"))
        noises = []
        for band in product["bands"]:
            name = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
            name = "noise-" + name[:-5] + ".xml"
            noise = directory / name
            noise.write_bytes(
                client.fetch(
                    base
                    + "/Nodes(annotation)"
                    + f"/Nodes(calibration)/Nodes({name})/$value"
                )
            )
            if (
                ET.parse(noise).findtext(".//adsHeader/polarisation", "").upper()
                != band["polarization"]
            ):
                raise ValueError("Noise polarization mismatch")
            noises.append({"polarization": band["polarization"], "source": ref(noise)})
        products.append(
            {
                "product_id": product["product_id"],
                "manifest": ref(path),
                "noise": noises,
            }
        )
        save(
            output / f"progress_{len(products)}.json",
            {"products": products, "network_bytes": client.bytes},
        )
    save(
        output / "metadata.json",
        {
            "policy": ref(POLICY),
            "prepared": policy["prepared"],
            "products": products,
            "network_bytes": client.bytes,
        },
    )


def wrapper(
    product: dict[str, Any],
    area: dict[str, Any],
    metadata: dict[str, Any],
    output: Path,
) -> Path:
    safe = output / str(product["name"])
    (safe / "measurement").mkdir(parents=True)
    (safe / "annotation/calibration").mkdir(parents=True)
    shutil.copyfile(checked(metadata["manifest"]), safe / "manifest.safe")
    for i, band in enumerate(product["bands"]):
        name = band["measurement_url"].split("/Nodes(")[-1].split(")")[0]
        shutil.copyfile(
            checked(band["annotation"]), safe / "annotation" / (name[:-5] + ".xml")
        )
        shutil.copyfile(
            checked(band["calibration"]),
            safe / "annotation/calibration" / ("calibration-" + name[:-5] + ".xml"),
        )
        noise = next(
            n for n in metadata["noise"] if n["polarization"] == band["polarization"]
        )
        path = checked(noise["source"])
        shutil.copyfile(path, safe / "annotation/calibration" / path.name)
        dn = np.load(checked(area["bands"][i]["dn"]), allow_pickle=False)
        sparse_measurement(
            safe / "measurement" / name,
            band["width"],
            band["height"],
            [(area["window"], dn)],
        )
    return safe / "manifest.safe"


def execute(gpt: Path, arguments: list[str], log: Path, timeout: int) -> dict[str, Any]:
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                [str(gpt), *arguments],
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            deadline = start + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired([str(gpt), *arguments], timeout)
                try:
                    code = process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            timed_out = False
        except subprocess.TimeoutExpired:
            # gpt is a launcher: stop its JVM descendants too, only for this PID.
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            process.kill()
            for child in reversed(descendants):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.wait()
            psutil.wait_procs(descendants, timeout=5)
            code, timed_out = None, True
    elapsed = time.monotonic() - start
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "elapsed_limit_exceeded": elapsed > timeout,
        "log": ref(log),
    }


def bands(path: Path, raw: bool = False) -> Any:
    try:
        description = ET.parse(path)
    except (OSError, ET.ParseError) as error:
        raise ValueError("SNAP DIMAP metadata missing/invalid") from error
    arrays = []
    for pol in ("HH", "HV"):
        name = f"Amplitude_{pol}" if raw else f"Sigma0_{pol}_db"
        entries = [
            b
            for b in description.findall(".//Spectral_Band_Info")
            if b.findtext("BAND_NAME", "").casefold() == name.casefold()
        ]
        files = [
            p
            for p in path.with_suffix(".data").glob("*.img")
            if p.stem.casefold() == name.casefold()
        ]
        if len(files) != 1 or len(entries) != 1:
            raise ValueError(f"SNAP band output missing/ambiguous: {name}")
        band = entries[0]
        unit = band.findtext("PHYSICAL_UNIT", "").lower()
        if (
            float(band.findtext("SCALING_FACTOR", "nan")) != 1
            or float(band.findtext("SCALING_OFFSET", "nan")) != 0
            or band.findtext("LOG10_SCALED", "").lower() != "false"
        ):
            raise ValueError("Unsupported DIMAP scaling; no guessed conversion")
        if unit not in ({"amplitude"} if raw else {"intensity", "intensity_db"}):
            raise ValueError("Unsupported SNAP physical unit")
        with rasterio.open(files[0]) as src:
            if (src.height, src.width, src.count) != (384, 384, 1):
                raise ValueError("SNAP subset dimensions differ")
            values = src.read(1, masked=True).astype("float64").filled(np.nan)
        if band.findtext("NO_DATA_VALUE_USED", "").lower() == "true":
            nodata = float(band.findtext("NO_DATA_VALUE", "nan"))
            values[values == nodata] = np.nan
        if not raw and unit == "intensity":
            # SNAP may append _dB while declaring/storing linear intensity.
            # The explicit unit, never a filename or a fitted distribution,
            # determines the standard sigma0 conversion to the comparison unit.
            valid = np.isfinite(values) & (values > 0)
            converted = np.full(values.shape, np.nan)
            converted[valid] = 10 * np.log10(values[valid])
            values = converted
        arrays.append(values)
    return np.stack(arrays)


def reinterpret(execution: Path, output: Path) -> None:
    """Replay existing SNAP output with explicit units; never rerun/change pixels."""
    policy, _, rows = inputs()
    original = json.loads(execution.read_text())
    if original["policy"] != ref(POLICY) or original["prepared"] != policy["prepared"]:
        raise ValueError("Execution provenance differs")
    if {a["id"] for a in original["areas"]} != set(policy["area_ids"]):
        raise ValueError("Complete original SNAP execution required")
    fresh(output)
    result = json.loads(execution.read_text())
    result["original_execution"] = ref(execution)
    result["interpretation"] = (
        "DIMAP PHYSICAL_UNIT controls conversion: positive linear intensity "
        "to sigma0 dB with 10*log10; amplitude unchanged; "
        "unknown/nonidentity scaling refused"
    )
    for area in result["areas"]:
        row = next(r for r in rows if r["id"] == area["id"])
        area.pop("failure", None)
        area.pop("comparison", None)
        area.pop("noise_sensitivity", None)
        for variant in area["variants"].values():
            checked(variant["graph"])
            checked(variant["log"])
            for artifact in variant.get("files", []):
                checked(artifact)
        directory = checked(area["variants"]["raw"]["graph"]).parent
        for name, variant in area["variants"].items():
            variant["elapsed_limit_exceeded"] = (
                variant["elapsed_s"] > policy["process_timeout_s"]
            )
            if variant["exit_code"] == 0:
                values = bands(directory / f"{name}.dim", raw=name == "raw")
                variant["finite"] = bool(np.isfinite(values).all())
                variant["unit_metadata"] = ref(directory / f"{name}.dim")
        try:
            if any(
                area["variants"][k]["exit_code"] != 0 for k in ("raw", "calibration")
            ):
                raise ValueError("Failed SNAP graph")
            dn = np.stack(
                [np.load(checked(b["dn"]), allow_pickle=False) for b in row["bands"]]
            )
            with rasterio.open(checked(row["chip"])) as ds:
                expected = ds.read()
            nominal = bands(directory / "calibration.dim")
            area["comparison"] = compare(
                bands(directory / "raw.dim", True),
                nominal,
                dn,
                expected,
                policy["calibration_tolerance_db"],
            )
            noise = bands(directory / "noise.dim")
            area["noise_sensitivity"] = []
            for i, pol in enumerate(("HH", "HV")):
                valid = np.isfinite(noise[i]) & np.isfinite(nominal[i])
                delta = noise[i][valid] - nominal[i][valid]
                area["noise_sensitivity"].append(
                    {
                        "polarization": pol,
                        "finite_pixels": int(np.isfinite(noise[i]).sum()),
                        "total_pixels": int(noise[i].size),
                        "median_delta_db": (
                            float(np.median(delta)) if delta.size else None
                        ),
                        "max_absolute_delta_db": (
                            float(np.abs(delta).max()) if delta.size else None
                        ),
                    }
                )
        except (ValueError, rasterio.errors.RasterioIOError) as error:
            area["failure"] = str(error)
    result["calibration_all_passed"] = all(
        a.get("comparison", {}).get("passed") is True for a in result["areas"]
    )
    save(output / "report.json", result)
    print(
        json.dumps(
            {
                "report": ref(output / "report.json"),
                "passed": result["calibration_all_passed"],
            }
        )
    )


def docker_arguments(output: Path, name: str, image: str) -> list[str]:
    return [
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        "2g",
        "--memory-swap",
        "2g",
        "--cpus",
        "1",
        "--pids-limit",
        "128",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "--tmpfs",
        "/root:rw,size=128m",
        "--mount",
        f"type=bind,src={output.resolve()},dst=/work",
        "--entrypoint",
        "/opt/snap/bin/gpt",
        image,
        "-J-Xmx1536m",
        "-J-XX:ActiveProcessorCount=1",
        "-J-Dsnap.userdir=/work/user",
        "-J-Djava.io.tmpdir=/tmp",
    ]


def run(receipt_path: Path, metadata_path: Path, output: Path) -> None:
    policy, manifest, rows = inputs()
    if (
        available_commit() < 3 * 1024**3
        or psutil.virtual_memory().available < 3 * 1024**3
    ):
        raise RuntimeError("Three GiB memory headroom required")
    if shutil.disk_usage(REPO_ROOT).free < 5 * 1024**3:
        raise RuntimeError("Five GiB free required")
    metadata = json.loads(metadata_path.read_text())
    if metadata["prepared"] != policy["prepared"] or not (
        metadata["policy"] == ref(POLICY)
        or ref(metadata_path) == policy["adopted_metadata"]
    ):
        raise ValueError("SNAP metadata lineage differs")
    receipt = json.loads(receipt_path.read_text())
    installer = checked(receipt["installer"])
    if (
        receipt["image_id"] != policy["image_id"]
        or ref(installer)["sha256"] != policy["installer_sha256"]
    ):
        raise ValueError("Pinned SNAP distribution required")
    command = shutil.which("docker")
    if command is None:
        raise ValueError("Docker required for the pinned cached reference")
    docker = Path(command)
    inspected = subprocess.run(
        [str(docker), "image", "inspect", policy["image_id"], "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if inspected.stdout.strip() != policy["image_id"]:
        raise ValueError("Pinned SNAP image unavailable")
    fresh(output)

    def launch(arguments: list[str], log: Path, timeout: int) -> dict[str, Any]:
        name = "tritoneye-snap-" + uuid4().hex
        result = execute(
            docker,
            [*docker_arguments(output, name, policy["image_id"]), *arguments],
            log,
            timeout,
        )
        if result["timed_out"]:
            subprocess.run(
                [str(docker), "container", "rm", "--force", name],
                capture_output=True,
                timeout=30,
                check=False,
            )
        result["container_name"] = name
        return result

    help_result = launch(["-h"], output / "gpt_help.log", 120)
    report: dict[str, Any] = {
        "policy": ref(POLICY),
        "prepared": policy["prepared"],
        "metadata": ref(metadata_path),
        "runtime": ref(receipt_path),
        "image_id": policy["image_id"],
        "installer": ref(installer),
        "help": help_result,
        "areas": [],
        "reference_scope": "SNAP pointwise processing of original cropped DN "
        "in sparse full-dimension wrappers; absent pixels never evaluated",
        "training_approved": False,
        "independent_review": False,
        "compatibility_established": False,
        "classifier_scored": False,
        "detector_scored": False,
    }
    for area in rows:
        directory = output / area["id"]
        directory.mkdir()
        product = next(
            p for p in manifest["products"] if p["product_id"] == area["product_id"]
        )
        meta = next(
            p for p in metadata["products"] if p["product_id"] == area["product_id"]
        )
        source = wrapper(product, area, meta, directory)
        result: dict[str, Any] = {
            "id": area["id"],
            "product_id": area["product_id"],
            "split": area["split"],
            "source_chip": area["chip"],
            "window": area["window"],
            "variants": {},
        }
        for variant in ("raw", "calibration", "noise"):
            target = directory / f"{variant}.dim"
            graph_path = directory / f"{variant}.xml"
            graph_path.write_text(
                graph(
                    Path("/work") / source.resolve().relative_to(output.resolve()),
                    area["window"],
                    Path("/work") / target.resolve().relative_to(output.resolve()),
                    variant,
                ),
                encoding="utf-8",
            )
            execution = launch(
                [
                    (
                        Path("/work")
                        / graph_path.resolve().relative_to(output.resolve())
                    ).as_posix(),
                    "-q",
                    "1",
                    "-c",
                    "128M",
                ],
                directory / f"{variant}.log",
                policy["process_timeout_s"],
            )
            result["variants"][variant] = {**execution, "graph": ref(graph_path)}
            if execution["exit_code"] == 0:
                try:
                    values = bands(target, raw=variant == "raw")
                    result["variants"][variant]["finite"] = bool(
                        np.isfinite(values).all()
                    )
                    result["variants"][variant]["files"] = [
                        ref(p)
                        for p in sorted(target.with_suffix(".data").glob("*.img"))
                    ]
                except (ValueError, rasterio.errors.RasterioIOError) as error:
                    result["variants"][variant]["failure"] = str(error)
        try:
            dn = np.stack(
                [np.load(checked(b["dn"]), allow_pickle=False) for b in area["bands"]]
            )
            with rasterio.open(checked(area["chip"])) as ds:
                expected = ds.read()
            result["comparison"] = compare(
                bands(directory / "raw.dim", True),
                bands(directory / "calibration.dim"),
                dn,
                expected,
                policy["calibration_tolerance_db"],
            )
            noise = bands(directory / "noise.dim")
            nominal = bands(directory / "calibration.dim")
            result["noise_sensitivity"] = []
            for i, pol in enumerate(("HH", "HV")):
                valid = np.isfinite(noise[i]) & np.isfinite(nominal[i])
                delta = noise[i][valid] - nominal[i][valid]
                result["noise_sensitivity"].append(
                    {
                        "polarization": pol,
                        "finite_pixels": int(np.isfinite(noise[i]).sum()),
                        "total_pixels": int(noise[i].size),
                        "median_delta_db": (
                            float(np.median(delta)) if delta.size else None
                        ),
                        "max_absolute_delta_db": (
                            float(np.abs(delta).max()) if delta.size else None
                        ),
                    }
                )
        except (ValueError, rasterio.errors.RasterioIOError) as error:
            result["failure"] = str(error)
        report["areas"].append(result)
        report["calibration_all_passed"] = len(report["areas"]) == 4 and all(
            a.get("comparison", {}).get("passed") is True for a in report["areas"]
        )
        save(output / f"progress_{len(report['areas'])}.json", report)
        print(
            json.dumps(
                {
                    "area": area["id"],
                    "passed": result.get("comparison", {}).get("passed", False),
                }
            ),
            flush=True,
        )
    save(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--interpret", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.fetch_metadata:
        fetch(args.output)
    elif args.interpret:
        reinterpret(args.interpret, args.output)
    elif args.runtime and args.metadata:
        run(args.runtime, args.metadata, args.output)
    else:
        parser.error("Choose metadata acquisition or SNAP execution")
