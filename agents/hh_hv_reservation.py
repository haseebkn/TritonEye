"""Reserve additional NL development acquisitions without accessing SAR pixels.

This is a sampling reservation, not an annotated dataset or a new locked test.
The seal is an integrity contract, not access control against a repository owner.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from agents.acquisition import native_acquisition_group, product_id, unique_acquisitions
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT, sha256_file
from agents.nl_benchmark import bounded_path, require_development_scene
from agents.region import load_region
from agents.run_versions import digest

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
POLICY = REPO_ROOT / "configs/reservations/hh_hv_development_v010.json"
DEFAULT = REPO_ROOT / "reservations/hh_hv/v0.1.0"


def save(path: Path, value: Any) -> None:
    """New generated metadata has stable LF bytes and cannot replace a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def envelopes(group: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "regime": regime,
            "geometry": mapping(
                box(
                    center[0] - policy["envelope_half_width_degrees"],
                    center[1] - policy["envelope_half_height_degrees"],
                    center[0] + policy["envelope_half_width_degrees"],
                    center[1] + policy["envelope_half_height_degrees"],
                )
            ),
        }
        for regime in ("coastal", "offshore")
        for center in (group[f"{regime}_center"],)
    ]


def exposure_inventory(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Read only recorded identities/extents, never labels or imagery arrays."""
    files: dict[str, str] = {}
    names = {p.name for p in (root / "data/raw").glob("S1*.SAFE")}
    regions: list[dict[str, Any]] = []
    for path in sorted((root / "datasets/nl_benchmark").glob("*/manifest.json")):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
        for scene in json.loads(path.read_text(encoding="utf-8"))["scenes"]:
            names.add(scene["name"])
            regions.extend(
                {
                    "source": path.relative_to(root).as_posix(),
                    "group": roi["group"],
                    "geometry": roi["geometry"],
                }
                for roi in scene["rois"]
            )
    release = root / "datasets/allenai_nl/v0.1.0/release.json"
    if release.exists():
        files[release.relative_to(root).as_posix()] = sha256_file(release)
        reference = json.loads(release.read_text(encoding="utf-8"))["local_manifest"]
        local = bounded_path(root, reference["path"])
        if sha256_file(local) != reference["sha256"]:
            raise ValueError("Quarantine metadata integrity mismatch")
        files[reference["path"]] = reference["sha256"]
        metadata = json.loads(local.read_text(encoding="utf-8"))
        names.update(s["name"] for s in metadata["sources"])
        # Conservative bounding boxes avoid redistributing source labels.
        regions.extend(
            {
                "source": reference["path"],
                "group": "previously_exposed_allenai",
                "geometry": mapping(box(*shape(c["physical_geometry"]).bounds)),
            }
            for c in metadata["cases"]
        )
    return {
        "files": files,
        "native_groups": sorted(
            {g for name in names if (g := native_acquisition_group(name))}
        ),
        "product_names": sorted(names),
        "regions": regions,
        "scope": "Known local exposures; not a guarantee against unrecorded access",
    }


def projected(geometry: dict[str, Any]) -> Any:
    value = shape(geometry)
    load_region(value)
    transformer = Transformer.from_crs(4326, 3347, always_xy=True)
    return transform(transformer.transform, value)


def validate_policy(policy: dict[str, Any], prior: dict[str, Any]) -> None:
    """Separate actual footprints as well as names; no same-area split aliases."""
    parameters = [
        policy[k]
        for k in (
            "minimum_group_separation_m",
            "minimum_prior_exposure_separation_m",
            "envelope_half_width_degrees",
            "envelope_half_height_degrees",
        )
    ]
    if (
        not all(math.isfinite(v) and v > 0 for v in parameters)
        or policy["minimum_group_separation_m"] < 10000
        or policy["minimum_prior_exposure_separation_m"] < 10000
        or len(policy["groups"]) < 4
        or len(policy["epochs"]) < 2
    ):
        raise ValueError("At least four separated groups and two epochs required")
    old = [projected(r["geometry"]) for r in prior["regions"]]
    seen: dict[str, list[Any]] = {}
    for group in policy["groups"]:
        if group["id"] in seen or group["split"] not in ("train", "validation"):
            raise ValueError("Unique development geographic groups required")
        if group["region"] not in ("Newfoundland", "Labrador"):
            raise ValueError("NL development only")
        areas = [projected(r["geometry"]) for r in envelopes(group, policy)]
        if areas[0].distance(areas[1]) < 1000:
            raise ValueError("Coastal and offshore envelopes must not overlap")
        for area in areas:
            if any(
                area.distance(other) < policy["minimum_prior_exposure_separation_m"]
                for other in old
            ):
                raise ValueError("Reserved geography overlaps prior exposure")
            if any(
                area.distance(other) < policy["minimum_group_separation_m"]
                for v in seen.values()
                for other in v
            ):
                raise ValueError("Reserved geographic groups are not separated")
        seen[group["id"]] = areas
    intervals: list[tuple[datetime, datetime]] = []
    identifiers = set()
    for epoch in policy["epochs"]:
        start, end = parse_utc(epoch["start"]), parse_utc(epoch["end"])
        if (
            start is None
            or end is None
            or start >= end
            or not epoch["id"]
            or epoch["id"] in identifiers
        ):
            raise ValueError("Valid UTC epoch intervals required")
        if any(start < b and end > a for a, b in intervals):
            raise ValueError("Sampling epochs must not overlap")
        identifiers.add(epoch["id"])
        intervals.append((start, end))
    for split in ("train", "validation"):
        if {g["region"] for g in policy["groups"] if g["split"] == split} != {
            "Newfoundland",
            "Labrador",
        }:
            raise ValueError("Each development split needs both NL regions")


def product_metadata(product: dict[str, Any]) -> dict[str, Any]:
    name = product["Name"]
    group = native_acquisition_group(name)
    attributes = {a["Name"]: a["Value"] for a in product.get("Attributes", [])}
    if (
        group is None
        or "_IW_GRDH_1SDH_" not in name
        or set(
            str(attributes.get("polarisationChannels", "")).replace("&", " ").split()
        )
        != {"HH", "HV"}
        or product.get("Online") is not True
    ):
        raise ValueError("Online native IW GRDH HH/HV metadata required")
    try:
        footprint = shape(product["GeoFootprint"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Complete valid catalogue footprint required") from error
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError("Complete valid catalogue footprint required")
    if parse_utc(product["ContentDate"]["Start"]) is None:
        raise ValueError("Acquisition time required")
    return {
        "product_id": product_id(product["Id"]),
        "name": name,
        "acquisition_group": group,
        "acquisition_time": product["ContentDate"]["Start"],
        "polarizations": ["hh", "hv"],
        "footprint": product["GeoFootprint"],
        "orbit_direction": attributes.get("orbitDirection"),
        "relative_orbit_number": attributes.get("relativeOrbitNumber"),
    }


def choose(
    products: list[dict[str, Any]],
    group: dict[str, Any],
    epoch: dict[str, Any],
    policy: dict[str, Any],
    excluded: set[str],
) -> dict[str, Any]:
    for product in sorted(
        unique_acquisitions(products),
        key=lambda p: (p["ContentDate"]["Start"], p["Name"], p["Id"]),
    ):
        try:
            item = product_metadata(product)
        except ValueError:
            continue
        if item["acquisition_group"] in excluded:
            continue
        acquired = parse_utc(item["acquisition_time"])
        start, end = parse_utc(epoch["start"]), parse_utc(epoch["end"])
        if (
            acquired is None
            or start is None
            or end is None
            or not start <= acquired < end
        ):
            continue
        rois = envelopes(group, policy)
        if not all(shape(item["footprint"]).covers(shape(r["geometry"])) for r in rois):
            continue
        for roi in rois:
            require_development_scene(
                item["product_id"], {**item, "study_roi": roi["geometry"]}
            )
        return {
            **item,
            "geographic_group": group["id"],
            "region": group["region"],
            "split": group["split"],
            "epoch": epoch["id"],
            "rois": rois,
        }
    raise ValueError(
        f"No eligible complete-footprint acquisition: {group['id']}/{epoch['id']}"
    )


def query(
    group: dict[str, Any], epoch: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lon, lat = group["coastal_center"]
    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-1' and contains(Name,'IW_GRDH_1SDH') "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
            f"and ContentDate/Start ge {epoch['start']} "
            f"and ContentDate/Start lt {epoch['end']}"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": "100",
        "$expand": "Attributes",
    }
    cache = (
        REPO_ROOT
        / "data/hh_hv_reservations/catalogue_queries"
        / f"{digest(params)}.json"
    )
    if cache.exists():
        receipt = json.loads(cache.read_text(encoding="utf-8"))
    else:
        response = requests.get(CATALOGUE, params=params, timeout=60)
        response.raise_for_status()
        receipt = {
            "url": response.url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "page": response.json(),
        }
        save(cache, receipt)
    page = receipt["page"]
    # Refuse truncation rather than pretend a partial search is deterministic.
    if page.get("@odata.nextLink"):
        raise ValueError("Catalogue query needs narrower predeclared epochs")
    return page["value"], {
        "url": receipt["url"],
        "result_count": len(page["value"]),
        "cached_response": {
            "path": cache.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(cache),
        },
    }


def reserve(output: Path, policy_path: Path = POLICY) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() or not output.is_relative_to(
        (REPO_ROOT / "reservations/hh_hv").resolve()
    ):
        raise ValueError("Fresh HH/HV reservation directory required")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    prior = exposure_inventory()
    validate_policy(policy, prior)
    selected, snapshots, searches = [], {}, []
    excluded = set(prior["native_groups"])
    for group in policy["groups"]:
        for epoch in policy["epochs"]:
            products, search = query(group, epoch)
            item = choose(products, group, epoch, policy, excluded)
            selected.append(item)
            snapshots[item["product_id"]] = next(
                p for p in products if product_id(p["Id"]) == item["product_id"]
            )
            searches.append({"group": group["id"], "epoch": epoch["id"], **search})
            excluded.add(item["acquisition_group"])
    manifest = {
        "version": policy["version"],
        "status": "reserved_before_pixels_and_scores",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reservation_code_sha256": sha256_file(Path(__file__)),
        "policy": policy,
        "prior_exposure": prior,
        "searches": searches,
        "scenes": selected,
        "generator_config_sha256": sha256_file(
            REPO_ROOT / "configs/candidates/hh_hv_local_contrast_v010.json"
        ),
        "imagery_downloaded_by_reservation": False,
        "imagery_inspected": False,
        "scored": False,
        "independent_review": False,
        "promotion_eligible": False,
        "attribution": "Contains Copernicus Sentinel metadata (2025, 2026)",
        "terms": "https://dataspace.copernicus.eu/terms-and-conditions",
        "conditions": (
            "Seasonal proxies only; wind, ice, sea state and radiometry unverified"
        ),
    }
    output.mkdir(parents=True)
    save(output / "manifest.json", manifest)
    for identity, snapshot in snapshots.items():
        save(output / f"catalogue/{identity}.json", snapshot)
    files = {
        p.relative_to(output).as_posix(): sha256_file(p)
        for p in sorted(output.rglob("*.json"))
    }
    save(
        output / "seal.json",
        {
            "version": policy["version"],
            "files": files,
            "selection_digest": digest(selected),
        },
    )
    return manifest


def load_reservation(directory: Path, expected_seal_sha256: str) -> dict[str, Any]:
    directory = directory.resolve()
    if not directory.is_relative_to((REPO_ROOT / "reservations/hh_hv").resolve()):
        raise ValueError("Reservation must remain in its project directory")
    if sha256_file(directory / "seal.json") != expected_seal_sha256:
        raise ValueError("Reservation seal mismatch")
    seal = json.loads((directory / "seal.json").read_text(encoding="utf-8"))
    for name, checksum in seal["files"].items():
        if sha256_file(bounded_path(directory, name)) != checksum:
            raise ValueError("Reservation bytes changed")
    manifest: dict[str, Any] = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    required_files = {"manifest.json"} | {
        f"catalogue/{product_id(s['product_id'])}.json" for s in manifest["scenes"]
    }
    if set(seal["files"]) != required_files:
        raise ValueError("Reservation seal must bind all required files")
    if (
        manifest["status"] != "reserved_before_pixels_and_scores"
        or any(
            manifest[k] is not False
            for k in (
                "scored",
                "imagery_inspected",
                "imagery_downloaded_by_reservation",
                "independent_review",
                "promotion_eligible",
            )
        )
        or seal["version"] != manifest["version"]
        or manifest["version"] != manifest["policy"]["version"]
        or manifest["generator_config_sha256"]
        != sha256_file(REPO_ROOT / "configs/candidates/hh_hv_local_contrast_v010.json")
    ):
        raise ValueError("Reservation status or frozen generator changed")
    validate_policy(manifest["policy"], manifest["prior_exposure"])
    if digest(manifest["scenes"]) != seal["selection_digest"]:
        raise ValueError("Reserved population changed")
    used = set(manifest["prior_exposure"]["native_groups"])
    pairs = set()
    for item in manifest["scenes"]:
        product = json.loads(
            bounded_path(
                directory, f"catalogue/{product_id(item['product_id'])}.json"
            ).read_text(encoding="utf-8")
        )
        metadata = product_metadata(product)
        if (
            any(item[k] != v for k, v in metadata.items())
            or item["acquisition_group"] in used
        ):
            raise ValueError("Reserved acquisition identity changed or reused")
        group = next(
            g
            for g in manifest["policy"]["groups"]
            if g["id"] == item["geographic_group"]
        )
        epoch = next(
            e for e in manifest["policy"]["epochs"] if e["id"] == item["epoch"]
        )
        expected = choose([product], group, epoch, manifest["policy"], used)
        if digest(item) != digest(expected):
            raise ValueError("Reserved split or geographic scope changed")
        pair = (item["geographic_group"], item["epoch"])
        if pair in pairs:
            raise ValueError("Repeated geographic group and epoch assignment")
        pairs.add(pair)
        used.add(item["acquisition_group"])
    if len(manifest["scenes"]) != len(manifest["policy"]["groups"]) * len(
        manifest["policy"]["epochs"]
    ):
        raise ValueError("Incomplete reservation")
    return manifest


def require_scope(
    manifest: dict[str, Any], scene: dict[str, Any], roi: dict[str, Any]
) -> None:
    matching = [s for s in manifest["scenes"] if s["product_id"] == scene["product_id"]]
    if len(matching) != 1:
        raise ValueError("Product is not reserved")
    selected = matching[0]
    if (
        scene["name"] != selected["name"]
        or scene["split"] != selected["split"]
        or scene["region"] != selected["region"]
        or scene["polarizations"] != ["hh", "hv"]
        or parse_utc(scene["acquisition_time"])
        != parse_utc(selected["acquisition_time"])
        or roi["group"] != selected["geographic_group"]
        or not any(
            r["regime"] == roi["regime"]
            and shape(r["geometry"]).covers(shape(roi["geometry"]))
            for r in selected["rois"]
        )
    ):
        raise ValueError("ROI or split escapes the reserved development scope")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    if args.reserve and not args.verify:
        manifest = reserve(args.reserve, args.policy)
    elif args.verify and not args.reserve and args.seal_sha256:
        manifest = load_reservation(args.verify, args.seal_sha256)
    else:
        parser.error("Use --reserve DIRECTORY or --verify DIRECTORY --seal-sha256 HASH")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "acquisitions": len(manifest["scenes"]),
                "scored": manifest["scored"],
            }
        )
    )


if __name__ == "__main__":
    main()
