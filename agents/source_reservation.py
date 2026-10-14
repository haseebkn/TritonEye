"""Seal source-supported annotation opportunities without assigning SAR labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from agents.acquisition import native_acquisition_group
from agents.artifacts import REPO_ROOT, sha256_file
from agents.hh_hv_reservation import save
from agents.nl_benchmark import require_development_scene
from agents.region import load_region
from agents.source_supported import POLICY

FORWARD = Transformer.from_crs(4326, 3347, always_xy=True)
INVERSE = Transformer.from_crs(3347, 4326, always_xy=True)


def ref(path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def checked(reference: dict[str, str]) -> Path:
    path = (REPO_ROOT / reference["path"]).resolve()
    if (
        not path.is_relative_to(REPO_ROOT.resolve())
        or sha256_file(path) != reference["sha256"]
    ):
        raise ValueError("Source artifact hash/path mismatch")
    return path


def projected(geometry: dict[str, Any]) -> Any:
    return transform(FORWARD.transform, shape(geometry))


def envelope(lon: float, lat: float, half_width: float) -> dict[str, Any]:
    x, y = FORWARD.transform(lon, lat)
    return dict(
        mapping(
            transform(
                INVERSE.transform,
                box(x - half_width, y - half_width, x + half_width, y + half_width),
            )
        )
    )


def prior_assignments() -> tuple[list[dict[str, Any]], set[str], dict[str, str]]:
    """Read identities and geometry only, including quarantine extent metadata."""
    areas: list[dict[str, Any]] = []
    exposed: set[str | None] = set()
    files = {}
    exposed.update(
        native_acquisition_group(path.name)
        for path in (REPO_ROOT / "data/raw").glob("S1*.SAFE")
    )
    for path in sorted((REPO_ROOT / "datasets/nl_benchmark").glob("*/manifest.json")):
        files[path.relative_to(REPO_ROOT).as_posix()] = sha256_file(path)
        for scene in json.loads(path.read_text())["scenes"]:
            exposed.add(native_acquisition_group(scene["name"]))
            areas.extend(
                {
                    "geometry": roi["geometry"],
                    "group": roi["group"],
                    "split": scene["split"],
                    "acquisition_group": native_acquisition_group(scene["name"]),
                    "quarantine": False,
                }
                for roi in scene["rois"]
            )
    path = REPO_ROOT / "reservations/hh_hv/v0.1.0/manifest.json"
    files[path.relative_to(REPO_ROOT).as_posix()] = sha256_file(path)
    for scene in json.loads(path.read_text())["scenes"]:
        exposed.add(native_acquisition_group(scene["name"]))
        areas.extend(
            {
                "geometry": roi["geometry"],
                "group": scene["geographic_group"],
                "split": scene["split"],
                "acquisition_group": native_acquisition_group(scene["name"]),
                "quarantine": False,
            }
            for roi in scene["rois"]
        )
    # This existing release binds extent-only local metadata. Do not open labels.
    path = REPO_ROOT / "datasets/allenai_nl/v0.1.0/release.json"
    release = json.loads(path.read_text())
    files[path.relative_to(REPO_ROOT).as_posix()] = sha256_file(path)
    local = checked(release["local_manifest"])
    files[local.relative_to(REPO_ROOT).as_posix()] = sha256_file(local)
    metadata = json.loads(local.read_text())
    exposed.update(native_acquisition_group(s["name"]) for s in metadata["sources"])
    areas.extend(
        {
            "geometry": case["physical_geometry"],
            "group": "quarantined_allenai",
            "split": "quarantine",
            "quarantine": True,
        }
        for case in metadata["cases"]
    )
    return areas, {g for g in exposed if g}, files


def assign(
    geometry: dict[str, Any],
    priors: list[dict[str, Any]],
    native_group: str | None = None,
) -> tuple[str, str]:
    area = projected(geometry)
    nearby = [p for p in priors if area.distance(projected(p["geometry"])) < 10000]
    if any(p["quarantine"] or p["split"] == "test" for p in nearby):
        raise ValueError("Locked/quarantine geographic separation")
    roles = {p["split"] for p in nearby}
    native_roles = {
        p["split"]
        for p in priors
        if native_group is not None and p.get("acquisition_group") == native_group
    }
    if native_roles & {"test", "quarantine"}:
        raise ValueError("Locked/quarantine acquisition assignment")
    if len(roles | native_roles) > 1:
        raise ValueError("Conflicting acquisition/geography assignments")
    if len(roles) > 1:
        raise ValueError("Conflicting development geography assignments")
    if nearby:
        groups = sorted({p["group"] for p in nearby})
        return next(iter(roles)), "+".join(groups)
    center = shape(geometry).centroid
    return (
        next(iter(native_roles), "train"),
        f"new_cell_{math.floor(center.x)}_{math.floor(center.y)}",
    )


def vessel_opportunities(
    audit: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = []
    for scene in audit["scenes"]:
        if scene["polarization"] != "HH/HV":
            continue
        for support in scene["support"]:
            nearest = min(
                support["reports"],
                key=lambda r: abs(
                    r["time"]
                    - datetime.fromisoformat(
                        scene["reference_midpoint_utc"]
                    ).timestamp()
                ),
            )
            try:
                speed = float(nearest["values"].get("speed_knots", "nan"))
            except (TypeError, ValueError):
                speed = math.nan
            eligible = (
                support.get("physical_surface") == "water"
                and support.get("signed_shore_distance_m") is not None
                and support.get("signed_shore_distance_m", -1)
                >= policy["ais_min_shore_distance_m"]
                and abs(support["nearest_offset_s"])
                <= policy["ais_nearest_max_seconds"]
                and support.get("bracket_gap_s") is not None
                and support["bracket_gap_s"] <= policy["ais_bracket_max_seconds"]
                and support["uncertainty_m"] <= policy["ais_max_uncertainty_m"]
                and math.isfinite(speed)
                and speed >= policy["ais_min_speed_knots"]
            )
            if not eligible:
                continue
            candidates.append(
                {
                    "stream": "vessel_support",
                    "product": scene["product"],
                    "reference_midpoint_utc": scene["reference_midpoint_utc"],
                    "lon": support["lon"],
                    "lat": support["lat"],
                    "source_identity": support["mmsi"],
                    "source": support,
                    "absolute_offset_s": abs(support["nearest_offset_s"]),
                    "search_uncertainty_m": support["uncertainty_m"],
                    "bracket_gap_s": support["bracket_gap_s"],
                    "speed_knots": speed,
                    "shore_distance_m": support["signed_shore_distance_m"],
                    "recorder_status": scene.get("recorder", {}).get(
                        "status", "unknown"
                    ),
                    "competing_identity_pool": scene["support"],
                    "label": None,
                    "provisional": True,
                }
            )
    candidates.sort(
        key=lambda r: (
            r["absolute_offset_s"],
            r["bracket_gap_s"],
            r["search_uncertainty_m"],
            -r["shore_distance_m"],
            r["product"]["Id"],
            r["source_identity"],
        )
    )
    return candidates


def reserve(
    iip: list[dict[str, Any]],
    ais: list[dict[str, Any]],
    priors: list[dict[str, Any]],
    exposed: set[str],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for stream, candidates in (("iceberg_support", iip), ("vessel_support", ais)):
        opportunities = []
        for row in candidates:
            if stream == "iceberg_support":
                if not row["temporal_eligible"]:
                    continue
                observation = row["observation"]
                row = {
                    **row,
                    "stream": stream,
                    "lon": observation["lon"],
                    "lat": observation["lat"],
                    "source_identity": observation["iceberg_id"],
                    "source": observation,
                }
            opportunities.append(row)
        # Prefer unseen native datatakes. Existing development examples remain
        # eligible but carry prior-exposure disclosure and existing split roles.
        opportunities.sort(
            key=lambda r: (
                native_acquisition_group(r["product"]["Name"]) in exposed,
                r["absolute_offset_s"],
                r["product"]["Id"],
                r["source_identity"],
            )
        )
        for row in opportunities:
            if (
                sum(s["stream"] == stream for s in selected)
                >= policy["maximum_examples_per_class"]
            ):
                break
            product = row["product"]
            identity = (stream, row["source_identity"])
            try:
                if (
                    product.get("Online") is not True
                    or "_IW_GRDH_1SDH_" not in product["Name"]
                ):
                    raise ValueError("Native online IW HH/HV required")
                geometry = envelope(
                    row["lon"], row["lat"], policy["reservation_envelope_half_width_m"]
                )
                load_region(geometry)
                if not shape(product["GeoFootprint"]).covers(shape(geometry)):
                    raise ValueError("Complete reservation envelope outside swath")
                split, group = assign(
                    geometry, priors, native_acquisition_group(product["Name"])
                )
                require_development_scene(
                    product["Id"],
                    {"name": product["Name"], "split": split, "study_roi": geometry},
                )
                if identity in identities:
                    raise ValueError("Repeated source identity")
                if counts[product["Id"]] >= policy["maximum_examples_per_product"]:
                    raise ValueError("Per-product batch limit")
                for previous in selected:
                    if (
                        native_acquisition_group(product["Name"])
                        == previous["native_acquisition_group"]
                        and previous["product_id"] != product["Id"]
                    ):
                        raise ValueError("Adjacent slice of already selected datatake")
                    if (
                        native_acquisition_group(product["Name"])
                        == previous["native_acquisition_group"]
                        and previous["split"] != split
                    ):
                        raise ValueError("Same native acquisition across splits")
                    distance = projected(geometry).distance(
                        projected(previous["geometry"])
                    )
                    if previous["split"] != split and distance < 10000:
                        raise ValueError("New cross-split geographic overlap")
                    if previous["product_id"] == product["Id"] and distance < 1000:
                        raise ValueError("Overlapping same-scene annotation areas")
                # Full competing identity pool is retained privately for AIS.
                item = {
                    **row,
                    "product_id": product["Id"],
                    "name": product["Name"],
                    "native_acquisition_group": native_acquisition_group(
                        product["Name"]
                    ),
                    "split": split,
                    "geographic_group": group,
                    "geometry": geometry,
                    "prior_datatake_exposure": native_acquisition_group(product["Name"])
                    in exposed,
                    "imagery_inspected": False,
                    "classifier_scored": False,
                    "label": None,
                    "ignore_for_accuracy": True,
                    "provisional": True,
                    "independent_review": False,
                    "promotion_eligible": False,
                }
                item["id"] = (
                    f"{stream}_{sum(s['stream'] == stream for s in selected)+1:02d}"
                )
                selected.append(item)
                identities.add(identity)
                counts[product["Id"]] += 1
            except ValueError as error:
                excluded.append(
                    {
                        "product_id": product["Id"],
                        "stream": stream,
                        "reason": str(error),
                    }
                )
    return selected, excluded


def public_entry(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "stream",
        "product_id",
        "name",
        "native_acquisition_group",
        "split",
        "geographic_group",
        "reference_midpoint_utc",
        "absolute_offset_s",
        "search_uncertainty_m",
        "prior_datatake_exposure",
        "imagery_inspected",
        "classifier_scored",
        "label",
        "ignore_for_accuracy",
        "provisional",
        "independent_review",
        "promotion_eligible",
    )
    value = {key: row[key] for key in keys if key in row}
    if row["stream"] == "iceberg_support":
        value["observation_method"] = row["source"]["method"]
        value["observation_source"] = row["source"]["source"]
    return value


def build(iip_path: Path, ais_path: Path, output: Path, publish: Path) -> None:
    if (
        output.exists()
        or not output.resolve().is_relative_to((REPO_ROOT / "data").resolve())
        or publish.exists()
        or publish.resolve().parent
        != (REPO_ROOT / "reservations/source_supported").resolve()
    ):
        raise ValueError("Fresh scoped output/release directories required")
    policy = json.loads(POLICY.read_text())
    iip = json.loads(iip_path.read_text())
    audit = json.loads(ais_path.read_text())
    normalized = json.loads(checked(iip["provenance"]["normalized"]).read_text())
    for source in normalized["sources"]:
        checked(source)
    for catalogue in iip["provenance"]["catalogues"]:
        checked(catalogue)
    if iip["provenance"]["policy_sha256"] != sha256_file(POLICY):
        raise ValueError("IIP ranking policy changed")
    for filename, expected in (
        ("snapshot_manifest.json", audit["summary"]["snapshot_manifest_sha256"]),
        ("catalogue_products.json", audit["summary"]["catalogue_sha256"]),
    ):
        if sha256_file(ais_path.parent / filename) != expected:
            raise ValueError("AIS audit inputs changed")
    priors, exposed, hashes = prior_assignments()
    vessels = vessel_opportunities(audit, policy)
    selected, excluded = reserve(iip["ranked"], vessels, priors, exposed, policy)
    output.mkdir(parents=True)
    private = {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "policy": ref(POLICY),
        "iip": ref(iip_path),
        "ais": ref(ais_path),
        "areas": selected,
        "exclusions": excluded,
        "prior_metadata_hashes": hashes,
        "prior_native_groups": sorted(exposed),
        "iip_provenance": iip["provenance"],
        "iip_sources": normalized["sources"],
        "ais_snapshot_manifest": ref(ais_path.parent / "snapshot_manifest.json"),
        "ais_catalogue": ref(ais_path.parent / "catalogue_products.json"),
        "annotation_complete": False,
        "independent_review": False,
    }
    save(output / "selection_private.json", private)
    save(
        output / "first_pass_template.json",
        {
            "reviewer": None,
            "independent_review": False,
            "prior_source_exposure": True,
            "classifier_scores_viewed": False,
            "areas": [
                {
                    "id": row["id"],
                    "product_id": row["product_id"],
                    "geometry": row["geometry"],
                    "source_chip_sha256": None,
                    "cells_reviewed": [],
                    "objects": [],
                    "complete": False,
                    "notes": "Await native crop; inspect both bands before scores. "
                    "Source evidence is in the separate selection file.",
                }
                for row in selected
            ],
        },
    )
    publish.mkdir(parents=True)
    iip_public = [
        {
            "rank": index + 1,
            "product_id": row["product"]["Id"],
            "name": row["product"]["Name"],
            "absolute_offset_s": row["absolute_offset_s"],
            "temporal_eligible": row["temporal_eligible"],
            "online": row["product"].get("Online"),
            "method": row["observation"]["method"],
            "source": row["observation"]["source"],
            "source_file": row["observation"]["source_file"],
            "source_line": row["observation"]["source_line"],
            "label": None,
        }
        for index, row in enumerate(iip["ranked"])
    ]
    ais_public = [
        {
            "rank": index + 1,
            "product_id": row["product"]["Id"],
            "name": row["product"]["Name"],
            "absolute_offset_s": row["absolute_offset_s"],
            "bracket_gap_s": row["bracket_gap_s"],
            "uncertainty_m": row["search_uncertainty_m"],
            "shore_distance_m": row["shore_distance_m"],
            "speed_knots": row["speed_knots"],
            "recorder_status": row["recorder_status"],
            "label": None,
        }
        for index, row in enumerate(vessels)
    ]
    save(
        publish / "rankings.json",
        {
            "iip_summary": iip["summary"],
            "iip_observation_summary": normalized["summary"],
            "iip_source_files": normalized["sources"],
            "iip_catalogue_files": iip["provenance"]["catalogues"],
            "iip_ranked": iip_public,
            "ais_archive": audit["summary"]["archive"],
            "ais_scene_statuses": audit["summary"]["statuses"],
            "ais_ranked_strict_hh_hv": ais_public,
            "interpretation": "Source-supported screening opportunities; "
            "SAR identities and class labels unverified",
        },
    )
    save(
        publish / "manifest.json",
        {
            "version": policy["version"],
            "selected_at": private["selected_at"],
            "policy": ref(POLICY),
            "private_selection": ref(output / "selection_private.json"),
            "first_pass_template": ref(output / "first_pass_template.json"),
            "rankings": ref(publish / "rankings.json"),
            "areas": [public_entry(row) for row in selected],
            "exclusions": excluded,
            "class_counts": dict(Counter(row["stream"] for row in selected)),
            "native_acquisitions": len(
                {row["native_acquisition_group"] for row in selected}
            ),
            "annotated_objects": 0,
            "scored": False,
            "independent_review": False,
            "promotion_eligible": False,
            "class_year_confounding": True,
        },
    )
    save(
        publish / "seal.json",
        {
            "manifest": ref(publish / "manifest.json"),
            "rankings": ref(publish / "rankings.json"),
        },
    )
    print(
        json.dumps(
            {
                "reserved": len(selected),
                "classes": dict(Counter(row["stream"] for row in selected)),
                "seal": ref(publish / "seal.json"),
            }
        )
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest["scored"] is not False
        or manifest["independent_review"] is not False
        or manifest["promotion_eligible"] is not False
        or manifest["annotated_objects"] != 0
    ):
        raise ValueError("Provisional reservation guard changed")
    for area in manifest["areas"]:
        if (
            area["label"] is not None
            or area["provisional"] is not True
            or area["ignore_for_accuracy"] is not True
            or area["independent_review"] is not False
            or area["classifier_scored"] is not False
            or area["imagery_inspected"] is not False
            or area["promotion_eligible"] is not False
            or area["split"] not in {"train", "validation"}
        ):
            raise ValueError("Unscored annotation reservation guard changed")
    areas = manifest["areas"]
    if (
        len({area["id"] for area in areas}) != len(areas)
        or dict(Counter(area["stream"] for area in areas)) != manifest["class_counts"]
        or len({area["native_acquisition_group"] for area in areas})
        != manifest["native_acquisitions"]
    ):
        raise ValueError("Reservation inventory mismatch")


def verify(directory: Path, expected_hash: str, local: bool = False) -> None:
    if sha256_file(directory / "seal.json") != expected_hash:
        raise ValueError("Trusted reservation seal changed")
    seal = json.loads((directory / "seal.json").read_text())
    manifest = json.loads(checked(seal["manifest"]).read_text())
    checked(seal["rankings"])
    checked(manifest["policy"])
    validate_manifest(manifest)
    if local:
        private = json.loads(checked(manifest["private_selection"]).read_text())
        iip = json.loads(checked(private["iip"]).read_text())
        ais = json.loads(checked(private["ais"]).read_text())
        checked(private["ais_snapshot_manifest"])
        checked(private["ais_catalogue"])
        checked(manifest["first_pass_template"])
        for catalogue in private["iip_provenance"]["catalogues"]:
            checked(catalogue)
        for source in private["iip_sources"]:
            checked(source)
        priors, _, _ = prior_assignments()
        policy = json.loads(checked(manifest["policy"]).read_text())
        replay, rejected = reserve(
            iip["ranked"],
            vessel_opportunities(ais, policy),
            priors,
            set(private["prior_native_groups"]),
            policy,
        )
        if (
            json.loads(json.dumps(replay)) != private["areas"]
            or rejected != private["exclusions"]
        ):
            raise ValueError("Reservation replay differs from frozen source selection")
        if [public_entry(row) for row in private["areas"]] != manifest["areas"]:
            raise ValueError("Public/private selections differ")
        for area in private["areas"]:
            require_development_scene(
                area["product_id"],
                {
                    "name": area["name"],
                    "split": area["split"],
                    "study_roi": area["geometry"],
                },
            )
            if assign(area["geometry"], priors, area["native_acquisition_group"]) != (
                area["split"],
                area["geographic_group"],
            ):
                raise ValueError("Prior geographic assignments changed")
    print(
        json.dumps(
            {
                "verified": True,
                "local": local,
                "reserved": len(manifest["areas"]),
                "annotated_objects": 0,
                "scored": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iip", type=Path)
    parser.add_argument("--ais", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.verify and args.seal_sha256:
        verify(args.verify, args.seal_sha256, args.local)
        if args.receipt:
            destination = args.receipt.resolve()
            if not destination.is_relative_to(REPO_ROOT.resolve()):
                raise ValueError("Verification receipt must stay in project")
            save(
                destination,
                {
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                    "seal_sha256": args.seal_sha256,
                    "local_sources_verified": args.local,
                    "selection_replayed": args.local,
                    "verification_code_sha256": sha256_file(Path(__file__)),
                    "scored": False,
                    "annotated_objects": 0,
                },
            )
    elif args.iip and args.ais and args.output and args.publish:
        build(args.iip, args.ais, args.output, args.publish)
    else:
        parser.error("Choose build or verify and its required paths")


if __name__ == "__main__":
    main()
