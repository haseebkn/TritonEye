"""Admit evidence-backed provisional C/D objects and export native training chips.

Evidence declarations are auditable assertions, not independent physical truth.
No annotation, ignore mask, split or model selection is changed by this module.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from rasterio.transform import GCPTransformer
from scipy.interpolate import LinearNDInterpolator
from shapely.geometry import Polygon, mapping, shape

from agents.acquisition import native_acquisition_group
from agents.ais_validation import parse_utc
from agents.artifacts import REPO_ROOT
from agents.calibration import Calibrator
from agents.cd_acceptance import annotation_utc, software
from agents.cd_reference import header
from agents.cd_training_reservations import freeze_priors
from agents.hh_hv_reservation import save
from agents.native_cd_input import SPEC, prepare, specification
from agents.nl_benchmark import require_development_scene
from agents.provisional_training import GUARDS, training_role
from agents.region import load_region
from agents.reserved_sar import annotation_metadata
from agents.source_reservation import checked, ref

POLICY = REPO_ROOT / "configs/reservations/cd_provisional_v010.json"
SELECTION = REPO_ROOT / "data/ca_sel/selection.json"


def policy() -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(POLICY.read_text())
    specification()
    if value["native_specification"] != ref(SPEC):
        raise ValueError("Frozen C/D specification required")
    base = json.loads(checked(value["base_policy"]).read_text())
    if (
        value["classes"] != base["admission"]["classes"]
        or value["split"] != "train"
        or value["label_status"] != "provisional"
        or value["normalization"] != "identity"
        or any(
            value.get(k) is not False
            for k in (
                "independent_review",
                "accuracy_truth_admitted",
                "promotion_eligible",
                "training_approved",
            )
        )
    ):
        raise ValueError("Provisional training policy changed")
    return value, base["admission"]


def inventory() -> dict[str, Any]:
    """Use existing metadata roles, including later reservations and C/D checks."""
    return freeze_priors(SELECTION)


def scope(subject: dict[str, Any], frozen: dict[str, Any]) -> None:
    UUID(subject["product_id"])
    geometry = subject["geometry"]
    load_region(geometry)
    if subject["split"] != "train":
        raise ValueError("Training role required")
    group = native_acquisition_group(subject["name"])
    if (
        not group
        or training_role(geometry, frozen["priors"], group)
        != subject["geographic_group"]
    ):
        raise ValueError("Acquisition/geographic training role differs")
    require_development_scene(
        subject["product_id"],
        {"name": subject["name"], "split": "train", "study_roi": geometry},
    )
    # Repeated objects previously assigned to validation cannot enter train.
    identity = subject.get("source_identity")
    if identity and any(
        str(a.get("source_identity")) == str(identity) and a["split"] != "train"
        for a in frozen["old"]
    ):
        raise ValueError("Source identity already assigned outside training")


def native_tensor(
    native: dict[str, Any], bbox: list[int], frozen: dict[str, Any]
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Replay local DN/LUTs and metadata before extracting an untouched chip."""
    scope(native, frozen)  # Before opening SAR arrays or native metadata.
    col, row, width, height = native["window"]
    if (
        any(type(x) is not int for x in native["window"])
        or min(col, row) < 0
        or not 75 <= min(width, height) <= max(width, height) <= 512
        or [b["polarization"] for b in native["bands"]] != ["HH", "HV"]
    ):
        raise ValueError("Bounded native HH/HV window required")
    processor = software(checked(native["manifest"]))
    headers, trees, gcps_list = [], [], []
    for band in native["bands"]:
        annotation = checked(band["annotation"]).read_bytes()
        headers.append(
            header(
                annotation,
                checked(band["calibration"]).read_bytes(),
                checked(band["noise"]).read_bytes(),
                native["name"][:3],
                band["polarization"],
            )
        )
        _, gcps = annotation_metadata(annotation, native, band["polarization"].lower())
        tree = ET.fromstring(annotation)
        h = tree.find(".//adsHeader")
        assert h is not None  # header() requires it.
        pieces = native["name"].split("_")
        if (
            int(h.findtext("absoluteOrbitNumber", "-1")) != int(pieces[6])
            or int(h.findtext("missionDataTakeId", "-1")) != int(pieces[7], 16)
            or col + width > int(tree.findtext(".//numberOfSamples", "0"))
            or row + height > int(tree.findtext(".//numberOfLines", "0"))
        ):
            raise ValueError("Native orbit/datatake or window identity differs")
        trees.append(tree)
        gcps_list.append(gcps)
    if headers[0] | {"polarization": "HV"} != headers[1] or [
        (g.row, g.col, g.x, g.y) for g in gcps_list[0]
    ] != [(g.row, g.col, g.x, g.y) for g in gcps_list[1]]:
        raise ValueError("HH/HV native metadata alignment differs")
    # Reconstruct the full window boundary from native metadata. Test the
    # resulting geography before pixels; a caller's safe-looking ROI is insufficient.
    rr = np.concatenate(
        [
            np.full(17, row),
            np.linspace(row, row + height, 17),
            np.full(17, row + height),
            np.linspace(row + height, row, 17),
        ]
    )
    cc = np.concatenate(
        [
            np.linspace(col, col + width, 17),
            np.full(17, col + width),
            np.linspace(col + width, col, 17),
            np.full(17, col),
        ]
    )
    with GCPTransformer(gcps_list[0], tps=True) as geo:
        xs, ys = geo.xy(rr.tolist(), cc.tolist(), offset="ul")
    geometry = mapping(Polygon(zip(xs, ys)))
    actual = native | {"geometry": geometry}
    scope(actual, frozen)
    if not shape(native["geometry"]).buffer(1e-9).covers(shape(geometry)):
        raise ValueError("Native pixels escape declared scope")
    if len(bbox) != 4 or any(type(v) is not int for v in bbox):
        raise ValueError("Integer object bbox required")
    x0, y0, x1, y1 = bbox
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("Object bbox outside native window")
    center_row, center_col = row + (y0 + y1) // 2, col + (x0 + x1) // 2
    points = trees[0].findall(".//geolocationGridPoint")
    angle = float(
        LinearNDInterpolator(
            [
                (float(p.findtext("line", "nan")), float(p.findtext("pixel", "nan")))
                for p in points
            ],
            [float(p.findtext("incidenceAngle", "nan")) for p in points],
        )(center_row, center_col)
    )
    start = parse_utc(annotation_utc(headers[0]["start"]))
    assert start is not None
    interval = float(trees[0].findtext(".//azimuthTimeInterval", "nan"))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Native azimuth timing invalid")
    metadata = {
        "name": native["name"],
        "mission": native["name"][:3],
        "mode": "IW",
        "product_type": "GRD",
        "band_order": ["HH", "HV"],
        "pixel_spacing_m": [
            headers[0]["range_spacing_m"],
            headers[0]["azimuth_spacing_m"],
        ],
        "thermal_noise_corrected": False,
        "unit": "sigma0_db",
        "processing_software": processor["name"],
        "processing_version": processor["version"],
        "native_start_utc": start.isoformat(),
        "native_stop_utc": annotation_utc(headers[0]["stop"]),
        "native_center_utc": (
            start + timedelta(seconds=center_row * interval)
        ).isoformat(),
        "incidence_center_deg": angle,
    }
    values = np.load(checked(native["values"]), allow_pickle=False, mmap_mode="r")
    if values.shape != (2, height, width) or values.dtype != np.dtype("float32"):
        raise ValueError("Saved nominal window shape/type differs")
    for i, band in enumerate(native["bands"]):
        dn = np.load(checked(band["dn"]), allow_pickle=False, mmap_mode="r")
        if dn.shape != (height, width) or not np.isfinite(dn).all() or np.any(dn <= 0):
            raise ValueError("Invalid or incomplete native DN")
        replay = Calibrator(str(checked(band["calibration"]))).to_sigma0_db(
            dn, row, col
        )
        if not np.array_equal(replay.astype("float32"), values[i]):
            raise ValueError("Native nominal pixels fail calibration replay")
    result = prepare(values, bbox, metadata, "train")
    if result["status"] != "research_tensor_only":
        raise ValueError("Native recipe refused: " + str(result["reason"]))
    return result["tensor"], metadata


def assess(
    dossier: dict[str, Any], frozen: dict[str, Any] | None = None
) -> tuple[dict[str, Any], np.ndarray[Any, Any] | None]:
    rules, admission = policy()
    reasons: list[str] = []
    tensor = None
    metadata = None
    if dossier.get("policy") != ref(POLICY):
        reasons.append("frozen_cd_policy_required")
    if dossier.get("native_specification") != rules["native_specification"]:
        reasons.append("frozen_native_recipe_required")
    if dossier.get("class") not in rules["classes"]:
        reasons.append("unresolved_or_unsupported_class")
    if dossier.get("provisional") is not True:
        reasons.append("provisional_status_required")
    if dossier.get("prior_exposure_disclosed") is not True:
        reasons.append("prior_exposure_disclosure_required")
    if any(
        not isinstance(dossier.get(k), str) or not dossier[k].strip()
        for k in ("object_id", "repeated_object_group", "source_identity")
    ):
        reasons.append("object_and_repeated_identity_required")
    annotated, scored = (
        parse_utc(dossier.get(k))
        for k in (
            "annotation_completed_utc",
            "first_score_utc",
        )
    )
    if (
        annotated is None
        or dossier.get("scores_viewed") is not False
        or (dossier.get("first_score_utc") is not None and scored is None)
        or (scored is not None and annotated >= scored)
    ):
        reasons.append("annotation_before_scores_required")
    checks = admission["required_checks"] + admission.get(
        f"{dossier.get('class')}_checks", []
    )
    for name in checks:
        item = dossier.get("checks", {}).get(name, {})
        if item.get("passed") is not True or not item.get("evidence"):
            reasons.append(name)
            continue
        for reference in item["evidence"]:
            evidence = json.loads(checked(reference).read_text())
            if (
                any(
                    evidence.get(k) != dossier.get(k)
                    for k in (
                        "product_id",
                        "object_id",
                        "source_identity",
                        "repeated_object_group",
                    )
                )
                or evidence.get("checks", {}).get(name) is not True
                or evidence.get("native") != dossier.get("native")
                or evidence.get("annotation") != dossier.get("annotation")
            ):
                reasons.append(name + ":evidence_subject_or_result_mismatch")
    try:
        prior = inventory() if frozen is None else frozen
        scope(dossier, prior)
        annotation = json.loads(checked(dossier["annotation"]).read_text())
        keys = (
            "product_id",
            "name",
            "object_id",
            "class",
            "split",
            "geometry",
            "geographic_group",
            "annotation_completed_utc",
            "bbox_px",
            "native",
            "source_identity",
            "repeated_object_group",
            "provisional",
        )
        if any(annotation.get(k) != dossier.get(k) for k in keys):
            raise ValueError(
                "Annotation identity, class, native source or role differs"
            )
        native = json.loads(checked(dossier["native"]).read_text())
        if any(
            native.get(k) != dossier.get(k)
            for k in (
                "product_id",
                "name",
                "split",
                "geographic_group",
                "geometry",
            )
        ):
            raise ValueError("Native receipt identity or scope differs")
        # A missing evidence check holds the label before any image read.
        if not reasons:
            tensor, metadata = native_tensor(native, dossier["bbox_px"], prior)
    except (KeyError, TypeError, ValueError, OSError) as error:
        reasons.append("native_or_annotation_unverified:" + str(error))
    result = {
        "disposition": "hold" if reasons else "provisional_exploratory_training",
        "exploratory_training_admitted": not reasons,
        "accuracy_truth_admitted": False,
        "verified_negative": False,
        "blockers": reasons,
        "native_metadata": metadata,
        **GUARDS,
    }
    return result, tensor if not reasons else None


def repeated_consistency(
    dossier: dict[str, Any],
    repeated: dict[Any, Any],
    repeated_groups: dict[Any, Any],
    repeated_classes: dict[Any, Any],
) -> None:
    """Keep repeated-object groups and source identities in a 1:1 mapping."""
    identity = dossier.get("source_identity")
    group = dossier.get("repeated_object_group")
    if identity in repeated and repeated[identity] != group:
        raise ValueError("Repeated source identity assigned inconsistent object groups")
    repeated[identity] = group
    if group in repeated_groups and repeated_groups[group] != identity:
        raise ValueError(
            "Repeated object group assigned to inconsistent source identities"
        )
    repeated_groups[group] = identity
    label = dossier.get("class")
    if identity in repeated_classes and repeated_classes[identity] != label:
        raise ValueError("Repeated source identity assigned inconsistent classes")
    repeated_classes[identity] = label


def export(batch_path: Path, output: Path) -> dict[str, Any]:
    """Publish a fresh directory only after every proposed label passes admission."""
    output = output.resolve()
    root = (REPO_ROOT / "data/native_training_exports").resolve()
    if output.exists() or output == root or not output.is_relative_to(root):
        raise ValueError("Fresh private training export directory required")
    batch = json.loads(batch_path.read_text())
    if batch.get("policy") != ref(POLICY) or not isinstance(
        batch.get("dossiers"), list
    ):
        raise ValueError("Frozen policy and explicit dossier list required")
    frozen = inventory()
    assessed = []
    identities: set[tuple[Any, Any]] = set()
    repeated: dict[Any, Any] = {}
    repeated_groups: dict[Any, Any] = {}
    repeated_classes: dict[Any, Any] = {}
    for reference in batch["dossiers"]:
        dossier = json.loads(checked(reference).read_text())
        key = (dossier.get("product_id"), dossier.get("object_id"))
        if key in identities:
            raise ValueError("Duplicate object in export")
        identities.add(key)
        repeated_consistency(dossier, repeated, repeated_groups, repeated_classes)
        decision, tensor = assess(dossier, frozen)
        if not decision["exploratory_training_admitted"]:
            raise ValueError("Admission held: " + ", ".join(decision["blockers"]))
        assert tensor is not None
        assessed.append((reference, dossier, decision, tensor))
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    entries = []
    for n, (reference, dossier, decision, tensor) in enumerate(assessed):
        path = staging / f"chip_{n:04d}.npy"
        np.save(path, tensor, allow_pickle=False)
        # File paths in a published manifest refer to the final location.
        file_ref = ref(path) | {
            "path": (output / path.name).relative_to(REPO_ROOT).as_posix()
        }
        entries.append(
            {
                "dossier": reference,
                "product_id": dossier["product_id"],
                "native_acquisition_group": native_acquisition_group(dossier["name"]),
                "object_id": dossier["object_id"],
                "class": dossier["class"],
                "geographic_group": dossier["geographic_group"],
                "repeated_object_group": dossier["repeated_object_group"],
                "tensor": file_ref,
                "decision": decision,
            }
        )
    manifest = {
        "version": "0.1.0",
        "status": "provisional_exploratory_training_only",
        "policy": ref(POLICY),
        "native_specification": ref(SPEC),
        "batch": ref(batch_path),
        "implementation": ref(Path(__file__)),
        "split": "train",
        "normalization": "identity",
        "entries": entries,
        "labels_admitted": len(entries),
        "class_counts": dict(Counter(e["class"] for e in entries)),
        "accuracy_truth_admitted": False,
        "verified_negative": False,
        **GUARDS,
        "limitations": [
            "No independent reviewer; source declarations do not establish "
            "physical truth.",
            "Classified target contexts only; no complete-area or negative "
            "labels implied.",
            "No Kaggle equivalence, fitted model, accuracy measurement or promotion.",
            "DN range hashes are retained; full publisher product checksums "
            "may be unavailable.",
        ],
    }
    save(staging / "inventory.json", frozen)
    manifest["inventory"] = ref(staging / "inventory.json") | {
        "path": (output / "inventory.json").relative_to(REPO_ROOT).as_posix()
    }
    save(staging / "manifest.json", manifest)
    # A failed write leaves only a visibly partial staging directory. It cannot
    # expose a completed export at the requested path.
    staging.rename(output)
    return {
        "exported": len(entries),
        "class_counts": manifest["class_counts"],
        **GUARDS,
    }


def verify(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if (
        manifest["policy"] != ref(POLICY)
        or manifest["native_specification"] != ref(SPEC)
        or manifest["split"] != "train"
        or manifest["normalization"] != "identity"
        or any(manifest.get(k) is not v for k, v in GUARDS.items())
        or manifest["accuracy_truth_admitted"] is not False
        or manifest["verified_negative"] is not False
    ):
        raise ValueError("Export guards differ")
    checked(manifest["implementation"])
    frozen = json.loads(checked(manifest["inventory"]).read_text())
    for filename, sha in frozen["prior_metadata_hashes"].items():
        checked({"path": filename, "sha256": sha})
    batch = json.loads(checked(manifest["batch"]).read_text())
    entries = manifest["entries"]
    if (
        [e["dossier"] for e in entries] != batch["dossiers"]
        or manifest["labels_admitted"] != len(entries)
        or manifest["class_counts"] != dict(Counter(e["class"] for e in entries))
    ):
        raise ValueError("Export inventory differs")
    live = inventory()
    identities: set[tuple[Any, Any]] = set()
    repeated: dict[Any, Any] = {}
    repeated_groups: dict[Any, Any] = {}
    repeated_classes: dict[Any, Any] = {}
    for entry in entries:
        dossier = json.loads(checked(entry["dossier"]).read_text())
        identity = (dossier["product_id"], dossier["object_id"])
        if identity in identities:
            raise ValueError("Duplicate object in export")
        identities.add(identity)
        repeated_consistency(dossier, repeated, repeated_groups, repeated_classes)
        # Recheck present role constraints as well as the frozen decision.
        scope(dossier, live)
        decision, tensor = assess(dossier, frozen)
        if (
            decision != entry["decision"]
            or tensor is None
            or entry["native_acquisition_group"]
            != native_acquisition_group(dossier["name"])
            or any(
                entry[k] != dossier[k]
                for k in (
                    "object_id",
                    "product_id",
                    "class",
                    "geographic_group",
                    "repeated_object_group",
                )
            )
        ):
            raise ValueError("Export admission replay differs")
        saved = np.load(checked(entry["tensor"]), allow_pickle=False)
        if saved.dtype != np.dtype("float32") or not np.array_equal(saved, tensor):
            raise ValueError("Export tensor replay differs")
    return {"verified": True, "labels_admitted": len(entries), **GUARDS}


def template(output: Path) -> None:
    _, admission = policy()
    names = (
        admission["required_checks"]
        + admission["vessel_checks"]
        + admission["iceberg_checks"]
    )
    save(
        output,
        {
            "policy": ref(POLICY),
            "native_specification": ref(SPEC),
            "product_id": None,
            "name": None,
            "object_id": None,
            "source_identity": None,
            "repeated_object_group": None,
            "geographic_group": None,
            "geometry": None,
            "split": "train",
            "class": None,
            "provisional": True,
            "prior_exposure_disclosed": True,
            "annotation_completed_utc": None,
            "first_score_utc": None,
            "scores_viewed": False,
            "bbox_px": None,
            "native": None,
            "annotation": None,
            "checks": {name: {"passed": False, "evidence": []} for name in names},
        },
    )


def native_templates(prepared: Path, output: Path) -> dict[str, Any]:
    """Bridge saved C/D native preparation to deliberately incomplete dossiers."""
    if not output.resolve().is_relative_to(
        (REPO_ROOT / "data/source_supported").resolve()
    ):
        raise ValueError("Private project source directory required")
    payload = json.loads(prepared.read_text())
    frozen = inventory()
    planned = []
    for product in payload["products"]:
        for area in product["areas"]:
            native = {
                "product_id": product["product_id"],
                "name": product["name"],
                "split": area["split"],
                "geographic_group": product["scene"]["geographic_group"],
                "geometry": area["original"]["geometry"],
                "window": area["window"],
                "manifest": product["metadata"]["manifest"],
                "values": area["chip"],
                "bands": [
                    b
                    | area["bands"][i]
                    | {
                        "noise": next(
                            n["source"]
                            for n in product["metadata"]["noise"]
                            if n["polarization"] == b["polarization"]
                        )
                    }
                    for i, b in enumerate(product["bands"])
                ],
                "prepared": ref(prepared),
                "area_id": area["id"],
            }
            scope(native, frozen)
            planned.append(native)
    output.mkdir(parents=True, exist_ok=False)
    for n, native in enumerate(planned):
        native_path = output / f"native_{n:04d}.json"
        save(native_path, native)
        template_path = output / f"template_{n:04d}.json"
        template(template_path)
        # Keep the generic template as a reference. A distinct draft binds the
        # actual preparation while leaving every observation/evidence unset.
        draft = json.loads(template_path.read_text())
        draft.update(
            {
                k: native[k]
                for k in (
                    "product_id",
                    "name",
                    "split",
                    "geographic_group",
                    "geometry",
                )
            }
        )
        draft["native"] = ref(native_path)
        save(output / f"dossier_{n:04d}.json", draft)
    summary = {
        "prepared": ref(prepared),
        "native_receipts": len(planned),
        "labels_admitted": 0,
        "scores_viewed": False,
        "prior_project_exposure_disclosed": True,
        **GUARDS,
    }
    save(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--assess", type=Path)
    action.add_argument("--export", type=Path, help="Hash-bound dossier batch JSON")
    action.add_argument("--verify", type=Path, help="Export manifest JSON")
    action.add_argument("--template", type=Path)
    action.add_argument("--templates-from-prepared", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.templates_from_prepared:
        if not args.output:
            parser.error("Native templates require a fresh private output directory")
        print(json.dumps(native_templates(args.templates_from_prepared, args.output)))
    elif args.template:
        template(args.template)
    elif args.assess:
        print(json.dumps(assess(json.loads(args.assess.read_text()))[0]))
    elif args.verify:
        print(json.dumps(verify(args.verify)))
    elif args.output:
        print(json.dumps(export(args.export, args.output)))
    else:
        parser.error("Export requires a fresh private output path")
