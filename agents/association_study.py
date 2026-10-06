"""Detector-blind identity review and held-out NL correlation comparisons.

Unreviewed objects, nearest-neighbour identities and AIS absence are never truth.
The locked detection test remains locked; validation is held out from fitting.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

from agents.acquisition import native_acquisition_group, product_id
from agents.ais_validation import parse_utc, valid_mmsi
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.association import GEOD, aligned_ais, one_to_one_matches
from agents.baseline_context import require_context_split
from agents.baseline_statistics import (
    binomial_interval,
    block_interval,
    dependence_blocks,
)
from agents.correlation.correlation_agent import CORRELATION_RADIUS_M
from agents.nl_benchmark import (
    DEFAULT_DATASET,
    bounded_path,
    load_dataset,
    require_development_scene,
)
from agents.region import load_region
from agents.run_versions import digest
from agents.uncertainty_association import UncertaintyConfig, uncertainty_matches


def case_digest(case: dict[str, Any]) -> str:
    """Bind adjudication to exact targets, telemetry, scope and source bytes."""
    return digest({k: v for k, v in case.items() if k not in {"labels", "review"}})


def pilot_bundle() -> dict[str, Any]:
    """Export existing image-first annotations, not detector predictions or truth."""
    manifest, annotations, _ = load_dataset(verify_files=True)
    cases = []
    for scene in manifest["scenes"]:
        if scene["split"] == "test":
            continue
        ais_path = (
            REPO_ROOT / f"data/benchmarks/nl/0.1.0/ais/{scene['product_id']}.json"
        )
        telemetry = json.loads(ais_path.read_text(encoding="utf-8"))
        for roi in scene["rois"]:
            targets = []
            for obj in annotations["objects"]:
                if obj["roi_id"] != roi["id"] or obj["kind"] != "object":
                    continue
                lon, lat = obj["geometry"]["coordinates"]
                targets.append({"id": obj["id"], "lon": lon, "lat": lat})
            chip = f"data/benchmarks/nl/0.1.0/{roi['id']}/sigma0.tif"
            case = {
                "id": roi["id"],
                "product_id": scene["product_id"],
                "name": scene["name"],
                "acquisition_group": native_acquisition_group(scene["name"]),
                "acquisition_time": scene["acquisition_time"],
                "split": scene["split"],
                "geographic_group": roi["group"],
                "region": scene["region"],
                "regime": roi["regime"],
                "polarizations": scene["polarizations"],
                "study_roi": roi["geometry"],
                "targets": targets,
                # Include all recorded reports, not just detector-selected or
                # nearest tracks. Matching itself performs bounded alignment.
                "ais_records": telemetry["rows"],
                "sources": [
                    {
                        "kind": "sar",
                        "path": chip,
                        "sha256": sha256_file(REPO_ROOT / chip),
                    },
                    {
                        "kind": "ais",
                        "path": str(ais_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(ais_path),
                    },
                ],
                "labels": [
                    {
                        "target_id": t["id"],
                        "object_class": "unresolved",
                        "state": "unresolved",
                        "mmsis": [],
                        "decision": "Independent identity adjudication unavailable",
                        "evidence": [],
                    }
                    for t in targets
                ],
            }
            case["review"] = {
                "case_sha256": case_digest(case),
                "initial_annotator": "codex_primary_machine_assisted",
                "reviewer": None,
                "independent": False,
                "complete_target_inventory": False,
                "reviewed_at": None,
                "evidence": [],
            }
            cases.append(case)
    return {
        "schema_version": "1.0.0",
        "dataset_version": "association-pilot-0.1.0",
        "detection_release_sha256": sha256_file(DEFAULT_DATASET / "release.json"),
        "purpose": "Detector-blind provisional review; no reviewed matching examples",
        "cases": cases,
        "review_history": [],
    }


def validate_bundle(
    bundle: dict[str, Any], *, artifact_root: Path = REPO_ROOT
) -> list[dict[str, Any]]:
    """Fail closed on stale review, split leakage, identity and provenance errors."""
    if bundle.get("schema_version") != "1.0.0" or not bundle.get("dataset_version"):
        raise ValueError("Versioned association dataset required")
    manifest, _, _ = load_dataset()
    cases = bundle["cases"]
    ids: set[str] = set()
    region = load_region()
    projection = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
    for case in cases:
        if case["id"] in ids:
            raise ValueError("Duplicate case identity")
        ids.add(case["id"])
        if case["split"] not in {"train", "validation"}:
            raise ValueError("Locked test cannot be exported or compared")
        product_id(case["product_id"])
        group = native_acquisition_group(case["name"])
        if not group or group != case["acquisition_group"]:
            raise ValueError("Native acquisition identity mismatch")
        if parse_utc(case["acquisition_time"]) is None:
            raise ValueError("Explicit UTC acquisition time required")
        known = next(
            (s for s in manifest["scenes"] if s["product_id"] == case["product_id"]),
            None,
        )
        if known and any(
            case[k] != known[k]
            for k in ("name", "split", "acquisition_time", "polarizations")
        ):
            raise ValueError("Product metadata differs from the released scene")
        if case["polarizations"] not in (["vv", "vh"], ["hh", "hv"]):
            raise ValueError("Preserve a supported polarization branch")
        for key in ("geographic_group", "region", "regime"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError("Geographic strata required")
        roi = shape(case["study_roi"])
        load_region(roi)
        require_development_scene(case["product_id"], case)
        require_context_split(roi, case["split"], manifest)
        sources = case["sources"]
        if not {"sar", "ais"}.issubset({s["kind"] for s in sources}):
            raise ValueError(
                "SAR and AIS source provenance required, even if AIS is empty"
            )
        for source in sources:
            path = bounded_path(artifact_root, source["path"])
            if sha256_file(path) != source["sha256"]:
                raise ValueError("Association source hash mismatch")
        target_ids: set[str] = set()
        for target in case["targets"]:
            if not isinstance(target["id"], str) or target["id"] in target_ids:
                raise ValueError("Unique target identifiers required")
            target_ids.add(target["id"])
            point = Point(target["lon"], target["lat"])
            if not roi.covers(point) or not region.covers(point):
                raise ValueError("Target outside declared NL image area")
            instant = parse_utc(target.get("timestamp", case["acquisition_time"]))
            start = parse_utc(case["acquisition_time"])
            if (
                instant is None
                or start is None
                or abs((instant - start).total_seconds()) > 300
            ):
                raise ValueError("Invalid per-target UTC acquisition time")
        labels = case["labels"]
        if (
            len(labels) != len(target_ids)
            or {label["target_id"] for label in labels} != target_ids
        ):
            raise ValueError("Exactly one decision required for every selected target")
        observed = {str(valid_mmsi(r.get("mmsi"))) for r in case["ais_records"]}
        identities: set[str] = set()
        for label in labels:
            state, mmsis = label["state"], label["mmsis"]
            if state not in {"matched", "no_association", "ambiguous", "unresolved"}:
                raise ValueError("Invalid association truth state")
            if label["object_class"] not in {"vessel", "non_vessel", "unresolved"}:
                raise ValueError("Invalid reviewed object class")
            if len(mmsis) != len(set(mmsis)) or any(
                valid_mmsi(m) is None or m != str(valid_mmsi(m)) or m not in observed
                for m in mmsis
            ):
                raise ValueError(
                    "Truth identities must occur in supplied AIS observations"
                )
            if (
                (state == "matched" and len(mmsis) != 1)
                or (state == "ambiguous" and len(mmsis) < 2)
                or (state in {"unresolved", "no_association"} and mmsis)
                or (
                    state in {"matched", "ambiguous"}
                    and label["object_class"] != "vessel"
                )
                or (state == "no_association" and label["object_class"] == "unresolved")
            ):
                raise ValueError("Truth state and class/identity cardinality disagree")
            if state == "matched":
                if mmsis[0] in identities:
                    raise ValueError(
                        "One reviewed identity cannot label two SAR objects"
                    )
                identities.add(mmsis[0])
            if state != "unresolved" and (
                not label["decision"] or not label["evidence"]
            ):
                raise ValueError(
                    "Resolved labels need a decision and independent evidence"
                )
        review = case["review"]
        if review["case_sha256"] != case_digest(case):
            raise ValueError(
                "Review refers to stale targets, telemetry or source bytes"
            )
        if review["independent"] is True and (
            not review["reviewer"]
            or not review["initial_annotator"]
            or review["reviewer"] == review["initial_annotator"]
            or parse_utc(review["reviewed_at"]) is None
            or not review["evidence"]
        ):
            raise ValueError("Independent review identity, time and evidence required")
        if review["independent"] is True and not any(
            event.get("case_id") == case["id"]
            and event.get("case_sha256") == case_digest(case)
            and event.get("labels_sha256") == digest(labels)
            and event.get("reviewer") == review["reviewer"]
            and event.get("reviewed_at") == review["reviewed_at"]
            for event in bundle.get("review_history", [])
        ):
            raise ValueError(
                "Reviewed decisions require a matching review history event"
            )
    for i, case in enumerate(cases):
        for other in cases[:i]:
            if case["split"] == other["split"]:
                continue
            overlap = (
                transform(projection.transform, shape(case["study_roi"]))
                .buffer(manifest["separation_m"])
                .intersects(transform(projection.transform, shape(other["study_roi"])))
            )
            if overlap or any(
                case[k] == other[k]
                for k in ("product_id", "acquisition_group", "geographic_group")
            ):
                raise ValueError("Association development/validation split leakage")
    return list(cases)


def metric_ready(case: dict[str, Any]) -> bool:
    review = case["review"]
    return bool(
        review["independent"] is True
        and review["complete_target_inventory"] is True
        and all(label["state"] != "unresolved" for label in case["labels"])
    )


def identity_metrics(
    case: dict[str, Any], selected: dict[int, str], ambiguous: set[int]
) -> dict[str, Any]:
    """Ambiguous assignments are tentative, not accepted identity successes."""
    labels = {label["target_id"]: label for label in case["labels"]}
    counts: Counter[str] = Counter()
    for i, target in enumerate(case["targets"]):
        label = labels[target["id"]]
        state, prediction = label["state"], selected.get(i)
        accepted = prediction is not None and i not in ambiguous
        counts["targets"] += 1
        counts["ambiguous_outputs"] += i in ambiguous
        counts["unassociated_candidates"] += prediction is None and i not in ambiguous
        counts["accepted_identities"] += accepted
        counts["known_associations"] += state == "matched"
        counts["reviewed_ambiguous_cases"] += state == "ambiguous"
        if state == "matched":
            correct = prediction == label["mmsis"][0]
            counts["correct_tentative_identities"] += correct
            counts["correct_accepted_identities"] += correct and accepted
            counts["wrong_tentative_identities"] += (
                prediction is not None and not correct
            )
            counts["wrong_accepted_identities"] += accepted and not correct
            counts["missed_associations"] += not (correct and accepted)
            counts["identity_errors"] += not (correct and accepted)
        elif state == "no_association":
            counts["false_associations"] += accepted
            counts["identity_errors"] += accepted
        elif state == "ambiguous":
            counts["unsafe_ambiguity_resolutions"] += accepted
            counts["identity_errors"] += accepted
    keys = (
        "targets accepted_identities known_associations reviewed_ambiguous_cases "
        "correct_tentative_identities correct_accepted_identities "
        "wrong_tentative_identities wrong_accepted_identities "
        "missed_associations false_associations unsafe_ambiguity_resolutions "
        "identity_errors ambiguous_outputs unassociated_candidates"
    ).split()
    result: dict[str, Any] = {k: counts[k] for k in keys}
    for name, denominator in (
        ("accepted_identity_precision", counts["accepted_identities"]),
        ("association_recall", counts["known_associations"]),
    ):
        result[name] = (
            counts["correct_accepted_identities"] / denominator if denominator else None
        )
        result[name + "_interval"] = binomial_interval(
            counts["correct_accepted_identities"], denominator
        )
    return result


def paired_error_reduction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Positive reduction favours the experiment; paired whole-block resampling."""
    blocks = dependence_blocks(rows)
    result: dict[str, Any] = {
        "interval": None,
        "blocks": len(blocks),
        "seed": 20261006,
        "resamples": 2000,
    }
    if len(blocks) < 5:
        result["reason"] = (
            "At least five independent acquisition/geography blocks required"
        )
        return result
    values = np.array(
        [
            [
                sum(
                    rows[i]["geometric_errors"] - rows[i]["uncertainty_errors"]
                    for i in block
                ),
                sum(rows[i]["targets"] for i in block),
            ]
            for block in blocks
        ],
        dtype=float,
    )
    rng = np.random.default_rng(20261006)
    draws = values[rng.integers(0, len(blocks), (2000, len(blocks)))].sum(axis=1)
    if (draws[:, 1] <= 0).any():
        result["reason"] = "Undefined target denominator in resamples"
        return result
    rates = draws[:, 0] / draws[:, 1]
    if np.ptp(rates) == 0:
        result["reason"] = "Degenerate empirical distribution; no uncertainty claim"
        return result
    result["interval"] = np.quantile(rates, [0.025, 0.975]).tolist()
    result["reason"] = (
        "Conditional on reviewed selected examples, not operational performance"
    )
    return result


def reviewed_pair_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    """Count possible supervised pairs, not negatives inferred from AIS silence.

    Only reviewed unique-identity or explicitly adjudicated no-association
    targets contribute, against aligned AIS within a fixed 2 km retrieval area.
    Ambiguous identities and missing telemetry never supply negative pairs.
    """
    positive = negative = 0
    for case in cases:
        labels = {label["target_id"]: label for label in case["labels"]}
        for target in case["targets"]:
            label = labels[target["id"]]
            if label["state"] not in {"matched", "no_association"}:
                continue
            positions, _ = aligned_ais(
                pd.DataFrame(case["ais_records"]),
                target.get("timestamp", case["acquisition_time"]),
            )
            for row in positions.to_dict("records"):
                if not load_region().covers(row["geometry"]):
                    continue
                _, _, distance = GEOD.inv(
                    target["lon"], target["lat"], row["lon"], row["lat"]
                )
                if distance <= 2000:
                    is_positive = row["mmsi"] in label["mmsis"]
                    positive += is_positive
                    negative += not is_positive
    return {"positive": positive, "negative": negative, "total": positive + negative}


def compare_matchers(
    targets: list[dict[str, Any]],
    records: pd.DataFrame,
    acquisition_time: str,
    *,
    analysis_region: Any = None,
) -> dict[str, Any]:
    """Keep the production geometric control and share its aligned identity pool.

    The pool is defined at scene time, after alignment, not by raw report
    coverage. Experimental per-target alignment retains every report belonging
    to those identities but cannot introduce identities absent from the control.
    """
    region = load_region(analysis_region)
    positions, diagnostics = aligned_ais(records, acquisition_time)
    positions = positions.loc[positions.geometry.map(region.covers)]
    positions = positions.reset_index(drop=True)
    points = [Point(target["lon"], target["lat"]) for target in targets]
    geometric, _, ambiguous = one_to_one_matches(
        points,
        list(positions.geometry),
        [CORRELATION_RADIUS_M + float(u) for u in positions.uncertainty_m],
    )
    selected = {i: str(positions.iloc[j].mmsi) for i, (j, _) in geometric.items()}
    shared = set(positions.mmsi.astype(str))
    experimental = uncertainty_matches(
        targets, records, acquisition_time, admissible_mmsis=shared
    )
    return {
        "positions": positions,
        "ais_diagnostics": diagnostics,
        "geometric_selected": selected,
        "geometric_ambiguous": ambiguous,
        "experimental": experimental,
        "candidate_pool": {
            "scope": "scene-time-aligned NL identities; full report histories retained",
            "count": len(shared),
            "mmsis": sorted(shared),
        },
    }


def compare_bundle(
    bundle: dict[str, Any], *, artifact_root: Path = REPO_ROOT
) -> dict[str, Any]:
    cases = validate_bundle(bundle, artifact_root=artifact_root)
    rows = []
    for case in cases:
        if case["split"] != "validation":
            continue
        records = pd.DataFrame(case["ais_records"])
        comparison = compare_matchers(
            case["targets"], records, case["acquisition_time"]
        )
        positions = comparison["positions"]
        diagnostics = comparison["ais_diagnostics"]
        selected = comparison["geometric_selected"]
        ambiguous = comparison["geometric_ambiguous"]
        experimental = comparison["experimental"]
        ready = metric_ready(case)
        rows.append(
            {
                "id": case["id"],
                "acquisition_group": case["acquisition_group"],
                "geographic_group": case["geographic_group"],
                "region": case["region"],
                "regime": case["regime"],
                "polarizations": case["polarizations"],
                "targets": len(case["targets"]),
                "candidate_pool": comparison["candidate_pool"],
                "aligned_ais": len(positions),
                "aligned_ais_in_roi": sum(
                    shape(case["study_roi"]).covers(point)
                    for point in positions.geometry
                ),
                "ais_diagnostics": diagnostics,
                "measured": ready,
                "geometric": (
                    identity_metrics(case, selected, ambiguous) if ready else None
                ),
                "uncertainty": (
                    identity_metrics(
                        case, experimental["selected"], set(experimental["ambiguous"])
                    )
                    if ready
                    else None
                ),
                "descriptive_assignments": {
                    "geometric": len(selected),
                    "uncertainty": len(experimental["selected"]),
                },
                "assignments": [
                    {
                        "target_id": target["id"],
                        "geometric_mmsi": selected.get(i),
                        "geometric_ambiguous": i in ambiguous,
                        "uncertainty_mmsi": experimental["selected"].get(i),
                        "uncertainty_ambiguous": i in experimental["ambiguous"],
                    }
                    for i, target in enumerate(case["targets"])
                ],
                "uncertainty_edges": experimental["edges"],
            }
        )
    paired = [
        {
            "acquisition_group": r["acquisition_group"],
            "geographic_group": r["geographic_group"],
            "targets": r["targets"],
            "geometric_errors": r["geometric"]["identity_errors"],
            "uncertainty_errors": r["uncertainty"]["identity_errors"],
        }
        for r in rows
        if r["measured"]
    ]
    training = [c for c in cases if c["split"] == "train" and metric_ready(c)]
    train_labels = [label for c in training for label in c["labels"]]
    pair_counts = reviewed_pair_counts(training)
    enough = (
        pair_counts["total"] >= 500
        and pair_counts["positive"] >= 50
        and pair_counts["negative"] >= 50
        and len({c["geographic_group"] for c in training}) >= 5
        and len({c["acquisition_group"] for c in training}) >= 5
    )
    return {
        "dataset_version": bundle["dataset_version"],
        "dataset_sha256": digest(bundle),
        "step_6_complete": False,
        "improvement_demonstrated": False,
        "production_method_changed": False,
        "automatic_alerts": False,
        "configuration": asdict(UncertaintyConfig()),
        "reviewed_validation_cases": len(paired),
        "validation_cases": rows,
        "error_rate_intervals": {
            method: block_interval(paired, method + "_errors", "targets")
            for method in ("geometric", "uncertainty")
        },
        "paired_error_reduction": paired_error_reduction(paired),
        "paired_error_reduction_by_polarization": {
            pol: paired_error_reduction(
                [
                    pair
                    for pair, row in zip(paired, [r for r in rows if r["measured"]])
                    if "/".join(row["polarizations"]).upper() == pol
                ]
            )
            for pol in ("VV/VH", "HH/HV")
        },
        "polarization_case_ids": {
            pol: [r["id"] for r in rows if "/".join(r["polarizations"]).upper() == pol]
            for pol in ("VV/VH", "HH/HV")
        },
        "learned_ranker": {
            "implemented": False,
            "training_readiness": enough,
            "reviewed_training_targets": len(train_labels),
            "reviewed_training_pairs": pair_counts,
            "provisional_minima": {
                "pairs": 500,
                "positive_pairs": 50,
                "negative_pairs": 50,
                "acquisitions": 5,
                "geographic_groups": 5,
            },
        },
        "reason": (
            "No reviewed validation identities; no improvement measurement"
            if not paired
            else (
                "Descriptive held-out comparison only; "
                "paired improvement decision still required"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "compare"])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Choose a new output; previous review/evaluation artifacts are immutable"
        )
    if args.command == "pilot":
        result = pilot_bundle()
        validate_bundle(result)
    else:
        if args.bundle is None:
            raise ValueError("Comparison requires --bundle")
        result = compare_bundle(json.loads(args.bundle.read_text(encoding="utf-8")))
        result["bundle_file_sha256"] = sha256_file(args.bundle)
    result["runtime"] = {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in ("numpy", "pandas", "scipy", "pyproj", "shapely")
        },
    }
    result["code_sha256"] = {
        name: sha256_file(REPO_ROOT / "agents" / name)
        for name in (
            "association.py",
            "association_study.py",
            "uncertainty_association.py",
        )
    }
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "step_6_complete": False}))


if __name__ == "__main__":
    main()
