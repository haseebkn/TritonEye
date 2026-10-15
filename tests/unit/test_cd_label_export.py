"""Execute provisional admission, actual calibration replay and atomic export."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box, mapping

from agents import cd_label_export as workflow
from agents.artifacts import REPO_ROOT, sha256_file
from agents.calibration import Calibrator
from agents.hh_hv_reservation import save
from agents.native_cd_input import SPEC
from agents.source_reservation import checked as real_checked
from agents.source_reservation import ref as real_ref


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    def reference(path: Path) -> dict[str, str]:
        path = path.resolve()
        if path.is_relative_to(tmp_path):
            return {
                "path": "fixture/" + path.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(path),
            }
        return real_ref(path)

    def checked(value: dict[str, str]) -> Path:
        if value["path"].startswith("data/native_training_exports/"):
            path = (tmp_path / value["path"]).resolve()
            if (
                not path.is_relative_to(tmp_path)
                or sha256_file(path) != value["sha256"]
            ):
                raise ValueError("Fixture hash/path mismatch")
            return path
        if not value["path"].startswith("fixture/"):
            return real_checked(value)
        path = (tmp_path / value["path"].removeprefix("fixture/")).resolve()
        if not path.is_relative_to(tmp_path) or sha256_file(path) != value["sha256"]:
            raise ValueError("Fixture hash/path mismatch")
        return path

    monkeypatch.setattr(workflow, "ref", reference)
    monkeypatch.setattr(workflow, "checked", checked)
    monkeypatch.setattr(workflow, "REPO_ROOT", tmp_path)
    frozen: dict[str, Any] = {
        "priors": [],
        "old": [],
        "exposed": [],
        "prior_metadata_hashes": {},
    }
    monkeypatch.setattr(workflow, "inventory", lambda: frozen)
    native: dict[str, Any] = {
        "product_id": "e4f3f146-305f-4dcf-bd9e-000000000001",
        "name": "S1C_IW_GRDH_1SDH_20260715T094924_20260715T094949_"
        "008552_010EE3_C13E.SAFE",
        "split": "train",
        "geographic_group": "new_cell_-56_50",
        "geometry": mapping(box(-55.501, 50.199, -55.489, 50.211)),
        "window": [0, 0, 100, 100],
        "bands": [],
    }
    manifest = tmp_path / "manifest.safe"
    manifest.write_bytes(
        b'<root><software name="Sentinel-1 IPF" version="004.03"/></root>'
    )
    native["manifest"] = reference(manifest)
    values = []
    for n, pol in enumerate(("HH", "HV")):
        header = (
            f"<adsHeader><missionId>S1C</missionId><productType>GRD</productType>"
            f"<polarisation>{pol}</polarisation><mode>IW</mode>"
            "<startTime>2026-07-15T09:49:24</startTime>"
            "<stopTime>2026-07-15T09:49:49</stopTime>"
            "<absoluteOrbitNumber>8552</absoluteOrbitNumber>"
            f"<missionDataTakeId>{int('010EE3', 16)}</missionDataTakeId></adsHeader>"
        )
        grid = "".join(
            f"<geolocationGridPoint><line>{y}</line><pixel>{x}</pixel>"
            f"<longitude>{-55.5 + x * 0.0001}</longitude>"
            f"<latitude>{50.2 + y * 0.0001}</latitude>"
            "<incidenceAngle>40</incidenceAngle></geolocationGridPoint>"
            for y in (0, 50, 100)
            for x in (0, 50, 100)
        )
        annotation = tmp_path / f"{pol}_annotation.xml"
        annotation.write_bytes(
            (
                f"<root>{header}<mode>IW</mode>"
                "<rangePixelSpacing>10</rangePixelSpacing><azimuthPixelSpacing>10</azimuthPixelSpacing>"
                "<incidenceAngleMidSwath>40</incidenceAngleMidSwath>"
                "<thermalNoiseCorrectionPerformed>false</thermalNoiseCorrectionPerformed>"
                "<numberOfSamples>100</numberOfSamples><numberOfLines>100</numberOfLines>"
                f"<azimuthTimeInterval>0.1</azimuthTimeInterval>{grid}</root>"
            ).encode()
        )
        calibration = tmp_path / f"{pol}_calibration.xml"
        vectors = "".join(
            f"<calibrationVector><line>{y}</line><pixel>0 99</pixel>"
            "<sigmaNought>10 10</sigmaNought></calibrationVector>"
            for y in (0, 99)
        )
        calibration.write_bytes(f"<root>{header}{vectors}</root>".encode())
        noise = tmp_path / f"{pol}_noise.xml"
        noise.write_bytes(f"<root>{header}</root>".encode())
        dn = np.full((100, 100), 100 + n * 20, dtype="uint16")
        path = tmp_path / f"{pol}_dn.npy"
        np.save(path, dn, allow_pickle=False)
        values.append(Calibrator(str(calibration)).to_sigma0_db(dn).astype("float32"))
        native["bands"].append(
            {
                "polarization": pol,
                "annotation": reference(annotation),
                "calibration": reference(calibration),
                "noise": reference(noise),
                "dn": reference(path),
            }
        )
    path = tmp_path / "values.npy"
    np.save(path, np.stack(values), allow_pickle=False)
    native["values"] = reference(path)
    save(tmp_path / "native.json", native)
    dossier: dict[str, Any] = {
        **{
            k: native[k]
            for k in ("product_id", "name", "geometry", "geographic_group", "split")
        },
        "policy": reference(workflow.POLICY),
        "native_specification": reference(SPEC),
        "native": reference(tmp_path / "native.json"),
        "object_id": "synthetic-01",
        "source_identity": "synthetic-ship",
        "repeated_object_group": "synthetic-ship",
        "class": "vessel",
        "provisional": True,
        "prior_exposure_disclosed": True,
        "annotation_completed_utc": "2026-10-15T10:00:00Z",
        "first_score_utc": None,
        "scores_viewed": False,
        "bbox_px": [48, 48, 52, 52],
    }
    dossier = json.loads(json.dumps(dossier))
    save(tmp_path / "annotation.json", dossier)
    dossier["annotation"] = reference(tmp_path / "annotation.json")
    _, rules = workflow.policy()
    names = rules["required_checks"] + rules["vessel_checks"] + rules["iceberg_checks"]
    save(
        tmp_path / "evidence.json",
        {
            **{
                k: dossier[k]
                for k in (
                    "product_id",
                    "object_id",
                    "source_identity",
                    "repeated_object_group",
                    "native",
                    "annotation",
                )
            },
            "checks": {name: True for name in names},
        },
    )
    dossier["checks"] = {
        name: {"passed": True, "evidence": [reference(tmp_path / "evidence.json")]}
        for name in names
    }
    return {
        "dossier": dossier,
        "native": native,
        "frozen": frozen,
        "tmp": tmp_path,
        "values": np.stack(values),
        "reference": reference,
    }


def batch(case: dict[str, Any], dossiers: list[dict[str, Any]]) -> Path:
    root = case["tmp"]
    refs = []
    for n, dossier in enumerate(dossiers):
        path = root / f"dossier_{n}.json"
        save(path, dossier)
        refs.append(case["reference"](path))
    path = root / "batch.json"
    save(path, {"policy": case["reference"](workflow.POLICY), "dossiers": refs})
    return Path(path)


def test_admits_only_provisional_training_and_exports_exact_chips(
    case: dict[str, Any],
) -> None:
    result, tensor = workflow.assess(case["dossier"])
    assert result["exploratory_training_admitted"] is True, result["blockers"]
    assert tensor is not None
    np.testing.assert_array_equal(tensor, case["values"][:, 13:88, 13:88])
    assert result["native_metadata"]["native_center_utc"] == "2026-07-15T09:49:29+00:00"
    assert result["training_approved"] is False
    assert result["accuracy_truth_admitted"] is False
    assert result["promotion_eligible"] is False
    output = case["tmp"] / "data/native_training_exports/batch1"
    assert workflow.export(batch(case, [case["dossier"]]), output)["exported"] == 1
    assert workflow.verify(output / "manifest.json")["labels_admitted"] == 1
    np.testing.assert_array_equal(np.load(output / "chip_0000.npy"), tensor)
    assert not (output / "ignore_mask.npy").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "validation"),
        ("split", "test"),
        ("class", "unresolved"),
        ("scores_viewed", True),
        ("first_score_utc", "2026-10-15T09:00:00Z"),
        ("first_score_utc", "invalid"),
        ("prior_exposure_disclosed", False),
        ("native_specification", {}),
        ("policy", {}),
        ("source_identity", ""),
        ("repeated_object_group", ""),
        ("bbox_px", [1, 1, 2, 2]),
    ],
)
def test_incomplete_and_leaking_dossiers_stay_held(
    case: dict[str, Any], field: str, value: Any
) -> None:
    case["dossier"][field] = value
    decision, tensor = workflow.assess(case["dossier"])
    assert not decision["exploratory_training_admitted"]
    assert tensor is None


def test_all_evidence_gates_are_effective_and_no_pixels_read_on_hold(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow, "native_tensor", lambda *args: pytest.fail("Pixels read on hold")
    )
    _, rules = workflow.policy()
    for name in rules["required_checks"] + rules["vessel_checks"]:
        dossier = deepcopy(case["dossier"])
        dossier["checks"][name]["passed"] = False
        decision, tensor = workflow.assess(dossier)
        assert name in decision["blockers"] and tensor is None


def test_validation_role_and_repeated_identity_preserved(case: dict[str, Any]) -> None:
    dossier = case["dossier"]
    case["frozen"]["old"] = [
        {"source_identity": dossier["source_identity"], "split": "validation"}
    ]
    assert not workflow.assess(dossier)[0]["exploratory_training_admitted"]
    case["frozen"]["old"] = []
    case["frozen"]["priors"] = [
        {
            "geometry": dossier["geometry"],
            "group": dossier["geographic_group"],
            "split": "validation",
            "quarantine": False,
        }
    ]
    assert not workflow.assess(dossier)[0]["exploratory_training_admitted"]


def test_calibration_and_metadata_tampering_are_detected(case: dict[str, Any]) -> None:
    native = deepcopy(case["native"])
    values = case["values"].copy()
    values[0, 50, 50] += 1
    path = case["tmp"] / "tampered.npy"
    np.save(path, values)
    native["values"] = case["reference"](path)
    with pytest.raises(ValueError, match="calibration replay"):
        workflow.native_tensor(native, [48, 48, 52, 52], case["frozen"])
    native = deepcopy(case["native"])
    native["name"] = native["name"].replace("008552", "008553")
    with pytest.raises(ValueError, match="orbit/datatake"):
        workflow.native_tensor(native, [48, 48, 52, 52], case["frozen"])


def test_native_geography_checked_before_pixel_read(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    native = deepcopy(case["native"])
    native["geometry"] = mapping(box(-55.501, 50.199, -55.499, 50.201))
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("Read escaped pixels")
    )
    with pytest.raises(ValueError, match="escape declared scope"):
        workflow.native_tensor(native, [48, 48, 52, 52], case["frozen"])


def test_iceberg_requires_physical_corroboration(case: dict[str, Any]) -> None:
    dossier = case["dossier"]
    dossier["class"] = "iceberg"
    dossier["checks"]["additional_physical_corroboration"]["passed"] = False
    decision, tensor = workflow.assess(dossier)
    assert "additional_physical_corroboration" in decision["blockers"]
    assert tensor is None


def test_wrong_evidence_subject_and_tampered_hash_cannot_admit(
    case: dict[str, Any],
) -> None:
    dossier = case["dossier"]
    name = "track_supports_vessel_identity"
    record = json.loads((case["tmp"] / "evidence.json").read_text())
    record["object_id"] = "another"
    path = case["tmp"] / "wrong.json"
    save(path, record)
    dossier["checks"][name]["evidence"] = [case["reference"](path)]
    assert (
        name + ":evidence_subject_or_result_mismatch"
        in workflow.assess(dossier)[0]["blockers"]
    )
    dossier["checks"][name]["evidence"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/path"):
        workflow.assess(dossier)


def variant(
    case: dict[str, Any], object_id: str, source_identity: str, group: str
) -> dict[str, Any]:
    """Build another otherwise-eligible dossier sharing the same native receipt."""
    dossier = deepcopy(case["dossier"])
    dossier["object_id"] = object_id
    dossier["source_identity"] = source_identity
    dossier["repeated_object_group"] = group
    root = case["tmp"]
    annotation_keys = (
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
    annotation_path = root / f"annotation_{object_id}.json"
    save(annotation_path, {k: dossier[k] for k in annotation_keys})
    dossier["annotation"] = case["reference"](annotation_path)
    evidence = {
        **{
            k: dossier[k]
            for k in (
                "product_id",
                "object_id",
                "source_identity",
                "repeated_object_group",
                "native",
                "annotation",
            )
        },
        "checks": {name: True for name in dossier["checks"]},
    }
    evidence_path = root / f"evidence_{object_id}.json"
    save(evidence_path, evidence)
    dossier["checks"] = {
        name: {"passed": True, "evidence": [case["reference"](evidence_path)]}
        for name in dossier["checks"]
    }
    return dossier


def test_repeated_observations_of_one_identity_share_a_group(
    case: dict[str, Any],
) -> None:
    first = variant(case, "repeated-a1", "vessel-shared", "group-shared")
    second = variant(case, "repeated-a2", "vessel-shared", "group-shared")
    output = case["tmp"] / "data/native_training_exports/shared_identity"
    assert workflow.export(batch(case, [first, second]), output)["exported"] == 2
    assert workflow.verify(output / "manifest.json")["labels_admitted"] == 2


def test_distinct_identities_cannot_share_one_repeated_group(
    case: dict[str, Any],
) -> None:
    first = variant(case, "conflict-a1", "vessel-A", "group-conflict")
    second = variant(case, "conflict-b1", "vessel-B", "group-conflict")
    output = case["tmp"] / "data/native_training_exports/conflicting_group"
    with pytest.raises(ValueError, match="inconsistent source identities"):
        workflow.export(batch(case, [first, second]), output)
    assert not output.exists()


def test_verify_rejects_conflicting_group_smuggled_into_a_manifest(
    case: dict[str, Any],
) -> None:
    first = variant(case, "smuggle-a1", "vessel-C", "group-smuggle")
    output = case["tmp"] / "data/native_training_exports/smuggled_group"
    workflow.export(batch(case, [first]), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    second = variant(case, "smuggle-b1", "vessel-D", "group-smuggle")
    second_path = case["tmp"] / "smuggled_dossier.json"
    save(second_path, second)
    second_ref = case["reference"](second_path)
    new_batch_path = case["tmp"] / "smuggled_batch.json"
    save(
        new_batch_path,
        {
            "policy": case["reference"](workflow.POLICY),
            "dossiers": [manifest["entries"][0]["dossier"], second_ref],
        },
    )
    manifest["batch"] = case["reference"](new_batch_path)
    manifest["entries"].append(
        {
            "dossier": second_ref,
            "product_id": second["product_id"],
            "native_acquisition_group": manifest["entries"][0][
                "native_acquisition_group"
            ],
            "object_id": second["object_id"],
            "class": second["class"],
            "geographic_group": second["geographic_group"],
            "repeated_object_group": second["repeated_object_group"],
            "tensor": manifest["entries"][0]["tensor"],
            "decision": manifest["entries"][0]["decision"],
        }
    )
    manifest["labels_admitted"] = len(manifest["entries"])
    manifest["class_counts"] = dict(Counter(e["class"] for e in manifest["entries"]))
    tampered_path = output / "tampered_manifest.json"
    save(tampered_path, manifest)
    with pytest.raises(ValueError, match="inconsistent source identities"):
        workflow.verify(tampered_path)


def test_duplicate_export_and_failure_publish_nothing(case: dict[str, Any]) -> None:
    output = case["tmp"] / "data/native_training_exports/duplicate"
    with pytest.raises(ValueError, match="Duplicate object"):
        workflow.export(batch(case, [case["dossier"], case["dossier"]]), output)
    assert not output.exists()


def test_failed_tensor_write_does_not_publish(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = batch(case, [case["dossier"]])
    output = case["tmp"] / "data/native_training_exports/failure"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        workflow.export(path, output)
    assert not output.exists()


def test_empty_export_is_honest_and_cannot_overwrite(case: dict[str, Any]) -> None:
    output = case["tmp"] / "data/native_training_exports/empty"
    path = batch(case, [])
    assert workflow.export(path, output)["exported"] == 0
    assert workflow.verify(output / "manifest.json")["labels_admitted"] == 0
    with pytest.raises(ValueError, match="Fresh private"):
        workflow.export(path, output)


def test_local_real_native_replay_when_available() -> None:
    prepared = REPO_ROOT / "data/ca_p2/prepared.json"
    if not prepared.exists():
        pytest.skip("Requires retained C/D native acceptance windows")
    payload = json.loads(prepared.read_text())
    frozen = workflow.inventory()
    for product in payload["products"]:
        for area in product["areas"]:
            native = {
                "product_id": product["product_id"],
                "name": product["name"],
                "split": "train",
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
            }
            tensor, metadata = workflow.native_tensor(
                native, [190, 190, 194, 194], frozen
            )
            assert tensor.shape == (2, 75, 75)
            assert metadata["processing_version"] == "004.03"


def test_quarantine_and_unsupported_processor_refuse_before_pixels(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = case["native"]
    case["frozen"]["priors"] = [
        {
            "geometry": native["geometry"],
            "group": "quarantined",
            "split": "quarantine",
            "quarantine": True,
        }
    ]
    original_load = np.load
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("Protected pixels read")
    )
    with pytest.raises(ValueError, match="quarantine"):
        workflow.native_tensor(native, [48, 48, 52, 52], case["frozen"])
    case["frozen"]["priors"] = []
    path = case["tmp"] / "unsupported.safe"
    path.write_bytes(b'<root><software name="Sentinel-1 IPF" version="004.04"/></root>')
    native["manifest"] = case["reference"](path)
    with pytest.raises(ValueError, match="processor version"):
        workflow.native_tensor(native, [48, 48, 52, 52], case["frozen"])
    monkeypatch.setattr(np, "load", original_load)


def test_input_source_hash_and_exported_tensor_tampering_detected(
    case: dict[str, Any],
) -> None:
    output = case["tmp"] / "data/native_training_exports/untampered"
    workflow.export(batch(case, [case["dossier"]]), output)
    saved = np.load(output / "chip_0000.npy")
    saved[0, 10, 10] += 1
    np.save(output / "chip_0000.npy", saved)
    with pytest.raises(ValueError, match="hash/path"):
        workflow.verify(output / "manifest.json")


def test_native_dossier_templates_remain_incomplete(case: dict[str, Any]) -> None:
    native = case["native"]
    prepared = case["tmp"] / "prepared.json"
    save(
        prepared,
        {
            "products": [
                {
                    "product_id": native["product_id"],
                    "name": native["name"],
                    "scene": {"geographic_group": native["geographic_group"]},
                    "metadata": {
                        "manifest": native["manifest"],
                        "noise": [
                            {"polarization": b["polarization"], "source": b["noise"]}
                            for b in native["bands"]
                        ],
                    },
                    "bands": native["bands"],
                    "areas": [
                        {
                            "id": "sample",
                            "split": "train",
                            "original": {"geometry": native["geometry"]},
                            "window": native["window"],
                            "chip": native["values"],
                            "bands": [{"dn": b["dn"]} for b in native["bands"]],
                        }
                    ],
                }
            ]
        },
    )
    output = case["tmp"] / "data/source_supported/templates"
    assert workflow.native_templates(prepared, output)["labels_admitted"] == 0
    dossier = json.loads((output / "dossier_0000.json").read_text())
    assert dossier["annotation_completed_utc"] is None
    assert dossier["class"] is None
    assert dossier["scores_viewed"] is False
    assert workflow.assess(dossier)[0]["disposition"] == "hold"
