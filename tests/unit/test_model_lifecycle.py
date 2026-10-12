"""Executable lifecycle contracts; all successful accuracy evidence is synthetic."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping

from agents import model_shadow, nl_benchmark
from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.landmask import LandMask
from agents.model_lifecycle import Lifecycle
from agents.nl_benchmark import objects_digest, selection_lock
from tests.unit.test_nl_benchmark import fixture


def register(store: Lifecycle, root: Path, status: str = "reference") -> str:
    path = root / f"{status}.jit"
    path.write_bytes(f"synthetic {status} artifact, never executed".encode())
    return store.register(
        path,
        {
            "adapter": "xview3",
            "polarizations": ["vv", "vh"],
            "status": status,
            "source": "synthetic fixture",
            "licence": "synthetic test only",
            "experiment": "simulation",
            "code_commit": "synthetic",
            "weights_sha256": sha256_file(path),
            "training_dataset_release_sha256": "a" * 64,
        },
        actor="synthetic test actor",
    )


def synthetic_dataset(root: Path, *, reviewed: bool = True) -> Path:
    """Five independent validation blocks with real hashed synthetic GeoTIFFs.

    The held-out test asset and mixed review bundle deliberately do not exist.
    Names/reviewer declarations are invented to exercise the guard, not SAR truth.
    """
    directory = root / "datasets/synthetic"
    directory.mkdir(parents=True)
    manifest, selection, _, annotations, history = fixture()
    base = copy.deepcopy(manifest["scenes"][1])
    manifest["scenes"] = [manifest["scenes"][2]]
    selection["scenes"] = [selection["scenes"][2]]
    annotations["objects"] = [annotations["objects"][2]]
    history["events"] = [history["events"][2]]
    manifest["licence"] = {"url": "synthetic licence", "snapshot": "licence.txt"}
    (root / "licence.txt").write_text("synthetic")
    manifest["licence"]["sha256"] = sha256_file(root / "licence.txt")
    for i in range(5):
        lon, lat = -55 + i * 0.4, 49.3
        scene = copy.deepcopy(base)
        name = (
            f"S1A_IW_GRDH_1SDV_2026090{i+1}T210000_2026090{i+1}T210025_"
            f"00001{i}_00001{i}_ABCD.SAFE"
        )
        rid = f"synthetic_{i}"
        roi = scene["rois"][0]
        roi.update(
            id=rid,
            group=rid,
            center=[lon, lat],
            geometry=mapping(box(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)),
        )
        scene.update(
            name=name,
            product_id=f"00000000-0000-4000-8000-{i+10:012d}",
            acquisition_group=f"S1A_00001{i}_00001{i}",
            acquisition_time=f"2026-09-0{i+1}T21:00:00Z",
            acquisition_end=f"2026-09-0{i+1}T21:00:25Z",
            footprint=mapping(box(lon - 0.15, lat - 0.15, lon + 0.15, lat + 0.15)),
        )
        spec = {k: roi[k] for k in ("id", "group", "regime", "center", "size")}
        selected = {
            k: scene[k]
            for k in ("name", "split", "region", "season", "prior_development_exposure")
        }
        selected["rois"] = [spec]
        selection["scenes"].append(selected)
        chip = root / f"{rid}.tif"
        with rasterio.open(
            chip,
            "w",
            driver="GTiff",
            width=256,
            height=256,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(lon - 0.1, lat + 0.1, 0.2 / 256, 0.2 / 256),
        ) as src:
            src.write(np.ones((2, 256, 256), dtype="float32"))
        item = {"path": chip.name, "sha256": sha256_file(chip)}
        scene.update(sources=[item], catalogue=item)
        scene["ais"]["snapshot"] = item
        roi.update(chip=item, calibration=[item], previews=[item], review_grid=item)
        objects = []
        for j in range(10):
            x, y = 20 + (j % 5) * 40, 50 + (j // 5) * 80
            objects.append(
                {
                    "id": f"{rid}_{j}",
                    "roi_id": rid,
                    "product_id": scene["product_id"],
                    "class": "vessel",
                    "kind": "object",
                    "bbox_px": [x - 1, y - 1, x + 1, y + 1],
                    "geometry": mapping(
                        Point(lon - 0.1 + x * 0.2 / 256, lat + 0.1 - y * 0.2 / 256)
                    ),
                    "decision": "synthetic vessel",
                    "evidence": ["synthetic"],
                    "annotator": "synthetic first",
                }
            )
        annotations["objects"].extend(objects)
        first = {
            "id": rid,
            "kind": "first_pass",
            "roi_id": rid,
            "timestamp": "2026-10-05T23:00:00Z",
            "reviewer": "synthetic first",
            "source_chip_sha256": item["sha256"],
            "annotation_digest": objects_digest(objects),
            "cells_reviewed": ["0_0"],
            "decision": "complete_first_pass",
            "notes": "synthetic full-area fixture",
        }
        history["events"].append(first)
        if reviewed:
            history["events"].append(
                {
                    **first,
                    "id": rid + "review",
                    "kind": "independent_review",
                    "reviewer": "synthetic second",
                    "independent": True,
                    "decision": "accepted",
                    "qualifications": "synthetic simulation, no real reviewer",
                }
            )
        manifest["scenes"].append(scene)
    write_json(directory / "selection.json", selection)
    manifest["selection_sha256"] = sha256_file(directory / "selection.json")
    files = {
        "manifest.json": manifest,
        "split_lock.json": selection_lock(selection),
        "annotations.json": annotations,
        "review_history.json": history,
    }
    for name, content in files.items():
        write_json(directory / name, content)
    for name in ("ANNOTATION_GUIDE.md", "DATASET_CARD.md"):
        (directory / name).write_text("Synthetic simulation only")
    names = ["selection.json", *files, "ANNOTATION_GUIDE.md", "DATASET_CARD.md"]
    write_json(
        directory / "release.json",
        {
            "dataset_version": "test",
            "files": {n: sha256_file(directory / n) for n in names},
            "review_bundle": {"path": "absent_test_pixels.html", "sha256": "1" * 64},
        },
    )
    return directory


def rehearsal(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, reviewed: bool = True
) -> tuple[Lifecycle, str, str, dict[str, Any]]:
    dataset = synthetic_dataset(root, reviewed=reviewed)
    store = Lifecycle(root / "state/lifecycle.sqlite3")
    reference, candidate = register(store, root), register(store, root, "exploratory")
    store.set_reference(reference, actor="fixture", reason="synthetic reference")
    # Real release validation, chip hashing/geolocation, scoring and threshold
    # selection run. Only inference and physical coastline use synthetic inputs.
    monkeypatch.setattr(nl_benchmark, "REPO_ROOT", root)
    monkeypatch.setattr(
        LandMask,
        "for_footprint",
        lambda *a, **kw: LandMask([], "EPSG:4326", "synthetic", "synthetic"),
    )
    monkeypatch.setattr(
        model_shadow,
        "XView3Detector",
        lambda *a, **kw: type("Detector", (), {"device": "cpu"})(),
    )
    _, annotation, _ = nl_benchmark.load_dataset(dataset)

    def inference(
        scene: dict[str, Any], roi: dict[str, Any], *args: Any
    ) -> tuple[dict[str, Any], list[Any]]:
        return (
            {
                "type": "FeatureCollection",
                "sar_product_id": scene["product_id"],
                "sar_product": scene["name"],
                "acquisition_time": scene["acquisition_time"],
                "study_roi": roi["geometry"],
                "split": "validation",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": obj["geometry"],
                        "properties": {"detection_id": obj["id"], "confidence": 0.95},
                    }
                    for obj in annotation["objects"]
                    if obj["roi_id"] == roi["id"]
                ],
            },
            [],
        )

    monkeypatch.setattr(model_shadow, "infer_roi", inference)
    result = model_shadow.run_shadow(store, candidate, dataset, root, root / "shadow")
    # Runtime comparisons are tested separately with deterministic resource data.
    # OS scheduling is not a deterministic criterion in this synthetic rehearsal.
    for role in ("reference", "candidate"):
        for case in result["shadow"]["cases"][role]:
            case["resources"] = {"wall_seconds": 1.0, "rss_sampled_peak_bytes": 1024}
    result["shadow_id"] = store.save("shadows", result["shadow"])
    return store, reference, candidate, result


def test_successful_shadow_promotion_and_rollback_survive_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, reference, candidate, result = rehearsal(tmp_path, monkeypatch)
    assert (
        store.resolve()["model_id"] == reference
    )  # shadow has no deployment side effect
    evaluation = model_shadow.evaluate_shadow(store, result["shadow"])
    assert not evaluation["blockers"] and evaluation["selected_threshold"] == 0.05
    assert (
        evaluation["reports"]["candidate"]["validation_baseline"]["post_policy"]["tp"]
        == 50
    )
    event = store.promote(
        result["shadow_id"], actor="synthetic approver", reason="simulation"
    )
    assert event["revision"] == 2
    restarted = Lifecycle(store.path)
    assert restarted.resolve()["model_id"] == candidate
    restarted.rollback(
        reference,
        expected_revision=2,
        actor="synthetic operator",
        reason="simulated incident",
    )
    assert Lifecycle(store.path).resolve()["model_id"] == reference
    assert len(store.status()["events"]) == 3
    assert not store.resolve()["operational_alerts_enabled"]
    assert not (tmp_path / "test.tif").exists()


def test_unreviewed_shadow_cannot_promote_even_if_report_claims_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, reference, _, result = rehearsal(tmp_path, monkeypatch, reviewed=False)
    result["shadow"]["passed"] = True
    identifier = store.save("shadows", result["shadow"])
    with pytest.raises(ValueError, match="Promotion blocked"):
        store.promote(identifier, actor="operator", reason="try")
    assert store.resolve()["model_id"] == reference and store.status()["revision"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "prediction",
        "weights",
        "release",
        "code",
        "config",
        "scope",
        "stale",
        "resources",
        "scene",
        "false_alarms",
        "failed",
    ],
)
def test_changed_or_regressing_evidence_preserves_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    store, reference, candidate, result = rehearsal(tmp_path, monkeypatch)
    shadow = result["shadow"]
    case = shadow["cases"]["candidate"][0]
    if change == "prediction":
        Path(case["prediction_path"]).write_text("{}")
    elif change == "weights":
        Path(store.record("packages", candidate)["weights"]).write_bytes(b"changed")
    elif change == "release":
        Path(shadow["dataset"]).joinpath("release.json").write_text("{}")
    elif change == "code":
        shadow["code_sha256"] = {}
    elif change == "config":
        shadow["configuration_sha256"] = {}
    elif change == "scope":
        shadow["cases"]["candidate"].pop()
    elif change == "stale":
        shadow["reference_revision"] = 0
    elif change == "resources":
        case["resources"]["wall_seconds"] = 3
    elif change == "failed":
        case["processing_complete"] = False
    elif change == "scene":
        path = Path(case["prediction_path"])
        data = json.loads(path.read_text())
        data["sar_product_id"] = "00000000-0000-4000-8000-000000000099"
        write_json(path, data)
        case["prediction_sha256"] = sha256_file(path)
    else:
        path = Path(case["prediction_path"])
        data = json.loads(path.read_text())
        extra = copy.deepcopy(data["features"][0])
        extra["properties"]["detection_id"] = "false"
        extra["geometry"] = mapping(Point(-55, 49.3))
        data["features"].append(extra)
        write_json(path, data)
        case["prediction_sha256"] = sha256_file(path)
    identifier = store.save("shadows", shadow)
    with pytest.raises(ValueError):
        store.promote(identifier, actor="operator", reason="try")
    assert store.resolve()["model_id"] == reference and store.status()["revision"] == 1


def test_stale_and_unknown_rollback_are_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, reference, candidate, result = rehearsal(tmp_path, monkeypatch)
    store.promote(result["shadow_id"], actor="operator", reason="simulation")
    unknown = register(store, tmp_path, "exploratory")
    for target, revision in ((reference, 1), (unknown, 2)):
        with pytest.raises(ValueError):
            store.rollback(
                target, expected_revision=revision, actor="operator", reason="try"
            )
    assert (
        store.resolve()["model_id"] == candidate and len(store.status()["events"]) == 2
    )


def test_real_release_stays_unreviewed_and_test_scope_is_rejected() -> None:
    dataset = REPO_ROOT / "datasets/nl_benchmark/v0.2.0"
    manifest, statuses = model_shadow.validation_scope(dataset)
    assert all(not s["metric_ready"] for s in statuses.values())
    scene = next(s for s in manifest["scenes"] if s["split"] == "test")
    with pytest.raises(ValueError, match="Held-out test"):
        nl_benchmark.require_development_scene(scene["product_id"], scene)


def test_quarantined_training_lineage_rejected_before_weights_read(
    tmp_path: Path,
) -> None:
    store = Lifecycle(tmp_path / "state/store.sqlite3")
    metadata = {
        "adapter": "xview3",
        "polarizations": ["vv", "vh"],
        "status": "exploratory",
        "source": "test",
        "licence": "test",
        "experiment": "test",
        "code_commit": "test",
        "training_dataset_release_sha256": sha256_file(
            REPO_ROOT / "datasets/allenai_nl/v0.1.0/release.json"
        ),
    }
    with pytest.raises(ValueError, match="Quarantined"):
        store.register(tmp_path / "missing.jit", metadata, actor="test")
