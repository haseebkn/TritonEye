"""Replay executes derived artifacts while preserving original missions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, mapping

from agents.coastal_benchmark import compare_buffers
from agents.coastal_replay import replay
from agents.nl_benchmark import DEFAULT_DATASET

PRODUCT = "66d3167d-240a-459a-8066-75b9d2a458f2"


def inputs(root: Path) -> Path:
    (root / "configs/coastline").mkdir(parents=True)
    (root / "configs/model.yaml").write_text(
        "landmask:\n  enabled: true\n  coastal_buffer_m: 300\n"
    )
    controls = {
        "type": "FeatureCollection",
        "reference": {"scope": "synthetic test"},
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(-55.05, 49.24)),
                "properties": {"id": "water", "expected": "water"},
            }
        ],
    }
    (root / "configs/coastline/site.geojson").write_text(json.dumps(controls))
    (root / "configs/coastline/registry.json").write_text(
        json.dumps(
            {
                "type": "ShorelineControlRegistry",
                "collections": [{"location": "synthetic", "path": "site.geojson"}],
            }
        )
    )
    raster = root / "vv.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint8",
        crs=4326,
        transform=from_origin(-55.1, 49.3, 0.001, 0.001),
    ) as dst:
        dst.write(np.ones((100, 100), dtype="uint8"), 1)
    detections = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(lon, 49.24)),
                "properties": {"target_id": str(i), "surface": "water"},
            }
            for i, lon in enumerate([-55.05, -55.03])
        ],
    }
    (root / "detections.geojson").write_text(json.dumps(detections))
    (root / "ais.csv").write_text("mmsi,lat,lon,timestamp\n")
    payload = {
        "sar_product_id": PRODUCT,
        "acquisition_time": "2026-09-27T21:30:23Z",
        "detections_geojson": str(root / "detections.geojson"),
        "sar_bands": {"VV": str(raster)},
        "ais_telemetry": str(root / "ais.csv"),
        "spatial_bounds": {"ais_coverage": "none"},
        "mode": "mock",
    }
    (root / "payload.json").write_text(json.dumps(payload))
    manifest = root / "inventory.json"
    manifest.write_text(
        json.dumps(
            {
                "controls_registry": "configs/coastline/registry.json",
                "buffers_m": [0, 300],
                "not_covered": ["Labrador"],
                "scenes": [
                    {
                        "product_id": PRODUCT,
                        "payload": "payload.json",
                        "split": "development",
                        "independent_labels": None,
                        "locations_to_check": ["synthetic"],
                    }
                ],
            }
        )
    )
    return manifest


def fake_classification(*args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return (
        ["coastal", "water"],
        np.array([100.0, 800.0]),
        {
            "status": "ok",
            "source": "synthetic",
            "coastal_buffer_m": 300,
            "physical_surfaces": ["water", "water"],
            "infrastructure_flags": [False, False],
            "shoreline_validation": {"status": "passed", "scope": "synthetic"},
        },
    )


def test_replay_preserves_originals_and_writes_separate_research_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = inputs(tmp_path)
    original = (tmp_path / "detections.geojson").read_bytes()
    monkeypatch.setattr("agents.coastal_replay.classify_surfaces", fake_classification)
    output = tmp_path / "derived"
    result = replay(tmp_path, manifest, output)
    assert result["status"] == "prebenchmark_unmeasured"
    assert (tmp_path / "detections.geojson").read_bytes() == original
    scene = result["scenes"][0]
    assert scene["regional_coverage"][0]["controls_on_valid_scene_pixels"] == 1
    assert scene["ais_evaluation"]["scored"] is False
    coastal = json.loads((output / PRODUCT / "coastal_research.geojson").read_text())
    open_water = json.loads(
        (output / PRODUCT / "open_water_research.geojson").read_text()
    )
    assert len(coastal["features"]) == len(open_water["features"]) == 1
    assert coastal["features"][0]["properties"]["physical_surface"] == "water"
    assert not coastal["features"][0]["properties"]["alert_eligible"]
    assert (
        "Harbour and coastal research (1)"
        in (output / PRODUCT / "report.html").read_text()
    )


def test_inventory_cannot_claim_nonimaged_control_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = inputs(tmp_path)
    data = json.loads(manifest.read_text())
    data["scenes"][0]["locations_to_check"] = ["Labrador"]
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr("agents.coastal_replay.classify_surfaces", fake_classification)
    with pytest.raises(ValueError, match="no controls on valid scene pixels"):
        replay(tmp_path, manifest, tmp_path / "derived")


@pytest.mark.parametrize("held_out", ["declared", "product", "datatake", "unresolved"])
def test_unlabelled_held_out_scene_cannot_compare_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held_out: str
) -> None:
    manifest_path = inputs(tmp_path)
    inventory = json.loads(manifest_path.read_text())
    scene = inventory["scenes"][0]
    payload_path = tmp_path / "payload.json"
    payload = json.loads(payload_path.read_text())
    detections_path = tmp_path / "detections.geojson"
    detections = json.loads(detections_path.read_text())
    released = json.loads((DEFAULT_DATASET / "manifest.json").read_text())
    locked = next(s for s in released["scenes"] if s["split"] == "test")
    if held_out == "declared":
        scene["split"] = detections["split"] = "test"
    elif held_out == "product":
        scene["product_id"] = payload["sar_product_id"] = locked["product_id"]
    elif held_out == "datatake":
        tokens = locked["name"].split("_")
        alias = "_".join(tokens[:8]) + "_FFFF.SAFE"
        payload["sar_product"] = detections["sar_product"] = alias
        detections["acquisition_group"] = "claimed_development_group"
    else:
        scene["product_id"] = payload["sar_product_id"] = (
            "00000000-0000-4000-8000-000000000099"
        )
    detections["sar_product_id"] = scene["product_id"]
    manifest_path.write_text(json.dumps(inventory))
    payload_path.write_text(json.dumps(payload))
    detections_path.write_text(json.dumps(detections))
    originals = {path: path.read_bytes() for path in (payload_path, detections_path)}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Held-out scene reached shoreline comparison")

    monkeypatch.setattr("agents.coastal_replay.classify_surfaces", forbidden)
    reason = "unresolved" if held_out == "unresolved" else "Held-out test"
    with pytest.raises(ValueError, match=reason):
        replay(tmp_path, manifest_path, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()
    for labels in (None, {"benchmark": {"split": "validation"}}):
        with pytest.raises(ValueError, match=reason):
            compare_buffers(detections, labels, scene["product_id"])
    assert all(path.read_bytes() == content for path, content in originals.items())
