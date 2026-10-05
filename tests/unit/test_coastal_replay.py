"""Replay executes derived artifacts while preserving original missions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, mapping

from agents.coastal_replay import replay

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
