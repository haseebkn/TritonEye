import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agents import coastal_benchmark
from agents.inference import inference_agent


@pytest.mark.parametrize(
    "identity",
    ["reprocessed_test", "adjacent_test", "development", "missing", "malformed"],
)
def test_production_artifacts_enforce_datatake_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    identity: str,
) -> None:
    product = "00000000-0000-4000-8000-000000000099"
    name = "S1A_IW_GRDH_1SDV_20260209T214114_20260209T214138_" "063148_07ED3C_FFFF.SAFE"
    if identity == "adjacent_test":
        name = name.replace("214114_20260209T214138", "214139_20260209T214204")
    elif identity == "development":
        name = name.replace("063148_07ED3C", "063149_07ED3D")
    elif identity == "malformed":
        name = "unresolvable_product_name"
    raster = tmp_path / "sar.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(-55.1, 49.3, 0.001, 0.001),
    ) as dst:
        dst.write(np.ones((16, 16), dtype="uint16"), 1)
    payload: dict[str, Any] = {
        "mission_id": "production_identity",
        "mode": "production",
        "sar_product_id": product,
        "acquisition_time": "2026-02-09T21:41:14Z",
        "sar_bands": {"VV": str(raster), "VH": str(raster)},
    }
    if identity != "missing":
        payload["sar_product"] = name
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")
    monkeypatch.setenv("TRITONEYE_DETECTOR", "xview3")
    monkeypatch.setattr(inference_agent, "mission_directory", lambda _: tmp_path)
    monkeypatch.setattr(
        inference_agent,
        "load_yaml_config",
        lambda _: {"landmask": {"enabled": False}},
    )
    monkeypatch.setattr(
        inference_agent,
        "run_xview3_inference",
        lambda *args: ([(2, 2, 6, 6, 0.8, 4)], 1, ""),
    )
    monkeypatch.setattr(sys, "argv", ["inference", "--payload", json.dumps(payload)])
    inference_agent.main()
    output = json.loads(capsys.readouterr().out)
    artifacts = [
        json.loads((tmp_path / filename).read_text())
        for filename in (
            "detections.geojson",
            "outside_study_area.geojson",
            "processing.json",
            "inference_payload.json",
        )
    ]
    assert output["mode"] == "production"
    assert len(artifacts[0]["features"]) == 1
    expected_group = (
        None
        if identity in {"missing", "malformed"}
        else "S1A_063149_07ED3D" if identity == "development" else "S1A_063148_07ED3C"
    )
    for artifact in artifacts:
        assert artifact["sar_product_id"] == product
        assert artifact.get("sar_product") == payload.get("sar_product")
        assert artifact["acquisition_group"] == expected_group
    destination = tmp_path / "trials.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--detections",
            str(tmp_path / "detections.geojson"),
            "--product-id",
            product,
            "--output",
            str(destination),
        ],
    )
    if identity == "development":
        coastal_benchmark.main()
        result = json.loads(destination.read_text())
        assert all(row["measured"] is False for row in result["trials"])
    else:
        reason = "unresolved" if expected_group is None else "Held-out test datatake"
        with pytest.raises(ValueError, match=reason):
            coastal_benchmark.main()
        assert not destination.exists()


def test_declared_group_without_native_identity_cannot_unlock_comparison() -> None:
    product = "00000000-0000-4000-8000-000000000099"
    detections = {
        "type": "FeatureCollection",
        "features": [],
        "sar_product_id": product,
        "acquisition_group": "S1A_063149_07ED3D",
    }
    with pytest.raises(ValueError, match="unresolved"):
        coastal_benchmark.compare_buffers(detections, None, product)
