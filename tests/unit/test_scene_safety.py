from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint

from agents.calibration import Calibrator
from agents.xview3_detector import XView3Detector


def paired_rasters(directory: Path) -> tuple[str, str]:
    paths = []
    for name in ("vv", "vh"):
        path = directory / f"{name}.tiff"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            gcps=[
                GroundControlPoint(row=0, col=0, x=-53, y=48),
                GroundControlPoint(row=0, col=2, x=-52.99, y=48),
                GroundControlPoint(row=2, col=0, x=-53, y=47.99),
                GroundControlPoint(row=2, col=2, x=-52.99, y=47.99),
            ],
            crs="EPSG:4326",
        ) as dst:
            dst.write(np.array([[1, 0], [1, 1]], dtype="uint16"), 1)
        paths.append(str(path))
    return paths[0], paths[1]


def fake_calibration(monkeypatch: Any) -> None:
    class Calibration:
        def to_sigma0_db(self, dn: Any, row: int, col: int) -> Any:
            return dn

    monkeypatch.setattr(Calibrator, "for_measurement", lambda _: Calibration())


def test_equivalent_gcp_objects_align_and_padding_cannot_generate_targets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    vv, vh = paired_rasters(tmp_path)
    fake_calibration(monkeypatch)
    detector = XView3Detector.__new__(XView3Detector)
    monkeypatch.setattr(
        detector,
        "detect_tile",
        lambda *_: [
            {"row": 0, "col": 0, "score": 0.5},  # real data
            {"row": 0, "col": 1, "score": 0.5},  # nodata
            {"row": 4, "col": 4, "score": 0.5},  # padding
        ],
    )
    result = list(detector.detect_scene(vv, vh, progress=False))
    assert len(result) == 1
    assert result[0]["row"] == result[0]["col"] == 0
    assert detector.scene_stats["nodata_detections_rejected"] == 2
    assert detector.scene_stats["tiles_processed"] == 1


def test_failed_model_tile_cannot_be_reported_as_complete(
    tmp_path: Path, monkeypatch: Any
) -> None:
    vv, vh = paired_rasters(tmp_path)
    fake_calibration(monkeypatch)
    detector = XView3Detector.__new__(XView3Detector)

    def failure(*args: Any) -> Any:
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(detector, "detect_tile", failure)
    with pytest.raises(RuntimeError, match="scene is incomplete"):
        list(detector.detect_scene(vv, vh, progress=False))
    assert detector.scene_stats["tiles_failed"] == 1
    assert detector.scene_stats["tiles_processed"] == 0
