import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rasterio.control import GroundControlPoint
from rasterio.io import MemoryFile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.evaluate.evaluate_agent import evaluate, flatten_metrics

ACQ = "2026-08-18 09:15:00"


def write_gcp_raster(path: Path) -> None:
    """A georeferenced raster whose GCP hull spans lon -53..-52, lat 47..48."""
    gcps = [
        GroundControlPoint(
            row=float(r), col=float(c), x=-53.0 + c / 1000.0, y=48.0 - r / 1000.0
        )
        for r in (0, 1000)
        for c in (0, 1000)
    ]
    profile = dict(driver="GTiff", dtype="uint8", width=1001, height=1001, count=1)
    with MemoryFile() as mem:
        with mem.open(gcps=gcps, crs="EPSG:4326", **profile) as dst:
            dst.write(np.zeros((1001, 1001), "uint8"), 1)
        data = mem.read()
    path.write_bytes(data)


def write_detections(path: Path, points: List[Tuple[float, float]]) -> None:
    """Small square polygons centred on each (lon, lat)."""
    feats = []
    for i, (lon, lat) in enumerate(points):
        d = 0.0005
        ring = [
            [lon - d, lat - d],
            [lon + d, lat - d],
            [lon + d, lat + d],
            [lon - d, lat + d],
            [lon - d, lat - d],
        ]
        feats.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"target_id": f"TRITON-{i:03d}", "confidence": 0.9},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def write_ais(path: Path, vessels: List[Tuple[int, float, float]]) -> None:
    lines = ["mmsi,lat,lon,timestamp,speed_knots,course_deg"]
    for mmsi, lon, lat in vessels:
        lines.append(f"{mmsi},{lat},{lon},{ACQ},10.0,90.0")
    path.write_text("\n".join(lines) + "\n")


def build_payload(
    tmp_path: Path,
    detections: List[Tuple[float, float]],
    vessels: List[Tuple[int, float, float]],
    coverage: str = "marinecadastre",
) -> Dict[str, Any]:
    raster = tmp_path / "vv.tif"
    det = tmp_path / "detections.geojson"
    ais = tmp_path / "ais.csv"
    write_gcp_raster(raster)
    write_detections(det, detections)
    write_ais(ais, vessels)
    return {
        "mission_id": "mission_test",
        "acquisition_time": ACQ,
        "sar_bands": {"VV": str(raster)},
        "detections_geojson": str(det),
        "ais_telemetry": str(ais),
        "spatial_bounds": {"ais_coverage": coverage},
    }


# --------------------------------------------------------------------------


def test_perfect_recall(tmp_path: Path) -> None:
    v = [(316000001, -52.5, 47.5), (316000002, -52.6, 47.6)]
    payload = build_payload(
        tmp_path, [(-52.5, 47.5), (-52.6, 47.6)], [(m, lon, lat) for m, lon, lat in v]
    )
    ev = evaluate(payload)
    assert ev["scored"]
    assert ev["ais_vessels_in_swath"] == 2
    assert ev["matched"] == 2
    assert ev["recall_all"] == 1.0


def test_zero_recall_when_detections_are_elsewhere(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path,
        [(-52.1, 47.1)],  # far from the vessel
        [(316000001, -52.9, 47.9)],
    )
    ev = evaluate(payload)
    assert ev["scored"]
    assert ev["matched"] == 0
    assert ev["recall_all"] == 0.0


def test_vessels_outside_swath_are_not_counted(tmp_path: Path) -> None:
    # A vessel well outside the GCP hull was never imaged, so counting it as a
    # miss would understate recall.
    payload = build_payload(
        tmp_path,
        [(-52.5, 47.5)],
        [(316000001, -52.5, 47.5), (316000002, -10.0, 10.0)],
    )
    ev = evaluate(payload)
    assert ev["ais_vessels_in_swath"] == 1
    assert ev["recall_all"] == 1.0


def test_no_ais_coverage_is_unscored_not_zero(tmp_path: Path) -> None:
    # The critical distinction: missing telemetry must not look like a detector
    # that found nothing.
    payload = build_payload(
        tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)], coverage="none"
    )
    ev = evaluate(payload)
    assert ev["scored"] is False
    assert "recall_all" not in ev
    assert "no real AIS ground truth" in ev["reason"]


def test_mock_coverage_is_unscored(tmp_path: Path) -> None:
    payload = build_payload(
        tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)], coverage="mock"
    )
    ev = evaluate(payload)
    assert ev["scored"] is False


def test_empty_detections_scores_zero_not_error(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [], [(316000001, -52.5, 47.5)])
    ev = evaluate(payload)
    assert ev["scored"]
    assert ev["detections"] == 0
    assert ev["recall_all"] == 0.0


def test_flatten_metrics_produces_numeric_only() -> None:
    ev = {
        "scored": True,
        "ais_vessels_in_swath": 10,
        "matched": 3,
        "recall_all": 0.3,
        "reason": "should be dropped",
        "by_length": {"50_100m": {"in_swath": 4, "matched": 1, "recall": 0.25}},
    }
    flat = flatten_metrics(ev)
    assert flat["eval.recall_all"] == 0.3
    assert flat["eval.matched"] == 3.0
    assert flat["eval.recall_50_100m"] == 0.25
    assert flat["eval.count_50_100m"] == 4.0
    assert all(isinstance(v, float) for v in flat.values())
    assert "reason" not in flat
