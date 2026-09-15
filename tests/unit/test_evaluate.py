import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.evaluate.evaluate_agent import evaluate, flatten_metrics, swath_hull

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
            dst.write(np.ones((1001, 1001), "uint8"), 1)
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
                "properties": {
                    "target_id": f"TRITON-{i:03d}",
                    "confidence": 0.9,
                    "surface": "water",
                },
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
    coverage: str = "partial",
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
    assert "no real AIS observations" in ev["reason"]


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


def test_one_detection_cannot_inflate_recall_for_two_ais_vessels(
    tmp_path: Path,
) -> None:
    payload = build_payload(
        tmp_path,
        [(-52.5, 47.5)],
        [(316000001, -52.5, 47.5), (316000002, -52.501, 47.5)],
    )
    ev = evaluate(payload)
    assert ev["matched"] == 1
    assert ev["recall_all"] == 0.5
    assert ev["raw_ambiguous_targets"] == 1
    assert ev["precision"] is None
    assert ev["false_positive_rate"] is None


def test_affine_projected_swath_is_nonempty_and_reprojected(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)])
    raster = Path(payload["sar_bands"]["VV"])
    x, y = Transformer.from_crs(4326, 32622, always_xy=True).transform(-52.5, 47.5)
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint16",
        crs="EPSG:32622",
        transform=from_origin(x - 500, y + 500, 10, 10),
    ) as dst:
        dst.write(np.ones((100, 100), dtype="uint16"), 1)
    hull = swath_hull(str(raster))
    assert not hull.is_empty
    assert hull.covers(Point(-52.5, 47.5))
    assert evaluate(payload)["recall_all"] == 1


def test_projected_gcp_hull_is_reprojected(tmp_path: Path) -> None:
    raster = tmp_path / "projected_gcps.tif"
    transform = Transformer.from_crs(4326, 32622, always_xy=True)
    gcps = []
    for row, lat in [(0, 48.0), (100, 47.0)]:
        for col, lon in [(0, -53.0), (100, -52.0)]:
            x, y = transform.transform(lon, lat)
            gcps.append(GroundControlPoint(row=row, col=col, x=x, y=y))
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=101,
        height=101,
        count=1,
        dtype="uint8",
        gcps=gcps,
        crs="EPSG:32622",
    ) as dst:
        dst.write(np.ones((101, 101), dtype="uint8"), 1)
    assert swath_hull(str(raster)).covers(Point(-52.5, 47.5))


def test_nodata_pixels_do_not_enter_ais_denominator(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)])
    with rasterio.open(payload["sar_bands"]["VV"], "r+") as dst:
        dst.write(np.zeros((1001, 1001), dtype="uint8"), 1)
    ev = evaluate(payload)
    assert ev["scored"] is False
    assert ev["ais_vessels_in_swath"] == 0


def test_raw_and_eligible_recall_are_distinct(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)])
    path = Path(payload["detections_geojson"])
    data = json.loads(path.read_text())
    data["features"][0]["properties"]["surface"] = "coastal"
    path.write_text(json.dumps(data))
    ev = evaluate(payload)
    assert ev["recall_raw"] == 1
    assert ev["recall_eligible"] == 0
    assert ev["ais_vessels_in_swath"] == 1


def test_stale_ais_cannot_be_evaluation_truth(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)])
    path = Path(payload["ais_telemetry"])
    path.write_text(path.read_text().replace(ACQ, "2026-08-18T08:00:00Z"))
    ev = evaluate(payload)
    assert ev["scored"] is False
    assert "recall_all" not in ev


def test_incomplete_and_mock_processing_are_unscored(tmp_path: Path) -> None:
    payload = build_payload(tmp_path, [(-52.5, 47.5)], [(316000001, -52.5, 47.5)])
    payload["processing"] = {"complete": False, "detector": "xview3"}
    assert evaluate(payload)["scored"] is False
    payload["processing"] = {"complete": True, "detector": "mock"}
    assert evaluate(payload)["scored"] is False
