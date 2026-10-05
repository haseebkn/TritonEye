"""Execute shoreline checks against a real local snapshot and faulty masks."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

from agents.inference.inference_agent import classify_surfaces
from agents.landmask import (
    LandMask,
    LandMaskUnavailable,
    local_aeqd_crs,
)
from agents.scene_watch import water_eligible

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHORELINE_CONTROLS = str(ROOT / "configs/coastline/st_johns_checks.geojson")
FIXTURE = ROOT / "tests/fixtures/st_johns_osm_land.geojson"
BOUNDS = (-52.73, 47.54, -52.66, 47.59)
WATER = (-52.68115769052795, 47.56640409090994)
LAND = (-52.69695255730004, 47.570044981458366)


def real_mask() -> LandMask:
    return LandMask.for_footprint(BOUNDS, shapefile=str(FIXTURE))


def test_harbour_and_narrows_are_physically_water() -> None:
    mask = real_mask()
    assert mask.validation["status"] == "passed"
    assert mask.validation["checked"] == 8
    assert mask.validation["failed"] == 0
    surfaces, distances = mask.classify([WATER[0], LAND[0]], [WATER[1], LAND[1]])
    assert surfaces == ["coastal", "land"]
    assert distances[0] == pytest.approx(133, abs=2)
    assert distances[1] < 0
    # Physical water verification cannot silently override the eligibility band.
    assert mask.classify([WATER[0]], [WATER[1]], coastal_buffer_m=0)[0] == ["water"]


def test_a_closed_harbour_mask_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "closed.geojson"
    gpd.GeoDataFrame(geometry=[box(*BOUNDS)], crs="EPSG:4326").to_file(path)
    with pytest.raises(LandMaskUnavailable, match="narrows_channel"):
        LandMask.for_footprint(BOUNDS, shapefile=str(path))


def test_a_mask_that_turns_quays_into_water_is_rejected() -> None:
    mask = LandMask([], local_aeqd_crs(BOUNDS), "faulty", "test")
    with pytest.raises(LandMaskUnavailable, match="north_quay"):
        mask.validate_controls(BOUNDS, DEFAULT_SHORELINE_CONTROLS)
    assert mask.validation["status"] == "failed"


def test_controls_outside_scene_do_not_claim_validation() -> None:
    mask = LandMask([], local_aeqd_crs(BOUNDS), "test", "test")
    mask.validate_controls((-52, 48, -51, 49), DEFAULT_SHORELINE_CONTROLS)
    assert mask.validation["status"] == "outside_scope"
    assert mask.validation["checked"] == 0


@pytest.mark.parametrize("bad_coordinate", [[float("nan"), 47], [-52, 100], ["x", 47]])
def test_bad_controls_cannot_disable_validation(
    tmp_path: Path, bad_coordinate: list[Any]
) -> None:
    document = json.loads(Path(DEFAULT_SHORELINE_CONTROLS).read_text())
    document["features"][0]["geometry"]["coordinates"] = bad_coordinate
    path = tmp_path / "invalid.geojson"
    path.write_text(json.dumps(document))
    with pytest.raises(LandMaskUnavailable, match="Invalid shoreline controls"):
        real_mask().validate_controls(BOUNDS, str(path))


def test_missing_control_file_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(LandMaskUnavailable, match="Invalid shoreline controls"):
        LandMask.for_footprint(
            BOUNDS, shapefile=str(FIXTURE), controls_path=str(tmp_path / "missing")
        )


def test_production_surface_metadata_records_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_mask_saved = real_mask()
    monkeypatch.setattr(
        LandMask, "for_footprint", lambda *args, **kwargs: real_mask_saved
    )
    surfaces, _, metadata = classify_surfaces(
        [WATER[0]], [WATER[1]], BOUNDS, str(tmp_path), {"landmask": {"enabled": True}}
    )
    assert surfaces == ["coastal"]
    assert metadata["shoreline_validation"]["status"] == "passed"
    assert metadata["shoreline_validation"]["checked"] == 8


def test_failed_controls_leave_production_targets_unassessable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*args: Any, **kwargs: Any) -> LandMask:
        raise LandMaskUnavailable("Shoreline imagery controls failed: narrows_channel")

    monkeypatch.setattr(LandMask, "for_footprint", broken)
    surfaces, _, metadata = classify_surfaces(
        [WATER[0]], [WATER[1]], BOUNDS, str(tmp_path), {"landmask": {"enabled": True}}
    )
    assert surfaces is None
    assert metadata["status"] == "unavailable"
    assert water_eligible([WATER], str(tmp_path)) == 0


def test_non_wgs84_controls_are_rejected(tmp_path: Path) -> None:
    document = json.loads(Path(DEFAULT_SHORELINE_CONTROLS).read_text())
    document["crs"] = {"type": "name", "properties": {"name": "EPSG:32181"}}
    path = tmp_path / "wrong_crs.geojson"
    path.write_text(json.dumps(document))
    with pytest.raises(LandMaskUnavailable, match="WGS84"):
        real_mask().validate_controls(BOUNDS, str(path))


@pytest.mark.parametrize(
    "document", [None, [], {"type": "FeatureCollection", "features": []}]
)
def test_empty_or_malformed_reference_is_not_a_pass(
    tmp_path: Path, document: Any
) -> None:
    path = tmp_path / "bad_document.geojson"
    path.write_text(json.dumps(document))
    with pytest.raises(LandMaskUnavailable, match="FeatureCollection"):
        real_mask().validate_controls(BOUNDS, str(path))
