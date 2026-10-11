"""Native-pixel coverage behavior; synthetic checks do not prove NL accuracy."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping

from agents.artifacts import sha256_file
from agents.landmask import LandMask, local_aeqd_crs
from agents.shoreline_coverage import (
    area_coverage,
    audit_coverage,
    valid_chip_points,
    verify_reference,
)
from scripts.inspect_regional_reference import export_bounds

BOUNDS = (-55.1, 49.2, -55.0, 49.3)
WATER = Point(-55.025, 49.275)
LAND = Point(-55.075, 49.275)


def chip(path: Path, *, bad: float | None = None) -> Path:
    data = np.zeros((2, 10, 10), dtype="float32")
    if bad is not None:
        data[1, 2, 7] = bad
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=2,
        width=10,
        height=10,
        dtype="float32",
        crs=4326,
        transform=from_origin(-55.1, 49.3, 0.01, 0.01),
        nodata=-999,
    ) as dst:
        dst.write(data)
    return path


def controls() -> list[dict[str, Any]]:
    return [
        {
            "location": "synthetic site",
            "document": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(point),
                        "properties": {"id": name, "expected": expected},
                    }
                    for point, name, expected in (
                        (WATER, "water", "water"),
                        (LAND, "land", "land"),
                    )
                ],
            },
        }
    ]


def mask() -> LandMask:
    crs = local_aeqd_crs(BOUNDS)
    polygons = gpd.GeoSeries([box(-55.1, 49.2, -55.05, 49.3)], crs=4326).to_crs(crs)
    return LandMask(polygons, crs, "synthetic", "test only")


def test_zero_db_is_valid_and_outside_raster_is_not(tmp_path: Path) -> None:
    assert valid_chip_points(
        chip(tmp_path / "chip.tif"), [WATER, LAND, Point(-56, 49)]
    ) == [True, True, False]


@pytest.mark.parametrize("bad", [-999, float("nan"), float("inf")])
def test_second_polarization_invalidates_control(tmp_path: Path, bad: float) -> None:
    assert valid_chip_points(chip(tmp_path / "chip.tif", bad=bad), [WATER, LAND]) == [
        False,
        True,
    ]


def test_passing_checks_do_not_claim_vessel_accuracy(tmp_path: Path) -> None:
    result = area_coverage(
        {"geometry": mapping(box(*BOUNDS))},
        chip(tmp_path / "chip.tif"),
        mask(),
        controls(),
    )
    assert result["status"] == "sparse_controls_passed"
    assert result["controls_on_joint_valid_pixels"] == 2
    assert all(c["passed"] for c in result["checks"])


def test_roi_membership_not_bounding_box_membership(tmp_path: Path) -> None:
    # The triangular ROI shares the raster bounding box but excludes LAND.
    roi = {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-55.1, 49.2], [-55, 49.2], [-55, 49.3], [-55.1, 49.2]]],
        }
    }
    result = area_coverage(roi, chip(tmp_path / "chip.tif"), mask(), controls())
    assert result["controls_in_roi"] == 1
    assert result["status"] == "insufficient_controls"


def test_joint_nodata_cannot_establish_land_and_water_coverage(tmp_path: Path) -> None:
    result = area_coverage(
        {"geometry": mapping(box(*BOUNDS))},
        chip(tmp_path / "chip.tif", bad=-999),
        mask(),
        controls(),
    )
    assert result["outside_joint_valid_pixels"] == 1
    assert result["status"] == "insufficient_controls"


def test_wrong_mask_is_a_failure(tmp_path: Path) -> None:
    wrong = LandMask([], local_aeqd_crs(BOUNDS), "synthetic", "test")
    result = area_coverage(
        {"geometry": mapping(box(*BOUNDS))},
        chip(tmp_path / "chip.tif"),
        wrong,
        controls(),
    )
    assert result["status"] == "failed"
    assert result["checks"][1]["passed"] is False


def audit_inputs(root: Path) -> tuple[Path, Path]:
    raster = chip(root / "chip.tif")
    collection = controls()[0]
    (root / "controls.geojson").write_text(json.dumps(collection["document"]))
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "type": "ShorelineControlRegistry",
                "collections": [
                    {"location": collection["location"], "path": "controls.geojson"}
                ],
            }
        )
    )
    scene = {
        "product_id": "66d3167d-240a-459a-8066-75b9d2a458f2",
        "name": (
            "S1D_IW_GRDH_1SDV_20260927T213023_20260927T213048_"
            "004769_008EFD_5767.SAFE"
        ),
        "acquisition_time": "2026-09-27T21:30:23Z",
        "split": "validation",
        "region": "Newfoundland",
        "polarizations": ["vv", "vh"],
        "rois": [
            {
                "id": "synthetic",
                "regime": "coastal",
                "geometry": mapping(box(*BOUNDS)),
                "chip": {"path": "chip.tif", "sha256": sha256_file(raster)},
            }
        ],
    }
    held = {
        "split": "test",
        "rois": [{"id": "locked", "chip": {"path": "../never_read.tif"}}],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"scenes": [scene, held]}))
    return manifest, registry


def test_audit_skips_locked_paths_before_resolution_and_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, registry = audit_inputs(tmp_path)
    monkeypatch.setattr(LandMask, "for_footprint", lambda *args, **kwargs: mask())
    result = audit_coverage(tmp_path, manifest, registry)
    assert result["locked_test_areas_not_opened"] == ["locked"]
    assert result["coastal_areas_with_passing_land_and_water_controls"] == 1
    assert result["step_2_complete"] is False
    assert result["buffer_selected"] is None
    assert result["province_wide_validation"] is False


def test_tampered_chip_is_unavailable_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, registry = audit_inputs(tmp_path)
    chip(tmp_path / "chip.tif", bad=-999)

    def do_not_load_mask(*args: Any, **kwargs: Any) -> LandMask:
        raise AssertionError("Tampered pixels must be rejected before shoreline use")

    monkeypatch.setattr(LandMask, "for_footprint", do_not_load_mask)
    result = audit_coverage(tmp_path, manifest, registry)
    assert result["areas"][0]["status"] == "unavailable"
    assert result["coastal_areas_with_passing_land_and_water_controls"] == 0


def test_registry_escape_is_rejected(tmp_path: Path) -> None:
    manifest, registry = audit_inputs(tmp_path)
    document = json.loads(registry.read_text())
    document["collections"][0]["path"] = "../elsewhere.geojson"
    registry.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="inside its root"):
        audit_coverage(tmp_path, manifest, registry)


def reference_input(root: Path) -> dict[str, Any]:
    image = root / "imagery.png"
    image.write_bytes(b"synthetic reference artifact")
    document: dict[str, Any] = controls()[0]["document"]
    document["reference"] = {
        "image_sha256": sha256_file(image),
        "local_provenance": "provenance.json",
        "image_extent_wgs84": list(BOUNDS),
        "image_size": [100, 100],
    }
    for feature, x in zip(document["features"], [75, 25]):
        feature["properties"]["image_pixel_xy"] = [x, 25]
    return document


def test_image_pixel_contract_uses_recorded_extent(tmp_path: Path) -> None:
    document = reference_input(tmp_path)
    assert (
        verify_reference(tmp_path, document)["status"]
        == "image_hash_and_recorded_pixels_verified"
    )
    document["reference"]["image_extent_wgs84"][0] -= 0.01
    with pytest.raises(ValueError, match="geography differs"):
        verify_reference(tmp_path, document)


def test_changed_reference_image_is_not_verified(tmp_path: Path) -> None:
    document = reference_input(tmp_path)
    (tmp_path / "imagery.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="imagery hash mismatch"):
        verify_reference(tmp_path, document)


def test_absent_historical_image_hash_is_disclosed(tmp_path: Path) -> None:
    assert (
        verify_reference(tmp_path, controls()[0]["document"])["status"]
        == "image_hash_not_recorded"
    )


def test_comparison_uses_expanded_actual_export_extent() -> None:
    metadata = {
        "extent": {
            "xmin": -52.9108,
            "ymin": 47.007,
            "xmax": -52.8612,
            "ymax": 47.038,
            "spatialReference": {"wkid": 4326},
        }
    }
    # This visually inspected offshore control is outside the requested bbox,
    # but inside the actual imagery returned by the service.
    assert box(*export_bounds(metadata)).covers(Point(-52.86647, 47.01196))


@pytest.mark.parametrize(
    "crs,bounds",
    [(3857, BOUNDS), (4326, (float("nan"), 49, -55, 50)), (4326, (-55, 50, -56, 49))],
)
def test_bad_export_cannot_drive_geographic_comparison(
    crs: int, bounds: tuple[float, ...]
) -> None:
    metadata: dict[str, Any] = {
        "extent": dict(zip(("xmin", "ymin", "xmax", "ymax"), bounds))
    }
    metadata["extent"]["spatialReference"] = {"wkid": crs}
    with pytest.raises(ValueError):
        export_bounds(metadata)


def test_non_nl_area_is_rejected_before_raster_use(tmp_path: Path) -> None:
    manifest, registry = audit_inputs(tmp_path)
    document = json.loads(manifest.read_text())
    document["scenes"][0]["rois"][0]["geometry"] = mapping(box(-71.1, 42.2, -71, 42.3))
    document["scenes"][0]["rois"][0]["chip"]["path"] = "must_not_open.tif"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="NL review area"):
        audit_coverage(tmp_path, manifest, registry)


def test_duplicate_roi_cannot_inflate_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, registry = audit_inputs(tmp_path)
    document = json.loads(manifest.read_text())
    document["scenes"][0]["rois"].append(document["scenes"][0]["rois"][0])
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(LandMask, "for_footprint", lambda *args, **kwargs: mask())
    with pytest.raises(ValueError, match="Duplicate review area"):
        audit_coverage(tmp_path, manifest, registry)
