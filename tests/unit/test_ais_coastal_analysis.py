"""Exercise fixed denominators, competing identities and native pixel validity."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping

from agents.ais_coastal_analysis import compare, run
from agents.artifacts import sha256_file
from agents.coastal_policy import annotations


def water(distance: float) -> dict[str, Any]:
    return annotations("water", distance)


def test_coastal_attrition_keeps_denominator_and_research_returns() -> None:
    points = [Point(-55, 49), Point(-55.02, 49)]
    result = compare(
        points,
        [water(100), water(800)],
        points,
        [water(100), water(800)],
        [0, 100, 300, 1000],
    )
    assert result["matched_raw"] == 2
    assert result["retained_returns"] == 2
    assert [r["observed_ais_denominator"] for r in result["trials"]] == [2] * 4
    assert [r["coastal_ais_withheld"] for r in result["trials"]] == [0, 1, 1, 2]
    assert [r["matched_eligible"] for r in result["trials"]] == [2, 1, 1, 0]
    assert all(
        r["false_alarms"] is None and r["total_missed_vessels"] is None
        for r in result["trials"]
    )


def test_withheld_ais_remains_competing_and_can_match_across_boundary() -> None:
    targets = [Point(-55, 49)]
    positions = [Point(-55, 49), Point(-55.003, 49)]
    result = compare(targets, [water(800)], positions, [water(100), water(800)], [300])
    row = result["trials"][0]
    assert row["strata"]["coastal_water"]["matched_eligible"] == 1
    assert row["strata"]["open_water"]["matched_eligible"] == 0
    assert row["eligible_ambiguous_returns"] == 1


def test_assignment_is_recomputed_after_policy_removes_a_return() -> None:
    result = compare(
        [Point(-55, 49), Point(-55.001, 49)],
        [water(100), water(800)],
        [Point(-55, 49)],
        [water(100)],
        [0, 300],
    )
    assert result["raw_ambiguous_returns"] == 2
    assert result["trials"][1]["matched_eligible"] == 1
    assert result["trials"][1]["eligible_ambiguous_returns"] == 0


def test_land_unknown_and_infrastructure_are_distinct_from_coastal_exclusion() -> None:
    points = [Point(-55 + i * 0.02, 49) for i in range(3)]
    props = [
        annotations("land", -10),
        annotations("unknown", None),
        annotations("water", 800, infrastructure=True),
    ]
    row = compare(points, props, points, props, [300])["trials"][0]
    assert row["coastal_ais_withheld"] == 0
    assert row["ais_policy_eligible"] == 0
    assert row["strata"]["land_mismatch"]["observed_ais"] == 1
    assert row["strata"]["unknown"]["observed_ais"] == 1


def test_no_overlap_and_zero_returns_are_not_zero_false_alarm_measurements() -> None:
    empty = compare([], [], [], [], [0])
    assert empty["status"] == "no_observed_overlap"
    assert empty["ais_subset_proximity_recall_raw"] is None
    result = compare([], [], [Point(-55, 49)], [water(800)], [300])
    assert result["ais_subset_proximity_recall_raw"] == 0
    assert result["trials"][0]["strata"]["open_water"]["unmatched_raw"] == 1


def test_missing_inference_is_not_reported_as_zero_matches_or_misses() -> None:
    result = compare(
        [], [], [Point(-55, 49)], [water(100)], [0, 300], detections_available=False
    )
    assert result["matched_raw"] is None
    assert result["trials"][1]["coastal_ais_withheld"] == 1
    assert result["trials"][1]["coastal_ais_with_raw_match"] is None
    assert result["trials"][1]["strata"]["coastal_water"]["unmatched_raw"] is None


@pytest.mark.parametrize(
    "buffers,radius", [([], 500), ([-1], 500), ([float("nan")], 500), ([0], 0)]
)
def test_invalid_trial_parameters_rejected(buffers: list[float], radius: float) -> None:
    with pytest.raises(ValueError):
        compare([], [], [], [], buffers, radius)


PRODUCT = "66d3167d-240a-459a-8066-75b9d2a458f2"
NAME = "S1D_IW_GRDH_1SDV_20260927T213023_20260927T213048_004769_008EFD_5767.SAFE"
TIME = "2026-09-27T21:30:23Z"


def fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bundle = root / "bundle"
    (bundle / "archive_snapshot").mkdir(parents=True)
    csv = bundle / "archive_snapshot/ais_stream_2026-09-27.csv"
    # Repeated reports count once per vessel; the second vessel is VH zero padding.
    csv.write_text(
        "mmsi,lon,lat,timestamp\n"
        "123456789,-55.05,49.25,2026-09-27T21:30:23Z\n"
        "123456789,-55.05,49.25,2026-09-27T21:30:23Z\n"
        "123456788,-55.02,49.25,2026-09-27T21:30:23Z\n"
        "123456787,-54,49.25,2026-09-27T21:30:23Z\n"
    )
    (bundle / "snapshot_manifest.json").write_text(
        json.dumps({"files": [{"file": csv.name, "sha256": sha256_file(csv)}]})
    )
    footprint = mapping(box(-55.1, 49.2, -55, 49.3))
    (bundle / "catalogue_products.json").write_text(
        json.dumps(
            [
                {
                    "Id": PRODUCT,
                    "Name": NAME,
                    "GeoFootprint": footprint,
                    "ContentDate": {"Start": TIME, "End": "2026-09-27T21:30:48Z"},
                }
            ]
        )
    )
    (root / "configs").mkdir()
    (root / "configs/model.yaml").write_text(
        "landmask:\n  enabled: true\n  coastal_buffer_m: 300\n"
    )
    bands = {}
    for pol in ("VV", "VH"):
        path = root / f"{pol}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=100,
            height=100,
            count=1,
            dtype="uint8",
            crs=4326,
            transform=from_origin(-55.1, 49.3, 0.001, 0.001),
        ) as dst:
            values = np.ones((100, 100), dtype="uint8")
            if pol == "VH":
                values[49:52, 79:82] = 0
            dst.write(values, 1)
        bands[pol] = str(path)
    detections = root / "detections.geojson"
    detections.write_text(
        json.dumps({"features": [{"geometry": mapping(Point(-55.05, 49.25))}]})
    )
    payload = {
        "sar_product_id": PRODUCT,
        "sar_product": NAME,
        "acquisition_time": TIME,
        "mode": "production",
        "sar_bands": bands,
        "detections_geojson": str(detections),
        "processing": {
            "complete": True,
            "detector": "xview3",
            "input_sha256": {k: sha256_file(p) for k, p in bands.items()},
        },
    }
    (root / "payload.json").write_text(json.dumps(payload))
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive_bundle": "bundle",
                "buffers_m": [0, 300],
                "scenes": [
                    {
                        "product_id": PRODUCT,
                        "payload": "payload.json",
                        "setting": "synthetic",
                    }
                ],
            }
        )
    )

    def classify(lons: list[float], *args: Any) -> tuple[Any, Any, Any]:
        return (
            ["water"] * len(lons),
            [100.0] * len(lons),
            {
                "status": "ok",
                "physical_surfaces": ["water"] * len(lons),
                "infrastructure_flags": [False] * len(lons),
            },
        )

    class Mask:
        shapefile = None

        def water_geometry(self, area: Any, buffer_m: float = 0) -> Any:
            return area

    monkeypatch.setattr("agents.ais_coastal_analysis.classify_surfaces", classify)
    monkeypatch.setattr(
        "agents.landmask.LandMask.for_footprint", lambda *a, **k: Mask()
    )
    return manifest


def test_replay_uses_latest_frozen_archive_and_joint_native_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    result = run(tmp_path, manifest, tmp_path / "report.json")
    scene = result["scenes"][0]
    assert scene["alignment"]["aligned_vessels"] == 3
    assert scene["ais_inside_nl_and_catalogue"] == 2
    assert scene["ais_outside_joint_valid_pixels"] == 1
    assert scene["comparison"]["observed_ais_denominator"] == 1
    assert scene["comparison"]["matched_raw"] == 1
    assert result["selected_buffer_m"] is None
    assert result["step_2_complete"] is False
    assert "mmsi" not in json.dumps(result)
    with pytest.raises(ValueError, match="Fresh output"):
        run(tmp_path, manifest, tmp_path / "report.json")


def test_ais_only_replay_reports_exclusions_without_inventing_detector_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    value = json.loads(manifest.read_text())
    value["scenes"][0].pop("payload")
    value["scenes"][0]["ais_only"] = True
    manifest.write_text(json.dumps(value))
    directory = tmp_path / "data/raw" / NAME / "measurement"
    directory.mkdir(parents=True)
    for pol in ("VV", "VH"):
        (tmp_path / f"{pol}.tif").rename(directory / f"test-{pol.lower()}-native.tiff")
    result = run(tmp_path, manifest, tmp_path / "report.json")
    scene = result["scenes"][0]
    assert scene["detections_available"] is False
    assert scene["comparison"]["matched_raw"] is None
    assert scene["comparison"]["trials"][1]["coastal_ais_withheld"] == 1


def test_original_and_cog_alias_cannot_count_as_two_physical_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    value = json.loads(manifest.read_text())
    alias_id = "4f6115d9-c1d8-45bc-b5da-0d37aeaa85f9"
    alias_name = NAME.replace("5767.SAFE", "FFFF_COG.SAFE")
    payload = json.loads((tmp_path / "payload.json").read_text())
    payload.update(sar_product_id=alias_id, sar_product=alias_name)
    (tmp_path / "alias.json").write_text(json.dumps(payload))
    value["scenes"].append(
        {"product_id": alias_id, "payload": "alias.json", "setting": "alias"}
    )
    manifest.write_text(json.dumps(value))
    catalogue_path = tmp_path / "bundle/catalogue_products.json"
    catalogue = json.loads(catalogue_path.read_text())
    catalogue.append({**catalogue[0], "Id": alias_id, "Name": alias_name})
    catalogue_path.write_text(json.dumps(catalogue))
    with pytest.raises(ValueError, match="Duplicate physical slice"):
        run(tmp_path, manifest, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "change", ["footprint", "incomplete", "snapshot", "band", "duplicate", "test"]
)
def test_invalid_inputs_refused_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    manifest = fixture(tmp_path, monkeypatch)
    if change == "footprint":
        path = tmp_path / "bundle/catalogue_products.json"
        value = json.loads(path.read_text())
        value[0]["GeoFootprint"] = {"type": "Polygon", "coordinates": []}
    elif change in {"incomplete", "band"}:
        path = tmp_path / "payload.json"
        value = json.loads(path.read_text())
        if change == "incomplete":
            value["processing"]["complete"] = False
        else:
            value["processing"]["input_sha256"]["VV"] = "incorrect"
    elif change == "snapshot":
        path = tmp_path / "bundle/snapshot_manifest.json"
        value = json.loads(path.read_text())
        value["files"][0]["sha256"] = "incorrect"
    else:
        path = manifest
        value = json.loads(path.read_text())
        if change == "duplicate":
            value["scenes"].append(value["scenes"][0])
        else:
            value["scenes"][0]["split"] = "test"
            # A forbidden test scene must be rejected before any SAR read/hash.
            monkeypatch.setattr(
                "agents.ais_coastal_analysis.valid_sar_points",
                lambda *a: pytest.fail("opened forbidden test raster"),
            )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        run(tmp_path, manifest, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()
