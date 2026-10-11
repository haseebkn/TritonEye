"""Behavioral shoreline/policy separation and benchmark guardrails."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import Point, box, mapping

from agents.coastal_benchmark import compare_buffers
from agents.coastal_policy import annotations, eligible
from agents.correlation.correlation_agent import correlate_targets
from agents.landmask import LandMask, LandMaskUnavailable
from agents.report.report_agent import build_html_report, prepare_report_data

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = "66d3167d-240a-459a-8066-75b9d2a458f2"
ACQ = "2026-09-27T21:30:23Z"


def feature(i: int, lon: float, distance: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(Point(lon, 49.24)),
        "properties": {"target_id": str(i), **annotations("water", distance)},
    }


def sample() -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthetic evaluator fixture, NOT measured regional evidence."""
    detections = {
        "type": "FeatureCollection",
        "sar_product_id": PRODUCT,
        "acquisition_time": ACQ,
        "shoreline_status": "ok",
        "features": [
            feature(0, -55.05, 50),
            feature(1, -55.03, 800),
            feature(2, -55.01, 80),
        ],
    }
    labels = {
        "type": "FeatureCollection",
        "benchmark": {
            "split": "validation",
            "product_id": PRODUCT,
            "acquisition_time": ACQ,
            "annotation_complete": True,
            "annotation_scope": "exhaustive_vessels_on_valid_imagery",
            "source_kind": "independent_sar_review",
            "source": "SYNTHETIC TEST ONLY",
            "reviewer": "test fixture",
            "licence": "test fixture",
            "valid_imagery_roi": mapping(box(-55.1, 49.2, -55.0, 49.3)),
        },
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(Point(lon, 49.24)),
                "properties": {
                    "label_id": str(i),
                    "class": "vessel",
                    **annotations("water", distance),
                },
            }
            for i, (lon, distance) in enumerate(
                [(-55.05, 50), (-55.03, 800), (-55.08, 900)]
            )
        ],
    }
    return detections, labels


def test_water_remains_water_when_buffer_changes() -> None:
    narrow = annotations("water", 133, 100)
    wide = annotations("water", 133, 300)
    assert narrow["physical_surface"] == wide["physical_surface"] == "water"
    assert narrow["surface"] == "water" and eligible(narrow)
    assert wide["surface"] == "coastal" and not eligible(wide)
    assert wide["research_retained"] and not wide["operational_alert"]


def test_infrastructure_does_not_change_physical_surface() -> None:
    result = annotations("water", 900, infrastructure=True)
    assert result["physical_surface"] == "water"
    assert result["surface"] == "infrastructure_proximity"
    assert not eligible(result)


@pytest.mark.parametrize("buffer", [-1, math.nan, math.inf])
def test_bad_buffers_fail(buffer: float) -> None:
    with pytest.raises(ValueError, match="buffer"):
        annotations("water", 900, buffer)


def test_unknown_distance_is_not_open_ocean() -> None:
    assert not eligible(annotations("water", None))
    assert eligible(annotations("water", math.inf))
    assert not eligible(annotations("unknown", math.inf))


def test_contradictory_policy_cannot_promote_land() -> None:
    props = annotations("land", -40)
    props.update({"surface": "water", "alert_eligible": True})
    assert not eligible(props)


@pytest.mark.parametrize(
    "site,bounds,count",
    [
        ("bonavista", (-53.134, 48.645, -53.103, 48.665), 5),
        ("lewisporte", (-55.070, 49.233, -55.042, 49.252), 5),
    ],
)
def test_actual_regional_controls_against_attributed_osm_snapshot(
    site: str,
    bounds: tuple[float, ...],
    count: int,
) -> None:
    mask = LandMask.for_footprint(
        bounds, shapefile=str(ROOT / f"tests/fixtures/{site}_osm_land.geojson")
    )
    assert mask.validation["status"] == "passed"
    assert mask.validation["checked"] == count
    checked = [r for r in mask.validation["collections"] if r["checked"]]
    assert len(checked) == 1
    assert checked[0]["reference"]["image_sha256"]
    assert checked[0]["reference"]["scope"]


def test_registry_cannot_claim_labrador_validation() -> None:
    from agents.landmask import local_aeqd_crs

    mask = LandMask([], local_aeqd_crs((-61, 55, -59, 57)), "test", "test")
    mask.validate_controls(
        (-61, 55, -59, 57), str(ROOT / "configs/coastline/regional_controls.json")
    )
    assert mask.validation["status"] == "outside_scope"
    assert mask.validation["checked"] == 0
    assert all(r["checked"] == 0 for r in mask.validation["collections"])


def test_closed_bonavista_fails_in_registry(tmp_path: Path) -> None:
    import geopandas as gpd

    bounds = (-53.134, 48.645, -53.103, 48.665)
    path = tmp_path / "closed.geojson"
    gpd.GeoDataFrame(geometry=[box(*bounds)], crs=4326).to_file(path)
    with pytest.raises(LandMaskUnavailable, match="bonavista_basin"):
        LandMask.for_footprint(bounds, shapefile=str(path))


def test_buffer_trials_measure_false_alarms_and_policy_misses() -> None:
    detections, labels = sample()
    rows = compare_buffers(detections, labels, PRODUCT, buffers_m=[0, 300, 1000])[
        "trials"
    ]
    assert [r["false_alarms"] for r in rows] == [1, 0, 0]
    assert [r["missed_vessels"] for r in rows] == [1, 2, 3]
    assert [r["policy_added_misses"] for r in rows] == [0, 1, 2]
    assert [r["vessels"] for r in rows] == [3, 3, 3]
    assert [r["retained_returns"] for r in rows] == [3, 3, 3]
    assert rows[1]["coastal_research_true_positives"] == 1
    assert rows[1]["coastal_research_false_returns"] == 1
    assert rows[1]["by_regime"]["harbour_coastal"]["missed_eligible"] == 1
    assert rows[1]["by_regime"]["open_water"]["vessels"] == 2
    assert rows[0]["false_alarms_per_km2"] is None


def test_unlabelled_returns_never_become_false_alarm_measurements() -> None:
    detections, _ = sample()
    result = compare_buffers(detections, None, PRODUCT)
    assert all(r["measured"] is False for r in result["trials"])
    assert all(
        r["false_alarms"] is None and r["missed_vessels"] is None
        for r in result["trials"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "holdout"),
        ("annotation_complete", False),
        ("source_kind", "ais_only"),
        ("reviewer", ""),
        ("product_id", "9d86c0e9-f457-4948-bb09-3f7d00b22d04"),
        ("acquisition_time", "2026-08-17T21:22:09Z"),
    ],
)
def test_invalid_label_provenance_is_rejected(field: str, value: Any) -> None:
    detections, labels = sample()
    labels["benchmark"][field] = value
    with pytest.raises(ValueError):
        compare_buffers(detections, labels, PRODUCT)


def test_missing_shoreline_does_not_measure_perfect_suppression() -> None:
    detections, labels = sample()
    detections["features"][0]["properties"].update(annotations("unknown", None))
    assert not compare_buffers(detections, labels, PRODUCT)["trials"][0]["measured"]


def test_one_return_cannot_match_two_labelled_vessels() -> None:
    detections, labels = sample()
    labels["features"].append(copy.deepcopy(labels["features"][0]))
    labels["features"][-1]["properties"]["label_id"] = "second_nearby"
    row = compare_buffers(detections, labels, PRODUCT, buffers_m=[0])["trials"][0]
    assert row["true_positives"] == 2 and row["missed_vessels"] == 2
    assert row["ambiguous_targets"] == 1


def test_only_reviewed_roi_is_scored() -> None:
    detections, labels = sample()
    detections["features"].append(feature(4, -54.8, 1000))
    row = compare_buffers(detections, labels, PRODUCT, buffers_m=[0])["trials"][0]
    assert row["retained_returns"] == 3
    assert row["false_alarms"] == 1


def test_coastal_return_survives_correlation_and_report(tmp_path: Path) -> None:
    detections, _ = sample()
    path = tmp_path / "detections.geojson"
    path.write_text(json.dumps(detections))
    targets = correlate_targets(str(path), str(tmp_path / "absent.csv"), ACQ)
    assert len(targets) == 3
    assert targets.coastal_review_required.sum() == 2
    assert targets.association_eligible.sum() == 1
    assert not targets.operational_alert.any()
    data = prepare_report_data({}, detections)
    assert len(data["targets"]["features"]) == 3
    assert data["review_count"] == 0
    rendered = build_html_report({}, detections)
    assert "Harbour and coastal research (2)" in rendered


def test_false_new_eligibility_flag_fails_closed() -> None:
    detections, _ = sample()
    detections["features"][1]["properties"]["alert_eligible"] = False
    data = prepare_report_data({}, detections)
    assert data["states"]["excluded_unknown_surface"] == 1


@pytest.mark.parametrize("bad", [None, [], {"type": "ShorelineControlRegistry"}])
def test_malformed_registry_collection_fails_explicitly(
    tmp_path: Path,
    bad: Any,
) -> None:
    child = tmp_path / "child.geojson"
    child.write_text(json.dumps(bad))
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "type": "ShorelineControlRegistry",
                "collections": [{"location": "bad", "path": "child.geojson"}],
            }
        )
    )
    from agents.landmask import local_aeqd_crs

    mask = LandMask([], local_aeqd_crs((-53, 47, -52, 48)), "test", "test")
    with pytest.raises(LandMaskUnavailable):
        mask.validate_controls((-53, 47, -52, 48), str(registry))


def test_missing_shore_distance_keeps_measurements_unavailable() -> None:
    detections, labels = sample()
    detections["features"][0]["properties"]["distance_to_shore_m"] = None
    rows = compare_buffers(detections, labels, PRODUCT)["trials"]
    assert all(not r["measured"] and r["missed_vessels"] is None for r in rows)


def test_detection_uuid_must_match_even_for_unlabelled_trials() -> None:
    detections, _ = sample()
    detections["sar_product_id"] = "9d86c0e9-f457-4948-bb09-3f7d00b22d04"
    with pytest.raises(ValueError, match="Detection product ID"):
        compare_buffers(detections, None, PRODUCT)


def test_no_shoreline_cannot_measure_zero_false_alarms() -> None:
    detections, labels = sample()
    detections["shoreline_status"] = "unavailable"
    detections["features"] = []
    rows = compare_buffers(detections, labels, PRODUCT)["trials"]
    assert all(not r["measured"] and r["false_alarms"] is None for r in rows)
