"""Synthetic evaluator contracts; these are not measured NL performance."""

import copy
import math
from typing import Any

import geopandas as gpd
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Point, box, mapping
from shapely.ops import transform

from agents.baseline_association import association_summary
from agents.baseline_context import area_km2, require_context_split, selected_origins
from agents.baseline_metrics import score_roi
from agents.baseline_report import (
    DEFAULT_TARGETS,
    choose_threshold,
    coverage,
    make_report,
)
from agents.baseline_statistics import (
    binomial_interval,
    block_interval,
    dependence_blocks,
    poisson_interval,
)
from agents.coastal_policy import annotations
from agents.landmask import LandMask, local_aeqd_crs


def fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    product = "66d3167d-240a-459a-8066-75b9d2a458f2"
    roi = mapping(box(-55.1, 49.2, -55.0, 49.3))
    detections = {
        "type": "FeatureCollection",
        "sar_product_id": product,
        "acquisition_time": "2026-09-27T21:30:23Z",
        "study_roi": roi,
        "shoreline_status": "ok",
        "features": [],
    }
    for i, (lon, distance, score) in enumerate(
        [(-55.05, 50, 0.8), (-55.03, 800, 0.8), (-55.01, 800, 0.15001)]
    ):
        detections["features"].append(
            {
                "type": "Feature",
                "geometry": mapping(Point(lon, 49.24)),
                "properties": {
                    "detection_id": str(i),
                    "confidence": score,
                    **annotations("water", distance),
                },
            }
        )
    labels = {
        "type": "FeatureCollection",
        "benchmark": {
            "split": "validation",
            "product_id": product,
            "acquisition_time": detections["acquisition_time"],
            "annotation_complete": True,
            "annotation_scope": "exhaustive_vessels_on_valid_imagery",
            "source_kind": "independent_sar_review",
            "source": "SYNTHETIC ONLY",
            "reviewer": "synthetic fixture",
            "licence": "test",
            "valid_imagery_roi": roi,
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


def test_raw_policy_density_and_misses_keep_fixed_truth() -> None:
    detections, labels = fixture()
    result = score_roi(
        detections,
        labels,
        threshold=0.15,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=True,
    )
    assert result["raw"]["tp"] == 2
    assert result["raw"]["fp"] == 1
    assert result["raw"]["recall"] == 2 / 3
    assert result["post_policy"]["tp"] == 1
    assert result["post_policy"]["fn"] == 2
    assert result["post_policy"]["false_alarms_per_km2"] == 0.1
    assert result["policy_added_misses"] == result["detector_missed_vessels"] == 1
    assert result["by_shore_regime"]["harbour_coastal"]["vessels"] == 1
    narrower = score_roi(
        detections,
        labels,
        threshold=0.15,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=True,
        buffer_m=0,
    )
    assert narrower["post_policy"]["tp"] == 2
    assert narrower["water_area_km2"] == result["water_area_km2"]
    assert not result["operational_alerts_enabled"]


@pytest.mark.parametrize("complete,has_labels", [(False, True), (True, False)])
def test_no_labels_or_failed_processing_never_becomes_accuracy(
    complete: bool, has_labels: bool
) -> None:
    detections, labels = fixture()
    result = score_roi(
        detections,
        labels if has_labels else None,
        threshold=0.15,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=complete,
    )
    assert not result["measured"]
    assert result["raw"] is result["post_policy"] is None


def test_strict_full_precision_floor_contract() -> None:
    detections, labels = fixture()
    above = score_roi(
        detections,
        labels,
        threshold=0.15,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=True,
    )
    exact = score_roi(
        detections,
        labels,
        threshold=0.15001,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=True,
    )
    assert above["returns"] == 3 and exact["returns"] == 2
    detections["features"][0]["properties"]["confidence"] = True
    with pytest.raises(ValueError, match="score-floor"):
        score_roi(
            detections,
            None,
            threshold=0.15,
            score_floor=0.05,
            water_area_km2=10,
            processing_complete=True,
        )


def cases(tp: int = 40) -> list[dict[str, Any]]:
    metric = {
        "tp": tp,
        "fp": 0,
        "fn": 40 - tp,
        "vessels": 40,
        "false_alarms_on_water": 0,
    }
    return [
        {
            "split": "validation",
            "region": "NL",
            "regime": "offshore",
            "polarization": "vv/vh",
            "season": "summer",
            "acquisition_group": str(i),
            "geographic_group": str(i),
            "water_area_km2": 100.0,
            "processing_complete": True,
            "metric_ready": True,
            "scores": [
                {
                    "threshold": t,
                    "measured": True,
                    "raw": metric,
                    "post_policy": metric,
                    "returns": tp,
                    "coastal_exclusions": 0,
                    "abstentions": 0,
                    "policy_added_misses": 0,
                    "detector_missed_vessels": 40 - tp,
                }
                for t in (0.15, 0.99)
            ],
        }
        for i in range(5)
    ]


def test_operating_point_requires_recall_not_only_zero_false_alarms() -> None:
    zero = choose_threshold(cases(0), [0.15, 0.99], DEFAULT_TARGETS)
    assert zero["selected_threshold"] is None
    good = choose_threshold(cases(), [0.15, 0.99], DEFAULT_TARGETS)
    assert good["selected_threshold"] == 0.15
    assert good["status"] == "research_candidate_requires_review"
    assert not good["configuration_changed"]


def test_unsupported_small_stratum_cannot_hide_in_total() -> None:
    rows = cases()
    unsupported = copy.deepcopy(rows[0])
    unsupported.update(
        water_area_km2=1.0,
        polarization="hh/hv",
        region="Labrador",
        processing_complete=False,
        metric_ready=False,
    )
    for score in unsupported["scores"]:
        score["measured"] = False
    rows.append(unsupported)
    assert coverage(rows)["processing"] > 0.99
    assert (
        choose_threshold(rows, [0.15, 0.99], DEFAULT_TARGETS)["selected_threshold"]
        is None
    )
    assert (
        make_report(rows, [0.15, 0.99])["stratified_validation"]["region"]["Labrador"][
            "raw"
        ]
        is None
    )


@pytest.mark.parametrize("split", ["train", "test"])
def test_no_train_or_test_threshold_tuning(split: str) -> None:
    rows = cases()
    rows[0]["split"] = split
    with pytest.raises(ValueError, match="validation"):
        choose_threshold(rows, [0.15, 0.99], DEFAULT_TARGETS)


def test_uncertainty_not_zero_for_no_events_and_dependence_transitive() -> None:
    assert binomial_interval(0, 0) is None
    assert poisson_interval(0, 0) is None
    interval = poisson_interval(0, 10)
    assert interval is not None and interval[1] > 0
    rows = [
        {"acquisition_group": "a", "geographic_group": "x"},
        {"acquisition_group": "a", "geographic_group": "y"},
        {"acquisition_group": "b", "geographic_group": "y"},
    ]
    assert len(dependence_blocks(rows)) == 1
    assert block_interval(rows, "num", "den")["interval"] is None
    with pytest.raises(ValueError):
        binomial_interval(True, 2)
    targets = {**DEFAULT_TARGETS, "max_false_alarms_per_km2": math.inf}
    with pytest.raises(ValueError, match="Finite"):
        choose_threshold(cases(), [0.15, 0.99], targets)


def test_context_split_guard_checks_pixels_not_only_roi_centers() -> None:
    manifest = {
        "separation_m": 1000,
        "scenes": [
            {
                "split": "test",
                "rois": [
                    {
                        "id": "held",
                        "geometry": mapping(box(-60.22, 55.45, -60.21, 55.46)),
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="another split"):
        require_context_split(box(-60.23, 55.44, -60.2, 55.47), "validation", manifest)
    require_context_split(box(-59.2, 55.0, -59.1, 55.1), "validation", manifest)
    assert selected_origins(4096, 4096, [2000, 2000, 512, 512])


def test_physical_water_denominator_does_not_follow_coastal_buffer() -> None:
    roi = box(-53.1, 48.0, -53.0, 48.1)
    crs = local_aeqd_crs(roi.bounds)
    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    land = transform(forward.transform, box(-53.1, 48.0, -53.05, 48.1))
    mask = LandMask([land], crs, "synthetic", "test")
    physical = area_km2(mask.water_geometry(roi))
    open_water = area_km2(mask.water_geometry(roi, 300))
    assert 0 < open_water < physical < area_km2(roi)
    assert area_km2(mask.water_geometry(roi)) == pytest.approx(physical)


def test_ais_proximity_is_not_identity_correctness() -> None:
    detections, _ = fixture()
    positions = gpd.GeoDataFrame(
        {"uncertainty_m": [100.0]}, geometry=[Point(-55.03, 49.24)], crs="EPSG:4326"
    )
    result = association_summary(
        detections, positions, threshold=0.15, processing_complete=True
    )
    assert result["ais_subset_proximity_recall_raw"] == 1
    assert result["association_correctness"] is None
    empty = association_summary(
        detections, positions.iloc[:0], threshold=0.15, processing_complete=True
    )
    assert empty["ais_subset_proximity_recall_raw"] is None
    assert empty["ambiguous_assignments"] is None


def test_raw_water_false_alarm_keeps_the_raw_assignment() -> None:
    detections, labels = fixture()
    labels["features"] = labels["features"][1:2]
    detections["features"] = detections["features"][:2]
    geod = Geod(ellps="WGS84")
    for feature, distance, surface in zip(
        detections["features"], [1, 80], ["land", "water"]
    ):
        lon, lat, _ = geod.fwd(-55.03, 49.24, 90, distance)
        feature["geometry"] = mapping(Point(lon, lat))
        feature["properties"].update(annotations(surface, 800))
    result = score_roi(
        detections,
        labels,
        threshold=0.15,
        score_floor=0.05,
        water_area_km2=10,
        processing_complete=True,
    )
    assert result["raw"]["tp"] == result["raw"]["fp"] == 1
    assert result["raw"]["false_alarms_on_water"] == 1
    assert result["raw"]["false_alarms_per_km2"] == 0.1
    assert result["post_policy"]["tp"] == 1
    assert result["post_policy"]["false_alarms_on_water"] == 0


@pytest.mark.parametrize(
    "distance,uncertainty,assigned", [(300, 0, 1), (550, 100, 1), (650, 100, 0)]
)
def test_ais_assignment_uses_production_gate_separately_from_proximity(
    distance: float, uncertainty: float, assigned: int
) -> None:
    detections, _ = fixture()
    detections["features"] = detections["features"][1:2]
    lon, lat, _ = Geod(ellps="WGS84").fwd(-55.03, 49.24, 90, distance)
    positions = gpd.GeoDataFrame(
        {"uncertainty_m": [uncertainty]},
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    )
    result = association_summary(
        detections, positions, threshold=0.15, processing_complete=True
    )
    assert result["ais_subset_proximity_recall_raw"] == 0
    assert result["ais_subset_proximity_recall_post_policy"] == 0
    assert result["geometric_assignments"] == assigned
    assert result["ais_proximity_radius_m"] == 100
    assert result["association_base_radius_m"] == 500
    assert result["association_correctness"] is None
    detections["features"].append(copy.deepcopy(detections["features"][0]))
    competing = association_summary(
        detections, positions, threshold=0.15, processing_complete=True
    )
    assert competing["geometric_assignments"] == assigned
    assert competing["ambiguous_assignments"] == (2 if assigned else 0)
