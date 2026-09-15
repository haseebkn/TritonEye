"""Report semantics, region containment and injection regression coverage."""

import json
import re
from typing import Any

from agents.report.report_agent import build_html_report, prepare_report_data


def feature(
    target: str,
    *,
    lon: float = -52.2,
    lat: float = 48.0,
    surface: str | None = "water",
    state: str | None = None,
    **props: Any,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "target_id": target,
            "surface": surface,
            "correlation_status": state,
            "confidence": 0.95,
            **props,
        },
    }


def collection(*features: dict[str, Any]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def test_no_ais_never_becomes_an_alert_or_false_association() -> None:
    detections = collection(
        feature("a", state="uncorrelated_candidate"),
        feature("b", state="ais_associated"),
    )
    payload = {"spatial_bounds": {"ais_coverage": "none"}}
    data = prepare_report_data(payload, detections)
    assert data["states"] == {"unassessable": 2}
    assert data["review_count"] == 0
    report = build_html_report(payload, detections)
    assert "OFFLINE" not in report
    assert "Dark Vessel Alerts" not in report
    assert "Iceberg discrimination is unresolved" in report
    assert "not calibrated probabilities" in report


def test_mock_report_is_prominently_marked_synthetic() -> None:
    rendered = build_html_report({"mode": "mock"}, collection())
    assert "<strong>SYNTHETIC DEMONSTRATION</strong>" in rendered
    assert "Not real surveillance or accuracy evidence" in rendered


def test_land_exclusions_are_not_counted_as_ais_associations() -> None:
    data = prepare_report_data(
        {"spatial_bounds": {"ais_coverage": "aisstream"}},
        collection(
            feature("land", surface="land", state="ais_associated"),
            feature("coast", surface="coastal", state="uncorrelated_candidate"),
            feature("platform", surface="infrastructure"),
            feature("unmasked", surface=None),
            feature("matched", state="ais_associated"),
            feature("review", state="uncorrelated_candidate"),
        ),
    )
    assert data["states"] == {
        "excluded_land": 1,
        "excluded_coastal": 1,
        "excluded_infrastructure": 1,
        "excluded_unknown_surface": 1,
        "ais_associated": 1,
        "uncorrelated_candidate": 1,
    }
    assert data["review_count"] == 1


def test_legacy_dark_artifacts_do_not_establish_association_by_subtraction() -> None:
    detections = collection(feature("a"), feature("b"))
    data = prepare_report_data(
        {"spatial_bounds": {"ais_coverage": "aisstream"}},
        detections,
        collection(feature("a")),
    )
    assert data["states"] == {"unassessable": 2}
    assert data["review_count"] == 0


def test_outside_nl_targets_and_ais_cannot_leak_into_report() -> None:
    detections = collection(
        feature("nl"),
        feature("boston", lon=-71.06, lat=42.36),
        feature("flagged", in_study_area=False),
    )
    data = prepare_report_data(
        {},
        detections,
        ais_data=[
            {"mmsi": "nl", "lon": -52.2, "lat": 48},
            {"mmsi": "boston", "lon": -71.06, "lat": 42.36},
            {"mmsi": "invalid", "lon": "nan", "lat": 48},
        ],
    )
    assert data["states"]["excluded_outside_region"] == 2
    assert [f["properties"]["target_id"] for f in data["targets"]["features"]] == ["nl"]
    assert [v["mmsi"] for v in data["ais"]] == ["nl"]
    assert "boston" not in build_html_report({}, detections)


def test_injection_is_inert_in_html_and_embedded_json() -> None:
    hostile = '</script><script>alert("attack")</script>'
    detections = collection(
        feature(hostile, state="uncorrelated_candidate", correlation_reason=hostile)
    )
    report = build_html_report(
        {"mission_id": hostile, "spatial_bounds": {"ais_coverage": "aisstream"}},
        detections,
        ais_data=[{"mmsi": hostile, "lon": -52.2, "lat": 48}],
    )
    assert hostile not in report
    assert "&lt;/script&gt;" in report
    match = re.search(r'id="report-data">(.*?)</script>', report, flags=re.DOTALL)
    assert match is not None
    data = json.loads(match.group(1))
    assert data["targets"]["features"][0]["properties"]["target_id"] == hostile
    assert data["ais"][0]["mmsi"] == hostile
    assert "row.textContent" in report
    assert "onclick=" not in report


def test_full_correlation_states_override_raw_detection_annotations() -> None:
    payload = {"spatial_bounds": {"ais_coverage": "aisstream"}}
    detections = collection(feature("a"), feature("b"))
    correlations = collection(
        feature("a", state="ais_associated"),
        feature("b", state="ambiguous_association"),
    )
    data = prepare_report_data(payload, detections, correlations)
    assert data["states"] == {"ais_associated": 1, "ambiguous_association": 1}
    assert data["review_count"] == 1
    assert all(
        not f["properties"]["operational_alert"] for f in data["targets"]["features"]
    )


def test_nonfinite_score_unknown_kinematics_and_type_are_not_invented() -> None:
    detections = collection(feature("a", confidence=float("nan"), class_name="cargo"))
    data = prepare_report_data(
        {}, detections, ais_data=[{"mmsi": "1", "lon": -52.2, "lat": 48}]
    )
    assert data["targets"]["features"][0]["properties"]["confidence"] is None
    assert data["targets"]["features"][0]["properties"]["class_name"] == (
        "unresolved SAR object"
    )
    assert data["ais"][0]["speed_knots"] is None
    assert data["ais"][0]["course_deg"] is None
    report = build_html_report({}, detections)
    assert "cargo" not in report
    assert '"precision": null' in report or "&quot;precision&quot;: null" in report


def test_missing_geometry_is_reported_separately_and_not_plotted() -> None:
    invalid = feature("broken")
    invalid["geometry"] = None
    data = prepare_report_data({}, collection(invalid))
    assert data["states"] == {"excluded_invalid_geometry": 1}
    assert data["targets"]["features"] == []


def test_infrastructure_proximity_is_uncertain_exclusion_not_association() -> None:
    data = prepare_report_data(
        {"spatial_bounds": {"ais_coverage": "aisstream"}},
        collection(
            feature(
                "near-platform",
                surface="infrastructure_proximity",
                state="ais_associated",
            )
        ),
    )
    assert data["states"] == {"excluded_infrastructure": 1}
    assert data["surfaces"] == {"infrastructure_proximity": 1}
    assert data["review_count"] == 0
    assert "uncertain" in data["targets"]["features"][0]["properties"]["state_label"]
