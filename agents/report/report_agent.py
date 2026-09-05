#!/usr/bin/env python3
"""Evidence-led Newfoundland/Labrador SAR and AIS research reports.

HTML tables and provenance work without a network. The optional Leaflet map
requires the CDN and Esri tiles. Association never establishes intent or identity.
"""

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.errors import ShapelyError
from shapely.geometry import Point, mapping, shape

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.artifacts import mission_directory  # noqa: E402
from agents.region import REGION_NAME, load_region  # noqa: E402
from agents.tracking import RunTracker  # noqa: E402

STATE_LABELS = {
    "ais_associated": "AIS associated (provisional)",
    "uncorrelated_candidate": "Uncorrelated candidate: analyst review",
    "ambiguous_association": "Ambiguous association: analyst review",
    "unassessable": "Unassessable",
    "excluded_land": "Excluded: land",
    "excluded_coastal": "Excluded: coastal uncertainty",
    "excluded_infrastructure": "Excluded: infrastructure proximity (uncertain)",
    "excluded_unknown_surface": "Excluded: surface unverified",
    "excluded_outside_region": "Outside study area",
    "excluded_invalid_geometry": "Invalid geometry",
}
REVIEW_STATES = {"uncorrelated_candidate", "ambiguous_association"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TritonEye Report Agent")
    parser.add_argument("--payload", help="JSON payload string")
    parser.add_argument("--payload-file", help="JSON payload file")
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload:
        value = json.loads(args.payload)
    elif args.payload_file:
        with open(args.payload_file, encoding="utf-8") as stream:
            value = json.load(stream)
    elif not sys.stdin.isatty():
        value = json.loads(sys.stdin.read())
    else:
        raise ValueError("Provide --payload, --payload-file, or stdin")
    if not isinstance(value, dict):
        raise ValueError("Payload must be an object")
    return value


def read_geojson_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection")
    if not isinstance(value.get("features"), list):
        raise ValueError("GeoJSON features must be a list")
    return value


def _text(value: Any) -> str:
    return html.escape(str(value if value is not None else "unavailable"), quote=True)


def _safe_json(value: Any) -> str:
    """HTML parsers terminate script tags even for application/json data."""
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def prepare_report_data(
    payload: dict[str, Any],
    detections: dict[str, Any],
    correlations: dict[str, Any] | None = None,
    ais_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize states conservatively and independently enforce the study area.

    Legacy candidate-only artifacts cannot establish which remaining detections
    were associated, so unspecified states remain unassessable. Map data never
    contain out-of-area features even when an old payload incorrectly marks them.
    """
    spatial = payload.get("spatial_bounds") or {}
    area = load_region(spatial.get("analysis_region"))
    coverage = str(spatial.get("ais_coverage", "unknown"))
    coverage_absent = coverage.lower() in {
        "none",
        "unknown",
        "n/a",
        "mock",
        "synthetic",
        "unavailable",
        "",
    }
    annotated: dict[str, dict[str, Any]] = {}
    for feature in (correlations or {}).get("features", []):
        props = feature.get("properties") or {}
        if props.get("target_id") is not None:
            annotated[str(props["target_id"])] = props

    features = []
    states: Counter[str] = Counter()
    surfaces: Counter[str] = Counter()
    for feature in detections.get("features", []):
        props = dict(feature.get("properties") or {})
        props.update(annotated.get(str(props.get("target_id")), {}))
        try:
            geom = shape(feature.get("geometry"))
            if geom.is_empty or not geom.is_valid:
                raise ValueError("Empty or invalid target geometry")
            center = geom.centroid
            if not all(math.isfinite(v) for v in (*geom.bounds, center.x, center.y)):
                raise ValueError("Nonfinite target geometry")
        except (TypeError, ValueError, AttributeError, ShapelyError):
            states["excluded_invalid_geometry"] += 1
            continue
        if props.get("in_study_area") is False or not area.covers(center):
            states["excluded_outside_region"] += 1
            continue
        # Clip boxes straddling the boundary so even rejected layers remain local.
        geom = geom.intersection(area)
        surface = str(props.get("surface") or "unknown")
        surfaces[surface] += 1
        state = str(props.get("correlation_status") or "unassessable")
        if surface in {"infrastructure", "infrastructure_proximity"}:
            state = "excluded_infrastructure"
        elif surface in {"land", "coastal"}:
            state = "excluded_" + surface
        elif surface != "water":
            state = "excluded_unknown_surface"
        elif state not in STATE_LABELS or state.startswith("excluded_"):
            state = "unassessable"
        elif coverage_absent:
            state = "unassessable"
        props["correlation_status"] = state
        props["state_label"] = STATE_LABELS[state]
        props["review_required"] = state in REVIEW_STATES
        props["operational_alert"] = False
        props["confidence"] = _finite_number(props.get("confidence"))
        # No SAR detector in this project can infer cargo/fishing vessel type.
        props["class_name"] = "unresolved SAR object"
        features.append(
            {"type": "Feature", "geometry": mapping(geom), "properties": props}
        )
        states[state] += 1

    local_ais = []
    for vessel in ais_data or []:
        lon, lat = _finite_number(vessel.get("lon")), _finite_number(vessel.get("lat"))
        if lon is None or lat is None or not area.covers(Point(lon, lat)):
            continue
        local_ais.append(
            {
                "mmsi": str(vessel.get("mmsi") or "unknown"),
                "lon": lon,
                "lat": lat,
                "timestamp": str(vessel.get("timestamp") or "unavailable"),
                "speed_knots": _finite_number(vessel.get("speed_knots")),
                "course_deg": _finite_number(vessel.get("course_deg")),
            }
        )
    return {
        "targets": {"type": "FeatureCollection", "features": features},
        "ais": local_ais,
        "region": mapping(area),
        "bbox": list(area.bounds),
        "states": dict(states),
        "surfaces": dict(surfaces),
        "ais_coverage": coverage,
        "raw_detections": len(detections.get("features", [])),
        "review_count": sum(states[s] for s in REVIEW_STATES),
    }


def _target_table(features: list[dict[str, Any]]) -> str:
    rows = []
    for feature in features:
        props = feature["properties"]
        score = props.get("confidence")
        score_text = "unavailable" if score is None else f"{score:.3f}"
        rows.append(
            "<tr><td>"
            + _text(props.get("target_id", "unknown"))
            + "</td><td>"
            + _text(props["state_label"])
            + "</td><td>"
            + score_text
            + "</td><td>"
            + _text(props.get("association_mmsi"))
            + "</td><td>"
            + _text(props.get("correlation_reason", "No association evidence supplied"))
            + "</td></tr>"
        )
    if not rows:
        return '<p class="muted">No targets in this category.</p>'
    return (
        '<div class="table-wrap"><table><thead><tr><th>Target</th><th>State</th>'
        "<th>Detector score</th><th>Associated MMSI</th><th>Reason</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def build_html_report(
    payload: dict[str, Any],
    detections: dict[str, Any],
    dark_vessels: dict[str, Any] | None = None,
    ais_data: list[dict[str, Any]] | None = None,
) -> str:
    """Keep the positional legacy API; third argument should be all correlations."""
    data = prepare_report_data(payload, detections, dark_vessels, ais_data)
    features = data["targets"]["features"]
    review = [f for f in features if f["properties"]["review_required"]]
    remaining = [
        f
        for f in features
        if not f["properties"]["review_required"]
        and not f["properties"]["correlation_status"].startswith("excluded_")
    ]
    rejected = [
        f
        for f in features
        if f["properties"]["correlation_status"].startswith("excluded_")
    ]
    states, surfaces = data["states"], data["surfaces"]
    provenance = {
        "mission_id": payload.get("mission_id", "mission_default"),
        "acquisition_time": payload.get("acquisition_time", "unavailable"),
        "region": REGION_NAME,
        "processing": payload.get("processing", {"status": "unavailable"}),
        "landmask": payload.get("landmask", {"status": "unavailable"}),
        "ais_coverage": data["ais_coverage"],
        "ais_coverage_completeness": "unverified",
        "correlation_summary": payload.get("correlation_summary", {}),
        "evaluation": payload.get(
            "evaluation",
            {"precision": None, "false_alarms_per_km2": None, "status": "not measured"},
        ),
        "mlflow_run_id": payload.get("mlflow_run_id", "not tracked"),
        "ice_context": payload.get("ice_context", {"status": "unavailable"}),
    }
    model = payload.get("processing") or {}
    synthetic_notice = (
        '<div class="notice"><strong>SYNTHETIC DEMONSTRATION</strong><br>'
        "Simulated imagery and AIS. Not real surveillance or accuracy evidence.</div>"
        if payload.get("mode") == "mock"
        or model.get("detector") == "mock"
        or data["ais_coverage"] in {"mock", "synthetic"}
        else ""
    )
    summary = " · ".join(
        f"{name}: {surfaces.get(name, 0)}"
        for name in (
            "water",
            "land",
            "coastal",
            "infrastructure",
            "infrastructure_proximity",
            "unknown",
        )
    )
    outside = states.get("excluded_outside_region", 0)
    invalid = states.get("excluded_invalid_geometry", 0)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TritonEye | {_text(provenance["mission_id"])}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#09131e;
color:#dce7ef;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;
padding:32px}}h1{{color:#83d5d6;font-size:32px;margin:0}}h2{{font-size:20px;
margin:28px 0 10px}}.muted,small{{color:#a5b6c7}}.eyebrow{{text-transform:uppercase;
letter-spacing:2px;color:#83d5d6;font-size:12px}}.notice{{border-left:4px solid #f4bf66;
background:#25241f;padding:16px;margin:22px 0}}.cards{{display:grid;
grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#132332;padding:16px;
border-radius:8px}}.value{{font-size:30px;color:#fff}}#map{{height:470px;
background:#132332;border-radius:8px}}#map-status{{margin:8px 0}}table{{width:100%;
border-collapse:collapse;font-size:13px}}th,td{{padding:11px;text-align:left;
border-bottom:1px solid #304252;overflow-wrap:anywhere}}th{{color:#83d5d6}}
.table-wrap{{overflow:auto}}details{{padding:12px 0}}summary{{cursor:pointer}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;
background:#132332;padding:16px}}.leaflet-popup-content{{color:#162635}}
@media(max-width:700px){{main{{padding:16px}}.cards{{grid-template-columns:repeat(2,1fr)}}
#map{{height:340px}}}}@media print{{#map,#map-status{{display:none}}
body{{background:white;
color:black}}.card,pre,.notice{{background:#eee;color:black}}.value{{color:black}}}}
</style></head><body><main>
<div class="eyebrow">Newfoundland and Labrador · Research prototype</div>
<h1>TritonEye mission review</h1>
{synthetic_notice}
<p class="muted">Mission {_text(provenance["mission_id"])} · Acquisition
{_text(provenance["acquisition_time"])}<br>{_text(REGION_NAME)}.
Study boundary is not a legal maritime boundary.</p>
<div class="notice"><strong>Analyst review required.
Iceberg discrimination is unresolved.</strong>
<br>A SAR return may be a vessel, iceberg, sea ice or clutter. An absent AIS match does
not establish that a vessel switched off its transponder, is non-compliant,
or poses a threat.
Detector scores are not calibrated probabilities of vessel identity. Operational alerts
are not enabled. This portfolio prototype is not certified or endorsed by C-CORE.</div>
<div class="cards">
<div class="card"><div class="value">{len(features)}</div>In-area SAR targets</div>
<div class="card"><div class="value">{states.get("ais_associated", 0)}</div>
AIS associations (provisional)</div>
<div class="card"><div class="value">{data["review_count"]}</div>Review candidates</div>
<div class="card"><div class="value">{states.get("unassessable", 0)}</div>
Unassessable</div>
</div>
<p>Recorded AIS source: <strong>{_text(data["ais_coverage"])}</strong>.
Coverage completeness is unverified. Missing or stale telemetry prevents
association assessment.
AIS points are historical observations, not current vessel positions.</p>
<p class="muted">Raw detections: {data["raw_detections"]}. In-area surface breakdown:
{_text(summary)}. Outside study area: {outside}; invalid geometry: {invalid}.
Land/coastal/infrastructure exclusions are not AIS associations.</p>
<h2>Study area and observations</h2><div id="map" role="img"
aria-label="Map of the Newfoundland and Labrador maritime study area"></div>
<p id="map-status" class="muted">The optional map requires the Leaflet CDN and online
Esri tiles. All target tables and provenance below remain available offline.</p>
<h2>Uncorrelated and ambiguous candidates</h2>
<p class="muted">Review the SAR chip, land/coast proximity, acquisition-matched ice
information and AIS history before assigning an identity. Suppression can also remove
real coastal vessels; it is not a measured precision improvement.</p>
{_target_table(review)}
<h2>Provisional associations and unassessable targets</h2>{_target_table(remaining)}
<details><summary>Excluded in-area returns ({len(rejected)})</summary>
{_target_table(rejected)}</details>
<h2>Evidence and reproducibility</h2>
<p>Detector: {_text(model.get("detector"))} · Actual threshold:
{_text(model.get("threshold"))}<br>Weights SHA-256:
<code>{_text(model.get("weights_sha256"))}</code></p>
<p>Precision and false alarms per square kilometre require independently labelled NL
scenes. Counts and AIS association rates do not measure precision.
Missing metrics remain
unavailable; the report supplies no inferred accuracy estimate.</p>
<details open><summary>Mission provenance</summary>
<pre>{_text(json.dumps(provenance, indent=2, ensure_ascii=True, allow_nan=False))}</pre>
</details><p class="muted">Method references and access requirements are documented in
docs/MDA_METHODS.md and docs/DATA_SOURCES.md in the project repository.</p>
</main>
<script type="application/json" id="report-data">{_safe_json(data)}</script>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(() => {{
  if (typeof L === 'undefined') return;
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const box = data.bbox;
  const map = L.map('map').fitBounds([[box[1],box[0]],[box[3],box[2]]]);
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/' +
    'World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{maxZoom:13,
    attribution:'Basemap © Esri; GEBCO, NOAA, CHS'}}).addTo(map);
  L.geoJSON(data.region, {{style:{{color:'#83d5d6',weight:1,fillOpacity:0,
    dashArray:'6,6'}}}}).addTo(map);
  const groups = {{'AIS associated (provisional)':L.featureGroup().addTo(map),
    'Analyst review':L.featureGroup().addTo(map),
    'Unassessable':L.featureGroup().addTo(map),
    'Excluded returns':L.featureGroup(), 'Recorded AIS':L.featureGroup()}};
  function popup(items) {{
    const box = document.createElement('div');
    for (const [label,value] of items) {{
      const row = document.createElement('div');
      row.textContent = label + ': ' + (value ?? 'unavailable');
      box.appendChild(row);
    }}
    return box;
  }}
  data.targets.features.forEach(feature => {{
    const p = feature.properties, s = p.correlation_status;
    const group = s.startsWith('excluded_') ? 'Excluded returns' :
      s === 'ais_associated' ? 'AIS associated (provisional)' :
      p.review_required ? 'Analyst review' : 'Unassessable';
    const color = group === 'Analyst review' ? '#eab35f' :
      group === 'AIS associated (provisional)' ? '#2c9da6' : '#9099a8';
    L.geoJSON(feature, {{style:{{color,weight:2,fillOpacity:0.15}},
      pointToLayer:(f,ll)=>L.circleMarker(ll,{{color,radius:5}}),
      onEachFeature:(f,layer)=>layer.bindPopup(popup([
        ['Target',p.target_id], ['State',p.state_label],
        ['Detector score (uncalibrated)',p.confidence],
        ['Associated MMSI',p.association_mmsi], ['Reason',p.correlation_reason],
        ['Iceberg discrimination','unresolved']]))}}).addTo(groups[group]);
  }});
  data.ais.forEach(v => L.circleMarker([v.lat,v.lon],{{radius:4,color:'#9372db'}})
    .bindPopup(popup([['Recorded MMSI',v.mmsi],['Observation time',v.timestamp],
      ['SOG (knots)',v.speed_knots],['COG (degrees)',v.course_deg]]))
    .addTo(groups['Recorded AIS']));
  L.control.layers(null,groups).addTo(map);
  document.getElementById('map-status').textContent =
    'Online basemap. Excluded returns and recorded AIS are optional layers. ' +
    'All observations are limited to the NL study area; no operational alerts.';
}})();
</script></body></html>"""


def main() -> None:
    payload = get_payload(parse_arguments())
    detection_path = payload.get("detections_geojson")
    if not detection_path:
        raise ValueError("Missing detections_geojson path")
    detections = read_geojson_file(detection_path)
    correlation_path = (
        payload.get("correlation_geojson")
        or payload.get("candidates_geojson")
        or payload.get("dark_vessels_geojson")
    )
    correlations = read_geojson_file(correlation_path) if correlation_path else None
    ais_data: list[dict[str, Any]] = []
    ais_path = payload.get("ais_telemetry")
    if ais_path:
        try:
            with open(ais_path, encoding="utf-8") as stream:
                ais_data = list(csv.DictReader(stream))
        except (OSError, csv.Error) as exc:
            print(f"AIS observations unavailable to report: {exc}", file=sys.stderr)
    report = build_html_report(payload, detections, correlations, ais_data)
    root = Path(__file__).resolve().parents[2]
    directory = mission_directory(
        payload.get("mission_id", "mission_default"), root / "reports"
    )
    report_path = directory / "report.html"
    report_path.write_text(report, encoding="utf-8")
    payload["report_html"] = str(report_path)
    tracker = RunTracker.resume(payload.get("mlflow_run_id"))
    try:
        tracker.set_tags({"stage": "report"})
        tracker.log_artifact(str(report_path), "report")
    finally:
        tracker.end()
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
