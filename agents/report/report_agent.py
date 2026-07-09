#!/usr/bin/env python3
"""
TritonEye Mission Report Agent
Generates an interactive HTML dashboard using Leaflet.js and Esri Ocean maps.
Presents cooperative targets alongside highlighted dark vessel alerts.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Standard python libraries (no third-party dependencies required)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="TritonEye Report Agent")
    parser.add_argument(
        "--payload", type=str, help="JSON payload string passed directly"
    )
    parser.add_argument(
        "--payload-file", type=str, help="Path to JSON file containing payload"
    )
    return parser.parse_args()


def get_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Retrieves the payload from CLI args or stdin."""
    if args.payload:
        return json.loads(args.payload)  # type: ignore[no-any-return]
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # Stdin fallback
    if not sys.stdin.isatty():
        return json.loads(sys.stdin.read())  # type: ignore[no-any-return]

    raise ValueError(
        "No input payload provided via --payload, --payload-file, or stdin."
    )


def read_geojson_file(path: str) -> Dict[str, Any]:
    """Reads a GeoJSON file and returns its parsed structure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON data not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def build_html_report(
    payload: Dict[str, Any],
    detections: Dict[str, Any],
    dark_vessels: Dict[str, Any]
) -> str:
    """Constructs the self-contained interactive Leaflet HTML dashboard string."""
    mission_id = payload.get("mission_id", "mission_default")
    acq_time = payload.get("acquisition_time", "N/A")
    bbox = payload.get("spatial_bounds", {}).get("bbox", [-52.6, 47.3, -51.5, 47.8])

    det_count = len(detections.get("features", []))
    dark_count = len(dark_vessels.get("features", []))
    active_count = max(0, det_count - dark_count)

    # Escape quotes and serialize data for injection into script tag
    detections_json = json.dumps(detections)
    dark_vessels_json = json.dumps(dark_vessels)
    bbox_json = json.dumps(bbox)

    # Construct sidebar list rows dynamically in python to facilitate styling
    vessel_rows = []
    for f in dark_vessels.get("features", []):
        props = f.get("properties", {})
        tid = props.get("target_id", "Unknown")
        cls_name = props.get("class_name", "unknown")
        conf = props.get("confidence", 0.0)

        # Extract coordinates of polygon center to pan to
        coords = f.get("geometry", {}).get("coordinates", [[[]]])[0]
        if coords and len(coords) >= 4:
            # Average the bounding box points for centroid estimation
            lons = [pt[0] for pt in coords]
            lats = [pt[1] for pt in coords]
            c_lon = sum(lons) / len(lons)
            c_lat = sum(lats) / len(lats)
        else:
            c_lon, c_lat = -52.0, 47.5

        row_html = (
            f'<div class="vessel-item" '
            f'onclick="panToTarget(\'{tid}\', {c_lat}, {c_lon})">'
            f'  <div class="vessel-id">{tid}</div>'
            f'  <div class="vessel-details">'
            f'    <span>Type: <b>{cls_name.upper()}</b></span>'
            f'    <span>Conf: <b>{conf:.2f}</b></span>'
            f'  </div>'
            f'</div>'
        )
        vessel_rows.append(row_html)

    if vessel_rows:
        vessels_list_html = "\n".join(vessel_rows)
    else:
        vessels_list_html = (
            '<div class="no-alerts">No Dark Vessels Isolated</div>'
        )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TritonEye Mission Brief - {mission_id}</title>

    <!-- Premium Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap"
          rel="stylesheet">

    <!-- Leaflet.js CDN -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
            height: 100vh;
            overflow: hidden;
            display: flex;
        }}
        #map {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}

        /* Glassmorphism sidebar panel */
        .sidebar {{
            position: absolute;
            top: 20px;
            left: 20px;
            bottom: 20px;
            width: 380px;
            z-index: 10;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            padding: 24px;
            overflow-y: auto;
        }}

        h1 {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #38bdf8;
            margin-bottom: 6px;
            text-transform: uppercase;
        }}

        .subtitle {{
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 12px;
        }}

        .meta-item {{
            margin-bottom: 12px;
            font-size: 13px;
        }}

        .meta-item span {{
            color: #94a3b8;
            display: inline-block;
            width: 100px;
        }}

        .meta-item b {{
            color: #f1f5f9;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 12px 6px;
            text-align: center;
        }}

        .stat-val {{
            font-size: 20px;
            font-weight: 700;
            color: #f8fafc;
        }}

        .stat-val.detections {{ color: #38bdf8; }}
        .stat-val.active {{ color: #4ade80; }}
        .stat-val.dark {{ color: #f87171; }}

        .stat-lbl {{
            font-size: 10px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-top: 4px;
        }}

        .alert-section-title {{
            font-size: 12px;
            font-weight: 700;
            color: #ef4444;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 20px;
            margin-bottom: 12px;
        }}

        .vessel-list {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .vessel-item {{
            background: rgba(239, 68, 68, 0.06);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .vessel-item:hover {{
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(239, 68, 68, 0.35);
            transform: translateX(4px);
        }}

        .vessel-id {{
            font-size: 14px;
            font-weight: 700;
            color: #f87171;
            margin-bottom: 4px;
        }}

        .vessel-details {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #cbd5e1;
        }}

        .no-alerts {{
            text-align: center;
            padding: 20px;
            font-size: 13px;
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            border: 1px dashed rgba(255, 255, 255, 0.05);
        }}

        /* Map custom styling */
        .leaflet-bar a {{
            background-color: rgba(15, 23, 42, 0.95) !important;
            color: #f1f5f9 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        .leaflet-bar a:hover {{
            background-color: #38bdf8 !important;
            color: #0f172a !important;
        }}

        /* Popup override styling */
        .leaflet-popup-content-wrapper {{
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(8px);
            color: #f1f5f9;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            font-family: 'Outfit', sans-serif;
        }}
        .leaflet-popup-tip {{
            background: rgba(15, 23, 42, 0.95);
        }}
        .popup-title {{
            font-weight: 700;
            margin-bottom: 4px;
            font-size: 13px;
        }}
        .popup-title.dark-alert {{
            color: #f87171;
        }}
        .popup-title.coop {{
            color: #38bdf8;
        }}
        .popup-body {{
            font-size: 11px;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>

    <div class="sidebar">
        <h1>TritonEye Brief</h1>
        <div class="subtitle">Maritime Domain Awareness Report</div>

        <div class="meta-item"><span>Mission ID</span><b>{mission_id}</b></div>
        <div class="meta-item"><span>Acquisition</span><b>{acq_time}</b></div>
        <div class="meta-item"><span>MLflow Run</span><b><a
            href="http://localhost:5000/#/experiments/0/runs/{mission_id}"
            target="_blank" style="color: #38bdf8; text-decoration: none;"
            >View Experiment ↗</a></b></div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val detections">{det_count}</div>
                <div class="stat-lbl">Targets</div>
            </div>
            <div class="stat-card">
                <div class="stat-val active">{active_count}</div>
                <div class="stat-lbl">Active</div>
            </div>
            <div class="stat-card">
                <div class="stat-val dark">{dark_count}</div>
                <div class="stat-lbl">Dark</div>
            </div>
        </div>

        <div class="alert-section-title">Dark Vessel Alerts</div>
        <div class="vessel-list">
            {vessels_list_html}
        </div>
    </div>

    <div id="map"></div>

    <script>
        // 1. Inject coordinates and GeoJSON sets
        const bbox = {bbox_json};
        const detections = {detections_json};
        const darkVessels = {dark_vessels_json};

        // 2. Initialize Leaflet Map centered on St. John's AOI bounds
        const map = L.map('map', {{
            zoomControl: false
        }}).fitBounds([
            [bbox[1], bbox[0]],
            [bbox[3], bbox[2]]
        ]);

        L.control.zoom({{
            position: 'topright'
        }}).addTo(map);

        const base_url = 'https://server.arcgisonline.com/ArcGIS/rest/' +
                         'services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}';
        const ref_url = 'https://server.arcgisonline.com/ArcGIS/rest/' +
                        'services/Ocean/World_Ocean_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}';

        L.tileLayer(base_url, {{
            maxZoom: 13,
            attribution: 'Basemap &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS'
        }}).addTo(map);

        L.tileLayer(ref_url, {{
            maxZoom: 13,
            attribution: 'Labels &copy; Esri'
        }}).addTo(map);

        // Reference mapping directory for quick pan lookup callbacks
        const targetsMap = {{}};

        // 4. Plot cooperative detections (Blue Polygons)
        L.geoJSON(detections, {{
            style: function(feature) {{
                return {{
                    color: '#0284c7',
                    weight: 2,
                    fillColor: '#0ea5e9',
                    fillOpacity: 0.15
                }};
            }},
            onEachFeature: function(feature, layer) {{
                const props = feature.properties || {{}};
                const tid = props.target_id || 'N/A';
                const cls = props.class_name || 'unknown';
                const conf = props.confidence || 0.0;

                layer.bindPopup(`
                    <div class="popup-title coop">Cooperative Target</div>
                    <div class="popup-body">
                        ID: <b>${{tid}}</b><br/>
                        Class: <b>${{cls.toUpperCase()}}</b><br/>
                        Confidence: <b>${{conf.toFixed(2)}}</b>
                    </div>
                `);

                targetsMap[tid] = layer;
            }}
        }}).addTo(map);

        // 5. Plot Dark Vessel anomalies (Thick Red Polygons + Highlight popup)
        L.geoJSON(darkVessels, {{
            style: function(feature) {{
                return {{
                    color: '#dc2626',
                    weight: 3,
                    fillColor: '#ef4444',
                    fillOpacity: 0.3,
                    dashArray: '4, 4'
                }};
            }},
            onEachFeature: function(feature, layer) {{
                const props = feature.properties || {{}};
                const tid = props.target_id || 'N/A';
                const cls = props.class_name || 'unknown';
                const conf = props.confidence || 0.0;

                layer.bindPopup(`
                    <div class="popup-title dark-alert">
                        🚨 ALERT: Dark Vessel Candidate
                    </div>
                    <div class="popup-body">
                        ID: <b>${{tid}}</b><br/>
                        Class: <b>${{cls.toUpperCase()}}</b><br/>
                        Confidence: <b>${{conf.toFixed(2)}}</b><br/>
                        AIS Telemetry Status: <b style="color:#ef4444">OFFLINE</b>
                    </div>
                `);

                targetsMap[tid] = layer;
            }}
        }}).addTo(map);

        // 6. Callback interface from the sidebar click events
        function panToTarget(targetId, lat, lon) {{
            map.setView([lat, lon], 12);
            const targetLayer = targetsMap[targetId];
            if (targetLayer) {{
                targetLayer.openPopup();
            }}
        }}
    </script>
</body>
</html>"""
    return html_template


def main() -> None:
    # 1. Parse arguments and configuration
    args = parse_arguments()
    payload = get_payload(args)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 2. Extract paths from the payload
    detections_geojson_path = payload.get("detections_geojson")
    dark_vessels_geojson_path = payload.get("dark_vessels_geojson")
    mission_id = payload.get("mission_id", "mission_default")

    if not detections_geojson_path or not dark_vessels_geojson_path:
        raise ValueError(
            "Invalid payload: Missing detections_geojson "
            "or dark_vessels_geojson path references."
        )

    # 3. Load actual GeoJSON structures
    detections = read_geojson_file(detections_geojson_path)
    dark_vessels = read_geojson_file(dark_vessels_geojson_path)

    # 4. Generate report HTML string
    report_html = build_html_report(payload, detections, dark_vessels)

    # 5. Save HTML output
    report_dir = os.path.join(base_dir, "reports", mission_id)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    # 6. Update payload and print to stdout
    payload["report_html"] = report_path
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
