"""Fetch small public imagery exports for NL shoreline development controls.

Sources remain local in data/reference/regional; do not redistribute basemap
pixels as open data. Saving an image is not visual review or benchmark labelling.
"""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer"
SITES = {
    "bay_of_exploits": [-55.170, 49.240, -55.110, 49.295],
    "ferryland": [-52.905, 47.007, -52.867, 47.038],
    "makkovik": [-59.200, 55.040, -59.120, 55.100],
    "bonavista": [-53.134, 48.645, -53.103, 48.665],
    "twillingate": [-54.782, 49.623, -54.750, 49.642],
    "lewisporte": [-55.070, 49.233, -55.042, 49.252],
}


def export_bounds(metadata: dict[str, Any]) -> list[float]:
    """Use the actual displayed WGS84 export, not the requested rectangle."""
    extent = metadata["extent"]
    if extent.get("spatialReference", {}).get("wkid") != 4326:
        raise ValueError("WGS84 imagery export required")
    bounds = [float(extent[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
    if not all(math.isfinite(value) for value in bounds) or not (
        -180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90
    ):
        raise ValueError("Valid imagery export bounds required")
    return bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--site", choices=sorted(SITES))
    parser.add_argument(
        "--compare", action="store_true", help="Compare cached OSM and CanVec locally"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data/reference/regional"
    for site, bounds in SITES.items():
        if args.site and site != args.site:
            continue
        directory = root / site
        directory.mkdir(parents=True, exist_ok=True)
        params: dict[str, str | int] = {
            "bbox": ",".join(map(str, bounds)),
            "bboxSR": 4326,
            "imageSR": 4326,
            "size": "1600,1000",
            "format": "png",
            "f": "image",
        }
        if args.fetch:
            if (directory / "imagery.png").exists():
                raise ValueError("Existing imagery evidence cannot be overwritten")
            response = requests.get(URL + "/export", params=params, timeout=90)
            response.raise_for_status()
            if not response.content.startswith(b"\x89PNG"):
                raise RuntimeError("Service did not return PNG imagery")
            (directory / "imagery.png").write_bytes(response.content)
            metadata = requests.get(
                URL + "/export", params={**params, "f": "pjson"}, timeout=90
            )
            metadata.raise_for_status()
            (directory / "export.json").write_text(metadata.text, encoding="utf-8")
            citation = requests.get(
                URL + "/identify",
                params={
                    "geometry": json.dumps(
                        {
                            "x": (bounds[0] + bounds[2]) / 2,
                            "y": (bounds[1] + bounds[3]) / 2,
                        }
                    ),
                    "geometryType": "esriGeometryPoint",
                    "sr": 4326,
                    "mapExtent": params["bbox"],
                    "imageDisplay": "1600,1000,96",
                    "tolerance": 1,
                    "returnGeometry": "false",
                    "f": "pjson",
                },
                timeout=90,
            )
            citation.raise_for_status()
            (directory / "citation.json").write_text(citation.text, encoding="utf-8")
            (directory / "provenance.json").write_text(
                json.dumps(
                    {
                        "service": URL,
                        "request": params,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "sha256": hashlib.sha256(response.content).hexdigest(),
                        "licence": (
                            "Esri and imagery suppliers; not open redistribution"
                        ),
                        "scope": (
                            "Sparse visual development checks only; not vessel truth"
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(site, directory)
        if args.compare:
            import geopandas as gpd
            import matplotlib
            from shapely.geometry import box

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            project = root.parents[2]
            metadata = json.loads((directory / "export.json").read_text())
            extent = metadata["extent"]
            # ArcGIS may expand the requested bounds to preserve image aspect.
            # Geometry and controls must use the extent actually displayed.
            bounds = export_bounds(metadata)
            osm = gpd.read_file(
                project / "data/reference/land-polygons-split-4326/land_polygons.shp",
                bbox=tuple(bounds),
            )
            osm.geometry = osm.geometry.intersection(box(*bounds))
            osm.to_file(directory / "osm_land.geojson", driver="GeoJSON")
            archive = (
                project / "data/reference/st_johns/source/canvec_50K_NL_Hydro_shp.zip"
            )
            water = gpd.read_file(
                f"zip://{archive}!canvec_50K_NL_Hydro/waterbody_2_2.shp",
                bbox=tuple(bounds),
            )
            water = water.loc[water.definit == 85].to_crs(4326)
            water.geometry = water.geometry.intersection(box(*bounds))
            water.to_file(directory / "canvec_ocean.geojson", driver="GeoJSON")
            image = plt.imread(directory / "imagery.png")
            controls = gpd.read_file(
                project / f"configs/coastline/{site}_checks.geojson"
            )
            import sys

            sys.path.insert(0, str(project))
            from agents.landmask import LandMask, LandMaskUnavailable, local_aeqd_crs

            crs = local_aeqd_crs(bounds)
            candidates = {
                "osm": LandMask(osm.to_crs(crs).geometry, crs, "osm", "ODbL"),
                "canvec": LandMask(
                    gpd.GeoSeries(
                        [box(*bounds).difference(water.geometry.union_all())], crs=4326
                    ).to_crs(crs),
                    crs,
                    "canvec",
                    "Open Government Licence - Canada",
                ),
            }
            checks = {}
            for name, mask in candidates.items():
                try:
                    mask.validate_controls(
                        bounds,
                        str(project / f"configs/coastline/{site}_checks.geojson"),
                    )
                except LandMaskUnavailable:
                    if mask.validation.get("status") != "failed":
                        raise
                checks[name] = mask.validation
            (directory / "verification.json").write_text(
                json.dumps(checks, indent=2, allow_nan=False), encoding="utf-8"
            )
            (directory / "comparison_sources.json").write_text(
                json.dumps(
                    {
                        "actual_image_bounds_wgs84": bounds,
                        "imagery_sha256": hashlib.sha256(
                            (directory / "imagery.png").read_bytes()
                        ).hexdigest(),
                        "controls_sha256": hashlib.sha256(
                            (
                                project / f"configs/coastline/{site}_checks.geojson"
                            ).read_bytes()
                        ).hexdigest(),
                        "osm_local_comparison_sha256": hashlib.sha256(
                            (directory / "osm_land.geojson").read_bytes()
                        ).hexdigest(),
                        "canvec_local_comparison_sha256": hashlib.sha256(
                            (directory / "canvec_ocean.geojson").read_bytes()
                        ).hexdigest(),
                        "osm_source": "https://osmdata.openstreetmap.de/data/land-polygons.html",
                        "osm_licence": "ODbL",
                        "canvec_source": "https://open.canada.ca/data/en/dataset/9d96e8c9-22fe-4ad2-b5e8-94a6991b744b",
                        "canvec_licence": "Open Government Licence - Canada",
                        "scope": (
                            "Local comparison only; no universal adoption "
                            "or independent accuracy claim"
                        ),
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            print(json.dumps(checks, indent=2, allow_nan=False))
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            for ax, layer, title in zip(
                axes, (osm, water), ("OSM land", "CanVec ocean (comparison only)")
            ):
                ax.imshow(
                    image,
                    extent=[
                        extent["xmin"],
                        extent["xmax"],
                        extent["ymin"],
                        extent["ymax"],
                    ],
                )
                layer.boundary.plot(ax=ax, color="red", linewidth=1)
                controls.plot(ax=ax, color="yellow", markersize=20)
                ax.set_title(title)
                ax.set_xlim(extent["xmin"], extent["xmax"])
                ax.set_ylim(extent["ymin"], extent["ymax"])
            fig.suptitle("Esri / Vantor imagery; sparse shoreline development checks")
            fig.savefig(directory / "comparison.png", dpi=120)
            plt.close(fig)


if __name__ == "__main__":
    main()
