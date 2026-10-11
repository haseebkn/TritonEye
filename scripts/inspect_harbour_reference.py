"""Extract local CanVec ocean geometry and compare it with municipal imagery.

Development inspection only. Source downloads and imagery stay in data/; no
claim of a surveyed shoreline or vessel ground truth is made by this script.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import requests
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.landmask import LandMask, LandMaskUnavailable  # noqa: E402

CANVEC_URL = (
    "https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/shp/Hydro/"
    "canvec_50K_NL_Hydro_shp.zip"
)
IMAGERY_URL = (
    "https://map.stjohns.ca/mapsrv/rest/services/Imagery/Imagery2022/MapServer"
)
BOUNDS = (-52.725, 47.55, -52.668, 47.585)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path, params: dict[str, Any] | None = None) -> None:
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(url, params=params, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch missing public sources"
    )
    parser.add_argument(
        "--update-fixture",
        action="store_true",
        help="Explicitly replace the attributed offline OSM test snapshot",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/reference/st_johns"
    (directory / "source").mkdir(parents=True, exist_ok=True)
    archive = directory / "source/canvec_50K_NL_Hydro_shp.zip"
    image_path = directory / "source/harbour_aerial2022.png"
    export = {
        "bbox": ",".join(map(str, BOUNDS)),
        "bboxSR": 4326,
        "imageSR": 32181,
        "size": "4096,4096",
        "format": "png",
        "f": "image",
    }
    if args.fetch:
        download(CANVEC_URL, archive)
        download(IMAGERY_URL + "/export", image_path, export)
        for name, url, params in (
            ("imagery_service.json", IMAGERY_URL, {"f": "pjson"}),
            ("imagery_export.json", IMAGERY_URL + "/export", {**export, "f": "pjson"}),
        ):
            download(url, directory / "source" / name, params)
    bounds = BOUNDS
    canvec_path = f"zip://{archive}!canvec_50K_NL_Hydro/waterbody_2_2.shp"
    water = gpd.read_file(canvec_path, bbox=bounds)
    water = water.loc[water.definit == 85].to_crs("EPSG:4326")
    water.geometry = water.geometry.intersection(box(*bounds))
    water.to_file(directory / "canvec_ocean_clip.geojson", driver="GeoJSON")
    if water.empty or not water.geometry.is_valid.all():
        raise RuntimeError("Missing or invalid CanVec ocean geometry")

    osm = gpd.read_file(
        root / "data/reference/land-polygons-split-4326/land_polygons.shp",
        bbox=bounds,
    )
    osm.geometry = osm.geometry.intersection(box(*bounds))
    osm.to_file(directory / "osm_land_clip.geojson", driver="GeoJSON")
    export_metadata = json.loads((directory / "source/imagery_export.json").read_text())
    frame = export_metadata["extent"]
    if frame["spatialReference"]["wkid"] != 32181:
        raise RuntimeError("Unexpected imagery coordinate system")
    extent = (frame["xmin"], frame["xmax"], frame["ymin"], frame["ymax"])
    image = plt.imread(image_path)
    if image.shape[:2] != (export_metadata["height"], export_metadata["width"]):
        raise RuntimeError("Imagery dimensions disagree with export metadata")
    controls_path = root / "configs/coastline/st_johns_checks.geojson"
    controls = gpd.read_file(controls_path).to_crs("EPSG:32181")
    # Small OSM snapshot is a source fixture for offline behavioral regressions.
    # Preserve its ODbL attribution in the adjacent documentation.
    fixture = root / "tests/fixtures/st_johns_osm_land.geojson"
    if args.update_fixture:
        fixture.parent.mkdir(exist_ok=True)
        osm.to_file(fixture, driver="GeoJSON")
    control_bounds = (-52.73, 47.54, -52.66, 47.59)
    mask = LandMask.for_footprint(
        control_bounds, cache_dir=str(root / "data/reference")
    )
    crs = mask.crs
    canvec_land = gpd.GeoDataFrame(
        geometry=[box(*bounds).difference(water.geometry.union_all())], crs="EPSG:4326"
    ).to_crs(crs)
    candidate = LandMask(
        canvec_land.geometry, crs, "canvec", "Open Government Licence – Canada"
    )
    try:
        candidate.validate_controls(control_bounds, str(controls_path))
    except LandMaskUnavailable:
        pass  # A failed candidate is an audit result, never an adopted replacement.
    sources = []
    for path, url, licence in (
        (archive, CANVEC_URL, "Open Government Licence – Canada"),
        (
            image_path,
            IMAGERY_URL + "/export",
            "City of St. John's; redistribution terms unverified",
        ),
        (
            root / "data/reference/land-polygons-split-4326/land_polygons.shp",
            "https://osmdata.openstreetmap.de/data/land-polygons.html",
            "ODbL; © OpenStreetMap contributors",
        ),
    ):
        sources.append(
            {
                "path": str(path.relative_to(root)),
                "url": url,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "local_file_time_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "licence": licence,
            }
        )
    checks = mask.validation["checks"]
    labels, distances = mask.classify(
        gpd.read_file(controls_path).geometry.x.tolist(),
        gpd.read_file(controls_path).geometry.y.tolist(),
    )
    policy = [
        {"id": row.id, "surface": labels[i], "distance_to_shore_m": float(distances[i])}
        for i, row in enumerate(controls.itertuples())
    ]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds_wgs84": bounds,
        "sources": sources,
        "imagery_export": export_metadata,
        "current_osm": mask.validation,
        "canvec_candidate": candidate.validation,
        "canvec_attributes": water.drop(columns="geometry").to_dict("records"),
        "coastal_policy_m": 300,
        "current_policy_classes": policy,
        "decision": (
            "Retain OSM; CanVec fails controls: "
            + ", ".join(
                c["id"] for c in candidate.validation["checks"] if not c["passed"]
            )
            if candidate.validation["failed"]
            else "Both pass sparse controls; no replacement automatically adopted."
        ),
        "limitations": (
            "Sparse visual controls do not establish surveyed accuracy, current "
            "shoreline, province-wide coverage or vessel detection performance."
        ),
    }
    (directory / "verification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "osm_checks": len(checks),
                "osm_failed": mask.validation["failed"],
                "canvec_failed": candidate.validation["failed"],
                "report": str(directory / "verification.json"),
            },
            indent=2,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, layer, title in zip(
        axes, (osm, water), ("Existing OSM land boundary", "CanVec ocean boundary")
    ):
        ax.imshow(image, extent=extent)
        layer.to_crs("EPSG:32181").boundary.plot(ax=ax, color="#ff4949", linewidth=1)
        for row in controls.itertuples():
            color = "#40ffff" if row.expected == "water" else "#ffff40"
            ax.scatter(
                [row.geometry.x],
                [row.geometry.y],
                c=color,
                s=28,
                edgecolors="black",
                zorder=5,
            )
            ax.annotate(
                row.id.replace("_", " "),
                (row.geometry.x, row.geometry.y),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 1},
            )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.suptitle("St. John's harbour: City Aerial2022 / NAD83 MTM zone 1")
    fig.tight_layout()
    fig.savefig(directory / "shoreline_comparison.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, layer, title in zip(
        axes, (osm, water), ("OSM quay detail", "CanVec quay detail")
    ):
        ax.imshow(image, extent=extent)
        layer.to_crs("EPSG:32181").boundary.plot(ax=ax, color="#ff4949", linewidth=2)
        ax.scatter([327600], [5270140], c="#ffff40", edgecolors="black", s=80)
        ax.set_xlim(327450, 327780)
        ax.set_ylim(5270020, 5270300)
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style="plain")
    fig.suptitle("North-quay land control / City Aerial2022")
    fig.tight_layout()
    fig.savefig(directory / "quay_detail.png", dpi=160)
    plt.close(fig)
    (directory / "imagery_extent.json").write_text(
        json.dumps({"crs": "EPSG:32181", "extent": list(extent)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
