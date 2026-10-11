"""Executable screening, isolation and private-output contracts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from agents import overlap_audit as audit
from agents.acquisition import native_acquisition_group
from agents.landmask import LandMask


def product(index: int = 1, cog: bool = False) -> dict[str, Any]:
    return {
        "Id": str(UUID(int=index)),
        "Name": (
            "S1D_IW_GRDH_1SDV_20261002T213855_20261002T213920_004842_00919B_"
            + ("6BFD_COG.SAFE" if cog else "7162.SAFE")
        ),
        "ContentDate": {"Start": "2026-10-02T21:38:55Z", "End": "2026-10-02T21:39:20Z"},
        "GeoFootprint": mapping(box(-54, 48, -53, 49)),
    }


def report(
    time: str, lon: float = -53.5, identity: str = "123456789"
) -> dict[str, Any]:
    instant = datetime.fromisoformat(time).replace(tzinfo=timezone.utc)
    return {
        "time": instant.timestamp(),
        "timestamp": instant.isoformat(),
        "mmsi": identity,
        "lon": lon,
        "lat": 48.5,
        "values": {
            "mmsi": identity,
            "lon": lon,
            "lat": 48.5,
            "timestamp": instant.isoformat(),
            "speed_knots": 0,
            "course_deg": 0,
        },
        "source": {"file": "private.csv", "line": 2},
        "legacy_naive_utc": False,
    }


def screen(
    products: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    groups: set[str] | None = None,
    geography: Any = None,
    region: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    return audit.audit_products(
        products,
        sorted(rows, key=lambda r: r["time"]),
        region if region is not None else box(-54, 48, -53, 49),
        groups or set(),
        geography if geography is not None else box(-60, 56, -59, 57),
    )


def test_original_cog_once_and_same_day_slices_preserved() -> None:
    original, cog, adjacent = product(), product(2, True), product(3)
    adjacent["Name"] = (
        adjacent["Name"].replace("T213855", "T213920").replace("T213920_", "T213945_")
    )
    adjacent["ContentDate"] = {
        "Start": "2026-10-02T21:39:20Z",
        "End": "2026-10-02T21:39:45Z",
    }
    scenes, rejected = screen(
        [cog, original, adjacent], [report("2026-10-02T21:39:07")]
    )
    assert not rejected and len(scenes) == 2
    assert {s["product"]["Id"] for s in scenes} == {original["Id"], adjacent["Id"]}
    assert sum(len(s["product"].get("catalogue_aliases", [])) for s in scenes) == 2


@pytest.mark.parametrize(
    "footprint",
    [
        None,
        mapping(box(-54, 48, -53, 49).centroid),
        {"type": "Polygon", "coordinates": []},
    ],
)
def test_missing_or_non_polygon_footprint_never_scored(footprint: Any) -> None:
    p = {**product(), "GeoFootprint": footprint}
    scenes, rejected = screen([p], [report("2026-10-02T21:39:07")])
    assert not scenes
    assert rejected[0]["reason"] == "missing_or_invalid_footprint"


def test_locked_alias_group_and_buffered_geography_excluded() -> None:
    p = product(2, True)
    group = native_acquisition_group(p["Name"])
    assert group is not None
    assert screen([p], [], {group})[1][0]["reason"] == "locked_test_group_or_geography"
    assert (
        screen([p], [], geography=box(-53.6, 48.4, -53.4, 48.6))[1][0]["reason"]
        == "locked_test_group_or_geography"
    )


def test_out_of_swath_stale_or_impossible_tracks_do_not_become_candidates() -> None:
    outside, _ = screen([product()], [report("2026-10-02T21:39:07", -54.5)])
    assert outside[0]["status"] == "no_observed_overlap"
    stale, _ = screen([product()], [report("2026-10-02T21:30:00")])
    assert stale[0]["status"] == "sparse_support_only"
    impossible, _ = screen(
        [product()],
        [report("2026-10-02T21:39:00", -53.9), report("2026-10-02T21:39:15", -53.1)],
    )
    assert impossible[0]["aligned_in_swath"] == 0
    assert impossible[0]["alignment_statistics"]["rejected_tracks"] == 1
    assert not impossible[0]["ground_truth_eligible"]


def test_align_before_boundary_filter_and_retain_competitors() -> None:
    # Positions either side of the selected narrow region interpolate inside.
    rows = [
        report("2026-10-02T21:39:00", -53.501),
        report("2026-10-02T21:39:15", -53.499),
        report("2026-10-02T21:39:07", -53.5, "987654321"),
    ]
    scenes, _ = screen([product()], rows, region=box(-53.5005, 48, -53.4995, 49))
    assert scenes[0]["aligned_in_swath"] == 2
    assert scenes[0]["bracket_240s"] == 1
    assert scenes[0]["close_120s"] == 2


def test_snapshot_drops_partial_tails_and_normalization_counts(tmp_path: Path) -> None:
    source, target = tmp_path / "archive", tmp_path / "result/archive_snapshot"
    source.mkdir()
    complete = (
        "mmsi,lon,lat,timestamp,speed_knots,course_deg\n"
        "123456789,-53.5,48.5,2026-10-02T21:39:07,0,0\n"
        "123456789,-53.5,48.5,2026-10-02T21:39:07,0,0\n"
        "bad,-53.5,48.5,2026-10-02T21:39:07,0,0\n"
    )
    (source / "ais_stream_2026-10-02.csv").write_text(
        complete + "partial", encoding="utf-8"
    )
    manifest = audit.snapshot(source, target)
    assert manifest["files"][0]["incomplete_tail_bytes_excluded"] == 7
    rows, stats = audit.observations(target)
    assert len(rows) == 1 and stats["duplicate_reports_removed"] == 1
    assert stats["invalid_rows"] == 1 and stats["legacy_naive_utc_reports"] == 1
    with pytest.raises(FileExistsError):
        audit.snapshot(source, target)


def test_journal_gap_disconnected_and_missing_are_not_healthy(tmp_path: Path) -> None:
    centre = datetime(2026, 10, 2, 21, 39, 0, tzinfo=timezone.utc).timestamp()
    assert audit.journal_support(tmp_path, centre)["status"] == "gap_or_unknown"
    journal = tmp_path / "coverage_a.jsonl"

    def beats(left: str, right: str, connected: bool = True) -> None:
        journal.write_text(
            "\n".join(
                json.dumps({"event": "heartbeat", "time": t, "connected": connected})
                for t in (left, right)
            ),
            encoding="utf-8",
        )

    beats("2026-10-02T21:30:00Z", "2026-10-02T21:50:00Z")
    assert audit.journal_support(tmp_path, centre)["status"] == "gap_or_unknown"
    beats("2026-10-02T21:38:30Z", "2026-10-02T21:39:30Z", False)
    assert audit.journal_support(tmp_path, centre)["status"] == "gap_or_unknown"
    beats("2026-10-02T21:38:30Z", "2026-10-02T21:39:30Z")
    result = audit.journal_support(tmp_path, centre)
    assert result["status"] == "connected_bracket"
    assert not result["area_wide_reception_proven"]


def test_public_aggregate_and_blind_output_never_contain_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    image = tmp_path / "band.png"
    image.write_bytes(b"image-payload")
    scenes, _ = screen([product()], [report("2026-10-02T21:39:07")])
    public = audit.public_scene(scenes[0])
    assert "support" not in public and "mmsi" not in public
    area = {
        "id": "area_01",
        "product_id": product()["Id"],
        "acquisition_time": "2026-10-02T21:39:07Z",
        "polarizations": ["VV", "VH"],
        "previews": [{"path": "band.png"}] * 2,
        "chip": {"sha256": "chip-hash"},
        "support": scenes[0]["support"],
        "detector_predictions": [{"confidence": 0.99}],
    }
    page = audit.blind_html([area])
    assert "123456789" not in page and "-53.5" not in page and "0.99" not in page
    assert page.count('src="data:image/png;base64,') == 2
    assert "independent or unbiased" in page and "No labels" in page


def test_private_output_path_is_bounded_before_any_writes(tmp_path: Path) -> None:
    target = tmp_path / "would_leak"
    with pytest.raises(ValueError, match="ignored data/research"):
        audit.run(target, tmp_path, None, 0, False)
    assert not target.exists()


def test_physical_water_is_not_coastal_policy_or_reviewed_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use real LandMask classification with a simple reference coast at -53.5.
    project = Transformer.from_crs("EPSG:4326", "EPSG:32622", always_xy=True).transform
    mask = LandMask(
        [transform(project, box(-54, 48, -53.5, 49))],
        "EPSG:32622",
        "synthetic",
        "test fixture",
    )
    monkeypatch.setattr(LandMask, "for_footprint", lambda *a, **kw: mask)
    scenes, _ = screen([product()], [report("2026-10-02T21:39:07", -53.499)])
    audit.enrich(scenes[0], tmp_path, tmp_path, tmp_path)
    assert scenes[0]["physical_counts"] == {"water": 1}
    assert scenes[0]["policy_counts"] == {"coastal": 1}
    assert scenes[0]["close_water"] == 1
    assert scenes[0]["close_open_water"] == 0
    assert not scenes[0]["ground_truth_eligible"]


def test_replay_rejects_tampered_snapshot_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        audit, "test_exclusion", lambda _: (set(), box(-60, 56, -59, 57))
    )
    replay = tmp_path / "data/research/source"
    (replay / "archive_snapshot").mkdir(parents=True)
    (replay / "archive_snapshot/ais_stream_day.csv").write_text("tampered")
    (replay / "snapshot_manifest.json").write_text(
        json.dumps({"files": [{"file": "ais_stream_day.csv", "sha256": "incorrect"}]})
    )
    with pytest.raises(ValueError, match="Snapshot hash/path"):
        audit.run(tmp_path / "data/research/replay", tmp_path, replay, 0, False)


def test_bundle_preserves_scene_competitors_separate_from_blind_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_nl_benchmark as builder

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/build_nl_benchmark.py").write_text("synthetic builder fixture")
    p = product()
    source = raw / p["Name"]
    (source / "measurement").mkdir(parents=True)
    (source / "manifest.safe").write_text("test fixture")
    for pol in ("vv", "vh"):
        (source / "measurement" / f"source-{pol}-fixture.tiff").write_bytes(b"fixture")
    rows = [
        report("2026-10-02T21:39:07", -53.5),
        report("2026-10-02T21:39:07", -53.51, "987654321"),
    ]
    scenes, _ = screen([p], rows)
    for fix in scenes[0]["support"]:
        fix["physical_surface"] = "water"
    scenes[0]["local_exact_product"] = True
    monkeypatch.setattr(
        LandMask,
        "for_footprint",
        lambda *a, **kw: LandMask([], "EPSG:32622", "synthetic", "test fixture"),
    )

    def fake_chip(
        roi: dict[str, Any], paths: list[Path], directory: Path
    ) -> dict[str, Any]:
        directory.mkdir(parents=True)
        image = directory / "band.png"
        image.write_bytes(b"test image")
        return {
            **roi,
            "geometry": mapping(box(-53.502, 48.49, -53.498, 48.51)),
            "previews": [{"path": str(image.relative_to(tmp_path))}] * 2,
            "chip": {"sha256": "test-hash"},
            "cells": [],
            "valid_pixels": 262144,
        }

    monkeypatch.setattr(builder, "chip", fake_chip)
    output = tmp_path / "data/research/run/image_first"
    result = audit.bundle(scenes, output, raw, box(-60, 56, -59, 57), 1, False)
    assert result["complete_areas"] == 1 and not result["failures"]
    telemetry = json.loads(
        (output.parent / "telemetry_second_pass/scene_01_area_01.json").read_text()
    )
    assert len(telemetry["support"]) == 2
    assert {s["inside_review_area"] for s in telemetry["support"]} == {True, False}
    assert not telemetry["correct_identities_reviewed"]
    assert "987654321" not in (output / "review.html").read_text()
    assert not list(output.rglob("*telemetry*"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["annotation_complete"] and not manifest["ground_truth_eligible"]
    with pytest.raises(FileExistsError):
        audit.bundle(scenes, output, raw, box(-60, 56, -59, 57), 1, False)


@pytest.mark.parametrize("size", [None, 100_000_000])
def test_unknown_size_or_insufficient_disk_prevents_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int | None
) -> None:
    from scripts import build_nl_benchmark as builder

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/build_nl_benchmark.py").write_text("synthetic builder fixture")
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    scenes, _ = screen(
        [{**product(), "ContentLength": size}], [report("2026-10-02T21:39:07")]
    )
    monkeypatch.setattr(audit.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("Must not start a download without space/size evidence")

    monkeypatch.setattr(builder, "download_product", forbidden)
    output = tmp_path / "data/research/run/image_first"
    result = audit.bundle(scenes, output, raw, box(-60, 56, -59, 57), 1, True)
    assert result["complete_areas"] == 0 and len(result["failures"]) == 1
    assert "download" in result["failures"][0]["reason"]
