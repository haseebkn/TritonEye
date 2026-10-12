"""Executable archive, coordinate and quarantined-release contracts."""

import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.control import GroundControlPoint
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from shapely.geometry import box

from agents import allenai_archive, allenai_subset
from agents.allenai_subset import (
    RESOLUTION,
    WORLD_M,
    archive_crosswalk,
    bounded_patch_center,
    publisher_xy,
    warp_publisher_window,
)
from agents.artifacts import sha256_file, write_json
from agents.nl_benchmark import load_dataset
from scripts import build_allenai_subset, publish_allenai_audit


class MemoryReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.requests = 0
        self.bytes = 0
        self.reads: list[tuple[int, int]] = []

    def read(self, offset: int, size: int) -> bytes:
        self.requests += 1
        self.bytes += size
        self.reads.append((offset, size))
        return self.payload[offset : offset + size]


def archive(name: str, *, symlink: bool = False, duplicate: bool = False) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for _ in range(2 if duplicate else 1):
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "elsewhere"
                tar.addfile(member)
            else:
                member.size = 100000
                tar.addfile(member, io.BytesIO(b"Z" * member.size))
    return stream.getvalue()


def test_index_skips_raster_payload_and_handles_long_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = archive("windows/train/100_200_1/layers/" + "a" * 120 + "/image.tif")
    monkeypatch.setattr(allenai_archive, "SIZE", len(payload))
    reader = MemoryReader(payload)
    result = allenai_archive.index(reader)
    assert result["windows"] == 1
    assert result["members"][0]["bytes"] == 100000
    assert result["raster_members_extracted"] == 0
    assert reader.bytes < 4096
    start = result["members"][0]["offset"]
    assert not any(start <= off < start + 100000 for off, _ in reader.reads)


@pytest.mark.parametrize(
    "name,symlink,duplicate,message",
    [
        ("../bad", False, False, "Unsafe"),
        ("/bad", False, False, "Unsafe"),
        ("windows/x", True, False, "unsupported"),
        ("windows/x", False, True, "Duplicate"),
    ],
)
def test_unsafe_archive_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    symlink: bool,
    duplicate: bool,
    message: str,
) -> None:
    payload = archive(name, symlink=symlink, duplicate=duplicate)
    monkeypatch.setattr(allenai_archive, "SIZE", len(payload))
    with pytest.raises(ValueError, match=message):
        allenai_archive.index(MemoryReader(payload))


class Response:
    def __init__(
        self,
        status: int = 206,
        *,
        etag: str = allenai_archive.ETAG,
        content_range: str = "bytes 0-65535/9589923840",
        size: int = 65536,
    ) -> None:
        self.status_code = status
        self.headers = {"ETag": etag, "Content-Range": content_range}
        self.raw = io.BytesIO(bytes(size))

    def raise_for_status(self) -> None:
        pass

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@pytest.mark.parametrize(
    "response,error",
    [
        (Response(200), "identity"),
        (Response(etag="changed"), "identity"),
        (Response(content_range="bytes 1-65536/9589923840"), "extent"),
        (Response(size=512), "Truncated"),
    ],
)
def test_range_identity_and_extent_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    response: Response,
    error: str,
) -> None:
    reader = allenai_archive.RangeReader()
    monkeypatch.setattr(reader.session, "get", lambda *a, **k: response)
    with pytest.raises(ValueError, match=error):
        reader.read(0, 512)


def test_range_cache_and_budget_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = allenai_archive.RangeReader()
    calls: list[dict[str, Any]] = []

    def get(*args: Any, **kwargs: Any) -> Response:
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(reader.session, "get", get)
    assert reader.read(0, 512) == bytes(512)
    assert reader.read(512, 512) == bytes(512)
    assert len(calls) == 1
    assert calls[0]["params"] == {"generation": allenai_archive.GENERATION}
    assert calls[0]["headers"]["If-Match"] == allenai_archive.ETAG
    monkeypatch.setattr(allenai_archive, "READ_BUDGET", 65536)
    with pytest.raises(ValueError, match="budget"):
        reader.read(70000, 512)
    assert len(calls) == 1


def test_grid_fraction_preserved_not_snapped() -> None:
    x, y = 1470000, 1440000
    affine = Affine(
        RESOLUTION,
        0,
        (x + 0.75) * RESOLUTION - WORLD_M / 2,
        0,
        -RESOLUTION,
        WORLD_M / 2 - (y + 0.25) * RESOLUTION,
    )
    assert publisher_xy(affine, (x, y), x + 123, y + 456) == pytest.approx(
        (affine.c + 123 * RESOLUTION, affine.f - 456 * RESOLUTION)
    )
    with pytest.raises(ValueError, match="origin"):
        publisher_xy(affine, (x + 1, y), x, y)
    with pytest.raises(ValueError, match="Finite"):
        publisher_xy(affine, (x, y), float("nan"), y)
    with pytest.raises(ValueError, match="north-up"):
        publisher_xy(Affine.identity(), (x, y), x, y)


def test_edge_context_moves_crop_not_label() -> None:
    assert bounded_patch_center(1652.05, 103.63, 25569, 16677) == (1652.05, 129)
    assert bounded_patch_center(500, 500, 1000, 1000) == (500, 500)
    assert bounded_patch_center(999, 999, 1000, 1000) == (871, 871)
    with pytest.raises(ValueError, match="outside"):
        bounded_patch_center(1000, 500, 1000, 1000)


def test_explicit_gcp_warp_has_pixels_not_blank_identity_transform() -> None:
    # A genuine GCP-only raster, like the raw Sentinel-1 measurements.
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=32, width=32, count=1, dtype="uint16"
        ) as writer:
            writer.write(np.full((32, 32), 125, dtype="uint16"), 1)
            writer.gcps = (
                [
                    GroundControlPoint(
                        row=r, col=c, x=-54 + c * 0.001, y=49 - r * 0.001
                    )
                    for r in (0, 16, 32)
                    for c in (0, 16, 32)
                ],
                rasterio.crs.CRS.from_epsg(4326),
            )
        with memory.open() as source:
            gcps, crs = source.gcps
            affine, w, h = calculate_default_transform(
                crs, 3857, 32, 32, gcps=gcps, resolution=RESOLUTION
            )
            pixels, valid = warp_publisher_window(
                source, affine, Window(w // 2, h // 2, 16, 16)
            )
            assert valid.all()
            assert (pixels == 125).all()


def test_crosswalk_exact_ids_and_coordinates(tmp_path: Path) -> None:
    database = tmp_path / "labels.sqlite3"
    with sqlite3.connect(database) as db:
        db.executescript(
            "CREATE TABLE images(id INTEGER,zoom INTEGER,projection TEXT,name TEXT);"
            "CREATE TABLE windows(id INTEGER,image_id INTEGER,dataset_id INTEGER,"
            "column INTEGER,row INTEGER,width INTEGER,height INTEGER,split TEXT);"
            "INSERT INTO images VALUES(1,13,'epsg:3857','scene');"
            "INSERT INTO windows VALUES"
            "(7,1,1,1470000,1440000,512,512,'original-train');"
        )
    index: dict[str, Any] = {
        "windows": 1,
        "members": [{"name": "windows/val/1470000_1440000_7/layers/tif"}],
    }
    result = archive_crosswalk(index, database, box(-55, 47, -51, 50))
    assert result["crosswalked_windows"] == 1
    assert result["nl_wholly_contained_windows"] == [7]
    assert result["original_database_splits"] == {"original-train": 1}
    assert result["annotation_payloads_checked"] is False
    index["members"][0]["name"] = "windows/val/1470001_1440000_7/layers/tif"
    with pytest.raises(ValueError, match="identity"):
        archive_crosswalk(index, database, box(-55, 47, -51, 50))


def test_release_loader_integrity_and_no_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(allenai_subset, "REPO_ROOT", tmp_path)
    root = tmp_path / "data/model_research/allenai_nl_v010/subset"
    artifact = root / "qa.json"
    write_json(artifact, {"unreviewed": True})
    local = root / "manifest.json"
    write_json(
        local,
        {
            "training_allowed": False,
            "evaluation_allowed": False,
            "artifacts": [
                {
                    "path": artifact.relative_to(tmp_path).as_posix(),
                    "sha256": sha256_file(artifact),
                }
            ],
        },
    )
    release = tmp_path / "public/release.json"
    public = release.with_name("manifest.json")
    write_json(public, {"prepared_points": 1})
    evidence = release.with_name("source_evidence.json")
    write_json(evidence, {"assets": []})
    write_json(
        release,
        {
            "version": "0.1.0",
            "status": "quarantined_qa",
            "manifest_sha256": sha256_file(public),
            "source_evidence_sha256": sha256_file(evidence),
            "local_manifest": {
                "path": local.relative_to(tmp_path).as_posix(),
                "sha256": sha256_file(local),
            },
        },
    )
    assert allenai_subset.load_alignment_subset(release)["training_allowed"] is False
    for purpose in ("training", "evaluation", "threshold_selection"):
        with pytest.raises(ValueError, match="blocked"):
            allenai_subset.load_alignment_subset(release, purpose=purpose)
    write_json(artifact, {"unreviewed": False})
    with pytest.raises(ValueError, match="artifact integrity"):
        allenai_subset.load_alignment_subset(release)
    write_json(local, {"training_allowed": True})
    with pytest.raises(ValueError, match="Local release integrity"):
        allenai_subset.load_alignment_subset(release)
    write_json(public, {"prepared_points": 99})
    with pytest.raises(ValueError, match="Public release integrity"):
        allenai_subset.load_alignment_subset(release)


def test_builder_rejects_locked_scene_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only committed lock metadata is read. No actual held-out image is used.
    dataset, _, _ = load_dataset()
    held = next(s for s in dataset["scenes"] if s["split"] == "test")
    audit = tmp_path / "audit"
    work = tmp_path / "work"
    audit.mkdir()
    with sqlite3.connect(audit / "metadata.sqlite3") as db:
        db.executescript(
            "CREATE TABLE images(name TEXT); CREATE TABLE labels(id INTEGER);"
        )
    inventory = audit / "sar_download_inventory_20261006.json"
    write_json(
        inventory,
        {
            "products": [
                {
                    "name": held["name"],
                    "product_id": held["product_id"],
                    "files": [
                        {"path": "must_not_access_pixels.tif", "sha256": "invalid"}
                    ],
                }
            ]
        },
    )
    write_json(
        work / "cc_archive_index.json",
        {
            "url": allenai_archive.URL,
            "generation": allenai_archive.GENERATION,
            "bytes": allenai_archive.SIZE,
            "etag": allenai_archive.ETAG,
        },
    )
    monkeypatch.setattr(build_allenai_subset, "ROOT", tmp_path)
    monkeypatch.setattr(build_allenai_subset, "AUDIT", audit)
    monkeypatch.setattr(build_allenai_subset, "WORK", work)
    monkeypatch.setattr(
        build_allenai_subset, "METADATA_SHA256", sha256_file(audit / "metadata.sqlite3")
    )
    monkeypatch.setattr(
        build_allenai_subset, "INVENTORY_SHA256", sha256_file(inventory)
    )
    monkeypatch.setattr(build_allenai_subset, "archive_crosswalk", lambda *args: {})
    monkeypatch.setattr(
        build_allenai_subset,
        "screen",
        lambda *args: {
            "cases": [
                {"source_name": held["name"], "geometry": held["rois"][0]["geometry"]}
            ]
        },
    )
    with pytest.raises(ValueError, match="Held-out test"):
        build_allenai_subset.build(work / "new_subset")
    assert not (work / "new_subset").exists()


def test_relative_output_normalized_and_released_versions_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_allenai_subset, "ROOT", tmp_path)
    monkeypatch.setattr(build_allenai_subset, "WORK", tmp_path / "work")
    target = build_allenai_subset.fresh_output(Path("work/subset"))
    assert target == tmp_path / "work/subset"
    assert (target / "windows/1/image.png").relative_to(
        tmp_path
    ).as_posix() == "work/subset/windows/1/image.png"
    with pytest.raises(ValueError, match="Fresh"):
        build_allenai_subset.fresh_output(Path("elsewhere"))
    write_json(tmp_path / "datasets/allenai_nl/v0.1.0/release.json", {})
    with pytest.raises(ValueError, match="immutable"):
        build_allenai_subset.fresh_output(Path("work/subset"))


def test_publisher_accepts_relative_manifest_without_partial_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Use the committed serialized public interface as a fixture template;
    # create every local source/artifact synthetically, without private data.
    public = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "datasets/allenai_nl/v0.1.0/manifest.json"
        ).read_text()
    )
    base = tmp_path / "data/model_research/allenai_nl_v010"
    target = tmp_path / "datasets/allenai_nl/v0.1.0"
    artifact = base / "subset/qa.json"
    write_json(artifact, {"fixture": True})
    data = dict(public)
    data.update(public["software"])
    data.update(
        {
            "prepared_windows": 1,
            "prepared_points": 1,
            "metadata_sha256": public["source_metadata_sha256"],
            "inventory_sha256": public["source_inventory_sha256"],
            "archive_index_sha256": public["cc_archive"]["index_sha256"],
            "cc_archive_crosswalk": public["cc_archive"]["crosswalk"],
            "whole_nl_candidate_windows": 73,
            "whole_nl_candidate_points": 39,
            "original_source_revision": "synthetic-fixture",
            "cases": [
                {
                    "publisher_previews": [{"valid_pixels": 1, "pixels": 1}],
                    "alignment_checks": [
                        {
                            "source_label_id": 1,
                            "visual_status": "unreviewed",
                            "grid_vs_native_tps_displacement_m": 0.5,
                            "point_px": [128, 128],
                            "crop_center_edge_clamped": False,
                            "patch": {"chip": {"sha256": sha256_file(artifact)}},
                        }
                    ],
                }
            ],
            "artifacts": [
                {
                    "path": artifact.relative_to(tmp_path).as_posix(),
                    "sha256": sha256_file(artifact),
                }
            ],
        }
    )
    local = base / "subset/manifest.json"
    write_json(local, data)
    index = dict(public["cc_archive"])
    index["members"] = []
    write_json(base / "cc_archive_index.json", index)
    write_json(base / "source_downloads.json", {"assets": []})
    for name in (
        "siv_README.md",
        "siv_warp.py",
        "siv_preprocess.py",
        "siv_dataset.py",
        "siv_download_imagery.py",
        "siv_sentinel1.py",
    ):
        write_json(base / "sources" / name, {"synthetic": True})
    monkeypatch.setattr(publish_allenai_audit, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(publish_allenai_audit, "BASE", base)
    monkeypatch.setattr(publish_allenai_audit, "PUBLIC", target)
    monkeypatch.chdir(tmp_path)
    publish_allenai_audit.publish(local.relative_to(tmp_path))
    descriptor = json.loads((target / "release.json").read_text())
    assert (
        descriptor["local_manifest"]["path"] == local.relative_to(tmp_path).as_posix()
    )
    assert descriptor["manifest_sha256"] == sha256_file(target / "manifest.json")
    assert descriptor["local_manifest"]["sha256"] == sha256_file(local)
    with pytest.raises(ValueError, match="immutable"):
        publish_allenai_audit.publish(local.relative_to(tmp_path))
