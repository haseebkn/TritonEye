"""Exercise image-blind packets, sequential seals and conservative QA masks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box, mapping

from agents import allenai_curation as cur
from agents.allenai_subset import RESOLUTION, WORLD_M
from agents.artifacts import sha256_file


def region(label: str, bounds: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "class": label,
        "geometry": mapping(box(*bounds)),
        "reason": "manual image interpretation",
    }


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cur, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cur, "ROOT", tmp_path / "curation")
    monkeypatch.setattr(cur, "require_development_scene", lambda *args: None)
    release = tmp_path / "release.json"
    release.write_bytes(b"frozen original release")
    original = tmp_path / "source"
    original.mkdir()
    valid = np.ones((8, 8), dtype="uint8") * 255
    valid[0, :] = 0
    previews = []
    for band in ("vv", "vh"):
        image = original / f"{band}.png"
        mask = original / f"valid_{band}.png"
        Image.fromarray(np.arange(64, dtype="uint8").reshape(8, 8)).save(image)
        Image.fromarray(valid).save(mask)
        previews.append(
            {
                "band": band,
                **cur.record(tmp_path, image),
                "valid_mask": cur.record(tmp_path, mask),
            }
        )
    col, row = 1445000, 1456000
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(
        (col + 3) * RESOLUTION - WORLD_M / 2, WORLD_M / 2 - (row + 3) * RESOLUTION
    )
    source = {
        "sources": [{"name": "source_SAFE", "product_id": "pinned-id"}],
        "cases": [
            {
                "source_name": "source_SAFE",
                "source_window_id": 50191,
                "physical_geometry": mapping(box(-53, 47, -52.9, 47.1)),
                "global_pixel_window": [col, row, 8, 8],
                "publisher_previews": previews,
                "points": [{"source_label_id": 155812, "lon": lon, "lat": lat}],
            }
        ],
    }
    monkeypatch.setattr(cur, "load_alignment_subset", lambda path: source)
    workspace = tmp_path / "curation/workspace"
    cur.prepare(release, workspace)
    return workspace


def decisions(workspace: Path, *, vessel: bool = False) -> Path:
    path = workspace / "first_pass/decisions.json"
    data = json.loads(path.read_text())
    data.update(actor="developer", method="self", prior_label_exposure=True)
    data["cases"][0].update(
        full_valid_area_reviewed=True, regions=[region("background", (0, 0, 8, 8))]
    )
    if vessel:
        data["cases"][0]["regions"].append(region("vessel", (2, 2, 5, 5)))
    cur.save(path, data)
    return path


def test_packet_contains_only_neutral_whole_images(prepared: Path) -> None:
    packet = prepared / "first_pass"
    index = json.loads((packet / "index.json").read_text())
    assert set(index) == {"version", "cases"}
    assert set(index["cases"][0]) == {
        "id",
        "width",
        "height",
        "images",
        "valid_mask",
        "initial_ignore_mask",
    }
    assert {
        p.relative_to(packet).as_posix() for p in packet.rglob("*") if p.is_file()
    } == {
        "index.json",
        "decisions.json",
        "index.html",
        "area_001/vv.png",
        "area_001/vh.png",
        "area_001/valid.png",
        "area_001/unreviewed_ignore.png",
    }
    assert b"155812" not in (packet / "index.html").read_bytes()
    assert b"50191" not in (packet / "index.json").read_bytes()
    assert (
        json.loads((packet / "decisions.json").read_text())["cases"][0]["regions"] == []
    )


def test_default_ignore_and_explicit_class_precedence() -> None:
    valid = np.ones((8, 8), dtype=bool)
    valid[0, :] = False
    assert not cur.masks(valid, []).any()
    result = cur.masks(
        valid,
        [
            region("background", (0, 0, 8, 8)),
            region("vessel", (2, 2, 5, 5)),
            region("ignore", (3, 3, 4, 4)),
        ],
    )
    assert result[2, 2] == 1
    assert result[3, 3] == 0
    assert result[7, 7] == 2
    assert not result[0].any()
    assert not cur.masks(valid, [region("ignore", (0, 0, 8, 8))]).any()


def test_outputs_cannot_leak_into_first_pass_packet(prepared: Path) -> None:
    with pytest.raises(ValueError, match="nested"):
        cur.fresh(prepared / "first_pass/revealed")


def test_nontext_annotation_reason_rejected() -> None:
    item = region("background", (0, 0, 8, 8))
    item["reason"] = None
    with pytest.raises(ValueError, match="reason"):
        cur.masks(np.ones((8, 8)), [item])


@pytest.mark.parametrize(
    "label,bounds",
    [("bogus", (0, 0, 2, 2)), ("vessel", (-1, 0, 2, 2)), ("vessel", (0, 0, 9, 9))],
)
def test_bad_mask_regions_rejected(
    label: str, bounds: tuple[int, int, int, int]
) -> None:
    with pytest.raises(ValueError):
        cur.masks(np.ones((8, 8)), [region(label, bounds)])


def test_conflicting_objects_fail_closed() -> None:
    regions = [region("vessel", (2, 2, 5, 5)), region("fixed_structure", (2, 2, 5, 5))]
    with pytest.raises(ValueError, match="Conflicting"):
        cur.masks(np.ones((8, 8)), regions)
    assert not cur.masks(
        np.ones((8, 8)), regions + [region("ignore", (2, 2, 5, 5))]
    ).any()


def test_incomplete_first_pass_cannot_reveal(prepared: Path) -> None:
    with pytest.raises(ValueError, match="actor"):
        cur.freeze_first_pass(
            prepared, prepared / "first_pass/decisions.json", cur.ROOT / "frozen"
        )
    path = decisions(prepared)
    data = json.loads(path.read_text())
    data["cases"][0]["regions"] = [region("background", (0, 0, 2, 2))]
    cur.save(path, data)
    with pytest.raises(ValueError, match="Unreviewed"):
        cur.freeze_first_pass(prepared, path, cur.ROOT / "incomplete")
    with pytest.raises(FileNotFoundError):
        cur.reveal(prepared, cur.ROOT / "missing", cur.ROOT / "reveal")


def test_freeze_reveal_and_ignore_unresolved_points(prepared: Path) -> None:
    frozen = cur.ROOT / "frozen"
    cur.freeze_first_pass(prepared, decisions(prepared), frozen)
    revealed = cur.ROOT / "revealed"
    assert cur.reveal(prepared, frozen, revealed)["points"] == 1
    path = revealed / "point_decisions.json"
    data = json.loads(path.read_text())
    point = data["cases"][0]["points"][0]
    assert np.allclose(point["publisher_pixel"], [3, 3])
    point["reason"] = "class unclear"
    cur.save(path, data)
    with pytest.raises(ValueError, match="explicitly ignored"):
        cur.seal_second_pass(prepared, frozen, path, cur.ROOT / "rejected")
    data["cases"][0]["ignore_regions"] = [region("ignore", (1, 1, 6, 6))]
    cur.save(path, data)
    output = cur.ROOT / "second_frozen"
    seal = cur.seal_second_pass(prepared, frozen, path, output)
    assert seal["counts"]["unresolved_class"] == 1
    assert seal["training_allowed"] is False
    with Image.open(output / "area_001_ignore.png") as image:
        assert np.asarray(image)[3, 3] == 255


def test_provisional_usable_point_requires_first_pass_alignment(prepared: Path) -> None:
    frozen = cur.ROOT / "frozen"
    cur.freeze_first_pass(prepared, decisions(prepared, vessel=True), frozen)
    revealed = cur.ROOT / "revealed"
    cur.reveal(prepared, frozen, revealed)
    path = revealed / "point_decisions.json"
    data = json.loads(path.read_text())
    data["cases"][0]["points"][0].update(
        state="usable_provisional",
        reason="provisional image interpretation",
        first_pass_region_index=1,
        alignment_evidence="matches first-pass region without relocating point",
    )
    cur.save(path, data)
    assert (
        cur.seal_second_pass(prepared, frozen, path, cur.ROOT / "usable")["counts"][
            "usable_provisional"
        ]
        == 1
    )
    data["cases"][0]["points"][0]["published_lon"] += 0.01
    cur.save(path, data)
    with pytest.raises(ValueError, match="coordinates changed"):
        cur.seal_second_pass(prepared, frozen, path, cur.ROOT / "relocated")


@pytest.mark.parametrize(
    "purpose", ["training", "evaluation", "threshold_selection", "promotion"]
)
def test_adoption_exports_remain_blocked(prepared: Path, purpose: str) -> None:
    with pytest.raises(ValueError, match="quarantine"):
        cur.load_workspace(prepared, purpose=purpose)


def test_tamper_seal_and_overwrite_rejected(prepared: Path) -> None:
    frozen = cur.ROOT / "frozen"
    cur.freeze_first_pass(prepared, decisions(prepared), frozen)
    before = sha256_file(frozen / "decisions.json")
    with pytest.raises(ValueError, match="overwritten"):
        cur.freeze_first_pass(prepared, decisions(prepared), frozen)
    assert sha256_file(frozen / "decisions.json") == before
    (frozen / "decisions.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        cur.reveal(prepared, frozen, cur.ROOT / "reveal")


def test_packet_tamper_and_path_escape_rejected(prepared: Path) -> None:
    (prepared / "first_pass/area_001/vv.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        cur.load_workspace(prepared)
    with pytest.raises(ValueError, match="escapes"):
        cur.member(prepared, "../elsewhere")
    with pytest.raises(ValueError, match="separate"):
        cur.fresh(prepared.parent.parent / "datasets/nl_benchmark/new")


def test_held_out_scope_guard_precedes_copy(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def blocked(*args: Any) -> None:
        raise ValueError("Held-out test geographic group")

    monkeypatch.setattr(cur, "require_development_scene", blocked)
    with pytest.raises(ValueError, match="Held-out"):
        cur.prepare(prepared.parent.parent / "release.json", cur.ROOT / "blocked")
    assert not (cur.ROOT / "blocked").exists()


def test_review_boolean_cannot_be_flipped(prepared: Path) -> None:
    path = prepared / "workspace.json"
    data = json.loads(path.read_text())
    data["independent_review"] = True
    cur.save(path, data)
    with pytest.raises(ValueError, match="adoption"):
        cur.load_workspace(prepared)
