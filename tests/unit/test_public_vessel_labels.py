"""Published point screening contract, with no source-text assertions."""

import math
import sqlite3
import sys
from pathlib import Path

import pytest
from shapely.geometry import box

from agents.artifacts import sha256_file
from agents.public_vessel_labels import main, pixel_lonlat, screen

NAME = "S1A_IW_GRDH_1SDV_20220421T212233_20220421T212258_042877_051E4D_68C1.SAFE"


def make_database(path: Path) -> tuple[float, float]:
    world = 512 * 2**13
    column = (-53 + 180) / 360 * world
    row = (1 - math.asinh(math.tan(math.radians(48))) / math.pi) / 2 * world
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE collections(id INTEGER,name TEXT);"
            "CREATE TABLE datasets(id INTEGER,collection_id INTEGER,"
            "name TEXT,task TEXT);"
            "CREATE TABLE images(id INTEGER,name TEXT,uuid TEXT,zoom INTEGER,"
            "projection TEXT,hidden INTEGER);"
            "CREATE TABLE windows(id INTEGER,dataset_id INTEGER,image_id INTEGER,"
            "column REAL,row REAL,width INTEGER,height INTEGER,"
            "hidden INTEGER,split TEXT);"
            "CREATE TABLE labels(id INTEGER,window_id INTEGER,column REAL,"
            "row REAL,properties TEXT);"
            "INSERT INTO collections VALUES(1,'sentinel1');"
            "INSERT INTO datasets VALUES(1,1,'vessels','point');"
        )
        connection.execute(
            "INSERT INTO images VALUES(1,?,'source-id',13,'epsg:3857',0)", (NAME,)
        )
        for identifier, split, x in [
            (1, "train", column),
            (2, "val", column + 1024),
            (3, "train", column - 1000000),
        ]:
            connection.execute(
                "INSERT INTO windows VALUES(?,1,1,?,?,512,512,0,?)",
                (identifier, x, row, split),
            )
        connection.execute(
            "INSERT INTO labels VALUES(1,1,?,?,'{}')", (column + 10, row + 20)
        )
        connection.execute(
            "INSERT INTO labels VALUES(2,2,?,?,'{\"OnKey\":true}')",
            (column + 1040, row + 20),
        )
    return column, row


def test_global_pixels_not_local_offsets() -> None:
    assert pixel_lonlat(2097152, 2097152, 13) == pytest.approx((0, 0))
    with pytest.raises(ValueError, match="zoom-13"):
        pixel_lonlat(2097152, 2097152, 12)
    with pytest.raises(ValueError, match="outside global"):
        pixel_lonlat(-1, 0, 13)
    with pytest.raises(ValueError, match="Finite"):
        pixel_lonlat(math.nan, 0, 13)


def test_screen_keeps_candidates_unapproved_and_detects_split_leak(
    tmp_path: Path,
) -> None:
    database = tmp_path / "metadata.sqlite3"
    make_database(database)
    original = sha256_file(database)
    result = screen(database, box(-55, 46, -50, 50))
    assert sha256_file(database) == original
    assert result["windows_wholly_inside_nl"] == 2
    assert result["published_vessel_points"] == 1
    assert result["empty_windows_not_confirmed_negatives"] == 1
    assert result["excluded"] == {
        "not_wholly_inside_nl": 1,
        "annotation_helper_point": 1,
    }
    assert len(result["acquisitions_with_multiple_upstream_splits"]) == 1
    assert result["adopted_as_training_data"] is False
    assert result["threshold_selection_allowed"] is False
    assert result["upstream_splits_adopted"] is False
    assert all(c["metric_ready"] is False for c in result["cases"])
    assert result["cases"][0]["points"][0]["metric_ready"] is False
    assert result["cases"][1]["empty_window_is_confirmed_negative"] is False
    assert result["cases"][0]["points"][0]["lon"] == pytest.approx(-52.9991417)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("UPDATE images SET projection='epsg:4326'", "projection"),
        ("UPDATE windows SET width=0 WHERE id=1", "dimensions"),
        ("UPDATE labels SET column=0 WHERE id=1", "outside its selected window"),
        ("UPDATE images SET name='unresolved'", "acquisition"),
        ("UPDATE datasets SET task='box'", "vessel-point"),
    ],
)
def test_invalid_source_contracts_fail_closed(
    tmp_path: Path, mutation: str, error: str
) -> None:
    database = tmp_path / "metadata.sqlite3"
    make_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(mutation)
    with pytest.raises(ValueError, match=error):
        screen(database, box(-55, 46, -50, 50))


def test_hidden_windows_not_adopted(tmp_path: Path) -> None:
    database = tmp_path / "metadata.sqlite3"
    make_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE images SET hidden=1")
    result = screen(database, box(-55, 46, -50, 50))
    assert result["cases"] == []
    assert result["excluded"] == {"hidden": 3}


def test_cli_rejects_unpinned_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "metadata.sqlite3"
    make_database(database)
    output = Path.cwd() / "data/model_research/test_unpublished_screen.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["public_vessel_labels", "--database", str(database), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="pinned published"):
        main()
    assert not output.exists()
