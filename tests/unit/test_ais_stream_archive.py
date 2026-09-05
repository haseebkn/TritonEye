import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.ingest.ingest_agent import (
    archive_coverage_details,
    filter_ais_stream_archive,
)

NORMALIZED_HEADER = "mmsi,lat,lon,timestamp,speed_knots,course_deg\n"


def write_day_file(archive_dir: str, date_str: str, rows: str) -> None:
    path = os.path.join(archive_dir, f"ais_stream_{date_str}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(NORMALIZED_HEADER)
        f.write(rows)


@pytest.fixture
def archive_dir(tmp_path: Path) -> str:
    return str(tmp_path)


def test_matches_records_within_window_and_bbox(archive_dir: str) -> None:
    write_day_file(
        archive_dir,
        "2026-08-18",
        # inside window/bbox, then out of bbox, then out of window
        "316000001,47.50,-52.70,2026-08-18 09:12:00,10.0,90.0\n"
        "316000002,10.00,10.00,2026-08-18 09:12:00,10.0,90.0\n"
        "316000003,47.50,-52.70,2026-08-18 10:00:00,10.0,90.0\n",
    )
    out_path = os.path.join(archive_dir, "out.csv")
    ok = filter_ais_stream_archive(
        archive_dir, "2026-08-18 09:15:00", [-53.0, 47.0, -52.0, 48.0], out_path
    )
    assert ok
    with open(out_path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 2  # header + one matching record
    assert "316000001" in lines[1]


def test_no_archive_directory_returns_false(tmp_path: Path) -> None:
    missing = str(tmp_path / "does_not_exist")
    ok = filter_ais_stream_archive(
        missing,
        "2026-08-18 09:15:00",
        [-53.0, 47.0, -52.0, 48.0],
        str(tmp_path / "out.csv"),
    )
    assert not ok


def test_no_matching_records_returns_false(archive_dir: str) -> None:
    write_day_file(
        archive_dir,
        "2026-08-18",
        "316000001,0.0,0.0,2026-08-18 09:12:00,10.0,90.0\n",
    )
    out_path = os.path.join(archive_dir, "out.csv")
    ok = filter_ais_stream_archive(
        archive_dir, "2026-08-18 09:15:00", [-53.0, 47.0, -52.0, 48.0], out_path
    )
    assert not ok


def test_window_straddling_midnight_checks_both_daily_files(archive_dir: str) -> None:
    # Acquisition at 00:02, window is 23:57 (previous day) .. 00:07 (same day).
    write_day_file(
        archive_dir,
        "2026-08-17",
        "316000001,47.50,-52.70,2026-08-17 23:58:00,10.0,90.0\n",
    )
    write_day_file(
        archive_dir,
        "2026-08-18",
        "316000002,47.50,-52.70,2026-08-18 00:03:00,10.0,90.0\n",
    )
    out_path = os.path.join(archive_dir, "out.csv")
    ok = filter_ais_stream_archive(
        archive_dir, "2026-08-18 00:02:00", [-53.0, 47.0, -52.0, 48.0], out_path
    )
    assert ok
    with open(out_path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3  # header + one record from each day file
    mmsis = {line.split(",")[0] for line in lines[1:]}
    assert mmsis == {"316000001", "316000002"}


def test_rerunning_stream_filter_does_not_duplicate_records(archive_dir: str) -> None:
    # _filter_ais_file appends so a window spanning several daily files
    # accumulates into one output. Re-ingesting the same acquisition must
    # still produce the same record count, not double it.
    write_day_file(
        archive_dir,
        "2026-08-18",
        "316000001,47.50,-52.70,2026-08-18 09:12:00,10.0,90.0\n",
    )
    out_path = os.path.join(archive_dir, "out.csv")
    args = (archive_dir, "2026-08-18 09:15:00", [-53.0, 47.0, -52.0, 48.0], out_path)

    assert filter_ais_stream_archive(*args)
    first = open(out_path, encoding="utf-8").read()
    assert filter_ais_stream_archive(*args)
    assert open(out_path, encoding="utf-8").read() == first


def test_invalid_rows_dropped_and_unavailable_velocity_preserved(
    archive_dir: str,
) -> None:
    import pandas as pd

    write_day_file(
        archive_dir,
        "2026-08-18",
        "316000001,47.5,-52.7,2026-08-18T06:42:00-02:30,102.3,360\n"
        "broken,47.5,-52.7,2026-08-18T09:12:00Z,10,90\n"
        "316000002,91,-52.7,2026-08-18T09:12:00Z,10,90\n"
        "316000003,47.5,-52.7,not-a-time,10,90\n"
        "316000004,47.5,181,2026-08-18T09:12:00Z,10,90\n",
    )
    output = os.path.join(archive_dir, "out.csv")
    assert filter_ais_stream_archive(
        archive_dir, "2026-08-18T09:15:00Z", [-180, -90, 180, 90], output
    )
    rows = pd.read_csv(output)
    assert len(rows) == 1
    assert rows.iloc[0]["timestamp"] == "2026-08-18T09:12:00Z"
    assert pd.isna(rows.iloc[0]["speed_knots"])
    assert pd.isna(rows.iloc[0]["course_deg"])
    assert rows.iloc[0]["timestamp_basis"] == "legacy_archive_utc_assumed"


def test_rerun_with_missing_archive_clears_stale_derived_rows(tmp_path: Path) -> None:
    output = tmp_path / "filtered.csv"
    output.write_text(
        NORMALIZED_HEADER + "316000001,47.5,-52.7,2026-08-18T09:12:00Z,10,90\n"
    )
    assert not filter_ais_stream_archive(
        str(tmp_path / "missing"),
        "2026-08-18T09:15:00Z",
        [-53, 47, -52, 48],
        str(output),
    )
    assert len(output.read_text().splitlines()) == 1


def test_single_observation_does_not_establish_complete_coverage(
    archive_dir: str,
) -> None:
    write_day_file(
        archive_dir, "2026-08-18", "316000001,47.5,-52.7,2026-08-18T09:12:00Z,10,90\n"
    )
    output = os.path.join(archive_dir, "filtered.csv")
    assert filter_ais_stream_archive(
        archive_dir, "2026-08-18T09:15:00Z", [-53, 47, -52, 48], output
    )
    coverage = archive_coverage_details(
        archive_dir, "2026-08-18T09:15:00Z", [-53, 47, -52, 48], output
    )
    assert coverage["status"] == "partial"
    assert coverage["coverage_known"] is False
    assert coverage["observations"] == 1


def test_filter_rejects_source_overwrite(archive_dir: str) -> None:
    write_day_file(
        archive_dir, "2026-08-18", "316000001,47.5,-52.7,2026-08-18T09:12:00Z,10,90\n"
    )
    with pytest.raises(ValueError, match="source AIS archive"):
        filter_ais_stream_archive(
            archive_dir,
            "2026-08-18T09:15:00Z",
            [-53, 47, -52, 48],
            os.path.join(archive_dir, "ais_stream_2026-08-18.csv"),
        )
