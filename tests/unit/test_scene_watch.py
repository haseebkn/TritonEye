"""
Scene-watch tests.

The point of this module is to refuse scenes that cannot yield a measurement,
so the tests concentrate on the three ways a scene looks usable and is not.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.scene_watch import (  # noqa: E402
    REQUIRED_POLARIZATION,
    ais_observations_near,
    assess,
    summarize,
)


def product(name: str, start: str) -> Dict[str, Any]:
    return {"Name": name, "ContentDate": {"Start": start}}


VV = "S1D_IW_GRDH_1SDV_20260903T213023_20260903T213048_004420_0082A1_1234"
HH = "S1C_IW_GRDH_1SDH_20260915T093243_20260915T093308_009300_0130AA_5678"


def write_archive(tmp: Path, day: str, rows: List[Dict[str, str]]) -> str:
    d = tmp / "ais_stream"
    d.mkdir(exist_ok=True)
    with open(d / f"ais_stream_{day}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, ["mmsi", "lat", "lon", "timestamp", "speed_knots", "course_deg"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(d)


def obs(ts: str, lon: float = -52.0, lat: float = 47.0) -> Dict[str, str]:
    return {
        "mmsi": "316000001",
        "lat": str(lat),
        "lon": str(lon),
        "timestamp": ts,
        "speed_knots": "8.0",
        "course_deg": "90.0",
    }


def test_hh_hv_is_never_scorable(tmp_path: Path) -> None:
    """
    The detector requires VV/VH. Over parts of this region HH/HV is all that is
    acquired, so this is the common rejection, not an edge case.
    """
    archive = write_archive(tmp_path, "2026-09-15", [obs("2026-09-15 09:32:43")])
    rows = assess([product(HH, "2026-09-15T09:32:43Z")], archive, str(tmp_path))
    assert rows[0]["scorable"] is False
    assert "VV/VH" in rows[0]["reason"]
    assert rows[0]["polarization"] == "HH/HV"


def test_vv_vh_without_ais_is_not_scorable(tmp_path: Path) -> None:
    archive = write_archive(tmp_path, "2026-09-03", [])
    rows = assess([product(VV, "2026-09-03T21:30:23Z")], archive, str(tmp_path))
    assert rows[0]["scorable"] is False
    assert rows[0]["ais_observations"] == 0
    assert "no recorded AIS" in rows[0]["reason"]


def test_ais_outside_the_window_does_not_count(tmp_path: Path) -> None:
    """20 minutes away is not coverage; the pipeline matches within +/-5 min."""
    archive = write_archive(
        tmp_path, "2026-09-03", [obs("2026-09-03 21:50:00"), obs("2026-09-03 21:10:00")]
    )
    assert ais_observations_near(archive, "2026-09-03T21:30:23Z") == []


def test_ais_inside_the_window_counts(tmp_path: Path) -> None:
    archive = write_archive(
        tmp_path, "2026-09-03", [obs("2026-09-03 21:28:00"), obs("2026-09-03 21:33:00")]
    )
    found = ais_observations_near(archive, "2026-09-03T21:30:23Z")
    assert len(found) == 2


def test_window_straddling_midnight_reads_both_days(tmp_path: Path) -> None:
    """
    An acquisition at 23:57 has half its window in the next day's file.

    Reading only the acquisition date would silently halve the coverage count
    and could make a scorable scene look unscorable.
    """
    write_archive(tmp_path, "2026-09-03", [obs("2026-09-03 23:58:00")])
    archive = write_archive(tmp_path, "2026-09-04", [obs("2026-09-04 00:01:00")])
    found = ais_observations_near(archive, "2026-09-03T23:59:00Z")
    assert len(found) == 2


def test_ais_present_but_all_on_land_is_not_scorable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """
    Regression for a real near-miss.

    A 2026-09-03 acquisition over St. John's had 75 observations from 18
    vessels, and every position fell inside the OSM coastline because the
    polygon encloses the harbour basin. The pipeline scores only water-
    classified detections, so those vessels would have produced a recall of
    zero by construction -- a meaningless measurement that looked like a real
    one. AIS existing is not the same as AIS being usable.
    """
    import agents.scene_watch as sw

    archive = write_archive(
        tmp_path, "2026-09-03", [obs("2026-09-03 21:30:00", -52.70, 47.56)]
    )
    monkeypatch.setattr(sw, "water_eligible", lambda positions, cache_dir: 0)
    rows = sw.assess([product(VV, "2026-09-03T21:30:23Z")], archive, str(tmp_path))
    assert rows[0]["ais_observations"] == 1
    assert rows[0]["ais_in_water"] == 0
    assert rows[0]["scorable"] is False
    assert "alert-eligible water" in rows[0]["reason"]


def test_ais_in_open_water_is_scorable(tmp_path: Path, monkeypatch: Any) -> None:
    import agents.scene_watch as sw

    archive = write_archive(
        tmp_path, "2026-09-03", [obs("2026-09-03 21:30:00", -51.0, 47.0)]
    )
    monkeypatch.setattr(sw, "water_eligible", lambda positions, cache_dir: 1)
    rows = sw.assess([product(VV, "2026-09-03T21:30:23Z")], archive, str(tmp_path))
    assert rows[0]["scorable"] is True
    assert "SCORABLE" in rows[0]["reason"]


def test_missing_coastline_does_not_claim_coverage(tmp_path: Path) -> None:
    """
    Without the coastline download, eligibility is unknowable. It must read as
    zero rather than defaulting to "eligible" and overstating what is scorable.
    """
    from agents.scene_watch import water_eligible

    assert water_eligible([(-52.0, 47.0)], str(tmp_path / "absent")) == 0
    assert water_eligible([], str(tmp_path)) == 0


def test_summarize_counts_each_category() -> None:
    rows = [
        {"polarization": "VV/VH", "scorable": True},
        {"polarization": "VV/VH", "scorable": False},
        {"polarization": "HH/HV", "scorable": False},
    ]
    assert summarize(rows) == {"total": 3, "vv_vh": 2, "hh_hv": 1, "scorable": 1}


def test_required_polarization_matches_the_detector() -> None:
    """1SDV is the product-name token for VV/VH; the detector accepts nothing else."""
    assert REQUIRED_POLARIZATION == "1SDV"
    assert REQUIRED_POLARIZATION in VV
    assert REQUIRED_POLARIZATION not in HH
