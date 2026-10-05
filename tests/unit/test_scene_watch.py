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
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.scene_watch import (  # noqa: E402
    REQUIRED_POLARIZATION,
    ais_observations_near,
    assess,
    summarize,
)


def product(name: str, start: str) -> Dict[str, Any]:
    return {
        "Id": str(uuid5(NAMESPACE_URL, name + start)),
        "Name": name,
        "ContentDate": {"Start": start},
    }


def covering(lon: float, lat: float, half: float = 0.5) -> Dict[str, Any]:
    """A GeoJSON swath containing (lon, lat), for tests about other gates."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - half, lat - half],
                [lon + half, lat - half],
                [lon + half, lat + half],
                [lon - half, lat + half],
                [lon - half, lat - half],
            ]
        ],
    }


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
    vessels, and no position met open-water eligibility. The pipeline scores only water-
    classified detections, so those vessels would have produced a recall of
    zero by construction -- a meaningless measurement that looked like a real
    one. AIS existing is not the same as AIS being usable.
    """
    import agents.scene_watch as sw

    archive = write_archive(
        tmp_path, "2026-09-03", [obs("2026-09-03 21:30:00", -52.70, 47.56)]
    )
    monkeypatch.setattr(sw, "water_eligible", lambda positions, cache_dir: 0)
    p = product(VV, "2026-09-03T21:30:23Z")
    p["GeoFootprint"] = covering(-52.7, 47.56)
    rows = sw.assess([p], archive, str(tmp_path))
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
    p = product(VV, "2026-09-03T21:30:23Z")
    p["GeoFootprint"] = covering(-51.0, 47.0)
    rows = sw.assess([p], archive, str(tmp_path))
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


SWATH = {
    "type": "Polygon",
    "coordinates": [
        [[-57.0, 47.8], [-53.2, 47.8], [-53.2, 49.7], [-57.0, 49.7], [-57.0, 47.8]]
    ],
}


def test_ais_outside_the_swath_is_not_scorable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """
    Regression for a real false positive.

    The recording envelope spans most of Atlantic Canada, so the time window
    almost always contains vessels somewhere. On 2026-09-27 this reported a
    scene scorable from 85 observations that were all outside its footprint --
    near St. John's and the Strait of Belle Isle while the swath covered the
    west coast. Two downloads and two pipeline runs could only return
    `scored: False`.
    """
    import agents.scene_watch as sw

    # St. John's: inside the time window, outside the swath.
    archive = write_archive(
        tmp_path, "2026-09-27", [obs("2026-09-27 21:30:00", -52.69, 47.56)]
    )
    monkeypatch.setattr(
        sw, "water_eligible", lambda positions, cache_dir: len(positions)
    )
    p = product(VV, "2026-09-27T21:30:23Z")
    p["GeoFootprint"] = SWATH
    rows = sw.assess([p], archive, str(tmp_path))
    assert rows[0]["scorable"] is False
    assert rows[0]["ais_observations"] == 1
    assert "inside the imaged swath" in rows[0]["reason"]


def test_ais_inside_the_swath_counts(tmp_path: Path, monkeypatch: Any) -> None:
    import agents.scene_watch as sw

    archive = write_archive(
        tmp_path, "2026-09-27", [obs("2026-09-27 21:30:00", -55.0, 48.5)]
    )
    monkeypatch.setattr(
        sw, "water_eligible", lambda positions, cache_dir: len(positions)
    )
    p = product(VV, "2026-09-27T21:30:23Z")
    p["GeoFootprint"] = SWATH
    rows = sw.assess([p], archive, str(tmp_path))
    assert rows[0]["scorable"] is True


def test_missing_footprint_is_never_scorable(tmp_path: Path, monkeypatch: Any) -> None:
    """
    An unknown swath must not default to "everything inside": a false
    SCORABLE costs a ~1.7 GB download and a full pipeline run.
    """
    import agents.scene_watch as sw

    archive = write_archive(
        tmp_path, "2026-09-27", [obs("2026-09-27 21:30:00", -55.0, 48.5)]
    )
    monkeypatch.setattr(
        sw, "water_eligible", lambda positions, cache_dir: len(positions)
    )
    rows = sw.assess([product(VV, "2026-09-27T21:30:23Z")], archive, str(tmp_path))
    assert rows[0]["scorable"] is False


def test_within_footprint_rejects_malformed_geometry() -> None:
    from agents.scene_watch import within_footprint

    assert within_footprint([(-55.0, 48.5)], None) == []
    assert within_footprint([(-55.0, 48.5)], {"type": "Polygon"}) == []
    assert within_footprint([], SWATH) == []
    assert within_footprint([(-55.0, 48.5), (-52.69, 47.56)], SWATH) == [(-55.0, 48.5)]
    assert (
        within_footprint(
            [(-55.0, 48.5)], {"type": "Point", "coordinates": [-55.0, 48.5]}
        )
        == []
    )
    assert (
        within_footprint(
            [(-55.0, 48.5)],
            {
                "type": "Polygon",
                "coordinates": [
                    [[-56, 48], [-54, 49], [-56, 49], [-54, 48], [-56, 48]]
                ],
            },
        )
        == []
    )


def test_catalogue_preserves_same_prefix_products_and_reads_next_page(
    monkeypatch: Any,
) -> None:
    from types import SimpleNamespace

    import requests

    import agents.scene_watch as sw

    first = product(VV, "2026-09-03T21:30:23Z")
    second = {**first, "Id": "66d3167d-240a-459a-8066-75b9d2a458f2"}
    next_link = sw.CATALOGUE + "?$skip=200"
    pages = [
        {"value": [first], "@odata.nextLink": next_link},
        {"value": [first, second]},
    ]
    calls: list[str] = []

    def get(url: str, **kwargs: Any) -> SimpleNamespace:
        calls.append(url)
        page = pages.pop(0)
        return SimpleNamespace(status_code=200, json=lambda: page)

    monkeypatch.setattr(
        sw, "_footprint_wkt", lambda _: "POLYGON((-51 47,-50 47,-50 48,-51 48,-51 47))"
    )
    monkeypatch.setattr(requests, "get", get)
    products = sw.search_acquisitions("unused", token="test")
    assert {row["Id"] for row in products} == {first["Id"], second["Id"]}
    assert calls == [sw.CATALOGUE, next_link]
