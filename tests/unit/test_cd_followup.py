"""Numerical regressions and admission/scope boundaries for the C/D follow-up."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import pytest
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping

from agents.artifacts import REPO_ROOT
from agents.cd_followup_evidence import LOCK, validate, verify
from agents.cd_noise_diagnosis import corrected_power, linear, mjd2000, timed_noise
from agents.cd_reference import retry, retry_directory
from agents.optical_lead import (
    classification_patch,
    geocoding,
    radiometry,
    require_permitted_geometry,
    window_contains,
)
from agents.snap_reference import execute
from agents.source_reservation import checked
from agents.target_noise_audit import noise_power


def regression_xml(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal LUT neighbours from the real D coastal pixel; no class labels."""
    annotation = tmp_path / "annotation.xml"
    annotation.write_text(
        """<product><thermalNoiseCorrectionPerformed>false</thermalNoiseCorrectionPerformed>
    <imageInformation><productFirstLineUtcTime>2026-07-02T09:56:07.478336</productFirstLineUtcTime>
    <azimuthTimeInterval>1.501426931060695e-03</azimuthTimeInterval></imageInformation></product>"""
    )
    noise = tmp_path / "noise.xml"
    noise.write_text("""<noise><adsHeader><polarisation>HH</polarisation></adsHeader>
    <noiseRangeVectorList><noiseRangeVector><azimuthTime>2026-07-02T09:56:09.478237</azimuthTime><line>1332</line>
    <pixel>23835 23875</pixel><noiseRangeLut>518.0865 527.5394</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector><azimuthTime>2026-07-02T09:56:10.478187</azimuthTime><line>1998</line>
    <pixel>23835 23875</pixel><noiseRangeLut>533.9415 543.7017</noiseRangeLut>
    </noiseRangeVector></noiseRangeVectorList>
    <noiseAzimuthVectorList><noiseAzimuthVector><firstRangeSample>17355</firstRangeSample><lastRangeSample>25634</lastRangeSample>
    <firstAzimuthLine>0</firstAzimuthLine><lastAzimuthLine>16650</lastAzimuthLine>
    <line>1590 1600</line><noiseAzimuthLut>1.003073 1.002741</noiseAzimuthLut>
    </noiseAzimuthVector></noiseAzimuthVectorList></noise>""")
    return annotation, noise


def test_real_d_near_cancellation_reproduces_old_failure_and_snap_agreement(
    tmp_path: Path,
) -> None:
    annotation, noise = regression_xml(tmp_path)
    bounds = [23848, 1596, 1, 1]
    dn = np.array([[23]], dtype="uint16")
    old = noise_power(noise, bounds, "HH", snap_precision=True)
    timed = timed_noise(annotation, noise, bounds, "HH")
    # Recorded independent SNAP DIMAP output at original (23848,1596).
    snap_db = -85.72225791053182
    sigma_lut = 603.7622400000001
    old_db = 10 * np.log10((dn.astype("float64") ** 2 - old) / sigma_lut**2)
    new_db = 10 * np.log10(corrected_power(dn, timed) / sigma_lut**2)
    assert abs(old_db.item() - snap_db) > 0.01
    assert abs(new_db.item() - snap_db) < 0.000001
    assert old.item() == pytest.approx(528.9990267055427, abs=1e-10)
    assert timed.item() == pytest.approx(528.9990238735444, abs=1e-10)


def test_negative_only_floor_preserves_small_positive_signal() -> None:
    actual = corrected_power(np.ones(3), np.array([1.1, 1 - 1e-7, 1]))
    assert actual[0] == pytest.approx(1e-5)
    assert 0 < actual[1] < 1e-5
    assert actual[2] == 0


def test_mjd2000_and_piecewise_time_interpolation() -> None:
    assert mjd2000("2000-01-01T00:00:00") == 0
    assert mjd2000("2000-01-02T00:00:00.500000Z") == pytest.approx(1 + 0.5 / 86400)
    assert np.array_equal(linear([0, 2], [1, 5], [-1, 0, 1, 2, 3]), [-1, 1, 3, 5, 7])
    with pytest.raises(ValueError, match="ordered"):
        linear([0, 0], [1, 2], [0])


def test_noise_refuses_unknown_block_and_wrong_polarization(tmp_path: Path) -> None:
    annotation, noise = regression_xml(tmp_path)
    with pytest.raises(ValueError, match="coverage"):
        timed_noise(annotation, noise, [0, 0, 1, 1], "HH")
    with pytest.raises(ValueError, match="polarization"):
        timed_noise(annotation, noise, [23848, 1596, 1, 1], "HV")


def test_optical_protected_geometry_is_rejected() -> None:
    protected = {
        "geometry": mapping(box(-53, 48, -52, 49)),
        "quarantine": True,
        "split": "quarantine",
    }
    with pytest.raises(ValueError, match="protected"):
        require_permitted_geometry(
            mapping(box(-52.5, 48.5, -51.5, 49.5)), [protected], 10000
        )
    distance = require_permitted_geometry(
        mapping(box(-55, 46, -54.9, 46.1)), [protected], 10000
    )
    assert distance > 10000
    with pytest.raises(ValueError, match="protected"):
        require_permitted_geometry(mapping(box(-55, 46, -54.9, 46.1)), [], 10000)


def test_classification_patch_refuses_mismatched_target_without_reading_pixels(
    tmp_path: Path,
) -> None:
    mask_window = Window(100, 100, 20, 20)
    missing_asset = tmp_path / "classification.jp2"
    # A mismatched target/area would compute a centre outside the permitted
    # window; the patch must be refused before any file is opened, so a
    # nonexistent asset path proves no pixel read is attempted.
    with pytest.raises(ValueError, match="outside permitted window"):
        classification_patch(missing_asset, mask_window, (5.0, 5.0))
    assert not missing_asset.exists()


def test_classification_patch_reads_when_contained(tmp_path: Path) -> None:
    asset = tmp_path / "classification.jp2"
    with rasterio.open(
        asset,
        "w",
        driver="GTiff",
        height=140,
        width=140,
        count=3,
        dtype="uint8",
    ) as ds:
        ds.write(np.ones((3, 140, 140), dtype="uint8"))
    mask_window = Window(100, 100, 20, 20)
    patch = classification_patch(asset, mask_window, (108.0, 109.0))
    assert patch.shape == (3, 5, 5)


def test_window_contains_boundaries() -> None:
    outer = Window(100, 100, 20, 20)
    assert window_contains(outer, Window(105, 105, 5, 5))
    assert window_contains(outer, Window(100, 100, 20, 20))
    assert not window_contains(outer, Window(95, 105, 5, 5))
    assert not window_contains(outer, Window(115, 115, 10, 10))


def test_collection1_offsets_and_native_tile_geocoding(tmp_path: Path) -> None:
    xml = tmp_path / "tile.xml"
    xml.write_text("""<root><HORIZONTAL_CS_CODE>EPSG:32622</HORIZONTAL_CS_CODE>
    <Size resolution="10"><NROWS>10980</NROWS><NCOLS>10980</NCOLS></Size>
    <Geoposition resolution="10"><XDIM>10</XDIM><YDIM>-10</YDIM>
    <ULX>300000</ULX><ULY>5300040</ULY></Geoposition>
    <QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>
    <RADIO_ADD_OFFSET band_id="1">-1000</RADIO_ADD_OFFSET>
    <RADIO_ADD_OFFSET band_id="2">-1000</RADIO_ADD_OFFSET>
    <RADIO_ADD_OFFSET band_id="3">-1000</RADIO_ADD_OFFSET>
    <RADIO_ADD_OFFSET band_id="7">-1000</RADIO_ADD_OFFSET></root>""")
    transform, crs, shape = geocoding(xml, 10)
    assert transform * (1, 1) == (300010, 5300030)
    assert crs == "EPSG:32622" and shape == (10980, 10980)
    q, offsets = radiometry(xml)
    assert (2000 + offsets["B02"]) / q == 0.1


def test_retry_rejects_windows_path_overflow_before_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agents.cd_reference.sys.platform", "win32")
    payload = {
        "products": [
            {
                "name": "S1C_" + "a" * 70,
                "areas": [{"id": "area"}],
                "bands": [
                    {
                        "measurement_url": "https://example/Nodes("
                        + "b" * 80
                        + ".tiff)/$value"
                    }
                ],
            }
        ]
    }
    destination = tmp_path / ("x" * 100)
    with pytest.raises(ValueError, match="shorter"):
        retry_directory(destination, payload)
    assert not destination.exists()


def test_retry_headroom_and_second_retry_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared.json"
    prepared.write_text('{"products":[]}')
    monkeypatch.setattr("agents.cd_reference.verify", lambda _: {"prepared": {}})
    monkeypatch.setattr("agents.cd_reference.checked", lambda _: prepared)
    monkeypatch.setattr("agents.cd_reference.available_commit", lambda: 0)
    output = tmp_path / "untouched"
    with pytest.raises(ValueError, match="headroom"):
        retry(Path("unused"), output)
    assert not output.exists()
    monkeypatch.setattr("agents.cd_reference.verify", lambda _: {"retry_of": {}})
    # A non-empty prior receipt triggers the one-retry boundary before any I/O.
    monkeypatch.setattr(
        "agents.cd_reference.verify", lambda _: {"retry_of": {"path": "prior"}}
    )
    with pytest.raises(ValueError, match="one automatic retry"):
        retry(Path("unused"), output)


def test_timeout_stops_only_its_launcher_and_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agents.source_reservation.REPO_ROOT", tmp_path)
    script = (
        "import os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(os.getpid(),p.pid,flush=True); time.sleep(30)"
    )
    log = tmp_path / "timeout.log"
    result = execute(Path(sys.executable), ["-c", script], log, 2)
    assert result["timed_out"] and result["exit_code"] is None
    for pid in map(int, log.read_text().split()):
        if psutil.pid_exists(pid):
            assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert psutil.Process().is_running()


def test_public_followup_and_guard_tampering() -> None:
    assert verify()["labels_admitted"] == 0
    report = json.loads(checked(json.loads(LOCK.read_text())["report"]).read_text())
    changed = deepcopy(report)
    changed["training_approved"] = True
    with pytest.raises(ValueError, match="guard"):
        validate(changed)
    changed = deepcopy(report)
    changed["noise_diagnosis"]["bands"][0]["time_replay_max_absolute_db"] = 0.1
    with pytest.raises(ValueError, match="tolerance"):
        validate(changed)


def test_local_followup_replay() -> None:
    if not (REPO_ROOT / "data/cd_r1/report.json").exists():
        pytest.skip("Retained local native C/D and optical evidence required")
    assert verify(local=True)["verified"]
