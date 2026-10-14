"""Verify native reference comparisons and conservative evidence accounting."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import box, mapping

from agents.iceberg_corroboration import (
    optical_query,
    optical_summary,
    query_optical,
    repeat_context,
)
from agents.snap_reference import bands, compare, execute, graph
from agents.source_reference_report import noise_floor_stats, summarize, verify
from agents.source_reservation import ref


def test_reference_compares_exact_pixels_and_order() -> None:
    dn = np.full((2, 384, 384), 100, dtype="uint16")
    dn[1] = 200
    db = np.full((2, 384, 384), -20, dtype="float32")
    db[1] = -26
    assert compare(dn, db, dn, db, 0.001)["passed"]
    with pytest.raises(ValueError, match="original DN"):
        compare(dn[::-1], db, dn, db, 0.001)
    changed = dn.copy()
    changed[0, 37, 29] += 1
    with pytest.raises(ValueError, match="original DN"):
        compare(changed, db, dn, db, 0.001)
    assert not compare(dn, db[::-1], dn, db, 0.001)["passed"]
    shifted = db.copy()
    shifted[1, 37, 29] += 0.01
    result = compare(dn, shifted, dn, db, 0.001)
    assert result["bands"][0]["passed"]
    assert not result["bands"][1]["passed"]
    shifted[1, 37, 29] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        compare(dn, shifted, dn, db, 0.001)
    with pytest.raises(ValueError, match="shape"):
        compare(dn[:, :75, :75], db[:, :75, :75], dn, db, 0.001)


def test_generated_graph_preserves_native_extent_and_noise_variant(
    tmp_path: Path,
) -> None:
    for variant in ("raw", "calibration", "noise"):
        root = ET.fromstring(
            graph(
                tmp_path / "a&b.safe",
                [1234, 6789, 384, 384],
                tmp_path / "out.dim",
                variant,
            )
        )
        nodes = {n.attrib["id"]: n for n in root.findall("node")}
        assert (
            nodes["read"].findtext("parameters/file")
            == (tmp_path / "a&b.safe").as_posix()
        )
        assert nodes["subset"].findtext("parameters/region") == "1234,6789,384,384"
        assert ("noise" in nodes) == (variant == "noise")
        assert ("calibrate" in nodes) == (variant != "raw")
        if "calibrate" in nodes:
            assert (
                nodes["calibrate"].findtext("parameters/selectedPolarisations")
                == "HH,HV"
            )
            assert (
                nodes["calibrate"].findtext("parameters/outputImageScaleInDb") == "true"
            )
    with pytest.raises(ValueError, match="native"):
        graph(tmp_path, [0, 0, 75, 75], tmp_path, "raw")


def observation() -> dict[str, Any]:
    return {
        "iceberg_id": "2018:1",
        "time": 10000,
        "lon": -53,
        "lat": 49,
        "method": "R/V",
        "source": "SHIP",
        "source_file": "iip.csv",
        "source_line": 2,
    }


def test_repeat_deduplication_and_identity_do_not_make_confirmation() -> None:
    selected = observation()
    primary = {**selected, "time": 10300, "source_line": 3}
    satellite = {**primary, "source": "S1A", "source_line": 4}
    other = {**primary, "iceberg_id": "2018:2", "source_line": 5}
    distant = {**primary, "time": 9999999}
    reports = repeat_context(
        selected,
        [selected, primary, deepcopy(primary), satellite, other, distant],
        10000,
        -53,
        49,
        86400,
    )
    assert len(reports) == 2
    assert sum(r["primary_method"] for r in reports) == 1
    assert all(not r["independent_confirmation"] for r in reports)
    far = {**primary, "lon": -52}
    assert not repeat_context(selected, [far], 10000, -53, 49, 86400)[0][
        "inside_original_gate"
    ]


def test_catalogue_discovery_is_not_optical_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agents.iceberg_corroboration as module

    # References remain project-scoped, as they do for real private receipts.
    monkeypatch.setattr("agents.source_reservation.REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    page = tmp_path / "page.json"
    product = {
        "Id": "scene",
        "Name": "S2_MSIL1C",
        "GeoFootprint": mapping(box(-54, 48, -52, 50)),
        "ContentDate": {"Start": "2018-05-04T10:00:00Z"},
        "Attributes": [{"Name": "cloudCover", "Value": 0.0}],
    }
    page.write_text(json.dumps({"value": [product]}))
    receipt = {"status": "complete", "response": ref(page)}
    result = optical_summary(receipt, -53, 49)
    assert result["covering_products"] == 1
    assert not result["object_confirmation"]
    assert not result["products"][0]["local_clear_sky_verified"]
    assert optical_summary(receipt, -60, 49)["covering_products"] == 0
    assert optical_summary({"status": "failed"}, -53, 49)["covering_products"] is None


def test_catalogue_failure_not_recorded_as_no_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setattr("agents.source_reservation.REPO_ROOT", tmp_path)

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise requests.Timeout()

    monkeypatch.setattr("agents.iceberg_corroboration.requests.get", failed)
    receipt = query_optical({}, tmp_path / "failure.json")
    assert receipt["status"] == "failed"
    assert optical_summary(receipt, -53, 49)["covering_products"] is None


def test_graph_process_failure_and_timeout_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("agents.source_reservation.REPO_ROOT", tmp_path)
    failure = execute(
        Path(sys.executable), ["-c", "raise SystemExit(3)"], tmp_path / "failed.log", 5
    )
    assert failure["exit_code"] == 3 and not failure["timed_out"]
    timeout = execute(
        Path(sys.executable),
        ["-c", "import time; time.sleep(10)"],
        tmp_path / "timeout.log",
        1,
    )
    assert timeout["exit_code"] is None and timeout["timed_out"]


def test_optical_query_is_time_bounded() -> None:
    from datetime import datetime, timezone

    params = optical_query(
        -53, 49, datetime(2020, 5, 23, 9, 41, tzinfo=timezone.utc), 3
    )
    query = params["$filter"]
    assert "2020-05-23T06:41:00Z" in query
    assert "2020-05-23T12:41:00Z" in query
    assert params["$top"] == "100"


def reports() -> tuple[dict[str, Any], dict[str, Any]]:
    from agents.snap_reference import POLICY

    policy = json.loads(POLICY.read_text())
    flags = dict.fromkeys(
        (
            "training_approved",
            "independent_review",
            "classifier_scored",
            "detector_scored",
        ),
        False,
    )
    snap: dict[str, Any] = {
        **flags,
        "compatibility_established": False,
        "policy": ref(POLICY),
        "reference_scope": "fixture",
        "areas": [],
    }
    evidence: dict[str, Any] = {
        **flags,
        "labels_modified": False,
        "policy": ref(POLICY),
        "areas": [],
        "ais_inventory": {},
        "limitations": [],
    }
    for area_id in policy["area_ids"]:
        snap["areas"].append(
            {
                "id": area_id,
                "product_id": area_id,
                "split": "train" if area_id.endswith("02") else "validation",
                "variants": {
                    v: {"exit_code": 1, "timed_out": False}
                    for v in ("raw", "calibration", "noise")
                },
            }
        )
        evidence["areas"].append(
            {
                "id": area_id,
                "repeat_reports": [],
                "optical": {"query_status": "complete", "covering_products": 0},
            }
        )
    return snap, evidence


def test_audit_failure_cannot_become_a_passing_reference() -> None:
    snap, evidence = reports()
    result = summarize(snap, evidence)
    assert not result["calibration_all_passed"]
    assert result["independently_confirmed"] == 0
    assert all(a["ignore_for_accuracy"] for a in result["areas"])
    snap["areas"][0]["comparison"] = {"passed": True}
    with pytest.raises(ValueError, match="Failed SNAP execution"):
        summarize(snap, evidence)


@pytest.mark.parametrize(
    "flag",
    [
        "training_approved",
        "independent_review",
        "classifier_scored",
        "detector_scored",
        "compatibility_established",
    ],
)
def test_report_does_not_approve_training_or_claim_compatibility(flag: str) -> None:
    snap, evidence = reports()
    snap[flag] = True
    with pytest.raises(ValueError):
        summarize(snap, evidence)


def test_partial_audit_is_not_complete() -> None:
    snap, evidence = reports()
    evidence["areas"].pop()
    with pytest.raises(ValueError, match="Complete four"):
        summarize(snap, evidence)


def test_numeric_agreement_does_not_conceal_elapsed_limit_overrun() -> None:
    snap, evidence = reports()
    for area in snap["areas"]:
        area["comparison"] = {"passed": True}
        for variant in area["variants"].values():
            variant.update(exit_code=0, elapsed_s=10.0)
    assert summarize(snap, evidence)["bounded_execution_all_passed"]
    snap["areas"][1]["variants"]["noise"]["elapsed_s"] = 1596.2
    result = summarize(snap, evidence)
    assert result["calibration_all_passed"]
    assert not result["bounded_execution_all_passed"]
    assert result["areas"][1]["snap_variants"]["noise"]["elapsed_limit_exceeded"]


def test_finite_noise_floor_values_are_not_hidden_as_valid_signal() -> None:
    result = noise_floor_stats(
        np.array([-90, -90.000001, -80, np.nan, -20]), np.full(5, -20), np.full(5, 10)
    )
    assert result["finite_pixels"] == 4
    assert result["pixels_near_reconstructed_power_1e_minus5"] == 2
    assert result["fraction_of_all_pixels"] == 0.4


def test_published_audit_integrity_and_provisional_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    verify(False)
    result = json.loads(capsys.readouterr().out)
    assert result["verified"]
    assert result["training_approved"] is False


def test_reference_band_reader_retains_nodata_as_missing(tmp_path: Path) -> None:
    import rasterio

    output = tmp_path / "noise.data"
    output.mkdir()
    values = np.full((384, 384), -25, dtype="float32")
    values[23, 49] = -9999
    for pol in ("HH", "HV"):
        with rasterio.open(
            output / f"Sigma0_{pol}_db.img",
            "w",
            driver="ENVI",
            width=384,
            height=384,
            count=1,
            dtype="float32",
            nodata=-9999,
        ) as dst:
            dst.write(values, 1)
    dimap(tmp_path / "noise.dim", "intensity_db", -9999)
    result = bands(tmp_path / "noise.dim")
    assert result.shape == (2, 384, 384)
    assert np.isnan(result[:, 23, 49]).all()
    assert (result[:, 0, 0] == -25).all()


def dimap(path: Path, unit: str, nodata: float) -> None:
    root = ET.Element("Dimap_Document")
    for pol in ("HH", "HV"):
        band = ET.SubElement(root, "Spectral_Band_Info")
        for key, value in {
            "BAND_NAME": f"Sigma0_{pol}_dB",
            "PHYSICAL_UNIT": unit,
            "SCALING_FACTOR": "1.0",
            "SCALING_OFFSET": "0.0",
            "LOG10_SCALED": "false",
            "NO_DATA_VALUE_USED": "true",
            "NO_DATA_VALUE": str(nodata),
        }.items():
            ET.SubElement(band, key).text = value
    path.write_text(ET.tostring(root, encoding="unicode"))


def test_linear_intensity_with_db_filename_uses_declared_unit(tmp_path: Path) -> None:
    import rasterio

    output = tmp_path / "calibration.data"
    output.mkdir()
    values = np.full((384, 384), 0.01, dtype="float32")
    values[23, 49] = 0
    for pol in ("HH", "HV"):
        with rasterio.open(
            output / f"Sigma0_{pol}_dB.img",
            "w",
            driver="ENVI",
            width=384,
            height=384,
            count=1,
            dtype="float32",
        ) as dst:
            dst.write(values, 1)
    dimap(tmp_path / "calibration.dim", "intensity", 0)
    result = bands(tmp_path / "calibration.dim")
    np.testing.assert_allclose(result[:, 0, 0], -20, atol=1e-6)
    assert np.isnan(result[:, 23, 49]).all()
    dimap(tmp_path / "calibration.dim", "unknown", 0)
    with pytest.raises(ValueError, match="physical unit"):
        bands(tmp_path / "calibration.dim")


def test_cached_installer_recovery_checks_source_and_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents.snap_image import recover

    monkeypatch.setattr("agents.snap_image.REPO_ROOT", tmp_path)
    monkeypatch.setattr("agents.source_reservation.REPO_ROOT", tmp_path)
    installer = b"test installer bytes"
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as content:
        info = tarfile.TarInfo("tmp/snap.sh")
        info.size = len(installer)
        content.addfile(info, io.BytesIO(installer))
    layer_bytes = layer.getvalue()
    layer_sha = hashlib.sha256(layer_bytes).hexdigest()
    config = json.dumps(
        {
            "os": "linux",
            "architecture": "amd64",
            "rootfs": {"diff_ids": ["sha256:" + layer_sha]},
        }
    ).encode()
    config_sha = hashlib.sha256(config).hexdigest()
    archive = tmp_path / "image.tar"
    with tarfile.open(archive, "w") as exported:
        for name, payload in ((config_sha, config), (layer_sha, layer_bytes)):
            member = tarfile.TarInfo("blobs/sha256/" + name)
            member.size = len(payload)
            exported.addfile(member, io.BytesIO(payload))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "image_id": "sha256:" + config_sha,
                "installer_sha256": hashlib.sha256(installer).hexdigest(),
                "checksum_url": "https://example.test/checksums",
            }
        )
    )
    monkeypatch.setattr("agents.snap_image.POLICY", policy)
    output = tmp_path / "data/tools/verified"
    recover(archive, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["installer_layer"] == "sha256:" + layer_sha
    assert (output / "esa-snap_sentinel_linux-14.0.0.sh").read_bytes() == installer
    changed = json.loads(policy.read_text())
    changed["installer_sha256"] = "0" * 64
    policy.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="installer checksum"):
        recover(archive, tmp_path / "data/tools/rejected")
    assert not (tmp_path / "data/tools/rejected/receipt.json").exists()
