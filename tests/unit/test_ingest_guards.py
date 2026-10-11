"""Production ingest must preserve source/time/region constraints."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from agents.ingest import ingest_agent as ingest


def test_missing_credentials_does_not_generate_synthetic_data(monkeypatch: Any) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    for key in (
        "COPERNICUS_USER",
        "COPERNICUS_PASS",
        "MOCK_INGEST",
        "TARGET_DATE",
        "AOI_NAME",
    ):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Production must not invoke synthetic ingestion")

    monkeypatch.setattr(ingest, "generate_synthetic_data", forbidden)
    with pytest.raises(SystemExit) as error:
        ingest.main()
    assert error.value.code == 1


def test_explicit_target_date_never_falls_back_to_latest(monkeypatch: Any) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("COPERNICUS_USER", "unit-test")
    monkeypatch.setenv("COPERNICUS_PASS", "unit-test")
    monkeypatch.setenv("MOCK_INGEST", "false")
    monkeypatch.setenv("TARGET_DATE", "2026-08-18")
    monkeypatch.setenv("AOI_NAME", "grand_banks")
    calls = []

    def no_scene(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs.get("target_date"))
        raise RuntimeError("No scene on requested day")

    monkeypatch.setattr(ingest, "query_copernicus_data", no_scene)
    with pytest.raises(SystemExit):
        ingest.main()
    assert calls == ["2026-08-18"]


def test_wrong_region_fails_before_authentication(tmp_path: Any) -> None:
    aoi = {
        "features": [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-72, 41], [-70, 41], [-70, 43], [-72, 43], [-72, 41]]
                    ],
                }
            }
        ]
    }
    with pytest.raises(ValueError, match="Newfoundland"):
        ingest.query_copernicus_data(aoi, str(tmp_path), "unused", "unused")


def test_exact_id_query_replays_requested_cached_product(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    identity = "42286d3d-cd80-4dc7-9aec-1613992b8256"
    name = "S1D_IW_GRDH_1SDV_20260922T212211_20260922T212236_TEST.SAFE"
    measurement = tmp_path / "data/raw" / name / "measurement"
    measurement.mkdir(parents=True)
    (measurement.parent / "manifest.safe").write_text("test cache manifest")
    for polarization in ("vv", "vh"):
        with rasterio.open(
            measurement / f"test-{polarization}-band.tiff",
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(-50, 48, 0.01, 0.01),
        ) as raster:
            raster.write(np.ones((1, 2, 2), dtype="uint16"))
    selected = {
        "Id": identity,
        "Name": name,
        "ContentDate": {"Start": "2026-09-22T21:22:11Z"},
    }
    calls: list[str] = []

    def response(value: Any) -> SimpleNamespace:
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)

    def get(url: str, **kwargs: Any) -> SimpleNamespace:
        calls.append(url)
        assert "download" not in url
        assert f"Id eq {identity}" in kwargs["params"]["$filter"]
        assert "ContentDate/Start ge" not in kwargs["params"]["$filter"]
        return response({"value": [selected]})

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: response({"access_token": "test-token"}),
    )
    monkeypatch.setattr(requests, "get", get)

    def directory(identifier: str) -> Path:
        target = tmp_path / "missions" / identifier
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(ingest, "mission_directory", directory)
    monkeypatch.delenv("AIS_ARCHIVE_DIR", raising=False)
    aoi = ingest.load_aoi(
        str(
            Path(ingest.__file__).resolve().parents[2]
            / "configs/aois/grand_banks.geojson"
        )
    )
    result = ingest.query_copernicus_data(
        aoi, str(tmp_path / "data/raw"), "test", "test", target_product_id=identity
    )
    assert result["sar_product_id"] == identity
    assert result["sar_product"] == name
    assert len(calls) == 1

    import io
    import json
    from contextlib import redirect_stdout
    from subprocess import CompletedProcess

    from agents import pipeline

    original = directory(result["mission_id"])
    (original / "ingest.json").write_text(json.dumps(result))
    (original / "ais_filtered.csv").write_text("previous mission AIS evidence\n")
    evidence = {path: path.read_bytes() for path in original.iterdir()}
    monkeypatch.setattr(
        ingest, "__file__", str(tmp_path / "agents/ingest/ingest_agent.py")
    )
    monkeypatch.setattr(ingest, "load_aoi", lambda _: aoi)
    monkeypatch.setattr(pipeline, "mission_directory", directory)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("TRITONEYE_TRACKING", "off")

    def execute(args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        if args[-1] == "agents.ingest.ingest_agent":
            with monkeypatch.context() as stage_env:
                for key, value in kwargs["env"].items():
                    stage_env.setenv(key, value)
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    ingest.main()
            return CompletedProcess(args, 0, stdout.getvalue())
        return CompletedProcess(args, 0, kwargs["input"])

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    reprocessed = pipeline.run_pipeline(
        env={
            "TARGET_PRODUCT_ID": identity,
            "COPERNICUS_USER": "test",
            "COPERNICUS_PASS": "test",
            "MOCK_INGEST": "false",
            "TARGET_DATE": "",
            "AOI_NAME": "grand_banks",
        },
        provenance={"version_key": "new-version"},
    )
    assert all(path.read_bytes() == content for path, content in evidence.items())
    assert reprocessed["source_mission_id"] == result["mission_id"]
    assert reprocessed["mission_id"] != result["mission_id"]
    destination = Path(reprocessed["execution_dir"])
    assert Path(reprocessed["ais_telemetry"]).parent == destination
    assert json.loads((destination / "ingest.json").read_text())["mission_id"] == (
        reprocessed["mission_id"]
    )


def test_catalogue_cannot_substitute_another_product(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def response(value: Any) -> SimpleNamespace:
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: response({"access_token": "test"}),
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: response(
            {"value": [{"Id": "66d3167d-240a-459a-8066-75b9d2a458f2"}]}
        ),
    )
    aoi = ingest.load_aoi(
        str(
            Path(ingest.__file__).resolve().parents[2]
            / "configs/aois/grand_banks.geojson"
        )
    )
    with pytest.raises(ValueError, match="different satellite product ID"):
        ingest.query_copernicus_data(
            aoi,
            str(tmp_path),
            "test",
            "test",
            target_product_id="42286d3d-cd80-4dc7-9aec-1613992b8256",
        )
    assert not list(tmp_path.iterdir())
