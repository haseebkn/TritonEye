"""Production ingest must preserve source/time/region constraints."""

from typing import Any

import pytest

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
