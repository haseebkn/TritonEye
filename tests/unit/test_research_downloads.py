"""Bounded public asset downloads preserve existing files and validate bytes."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

import scripts.download_model_research_assets as assets


class Response:
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, size: int) -> Iterator[bytes]:
        yield b"download fixture"


def test_download_publishes_verified_file_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    target = tmp_path / "model.pt"
    checksum = hashlib.sha256(b"download fixture").hexdigest()
    report = assets.download(
        "https://example.invalid/file", target, max_bytes=100, checksum=checksum
    )
    assert target.read_bytes() == b"download fixture"
    assert report["sha256"] == checksum
    assert assets.download(
        "unused", target, max_bytes=100, checksum=checksum
    ) == report | {"url": "unused"}
    assert not list(tmp_path.glob("*.download"))


@pytest.mark.parametrize("budget,checksum", [(1, None), (100, "wrong")])
def test_failed_download_does_not_publish_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    checksum: str | None,
) -> None:
    monkeypatch.setattr(assets, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    target = tmp_path / "model.pt"
    with pytest.raises(ValueError):
        assets.download(
            "https://example.invalid/file", target, max_bytes=budget, checksum=checksum
        )
    assert not target.exists()
    assert not list(tmp_path.glob("*.download"))


def test_existing_unverified_file_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assets, "REPO_ROOT", tmp_path)
    target = tmp_path / "model.pt"
    target.write_bytes(b"previous research evidence")
    with pytest.raises(ValueError, match="preserve"):
        assets.download("unused", target, max_bytes=100)
    assert target.read_bytes() == b"previous research evidence"


def test_download_refuses_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(assets, "REPO_ROOT", root)
    with pytest.raises(ValueError, match="project"):
        assets.download("unused", tmp_path / "outside", max_bytes=100)
    assert not (tmp_path / "outside").exists()
