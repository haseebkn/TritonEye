"""Real TIFF encoding exercises the bounded native strip consumer."""

import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio

from agents.reserved_sar import DOWNLOAD, Cdse, NativeTiff


def fixture_tiff(tmp_path: Path, **options: Any) -> bytes:
    path = tmp_path / "native.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=800,
        height=800,
        count=1,
        dtype="uint16",
        **options,
    ) as dst:
        dst.write(np.arange(800 * 800, dtype="uint16").reshape(800, 800), 1)
    return path.read_bytes()


def test_exact_native_window_and_bounded_reads(tmp_path: Path) -> None:
    raw = fixture_tiff(tmp_path)
    requests = []

    def read(start: int, count: int) -> bytes:
        assert 0 <= start < start + count <= len(raw)
        requests.append((start, count))
        return raw[start : start + count]

    source = NativeTiff(read, len(raw))
    expected = np.arange(800 * 800, dtype="uint16").reshape(800, 800)[101:613, 31:543]
    np.testing.assert_array_equal(source.window(31, 101, 512), expected)
    assert sum(n for _, n in requests) < len(raw)


def test_compressed_data_refused(tmp_path: Path) -> None:
    raw = fixture_tiff(tmp_path, compress="deflate")
    with pytest.raises(ValueError, match="uncompressed"):
        NativeTiff(lambda a, n: raw[a : a + n], len(raw))


@pytest.mark.parametrize("window", [(-1, 0, 512), (0, 0, 1024), (400, 400, 512)])
def test_window_refused_before_data_read(
    tmp_path: Path, window: tuple[int, int, int]
) -> None:
    raw = fixture_tiff(tmp_path)
    source = NativeTiff(lambda a, n: raw[a : a + n], len(raw))
    source.read = lambda a, n: pytest.fail("Pixels must not be read")
    with pytest.raises(ValueError, match="bounded"):
        source.window(*window)


@pytest.mark.parametrize(
    "status,content_range,body",
    [
        (200, "bytes 0-3/100", b"abcd"),
        (206, "bytes 1-4/100", b"abcd"),
        (206, "bytes 0-3/100", b"abc"),
        (206, "bytes 0-3/100", b"abcde"),
    ],
)
def test_range_transport_rejects_wrong_or_partial_response(
    monkeypatch: pytest.MonkeyPatch, status: int, content_range: str, body: bytes
) -> None:
    class Response:
        status_code = status
        headers = {"Content-Range": content_range}
        raw = BytesIO(body)

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    client = Cdse()
    client.authenticated = time.monotonic()
    monkeypatch.setattr(client.session, "get", lambda *a, **kw: Response())
    with pytest.raises(ValueError):
        client.fetch(DOWNLOAD + "(fixture)/Nodes(native)/$value", 0, 4, 100)
    assert client.bytes == 0
