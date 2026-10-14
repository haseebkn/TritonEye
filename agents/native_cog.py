"""Bounded exact HTTP ranges for GDAL's native compressed TIFF decoder.

The loopback endpoint exposes one source only, never credentials. Remote ranges
are validated by Cdse; GDAL may decode tiles but cannot resample or select overviews.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from agents.reserved_sar import Cdse

Array = np.ndarray[Any, Any]


def read_cog(
    client: Cdse,
    url: str,
    size: int,
    dimensions: tuple[int, int],
    windows: list[tuple[int, int, int]],
) -> tuple[list[Array], list[dict[str, Any]]]:
    path = "/native_" + secrets.token_hex(8) + ".tif"
    receipts: list[dict[str, Any]] = []
    errors: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_HEAD(self) -> None:  # noqa: N802
            if self.path != path:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if self.path != path or match is None:
                self.send_error(400)
                return
            start, end = map(int, match.groups())
            if not 0 <= start <= end < size or end - start + 1 > 128 * 1024**2:
                self.send_error(416)
                return
            try:
                raw = client.fetch(url, start, end - start + 1, size)
                receipts.append(
                    {
                        "offset": start,
                        "length": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                self.send_response(206)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                self.wfile.write(raw)
            except Exception as error:
                errors.append(type(error).__name__)
                self.send_error(502)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    arrays = []
    try:
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MULTIRANGE="NO",
            GDAL_NUM_THREADS="1",
            GDAL_CACHEMAX=16 * 1024**2,
        ):
            with rasterio.open(
                f"/vsicurl/http://127.0.0.1:{server.server_port}{path}"
            ) as dataset:
                if (
                    (dataset.width, dataset.height) != dimensions
                    or dataset.count != 1
                    or dataset.dtypes != ("uint16",)
                ):
                    raise ValueError(
                        "COG base raster incompatible with native annotation"
                    )
                for col, row, length in windows:
                    if (
                        min(col, row) < 0
                        or length < 75
                        or length > 512
                        or col + length > dataset.width
                        or row + length > dataset.height
                    ):
                        raise ValueError("Native compressed window out of bounds")
                    arrays.append(
                        dataset.read(1, window=Window(col, row, length, length))
                    )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    if errors or not receipts:
        raise ValueError("Compressed native range transport failed")
    return arrays, receipts
