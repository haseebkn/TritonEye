"""Index the pinned public CC BY TAR without extracting any raster members.

Range read-ahead can include incidental raster bytes; those are discarded,
never saved, decoded or used. Only TAR names, sizes and offsets are published.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from agents.artifacts import REPO_ROOT, write_json

URL = "https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/sentinel1_vessels.tar"
GENERATION = "1762098182799064"
SIZE = 9589923840
ETAG = '"45840224423a0f3a6bd6fd7b61d373fe"'
READ_BUDGET = 600_000_000


class RangeReader:
    """Pin GCS generation and ETag; reject ignored/truncated range responses."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.start = -1
        self.cache = b""
        self.requests = 0
        self.bytes = 0

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > SIZE:
            raise ValueError("Range outside pinned archive")
        if not self.start <= offset or offset + size > self.start + len(self.cache):
            length = min(max(65536, size), SIZE - offset)
            if self.bytes + length > READ_BUDGET:
                raise ValueError("Archive metadata read budget exceeded")
            with self.session.get(
                URL,
                params={"generation": GENERATION},
                headers={
                    "Range": f"bytes={offset}-{offset + length - 1}",
                    "If-Match": ETAG,
                },
                stream=True,
                timeout=(30, 90),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("ETag") != ETAG:
                    raise ValueError("Range or immutable object identity mismatch")
                expected = f"bytes {offset}-{offset + length - 1}/{SIZE}"
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("Range extent mismatch")
                self.cache = response.raw.read(length + 1)
            if len(self.cache) != length:
                raise ValueError("Truncated/oversized range response")
            self.start = offset
            self.requests += 1
            self.bytes += length
        return self.cache[offset - self.start : offset - self.start + size]


def index(reader: Any) -> dict[str, Any]:
    offset = 0
    members = []
    pending = None
    names = set()
    digest = hashlib.sha256()
    windows = set()
    while offset + 1024 <= SIZE:
        header = reader.read(offset, 512)
        if header == bytes(512):
            if reader.read(offset + 512, 512) != bytes(512):
                raise ValueError("Missing second TAR end marker")
            break
        digest.update(header)
        info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if info.size < 0 or offset + 512 + info.size > SIZE:
            raise ValueError("Invalid TAR member extent")
        if info.type == tarfile.GNUTYPE_LONGNAME:
            if info.size > 4096 or pending is not None:
                raise ValueError("Invalid long-name record")
            payload = reader.read(offset + 512, info.size)
            digest.update(payload)
            pending = payload.rstrip(b"\0").decode("utf-8")
        else:
            name = pending or info.name
            pending = None
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe member name")
            if name in names or not (info.isfile() or info.isdir()):
                raise ValueError("Duplicate or unsupported TAR member")
            names.add(name)
            members.append(
                {
                    "name": name,
                    "bytes": info.size,
                    "offset": offset + 512,
                    "kind": "file" if info.isfile() else "directory",
                }
            )
            if len(path.parts) >= 3 and path.parts[0] == "windows":
                window = (path.parts[1], path.parts[2])
                if window not in windows:
                    windows.add(window)
                    if len(windows) % 100 == 0:
                        print(f"indexed {len(windows)} windows", flush=True)
        offset += 512 + ((info.size + 511) // 512) * 512
    else:
        raise ValueError("Incomplete archive index")
    if pending:
        raise ValueError("Unbound long-name record")
    return {
        "url": URL,
        "generation": GENERATION,
        "bytes": SIZE,
        "etag": ETAG,
        "header_sequence_sha256": digest.hexdigest(),
        "members": members,
        "windows": len(windows),
        "range_requests": reader.requests,
        "bytes_transferred": reader.bytes,
        "raster_members_extracted": 0,
        "limits": (
            "Only headers indexed; full archive MD5 not recomputed. Range "
            "read-ahead includes discarded incidental raster bytes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(
        REPO_ROOT.resolve()
    ):
        raise ValueError("Fresh project-local index path required")
    write_json(args.output, index(RangeReader()))


if __name__ == "__main__":
    main()
