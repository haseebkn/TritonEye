"""Download pinned public research assets inside this project, not a training set."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import requests

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.croma_features import WEIGHTS_SHA256, WEIGHTS_URL

LABEL_REVISION = "9a06750051ab61ff0f8f86cf4317788295c8a909"
LABEL_BASE = (
    "https://raw.githubusercontent.com/John-J-Tanner/Extract-SARFish-Data/"
    f"{LABEL_REVISION}/Labels"
)


def download(
    url: str, destination: Path, *, max_bytes: int, checksum: str | None = None
) -> dict[str, Any]:
    """Atomic bounded download; never replace pre-existing evidence."""
    destination = destination.resolve()
    if not destination.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Research downloads must stay inside the project")
    if destination.exists():
        if checksum is None or sha256_file(destination) != checksum:
            raise ValueError(
                "Existing asset is not the pinned file; preserve and inspect it"
            )
        return {
            "url": url,
            "path": str(destination.relative_to(REPO_ROOT)),
            "sha256": checksum,
            "bytes": destination.stat().st_size,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    try:
        count = 0
        with (
            os.fdopen(fd, "wb") as stream,
            requests.get(url, stream=True, timeout=(30, 90)) as response,
        ):
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("Public asset exceeded declared download budget")
                stream.write(chunk)
        actual = sha256_file(temporary)
        if checksum is not None and checksum != actual:
            raise ValueError("Downloaded checkpoint failed its pinned SHA256 check")
        # Caller starts with absent targets; exclusive publication prevents a
        # concurrent downloader overwriting evidence after the initial check.
        os.link(temporary, destination)
        return {
            "url": url,
            "path": str(destination.relative_to(REPO_ROOT)),
            "sha256": actual,
            "bytes": count,
        }
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-label-audit", action="store_true")
    args = parser.parse_args()
    record = REPO_ROOT / "data/model_research/downloads.json"
    if record.exists():
        raise ValueError(
            "Download record exists; inspect/reuse its files, do not overwrite"
        )
    assets = [
        download(
            WEIGHTS_URL,
            REPO_ROOT / "models/croma/CROMA_base.pt",
            max_bytes=800_000_000,
            checksum=WEIGHTS_SHA256,
        )
    ]
    assets.append(
        download(
            "https://raw.githubusercontent.com/antofuller/CROMA/"
            "59505a6bcadbf36ba20767270154bf9f3067c5e7/LICENSE",
            REPO_ROOT / "data/model_research/CROMA_LICENSE.txt",
            max_bytes=10_000,
        )
    )
    if args.include_label_audit:
        for partition in ("train", "validation"):
            name = f"GRD_{partition}.csv"
            assets.append(
                download(
                    f"{LABEL_BASE}/{name}",
                    REPO_ROOT / f"data/model_research/public_label_audit/{name}",
                    max_bytes=30_000_000,
                )
            )
        assets.append(
            download(
                "https://raw.githubusercontent.com/DIUx-xView/SARFish/"
                "bfef9694946d192ce7f66c5c5f97d5be364de02c/SARFish_Terms_and_Conditions.md",
                REPO_ROOT / "data/model_research/public_label_audit/SARFish_terms.md",
                max_bytes=100_000,
            )
        )
    write_json(
        record,
        {
            "assets": assets,
            "labels_adopted": False,
            "label_repository_revision": LABEL_REVISION,
        },
    )
    print(record)


if __name__ == "__main__":
    main()
