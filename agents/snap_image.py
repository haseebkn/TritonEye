"""Recover and verify ESA's original installer from a cached reference image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT
from agents.hh_hv_reservation import save
from agents.snap_reference import POLICY
from agents.source_reservation import ref


def recover(archive: Path, output: Path) -> None:
    policy = json.loads(POLICY.read_text())
    if output.exists() or not output.resolve().is_relative_to(
        (REPO_ROOT / "data/tools").resolve()
    ):
        raise ValueError("Fresh project tool output required")
    output.mkdir(parents=True)
    image_id = policy["image_id"].removeprefix("sha256:")
    with tarfile.open(archive) as exported:
        manifest_stream = exported.extractfile("blobs/sha256/" + image_id)
        if manifest_stream is None:
            raise ValueError("Pinned OCI image manifest missing")
        raw = manifest_stream.read()
        if hashlib.sha256(raw).hexdigest() != image_id:
            raise ValueError("OCI manifest hash mismatch")
        manifest = json.loads(raw)
        found: dict[str, Any] | None = None
        if manifest.get("os") != "linux" or manifest.get("architecture") != "amd64":
            raise ValueError("Pinned Linux amd64 reference required")
        for layer in manifest["rootfs"]["diff_ids"]:
            member = "blobs/sha256/" + layer.removeprefix("sha256:")
            stream = exported.extractfile(member)
            if stream is None:
                raise ValueError("OCI layer missing")
            with tarfile.open(fileobj=stream, mode="r:*") as content:
                matches = [
                    m
                    for m in content.getmembers()
                    if m.name.lstrip("./") == "tmp/snap.sh" and m.isfile()
                ]
                if not matches:
                    continue
                if len(matches) != 1 or found is not None:
                    raise ValueError("Ambiguous installer in cached image")
                source = content.extractfile(matches[0])
                if source is None:
                    raise ValueError("Installer bytes unavailable")
                path = output / "esa-snap_sentinel_linux-14.0.0.sh"
                with path.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024**2)
                if ref(path)["sha256"] != policy["installer_sha256"]:
                    raise ValueError("Official ESA installer checksum mismatch")
                check = exported.extractfile(member)
                digest = hashlib.sha256()
                if check is None:
                    raise ValueError("OCI installer layer missing")
                while block := check.read(1024**2):
                    digest.update(block)
                if "sha256:" + digest.hexdigest() != layer:
                    raise ValueError("OCI installer layer checksum mismatch")
                found = {
                    "image_id": policy["image_id"],
                    "installer": ref(path),
                    "installer_layer": layer,
                    "checksum_url": policy["checksum_url"],
                    "image_config_sha256": image_id,
                }
        if found is None:
            raise ValueError("No original installer in the cached image")
    save(output / "receipt.json", found)
    print(json.dumps({"receipt": ref(output / "receipt.json"), "verified": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.archive, args.output)
