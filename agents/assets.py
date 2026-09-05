"""Download or verify the pinned public xView3 TorchScript release.

Run explicitly: python -m agents.assets. No credentials required.
"""

import os
from pathlib import Path

import requests
import yaml

from agents.artifacts import REPO_ROOT, sha256_file

MODEL_URL = (
    "https://github.com/BloodAxe/xView3-The-First-Place-Solution/"
    "releases/download/1.0/traced_ensemble.jit"
)


def fetch_model(destination: Path, expected_sha256: str) -> Path:
    """Verify identity before installation; preserve any existing model on failure."""
    if destination.is_file():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(
                "Existing model checksum mismatch; inspect before replacing"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(MODEL_URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with open(temporary, "wb") as stream:
            for chunk in response.iter_content(1 << 20):
                stream.write(chunk)
    if sha256_file(temporary) != expected_sha256:
        raise ValueError(
            "Downloaded model checksum mismatch; .part retained for inspection"
        )
    os.replace(temporary, destination)
    return destination


def main() -> None:
    with open(REPO_ROOT / "configs" / "model.yaml", encoding="utf-8") as stream:
        expected = str(yaml.safe_load(stream)["model"]["xview3_sha256"])
    path = fetch_model(
        REPO_ROOT / "models" / "xview3" / "traced_ensemble.jit", expected
    )
    print(f"Verified: {path}\nSHA-256: {expected}")


if __name__ == "__main__":
    main()
