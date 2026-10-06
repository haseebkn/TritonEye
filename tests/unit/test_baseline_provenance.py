"""Executing-checkout and external-asset hashes are separate report contracts."""

import json
from pathlib import Path
from typing import Any

import pytest

from agents import baseline
from agents.artifacts import REPO_ROOT, sha256_file
from agents.landmask import SOURCES


def baseline_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assets = tmp_path / "assets"
    weights = assets / "models/xview3/traced_ensemble.jit"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic weights for provenance-only orchestration")
    monkeypatch.setattr(baseline, "MODEL_SHA256", sha256_file(weights))
    (assets / "agents").mkdir()
    (assets / "agents/baseline.py").write_text("different implementation")
    (assets / "configs/coastline").mkdir(parents=True)
    (assets / "configs/model.yaml").write_text(
        "inference:\n  detector: xview3\nmodel: {}\n"
    )
    (assets / "configs/coastline/regional_controls.json").write_text("{}")
    shoreline = assets / "data/reference" / SOURCES["osm"]["shapefile"]
    shoreline.parent.mkdir(parents=True)
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        shoreline.with_suffix(suffix).write_bytes(f"synthetic {suffix}".encode())
    (tmp_path / "release.json").write_text("{}")

    def empty_dataset(
        dataset: Path, *, verify_files: bool, artifact_root: Path
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assert dataset == tmp_path
        assert verify_files
        assert artifact_root == assets
        return {"dataset_version": "synthetic", "scenes": []}, {}, {}

    monkeypatch.setattr(baseline, "load_dataset", empty_dataset)
    return assets


def test_alternate_assets_cannot_supply_executing_code_or_control_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = baseline_inputs(tmp_path, monkeypatch)
    output = tmp_path / "first"
    baseline.run_baseline(tmp_path, assets, output, device="cpu")
    report = json.loads((output / "baseline.json").read_text())
    provenance = report["provenance"]
    versions = provenance["processing_versions"]
    assert provenance["baseline_protocol_version"] == 2
    for category, name in (
        ("code", "agents/baseline.py"),
        ("configuration", "configs/model.yaml"),
        ("configuration", "configs/coastline/regional_controls.json"),
    ):
        assert versions[category][name] == sha256_file(REPO_ROOT / name)
        assert versions[category][name] != sha256_file(assets / name)
    assert "model" not in versions and "shoreline" not in versions
    assert provenance["assets"]["model"]["actual_sha256"] == sha256_file(
        assets / "models/xview3/traced_ensemble.jit"
    )
    shoreline = assets / "data/reference" / SOURCES["osm"]["shapefile"]
    assert provenance["assets"]["shoreline"]["files_sha256"][
        shoreline.name
    ] == sha256_file(shoreline)
    (assets / "agents/baseline.py").write_text("another implementation")
    (assets / "configs/model.yaml").write_text(
        "inference:\n  detector: yolo\nmodel: {}\n"
    )
    shoreline.write_bytes(b"different reference")
    second = baseline.run_baseline(tmp_path, assets, tmp_path / "second", device="cpu")
    assert second["provenance"]["processing_versions"] == versions
    assert (
        second["provenance"]["assets"]["shoreline"]
        != provenance["assets"]["shoreline"]
    )
    assert report["threshold_selection"]["selected_threshold"] is None


@pytest.mark.parametrize(
    "changed_file", ["agents/baseline.py", "configs/coastline/regional_controls.json"]
)
def test_final_integrity_check_uses_executing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_file: str
) -> None:
    assets = baseline_inputs(tmp_path, monkeypatch)

    def changed_checksum(path: str | Path) -> str:
        if Path(path) == REPO_ROOT / changed_file:
            return "0" * 64
        return sha256_file(path)

    monkeypatch.setattr(baseline, "sha256_file", changed_checksum)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="Code or configuration changed"):
        baseline.run_baseline(tmp_path, assets, output, device="cpu")
    assert not (output / "baseline.json").exists()
