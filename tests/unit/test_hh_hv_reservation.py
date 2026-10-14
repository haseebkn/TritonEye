"""Metadata-only reservation, geography separation and pre-pixel scope guards."""

import copy
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from agents import hh_hv_candidate_run as runner
from agents import hh_hv_reservation as reservation
from agents.artifacts import REPO_ROOT, sha256_file


@pytest.fixture
def policy() -> dict[str, Any]:
    return json.loads(reservation.POLICY.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def metadata(
    group: dict[str, Any], epoch: dict[str, Any], ordinal: int
) -> dict[str, Any]:
    date = epoch["start"][:10].replace("-", "")
    return {
        "Id": str(uuid5(NAMESPACE_URL, f"fixture/{group['id']}/{epoch['id']}")),
        "Name": (
            f"S1A_IW_GRDH_1SDH_{date}T100000_{date}T100025_"
            f"{100000+ordinal:06d}_ABCD12_1234.SAFE"
        ),
        "Online": True,
        "ContentDate": {
            "Start": f"{epoch['start'][:10]}T10:00:00Z",
            "End": f"{epoch['start'][:10]}T10:00:25Z",
        },
        "GeoFootprint": mapping(box(-61.4, 45.5, -50, 58)),
        "Attributes": [
            {"Name": "polarisationChannels", "Value": "HH&HV"},
            {"Name": "orbitDirection", "Value": "DESCENDING"},
            {"Name": "relativeOrbitNumber", "Value": 54},
        ],
    }


@pytest.fixture
def sealed(
    tmp_path: Path, monkeypatch: Any, policy: dict[str, Any]
) -> tuple[Path, str, dict[str, Any]]:
    frozen_config = REPO_ROOT / "configs/candidates/hh_hv_local_contrast_v010.json"
    config = tmp_path / "configs/candidates/hh_hv_local_contrast_v010.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(frozen_config.read_bytes())
    monkeypatch.setattr(reservation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        reservation,
        "exposure_inventory",
        lambda: {"files": {}, "native_groups": [], "regions": []},
    )

    def query(
        group: dict[str, Any], epoch: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ordinal = (
            next(i for i, g in enumerate(policy["groups"]) if g["id"] == group["id"])
            * 2
        )
        ordinal += next(
            i for i, e in enumerate(policy["epochs"]) if e["id"] == epoch["id"]
        )
        return [metadata(group, epoch, ordinal)], {"fixture": True}

    monkeypatch.setattr(reservation, "query", query)
    policy_path = tmp_path / "policy.json"
    reservation.save(policy_path, policy)
    path = tmp_path / "reservations/hh_hv/fixture"
    manifest = reservation.reserve(path, policy_path)
    return path, sha256_file(path / "seal.json"), manifest


def test_reservation_is_complete_and_unscored(
    sealed: tuple[Path, str, dict[str, Any]],
) -> None:
    directory, checksum, _ = sealed
    manifest = reservation.load_reservation(directory, checksum)
    assert len(manifest["scenes"]) == 8
    assert len({s["acquisition_group"] for s in manifest["scenes"]}) == 8
    assert {
        s["geographic_group"] for s in manifest["scenes"] if s["split"] == "train"
    }.isdisjoint(
        {
            s["geographic_group"]
            for s in manifest["scenes"]
            if s["split"] == "validation"
        }
    )
    assert all(
        {r["regime"] for r in s["rois"]} == {"coastal", "offshore"}
        for s in manifest["scenes"]
    )
    assert manifest["scored"] is False and manifest["imagery_inspected"] is False
    assert (
        manifest["independent_review"] is False
        and manifest["promotion_eligible"] is False
    )


def test_altered_manifest_or_wrong_seal_rejected(
    sealed: tuple[Path, str, dict[str, Any]],
) -> None:
    directory, checksum, _ = sealed
    with pytest.raises(ValueError, match="seal mismatch"):
        reservation.load_reservation(directory, "0" * 64)
    path = directory / "manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        reservation.load_reservation(directory, checksum)


def test_prior_geography_and_renamed_group_overlap_rejected(
    policy: dict[str, Any],
) -> None:
    prior = {
        "regions": [
            {
                "geometry": reservation.envelopes(policy["groups"][0], policy)[0][
                    "geometry"
                ]
            }
        ]
    }
    with pytest.raises(ValueError, match="prior exposure"):
        reservation.validate_policy(policy, prior)
    policy["groups"][2]["coastal_center"] = policy["groups"][0]["coastal_center"]
    policy["groups"][2]["offshore_center"] = policy["groups"][0]["offshore_center"]
    with pytest.raises(ValueError, match="not separated"):
        reservation.validate_policy(policy, {"regions": []})


@pytest.mark.parametrize(
    "fault",
    ["vv_vh", "missing_footprint", "out_of_swath", "old_acquisition", "outside_epoch"],
)
def test_ineligible_catalogue_rows_cannot_be_reserved(
    policy: dict[str, Any], fault: str
) -> None:
    group, epoch = policy["groups"][0], policy["epochs"][0]
    product = metadata(group, epoch, 0)
    excluded: set[str] = set()
    if fault == "vv_vh":
        product["Attributes"][0]["Value"] = "VV&VH"
    if fault == "missing_footprint":
        product["GeoFootprint"] = mapping(box(0, 0, 0, 0))
    if fault == "out_of_swath":
        product["GeoFootprint"] = mapping(box(-54, 48, -53, 49))
    if fault == "old_acquisition":
        excluded.add(reservation.product_metadata(product)["acquisition_group"])
    if fault == "outside_epoch":
        product["ContentDate"]["Start"] = "2024-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="No eligible"):
        reservation.choose([product], group, epoch, policy, excluded)


def test_scope_refuses_aliases_splits_or_escaped_geometry(
    sealed: tuple[Path, str, dict[str, Any]],
) -> None:
    directory, checksum, _ = sealed
    manifest = reservation.load_reservation(directory, checksum)
    scene = manifest["scenes"][0]
    roi = {**scene["rois"][0], "group": scene["geographic_group"]}
    reservation.require_scope(manifest, scene, roi)
    for key, value in [
        ("product_id", str(uuid5(NAMESPACE_URL, "unreserved"))),
        ("split", "validation"),
        ("name", scene["name"].replace("1234", "5678")),
    ]:
        with pytest.raises(ValueError):
            reservation.require_scope(manifest, {**scene, key: value}, roi)
    with pytest.raises(ValueError):
        reservation.require_scope(
            manifest, scene, {**roi, "geometry": mapping(box(-54, 48, -53, 49))}
        )


def test_runner_refuses_out_of_reservation_before_pixels(
    sealed: tuple[Path, str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    directory, checksum, manifest = sealed
    scene = copy.deepcopy(manifest["scenes"][0])
    scene["rois"][0]["group"] = "renamed_escape"
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(runner, "load_dataset", lambda _: ({"scenes": [scene]}, {}, {}))
    output = tmp_path / "data/should_not_exist"
    with pytest.raises(ValueError, match="reserved development scope"):
        runner.run(output, tmp_path / "dataset", directory, checksum)
    assert not output.exists()
    with pytest.raises(ValueError, match="together"):
        runner.run(output, tmp_path / "dataset", directory)


def test_committed_reservation_consumes_without_network(monkeypatch: Any) -> None:
    directory = reservation.DEFAULT
    lock = json.loads(
        (
            REPO_ROOT / "configs/reservations/hh_hv_development_v010.lock.json"
        ).read_text()
    )

    def no_network(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Reservation verification must stay offline")

    monkeypatch.setattr(requests, "get", no_network)
    manifest = reservation.load_reservation(directory, lock["seal_sha256"])
    assert len(manifest["scenes"]) == 8
    assert {s["split"] for s in manifest["scenes"]} == {"train", "validation"}


@pytest.mark.parametrize("outside", [False, True])
def test_reserved_runner_checks_actual_raster_extent_before_array_access(
    sealed: tuple[Path, str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: Any,
    outside: bool,
) -> None:
    directory, checksum, manifest = sealed
    scene = copy.deepcopy(manifest["scenes"][0])
    images = np.zeros((2, 101, 101), dtype="float32")
    images[0, 50, 50] = 12
    path = tmp_path / "data/chip.tif"
    path.parent.mkdir()
    lon, lat = (-54.0, 48.0) if outside else (-55.16, 47.03)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=101,
        height=101,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(lon, lat, 0.0001, 0.0001),
    ) as dst:
        dst.write(images)
    roi = {
        **scene["rois"][0],
        "id": "reserved_fixture",
        "group": scene["geographic_group"],
        "window": [0, 0, 101, 101],
        "chip": {"path": "data/chip.tif", "sha256": sha256_file(path)},
    }
    scene["rois"] = [roi]
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(runner, "runtime_check", lambda: None)
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda _: (
            {"scenes": [scene], "licence": {"attribution": "generated fixture"}},
            {},
            {},
        ),
    )
    calls = []

    def audit(*_: Any) -> tuple[Any, dict[str, Any]]:
        calls.append("pixels")
        return images, {"calibration_replay_max_abs_db": [0, 0]}

    monkeypatch.setattr(runner, "native_audit", audit)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reservation.save(dataset / "release.json", {"fixture": True})
    output = tmp_path / "data/verified_run"
    result = runner.run(output, dataset, directory, checksum)
    if outside:
        assert calls == []
        assert result["status"] == "incomplete"
        assert result["summary"]["failed_areas"] == 1
    else:
        assert calls == ["pixels"]
        assert result["status"] == "complete"
        definition = json.loads((output / "definition.json").read_text())
        assert definition["reservation"]["seal_sha256"] == checksum
        assert result["areas"][0]["reserved_raster_bounds"]


@pytest.mark.parametrize(
    "field", ["scored", "independent_review", "promotion_eligible", "imagery_inspected"]
)
def test_even_resealed_false_status_cannot_pass_contract(
    sealed: tuple[Path, str, dict[str, Any]],
    field: str,
) -> None:
    directory, _, manifest = sealed
    manifest[field] = True
    path = directory / "manifest.json"
    path.write_bytes((json.dumps(manifest) + "\n").encode())
    seal = json.loads((directory / "seal.json").read_text())
    seal["files"]["manifest.json"] = sha256_file(path)
    (directory / "seal.json").write_bytes((json.dumps(seal) + "\n").encode())
    with pytest.raises(ValueError, match="status or frozen generator"):
        reservation.load_reservation(directory, sha256_file(directory / "seal.json"))
