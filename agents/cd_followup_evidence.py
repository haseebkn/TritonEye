"""Publish additive C retry, noise diagnosis and provisional optical evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from agents.artifacts import REPO_ROOT
from agents.cd_evidence import mission_decisions
from agents.cd_noise_diagnosis import diagnose
from agents.cd_reference import GUARDS
from agents.cd_reference import verify as verify_snap
from agents.hh_hv_reservation import save
from agents.optical_lead import POLICY, SCOPE, render_data, validate_download
from agents.source_reservation import checked, ref

OUTPUT = REPO_ROOT / "docs/evidence/cd_followup_v010.json"
LOCK = REPO_ROOT / "configs/reservations/cd_followup_v010.lock.json"
OBSERVATION = REPO_ROOT / "configs/reservations/optical_lead_observation_v010.json"


def guards(record: dict[str, Any]) -> None:
    if (
        any(record.get(k) is not False for k in GUARDS)
        or record.get("labels_admitted", 0) != 0
    ):
        raise ValueError("Provisional evidence guard changed")


def semantic(value: Any) -> Any:
    """Compare content hashes while allowing a temporary replay directory."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return value["sha256"]
        return {k: semantic(v) for k, v in value.items()}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def build(
    snap_path: Path, noise_path: Path, optical_path: Path, local: bool
) -> dict[str, Any]:
    rules = json.loads(POLICY.read_text())
    snap = verify_snap(snap_path) if local else json.loads(snap_path.read_text())
    noise = json.loads(noise_path.read_text())
    optical = json.loads(optical_path.read_text())
    observation = json.loads(OBSERVATION.read_text())
    for record in (snap, noise, optical, observation):
        guards(record)
    if (
        snap.get("retry_of") != rules["original_snap"]
        or noise["snap"] != ref(snap_path)
        or noise["policy"] != ref(POLICY)
    ):
        raise ValueError("Exact original attempt and retry binding required")
    if (
        observation["render"] != ref(optical_path)
        or optical["inspection_policy"] != ref(SCOPE)
        or optical["policy"] != ref(POLICY)
    ):
        raise ValueError("Permitted optical inspection binding differs")
    if (
        optical["target_id"] != rules["target_id"]
        or observation["target_id"] != rules["target_id"]
    ):
        raise ValueError("Optical object identity changed")
    if observation["admission_decision"] != "hold" or any(
        observation[k] is not False
        for k in (
            "physical_corroboration",
            "same_reflector_identified",
            "local_clear_sky_verified",
        )
    ):
        raise ValueError("Optical evidence remains unresolved")
    if local:
        if diagnose(snap) != noise["bands"]:
            raise ValueError("Time-domain numerical replay differs")
        for source in noise["sources"]:
            checked(source["artifact"])
        with zipfile.ZipFile(checked(noise["runtime_calibration_jar"])) as archive:
            manifest = archive.read("META-INF/MANIFEST.MF").decode().replace("\r", "")
        if "OpenIDE-Module-Implementation-Version: 14.0.0" not in manifest:
            raise ValueError("Pinned reference module version differs")
        validate_download(checked(optical["download"]))
        for old in observation["exposure_record"]["renders"]:
            checked(old)
        for name in ("rgb_native", "overview"):
            checked(optical[name])
        for tile in optical["review_tiles"]:
            checked(tile["image"])
        with tempfile.TemporaryDirectory(
            prefix="cd-optical-replay-", dir=REPO_ROOT / "data"
        ) as directory:
            replay = render_data(checked(optical["download"]), Path(directory))
            if semantic(replay) != semantic(optical):
                raise ValueError("Optical pixels, masks or display replay differ")
    return {
        "version": "0.1.0",
        "policy": ref(POLICY),
        "inspection_policy": ref(SCOPE),
        "local_inputs": {
            "snap": ref(snap_path),
            "noise": ref(noise_path),
            "optical": ref(optical_path),
        },
        "original_attempt": snap["retry_of"],
        "retry": {
            "nominal_mission_decisions": mission_decisions(snap["areas"]),
            "areas": [
                {
                    "id": a["id"],
                    "mission": a["mission"],
                    "regime": a["regime"],
                    "comparison": a.get("comparison"),
                    "executions": {
                        k: {
                            f: v[f]
                            for f in (
                                "exit_code",
                                "timed_out",
                                "elapsed_s",
                                "elapsed_limit_exceeded",
                                "log",
                                "graph",
                            )
                        }
                        for k, v in a["variants"].items()
                    },
                }
                for a in snap["areas"]
            ],
        },
        "noise_diagnosis": noise,
        "optical": optical,
        "observation": observation,
        "noise_replay_adopted_as_model_input": False,
        "labels_admitted": 0,
        "accuracy_truth_admitted": False,
        **GUARDS,
    }


def validate(report: dict[str, Any]) -> None:
    guards(report)
    if (
        report["policy"] != ref(POLICY)
        or report["inspection_policy"] != ref(SCOPE)
        or report["observation"] != json.loads(OBSERVATION.read_text())
    ):
        raise ValueError("Frozen follow-up policies or observation changed")
    for record in (report["noise_diagnosis"], report["optical"], report["observation"]):
        guards(record)
    if (
        report["accuracy_truth_admitted"] is not False
        or report["noise_replay_adopted_as_model_input"] is not False
    ):
        raise ValueError("No truth or input adoption permitted")
    areas = report["retry"]["areas"]
    if len(areas) != 4 or len({a["id"] for a in areas}) != 4:
        raise ValueError("Four distinct native contexts required")
    decisions = mission_decisions([{**a, "variants": a["executions"]} for a in areas])
    if report["retry"]["nominal_mission_decisions"] != decisions:
        raise ValueError("Mission decisions differ")
    expected = {(a["id"], pol) for a in areas for pol in ("HH", "HV")}
    bands = report["noise_diagnosis"]["bands"]
    if len(bands) != 8 or {(b["id"], b["polarization"]) for b in bands} != expected:
        raise ValueError("Complete eight-band diagnostic required")
    for band in bands:
        passed = (
            band["pixels"] == 147456
            and band["time_replay_within_tolerance"] == 147456
            and 0 <= band["time_replay_max_absolute_db"] <= 0.001
        )
        if band["time_replay_passed"] is not passed:
            raise ValueError("Noise decision differs from frozen tolerance")
    optical = report["optical"]
    if (
        optical["same_object_confirmed"] is not False
        or optical["inspection_policy"] != ref(SCOPE)
        or optical["minimum_protected_distance_m"] < 10000
    ):
        raise ValueError("Optical identity/scope guard changed")


def publish(snap: Path, noise: Path, optical: Path) -> None:
    if OUTPUT.exists() or LOCK.exists():
        raise ValueError("Additive follow-up release required")
    report = build(snap, noise, optical, True)
    validate(report)
    save(OUTPUT, report)
    save(LOCK, {"report": ref(OUTPUT), "observation": ref(OBSERVATION)})


def verify(local: bool = False) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    checked(lock["observation"])
    report = json.loads(checked(lock["report"]).read_text())
    validate(report)
    if local:
        paths = report["local_inputs"]
        if (
            build(
                checked(paths["snap"]),
                checked(paths["noise"]),
                checked(paths["optical"]),
                True,
            )
            != report
        ):
            raise ValueError("Follow-up evidence replay differs")
    return {
        "verified": True,
        "local": local,
        "nominal_bands": 8,
        "noise_diagnostic_bands": 8,
        "labels_admitted": 0,
        "optical_corroboration": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--snap", type=Path)
    parser.add_argument("--noise", type=Path)
    parser.add_argument("--optical", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.publish and args.snap and args.noise and args.optical:
        publish(args.snap, args.noise, args.optical)
    elif args.verify:
        print(json.dumps(verify(args.local)))
    else:
        parser.error("Choose publish with inputs or verify")
