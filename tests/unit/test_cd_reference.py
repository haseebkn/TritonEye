"""Exercise C/D processing and conditional evidence boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from shapely.geometry import box, mapping

from agents.artifacts import REPO_ROOT
from agents.cd_evidence import LOCK, mission_decisions, verify
from agents.cd_reference import POLICY, header
from agents.cd_source_refresh import response_status, search_params, summarize_optical
from agents.native_nl_input import prepare
from agents.provisional_training import policy, select
from agents.source_reservation import envelope, ref


def xml(mission: str = "S1C", pol: str = "HH", corrected: str = "false") -> bytes:
    return f"""<product><adsHeader><missionId>{mission}</missionId>
    <productType>GRD</productType><polarisation>{pol}</polarisation><mode>IW</mode>
    <startTime>2026-07-03T09:49:24</startTime><stopTime>2026-07-03T09:49:49</stopTime>
    <absoluteOrbitNumber>8377</absoluteOrbitNumber><missionDataTakeId>10936</missionDataTakeId>
    </adsHeader><rangePixelSpacing>10</rangePixelSpacing>
    <azimuthPixelSpacing>10</azimuthPixelSpacing>
    <thermalNoiseCorrectionPerformed>{corrected}</thermalNoiseCorrectionPerformed>
    </product>""".encode()


@pytest.mark.parametrize("mission", ["S1C", "S1D"])
def test_matching_cd_metadata_is_only_a_processing_input(mission: str) -> None:
    raw = xml(mission)
    result = header(raw, raw, raw, mission, "HH")
    assert result["mission"] == mission and result["thermal_noise_corrected"] == "false"
    # Existing frozen native consumer continues to reject C/D.
    assert (
        prepare(None, [0, 0, 1, 1], {"mission": mission}, "train")["status"]
        == "abstain"
    )


@pytest.mark.parametrize("part", [0, 1, 2])
@pytest.mark.parametrize("change", ["mission", "polarization", "time"])
def test_metadata_subject_mismatch_is_refused(part: int, change: str) -> None:
    values = [xml()] * 3
    if change == "mission":
        values[part] = xml("S1D")
    elif change == "polarization":
        values[part] = xml(pol="HV")
    else:
        values[part] = xml().replace(b"09:49:24", b"09:48:24")
    with pytest.raises(ValueError, match="identity"):
        header(*values, "S1C", "HH")


@pytest.mark.parametrize("corrected", ["true", "", "unknown"])
def test_missing_or_changed_noise_state_is_not_nominal(corrected: str) -> None:
    with pytest.raises(ValueError, match="nominal"):
        header(xml(corrected=corrected), xml(), xml(), "S1C", "HH")


def area(mission: str, regime: str) -> dict[str, Any]:
    return {
        "mission": mission,
        "regime": regime,
        "comparison": {"passed": True, "noise": [{"passed": True}, {"passed": True}]},
        "variants": {
            v: {"exit_code": 0, "timed_out": False, "elapsed_limit_exceeded": False}
            for v in ("raw", "calibration", "noise")
        },
    }


def test_mission_specific_results_and_runtime_are_separate() -> None:
    rows = [area(m, r) for m in ("S1C", "S1D") for r in ("coastal", "offshore")]
    rows[2]["comparison"]["noise"][1]["passed"] = False
    rows[0]["variants"]["raw"]["elapsed_limit_exceeded"] = True
    c, d = mission_decisions(rows)
    assert c["nominal_reference_passed"] and c["noise_reference_passed"]
    assert not c["bounded_execution_passed"]
    assert d["nominal_reference_passed"] and not d["noise_reference_passed"]
    assert d["bounded_execution_passed"]
    assert not c["input_contract_extended"] and not d["training_admitted"]


def test_missing_or_duplicate_regime_does_not_complete_mission() -> None:
    assert not mission_decisions([area("S1C", "coastal")])[0][
        "nominal_reference_passed"
    ]
    assert not mission_decisions([area("S1C", "coastal")] * 2)[0][
        "nominal_reference_passed"
    ]
    rows = [area("S1C", r) for r in ("coastal", "offshore")]
    rows[0]["variants"].pop("noise")
    assert not mission_decisions(rows)[0]["bounded_execution_passed"]


def test_failed_execution_cannot_pass_from_a_saved_comparison_flag() -> None:
    rows = [area("S1C", r) for r in ("coastal", "offshore")]
    rows[0]["variants"]["calibration"].update(exit_code=None, timed_out=True)
    assert not mission_decisions(rows)[0]["nominal_reference_passed"]


def candidate() -> dict[str, Any]:
    return {
        "stream": "vessel_support",
        "product": {
            "Id": "e4f3f146-305f-4dcf-bd9e-000000000001",
            "Name": "S1D_IW_GRDH_1SDH_20260702T095607_20260702T095632_"
            "003493_0062D9_B1F9.SAFE",
            "Online": True,
            "GeoFootprint": mapping(box(-58, 46, -50, 54)),
        },
        "source_identity": "example",
        "lon": -55.5,
        "lat": 50.2,
        "absolute_offset_s": 10,
        "search_uncertainty_m": 200,
        "source": {},
    }


@pytest.mark.parametrize("role", ["validation", "test", "quarantine"])
def test_conditional_cd_screen_cannot_reassign_protected_roles(role: str) -> None:
    row = candidate()
    prior = {
        "geometry": envelope(row["lon"], row["lat"], 3000),
        "group": "prior",
        "split": role,
        "quarantine": role == "quarantine",
    }
    rules = policy() | {"native_specification": ref(POLICY)}
    selected, ranked = select([row], [prior], set(), [], rules)
    assert selected == [] and ranked[0]["decision"] == "excluded"


def test_conditional_reservation_never_admits_label() -> None:
    selected, _ = select(
        [candidate()], [], set(), [], policy() | {"native_specification": ref(POLICY)}
    )
    assert len(selected) == 1
    assert selected[0]["label"] is None and selected[0]["training_approved"] is False


def target() -> dict[str, Any]:
    return {
        "id": "case",
        "lon": -53.0,
        "lat": 49.0,
        "native_time_utc": "2020-05-23T10:00:00+00:00",
    }


def test_bounded_search_spans_day_boundary_and_retains_identity_uncertainty() -> None:
    s2, stac = search_params(target())
    assert stac["datetime"] == "2020-05-22T22:00:00Z/2020-05-23T22:00:00Z"
    assert stac["limit"] == 100 and s2["$top"] == "100"
    item = {
        "id": "optical",
        "geometry": mapping(box(-54, 48, -52, 50)),
        "properties": {"datetime": "2020-05-23T14:00:00Z", "eo:cloud_cover": 0},
    }
    summary = summarize_optical({"features": [item]}, target(), True)[0]
    assert summary["covers_original_position"] and summary["offset_s"] == 14400
    assert not summary["same_object_confirmed"] and not summary["pixels_inspected"]
    item = deepcopy(item)
    item["geometry"] = mapping(box(-56, 48, -55, 49))
    assert not summarize_optical({"features": [item]}, target(), True)[0][
        "covers_original_position"
    ]


def test_public_cd_release() -> None:
    if not LOCK.exists():
        pytest.skip("Evidence release not generated yet")
    result = verify()
    assert result["labels_admitted"] == 0


@pytest.mark.parametrize("stac", [False, True])
def test_failed_and_truncated_search_are_not_empty_complete_search(stac: bool) -> None:
    key = "features" if stac else "value"
    assert response_status({key: []}, stac) == "complete"
    assert response_status({"error_type": "Timeout"}, stac) == "failed"
    more = (
        {"links": [{"rel": "next", "href": "next-page"}]}
        if stac
        else {"@odata.nextLink": "next-page"}
    )
    assert response_status({key: [], **more}, stac) == "truncated"
    with pytest.raises(ValueError, match="schema"):
        response_status({}, stac)


def test_private_cd_pixels_and_evidence_replay() -> None:
    path = REPO_ROOT / "data/source_supported/cd_snap_v010_01/report.json"
    if not path.exists() or not LOCK.exists():
        pytest.skip("Retained local C/D reference artifacts required")
    assert verify(local=True)["verified"]
