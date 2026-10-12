"""Presentation output contracts with synthetic inputs, no private data."""

import copy
from pathlib import Path
from typing import Any

import pytest

from agents import portfolio_demo as demo
from agents.artifacts import sha256_file


def synthetic_case() -> tuple[dict[str, Any], dict[str, Any]]:
    features = []
    for number, surface in enumerate(
        ("water", "coastal", "coastal", "land", "coastal")
    ):
        features.append(
            {
                "properties": {
                    "confidence": 0.2 + number * 0.01,
                    "source_col": 10 + number * 10,
                    "source_row": 10 + number * 10,
                    "surface": surface,
                    "distance_to_shore_m": 400 if surface == "water" else 10,
                    "alert_eligible": surface == "water",
                    "mmsi": "PRIVATE_ID_SHOULD_NOT_EXPORT",
                    "ais_track": "PRIVATE_TRACK_SHOULD_NOT_EXPORT",
                },
            }
        )
    case = {
        "roi_id": demo.ROI_ID,
        "product_id": demo.PRODUCT_ID,
        "native_product": "synthetic source",
        "acquisition_time": "2026-09-22T21:21:46Z",
        "split": "train",
        "region": "Newfoundland",
        "state": "processed_unmeasured",
        "metric_ready": False,
        "scores": [
            {
                "threshold": 0.15,
                "measured": False,
                "returns": 5,
                "coastal_exclusions": 3,
                "physical_land_returns": 1,
                "eligible_returns": 1,
            }
        ],
        "association": {
            "aligned_ais_in_roi": 28,
            "ais_subset_proximity_matches_raw": 1,
            "ais_subset_proximity_matches_post_policy": 0,
            "ais_proximity_radius_m": 100,
            "geometric_assignments": 0,
            "ambiguous_assignments": 0,
            "association_correctness": None,
            "mmsi": "PRIVATE_ID_SHOULD_NOT_EXPORT",
        },
    }
    prediction = {
        "sar_product_id": demo.PRODUCT_ID,
        "sar_product": "synthetic source",
        "acquisition_time": case["acquisition_time"],
        "split": "train",
        "features": features,
    }
    return case, prediction


def test_generated_presentation_has_sar_panels_without_private_identity() -> None:
    case, prediction = synthetic_case()
    summary = demo.public_case(case, prediction)
    summary.update(window=[0, 0, 512, 512], calibration_max_abs_error_db=0)
    page = demo.render(summary, [b"synthetic PNG one", b"synthetic PNG two"])
    # Inspect the emitted public HTML contract, not implementation source.
    assert page.count("data:image/png;base64,") == 2
    assert page.count("<circle ") == 10
    assert "PRIVATE_ID_SHOULD_NOT_EXPORT" not in page
    assert "PRIVATE_TRACK_SHOULD_NOT_EXPORT" not in page
    assert summary["accuracy_measured"] is False
    assert summary["policy_counts"]["coastal_exclusions"] == 3
    assert summary["ais_support"]["association_correctness"] is None


@pytest.mark.parametrize("change", ["split", "product", "time", "measured", "counts"])
def test_mismatched_evidence_cannot_be_presented_as_the_real_case(change: str) -> None:
    case, prediction = synthetic_case()
    if change == "split":
        case["split"] = "test"
    elif change == "product":
        prediction["sar_product_id"] = "another acquisition"
    elif change == "time":
        prediction["acquisition_time"] = "2026-09-23T21:21:46Z"
    elif change == "measured":
        case["scores"][0]["measured"] = True
    else:
        case["scores"][0]["coastal_exclusions"] = 2
    with pytest.raises(ValueError):
        demo.public_case(case, prediction)


def test_source_hash_and_escape_checks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"synthetic source bytes")
    reference = {"path": "source", "sha256": sha256_file(source)}
    assert demo.checked(tmp_path, reference) == source
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        demo.checked(tmp_path, reference)
    escape = copy.deepcopy(reference)
    escape["path"] = "../outside"
    with pytest.raises(ValueError, match="outside project"):
        demo.checked(tmp_path, escape)


def test_existing_output_is_preserved_before_loading_data(tmp_path: Path) -> None:
    output = tmp_path / "demo.html"
    output.write_text("existing artifact")
    with pytest.raises(ValueError, match="Fresh"):
        demo.build(tmp_path, output)
    assert output.read_text() == "existing artifact"
