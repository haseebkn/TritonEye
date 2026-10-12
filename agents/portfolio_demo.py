"""Build one permitted, sanitized historical NL presentation without inference."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from agents.artifacts import REPO_ROOT, sha256_file, write_json
from agents.calibration import Calibrator
from agents.nl_benchmark import load_dataset, require_development_scene

ROI_ID = "scene_01_area_01"
PRODUCT_ID = "bae55045-b7cb-46f5-8ed9-f504e42650b3"
LEDGER = "docs/evidence/nl_baseline_v020_20261011.json"


def checked(root: Path, reference: dict[str, Any]) -> Path:
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Presentation input outside project")
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("Presentation input hash mismatch")
    return Path(path)


def public_case(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Allowlist aggregate evidence and SAR pixels; never export AIS tracks."""
    if (
        case["roi_id"] != ROI_ID
        or case["product_id"] != PRODUCT_ID
        or case["split"] != "train"
        or case["region"] != "Newfoundland"
        or case["state"] != "processed_unmeasured"
        or prediction["sar_product_id"] != PRODUCT_ID
        or prediction["sar_product"] != case["native_product"]
        or prediction["acquisition_time"] != case["acquisition_time"]
        or prediction["split"] != "train"
    ):
        raise ValueError("Matching unmeasured Newfoundland development case required")
    score = next(s for s in case["scores"] if s["threshold"] == 0.15)
    if score["measured"] is not False or case["metric_ready"] is not False:
        raise ValueError("Presentation must retain provisional evidence status")
    returns = []
    for feature in prediction["features"]:
        p = feature["properties"]
        if not math.isfinite(p["confidence"]):
            raise ValueError("Non-finite detection score")
        if p["confidence"] > 0.15:
            if p["surface"] not in {"water", "coastal", "land"}:
                raise ValueError("Unsupported saved physical/policy classification")
            returns.append(
                {
                    k: p[k]
                    for k in (
                        "confidence",
                        "source_row",
                        "source_col",
                        "surface",
                        "distance_to_shore_m",
                        "alert_eligible",
                    )
                }
            )
    if (
        len(returns) != score["returns"]
        or sum(r["surface"] == "coastal" for r in returns)
        != score["coastal_exclusions"]
        or sum(r["surface"] == "land" for r in returns)
        != score["physical_land_returns"]
        or sum(r["alert_eligible"] is True for r in returns)
        != score["eligible_returns"]
    ):
        raise ValueError("Saved policy counts disagree with selected returns")
    association = case["association"]
    return {
        "product_id": PRODUCT_ID,
        "native_product": case["native_product"],
        "acquisition_time": case["acquisition_time"],
        "roi_id": ROI_ID,
        "split": "train",
        "accuracy_measured": False,
        "returns": returns,
        "policy_counts": {
            k: score[k]
            for k in (
                "returns",
                "coastal_exclusions",
                "physical_land_returns",
                "eligible_returns",
            )
        },
        "ais_support": {
            k: association[k]
            for k in (
                "aligned_ais_in_roi",
                "ais_subset_proximity_matches_raw",
                "ais_subset_proximity_matches_post_policy",
                "ais_proximity_radius_m",
                "geometric_assignments",
                "ambiguous_assignments",
                "association_correctness",
            )
        },
    }


def render(summary: dict[str, Any], images: list[bytes]) -> str:
    """Self-contained HTML with only public SAR derivatives and aggregate AIS."""
    markers, rows = [], []
    col, row, width, height = summary["window"]
    for number, p in enumerate(summary["returns"], 1):
        x, y = p["source_col"] - col, p["source_row"] - row
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Return outside selected native crop")
        color = {"water": "#70f5be", "coastal": "#ffd166", "land": "#ff8f96"}[
            p["surface"]
        ]
        markers.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="none" '
            f'stroke="{color}" stroke-width="2"/><text x="{x+11:.2f}" '
            f'y="{y+4:.2f}" fill="{color}" font-size="16">{number}</text>'
        )
        policy = (
            "eligible candidate"
            if p["alert_eligible"]
            else "withheld; retained for review"
        )
        rows.append(
            f'<tr><td>{number}</td><td>{p["confidence"]:.3f}</td>'
            f'<td>{p["surface"]}</td><td>{p["distance_to_shore_m"]:.1f} m</td>'
            f"<td>{policy}</td></tr>"
        )
    panels = []
    for band, data in zip(("VV", "VH"), images, strict=True):
        encoded = base64.b64encode(data).decode("ascii")
        panels.append(
            f'<figure><svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Real {band} SAR crop with five numbered detector returns">'
            f'<image width="{width}" height="{height}" '
            f'href="data:image/png;base64,{encoded}"/>'
            + "".join(markers)
            + f"</svg><figcaption>{band}, native radar grid; "
            "display stretch −30 to +5 dB. Markers are points, not vessel hulls."
            "</figcaption></figure>"
        )
    support = summary["ais_support"]
    provenance = html.escape(json.dumps(summary, indent=2, allow_nan=False))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TritonEye real Newfoundland case</title><style>
body{{margin:0;background:#101923;color:#edf3fa;font:17px/1.55 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:32px 24px}}h1,h2{{line-height:1.2}}
.lead{{max-width:850px}}.note{{border-left:4px solid #ffd166;
padding:12px 18px;background:#1d2937}}
.panels{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}
svg{{width:100%;background:#000}}figcaption{{font-size:14px;color:#bfcbda}}
table{{width:100%;border-collapse:collapse}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #3d4b5a}}
code,pre{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;font-size:12px}}a{{color:#70f5be}}
@media(max-width:700px){{.panels{{grid-template-columns:1fr}}
main{{padding:20px 12px}}table{{font-size:13px}}}}
</style><main><p>TRITONEYE · HISTORICAL REAL DATA</p>
<h1>St. John’s harbour and approaches</h1><p class="lead">September 22, 2026,
Sentinel-1D VV/VH. This selected training area illustrates preprocessing,
detector returns and coastal policy. The images are real. The target classes and
identities remain unverified. Saved inference: October 11, 2026.</p>
<p class="note">Five returns above the provisional score 0.15: three coastal
withheld, one reference-land return, one eligible candidate. Detector scores
are uncalibrated objectness. Vessel precision, recall and false alarms are unknown.</p>
<h2>1 Acquisition and calibration</h2><p>Exact product UUID:
<code>{PRODUCT_ID}</code>.
Acquisition start: {html.escape(summary["acquisition_time"])}.
The 512-pixel crop uses the original native window. For both bands, DN²/LUT²
and 10log10 calibration were replayed against the frozen sigma0 chip; maximum
stored-float32 difference: {summary["calibration_max_abs_error_db"]:.6g} dB.
No terrain or thermal-noise correction is claimed.</p>
<p>Contains modified Copernicus Sentinel data (2026).
<a href="https://cds.climate.copernicus.eu/licences/ec-sentinel">Source licence</a>.
Policy classifications use the saved OpenStreetMap coastline:
© OpenStreetMap contributors, ODbL. No municipal or commercial basemap is embedded.</p>
<h2>2 Saved detector returns and policy</h2><div class="panels">{"".join(panels)}</div>
<p>Green: eligible water candidate. Amber: water within the 300 m coastal buffer.
Pink: reference land. Both images use the same native pixel positions.</p>
<table><thead><tr><th>Return</th><th>Score</th><th>Saved zone</th>
<th>Signed shore distance</th><th>Policy</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table><p>Coastal withholding is a policy decision;
it does not make harbour water into land or prove a return is a false alarm.
The underlying inference used one native 2048-pixel context and then clipped
returns to this 512-pixel area. These display images are not model inputs.</p>
<h2>3 Association uncertainty</h2><p>{support["aligned_ais_in_roi"]} aligned AIS
vessel references were present inside the selected area. At a 100 m proximity
radius, {support["ais_subset_proximity_matches_raw"]} had a raw detector match;
{support["ais_subset_proximity_matches_post_policy"]} remained after policy.
The separate eligible geometric association reported {support["geometric_assignments"]}
assignments and {support["ambiguous_assignments"]} flagged ambiguities.
Zero flagged ambiguity does not establish identity correctness. Outside-area
competitors were not evaluated; timing and motion allowances are heuristic.
AIS identities, tracks and coordinates are omitted from this presentation.</p>
<h2>4 Local model lifecycle evidence</h2><p>The recorded October 12 real shadow
comparison used the reference model against itself: six VV/VH validation areas
processed, four HH/HV unsupported, 62.23% water processing coverage and 0%
independently measured coverage. Promotion was refused; selection and history
remained unchanged. This record is a historical refusal,
not a fresh promotion attempt.</p>
<p>Successful rollback and a failed history-write rollback are executable
synthetic rehearsals. Follow the <a href="../docs/PORTFOLIO_DEMO.md">walkthrough</a>
for the isolated test commands and expected selection/history behavior.
No real candidate has been approved.</p>
<h2>Evidence and scope</h2><p>This presentation verifies frozen bytes and replays
calibration. It does not rerun discovery, inference, association or coastal
classification. The area was selected for explanatory value and previously
used in development. It is not independent benchmark evidence.</p>
<p><a href="../docs/REVIEWER_GUIDE.md">Reviewer guide</a> ·
<a href="../examples/nl-synthetic-report.html">Separate synthetic workflow demo</a> ·
<a href="../docs/TECHNICAL_EXPLANATION.md">Technical explanation
and contributions</a></p>
<details><summary>Hashes, native window and aggregate evidence</summary>
<pre>{provenance}</pre></details>
</main></html>"""


def build(root: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists() or not output.is_relative_to(root.resolve()):
        raise ValueError("Fresh project-local presentation output required")
    manifest, _, _ = load_dataset(root / "datasets/nl_benchmark/v0.2.0")
    scene, roi = next(
        (s, r) for s in manifest["scenes"] for r in s["rois"] if r["id"] == ROI_ID
    )
    require_development_scene(
        scene["product_id"], {**scene, "study_roi": roi["geometry"]}
    )
    if scene["product_id"] != PRODUCT_ID or scene["split"] != "train":
        raise ValueError("Fixed development demonstration case changed")
    ledger = json.loads((root / LEDGER).read_text())
    report_path = checked(
        root,
        {"path": ledger["full_report_path"], "sha256": ledger["full_report_sha256"]},
    )
    report = json.loads(report_path.read_text())
    case = next(c for c in report["cases"] if c["roi_id"] == ROI_ID)
    release_hash = sha256_file(root / "datasets/nl_benchmark/v0.2.0/release.json")
    if (
        report["provenance"]["release_sha256"] != release_hash
        or case["native_product"] != scene["name"]
        or case["acquisition_time"] != scene["acquisition_time"]
    ):
        raise ValueError("Saved inference and current frozen dataset disagree")
    for context in case["contexts"]:
        require_development_scene(
            scene["product_id"], {**scene, "study_roi": context["geometry"]}
        )
    prediction_path = checked(
        root,
        {
            "path": str(report_path.parent.relative_to(root) / f"{ROI_ID}.geojson"),
            "sha256": case["detections_sha256"],
        },
    )
    summary = public_case(case, json.loads(prediction_path.read_text()))
    licence = checked(
        root,
        {
            "path": manifest["licence"]["snapshot"],
            "sha256": manifest["licence"]["sha256"],
        },
    )
    col, row, width, height = roi["window"]
    chip = checked(root, roi["chip"])
    with rasterio.open(chip) as src:
        db = src.read()
        if (
            src.descriptions != ("sigma0_db_vv", "sigma0_db_vh")
            or db.shape != (2, height, width)
            or not (src.read_masks() > 0).all()
            or not np.isfinite(db).all()
        ):
            raise ValueError("Invalid calibrated crop")
    errors = []
    for band, pol in enumerate(("vv", "vh")):
        measurement = next(s for s in scene["sources"] if f"-{pol}-" in s["path"])
        calibration = roi["calibration"][band]
        if f"-{pol}-" not in calibration["path"]:
            raise ValueError("Calibration polarization mismatch")
        with rasterio.open(checked(root, measurement)) as src:
            dn = src.read(1, window=Window(col, row, width, height))
        replay = (
            Calibrator(str(checked(root, calibration)))
            .to_sigma0_db(dn, row, col)
            .astype("float32")
        )
        if replay.shape != db[band].shape or not np.isfinite(replay).all():
            raise ValueError("Invalid native calibration replay")
        error = float(np.max(np.abs(replay - db[band])))
        if error > 1e-5:
            raise ValueError("Calibration replay mismatch")
        errors.append(error)
    images = [checked(root, p).read_bytes() for p in roi["previews"]]
    lifecycle_path = root / "docs/evidence/model_lifecycle_20261012.json"
    lifecycle = json.loads(lifecycle_path.read_text())
    if (
        lifecycle["status"] != "blocked"
        or lifecycle["paired_same_model"] is not True
        or lifecycle["processed_areas_per_role"] != 6
        or lifecycle["unsupported_hh_hv_areas_per_role"] != 4
        or lifecycle["measured_water_coverage_per_role"] != 0.0
        or round(lifecycle["processing_water_coverage_per_role"] * 100, 2) != 62.23
        or lifecycle["promotion_attempt"] != "refused; selection and journal unchanged"
    ):
        raise ValueError("Historical lifecycle presentation record changed")
    checked(
        root, {"path": lifecycle["report_path"], "sha256": lifecycle["report_sha256"]}
    )
    summary.update(
        window=roi["window"],
        calibration_max_abs_error_db=max(errors),
        baseline_report_sha256=sha256_file(report_path),
        dataset_release_sha256=release_hash,
        chip_sha256=sha256_file(chip),
        preview_sha256=[p["sha256"] for p in roi["previews"]],
        source_measurements_sha256=[p["sha256"] for p in scene["sources"]],
        calibration_sha256=[p["sha256"] for p in roi["calibration"]],
        licence_snapshot_sha256=sha256_file(licence),
        model_sha256=report["provenance"]["model_sha256"],
        inference_commit=report["provenance"]["git_head"],
        lifecycle_ledger_sha256=sha256_file(lifecycle_path),
        presentation_builder_sha256=sha256_file(__file__),
        locked_test_pixels_read=False,
        ais_tracks_exported=False,
    )
    page = render(summary, images)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8", newline="\n")
    write_json(sidecar, {**summary, "html_sha256": sha256_file(output)})
    return {
        "report_html": str(output),
        "evidence": str(sidecar),
        "calibration_max_abs_error_db": max(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(REPO_ROOT, args.output)))


if __name__ == "__main__":
    main()
