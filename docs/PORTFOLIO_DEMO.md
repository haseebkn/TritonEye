# TritonEye presentation walkthrough

Use this five-minute walkthrough to show one real Newfoundland development case,
then the local model lifecycle. Download and open
[the self-contained real case](../examples/nl-real-case.html) before the interview.
It needs no credentials, model loading, live AIS or network connection to display.
Keep [the existing synthetic report](../examples/nl-synthetic-report.html) as a
separate workflow example; its targets are simulated.

## Real Newfoundland case

**0:00–0:45 — Acquisition.** Show the product UUID and September 22, 2026 UTC
acquisition start. This is Sentinel-1D VV/VH, St. John's harbour and approaches,
`scene_01_area_01`, a previously exposed **training** area. Its 512px image is
selected for demonstration, not sampled for an accuracy claim. Saved detector
inference occurred on October 11 using code `d66d3a6a` and the pinned xView3
ensemble. Product identity and hashes prevent a date-only substitution.

**0:45–1:30 — Calibration.** Explain that the original integer DN is calibrated
with its matching polarization LUT: sigma0 = DN²/LUT², then 10log10. The builder
replays both native windows against the frozen calibrated chip, with actual row
and column offsets, before producing the presentation. Its sidecar reports the
maximum float32 difference. This checks reproducibility of our implementation;
it supplies no independent radiometric or absolute geolocation validation.
The display stretch is −30 to +5 dB. The detector uses sigma0 dB normalized by
`sigmoid((dB + 20) * 0.18)`, **VH then VV**, in native 2048px contexts. It never
uses the clipped presentation PNG as its calibrated input.

**1:30–2:30 — Detections and exclusions.** Point to the five numbered returns at
strict score >0.15. Green is the single eligible water candidate; amber marks
three coastal-water returns withheld by the 300m policy; pink marks one
reference-land return. The signed distances come from the saved OSM reference.
All returns remain available for research. Eligibility is not vessel truth:
neither the green target nor the excluded returns are independently confirmed.
The actual model ran one 2048px context, then clipped its predictions to this area.
This presentation reuses that output; it does not rerun inference or policy.

**2:30–3:15 — Association uncertainty.** The saved case had 28 aligned AIS
references in the ROI. At 100m, raw proximity matched 1/28; after policy it matched
0/28. The separate eligible geometric assignment recorded zero assignments and
zero flagged ambiguities. Explain why zero flags is not identity validation:
outside-ROI competitors were not considered, timestamps/motion are uncertain,
and independent vessel identities are unavailable. Do not invent an ambiguous
pair to make the demonstration more interesting. An eligible unmatched return
remains an **unassociated candidate**. AIS positions and identities are omitted.

The full-scene 3/3 result at 500m is a different sample and radius. Its conditional
95% interval is 29.2%–100%; it must not be pooled with this selected-area result.
See [the complete explanation](AIS_SUBSET_RESULT.md).

## Model lifecycle demonstration

**3:15–4:00 — Historical real refusal.** Show the lifecycle panel and
[the October 12 ledger](evidence/model_lifecycle_20261012.json). It compares the
same reference model in both branches, includes all ten validation areas, and
records six processed VV/VH areas and four unsupported HH/HV areas. Coverage is
62.23% processed and 0% independently measured. The promotion attempt was
refused and the selection/history remained unchanged. This is a saved real-data
rehearsal, not a new candidate approval or a fresh execution of old shadow evidence
against changed code.

**4:00–5:00 — Executable synthetic recovery.** In a prepared environment, run:

```sh
python -m pytest tests/unit/test_model_lifecycle.py::test_successful_shadow_promotion_and_rollback_survive_restart tests/unit/test_model_lifecycle.py::test_unreviewed_shadow_cannot_promote_even_if_report_claims_pass tests/unit/test_model_lifecycle.py::test_rollback_history_write_failure_preserves_selection_after_restart -v
```

On a Windows sandbox with an unusable system temporary directory, supply a fresh
`--basetemp` directory under `data/`; never reuse an existing populated directory.
These tests use temporary databases and invented review actors. They execute
the lifecycle implementation, real release/metric validation and synthetic
inference. They do not approve a real NL model or change the active research store.

Explain the three observable checks as the named tests pass:

1. Synthetic reference → candidate promotion → restart → rollback restores the
   reference, retaining three journal events.
2. Unreviewed evidence cannot promote even when a saved report claims success.
3. A SQLite trigger fails the history insert after the selection update is
   visible inside the transaction. Reopening the database shows the full model,
   threshold, revision and decision history unchanged. Removing the trigger
   permits rollback; one new event records the restoration.

Finish with the current boundary: engineering and reproducible experiments are
demonstrated; regional accuracy and model improvement still require defensible
labels and evaluation. Optical/RF and operational cloud deployment remain open.

## Evidence and reproduction

The real presentation includes only Sentinel SAR derivatives, detector pixel
positions, saved policy classifications and AIS **aggregates**. The
[Sentinel licence](https://cds.climate.copernicus.eu/licences/ec-sentinel) allows
distribution and adaptation with source acknowledgement. The images carry
“Contains modified Copernicus Sentinel data (2026).” OSM-derived policy values
retain contributor and ODbL attribution. No municipal/commercial imagery,
AllenAI labels, AIS tracks, credentials or model weights are embedded.

[The presentation sidecar](../examples/nl-real-case.json) binds the HTML, crop,
previews, measurement TIFFs, calibration LUTs, baseline report, dataset release,
model identity and lifecycle ledger. Those hashes identify frozen inputs;
the original large files remain local. With the authorized project inputs:

```sh
python -m agents.portfolio_demo --output reports/NEW_presenter_replay.html
```

Choose a fresh output. The builder verifies the benchmark release and development
scene/context boundaries before reading selected image inputs, checks their
hashes, replays calibration, and renders only the chosen training area. It never
reads locked test pixels or source AIS files. Missing or altered local inputs
fail the replay; the committed HTML remains usable offline.

Open the two band panels and table before presenting. Rehearse the lifecycle
command once in your interview environment. If dependencies are unavailable,
show the saved evidence and explain the command; do not describe a displayed
ledger as a live test.

Preparation validation on 12 October 2026: both real 512px band calibrations
replayed with maximum stored-float32 difference 0 dB; presentation and lifecycle
regressions passed **25 tests**, including successful rollback and history-write
failure recovery. Ruff, Black and strict Mypy passed for the added code. Browser
checks rendered two panels and five table rows with no network requests and no
horizontal overflow at a 390px phone viewport. This is software and presentation
verification; no new detection accuracy or independent review is claimed.
