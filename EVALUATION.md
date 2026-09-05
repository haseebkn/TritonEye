# Evaluation and limitations

## Current claim boundary

TritonEye is an NL-only research pipeline, not an operational dark-vessel detector.
There is no independently labelled, co-temporal NL holdout in this repository.
**Vessel precision, false positives per square kilometre, and overall vessel recall
are unknown.** A zero automatic-alert count reflects disabled alerting, not perfect
accuracy. A reduction in displayed candidates is not itself an accuracy metric.

AIS observations are an incomplete, self-reported comparison subset. The evaluator
reports raw and water-eligible **AIS-subset proximity recall** when valid co-temporal
observations exist. This is neither a lower nor an upper bound on overall recall.
One-to-one matches may still be accidental; labels are needed to assess correctness.

## Verification evidence

The September 2026 audit introduced behavioral regressions for NL scope, missing
credentials/date failure, Class A/B AIS parsing, timestamps/sentinels, bounded
recording, georeferencing, deduplication, ambiguous one-to-one association,
missing coverage, JSON/report safety, and incomplete-processing handling.
Local CPU regression results and Docker/CI status are recorded in
[the audit log](docs/AUDIT.md). Synthetic end-to-end success establishes interfaces
and failure handling only; it is not an ML benchmark.

A fresh full-scene replay was attempted on the existing public Newfoundland
acquisition `S1D_IW_GRDH_1SDV_20260817T212209_20260817T212234_004171_007A29_97C4`.
The 8 GB RTX 4070 Laptop GPU encountered memory pressure in full precision.
The audit log records the eventual replay outcome; until a completed processing
manifest exists, no fresh full-scene detection totals are claimed.

Historical outputs from earlier commits included 298 targets on this acquisition
and 217 land classifications. These are legacy detector/mask counts, **not verified
vessel labels or a measured current-model false-positive rate**. They are not
carried forward as a benchmark after decoding and filtering changes.

## Known scientific limitations

- The ensemble's objectness is not a calibrated probability. Its threshold 0.15
  is provisional and not optimized on an NL test set.
- The model cannot be claimed to distinguish vessels from icebergs in NL.
  Regional ice products provide context, not per-target truth at acquisition time.
- Nominal output squares are visualization markers, not measured hull extents.
- GCP interpolation consistency is not independent geolocation accuracy.
  Terrain, moving-target SAR displacement, interpolation between control points,
  and sensor timing contribute uncertainty.
- Coastal buffering and infrastructure proximity may withhold genuine harbour or
  supply-vessel targets. Exclusions remain inspectable. Provisional installation
  coordinates are not authoritative current surveyed footprints.
- Cross-tile point deduplication is an adaptation, not the upstream competition
  solution's weighted heatmap stitching. Near-seam targets can still split/merge.
- AIS velocity propagation is deliberately bounded. Provider receipt/observation
  timestamps cannot prove exact onboard fix time; AIS absence cannot prove silence.
- The pipeline is batch processing, not a real-time multi-sensor service.

## How to obtain an honest low-false-positive operating point

1. Collect multiple NL acquisitions across coast/offshore, day, season, incidence
   angle, sea state and ice conditions; include Labrador as well as Newfoundland.
2. Obtain legitimate co-temporal AIS plus independent target labels (analyst-adjudicated
   SAR, suitable optical imagery or authorized reference observations). Explicitly
   label vessels, icebergs, fixed structures, clutter and unresolvable targets.
3. Separate scenes/dates/geographic clusters into development, validation and a
   locked holdout to avoid spatial/temporal leakage. Record licence and lineage.
4. Compare the current model and a geospatial-foundation-model baseline. Measure
   precision, recall, false alarms per valid-water km², missed vessels, association
   accuracy/ambiguity, latency and failure rate, stratified by size and conditions.
5. Choose a threshold on validation data under an agreed precision/false-alarm
   constraint; report confidence intervals and abstention/coverage tradeoffs.
   Do not optimize by suppressing all alerts or by treating AIS absence as truth.
6. Evaluate once on the holdout, then consider staged deployment with human review,
   rollback and monitoring. No automatic retraining/promotion is implemented.

Public input sources and credential requirements: [DATA_SOURCES.md](docs/DATA_SOURCES.md).
Public MDA principles and their implementation limits: [MDA_METHODS.md](docs/MDA_METHODS.md).
