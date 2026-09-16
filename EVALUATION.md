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

A fresh full-scene replay **completed on 2026-09-15** on the public Newfoundland
acquisition `S1D_IW_GRDH_1SDV_20260817T212209_20260817T212234_004171_007A29_97C4`
(`status: success`, 16m43s, 6.4 GB peak VRAM). Current-code surface
classification: **17 water, 67 coastal, 228 land, 0 infrastructure of 312**.

The earlier memory failures were host RAM, not VRAM — the GPU was idle at 0 MiB
while host memory sat at 91% used. Allocator tuning
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) was sufficient.

**These are surface classifications, not vessel labels.** The scene carried no
AIS coverage, so it is not scored: precision, false alarms per km² and overall
recall remain unmeasured. The `infrastructure` count of zero reflects that no
installation falls in this scene's footprint, and separately that the offshore
production area is not acquired in VV/VH at all.

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
- **Ground-truth collection is bounded by host uptime, not just recorder
  uptime.** The recorder cannot backfill, and a sleeping machine freezes it
  mid-stream while logging nothing -- observed as a 9-hour gap containing only
  three disconnects, all recovered within 14 seconds. Sentinel-1 crosses this
  region near 09:30 and 21:30 UTC, so an overnight sleep lands on the morning
  pass. Every hour the host is asleep is permanently unscorable.
- **The coastline mask closes St. John's harbour, including its entrance.**
  Classified by the pipeline: harbour berths `land` (-39 m), mid-basin
  `coastal` (+122 m), **The Narrows entrance `land` (-220 m)**, open water only
  beyond (+444 m). OSM's coastline generalisation does not resolve a ~200 m
  entrance channel, so the polygon encloses the basin. Two consequences: the
  port and its approaches are outside coverage until a vessel clears the
  entrance, and **harbour AIS cannot serve as ground truth** -- a 2026-09-03
  acquisition carried 75 observations from 18 vessels, every one inside the
  coastline, which would have scored recall of zero by construction rather than
  by detector performance. A higher-resolution coastline (CanVec resolves
  narrow channels better than OSM here) or explicit port-water polygons would
  address it; neither is implemented.
- **The detector's VV/VH requirement excludes the offshore production area
  entirely.** Of 75 Sentinel-1 IW GRDH scenes covering the Jeanne d'Arc Basin
  installations between 2026-01-06 and 2026-09-10, **100% are HH/HV and none are
  VV/VH**; across the wider NL AOIs only about a third are VV/VH. The
  infrastructure-proximity reference list therefore cannot be exercised on real
  imagery, and a zero proximity count means "not processable", not "nothing
  found". See [DATA_SOURCES.md](docs/DATA_SOURCES.md).

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
