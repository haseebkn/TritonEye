# Historical NL benchmark implementation — 2026-10-05

## Delivered boundary

The [v0.1.0 dataset card](../datasets/nl_benchmark/v0.1.0/DATASET_CARD.md),
[annotation guide](../datasets/nl_benchmark/v0.1.0/ANNOTATION_GUIDE.md), source
manifest, selection lock, annotations, author decisions and review history are
versioned together. SHA-256 release hashes cover the small artifacts; Git
attributes preserve their exact bytes on Windows and Linux.

The dataset is **a provisional review pilot, not an independently reviewed
accuracy benchmark**. The owner confirmed no independent reviewer is available.
No human expert or second-agent annotation review is claimed. The complete
image-first pass covers twelve selected areas, not all six full swaths or all
objects in Newfoundland and Labrador. Known uncertainty is retained, including
possible missed objects. There are no confidently confirmed vessel, structure
or ice labels; 22 unresolved annotations include one context area, and three
clutter labels are context areas. Counts are not vessel counts.

Two historical Labrador products were downloaded into `data/raw/`, bringing
real-image coverage to Makkovik (summer) and Hopedale (winter). Four real cached
Newfoundland acquisitions add Ferryland, St. John's, Bonavista and Lewisporte
with geographically selected nearshore/offshore areas. Boston and synthetic
products were excluded. Approximately 2.04 GB of downloaded Labrador
measurement/SAFE-manifest bytes are retained locally; calibration/other product
metadata and derived review chips are additional. Large products stay ignored
by Git and Docker, not uploaded as repository source.

No co-temporal AIS was available in any selected area. Five acquisition dates
have no local archive; the September 27 frozen ±5-minute snapshot has 86
observations, all outside the selected swath. This is missing supporting evidence,
not a negative class. Exact snapshot hashes/counts are in the manifest; private
AIS identities are not included in Git or the review bundle. Historical annotation
completed without waiting for the watcher.

## Reproducibility and safeguards

`agents.nl_benchmark` verifies selection identity, source/derived hashes, native
polarization, valid-image area, complete review grids, author/decision provenance,
stale-review digests, full-datatake grouping and geographic group separation.
Different splits have at least 1 km separation between selected ROI polygons;
duplicate/reprocessed products and adjacent datatake frames cannot cross splits.
Previously exposed scenes cannot be promoted to the locked test. Source full
swaths may overlap, so unselected pixels must not be added to training casually.

The labelled test acquisition receives image-only annotation/QA, not inference,
tuning or score calculation. The export interface permits only resolved,
independently full-area-reviewed **validation** labels. The current dataset
therefore deliberately refuses all metric exports. A future approved export
feeds `coastal_replay`, which applies shoreline properties to label positions
before `coastal_benchmark` compares buffers. Test release needs a separate,
predeclared versioned evaluation protocol, not an override flag.

The portable, offline, detector-blind review bundle is at
`data/benchmarks/nl/0.1.0/review_bundle.html`. It embeds both bands without network
tiles or AIS identities. Provisional first-pass decisions are collapsed so a
reviewer can inspect complete images before seeing the draft. This does not
itself prove reviewer independence or expertise.

Source metadata and the Sentinel legal notice are frozen locally. Calibration is
window-offset-aware sigma-nought dB in native radar geometry with GCP/TPS mapping.
Noise/terrain correction, weather labels and province-wide shoreline validation
are not claimed. HH/HV samples are retained for annotation, not mislabelled as
VV/VH or sent through the current VV/VH-only detector.

## Verification at implementation

- Full Windows suite: **290 passed**, 2 heavy-model tests deselected.
- Offline Linux Python 3.12 Docker suite: **288 passed**, 2 local-data tests
  skipped, 2 heavy-model tests deselected.
- Strict types: 61 agent/test files pass; project lint and formatting pass.
- Full local verification rehashed every source measurement, calibration XML,
  catalogue record, licence, AIS snapshot, calibrated chip and preview successfully.
- Container `tritoneye:nl-benchmark` built successfully; source-image digest
  `bd99fca10e24724e40f52f2a7790fc5bb0ebe76a8a252d3b8cfc5f1a7a0683ea` at this run.

These checks verify software and dataset accounting, **not label correctness or
measured model performance**. Independent adjudication and confirmed vessel
positives are still needed. Next expansion should add more acquisitions and
geographic groups, corroborated structures/ice where identifiable, and a
pretraining-overlap audit before any model-improvement claim.
