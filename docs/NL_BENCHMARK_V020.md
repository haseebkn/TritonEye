# Step 3 benchmark expansion — 11 October 2026

The [0.2.0 dataset card](../datasets/nl_benchmark/v0.2.0/DATASET_CARD.md) and
[annotation guide](../datasets/nl_benchmark/v0.2.0/ANNOTATION_GUIDE.md) describe a
versioned provisional expansion, with 21 complete selected areas from 11 exact
products / 10 native acquisition groups. The dataset has a complete machine
first pass and is prepared for independent review. **No independent SAR reviewer
is available, so Step 3's independent benchmark acceptance remains incomplete.**

## Delivered evidence

- Twelve frozen v0.1.0 areas, labels and review events inherited unchanged,
  plus nine newly inspected areas. Newfoundland has 17 areas; Labrador has four
  inherited Makkovik/Hopedale areas. This extension did not open Hopedale test
  imagery or change the parent release.
- Train / validation / locked test: 9 / 10 / 2 areas. Datatake grouping, geographic
  assignments and at least 1 km cross-split separation are checked. Adjacent
  September 27 slices keep the same validation assignment.
- All 84 review cells have recorded first-pass coverage: 48 inherited and 36
  new. Both native bands were inspected in all nine additions. Coverage records
  are annotator declarations and do not establish exhaustive object correctness.
- Five new **provisional vessel interpretations**, 40 unresolved annotations
  including inherited uncertainty, and seven clutter context areas. Of the 40
  unresolved annotations, one is an inherited winter context area. Clusters
  do not establish vessel counts; context areas are not object-centre negatives.
  No fixed structures or ice targets are independently confirmed.
- 13 VV/VH areas and eight experimental HH/HV areas. Separate metadata and native
  calibration are preserved; HH/HV detector compatibility remains unvalidated.
- Source/licence/product/time/image hashes, pixel boxes, geographic centres,
  decisions and append-only review history are frozen in the release.

## AIS after the image pass

The [supporting ledger](../datasets/nl_benchmark/v0.2.0/ais_support.json) binds to
the hash of the persisted first-pass annotations. It retains the whole scene's
competing identity pool for the calculation, publishing only counts and distances.
Each of the five provisional vessel interpretations has one observed aligned
track within the illustrative 500 m gate. Nearest distances are 109.7, 253.1,
432.1, 347.6 and 320.8 m. Correct identities remain **unreviewed**.

These are image-annotation corroboration observations from four acquisition
groups, not five new independently verified matching cases or a 5/5 detector
result. Two northern VV/VH objects belong to the same September 27 scene already
used in the historical [3/3 detector proximity result](AIS_SUBSET_RESULT.md);
they must not increase that denominator. The other three interpretations were
not tested against detector outputs in this work, including two HH/HV cases.

The frozen support uses catalogue-midpoint alignment from the overlap review.
It differs from scene-start detector evaluation and lacks per-pixel radar times.
AIS has gaps, motion/location uncertainty and incomplete reception. The enrichment
was selected using AIS availability; proximity could be coincidental. In dense
harbour areas, supporting AIS does not resolve bright shoreline clusters into
individual vessel identities.

## Reviewer package and reproduction

Open `data/benchmarks/nl/0.2.0/review_bundle.html` for the nine additions. It embeds
18 unmarked native previews and keeps first-pass decisions collapsed. The parent
bundle at `data/benchmarks/nl/0.1.0/review_bundle.html` contains the 12 inherited
areas. Read the guide, inspect all image cells before revealing decisions, then
consult authorized telemetry afterward. The packet is detector-blind;
AIS-informed sampling and prior preparer exposure remain disclosed.

Large SAR data, AIS histories and local HTML remain ignored by Git. The repository
contains small release artifacts and reusable software. The isolated code gate
uses these public contracts and synthetic executable inputs; it is not authorized
to receive linked/copied private data. Real source checks occur in the project
checkout. Software review is separate from independent scientific label review.

```sh
# Check the versioned release and its annotation/split contracts without raw data.
python -m agents.nl_benchmark --dataset datasets/nl_benchmark/v0.2.0
python -m pytest tests/unit/test_benchmark_revision.py tests/unit/test_nl_benchmark.py -q

# A NEW version/plan is required to extend again; never overwrite a release.
python -m agents.benchmark_revision draft --plan configs/benchmarks/new_plan.json --dataset datasets/nl_benchmark/vNEW
# After complete image inspection, fill annotation_decisions.json and the guide/card.
python -m agents.benchmark_revision finalize --dataset datasets/nl_benchmark/vNEW
```

The initial draft producer hash was recorded before follow-up formatting,
polarization-case normalization and support-summary code changes; the release
separately records the actual finalization method hash. All physical polarization
identities are unchanged. The new assignment lock was written before new image
inspection, while the source bundle's older lock records its AIS-informed selection.

Independent complete-area review and adjudication remain necessary before
exporting validation truth, selecting thresholds/buffers, fitting an accepted
detector or claiming improvement. The existing evaluator rejects all current
provisional metric exports. This work advances data preparation and annotation
for this independent project while preserving that evidence boundary.

Initial verification: 535 local tests passed (two slow tests deselected), lint,
formatting and strict types passed. The project checkout rehashed 62 unique
source/derived inputs and nine private supporting files, checked native pixels
and geographic object centres in every new area, preserved parent objects/events
and confirmed that provisional metric export is rejected. No locked test image
was opened. The [sanitized evidence ledger](evidence/nl_benchmark_v020_20261011.json)
records release/method/bundle hashes and scoped counts. These checks verify
accounting and software; scientific label correctness remains unreviewed.
