# NL SAR–AIS overlap refresh and detector-blind review

Prepared 11 October 2026 for independent TritonEye target-correlation research.
This is **not operational acceptance evidence, reviewed vessel
truth, an operational surveillance capability or measured model improvement**.
The existing [3-of-3 proximity result](AIS_SUBSET_RESULT.md) remains unchanged.

## Refreshed archive and ranked candidates

The frozen archive contains **198,447 valid unique reports**, on 37 recording
dates, from 18 August 20:08:37 UTC through **11 October 09:45:29.661491 UTC**.
Recording was intermittent, not continuous. 19,621 legacy timezone-naive reports
were explicitly interpreted as UTC, matching the existing aligner. Files were
captured sequentially as complete-line prefixes, not as a transactional snapshot.
The live recorder reported fresh at the start; that does not establish past
reception coverage. A replay uses the frozen bytes, not later live observations.

Fresh public Copernicus metadata returned 1,794 catalogue entries: **897 physical
slices after original/COG deduplication**. Before overlap scoring, 41 slices were
excluded by the locked Hopedale acquisition group or buffered test geography.
Of the remaining 856 slices:

- **15 temporal candidates, across 10 native acquisition groups**;
- 9 with sparse supporting observations only;
- 832 with no observed overlap in this partial archive.

Those are screening states, not measured scenes or vessel detections. Native
groups contain adjacent slices; neither slices nor repeated identities are
independent accuracy trials. The [aggregate ledger](evidence/overlap_review_20261011.json)
provides exact UUIDs, start/end times, versions and hashes without private AIS.

Ranking is lexicographic: close physical-water tracks, short brackets, nearby
competing-track opportunities, total aligned tracks, then UTC and UUID. The
provisional research screen retrieves ±30 minutes, aligns using the unchanged
production motion/freshness checks, counts reports within 120 seconds and
brackets no wider than 240 seconds. Full histories are aligned **before** NL and
exact-footprint filtering. The reference is the catalogue interval midpoint,
not an extracted per-pixel radar time and not the production evaluator's scene
start time. A pair within 5 km is a scene-level opportunity to review competing
identities, not a demonstrated ambiguous SAR association.

| Rank | Start UTC (2026) | UUID prefix | Pol. | Close water / open water | Short brackets | Complete bundle areas |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 09-22 21:21:46 | bae55045 | VV/VH | 23 / 4 | 14 | 2 |
| 2 | 10-04 21:21:46 | 0b5ad75c | VV/VH | 16 / 0 | 10 | 1 |
| 3 | 10-02 09:41:05 | 19416098 | HH/HV | 12 / 1 | 9 | 2 |
| 4 | 10-08 09:41:01 | 5415d10e | HH/HV | 9 / 1 | 3 | Not downloaded |
| 5 | 09-18 09:56:56 | 01876305 | HH/HV | 8 / 2 | 5 | Not downloaded |
| 6 | 10-02 21:38:30 | 75bcc3eb | VV/VH | 7 / 0 | 1 | Failed raster boundary |
| 7 | 09-18 09:56:31 | 4b6f0ce7 | HH/HV | 6 / 4 | 2 | 2 |
| 8 | 09-20 21:38:29 | 18f5ab4b | VV/VH | 5 / 0 | 3 | Not downloaded |
| 9 | 10-05 10:05:25 | b2c7b3b8 | HH/HV | 5 / 0 | 2 | Not downloaded |
| 10 | 09-27 21:30:48 | 31773a03 | VV/VH | 3 / 0 | 0 | Not downloaded |
| 11 | 09-27 21:31:13 | bca72678 | VV/VH | 2 / 2 | 2 | 2 |
| 12 | 10-05 10:04:35 | dd031ca1 | HH/HV | 2 / 2 | 2 | Not in this bundle |
| 13 | 09-19 09:48:24 | 18b64506 | HH/HV | 2 / 1 | 1 | Not downloaded |
| 14 | 09-20 21:39:19 | d69ee41c | VV/VH | 2 / 0 | 0 | Not downloaded |
| 15 | 10-05 10:05:00 | a0430ef4 | HH/HV | 2 / 0 | 0 | Not downloaded |

Water is OSM-reference-relative. Open water applies the separate 300 m policy;
coastal cases are retained for review. All candidates have **gap-or-unknown
historical heartbeat bracketing**, not established continuous receiver coverage.
Supporting reports can still exist. The eight St. John's shoreline controls do
not validate northern NL waters or Labrador; `outside_scope` is not a pass.
CanVec remains comparison-only; no shoreline, threshold or alert policy changed.

## What the bundle actually contains

Local directory: `data/research/ais_sar_overlap_20261011_final_v01/`.
Open **`image_first/review.html`**. It embeds 18 native previews, with no detector
predictions, AIS markers, identities or presumed labels. The preparer previously
saw telemetry; the area selection is AIS-informed. It is detector-blind, **not
fully AIS-blind, independent review or an unbiased detector-recall benchmark**.

Six cached acquisition groups were proposed, one slice per group. Five products
yielded nine non-overlapping complete areas: **five VV/VH and four experimental
HH/HV areas**. All have 262,144 valid pixel positions in both bands. The sixth
product's proposed area failed the native-raster boundary check and is not in
the annotation template. Catalogue footprints alone would have missed that.
The successfully prepared areas cover about **238.62 km²**, including **158.68 km²
of reference physical water** and **143.84 km² after the 300 m policy**. These are
geometric preparation denominators, not reviewed or successfully detected water.
The packet includes southeastern Newfoundland harbour/coastal areas and northern
NL open-water/coastal areas; no province-wide accuracy or Labrador-specific
benchmark result is established.

The October 2 HH/HV SAFE was downloaded into `data/raw/`. Further downloads were
deferred because disk space was limited; only the interrupted temporary ZIP was
removed. Exact cached products and previous archives were preserved. `--download`
now checks product size and free space for archive plus extraction, reserving
1 GiB. The ranking still includes uncached and failed candidates. No replacement
UUID is silently used. HH/HV imagery preparation does not validate production
detector compatibility or usable offshore detection coverage.

The bundle records source measurement/calibration hashes, selection before image
inspection, native windows, GCP geometry, valid pixels, polarization, source
licence, per-chip hashes, all-area review cells and explicit failures. Calibration
is sigma0 LUT only, without thermal-noise or terrain correction; previews use
the fixed −30 to +5 dB stretch. The bundle's manifest and hash inventory bind the
first-pass template to the images. None of the templates are completed labels.

## Review workflow and limits

1. Give a reviewer **only `image_first/`** and the repository
   [annotation guide](../datasets/nl_benchmark/v0.1.0/ANNOTATION_GUIDE.md).
   Use the manifest-listed areas, not failed/partial directories from other runs.
2. Inspect every 256×256 cell in both native bands. Fill
   `first_pass_template.json`: vessels, fixed structures, clutter, identifiable
   ice and unresolved objects, including objects without AIS or missed by a
   detector. Record empty-area decisions, reviewed cells, qualifications,
   timestamps and the source chip hash; do not force uncertain objects to vessels.
3. Freeze and hash those decisions before opening `telemetry_second_pass/`.
   That separate directory retains the **whole aligned scene identity pool** and
   report histories, with ROI membership flags—not only the expected vessel or
   tracks inside a small chip. Review wrong/missing/ambiguous identities there.
4. No independent reviewer is currently available. A self/AI pass remains
   provisional; do not feed it through an independent-truth metric gate. New
   areas are a development review queue, not a released training/validation/test
   dataset. Preserve existing acquisition/geographic assignments: a released
   validation group must not become training data. Hopedale test pixels were not
   opened, relabelled, tuned against or scored by this workflow.
5. Only after suitable reviewed examples and a held-out split exist, compare the
   unchanged geometric matcher with the uncertainty-aware control using the
   [correlation protocol](TARGET_CORRELATION.md). Report wrong identities, missed
   associations, ambiguity and uncertainty. Unmatched returns remain
   **unassociated candidates**. There is no new measured improvement here.

Private reports and images remain ignored by Git; main contains the reusable
method, tests and aggregate evidence. An external reviewer needs the local
image-only package, not `.env`, raw AIS, mission outputs or the second-pass
telemetry directory before their first pass.

## Reproduce and validate

From the repository checkout with its dependencies and OSM cache:

```bash
# New snapshot and fresh public catalogue; output must be a NEW directory.
python -m agents.overlap_audit --output data/research/overlap_new --bundle-count 6

# Exact frozen input replay; no new catalogue query or live archive rows.
python -m agents.overlap_audit --replay data/research/ais_sar_overlap_20261011_01 --output data/research/overlap_replay --bundle-count 6

# Add --download only to fetch missing selected SAFE products with existing
# COPERNICUS_USER/PASS in the ignored .env and sufficient free disk space.
pytest tests/ -q -m "not slow"
ruff check agents tests
black --check agents tests
mypy agents tests
```

The saved execution began before the feature commit; its method file SHA-256,
not the baseline `code_commit` field alone, identifies the executed implementation.
The unchanged method was subsequently committed in `8332e43`. Replay matched all
15 ranked UUIDs, acquisition intervals and support counts. All bundle file hashes
and 18 preview dimensions were checked; no vessel annotation was inferred from
that validation. Historical frozen benchmark releases and measured outputs were
not overwritten.
