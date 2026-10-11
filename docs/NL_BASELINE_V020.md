# Provisional v0.2.0 development detector baseline

This Step 4 replay uses the pinned pretrained xView3 ensemble on the frozen
[v0.2.0 NL benchmark](../datasets/nl_benchmark/v0.2.0/DATASET_CARD.md).
**Results are provisional. No independent SAR reviewer is available. Vessel
precision, total recall, false-alarm density and correct identities remain
unmeasured; no operating threshold is selected.**

The development scope is 19 complete selected areas: nine training and ten
validation areas. Eleven have VV/VH input; eight have HH/HV input unsupported by
this detector. The two Hopedale test areas stay locked. Their metadata enters
split checks; their imagery and embedded review previews are excluded from
baseline verification and inference. Frozen dataset labels and hashes are
unchanged.

## Saved replay results — 11 October 2026

Run `20261011_v020_replay_01` executed code revision
`d66d3a6aeb63a39edf9fb4cf98b16afac37e7ef9`, on
CUDA with Python 3.13.7 / PyTorch 2.6.0+cu124 and an RTX 4070 Laptop GPU.
Eleven VV/VH areas completed 17 native contexts, with zero failed areas; eight
HH/HV areas were unsupported. All 19 cases remain unmeasured.

Overall development water processing coverage is **55.79%** of 320.058 km².
Validation processing coverage is **62.23%**, or 111.283 of 178.819 km².
Independently measured coverage is **0%**. These are acquisition-area exposures,
not the union of provincial sea area or surveillance coverage. Selected areas
within the same native acquisition group have no positive-area overlap; repeat
geography on different dates remains a separate exposure.

At the unchanged 0.15 descriptive threshold and 300 m policy, the supported
areas produced:

| Area | Split | Water km² | Returns | Coastal withheld | Reference-land returns | Eligible | Inference seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical St. John's harbour | Train | 4.626 | 1 | 0 | 0 | 1 | 68.19 |
| Historical Eastern Avalon offshore | Train | 26.611 | 0 | 0 | 0 | 0 | 20.90 |
| Bonavista coast | Validation | 10.096 | 4 | 1 | 3 | 0 | 11.22 |
| Bonavista offshore | Validation | 26.554 | 0 | 0 | 0 | 0 | 25.65 |
| Lewisporte coast | Validation | 8.257 | 4 | 3 | 1 | 0 | 24.66 |
| Bay of Exploits water | Validation | 13.103 | 2 | 0 | 2 | 0 | 48.26 |
| September 22 St. John's, `scene_01_area_01` | Train | 3.695 | 5 | 3 | 1 | 1 | 13.21 |
| September 22 offshore, `scene_01_area_02` | Train | 26.330 | 2 | 0 | 0 | 2 | 10.61 |
| October 4 St. John's, `scene_02_area_01` | Train | 6.010 | 5 | 3 | 1 | 1 | 11.12 |
| September 27 northern NL, `scene_06_area_01` | Validation | 26.636 | 1 | 0 | 0 | 1 | 11.48 |
| September 27 northern NL, `scene_06_area_02` | Validation | 26.638 | 1 | 0 | 0 | 1 | 10.60 |

Training produced 13 returns; validation produced 12. In validation, four were
withheld coastal-water returns, six were reference-land returns and two remained
eligible. There were zero policy abstentions in the processed areas. These
counts cannot establish how many returns are vessels, how many vessels were
missed, whether exclusions were correct, or a false-alarm rate. Unsupported
HH/HV cases have no predictions: zero placeholder counts in their serialized
scores must not be interpreted as empty or correctly rejected images.

Validation strata, with independently measured coverage **0% in every row**:

| Dimension | Stratum | Areas | Water km² | Processing coverage |
| --- | --- | ---: | ---: | ---: |
| Region | Newfoundland | 8 | 148.123 | 75.13% |
| Region | Labrador | 2 | 30.696 | 0% |
| Selection setting | Coastal | 6 | 72.404 | 43.45% |
| Selection setting | Offshore | 4 | 106.416 | 75.01% |
| Polarization | VV/VH | 6 | 111.283 | 100% |
| Polarization | HH/HV | 4 | 67.536 | 0% |
| Acquisition season | Autumn | 6 | 111.474 | 66.95% |
| Acquisition season | Summer | 4 | 67.346 | 54.42% |

Wall time was 372.11 seconds, excluding the initial dataset/model/reference
integrity checks. Model loading took 14.11 seconds. Per-area sampled process RSS
peaked at 3.16 GiB; CUDA allocated/reserved peaks were 6.43/6.57 GiB. Inference
timings exclude shoreline loading and later scoring, vary by context count and
include first-area warm-up. These are one replay's observations, not statistical
latency bounds or full-swath throughput; host load and repeated execution can
change them. Sampled RSS is absolute process memory at 100 ms intervals and can
miss shorter peaks.

## AIS supporting observations and uncertainty

The new supported areas have the following scene-start aligned subset results
at 100 m. Parent areas have no aligned AIS in their frozen snapshots and retain
null supporting metrics. The denominator includes all aligned positions within
the image area, including berth/shoreline positions; it is not an exhaustive
independently confirmed water-vessel count.

| Area | Aligned AIS | Raw proximity matches | Conditional 95% interval | Post-policy matches | Conditional 95% interval |
| --- | ---: | ---: | --- | ---: | --- |
| `scene_01_area_01` harbour | 28 | 1/28 | 0.09%–18.35% | 0/28 | 0%–12.34% |
| `scene_01_area_02` offshore | 1 | 1/1 | 2.5%–100% | 1/1 | 2.5%–100% |
| `scene_02_area_01` harbour | 23 | 1/23 | 0.11%–21.95% | 0/23 | 0%–14.82% |
| `scene_06_area_01` northern | 1 | 0/1 | 0%–97.5% | 0/1 | 0%–97.5% |
| `scene_06_area_02` northern | 1 | 0/1 | 0%–97.5% | 0/1 | 0%–97.5% |

The offshore training area has two returns and one observed track under the
wider assignment gate: one geometric assignment and two ambiguous candidate
returns, including the unmatched competitor. Independent review must determine
whether these returns are separate targets, duplicates or clutter. Each northern
validation area has one geometric assignment under the wider gate, despite its
zero matches at 100 m. A proximity-gate disagreement does not itself establish a
missed vessel or wrong identity. The northern cases reuse the scene represented
in the historical 3/3 evidence and add no independent examples to that result.

Intervals assume independent binomial observations conditional on this partial
AIS subset. They do not correct shared scenes, purposive sampling, label error,
motion/geolocation uncertainty or correlated berthed vessels. Results are kept
per area; no pooled regional or cluster-robust accuracy interval is claimed.
HH/HV areas have supporting AIS counts but null detector proximity/association
metrics because inference is unsupported.

Accuracy metrics, vessel misses, policy-added vessel misses and association
correctness are null in every case. All nine validation threshold trials are
inadmissible; the selected threshold is null. Published software defaults and
operational alert settings are unchanged.

The [public evidence ledger](evidence/nl_baseline_v020_20261011.json) records the
case/grid accounting, supporting intervals, resource observations and source
hashes. The complete local report is
`data/benchmarks/nl_baseline/20261011_v020_replay_01/baseline.json`, SHA-256
`49cedf985a9929d848b86ee3ef0e3a3442313c9d4824f6f5e04d1588da7ac8ec`.
Each of its eleven prediction files was rehashed against its report entry, and
executing source/configuration hashes matched after the run. Frozen dataset
release bytes and assignments remain unchanged.

Local validation passed 539 tests (two slow model-loading tests deselected),
with two expected Rasterio georeferencing warnings. Behavioral regressions verify
that missing test assets cannot prevent development-only verification, altered
development/frozen AIS bytes are rejected, low-score cached returns do not enter
the fixed-threshold buffer trials, and provisional accuracy/threshold outputs
remain unavailable. Isolated review and CI verify software separately from the
real project replay and independent scientific review.

## Method and evidence boundary

Protocol 3 uses native 2048-pixel production-aligned contexts, joint VV/VH
validity, LUT sigma0 calibration, GCP geolocation and production deduplication.
Every context is checked against other splits and the 1000 m separation zone
before native pixels are read. Complete selected areas supply denominators;
unsupported and failed areas remain in processing coverage.

The unchanged descriptive baseline uses strict objectness greater than 0.15 and
a 300 m coastal buffer. Frozen raw caches retain scores above 0.05, allowing the
documented threshold grid to be inspected without rerunning inference. These
objectness scores are uncalibrated. Returns and coastal/land exclusions are
descriptive counts; their class correctness requires reviewed labels.

Physical-water area is conditional on OSM shoreline polygons and valid imagery.
Sparse regional imagery controls do not establish full-area boundary correctness.
Area names, geographic and coastal/offshore strata reflect selection intent.
Season does not establish sea state, ice, weather or incidence angle. The newer
areas were enriched using AIS availability, and prior development exposure and
unverified upstream pretraining overlap limit generalization.

AIS is aligned to the exact scene start using the bounded production motion
rules. Frozen archive files are verified against their recorded hashes before
use; all geographic competitors are retained before alignment. Selected-area
support uses a fixed 100 m proximity rule and conditional 95% Clopper–Pearson
intervals. Geometric association uses the separate 500 m base gate plus heuristic
AIS uncertainty. Its competitor pool is restricted to the selected area and
cannot characterize ambiguity at area boundaries or throughout a swath.

The [Step 3 annotation support](NL_BENCHMARK_V020.md) used catalogue-midpoint
alignment and an illustrative 500 m gate. The historical [3/3 full-scene
result](AIS_SUBSET_RESULT.md) also uses 500 m. These have different denominators
and must not be pooled with this selected-area replay. AIS silence, rejected
tracks and unmatched returns cannot label a detection as false or establish
vessel intent.

## Reproduction

Run from the project containing the authorized frozen SAR/AIS inputs and pinned
model. Use a fresh output path for each run:

```sh
python -m agents.baseline --dataset datasets/nl_benchmark/v0.2.0 --artifact-root E:/TritonEye --output data/benchmarks/nl_baseline/NEW_v020_replay --device cuda
python -m agents.baseline_evidence --report data/benchmarks/nl_baseline/NEW_v020_replay/baseline.json --output docs/evidence/NEW_v020_baseline.json
python -m pytest tests/unit -q -m "not slow"
```

The report records dataset/release, code, configuration, shoreline, model, input
and frozen telemetry hashes, exact product IDs, context footprints and software
versions. Public evidence omits prediction coordinates, native imagery and AIS
identities/positions. Large local caches remain ignored by Git. Synthetic
fixtures validate the isolated code gate; private data stays in this project.

## Completion boundary

The [provisional research objective](NL_BASELINE.md#provisional-operating-objective)
requires a false-alarm upper bound at most 0.1/km², recall lower bound at least
70%, at least 90% processing and measured coverage in each required stratum,
30 independently labelled vessels and five independent connected blocks.
These are portfolio research choices, not C-CORE acceptance requirements.
Threshold search stays blocked; the independent-review export guard remains in
force. HH/HV compatibility, sufficient reviewed complete-area validation and
vessel identity adjudication are required before completing Step 4's accuracy
and operating-point deliverables. Locked test scoring follows a separately
frozen eligible operating point and evaluation protocol.
