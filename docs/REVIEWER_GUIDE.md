# TritonEye reviewer guide

Independent Newfoundland and Labrador maritime AI portfolio, relevant to the
C-CORE COREsight AI Engineer role. **Research prototype, not a C-CORE product,
certified surveillance system or operational dark-vessel detector.** Internal
C-CORE acceptance criteria were not available; no endorsement or compliance is
claimed. The NL study polygon is not an authoritative jurisdiction boundary.

Evidence reviewed: **11 October 2026**, against main baseline
`04cc59b0f7322831199501c86e8c8271ad2776d5`. This guide is a documentation package,
not a new inference, annotation or training experiment. Historical results below
retain their own acquisition dates, execution versions and source hashes.

For the subsequent archive refresh and local image-only handoff, see the
[ranked SAR–AIS overlap audit and review bundle](OVERLAP_REVIEW.md). Its preparation
counts are separate from the measured results below; independent labels and
correlation improvement are still blocked.

The subsequent [AIS-supported coastal-buffer analysis](AIS_COASTAL_ANALYSIS.md)
measures exclusions within two observed coastal AIS samples and replays policy
effects on cached detector proximity matches. It provides supporting sensitivity
evidence; complete-area false alarms and missed vessels remain unavailable.

The subsequent [Step 3 benchmark expansion](NL_BENCHMARK_V020.md) freezes 21
selected areas, including nine new complete image-first passes and five
provisional vessel interpretations with supporting AIS proximity. Independent
review is still unavailable; validation truth export and accuracy remain blocked.
The frozen v0.1.0 statements below retain their historical scope.

The [provisional v0.2.0 detector baseline](NL_BASELINE_V020.md) subsequently
runs the expanded development set and records processing, resources, policy
exclusions and AIS supporting proximity. Its accuracy metrics remain null and
threshold selection remains blocked by review and coverage requirements.

The [exploratory development experiment protocol](DEVELOPMENT_EXPERIMENTS.md)
records three limited xView3 fits and three frozen-CROMA token-head fits. Provisional
positive-only fits cannot establish false-alarm reduction or measured improvement.
The independent-review guard and locked test assignments still apply.

The subsequent [AllenAI NL source audit](ALLENAI_NL_AUDIT.md) prepares 31
alignment-review windows and 23 point patches from three exact 2022 acquisitions.
It verifies the separate CC BY archive's licence declaration but finds no NL
source-window overlap there. Older NL annotation permission and completeness
remain unresolved; this quarantined package is not adopted training data or
independent benchmark truth.

## 1. Built and verified — engineering evidence

The working system is a sequential Python pipeline:
`ingest → inference → correlate → evaluate → report`. Software tests, processing
replays and provenance checks establish behavior, **not maritime accuracy**.

| Component | Implemented boundary and verification | Evidence to inspect |
| --- | --- | --- |
| SAR ingestion and inference | Real Sentinel-1 VV/VH calibration, native tiling, georeferencing, pinned pretrained xView3 model, NL-restricted GeoJSON and explicit incomplete-processing handling. Not a detector trained by this project. | [Pipeline](../agents/pipeline.py), [audit and replay ledger](AUDIT.md), [evaluation limitations](../EVALUATION.md) |
| Acquisition reliability | Product UUID propagation, versioned outcomes, recorder freshness/gaps and distinct processing/measurement states. Original SAFE/COG aliases do not create independent physical slices; malformed metadata is guarded. | [Reliability and reproduction](ACQUISITION_RELIABILITY.md), [watcher regressions](../tests/unit/test_scene_watch.py), [outcome regressions](../tests/unit/test_watch.py), [exact-ID replay evidence](evidence/acquisition_replay_20261005.json) |
| Shoreline and coastal policy | Physical reference-land/water classification is separate from candidate eligibility; coastal returns are retained for research. St. John's basin and The Narrows are open water. CanVec is comparison-only, not a province-wide replacement. | [Coastal policy and coverage](COASTAL_POLICY.md), [St. John's checks](ST_JOHNS_SHORELINE.md), [policy regressions](../tests/unit/test_coastal_policy.py) |
| SAR–AIS correlation | Bounded UTC/motion alignment, one-to-one geometric assignment and ambiguity handling. A separate uncertainty-aware research comparison and detector-blind identity review workflow exist; production matching is unchanged. | [Correlation study and review contract](TARGET_CORRELATION.md), [comparison regressions](../tests/unit/test_association_study.py) |
| Benchmark and experiments | Frozen v0.1.0 SAR review pilot with source licences/hashes and acquisition/geographic split locks; baseline/threshold evaluation code; real frozen CROMA radar-feature integration. Independent annotation, fitting and model improvement are not delivered. | [Dataset card](../datasets/nl_benchmark/v0.1.0/DATASET_CARD.md), [baseline protocol](NL_BASELINE.md), [model development](MODEL_DEVELOPMENT.md) |
| Reproducibility and analyst handoff | Mission provenance, optional MLflow tracking, retained failure evidence, HTML reports, unit tests, lint/type checks and Docker/CI. This is not a complete production MLOps service. | [Synthetic report](../examples/nl-synthetic-report.html), [CI configuration](../.github/workflows/ci.yml), [software CI for the baseline commit](https://github.com/haseebkn/TritonEye/actions/runs/38126241155) |

The linked baseline CI completed successfully for unit tests, lint/types and
Docker. It verifies that commit's software, not an independently labelled model
benchmark. Use the commands below to check another checkout; do not assume an
older green run certifies newer code.

## 2. Measured — small and explicitly scoped

| Saved experiment | Observation | What it does not establish |
| --- | --- | --- |
| Full-scene SAR–AIS proximity comparison | **3/3 AIS-supported vessel-scene observations across two physical VV/VH scenes**, at **500 m**: 2/2 on 27 September and 1/1 on 2 October 2026. Raw and open-water-eligible subset recall are 100%; conditional exact two-sided **95% Clopper–Pearson interval: 29.2%–100%**. | Not three independently confirmed vessels, correct identities, overall recall, precision, false-alarm rate, improvement, coastal performance or province-wide/Labrador validation. |
| Frozen CROMA radar-feature probe | Six VV/VH development areas from three Newfoundland acquisitions; 96 windows encoded completely, finite features and zero maximum repeated-window token difference on that run. | No vessel head, gradient update, fine-tuning or detection improvement. Cross-device bitwise repeatability is not promised. |
| Historical selected-area detector baseline | Six supported VV/VH areas completed; four HH/HV areas were unsupported and two locked-test areas excluded. Selected validation water processing coverage was 65.40%, with 0% independently measured coverage. | Processing coverage and descriptive returns are not vessel accuracy or proof that empty offshore predictions were correct. |

The SAR–AIS result comes from saved 6 October executions, inspected on 11 October.
Its AIS coverage is **partial and self-reported**; matches can be accidental.
`n=3` counts vessel-scene examples, not a verified cross-date distinct-identity
count. Repeat runs and original SAFE/COG aliases are counted once. The interval
assumes independent trials and does not correct clustering, selection bias or
AIS/geolocation error. No cluster-robust regional interval is claimed from two
scenes. All three references were source-relative open water; shoreline imagery
controls were outside those masks' scope. No independent reviewer is available.

Inspect [the complete AIS result and local verification instructions](AIS_SUBSET_RESULT.md)
and its [sanitized, hashed ledger](evidence/ais_subset_20261011.json). It uses a
different radius from the baseline's 100 m selected-area AIS metric; do not pool
them. Feature measurements and processing/resource observations have separate
owners in [model development](MODEL_DEVELOPMENT.md#real-feature-probe-results)
and [the historical baseline](NL_BASELINE.md#real-replay-results), with
[feature evidence](evidence/model_research_20261006.json) and
[baseline evidence](evidence/nl_baseline_20261005.json).

Sparse imagery controls and coastal-buffer replays are additional engineering
observations, not measured vessel performance or province-wide validation.
See the authoritative [current coverage inventory, source provenance and
remaining Step 2 blocker](COASTAL_POLICY.md#current-frozen-pilot-development-coverage-11-october).

## 3. Blocked or not implemented — why and what would resolve it

| Missing outcome | Current reason | Required next evidence or implementation |
| --- | --- | --- |
| Vessel precision, overall recall and false alarms per water km² | No independently reviewed, complete-area vessel/non-vessel truth. Frozen v0.1.0 has zero confirmed training vessel labels; later AIS proximity matches do not change that release. | A new versioned, reviewed NL dataset including detector misses, clutter/ice uncertainty and acquisition/geographic holdouts. [Annotation boundary](NL_BENCHMARK.md) |
| Correct SAR–AIS identities and measured correlation improvement | No independently reviewed association cases; sparse local AIS lacks sufficient competition/track context. A proximity match is not a correct-identity label. | Area-wide co-temporal tracks, supported identity decisions and a paired geometric/uncertainty comparison on held-out cases, including wrong identities, ambiguity, misses and coverage. [Association protocol](TARGET_CORRELATION.md) |
| Supervised detector or foundation-model improvement | Exploratory bias/token-head fits now use one provisional VV/VH training vessel. They demonstrate gradient updates, but lack legitimate negative supervision, independently reviewed accuracy and a useful operating-point comparison. | Reviewed, diverse development data, meaningful detector fine-tuning and reproducible comparisons under the same splits and operating constraints. [Executed development study](DEVELOPMENT_EXPERIMENTS.md) |
| Low-false-positive threshold and coastal buffer selection | Missing reviewed labels and sufficient independent validation blocks; partial/unsupported sensor coverage. No operating point has been selected from detector counts. | Validation under documented false-alarm **and minimum recall/coverage** constraints, then a separately frozen final-test protocol. These are provisional research targets, not C-CORE requirements. [Operating objective](NL_BASELINE.md#provisional-operating-objective) |
| General NL offshore/HH/HV capability and shoreline correctness | The current detector and feature model have no validated HH/HV compatibility; sparse shoreline controls, now including Makkovik, leave large regions unchecked. Historical annotation coverage is not detector support. | Separate labelled HH/HV model/input experiments and broader regional imagery checks beyond these sparse points. [Polarization limits](DATA_SOURCES.md#polarization-availability-constrains-what-can-be-processed) |
| Optical/RF fusion, retraining and operational MLOps | Not implemented: optical/RF fusion, automated retraining/promotion, monitored staged deployment, AWS training, DVC data versioning, the SQLite task bus and cloud commissioning. These are implementation gaps, not features merely awaiting credentials. | Authorized data, implemented pipelines and deployment/monitoring/rollback validation. The [agent-engine document](../agents.md) is explicitly a future blueprint, not delivery evidence. |

Upstream detector-pretraining overlap must also be audited before claiming an
independent validation set; split locks alone do not prove absence of pretraining
leakage. The locked final test has not been scored or used to choose a threshold.

Independent review is currently unavailable. Prepare additional labels and review
bundles without pretending machine/self review satisfies that requirement.
Historical SAR work can continue while prospective AIS recording proceeds;
additional licensed historical AIS may be needed, and access/coverage must be
confirmed. [Sources and credentials](DATA_SOURCES.md). Existing locked test data
must not be used for tuning or quietly reassigned.

Automatic operational alerts remain disabled. Unmatched returns are
**unassociated candidates**, not confirmed dark vessels or evidence of intent.
Fewer displayed candidates, withheld coastal vessels, an empty output or a
running AIS container do not establish low false positives or complete coverage.

## Reviewer walkthrough and access limits

1. Read the three sections above, then the linked AIS result for its exact claim.
2. Open the [offline synthetic report](../examples/nl-synthetic-report.html) to
   inspect analyst handoff. It uses simulated targets and is not real evidence.
3. Inspect the dataset card, saved evidence ledgers and source/test links. Dates,
   versions and hashes distinguish historical experiments from current code.
4. With Python 3.12/3.13 and the [README environment instructions](../readme.md#offline-synthetic-demonstration),
   run from the repository root:

```sh
python -m agents.pipeline --mock
python -m pytest tests/ -q -m "not slow"
python -m ruff check agents tests
python -m black --check agents tests
python -m mypy agents tests
```

The mock run creates a new synthetic mission under `missions/`; it requires no
credentials or trained model once dependencies are installed. Real replay and
image-byte verification additionally require local SAR products, model weights,
shoreline data and, where applicable, authorized AIS snapshots. Public metadata
and hash summaries are not full raw inputs. Keep credentials, private identities,
AIS positions and restricted imagery out of Git; do not redistribute data just
to make a reviewer bundle portable. See [verification and limits](../EVALUATION.md).

For C-CORE, this demonstrates relevant SAR/geospatial engineering, auditable
target association and reproducible ML experiments. It does **not yet demonstrate
the role's complete supervised model-development or production MLOps scope**.
