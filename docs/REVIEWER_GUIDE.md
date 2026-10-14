# TritonEye reviewer guide

Independent Newfoundland and Labrador maritime AI portfolio, relevant to the
C-CORE COREsight AI Engineer role. **Research prototype, not a C-CORE product,
certified surveillance system or operational dark-vessel detector.** Internal
C-CORE acceptance criteria were not available; no endorsement or compliance is
claimed. The NL study polygon is not an authoritative jurisdiction boundary.

Current summary refreshed **14 October 2026**, against main baseline
`d44fe4a6cdaf717e34223ba4f03c80323cf9be02` (PR #25), plus the reserved-scene
transfer collection described below. The current benchmark is
v0.2.0: 21 areas, six exploratory fits, local lifecycle rehearsals and quarantined
AllenAI adoption checks. Independent accuracy and model improvement remain
unmeasured for vessel detection. A separate supervised
[Statoil HH/HV ship versus iceberg baseline](STATOIL_BASELINE.md) now reports
87.97% exploratory chip classification accuracy against publisher labels; this
does not validate full-scene detection, geographic independence or promotion.
The subsequent [real HH/HV transfer audit](STATOIL_TRANSFER.md) scores fifteen
existing development objects across eight NL areas using that frozen classifier.
Input calibration replays match, but missing reviewed truth and iceberg labels
keep native accuracy, false-alarm reduction and detection improvement unmeasured.
See the [experimental HH/HV candidate generator](HH_HV_CANDIDATES.md) for the
subsequent frozen population, replay evidence and future filter-comparison limits.
The [additional HH/HV development reservation](HH_HV_DEVELOPMENT.md) fixes eight
new acquisitions and four geographic groups before scoring. The subsequent
[transfer collection](RESERVED_HH_HV_TRANSFER.md) contains sixteen inspected
native windows, 25 unresolved returns and sixteen context samples. All 32 native
band calibrations replay exactly. No vessel or plausible iceberg label could be
supported; no classifier scores were generated. This separate provisional
collection does not replace the benchmark or locked test.
The subsequent [source-supported native batch](NATIVE_SOURCE_ANNOTATION.md) adds nine
inspected windows across five acquisitions: two provisional vessels, seven unresolved
compact returns and two background contexts. Eighteen calibration bands replay exactly;
no iceberg label was defensible and no scores were generated. All remain ignored
for accuracy and training approval, with no independent reviewer.
Historical results below retain their acquisition dates, execution
versions and source hashes; this documentation refresh does not rerun inference.

For a five-minute presentation, open the [real Newfoundland case](../examples/nl-real-case.html)
and follow the [demonstration walkthrough](PORTFOLIO_DEMO.md). The separate
[technical explanation and contribution notes](TECHNICAL_EXPLANATION.md) prepare
the evidence questions and distinguish recorded AI assistance from the owner's
personal implementation and understanding, which must be demonstrated by the owner.

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

The [AllenAI image-first curation workflow](ALLENAI_CURATION.md) subsequently
prepares those 31 whole areas without publisher points or detector/AIS overlays.
All pixels initially remain ignored. The subsequent
[AI-only first pass](ALLENAI_FIRST_PASS.md) inspects all 31 areas, discloses prior
label exposure and freezes five provisional vessel-like interpretations with
explicit uncertain/nodata ignores. It is not independent truth or usable training
data. See the subsequent [adoption checks](ALLENAI_ADOPTION.md) for AI-only
publisher-point decisions, native calibration replay and remaining adoption blockers.
Hash-bound sequential review, provisional class/ignore masks and an
[unsent licence/completeness clarification request](ALLENAI_CLARIFICATION_REQUEST.md)
are implemented. The original quarantine, independent-review guard and locked
test set remain unchanged; training and accuracy exports stay blocked.

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
| Benchmark and experiments | Frozen v0.2.0 pilot with 21 selected areas, source licences/hashes and acquisition/geographic split locks; expanded detector baseline; three limited xView3 bias fits and three frozen-CROMA token-head fits. Training used one provisional VV/VH vessel. Independent accuracy and useful model improvement remain unmeasured. | [Current dataset card](../datasets/nl_benchmark/v0.2.0/DATASET_CARD.md), [executed baseline](NL_BASELINE_V020.md), [six exploratory fits](DEVELOPMENT_EXPERIMENTS.md) |
| Local model lifecycle | Paired validation shadow inference, reviewed promotion gates and transactional model/threshold rollback with operator history. Successful promotion is verified with synthetic fixtures; actual NL promotion remains blocked. | [Lifecycle commands and limits](MODEL_LIFECYCLE.md), [behavioral regressions](../tests/unit/test_model_lifecycle.py) |
| Reproducibility and analyst handoff | Mission provenance, optional MLflow tracking, retained failure evidence, HTML reports, unit tests, lint/type checks and Docker/CI. This is not a complete production MLOps service. | [Real case presentation](../examples/nl-real-case.html), [synthetic report](../examples/nl-synthetic-report.html), [CI configuration](../.github/workflows/ci.yml), [PR #20 main CI](https://github.com/haseebkn/TritonEye/actions/runs/38227936504) |

The linked PR #20 main CI completed successfully for unit tests, lint/types and
Docker. It verifies that commit's software, not an independently labelled model
benchmark. Use the commands below to check another checkout; do not assume an
older green run certifies newer code.

## 2. Measured — small and explicitly scoped

| Saved experiment | Observation | What it does not establish |
| --- | --- | --- |
| Full-scene SAR–AIS proximity comparison | **3/3 AIS-supported vessel-scene observations across two physical VV/VH scenes**, at **500 m**: 2/2 on 27 September and 1/1 on 2 October 2026. Raw and open-water-eligible subset recall are 100%; conditional exact two-sided **95% Clopper–Pearson interval: 29.2%–100%**. | Not three independently confirmed vessels, correct identities, overall recall, precision, false-alarm rate, improvement, coastal performance or province-wide/Labrador validation. |
| Frozen CROMA radar-feature probe | Six VV/VH development areas from three Newfoundland acquisitions; 96 windows encoded completely, finite features and zero maximum repeated-window token difference on that run. | No vessel head, gradient update, fine-tuning or detection improvement. Cross-device bitwise repeatability is not promised. |
| Historical selected-area detector baseline | Six supported VV/VH areas completed; four HH/HV areas were unsupported and two locked-test areas excluded. Selected validation water processing coverage was 65.40%, with 0% independently measured coverage. | Processing coverage and descriptive returns are not vessel accuracy or proof that empty offshore predictions were correct. |
| Current v0.2.0 detector baseline | Eleven VV/VH development areas processed; eight HH/HV areas unsupported; two locked-test areas excluded. Validation water processing coverage was 62.23%, with 0% independently measured coverage. | The expanded selection changes the denominator; do not interpret its coverage difference as a detector regression or improvement. |
| Exploratory development on v0.2.0 | Six fits and 176 frozen CROMA windows across eleven VV/VH areas. Bias fitting changed scores but retained 14 validation returns at >0.15. CROMA heads produced 2,487–2,601 validation returns. | One provisional training positive and no accepted negatives cannot establish useful generalization, accuracy or false-alarm reduction. |
| Separate Statoil HH/HV classification | 1,604 publisher-labelled chips; angle/duplicate proxy groups kept within splits. On 241 holdout chips, the CNN ensemble had 87.97% accuracy versus 58.92% for band-statistics logistic regression; 17 ships called icebergs and 12 icebergs called ships. | Target-chip classification only. Acquisition metadata and native preprocessing transfer are unavailable; repeated protocol runs do not add independent examples. No full-scene false-alarm or detector-improvement claim. |

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

Roadmap step 7 now has a [local model lifecycle](MODEL_LIFECYCLE.md): versioned
registration, paired validation shadow inference, manual promotion gates and
transactional rollback with incident history. Synthetic rehearsals verify the
successful state transitions. Current NL labels cannot authorize promotion, and
production model configuration is unchanged.

| Missing outcome | Current reason | Required next evidence or implementation |
| --- | --- | --- |
| Vessel precision, overall recall and false alarms per water km² | No independently reviewed, complete-area vessel/non-vessel truth. Frozen v0.1.0 has zero confirmed training vessel labels; later AIS proximity matches do not change that release. | A new versioned, reviewed NL dataset including detector misses, clutter/ice uncertainty and acquisition/geographic holdouts. [Annotation boundary](NL_BENCHMARK.md) |
| Correct SAR–AIS identities and measured correlation improvement | No independently reviewed association cases; sparse local AIS lacks sufficient competition/track context. A proximity match is not a correct-identity label. | Area-wide co-temporal tracks, supported identity decisions and a paired geometric/uncertainty comparison on held-out cases, including wrong identities, ambiguity, misses and coverage. [Association protocol](TARGET_CORRELATION.md) |
| Supervised detector or foundation-model improvement | Exploratory bias/token-head fits now use one provisional VV/VH training vessel. They demonstrate gradient updates, but lack legitimate negative supervision, independently reviewed accuracy and a useful operating-point comparison. | Reviewed, diverse development data, meaningful detector fine-tuning and reproducible comparisons under the same splits and operating constraints. [Executed development study](DEVELOPMENT_EXPERIMENTS.md) |
| Low-false-positive threshold and coastal buffer selection | Missing reviewed labels and sufficient independent validation blocks; partial/unsupported sensor coverage. No operating point has been selected from detector counts. | Validation under documented false-alarm **and minimum recall/coverage** constraints, then a separately frozen final-test protocol. These are provisional research targets, not C-CORE requirements. [Operating objective](NL_BASELINE.md#provisional-operating-objective) |
| General NL offshore/HH/HV capability and shoreline correctness | The current detector and feature model have no validated HH/HV compatibility; sparse shoreline controls, now including Makkovik, leave large regions unchecked. Historical annotation coverage is not detector support. | Separate labelled HH/HV model/input experiments and broader regional imagery checks beyond these sparse points. [Polarization limits](DATA_SOURCES.md#polarization-availability-constrains-what-can-be-processed) |
| Optical/RF fusion, retraining and operational MLOps | Local research promotion, shadow evaluation and rollback are implemented; no real candidate is approved. Optical/RF fusion, automated retraining/production promotion, continuous production monitoring, AWS training, DVC data versioning, the SQLite task bus and cloud commissioning remain unimplemented. | Reviewed validation evidence for local promotion; implemented and commissioned infrastructure for operational deployment. The [agent-engine document](../agents.md) is explicitly a future blueprint, not delivery evidence. |

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
2. Open the [offline real case](../examples/nl-real-case.html) and follow
   [the short walkthrough](PORTFOLIO_DEMO.md). It reuses saved September 22 imagery
   and October 11 inference; it is a selected training-area example, not a test.
   The separate [synthetic report](../examples/nl-synthetic-report.html) uses
   simulated targets to demonstrate the workflow.
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
