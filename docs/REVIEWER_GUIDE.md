# TritonEye reviewer guide

Independent Newfoundland and Labrador maritime AI research project.
**Research prototype, not a certified surveillance system or operational
dark-vessel detector.** No organizational endorsement or operational compliance
is claimed. The NL study polygon is not an authoritative jurisdiction boundary.

## Current summary — 15 October 2026

Engineering baseline: `259d5923ba233e3ba79fb8942e006d6937669ca2`
([PR #44](https://github.com/haseebkn/TritonEye/pull/44)).
[That main commit passed CI](https://github.com/haseebkn/TritonEye/actions/runs/38810857035).
The project can process real supported SAR scenes, retain uncertainty and
provenance, run exploratory model experiments and demonstrate guarded local
model operations. **It has not demonstrated low-false-positive native-NL vessel
detection or controlled detector/correlation improvement.**

- **Built:** sequential SAR/AIS pipeline, coastal policy separation, provisional
  benchmark, six exploratory fits, HH/HV candidate and preprocessing experiments,
  local promotion/shadow/rollback tooling, tests and an analyst demonstration.
- **Measured:** a tiny 3/3 AIS proximity sample across two scenes (conditional
  exact 95% interval 29.2%–100%), processing/resource observations, and a separate
  Kaggle chip-classification comparison. These are not interchangeable metrics.
- **Blocked:** insufficient eligible co-temporal AIS, unresolved source-label
  permission/completeness, uncertain iceberg identities, no independent reviewer,
  and unvalidated model transfer. Public imagery exists; access alone is not the
  whole problem.
- **Preserved:** review/promotion guard, AllenAI quarantine, original hashes and
  locked test assignments. No approved native-NL model or operational alerts.

The current v0.2.0 benchmark has 21 selected areas and five provisional vessel
interpretations. Six VV/VH exploratory fits used only one provisional training
vessel. Separately, the [31-area native HH/HV register](LABEL_ADMISSION.md) records
58 provisional annotations/contexts and zero training/accuracy admissions; the
[later two-window preparation](TRAINING_NATIVE_ANNOTATION.md) adds one unconfirmed
iceberg hypothesis and one coverage abstention, not approved training labels.
Do not combine these different releases into a claimed independently labelled dataset.

The [recorder deployment and collection reservations](TARGETED_AIS_COLLECTION.md)
document a point-in-time working subscription and three planned passes on
October 17, 18 and 20, **after the October 16 application deadline**. Actual
products and usable labels remain pending; no future reception is guaranteed.

For a five-minute presentation, open the [saved real Newfoundland case](../examples/nl-real-case.html)
and follow [the demonstration walkthrough](PORTFOLIO_DEMO.md), including blocked
promotion and successful rollback. Keep the [synthetic example](../examples/nl-synthetic-report.html)
separate. [Technical explanation and contribution notes](TECHNICAL_EXPLANATION.md)
disclose AI assistance; the owner must personally demonstrate their understanding.
Historical evidence below keeps its acquisition dates, execution versions and
source hashes. This documentation refresh does not rerun inference or admit labels.

Current presentation describes an independent project. Source attribution and
frozen historical dataset cards, reservation policies/manifests and evidence
ledgers retain their original wording and hashes; they are historical provenance,
not statements of current affiliation. Earlier Git commits and PR discussions
are not rewritten. New generated reports use independent research wording.

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
| Reproducibility and analyst handoff | Mission provenance, optional MLflow tracking, retained failure evidence, HTML reports, unit tests, lint/type checks and Docker/CI. This is not a complete production MLOps service. | [Real case presentation](../examples/nl-real-case.html), [synthetic report](../examples/nl-synthetic-report.html), [CI configuration](../.github/workflows/ci.yml), [PR #44 main CI](https://github.com/haseebkn/TritonEye/actions/runs/38810857035) |

The main CI run linked in the current summary passed unit tests, lint/types and
Docker for `259d592`. Local validation passed 1,113 tests (two slow tests excluded);
a subsequent focused regression/guard run passed 260. This verifies software,
not an independently labelled model benchmark. Use the commands below to check
another checkout; an older green run does not certify newer code.

PR #44 requires an exact calibration measurement/polarization match and rejects
overlap heartbeat spans containing disconnect, reconnect or session-stop events,
including equal-time boundaries. All 20 permitted development measurements still
resolve to exact calibration files. The original numerical calibration is unchanged.
See [regressions and historical replay limits](ACQUISITION_RELIABILITY.md):
historical v0.3.0/v0.4.0 reservation manifests retain the pre-fix source hash and
therefore refuse verification with changed code. Exact historical replay needs
the documented original commit, not rewritten hashes. The three-pass targeted
collection replay still verifies with zero labels admitted.

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
| Vessel precision, overall recall and false alarms per water km² | No independently reviewed, complete-area vessel/non-vessel truth. Current v0.2.0 has five provisional vessel interpretations, not accepted accuracy truth. Historical v0.1.0 retains its zero-confirmed-label scope. | A new versioned, reviewed NL dataset including detector misses, clutter/ice uncertainty and acquisition/geographic holdouts. [Annotation boundary](NL_BENCHMARK.md) |
| Correct SAR–AIS identities and measured correlation improvement | No independently reviewed association cases; sparse local AIS lacks sufficient competition/track context. A proximity match is not a correct-identity label. | Area-wide co-temporal tracks, supported identity decisions and a paired geometric/uncertainty comparison on held-out cases, including wrong identities, ambiguity, misses and coverage. [Association protocol](TARGET_CORRELATION.md) |
| Supervised detector or foundation-model improvement | Exploratory bias/token-head fits now use one provisional VV/VH training vessel. They demonstrate gradient updates, but lack legitimate negative supervision, independently reviewed accuracy and a useful operating-point comparison. | Reviewed, diverse development data, meaningful detector fine-tuning and reproducible comparisons under the same splits and operating constraints. [Executed development study](DEVELOPMENT_EXPERIMENTS.md) |
| Low-false-positive threshold and coastal buffer selection | Missing reviewed labels and sufficient independent validation blocks; partial/unsupported sensor coverage. No operating point has been selected from detector counts. | Validation under documented false-alarm **and minimum recall/coverage** constraints, then a separately frozen final-test protocol. These are provisional research targets, not operational requirements. [Operating objective](NL_BASELINE.md#provisional-operating-objective) |
| General NL offshore/HH/HV capability and shoreline correctness | The xView3/CROMA VV/VH study does not support HH/HV. A separate frozen HH/HV contrast candidate generator and native input recipes exist, but native classification/filter improvement is unvalidated. Thirty-three shoreline controls across six locations leave large regions unchecked. | Labelled native HH/HV comparisons with shared candidates, vessel-loss accounting and broader imagery checks. [Candidate generator](HH_HV_CANDIDATES.md), [shoreline scope](COASTAL_POLICY.md) |
| Optical/RF fusion, retraining and operational MLOps | Local research promotion, shadow evaluation and rollback are implemented; no real candidate is approved. Optical/RF fusion, automated retraining/production promotion, continuous production monitoring, AWS training, DVC data versioning, the SQLite task bus and cloud commissioning remain unimplemented. | Reviewed validation evidence for local promotion; implemented and commissioned infrastructure for operational deployment. The [agent-engine document](../agents.md) is explicitly a future blueprint, not delivery evidence. |

### Data access is only one blocker

Public-source searches and downloaded imagery have enabled substantial work.
The repository does **not** establish that suitable public NL data is universally
unavailable. Its recorded searches have not yielded enough eligible evidence for
the selected acquisitions, regions and split constraints.

| Source or constraint | What is available in this project | What is still missing |
| --- | --- | --- |
| Sentinel-1 imagery | Real products, native calibration and A/B plus separate C/D research recipes; [C/D acceptance checks](CD_INPUT_ACCEPTANCE.md) | Numerical agreement is not absolute geolocation, all-condition qualification, label correctness or learned model transfer. HH/HV cannot simply replace VV/VH. |
| AIS access and reception | Live keyed recorder and retained local reports; [training-region coverage audit](TRAINING_AIS_AND_CD_EXPORT.md), [reception map](NL_RECEPTION_AND_ICEBERG_SEARCH.md) and [neighbour search](PASS_COLLECTION_AND_NEIGHBOURS.md) | Dense, co-temporal, split-safe tracks including competitors. Subscription bounds and total report counts are not area-wide reception. The live feed cannot backfill laptop-off or historical gaps. Authorized historical data or targeted collection may help, without guaranteeing usable labels. |
| AllenAI NL annotations | 31 curated review areas and 23 publisher-point checks: five provisional consistencies, four unresolved alignments and fourteen unresolved classes | Annotation-specific permission and completeness, defensible alignment/classes and full native contexts. Twelve unmarked windows are not confirmed negatives; clipped PNGs are not direct xView3 inputs. [Adoption audit](ALLENAI_ADOPTION.md). The [clarification request](ALLENAI_CLARIFICATION_REQUEST.md) remains unsent; quarantine remains intact. |
| Iceberg corroboration | Public IIP sightings, native candidate windows and bounded optical searches | Same-object evidence after source uncertainty/drift and cloud/coverage checks. Four plausible iceberg interpretations are not confirmed labels; AIS absence is not iceberg evidence. [Corroboration audit](SNAP_CORROBORATION.md), [optical/noise follow-up](CD_RETRY_NOISE_OPTICAL.md) |
| Kaggle-to-native compatibility | Separate publisher-labelled chip baseline, frozen shadow scores and native processing comparisons | Native preprocessing/domain equivalence and a labelled paired filtering experiment. Matching dimensions or passing calibration does not prove that an iceberg filter preserves vessels. [Transfer audit](STATOIL_TRANSFER.md), [input decisions](INPUT_COMPATIBILITY.md) |
| Scientific review and engineering | Provisional admission/export checks and local lifecycle are implemented | No independent reviewer is available. More downloaded data does not satisfy the review guard. Optical/RF fusion, automated retraining and commissioned cloud operations are unimplemented work, not merely inaccessible datasets. |

No new native HH/HV labels have passed the provisional training admission/export
workflow. That is distinct from the one provisional VV/VH positive already used
in the six exploratory fits. A future evidence-qualified training-only release
may support exploratory work without claiming independent accuracy or promotion;
the benchmark review requirement must not be bypassed.

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

## Supporting evidence by topic

- Benchmark and coast: [v0.2.0 expansion](NL_BENCHMARK_V020.md),
  [AIS buffer sensitivity](AIS_COASTAL_ANALYSIS.md), [blind overlap bundle](OVERLAP_REVIEW.md).
- Native HH/HV collection: [development reservations](HH_HV_DEVELOPMENT.md),
  [16-window transfer collection](RESERVED_HH_HV_TRANSFER.md),
  [nine source-supported windows](NATIVE_SOURCE_ANNOTATION.md),
  [six follow-up windows](FOLLOWUP_SOURCE_ANNOTATION.md).
- Preprocessing: [source-error cross-check](SOURCE_ERROR_PREPROCESSING.md),
  [target noise audit](TARGET_NOISE_AUDIT.md), [A/B native recipe](NATIVE_NL_INPUT.md),
  [C/D pilot](CD_REFERENCE_AND_SOURCES.md). The noise audit's two small tolerance
  failures and the optical scope/exposure incident in the follow-up report remain
  disclosed; no independent identity or Kaggle equivalence follows from these checks.
- Admission and access: [provisional policy](PROVISIONAL_NATIVE_TRAINING.md),
  [label register](LABEL_ADMISSION.md), [AllenAI source audit](ALLENAI_NL_AUDIT.md),
  [curation workflow](ALLENAI_CURATION.md), [AI first pass and exposure](ALLENAI_FIRST_PASS.md),
  [historical vessel reservation attempt](RECEPTION_VESSEL_BATCH.md),
  [source-supported reservation](SOURCE_SUPPORTED_ANNOTATION.md).

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

This demonstrates SAR/geospatial engineering, auditable target association and
reproducible ML experiments. It does **not yet demonstrate validated native-NL
detector improvement or a deployed production MLOps service**.
