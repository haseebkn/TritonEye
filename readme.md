# TritonEye
Newfoundland and Labrador SAR/AIS maritime target research pipeline.

**Start here: [Reviewer guide — built and verified, measured, and blocked](docs/REVIEWER_GUIDE.md).**
One entry point for evidence, limitations and a short review walkthrough.
Current data decisions: [NL label-admission register](docs/LABEL_ADMISSION.md)
and [input-compatibility decision report](docs/INPUT_COMPATIBILITY.md).
The subsequent [target noise audit](docs/TARGET_NOISE_AUDIT.md) and
[frozen native-NL input specification](docs/NATIVE_NL_INPUT.md) document
noise-limited target boxes and an executable research recipe; labels remain on hold.
The [provisional training policy and source-ranked training-only reservation](docs/PROVISIONAL_NATIVE_TRAINING.md)
add an executable evidence checklist and two fresh annotation opportunities;
no existing labels are admitted, and training-vessel coverage remains missing.
The [two-window native preparation and annotation report](docs/TRAINING_NATIVE_ANNOTATION.md)
records one new plausible iceberg hypothesis, one coverage abstention, four exact
calibration replays and no admitted training labels or new model scores.
The [C/D reference-processing pilot and source refresh](docs/CD_REFERENCE_AND_SOURCES.md)
compares existing training windows with SNAP and records an empty refreshed vessel
reservation plus optical iceberg leads. No input-contract extension or labels are approved.
The [C retry, noise diagnosis and optical inspection](docs/CD_RETRY_NOISE_OPTICAL.md)
complete the pilot's nominal comparisons and explain the noise mismatch;
cloud and coverage prevent iceberg corroboration. An initial optical scope
error is disclosed, and protected-geography checks now precede pixel reads.
The [C/D input specification and additional acceptance checks](docs/CD_INPUT_ACCEPTANCE.md)
add a separate executable research recipe and refresh vessel-training reservations;
object admission, native model transfer and independent accuracy remain unresolved.
The [training-region AIS coverage audit and provisional C/D admission/export workflow](docs/TRAINING_AIS_AND_CD_EXPORT.md)
find almost no received AIS in the four training envelopes and add guarded native
training exports for future evidence-qualified objects. No real labels are admitted.
The 31 reserved areas contain 58 provisional annotations/contexts; none is
admitted for training or accuracy truth. Native model transfer remains unvalidated.
Five-minute presentation: [real Newfoundland case](examples/nl-real-case.html),
[walkthrough and lifecycle rehearsal](docs/PORTFOLIO_DEMO.md), and
[technical explanation and contribution notes](docs/TECHNICAL_EXPLANATION.md).
Local model operations: [promotion, paired shadow evaluation and rollback](docs/MODEL_LIFECYCLE.md).
Promotion remains blocked by the current independent-review and coverage gaps.
Latest preparation: [SAR–AIS overlap ranking and detector-blind review bundle](docs/OVERLAP_REVIEW.md).
New label collection: [IIP and local AIS overlap audit with a sealed annotation batch](docs/SOURCE_SUPPORTED_ANNOTATION.md).
Its [nine native windows are now provisionally annotated before scoring](docs/NATIVE_SOURCE_ANNOTATION.md):
two provisional vessels, seven unresolved returns, two background contexts, no defensible iceberg labels.
All remain excluded from accuracy and training approval; independent review is still missing.
The [association-error and preprocessing cross-check](docs/SOURCE_ERROR_PREPROCESSING.md)
adds 18 numerical calibration comparisons and a sealed six-area follow-up reservation.
The later SNAP audit below checks external numerical agreement; Kaggle input
equivalence remains unresolved.
The subsequent [six-window native preparation and provisional annotations](docs/FOLLOWUP_SOURCE_ANNOTATION.md)
yield four source-supported plausible iceberg interpretations and two unresolved returns.
Four windows cover all frozen uncertainty scenarios; two require association abstention.
All six remain ignored for accuracy and unapproved for training or promotion.
The subsequent [SNAP reference and four-candidate corroboration audit](docs/SNAP_CORROBORATION.md)
passes eight native calibration bands across four acquisitions and records
substantial noise sensitivity, remaining identity/input-equivalence limits and
one elapsed-limit overrun. No labels are approved by these numerical checks.
Latest supporting measurement: [AIS coastal-buffer sensitivity and its limits](docs/AIS_COASTAL_ANALYSIS.md).
Latest detector replay: [Provisional v0.2.0 development baseline](docs/NL_BASELINE_V020.md).
Development study: [Exploratory model fits and comparison protocol](docs/DEVELOPMENT_EXPERIMENTS.md).
Separate supervised task: [Audited HH/HV ship versus iceberg classification baseline](docs/STATOIL_BASELINE.md).
Its [real NL HH/HV compatibility audit and frozen shadow study](docs/STATOIL_TRANSFER.md)
reports provisional native-image scores; detection improvement remains unmeasured.
The [experimental HH/HV candidate generator](docs/HH_HV_CANDIDATES.md) now provides
a frozen, shared candidate population for a future paired filtering experiment.
Additional [NL HH/HV development acquisitions are reserved before scoring](docs/HH_HV_DEVELOPMENT.md),
with separated native datatakes and geographic groups; no new imagery is scored.
The [reserved-scene transfer collection](docs/RESERVED_HH_HV_TRANSFER.md) adds
sixteen inspected native windows and 32 exact calibration replays. Its 25
unresolved returns remain ignored; vessel and iceberg class coverage is missing.
Its chip results do not measure full-scene vessel detection or approve model promotion.
Public-label audit: [AllenAI licence, NL coverage, alignment and completeness](docs/ALLENAI_NL_AUDIT.md).
Curation preparation: [AllenAI image-first workflow and provisional ignore masks](docs/ALLENAI_CURATION.md),
with an [unsent clarification request](docs/ALLENAI_CLARIFICATION_REQUEST.md).
Completed curation: [31-area provisional AI first pass and its limitations](docs/ALLENAI_FIRST_PASS.md).
Adoption checks: [23 publisher-point decisions, input compatibility and remaining blockers](docs/ALLENAI_ADOPTION.md).

Built as an independent portfolio project relevant to C-CORE's COREsight AI
Engineer role. **Research prototype, not a certified surveillance system or a
C-CORE product. Vessel precision and false-alarm rate are not yet measured.**

## What the project demonstrates

- Calibrated Sentinel-1 VV/VH processing, fixed-size model tiling, GCP/affine
  georeferencing, and GeoJSON outputs restricted to an explicit NL study polygon.
- A pinned xView3 SAR ensemble; experimental SAR-only YOLO is optional.
  No generic detector fallback and no invented vessel-type labels.
- UTC AIS validation, bounded motion alignment and geodesic one-to-one target
  assignment, retaining ambiguous matches and missing-coverage states.
- Separate physical land/water classification and coastal eligibility policy,
  preserving coastal research returns independently of open-water candidates.
- Auditable policy exclusions and provisional fixed-infrastructure
  proximity flags. Nominal mobile FPSO positions are not permanent masks.
- Per-mission input/model hashes, processing parameters, stage logs, MLflow
  experiment/model provenance, unit tests, and Docker/CI configuration.
- An analyst report with offline-readable evidence tables and an optional map.

The pipeline is **ingest → inference → correlate → evaluate → report**. A failed
stage stops the mission; failed model tiles never silently become a complete scan.
A target without an AIS match is **not proof of a dark vessel or intent**.
Automatic operational alerts are disabled pending regional validation.

Saved real-scene AIS comparisons provide supporting proximity evidence;
see [the observed result, interval, hashed evidence and limitations](docs/AIS_SUBSET_RESULT.md).

Regional shoreline controls, coastal/open-water replay counts and remaining
label/coverage gaps: [COASTAL_POLICY.md](docs/COASTAL_POLICY.md).

SAR annotation now has a [versioned NL benchmark draft](datasets/nl_benchmark/v0.2.0/DATASET_CARD.md),
with 21 complete selected areas, acquisition/geographic split locks and hashed
provenance. The [Step 3 expansion report](docs/NL_BENCHMARK_V020.md) documents nine
new image-first passes and five provisional vessel interpretations with AIS
support. Independent review is unavailable; accuracy remains unmeasured.

The [NL baseline and operating-point protocol](docs/NL_BASELINE.md) separates
processing coverage, resource use and descriptive returns from unavailable
accuracy. Validation-only threshold selection requires useful recall and coverage
as well as a low false-alarm bound; it cannot succeed by withholding everything.
The [v0.2.0 replay](docs/NL_BASELINE_V020.md) extends this accounting to the
expanded development set; independent accuracy remains unavailable.

The [model development status](docs/MODEL_DEVELOPMENT.md) includes frozen CROMA
radar features and an [exploratory positive-only training protocol](docs/DEVELOPMENT_EXPERIMENTS.md)
for a limited xView3 head fine-tune and a CROMA token head. Reviewed accuracy and
controlled improvement remain blocked by label and coverage gaps. HH/HV remains
separate and unsupported; locked test imagery is excluded from development.

The [target correlation study](docs/TARGET_CORRELATION.md) adds detector-blind
identity review bundles and an experimental uncertainty-aware comparison. No
independently reviewed SAR–AIS identities or measured improvement are available;
the production geometric method is unchanged and no learned ranker is trained.

## Offline synthetic demonstration

For a no-install preview, download and open
[the synthetic example report](examples/nl-synthetic-report.html).
It contains simulated targets and must not be used as real surveillance evidence.

Use Python 3.12 or 3.13 in a virtual environment, from this repository root:

```sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m agents.pipeline --mock
```

Open the `report_html` path printed in the final JSON. No credentials, network
data downloads or trained model are required for this **synthetic** demonstration.
It proves workflow behavior, not vessel detection accuracy. The demo uses
deterministic NL imagery/AIS; missing coastline data stays explicitly unverified.
Paths to each stage's JSON and log are under `missions/execution_*/`.

## Finding acquisitions eligible for evaluation

```bash
python -m agents.scene_watch --days 12 --aoi eastern_newfoundland
```

Discovery checks metadata and recorded observations before downloading a product
(often about 1.7 GB). Eligibility requires:

1. **VV/VH** — the detector cannot use HH/HV, and over parts of this region
   HH/HV is all that is acquired ([DATA_SOURCES.md](docs/DATA_SOURCES.md))
2. **Recorded AIS in the ±5 min window and actual product footprint**, inside
   the NL study region. The integrated live feed cannot backfill gaps.
3. **Open-water reference observations** under the current coastal policy.
   Harbour AIS can support a separate coastal/raw-detection experiment.

This is a preliminary selection check. Measurement still requires aligned AIS
on valid SAR pixels. The legacy JSON field `scorable` is an eligibility alias;
it does not mean an evaluation has been produced.

Harbour observations require particular care: physical water inside the 300 m
coastal band is not eligible under the current policy. The earlier claim that
the OSM polygon closed St. John's harbour was incorrect. A 2026-10-05 comparison
with municipal aerial imagery confirms an open basin and entrance. Eight
imagery controls now guard against harbour closure and obvious land leakage;
CanVec was evaluated but not adopted because it misses a quay control.
See [shoreline evidence and reproduction](docs/ST_JOHNS_SHORELINE.md).

Exit status is 0 when something is scorable and 3 when nothing is, so a
scheduled job can branch on it without parsing text.

### Catching one unattended

`scripts/watch_and_score.sh` supervises the recorder and runs the Python watcher.
It checks heartbeat and observation freshness, discovers eligible products and
processes at most one pending product per invocation. Both default to the whole
`newfoundland_labrador` study polygon. Override the Python watcher's area with
`--aoi`, or the launcher's default with `TRITONEYE_WATCH_AOI`:

```bash
scripts/watch_and_score.sh              # discover and process an eligible product
scripts/watch_and_score.sh --check-only # update discovery without processing
python -m agents.watch --check-only     # discovery without Docker supervision
```

Product UUIDs select downloads and survive every pipeline stage. JSON records
under `data/watch/records/<aoi>/<product-id>.json` distinguish discovered,
eligible, processed, measured and failed outcomes. Content hashes cover code,
model, configuration, shoreline and the acquisition's AIS window. Identical
completed versions are skipped; changed versions can be evaluated again.
Legacy date stamps remain on disk but no longer control scheduling. Attempts
use frozen AIS snapshots and separate execution artifacts. OS locks prevent
overlapping watchers. See the [per-version retry policy](docs/ACQUISITION_RELIABILITY.md#product-records)
for retry delays, exhaustion and retained attempt history.

The recorder publishes a heartbeat every 60 seconds. A heartbeat older than
180 seconds, observations/receipts older than 600 seconds, and recorded gaps are
reported independently of Docker running state. Freshness does not establish
complete AIS coverage. An old recorder image must be rebuilt and restarted to
publish this evidence. See [reliability and replay](docs/ACQUISITION_RELIABILITY.md).

Register it on Windows so it runs without supervision:

```bash
schtasks /create /tn TritonEyeSceneWatch /sc hourly /f /tr "\"C:\Program Files\Git\bin\bash.exe\" E:\TritonEye\scripts\watch_and_score.sh"
```

## Real Newfoundland/Labrador data

1. Copy `.env.example` to `.env`. Supply a free Copernicus Data Space account
   for scene downloads; never commit credentials.
2. Install a matching CUDA-enabled PyTorch build if using an NVIDIA GPU.
   The Docker image is deliberately CPU-only. The ensemble is computationally
   expensive; see [evaluation and limitations](EVALUATION.md).
3. Download/verify the pinned public model and the optional public coastline:

```sh
python -m agents.assets
python -m agents.landmask --fetch --source osm
python -m agents.pipeline --date 2026-08-17 --aoi eastern_newfoundland
```

Imaging AOIs: `grand_banks` (pipeline default), `eastern_newfoundland`,
`st_johns_offshore`, `labrador_shelf`, `newfoundland_labrador`. Dates are UTC;
if no compatible VV/VH scene is found for the requested date, ingestion fails
rather than changing it.

The `newfoundland_labrador.geojson` polygon is a hand-defined **study area**,
not an EEZ, provincial jurisdiction boundary, or navigation chart.
`nl_shelf` is the larger AIS subscription envelope, not an imaging request.
Out-of-region SAR returns are saved separately and excluded from results.

For future co-temporal AIS observations, put a free AISStream key in `.env`:

```sh
python -m agents.ais_recorder --duration 3600
```

The recorder cannot backfill historical acquisitions. Successful subscription or
one received message does not establish complete spatial/temporal coverage.
Provider timestamps are not independently verified onboard position-fix times.
See [data sources and access](docs/DATA_SOURCES.md).

Existing ingest JSON can be replayed with
`python -m agents.pipeline --input path/to/ingest.json`.
Do not compare changed thresholds/preprocessing as if they were the same experiment.

## Verification and container demo

```sh
python -m pytest tests/ -q -m "not slow"
python -m ruff check agents tests
python -m black --check agents tests
python -m mypy agents tests
docker compose config --quiet
docker compose run --rm pipeline python -m agents.pipeline --mock
```

Direct dependency versions are pinned in `requirements.txt` and
`pyproject.toml`; transitive dependencies and the base image are not fully locked.
Large SAR rasters, model weights, coastline databases, and credentials are excluded
from Git and the Docker build context. Persistent SQLite state uses a directory
mount, not a nonexistent database-file mount.

## C-CORE relevance and honest gaps

This project emphasizes the role's ML/data-fusion core rather than a frontend:
SAR preprocessing, target association, operationally meaningful uncertainty,
reproducible experiments and analyst handoff. The xView3 detector is a supervised
SAR model, **not a geospatial foundation model**. MLflow records provenance. A
separate local research lifecycle implements guarded manual promotion, paired
shadow evaluation and rollback; no real candidate is approved. Automatic
production promotion and monitored cloud deployment are not implemented.

Not implemented/validated: RF or optical fusion, regional iceberg/vessel
classification, learned target correlation, a geospatial foundation-model
benchmark, labelled NL precision/recall, automated retraining, cloud commissioning,
or an operational service-level agreement. These are concrete next experiments,
not features hidden behind an attractive dashboard.

Read [methods and public standards](docs/MDA_METHODS.md),
[evidence and validation plan](EVALUATION.md), and
[portfolio audit](docs/AUDIT.md) before presenting results.
C-CORE's internal acceptance criteria were not available; public references guide
the design but do not confer compliance or endorsement.
