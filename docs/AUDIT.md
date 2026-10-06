# Portfolio audit — September 2026

## Scope and claim boundary

Independent portfolio engineering review oriented to C-CORE COREsight's ML role.
No access to C-CORE internal protocols, customers, protected data or acceptance
tests was available. This is not a compliance certification or product endorsement.

## Addressed issues

- October 5 shoreline/policy separation: retained the eight St. John's controls,
  added five Bonavista and five Lewisporte imagery controls, and checked their
  coverage against actual valid SAR pixels in two cached scenes. CanVec remains
  comparison-only. Coastal returns now have separate review artifacts/table;
  versioned 0/100/300/500/1,000 m replay counts remain unmeasured for accuracy.
  No province-wide validation or Labrador scene coverage is claimed. Details:
  [COASTAL_POLICY.md](COASTAL_POLICY.md).

  Local verification: 274 non-heavy tests passed on Windows Python 3.13;
  network-isolated Linux Python 3.12 container: 272 passed, two local-data skips,
  two heavy tests deselected. Ruff, Black, strict typing (59 files), Docker build
  and watcher-shell syntax passed. Real cached-scene replay retained identical
  source artifacts and correctly produced no accuracy measurement.

- Removed silent synthetic ingestion and wrong-date fallback from production.
- Enforced VV/VH, UTC/sentinel validation, official AIS Class A/B message parsing,
  and bounded recording with explicit connection/coverage provenance.
- Added a shared NL-only study polygon, including Labrador, and rejected expanded
  external AOIs. Preserved excluded targets without advertising them as vessels.
- Pinned detector hashes, removed generic YOLO fallback and fabricated vessel types.
- Made failed model tiles abort incomplete inference; validated VV/VH geolocation
  by coordinate values rather than Python object identity; rejected nodata centers.
- Preserved nearby same-tile targets during cross-tile deduplication.
- Corrected land/coastal classification edge context and made infrastructure
  proximity uncertain; movable FPSO reference positions are not permanent masks.
- Replaced many-to-one spatial matches with bounded time-aligned, geodesic,
  one-to-one assignments; retained alternatives as ambiguous.
- Missing or stale AIS no longer creates dark-vessel claims. Automatic alerts are
  disabled. Evaluation separates raw/eligible AIS-subset recall from unknown precision.
- Added atomic JSON artifacts, safe mission identifiers, input/model/config hashes,
  MLflow model identity reuse, a sequential pipeline and per-stage failure logs.
- Rebuilt the report with offline evidence, explicit uncertainty and safe HTML/JSON.
- Excluded secrets/large datasets from the Docker context and corrected persistent
  database mounts and evaluation-before-report ordering.

## Verification ledger

- Local regression suite: 176 passed, 2 heavyweight model tests deselected.
- Strict mypy: all 45 Python source/test files passed. Ruff and Black passed.
- Offline five-stage Grand Banks mock replay: completed; clearly synthetic.
- MLflow: actual temporary SQLite integration verifies two runs reuse one model
  version for identical weights; no model inference performed by that test.
- Cached public xView3 model: SHA-256 verified against the pinned configuration.
- Fresh full-scene NL replay: **completed 2026-09-15**, `status: success` in
  16m43s on the 8 GB RTX 4070 Laptop GPU, peak 6.4 GB VRAM. Mission
  `mission_20260817_212209_9d86c0e9` on acquisition
  `S1D_IW_GRDH_1SDV_20260817T212209`. Surface classification: 17 water, 67
  coastal, 228 land, 0 infrastructure of 312.

  The earlier failures were diagnosed as **host** memory exhaustion, not VRAM:
  at the time of the failing attempts the GPU was idle at 0 MiB while host RAM
  sat at 91% used with 2.3 GB free. Setting
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator
  fragmentation across the 126-tile loop was sufficient; no precision or batch
  change was needed.

  These are current-code detection and classification counts. They remain
  **surface classifications, not verified vessel labels**: precision, false
  alarms per km² and overall recall are still unmeasured. The scene carried no
  AIS coverage, so `evaluation.scored` is false.
- Clean Docker build: passed after correcting a type-stub version pin and missing
  OpenCV system libraries discovered by the build/run checks.
- Container suite: 173 passed, 2 real-local-data tests skipped, 2 heavyweight
  model tests deselected. Compose configuration validates successfully.
- Remote CI and push: reported by the delivery PR, not inferred from local checks.

This ledger is updated before delivery; an attempted check is not a passing check.
ML precision/false-alarm performance remains unmeasured regardless of test results.

## St. John's shoreline verification — 2026-10-05

- Downloaded CanVec 1:50,000 NL hydrography and a public City Imagery2022
  harbour export. Raw sources, export metadata, clipped comparisons, SHA-256
  provenance and visual overlays are saved under `data/reference/st_johns/`.
- Corrected the prior unsupported assertion that OSM enclosed the basin and
  The Narrows. The cached OSM shoreline passes all eight imagery controls.
  Basin/channel/entrance controls are physically water but remain in the
  300 m coastal exclusion band. No geometry replacement or buffer reduction
  was warranted by this evidence.
- CanVec misses the north-quay land control and omits a visible pier. Its
  local ocean feature has 1979 source attributes and nominal 26 m horizontal
  accuracy; the point discrepancy alone is not an overall accuracy ranking.
  CanVec is retained for inspection, not adopted as an automatic replacement.
- Added default footprint-local shoreline checks, including versioned controls
  and provenance. Known channel closure/land leakage and malformed/missing
  controls make the mask unavailable; downstream eligibility is not inferred.
- Final Windows CPU suite: **204 passed, 2 heavyweight tests deselected**.
  Two expected Rasterio warnings originate from the intentionally ungeoreferenced
  raster regression. Strict mypy passed all 48 source/test files; Ruff and Black
  passed, including the audit script. Diff whitespace check passed.
- Existing Linux image with the updated project mounted read-only, networking
  disabled: **43 shoreline/scene-watch tests passed**. This was a runtime check,
  not a fresh Docker build or remote CI run.
- Offline audit reproduction passed: OSM 8/8 controls, CanVec 7/8. No satellite
  inference was rerun and no vessel precision or false-alarm improvement was
  measured. Broader NL coastal geometry remains unverified.

Evidence and reproduction: [ST_JOHNS_SHORELINE.md](ST_JOHNS_SHORELINE.md).

## Acquisition reliability — 2026-10-05

- Product UUID now constrains discovery, ingestion and every pipeline stage.
  Distinct same-day products retain separate records; pagination is followed.
  Missing/invalid footprints and out-of-swath/out-of-region AIS are rejected.
  Legacy and expanded-schema AIS files are both read.
- Replaced date completion stamps with product-specific JSON transition and
  attempt histories. A completed pipeline with `scored: false` remains processed.
  Measurement requires matching evaluation identity and actual valid metrics.
- Content versions include code, model, settings, dependency versions, shoreline
  data and acquisition-window AIS. Changed versions can run again. AIS windows
  and execution artifacts are retained separately; overlapping watchers are
  excluded by an OS lock. Failed attempts have a one-hour retry delay.
- Added recorder heartbeat, observation/receipt freshness and persisted gap
  evidence. Updated the local recorder to the validated image; real heartbeat
  and incoming AIS were observed. Receiver coverage remains incomplete/unknown.
- Final Windows CPU suite: **241 passed, 2 heavyweight tests deselected**.
  Strict mypy passed all 54 source/test files; Ruff and Black passed. Two expected
  Rasterio warnings came from the deliberately ungeoreferenced regression.
- Fresh Docker build passed. Offline Linux image: **239 passed, 2 local-data
  tests skipped, 2 heavyweight tests deselected**. Linux lint, formatting, strict
  types, launcher syntax and Compose configuration also passed.
- Live exact-ID query verified product
  `66d3167d-240a-459a-8066-75b9d2a458f2`, reused its satellite cache without a
  download and correctly found no AIS in the scene/time window. A live 12-day
  discovery check produced no eligible pending products and no measurement.
- Cached real-scene replay completed all 126 inference tiles and subsequent
  correlation, evaluation and reporting. Selected, processed, inference and
  evaluation UUIDs all agree. Inference took 963.69 seconds; evaluation correctly
  remained `scored: false` with `coverage=none`. No vessel precision/false-alarm
  claim follows. [Compact replay evidence](evidence/acquisition_replay_20261005.json)
  records model/config/input hashes; full artifacts stay in the project data.

Reproduction and state definitions: [ACQUISITION_RELIABILITY.md](ACQUISITION_RELIABILITY.md).

## NL baseline implementation on 5 October 2026

The follow-up closes three shipping-review gaps: every pipeline invocation now
uses a fresh execution directory even without watcher provenance; development
comparisons reject held-out geography from other datatakes, including empty
detection files; and coastal replay rejects conflicting UUIDs, native products
and acquisition times before producing output. Production GeoJSON retains scene
scope and full-precision confidence scores.

The baseline adds fixed-area raw/post-policy metrics, conditional and connected
block uncertainty, explicit unsupported-sensor coverage, AIS supporting metrics,
resource measurements and validation-only operating-point search. Targets are
provisional research choices. Missing independent labels and HH/HV coverage
correctly block selection; no operational settings changed.

- Windows CPU regressions: **342 passed, 2 heavyweight tests deselected**.
- Strict mypy: **74 source/test files passed**. Ruff and Black passed.
- Fresh Docker build passed; its offline CPU suite passed with **340 tests,
  2 local-data skips and 2 heavyweight tests deselected**.
- Immutable local dataset files were rehashed and pixel/geography contracts
  verified before the real replay. Independent annotation review remains absent.
- Real bounded replay: six complete VV/VH areas, twelve native contexts; four
  HH/HV areas unsupported; no test-area inference. Validation water coverage
  **65.40% processed, 0% independently measured**.

[Baseline methods and results](NL_BASELINE.md) and
[hashed replay evidence](evidence/nl_baseline_20261005.json) document the remaining
accuracy, identity-adjudication, sensor and sample-size limits. These software
checks establish workflow behavior, not verified vessel-detection performance.
