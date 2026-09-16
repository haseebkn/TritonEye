# Portfolio audit — September 2026

## Scope and claim boundary

Independent portfolio engineering review oriented to C-CORE COREsight's ML role.
No access to C-CORE internal protocols, customers, protected data or acceptance
tests was available. This is not a compliance certification or product endorsement.

## Addressed issues

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
