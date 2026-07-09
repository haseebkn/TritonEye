# Project TritonEye — Full Audit Report
> Phase-by-phase review of all 5 pipeline stages. All findings are based on live code inspection and automated checks.

---

## 🟢 Overall Rating: **7.8 / 10**

A well-structured, architecturally sound pipeline for a v0.1 prototype. The mock-mode data path is exceptionally well-designed and the CRS alignment strategy is solid. The main gaps are a confirmed HTML bug, absent test suite, and a hardcoded timestamp that prevents true nightly re-execution.

---

## Phase 1 — Foundations & Configuration

**Rating: 8/10** ✅ Mostly Good

| Check | Status | Notes |
|---|---|---|
| `agents.md` blueprint | ✅ Pass | Comprehensive crew roster |
| `readme.md` architecture | ✅ Pass | Well-structured technical docs |
| Directory layout | ✅ Pass | Clean agent-per-folder structure |
| `pyproject.toml` dependencies | ⚠️ Warning | `psycopg2-binary`, `fastapi`, `uvicorn` declared but **no code uses them yet** — orphaned deps |
| `pyproject.toml` ruff config | ⚠️ Warning | `select`/`ignore` keys are **deprecated** in favour of `lint.select`/`lint.ignore` (shows up as a warning on every ruff run) |
| `configs/model.yaml` | ✅ Pass | Clean, well-commented |
| `__init__.py` files | ❌ Missing | No `__init__.py` exists in any agent package — prevents Python package imports and blocks pytest discovery |
| Test suite | ❌ Missing | `tests/unit/` and `tests/integration/` directories are **completely empty** |

### Fixes needed
```diff
# pyproject.toml — fix deprecated ruff keys
-[tool.ruff]
-select = ["E", "F", "W", "I", "N"]
-ignore = []
+[tool.ruff]
+[tool.ruff.lint]
+select = ["E", "F", "W", "I", "N"]
+ignore = []
```

---

## Phase 2 — Data Ingestion Layer (`ingest_agent.py`)

**Rating: 9/10** ✅ Excellent

| Check | Status | Notes |
|---|---|---|
| AOI GeoJSON geometry | ✅ Pass | Valid coordinates, correct CRS |
| Mock data determinism | ✅ Pass | `seed=42` confirmed reproducible across runs |
| UTM Zone 22N projection | ✅ Pass | Correct EPSG:32622 setup |
| Target grid density | ✅ Pass | 100km×100km grid, 5km edge buffer |
| AIS CSV schema | ✅ Pass | 200 records, 40 MMSI, 5 track points each |
| Timestamp hardcoding | ⚠️ Bug | `timestamp_str = "20260708T050000"` is **hardcoded** — every mock run generates the same `mission_id`. Production re-runs would overwrite prior missions |
| Graceful import fallback | ⚠️ Warning | `except ImportError: pass` at module-level **silently swallows** import errors — scripts appear to import cleanly but then crash with `NameError` at runtime |
| Copernicus API URL | ⚠️ Outdated | `https://dataspace.copernicus.eu/odata/v1` is the OData endpoint; `sentinelsat` typically uses `dhus` API URL format |

### Fixes needed
```python
# Use datetime.utcnow() for a live timestamp in mock mode
timestamp_str = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
```

---

## Phase 3 — Inference Agent (`inference_agent.py`)

**Rating: 8/10** ✅ Good

| Check | Status | Notes |
|---|---|---|
| Windowed raster tiling | ✅ Pass | `generate_raster_windows()` correctly clamps tile edges |
| NMS implementation | ✅ Pass | Coordinate-space NMS in UTM meters is correct |
| Reprojection to WGS84 | ✅ Pass | `EPSG:32622 → EPSG:4326` via pyproj |
| GeoJSON schema | ✅ Pass | No duplicate IDs, valid coordinates confirmed |
| Confidence score formula | ⚠️ Bug | `score = 0.95 + offset` where offset = `(row + col) % 100 / 2500` — this always produces values **between 0.95 and 0.989**, never testing low-confidence filtering. The real `conf_threshold: 0.35` check is never exercised |
| VH band usage | ⚠️ Issue | VH data is read (`_ = src_vh.read(...)`) but **completely discarded** — dual-polarization ratio (VV/VH) is the core SAR vessel feature and is unused |
| Class assignment | ⚠️ Issue | `class_id = int((global_row + global_col) % 5)` — deterministic but **not representative**; 100% of vessels end up classified based on pixel coordinate parity, not any feature |
| `pass` after try/except | ⚠️ Style | Loose `pass` after import try-block is unnecessary noise |

### Key fix: VH fusion
```python
# Instead of discarding VH:
vv_win = src_vv.read(1, window=win)
vh_win = src_vh.read(1, window=win)
# Use polarization ratio as a feature
ratio = np.where(vh_win > 0, vv_win.astype(float) / vh_win, 0.0)
rows, cols = np.where((vv_win == 255) | (ratio > 1.5))
```

---

## Phase 4 — Correlation Agent (`correlation_agent.py`)

**Rating: 9/10** ✅ Excellent

| Check | Status | Notes |
|---|---|---|
| CRS alignment (EPSG:32622) | ✅ Pass | Both GDFs reprojected before spatial join |
| 2000m buffer logic | ✅ Pass | Correct use of `.geometry.buffer()` |
| Left spatial join | ✅ Pass | Correctly identifies unmatched detections |
| Dark vessel isolation | ✅ Pass | 8/10 dark vessels correctly isolated |
| Schema cleanup | ✅ Pass | Joined columns correctly stripped |
| Time-based AIS filtering | ⚠️ Missing | All 5 timestamps per vessel are used in the buffer join — AIS points from ±10 minutes are counted as "present". **A vessel that turned off AIS 20 minutes ago would be missed.** The join should filter `WHERE timestamp BETWEEN t-5m AND t+5m` |
| Multi-MMSI per detection | ⚠️ Edge case | A detection that intersects multiple AIS tracks (e.g., ships in convoy) produces **duplicate rows** in `joined_gdf`. The current `dark_mask` handles this correctly via `isna()` but upstream processing doesn't deduplicate cooperative matches |

---

## Phase 5 — Report Agent (`report_agent.py`)

**Rating: 7/10** ✅ Good but has a confirmed visual bug

| Check | Status | Notes |
|---|---|---|
| Leaflet.js map renders | ✅ Pass | Correct structure |
| Esri Ocean Base tiles | ✅ Pass | URL format correct |
| GeoJSON inline injection | ✅ Pass | Fully self-contained HTML |
| Sidebar click callbacks | ✅ Pass | `panToTarget()` correctly resolves layer |
| **Google Fonts URL** | ❌ **Bug** | URL contains `&amp;display=swap` — **HTML-entity encoded inside a `<link href>` attribute**, which causes the font to silently fail to load in most browsers. Should be literal `&display=swap` |
| `base_dir` unused variable | ⚠️ Waste | `base_dir = os.path.abspath(...)` computed in `main()` but **never used** (report_dir is computed from same path separately) |
| `active_count` calculation | ⚠️ Logic | `active_count = det_count - dark_count` — this assumes dark vessels are a strict subset of total detections. This is true in current data but conceptually fragile |
| Missing MLflow links | ⚠️ Incomplete | The blueprint specifies MLflow experiment links in the report — these are absent |
| `pass` statement at line 15 | ⚠️ Style | Orphaned `pass` after a comment with no code to guard |

### Confirmed font bug fix
```python
# In build_html_report() HTML template — change:
#  &amp;display=swap  →  &display=swap
f'<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap"'
```

---

## Summary of All Issues

### 🔴 Confirmed Bugs
| # | Location | Issue |
|---|---|---|
| B1 | `report_agent.py` | Google Fonts URL uses `&amp;` — font fails to load |
| B2 | `ingest_agent.py` | Hardcoded mock timestamp → same `mission_id` every run |

### 🟠 Functional Gaps
| # | Location | Issue |
|---|---|---|
| F1 | `inference_agent.py` | VH band read and discarded — dual-pol unused |
| F2 | `correlation_agent.py` | No time-window filter on AIS records |
| F3 | All phases | No unit or integration tests exist |
| F4 | `report_agent.py` | MLflow experiment links missing from report |

### 🟡 Code Quality Warnings
| # | Location | Issue |
|---|---|---|
| Q1 | All agents | `except ImportError: pass` silently hides import failures |
| Q2 | `pyproject.toml` | Deprecated `ruff` config keys |
| Q3 | `pyproject.toml` | Orphaned dependencies (`fastapi`, `uvicorn`, `psycopg2-binary`) |
| Q4 | All agents | Missing `__init__.py` — no package importability |
| Q5 | `report_agent.py` | `base_dir` unused variable |

---

## Top 5 Suggested Improvements

### 1. 🔧 Fix the Font Bug Immediately (5 min fix)
The `&amp;` in the Google Fonts URL is the highest-impact visual bug — it makes the premium Outfit typeface silently fail on any browser, reverting to a generic sans-serif.

### 2. 🧪 Add a Pytest Test Suite (High Priority)
The `tests/unit/` and `tests/integration/` directories are both empty. A minimal suite covering at minimum:
- `test_nms.py` — unit test NMS with synthetic overlapping boxes
- `test_crs_reprojection.py` — verify UTM→WGS84 roundtrip accuracy within 1m
- `test_dark_vessel_isolation.py` — end-to-end mock pipeline assertion (`dark_count == 8`)
- `test_ais_time_filter.py` — verify only temporally relevant AIS records are joined

### 3. ⏱️ Fix Hardcoded Mock Timestamp
Replace `timestamp_str = "20260708T050000"` with `datetime.utcnow().strftime(...)` so mock pipeline runs generate unique mission IDs and don't overwrite each other's outputs.

### 4. 📡 Use VH Band in Inference (SAR Best Practice)
The dual-polarization backscatter ratio (VV/VH) is the core discriminator between ocean clutter and metallic vessel returns in SAR imagery. Currently VH is read and thrown away. Even in mock mode, incorporating a simple `ratio = vv_win / (vh_win + 1e-6)` feature would make the inference logic more faithful to real SAR processing.

### 5. 🔌 Add AIS Temporal Filtering in Correlation
Filter AIS records to within ±5 minutes of the SAR acquisition timestamp before buffering. This prevents stale tracks from masking dark vessels that turned off their transponders before the image was acquired:
```python
img_time = pd.to_datetime(payload["acquisition_time"])
ais_df["timestamp"] = pd.to_datetime(ais_df["timestamp"])
ais_df = ais_df[abs(ais_df["timestamp"] - img_time) <= pd.Timedelta(minutes=5)]
```

---

## Bonus Improvements (Phase 6+)

| Priority | Improvement |
|---|---|
| Medium | Add a `--dry-run` flag to each agent to validate payload schema without executing |
| Medium | Implement `tasks-axi` SQLite backend integration for proper pipeline state tracking |
| Medium | Add `gnhf.yaml` nightly schedule definition for the North Atlantic default region |
| Low | Add a `Layer Control` toggle on the Leaflet map to independently show/hide Cooperative vs Dark layers |
| Low | Publish `report.html` artifact path back to `tasks-axi` task record for audit trail |
| Low | Add `firstmate` orchestrator dispatcher script that chains agents without shell pipes |
