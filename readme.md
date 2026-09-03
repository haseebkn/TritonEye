# TritonEye

> Satellite-based maritime domain awareness for **Newfoundland and Labrador** —
> Sentinel-1 SAR vessel detection correlated against AIS, to surface targets
> that are present in the imagery but silent on their transponders.

**Status: research prototype with measured false-positive suppression and
unmeasured detection recall.** Both halves of that sentence are load-bearing;
[EVALUATION.md](EVALUATION.md) is the evidence for each.

## Operating area

Newfoundland and Labrador only — the Grand Banks, the Jeanne d'Arc Basin
production installations, and the approaches to St. John's. That is not a
cosmetic scope; three regional facts drive most of the engineering here:

- **No historical AIS archive exists for these waters.** MarineCadastre, the
  usual free bulk source, is US Coast Guard data with **zero records east of
  −67.4°W**. Ground truth has to be collected prospectively, which is why this
  project ships its own AIS recorder — and why recall is not yet measured.
- **The scene is mostly land.** On a real acquisition, **72.8% of raw
  detections fell on land**, some 35.4 km inland. Suppressing that is not
  polish; without it the output is unusable.
- **This is Iceberg Alley.** An iceberg is a bright compact target with no
  transponder, so it satisfies this pipeline's dark-vessel definition exactly.
  See [§7.1](EVALUATION.md) — the limitation is structural, not incidental.

## What it does

Downloads a Sentinel-1 Level-1 GRD scene from the Copernicus Data Space,
calibrates it to σ⁰, runs a SAR-specific detector over it in overlapping tiles,
georeferences the detections, suppresses land and known fixed infrastructure,
and flags the survivors with no matching AIS transponder signal as candidate
**AIS-uncorrelated targets**. Outputs GeoJSON plus an interactive HTML brief,
and records parameters, metrics and artifacts to MLflow.

| Component | Status |
|---|---|
| Copernicus CDSE ingest (OAuth2, OData, download) | **Working** |
| GCP/TPS georeferencing of radar-geometry products | **Working** — 0.00 m residual at 210 GCPs; affine is 801 m out |
| Radiometric calibration to σ⁰ dB | **Working** — exact, 3.4× less memory than the naive path |
| xView3 ensemble detector | **Working** — stride-2 dense prediction |
| Land masking | **Working** — rejected 72.8% of a real scene; alerts 298 → 17 |
| Fixed infrastructure masking | **Working** — 4 Jeanne d'Arc Basin installations; unit-tested, not yet run on a Grand Banks scene |
| AIS recorder (`aisstream.io`) | **Working** — forward-recording only; cannot backfill |
| Dark-vessel correlation | **Working** — water-classified detections only |
| HTML mission report | **Working** |
| MLflow tracking + model registry | **Working** |
| Evaluation vs AIS ground truth | **Implemented, not yet exercised** — no NL scene has coincided with recorder uptime |
| **Detection recall over NL** | **UNMEASURED** — the project's central open gap |
| **Precision / false positives** | **UNMEASURED** — land is masked; clutter and ice are not |
| **Ship / iceberg discrimination** | **Not implemented** — every detected iceberg becomes an alert |

### What the numbers do and do not say

The alert list for the 2026-08-17 eastern Newfoundland acquisition went from
**298 to 17** once land was masked. That measures **false-positive suppression**,
which is real and reproducible.

It does **not** mean 17 vessels. It means 17 AIS-uncorrelated targets, which is
an *upper bound* on marine targets in that scene — sea clutter, wind streaks and
icebergs are unquantified within it, and no AIS existed for that acquisition to
correlate against in the first place.

---

## Architecture

```
  Copernicus Data Space                aisstream.io live feed
  (OAuth2 / OData / download)          (recorded locally)
            |                                    |
            v                                    v
  +---------------------------------------------------------+
  |  ingest_agent            data/raw/*.SAFE, ais_*.csv      |
  |    - queries + downloads Sentinel-1 IW GRD               |
  |    - measures scene footprint from GCPs                  |
  |    - selects AIS for the imaged swath, records coverage  |
  +---------------------------------+-------------------------+
                                    v
  +---------------------------------------------------------+
  |  inference_agent         missions/<id>/detections.geojson|
  |    - 640x640 overlapping tiles, rasterio windowed read   |
  |    - YOLOv8 via Ultralytics, weights from HF Hub         |
  |    - NMS in pixel space                                  |
  |    - georeference surviving boxes (geo.py: TPS or affine)|
  +---------------------------------+-------------------------+
                                    v
  +---------------------------------------------------------+
  |  correlation_agent      missions/<id>/dark_vessels.geojson|
  |    - AIS -> points, +/-5 min window                      |
  |    - buffer 2 km in a UTM zone derived from the data     |
  |    - spatial join; unmatched detections = dark candidates|
  +---------------------------------+-------------------------+
                                    v
  +----------------------------+  +--------------------------+
  |  report_agent              |  |  evaluate_agent          |
  |  reports/<id>/report.html  |  |  recall vs AIS truth     |
  |  Leaflet map + sidebar     |  |  stratified by vessel len|
  +----------------------------+  +--------------------------+
                     \\                    /
                      v                  v
              MLflow: params, metrics, artifacts,
                      model registry (mlflow.db)
```

Agents are standalone scripts chained by a JSON payload on stdout/stdin.

---

## Pipeline stages

### 1 — Ingest (`agents/ingest/ingest_agent.py`)

**Sentinel-1**
- Level-1 GRD, IW mode, dual-pol VV/VH, ~10 m ground spacing.
- Copernicus Data Space Ecosystem: OAuth2 (Keycloak password grant) → OData
  catalogue query filtered by collection, product type, AOI intersection and
  date → download with manual redirect resolution to preserve the auth header.
- Local cache check before re-downloading.
- **Radiometric calibration to σ⁰ (dB)** is implemented in
  `agents/calibration.py`, using the `sigmaNought` LUT from each product's own
  `annotation/calibration/*.xml`. It is required by — and currently used only
  by — the `xview3` backend. Orbit file application, thermal/border noise
  removal, speckle filtering and terrain correction remain *not* implemented.

**AIS.** There is exactly one source, because there is only one that reaches
this operating area:
1. **Recorded `aisstream.io` archive** (`data/raw/ais_stream/`), built by
   `agents/ais_recorder.py`. aisstream.io is a live feed with **no historical
   API**, so this covers only time during which the recorder was actually
   running, and it cannot backfill imagery acquired earlier. MarineCadastre —
   the usual free bulk archive — is not consulted: it is US Coast Guard data
   with zero records east of −67.4°W and does not reach Newfoundland at all.
2. **Nothing.** A production run with no real coverage gets an *empty* — not
   fabricated — AIS file, and `spatial_bounds.ais_coverage` is set to `"none"`.
   The report shows a red **NO COVERAGE** badge, so an all-dark result is
   auditable as a telemetry gap rather than an intelligence finding.

Synthetic mock data (`MOCK_INGEST=true`) is used only for offline testing and is
never a silent fallback in production mode.

### 2 — Inference (`agents/inference/inference_agent.py`)

Two backends, selected by `inference.detector` in `configs/model.yaml` or the
`TRITONEYE_DETECTOR` env var.

**`yolov8`** (default) — Ultralytics YOLOv8, weights from Hugging Face Hub with
fallback to a local file then stock `yolov8m.pt`. Overlapping 640×640 tiles via
`rasterio` windowed reads, rendered as 3-channel `(VV, VH, VV)` uint8. ~47 s per
scene. **Measured recall on vessels ≥50 m: 4%.**

**`xview3`** (opt-in) — the xView3-SAR challenge-winning ensemble (MIT licence),
a CircleNet encoder-decoder producing stride-2 dense predictions. **Measured
recall on the same vessels: 59%** — a 14.75× improvement, and it recovers the
20–100 m band the YOLO path has never detected anything in. Costs ~28 min per
scene and requires:

- `models/xview3/traced_ensemble.jit` (1.3 GB, gitignored) — see
  [the release page](https://github.com/BloodAxe/xView3-The-First-Place-Solution/releases)
- radiometric calibration (`agents/calibration.py`), because it consumes σ⁰ in
  dB rather than raw digital numbers

It cannot run on mock data, which carries no calibration LUT; requesting it
there falls back to the mock detector with a warning. Detections are reported
as class `unknown` rather than a fabricated vessel type.

Both backends return whole-raster pixel boxes, so detection and NMS run in
**pixel space** and georeferencing is applied once, to the surviving boxes.

```bash
TRITONEYE_DETECTOR=xview3 python agents/inference/inference_agent.py --payload-file p1.json
```

### 3 — Georeferencing (`agents/geo.py`)

The part worth reading. Level-1 GRD measurement rasters are stored in **radar
geometry**: no CRS, no affine geotransform, only a grid of ~210 ground control
points in WGS-84.

- GCP products are georeferenced by **thin plate spline** interpolation, which
  reproduces every control point exactly. A first-order affine fit to the same
  points (`rasterio.transform.from_gcps`) is off by a mean of ~700 m on a full
  IW scene and is not used.
- Terrain-corrected and synthetic mock products carry a real CRS and affine
  transform; those take an affine path and are reprojected via `pyproj`.
- A raster with neither is **rejected**, not assigned an assumed projection — a
  detection at invented coordinates is worse than no detection.
- Each box is carried through as its four corners, since an axis-aligned box in
  pixel space is a rotated quadrilateral on the ground.

### 4 — False-positive suppression (`agents/landmask.py`, `agents/infrastructure.py`)

Two masks run after georeferencing, on detection centroids rather than the
raster. Both **annotate rather than delete** — only `water` reaches the
correlator, and the rejection tally goes to the payload and the report header,
because what was discarded is itself the evidence.

**Fixed infrastructure.** The Grand Banks carries four production installations
(Hibernia, Hebron, Terra Nova, White Rose). Each is a large radar-hard target
that appears in every acquisition over its field, does not move, and carries no
AIS transmitter — so each satisfies the dark-vessel definition on **every single
pass**. A false positive that repeats on a fixed schedule erodes trust in an
alert list faster than a sporadic one. Detections within 1 km of a published
position are attributed to the installation by name; the radius covers the
structure and its 500 m safety zone while leaving the supply and standby vessels
working the field visible, because those are real traffic.

#### Land masking

- Detections are classified **water / coastal / land** against open coastline
  data (OSM land polygons, ODbL; GSHHG, public domain, as a fallback).
- Runs on detection centroids in geographic space, **not on the raster** —
  point-in-polygon over a few hundred points takes ~0.6 s, where rasterising a
  coastline to a 25000×16000 scene grid would cost hundreds of MB.
- Distances are metric via an **azimuthal equidistant projection centred on the
  scene**; a fixed UTM zone is wrong for footprints spanning a zone boundary,
  which the Newfoundland scenes do.
- Detections are **annotated, never deleted**. Only `water` reaches the
  correlator; the rejection tally goes to the mission payload and the report
  header, because what was thrown away is itself the evidence.
- On the 2026-08-17 Newfoundland scene this rejected **72.8% of 298 detections
  as land**, reaching 35.4 km inland. See EVALUATION.md §5a.8.

Fetch the coastline once before first use (~900 MB, gitignored):

```bash
python -m agents.landmask --fetch --source osm
```

Disable with `landmask.enabled: false` in `configs/model.yaml`, which reproduces
the previous unmasked behaviour exactly.

### 5 — Correlation (`agents/correlation/correlation_agent.py`)

- AIS records → points, filtered to ±5 min of acquisition.
- Both layers projected to a UTM zone **derived from the detections**, so the
  buffer distance is always metric over the data.
- AIS points buffered by 2 km (~1 nmi); detections with no intersecting buffer
  are dark-vessel candidates.
- Only water-classified detections are eligible: a rock outcrop has no AIS
  transmitter and would otherwise satisfy the dark-vessel definition perfectly.

### 6 — Report & evaluation

- `report_agent` renders a self-contained Leaflet brief with detections, dark
  vessels, AIS positions, and the AIS coverage badge.
- `evaluate_agent` scores the mission against AIS-derived ground truth. Every
  AIS-broadcasting vessel inside the swath was definitely there, so the fraction
  detected is a conservative **lower bound on recall** needing no hand labels.
  Stratified by vessel length. Missions without real AIS are marked **unscored**
  rather than 0%, so a coverage gap can never look like a detector regression.

---

## Experiment tracking

One MLflow run per mission, spanning all five stages. Ingest opens the run and
its id travels downstream in the payload (`mlflow_run_id`).

- **Params**: model repo/file/classes, conf & IoU thresholds, tile size and
  overlap, correlation radius, AOI, georeferencing method, AIS coverage, product id.
- **Metrics**: detections, raw detections, NMS suppressions, dark vessels, dark
  ratio, AIS records, tiles processed, inference seconds, plus the full
  `eval.*` block.
- **Artifacts**: both GeoJSONs, the HTML report, the detector weights.
- **Registry**: each mission registers a version of `tritoneye-sar-detector`
  tagged with repo, class map and threshold, so a change in results is traceable
  to a change in model.

Backend defaults to a repo-local SQLite database — no server needed, and unlike
a file store it supports the registry. Set `MLFLOW_TRACKING_URI` for a shared
server, or `TRITONEYE_TRACKING=off` to disable. Tracking failures never break
the pipeline.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in credentials

# offline, no credentials needed
MOCK_INGEST=true python agents/ingest/ingest_agent.py > p1.json

# start collecting ground truth -- nothing can be scored without this
python agents/ais_recorder.py --aoi grand_banks &

# real acquisition
export TARGET_DATE=2026-08-17 AOI_NAME=eastern_newfoundland
export TRITONEYE_DETECTOR=xview3
python agents/ingest/ingest_agent.py            > p1.json
python agents/inference/inference_agent.py    --payload-file p1.json > p2.json
python agents/correlation/correlation_agent.py --payload-file p2.json > p3.json
python agents/report/report_agent.py          --payload-file p3.json > p4.json
python agents/evaluate/evaluate_agent.py      --payload-file p4.json > p5.json

pytest tests/ -m 'not slow'                    # skips the 1.3 GB ensemble load
```

Docker:

```bash
docker compose run --rm pipeline        # all five stages, pipefail-guarded
docker compose run --rm ingest          # a single stage
docker compose up ais-recorder          # the one long-running service
docker compose run --rm tests           # unit tests inside the image
```

Every agent is a one-shot stage, so they sit behind a compose profile: a bare
`docker compose up` starts only `ais-recorder` rather than launching all of
them to block on empty stdin.

### Checks

```bash
ruff check agents tests      # lint          (0 findings)
black --check agents tests   # formatting
mypy agents tests            # strict types  (0 findings)
pytest tests/ -q -m "not slow"   # ~10 s
pytest tests/ -q                 # + 2 that load the 1.3 GB xView3 ensemble
```

CI (`.github/workflows/ci.yml`) runs all four on every push and PR, plus builds
the Docker image and runs its test suite inside the container. `pre-commit
install` wires the same gates to run locally before a commit.

### Environment

| Variable | Purpose |
|---|---|
| `COPERNICUS_USER` / `COPERNICUS_PASS` | CDSE credentials |
| `HUGGINGFACE_HUB_TOKEN` / `_MODEL_REPO` / `_MODEL_FILE` | detector weights |
| `AISSTREAM_API_KEY` | live AIS recorder (free, GitHub sign-in) |
| `TARGET_DATE`, `AOI_NAME` | which acquisition to process |
| `MOCK_INGEST` | offline synthetic mode |
| `MLFLOW_TRACKING_URI`, `TRITONEYE_TRACKING` | tracking backend / disable |

Note: with `TARGET_DATE` unset, ingest iterates the `ais-YYYY-MM-DD`
directories present in `data/raw/` rather than fetching the newest scene. Set
`TARGET_DATE` explicitly to process a specific acquisition.

---

## Layout

```
agents/
  geo.py                  georeferencing: TPS over GCPs, or affine
  tracking.py             MLflow run spanning all stages
  ais_recorder.py         aisstream.io live recorder -> daily CSVs
  ingest/  inference/  correlation/  report/  evaluate/
configs/
  aois/*.geojson          areas of interest
  model.yaml              tiling + inference thresholds
tests/unit/               unit + integration tests
data/raw/                 SAFE products, AIS archives (gitignored)
missions/<id>/            detections.geojson, dark_vessels.geojson
reports/<id>/report.html
mlflow.db                 tracking store (gitignored)
EVALUATION.md             measured performance
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detector |
| `rasterio` | windowed raster IO, GCP transforms |
| `geopandas` / `shapely` / `pyproj` / `rtree` | spatial joins, buffering, CRS |
| `pandas` / `numpy` | AIS filtering, array work |
| `requests` | Copernicus OData + OAuth2 |
| `huggingface-hub` | weight fetching |
| `websockets` | aisstream.io live feed |
| `mlflow` | experiment tracking + model registry |
| `python-dotenv` | configuration |

---

## Not implemented

Listed explicitly because earlier versions of this README claimed them.

**SAR preprocessing** — radiometric calibration to σ⁰ **is** implemented
(`agents/calibration.py`) and is used by the `xview3` backend. Still missing:
orbit file application, border and thermal noise removal, speckle filtering, and
terrain correction. The default `yolov8` backend still consumes raw digital
numbers.

**Ship / iceberg discrimination** — the AOI is Iceberg Alley. An iceberg is a
bright compact target against a dark sea, which is the signature the detector
looks for, and it carries no AIS transmitter — so it satisfies this pipeline's
definition of a dark vessel exactly. Every detected iceberg becomes a confident
false alert **by construction**, and no threshold can fix it. See
[EVALUATION.md](EVALUATION.md) §8.1. Dark-vessel output over Newfoundland should
be read as *AIS-uncorrelated targets*, not as vessels.

**Precision** — land is now masked, but sea clutter, wind streaks, and ice are
not. The surviving water detections are an upper bound on marine targets, not a
vessel count.

**Vessel classification** — `configs/model.yaml` declares five classes but the
deployed model has one (`ship`), which the code maps to `cargo`. Class labels in
the output are not meaningful.

**DVC data versioning**, **automated retraining / staged model promotion**,
**cloud or Spot compute**, **RF signal fusion**, **scheduling** — none of
these exist. No `.dvc`, no scheduler. Every pipeline run to date has been
launched by hand.

**Multi-scene batching** — the OData query uses `$top: 1`, so only the newest
matching scene per query is processed even when many intersect the AOI.

## Known limitations

- **Detector recall is 4%** at the deployed threshold on vessels ≥50 m; nothing
  below 100 m has ever been detected. See [EVALUATION.md](EVALUATION.md).
- **Newfoundland AOIs cannot be validated retroactively** — MarineCadastre has
  no Canadian coverage and the aisstream.io recorder only captures forward in
  time.
- **The AIS recorder subscribes to the AOI box**, which is smaller than the
  swath ingest requests AIS for; widen it to cover intersecting scenes.
- **Report requires network** — Leaflet, fonts and basemap tiles load from CDNs.
- Revisit is 1–4.5 days depending on AOI; CDSE publication latency is ~2.3 h
  median. Near-real-time is feasible, live is not.
