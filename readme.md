# TritonEye
Newfoundland and Labrador SAR/AIS maritime target research pipeline.

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
- Auditable land/coastal exclusions and provisional fixed-infrastructure
  proximity flags. Nominal mobile FPSO positions are not permanent masks.
- Per-mission input/model hashes, processing parameters, stage logs, MLflow
  experiment/model provenance, unit tests, and Docker/CI configuration.
- An analyst report with offline-readable evidence tables and an optional map.

The pipeline is **ingest → inference → correlate → evaluate → report**. A failed
stage stops the mission; failed model tiles never silently become a complete scan.
A target without an AIS match is **not proof of a dark vessel or intent**.
Automatic operational alerts are disabled pending regional validation.

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

## Finding a scene that can actually be scored

```bash
python -m agents.scene_watch --days 12 --aoi eastern_newfoundland
```

A scene is only scorable if three things hold at once, and a source product is
~1.7 GB, so this checks the free ones first rather than recommending a download
that cannot yield a measurement:

1. **VV/VH** — the detector cannot use HH/HV, and over parts of this region
   HH/HV is all that is acquired ([DATA_SOURCES.md](docs/DATA_SOURCES.md))
2. **Recorded AIS in the ±5 min window** — no historical archive covers these
   waters, so coverage exists only where the recorder was running
3. **Those AIS positions in alert-eligible water** — vessels berthed in a
   harbour cannot contribute to recall, because the pipeline excludes those
   positions by design

The third gate exists because of a real near-miss. A 2026-09-03 acquisition
reported 75 AIS observations from 18 vessels and looked scorable, but every
position lay inside the coastline in St. John's harbour. Scoring it would have
produced a recall of zero **by construction** and read like a detector result.

Exit status is 0 when something is scorable and 3 when nothing is, so a
scheduled job can branch on it without parsing text.

### Catching one unattended

A qualifying scene cannot be arranged, only caught — it needs a VV/VH pass over
open water while the recorder happens to be running. `scripts/watch_and_score.sh`
checks and, if one qualifies, processes it:

```bash
scripts/watch_and_score.sh              # check, and score if possible
scripts/watch_and_score.sh --check-only # never download, just report
```

Safe to run repeatedly: the common case costs one catalogue query. A scene is
only downloaded (~1.7 GB) and processed (~17 min GPU) when all three gates pass,
and a stamp file in `data/watch/` stops the same acquisition being reprocessed.

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

Imaging AOIs: `grand_banks` (default), `eastern_newfoundland`,
`st_johns_offshore`, `labrador_shelf`. Dates are UTC; if no compatible VV/VH
scene is found for the requested date, ingestion fails rather than changing it.

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
SAR model, **not a geospatial foundation model**. MLflow records provenance; it
does not implement automatic champion promotion or monitored staged deployment.

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
