# TritonEye
> **Autonomous Geospatial Target Correlation and Dark Vessel Detection Pipeline**  
> A production-grade Maritime Domain Awareness (MDA) application

---

## Overview

TritonEye is an end-to-end autonomous pipeline for detecting and tracking maritime
vessels — including "dark" vessels that disable their AIS transponders — using
Sentinel-1 Synthetic Aperture Radar (SAR) imagery fused with live AIS data streams.

By utilizing SAR imagery (GRD products), TritonEye operates under all-weather and
nighttime conditions, overcoming the limitations of traditional optical satellite
imagery. The system is designed to simulate a production MDA environment: data is
ingested continuously, processed through a deep-learning computer-vision core,
correlated across geospatial layers, and reported through an automated MLOps pipeline with
DVC data versioning and MLflow model tracking.

---

## Architecture

```
+-----------------------+      +-----------------------------+
|  DATA INGESTION LAYER |      |  GEOSPATIAL CV CORE         |
|                       |      |                             |
|  Sentinel-1 SAR GRD   |----> |  PyTorch YOLOv8 for SAR      |
|  (single/dual-pol)    |      |  (dual-polarization inputs)  |
|                       |      |  Windowed rasterio tiling    |
|  AIS feed             |----> |  Non-Maximum Suppression     |
|  (aisstream.io API)   |      |  ONNX Runtime Inference Core |
+-----------+-----------+      +------------+----------------+
            |                               |
            v                               v
+-----------+-------------------------------+----------------+
|  TARGET CORRELATION ENGINE                                 |
|                                                            |
|  rasterio  - coordinate transformation and projection      |
|  geopandas - Coordinate Reference System (CRS) alignment   |
|              and spatial joins on AIS vector points        |
|  shapely   - vessel footprint geometry buffering           |
|                                                            |
|  Dark vessel rule: detection in image with no AIS          |
|  transponder signal within 2 nm / 30-min window           |
+---------------------------+--------------------------------+
                            |
                            v
+--------------------------++--------------------------------+
|  PRODUCTION MLOPS PIPELINE                                 |
|                                                            |
|  DVC             - data versioning of large GeoTIFFs to S3 |
|  MLflow          - experiment tracking, model registry     |
|  Docker          - containerised agents and inference      |
|  AWS (ECR/S3/RDS/EC2 Spot) - spot-optimized cloud running  |
|  GitHub Actions  - CI/CD, model promotion gates           |
+-----------+------------------------------------------------+
            |
            v
+---------------------+
|  MISSION REPORT     |
|                     |
|  lavish-axi HTML    |
|  Leaflet map embed  |
|  Dark vessel table  |
|  MLflow links       |
+---------------------+
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion Layer

**Satellite: Sentinel-1 Synthetic Aperture Radar (SAR)**
- Product Type: Level-1 Ground Range Detected (GRD) in Interferometric Wide (IW) mode.
- Polarization: Dual-polarization (VV+VH) or single-polarization (HH or VV) backscatter data.
- Resolution: ~10 m spatial resolution.
- Access: Programmatic Copernicus DataSpace ecosystem APIs and `sentinelsat`.
- Pre-processing: Apply orbit file, GRD border noise removal, thermal noise removal, calibration to beta0 or sigma0, speckle filtering, and terrain correction.
- Dataset Versioning: Managed by DVC (Data Version Control) to version massive raw/processed GeoTIFF datasets to an AWS S3 remote.

**AIS Stream**
- Source: Live aisstream.io WebSocket API + historical CSV snapshots.
- Fields: MMSI, name, vessel type, position, SOG, COG, heading, and timestamp.
- Storage: PostgreSQL with PostGIS extension.

**Ingestion Agent (`IngestAgent`)**
- Programmatically queries Copernicus DataSpace for SAR tiles intersecting configured AOIs.
- Downloads files directly and versions them with DVC (`dvc add`).
- Fetches matching AIS logs and creates task records in `tasks-axi`.

---

### Stage 2 — Geospatial Computer Vision Core

**Model Architecture**
- Core Model: YOLOv8 customized for SAR target detection (dual-polarization VV/VH backscatter input channels).
- Pre-processing: Large Sentinel-1 GRD GeoTIFF files are split into overlapping chunks (e.g., 640×640 pixels) using `rasterio` windowed reading.
- Inference Core: Accelerates execution with ONNX Runtime.
- Stitching & Post-processing: Combines overlapping tile predictions and applies Non-Maximum Suppression (NMS) to output clean, non-redundant vessel target bounding boxes.
- Output: GeoJSON containing bounding boxes with properties: class, confidence, coordinates, and classification.

---

### Stage 3 — Target Correlation Engine

**Geospatial CRS Alignment**
- Coordinates of the SAR detections (often in UTM zone projections) are reprojected and aligned to coordinate reference systems (CRS) matching the AIS track vector points (WGS-84 / EPSG:4326) using `geopandas` and `pyproj`.

**Correlation & Scoring**
- Reprojected vessel footprints are buffered by 2 nautical miles using `shapely` to account for timing offsets and drift.
- Spatial intersection joins are executed between the buffered vessel polygons and time-matched AIS points.
- Detections with no correlating AIS signal are identified as dark vessels.

---

### Stage 4 — Production MLOps Pipeline

**Data Version Control (DVC)**
- DVC versions the massive raw/processed GeoTIFF datasets to an AWS S3 remote, working in tandem with Git. Git tracks the `.dvc` pointer files, while large binary data lives securely on S3.

**MLflow**
- Centralized tracking server log parameters: hyperparameters, dataset DVC hash, model metrics (mAP@0.5), inference latency, and hardware metrics.
- Model Registry holds candidate versions; promotion is automated via GitHub Actions pipelines.

**Docker & AWS Cloud Infrastructure**
- Microservices: Containers for `ingest`, `inference`, `correlation`, and `report` services built from `python:3.12-slim` bases.
- Compute Optimization: Nightly pipelines run sequentially on single-region configurations (defaulting to the North Atlantic).
- Autoresearch Loop: Leverages AWS Spot Instances to scale up compute dynamically for training runs and spin down immediately after registration.

---

## Directory Structure

```
TritonEye/
|-- agents.md                   # Agent blueprint and repository roles
|-- readme.md                   # This architecture overview
|-- pyproject.toml              # Project dependency and tool configuration
|-- requirements.txt            # Python environment locked requirements
|-- data.dvc                    # DVC dataset tracker (pointer to S3 bucket)
|-- agents/                     # Agent source modules
|   |-- ingest/
|   |-- inference/
|   |-- correlation/
|   |-- report/
|   `-- skills/                 # Custom anthropics/skills extensions
|-- configs/
|   |-- aois/                   # Area-of-interest GeoJSON polygons
|   |-- model.yaml              # Model hyperparameters & tiling configs
|   `-- gnhf.yaml               # Overnight mission schedule config
|-- data/
|   |-- raw/                    # Downloaded Sentinel-1 SAR tiles (DVC tracked)
|   `-- processed/              # Calibrated and terrain-corrected tiles (DVC tracked)
|-- missions/                   # Per-mission geojson outputs (S3 synced)
|-- models/                     # Local model weight cache (gitignored)
|-- reports/                    # lavish-axi HTML reports
|-- tests/
|   |-- unit/
|   `-- integration/
|-- Dockerfile.ingest
|-- Dockerfile.inference
|-- Dockerfile.correlation
|-- Dockerfile.report
|-- docker-compose.yml
|-- pyproject.toml
`-- .pre-commit-config.yaml     # no-mistakes hooks
```

---

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.3+ | Deep learning training and backscatter processing |
| `ultralytics` | 8.x | YOLOv8 object detection model API |
| `rasterio` | 1.3+ | Geospatial windowed raster tiling and processing |
| `geopandas` | 0.14+ | Spatial dataframes, projection (CRS) alignment, and joins |
| `shapely` | 2.x | Geometric buffering and polygon operations |
| `sentinelsat` | 1.3+ | Copernicus API integration |
| `dvc[s3]` | 3.x | Large data versioning to AWS S3 storage |
| `mlflow` | 2.x | Experiment tracking and model registry integration |
| `psycopg2-binary` | 2.9+ | PostgreSQL/PostGIS database adapter |
| `fastapi` | 0.110+ | Internal microservice coordination |
| `boto3` | 1.34+ | AWS SDK for Python |

---

## Reference Repositories

| Repository | Role in TritonEye |
|-----------|------------------|
| [kunchenguid/firstmate](https://github.com/kunchenguid/firstmate) | Multi-agent orchestration (Captain) |
| [kunchenguid/tasks-axi](https://github.com/kunchenguid/tasks-axi) | Task & backlog management |
| [kunchenguid/lavish-axi](https://github.com/kunchenguid/lavish-axi) | HTML mission report rendering |
| [kunchenguid/no-mistakes](https://github.com/kunchenguid/no-mistakes) | Git safety & CI guardrails |
| [kunchenguid/treehouse](https://github.com/kunchenguid/treehouse) | Git worktree management |
| [kunchenguid/gnhf](https://github.com/kunchenguid/gnhf) | Overnight autonomous scheduling |
| [anthropics/skills](https://github.com/anthropics/skills) | Claude agent skill library |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Automated ML experiment loop |
