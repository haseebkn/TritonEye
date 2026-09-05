# Current implementation boundary (September 2026)

This file records a **future blueprint**, not an implemented autonomous system.
The working pipeline is `python -m agents.pipeline`: sequential Python ingestion,
inference, correlation, evaluation and reporting with optional MLflow provenance.
The eight repository integrations below, SQLite task bus, unattended research,
AWS training, DVC data versioning and notification delivery are not implemented
here. Do not present this blueprint as evidence of completed MLOps infrastructure.
See readme.md, EVALUATION.md and docs/AUDIT.md for verified behavior and limitations.

# TritonEye — Autonomous Workflow Engine Blueprint
> **AGENTS.MD** · Project TritonEye · Maritime Domain Awareness Pipeline  
> Status: Blueprint v0.1 — Blueprint Updated & Approved

---

## 1. Purpose

This document defines the autonomous workflow engine that drives TritonEye end-to-end:
from raw satellite/AIS data ingestion, through computer-vision inference, geospatial
target correlation, and dark-vessel detection, to MLOps reporting and overnight
unattended operation.

Every agent, memory structure, and task-tracking loop is mapped explicitly to one or
more of the eight reference repositories below.

---

## 2. Repository Role Assignments

### 2.1 `kunchenguid/firstmate` — **Orchestrator Agent (Captain)**
> *"Talk to one agent. Ship with a crew."*

`firstmate` is the **single entry-point** for the entire TritonEye pipeline.  A human
operator (or a CI trigger) issues one top-level command to the Orchestrator; firstmate
decomposes it into subtasks and fans out to specialist crew agents.

**TritonEye integration:**
- Receives pipeline triggers: `RUN_INGEST`, `RUN_INFERENCE`, `RUN_CORRELATION`,
  `RUN_REPORT`, or the all-in-one `RUN_FULL_PIPELINE`.
- Maintains a **crew roster** (YAML manifest) listing all subordinate agent identities
  and their capabilities.
- Returns a single structured mission summary to the operator after all crew agents
  complete.
- Handles crew-level error escalation and retry decisions.

```
Operator / CI
     |
     v
 firstmate (Captain)
     |-- IngestAgent      (--> tasks-axi task queue)
     |-- InferenceAgent   (--> tasks-axi task queue)
     |-- CorrelationAgent (--> tasks-axi task queue)
     `-- ReportAgent      (--> lavish-axi HTML artifact)
```

---

### 2.2 `kunchenguid/tasks-axi` — **Task & Backlog Manager**
> *"Agent-ergonomic CLI for task/backlog management with pluggable backends
> (markdown, sqlite, remote trackers)."*

`tasks-axi` is the **shared task bus and state machine** that all crew agents read and
write.  Every pipeline stage creates, claims, and closes tasks through this interface.

**TritonEye integration:**
- **Backend:** SQLite for local development; remote tracker (GitHub Issues or Jira)
  for production.
- **Task schema** (per pipeline item):
  ```
  id          UUID
  stage       ENUM(ingest, inference, correlation, report)
  status      ENUM(pending, in_progress, done, failed)
  payload     JSON  (tile path, vessel ID, confidence score, ...)
  assigned_to STRING
  created_at  TIMESTAMP
  updated_at  TIMESTAMP
  ```
- Agents poll `tasks-axi list --stage <stage> --status pending` to claim work.
- After completion, agents call `tasks-axi done <id> --output <result_path>`.
- MLflow run IDs are written back to the task payload for audit trail linkage.

---

### 2.3 `kunchenguid/lavish-axi` — **HTML Artifact Publisher (Report Agent)**
> *"HTML is the new markdown. Lavish is the new editor for your HTML artifacts."*

`lavish-axi` is the **rendering layer** for all TritonEye mission reports.  After
inference and correlation complete, the Report Agent uses lavish-axi to render an
interactive HTML mission brief that operators can view in any browser.

**TritonEye integration:**
- Accepts a structured JSON payload (vessel detections, confidence maps, AIS
  correlation gaps, geographic bounding boxes).
- Renders an HTML report containing:
  - Leaflet.js maritime map with vessel detections overlaid.
  - Dark-vessel highlight table (vessels with AIS gap > threshold).
  - Model performance metrics (mAP, inference latency).
  - MLflow experiment links.
- Output artifact is saved to `reports/<mission_id>/report.html` and linked from the
  tasks-axi task record.

---

### 2.4 `kunchenguid/no-mistakes` — **Git Safety & Guardrails Layer**
> *"git push no-mistakes."*

`no-mistakes` provides **pre-push and pre-commit hooks** that prevent accidental
commits of sensitive data (API keys, model weights, raw satellite tiles) and enforce
pipeline code quality gates.

**TritonEye integration:**
- Pre-commit hooks:
  - Block commits containing AWS credentials, Copernicus API tokens, or raw data assets.
  - Enforce Python type annotations on all agent modules (`mypy --strict`).
  - Run `ruff` linting on every staged file.
- Pre-push hooks:
  - Run the full unit-test suite (`pytest tests/unit/`).
  - Verify Docker image builds cleanly before pushing to registry.
- Branch protection rules encoded as `no-mistakes` config so CI mirrors local policy.

---

### 2.5 `kunchenguid/treehouse` — **Parallel Development Worktree Manager**
> *"Manage worktrees without managing worktrees."*

`treehouse` abstracts `git worktree` to let multiple TritonEye feature branches be
developed and tested simultaneously without context-switching overhead.

**TritonEye integration:**
- Each pipeline stage (ingest, inference, correlation, report) can be developed in its
  own worktree:
  ```
  treehouse spawn feature/dark-vessel-classifier   # new worktree in .wt/
  treehouse spawn fix/ais-parser-timezone-bug
  treehouse list                                   # shows all active worktrees
  treehouse prune                                  # cleans merged worktrees
  ```
- CI matrix jobs check out individual worktrees, enabling stage-isolated testing.
- **Critical for TritonEye:** Model-retraining experiments run in isolated worktrees so
  production inference weights are never overwritten mid-experiment.

---

### 2.6 `kunchenguid/gnhf` — **Overnight Agent Scheduler ("Good Night, Have Fun")**
> *"Before I go to bed, I tell my agents: good night, have fun."*

`gnhf` is the **unattended overnight execution controller**.  An operator schedules the
full TritonEye pipeline to run autonomously while offline, with `gnhf` managing
resource limits, progress logging, and morning summary delivery.

**TritonEye integration:**
- Nightly mission definition file (`gnhf.yaml`):
  ```yaml
  mission: triton_nightly_scan
  trigger: "02:00 UTC"
  agents:
    - firstmate run RUN_FULL_PIPELINE --region North_Atlantic
  budget:
    max_gpu_hours: 4
    max_cost_usd: 2.50
  on_complete:
    notify: slack://ops-channel
    artifact: reports/latest/report.html
  ```
- Defaults to sequential, single-region missions (default region: `North_Atlantic`).
- `gnhf` enforces cost and time budgets, terminating runaway inference jobs.
- Morning summary is posted to Slack/email with mission statistics and the lavish-axi
  HTML report link.

---

### 2.7 `anthropics/skills` — **Claude Agent Skill Library**
> *"Public repository for Agent Skills."*

`anthropics/skills` provides **reusable, composable skill modules** for Claude-based
agents.  TritonEye uses these skills to give its crew agents structured capabilities
without re-implementing common agentic patterns.

**TritonEye integration — adopted skills:**

| Skill | TritonEye Usage |
|-------|----------------|
| `bash` | IngestAgent executes Sentinel-1 SAR tile downloads via `sentinelsat` CLI / Copernicus APIs |
| `text_editor` | Agents patch config files (thresholds, AOIs) between pipeline runs |
| `web_search` | ReportAgent enriches vessel identity with live maritime registry lookups |

*Note: The `computer_use` UI skill is removed. The IngestAgent relies purely on stable programmatic APIs (Copernicus API / sentinelsat).*

---

### 2.8 `karpathy/autoresearch` — **MLOps Research Automation Loop**
> *"AI agents running research on single-GPU nanochat training automatically."*

`autoresearch` provides the **self-improving experiment loop** that drives TritonEye's
computer-vision model improvement.  Inspired by Karpathy's framework, TritonEye runs
automated hypothesis-test cycles on its vessel-detection models.

**TritonEye integration:**
- **Hypothesis generation:** An LLM agent proposes model architecture changes
  (e.g., "increase anchor scales for small vessel detection").
- **Automated experiment:** The agent modifies `configs/model.yaml`, requests a short-lived AWS GPU Spot Instance, launches a training run, waits for completion, and reads MLflow metrics.
- **Evaluation:** Results are compared against the current champion model using a held-out validation set.
- **Commit or discard:** If mAP improves beyond a threshold, the agent opens a PR via
  `no-mistakes`-guarded git push; otherwise the worktree (via `treehouse`) is pruned.
- Single-GPU training runs complete overnight, managed by `gnhf` on cost-effective AWS Spot Instances.

---

## 3. Agent Interaction Diagram

```
+-------------------------------------------------------------------------+
|                        TRITONEYE AGENT MESH                             |
|                                                                         |
|   Human/CI --> firstmate (Captain)                                      |
|                    |                                                    |
|                    |-- IngestAgent                                      |
|                    |     uses: programmatic APIs (sentinelsat)          |
|                    |     claims: tasks-axi[stage=ingest]                |
|                    |                                                    |
|                    |-- InferenceAgent                                   |
|                    |     uses: PyTorch model (autoresearch-improved)    |
|                    |           + rasterio windowed tiling & NMS stitching|
|                    |     claims: tasks-axi[stage=inference]             |
|                    |                                                    |
|                    |-- CorrelationAgent                                 |
|                    |     uses: geopandas, shapely (CRS alignment)       |
|                    |     claims: tasks-axi[stage=correlation]           |
|                    |                                                    |
|                    `-- ReportAgent                                      |
|                          renders: lavish-axi HTML artifact              |
|                          claims: tasks-axi[stage=report]               |
|                                                                         |
|   gnhf --> schedules firstmate nightly --> delivers morning report      |
|                                                                         |
|   autoresearch --> model improvement loop (AWS Spot Instance)           |
|       `-- treehouse (isolated worktree) + no-mistakes (PR gates)        |
+-------------------------------------------------------------------------+
```

---

## 4. Memory Structures

### 4.1 Short-Term: Task State (tasks-axi / SQLite)
Live pipeline state; wiped after each completed mission.

### 4.2 Mid-Term: Mission Archive & Datasets (S3 / local `missions/` + DVC versioning)
One folder per mission: raw tiles, inference outputs, correlation GeoJSON,
lavish-axi HTML report, MLflow run ID. Large dataset files are tracked via DVC and versioned to AWS S3.

### 4.3 Long-Term: Model Registry (MLflow)
Champion model tracked with full lineage (dataset version, hyperparameters,
autoresearch experiment ID that produced it).

---

## 5. Automated Task-Tracking Loops

| Loop | Trigger | Agent | Output |
|------|---------|-------|--------|
| **Ingest Loop** | New Sentinel-1 SAR pass / AIS snapshot | IngestAgent | Raw GRD tiles + AIS CSV in `missions/<id>/raw/` |
| **Inference Loop** | Ingest task marked `done` | InferenceAgent | `rasterio` windowed chunking + YOLOv8 + NMS stitching → GeoJSON + confidence raster |
| **Correlation Loop** | Inference task marked `done` | CorrelationAgent | CRS-aligned intersection of SAR bboxes and AIS points → Dark-vessel list |
| **Report Loop** | Correlation task marked `done` | ReportAgent | lavish-axi HTML report |
| **Research Loop** | Weekly cron / mAP degradation alert | autoresearch agent | Spot Instance training run → Improved weights / PR |
| **Overnight Loop** | gnhf nightly trigger (02:00 UTC) | firstmate + all crew | Morning Slack summary + report link (North Atlantic) |
