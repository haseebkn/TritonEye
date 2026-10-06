# Acquisition identity and scoring reliability

Discovery retains distinct Copernicus UUIDs, including slices on the same date,
and follows catalogue pagination. AIS comes from legacy and `_v2` daily CSVs,
including UTC day boundaries. Invalid identities/coordinates, missing or invalid
footprints, and observations outside the footprint/NL area cannot establish
eligibility. Shoreline controls and the current coastal policy select an
open-water experiment. This precheck does not guarantee a measurement.

Ingestion queries the exact selected UUID with AOI, polarization and optional
date constraints. A different catalogue UUID fails before download. The pipeline
checks identity after every stage. Inference and evaluation include the UUID.

## Product records

`data/watch/records/<aoi>/<uuid>.json` retains discovery metadata, eligibility,
recorder health, transition history and attempts.

| State | Evidence |
| --- | --- |
| discovered | Product seen; it may be ineligible |
| eligible | Open-water metadata/AIS precheck passed |
| processed | Pipeline completed with complete inference for the selected UUID |
| measured | Evaluation explicitly scored, matching UUID, positive AIS denominator and finite recall in [0, 1] |
| failed | Stage failure, identity mismatch or invalid claimed measurement |

An evaluation with `scored: false` stays processed, retaining its reason.
Measurement refers to observed-AIS-subset proximity recall, not independently
labelled vessel precision or overall recall.

The attempt's `version_key` hashes Python sources, watcher launcher, configuration,
AOIs/shoreline controls, installed dependency versions, actual local weights,
shoreline files, catalogue/checksum metadata and acquisition-window AIS rows.
Git commit is retained for navigation but committing identical content does not
change the key. New AIS outside the historical window does not trigger a run;
changed code, model, configuration, reference data or window observations does.

Window observations are frozen in `data/watch/inputs/<uuid>/<version>/` before
processing. Snapshot files are published atomically; reuse requires byte-for-byte
agreement with the selected observations and no extra daily files. Every pipeline
invocation, including date-only and unversioned runs, assigns a fresh
`missions/execution_*/` destination before ingestion writes mission files.
Executions retain filtered AIS, actual SAR hashes, logs and stage outputs;
watcher executions also retain their versioned provenance. Prior detections and
reports survive re-evaluation. Source manifests and satellite caches remain reusable.

One pending product runs per invocation. Identical processed/measured versions
are skipped. Failed versions become retryable after one hour. Unattempted
products take priority, followed by the least recently attempted product, so a
repeatedly failing acquisition does not monopolize processing. Interrupted
eligible attempts can retry after the OS releases their lock. Legacy date stamps
are retained but ignored by scheduling.

## Recorder freshness and gaps

`data/raw/ais_stream/recorder_status.json` is updated atomically every 60 seconds,
including on a silent connection. It records session, connection state, latest
observation and receipt times, count and recent gaps. Journals retain periodic
heartbeats and gap events. Heartbeat intervals above 180 seconds indicate a
recording/liveness gap, including suspension. Receipt intervals above 600 seconds
record an observation gap; quiet waters and provider outages cannot be
distinguished from this evidence alone.

The watcher reports unknown, stale heartbeat, stale observations, disconnected
or fresh status. Historical eligible observations may still be processed while
the current recorder is unhealthy; the diagnostic is retained. Starting a
stopped recorder cannot recover missing history. The shell launcher supervises
Docker, while the Python watcher assesses actual evidence. A quiet connection
is not repeatedly restarted. Rebuild old recorder images for heartbeat support:

```sh
docker compose up -d --build ais-recorder
python -m agents.watch --check-only --aoi eastern_newfoundland
```

A heartbeat establishes process activity, and messages establish observations.
Neither establishes complete receiver coverage or vessel silence.

## Reproduction

Offline behavioral tests exercise catalogue identity/pagination, same-day
products, missing footprints, out-of-swath AIS, outcomes, changed versions,
frozen AIS, lock exclusion, recorder freshness and shoreline controls:

```sh
pytest tests/unit/test_watch.py tests/unit/test_scene_watch.py tests/unit/test_pipeline.py tests/unit/test_ingest_guards.py tests/unit/test_recorder_health.py tests/unit/test_shoreline_controls.py -q
```

Some tests simulate catalogue/stage responses; they establish software behavior.
For a real cached NL acquisition, replay the saved ingest manifest:

```sh
python -m agents.pipeline --input missions/mission_20260927_213023_66d3167d/ingest.json --product-id 66d3167d-240a-459a-8066-75b9d2a458f2 --aoi eastern_newfoundland --output data/watch/replay-66d3167d.json
```

That command requires the existing local scene/manifest and pinned model; it
does not download a replacement scene. Verify the selected, top-level processed,
inference and evaluation UUIDs agree. It saves the full result and separate
execution artifacts. This acquisition had no usable recorded AIS, so successful
processing must not be reported as a measured benchmark.

Saved-payload replay assigns a new execution mission and copies the filtered AIS
into it. It retains the input identity as `source_mission_id` and
`source_mlflow_run_id`. When tracking is available, the replay starts a new
`mlflow_run_id` and records those lineage fields as tags; downstream stages and
failure handling use only that new run. If tracking is disabled or unavailable,
the replay proceeds without resuming the source run.

Results are recorded in [AUDIT.md](AUDIT.md). The NL scope and
[scientific evaluation limits](../EVALUATION.md) continue to apply.

The real replay completed on 2026-10-05: all four UUIDs matched, inference was
complete, and evaluation was explicitly unmeasured. The repository retains a
[compact evidence record](evidence/acquisition_replay_20261005.json) with model,
configuration and input hashes. Full scene artifacts remain locally under data,
missions and reports; they are not committed.
