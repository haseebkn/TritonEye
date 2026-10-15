# Targeted AIS collection: deployment and three planned HH/HV passes

On **15 October 2026**, the existing AIS recorder was rebuilt from main
`b81d91b579b122ce812ebcdc2550ab335851e02b`, tested, and replaced in place.
Three future collection opportunities were reserved from official ESA plans
before imagery or scores. **Zero vessel labels or training exports are created.**

## Verified deployment

The [deployment receipt](evidence/targeted_recorder_deployment_v010.json) records
a point-in-time check at **14:36:41 UTC**. The running image is
`sha256:b1a7da049a5fde12f819d7a1e70732bc9e7f1cf4936cf4f05d3d14f5baf297ae`.
Its recorder source hash exactly matches the previously validated main source.
The image passed **47 recorder, heartbeat and pass-audit tests** before deployment.

- Exactly one running Compose AIS recorder was found, with one recorder process.
- The live server acknowledged the subscription and confirmed compression;
  the negotiated WebSocket extension also recorded compression as enabled.
- Heartbeats and observation timestamps were fresh. The bounded check found
  **27 new-session reports**, and all 27 were present in the captured CSV prefix.
- The previous session stopped at 14:32:35.274670 UTC; the new one started at
  14:32:37.216066 UTC. This **1.941396-second session transition** is not a count
  or duration estimate of lost vessel reports. First new receipt was at
  14:33:00.727390 UTC, for a provider observation at 14:32:58.449439 UTC.
- Existing data mounts, NL subscription envelope, Class A/Class B message types
  and restart policy remain in place. No second recorder was launched, no
  unrelated service was restarted, and no existing container/data was pruned.

This proves successful deployment and reception at the check time. It cannot
prove future availability, packet-loss rate or reception within the reserved
cells during future acquisitions. Historical acknowledgement/compression remain
unknown; new telemetry does not repair historical evidence.

## Frozen collection reservations

The [manifest](../reservations/targeted_collection/v0.1.0/manifest.json) and
[policy](../configs/reservations/targeted_collection_v010.json) freeze the next
three eligible **planned native acquisition groups**, using the official
[Sentinel-1 acquisition plans](https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans).
The downloaded C plan covers 14 October–5 November; the D plan covers
13 October–2 November. Original KML files and source URLs/hashes are retained.
Plan times without timezone suffix are interpreted as UTC.

| Date (2026) | Mission / planned native group | Full planned segment UTC | Collection context UTC | Target cell |
| --- | --- | --- | --- | --- |
| October 17 | S1C / `S1C_009923_013C84` | 09:55:43–10:06:50 | **09:25:43–10:36:50** | Western NL: 58.0–57.9°W, 48.9–49.0°N |
| October 18 | S1D / `S1D_005068_009965` | 09:50:59–09:58:48 | **09:20:59–10:28:48** | Same western NL cell |
| October 20 | S1D / `S1D_005097_009A65` | 09:34:37–09:42:32 | **09:04:37–10:12:32** | East of Avalon: 52.5–52.4°W, 47.5–47.6°N |

The context spans the entire planned segment plus 30 minutes each side. A long
segment's midpoint is **not** used as a local vessel acquisition time. Actual
product UUIDs/names remain null until catalogue reconciliation. A plan can change,
fail to execute or yield incompatible processing; reservation is not delivery.

Selection reuses the v0.4.0 frozen AIS archive through 15 October 13:45:40 UTC,
224,917 unique reports, and its bounded 132-cell neighbourhood. All current
acquisition/geographic roles are rechecked. A whole cell must fit the planned
footprint, pass existing NL/training/protected checks, and contain at least one
finite moving AIS report (1 to below 102.3 knots) in the preceding seven days.
These are prospective collection criteria, **not relaxed object-admission rules**.

The western target has 33 such reports from four identities; the Avalon target
has 34 from six. Reports and repeated vessels are not independent examples.
The earliest passes are selected, with existing cell rank/ID breaking target
ties. The October 19 plan covers other eligible cells but none has qualifying
recent moving reception. Thirteen future opportunities satisfy the collection
screen in the downloaded plans; only the first three are reserved. The first
two share geography, so the batch has three planned acquisition groups and two
geographic groups, not three geographically independent samples.

## Collection and post-pass handoff

The existing continuous recorder already covers all target cells and their
surroundings. It is left running across these windows, without MMSI filtering
or a separate scheduled recording process. No automatic imagery download,
scoring, annotation or promotion is enabled by this manifest. The host and
Docker must remain awake and running; sleep/power settings were not changed.

Before each pass, recheck the official plan and actual local heartbeat,
acknowledgement, compression and recent reports. Preserve revisions additively;
do not silently replace the sealed plan. After each pass:

1. Snapshot AIS/journals through the context end. Audit regional receipts,
   source times, gaps and the available competing tracks; global traffic alone
   does not establish target-cell reception.
2. Match actual catalogue products to the planned satellite/orbit/datatake and
   actual time/footprint. Retain exact UUIDs and native group identity; adjacent
   slices do not become independent passes.
3. Reconcile this manifest's training intent with **all current** split,
   exposure/quarantine and native-input checks before pixel access. Only target
   cells are reserved for training intent; protected areas elsewhere in a wide
   satellite swath are not authorized for inspection. The collection manifest
   is not automatically an annotation reservation or an admission override.
4. Apply unchanged source requirements at native object time: close/bracketing
   AIS, motion/shore constraints, uncertainty coverage and competing-return
   review. Reserve actual qualifying windows, then annotate before scores.
   Unknown objects/background remain ignored; all labels remain provisional.
5. Use the existing provisional admission/export workflow only if its evidence
   checks pass. Zero qualifying examples after these three passes is a reason
   to reassess the data source, not to relax labels or claim model improvement.

All three passes are after the earlier October 16 portfolio deadline. The
current presentation should describe this as a pending collection trial.

## Reproduction and validation

Private source plans, frozen priors and bounded deployment evidence are in
`data/research/targeted_passes_v010_01/`; public outputs omit vessel identities
and individual positions. Private CSV/journal snapshots preserve original bytes.
The public lock binds the manifest, inputs and implementation. Local replay
checks the retained archive hashes, recomputes moving reception, split decisions
and chronological selection, and compares the complete serialized result.

Local validation passed **124 focused tests**, including twenty collection
cases (ten covering selection/reception, ten covering policy-limit validation
and tampered-admission-flag rejection), plus the separate 47-test
pre-deployment image check. Ruff and mypy passed across 179 Python files; the
new files pass Black. Retained-data local replay selected the same three
groups. Automated review and CI are additional delivery gates.

```sh
python -m agents.targeted_collection --verify
python -m agents.targeted_collection --verify --local
pytest tests/unit/test_targeted_collection.py tests/unit/test_ais_recorder.py tests/unit/test_recorder_health.py tests/unit/test_ais_pass_audit.py -q
```

Publication refuses existing release paths. New plan versions or later evidence
require an additive release. No existing guard implementation, quarantine file,
locked-test area, original annotation or input specification was changed.
