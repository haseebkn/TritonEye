# St. John's shoreline verification

Checked on 2026-10-05. The previous diagnosis that OSM closed the harbour and
The Narrows was incorrect. The actual cached shoreline follows the basin and
channel. The restrictive 300 m coastal eligibility band is a separate issue.

## Source comparison and decision

The public City of St. John's Imagery2022 export covers the harbour, channel,
immediate approaches and surrounding land. Image extent and coordinate system
are retained with the downloaded image. Eight control points were located on
visible water or land, transformed from EPSG:32181 to WGS84 with longitude first,
and checked against both shoreline datasets.

| Imagery control | OSM signed shore distance | Default surface |
| --- | ---: | --- |
| Basin centre | +112 m | coastal |
| The Narrows channel | +133 m | coastal |
| Entrance water | +92 m | coastal |
| Approach open water | +562 m | water |
| North quay | −7 m | land |
| Downtown land | −204 m | land |
| South industrial land | −121 m | land |
| Signal Hill land | −455 m | land |

All eight physical land/water checks pass for the current OSM geometry.
Positive shore distance establishes water relative to that geometry; the
coastal band remains an uncertainty/exclusion policy, not a land classification.
No claim of measured vessel detection accuracy follows from these results.

CanVec 1:50,000 NL was also downloaded and clipped to this area. Its local ocean
feature has 1979 source dates and nominal 26 m horizontal-accuracy attributes.
Both datasets show an open channel, but CanVec treats the north-quay control
as ocean where municipal imagery shows built land. At that control, OSM's
land boundary is only 7 m away, so this is a local visual/topology warning,
not a statistically established accuracy ranking. CanVec is retained for
inspection and **not adopted** as an automatic replacement. Narrowing the
coastal band on this evidence alone would not establish safe harbour operation.

## Implemented behavior

`LandMask.for_footprint` checks the versioned imagery controls inside the actual
scene bounding box after loading and projecting the coastline. A closed channel,
known land classified as water, malformed controls or missing control file
raises `LandMaskUnavailable`. Inference records the failure and leaves surface
eligibility unassessable; scene-watch does not claim water coverage. No target is
promoted to a confirmed vessel or automatic alert by these checks.

Passing checks are retained in mission `landmask.shoreline_validation`, including
the control-file SHA-256 and reference description. A scene containing none of
these controls reports `outside_scope`, not regional validation. Controls are
sample locations; they do not certify every shoreline segment between samples.
The coastal band is unchanged, so genuine nearshore vessels can remain withheld.

## Reproduce locally

```bash
python -m agents.landmask --fetch --source osm
python scripts/inspect_harbour_reference.py --fetch
pytest tests/unit/test_landmask.py tests/unit/test_shoreline_controls.py tests/unit/test_scene_watch.py -q
```

The explicit `--fetch` flag downloads missing public sources. Running without it
uses the locally saved files and makes no network requests. Downloads, clips,
metadata, `verification.json` and `shoreline_comparison.png` are kept under
`data/reference/st_johns/`, which is ignored by Git. The NRCan archive is about
1.09 GB and the municipal image about 13.5 MB. No credentials are required.
The audit script uses GeoPandas, requests and Matplotlib from the project runtime.

A small attributed OSM clip in `tests/fixtures/` supplies offline regression
input; CI never downloads the large datasets or municipal imagery. The source
URLs, hashes, licences and local file times are recorded in `verification.json`.
Local file time is not necessarily an upstream publication or acquisition date.

## Limits and follow-up

Imagery identified by the service as 2022 cannot prove the current 2026 quay
layout or waterline at a particular tide. These are visual checks performed for
this project, not independent expert adjudication or surveyed accuracy tests.
Other Newfoundland and Labrador harbours remain unverified. Fresh imagery and
more shoreline controls are needed before broader replacement of the mask.
Changes in the real shoreline may require re-reviewing controls; do not bypass
a failed check by labelling an old reference as current.

Source attribution and access: [DATA_SOURCES.md](DATA_SOURCES.md).

## Separate operating policy and expanded sparse controls

The later implementation retains these eight controls unchanged and adds
physical-surface versus policy fields, dedicated coastal research artifacts,
and Bonavista/Lewisporte imagery controls for the selected real replays. No
CanVec replacement or buffer reduction was adopted. Neither these eight checks
nor the expanded sparse registry establishes province-wide accuracy. See
[COASTAL_POLICY.md](COASTAL_POLICY.md) for coverage, buffer counts and the
still-missing independent validation labels.
