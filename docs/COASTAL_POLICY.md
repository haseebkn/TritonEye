# Physical shoreline and coastal operating policy

Implemented 2026-10-05. St. John's harbour and The Narrows remain open water in
the existing OSM geometry. The eight municipal-imagery controls and their source
provenance are unchanged. No replacement shoreline or buffer reduction was adopted.

## Separate physical classification from operating policy

`physical_surface` is land, water or unknown **relative to the reference polygon**.
Infrastructure proximity and buffer changes do not change it. Boundary points
are conservatively land, not a surveyed tidal waterline. Signed shore distance
is retained without rounding before policy decisions.

`surface` remains a compatibility **operating-zone** field: open water, coastal,
land, infrastructure proximity or unknown. New outputs also carry `coastal_zone`,
`coastal_buffer_m`, `coastal_policy_version`, `alert_eligible`,
`alert_eligibility_reason`, `infrastructure_proximity`, `shore_distance_unbounded`
and `research_retained`. Unknown distance does not mean open ocean. Consumers
recompute eligibility and fail closed on contradictory new fields. Legacy
artifacts remain readable but are not silently upgraded into physical evidence.

`alert_eligible` means eligibility for the **open-water research candidate path**,
not permission to issue an operational alert. `operational_alert` stays false.
The default buffer remains 300 m; no accuracy evidence justifies changing it.

All in-region detections stay in `detections.geojson` and `correlation.geojson`.
`coastal_research.geojson` preserves coastal returns with
`coastal_review_required`, even without AIS. They are not mixed into the
open-water uncorrelated/ambiguous `review_candidates.geojson` path. HTML reports
have a dedicated harbour/coastal research table. Cached replay additionally
exports `open_water_research.geojson`.

## Coverage: sparse controls, not certification

Production registry: `configs/coastline/regional_controls.json`. Runtime checks
controls inside each scene bounding box and records each source collection.
Mission `landmask.shoreline_validation` retains registry and control hashes,
per-collection provenance and results. A scene containing no registered controls
reports `outside_scope`; passing sample checks does not certify intervening shoreline.
Replay separately tests **actual valid SAR pixels**. A bounding box alone does
not prove that a site was imaged. Mismatches and missing or malformed controls
make the mask unavailable; unknown shoreline detections do not become eligible.

| Location | Controls | Reference | Selected real-scene coverage |
| --- | ---: | --- | --- |
| St. John's basin/Narrows/approaches | 8 | City Imagery2022 | Neither selected scene; retained context/regressions |
| Bonavista harbour/bays | 5 | Esri/Vantor; centre citation 2025-08-01 | August 17: 5/5 on valid SAR pixels |
| Lewisporte marina/bay | 5 | Esri/Vantor; centre citation 2025-06-04 | September 27: 5/5 on valid SAR pixels |
| Labrador | 0 | None in this inventory | No selected scene or checks |

OSM passes these sparse controls. CanVec passes five Bonavista and five
Lewisporte controls, but that does not establish complete quay topology or
justify adoption. At St. John's it still fails the north-quay control (7/8).
CanVec remains a local **comparison** source. Source geometry, metadata and
imagery overlays are saved in `data/reference/regional/`. Basemap imagery is
not redistributed with the repository.

An initial Bonavista candidate at image pixel [950,825] was rejected after
overlay review: it lies at the quay/water edge, about 6 m inside OSM and 3 m
inside CanVec, not clearly in the basin. It is documented in the reference
metadata and did **not** justify a polygon correction. The retained basin
control is in visibly clear water about 89 m from the OSM shoreline.

Four additional Twillingate western-approach comparison controls pass, but
neither selected scene images them. Their separate file is not in the production
registry or benchmark coverage. They were not adopted based on rectangular bounds.

These are visual development checks, not independent expert adjudication or
surveyed geolocation. Dates are citations at image centres, not every pixel.
Older imagery cannot prove a 2026 tidal waterline. Other swath shoreline
segments and western/southern Newfoundland remain unchecked.
**18 registered controls are not province-wide validation.**

## Real buffer replay: counts, not accuracy

Inventory: `configs/benchmarks/nl_coastal_inventory.json`. These cached scenes
are **development replays**, not a labelled validation benchmark. Derived
outputs are separate from original missions in
`data/validation/coastal_policy_20261005/`.

| Buffer | Aug 17 open-water / coastal returns | Sept 27 open-water / coastal returns |
| ---: | ---: | ---: |
| 0 m | 84 / 0 | 144 / 0 |
| 100 m | 27 / 57 | 32 / 112 |
| 300 m (unchanged) | 17 / 67 | 12 / 132 |
| 500 m | 10 / 74 | 6 / 138 |
| 1,000 m | 6 / 78 | 3 / 141 |

Total valid-pixel returns: 312 and 1,032, including 228 and 888 physical-land
returns. **These counts are not vessel labels or false alarms.** Neither scene
has usable in-swath AIS or independent SAR vessel labels. Both correctly stay
unmeasured; precision, false alarms and missed vessels are null. No buffer is
selected from these counts. Compact versioned evidence:
`docs/evidence/coastal_policy_replay_20261005.json`.

With aligned AIS, evaluation separately reports harbour/coastal, open-water and
land/shoreline-mismatch denominators, raw matches, eligible matches and misses.
That remains AIS-subset proximity recall, not precision or complete truth.
Harbour here includes nearshore coastal waters, not an official port boundary.
Stratification reloads the shoreline source and cache recorded by inference in
the payload's `landmask` metadata; a relative cache path is repository-relative.
If the physical reference cannot be loaded, strata remain unmeasured.

## Independent validation-label contract

`python -m agents.coastal_benchmark` compares fixed physically annotated returns
against an **exhaustively reviewed valid-imagery ROI**. A label GeoJSON
FeatureCollection must have a `benchmark` object containing:

```text
split: validation
product_id: exact Copernicus UUID
acquisition_time: exact detection acquisition time
annotation_complete: true
annotation_scope: exhaustive_vessels_on_valid_imagery
source_kind: independent_sar_review
source, reviewer, licence: actual provenance
valid_imagery_roi: valid WGS84 Polygon/MultiPolygon inside NL
```

Each label is a WGS84 Point with unique `label_id` and `class` (`vessel` or
`non_vessel`). Resolve uncertain objects before declaring exhaustive review;
exclude unresolved/nodata/unreviewed areas from the ROI, not as negatives.
`coastal_replay` adds physical annotations and checks label pixels. The direct
benchmark command expects these annotations already present, plus detection
UUID/time and `shoreline_status: ok` metadata. False alarms require genuinely
exhaustive independent review, not merely setting a boolean in a file.

Buffer comparison and replay enforce held-out product, datatake and geographic
assignments even without labels or detections. Native acquisition identity must
resolve from a SAFE product name (`sar_product` or `name`) or the registered
dataset manifest; a declared `acquisition_group` alone is insufficient.
Production detection GeoJSON and
processing records retain `sar_product`, `sar_product_id`, `acquisition_time`
and the derived `acquisition_group`. Unresolved or conflicting datatake identity,
an explicit test split, or a held-out UUID/datatake rejects comparison.

Geographic scope must resolve from `study_roi` or `valid_imagery_roi`, otherwise
`footprint` or `scene_footprint`, or the registered product's footprint. Both scope
and feature geometries are checked against held-out areas and their locked
separation zone, including acquisitions from other datatakes. Declared held-out
geographic groups and ROI IDs also reject comparison. Production detection
GeoJSON retains `scene_footprint` even when its feature list is empty. Legacy
inputs need resolvable native identity and geographic scope.

Replay preflights all input scenes before writing derived outputs. Existing
detection UUIDs, native product names and acquisition times must agree with the
selected scene and payload; only absent legacy fields may be populated. The
output directory must be new and must not alias source evidence. Existing
directories, symlink destinations and duplicate product destinations are rejected.

The evaluator rejects AIS-only truth, incomplete/uncertain annotations,
holdout tuning, mismatched UUID/time and non-NL geometry. All labelled vessels
remain in the fixed denominator, including policy-withheld vessels. It reports
true positives, false-alarm counts, missed vessels, detector misses,
policy-added misses, per-regime label results and coastal research matches.
One-to-one matching prevents double credit; ambiguous associations are reported.
False alarms/km² remain null without a valid-water-area denominator.
Synthetic tests establish accounting behavior, **not regional performance**.
No buffer/configuration/model is automatically selected or promoted.

Remaining measurement work: independently reviewed labels for untouched NL
validation scenes and Labrador coastal-policy replays/imagery controls, then
buffer selection under an explicit false-alarm constraint while reporting misses.
Use locked holdout groups from the [historical review pilot](NL_BENCHMARK.md).
Public training datasets cannot replace labels of the 2026 acquisitions.
The deployed ensemble's upstream data also requires a
training-overlap audit before reuse as validation:
[official xView3 model documentation](https://github.com/DIUx-xView/xView3_first_place).

## Reproduce

```bash
# Explicit public imagery fetch; no credentials were used.
python scripts/inspect_regional_reference.py --fetch --site bonavista
python scripts/inspect_regional_reference.py --fetch --site lewisporte
# Reuses existing cached CanVec archive and OSM; no download.
python scripts/inspect_regional_reference.py --compare --site bonavista
python scripts/inspect_regional_reference.py --compare --site lewisporte
python -m agents.coastal_replay --output-dir data/validation/coastal_policy_replay
pytest tests/unit/test_coastal_policy.py tests/unit/test_coastal_replay.py -q
```

SAR inputs, weights, raw AIS and derived mission outputs stay ignored by Git.
Evidence records product IDs, hashes of source/derived artifacts, code,
configurations and controls. Original mission artifacts are not overwritten.
Change the inventory's explicit local source paths when replaying on another host.

Source attribution: [OSM](https://osmdata.openstreetmap.de/data/land-polygons.html),
[CanVec](https://open.canada.ca/data/en/dataset/9d96e8c9-22fe-4ad2-b5e8-94a6991b744b),
[municipal imagery](https://map.stjohns.ca/mapsrv/rest/services/Imagery/Imagery2022/MapServer),
[Esri metadata guidance](https://www.esri.com/arcgis-blog/products/arcgis-enterprise/imagery/learning-more-about-the-world-imagery-basemap).
