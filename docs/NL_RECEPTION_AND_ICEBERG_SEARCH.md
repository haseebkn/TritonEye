# NL reception map and source-first iceberg search

Research evidence collected on **15 October 2026**, against main `1e1362e`
(PR #39). All labels remain provisional. This release creates no annotations,
training labels, training export or acquisition reservation. No model runs or
scientific independent review took place.

## Observed AIS reception

The [reception evidence](evidence/nl_reception_v010.json),
[self-contained map](evidence/nl_reception_v010.html) and
[GeoJSON](evidence/nl_reception_v010.geojson) cover the existing NL study boundary.
The archive snapshot contains **224,120 unique reports from 345 identities**, from
18 August 2026 at 20:08:37 UTC through 15 October at 11:55:59.862116 UTC.
All normalized reports are in scope. There are **374 observed 0.1-degree cells**;
**255 pass the whole-cell training-geography check**. Cells have unequal physical
area. Message counts are reception evidence, not traffic density, coverage
completeness or independent sample counts. Grey areas on the map contain no
received reports in this snapshot; this cannot distinguish absent vessels,
receiver limits and recorder outages.

The snapshot freezes complete-line file prefixes sequentially rather than
transactionally, without stopping the recorder. It retains the existing explicit
UTC interpretation of **19,621 legacy naive timestamps**. Exact source bytes,
coverage journals, full competing-track pools, catalogue responses and frozen
prior roles remain under `data/research/nl_reception_v010_01/`; public outputs
contain aggregate cells and product identities, with no MMSIs or individual tracks.

## Training-area shortlist

The [frozen discovery rules](../configs/reservations/nl_reception_v010.json)
require at least two reporting days and two identities per whole cell before
shortlisting. They rank motion/uncertainty-qualified native groups, eligible
HH/HV timing groups, days, reporting hours and identities. These are provisional
research screening choices, not C-CORE criteria. Eight cells are shortlisted;
the three with eligible SAR timing are:

| Cell bounds, longitude / latitude | Received reports | Reporting days | Identities | Eligible HH/HV native groups | Motion-qualified groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| −58.0 to −57.9 / 48.9 to 49.0 | 30,842 | 31 | 21 | 2 | 0 |
| −58.1 to −58.0 / 48.9 to 49.0 | 13,908 | 31 | 17 | 2 | 0 |
| −55.6 to −55.5 / 51.3 to 51.4 | 1,089 | 2 | 7 | 1 | 0 |

The first two western Newfoundland cells share the same two native acquisitions:
**11 October, Sentinel-1D**, product `0091a45d-08d2-44f4-9bdd-61e01f24f082`,
and **12 October, Sentinel-1C**, product `e15bd497-5e70-41b3-bbe6-b51e53508290`.
The northern Newfoundland cell has **19 September, Sentinel-1D**, product
`18b64506-e455-4d65-a171-16aadfcdc2d7`.
This is **three distinct native acquisition groups**, not five independent
acquisitions. The adjacent western cells must be considered together when defining
future geographic groups; these proposed cell names do not freeze new split roles.

Five further shortlisted cells have repeated reception but no eligible HH/HV
timing overlap. One additional eligible timing lead falls in a cell that does
not meet the repeated-reception shortlist screen; it remains in the complete
map/evidence. The shortlist is therefore not an exhaustive list of timing leads.

**No current lead passes the motion screen.** Across eligible timing leads,
nearest recorded speeds range from **0.0 to 0.8 knots**, below the existing
1-knot source-discovery criterion. The screen also requires finite AIS alignment
uncertainty at most 500 m. Strong harbour reception must not be described as
usable moving-vessel training evidence. These speeds do not prove that a SAR
reflector is a vessel, or that a reported slow vessel was berthed.

Track histories are aligned before spatial filtering, so interpolation can cross
a cell boundary. A timing opportunity needs a report within 120 seconds and an
AIS bracket no wider than 240 seconds at catalogue midpoint. Actual training
admission still requires native object-time support, competing-vessel review,
physical-water/shoreline inspection, native input replay and the remaining
hash-bound evidence. No SAR pixels were opened to select these areas.

Whole-cell geometry and separate 3 km candidate envelopes use the existing
10 km protected-geography separation. Existing validation/test/quarantine roles,
acquisition assignments and exposure remain binding. Native/COG aliases count
once; adjacent slices retain their shared native group. A swath intersecting
protected geography is inspected as metadata only; every proposed envelope must
pass its own role checks before any future pixel access. The evidence records
15 protected-geography exclusions, 10 validation/conflicting-role exclusions and
one incomplete-swath-envelope exclusion among tight track opportunities.

## Iceberg evidence search

The [source-search evidence](evidence/nl_iceberg_search_v010.json) preserves a
bounded search and all request/response receipts. The existing ranked IIP/SAR
overlap pool provides **zero new eligible source anchors** under the chosen
2018-or-later, medium/large, visual/radar-visual, at-most-900-second screen and
existing exposure/split restrictions. This is not a claim that no other NL
iceberg evidence exists.

The source-first fallback then freezes **six previously unreserved 2020 IIP
radar/visual sightings in eligible training geography**, before querying optical
catalogues. These are source leads with **no established SAR overlap**, rather
than new SAR object labels. Their dates are 27 September, 20 September,
9 September, 23 August (two sightings) and 17 August. Repeated iceberg identities
are grouped; no more than two anchors come from one provisional geographic group.

For each source, Sentinel-2 L1C and Landsat Collection 2 Level 2 were searched
within **±1 hour**, using a **3,900 m discovery envelope**: 300 m plus 1 m/s times
one hour. This is an exploratory search allowance, not a calibrated drift model
or enlarged label-association gate. All **12 catalogue requests completed**,
with **zero optical products returned**. No optical or SAR pixels were inspected
and no iceberg corroboration was established. The saved report distinguishes
failed/truncated responses from a complete empty response; broader dates,
other sensors and other source observations remain unsearched.

Four primary public pages were downloaded to
`data/source_supported/nl_iceberg_search_v010_03/`, with hashes and retrieval times:

- [NSIDC IIP dataset description](https://nsidc.org/data/g00807/versions/1)
  and [archive listing](https://noaadata.apps.nsidc.org/NOAA/G00807/): the listing
  currently ends at **2021**, consistent with the published temporal extent.
  These files offer no direct C/D-era observations.
- [IIP photo gallery](https://www.navcen.uscg.gov/international-ice-patrol-photos):
  the retrieved page offers photographs without an object-linked observation
  time/position table. Images and embedded EXIF were not downloaded or inspected;
  this is not proof that underlying originals lack metadata.
- [IIP annual-report index](https://www.navcen.uscg.gov/international-ice-patrol-annual-reports):
  later reports exist, but index links do not establish same-reflector evidence.
  Their PDF contents were not audited in this bounded search.

No organization was contacted. The IIP citation is: International Ice Patrol
(1995), *International Ice Patrol (IIP) Iceberg Sightings Database*, G00807,
Version 1, NSIDC, [DOI 10.7265/N56Q1V5R](https://doi.org/10.7265/N56Q1V5R),
accessed 15 October 2026; subset and decisions are recorded in the search policy.
Original source permissions and provenance remain subject to existing admission
checks; gallery/report availability alone grants no training-label permission.

## What the findings support next

Prioritize checking prospective reception and alternative historical AIS samples
around the western Newfoundland cells, requesting the full competing-vessel pool
when another source is available. Freeze any new reservation before pixels or
classifier scores, with the adjacent cells in a consistent geographic group.
The present timing leads do not justify bypassing motion, identity or native
input requirements. Native C/D processing remains limited to the previously
accepted recipe and processor version until each new product is checked.

For icebergs, additional observer records or another bounded source-first optical
search is needed. Historical A/B source evidence cannot establish C/D class
performance. Prior project exposure is disclosed; further AI inspection does not
provide physical corroboration or independent scientific review. Zero real labels
are admitted here. Original annotation masks, quarantine, locked test and review/
promotion implementations are unchanged.

## Replay and validation

Public replay needs the committed report, locks, policies and map only:

```sh
python -m agents.nl_reception --verify
python -m agents.nl_iceberg_search --verify
```

Retained private data additionally enables deterministic reception, split,
source-selection and request/response replay without contacting providers:

```sh
python -m agents.nl_reception --verify --local
python -m agents.nl_iceberg_search --verify --local
python -m pytest tests/unit/test_nl_reception.py tests/unit/test_training_ais_coverage.py -q
```

Acquisition commands `--build` and `--search` need fresh private directories and
fresh public versions. `--publish` seals retained data only when release paths
do not exist; it never overwrites an existing release. Code hashes bind each
release to its implementation. Public map replay recomputes the HTML and GeoJSON;
local replay checks snapshot bytes and reconstructs overlaps before ranking.
Behavioral tests exercise protected and conflicting roles, exposure, incomplete
swaths, stationary/uncertain tracks, boundary crossing and repeated sources.
Local validation passed **82 focused tests**, including retained-data admission
regressions. Both new evidence releases replay with the private inputs. Ruff,
Black and mypy pass across **172 Python files**. The publication workflow also
runs automated review, portable tests and CI before merging.
