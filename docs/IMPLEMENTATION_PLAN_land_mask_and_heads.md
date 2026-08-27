# Implementation Plan — Land Masking, Head Decoding, and Iceberg Disclosure

Status: proposed, not yet implemented
Author: drafted 2026-08-24
Target: close the four gaps that make the current output unpresentable to a SAR
domain audience.

---

## Motivation

The 2026-08-17 eastern Newfoundland scene produced **298 detections, essentially
all of them flagged dark, with the majority clustered over the Newfoundland
landmass rather than open water.** The open Newfoundland Sea to the east is
almost empty of detections. That spatial distribution is diagnostic: the
detector is responding to terrain, not vessels.

Two independent defects produce that outcome, and a third and fourth limit how
the surviving detections can be described:

| # | Gap | Consequence |
|---|-----|-------------|
| 1 | No land masking anywhere in the pipeline | Terrain returns enter the alert list as vessels |
| 2 | That scene has never been re-scored | The headline number (298) is known-wrong and still published |
| 3 | `SIZE` / `VESSEL` / `FISHING` heads sampled but never decoded | Every target reports `class_name: unknown`, nominal 15 px box |
| 4 | Iceberg ambiguity undocumented | In Iceberg Alley, an iceberg is indistinguishable from a dark vessel |

Items 1 and 2 are correctness. Items 3 and 4 are about saying only what the
evidence supports.

---

## Task 1 — Land masking

### 1.1 Design decision: mask points, not pixels

The obvious implementation is to rasterize a coastline into a scene-sized
boolean array and mask the imagery before inference. **Do not do this.** A full
IW scene is roughly 25000 x 16000 pixels; rasterizing to that grid costs
hundreds of MB and a great deal of time, and this pipeline has already been bitten
once this month by full-raster allocations (see `interpolate_lut_grid` in
`agents/calibration.py`).

Instead, mask **after** georeferencing, in geographic space, against the ~300
detection centroids. Point-in-polygon over 300 points against a clipped
coastline is milliseconds. The detector still wastes compute on land tiles, but
that is a throughput problem, not a correctness one, and can be optimised later
by skipping tiles whose footprint is fully inland.

### 1.2 Design decision: flag, do not silently drop

Detections are annotated, never deleted:

```
properties:
  surface:            "water" | "land" | "coastal"
  distance_to_shore_m: float      # negative = inland
```

Only `surface == "water"` enters the dark-vessel alert list. The others stay in
`detections.geojson`.

Three reasons this matters more than it looks:

- **The rejection count is the evidence.** "298 raw → N marine after land
  masking" demonstrates you found and fixed a false alarm problem. A bare `N`
  demonstrates nothing.
- Georeferencing is `gcp_tps` with residual error; a hard cut would silently
  delete genuine harbour and nearshore traffic.
- It is auditable. A reviewer can inspect what was rejected and why.

`coastal` covers detections within a configurable distance of shore (default
**300 m**, roughly 30 pixels at 10 m spacing — comfortably above the measured
TPS georeferencing residual). These are genuinely ambiguous: real nearshore
vessels and land bleed both live there. Reporting them as a distinct third
class is more honest than forcing a binary.

### 1.3 Data sources — all open, none paid

Primary and fallback, in preference order:

**A. OSM land polygons — recommended primary**
- <https://osmdata.openstreetmap.de/data/land-polygons.html>
- File: `land-polygons-split-4326.zip` (WGS84, split; the split version keeps
  per-feature geometry small, which makes the spatial index far more effective
  than the single continent-sized polygons in the `complete` build)
- Licence: **ODbL** — free, attribution + share-alike on derived databases
- Derived from ways tagged `natural=coastline`. Newfoundland's coastline is
  well mapped in OSM.
- Useful property: because it is coastline-derived, **inland lakes are part of
  the land polygons**, not holes in them. Newfoundland's interior is full of
  ponds, and this is exactly the behaviour we want — no special handling.

**B. CanVec Hydrographic Features — recommended refinement for NL**
- <https://open.canada.ca/data/en/dataset/9d96e8c9-22fe-4ad2-b5e8-94a6991b744b>
- Prepackaged shapefiles: <https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/shp/Hydro/>
- Licence: **Open Government Licence – Canada** — free, attribution only
- Natural Resources Canada's national topographic vector product. Higher
  positional fidelity over Canada than OSM in less-mapped stretches. Packaged
  per NTS tile, so it needs tile selection logic for a scene footprint — more
  plumbing than A, which is why it is the refinement rather than the default.

**C. Newfoundland & Labrador provincial open data — for provincial context layers**
- <https://opendata.gov.nl.ca/> and the ArcGIS Hub at
  <https://gis-and-mapping-gnl.hub.arcgis.com/> (exports GeoJSON/shapefile directly)
- Provincial coverage: protected areas, water resources, ecoregions. Not a
  better coastline than A or B, but the right source if the report ever needs
  protected-area or fishing-zone overlays.

**D. GSHHG — public-domain fallback**
- <https://www.soest.hawaii.edu/pwessel/gshhg/>
- Licence: **public domain** (data; the tooling is LGPL)
- Long-established in the SAR community. Single global download, no ODbL
  share-alike obligation. Worth wiring as a fallback specifically because its
  licence is the least encumbered — if the ODbL share-alike on A ever becomes
  awkward, this is the escape hatch.

**Recommendation:** implement against **A** now, structure the loader so **B**
and **D** are drop-in alternates, and record the choice in the mission payload
so any report states which coastline produced its mask.

### 1.4 New module — `agents/landmask.py`

```python
LandMask.fetch(source: str = "osm", cache_dir: str = "data/reference") -> str
    """Downloads and caches the coastline archive. Idempotent; no-op if present."""

LandMask.for_footprint(bounds: Sequence[float], source: str = "osm") -> "LandMask"
    """Loads only polygons intersecting the scene bbox, into an STRtree."""

LandMask.classify(lons, lats, coastal_buffer_m: float = 300.0)
    -> Tuple[np.ndarray, np.ndarray]
    """Returns (surface_labels, distance_to_shore_m) for detection centroids."""
```

Notes on implementation:

- `geopandas` 1.1.4, `shapely` 2.1.2, `pyproj` 3.7.2, `pyogrio` 0.13.0 are all
  already installed — **verified, no new dependencies required.** (`fiona` is
  absent, but geopandas 1.x reads shapefiles through `pyogrio`, so this is fine.)
- Use `shapely.STRtree` for the index. With shapely 2.x, `tree.query()` is
  vectorised over point arrays — do not loop.
- Distance-to-shore must be metric, not degrees. Reproject the clipped
  coastline and the points to a local metric CRS.

  **Do not use a fixed UTM zone.** The 2026-08-17 footprint spans
  lon −55.00…−51.17, which straddles the zone 21N/22N boundary at −54°W;
  `pyproj` returns both `EPSG:26921` and `EPSG:26922` as candidates for it.
  Picking either distorts the far edge of the scene.

  Use an **azimuthal equidistant projection centred on the scene footprint
  centroid**, built per mission:

  ```python
  CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
                 f"+datum=NAD83 +units=m +no_defs")
  ```

  Verified against geodesic truth on this footprint: **0.0 m error at both
  corners and mid-scene** (176.59 km planar vs 176.59 km geodesic at the SW
  corner). AEQD is exact for distance-from-centre by construction, which is
  precisely the quantity being measured, and it works for any AOI without zone
  selection logic.
- Cache the clipped-and-reprojected coastline per AOI — reloading a global
  shapefile every mission is wasteful.
- The download is large (hundreds of MB). It must be **explicitly invoked**, not
  triggered silently inside a mission run, and the cache directory must be
  gitignored.

### 1.5 Wiring into the pipeline

In `agents/inference/inference_agent.py`, in the feature assembly loop around
line 583, after `lons`/`lats` are computed and before features are built:

1. Compute polygon centroids from the already-georeferenced ring coordinates.
2. `surface, dist = landmask.classify(centroid_lons, centroid_lats)`
3. Add `surface` and `distance_to_shore_m` to each feature's `properties`.
4. Write the tally to the payload:
   `payload["landmask"] = {"source", "water", "land", "coastal", "buffer_m"}`

In `agents/correlation/correlation_agent.py`: restrict dark-vessel candidacy to
`surface == "water"`. Land and coastal detections stay in the detections file
but must not generate alerts.

In `agents/report/report_agent.py`: show the rejection tally in the header
(e.g. `298 raw · 41 water · 12 coastal · 245 land`) and render land/coastal
detections in a muted style, distinct from marine targets.

### 1.6 Config

Add to `configs/model.yaml`:

```yaml
landmask:
  enabled: true
  source: "osm"            # osm | canvec | gshhg
  coastal_buffer_m: 300    # water/coastal boundary; ~30 px at 10 m spacing
  cache_dir: "data/reference"
```

`enabled: false` must reproduce today's behaviour exactly, so the before/after
comparison in Task 2 is a single config flip.

### 1.7 Tests — `tests/unit/test_landmask.py`

Build a synthetic coastline (a square "island" polygon) rather than depending on
a network download, and assert:

1. A point well inside the island → `land`, negative distance.
2. A point well offshore → `water`, positive distance beyond the buffer.
3. A point 100 m offshore with a 300 m buffer → `coastal`.
4. Distances are metric and sane — a point 1 km offshore returns ~1000 m, not a
   degree-valued number. (This is the test that catches the most likely bug in
   the whole task.)
5. `classify` on an empty input returns empty arrays, not a crash.
6. `enabled: false` leaves feature properties untouched.
7. Antimeridian / empty-intersection footprint degrades gracefully rather than
   raising.

Add an integration test asserting the correlation agent raises no dark-vessel
alert for a `surface: land` detection.

**Effort: ~1 day.** This is the highest-value item in the plan by a wide margin.

---

## Task 2 — Re-score the Newfoundland scene

Once Task 1 lands:

```bash
TRITONEYE_DETECTOR=xview3 python -m agents.ingest.ingest_agent --aoi eastern_newfoundland
```

Run the full chain on the same 2026-08-17 product, unchanged except the mask.

**Report both numbers.** The deliverable is the delta, not the new count:

```
2026-08-17 eastern Newfoundland, xView3 @ 0.15
  before land masking:  298 detections, 298 flagged dark
  after  land masking:  N water, M coastal, K land-rejected
```

Then, and only then, revisit the operating point. `configs/model.yaml` already
carries the note that the 0.15 threshold was chosen partly to suppress coastal
false positives, and that it should move toward 0.05 once a mask exists. With
land removed, re-run the sweep on the Boston scene — 0.05 gave 59% recall at
10.1 detections per AIS vessel, and a large share of that 10.1x was almost
certainly land. **Do not assume the ratio improves; measure it.**

Update `EVALUATION.md` §5a.4 (operating point) and §5a.5 (the "count is not a
vessel count" caveat) with the measured result. If precision is still
unmeasurable, §5a.5 stays — masking land does not turn recall into precision.

**Effort: ~2 hours wall clock**, of which ~90 min is unattended compute.

---

## Task 3 — Decode the SIZE / VESSEL / FISHING heads

### 3.1 Current state

`agents/xview3_detector.py:detect_tile` already samples two of the three heads
at each peak and stores them honestly:

```python
"size_raw":   float(size_map[hy, hx]),
"vessel_raw": float(vessel_map[hy, hx]),
```

`KEY_FISHING` is declared but never read. The existing comment is correct that
these are undecoded — the task is to decode them against evidence, not to
rename them and hope.

### 3.2 The schema bottleneck

`run_xview3_inference` returns a fixed 6-tuple
`(x1, y1, x2, y2, score, class_id)` shared with the YOLO path, so there is
nowhere to put extra attributes.

**Minimal change (recommended):** return a parallel `attrs: List[Dict]`
alongside `boxes`; the YOLO path returns `[{} for _ in boxes]`. The feature
assembly merges `attrs[idx]` into `properties`. ~30 lines, no refactor, no risk
to the YOLO path.

**Cleaner long-term:** a `Detection` dataclass in `agents/detections.py` that
both backends emit. Better design, wider blast radius. The 83-test suite makes
it tractable — but do it as its own change, not folded into this one.

### 3.3 Calibrating SIZE — use the data you already have

The xView3 challenge scored vessel **length in metres**, so the head is
predicting length in some parameterisation. Its exact scaling in this traced
ensemble is **not known** and must not be guessed.

You already have the ground truth to settle it: the Boston 2025-01-08 scene has
141 AIS vessels in swath, and AIS position reports carry vessel dimensions.

1. Extend the AIS parsing to retain reported vessel length where present.
   (Note: `parse_position_report` in `agents/ais_recorder.py` handles
   `PositionReport` only — dimensions arrive in `ShipStaticData` messages, so
   the recorder needs to subscribe to and join that second message type. This is
   a real subtask, not a footnote.)
2. For matched detection/AIS pairs from the Boston run, scatter `size_raw`
   against AIS length.
3. Fit the relationship. If it is linear with tight residuals, publish
   `length_m` with a stated error band. If it is not, **publish nothing** and
   say in EVALUATION.md that the head did not calibrate against available ground
   truth.

Only if calibration succeeds should `NOMINAL_HALF_EXTENT_PX` be replaced by a
measured extent.

### 3.4 Decoding VESSEL / FISHING — and what they do not mean

Sample all three heads. Apply a sigmoid only if values fall outside `[0, 1]`,
matching the existing objectness handling. Emit:

```
is_vessel_score:   float
is_fishing_score:  float
class_name:        "vessel" | "fishing_vessel" | "non_vessel" | "unknown"
```

Set `class_name` from the heads **only above a threshold justified by the same
AIS-matched data**; below it, keep `unknown`. Reporting an uncalibrated
probability as a class label would repeat exactly the failure this project
already documented in the incumbent YOLO path, which "silently labels everything
cargo."

**The critical caveat, which belongs in both the code and EVALUATION.md:**
xView3's `is_vessel` head separates vessels from *fixed marine infrastructure*
(rigs, platforms, wind turbines) — the negative class it was trained on. **It is
not an iceberg discriminator, and it is not a land-clutter discriminator.** A
low `is_vessel` score means "not a vessel by xView3's training distribution,"
which lumps together infrastructure, ice, and terrain. Do not present it as
target classification.

**Effort: ~half a day** for decoding and plumbing; the SIZE calibration and the
`ShipStaticData` join are a separate half-day and may conclude "not
calibratable," which is a valid outcome to publish.

---

## Task 4 — Document the iceberg ambiguity

Add **§8.x** to `EVALUATION.md` (Limitations). This is the item most specific to
a Newfoundland deployment and the one a C-CORE reviewer is most likely to raise
first.

The paragraph must state plainly:

- The AOI is in **Iceberg Alley**. Icebergs and bergy bits are routine in these
  waters, seasonally concentrated roughly spring through mid-summer.
- Icebergs are **bright, compact targets against a dark sea** — the same
  signature the detector is trained to find. There is no reason to expect the
  xView3 ensemble to reject them; its training negatives are fixed
  infrastructure, not ice.
- **The failure mode is structural, not incidental.** This system flags a target
  as a dark vessel when it detects a target with no correlating AIS transmission.
  An iceberg has no AIS transmitter. **Every detected iceberg therefore satisfies
  the exact definition of a dark vessel that this pipeline uses.** It is not that
  icebergs are hard to classify — it is that the dark-vessel logic converts each
  one into a false alert by construction.
- No claim is currently made about ship/iceberg discrimination, and **none
  should be inferred from these results**.

Then state what would address it, without claiming any of it is done:

- **Polarimetric discrimination** — VV/VH ratio and dual-pol behaviour differ
  between ice and metal hulls; both channels are already ingested and calibrated,
  so the input exists.
- **Temporal behaviour** — icebergs drift with current and wind; vessels
  transit independently. Multi-pass tracking (`agents/tracking.py` exists)
  separates them over repeat cycles.
- **Ancillary data** — the Canadian Ice Service publishes open iceberg and ice
  charts that could be joined as a prior.
- **Labelled ice data** — no ship/iceberg-labelled SAR training set for this AOI
  is currently in the project.

Cross-reference this section from `agents/xview3_detector.py`'s module docstring
so the caveat is visible from the code, not just the report.

**Effort: ~1 hour.** Cheapest item here, and the one that most changes how a
domain expert reads the whole project — naming the limitation yourself is far
stronger than being caught by it.

---

## Sequencing

```
Task 1 (land mask)  ──▶  Task 2 (re-score + threshold sweep)
                                  │
Task 3 (heads)  ──────────────────┘   independent, can run in parallel
Task 4 (iceberg doc)                  independent, do first — it is an hour
```

Do Task 4 first: it costs an hour and immediately removes the largest
credibility gap in the written record. Then Task 1 → Task 2 as the main line of
work. Task 3 is independent and can slot in around the ~90 min compute in Task 2.

## Definition of done

- [ ] `landmask.enabled: false` reproduces current output byte-for-byte
- [ ] Newfoundland report shows a marine-only alert list, with the rejection
      tally visible in the header
- [ ] Before/after counts recorded in `EVALUATION.md` with the delta explicit
- [ ] Threshold re-swept on Boston with land removed; §5a.4 updated with
      measured numbers, or explicitly stated as unchanged
- [ ] Head-derived fields appear only where calibrated; uncalibrated fields keep
      their `_raw` suffix and are absent from user-facing report text
- [ ] §8.x iceberg limitation written and cross-referenced from the detector
      docstring
- [ ] Data source, licence, and download date recorded in the mission payload
- [ ] `data/reference/` gitignored — **already covered**, `.gitignore:21`
      ignores `data/`; confirm no coastline archive is force-added
- [ ] Attribution present for whichever source ships: ODbL for OSM, OGL-Canada
      for CanVec

## Licence obligations to honour

| Source | Licence | Obligation |
|--------|---------|------------|
| OSM land polygons | ODbL | Attribute OpenStreetMap contributors; share-alike applies to derived *databases*, not to detection outputs |
| CanVec | Open Government Licence – Canada | Attribute Natural Resources Canada |
| NL Open Data | Provincial open licence — verify per dataset before shipping | Attribution |
| GSHHG | Public domain (data) | None; attribution courteous |

Add attribution to the report footer and `readme.md`. The Leaflet basemap
already carries its own Esri/GEBCO/NOAA/CHS attribution — extend that line
rather than adding a second one.

## Sources

- [OSM land polygons](https://osmdata.openstreetmap.de/data/land-polygons.html)
- [OSM coastline wiki](https://wiki.openstreetmap.org/wiki/Coastline)
- [CanVec Hydrographic Features (Open Canada)](https://open.canada.ca/data/en/dataset/9d96e8c9-22fe-4ad2-b5e8-94a6991b744b)
- [CanVec Topographic Data of Canada](https://open.canada.ca/data/en/dataset/8ba2aa2a-7bb9-4448-b4d7-f164409fe056)
- [CanVec Hydro shapefile FTP](https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/shp/Hydro/)
- [Open Data Newfoundland and Labrador](https://opendata.gov.nl.ca/)
- [GIS and Mapping NL (ArcGIS Hub)](https://gis-and-mapping-gnl.hub.arcgis.com/)
- [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
- [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada)
