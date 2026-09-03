# TritonEye — Evaluation Report

Measured behaviour of the SAR vessel detection and AIS correlation pipeline over
**Newfoundland and Labrador**, on real Sentinel-1 acquisitions.

**Operating area:** Newfoundland and Labrador only — the Grand Banks, the
Jeanne d'Arc Basin, and the approaches to St. John's.

---

## Summary

| Question | Answer |
|---|---|
| Is geolocation correct? | **Yes.** Sub-metre at all 210 control points on the NL product; an affine fit on the same product is out by 801 m mean, 2488 m max. |
| Are the detections vessels? | **Unknown, and the raw count is not a vessel count.** 72.8% of a real NL scene's detections were land. |
| Does false-positive suppression work? | **Yes, measured.** Land masking removed 217 of 298 detections; the alert list went from 298 to 17. |
| Is recall measured? | **No.** No historical AIS archive covers this region. This is the project's central open gap. |
| Is precision measured? | **No.** Land is masked, but clutter and ice are not. |
| Can it tell a ship from an iceberg? | **No** — and in Iceberg Alley that is structural, not incidental (§7.1). |

**The honest position: this is a working, instrumented detection and
false-positive-suppression pipeline whose detection quality over its own
operating area has not yet been quantified, because the ground truth to quantify
it does not exist yet and is being collected.**

---

## 1. Operating area and why it constrains everything

Newfoundland and Labrador, and nowhere else. That choice drives three
consequences that shape this entire report.

**There is no historical AIS archive for these waters.** MarineCadastre — the
usual free bulk source — is US Coast Guard data and holds **zero records east of
−67.4°W**. Newfoundland sits near −53°W. There is no Canadian public equivalent
with comparable coverage. Ground truth therefore has to be *collected*
prospectively, which `agents/ais_recorder.py` does against the live aisstream.io
feed, and which cannot be backfilled onto acquisitions already on disk.

**Land dominates the scene.** An island operating area means a Sentinel-1 swath
is mostly terrain. On the 2026-08-17 acquisition, **72.8% of raw detections fell
on land**, some **35.4 km inland**. Suppressing that is not a refinement; without
it the output is unusable.

**The region is Iceberg Alley.** Icebergs are bright compact targets against a
dark sea — the exact signature a vessel detector is built to find — and they
carry no AIS transmitter. See §7.1 for why this is structural.

### 1.1 Ground truth, and why so little of it exists

Where AIS coverage exists, every AIS-broadcasting vessel inside the swath at
acquisition time was **definitely present**, so the fraction detected is a
conservative **lower bound on recall** requiring no hand labels. Dark vessels
are absent from AIS by definition, so true recall can only be higher.

That method is sound and implemented (`agents/evaluate/evaluate_agent.py`). It
has not yet been *applied* to an NL scene, because no acquisition has yet
coincided with recorder uptime:

| recorded AIS window (eastern NL) | duration |
|---|---|
| 2026-08-18 20:08 → 2026-08-19 04:25 | 8.3 h |
| 2026-08-22 14:31 → 17:29 | 3.0 h |
| 2026-08-23 20:12 → 20:24 | 0.2 h |

Against six Sentinel-1 IW acquisitions over the AOI in the same week, **none
fell inside a coverage window**, and four of the six were HH/HV rather than the
VV/VH the detector requires. The binding constraint is recorder uptime, not
imagery availability. Sentinel-1 crosses this AOI near **09:30 and 21:30 UTC**.

---

## 2. Georeferencing

Sentinel-1 L1 GRD measurement rasters are stored in **radar geometry**: no CRS,
no affine geotransform, only a grid of ground control points. The NL product
reports `crs = None` with 210 GCPs in EPSG:4326.

Measured on `S1D_IW_GRDH_1SDV_20260817T212209` (25579 × 16677, 210 GCPs):

| method | mean residual | max residual |
|---|---|---|
| Affine (`rasterio.transform.from_gcps`) | **801.49 m** | **2488.22 m** |
| **Thin plate spline** (production) | **0.00 m** | **0.00 m** |

TPS interpolates exactly through every control point. The affine fit is off by
~800 m on average — inside the 2 km correlation buffer, but easily enough to
mis-associate vessels in traffic.

---

## 3. Detector selection

The pipeline runs the **xView3-SAR challenge-winning ensemble** (Eugene
Khvedchenya, MIT licence) rather than the general-purpose YOLOv8 it shipped with.

**The argument is geometric, and it does not depend on any particular scene.**
At 10 m ground spacing a 70 m vessel spans **7 pixels**. A stride-8 detector
reduces that to less than one output cell — the target is gone before the
detection head sees it. The xView3 ensemble predicts densely at **stride 2**.
It also won a challenge whose task (vessel detection in Sentinel-1 GRD) is
exactly the task here.

**What is deliberately not claimed:** any recall figure for this operating area.
Recall was not measured over Newfoundland and Labrador, because §1.1's ground
truth does not exist yet.

### 3.1 Three input details, each load-bearing

Each of these, when wrong, looks exactly like "the model is broken" rather than
"the input is wrong":

| | required | consequence of getting it wrong |
|---|---|---|
| Radiometry | calibrated σ⁰ in dB | normalisation saturates above ~+10 dB; raw DN (median ~165) drives every pixel to 1.0 — a uniformly white image |
| Tile size | exactly 2048×2048 | concrete sizes were baked in at trace time; anything else fails inside the ensembling stack |
| Channel order | **(VH, VV)** | measured on a known vessel: (VH,VV) peaked at 0.33 objectness 56 px from target; (VV,VH) at 0.11 and 918 px away |

### 3.2 Radiometric calibration

σ⁰ is computed from the product's calibration LUT, not from raw digital
numbers. Bilinear interpolation on the LUT grid is separable, so it is evaluated
once per axis rather than per pixel.

Measured on a 2048×2048 tile:

| variant | peak memory | dB error |
|---|---|---|
| pointwise float64 (original) | 704.6 MB | exact |
| float32 throughout | 67.4 MB | **1.5e-6** — rejected |
| **float32 LUT + float64 dB** | **205.6 MB** | **0.0** |

The middle row is recorded because it was briefly adopted: the claim that "the
result feeds a log so precision does not matter" was measured and found false —
float32 through the log exceeded this module's own accuracy tolerance, and
passed locally only on a numpy version that happened to round favourably.

### 3.3 Robustness

Two failure modes found by running full scenes end to end:

- **JIT kernel fusion is disabled.** The ensemble was traced under torch 1.10
  and runs under 2.x, so TorchScript compiles fused CUDA kernels at runtime via
  nvrtc/ptxas. Under host memory pressure that dies with `ptxas fatal: Memory
  allocation failure` partway through a scene. Running unfused costs throughput
  and removes the failure mode.
- **`detect_scene` survives per-tile failures.** A transient CUDA OOM should
  cost one tile, not a 90-minute run.

### 3.4 Cost

| | per scene |
|---|---|
| yolov8n | 47 s |
| xView3 ensemble, RAM headroom | **20.4 min** (126 tiles) |
| xView3 ensemble, host memory contended | 89 min |

Both xView3 figures are the same scene and the same detections
(332 raw → 298 deduped, bit-identical); only host memory differed. This is a
scheduled batch job, not an interactive one, which is why the backend is opt-in.

---

## 4. False positive suppression

This is where the measured work is, and it is the part of the system that most
directly serves an operator.

### 4.1 Land masking — the dominant mode

The 2026-08-17 eastern Newfoundland scene returned **298 detections at threshold
0.15, every one reported dark.** The spatial distribution was diagnostic before
any mask existed: dense clustering over the landmass, while the open sea east of
the island was nearly empty — the inverse of real traffic.

`agents/landmask.py` classifies detection centroids against OSM land polygons
(ODbL, derived from `natural=coastline`):

| surface | count | share |
|---|---|---|
| water (alert-eligible) | **17** | 5.7% |
| coastal (within 300 m of shore) | 64 | 21.5% |
| **land-rejected** | **217** | **72.8%** |

**Land detections reached 35.4 km inland, median 7.2 km.** That is not
georeferencing slop; it is terrain — rock, structures, radar-facing slopes —
detected as vessels tens of kilometres from navigable water.

End-to-end effect on the alert list: **298 → 17.**

**Method.** Classification runs on detection centroids in geographic space, not
on the raster: **581 ms** for 298 points, against the hundreds of MB a coastline
rasterised to the 25579×16677 scene grid would cost. Distances are metric via an
azimuthal equidistant projection centred on the footprint — a fixed UTM zone was
rejected because this scene spans −55.00…−51.17, straddling the 21N/22N boundary
at −54°W. Verified at **0.0 m error** against geodesic truth.

Detections are **annotated, never deleted**: `surface` and
`distance_to_shore_m` are written to every feature, and only `water` reaches the
correlator. The rejection tally appears in the mission payload and the report
header, because what was discarded is itself the evidence.

### 4.2 Confidence does not separate land from water here

Land contamination by confidence band, 2026-08-17:

| confidence | detections | on land |
|---|---|---|
| 0.15–0.20 | 185 | **78%** |
| 0.20–0.30 | 87 | **67%** |
| 0.30–0.50 | 21 | **57%** |
| 0.50+ | 5 | **60%** |

**Even the highest-confidence detections are 60% land.** A bright rock face
scores as well as a hull.

This matters for how the system should be tuned: **no threshold separates land
from water over this terrain**, so raising the operating point is not a
substitute for masking, and the threshold should not be tuned against
false-alarm counts that land masking already handles.

### 4.3 Fixed offshore infrastructure

The Grand Banks carries four production installations in the Jeanne d'Arc Basin.
Each is a large, bright, radar-hard target that appears in **every** acquisition
over its field and does not move. With no AIS transmitter, each satisfies the
dark-vessel definition on every pass.

| installation | type | position | source |
|---|---|---|---|
| Hibernia | gravity base structure | 46.7504 N, 48.7829 W | public record |
| Hebron | gravity base structure | 46.5439 N, 48.4981 W | public record |
| Terra Nova | FPSO | 46.4750 N, 48.4794 W | public record |
| White Rose (SeaRose) | FPSO | 46.7886 N, 48.0150 W | public record |

`agents/infrastructure.py` attributes detections within **1 km** of a published
position to the installation. That radius is a stated trade: it covers the
structure and its 500 m safety zone plus georeferencing slop, while leaving the
**supply and standby vessels working the field visible** — those are real
traffic and exactly what an MDA system exists to show.

This is a false positive that **repeats on a fixed schedule**, which erodes
confidence in an alert list faster than a sporadic one.

*Not yet exercised on a real acquisition — no Grand Banks scene has been
processed. The mechanism and its geometry are unit-tested; the operational
result is pending.*

---

## 5. Correlation

AIS records → points, filtered to ±5 min of acquisition. Both layers projected
to a metric CRS derived from the detections. AIS points buffered by **2 km**
(~1 nmi); detections with no intersecting buffer are dark-vessel candidates.

**Only `water`-classified detections are eligible.** A rock outcrop or a
production platform has no AIS transmitter and would otherwise satisfy the
dark-vessel definition perfectly.

Missions with no AIS coverage are reported with an explicit **NO COVERAGE**
badge, and every detection is marked dark — reflecting *missing telemetry*, not
confirmed silence. This distinction is easy to lose once it is a number on a
map, so it is surfaced rather than folded into the dark count.

---

## 6. End-to-end verification

`S1D_IW_GRDH_1SDV_20260817T212209`, eastern Newfoundland, 22.9 h old at time of
processing. Full pipeline, land mask enabled:

```
ingest      -> product located, gcp_tps georeferencing,
               footprint lon -55.0038..-51.1668, lat 47.8795..49.7709
inference   -> 332 raw detections, 298 after cross-tile dedup   [20m 24s]
land mask   -> 17 water, 64 coastal, 217 land-rejected
correlation -> 281 of 298 excluded from dark-vessel candidacy
report      -> 17 alerts, NO COVERAGE badge
evaluate    -> not scored: no real AIS ground truth
```

Re-running produced **bit-identical detections** (332 → 298), confirming
determinism.

---

## 7. Limitations

Stated explicitly, because they bound how far any of this should be trusted.

1. **Recall over this operating area is unmeasured.** No AIS-scored NL
   acquisition exists yet. This is the single most important open gap.
2. **Precision is unmeasured.** Land is masked; sea clutter, wind streaks, and
   ice are not. The 17 surviving water detections are an **upper bound** on
   marine targets, not a vessel count.
3. **One scene, one season.** August 2026, eastern Newfoundland.
4. **The `coastal` band is a conservative judgement, not a measurement.** 64
   detections (21.5%) sit within 300 m of shore, where real nearshore vessels
   and land bleed both live. Excluding them from alerts is caution, not evidence.
5. **Vessel classification is not implemented.** Every detection reports
   `unknown`. The ensemble's SIZE, VESSEL and FISHING heads are sampled but not
   decoded, because decoding them requires labelled data this project does not
   hold.
6. **The infrastructure mask is unexercised operationally.** No Grand Banks
   scene has been processed.
7. **Sea state is uncontrolled.** Detectability varies with wind and waves,
   which were neither recorded nor normalised for.
8. **Icebergs are not discriminated from vessels**, and the dark-vessel rule
   converts every one into a false alert — see §7.1.

### 7.1 Iceberg ambiguity — structural, not incidental

**No claim is made about ship/iceberg discrimination, and none should be
inferred from any result in this report.**

The operating area is **Iceberg Alley**, the corridor along the Labrador Current
where icebergs calved from Greenland drift south past Newfoundland. Presence is
routine and seasonal, concentrated roughly spring through mid-summer. The
2026-08-17 acquisition sits at the tail of that season — contamination is less
likely than in May or June, but not excluded, and any spring acquisition faces
it directly.

**Why the detector cannot be expected to reject them.** An iceberg in SAR is a
bright, compact, high-backscatter target against a dark sea surface — the same
signature the detector is trained to find. The xView3 ensemble's negative class
is *fixed marine infrastructure*, not ice.

**Why this is structural.** This pipeline defines a dark vessel as *a detected
target with no correlating AIS transmission*. An iceberg carries no transmitter:

```
detected target  +  no AIS correlation  ->  "dark vessel"
iceberg          +  no AIS transmitter  ->  "dark vessel"   (by construction)
```

The failure is not that icebergs are hard to classify. It is that the
dark-vessel rule **converts each detected iceberg into a confident false alert
automatically**, using exactly the same evidence it would use for a genuine
AIS-silent vessel. No confidence threshold, NMS setting, or correlation radius
separates them, because the discriminating information is not present in the
inputs the rule consumes.

This compounds with the coverage gap: on an acquisition with no AIS at all,
every detection is dark by default, so an iceberg is indistinguishable from a
vessel *twice over*.

**What would address it** — none of it implemented:

- **Polarimetric discrimination.** Ice and metal hulls differ in dual-pol
  behaviour including VV/VH ratio and depolarisation. Both channels are already
  ingested and calibrated to σ⁰, so the input exists; only the discriminant is
  missing.
- **Temporal behaviour.** Icebergs drift with current and wind; vessels transit
  under power on independent headings. Multi-pass association over the
  Sentinel-1 repeat cycle separates them on motion alone.
- **Ancillary ice data.** The Canadian Ice Service publishes open iceberg and
  sea-ice charts usable as a spatial prior.
- **Labelled ice data.** No ship/iceberg-labelled SAR training set covering this
  AOI is held by the project.

**Interim honest reporting.** Until a discriminant exists, dark-vessel output
over this area should be described as **AIS-uncorrelated targets**, not as
vessels. In Iceberg Alley during drift season, an unknown and potentially large
share of those targets are ice.

### 7.2 The `VESSEL_MAP` head is not an ice or land discriminator

The ensemble exposes a `VESSEL_MAP` head whose training task separates vessels
from **fixed marine infrastructure** — the negative class the xView3 challenge
defined. A low score therefore means only *"not a vessel under xView3's training
distribution"*, a category that silently lumps together infrastructure, ice, and
terrain clutter.

It must not be presented as target classification, and specifically must not be
used as evidence that a detection is not an iceberg.

---

## 8. Recommended work, ordered by impact

1. **Score a real Newfoundland acquisition.** Everything else is bounded by
   this. Requires the recorder running continuously through a VV/VH pass near
   09:30 or 21:30 UTC. Converts every "unmeasured" above into a number.
2. **Build a ship/iceberg discriminant (§7.1).** Not a tuning task and not a
   day's work; it is the precondition for any operational claim in this region.
   Dual-pol input is already available and calibrated, so a polarimetric probe
   is the cheapest first step.
3. **Process a Grand Banks scene** to exercise the infrastructure mask against
   real installations rather than unit tests.
4. **Decode the SIZE head against AIS-reported vessel length** once ground truth
   exists. Requires joining `ShipStaticData` messages, which the recorder does
   not yet subscribe to.
5. **Quantify the `coastal` band** rather than excluding it on caution.

---

## 9. Reproduction

```bash
cp .env.example .env          # fill in COPERNICUS_USER / _PASS, AISSTREAM_API_KEY
python -m agents.landmask --fetch --source osm    # ~900 MB, once

# Start collecting ground truth. Nothing can be scored without this.
python agents/ais_recorder.py --aoi grand_banks

# Process an acquisition
export TARGET_DATE=2026-08-17 AOI_NAME=eastern_newfoundland
export TRITONEYE_DETECTOR=xview3
python agents/ingest/ingest_agent.py > m.json
python agents/inference/inference_agent.py --payload-file m.json > m2.json
python agents/correlation/correlation_agent.py --payload-file m2.json > m3.json
python agents/report/report_agent.py --payload-file m3.json > m4.json
python agents/evaluate/evaluate_agent.py --payload-file m4.json
```

The xView3 backend needs `models/xview3/traced_ensemble.jit` (1.3 GB,
gitignored) from the
[upstream release](https://github.com/BloodAxe/xView3-The-First-Place-Solution/releases).
