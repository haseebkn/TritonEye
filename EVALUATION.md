# TritonEye — Evaluation Report

Measured performance of the SAR vessel detection and AIS correlation pipeline
against ground truth derived from AIS, on real Sentinel-1 acquisitions.

**Date:** 18 August 2026 · **Branch:** `main` · **Commit:** `d53bf18` (working tree)

---

## Summary

| Question | Answer |
|---|---|
| Is geolocation correct? | **Yes, after fixing it.** Was 34–277 km out; now sub-pixel at control points, verified by two independent AIS matches at 41 m and 109 m. |
| Does the detector work? | **No.** 4% recall on vessels ≥50 m in the deployed configuration; 11% at best-case threshold. |
| Is that the imagery's fault? | **No.** 48% of those vessels sit 3–34× above local sea clutter and are missed anyway. |
| Is that the preprocessing's fault? | **No.** Tested and falsified — a corrected normalisation changes nothing. |
| Is it the confidence threshold? | **Partly.** 0.35 → 0.10 raises recall 4% → 11%, then saturates. |
| So what is it? | **The weights.** A 3.0 M-parameter `yolov8n` with a resolution domain gap it cannot close. |
| Does a better model fix it? | **Yes.** The xView3 challenge winner reaches **59%** recall on ≥50 m vessels versus 4% — a 14.75× improvement, full-scene verified (§5a). |

The pipeline's engineering is sound and its geolocation is now trustworthy. Its
detection model is not fit for purpose, and this report is the evidence for that
claim rather than an impression of it.

---

## 1. Method

### 1.1 Ground truth

Every AIS-broadcasting vessel inside the imaged swath at acquisition time is a
vessel that was **definitely present**. The fraction of those the detector finds
is therefore a **lower bound on recall** — dark vessels are by definition absent
from AIS, so true recall could be higher, but it cannot be lower.

Construction, per scene:

1. Filter the MarineCadastre daily archive to ±5 min of acquisition and to the
   scene's ground footprint.
2. Keep one position per MMSI — the record closest in time to acquisition.
3. Clip to the **actual swath footprint** (convex hull of the product's 210 GCPs),
   not the padded rectangle used for the AIS query.
4. Join vessel `length` from the raw archive's static fields.
5. A vessel counts as detected if a detection centroid falls within **500 m**
   (end-to-end tests) or **400 m** (per-tile tests) of its reported position.

### 1.2 Scenes

| Scene | Product | Size | AIS in swath |
|---|---|---|---|
| 2025-01-01 Boston | `S1A_IW_GRDH_1SDV_20250101T224342` | 25931 × 16689 | 25 vessels |
| 2025-01-08 Boston | `S1A_IW_GRDH_1SDV_20250108T223513` | 25846 × 16687 | 141 vessels |

Both are Level-1 IW GRD, dual-pol VV/VH, uint16, ~10 m ground spacing.

### 1.3 System under test

- **Model:** `MeWan2808/yolov8n-sar-vessel-detection` — **3.0 M parameters**, one
  class (`ship`)
- **Deployed config:** `conf_threshold: 0.35`, `iou_threshold: 0.45`,
  640 × 640 tiles, 128 px overlap
- **Input rendering:** 3-channel `(VV, VH, VV)` with fixed divisors,
  `clip(DN/10000 × 255)` and `clip(DN/5000 × 255)`

---

## 2. Georeferencing

### 2.1 The defect

Sentinel-1 L1 GRD measurement rasters are stored in **radar geometry**: no CRS,
no affine geotransform, only a grid of ground control points. Every real product
in `data/raw/` reports `crs = None` with 210 GCPs in EPSG:4326. Only the
synthetic mock products carry a CRS — which is why every test passed while the
real path was broken.

The original code assumed a projection (`EPSG:32622`), detected the raster was
not georeferenced, fell back to treating pixel indices as metres, then linearly
stretched the entire swath into the AOI search box. Measured error on the
2025-01-01 scene:

| Point in image | True (GCPs) | Reported | Error |
|---|---|---|---|
| Top-left | −73.897, 43.866 | −71.000, 42.500 | **277.3 km** |
| Centre | −72.154, 42.916 | −70.500, 42.250 | **153.7 km** |
| Bottom-right | −70.411, 41.966 | −70.000, 42.000 | **34.2 km** |

### 2.2 Interpolation method selection

Residuals against all 210 GCPs of the 2025-01-01 scene:

| Method | Mean residual | Max residual | Held-out median |
|---|---|---|---|
| Affine (`rasterio.transform.from_gcps`) | 696.58 m | 2021.30 m | — |
| GDAL polynomial | 118.61 m | 555.85 m | 80.29 m |
| **Thin plate spline** | **0.00 m** | **0.00 m** | **49.28 m** |

TPS interpolates exactly through every control point. The held-out column fits
on three quarters of the GCPs and tests on the withheld quarter — a strictly
harder problem than production, where all 210 are used.

The affine fit is off by a mean of ~700 m, which is inside the 2 km correlation
buffer but large enough to mis-associate vessels in traffic. TPS was adopted.

### 2.3 Validation

Detection counts are unchanged before and after — the original code from `HEAD`
and the corrected code both return 8 detections on the 2025-01-08 raster with
identical weights, confirming the change is coordinate-only. Only positions moved
(mean **38.9 km**, max 55.5 km).

| Scene | Detections | Inside true swath | Dark before → after | Closest AIS match |
|---|---|---|---|---|
| 2025-01-01 | 6 | 6/6 | 6 (100%) → 4 | **41 m, 109 m** |
| 2025-01-08 | 8 | 8/8 | — → 6 | 137 m, 1809 m |

Two detections agreeing with independent AIS contacts to **41 m and 109 m** is
the strongest available evidence the transform is correct; agreement at that
distance does not occur by chance. Before the fix, the correlation engine had
never matched a real vessel on any acquisition.

---

## 3. Detector performance

### 3.1 Recall by vessel length

Raw recall against all AIS contacts understates the model, because many AIS
transmitters are small craft below what 10 m imagery can resolve. Stratified by
length, at the deployed `conf_threshold: 0.35`:

**2025-01-01** — 6 detections produced

| Vessel length | In swath | Detected | Recall |
|---|---|---|---|
| 0–20 m | 3 | 0 | 0% |
| 20–50 m | 14 | 0 | 0% |
| 50–100 m | 2 | 0 | 0% |
| 100 m+ | 6 | 2 | 33% |
| **≥50 m** | **8** | **2** | **25%** |

**2025-01-08** — 8 detections produced

| Vessel length | In swath | Detected | Recall |
|---|---|---|---|
| 0–20 m | 28 | 0 | 0% |
| 20–50 m | 83 | 0 | 0% |
| 50–100 m | 14 | 0 | 0% |
| 100 m+ | 13 | 1 | 8% |
| length unknown | 3 | 0 | 0% |
| **≥50 m** | **27** | **1** | **4%** |

**The model detects nothing below 100 m.** Zero across 111 vessels in the
20–100 m band over two scenes. A 50 m ship spans five pixels at 10 m ground
spacing and is well within what the sensor resolves.

### 3.2 Is the signal present?

For each ≥50 m vessel in the 2025-01-08 swath, peak backscatter in a ±9 px box at
the AIS position versus the local ocean background (121 px window):

```
        MMSI   len   px    peak  bg med  contrast  visible?
   367174060    71    7    7289     214     34.1x       YES   <- missed
   367006540   145   14    2351     138     17.0x       YES   <- missed
   224941000   284   28    2998     192     15.6x       YES      found
   338767000    72    7    1735     187      9.3x       YES   <- missed
   356068000   140   14    1977     234      8.4x       YES   <- missed
   368354510    76    8    1684     195      8.6x       YES   <- missed
   367691010    70    7    1354     194      7.0x       YES   <- missed
   ...
  clearly visible (>=3x local clutter) : 13/27 = 48%
  marginal (1.8-3x)                    :  3/27
  model actually detected              :  3/27 = 11%
```

(Eight entries return zero because they fall in the product's zero-fill border.)

**Ships at 34× above sea clutter are being missed.** The information is
unambiguously present in the pixels; the detector is not using it.

---

## 4. Diagnostic sequence

Three hypotheses were tested. Two were falsified.

### 4.1 Hypothesis A — the input rendering destroys the signal · **FALSIFIED**

The pipeline maps DN to 8-bit with fixed divisors. Measured DN distribution over
a 3000 × 3000 window of a real scene:

| | VV | VH |
|---|---|---|
| Real DN (median / p99 / max) | 162 / 281 / 8523 | 37 / 111 / 984 |
| After `DN/10000 × 255` | **4/255** | **1/255** |
| Distinct grey levels used | **66 of 256** | **37 of 256** |
| A p1–p99 stretch instead | 107/255, 207 levels | 52/255, 94 levels |

The model is genuinely fed near-black images using a quarter of the available
dynamic range. **This is a real defect** — but A/B testing both renderings
through identical weights on 27 tiles centred on known vessels gave **3/27 either
way**. One vessel recovered, one lost.

Conclusion: real bug, not the cause. Worth fixing; will not move recall alone.

### 4.2 Hypothesis B — the confidence threshold is miscalibrated · **PARTLY TRUE**

Swept against ground truth, position-matched within 400 px:

| conf | Ships found | Recall | False boxes/tile |
|---|---|---|---|
| 0.02 | 3 | 11% | 0.7 |
| 0.05 | 3 | 11% | 0.3 |
| **0.10** | **3** | **11%** | **0.1** |
| 0.15 | 1 | 4% | 0.1 |
| **0.35 (deployed)** | **1** | **4%** | **0.1** |
| 0.50 | 1 | 4% | 0.1 |

Dropping 0.35 → 0.10 nearly triples recall at no false-positive cost. The 0.35
default was inherited from natural-image detection and never calibrated for this
model on this sensor. **Recommended change** — but the curve saturates at 11%.

### 4.3 Hypothesis C — the weights are inadequate · **SUPPORTED**

With A and B eliminated, and §3.2 establishing that 48% of the targets are
plainly visible, the remaining explanation is the model itself.

The likely mechanism is a **resolution domain gap**. Public SAR ship datasets
(SSDD, HRSID) are 0.5–3 m imagery from Gaofen-3/TerraSAR-X, where a ship spans
50–200 px. Sentinel-1 IW GRD is 10 m, so a 100 m ship is **10 px** and a 70 m
ship is **7 px**. YOLOv8's finest detection stride is 8 — meaning a 70 m vessel
occupies **less than one output cell**.

This is corroborated independently by the xView3-SAR challenge winner, who
explicitly abandoned grid-based detection for **stride-2 dense point
prediction**, citing that at 10 m/px *"ships occupy only a few pixels"* and that
a finer output map was required *"to avoid predictions being suppressed during
NMS."* The same solution also uses 2-channel VV/VH input and a tuned sigmoid
normalisation rather than fixed divisors — matching both defects identified here.

---

## 5. Candidate model comparison

A larger maritime model, `dronefreak/seadronessee-yolov8m` (25.9 M params, 8× the
current model), was evaluated before deployment. Identical 27 tiles, identical
ground truth, position-matched:

| Model | Render | conf | Found | Recall | Boxes |
|---|---|---|---|---|---|
| SAR yolov8n (current, 3.0 M) | fixed div | 0.35 | 1 | 4% | 3 |
| **SAR yolov8n (current)** | fixed div | 0.10 | **3** | **11%** | 7 |
| SeaDronesSee yolov8m (25.9 M) | fixed div | 0.35 | 0 | **0%** | 0 |
| SeaDronesSee yolov8m | fixed div | 0.02 | 0 | **0%** | 0 |
| SeaDronesSee yolov8m | stretched | 0.10 | 1 | 4% | 6 |
| SeaDronesSee, all classes | stretched | 0.02 | 3 | 11% | **100** |

**Zero detections** on the imagery the pipeline actually produces, even at a 0.02
threshold. SeaDronesSee is optical RGB captured from drones at centimetre GSD,
detecting swimmers, jetskis and buoys — its published 62% mAP@50 is on that
domain and transfers nothing to spaceborne C-band SAR. Reaching 11% requires
accepting every class (calling ships "swimmers" and "buoys") at a ~33:1
false-positive rate.

**A model with 8× the capacity and a strong benchmark score performed worse than
the incumbent.** Benchmark figures transfer only within the domain they were
measured in — which is the argument for this evaluation harness existing.

---

## 5a. Replacement detector: xView3 challenge winner

The xView3-SAR challenge-winning ensemble (Khvedchenya, MIT licence; CircleNet —
EfficientNet B4/B5/V2S encoders with U-Net decoders, stride-2 dense prediction)
was evaluated as a drop-in replacement. Weights are public and require no
dataset registration.

### 5a.1 Three input details, each load-bearing

Every one was invisible until measured, and each in isolation makes the model
look broken rather than mis-fed:

1. **Calibrated σ⁰ in dB.** Their normalisation is `sigmoid((x + 20) × 0.18)`,
   which saturates above ~+10 dB. Raw GRD digital numbers (VV median 165) drive
   *every* pixel to 1.0 — a uniformly white image. This forced the radiometric
   calibration in §5a.2.
2. **Exactly 2048×2048 input.** Concrete sizes were baked in when the ensemble
   was traced, so its twelve sub-models only agree at that size. Any other size
   fails with a shape mismatch deep inside the ensembling stack.
3. **Channel order (VH, VV)** — reversed from the rest of this pipeline.
   Measured on one known vessel:

   | Variant | Objectness | Distance from target |
   |---|---|---|
   | **sigmoid-norm (VH, VV)** | **0.329** | **56 px** |
   | sigmoid-norm (VV, VH) | 0.112 | 918 px |
   | raw dB (VV, VH) | 0.202 | 593 px |
   | raw dB (VH, VV) | 0.184 | 477 px |

### 5a.2 Radiometric calibration (`agents/calibration.py`)

Required by the above, and a gap in its own right — the pipeline previously
consumed uncalibrated DN. σ⁰ = DN²/A², where A is the `sigmaNought` LUT from the
product's own `annotation/calibration/*.xml` (a 27×648 grid, bilinearly
interpolated to full resolution, sampled at the correct window offset because
the LUT varies strongly across range).

| | before | after |
|---|---|---|
| VV | median 165 DN | **median −11.2 dB** (p1 −17.2, p99 −6.5) |
| After xView3 normalisation | 1.000 everywhere | **0.62 – 0.92** |

−11 dB is a textbook ocean VV value at moderate incidence.

### 5a.3 Full-scene result

Full 2025-01-08 Boston acquisition: 126 tiles, 28.6 min on an RTX 4070 Laptop,
1760 raw detections deduplicated to 1419. Same ground-truth construction as §1.1,
same 500 m match radius.

| Vessel length | In swath | xView3 found | xView3 recall | yolov8n recall |
|---|---|---|---|---|
| 0–20 m | 28 | 6 | 21% | 0% |
| 20–50 m | 83 | 49 | **59%** | **0%** |
| 50–100 m | 14 | 10 | **71%** | **0%** |
| 100 m+ | 13 | 6 | 46% | 8% |
| **≥50 m (headline)** | **27** | **16** | **59%** | **4%** |

**A 14.75× improvement on the headline metric**, and the 20–100 m band — where
the deployed model had never detected anything across 111 vessels — is recovered
at 59–71%.

The 100 m+ row (46%) being *lower* than 50–100 m (71%) is unexplained and rests
on 13 vessels; plausibly large vessels berthed in harbour clutter, but it is not
established.

### 5a.4 Operating point

Threshold swept on the saved detections, no re-inference needed:

| Threshold | Detections | Recall (all) | Recall ≥50 m | Detections per AIS vessel |
|---|---|---|---|---|
| 0.05 | 1419 | 51% | 59% | 10.1× |
| 0.10 | 313 | 33% | 52% | 2.2× |
| **0.15** | **160** | 21% | **48%** | **1.1×** |
| 0.25 | 97 | 13% | 44% | 0.7× |
| 0.40 | 40 | 8% | 30% | 0.3× |

Recall on resolvable vessels decays far more slowly than the detection count:
0.05 → 0.15 cuts detections **8.9×** for an 11-point drop in ≥50 m recall. The
default was therefore set to **0.15**. This is the first operating-point analysis
the project has had; the previous detector's 0.35 was an inherited default that
had never been swept.

### 5a.5 Precision remains unmeasured, and the count is not a vessel count

At threshold 0.05 the model returns **1419 detections against 141 AIS vessels** —
1347 unmatched. Some are genuine dark vessels; on a scene covering this much
Massachusetts coastline, most are likely land returns. AIS-derived ground truth
measures recall and **cannot measure precision at all** (§8.2).

Two mitigations exist and neither is yet usable: the ensemble's `VESSEL_MAP` head
(reads a near-uniform ~0.7 across detections, so the current decoding is not
trusted) and a land mask (§9.2). **Until one lands, a raw detection count from
this model must not be reported as a vessel count.**

### 5a.6 End-to-end confirmation through the production pipeline

The figures in §5a.3–4 came from a standalone harness. Running the same scene
through the shipped pipeline (`TRITONEYE_DETECTOR=xview3`, all five agents) at
the configured default threshold of 0.15 reproduces them exactly:

| | Standalone sweep @ 0.15 | Production pipeline |
|---|---|---|
| Detections | 160 | **160** |
| Recall (all) | 0.21 | **0.213** |
| Recall (≥50 m) | 0.48 | **0.481** |

Detections are reported as class `unknown` — the ensemble locates vessels
without typing them, unlike the incumbent path which clamps every detection to
`cargo`.

The mission reported **117 dark vessels of 160 detections**. That number should
not be read as an intelligence product: with precision unmeasured and no land
mask (§5a.5), an unknown share of those 117 are coastal returns rather than
vessels evading AIS.

### 5a.7 Cost

| | per scene |
|---|---|
| yolov8n (deployed) | **47 s** |
| xView3 ensemble, idle GPU | **28.6 min** (126 tiles, 6.94 GB peak VRAM) |
| xView3 ensemble, GPU contended | **107.7 min** |

~36× slower on an idle card. The contended figure was measured with a second
process holding the GPU, and is recorded because it is the more realistic number
for a workstation running anything else — VRAM headroom is only ~1.7 GB, so the
ensemble does not share the device gracefully. That still fits inside the 2.3 h median CDSE publication latency and
the 1–4.5 day revisit, so near-real-time survives — but it makes this a scheduled
batch job, which is why the xView3 path should be opt-in rather than default.

### 5a.8 Land masking — the dominant false-positive mode, measured

The 2026-08-17 eastern Newfoundland scene (§6) returned **298 detections at
threshold 0.15, every one of them reported dark.** The spatial distribution was
diagnostic before any mask existed: detections clustered densely over the
Newfoundland landmass while the open sea east of the island was nearly empty —
the inverse of what real traffic looks like.

`agents/landmask.py` classifies detection centroids against OSM land polygons
(ODbL, derived from `natural=coastline`). Applied to the same 298 detections:

| surface | count | share |
|---|---|---|
| water (alert-eligible) | **17** | 5.7% |
| coastal (within 300 m of shore) | 64 | 21.5% |
| **land-rejected** | **217** | **72.8%** |

**Land detections reached 35.4 km inland, with a median of 7.2 km.** These are
not near-shore ambiguity or georeferencing slop; they are terrain — rock, built
structures, radar-facing slopes — being detected as vessels tens of kilometres
from any water a ship could reach.

**Nearly three quarters of the raw output was land.** The headline count of 298
overstated the marine target count by a factor of roughly 17.

Three consequences worth stating plainly:

1. **The 298 figure should never have been reported as a vessel count**, and
   §5a.5 already said so in general terms. This is the specific magnitude.
2. **The `coastal` band is large (64, 21.5%) and genuinely ambiguous.** Real
   nearshore vessels and land bleed both live there, so it is reported as a
   third class rather than forced into the binary. It is excluded from alerts,
   which is the conservative choice, not a measured one.
3. **This does not convert recall into precision.** 17 water detections is an
   upper bound on true marine targets in this scene, not a count of vessels —
   sea clutter, wind streaks, and (per §8.1) icebergs remain unmeasured within
   it. §5a.5 stands unchanged.

**Method.** Classification runs on detection centroids in geographic space, not
on the raster: point-in-polygon over 298 points took **581 ms**, against the
hundreds of MB a coastline rasterised to the 25000×16000 scene grid would have
cost. Distances are metric via an azimuthal equidistant projection centred on
the footprint — a fixed UTM zone was rejected because this scene spans
−55.00…−51.17, straddling the 21N/22N boundary at −54°W.

Detections are **annotated, never deleted**: `surface` and
`distance_to_shore_m` are written to every feature, and only `water` reaches the
dark-vessel correlator. The rejection tally is recorded in the mission payload
and rendered in the report header, because the count of what was thrown away is
itself the evidence that the false-alarm mode was found.

**Caveat on the operating point.** Threshold 0.15 was chosen partly to suppress
coastal false positives (§5a.4). With land now removed at the output stage, that
trade should be re-swept — a lower threshold may now be affordable. That
measurement has not yet been made, and the threshold is unchanged pending it.


---

## 6. Independent verification

The pipeline was run on the most recent acquisition available at time of writing
— `S1D_IW_GRDH_1SDV_20260817T212209`, Sentinel-1D, 22.9 h old, northeast
Newfoundland. Georeferencing succeeded (`gcp_tps`). **Detections: 0.**

A CFAR-style bright-target scan over a decimated read of the same raster, using
no learned model at all — any pixel ≥6× its local background and ≥400 DN:

```
  candidate hard targets: 47

           lon       lat   peak DN    x bg
      -54.5767   48.9425      6983   32.5x
      -53.9584   48.0353      3679   14.8x
      -54.1563   48.2677      3420   18.8x
      -53.9811   48.1716      3350   13.3x
      ...
  model reported: 0 detections
```

Without a land mask an unknown share of these are shoreline returns, so 47 is not
a vessel count. But a twenty-line threshold filter surfaces targets where the
deployed detector surfaces none.

---

## 7. Correlation engine

Evaluated separately against the synthetic generator's known ground truth
(50 vessels planted, 40 with AIS, **10 dark by construction**):

| | |
|---|---|
| Detections | 50/50 |
| Dark vessels reported | **8** of 10 |
| **False-negative rate** | **20%** |

Cause: the 2 km correlation radius means a dark vessel within 2 km of any
cooperative vessel is absorbed by that vessel's buffer and silently reclassified
as correlated. The radius is a fixed constant; it should be derived from vessel
speed and time offset (`SOG × Δt` plus a fixed error term) and swept against this
ground truth.

---

## 8. Limitations of this evaluation

Stated explicitly, because they bound how far these numbers should be trusted:

1. **Recall is a lower bound, not a point estimate.** Dark vessels are absent
   from AIS by definition, so the denominator omits exactly the targets the
   system exists to find.
2. **Precision is not measured.** A land mask now exists (§5a.8) and removes the
   dominant confound — 72.8% of the 2026-08-17 detections were land, reaching
   35.4 km inland, and on 2025-01-01 four of six detections fell 25–60 km
   inland. Masking land is not the same as measuring precision: sea clutter,
   wind streaks, and icebergs (§8.1) remain unquantified among the surviving
   water detections. An exhaustive labelled scene is still needed for a
   precision figure.
3. **Two scenes, one region, one season.** Both are January, Massachusetts Bay.
   No Newfoundland scene could be evaluated because MarineCadastre is US Coast
   Guard data and has zero coverage east of −67.4°W.
4. **AIS position has slop.** Reported positions can be several hundred metres
   off at acquisition instant; match radii of 400–500 m absorb this but may admit
   coincidental matches in dense traffic.
5. **Vessel length is self-reported** AIS static data and is sometimes absent
   (3 of 141 on 2025-01-08) or wrong.
6. **Sea state is uncontrolled.** Detectability varies with wind and wave
   conditions, which were not recorded or normalised for.
7. **Icebergs are not discriminated from vessels, and in this AOI the
   dark-vessel logic converts every one into a false alert.** Expanded in §8.1,
   because it is structural rather than incidental.

### 8.1 Iceberg ambiguity in the Newfoundland AOI

**No claim is made about ship/iceberg discrimination, and none should be
inferred from any result in this report.**

The operating area is **Iceberg Alley** — the corridor along the Labrador
Current where icebergs calved from Greenland drift south past Newfoundland.
Iceberg presence is routine and seasonal, concentrated roughly spring through
mid-summer and tapering through late summer. The 2026-08-17 acquisition sits at
the tail of that season: contamination is less likely than it would be for a
May or June scene, but it is not excluded, and any spring acquisition would face
it directly.

**Why the detector cannot be expected to reject them.** An iceberg in SAR is a
bright, compact, high-backscatter target against a dark sea surface. That is the
same signature the detector is trained to find. The xView3 ensemble's negative
class is *fixed marine infrastructure* — rigs, platforms, turbines — not ice.
Nothing in its training distribution teaches it that ice is not a vessel, so its
`VESSEL_MAP` head should not be read as an ice filter (see §8.2).

**Why this is structural, not a tuning problem.** This pipeline defines a dark
vessel as *a detected target with no correlating AIS transmission*. An iceberg
carries no AIS transmitter. An iceberg therefore satisfies that definition
exactly and completely:

```
detected target  +  no AIS correlation  ->  "dark vessel"
iceberg          +  no AIS transmitter  ->  "dark vessel"     (by construction)
```

The failure is not that icebergs are hard to classify. It is that the
dark-vessel rule **converts each detected iceberg into a confident false alert
automatically**, and does so with exactly the same evidence it would use for a
genuine AIS-silent vessel. No confidence threshold, NMS setting, or correlation
radius can separate the two, because the discriminating information is not
present in the inputs the rule consumes.

This compounds with the AIS coverage gap. On an acquisition with no AIS
coverage at all — as on 2026-08-17 — every detection is reported dark by
default, so an iceberg is indistinguishable from a vessel *twice over*: once for
lacking a transmitter, and once for there being no telemetry to check against.

**What would address it.** None of the following is implemented; they are listed
to show the problem is tractable, not to imply progress against it:

- **Polarimetric discrimination.** Ice and metal hulls differ in dual-pol
  behaviour, including VV/VH ratio and depolarisation. Both channels are already
  ingested and radiometrically calibrated to sigma-nought (§5a.2), so the
  required input exists in the pipeline today — only the discriminant is missing.
- **Temporal behaviour.** Icebergs drift largely with current and wind; vessels
  transit under power on independent headings. Multi-pass association over the
  Sentinel-1 repeat cycle separates the two on motion alone.
  `agents/tracking.py` exists but is not yet applied to this problem.
- **Ancillary ice data.** The Canadian Ice Service publishes open iceberg and
  sea-ice charts that could be joined as a spatial prior for the acquisition
  date.
- **Labelled ice data.** No ship/iceberg-labelled SAR training set covering this
  AOI is currently held by the project. Acquiring or building one is the
  precondition for any supervised approach.

**Interim honest reporting.** Until a discriminant exists, dark-vessel output
over this AOI should be described as *AIS-uncorrelated targets*, not as vessels.
The distinction is not pedantic: in Iceberg Alley during the drift season, an
unknown and potentially large share of those targets are ice.

### 8.2 The `VESSEL_MAP` head is not an ice or land discriminator

Related to §8.1 and stated separately because it is an easy misreading of the
model's outputs.

The xView3 ensemble exposes a `VESSEL_MAP` head. Its training task separates
vessels from **fixed marine infrastructure**, which is the negative class the
xView3 challenge defined. A low `is_vessel` score therefore means only *"not a
vessel under xView3's training distribution"* — a category that silently lumps
together infrastructure, ice, and terrain clutter.

It must not be presented as target classification, and specifically must not be
used as evidence that a detection is not an iceberg. See
`agents/xview3_detector.py` for the corresponding note at the point of use.

---

## 9. Conclusions and recommended work

**Ordered by impact.**

1. **Replace the detector — done and measured (§5a).** The xView3 1st-place
   ensemble reaches **59% recall on vessels ≥50 m against 4%** for the incumbent,
   full-scene verified, and recovers the 20–100 m band entirely. Remaining work
   to productionise it: decode the `SIZE` head into metres and the `VESSEL_MAP`
   head into a usable probability (both currently return uninterpreted values),
   and wire it into `inference_agent` as an opt-in path given the 28.6 min/scene
   cost.
2. **Add a land mask — done and measured (§5a.8).** 72.8% of the 2026-08-17
   detections were land, reaching 35.4 km inland; the raw count of 298
   overstated marine targets by roughly 17×. Remaining work: re-sweep the
   operating point now that land no longer has to be suppressed by threshold.
3. **Lower `conf_threshold` to 0.10.** Measured 4% → 11% at no false-positive cost.
4. **Report honest classes.** The model has one class (`ship`); the config
   declares five and the code clamps every detection to `cargo`.
5. **Fix the input rendering** to a percentile stretch or dB scale. Latent defect;
   matters once the detector can see.
6. **Calibrate the correlation radius** against the synthetic ground truth.
7. **Build a ship/iceberg discriminant before any operational claim in this
   AOI (§8.1).** Not a tuning task and not under a day: the dark-vessel rule
   converts every detected iceberg into a confident false alert by construction,
   and no threshold can fix it. Dual-pol input is already available and
   calibrated, so a polarimetric discriminant is the cheapest first probe;
   multi-pass drift association via `agents/tracking.py` is the more robust one.
   Until then, report output over Newfoundland as *AIS-uncorrelated targets*
   rather than as vessels.

Items 2–6 are each under a day. Item 1 is the project. Item 7 is the
precondition for deploying it where this project is aimed.

---

## 10. Reproduction

The recall measurement in §3.1 is not a one-off script — it runs as a pipeline
stage (`agents/evaluate/evaluate_agent.py`) and logs to MLflow on every mission,
so it re-computes automatically whenever the model, thresholds or imagery
change. Missions with no real AIS coverage are recorded as **unscored** rather
than 0%, so a telemetry gap can never be mistaken for a detector regression.

```bash
pytest tests/                                   # 43 tests, incl. georeferencing regression
export TARGET_DATE=2025-01-08 AOI_NAME=boston_offshore
python agents/ingest/ingest_agent.py           > p1.json
python agents/inference/inference_agent.py   --payload-file p1.json > p2.json
python agents/correlation/correlation_agent.py --payload-file p2.json > p3.json
python agents/report/report_agent.py         --payload-file p3.json > p4.json
python agents/evaluate/evaluate_agent.py     --payload-file p4.json > p5.json
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Reproducing the 2025-01-08 figures above gives, on the run:

```
eval.ais_vessels_in_swath   141.0     eval.recall_0_20m        0.0
eval.resolvable_in_swath     27.0     eval.recall_20_50m       0.0
eval.resolvable_matched       1.0     eval.recall_50_100m      0.0
eval.recall_resolvable      0.037     eval.recall_100m_plus    0.0769
eval.recall_all            0.0071     dark_ratio               0.75
```

Each run also records the detector that produced it in the MLflow model registry
(`tritoneye-sar-detector`), tagged with repo, class map and confidence
threshold — so a change in these numbers can be attributed to a change in model
rather than guessed at.

The georeferencing regression test (`tests/unit/test_georef.py`) asserts every
GCP of every real product in `data/raw/` round-trips to under 1 m, and skips
cleanly where the (gitignored) archive is absent. It is the test whose absence
allowed the 277 km error to survive two prior code reviews.
