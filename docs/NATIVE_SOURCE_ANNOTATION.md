# Native source supported NL annotations

On 14 October 2026, all **nine reserved areas across five HH/HV acquisitions**
were prepared and provisionally annotated before scoring. The result is **two
provisional vessel labels, seven unresolved compact returns and two diffuse
background contexts**. No plausible iceberg label was defensible under the frozen
source allowances. Ten unresampled 75×75 two-band contexts were extracted; one
shore-adjacent return was too close to an image edge and was not padded.

The [versioned collection](../datasets/source_supported/v0.1.0/collection.json)
is separate from the benchmark. All nine areas and every annotation remain
ignored for accuracy, with independent review, training approval, compatibility
and promotion eligibility false. No detector or classifier ran on this batch.
These are source-supported AI interpretations, not reviewed SAR–AIS matches or
measured model improvement.

## Exact acquisition preparation

The [sealed source reservation](../reservations/source_supported/v0.1.0/manifest.json)
and all earlier reservations retain their original bytes. No UUID, centre, split
or geographic assignment was substituted. A metadata-only preparation check found
that a rotated 512-pixel native square extends outside the six-kilometre projected
envelope. Before any pixels were inspected, commit `123d2fb` froze a centred
**384-pixel** square for every area in the
[amended preparation protocol](../configs/reservations/source_windows_v011.json).
The earlier protocol and failed attempt remain available; the envelopes were not
enlarged and windows were not moved to favourable returns.

All nine final footprints lie inside their original envelope, catalogue swath
and NL study boundary and pass the existing development-scene guard. The first
product remains its exact reserved COG UUID. An attempted full download produced
no image bytes and was stopped. Bounded range reads then obtained its native base
pixels through GDAL's compressed TIFF decoder. The other products use the existing
native strip reader. No overviews, resampling, padding or speckle filtering were
used. The final preparation transferred 312,438,406 bytes, including metadata and
range headers; this is not the on-disk dataset size.

Original cropped uint16 DN, annotation XML, calibration LUTs, range receipts,
calibrated two-band GeoTIFFs and fixed −30 to +5 dB HH/HV previews are retained in
`data/source_supported/native_v011_02/`. All **18 band calibrations replay exactly**
from the saved DN and original LUT offsets. Native mode, spacing, thermal-noise
correction flag, scene start, local incidence interpolation and window-row times
are recorded. Pixel spacing is not physical resolution. No additional thermal
noise removal, terrain correction or moving-target SAR displacement correction
was introduced. Partial reads do not validate the publisher's full-file checksum.
All measurements are IW GRDH HH/HV with 10 m range spacing and 10–10.001 m
azimuth spacing; native centre incidence spans 32.149–40.929 degrees. Every native
annotation reports thermal-noise correction false. A fresh compressed-source read
of one complete reserved HH window also reproduced the saved DN exactly.

## Annotation sequence and source reconciliation

All nine full paired previews were inspected in four quadrants in each band.
Coordinates exclude preview headers and gutters. The
[image-first decisions](../configs/reservations/source_first_pass_v010.json)
were committed in `0e2c2ef` **before** calculating the native-time source distances
or opening source overlays. That pass recorded nine unresolved compact returns
and two texture contexts; continuous shore/land-like features remained ignored.
Scanning an entire image does not establish exhaustive object or class truth.

The annotator was Codex, with no independent SAR reviewer. Source-informed
selection, source-centred windows and stream IDs were already known. One source
record had been opened earlier for schema inspection, and the AIS datatakes have
prior project imagery/model exposure. This was score-free inspection for this
pass, not independent or fully naive review. AI self-review does not satisfy the
production review guard.

Annotation coordinates are integer `[x0, y0, x1, y1]` boxes with exclusive upper
bounds in the 384×384 native window. `unresolved` means a compact return with
unsettled class or source association; `clutter` here means a diffuse texture
context, never a verified negative. `provisional_vessel` requires compatible image
appearance and one consistent source interpretation in the frozen pool. No class
is independent truth. Ambiguities and omitted fine speckle stay inside whole-area
ignore masks. A 75×75 chip is centred on the frozen box centre; boundary or size
failures abstain instead of stretching, padding or shifting. Review history is
the image-first commit followed by the separate reconciliation record; no second
AI pass is represented as independent review.

Subsequent [source reconciliation](../configs/reservations/source_reconciliation_v010.json)
uses native object-row times and TPS locations. All frozen competing AIS histories
are realigned at each object time; the IIP allowance retains 300 m plus 1 m/s of
absolute observation separation. No allowance was enlarged after viewing images.
Private evidence preserves source identities, positions, competing tracks and
contradictions. Public records omit those identities, geographic positions,
track histories and native raster offsets.

| Area | Source interpretation after image-first annotation |
| --- | --- |
| iceberg_support_01 | Two nearshore compact returns about 965 m and 1769 m away, outside the approximately 353 m allowance; unresolved |
| iceberg_support_02 and 03 | Source position overlaps diffuse texture, but no isolated object was identified; contexts are not negatives or iceberg labels |
| iceberg_support_04 | Compact return about 598 m away versus approximately 486 m allowance; unresolved |
| iceberg_support_05 | Return about 941 m away versus approximately 621 m allowance; unresolved |
| iceberg_support_06 | Two competing returns about 943 m and 673 m away, both outside approximately 620 m allowance; unresolved |
| vessel_support_01 | One compatible selected AIS identity in the frozen pool, about 384 m away within approximately 457 m allowance; provisional vessel |
| vessel_support_02 | Nearest AIS track about 243 m away, outside approximately 200 m allowance; unresolved |
| vessel_support_03 | One compatible selected AIS identity in the frozen pool, about 221 m away within approximately 311 m allowance; provisional vessel |

The allowances are heuristics, not confidence intervals or identity probabilities.
Coarse IIP coordinate precision, drift, source errors and SAR moving-target
displacement remain possible explanations for failures, not established causes.
AIS reception is incomplete; absence of AIS never implies iceberg. Background
texture near a source coordinate cannot inherit its class. The two provisional
vessel labels stay ignored, including when the source interpretation appears unique.

## What this batch can and cannot support

The collection provides reproducible native inputs, explicit first-pass decisions,
source-association failures and two provisional vessel contexts for later research.
Its whole-area ignore masks contain one everywhere: one means excluded. Neither
clutter contexts nor dark areas are hard negatives. Ten extracted contexts include
unresolved/background samples and must not be described as ten usable class labels.

There is still no usable iceberg class in this batch, no independently reviewed
truth and no evidence of false-positive reduction. Historical IIP areas retain
training roles; 2026 AIS areas retain validation roles, leaving class, year and
split confounded. The 2015/2017 acquisitions may overlap the Kaggle competition's
undisclosed parent imagery. Kaggle physical spacing and preprocessing remain
unknown; equal chip dimensions do not establish input compatibility. This work
does not approve two-class training, unlock the test set or lift AllenAI quarantine.

Next, reserve new source-supported opportunities without using classifier scores,
including training-role vessels and validation-role ice observations. Resolve
coordinate precision, time and SAR displacement assumptions before another
source-association pass. Seek higher-resolution permitted supporting imagery or
better source observations where possible. Without a qualified reviewer, any
future labels and development comparisons must remain explicitly provisional.

## Replay and integrity

```sh
python -m agents.source_windows --output data/source_supported/fresh_native_replay
python -m agents.source_annotations --support data/source_supported/native_v011_02/manifest.json --first-pass configs/reservations/source_first_pass_v010.json --output data/source_supported/fresh_support.json
python -m agents.source_annotations --verify datasets/source_supported/v0.1.0 --seal-sha256 8c985d6cc1b91d1c5807046b7be9e55f139ce477d8980c11623a3504bbad67f9 --local
```

Acquisition requires existing `COPERNICUS_USER` and `COPERNICUS_PASS` in the ignored
`.env`; no new credentials were required. Local verification requires original
hash-matching private evidence, source DN, calibration and annotation artifacts.
Without `--local`, verification checks the sealed public metadata, decision lineage
and provisional guards without loading image pixels or private AIS histories.
The [collection lock](../configs/reservations/source_collection_v010.lock.json)
pins the release. The [source audit](SOURCE_SUPPORTED_ANNOTATION.md) retains source
citations, licences, original hashes and reservation methodology.
