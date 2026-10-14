# Two reserved native training windows: provisional annotation v0.1.0

Both source-ranked training reservations have been prepared and inspected before
detector or classifier scoring. The [versioned collection](../datasets/native_training/v0.1.0/collection.json)
contains **two areas, one source-compatible plausible iceberg hypothesis and no
admitted training or accuracy-truth labels**. No independent reviewer is available.
The [original reservation](PROVISIONAL_NATIVE_TRAINING.md), 31-area admission
register, prior datasets/masks, AllenAI quarantine and original locked test remain
unchanged. This collection is additive; its counts are not project-wide totals.

## What the two windows showed

| Area | Acquisition UTC | Source coverage | Image-first observation | Reconciled interpretation |
| --- | --- | --- | --- | --- |
| `training_iceberg_support_01` | 2019-04-29, native target time recorded in collection | All three frozen scenarios complete | One elongated bright reflector in both HH/HV, initially unresolved | One plausible iceberg hypothesis; unconfirmed identity and no additional physical corroboration |
| `training_iceberg_support_02` | 2020-06-27 | Original and nearest-rounding complete; widest scenario incomplete | No comparably distinct object confidently isolated | Empty object list with source-association abstention; not a verified negative |

The 2019 return is **894.82 m** from the selected IIP radar/visual observation at
a **686.66 s** native target-time separation. It falls within the unchanged
**986.66 m heuristic search allowance**, with one compatible reported identity
in the available, incomplete IIP pool. Its elongated response also admits a
vessel or other reflector explanation. Source consistency is not physical
identity confirmation. The 22x55-pixel annotation includes radar response and
background; it does not measure an iceberg or hull.

For the 2020 window, the one-step coordinate/time scenario needs a radius of
1,034.42 m around its coordinate cell, but the minimum native-window clearance
is 837.23 m. About **95.92%** of that heuristic search region is covered.
The complete-coverage gate fails despite the high area fraction. The crop was
neither shifted nor enlarged after inspection. No object was invented at the
source centre, and absent annotation cannot establish absence of a vessel or
iceberg. Both windows remain entirely ignored, including apparently empty pixels.

These scenarios retain the previous 300 m base plus 1 m/s worst native-row time
separation, literal coordinate rounding/truncation sensitivity, and minute-time
sensitivity. They are disclosed search allowances, not calibrated error bounds.

## Corroboration and remaining label limits

The [sealed first pass](../configs/reservations/training_first_pass_v010.json)
records both bands and all four quadrants, with original fixed-scale previews
and display-only percentile/nearest-neighbour enlargement. The AI annotator had
prior project label/model-output exposure and knew the source-centred selection.
No source overlay, target/source-distance reconciliation or new model output was
viewed before this pass. It is not independent or wholly naive annotation.
The [later reconciliation](../configs/reservations/training_reconciliation_v010.json)
retains the original box and zero-object area, recording how source evidence
changed the 2019 interpretation. A visual scan is not verified annotation
completeness; weak or unresolved objects may have been missed.

No repeated sighting of the selected 2019 iceberg identity occurs in the retained
IIP observations within **+/-24 hours** of the target time. Complete public
Sentinel-2 L1C catalogue queries over each full native window within **+/-3 hours**
returned no product covering the annotated target or the respective source
position. This bounded metadata search supplies no optical confirmation; it is
not an exhaustive search across sensors, times or drift locations. No optical
pixels were inspected because these queries provided no covering product.

The known local AIS archive dates are 2026, with historical directories dated
2025. This is a date-name inventory, not a historical receiver-coverage audit;
no co-temporal 2019/2020 local AIS evidence was available. Missing AIS cannot
distinguish ship from iceberg. Original observer logs, photographs and calibrated
IIP position errors were not available. No source organization was contacted.

The [provisional admission policy](PROVISIONAL_NATIVE_TRAINING.md) therefore
keeps the new hypothesis on hold: additional physical corroboration identifying
the same reflector is missing. The second window supplies neither an object
label nor a negative. Training-vessel coverage also remains missing. No fitting,
model selection, threshold tuning, detection metric or false-positive improvement
is reported by this work.

## Native input and integrity evidence

The two exact reserved Sentinel-1 A COG products were accessed through bounded
native compressed ranges: **54,172,512 bytes** for pixel preparation, plus the
separate metadata plan. The original complete products were not downloaded or
checksum-verified. Source range receipts, original annotation/calibration XML,
raw DN arrays, calibrated GeoTIFFs, previews, source-query responses and ignore
masks remain under `data/source_supported/` in the project folder. Their hashes
are bound by the public collection and private receipts.

Each crop is 384x384 original samples with HH followed by HV, 10 m native
sampling, nominal uncorrected sigma0 dB and no padding, resizing, filtering or
terrain correction. Sampling is not effective resolution. Metadata and crop
coordinates were frozen before reading pixels and checked against the reserved
envelope, exact catalogue product, NL scope and existing locked-test guard.

All **four saved calibration bands replay exactly** from their original DN and
LUTs. This is a reproducibility check using the existing calibrator, not a new
independent SNAP comparison or validation of absolute SAR geolocation/noise.
The crop-centre incidence angles are **37.01 and 45.36 degrees**. These angles
describe the prepared inputs; they do not establish an operating envelope.

One complete 2x75x75 float32 context is copied from the original calibrated
2019 pixels under the frozen [native input specification](NATIVE_NL_INPUT.md).
The tensor receipt and exact-pixel replay are retained. No context is extracted
from the unannotated 2020 area. Input extraction does not approve its class,
establish Kaggle equivalence, resolve thermal-noise sensitivity or permit fitting.
Display contrast and enlargement affect previews only, never stored model inputs.

Public verification validates the frozen policy/annotation references, counts and
provisional guards. Local replay additionally reconstructs native crop planning,
coverage, all calibrations, target times/distances, competing source identities,
repeat context, optical response interpretation and exact native tensor pixels.
All-area uint8 ignore masks must remain ones. Optical responses are replayed
offline; verification does not repeat network calls. Publication refuses existing
release/output paths; future changes need a new collection version.

```sh
python -m agents.training_annotation --verify
python -m agents.training_annotation --verify --local
python -m agents.training_windows --verify data/source_supported/training_native_v010_01/manifest.json
python -m pytest tests/unit/test_training_annotation.py -q
```

Source attribution: contains modified Copernicus Sentinel data (2019, 2020),
retrieved from Copernicus Data Space; the IIP observations and original annual
CSV hashes are retained from the [public NSIDC IIP archive](https://noaadata.apps.nsidc.org/NOAA/G00807/)
as documented in the [source audit](SOURCE_SUPPORTED_ANNOTATION.md). These source
organizations have not reviewed or endorsed the annotations or project.
