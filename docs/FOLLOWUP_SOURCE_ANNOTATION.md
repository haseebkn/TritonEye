# Six native follow-up SAR windows

On 14 October 2026, the six reserved NL HH/HV windows across five acquisitions
were prepared and annotated before scoring. Four returns have **plausible
iceberg** interpretations supported by IIP observations; two remain **unresolved**
because their source-uncertainty scenarios are not fully covered. These are
provisional AI interpretations, with prior source exposure and no independent
review. None is approved training data, accuracy truth or a confirmed iceberg.

The [sealed collection](../datasets/source_supported/v0.2.0/collection.json)
contains decisions and coverage diagnostics, while the original
[six-area reservation](../reservations/source_supported/v0.2.0/manifest.json)
and [earlier nine-window collection](NATIVE_SOURCE_ANNOTATION.md) retain their
original bytes. All six whole-area ignore masks contain one everywhere
(one means excluded). The original benchmark/test split, AllenAI quarantine,
review guard and active model configuration are unchanged.

## Metadata first, then exact native pixels

Commit `03a7519` froze the
[preparation and coverage protocol](../configs/reservations/source_followup_v010.json)
before acquiring annotation metadata. A metadata-only plan retained all six
original centres, product UUIDs, native datatakes and split/geographic roles.
Commit `f2e1d4a` pinned that plan's SHA256 before the first new measurement pixels
were read. No substitution, recentering, padding, resampling or window expansion
was used. The 384×384 footprints fit their reserved envelopes, catalogue swaths
and NL study boundary and pass the existing development-scene guard.

The preparation transferred **264,999,386 bytes**, plus the earlier metadata-only
planning requests. Cropped original DN, original calibration/annotation XML,
byte-range receipts, sigma0-dB HH/HV GeoTIFFs and paired fixed −30 to +5 dB
previews remain under `data/source_supported/followup_native_v010_01/`.
All **12 native band calibrations replay exactly**. All six unresampled
75×75 contexts were extracted with no edge/size abstention. They are six image
contexts, not six confirmed class examples.

The inputs are native IW GRDH HH/HV; local centre incidence ranges from
32.64 to 39.67 degrees. Pixel spacing is not physical resolution. This work adds
no thermal-noise subtraction, terrain correction or moving-target displacement
correction. Partial downloads do not verify publisher full-file checksums.
The earlier [preprocessing cross-check](SOURCE_ERROR_PREPROCESSING.md) did not
establish external calibrated-processor or Kaggle input equivalence. The subsequent
[SNAP/source audit](SNAP_CORROBORATION.md) passes eight numerical calibration bands
for four candidates; input equivalence and independent class review remain unresolved.

## Source-uncertainty coverage

The check uses each source's literal CSV decimal representation, observation
time, native first/last window-row times and densified native footprint. In a
local WGS84 azimuthal-equidistant plane it evaluates three frozen scenarios:

- Original point allowance: 300 m plus 1 m/s of the largest absolute observation
  separation across the first and last native rows.
- Nearest-rounding sensitivity: a coordinate half-step cell plus the original
  allowance and 30 seconds of time sensitivity.
- Broader sensitivity: a symmetric coordinate one-step cell plus the original
  allowance and 60 seconds of time sensitivity.

Complete coverage uses cell containment and minimum distance to the footprint
boundary, with a small numerical margin. Approximate area fractions are
descriptive only. Decimal representation does not establish source accuracy;
the rounding convention and actual position/motion errors remain unknown.
These scenarios are not calibrated confidence regions or identity probabilities.
Even complete coverage cannot prove that the observed reflector is the source.

| Window suffix | Original coverage | Nearest-rounding sensitivity | Broader sensitivity | Source-class policy |
| --- | ---: | ---: | ---: | --- |
| 01 | 100% | 100% | 100% | Provisional interpretation allowed |
| 02 | 100% | 100% | 100% | Provisional interpretation allowed |
| 03 | 100% | 100% | 100% | Provisional interpretation allowed |
| 04 | 100% | 100% | 100% | Provisional interpretation allowed |
| 05 | 100% | 96.3% | 85.3% | Abstain |
| 06 | 98.0% | 81.2% | 64.7% | Abstain |

The two incomplete windows were retained for research inspection. Their nearby
returns do not override the pre-pixel abstention. No claim of a missing object,
negative background or false alarm follows from incomplete coverage.

## Image observations sealed before source reconciliation

All six paired previews were scanned in all four quadrants in both bands.
One compact return per window was recorded as unresolved in
[the first pass](../configs/reservations/source_followup_first_pass_v010.json),
sealed in commit `3632501` before calculating target-to-source distances.
Boxes use native coordinates with exclusive upper bounds, excluding preview
headers/gutters. Fixed display limits, sparse boxes and a complete visual scan
do not establish exhaustive annotation. Fine speckle/background and possible
unnoticed targets remain ignored. No verified vessel-negative areas were created.

The annotator was Codex, not a qualified independent SAR reviewer. Source-centred
selection, stream names, source precision and coverage results were already
known. No source overlays, detector outputs or classifier scores were viewed in
the first pass. This is disclosed score-free curation, not fully naive or
independent review. No second AI pass is represented as independent validation.

After that commit, source reconciliation used each object's native acquisition
time and TPS location. It screened all available visual/RV IIP observations
within 30 minutes, excluding satellite-source codes. Matching retains the
original 300 m plus 1 m/s rule; sensitivity scenarios do not widen matching gates.
The pool is incomplete supporting evidence, not a census of all physical objects.
No co-temporal area-wide AIS or independent optical confirmation was available.

| Window suffix | Distance to selected source | Native-time allowance | Decision |
| --- | ---: | ---: | --- |
| 01 | 294 m | 780 m | Plausible iceberg, provisional |
| 02 | 450 m | 1,453 m | Plausible iceberg, provisional |
| 03 | 608 m | 1,493 m | Plausible iceberg, provisional |
| 04 | 988 m | 1,587 m | Plausible iceberg, provisional |
| 05 | 269 m | 1,720 m | Unresolved: incomplete coverage |
| 06 | 363 m | 2,018 m | Unresolved: incomplete coverage |

Each return has one compatible IIP identity in that frozen pool. A source-consistent
return can still be an unrelated vessel or clutter; absent AIS cannot establish
an iceberg. In particular, the long time gaps and kilometre-scale allowances in
windows 02–04 weaken identity confidence. Their classes express plausible source
interpretations, not four correct matches or an accuracy numerator.
The [reconciliation record](../configs/reservations/source_followup_reconciliation_v010.json)
preserves every original box and states these alternatives.

The four plausible examples span four acquisitions: one training and three
validation areas, retaining their frozen roles. The two unresolved examples share
the fifth acquisition. There are still no training-role vessels in this new
batch, no independently reviewed iceberg labels, no measured classifier transfer
accuracy or false-positive reduction, and no training/promotion approval.

## Replay and integrity

```sh
python -m agents.source_followup --verify data/source_supported/followup_native_v010_01/manifest.json
python -m agents.source_followup_annotations --verify datasets/source_supported/v0.2.0 --seal-sha256 35c46df881403aff0b73149262a9ea07574472d85b890553504b9aca23a568d2 --local
```

Public verification omits `--local` and validates the pinned release, decisions
and provisional guards without loading private evidence or imagery. Local replay
also checks the pinned pre-pixel plan, original source metadata, scope, 12
calibration bands, source reconciliation, all ignore masks and exact native
75×75 extraction. The [collection lock](../configs/reservations/source_followup_collection_v020.lock.json)
pins the release. Reacquisition uses the existing ignored Copernicus credentials
and a fresh project-local output directory. No new credentials or contact request
were required. Original licences and IIP/Copernicus provenance remain in the
[source audit](SOURCE_SUPPORTED_ANNOTATION.md).
