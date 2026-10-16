# C retry, D noise diagnosis and optical inspection

Executed on 15 October 2026 against main baseline `3038bf4` (PR #36).
This additive follow-up completes the missing C nominal comparison, explains the
D noise discrepancy and records why the optical iceberg lead remains on hold.
The [versioned evidence](evidence/cd_followup_v010.json) retains the original
attempt, source hashes, numerical failures and subsequent results. No labels,
model inputs, operating thresholds or model-selection state were changed.

## C reference retry

The unchanged three-GiB physical/commit headroom and five-GiB disk guards passed.
The first setup attempt under `data/source_supported/cd_snap_retry_v010_02`
failed before launching SNAP because its SAFE calibration path exceeded Windows'
path-length limit. That partial directory is retained. A path-length preflight
now refuses an oversized retry directory before creating a wrapper.

Using the shorter `data/cd_r1` directory, the C offshore calibration graph exited
successfully in **132.33 seconds**, within the original 600-second deadline.
The original incomplete attempt and its 1,308-second overrun remain retained;
this retry does not establish the cause of that earlier overrun. A regression
checks that a timed-out launcher and its child process are terminated.

The retry reran only the failed calibration graph, reusing and hashing the
successful raw/noise outputs from the first attempt. Exact original DN equality
and the frozen 0.001 dB nominal-calibration tolerance pass in both polarizations.
The C offshore maximum differences are approximately 0.00000218 dB HH and
0.00000244 dB HV. **All eight nominal bands now pass**, across two existing
training acquisitions and four previously inspected contexts. This is a small
processing pilot; absolute geolocation and learned model transfer remain open.

## D noise discrepancy

The historical line-index diagnostic still reproduces D coastal HH's
**0.01261855 dB** mismatch at local row 263, column 315, corresponding to native
column 23,848, line 1,596. Nothing overwrites that historical result.

SNAP 14's published implementation interpolates GRD range-noise vectors using
their stored acquisition timestamps in MJD2000, then multiplies the azimuth
factor. The earlier local diagnostic interpolated those vectors by line index.
The stored UTC timestamps and the line-index timing approximation are not
numerically interchangeable near cancellation. The new diagnostic independently
implements the timestamp calculation and records float32 LUT/residual storage.

| Quantity at the original failing pixel | Value |
| --- | ---: |
| Original DN power | 529 |
| Line-index noise estimate | 528.9990267055427 |
| Time-interpolated noise estimate | 528.9990238735444 |
| Time residual after float32 storage | 0.0009761264664 |
| SNAP corrected sigma0 | -85.7222579105 dB |
| Time replay corrected sigma0 | -85.7222580823 dB |

An ablation changing residual storage alone retains a 0.01261867 dB maximum
error. Changing time interpolation while retaining the earlier storage convention
reduces that maximum to approximately 0.00000084 dB. The full reference replay
reduces it to approximately **0.00000067 dB**. This isolates time interpolation as
the material cause in this pilot, rather than an incorrectly matched native pixel.

SNAP clips *negative* residuals to its positive floor; it does not raise every
small positive residual to that floor. This distinction and float32 storage are
explicit in the new replay and have separate regressions. No tolerance was relaxed.
All **eight noise diagnostic bands** pass 0.001 dB; their largest remaining
difference is about **0.00000070 dB**. These are replays against the already saved
SNAP output, not eight new processor runs or new independent acquisitions.

The runtime module JAR is hashed and reports implementation version 14.0.0.
The audit retains the published 14.0.0
[noise operator source](https://github.com/senbox-org/microwave-toolbox/blob/38998c9baecfc1d568647abbd6424b054f423fdb/sar-op-calibration/src/main/java/eu/esa/sar/calibration/gpf/Sentinel1RemoveThermalNoiseOp.java)
and [UTC representation source](https://github.com/senbox-org/snap-engine/blob/14.0.0/snap-core/src/main/java/org/esa/snap/core/datamodel/ProductData.java).
Matching version metadata and numerical output does not prove a reproducible
source-to-binary build. The original nominal A/B input contract remains unchanged;
the noise replay is diagnostic and is not substituted into model inputs.

## Optical lead and exposure record

The exact previously discovered Sentinel-2 L1C tile was retrieved:
`S2A_MSIL1C_20180504T142911_N0500_R096_T22TCT_20230818T152412.SAFE`,
UUID `4600e3a9-b973-4204-b60f-9a917bc9fd90`, supporting the existing training-role
candidate `next_iceberg_support_02`. Four native bands (B02/B03/B04/B08), the
three-plane classification mask and product/tile metadata are retained under
project `data/`. Total transfer was **91,465,185 bytes**. Complete downloaded
pixel assets pass the publisher manifest's SHA3-256 checksums. The entire SAFE
product was not downloaded.

Collection-1 reflectance uses the explicit -1,000 offsets and 10,000
quantification value. RGB rendering uses a fixed 0–0.3 reflectance range and
gamma 2.2 solely for inspection; native samples are retained. Tile sensing time
is 14:29:35.174344 UTC, about 24 seconds later than the catalogue product time.
Neither value establishes per-pixel timing or object identity. The original
catalogue-based gap of 4.79 hours and motion-search heuristic remain disclosed.

**An avoidable scope error occurred:** initial optical previews used the full
17.54 km heuristic motion-search radius. That expanded area intersected
quarantined AllenAI geographic footprints. No AllenAI SAR pixels or labels were
opened, but the optical exposure cannot be undone. The two original previews
are retained by hash as exposure records, excluded from label admission and
absent from the published inspection images. All original quarantine files,
class decisions, ignore masks and split assignments remain unchanged.

A separate frozen inspection policy now restricts optical reads to the original
permitted SAR context's bounding rectangle, with an explicit **10 km separation**
from protected test/quarantine footprints checked before raster reads. The actual
minimum separation is **11,410.95 m**. The 455 × 458 native optical window covers
only **1.69%** of the requested motion-search square; it cannot establish complete
competing-object coverage. The full compressed JP2 files remain local provenance,
not permission to inspect other geographic areas.

The permitted RGB window shows clouds at the SAR location and no-data pixels to
the west. Only **63.10%** of that window has valid values in all four bands.
All **25 central 60 m classification samples** flag opaque cloud. Unflagged mask
values elsewhere are not treated as verified clear sky. The imagery was inspected
by an AI assistant with prior SAR-label exposure, without classifier scores or
an independent reviewer. It does not permit reliable surface-object identification.

The [inspection decision](../configs/reservations/optical_lead_observation_v010.json)
therefore holds the candidate: cloud, missing data, incomplete motion coverage,
unresolved reflector identity and prior unscoped optical exposure prevent
admission. No iceberg or vessel label, confirmed negative or accuracy truth was
created. No additional corroboration was established.

## Replay and remaining work

```sh
python -m agents.cd_reference --retry data/source_supported/cd_snap_v010_01/report.json --output data/short_fresh_retry
python -m agents.cd_noise_diagnosis --snap data/cd_r1/report.json --output data/fresh_noise_diagnosis
python -m agents.optical_lead --download --output data/fresh_optical_download
python -m agents.optical_lead --render data/fresh_optical_download/download.json --output data/fresh_permitted_optical_review
python -m agents.cd_followup_evidence --verify
python -m agents.cd_followup_evidence --verify --local
```

Execution/download commands require fresh directories. The first attempt can
have one bounded execution retry; the retry report itself cannot be retried.
Local verification replays original native values, all reference comparisons,
publisher checksums, permitted optical geometry, masks and render hashes without
network requests, Java execution or classifier inference. Public verification
checks the committed release and guard/decision consistency without private data.

The immediate numerical blockers in this pilot are now explained and replayed.
A new executable C/D input specification still needs its own scoped acceptance
checks, preferably another permitted acquisition per mission, before experiments
use those missions. Newly eligible AIS-supported training acquisitions remain
needed. Iceberg labels require another usable physical observation that identifies
the same reflector. This cloudy tile supplies none. Independent review and
measured native vessel-detection improvement remain outstanding.

Contains modified Copernicus Sentinel data (2018, 2026). The results and acceptance
criteria are provisional portfolio research and are not externally endorsed operational requirements.
