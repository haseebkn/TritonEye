# Source association and preprocessing cross-check

The subsequent [SNAP reference and source audit](SNAP_CORROBORATION.md) now passes
eight calibration bands for four follow-up development windows, while recording
large noise sensitivity and an elapsed-limit overrun. The earlier GDAL failure
below remains historical evidence; no independent class approval or Kaggle input
equivalence follows from the later numerical comparison.

Subsequent [native preparation and provisional annotation](FOLLOWUP_SOURCE_ANNOTATION.md)
is complete for the six reserved windows: four plausible iceberg interpretations,
two unresolved returns and two source-coverage abstentions. All remain ignored
and unapproved; the reservation-stage statements below retain their original scope.

On 14 October 2026, the nine sealed native areas were audited without changing
their annotations, source allowances or ignore masks. The separate float64
calibration calculation passes all 18 bands. The next batch reserves six
iceberg-source opportunities across five acquisitions, with no image labels yet.
Input compatibility and independent review remain unresolved.

The [sealed audit evidence](../reservations/source_supported/v0.2.0/audit_summary.json)
contains per-object diagnostics, calibration differences, noise metadata and
descriptive distributions. The [reservation manifest](../reservations/source_supported/v0.2.0/manifest.json)
records the selected products, split roles and exclusions. Neither release
approves training, scoring, a threshold or promotion.

## What the association audit establishes

Original IIP CSV text has two decimal places for the six previous sighting
positions. A hypothetical nearest-rounding cell extends roughly 661–671 m from
its centre to a corner. The original 300 m base allowance did not explicitly
represent this cell. Decimal representation is not positional accuracy: the
publisher's actual rounding/truncation method and observation errors are unknown.
The audit therefore reports half-step and symmetric one-step sensitivity,
plus 30/60-second time sensitivity, without widening the old association gates.

At the eleven inspected object/context locations, TPS versus piecewise-linear
annotation-grid positions differ by 1.3–17.4 m; a half-pixel convention changes
positions by about 7.1 m. Removing nine interior annotation-grid points per
product yields maximum interpolation residuals of 7.4–189.8 m. The nearly exact
inverse round trips use the same grid and are not independent validation.
No municipal imagery ground control or absolute shoreline check was performed
in this audit. Source-coordinate error, grid interpolation, motion displacement
and competing objects still cannot be disentangled into a proven failure cause.

Some returns outside the original gates are closer to hypothetical rounding
cells, but that does not establish iceberg identity. Two sources overlap diffuse
texture rather than isolated returns. The previous result stays **two provisional
vessels, seven unresolved returns, two background contexts and zero defensible
iceberg labels**, with whole-area ignore masks.

## What the preprocessing cross-check establishes

The numerical reference separately parses the original calibration XML, checks
regular pixel grids, interpolates in float64 with SciPy and applies
`20*log10(DN/sigmaNoughtLUT)` at the original native offsets. It shares the ESA
equation and source LUT with the pipeline but does not call the pipeline's
calibration/interpolation functions. All 18 bands pass the frozen 0.001 dB
tolerance; maximum absolute difference is **0.000002608 dB**. This checks
calibration arithmetic, not physical equivalence to Kaggle inputs.

Original SAFE manifests and noise XML were downloaded (3,737,949 bytes for the
completed metadata acquisition). GDAL 3.12.1, through rasterio 1.5.0, independently
reads the saved native DN exactly through sparse full-dimension reference
wrappers. Only the saved windows contain real pixels; the rest of each wrapper
is absent/zero and must never be evaluated. Attempting the calibrated GRD
`SIGMA0:...:INTENSITY` view fails on all 18 windows with "Measurement bands not
found." This failure is retained explicitly. **No successful external calibrated
processor or SNAP comparison is claimed.** No source annotation was edited to
force that view to open.

Six of ten product-band noise XMLs use legacy noise vectors; four have range and
azimuth vectors. All ten native annotations report thermal-noise correction
false. The audit inventories these schemas without subtracting a guessed noise
model or changing the frozen pixels. A complete processor/noise-treatment
comparison remains a compatibility dependency.

The final distribution comparison reuses the hash-verified summaries of the
frozen **1,122 training chips** (595 ships, 527 icebergs). It computes statistics
for the ten native extracted contexts, which include unresolved returns and
backgrounds. The first diagnostic run used all 1,604 publisher-labelled chips;
that run remains an unused local artifact. Its statistics are not the published
reference and were not used to select transformations. No model was trained,
no scores were calculated, and no NL locked-test pixels were opened.

| Descriptive quantity | Kaggle training ships | Kaggle training icebergs | Native mixed contexts |
| --- | ---: | ---: | ---: |
| Median HH dB | −19.87 | −22.11 | −20.47 |
| Median HV dB | −26.07 | −26.71 | −25.11 |
| Median contrast-proxy width, pixels | 8 | 10 | 17 |
| Median contrast-proxy height, pixels | 14 | 14 | 21.5 |

The size proxy is a fixed descriptive component calculation, not a hull
measurement or class label. Broad overlap does not prove compatibility. Native
10 m spacing is known; physical resolution is different, and Kaggle's physical
scale and complete preprocessing remain unknown. No histogram matching,
resampling or input adaptation was selected from these results.

## Next reservation and remaining gaps

Commit `a5f133e` froze the new policy before creating the reservation. It prioritizes
post-2017 visual or radar/visual reports of medium/large/very-large icebergs,
first within 15 minutes and then within the existing 30-minute limit. Small
icebergs within 15 minutes are a lower-priority fallback; growlers, bergy bits,
unknown sizes and satellite sources are excluded. Larger reported size is a
selection preference, not proof of SAR visibility.

| Acquisition UTC | Reported size opportunities | Existing/frozen role | Absolute source offset |
| --- | --- | --- | --- |
| 2020-05-23 09:40:58 | 1 medium | validation | 469 s |
| 2018-05-04 09:41:26 | 1 medium | train | 1,161 s |
| 2020-04-05 09:40:56 | 1 medium | validation | 1,191 s |
| 2018-04-10 09:41:25 | 1 large | validation | 1,282 s |
| 2019-05-29 09:41:17 | 1 medium, 1 large | validation | 1,410–1,710 s |

Five of six opportunities exceed 15 minutes: larger reported size trades against
temporal uncertainty here. Before acquiring their pixels, check native 384-pixel
footprints and whether each source uncertainty region is adequately covered.
Incomplete coverage must be recorded or the area abstained; it must not trigger
post-inspection relocation or a wider identity gate.

Previously inspected/reserved datatakes and source identities are excluded.
Original geography, native-acquisition roles, the ten-kilometre protected
separation and quarantine exclusions are enforced. New disjoint geographic
groups receive deterministic roles from a frozen hash rule; existing assignments
take precedence. This creates **one training and five validation iceberg
opportunities**. No fresh AIS opportunity passes: four are previously exposed
datatakes and one conflicts with protected geography. This is an honest vessel
coverage gap, not permission to relabel earlier validation vessels as training.

The reservation contains zero annotations and no newly inspected imagery.
Its source-centred selection is disclosed. After scoped preparation, inspect both
bands before detector/classifier scores; freeze image observations, then
reconcile with sources. All later labels remain provisional and ignored by the
independent-truth guard. Full class/split balance, independent review, successful
external preprocessing comparison and demonstrated false-positive reduction
remain outstanding.

## Replay

Original artifacts and failed attempts remain under project `data/`. Each command
requires a fresh output directory. No credentials are needed for local replay;
metadata reacquisition uses the existing ignored Copernicus credentials.

```sh
python -m agents.source_error_audit --prepared data/source_supported/native_v011_02/manifest.json --evidence data/source_supported/annotation_v010_01/source_support.json --iip data/source_supported/v010_01/iip --output data/source_supported/error_replay
python -m agents.source_preprocessing --prepared data/source_supported/native_v011_02/manifest.json --metadata data/source_supported/reference_metadata_v010_02/metadata.json --collection datasets/source_supported/v0.1.0/collection.json --output data/source_supported/preprocessing_replay
python -m agents.source_next_batch --verify reservations/source_supported/v0.2.0 --seal-sha256 30d4f8987fa84954578a0604274b386802e7b1189c07b7be0f75fb62e753a905 --local
```

Public verification omits `--local`; it checks the pinned release and provisional
contract. Local verification checks hashed inputs and replays the selection from
the frozen source pool and assignment snapshot. It does not query current
catalogue availability or establish new ground truth.

Sources: [NSIDC IIP user guide](https://nsidc.org/sites/default/files/g00807-v001-userguide_1_1.pdf),
[ESA Sentinel-1 processing](https://sentiwiki.copernicus.eu/web/s1-processing),
[GDAL SAFE driver](https://gdal.org/en/stable/drivers/raster/safe.html), and
[Kaggle data description](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data).
Original source licences and acquisition provenance remain in the
[source audit](SOURCE_SUPPORTED_ANNOTATION.md).
