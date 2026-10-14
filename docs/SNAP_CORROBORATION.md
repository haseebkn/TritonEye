# SNAP reference and four-candidate evidence audit

The four `plausible_iceberg` interpretations remain provisional, ignored for
accuracy and unapproved for training. This audit preserves the sealed image-first
decisions, original source allowances, review guard, AllenAI quarantine and locked
test. No classifier or detector scores are generated. One candidate retains its
training role and three retain their validation roles.

The initial protocol was frozen in commit `a53bc44`. The
[final runtime protocol](../configs/reservations/snap_corroboration_v020.json),
commit `45593e6`, pins the existing cached SNAP image before comparison execution;
scientific settings and source-search bounds are unchanged. The exact preparation
and [six-window annotation collection](FOLLOWUP_SOURCE_ANNOTATION.md) supply the
four candidates. This audit does not change the two source-coverage abstentions.

## Source corroboration

The source audit reuses the hash-checked IIP seasonal CSVs and their normalized
observations, preserving sighting positions and times. It checks additional
same-identity reports within 24 hours of the native SAR object time, deduplicates
identical sightings, and identifies visual/radar-visual reports separately from
satellite sources. Same-identity publisher reports are correlated context, not
independent confirmation. The original 30-minute and `300 m + 1 m/s × time gap`
association rules are unchanged. No inferred drift track is treated as truth.

| Candidate suffix | Additional same-identity sightings within 24 h | Optical catalogue search |
| --- | ---: | --- |
| 01 | 0 | Complete; 0 covering Sentinel-2 L1C products within ±3 h |
| 02 | 0 | Complete; 0 covering Sentinel-2 L1C products within ±3 h |
| 03 | 1, approximately 14.7 h earlier | Complete; 0 covering Sentinel-2 L1C products within ±3 h |
| 04 | 0 | Complete; 0 covering Sentinel-2 L1C products within ±3 h |

All four original sources use the R/V (radar/visual) method. The additional
candidate-03 report also uses R/V; it does not independently identify the SAR
reflector. The available CSV does not include original observer photographs,
logs or calibrated position errors. No email or other contact was sent.

Each optical query uses the already inspected SAR object's native TPS position,
the same ±3-hour window, the public Copernicus catalogue and an exact point-in-
footprint check on returned products. Raw responses, request URLs and retrieval
times are retained. Failure or truncated pagination cannot become a zero-coverage
claim. No optical pixels were inspected because no product passed this bounded
search. This is not an exhaustive search across sensors, dates or possible drift
positions, and does not establish that no optical evidence exists elsewhere.

The known local date-named AIS archives begin in August 2026; historical local
directories cover January 2025. They do not provide known co-temporal support
for these 2018/2020 acquisitions. This is an archive-name inventory, not a content
or reception-completeness audit. It supplies no evidence that vessels were absent.

**Result: zero newly confirmed iceberg identities.** The four plausible
interpretations and every original ignore mask remain unchanged. Repeated AI
inspection, source proximity and uniqueness in an incomplete source list do not
satisfy independent review.

## SNAP comparison method

Use the official ESA Sentinel Toolboxes 14.0.0 distribution in an already cached
Linux image, pinned by image ID
`sha256:ac08abe1c070613c2336d9cfa06746ef9eb7d12b99373f28fe878f27a2a70fa7`.
The original Linux installer was recovered from its cached COPY layer and matches
ESA's published SHA256
`786ce26a464ce32aef133af25b81281ad1b0af320e3614c0b465939fffbb4ee6`.
Both the image configuration digest and installer layer digest are verified.
The export, original installer and verification receipt are retained under
ignored project `data/tools/`. No new Docker image is downloaded, and existing
services are unchanged. Earlier slow Windows-installer transfers are retained
as unused partial artifacts; no Windows SNAP installation was performed.

Each original 384×384 HH/HV window is placed at its original native offset in a
sparse, full-dimension TIFF wrapper. Original manifest, annotation, calibration
and noise metadata accompany it. All other wrapper pixels are absent and must
never be evaluated. Only pointwise calibration/noise operators and the exact
output subset are permitted; these wrappers are unsuitable for contextual
filters, geolocation validation or full-scene analysis. Full publisher file
checksums remain unverified by partial acquisition.

Three separate graphs per acquisition read original amplitude, calculate sigma0
in dB without noise removal, and calculate sigma0 after thermal-noise removal.
Raw HH/HV output must equal original DN exactly before numerical agreement can
pass. Calibration uses the frozen 0.001 dB maximum absolute difference limit
against the existing sealed pixels. Band swaps, wrong dimensions, changed raw
pixels, failed processing and nonfinite calibration values cannot pass.
Thermal-noise output is a sensitivity variant, with nodata accounted for; it does
not replace the sealed input or select a model preprocessing recipe.

An initial diagnostic exposed a bug in the new comparison reader: these SNAP
outputs have `_dB` names but declare `PHYSICAL_UNIT=intensity` and store positive
linear sigma0. The initial reader compared those linear values directly with
dB, producing large apparent differences. The corrected reader requires explicit
DIMAP units and identity scaling, converts declared intensity using `10*log10`,
retains declared nodata as missing, and rejects unknown units/scaling. It never
infers units from filenames or a fitted value distribution. Regression tests
reproduce the misleading name with known linear values. The original execution
report remains intact; a separate interpretation report replays the same saved
SNAP output. No native pixel or calibration parameter is changed.

Graphs execute one at a time, with one processing thread, a 1,536 MiB Java heap,
128 MiB tile cache and project-local temporary files. Process waits are configured
for ten minutes; elapsed-limit compliance is reported separately from numerical
agreement.
The container has a two-GiB memory/swap cap, one CPU, read-only root filesystem
and no network. Only the new run directory is mounted. Timeout cleanup removes
only that run's uniquely named container. Processing requires three
GiB of available physical/commit memory and five GiB of free project storage.

One original noise graph completed successfully but recorded **1,596.2 seconds**
elapsed, exceeding the configured 600-second wait. The cause is not established;
the record does not prove that the elapsed limit was met. The runner now polls
against a monotonic deadline and explicitly records elapsed-limit overruns,
including after a host pause; a program cannot enforce a deadline while its host
is suspended. The public evidence keeps numerical agreement separate from
`bounded_execution_all_passed`. This is not a successful runtime-limit result.

All **eight calibration bands across four acquisitions pass**, and raw amplitude
equals original DN exactly. Each band compares 147,456 native pixels. The maximum
absolute difference is **0.0000025064 dB**, against the predeclared 0.001 dB limit.
This establishes numerical agreement for these windows with noise removal off.
All twelve graphs exited successfully; the elapsed-limit overrun above remains
an explicit failure of that separate resource target.

| Candidate suffix | Maximum error, HH / HV, dB | Median HV change with noise removal, dB | HV pixels consistent with inferred power floor |
| --- | --- | ---: | ---: |
| 01 | 0.000001423 / 0.000002298 | −6.32 | 44,605 / 147,456 (30.25%) |
| 02 | 0.000001590 / 0.000002282 | −74.61 | 74,776 / 147,456 (50.71%) |
| 03 | 0.000002307 / 0.000002506 | −72.92 | 77,802 / 147,456 (52.76%) |
| 04 | 0.000001593 / 0.000002450 | −5.08 | 32,431 / 147,456 (21.99%) |

The post-run noise diagnostic reconstructs pre-calibration surviving power as
`DN² × 10^((noise_dB − nominal_dB)/10)`: the matching calibration LUT cancels.
It counts values near `1e-5` at relative tolerance `1e-5`. This is an exploratory
description of the observed numerical pile-up, chosen after examining the noise
results, not a classifier filter or a preregistered scientific accuracy test.
HH fractions are 0%, 4.20%, 0.55% and 0.01%, respectively. The corresponding dB
floor varies with the LUT (approximately −105 to −106 dB here); a fixed −100 dB
check would miss it. These fractions describe the full 384×384 contexts, not
the labelled objects or a missed-vessel rate.

All noise-output pixels are finite, but finite floor values are not evidence of
useful signal. The marked HV change prevents automatic adoption as a Kaggle
classifier input recipe. Correct physical noise treatment, publisher processing
equivalence and class transfer remain unresolved. This audit selects no new
normalization, resampling, masking or score threshold.

The [public calibration/source evidence](evidence/snap_corroboration_20261014.json)
is pinned at SHA256
`99d60a8616a4458667fbee05aba919656ca9d9440add8773a05c9b9c480d4c80`.
The separate [noise diagnostic](evidence/snap_noise_floor_20261014.json) has its
own [integrity lock](../configs/reservations/snap_noise_floor_v010.lock.json).
The public release binds 204 SNAP output files, including DIMAP metadata and
ENVI headers. Private execution logs, exact graphs, original diagnostic report,
corrected interpretation report, original source evidence and catalogue receipts
remain available for local replay. Original SAFE/noise metadata acquisition
transferred 3,388,628 bytes; it acquired no new measurement imagery.

## Interpretation and replay

Agreement with SNAP would establish numerical consistency for these native
windows under matching settings. It would not establish the Kaggle source's
physical scale, full preprocessing, object identity, classification accuracy or
false-positive reduction. No histogram matching, resizing, training or promotion
is authorized by this report. Source licences and original provenance remain in
[the source audit](SOURCE_SUPPORTED_ANNOTATION.md).

```powershell
docker image load --input data/tools/snap14_cached/image.tar
python -m agents.snap_image --archive data/tools/snap14_cached/image.tar --output data/tools/snap14_verified_replay
python -m agents.snap_reference --runtime data/tools/snap14_verified_replay/receipt.json --metadata data/source_supported/snap_metadata_v010_02/metadata.json --output data/source_supported/snap_reference_replay
python -m agents.iceberg_corroboration --output data/source_supported/corroboration_replay
python -m agents.source_reference_report --verify --local
```

Acquisition/execution commands require fresh output directories. The exact cached
image export and ignored native/source evidence are needed for local replay;
they are not distributed through Git. A rebuilt image would require its own
recorded runtime identity and audit rather than silently replacing the frozen
reference. Verification replays saved pixels
and catalogue receipts without rerunning Java or querying the live catalogue;
omit `--local` to check only the public, hash-pinned release.

Sources: [ESA SNAP download and checksums](https://step.esa.int/main/download/snap-download/),
[SNAP calibration](https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.calibration.ui/operators/CalibrationOp.html),
[thermal-noise removal](https://step.esa.int/main/wp-content/help/versions/13.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.calibration.ui/operators/ThermalNoiseRemoval.html),
and [IIP user guide](https://nsidc.org/sites/default/files/g00807-v001-userguide_1_1.pdf).
