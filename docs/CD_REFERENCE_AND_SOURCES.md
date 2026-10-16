# Sentinel-1 C/D reference pilot and source refresh

This is a provisional research audit, executed on 15 October 2026. It compares
existing C/D training pixels with SNAP, refreshes vessel reservation screening,
and searches for physical corroboration of five existing iceberg hypotheses.
It produces no detector/classifier scores, fitted model, new class annotation,
training admission or promotion. The original A/B input contract, review guard,
AllenAI quarantine, original locked test, historical reservations and masks remain
unchanged. These acceptance criteria are provisional research choices, not externally endorsed requirements.

## Reference-processing design

The [frozen pilot protocol](../configs/reservations/cd_reference_v010.json) selects
two already reserved training acquisitions, each with coastal and offshore
contexts. These development images were inspected in earlier work. The pilot
copies the central `[64:448,64:448]` original samples from each existing 512-pixel
window, before comparison, with no target-based repositioning or resampling.

| Mission | Exact product UUID | Acquisition start UTC | Region |
| --- | --- | --- | --- |
| C | `7e9631d8-2ac4-4718-9d67-f7b0f65f7039` | 2026-07-03 09:49:24 | Burin shelf, Newfoundland |
| D | `4689632a-a8a0-42a1-b6e2-476663671005` | 2026-07-02 09:56:07 | Cartwright shelf, Labrador |

Both products report IPF 004.03. The original SAFE manifests, calibration/noise
XML and referenced auxiliary-file names are retained. Annotation, calibration and
noise identities must agree on mission, polarization, mode, orbit, datatake and
time. The pilot requires HH/HV, IW GRD, 10 m sampling and no prior thermal-noise
correction. These conditions describe the native samples; 10 m sampling does not
mean 10 m effective resolution.

The cached SNAP 14.0.0 image and original installer are pinned to the same hashes
as the [earlier reference audit](SNAP_CORROBORATION.md). Twelve graphs run
sequentially: raw amplitude, nominal sigma0, and noise-removed sigma0 for each
window. A two-GiB container cap, 1,536 MiB JVM heap, one CPU, 128 MiB tile cache,
disabled container network and 600-second per-process deadline bound execution.
No existing service or Docker setting is changed.

Original DN must match SNAP raw pixels exactly. Nominal sigma0 must agree within
the predeclared 0.001 dB maximum absolute difference. A separate metadata-LUT
implementation is compared against SNAP noise removal with the same tolerance;
noise-floor counts and nominal-to-corrected differences remain diagnostics.
DIMAP units determine conversion, never the band filename. Sparse full-dimension
wrappers contain only these windows; they cannot support contextual filters or
full-scene evaluation. Full publisher file checksums remain unverified.

Nine pixel-centre positions per window compare SNAP's bilinear tie-point
interpolation with the existing TPS method. This is an interpolation diagnostic,
not absolute geolocation validation. It records HH/HV positional agreement and
native incidence. No post-hoc tolerance turns that diagnostic into ground truth.

The [versioned evidence](evidence/cd_reference_v010.json) reports each mission,
band and regime separately, including failures and elapsed-limit overruns.
Eleven of twelve graphs exited successfully. C coastal and both D contexts pass
nominal calibration: **six measured bands pass** the 0.001 dB limit. The C offshore
nominal graph timed out and did not produce a complete band inventory; its two
bands are unmeasured. The configured deadline was 600 seconds, but observed
elapsed time was **1,308.1 seconds**. The reason for that delay is not established;
this is not successful deadline compliance. The runner cannot enforce a deadline
while its host is suspended. The timed-out task's container was removed; original
partial outputs and logs remain retained.

A retry command was implemented, but the actual retry was refused before launch
because physical-memory headroom was below the unchanged three-GiB requirement
(a subsequent check showed about 2.13 GiB). No limit was relaxed and no unrelated
service was stopped. There is **no successful retry** in this release. A later
same-settings retry must create a fresh artifact and retain this first attempt.

For the three fully compared contexts, five of six noise-removal bands meet the
same strict tolerance. D coastal HH fails at one of 147,456 pixels, with a maximum
difference of **0.01262 dB**; the other 147,455 pixels are within tolerance. At the
worst pixel, the local reconstruction leaves approximately 0.000973 DN-power
units from original power 529 after noise subtraction. This is a near-cancellation
diagnostic, not an established cause or permission to relax the tolerance. C
offshore noise output exists, but the combined comparison is incomplete and does
not count as a passed noise reference. The report therefore records D nominal
processing as passing, C nominal completion as failing, and neither mission as
passing the complete noise comparison.

HH/HV metadata positions agree at the nine checked positions in every context.
The two interpolation methods differ by maxima of **5.71–7.31 m** across the four
windows. Native centre incidence spans approximately **37.06–44.90 degrees**.
Those ranges describe this pilot, not a validated operating envelope.

Reference agreement does not establish Kaggle preprocessing equivalence,
classification transfer or false-positive reduction. The default A/B consumer
continues to reject C/D; the pilot configuration is not a replacement input
contract or permission to train.

## Refreshed vessel reservation

The new immutable archive snapshot contains **221,955 unique AIS reports**,
through **2026-10-15 05:46:34.955133 UTC**. The live recorder reported fresh at
audit time; freshness does not establish area-wide reception. The fresh catalogue
audit found 21 temporal candidate slices, 12 sparse-support slices and 890 slices
without observed overlap. These are screening categories, not measured scenes
or verified empty waters.

Five HH/HV source opportunities pass the original AIS timing, speed, water and
uncertainty screen. A conditional C/D screen then applies the original geographic,
acquisition, exposure and identity exclusions. It also includes the subsequent
two native-training reservations in the prior inventory.

**Zero new vessel areas can be reserved for training.** Four opportunities are
in previously exposed/reserved C acquisitions with validation roles. The D
opportunity fails protected/quarantine geographic separation. Compatibility
alone would not remove those blockers. The
[new reservation manifest](../reservations/cd_vessels/v0.1.0/manifest.json)
retains every exclusion; it is an explicit empty reservation, not a silent
failure or a reassignment of validation data. Complete competing-track pools and
exact snapshot hashes remain private under project `data/`.

## Iceberg corroboration search

The [separate discovery protocol](../configs/reservations/cd_evidence_search_v010.json)
expands the five existing hypotheses to Sentinel-2 L1C and Landsat Collection 2
Level 2 within +/-12 hours. A 43.5 km half-width envelope is a search heuristic
(300 m + 1 m/s for 12 hours), not a calibrated drift prediction or a change to
the original association rules. Queries are limited to 100 records and report
complete, truncated or failed status. All ten queries completed without
pagination; raw responses and request parameters are retained for offline replay.

| Existing hypothesis | S2 tile records in search envelope | Cover original SAR position | Landsat records |
| --- | ---: | ---: | ---: |
| Follow-up 01 | 1 | 0 | 0 |
| Follow-up 02 (training role) | 3 | 1 | 0 |
| Follow-up 03 | 0 | 0 | 0 |
| Follow-up 04 | 4 | 0 | 0 |
| New training 01 | 0 | 0 | 0 |

The eight S2 records represent tiles from **three acquisition times**, not eight
independent observations. The sole tile covering an original position is
`S2A_MSIL1C_20180504T142911_N0500_R096_T22TCT_20230818T152412.SAFE`,
UUID `4600e3a9-b973-4204-b60f-9a917bc9fd90`. Its catalogue time is 17,243.82 seconds
(4.79 hours) after the SAR target; scene cloud cover is 11.40%. Scene cloud cover
does not establish local visibility, and catalogue acquisition time is not a
pixel-level timing measurement. Optical pixels were not inspected. The motion
gap and incomplete competing-object evidence prevent same-reflector confirmation.
This is a documented optical lead, not a newly corroborated training label.

Public-source discovery also checked:

- [NSIDC IIP sightings](https://nsidc.org/data/g00807/versions/1): the catalogue
  describes dated sighting coordinates and class codes, with coverage ending in
  2021. It does not supply the missing target-specific photographs in this audit.
- [Canadian Ice Service chart descriptions](https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/products-guides/chart-descriptions.html):
  iceberg charts aggregate observations and forecast conditions. They can support
  regional context, but cannot independently identify our SAR reflector.
- [Fisser's Sentinel-1 Iceberg Areas dataset](https://data.mendeley.com/datasets/z9jdmxrkvn/1):
  published under CC BY 4.0 with SAR/optical matched-object statistics. Its
  [associated implementation](https://github.com/hfisser/S1.IcebergArea) targets
  EW imagery. The file-metadata endpoint returned HTTP 403; NL overlap, exact
  product identities and annotation applicability were not established. No data
  were adopted or assumed to be NL IW labels.

No organization was contacted. All five iceberg hypotheses retain their original
provisional classes, splits and ignore masks. No repeat sighting, broad regional
chart, absent AIS or model score is promoted into physical corroboration.

## Replay and next decision

```sh
python -m agents.cd_reference --prepare --output data/source_supported/fresh_cd_metadata
python -m agents.cd_reference --run data/source_supported/fresh_cd_metadata/prepared.json --output data/source_supported/fresh_cd_snap
python -m agents.cd_reference --retry data/source_supported/cd_snap_v010_01/report.json --output data/source_supported/fresh_cd_retry
python -m agents.overlap_audit --output data/research/fresh_cd_ais
python -m agents.cd_source_refresh --reserve data/research/fresh_cd_ais/audit_private.json --output data/source_supported/fresh_cd_reservation
python -m agents.cd_source_refresh --search --output data/source_supported/fresh_cd_search
python -m agents.cd_evidence --verify
python -m agents.cd_evidence --verify --local
```

Acquisition/execution commands require fresh project directories and public
network access; metadata retrieval uses the existing local CDSE credentials.
Offline `--verify --local` checks saved source bytes, DN, calibrated values,
SNAP outputs, source screening and optical summaries; it does not query APIs,
run Java or inspect the locked test. Public verification needs only committed
release artifacts. New evidence requires an additive release.

Next, retry the incomplete C graph when the documented memory headroom is
available, retaining the original failure. Seek fresh vessel acquisitions in
eligible training geography. Before
fitting C/D, freeze a new executable input contract with a bounded mission and
processing-baseline scope and test it on permitted examples. The optical lead
needs local visibility, motion and identity evidence before label admission;
continued inability to establish that identity is a valid hold result. Independent
review, defensible class coverage and native model-transfer measurements remain
open. This audit does not establish measured detection improvement.

Contains modified Copernicus Sentinel data (2018, 2019, 2020, 2026).
Processing context: [C calibration update](https://dataspace.copernicus.eu/news/2026-2-5-update-sentinel-1c-radiometric-calibration),
[D operational data opening](https://sentinels.copernicus.eu/-/sentinel-1d-user-data-opening-from-17-april-2026),
and [D May/June quality notice](https://sentinels.copernicus.eu/web/sentinel/-/sentinel-1d-sar-data-quality-degradation).
The selected July D product is outside that notice's affected interval; the
notice is not a substitute for per-product quality checks. Landsat metadata
access uses the [public Planetary Computer catalogue](https://planetarycomputer.microsoft.com/docs/).
