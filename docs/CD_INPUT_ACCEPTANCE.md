# Experimental C/D inputs and refreshed vessel reservations

Implemented on 15 October 2026 against main baseline `9144327` (PR #37).
The additive [C/D specification](../configs/native_nl/input_v020.json) and
[consumer](../agents/native_cd_input.py) define native research tensors. The
original A/B specification, annotations, quarantine, locked test and independent
review/promotion requirements retain their original bytes and behavior.

## Input scope

The v0.2.0 recipe accepts Sentinel-1 C/D IW GRD HH/HV with native 10 m sampling
and IPF **004.03** metadata. It preserves nominal uncorrected sigma0 dB, channel
order HH then HV, identity normalization and complete, unresampled 75x75 float32
contexts. There is no padding, resizing, clipping or noise-corrected substitution.
Sampling does not establish effective spatial resolution or equivalence to Kaggle.

The consumer checks mission/product naming, acquisition timestamps, processing
software/version, polarization order, spacing, units, noise status, incidence
validity and complete target/context support. Unsupported metadata abstains;
test-role requests are rejected. Returned tensors carry no training or promotion
approval. The caller must authenticate original pixels and metadata, enforce NL
and split scope, and obtain a separate object admission decision. A centre crop
used to exercise extraction is not an object annotation.

The C/D recipe is separate from the historical A/B consumer and Kaggle model.
Other processor versions require an explicit new acceptance decision. The
existing provisional admission policy remains pinned to A/B: a future C/D
training export needs a separately versioned policy retaining its evidence
requirements. No fitting or classifier/detector scoring occurs here.

## Additional acceptance design

The [predeclared policy](../configs/reservations/cd_acceptance_v020.json) chooses
the earliest unseen online native HH/HV acquisition per mission between July 10
and August 1 covering both original training envelopes. C uses Burin; D uses
Cartwright. Acquisitions, windows, tolerances and input settings were fixed before
the comparison. These are additional dates in existing training geography,
not new geographically held-out examples. Prior project exposure is disclosed.

| Mission | Additional acquisition UTC | Original training group |
| --- | --- | --- |
| C | 2026-07-15 09:49:24 | Burin shelf, Newfoundland |
| D | 2026-07-14 09:56:08 | Cartwright shelf, Labrador |

Both coastal and offshore windows are prepared with the existing scoped native
reader. Each comparison uses a fixed central 384x384 subset of its 512x512
context. Sparse SAFE wrappers contain only saved native windows and are valid
only for these pointwise comparisons. Full publisher SAR file checksums are not
verified by partial downloads; local DN, XML and output hashes are retained.

The pinned SNAP 14 image runs sequentially with 2 GiB container memory, one CPU,
1536 MiB Java heap, 128 MiB cache and a 600-second deadline per graph. Existing
three-GiB physical/commit and five-GiB disk headroom requirements apply. Raw DN
must agree exactly and both calibrated bands must pass the unchanged **0.001 dB**
tolerance. A nine-position metadata interpolation comparison requires matching
HH/HV grids and separation at most 25 m. That predeclared engineering limit
does not measure absolute geolocation or validate shoreline alignment.

The first preparation stopped on identical repeated IPF software entries;
the parser now accepts identical repetitions and rejects conflicting versions.
That partial directory remains under `data/ca_p`. The first analysis also rejected
native XML UTC fields without a timezone suffix. The explicit conversion now
records UTC offsets. The initial processing report and analysis failures are
retained; a separate replay recomputes checks on the identical saved SNAP outputs.
No reference graph, pixels, crop, processor or tolerance changes in that repair.

The [versioned acceptance evidence](evidence/cd_acceptance_v020.json) records
every execution, comparison, metadata value, tensor hash and decision. Noise
estimates use the corrected timestamp interpolation solely to describe
below-noise content. These additional runs do not repeat SNAP noise removal;
the prior eight-band noise replay remains in the
[C retry and D noise diagnosis](CD_RETRY_NOISE_OPTICAL.md).

**Result: all eight nominal bands pass across the four additional contexts.**
The largest absolute calibration difference is **0.000002603 dB**. Original DN
matches exactly, all four centre contexts produce unchanged 2x75x75 float32
research tensors, and all eight reference jobs exit successfully in **41.81–96.85
seconds** each. There are no timeouts or elapsed-limit failures in these new jobs.
The metadata interpolators differ by **5.75–7.32 m** at the sampled positions;
HH/HV metadata positions agree. Centre incidence spans **37.06–44.91 degrees**.
These figures describe these four contexts, not the mission-wide operating range.
The successful preparation transferred **225,418,515 bytes**; the earlier partial
preparation remains retained separately.

Combined with the earlier pilot, nominal reference comparisons now cover **four
acquisitions and eight contexts**, with two acquisitions per mission. The earlier
timeout and line-index noise discrepancy retain their original records. The
new input scope is experimental nominal processing under the declared metadata
checks; the prior noise diagnostic is not a corrected-input qualification.

Absolute geolocation, all-condition C/D qualification, class identities, Kaggle
transfer and learned detection improvement remain unestablished. These are
provisional portfolio research criteria, not C-CORE requirements or endorsement.

## Vessel reservation refresh

The latest local AIS archive is frozen before overlap analysis; the public
catalogue query is refreshed. The
[v0.2.0 vessel reservation](../reservations/cd_vessels/v0.2.0/manifest.json)
records the exact archive endpoint, all source-eligible opportunities, exclusion
reasons and any selected training areas. Full identities, competing tracks,
positions and annotation templates remain under ignored project `data/`.

Selection preserves acquisition/geographic roles, excludes validation, locked
test and quarantine geography, applies the existing 10 km protected separation,
and excludes previously exposed/reserved acquisitions and identities. Additional
acceptance acquisitions are also excluded from the new reservation. Existing AIS
screening limits, moving-vessel requirement and complete 6 km envelope remain
unchanged. These rules are research sampling choices.

Any selected area is an annotation opportunity. Its processor version, native
target timing, uncertainty coverage, visible competing returns and identity still
require checks before provisional admission. AIS reception is incomplete; an
empty admissible reservation does not imply empty water. Source timing at scene
midpoint is not bracketing at the native target time. All labels remain
provisional, with zero labels or verified negatives admitted by this release.

The refreshed archive contains **223,047 unique reports**, ending **15 October
2026 at 08:52:36.096744 UTC**. The catalogue returned 1,932 records, representing
923 audited physical slices after exclusions: 21 temporal candidates, 12 sparse
support slices and 890 without observed overlap. Reception limits prevent
interpreting those last 890 as empty scenes.

The unchanged strict AIS source screen yields **five vessel opportunities**.
**Zero can be reserved for new training**: four reuse already exposed/reserved
native acquisitions, and one fails locked/quarantine geographic separation.
C/D mission eligibility therefore removes one earlier input blocker but does
not remove the remaining evidence/split constraints. No new vessel or iceberg
label, training export or model improvement is claimed. Fresh AIS support in
eligible training geography is still required.

## Replay

```sh
python -m agents.cd_acceptance --verify
python -m agents.cd_acceptance --verify --local
python -m agents.cd_training_reservations --verify
python -m agents.cd_training_reservations --verify --local
```

Public verification checks committed specifications, decisions and hash-bound
release evidence. Local verification replays the original calibrated pixels,
SNAP results, metadata/tensor checks and frozen AIS ranking without Java, network
requests or model inference. Fresh discovery/execution needs fresh project data
directories; old attempts must not be overwritten. The original locked test is
never opened for processing, annotation or model selection.

Implementation validation: **124 focused tests passed**, including retained-data
replay of the new acceptance and vessel ranking, the earlier C/D reports, native
input/noise checks and provisional admission regressions. Ruff, Black and mypy
passed for the changed Python modules and tests. The delivery pipeline additionally
checks the complete portable suite and Docker/CI before merging.

Contains modified Copernicus Sentinel data (2026). Source licences and retained
evidence limitations continue to apply. No organization was contacted.
