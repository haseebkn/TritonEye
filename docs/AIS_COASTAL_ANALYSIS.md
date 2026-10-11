# AIS-supported coastal-buffer analysis

Computed on 11 October 2026 from the frozen archive used by the
[latest overlap audit](OVERLAP_REVIEW.md), with **198,447 unique reports**.
This is an observed-AIS exclusion sensitivity study and a separate cached
detector proximity comparison. It does not select an operating buffer or
complete Step 2's independent false-alarm/missed-vessel validation.

## Coastal AIS exclusion sensitivity

Two cached eastern Newfoundland VV/VH slices have usable coastal observations.
After motion alignment, exact catalogue footprint, NL study polygon and valid
nonzero native pixels in **both** bands:

| Scene start UTC | Exact product UUID | Aligned AIS positions | Reference water | Reference land/mismatch |
| --- | --- | ---: | ---: | ---: |
| 2026-09-22 21:21:46 | bae55045-b7cb-46f5-8ed9-f504e42650b3 | 32 | 25 | 7 |
| 2026-10-04 21:21:46 | 0b5ad75c-9a10-4f0c-8bcc-f145aa1738e4 | 23 | 18 | 5 |

Each row contains at most one position per MMSI for that acquisition. These are
vessel-scene observations, not a claim of 55 distinct vessels across dates.
Reference-land positions stay separate; AIS positions near berths may disagree
with a mapped quay, the tidal waterline or motion alignment.

| Buffer | Sept 22 water positions withheld / eligible | Oct 4 water positions withheld / eligible |
| ---: | ---: | ---: |
| 0 m | 0 / 25 | 0 / 18 |
| 100 m | 21 / 4 | 18 / 0 |
| 300 m | 21 / 4 | 18 / 0 |
| 500 m | 21 / 4 | 18 / 0 |
| 1,000 m | 21 / 4 | 18 / 0 |

The water denominators stay fixed at 25 and 18. The 7 and 5 land/mismatch cases
are excluded by the physical reference at every buffer and are not counted as
additional coastal-buffer exclusions. No infrastructure flags or unknown
classifications occurred in these two samples.

At 300 m this withholds **84% and 100% of the observed reference-water subset**.
The sampled positions cannot distinguish 100 m from 300 or 500 m. They establish
neither that all withheld vessels were SAR-visible nor that narrowing the buffer
would improve usable recall. These exact slices have **no cached detector
outputs**. Their raw matches, matches after policy and detector misses remain
**null**, including when zero positions are eligible.

## Cached detector proximity comparison

Four other exact VV/VH products have complete cached xView3 inference. The model
and original inference settings are fixed; the comparison reclassifies the
retained returns with the current physical reference and changes only the buffer.
The two coastal AIS-only products above are not scored against predictions from
their adjacent slices.

| Scene start UTC | UUID prefix | Observed AIS on valid pixels | Raw proximity matches | Eligible matches at 0 / 100 / 300 / 500 / 1,000 m |
| --- | --- | ---: | ---: | --- |
| Sept 22 21:22:11 | 42286d3d | 0 | 0 | 0 / 0 / 0 / 0 / 0; recall unavailable |
| Sept 27 21:30:23 | 66d3167d | 0 | 0 | 0 / 0 / 0 / 0 / 0; recall unavailable |
| Sept 27 21:31:13 | 4f6115d9 | 2 | 2 | 2 / 2 / 2 / 2 / 2 |
| Oct 2 21:38:55 | 423801d3 | 1 | 1 | 1 / 1 / 1 / 1 / 0 |

Matching uses a fixed **500 m geodesic one-to-one gate**. For each buffer, the
assignment is recomputed with eligible returns and **all** observed AIS positions
as competitors, including withheld coastal and reference-land positions. An
eligible return can therefore still match a coastal AIS position across the
boundary. AIS policy exclusion counts and lost proximity matches are different
observations, and both are recorded.

All three supporting matches survive through 500 m. At 1,000 m, the Oct 2
position becomes coastal and its previously matching return is withheld; that
scene's eligible subset proximity recall changes from 1/1 to 0/1. This is a
specific policy effect in one observation, not evidence that 500 m is optimal.
The existing [3/3 result and conditional 29.2%–100% interval](AIS_SUBSET_RESULT.md)
remain small-sample supporting evidence. The new AIS-only counts do not enlarge
the detector recall denominator or supply confirmed identities.

## Reproduction and evidence

```sh
python -m agents.ais_coastal_analysis --output data/validation/ais_coastal_analysis_new.json
python -m pytest tests/unit/test_ais_coastal_analysis.py -q
```

The output must be a fresh path. The default
[manifest](../configs/benchmarks/ais_coastal_analysis.json) selects all six exact
products and buffers. The locally retained full run is
`data/validation/ais_coastal_analysis_20261011_v02.json`; the
[public aggregate ledger](evidence/ais_coastal_analysis_20261011.json) preserves
counts, exact product identities/times, model/configuration provenance, archive,
band, shoreline-component and method hashes. It omits verbose local shoreline
control details, while retaining their hashes and check summaries. Private MMSIs,
AIS positions, credentials, raw satellite imagery and basemap pixels are absent.

The public dependency hashes use the committed LF source bytes, pinned by
`.gitattributes` for reproducible checkouts. The original local full report and
its `local_full_report_sha256` remain unchanged; its historical dependency
hashes recorded CRLF execution bytes. This byte-contract correction changes
no measurements or original inference artifacts.

Original/COG aliases are rejected as duplicate physical slices. Full temporal
histories are aligned before spatial filtering; conflicting and stale reports
follow the existing bounded aligner. Raw DN zero padding, masks and nonfinite
values in either polarization reject a position/return. The current 300 m default
and original inference artifacts are preserved. Test identities, datatakes and
buffered test geography are checked before SAR access; locked test images are
not used. Source files are verified against the frozen archive and recorded
inference input hashes. AIS-only scenes record fresh native measurement hashes.

## Limits and interpretation

- Reception is incomplete and self-reported. Every scene has gap-or-unknown
  historical recorder bracketing; area-wide continuous reception is unproven.
- This convenience sample was chosen using AIS availability and cached data.
  Dense nearshore/possible berth observations dominate the exclusion comparison;
  vessel operating state was not independently adjudicated.
- Scene-start alignment uses catalogue precision for AIS-only scenes and the
  recorded whole-second start for cached inference. Per-pixel radar times are
  unavailable. Motion and location uncertainty can move cases across a buffer.
- The six slices belong to **four native acquisition groups**; adjacent slices
  and repeated identities are not independent trials. No pooled uncertainty
  interval is claimed for these selected coastal exclusion counts.
- OSM classification is reference-relative. Sparse controls support local checks
  but do not validate complete shorelines. These results cover Newfoundland
  scenes and do not establish Labrador performance or province-wide coverage.
- A 500 m proximity match may be coincidental or ambiguous and does not establish
  correct vessel identity. An unassociated return does not establish a dark vessel.
- False alarms, overall vessel recall, total missed vessels and a recommended
  buffer remain unavailable. Lower eligible-return counts do not measure fewer
  false alarms. Complete-area independent SAR labels remain necessary.

For the C-CORE portfolio, this demonstrates reproducible policy sensitivity and
honest handling of unavailable measurements. It does not claim C-CORE approval,
operational performance or compliance with unpublished acceptance criteria.
