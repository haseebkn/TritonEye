# Source supported NL annotation reservation

Subsequent preparation is complete: the [native annotation report](NATIVE_SOURCE_ANNOTATION.md)
records nine inspected windows, two provisional vessel labels, seven unresolved
returns and two background contexts, all before scoring. The reservation audit
below describes the historical selection stage; its sealed zero-label records
remain unchanged. No iceberg class or independent truth was established.

The 14 October 2026 audit reserves **nine annotation areas across five native
HH/HV acquisitions**: six opportunities supported by IIP iceberg observations and
three supported by local AIS. These are candidate locations for image annotation;
there are **zero newly annotated SAR objects**. No imagery, detector outputs or
classifier scores were opened to make this selection.

The [sealed reservation](../reservations/source_supported/v0.1.0/manifest.json)
and [rankings](../reservations/source_supported/v0.1.0/rankings.json) retain the
selection and exclusions. Private AIS identities, coordinates, report histories,
competing-vessel pools and annotation envelopes remain under
`data/source_supported/v010_01/reservation/`. Public metadata contains no AIS
identities or exact positions. The original review guard, AllenAI quarantine,
locked test set and previously sealed reservations remain unchanged.

## IIP observation audit

Eight original annual files, 2014–2021, were downloaded from the public
[NSIDC IIP archive](https://noaadata.apps.nsidc.org/NOAA/G00807/).
The 101,868 input rows yield 81,113 unique valid sightings inside the NL study
area after 1 October 2014. Exclusions comprise 17,139 earlier records, 3,360
outside the study area, 243 invalid records and 13 exact duplicate sightings.
The original files and their hashes are retained. Signed west longitudes are
preserved, UTC times are read at their supplied minute precision, and the 2018
quoted multiline column headings are normalized explicitly. Invalid timestamps
or coordinates are excluded rather than repaired by guessing.

The immutable original rankings recorded 81,114 valid and 242 invalid rows.
Validation found one blank sighting time that the initial parser treated as
midnight: IIP record 2020:165, a SAT-LOW report on 8 September 2020. The parser
now rejects blank times, and the counts above include that correction. The row
was outside the primary VIS/R/V pool, so primary overlap counts and the nine
reserved areas are unchanged. The sealed rankings retain their original bytes;
the corrected audit is an additive record rather than a replacement reservation.
The [correction ledger](evidence/source_supported_correction_20261014.json) binds
the corrected local artifacts and confirms that the primary ranked rows are identical.

Only sighting positions and times are used. Predicted drift positions are not
substituted for observations. The primary pool contains **46,404 VIS or R/V
sightings** after excluding 15 records whose visual-method flag conflicts with
a satellite reporting source. An observer's call sign identifies the reporting
source; it is not the identity of the iceberg or a matched vessel.

The [IIP user guide](https://nsidc.org/sites/default/files/g00807-v001-userguide_1_1.pdf)
distinguishes visual, radar/visual, radar and satellite methods. Even a visual
report remains a provisional source assertion here: reporting accuracy, repeated
sightings, mistaken identity and the particular SAR return require review.
Satellite-only records can inform later comparisons but are excluded from the
primary label-support pool to reduce dependence on the same SAR interpretation.

Public Copernicus queries returned **73,368 catalogue entries**, reduced to
48,188 slice representatives by the existing identity/time/footprint alias rule.
Conflicting footprints are deliberately not collapsed. Native datatakes are
handled separately when reserving areas so adjacent slices do not count as
independent acquisitions.

There are **1,628 visual-source overlap pairs within two hours**, of which
154 fall within thirty minutes and involve 66 reported iceberg identities.
Neither a pair nor a repeated sighting is a unique verified SAR object. Ranking
prioritizes the thirty-minute screen, absolute time offset, radar/visual evidence,
online availability, UUID and reported iceberg identity. Reservation then prefers
datatakes outside known exposure and applies the complete-envelope and split checks.

The initial 2019 and 2020 queries silently stopped after 11,000 entries, in August
and July respectively. Date-based continuation recovered another 7,831 and 9,313
entries. A full page without a next link now triggers inclusive timestamp
continuation with UUID deduplication; no temporal progress causes an explicit
failure. The initial partial snapshots remain in `data/source_supported/v010_01/audit/`;
the completed snapshots and query receipts are in `complete_audit/`. Only those
completed snapshots enter `final_audit/iip_ranked.json` and this reservation.

## Local SAR AIS audit

The fresh complete-line archive snapshot contains **216,050 valid unique reports**
from 18 August 20:08:37 UTC through **14 October 09:32:35.714870 UTC**, covering
40 recording dates. The 19,621 legacy naive timestamps use the existing explicit
UTC interpretation. Recording remains intermittent; current recorder freshness
does not establish historical or area-wide reception.

The local audit queried 1,860 catalogue entries, yielding 930 slice representatives.
After 42 locked-test group/geography exclusions, 888 were screened: 19 temporal
candidates across 12 native datatakes, ten scenes with sparse support, and 859
with no observed overlap in this partial archive. The existing audit aligns full
track histories before checking the exact swath and NL study boundary.

The stricter HH/HV selection requires a report within 120 seconds, a bracketing
gap no wider than 240 seconds, estimated positional uncertainty no greater than
500 m, speed at least one knot, physical-water classification and reference
shore distance at least 500 m. Unknown shore distances are excluded. This yields
five track–scene opportunities across four products before reservation checks.
Ranking uses nearest-report offset, bracket width, uncertainty, offshore distance,
UUID and private identity. All competing aligned scene tracks remain available
in the private evidence; a selected expected vessel is not the only match option.

AIS is supporting evidence, not a complete vessel census or proof that a radar
return is a ship. No AIS absence establishes an iceberg. Source-relative shoreline
and scene-midpoint timing also require native-image checks during annotation.

## Reserved annotation batch

| Acquisition reference UTC | UUID prefix | Evidence opportunities | Existing role | Time offsets |
| --- | --- | ---: | --- | --- |
| 2017-05-02 09:49:54 | 8ecbf406 | 2 iceberg, VIS | New training groups | 54–185 s |
| 2015-06-30 09:49:12 | 5764b504 | 2 iceberg, R/V | New training groups | 108 s |
| 2017-06-01 09:48:48 | c75c517d | 2 iceberg, R/V | New training groups | 311 s |
| 2026-09-18 09:56:43 | 4b6f0ce7 | 1 vessel, AIS | Northern NL validation | 6.5 s |
| 2026-10-05 10:04:47 | dd031ca1 | 2 vessels, AIS | Northern NL validation | 11–89 s |

Each opportunity reserves a six-kilometre projected envelope, entirely inside
the catalogue footprint and NL study area. A later native window must fit inside
that envelope; it must not move to a more favourable model output. The IIP search
allowance is 300 m plus one metre per second of observation–SAR time separation.
This is a provisional search allowance, not a calibrated confidence interval or
an assertion about actual drift speed. Source observations have minute precision;
the catalogue midpoint is not the target's native radar acquisition time.

The batch caps selection at two opportunities per product and six per source
class, excludes repeated source identities, and separates same-scene envelopes
by at least one kilometre. Prior geographic and datatake split assignments are
retained. Locked and quarantined geography is excluded within ten kilometres;
conflicting roles and new cross-split geographic overlap are refused. The original
production review guard additionally checks the actual locked datatake and scope.

The exclusion ledger records swath-edge, overlap, identity, datatake and quota
failures. Six vessel examples were allowed by policy, but only three passed the
combined checks. The original AIS-supported datatakes have prior project exposure;
the chosen IIP datatakes have no recorded exposure. This is source-informed
selection, with preparer exposure to source records, not a fully naive or
independently reviewed sample.

**Both class and split are confounded in this batch:** iceberg opportunities
are historical training areas, while vessel opportunities retain existing 2026
validation roles. No two-class training, generalization or seasonal-improvement
claim can be made from that arrangement. Training vessels, validation icebergs
and class coverage under comparable conditions are still needed before a model
comparison. The batch first tests whether usable image labels and native inputs
can be prepared from these sources.

The 2015/2017 iceberg imagery also cannot be certified unseen by the Kaggle
classifier: individual competition chip dates and parent acquisitions are absent.
Any later classifier transfer report must disclose that possible overlap; a fresh
evaluation cohort should prefer acquisitions after the original 2017 competition.

## Annotation and input checks next

Use `first_pass_template.json` to record a complete inspection of the selected
native areas in both bands before viewing detector or classifier outputs. The
template contains no presumed class labels. Source selection is already disclosed;
freeze initial image decisions before consulting the separate supporting tracks
and IIP records for reconciliation. Retain the uncertainty region, competing
objects and contradictory evidence rather than assigning the nearest bright pixel.

Record native polarization, calibration LUTs and offsets, mode, local incidence
angle, spacing, pixel validity, target centring and edge truncation. Save original
pixels and extract native 75×75 contexts without stretching or guessed resampling.
Treat structures, clutter and unresolved targets explicitly. Any class supported
after reconciliation remains provisional and ignored by independent-truth gates.

Then compare isolated, supported targets against the frozen Kaggle training
distribution. Source support alone does not resolve the unknown Kaggle physical
spacing and preprocessing. Any native-input adaptation or preprocessing variants
must be developed and frozen before validation. The existing locked test set
remains outside this work.

## Reproduce and verify

Original CSVs, the user guide and archive index are stored in
`data/source_supported/v010_01/`. Public source hashes, acquisition UUIDs and
ranking criteria are retained in Git. Reservation builds require fresh private and
public output directories; use a fresh IIP audit directory to preserve prior evidence.
Catalogue and IIP metadata are public; this audit needed no new
credentials and sent no email or contact request.

```sh
python -m agents.source_supported --iip data/source_supported/v010_01/iip --output data/source_supported/replay_iip --catalogue data/source_supported/v010_01/complete_audit
python -m agents.source_reservation --verify reservations/source_supported/v0.1.0 --seal-sha256 eb2e49ff3e469940f3ca9f9aff3c6f585a4010388130da1ddbe7f59fdc18e521 --local
```

Verification without `--local` checks the public seal, policy and provisional
flags. Local verification requires the original hash-matching private bundle and
prior-assignment metadata at their recorded project paths. It additionally hashes
the frozen source files, private selection and template, including every CSV and
coverage journal listed in the AIS snapshot manifest. Snapshot entries must resolve
to files directly inside its `archive_snapshot/` directory. Verification then
replays reservation decisions from the saved IIP ranking and AIS audit. It does
not download new data, view imagery or create labels.
The [historical local verification receipt](evidence/source_supported_verification_20261014.json)
records a successful replay with the implementation identified by its code hash;
it predates the added AIS archive-file checks and does not attest to those checks.
The [post-review verification receipt](evidence/source_supported_verification_after_review_20261014.json)
records successful local replay with the strengthened archive-file checks.
To record verification with the current implementation, add `--receipt` with a
fresh project-local JSON path to the verification command.
Fresh local AIS collection can be audited with `python -m agents.overlap_audit
--output data/research/fresh_overlap --bundle-count 0`. Use `--fetch` with the IIP
auditor for fresh yearly catalogue queries rather than a frozen replay.

Data citation: International Ice Patrol (1995), *International Ice Patrol Iceberg
Sightings Database*, G00807 Version 1, NSIDC,
[doi:10.7265/N56Q1V5R](https://doi.org/10.7265/N56Q1V5R), accessed 14 October 2026;
2014–2021 annual files, filtered to the documented NL study area and period.
Contains Copernicus Sentinel metadata under the
[Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions).
