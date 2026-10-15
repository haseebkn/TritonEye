# Training-region AIS reception and provisional C/D exports

Implemented against main `9b2430420f47ccb4bfd109f8d197f08edeba9a44`
(PR #38). This release adds an observed-reception audit and an executable
training admission/export workflow. All object labels remain provisional.

## What the fresh AIS audit found

The [versioned coverage evidence](evidence/training_ais_coverage_v010.json)
freezes **223,468 unique reports** from 18 August to 15 October 2026,
ending **09:50:12.615035 UTC**. The public Sentinel-1 catalogue query returned
1,932 records. Source CSVs, connection journals, catalogue responses and role
metadata are hash-bound in `data/research/training_ais_coverage_v010_01/`.
Files were captured sequentially at complete-line boundaries; the snapshot is
not a transaction across all recorder files. Legacy timestamps retain the
existing disclosed UTC interpretation (19,621 reports).

The four envelopes are the original training areas, each 0.2 degrees longitude
by 0.13 degrees latitude. They sample Burin and Cartwright coastal/offshore
water; they do not describe reception throughout either region or province.

| Training envelope | Received reports | Identities | C/D HH/HV physical slices | Slices with tight AIS support |
| --- | ---: | ---: | ---: | ---: |
| Burin coastal | 0 | 0 | 10 | 0 |
| Burin offshore | 0 | 0 | 21 | 0 |
| Cartwright coastal | 0 | 0 | 23 | 0 |
| Cartwright offshore | 1 | 1 | 23 | 0 |

The sole Cartwright offshore report is at **10 October 2026, 05:19:14.665601 UTC**.
The longest observed silent intervals, including the snapshot boundaries, are
57.57 days in the three empty envelopes and 52.38 days in Cartwright offshore.
These are intervals without received reports, not proven recorder outages or
intervals without vessels. Fresh global recorder activity cannot establish
reception in a particular training envelope.

The evidence includes hourly report counts, time since the last local report,
exact product IDs, native acquisition groups, whole/partial envelope swath
coverage and training-role exclusions. Native/COG aliases count once. Adjacent
slices remain separate physical slices, with native group counts reported
separately; rows in the table must not be added as independent acquisitions.

For each pass, complete nearby track histories are aligned before filtering to
the envelope and exact swath. A tight source screen needs a report within 120
seconds and bracketing reports separated by at most 240 seconds around the
catalogue midpoint. This is a metadata sampling screen, not an admission check
at native object time, a physical-water verification or guaranteed input
compatibility. No new eligible training pass has tight support in these
envelopes. Missing footprints produce no pass. No SAR images are read by the
coverage audit, including excluded acquisitions.

**Consequence:** the current feed provides no usable vessel training opportunity
in these four envelopes. More recording time is useful only if reception reaches
eligible training geography. A future source should be checked for actual tracks,
temporal density and competing identities before downloading more SAR windows.
Changing a subscription boundary alone does not establish reception. Existing
validation opportunities retain their roles.

## C/D admission and export

The new [policy](../configs/reservations/cd_provisional_v010.json) is additive.
It binds the accepted C/D v0.2.0 input specification and inherits every general,
vessel and iceberg evidence check from the unchanged A/B admission policy.
It supports Sentinel-1 C/D IW GRDH HH/HV, IPF 004.03, native 10 m sampling,
nominal uncorrected sigma0 dB, complete 2x75x75 float32 contexts and identity
normalization. Future processing versions need another acceptance decision.

Every proposed object needs an annotation completed before scoring, prior
exposure disclosure, acquisition/geographic/repeated-object grouping, permission
evidence and source uncertainty/alternative-return checks. Vessels need AIS
bracketing at native target time and competing-track review. Icebergs still need
additional physical corroboration identifying the same reflector. Proximity to
IIP, missing AIS, model confidence and AI agreement cannot replace that evidence.

Evidence records bind the exact object, source identity, repeated-object group,
annotation and native receipt. The consumer checks declarations and local
receipts; it cannot establish their physical truth or independent review.
Current source-supported iceberg hypotheses therefore remain on hold.

Before opening SAR arrays, the workflow checks NL scope and existing split and
quarantine metadata. It reconstructs a densely sampled native window boundary
from both bands' matching geolocation grids, checks that against the declared
scope and protected geography, and verifies mission, orbit, datatake, timing,
processor, band order, spacing and noise status. It replays both DN/LUT
calibrations and requires exact float32 agreement with saved nominal pixels.
Invalid DN, incomplete contexts and incompatible metadata cause refusal. Native
metadata agreement still does not establish absolute geolocation accuracy.

An admitted export contains only classified object chips, with dossier hashes,
native timing/incidence, class, acquisition, geographic and repeated-object
groups. These groups must remain intact in later development comparisons.
Unknown pixels do not become negative examples. No original annotation or ignore
mask is rewritten. `exploratory_training_admitted` may become true for qualifying
objects; `training_approved`, `independent_review`, `accuracy_truth_admitted` and
`promotion_eligible` remain false, and `ignore_for_accuracy` remains true.

Exports require a fresh directory under `data/native_training_exports/`.
All proposed dossiers must pass before any export is published. Duplicate
objects are rejected, and within a batch each source identity and each
repeated-object group bind to exactly one counterpart on the other side: one
source identity cannot carry two different repeated-object groups, and one
repeated-object group cannot be shared by two different source identities or
classes. The same source identity may repeat its own group and class across
multiple observations. Writes occur in a temporary directory and the final
directory appears only after all writes succeed. An interrupted write may
leave a `.partial-*` directory, which is not a published export. Verification
replays the admission and pixels, re-applies the same one-to-one identity/group
consistency check across the manifest's entries, and checks present split
constraints as well as the frozen decision.

## Use and replay

Audit creation queries public catalogue metadata and snapshots the live archive;
it does not change recorder settings. Current evidence is immutable: another
audit release needs a new version/path rather than overwriting this one.

```sh
python -m agents.training_ais_coverage --verify
python -m agents.training_ais_coverage --verify --local
```

Prepare metadata receipts and intentionally incomplete dossier drafts from a
saved native preparation in the existing C/D preparation format:

```sh
python -m agents.cd_label_export --templates-from-prepared data/ca_p2/prepared.json --output data/source_supported/cd_admission_v010_01
```

These four already-inspected acceptance contexts are engineering examples,
not new reserved annotation areas or class labels. Prior exposure is disclosed.
Generic blank dossiers are also available:

```sh
python -m agents.cd_label_export --template data/source_supported/new_cd_dossier.json
python -m agents.cd_label_export --assess data/source_supported/cd_admission_v010_01/dossier_0000.json
```

For an actual object, retain the draft and save a new annotated dossier/version.
Its `annotation` reference points to a JSON record with identical product/name,
scope/group, split, object ID, class, provisional status, timestamp, native
reference, half-open `bbox_px`, source identity and repeated-object group. Each
required check cites a JSON evidence record carrying those object/source/group
identities, exact `native` and `annotation` references, and a `checks` mapping.
Every reference uses project-relative `path` and SHA-256 `sha256` fields.

Native receipts produced by the bridge contain the exact full-raster
`window: [column, row, width, height]`, saved calibrated `values` reference,
SAFE `manifest` reference and ordered `bands`. Each band supplies
`polarization`, `annotation`, `calibration`, `noise` and original-window `dn`
references. `bbox_px` is relative to that window. Neither preview images nor
Kaggle-normalized values can supply these pixels.

A batch JSON contains `policy` (the new policy reference) and `dossiers` (a list
of completed dossier references). Do not include unresolved or held objects.

```sh
python -m agents.cd_label_export --export data/source_supported/my_batch.json --output data/native_training_exports/my_batch_v001
python -m agents.cd_label_export --verify data/native_training_exports/my_batch_v001/manifest.json
```

The delivered workflow is exercised with synthetic evidence for successful
admission/export, failure and tampering tests, and all four retained real C/D
acceptance windows for calibration/input replay. Synthetic evidence validates
software behavior only. A local empty-batch export at
`data/native_training_exports/cd_rehearsal_v010/` was captured and verified
against the initial submitted implementation (commit `15264f1`) and explicitly
contains zero chips. Its manifest pins that commit's `cd_label_export.py`
hash, so it is historical evidence of that commit only and does not replay
`--verify` against a later implementation; each subsequent implementation
change needs its own fresh rehearsal output at a new path, created and
verified with the matching code, rather than reusing or overwriting this one.
This release admits **zero real training labels**, runs
no fitting or classifier scoring, and establishes no detection improvement.
Independent scientific review remains unavailable. Original A/B policies,
quarantine, locked test, annotation masks and promotion/review guards retain
their existing contents and behavior. No organization was contacted.

Contains modified Copernicus Sentinel metadata/data (2026). AIS observations are
partial supporting evidence under the project's existing source-use limits.

Implementation validation: **110 focused tests passed**, including the frozen
AIS replay, four retained real C/D native contexts, export failure/tampering
cases and existing provisional admission regressions. Ruff, Black and mypy pass
for the new modules and tests. The repository delivery gate additionally runs
review, portable tests and CI before merge.
