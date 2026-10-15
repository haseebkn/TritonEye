# Reception-prioritized vessel-training reservation attempt

On **15 October 2026**, the eight areas in the frozen
[NL reception shortlist](NL_RECEPTION_AND_ICEBERG_SEARCH.md) were checked against
a fresh AIS snapshot and public Sentinel-1 catalogue metadata. **Zero new
acquisitions qualify for reservation.** This is a completed source-selection
attempt with a documented shortfall, not a usable vessel-training batch.

The [v0.3.0 manifest](../reservations/cd_vessels/v0.3.0/manifest.json) records
the priorities and each source-screen outcome; the
[policy](../configs/reservations/reception_vessels_v030.json) and lock preserve
the decision. All labels remain provisional. No SAR/optical pixels or classifier
scores were inspected, no annotations were created, and no training labels were
admitted. Review requirements, quarantine and locked test assignments are intact.

## What was screened

The frozen snapshot contains **224,622 unique reports**, 502 more than the
previous reception-map snapshot. It spans 18 August 2026 at 20:08:37 UTC to
15 October at **13:04:07.758005 UTC**. Its 19,621 legacy naive timestamps retain
the explicitly disclosed UTC interpretation. Complete-line prefixes were captured
sequentially while the recorder continued running; this is not a transactional
snapshot or evidence of continuous area-wide AIS coverage.

The public catalogue returned **1,965 records**; the shared alias-aware overlap
audit produced **771 HH/HV scene records** over that archive interval. These
are catalogue/processing units, not independent vessel examples. Full available
swath-wide aligned competing histories are retained privately.

Selection preserves the original eight-cell order, then ranks nearest AIS time,
bracket gap, uncertainty, product UUID and private identity. Source requirements
remain nearest report at most 120 seconds, bracket at most 240 seconds, finite
uncertainty at most 500 m, speed at least 1 knot, and reference water at least
500 m from shore. Unavailable AIS speed values are rejected. Whole-cell and
candidate-envelope roles are checked separately; existing acquisition roles,
source identities, exposure and protected geographic separation remain binding.
The batch cap is four, with one window per native acquisition and no repeated
source identities. These are research discovery criteria, not C-CORE standards.

**24 aligned supports** fall in the eight priority cells. Reasons overlap:

| Failed source check | Supports failing |
| --- | ---: |
| Speed below 1 knot or unavailable | 22 |
| Reference-water/500 m shore separation | 22 |
| Bracket over 240 seconds, missing or invalid | 11 |
| Alignment uncertainty over 500 m or invalid | 3 |
| Nearest report over 120 seconds or missing | 1 |

These are source-support counts, not unique vessels, independently annotated
targets, false alarms or missed detections. Twenty-three supports fail at least
one source check. The sole remaining support is on
`S1C_IW_GRDH_1SDH_20260918T095656_20260918T095721_009500_012E71_F89B.SAFE`,
UUID `01876305-6ca7-4a24-9004-b9bf7a2d9848`. It has a 9.40 s nearest offset,
79.01 s bracket and 239.22 m estimated alignment uncertainty, but its native
acquisition was previously exposed/reserved. It is excluded rather than reused
as a fresh training example. Passing this source screen would still not identify
a vessel reflector or satisfy native object-time/permission checks.

## Retained priorities and the remaining blocker

The manifest retains all eight cells with a fresh whole-cell role check.
The first priorities remain the two adjacent western Newfoundland cells
−58.1 to −57.9 longitude / 48.9 to 49.0 latitude, followed by the northern cell
−55.6 to −55.5 / 51.3 to 51.4. The earlier report contains exact bounds and
reception counts for every cell. Adjacent western cells must retain consistent
geographic grouping when a future acquisition is reserved.

These are **prospective collection priorities**, with
`acquisition_reserved: false`. No future pass, usable AIS reception or object
identity is promised. No scheduled task or receiver subscription was changed.
The empty annotation template is an explicit no-work handoff; it cannot supply
training examples or justify scoring.

Progress now requires a new, tightly supported moving-vessel overlap in eligible
training geography, or permitted historical tracks meeting the same checks.
Replay this workflow with a fresh version when that evidence arrives. Before
any later admission, freeze exact products/envelopes, verify native processor
and input compatibility, annotate all visible returns before classifier scores,
check native target-time uncertainty and competing identities, and retain
ambiguous objects as ignored. AIS sampling is incomplete and source-centred
selection is biased. Slow/nearshore vessels remain real possible vessels; these
discovery exclusions do not define their class or measure detector performance.

## Reproduction and validation

Raw AIS, identities, positions, catalogue responses, original/enriched overlaps,
annotation template and frozen prior roles remain under
`data/research/reception_vessels_v030_01/`. Public outputs contain cell envelopes,
product identities and source-screen decisions, with no vessel identities,
individual tracks or object positions. Original source access and usage
conditions still apply; this metadata reservation grants no permission to
redistribute tracks or train on a future object.

Public verification checks the lock, source/implementation hashes and guards:

```sh
python -m agents.reception_vessel_batch --verify
```

Full retained-data replay reconstructs the overlaps, physical reference
classification, priority ranking, split checks and reservation outcome:

```sh
python -m agents.reception_vessel_batch --verify --local
python -m pytest tests/unit/test_reception_vessel_batch.py -q
```

Behavioural tests exercise successful synthetic reservations, full available
competing-pool retention, privacy, failed source checks, protected geography,
remote acquisition roles, previous exposure, repeated identities/slices, separate
same-day acquisitions and the blocked empty outcome. Synthetic passing fixtures
are engineering tests, not additional NL observations. New metadata uses stable
LF bytes; two existing implementation inputs explicitly retain their CRLF
checkout bytes so evidence verification behaves consistently on Windows/Linux.

Local validation passed **103 focused tests**, including the existing reception,
provisional admission and C/D export regressions. Retained-data replay passed;
Ruff, Black and mypy passed across **174 Python files**. An initial sandbox run
could not create pytest temporary directories; the successful run used a fresh
writable directory. Publication additionally runs automated review and CI.

`--capture FRESH_PRIVATE_DIRECTORY` collects fresh metadata into an ignored
directory. `--publish RETAINED_PRIVATE_DIRECTORY` creates the public reservation
only if its version/lock do not already exist. A later refresh needs fresh public
version constants/policy and a fresh private directory; existing releases are
never overwritten. Capture and publication are separate so input collection can
be diagnosed without silently replacing an existing decision.
