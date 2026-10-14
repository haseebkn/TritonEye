# Provisional native NL training policy and reservation v0.1.0

Subsequent [native preparation and provisional annotation](TRAINING_NATIVE_ANNOTATION.md)
is complete for both reserved windows: one new source-compatible iceberg hypothesis,
one incomplete-coverage area without a distinct annotated target, and zero training
admissions. The sealed reservation below retains its original selection-stage state.

The [executable policy](../configs/reservations/provisional_training_v010.json)
permits separately recorded **exploratory training decisions** when a new object
dossier satisfies explicit evidence requirements. Independent review remains
unavailable. These decisions never admit accuracy truth or approve promotion.
The original [31-area admission register](LABEL_ADMISSION.md), annotations,
ignore masks and split assignments retain their original bytes and hold decisions.
No existing label is admitted by this release; no model is trained or scored.

## Requirements for a provisional training label

The [consumer](../agents/provisional_training.py) requires a training role,
disclosed exposure, annotation completed before scores, a hash-bound annotation
receipt with the same product/object/class/role/time, and the frozen
[native HH/HV recipe](NATIVE_NL_INPUT.md). It checks NL scope, the existing locked
test guard and prior acquisition/geographic roles. A validation example cannot
become training data by changing a declaration. Each evidence check must cite
a hash-bound JSON record identifying the same product and object and recording
the named check as passed. A missing check leaves the object on hold; changed
evidence bytes are refused.

Required checks cover native calibration and extraction, complete 75x75 support,
source uncertainty coverage, both-band area inspection, recording all visible
returns, alternative returns, source error/motion, association ambiguity,
acquisition/geographic/repeated-object groups and permitted use. These receipts
document the preparer's evidence and decisions. They do not independently prove
physical identity, exhaustive annotation or the correctness of an assertion.

Vessel evidence must include AIS bracketing the native target acquisition time,
competing aligned AIS tracks, and a defensible association to the visible return.
An iceberg needs a visual or radar/visual observation independent of the SAR
interpretation **and additional physical corroboration identifying the same
reflector**, with alternative vessel explanations considered. Another AI opinion,
nearby source coordinates, uniqueness in a partial pool, absent AIS, a predicted
drift position or a model score cannot satisfy that corroboration requirement.
The existing four plausible icebergs do not acquire new evidence through this policy.

A positive result is `provisional_exploratory_training`, with
`exploratory_training_admitted: true`. It retains `training_approved: false`,
`ignore_for_accuracy: true`, `independent_review: false` and
`promotion_eligible: false`. The first flag describes this restricted research
decision; the others retain the existing approval boundary. This command creates
no training export and changes no original masks. Any eventual fitting collection
needs an additive, versioned export, original receipt history, acquisition/object/
geography separation and an uncertainty sensitivity analysis. Unknown objects and
unreviewed background cannot be exported as verified negatives.

## Source-ranked training-only reservation

The [sealed manifest](../reservations/native_training/v0.1.0/manifest.json) and
[full screened ranking](../reservations/native_training/v0.1.0/rankings.json)
reserve **two previously unexposed acquisitions, both training only**:

| Catalogue acquisition start UTC | Product UUID | Source opportunity | Midpoint offset | Search allowance | Geographic group |
| --- | --- | --- | ---: | ---: | --- |
| 2019-04-29 09:41:08.194 | `29af29e1-2c3a-4c69-9503-d8bbac05788a` | IIP radar/visual, small | 680.693 s | 980.693 m | `new_cell_-53_49` |
| 2020-06-27 09:48:24.120 | `504f95af-902a-56d6-8743-061e4b826d78` | IIP radar/visual, small | 683.382 s | 983.382 m | `new_cell_-56_53` |

The manifest supplies full native product names. These are annotation opportunities,
not SAR labels. The source-to-scene midpoint is not the native target time.
Later preparation must verify source uncertainty coverage at the actual target
time and retain enough native context; no pixels were opened for this reservation.
Both observations report small icebergs, which may be difficult to resolve;
reservation does not establish that either object is identifiable in SAR.

The refreshed local AIS snapshot contains **219,136 unique reports**, ending
**14 October 2026 at 21:28:39.934427 UTC**. A fresh public catalogue query and
footprint audit produced 20 temporal candidate slices, ten sparse-support slices
and 886 slices without observed overlap. Reception is incomplete, so those 886
are not verified empty scenes. The shortlist reuses the complete, corrected
historical 2014–2021 IIP audit and its original source hashes; those historical
catalogue snapshots were not refreshed. No account credentials were used for
public catalogue access and no source organization was contacted.

The stricter source screen yields **five AIS opportunities and eleven IIP
opportunities**. Within each stream, ranking uses observation time separation,
source uncertainty, AIS bracket width where relevant, product UUID and private
source identity. The vessel stream is evaluated first. This is a source-based
sampling rule, not a learned confidence score or statistical confidence interval.

Selection preserves every known acquisition and named geographic-group role,
rejects protected/quarantined geography within 10 km, requires the entire 6 km
reservation envelope inside the scene and NL study area, excludes exposed or
previously reserved datatakes/identities, and caps the batch at one area per native
acquisition and four per source stream. Genuinely new groups are assigned training
before image inspection. The two selected acquisitions have distinct geographic
groups. These development reservations create no new locked test.

**Zero vessel opportunities survive the reservation checks.** All five AIS
opportunities use Sentinel-1 C/D, outside the frozen A/B native input contract.
Several also have prior exposure or protected geographic assignments; these roles
must still be preserved if a later input recipe supports C/D. Among the eleven
IIP opportunities, one is excluded for prior datatake exposure/reservation, four
for protected/quarantine geography, and four for previously reserved identity. The two
remaining IIP opportunities are reserved. The shortfall is four vessel and two
iceberg opportunities relative to a batch cap, not a scientific sample-size target.
No training-role vessel examples, supported iceberg labels or balanced training
collection are claimed. Historical iceberg and modern AIS evidence also confound
era and class; a successful future fit will need that limitation evaluated.

## Preparation and replay

Private source identities, positions, full AIS competing-track pools, envelopes,
complete ranking inputs and blank annotation/dossier templates are retained in
`data/source_supported/training_v010_01/`. The public manifest pins their hashes;
public rankings omit AIS identities and positions. Copy a template to a fresh
versioned working directory before annotation: sealed templates must not be edited.
Inspect both bands and record all visible returns, uncertainty and alternatives
before viewing scores. Complete the dossier only after source and input checks.

```sh
python -m agents.provisional_training --verify
python -m agents.provisional_training --verify --local
python -m agents.provisional_training --assess data/path/to/completed_dossier.json
python -m pytest tests/unit/test_provisional_training.py -q
```

Public verification checks policy/source-code/release hashes and reservation
semantics without private data. Local verification checks original IIP files,
complete catalogue snapshots, every frozen AIS file, prior metadata and templates,
then reconstructs the ranking and selection exactly. Publication refuses existing
release or private output paths. Further evidence needs a new version; this release
neither changes the review guard nor resolves AllenAI permission/completeness.

At reservation time, the next task was preparing the two native windows; that
preparation is now documented above. Neither area currently supplies an admitted
training label. Additional vessels require fresh
source-supported training acquisitions; existing validation vessels retain their
roles. Vessel detection remains the primary goal. A vessel/clutter feasibility
study may proceed separately if iceberg class evidence remains insufficient,
provided its own labels, candidate population and comparison rules are documented.
