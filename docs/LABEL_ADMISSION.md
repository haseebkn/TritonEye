# NL label-admission register v0.1.0

The [versioned register](evidence/label_admission_v010.json) accounts for **31
reserved native HH/HV areas and 58 annotations or context samples**. It admits
**zero training labels, zero accuracy-truth objects and zero verified negatives**.
This is an evidence-based hold decision, not a new annotation pass. All original
labels, split roles, ignore masks and source bytes remain unchanged.

The [input-compatibility decision report](INPUT_COMPATIBILITY.md) explains why
successful calibration alone does not permit transfer training or filter selection.
The register is a reporting consumer, not a training loader or promotion override.

## Scope and disposition

| Existing collection | Areas | Annotations / contexts | Current interpretations |
| --- | ---: | ---: | --- |
| Reserved HH/HV v0.1.0 | 16 | 41 | 25 unresolved returns, 8 structure contexts, 8 clutter contexts |
| Source-supported v0.1.0 | 9 | 11 | 2 provisional vessels, 7 unresolved returns, 2 clutter contexts |
| Source-supported v0.2.0 | 6 | 6 | 4 plausible icebergs, 2 unresolved returns |
| Total | 31 | 58 | 34 unresolved, 18 contexts, 6 source-supported interpretations |

Each register row preserves collection/hash, area and object identity, product,
original split, available acquisition/geographic group, box, interpretation,
annotation reason, source consistency, uniqueness in the available source pool,
coverage abstention and decision blockers. Missing public metadata is null or
points to a hash-bound local report; it is not inferred to be verified.
The area inventory includes the original preparation references and explicitly
records that no whole area is a verified negative or exhaustively annotated.

Scope is the three additional reserved collections. The older eight HH/HV areas
already scored in the [shadow study](STATOIL_TRANSFER.md), the VV/VH benchmark,
Kaggle chips and quarantined AllenAI annotations are outside this register.
Counts here are not totals for every label ever created in the project. The original
locked test set is neither opened nor reassigned by this work.

## Six positive interpretations still on hold

| Area / object | Interpretation | Frozen role | Evidence and unresolved issue |
| --- | --- | --- | --- |
| `vessel_support_01/u1` | Provisional vessel | Validation | Compact two-band return and one compatible moving AIS identity; reception and SAR/AIS errors are incomplete |
| `vessel_support_03/u1` | Provisional vessel | Validation | Elongated return and one compatible moving AIS identity; no independently verified identity |
| `next_iceberg_support_01/u1` | Plausible iceberg | Validation | IIP-compatible return, 294 m separation within 780 m allowance; no new identity confirmation |
| `next_iceberg_support_02/u1` | Plausible iceberg | Train | 450 m separation within 1,453 m allowance; long time gap and alternative vessel/clutter explanations |
| `next_iceberg_support_03/u1` | Plausible iceberg | Validation | 608 m separation within 1,493 m allowance; earlier same-publisher report is correlated context |
| `next_iceberg_support_04/u1` | Plausible iceberg | Validation | 988 m separation within 1,587 m allowance; source error and motion remain unresolved |

Rounded distances above retain the original gates. The [SNAP/corroboration audit](SNAP_CORROBORATION.md)
found no covering Sentinel-2 L1C product in its bounded searches and no additional
independent identity confirmation. A unique match in an incomplete source pool
cannot rule out another vessel or reflector. Four complete uncertainty-coverage
scenarios are not four proven identities. Follow-up windows 05 and 06 retain their
incomplete-coverage abstentions and unresolved classes.

The sole training-role plausible iceberg is not admitted for fitting. Both new
vessel interpretations are validation examples and cannot be moved into training
to balance classes. Every row remains provisional and ignored for accuracy;
context annotations cannot become binary non-vessel or iceberg labels. A missing
AIS message does not support an iceberg class.

## Admission rules and future decisions

The [pinned policy](../configs/reservations/label_admission_v010.json) states the
current hold decision and requirements for a future version:

1. Preserve original annotation history, masks and frozen splits. Record class
   evidence, competing returns, source uncertainty and prior exposure before
   viewing model scores. Ambiguous objects remain ignored.
2. Resolve or explicitly bound the input recipe for the intended experiment.
   Freeze calibration/noise, scale, cropping and normalization choices before
   comparing models. A recipe cannot be selected using held-out outcomes.
3. Preserve acquisition and geographic separation and group repeated physical
   objects together. Missing public group metadata must be resolved before any
   new training export; this report does not certify a new split.
4. With no independent reviewer, a future authorized weak-label experiment can
   remain exploratory. It needs its own version, admission decisions and label
   uncertainty analysis; this register grants no training permission. Human or AI
   self-review cannot satisfy the existing independent-review/promotion guard.

For vessels, improve co-temporal track support and account for competing vessels.
For icebergs, seek tighter source timing/position evidence and additional
corroboration that identifies the SAR reflector. Increasing the number of repeated
AI inspections does not supply the missing evidence. Report disagreement and
abstention rather than allowing confident model scores to determine the label.

The annotator had previously viewed other project labels and model outputs.
These three collections were curated before their own classifier scoring, with
source-centred selection disclosed. This later synthesis is neither independent
review nor a new blind pass. No new model scores, fitting or candidate filtering
were performed. Original source licences/provenance remain in the linked
[source audit](SOURCE_SUPPORTED_ANNOTATION.md) and [reserved collection](RESERVED_HH_HV_TRANSFER.md).

## Replay and integrity

```sh
python -m agents.label_admission --verify
python -m agents.label_admission --verify --local
python -m pytest tests/unit/test_label_admission.py -q
```

Public verification hash-checks the pinned collections, SNAP/noise evidence and
Kaggle baseline ledger, recomputes every decision and requires an exact match to
the released artifact. It needs no private imagery, classifier or network.
Local verification additionally invokes existing collection/source/SNAP replay
checks on retained metadata, pixels, ignore masks and receipts. The command prints
verification results and writes no source data. Fresh publication refuses existing
release/lock paths; later evidence requires a new version.

The [release lock](../configs/reservations/label_admission_v010.lock.json) pins the
report. Original benchmark/review code, all datasets and reservations, quarantine,
locked test and active model settings are unchanged.
