# Local model lifecycle

TritonEye can register model artifacts, run paired shadow inference, promote a
model for local research and restore a previous selection. Promotion requires
independently reviewed validation evidence and the existing operating constraints.
The current NL benchmark remains provisional, so it cannot authorize promotion.

This implements roadmap step 7's small lifecycle. The scope is a local research
workflow, with a persistent SQLite journal and explicit operator decisions.
Production `agents.pipeline`, `configs/model.yaml`, MLflow registry aliases and
operational alerts are unchanged. Cloud rollout, automatic retraining, continuous
production monitoring and service commissioning remain future work.

## Model registration and selection

`agents.model_lifecycle` stores immutable, content-addressed package and shadow
records in `state/model_lifecycle.sqlite3`, which is excluded from Git. Model
weights remain at their original local paths; registration records their SHA-256
and checks it whenever the package is selected. Moving or changing a file requires
a new package. Model bytes, SAR imagery and credentials are not published.

Registration requires an actor, source, licence declaration, experiment identifier,
code commit and the exact weights hash. An exploratory package also requires a
training dataset release hash. Known quarantined AllenAI releases are refused as
training lineage. These declarations record provenance; they do not independently
verify licence entitlement, model compatibility or the operator's identity.

The implemented adapter accepts the xView3 TorchScript interface and native VV/VH
inputs. A compatible model must retain the existing calibration, normalization,
output keys and decoding. CROMA token heads and HH/HV have no deployment adapter.
A registered incompatible artifact fails during shadow inference.

Example metadata for the existing reference weights:

```json
{
  "adapter": "xview3",
  "polarizations": ["vv", "vh"],
  "status": "reference",
  "source": "BloodAxe xView3 reference ensemble",
  "licence": "MIT; retain upstream attribution",
  "experiment": "pretrained reference",
  "code_commit": "REPLACE_WITH_ACTUAL_COMMIT",
  "weights_sha256": "c21d6b4204e2803b56014c9780d8d62190ed81d9c2ec56d0cbf437b439307d38"
}
```

Save this locally, replace the commit, and run:

```sh
python -m agents.model_lifecycle register --weights models/xview3/traced_ensemble.jit --metadata state/reference.json --actor "operator name"
python -m agents.model_lifecycle reference --model MODEL_ID --actor "operator name" --reason "Initialize the existing pretrained research reference"
python -m agents.model_lifecycle resolve
```

Initialization records an **unvalidated reference**. It sets no approved champion
or selected operating threshold. The reference uses the existing provisional 0.15
threshold for the subsequent comparison. Initialization cannot replace an existing
reference. Use a separate `--store` for an isolated rehearsal.

## Shadow evaluation

Register a candidate with status `exploratory` and its actual training release
hash, then execute:

```sh
python -m agents.model_shadow --candidate CANDIDATE_ID --dataset datasets/nl_benchmark/v0.2.0 --output state/shadow/run_001 --device cuda
```

Each run resolves the current research reference or champion and executes both
models on the same complete selection of validation areas. Outputs go to a fresh
directory. Training and locked-test areas are excluded from inference. The shared
development scene and context guards reject held-out products, datatakes and
geography before inference. Development source files and chips are verified;
locked test pixels and the mixed review bundle are not opened.

All selected validation areas stay in coverage accounting. Unsupported HH/HV,
missing assets and failed processing cannot disappear from the denominator.
Predictions, scene identity, ROI geometry, model hashes, code/configuration hashes,
dataset release, coastline files, processing state and sampled resources are
recorded. Model load time is recorded separately from inference time. RSS is
sampled absolute process memory; timing includes bounded ROI inference rather
than full-scene commissioning performance.

The shadow evaluator obtains labels only through the existing
`validation_labels` review guard. It recomputes metrics from the hashed prediction
files and preserves the distinction between processed and measured. Missing review
leaves precision, recall and false-alarm measurements unavailable. Counts of
returns and AIS proximity never supply approval evidence.

## Promotion criteria

The operator requests promotion using the recorded shadow identifier:

```sh
python -m agents.model_lifecycle promote --shadow SHADOW_ID --actor "operator name" --reason "Document the reviewed decision"
```

Promotion rechecks artifact integrity, dataset review status and current reference
revision, then recomputes the comparison. A previously saved success flag or
edited report cannot authorize it. Code, configuration, weights, source imagery,
release or coastline changes require a new valid comparison.

The fixed threshold grid and existing [provisional operating objective](NL_BASELINE.md#provisional-operating-objective)
apply: false-alarm upper bound at most 0.1 per water km², recall lower bound at
least 70%, processing and measured water coverage at least 90% in each required
region, coastal setting and polarization stratum, at least 30 reviewed vessels
and at least five independent acquisition/geography blocks. These are research
targets, not C-CORE requirements.

At the selected candidate threshold, precision and recall must not decrease and
false-alarm density must not increase relative to the current reference's recorded
threshold, initially 0.15. Its reviewed metrics must be available. Candidate
runtime and sampled RSS must remain within a provisional **2×** budget in each
paired processed area, and it must not lose a reference's successfully processed
area. Timing noise can block a run; investigate and repeat the whole comparison
rather than edit the measurements.

Passing allows a manual local research promotion. It does not establish a
statistically significant improvement, province-wide performance or operational
certification. Review qualifications are recorded declarations, as in the existing
benchmark contract; repository owners can intentionally edit state and releases.

SQLite commits the new model, threshold, revision and decision journal together.
Concurrent or stale decisions fail without changing the selection. `resolve`
returns the selected package and threshold, and the next shadow run consumes that
selection as its reference. No environment variable or production configuration
is silently rewritten.

## Rollback and incident history

Inspect the journal and restore a previous reference or promoted package:

```sh
python -m agents.model_lifecycle status
python -m agents.model_lifecycle rollback --model PREVIOUS_MODEL_ID --expected-revision CURRENT_REVISION --actor "operator name" --reason "Describe the incident and restoration decision"
```

Rollback verifies the target weights and restores its previously recorded
threshold without retuning. A target never previously selected, missing or changed
weights, a stale revision or a request to restore the current model is refused.
The restoration and incident reason append another journal entry; prior records
are retained. Restoring the original reference also restores its unvalidated
threshold status. Rollback authorizes recovery of local research state, not a new
claim of detector accuracy.

## Verification and current limits

The [12 October real-data replay](evidence/model_lifecycle_20261012.json) compared
the pinned reference against itself. Each branch retained all ten validation
areas, processed six VV/VH areas, recorded four HH/HV areas as unsupported and
had zero processing failures. The paired prediction files were identical.
Processing covered 62.23% of selected validation water; independently measured
coverage was 0%. Each branch produced 12 returns above the provisional 0.15
threshold. These are descriptive returns, not confirmed vessels or false alarms.

The actual promotion request was refused, with the reference revision and journal
unchanged. No threshold was selected. Bounded inference took 122.26 seconds in
the reference branch and 99.42 seconds in the second branch; warm-up, execution
order and system load prevent interpreting that difference as model improvement.
The isolated store and full report remain under `state/`, outside Git.

[Lifecycle regressions](../tests/unit/test_model_lifecycle.py) execute registration,
paired shadow processing, real release/review validation, real chip hash and
geolocation checks, metric computation, threshold selection, promotion, restart
and rollback with synthetic imagery and invented review declarations. Successful
promotion in those tests proves the state machine; it adds no NL measurement or
independent review. Failure cases exercise unreviewed labels, changed predictions,
weights and releases, stale state, failed comparisons and quarantined lineage.

The public NL v0.2.0 metadata is also checked through the lifecycle's validation
scope: its labels remain unreviewed and the test boundary remains enforced.
No real candidate has been approved or deployed through this lifecycle. The
production operating threshold stays provisional; model improvement, identity
correctness and low false-alarm performance remain unmeasured.
