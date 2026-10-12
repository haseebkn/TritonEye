# Exploratory NL development experiments

This study uses the frozen v0.2.0 development assignments. Results are
**exploratory**. Independent label review is unavailable, and the locked test
set remains excluded. No performance improvement, selected operating threshold
or completed Step 5 is claimed.

## Executed results — 12 October 2026 UTC

The successful local run is
`data/model_research/development_20261012_v020_03/`, executing revision
`971bec6`. [Public experiment evidence](evidence/development_20261012_v020.json)
retains the complete settings, six fits, loss histories, per-area score trials,
review rejections and provenance. The frozen release SHA256 is
`29ab2db4b0ec8d969db4200c3475dd67699bc2de1f7244d66643737bc05013c5`.

One pretrained member control and six fitted runs produced **77 area/model
records across eleven distinct VV/VH development areas**: five train and six
validation. Eight HH/HV development areas remained unsupported; two locked
test areas were excluded. These are repeated experiments on the same areas,
not 77 independent acquisitions. All ten validation truth exports remained
blocked by the scientific review guard. No test images or combined review HTML
were opened by this study, and frozen dataset files were unchanged.

The bias-only fits reduced recorded training loss from **1.714714 to 1.666663**.
All three seeds produced the same parameter digest and bias increase,
**0.0599117279**. This objective has no stochastic initialization, so these
identical runs provide no independent uncertainty estimate. Native replay
probabilities actually increased: the
[saved-artifact verifier](evidence/development_verification_20261012_v020.json)
checks every retained control peak against its expected sigmoid/logit shift.
It also binds all 77 prediction files, six parameter artifacts and eleven
frozen feature archives to their hashes and checks the published return counts.

CROMA encoded **176 windows**, covering all eleven selected chips. Its encoder
state digest stayed unchanged. The 769-parameter heads reduced their recorded
training losses as follows:

| Seed | First recorded training loss | Last recorded training loss |
| --- | ---: | ---: |
| 20261012 | 0.569503 | 0.000362 |
| 20261013 | 0.316859 | 0.000231 |
| 20261014 | 0.897621 | 0.000485 |

At strict score **>0.15**, across the six processed validation areas and
111.283 km² of reference-relative water:

| Approach | Raw returns | Open-water eligible | Coastal withheld | Reference land |
| --- | ---: | ---: | ---: | ---: |
| Pretrained member zero | 14 | 2 | 7 | 5 |
| Bias-fitted member zero, each seed | 14 | 2 | 7 | 5 |
| Frozen CROMA head, seed 20261012 | 2,601 | 1,671 | 174 | 756 |
| Frozen CROMA head, seed 20261013 | 2,487 | 1,585 | 143 | 759 |
| Frozen CROMA head, seed 20261014 | 2,574 | 1,639 | 154 | 781 |

These are **descriptive returns, not false alarms or confirmed vessels**.
The bias fit changed scores but did not change counts at this threshold.
CROMA's very low one-example training loss accompanied a large candidate load;
its positive-only head lacks evidence of useful background rejection. Scores
are not calibrated between architectures, so these counts cannot rank accuracy
or estimate false-alarm improvement. Neither experimental fit is promoted.
The earlier production ensemble returned twelve validation detections at this
threshold; comparing that different ensemble with this member is not a
fine-tuning effect.

The measured compute block took **189.79 seconds**, including model loading,
all 77 area/model passes, fitting, feature extraction and serialization. Initial
source integrity checks and shoreline-mask preparation are outside that block.
Sampled process RSS peaked at **2.31 GiB**. The member phase took 166.03 seconds
and peaked at **5.58 GiB CUDA allocation**; the separately recorded CROMA feature
phase took 14.48 seconds and peaked near **230.08 MiB**. This is one execution
on an RTX 4070 Laptop GPU, Python 3.13.7 and Torch 2.6.0+cu124; it provides no
runtime uncertainty or full-swath throughput estimate.

Two earlier execution attempts are retained:

- [Attempt 1](evidence/development_failure_20261012_01.json) stopped before
  detector inference because TorchScript does not support module-level
  `requires_grad_`. Parameter-level freezing fixed it; an actual ScriptModule
  regression verifies freezing and preserved inference.
- [Attempt 2](evidence/development_failure_20261012_02.json) was stopped before
  training after the individual member tuple was incorrectly interpreted in
  ensemble order. Its partial fishing-head predictions are invalid and excluded.
  A runtime head-identity regression now exercises the corrected mapping.

These failures are software execution history, not accuracy comparisons.
Current evidence demonstrates limited gradient training and reproducible
foundation-head experimentation. Completing Step 5 still requires reviewed,
diverse training/validation data with legitimate negatives, meaningful detector
fine-tuning and an operating-point comparison meeting the Step 4 requirements.

Local verification: **548 tests passed**, two heavy tests deselected. Ruff,
Black and strict Mypy passed across 97 source/test files. Scientific independent
review remains unavailable; software verification does not supply image truth.

## Protocol declared before execution

The available VV/VH training supervision consists of one provisional vessel
interpretation in `scene_01_area_02`, from September 22, 2026. Its image-box
centre supplies one positive training location. The HH/HV training vessel is
incompatible with these models. Unresolved objects, contextual clutter and
unlabelled pixels receive no training loss. No AIS track is converted into
image truth, and validation annotations never supervise training.

1. Control: member zero of the pinned xView3 checkpoint, before fitting. It uses
   the existing production normalization, native 2048px contexts, offset
   decoding, cross-context deduplication and selected-area clipping. The
   12-member production ensemble with flip augmentation is a different model;
   its [Step 4 replay](NL_BASELINE_V020.md) remains a separate reference.
2. Fine-tune: update **one existing final objectness bias** in that same member.
   Cached centre logits supply the exact positive-only bias objective; the
   actual checkpoint parameter changes, and each fitted member is inferred
   again. All backbone, classifier, offset and other head parameters are
   frozen. This tests limited head fitting, rather than a full detector recipe.
3. Foundation approach: extract the pinned CROMA radar features for the same
   eleven VV/VH development areas, then fit a 769-parameter linear token head.
   The radar encoder remains frozen. Select the 8px token containing the
   provisional training centre; other tokens receive no loss. Infer all 176
   disjoint 128px windows across the eleven 512px selected areas. Decode local
   score maxima on the 8px token grid, without subpixel offset refinement.

Each fitted approach uses seeds `20261012`, `20261013`, `20261014`, Adam,
learning rate 0.001, zero weight decay, 60 updates, float32 and no augmentation.
Use the final update; there is no validation-based model or epoch selection.
The feature extractor retains its separately recorded seed `20261006`.
All models use the fixed strict score grid 0.05, 0.10, 0.15, 0.25, 0.35, 0.50,
0.70, 0.90, 0.99, a 300m coastal policy and reference-relative OSM land/water.
Infrastructure flags are disabled consistently for this comparison. The
reviewed performance protocol retains its 100m matching rule and provisional
false-alarm, recall and coverage targets; it cannot be evaluated here.

These architectures have different context and output resolution. Identical
raw score thresholds do not give equivalent false-alarm operating points.
Three random seeds do not enlarge a one-vessel training sample or establish
statistical uncertainty. All scored development areas are Newfoundland;
Labrador's HH/HV areas remain unsupported. No HH/HV channels are renamed.

## Expected interpretation and safeguards

Positive-only BCE rewards increasing the score of the single training example.
It cannot teach rejection of background, fixed objects, ice or clutter.
Decreasing training loss or increasing candidate counts therefore provides no
evidence of improved vessel detection. The report retains every loss history,
parameter digests, checkpoint hashes, descriptive returns by area/policy/score,
resource telemetry, code/configuration/source identities and validation-export
rejections. Precision, recall, false-alarm rates and improvement remain null.

The public report omits image arrays, prediction coordinates and training pixel
coordinates. Local immutable experiment directories contain model parameter
patches, CROMA heads/features and GeoJSON outputs. To apply a member patch,
load the pinned checkpoint, select `model.models._modules['0']`, and copy the
saved tensor into its named `model.head.objectness.final.bias`; never substitute
the member patch for the entire ensemble. Production configuration is unchanged.
Encoder adaptation is deferred until a frozen-head comparison justifies it.

Failures retain `failure.json` and any partial artifacts. A fresh run directory
is required for each attempt. Neither a blocked fit nor an execution failure is
an unsuccessful accuracy comparison. Data must stay in the original project;
validation worktrees use synthetic fixtures and public metadata only.

## Reproduction

Use the existing calibrated development data and pinned local model assets:

```sh
python -m agents.development_experiments --dataset datasets/nl_benchmark/v0.2.0 --output data/model_research/new_development_run --device cuda --publish docs/evidence/new_development_run.json
python -m agents.development_evidence --experiment data/model_research/new_development_run --output docs/evidence/new_development_verification.json
```

CPU is supported but native detector inference can be much slower. Whole
development-source integrity checks precede compute telemetry. Single-run
timings include model loading, replay and serialization, and are not
full-swath or statistical latency estimates. RSS is sampled at 100ms; CUDA
memory is reported separately for the member phase and foundation phase.

Independent full-area review, sufficiently diverse labelled positives and
negatives, compatible HH/HV experiments and reviewed validation comparisons
are still required to complete Step 5 and select an operating threshold.
