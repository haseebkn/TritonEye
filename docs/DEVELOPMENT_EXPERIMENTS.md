# Exploratory NL development experiments

This study uses the frozen v0.2.0 development assignments. Results are
**exploratory**. Independent label review is unavailable, and the locked test
set remains excluded. No performance improvement, selected operating threshold
or completed Step 5 is claimed.

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
```

CPU is supported but native detector inference can be much slower. Whole
development-source integrity checks precede compute telemetry. Single-run
timings include model loading, replay and serialization, and are not
full-swath or statistical latency estimates. RSS is sampled at 100ms; CUDA
memory is reported separately for the member phase and foundation phase.

Independent full-area review, sufficiently diverse labelled positives and
negatives, compatible HH/HV experiments and reviewed validation comparisons
are still required to complete Step 5 and select an operating threshold.
