# Statoil ship and iceberg classification baseline

This experiment uses the user-downloaded C-CORE competition training archive
to classify a supplied radar target as a ship or iceberg. Vessel detection and
SAR–AIS correlation remain TritonEye's primary scope. Chip classification supplies
a separate supervised development task; it cannot establish full-scene vessel
recall, false alarms per water area or correctly associated AIS identities.

The subsequent [real NL HH/HV compatibility audit and frozen shadow study](STATOIL_TRANSFER.md)
tests native input transfer on development imagery. Its descriptive scores do not
extend the classification holdout or establish vessel detection improvement.

The [official data description](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data)
specifies expert-supplied labels, HH then HV, two 75 by 75 float-dB bands,
and missing incidence angles. C-CORE's authors describe collection along
[Newfoundland and Labrador](https://www.mathworks.com/company/technical-articles/analyzing-satellite-radar-imagery-with-deep-learning.html).
Individual chip locations, dates and parent acquisitions are absent from the
downloaded schema, so regional provenance cannot be independently checked per chip.

## Access and data handling

The local file is
`statoil-iceberg-classifier-challenge/train.json/data/processed/train.json`.
The [competition rules](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/rules)
allow research, education and other non-commercial uses after acceptance, subject
to their conditions. They restrict disclosure to people who have not accepted the
rules. The archive was supplied by the owner; acceptance was not independently
verified. Original chips, identifiers, predictions and model parameters stay in
ignored local storage. Public evidence contains aggregate counts, metrics and hashes.

Only the labelled training archive enters this experiment. The competition
`test.json` includes machine-generated images and has no supplied labels; it is
excluded before file opening. The original NL detection benchmark's test set,
review guard, AllenAI quarantine and production operating configuration remain
outside this study.

## Audit and fixed protocol

The source audit checks the schema, duplicate IDs, both classes, native band
dimensions, finite values and valid incidence angles. Float32 band quantiles
describe the actual supplied values. Publisher dB is not treated as original
native sigma0 calibration: the original preprocessing, pixel spacing, noise
corrections and geolocation are unavailable. No guessed DN conversion,
replacement polarizations or xView3 normalization is applied.

Exact image duplicates use float32 content hashes. Near-duplicate screening uses
Gaussian smoothing and rotation-equivariant 3 by 3 block means, then all eight
rotation/reflection transforms. A pair is grouped when correlation is at least
0.995 and native two-band RMSE is at most 0.5 dB. HH/HV are never swapped.
This bounded definition can miss shifted, resampled or related crops; a zero
count would not certify complete independence. Conflicting duplicate labels
stop the study for review.

Equal incidence angles rounded to four decimals share a group; all missing-angle
chips share one conservative group. Duplicate and angle connections are joined
transitively. A single seeded, group-stratified allocation approximates 70% training,
15% validation and 15% local holdout. No group crosses partitions. This is a
proxy for related observations; unknown acquisitions may still span groups.
The [published competition solution](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/writeups/beluga-2nd-place-solution-overview)
identifies angle-related leakage as a reason for grouped evaluation.

The split seed is 20261013. It is frozen locally before fitting. Incidence angles,
missing-angle flags and identifiers are excluded from every model. The models are:

- A training-class-prior reference.
- Logistic regression on each band's mean, standard deviation and 10th, 50th
  and 90th percentiles. Feature means/scales use training rows only. Validation
  log loss selects L2 penalty from 0.001, 0.01, 0.1 and 1.0.
- Three small CNNs trained from scratch with seeds 20261013, 20261014 and
  20261015. They use two convolutions, 12 and 24 channels, pooling, a 32-unit
  dense layer and binary output. Adam uses learning rate 0.001, weight decay
  0.001 and batch size 64. Each runs at most 40 epochs; its checkpoint minimizes
  validation log loss. The ensemble averages three probabilities.

CNN inputs preserve 75 by 75 HH/HV and use training-pixel band means/scales.
There is no resizing, per-chip normalization, augmentation or pretrained model.
Validation log loss chooses between the statistics model and CNN ensemble before
holdout inference. A classification threshold of 0.5 is prespecified; it is not
an operational detection threshold.

Each run evaluates the holdout after choices are saved. Reports include log loss,
accuracy, confusion, ship recall, iceberg recall and both error directions.
Exact 95% recall intervals assume independent chips and are conditional on the
publisher labels. Paired 1000-replicate resampling of whole proxy groups reports
the CNN-minus-statistics differences, with actual acquisition dependence still
unknown. Training histories and every seed's result are retained. Further tuning
consumes this exploratory holdout rather than creating new unseen evidence.

## Observed training data and classification results

The 13 October audit found 1,604 labelled chips: 851 ships and 753 icebergs.
Every band has 5,625 finite values; IDs are unique. There were zero exact image
duplicates and zero near pairs under the fixed screening rule. The 133 missing
angles all belong to ships. Of 943 chips in repeated-angle groups, 926 belong to
groups containing only one class. These are substantial metadata confounds;
angle-only random-row evaluation would be especially vulnerable to leakage.

There are 879 connected proxy groups; the largest contains the 133 missing-angle
chips and is assigned to training. The frozen split contains:

| Partition | Chips | Ships | Icebergs | Proxy groups |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,122 | 595 | 527 | 608 |
| Validation | 241 | 128 | 113 | 135 |
| Exploratory holdout | 241 | 128 | 113 | 136 |

HH minimum/median/maximum are −45.594/−21.060/+34.575 dB; HV values are
−45.655/−26.345/+20.154 dB. Positive and extreme values are retained as supplied;
they do not establish that the chips are calibrated like native Sentinel sigma0.

The CPU experiment selected logistic penalty 0.001 and CNN checkpoints at
epochs 33, 34 and 36. Validation log loss was 0.5914 for statistics and 0.2091
for the CNN ensemble, selecting the ensemble before holdout scoring.

| Model | Holdout log loss | Accuracy | Ships called icebergs | Icebergs called ships |
| --- | ---: | ---: | ---: | ---: |
| Training class prior | 0.6912 | 53.11% | 0 / 128 | 113 / 113 |
| Logistic band statistics | 0.6405 | 58.92% | 60 / 128 | 39 / 113 |
| Three CNN ensemble | 0.2855 | 87.97% | 17 / 128 | 12 / 113 |

The ensemble's ship recall is 111/128 = 86.72%, with conditional exact 95%
interval 79.59%–92.07%; iceberg recall is 101/113 = 89.38%, interval
82.18%–94.39%. Individual CNN accuracies are 87.14%, 89.21% and 86.72%.
The second seed's higher holdout accuracy was not used to replace the
validation-selected ensemble.

Paired proxy-group bootstrap gives an ensemble-minus-statistics accuracy
difference of +29.05 percentage points, conditional 95% interval +19.80 to
+37.85 points. Log-loss difference is −0.3550, interval −0.4545 to −0.2338.
These support improvement on this exploratory publisher-labelled chip task.
Unknown scene dependence, upstream processing and label error limit the inference.
Twelve of 113 icebergs still received a ship classification, and 17 of 128 ships
were rejected: this is not an approved suppression component or an operational
false-alarm result. Performance on native detector crops remains untested.

The first execution arose from a test-isolation mistake: a guard test's
project-local temporary output passed the runner's valid-output rule. Three
fixed-protocol executions completed before the fixture was corrected; their
partitions, selected epochs and metrics agree exactly. No model settings were
changed in response to their holdout results. The first run was preserved at
`data/statoil_baseline/20261013_cpu_01/`; the original test-generated outputs
remain local. These replays are one consumed sample, not three independent
measurements. The corrected tests use generated fixtures only.

The [aggregate ledger](evidence/statoil_baseline_20261013.json) binds the
restricted source, split, parameters, predictions and full report. The full
report SHA-256 is
`39b79185ab6e30d998385e0cdd0d983e8785dd89a97f110de524a92df7a8a521`.
Execution used uncommitted new research code while the checkout HEAD was
`4d973e7`; the ledger's executing-file hash identifies the actual implementation,
not code already present at that older commit. PyTorch 2.6.0+cpu, Python 3.13.7
and NumPy 2.2.6 were used on an Intel Family 6 Model 191 CPU with four Torch
threads. The preserved run took 80.91 seconds. Timing excludes environment
installation and is not an operational latency bound.

## Reproduction

Use CPU-only PyTorch 2.6.0, project dependencies and authorized local training
data, and choose a fresh path. The isolated Windows interpreter for this run is
`data/statoil_runtime/Scripts/python.exe`; activating that environment makes
`python` resolve to it. A CPU device flag alone does not prevent a CUDA PyTorch
installation from loading large CUDA libraries.

```sh
python -m agents.statoil_baseline --output data/statoil_baseline/NEW_run --device cpu --epochs 40
python -m agents.statoil_evidence --report data/statoil_baseline/NEW_run/report.json --output docs/evidence/NEW_statoil.json
python -m pytest tests/unit/test_statoil_baseline.py -q
```

The run writes source and code hashes, software versions, compute identity,
partitions, normalization, fitted parameters, histories and predictions under
ignored `data/`. Existing output directories are refused. File integrity and
behavioral tests establish software reproducibility, not label accuracy or
operational transfer.

The CPU run requires at least 3 GiB system commit headroom, 2 GiB available
physical memory and 1 GiB free project/temporary storage before loading data.
CNN training stops if process RSS exceeds 2 GiB or system commit headroom falls
below 1 GiB. These checks reduce resource pressure; they cannot guarantee system
stability when other applications allocate memory between checks.

Windows initially returned low-memory and temporary-storage errors loading the
system CUDA installation. A separate CPU-only virtual environment was installed
under ignored `data/statoil_runtime/`, leaving the system CUDA installation intact.
The machine subsequently restarted; no crash log established causation. The
bounded CPU executions above ran after restart. Resource and test-isolation
regressions passed with the CPU runtime.
