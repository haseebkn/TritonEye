# Native NL HH/HV input-compatibility decision v0.1.0

**Decision: hold native transfer training and classifier-filter selection.**
No preprocessing recipe is approved by this report. The decision consumes the
same pinned evidence as the [label-admission register](LABEL_ADMISSION.md); its
machine-readable `compatibility` section contains the checks and supporting
training-only summaries. No new classifier scores or fitting were performed.

## What is established and what is unresolved

| Dimension | Evidence | Decision / consequence |
| --- | --- | --- |
| Polarization and layout | Native calibrated HH, then HV; unresampled 75×75 contexts where complete crops exist | Layout documented; dimensions alone do not establish physical equivalence |
| Nominal calibration | Existing exact pipeline replays; SNAP agrees in eight bands across four follow-up acquisitions within 0.001 dB, maximum error about 0.00000251 dB | Passes this numerical scope; does not prove physical calibration accuracy or equivalence to Kaggle |
| Physical scale / resolution | Native IW GRDH sampling is 10 m; Kaggle physical spacing, resolution and full preparation are unknown | Unresolved; do not resize solely to make apparent targets resemble Kaggle |
| Mode / incidence / conditions | Native annotation inventories record IW mode and incidence; follow-up centre incidence spans 32.64–39.67° | Geometry and condition equivalence unresolved; matching incidence does not establish matching resolution or sea state |
| Thermal-noise treatment | Native inputs uncorrected; separate SNAP sensitivity outputs show large HV changes and inferred low-power pile-up | Unresolved; retain original pixels and do not adopt corrected outputs automatically |
| Values / apparent extent | Reserved comparison uses 1,122 Kaggle training chips, 595 ships and 527 icebergs; native mixed returns differ in brightness and component extent | Descriptive warning only; no histogram matching or compatibility cut-off selected |
| Normalization | Existing frozen Kaggle training means/standard deviations support reproducible shadow inference | Retain for reproducing that study; no transfer-validity claim |
| Execution limits | One SNAP noise run took approximately 1,596 s despite a 600 s limit; the cause remains unresolved | Numerical pass and bounded execution are separate; do not describe every runtime check as passing |

The [reserved sixteen-area report](RESERVED_HH_HV_TRANSFER.md),
[nine-window cross-check](SOURCE_ERROR_PREPROCESSING.md),
[six-window preparation](FOLLOWUP_SOURCE_ANNOTATION.md) and
[SNAP audit](SNAP_CORROBORATION.md) retain the detailed metadata, historical
dates, values, failed attempts and original hashes. The SNAP result applies to
four development windows; it is not a universal reference test for all 31 areas.
Pixel spacing is distinct from effective spatial resolution, and apparent bright
component extent is distinct from hull or iceberg size.

## Noise findings that constrain the next recipe

| Follow-up candidate suffix | Median HV change after noise removal | Context HV pixels near inferred power 1e-5 |
| --- | ---: | ---: |
| 01 | −6.32 dB | 30.25% |
| 02 | −74.61 dB | 50.71% |
| 03 | −72.92 dB | 52.76% |
| 04 | −5.08 dB | 21.99% |

The power diagnostic reconstructs `DN² × 10^((noise_dB − nominal_dB)/10)` because
the paired calibration LUT cancels. These are whole-context descriptive results;
they are not object-class measurements or proof of correct physical subtraction.
Finite output values alone do not resolve the issue. The [sealed diagnostic](evidence/snap_noise_floor_20261014.json)
retains counts, quantiles and the post-run interpretation. No inferred floor mask
was added to the images or classifier inputs.

## Conditions for the next experimental decision

First investigate the low surviving HV power against the original per-product
noise metadata and target neighbourhoods, retaining nominal/corrected outputs and
all failure cases. Establish publisher preparation and physical scale where that
evidence is obtainable. If it remains unknown, explicitly define a separate native
NL experiment; do not claim that Kaggle equivalence has been recovered.

Choose any experimental noise/scale/normalization adaptation using training
acquisitions alone and record its assumptions and compute settings. Freeze the
recipe before validation. Current candidates include only one training-role
plausible iceberg and no new training-role vessel, all unapproved. That is
insufficient support for a class-balanced native transfer comparison. Validation
images already inspected during diagnosis retain that exposure history and cannot
be used to claim an untouched test.

After sufficiently supported provisional labels and an explicit input recipe
exist, compare the frozen HH/HV candidate population with and without classifier
filtering. Track vessel losses, supported non-vessel removals, unknown rejections
and abstentions separately, across held-out development acquisitions. Unknown
rejections are not measured false-positive reductions. Independent-review and
promotion guards, AllenAI quarantine and the original locked test remain intact.

Run `python -m agents.label_admission --verify --local` for the retained-data
replay, or omit `--local` for public evidence verification. No SNAP rerun, new
data download or account credential is needed for this decision report.
