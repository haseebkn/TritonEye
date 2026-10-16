# TritonEye technical explanation and contributions

Use these notes to explain the engineering decisions and the limits of the
evidence. They describe repository behavior; they do not certify the owner's
personal implementation history or understanding. Practice the explanations
against the linked code and saved artifacts before using them in an interview.

## Project introduction

TritonEye is an independent Newfoundland and Labrador maritime AI research
prototype. It processes Sentinel-1 VV/VH, runs a pretrained vessel detector,
classifies returns relative to a coastline, and associates eligible targets with
co-temporal AIS using bounded motion and one-to-one assignment. It records
processing failures, uncertainty and provenance, produces analyst reports, and
provides guarded local model promotion, paired shadow inference and rollback.

Its strongest evidence is reproducible SAR/geospatial engineering. Regional
precision, recall, false-alarm density and useful model improvement remain
unmeasured. This is independent research, not an endorsed operational system.

## Why the harbour diagnosis changed

The earlier diagnosis said the coastline closed St. John's harbour. It relied
on manually placed coordinates that were inadequate evidence for that claim.
Comparison with municipal imagery and recorded positions showed the basin and
The Narrows were water. The mechanism was the coastal eligibility policy:
physical water within 300m of shore was withheld from open-water candidacy.
Berthed AIS positions can also lie just inside a reference quay polygon.

The correction was to retain the open harbour, verify sparse imagery controls,
and separate physical land/water classification from operating eligibility.
Coastal detections now remain available for research. CanVec stayed a comparison
source because it missed a quay control; no province-wide replacement was
justified. The [shoreline report](ST_JOHNS_SHORELINE.md) and
[coastal policy](COASTAL_POLICY.md) retain provenance and limitations.

Be able to explain signed shore distance, the source imagery's age, sparse
control coverage, and why an AIS-supported exclusion count cannot establish
false alarms or all missed vessels.

## Why AIS proximity is incomplete evidence

AIS is partial, self-reported supporting data. Reception gaps, stale positions,
receipt versus onboard-fix timing, vessel motion and competing nearby vessels
can affect a match. SAR geolocation and moving-target displacement add further
uncertainty. A nearby SAR return may be a different vessel or another reflector;
an unmatched return may reflect missing AIS coverage.

The 3/3 observation across two physical VV/VH scenes at 500m has a conditional
exact 95% interval of 29.2%–100%. It measures a selected AIS proximity sample.
It does not establish three independently confirmed identities, total vessel
recall, precision or low false positives. Its interval also assumes independent
trials and does not correct scene clustering or selection bias.
See [the result](AIS_SUBSET_RESULT.md) and
[association protocol](TARGET_CORRELATION.md).

Explain one-to-one assignment and abstention, why zero flagged ambiguities is
not zero wrong identities, and how denser AIS plus independently reviewed
matching cases would support a held-out comparison.

## Why the CROMA heads produced thousands of returns

Three frozen-feature heads trained on **one provisional VV/VH positive**.
Unresolved and unlabelled pixels had no training loss; they supplied no legitimate
negative supervision. The loss could decrease by raising positive scores without
learning reliable background rejection. This is a plausible explanation of the
observed behavior, not a completed causal diagnosis across different datasets.

At >0.15, the three heads produced 2,487–2,601 validation returns. Their low
training losses therefore do not establish useful generalization. Those returns
are not counted as known false alarms without complete reviewed labels, and the
heads' scores cannot be ranked against xView3 as calibrated probabilities.
The xView3 experiment only changed one objectness bias: scores increased while
the validation return count stayed at 14. See
[the six fits, controls and failures](DEVELOPMENT_EXPERIMENTS.md).

Explain the frozen encoder versus trained head, which parameter was changed,
the train/validation separation, why identical bias-fit seeds add no uncertainty
estimate, and what legitimate negatives and a useful recall/false-alarm operating
point would add to the next experiment.

## Why AllenAI remains quarantined

Software and dataset licences are distinct. Annotation-specific permission for
the older 2022 database remains unresolved; a newer licensed archive did not
establish permission for those exact NL labels. The clarification request remains
unsent at the owner's instruction.

AI inspection found five provisional source/first-pass consistencies, four
unresolved alignments and fourteen unresolved classes. Original coordinates
stay unchanged and uncertain contexts stay ignored. Twelve unmarked windows
cannot be accepted as complete negative examples. Prior publisher-label exposure
is disclosed; there is no qualified independent reviewer.

Clipped uint8 DN previews lose original amplitude information and cannot serve
directly as native calibrated xView3 input. Replaying 23 native 256px patch
calibrations establishes implementation consistency, not complete 2048px model
contexts or independently confirmed label alignment. Permission, completeness,
alignment, valid native contexts and pretraining overlap all require resolution.
See [the adoption report](ALLENAI_ADOPTION.md).

## Contributions and AI assistance

The conversation records the owner's project objective,
Newfoundland/Labrador scope, low-false-positive priority, staged roadmap requests,
approval decisions and insistence on honest limitations. Those are supported
project-direction contributions.

The assistant inspected data and code, proposed and implemented substantial
engineering changes, authored tests/documentation, ran experiments and performed
AI-assisted image interpretation. Upstream xView3 and CROMA models were reused;
their architecture and pretraining are not original TritonEye contributions.
Software tests and CI check executable behavior. Further AI review does not
satisfy independent SAR annotation review.

The record does **not** establish which modules the owner personally wrote,
which design details they independently developed, or what they can explain
without assistance. Do not substitute assistant work for personal authorship or
claim understanding solely because a task was approved. Add specific personal
contributions only when true, with commit/artifact references where available.

A truthful starting account is: “I directed the project toward Newfoundland and
Labrador MDA and used substantial AI assistance for implementation, debugging,
tests, experiments and documentation. The repository records what ran and what
remains unverified.” Follow it with your actual personal design, coding and
evaluation contributions; avoid claiming sole implementation.

Before presenting, demonstrate your understanding by doing these tasks yourself:

1. Trace the real case's product UUID, native window, polarization LUTs and output
   hashes, and explain the distinction between display pixels and model inputs.
2. Follow one return's physical classification and policy eligibility. Explain
   the effect of a buffer change without equating fewer candidates with accuracy.
3. Explain a training objective, its lack of negative supervision, the controls,
   and why the recorded loss reduction cannot select a deployment candidate.
4. Walk through promotion refusal and rollback. Identify the transaction that
   keeps model selection and decision history consistent after failure.

If you cannot yet explain a component, state that it was AI-assisted and remains
an area you are studying. The [presentation walkthrough](PORTFOLIO_DEMO.md)
provides a concrete rehearsal without overstating personal mastery.
