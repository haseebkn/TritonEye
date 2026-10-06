# Newfoundland and Labrador detector baseline

TritonEye now has a reproducible baseline evaluation and validation-only threshold
search. The historical pilot still cannot establish vessel precision, recall,
false-alarm density or association correctness: no independent SAR reviewer is
available, and its machine-assisted labels remain provisional. No operating
threshold has been selected and automatic operational alerts remain disabled.

This is relevant engineering evidence for C-CORE's maritime AI role, not a claim
of operational readiness, C-CORE approval or compliance with an unnamed maritime
standard. The benchmark remains confined to Newfoundland and Labrador.

## Evaluation scope

The frozen pilot contains six acquisitions and twelve complete 512 by 512 pixel
review areas. The baseline excludes both locked Hopedale test areas. It attempts
all ten training and validation areas; unsupported HH/HV areas stay in processing
coverage denominators. Only validation areas enter threshold selection. Previously
used Newfoundland scenes retain their disclosed development-exposure flags.

Inference uses native 2048 pixel contexts at production tile origins, actual
VV/VH calibration and georeferencing, joint valid-pixel masks and the production
deduplication routine. Contexts must not intersect another split's selected areas
or their 1000 m separation zone. Chips are not padded or resized to imitate native
contexts. This bounded replay is not a full-swath benchmark: its edge stitching
and runtime can differ from processing an entire acquisition.

The pinned xView3 model supports VV/VH. HH/HV inputs are explicitly unsupported,
not renamed to VV/VH, treated as negative images or removed from the evaluation
scope. Scores are uncalibrated objectness values, not vessel probabilities.

## Metrics and missing evidence

| Measure | Definition and evidence requirement |
| --- | --- |
| Raw vessel precision and recall | Geodesic one-to-one matching within 100 m to all independently reviewed vessels in the fixed valid-imagery area. Raw precision includes land returns. |
| Post-policy precision and recall | Same vessel denominator, after coastal and infrastructure eligibility rules. Withheld coastal vessels remain misses. |
| False alarms per water square kilometre | Physical-water returns unmatched in the raw assignment divided by fixed evaluated physical-water area. Post-policy density counts unmatched water returns in the post-policy assignment using that same area denominator. |
| Policy exclusions and abstentions | Descriptive retained coastal returns, physical-land returns and unknown-surface or unknown-distance abstentions. These are not false positives. |
| Detector and policy misses | All reviewed vessels missed by raw detections, and extra misses introduced by eligibility policy. Both remain unavailable without reviewed labels. |
| AIS subset proximity recall | Matches within a fixed 100 m to the bounded, time-aligned AIS subset in the area; raw and post-policy versions are separate. Not overall detector recall. |
| Association ambiguity | Competing geometric candidates under the production 500 m base gate plus each AIS position's heuristic uncertainty allowance. Unmatched competing returns count as ambiguous. |
| Association correctness | Requires independent adjudication of vessel identities. A nearest AIS match is not a correct-identity label. Currently unavailable. |
| Processing completeness and resources | Complete selected-area processing, native context count, wall time, process RSS sampled every 100 ms and CUDA peak allocated/reserved memory. Unsupported and failed areas remain in coverage. |

Physical-water area is a geodesic area derived from the frozen valid imagery and
OSM land polygons. It is conditional on that shoreline source, not independently
verified water truth. Sparse controls in St. John's, Bonavista and Lewisporte do
not validate the whole province; Labrador remains outside those controls. CanVec
remains a local comparison source, not a universal replacement.

Reports separate region, coastal/offshore setting, polarization and acquisition
season. Season does not certify ice, sea state, weather or incidence angle: these
conditions remain unverified. The retained coastal buffer comparison evaluates
0, 100, 300, 500 and 1000 m without automatically changing policy.

## Provisional operating objective

These are research design choices, **not C-CORE requirements**:

- False-alarm upper confidence bound no greater than 0.1 per evaluated water km².
- Recall lower confidence bound at least 70%, including coastal vessels withheld
  by policy.
- At least 90% processing and independently measured water coverage, both overall
  and within every represented region, polarization and coastal/offshore stratum.
- At least 30 independently labelled vessels and five independent connected
  acquisition/geographic blocks.

The fixed objectness grid is 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90 and
0.99. Selection is validation-only and uses strict score greater than threshold.
The cache floor is 0.05; lower thresholds require fresh inference. Among admissible
points, choose highest recall, then lowest density, then lowest threshold. Zero
returns cannot satisfy minimum recall. Missing labels, area or supported sensor
coverage block selection; no best-looking threshold is substituted.

Conditional 95% Clopper-Pearson binomial and exact Poisson intervals accompany
precision, recall and density. Selection adjusts the interval tail probability
over the fixed grid and its two statistical constraints using Bonferroni.
Connected-block resampling joins areas sharing any acquisition or geographic
group, including transitive links; fewer than five blocks, undefined denominators
and degenerate resamples produce an explicit unavailable interval. Conditional
intervals assume independent events and do not correct label error, shoreline
error, purposive selection or clustering. The minimum block count is an evidence
floor, not proof of independence or a cluster-robust operating guarantee.

Method references: [NIST binomial intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[exact Poisson intervals](https://www.statsmodels.org/v0.14.0/generated/statsmodels.stats.rates.confint_poisson.html)
and [paired bootstrap and degenerate intervals](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.bootstrap.html).

## Reproduction

```sh
python -m agents.nl_benchmark --verify-files --artifact-root E:/TritonEye
python -m agents.baseline --artifact-root E:/TritonEye --output data/benchmarks/nl_baseline/fresh_replay
pytest tests/unit -q -m "not slow"
```

Use a new output directory; previous evidence is never overwritten. The verified
SAR inputs, model weights, frozen private AIS snapshots and raw prediction cache
remain in the project folder and outside Git. A validator in a separate checkout
can use the explicit artifact root without copying private data or weakening path
bounds. Source, model, configuration, code and dataset hashes accompany the report.
Protocol version 2 records executing code, configuration and shoreline controls
from the running checkout, separately from model, SAR, reference and AIS assets
at the artifact root. The final integrity check uses the executing checkout.
Model loading is timed separately; per-area inference times exclude shoreline
loading and subsequent scoring. Sampled RSS is absolute process memory, not a
guaranteed operating-system high-water mark.

## Real replay results

The preserved `20261005_replay_02` report and public evidence are historical,
preceding protocol version 2's provenance separation, fresh replay destination
guard, raw-assignment water false alarms and production AIS assignment gate.
Their original hashes and values remain unchanged; they do not validate the
corrected implementation. Fresh SAR inference is unnecessary to verify these
fixes: inference is unchanged and every independent accuracy/false-alarm metric
and AIS subset metric in this pilot is null. Behavioral regressions exercise the
corrected cases with synthetic inputs; new measured claims require a new run
with independent labels and co-temporal AIS.

The 5 October 2026 replay used the pinned model on CUDA. Six VV/VH areas
completed all twelve selected native contexts. Four HH/HV areas were unsupported;
both locked Hopedale test areas were excluded. Validation physical-water area is
88.706 km²: 58.010 km² processed, or 65.40%. Independently measured coverage is
0%. Newfoundland VV/VH validation processing coverage is 100%; Labrador HH/HV
validation processing coverage is 0%. These are selected-area coverage figures,
not province-wide surveillance coverage.

At the unchanged provisional 0.15 threshold and 300 m policy:

| Area | Split | Water km² | Returns | Coastal withheld | Land returns | Eligible returns | Inference seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| St. John's harbour | Training | 4.626 | 1 | 0 | 0 | 1 | 41.59 |
| Eastern Avalon offshore | Training | 26.611 | 0 | 0 | 0 | 0 | 14.73 |
| Bonavista coast | Validation | 10.096 | 4 | 1 | 3 | 0 | 7.86 |
| Bonavista offshore | Validation | 26.554 | 0 | 0 | 0 | 0 | 14.80 |
| Lewisporte coast | Validation | 8.257 | 4 | 3 | 1 | 0 | 15.71 |
| Bay of Exploits water | Validation | 13.103 | 2 | 0 | 2 | 0 | 29.49 |

The validation areas produced ten returns: four reference-relative coastal-water
returns and six reference-relative land returns, with no open-water eligible
returns or abstentions. **This is not ten false positives, nor evidence of perfect
filtering.** Without reviewed truth, vessel counts, detector misses, policy misses,
precision, recall and false-alarm density remain unavailable. Zero offshore returns
cannot distinguish correct rejection from missed vessels. All areas have zero
aligned AIS positions in their selected imagery, so AIS subset proximity recall
and association ambiguity/correctness are also unmeasured.

Model loading took 8.56 seconds. Replay wall time was 170.46 seconds excluding
the initial file-integrity check. Per-area sampled process RSS peaked at 3.17 GiB;
CUDA allocated memory peaked at 6.43 GiB, with 6.57 GiB reserved. The first area
includes CUDA warm-up and is not comparable to a steady-state single-context
time. Concurrent work and host load can affect these timings.

[Machine-readable evidence](evidence/nl_baseline_20261005.json) records the source,
model, code, configuration and dataset hashes, per-area results and blocking
threshold-selection decision. The full local report and prediction caches remain
under `data/benchmarks/nl_baseline/20261005_replay_02/`. Research defaults and
operational alert settings were not changed.

## Remaining requirements

Independent complete-area labels and vessel-identity adjudication are needed
before accuracy and correlation correctness can be reported. More independent
validation acquisitions and geographic groups are needed before threshold
selection, even after reviewing this pilot. HH/HV support requires a separately
validated model/preprocessing experiment. The test split remains locked until an
eligible validation operating point and evaluation protocol are frozen.
