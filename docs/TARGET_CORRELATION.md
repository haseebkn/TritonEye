# NL target correlation research

Step 6 is **partially implemented, not complete**. TritonEye can compare its
geometric matching algorithm with an experimental uncertainty-aware algorithm
and record independent identity decisions. It has **no independently reviewed
SAR–AIS matching examples and no measured improvement**. A learned ranker is not
implemented or trained. These are prerequisites for a defensible C-CORE portfolio
claim, not C-CORE operational requirements or certification.

## Current evidence

The detector-blind association pilot references ten development areas from five
acquisitions in the frozen NL detection benchmark. It preserves twenty unresolved
image-first object locations and excludes the two locked Hopedale test areas.
Context clutter areas are not discrete object identities or negative pairs.
No new independent review is claimed. The complete-area inventory remains
unverified, including areas with no marked objects.

The [shared-pool replay](evidence/association_shared_pool_20261006.json)
contains six validation areas: four Newfoundland VV/VH areas and two Labrador
HH/HV areas. It has zero reviewed validation cases. The September 27 snapshot
contains 86 reports; bounded alignment retains 26 NL identities, **not 26 vessels
inside the selected imagery**. Neither matcher selects an association to the
pilot targets. Other selected acquisitions have no AIS snapshot observations.
Identity metrics and uncertainty intervals remain unavailable, rather than zero
errors or perfect precision. HH/HV results stay separately identified; coordinate
matching does not validate HH/HV vessel detection or offshore coverage.

The review bundle is saved locally at
`data/association_research/20261006_review_02.json`; the replay records its file
hash, dataset content digest and matching-code hashes. Existing raw imagery,
AIS snapshots, released annotations and historical baseline evidence remain
unchanged. The bundle contains SAR and AIS source paths and SHA256 hashes.
The [earlier replay](evidence/association_comparison_20261006.json) remains
historical evidence from before the shared-pool correction. Its recorded code
hashes refer to that earlier implementation, not the corrected comparison.

## Algorithms and uncertainty assumptions

Both methods use validated UTC AIS, reject conflicting fixes, interpolate only
plausible bracketing tracks, limit velocity extrapolation to 120 seconds and
reject old fixes without usable motion. The geometric control retains the
production algorithm: maximum-cardinality, minimum-distance one-to-one geodesic
assignment with a 500 m base gate plus the existing heuristic AIS allowance.
This is an algorithm comparison over image-first objects, **not an end-to-end
production-policy replay**. Coastal objects remain available for review; alert
eligibility is not used to discard difficult matching examples.

Both comparison methods use the **same scene-time-aligned NL identity pool**.
The geometric control aligns the full telemetry first, then filters aligned
positions to NL. The experimental method retains the full report histories of
only those identities, including raw reports outside NL. Reports outside the
study boundary can therefore move into the shared pool. Per-target refinement
cannot introduce an identity absent from that pool; this is a controlled
comparison, not evaluation of a wider experimental retrieval universe. Reports
record the shared pool's scope, identity count and MMSIs. This fixes the earlier
asymmetry between aligned-position filtering and raw-report filtering without
changing the production geometric matcher.

The experimental algorithm aligns AIS separately to each target's UTC timestamp
when supplied. Otherwise it uses scene start time and an explicit timing
allowance. It does not derive per-pixel azimuth times from SAFE annotation.
Its timestamp sanity bound of 300 seconds is not proof of a valid pixel time:
reviewers must substantiate the actual timestamp and its uncertainty.

Displacement is represented in local east/north metres using geodesic distance
and azimuth. The assumed covariance adds SAR and AIS location variance, motion
caused by timestamp uncertainty, alignment-age speed/course errors and an
acceleration allowance. Unknown speed is not treated as stationary: it uses the
existing 60-knot plausible-motion bound isotropically. Bracketing reports supply
an estimated course and speed, not measured motion-error statistics.

The score combines squared [Mahalanobis distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html)
and a covariance determinant penalty. The latter prevents diffuse tracks from
winning merely because they have larger error ellipses. Assignment has an
explicit abstention option, a squared-distance gate and a hard 2 km distance cap.
All targets sharing gated identities remain ambiguous, including unmatched
competitors. A selected identity is tentative, not verified.

| Provisional setting | Value |
| --- | --- |
| SAR and AIS location standard deviations | 100 m and 30 m |
| SAR and AIS timestamp standard deviations | 15 s and 5 s |
| Speed and course standard deviations | 1 m/s and 10 degrees |
| Acceleration standard deviation | 0.01 m/s² |
| Squared-distance gate | 9.21 |
| Maximum displacement and abstention cost | 2 km and 20 |

These are **uncalibrated research assumptions**, not estimated sensor accuracy,
probabilities of identity, tuned thresholds or employer protocols. Provider AIS
timestamps are not necessarily vessel GNSS fix times. Spatial bias, vessel turns,
SAR moving-target displacement and correlated errors are not fully modelled.
Production positions, gates and assignment are unchanged; automatic alerts remain
disabled. Human reports use **unassociated candidates**. The legacy internal
`uncorrelated_candidate` value remains compatible with existing artifacts.

## Independent review contract

Review native co-polarized and cross-polarized imagery without detector output or
either matcher's selections. Inspect all selected areas and add missed objects
before declaring the inventory complete. Use co-temporal track evidence,
timestamp basis, vessel motion and available independent identity evidence.
Proximity alone cannot establish identity; missing AIS cannot establish intent.

Each target needs one decision with object class, rationale and evidence:

- `matched`: independently supported vessel identity, exactly one observed MMSI.
- `no_association`: resolved vessel or non-vessel with no association to the
  supplied observations; this does not certify receiver completeness or darkness.
- `ambiguous`: a vessel with at least two supported possible observed identities.
- `unresolved`: insufficient object or identity evidence; not a negative label.

The case contains product UUID, native acquisition group, image area, split,
region, coastal/offshore regime, polarization, source hashes, object locations
and complete telemetry. Review binds a distinct reviewer, explicit UTC review
time and evidence to `case_digest(case)`. Append a review-history event containing
`case_id`, `case_sha256`, `labels_sha256=digest(case['labels'])`, `reviewer` and
`reviewed_at`. Changing targets, telemetry or sources invalidates the case
binding; changing decisions invalidates the corresponding history event.
Only independent review plus a complete resolved inventory permits metrics.
These are auditable declarations, not authentication of a human reviewer.

Training and validation cannot share acquisitions, geographic groups or areas
within 1 km. Existing detection test acquisitions and geographic buffers remain
locked, including alternate catalogue names. Validation is held out from fitting;
it is not the locked final test. Freeze any future reviewed release before tuning,
retain all revisions and decisions, and do not rewrite detection release v0.1.0.

## Metrics and promotion conditions

Each validation area reports accepted-identity precision and association recall,
wrong tentative and accepted identities, false associations, missed associations,
unsafe resolutions of reviewed ambiguous cases, ambiguity and unassociated
candidates. Ambiguous selections do not count as accepted correct identities.
A wrong identity also misses the true association; the total error count counts
that target once. Withholding everything leaves recall at zero when known
associations exist, and precision undefined when no identities are accepted.
Unresolved or incompletely reviewed cases have null metrics.

Binomial intervals are conditional independence summaries. Connected acquisition
and geographic blocks supply paired resampling, including the difference in
error rates; fewer than five blocks or a degenerate empirical distribution cannot
support an uncertainty claim. Cases retain geographic, operating-regime and
polarization strata, with separate VV/VH and HH/HV comparison intervals.

Promotion requires a predeclared validation protocol, lower error with paired
uncertainty supporting improvement, and no loss of useful association recall or
coverage, including important regional and coastal strata. No automatic promotion
or improvement claim is implemented. Configuration changes after looking at
validation outcomes must be recorded as new experiments; a repeatedly tuned
validation set is not an untouched final test.

A provisional learned-ranker readiness check requires at least 500 reviewed
training pairs, 50 positive and 50 negative pairs, five acquisitions and five
geographic groups. Pairs must have actual aligned telemetry within the fixed
2 km retrieval area. Reviewed unique identities support positive and alternative
negative pairs; ambiguous/unresolved objects and missing telemetry do not.
These screening minima do not establish statistical sufficiency, trigger training
or implement a ranker. Current reviewed training pairs: zero.

## Reproduction

From the repository root, use new output names to retain prior artifacts:

```sh
python -m agents.association_study pilot --output data/association_research/review_new.json
python -m agents.association_study compare --bundle data/association_research/review_new.json --output data/association_research/comparison_new.json
python -m pytest tests/unit/test_association_study.py tests/unit/test_association.py tests/unit/test_correlation.py -q
```

Tests use explicitly synthetic identities to check motion, covariance, timing,
competition, abstention, review integrity and error accounting. They are software
correctness evidence, not held-out maritime performance evidence.
