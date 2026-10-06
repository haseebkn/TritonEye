# Regional validation backlog (supersedes the earlier implementation plan)

The previous plan contained out-of-region experiments and overly strong claims.
Its history is retained in Git, but it is not the current portfolio method.

Implemented: calibrated SAR preprocessing, conservative land/coastal handling,
provisional fixed-infrastructure proximity, bounded AIS association, provenance,
offline reports and a reproducible synthetic NL workflow.

Full real-scene replay is complete; evidence and measurement limits are recorded
in [the acquisition audit](AUDIT.md#acquisition-reliability--2026-10-05).
The [historical review pilot](NL_BENCHMARK.md) provides provisional annotations,
with independent adjudication still outstanding.

Still required, in order:

1. Build independently adjudicated, co-temporal NL vessel/ice/structure/clutter
   labels, including Labrador and ice-season scenes, with clear data licences.
2. Measure detection precision/recall and false alarms per valid-water area;
   select a validation operating point and evaluate once on a scene-held-out set.
3. Validate reported geolocation against independent references, including moving
   targets, and measure association accuracy and ambiguity under AIS gaps.
4. Compare a geospatial foundation-model baseline and document domain adaptation.
5. Only after regional acceptance criteria are agreed, add monitored staged model
   deployment, rollback and a separately authorized retraining workflow.

Do not treat vessel-head scores as an iceberg classifier, AIS absence as a label,
or candidate suppression as measured precision. Current methods and references:
[MDA_METHODS.md](MDA_METHODS.md), [DATA_SOURCES.md](DATA_SOURCES.md),
[../EVALUATION.md](../EVALUATION.md).
