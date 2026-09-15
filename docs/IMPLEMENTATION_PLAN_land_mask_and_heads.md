# Regional validation backlog (supersedes the earlier implementation plan)

The previous plan contained out-of-region experiments and overly strong claims.
Its history is retained in Git, but it is not the current portfolio method.

Implemented: calibrated SAR preprocessing, conservative land/coastal handling,
provisional fixed-infrastructure proximity, bounded AIS association, provenance,
offline reports and a reproducible synthetic NL workflow.

Still required, in order:

1. Complete a full real-scene replay on adequately provisioned hardware; retain
   processing manifests and failure/latency evidence.
2. Build independently adjudicated, co-temporal NL vessel/ice/structure/clutter
   labels, including Labrador and ice-season scenes, with clear data licences.
3. Measure detection precision/recall and false alarms per valid-water area;
   select a validation operating point and evaluate once on a scene-held-out set.
4. Validate reported geolocation against independent references, including moving
   targets, and measure association accuracy and ambiguity under AIS gaps.
5. Compare a geospatial foundation-model baseline and document domain adaptation.
6. Only after regional acceptance criteria are agreed, add monitored staged model
   deployment, rollback and a separately authorized retraining workflow.

Do not treat vessel-head scores as an iceberg classifier, AIS absence as a label,
or candidate suppression as measured precision. Current methods and references:
[MDA_METHODS.md](MDA_METHODS.md), [DATA_SOURCES.md](DATA_SOURCES.md),
[../EVALUATION.md](../EVALUATION.md).
