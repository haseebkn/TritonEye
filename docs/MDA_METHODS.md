# MDA methods and limits

TritonEye is an independent portfolio research prototype focused on the
Newfoundland and Labrador maritime study area. It is not C-CORE software, an
operational navigation service, or a certified MDA system. No C-CORE internal
protocol, acceptance criterion, customer data or endorsement was available for
this work. Public standards inform the interfaces and evidence handling below;
they do not establish operational or NATO compliance.

## Why these choices matter for C-CORE

C-CORE's public description of Coresight emphasizes satellite analysis, human
quality control, reports, and ship/iceberg discrimination. Those needs motivate
TritonEye's explicit uncertainty states, reproducible processing metadata, and
analyst review workflow. This is our interpretation of public material, not an
implementation of C-CORE's internal procedures.
Source: [C-CORE maritime surveillance](https://c-core.ca/solutions/maritime-surveillance/),
accessed 2026-09-04.

## Geographic and temporal contract

- The versioned `configs/aois/newfoundland_labrador.geojson` is an approximate
  research boundary around Newfoundland, Labrador and adjacent waters. It is not
  an official provincial, territorial-sea, EEZ or fisheries-management boundary.
  Smaller imaging AOIs must stay inside it. SAR centroids and AIS observations
  outside it are excluded; reports do not display them. Land/coastal filtering
  is a separate operation.
- Output GeoJSON uses WGS 84 decimal degrees with longitude first, latitude
  second. Association distances use WGS 84 ellipsoidal geodesics in metres,
  never angular degrees. Coastline proximity uses a local metric projection.
  GeoTIFF/GCP transformations retain source
  georeferencing provenance. Reference:
  [IETF RFC 7946, sections 3.1.1 and 4](https://www.rfc-editor.org/rfc/rfc7946).
- Sensor timestamps must identify their UTC offset; UTC output uses `Z` or
  `+00:00`. Image acquisition time, AIS observation time and processing time are
  different quantities. A current processing timestamp must never replace an
  unknown acquisition or observation time. Reference:
  [IETF RFC 3339, sections 4 and 5.6](https://www.rfc-editor.org/rfc/rfc3339).

## Detection and association are separate evidence

A detector score is an uncalibrated model output, not a probability that a target
is a vessel and not a probability of illegal activity. The project does not infer
cargo/fishing vessel type from xView3 objectness. SAR bright returns can include
icebergs, sea ice, land, offshore installations, artefacts and vessels.

The processing record identifies the actual model, weight digest, detection
threshold and preprocessing used for each run. Changing thresholds can reduce
output volume while also missing real vessels; fewer alerts is not evidence of
higher precision. Failed tiles and unavailable input prerequisites must remain
visible rather than being represented as a clean, empty scene.

AIS is incomplete and can contain incorrect or stale data. Some vessels do not
carry AIS and a missing report does not establish a transponder shutdown. We
therefore treat association as a provisional spatial/temporal hypothesis and
retain ambiguity rather than declaring a vessel cooperative or suspicious.
Reference: [IMO Resolution A.1106(29), annex paragraphs 3 and 40–44](https://wwwcdn.imo.org/localresources/en/OurWork/Safety/Documents/AIS/Resolution%20A.1106(29).pdf).
The IMO guidance is for shipborne use; its documented limitations are relevant
to this research pipeline but do not certify it.

| Output state | Interpretation |
| --- | --- |
| `ais_associated` | A unique association under the configured gates; identity still provisional. |
| `ambiguous_association` | Competing associations need analyst review. |
| `uncorrelated_candidate` | No eligible AIS association in the available observations; analyst review, not an operational alert. |
| `unassessable` | Evidence is missing, stale, synthetic or otherwise insufficient for association assessment. |
| `excluded_*` | Invalid geometry, outside study area, or a surface/quality exclusion; not an AIS association. |

All states stay in the correlation audit artifact. The candidate artifact contains
only review-eligible states. The legacy `dark_vessels_geojson` payload name is a
deprecated alias and does not imply confirmed dark vessels. Receiver presence is
not proof of complete AIS coverage. Iceberg discrimination remains unresolved,
including for high detector scores; reports state this prominently.

Land/coastal and infrastructure proximity filters lower the analyst's immediate
workload but can hide real vessels near shore or platforms. Known positions of
movable FPSOs are time-dependent. Proximity to a nominal field coordinate must
not be presented as proof that a return is a fixed platform.

## Evidence needed to claim low false-positive performance

The acceptance target is to be chosen on labelled Newfoundland/Labrador scenes,
not by choosing a high score threshold and claiming success. Required evaluation:

1. Independently label vessels, icebergs, clutter and uncertain objects using
   image chips and time-matched corroborating evidence. Record reviewer, source,
   date, acquisition ID and uncertainty. Do not label every AIS-absent return a
   false positive or every AIS-present return a true detection.
2. Split by acquisition, geography and time before tuning. Keep repeated passes
   and neighbouring chips from leaking across train/validation/test splits.
   Include coastal Newfoundland, Grand Banks, Labrador, and ice-season scenes.
3. Select the operating point on validation data for an explicit precision /
   false-alarm workload target, report the accompanying recall, and evaluate
   the frozen setting once on the held-out test set.
4. Report precision, recall, false detections per valid water square kilometre,
   and candidates per scene, with sample sizes and uncertainty intervals. Break
   down results by vessel size, near-shore distance, sensor/polarization and ice
   conditions. Count exclusions and unusable scene area separately.
5. Report association accuracy separately from detector accuracy. AIS-confirmed
   recall is a partial-reference diagnostic; it cannot establish precision or
   recall for vessels absent from the AIS reference.

Until that benchmark exists, precision and false alarms per square kilometre are
unmeasured. No maximum false-positive rate, calibrated identity confidence, or
operational detection guarantee is claimed. Sea-ice charts and iceberg limits
provide contextual risk information; they are not per-object ground truth and
cannot justify declaring every target inside a limit to be ice.

## Analyst handoff and reproducibility

Retain raw detection geometry, surface/proximity flags, association reason,
candidate identities, observation ages, motion/uncertainty assumptions, model
digest, configuration, acquisition ID, source assets and MLflow run ID. Preserve
immutable input manifests and software/environment versions with evaluation
results. Do not log credentials into payloads or reports.

The HTML report embeds its filtered data and renders readable tables/provenance
without a network. The optional map still depends on the Leaflet CDN and Esri
basemap tiles; it is not a fully offline map. External identifiers and metadata
are escaped as text, including script-embedded JSON, to prevent input values
from becoming executable markup. There is no automatic operational alert,
enforcement action, or model promotion in this prototype.
