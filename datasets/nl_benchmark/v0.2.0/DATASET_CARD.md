# NL SAR benchmark draft 0.2.0

Versioned on 11 October 2026. Status: provisional machine-assisted review pilot.
Independent SAR review is unavailable, as confirmed by the owner. Step 3's
independently reviewed benchmark outcome remains incomplete; metric export is
blocked. This dataset supports the TritonEye portfolio relevant to C-CORE's
maritime AI role; it has no C-CORE endorsement or operational acceptance claim.

## Coverage and sampling

21 complete 512x512 native-radar areas from 11 exact Sentinel-1 products,
representing 10 platform/orbit/datatake acquisition groups. Adjacent September 27
slices share an acquisition. The selection includes 17 Newfoundland areas and
four Labrador areas, including inherited Makkovik summer and Hopedale winter
imagery. Coverage is purposive and sparse, not province-wide validation.

The 12 inherited v0.1.0 areas were selected by named geography before detector
inspection. Nine additions from the frozen October 11 overlap bundle are
AIS-informed development enrichment. Those additions intentionally make it easier
to examine possible vessels and competing tracks. They cannot be pooled with the
older geographic sample as an unbiased detector-recall estimate. No area was
selected from a detector prediction. All new areas had earlier implementation
exposure; no new pristine test acquisition is claimed.

Regime tags describe selection intent, not independently verified physical
land/water or coastal eligibility. Each entire selected area is reviewed,
including land texture, dark water and returns lacking a detector or AIS match.
Four 256x256 cells account for every pixel in each band. Complete first pass does
not establish that every vessel was found.

## Splits and inheritance

Nine training areas, ten validation areas and two locked test areas. Existing
v0.1.0 objects, review events, scene assignments and selected image windows are
inherited unchanged. Hopedale test images were not opened or reannotated by this
extension. Their historical annotation/QA exposure remains documented in v0.1.0.

Absolute orbit and datatake group whole acquisitions; geography groups and at
least 1 km separation between cross-split selected polygons prevent overlapping
tiles across splits. The newer northern Newfoundland VV/VH slice inherits the
September 27 validation acquisition assignment. All northern additions remain
validation; eastern Avalon additions remain training. Full source swaths may
overlap: only the selected windows are included, and other pixels are not free
training material. Pretrained-detector and CROMA pretraining overlap has not been
verified; split separation alone does not prove independent generalization.

## Annotations and supporting evidence

The new first pass interprets five isolated compact/elongated two-band returns
as provisional vessels. Harbour and shore-connected returns remain unresolved.
These are AI morphology decisions with possible fixed-target/artefact alternatives,
not confirmed vessel positives. No independently confirmed fixed structures or
ice objects are introduced. Winter does not by itself establish ice.

`annotations.json` combines historical decisions and new pixel boxes/geographic
centres. `review_history.json` preserves earlier author/timestamp/digest records
and appends new complete-area first-pass events. `annotation_decisions.json`
contains only the nine new image-first drafts; historical decisions remain in
the parent release, whose SHA-256 is recorded in the manifest.

`ais_support.json` is generated after the first-pass annotations are persisted.
It records counts from each frozen whole-scene competitor pool and distances
to provisional vessel interpretations, without publishing MMSIs or AIS positions.
Its alignment reference is the catalogue midpoint used by the overlap bundle,
not scene-start or per-pixel radar time. A nearby track supports investigation
but does not establish correct identity, exhaustive vessel truth or detector
performance. Reception gaps and possible coincidence remain material limits.

## Sources, processing and local artifacts

Contains modified Copernicus Sentinel data (2025, 2026). The parent Sentinel
licence snapshot, catalogue times, native measurements, calibration XML, derived
chips and image previews retain source hashes. New areas use the frozen overlap
bundle, including its prior selection lock, image inventory, exact product IDs
and explicit failed raster-boundary proposal. That failed area is not annotated.

Calibration is sigma0 LUT, in native radar geometry with GCP/TPS geolocation;
thermal-noise/terrain correction and surveyed moving-target locations are not
claimed. Fixed display stretch is -30 to +5 dB. Thirteen areas are VV/VH and eight
are HH/HV; HH/HV remains a separate experimental annotation branch without
validated detector compatibility. Sea state and ice conditions lack independent
adjudication; morphology/season are context only.

Large images and private AIS remain in ignored project data directories. The
new offline image-first bundle is `data/benchmarks/nl/0.2.0/review_bundle.html`.
It embeds all 18 new native previews, shows provisional decisions only in
collapsed sections, and contains no detector predictions or AIS markers.
Inherited images use `data/benchmarks/nl/0.1.0/review_bundle.html`. Give a reviewer
the image packages and guides first, then authorized supporting context after
their independent image pass. Do not transfer credentials or raw AIS by default.

## Release and limits

Release hashes freeze the small metadata, annotations, guide, card and review
bundle. The draft assembly method and finalization method retain separate
provenance; code changes never rewrite a released version. Revisions and reviews
need a new version preserving the parent and final-test assignments.

Precision, total recall, false alarms per water area, correct AIS identities,
coastal buffer selection and measured model improvement remain unavailable.
An unresolved object is retained for adjudication rather than silently dropped
from an evaluation denominator. Unmatched returns remain unassociated candidates.
