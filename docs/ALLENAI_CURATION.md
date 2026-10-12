# AllenAI provisional curation and ignore masks

The workflow prepares image-first annotation sidecars without adopting the
quarantined AllenAI labels. The local packet contains **31 whole areas from three
Newfoundland acquisitions**, 62 VV/VH previews, 31 intersection-validity masks
and 31 initial ignore masks. All **32,505,856 rectangular pixels start ignored**;
30,232,016 have valid imagery in both bands. No areas have been curated, no
publisher points have been revealed through this workflow, and no training,
evaluation or threshold-selection data have been exported.

See the [unsent clarification request](ALLENAI_CLARIFICATION_REQUEST.md) and
[source audit](ALLENAI_NL_AUDIT.md). Licence scope, annotation completeness and
independent local review remain unresolved. This work supports the C-CORE
portfolio's reproducible geospatial ML engineering; it is not operational
validation or a claim of compliance with C-CORE's internal protocols.

## First pass without publisher points

Open the local packet:

```text
data/model_research/allenai_curation/v0.1.0_20261012_04/first_pass/index.html
```

Provide **only the `first_pass/` directory** to the annotator, not its parent
workspace, source map, original contact sheets or point-centred patches.
Neutral IDs and fixed whole-window previews contain no publisher points,
detector predictions, AIS overlays or source-window IDs. The separate
`source_map.json` preserves exact source identities, acquisition groups and
geometry for the later alignment pass. This separation is a workflow guard,
not a security boundary against the owner opening local source files.

The existing preparer has previously seen publisher-point QA material. A new
annotator can avoid that exposure; existing self/AI curation must disclose it
through `prior_label_exposure: true`. Neither is independent expert review.
The displays reproduce clipped publisher raw DN, **not calibrated sigma0**;
saturation, source-edge missingness and subtle signatures limit interpretation.
Review the original 1024-pixel PNGs at native resolution as well as the HTML's
scaled view. These are Newfoundland areas, not Labrador or province-wide coverage.

Edit `first_pass/decisions.json` using an image viewer and pixel-coordinate
polygons. This release supplies a viewer and validated JSON interchange, not an
interactive drawing editor. Set the actual actor, `method` (`self` or
`ai_assisted`) and honest prior exposure. `independent_review` stays false.
Inspect the **entire valid area**, including objects the detector could miss.
Record reasons and polygon regions for vessels, background, fixed structures,
clutter, identifiable ice and unresolved/uncertain areas (`ignore`).

Coordinates are pixel edges: x increases right, y increases down, top-left is
(0,0). A polygon can span nodata; nodata always stays ignored. Pixel centres
determine raster membership. An explicit background region may cover the full
rectangle only after complete inspection, with object and ignore regions
overriding it. Never infer background from missing publisher points, coastal
policy exclusions, AIS absence or black invalid borders. Use ignore polygons
large enough to cover uncertain object extent and alignment context.

An illustrative region, **not an annotation of a real image**, is:

```json
{
  "class": "ignore",
  "reason": "Unresolved return and surrounding alignment context",
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[10,10],[50,10],[50,50],[10,50],[10,10]]]
  }
}
```

Set `full_valid_area_reviewed: true` only after inspection. The freeze command
also requires polygons to cover every valid pixel; a checkbox alone is
insufficient. Complete coverage with explicit ignore is allowed, but supplies
**no negative supervision**. Software cannot certify the quality of a visual
review merely from coverage or reasons.

## Freeze and then check publisher alignment

From the project root, use fresh output paths for each attempt:

```sh
python -m agents.allenai_curation check --workspace data/model_research/allenai_curation/v0.1.0_20261012_04
python -m agents.allenai_curation freeze --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --decisions data/model_research/allenai_curation/v0.1.0_20261012_04/first_pass/decisions.json --output data/model_research/allenai_curation/first_pass_01
python -m agents.allenai_curation reveal --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --first-pass data/model_research/allenai_curation/first_pass_01 --output data/model_research/allenai_curation/alignment_01
```

The freeze saves actor/exposure history, timestamp, input/workspace hashes,
decision bytes, class masks and binary ignore masks. All outputs remain
provisional. Reveal verifies those hashes before exposing the 23 published
points and their original coordinates mapped onto the publisher grid, not the
native TPS grid or centred crops. Never snap them to a bright return.

For every point, record one state and its reason in the revealed
`point_decisions.json`: `usable_provisional`, `unresolved_alignment`,
`unresolved_class` or `unusable`. A usable point must align with an already
recorded first-pass vessel region and valid vessel pixels, with written
alignment evidence. This is provisional consistency, not an independent class
confirmation. An unresolved or unusable point must remain inside an explicit
ignored region; include its uncertain extent, not merely a one-pixel exclusion.
The second pass can **only add ignore regions**, not invent new vessel truth or
rewrite the first pass. Discovering a first-pass omission requires a fresh,
explicitly label-exposed curation attempt, not silently modifying the sealed
history or pretending the revised pass is blind.

```sh
python -m agents.allenai_curation seal-points --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --first-pass data/model_research/allenai_curation/first_pass_01 --decisions data/model_research/allenai_curation/alignment_01/point_decisions.json --output data/model_research/allenai_curation/point_pass_01
```

No freeze/reveal/seal commands above have been performed on real labels: the
packet is intentionally unreviewed. Re-running `prepare --output <fresh-path>`
creates a new packet; no output can replace a frozen dataset or existing bundle.
Intermediate local preparations `_02`, `_03` and the initial unsuffixed directory
are retained, not training releases; `_04` is the documented preparation.

## Mask meaning and adoption boundary

| Class mask value | Provisional meaning |
| --- | --- |
| 0 | Ignored: nodata, unreviewed or explicitly uncertain |
| 1 | Vessel interpretation |
| 2 | Explicitly reviewed background |
| 3 | Fixed structure interpretation |
| 4 | Clutter interpretation |
| 5 | Identifiable ice interpretation |

The binary ignore mask is 255 for ignored pixels, 0 otherwise. Background is
painted first, object interpretations override it, and ignore/nodata override
everything. Conflicting object classes fail closed unless explicitly ignored.
These polygons describe reviewed regions; publisher point boxes are **not hull
truth**, and this module does not export synthetic vessel bounding boxes.

These are **curation QA artifacts only**, not inputs accepted by current model
training or benchmark evaluation. `load_workspace(..., purpose="training")`,
evaluation, threshold selection and promotion requests fail closed. Metadata
booleans cannot approve adoption or assert independent review. A future
training-only release needs resolved permission, documented completeness and
missing-label policy, imagery compatibility and acquisition/geographic split
checks. Independent evaluation and model-promotion requirements remain separate.

The frozen `datasets/allenai_nl/v0.1.0` bytes/hashes, existing NL benchmark
versions, `agents.nl_benchmark` review guard, locked test images and production
model/configuration are unchanged. The held-out product/datatake/geographic
guard executes before packet pixels are read and on subsequent loads. No
detector run, AIS association or model training occurs here.

## Validation evidence

The [preparation ledger](evidence/allenai_curation_preparation_20261012.json)
records the local packet and original release hashes. Behavioral regressions
exercise neutral packet contents, full-area coverage, mask precedence/nodata,
conflicts, point-coordinate preservation, unresolved ignores, tampering,
overwrite refusal, held-out scope rejection and blocked adoption purposes.
Synthetic fixtures run without copying or accessing private imagery in CI.
Software validation is not annotation truth or evidence of false-alarm reduction.
