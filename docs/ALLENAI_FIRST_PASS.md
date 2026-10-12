# Newfoundland provisional first pass curation

All **31 selected Newfoundland areas** have a completed AI-only visual first
pass, frozen at **12 October 2026, 04:56:39 UTC**. Both whole 1024×1024 VV/VH
images were inspected for each area. The authored decisions include **five
provisional vessel-like regions**, one provisional shore-connected structure
region, 29 background regions and 64 ignore regions. These are annotation
decisions, **not confirmed vessels, unique vessel counts or model performance**.

The annotator was the OpenAI Codex assistant, not a qualified SAR reviewer.
The schema records `method: ai_assisted`; no human took part in this pass.
**Prior publisher-label exposure is explicitly declared.** Publisher points,
detector outputs and AIS were withheld from the current whole-image displays,
but the assistant retained earlier project/point-QA context. This is therefore
not label-naive curation, independent review or unbiased benchmark truth.

The [verified aggregate ledger](evidence/allenai_first_pass_20261012.json)
contains per-area input/mask hashes and counts, without distributing imagery,
annotation polygons or original publisher coordinates. The original
[quarantined AllenAI release](../datasets/allenai_nl/v0.1.0/DATASET_CARD.md)
and [NL benchmark](../datasets/nl_benchmark/v0.2.0/DATASET_CARD.md) are unchanged.

## Decisions and uncertainty

| Authored interpretation | Areas or regions | Boundary |
| --- | --- | --- |
| Vessel-like | One region each in `area_001`, `area_006`, `area_016`, `area_022`, `area_024` | Visually separated narrow/elongated returns consistent across VV/VH; provisional class only, not identity or surveyed hull extent |
| Fixed structure | One region in `area_004` | Angular shore-connected line interpreted as a jetty/breakwater; provisional, not an operational exclusion rule |
| Fully ignored valid imagery | `area_002`, `area_011` | Narrow, poorly interpretable source-edge wedges; black nodata is never background |
| Other ignores | Small, saturated, cross-shaped, irregular and nearshore returns; land/coastal context envelopes | Unresolved class/extent, not rejected vessels or confirmed negatives |
| Background | 29 explicitly authored whole-area regions, overridden by object/ignore masks | Visibly diffuse texture after inspection, not certified exhaustive negatives |

Five vessel-like polygons in five areas do **not** mean five new independently
confirmed vessels, five correct publisher matches, five previously missed
detections or five unique identities. Repeated areas/acquisitions and prior
exposure limit interpretation. The 31 areas come from **three acquisitions**;
they are not 31 independent scenes or Labrador/province-wide coverage.

The displays are clipped publisher DN, not calibrated sigma0. Saturation,
speckle, small targets and missing edge context limit class interpretation.
No ice or clutter class was confidently authored; unresolved possibilities were
ignored. Some true vessels may be inside ignores or missed in apparently
diffuse background. No recall, precision or false-alarm rate can be inferred.

## Mask accounting

| Pixel category | Count |
| --- | ---: |
| Rectangular extent | 32,505,856 |
| Valid in both bands | 30,232,016 |
| Nodata, always ignored | 2,273,840 |
| Valid but ignored | 7,904,224 |
| Provisional background after overrides | 22,299,567 |
| Vessel-like review context | 23,665 |
| Fixed-structure review context | 4,560 |

Ignore has precedence over all classes. The vessel/structure pixels describe
**generous review context envelopes, not object segmentation or vessel boxes**.
These counts are in image pixels, not evaluated water area in square kilometres,
and are not training supervision admitted by the project. Broad coastal ignores
are local annotation uncertainty, not a revised shoreline mask or a change to
the 300 m operational research policy.

## Frozen local artifacts and replay

The original packet/template remains intact. Authored decisions were saved
separately, then copied into a fresh frozen pass:

```text
data/model_research/allenai_curation/
  v0.1.0_20261012_04/first_pass/     original neutral images and blank template
  first_pass_decisions_20261012.json   authored provisional input, ignored by Git
  first_pass_20261012_01/
    decisions.json                sealed authored decisions and observations
    seal.json                     hashes, time, counts and quarantine flags
    area_001_classes.png          class values 0–5; analogous files for all areas
    area_001_ignore.png           255 ignored, 0 not ignored; all 31 areas
```

The seal SHA256 is
`1a480b9749fbac064c11b734fe6ce45f19968fc50b43da9a0119e606c8dcc189`;
the frozen decisions SHA256 is
`ae271c0b97eaa65c162ddcf198fa4e5e224bba6349e3153523cb19b85a7546c1`.
The ledger binds these to the original packet/source release and executing
verifier. Published input image hashes are unchanged.

Replay from the original project, with cached local inputs present:

```sh
python -m agents.allenai_first_pass --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --first-pass data/model_research/allenai_curation/first_pass_20261012_01
```

The verifier checks all artifact hashes, actor/exposure/current-display
declarations, coverage of every valid pixel, class and binary-ignore masks,
nodata precedence and aggregate counts. All **62 masks** replay exactly from
the authored polygons. Software checks establish integrity and consistency,
**not visual-review quality, complete labels or vessel truth**. Automated code
review is not an independent review of SAR annotations.

## Adoption remains blocked

This first pass did not reveal or decide publisher points. The subsequent
[adoption checks](ALLENAI_ADOPTION.md#publisher-point-decisions) own the AI-only
second-pass decisions and their limitations. Original coordinates and this first
pass remain unchanged. Independent class/alignment review is still unavailable. The
[clarification request](ALLENAI_CLARIFICATION_REQUEST.md) remains unsent and
licence/completeness questions remain unresolved.

Training, evaluation, threshold selection and promotion exports remain blocked.
The independent-review guard, locked test pixels/geography, existing frozen
dataset bytes/hashes and production detector/configuration are unchanged. No
detector run, model fit, AIS association or measured-improvement experiment
occurred. For the C-CORE portfolio, this is reproducible, uncertainty-aware
curation work—not operational accuracy or compliance with internal protocols.

The separately recorded second-pass alignment review is now available in the
[adoption report](ALLENAI_ADOPTION.md). Permission and training-only adoption
requirements remain separate from independent benchmark/promotion requirements.
