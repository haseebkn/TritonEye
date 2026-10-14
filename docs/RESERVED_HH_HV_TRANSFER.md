# Reserved HH HV transfer collection

On 14 October 2026, all eight reserved Newfoundland and Labrador acquisitions
were materialized as sixteen native 512×512 windows. The AI first pass inspected
both polarizations across each selected window before candidate generation or
classifier scoring. This is a **provisional, unscored collection**, not a completed
ship-versus-iceberg transfer benchmark.

## What the collection contains

The [versioned collection](../datasets/hh_hv_transfer/v0.1.0/collection.json)
records 25 unresolved compact-return boxes, eight coastal-structure contexts and
eight sea-clutter contexts. None supports a defensible vessel or plausible iceberg
label. All samples remain ignored for accuracy. Whole-area visual inspection does
not establish exhaustive object truth, and the context samples are not confirmed
negatives. No AIS absence was used to infer an iceberg.

Twenty-three unresolved returns have native 75×75 HH/HV chips. Two boxes lack a
complete context and remain explicit boundary abstentions; they were not padded,
resized or moved. Sixteen additional context chips describe their surroundings.
These are not 41 independently labelled targets.

The collection retains the four geographic groups, eight distinct acquisitions
and train/validation assignments in the [reservation](HH_HV_DEVELOPMENT.md).
Each window is centred on its original envelope centroid. No candidate or model
score selected its position. The windows cover only part of each reserved
envelope, not full acquisitions. Winter 2025 and summer 2026 also differ in
platform, so season, year and platform effects cannot be separated.

## Input audit

All scenes are IW GRD HH/HV. Native annotations record 10 m range and azimuth
pixel spacing and mid-swath incidence angles of approximately 38.81–39.08°.
The latter are scene-level values, not per-target incidence estimates. Pixel
spacing is not effective resolution. Native annotations report no thermal-noise
correction; no extra terrain or noise correction was introduced here.

All **32 band calibration replays match exactly** after float32 conversion,
using the original digital numbers, calibration lookup tables and native row and
column offsets. Chips retain sigma0 dB values, HH/HV order and native sampling.
No resampling, model refitting or classifier evaluation was performed.

Bounded Copernicus requests transferred 895,887,548 bytes, including complete TIFF
strips needed for the selected windows. Original byte-range hashes, source names,
product UUIDs, XML, digital-number arrays, calibration and output hashes remain
under `data/hh_hv_transfer/v010_native_01/`. Partial downloads do **not** verify
the complete source TIFF checksum. Original download-code bytes had SHA256
`38870f9a3ace4f6ec9e65f58337603029e4bacbbf3f423e10359812d92a185b0`;
subsequent formatting and validation changes do not retroactively change that
historical implementation identity.

## Comparison with Kaggle training chips

Only the frozen training partition—595 ships and 527 icebergs—contributes to the
reference distributions. The source JSON contains all competition rows, but
validation and holdout rows are excluded before converting bands for this audit.
No classifier weights or outputs are used.

| Population | Median HH / HV dB | Median apparent width × height in pixels | Components touching chip edge |
| --- | --- | --- | --- |
| 23 unresolved NL chips | −13.73 / −21.53 | 19 × 26 | 10 / 23 |
| 595 Kaggle training ships | −19.87 / −26.07 | 8 × 14 | 19 / 586 components found |
| 527 Kaggle training icebergs | −22.11 / −26.71 | 10 × 14 | 1 / 523 components found |

The frozen descriptive size proxy measures a connected bright component near the
chip centre after 3-pixel median filtering, with a 6 dB contrast above each band's
10-pixel border median. Either band may contribute. This measures apparent radar
extent, **not hull size or iceberg dimensions**. Shore-connected returns can
include structures, rock and sidelobes; edge-touching components are truncated.
All components, including truncation flags and failures, remain in the report.

The brighter NL values and larger apparent components warn of a distribution
difference; they do not diagnose its cause or prove classifier incompatibility.
Kaggle physical pixel spacing and preprocessing equivalence remain unresolved.
Matching 75×75 dimensions alone therefore does not establish compatible inputs.

## Annotation history and safeguards

Commit `08496ea` froze the window and comparison protocol before imagery
inspection. Commit `4793346` recorded all first-pass decisions before the
descriptive chip comparison. The annotator had previously seen other project
labels and classifier results; this is disclosed prior exposure, not independent
or fully naive review. No scores were viewed for these reserved images.

The release seal binds the collection metadata. Every sample remains provisional,
ignored for accuracy and ineligible for promotion. The existing review guard,
AllenAI quarantine, original benchmark and locked test set are unchanged.
The collection is separate from `nl_benchmark` and is not yet a drop-in candidate
runner dataset. No classifier-score or false-alarm claim follows from its creation.

## Reproduce and verify

Raw pixels stay in the project data folder; Git stores provenance and decisions.
Copernicus access credentials are needed only to repeat acquisition.

```sh
python -m agents.reserved_sar --output data/hh_hv_transfer/fresh_native
python -m agents.hh_hv_transfer --prepared data/hh_hv_transfer/fresh_native/manifest.json --output data/hh_hv_transfer/fresh_collection
python -m agents.hh_hv_transfer --verify datasets/hh_hv_transfer/v0.1.0 --seal-sha256 52db14c54ac0fcf3f83187e2dd17d6dfc547ee30b83a9d2c2caac0ccdefe5d57
```

Add `--local` to verification to hash the retained native artifacts and extracted
chips. Rebuilding the collection replays calibration; fresh outputs never overwrite
the sealed original. Reference training data must be present to rebuild comparisons.

## What is still needed

Step 1 now has native inputs, calibration replay, metadata and descriptive value
and apparent-size comparisons. Physical compatibility remains unestablished.
Step 2 has a versioned first-pass collection, but **vessel and plausible-iceberg
coverage is still missing**. The frozen windows cannot supply those classes by
assertion. Before testing false-positive reduction, reserve further windows or
acquisitions using non-score evidence, annotate before scoring, and retain the
current cohort and exclusions. Co-temporal AIS may support vessels; independent
ice evidence and interpretable imagery are needed for plausible iceberg labels.
Without independent review, any resulting measurements must stay provisional.

Contains Copernicus Sentinel data (2025, 2026), accessed through the
[Copernicus Data Space OData API](https://documentation.dataspace.copernicus.eu/APIs/OData.html)
under the [applicable terms](https://dataspace.copernicus.eu/terms-and-conditions).
No Kaggle or AllenAI raw data is redistributed in Git.
