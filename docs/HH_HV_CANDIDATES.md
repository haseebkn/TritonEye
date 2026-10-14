# Experimental HH HV candidate generator

TritonEye now has a separate HH/HV local radar-background contrast generator
for the ship-versus-iceberg filtering experiment. The frozen 0.1.0 configuration
produced **1,642 unassociated candidates across eight NL development areas**.
Both comparison arms load the same saved, hash-verified population. No classifier
filter has been applied and no vessel accuracy or false-alarm rate is measured.

The existing VV/VH production detector, review guard, AllenAI quarantine, locked
test set and coastal operating policy remain unchanged. These candidates retain
land and coastal returns for subsequent shared policy annotation and research.

## Frozen algorithm

The [versioned configuration](../configs/candidates/hh_hv_local_contrast_v010.json)
was committed at `4553105` before the first generation attempt. Its byte SHA-256 is
`59360b88d1f07056d54e82e3fd6dbc33f6923f31eb6aa9797a1d36b161b5245d`.
The loader rejects changed bytes; intentional parameter changes need a new
configuration version and recorded protocol. The runner writes a definition
containing the configuration, code hashes and dataset release hash before
reading SAR arrays. Previously viewed regional images and the earlier shadow
study informed this development context; this is an engineering freeze, not
independent preregistration or a new blind evaluation.

| Parameter | Fixed value |
| --- | --- |
| Bands and units | Native HH then HV, calibrated sigma0 dB |
| Background ring | 61 × 61 pixels, excluding a central 11 × 11 guard |
| Background statistic | Arithmetic mean of **linear power**, not mean dB |
| Trigger | HH contrast ≥ 8 dB **or** HV contrast ≥ 6 dB |
| Grouping | Eight-connected threshold pixels, minimum size one pixel |
| Candidate location | Largest threshold margin; row-major tie breaking |
| Size or ranking exclusions | None; connected components are retained |
| Shoreline exclusions | None in the generator |
| ROI bound | At most 1,048,576 pixels |

For each band, contrast is `10 log10(pixel power / ring mean power)`, with
power calculated as `10 ** (sigma0_db / 10)`. A complete finite, non-nodata
background ring is required. Unsupported edges or gaps are not padded as ocean.
These heuristic contrast thresholds have no calibrated false-alarm probability;
the method is not a statistically validated CFAR operating point. Speckle,
structures, islands, ice and vessels can all trigger it.

One connected component becomes one candidate. Nearby targets can merge, and
large clutter can form large components; component boxes are not hull dimensions.
Peak pixel centres are georeferenced through the frozen raster's GCP/affine
metadata. Every candidate records its product/ROI identity, native peak, component
box, pixel count, per-band contrast, triggering bands and retained research status.
It is labelled `unassociated_candidate`, with physical surface unknown and
operational alerts false.

## Real NL replay

The successful replay used runner commit `38ed60e`; preceding fixes introduced
a separate bounded NumPy memory policy and passed each exact ROI, rather than
the whole swath, to the existing locked-geography guard. The guard itself was
not changed. Native source and LUT hashes and calibration replays use the same
audited path as the [frozen classifier transfer study](STATOIL_TRANSFER.md).
All sixteen band replays matched the frozen chips at 0.0 dB maximum difference.
No classifier checkpoints, annotation boxes, class labels or AIS observations
are used to generate candidate locations.

| Area | Candidates |
| --- | ---: |
| Ferryland coast | 386 |
| Southern Avalon offshore | 40 |
| Makkovik coast | 194 |
| Makkovik offshore | 5 |
| scene_03_area_01 | 638 |
| scene_03_area_02 | 6 |
| scene_05_area_01 | 11 |
| scene_05_area_02 | 362 |
| Total | **1,642** |

Eight areas processed, zero failed. Two locked test areas and eleven VV/VH areas
were excluded before image access. A 30-pixel border lacks full background-ring
support: each 512 × 512 area has 452 × 452 evaluated pixels. Total support is
1,634,432 of 2,097,152 pixels, **77.94%**. This is pixel processing coverage, not
evaluated water area or vessel recall; the unevaluated edge band cannot be
counted as empty water.

Of the candidates, 1,552 have a geometrically available 75 × 75 classifier
context, 89 lack ROI-boundary context and one connected component exceeds the
classifier chip size. All 1,642 remain in the shared population. Context
availability establishes neither input-domain compatibility nor a correct class;
later filtering must retain abstentions. The high coastal counts are raw returns,
not confirmed false positives. Classifier filtering and threshold comparisons
remain future work requiring provisional labels and explicit uncertainty.

The preserved successful run is `data/hh_hv_candidates/v010_20261013_03/`.
Earlier attempts are retained: the first stopped at the training-memory preflight;
the second left two partial area files when whole-swath geography was refused.
The partial attempt has no complete bundle and is not used for comparison.
The successful replay took 9.36 seconds locally, including integrity checks and
calibration replay; cache conditions prevent treating this as a latency bound.

## Identical comparison inputs

The [aggregate evidence](evidence/hh_hv_candidates_v010_20261013.json) binds the
complete `bundle.json`, frozen definition and eight candidate GeoJSON files by
hash. Candidate coordinates remain in local storage; the public ledger contains
counts, coverage, source hashes and the population digest. Contains modified
Copernicus Sentinel data (2025, 2026).

```python
from pathlib import Path
from agents.hh_hv_candidate_run import paired_inputs

arms = paired_inputs(
    Path("data/hh_hv_candidates/v010_20261013_03/bundle.json"),
    "88ca06cd2c01e84780c72edc6d2fe6c35c1558b51e412683552be297dced17cc",
)
assert arms["unfiltered"] == arms["classifier_filter"]
```

Use the expected bundle hash recorded in the versioned evidence, rather than
recalculating an expected hash after editing inputs. The loader rejects altered
candidate bytes, inconsistent settings/identities, duplicate IDs and incomplete
populations. It returns independent copies of the same collections so annotating
one arm cannot change the other. Both arms must use this population; do not
regenerate candidates separately or discard unscorable returns before comparison.

## Reproduction and resource limits

```sh
python -m agents.hh_hv_candidate_run --dataset datasets/nl_benchmark/v0.2.0 --output data/hh_hv_candidates/NEW_run
python -m pytest tests/unit/test_hh_hv_candidates.py -q
```

Use a fresh output path and authorized local development imagery. Configuration
and checkpoint-free NumPy/SciPy processing are separate from the Torch classifier.
The runner requires 1 GiB available commit and physical memory, 1 GiB free
project/temp storage, and checks a 512 MiB process RSS budget and 512 MiB commit
floor during processing. Checks bound the workload but cannot guarantee stability
when other applications allocate memory between samples. Resource limits do not
change the classifier's existing safeguards or detection thresholds.

Tests exercise analytic per-band contrast, exclusion of the guard region,
connectivity/merging, retained oversized components, nodata/edge coverage,
georeferencing, frozen-config tampering, memory refusal and byte-identical paired
inputs. Generated fixtures are used; no locked test imagery or Kaggle training
data enters these tests.
