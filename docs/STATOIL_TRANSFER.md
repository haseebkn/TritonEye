# Statoil classifier transfer to real NL SAR imagery

The frozen ship-versus-iceberg ensemble processed eight Newfoundland and Labrador
HH/HV development areas on 13 October 2026. Fifteen existing object centres were
scored. Both provisional vessel interpretations received ship predictions; five
unresolved objects received iceberg predictions. There are no independently
reviewed labels or labelled icebergs in this selection, so classification accuracy,
false-alarm reduction and vessel detection improvement remain unmeasured.

This is a provisional input audit and shadow experiment. Every return is preserved.
The [original Kaggle classification baseline](STATOIL_BASELINE.md) remains a
separate task. TritonEye's current VV/VH detector still requires a compatible
candidate generator or separate VV/VH adaptation before this classifier could
participate in a detection comparison.

## Selection and input compatibility

The unchanged v0.2.0 release supplies four HH/HV acquisitions, eight development
areas, thirteen unresolved object interpretations and two provisional vessels.
The two locked Hopedale test areas are excluded before image hashes, rasters or
native annotation XML are opened. Eleven VV/VH development areas are excluded.
The original review guard, AllenAI quarantine and model operating configuration
are unchanged. Existing labels were machine assisted and previously viewed;
this scoring exercise is not an independent or new blind annotation pass.

| Acquisition UTC | Region | Development areas | Object centres |
| --- | --- | ---: | ---: |
| 3 January 2025 09:42:20 | Newfoundland, Ferryland and southern Avalon | 2 training | 3 unresolved |
| 6 July 2026 10:11:53 | Labrador, Makkovik | 2 validation | 2 unresolved |
| 2 October 2026 09:41:05 | Newfoundland, St. John's selection | 2 training | 4 unresolved, 1 provisional vessel |
| 18 September 2026 09:56:31 | Northern Newfoundland | 2 validation | 4 unresolved, 1 provisional vessel |

The native annotation XML records IW acquisition mode, 10 m range and azimuth
pixel spacing, and mid-swath incidence angles from 38.79 to 38.95 degrees. These
are pixel spacing and scene metadata, not effective spatial resolution or the
exact incidence angle of each object. A 75 by 75 native chip spans approximately
750 by 750 m in the raster sampling grid. The original Kaggle chip spacing is
unknown, so physical target scale cannot be declared equivalent.

Both bands are explicitly identified as calibrated HH then HV. Original raster,
calibration LUT and frozen ROI hashes are checked. Sigma0 calibration is replayed
at the exact original row and column offsets across each 512 by 512 ROI. All
sixteen band comparisons have a maximum absolute difference of **0.0 dB** after
the same float32 conversion. Native annotation hashes and calibrator code hashes
are saved. This verifies the project's calibration replay; it does not recover
the publisher's unknown preprocessing.

Project imagery has no additional terrain or thermal-noise correction in this
study. Native annotation records thermal-noise correction as false. The model
uses the original training-only band means and standard deviations. There is no
resizing, padding, per-chip stretching, polarization substitution or refitting.
The entire native ROI and each object chip have min/1st/median/99th/max dB values
and fractions outside the Kaggle 1st-to-99th percentile interval recorded.

Coastal scenes show substantial value differences: 27.28% of Ferryland HV pixels,
40.77% of Makkovik coast HV pixels and 37.88% of the St. John's coastal ROI HV
pixels lie outside that reference interval. The two provisional vessel ROIs have
much smaller HV fractions, 0.44% and 1.01%. These describe mixed scene content;
percentile overlap is neither proof of compatibility nor a calibrated rejection
rule. All scores retain the status `provisional_transfer_only`.

## Frozen scoring and observed outputs

The validation-selected three-model ensemble, seeds 20261013 to 20261015, is
loaded with its original saved checkpoints. The published Kaggle ledger binds
the full training report, split, normalization, parameters and predictions by
hash. The shadow loader checks all original artifact hashes before inference,
loads weights on CPU with `weights_only=True`, disables gradients and uses
evaluation/inference mode. A regression compares the loaded architecture's
outputs with the original training model. Parameter hashes agree before and
after scoring.

Each existing object box determines a native 75-pixel crop centred on its
floor-rounded midpoint. Oversized boxes, boundaries and nodata cause explicit
abstention rather than padding or recentering. None of the fifteen selected
objects required abstention. Three area-level clutter annotations are ignored;
they are context, not centred objects or confirmed negative examples. The two
offshore areas without object annotations are audited but supply no classifier
targets; this does not establish that they contain no vessels.

| Object | Existing provisional interpretation | Ensemble iceberg score | Prediction at 0.5 |
| --- | --- | ---: | --- |
| scene_03_area_02_001 | Vessel | 0.01202 | Ship |
| scene_05_area_01_001 | Vessel | Approximately 0.00003 | Ship |
| Remaining 13 objects | Unresolved | Full scores in the evidence ledger | 8 ship, 5 iceberg |

Six objects have ensemble members on different sides of 0.5. Scores close to
zero or one are still uncalibrated under this transfer; agreement does not prove
the class. The binary head has no clutter, fixed-structure or unresolved class.
The five iceberg predictions therefore cannot be counted as detected icebergs,
removed false alarms or ground truth. The two ship predictions are descriptive
agreement with provisional interpretations, not measured recall.

There were eight processed areas, zero failed areas, fifteen scored centres,
zero abstentions, zero suppressed returns and zero independently reviewed
objects. Accuracy, false alarms per water area and change in vessel recall are
explicitly null; promotion eligibility is false. The reported runtime covers
hashing, calibration replay and scoring on this local CPU execution; cache and
environment conditions prevent treating it as an operational latency bound.

## Evidence and reproduction

The [published shadow ledger](evidence/statoil_shadow_20261013.json) contains
native-scene scores and distributions, input/code/model hashes, frozen
normalization, excluded area IDs and parameter integrity evidence. Kaggle
pixels, original chip IDs, training predictions and weights remain in ignored
local storage. Contains modified Copernicus Sentinel data (2025, 2026).

The preserved final run is `data/statoil_shadow/20261013_native_03/shadow.json`.
Earlier development replays remain local; scoring settings and class decisions
were not tuned in response to them. These are repeated observations of the same
images, not additional independent examples. The original classification
holdout and original training hashes were not rewritten.

Use the existing CPU-only environment and authorized local model/data assets,
choosing fresh output and evidence names:

```sh
python -m agents.statoil_shadow --dataset datasets/nl_benchmark/v0.2.0 --output data/statoil_shadow/NEW_run --evidence docs/evidence/NEW_shadow.json
python -m pytest tests/unit/test_statoil_shadow.py -q
```

The runner validates the versioned dataset contract, refuses existing outputs
and applies the classification study's memory/storage preflight and a 2 GiB
process budget. A failed area is recorded and produces a nonzero CLI exit.
Behavioral tests cover exact native calibration replay, input tampering,
HH/HV descriptions, crop abstentions, unchanged parameters and labels, locked
image exclusion, publication guards and frozen-checkpoint integrity. Synthetic
fixtures are used; unit tests do not train on or open the real Kaggle archive.

## What this enables next

A useful next comparison requires a permitted HH/HV candidate generator and
complete selected-area labels, including missed targets and clutter. Compare
identical candidates with and without classifier filtering, selecting thresholds
on validation data under both false-alarm and useful recall/coverage constraints.
Retain unresolved returns and measure vessels wrongly rejected alongside
non-vessel returns removed. Real iceberg examples with defensible class evidence
are still needed. With no independent reviewer, that comparison must stay
exploratory and promotion must remain blocked.
