# AllenAI adoption checks

The 23 cached Newfoundland publisher points now have a sealed AI-only second
pass, and the native radar calibration has been replayed for each QA patch.
**Training adoption remains blocked.** Annotation-specific permission,
exhaustive annotation, intended-object alignment and complete native model
contexts are not established. No independent SAR reviewer is available.

The [clarification request](ALLENAI_CLARIFICATION_REQUEST.md) is addressed to
Skylight's published support route and remains unsent. Support can route the
questions to the dataset maintainers; it has not granted permission.

## Publisher point decisions

All 23 unchanged points were inspected in four local sheets. Each sheet pairs
publisher-grid clipped-DN VV/VH crops with native calibrated VV/VH patches.
The markers retain the original coordinates in each representation. No detector
predictions, AIS or snapping to bright returns was used.

| Second pass state | Points | Interpretation |
| --- | ---: | --- |
| Usable provisional | 5 | Within previously sealed vessel-like regions, with visual consistency across representations. No independent class or absolute geolocation confirmation. |
| Unresolved alignment | 4 | Intended target or cross-grid placement cannot be established. Coordinates stay unchanged. |
| Unresolved class | 14 | Insufficient evidence to distinguish vessel from clutter, coastal structure or other responses. Not rejected source labels. |
| Unusable | 0 | No source point was declared definitively unusable. |

The annotator was the Codex AI assistant. Prior publisher-label exposure and
current second-pass display exposure are declared. Five provisional consistencies
are not five approved training examples or new benchmark vessels.

The second pass preserves the first pass and adds only ignore polygons around
the 18 uncertain points. Each new context extends 64 publisher pixels from its
point, clipped to the area. This is conservative review context, not hull extent
or an estimated sensor-error bound. All 31 class/ignore mask pairs replay from
the recorded decisions. Frozen datasets and original source labels are unchanged.

## Annotation completeness

The 31 areas span three acquisitions and include 12 windows with no publisher
points. Their absence does not establish negative imagery. The full-area first
pass has recorded AI inspection and explicit background/ignore regions, but
neither local independent completeness nor upstream exhaustive annotation is
verified. Difficult source points remain unresolved; no false-alarm or recall
measurement follows from this pass.

Unresolved objects, alignment context and nodata stay ignored. The quarantined
release supplies no admitted positive or negative supervision. Clarification
must establish upstream completeness or a documented missing-label policy;
independent complete-area validation remains required for accepted accuracy and
promotion claims. A future permitted training-only release must state its own
supervision policy without relaxing the benchmark review guard.

## Model input compatibility

The primary xView3 adapter expects native 2048 by 2048 sigma0 dB arrays,
normalized as `sigmoid((dB + 20) * 0.18)` in **VH, VV** order.
Publisher previews are clipped uint8 DN in a reconstructed Web Mercator grid.
Treating their pixel values as dB would violate that contract. Clipping different
original amplitudes to 255 loses information; a display PNG cannot be inverted
into original DN and calibrated with a guessed scale.

The input audit checks all 62 whole-band previews and replays the original native
DN-to-sigma0 calculation for all 23 point patches using their actual window
offsets and polarization-specific LUTs. It verifies patch hashes, validity,
band descriptions, finite calibration and the normalization arithmetic.
The public ledger records the replay error and display-ceiling counts.
For this replay, stored float32 native calibration matched exactly
(maximum absolute difference **0 dB**). Across 60,464,032 joint-valid band-pixel
exposures, 822,573 (**1.36%**) equal the display ceiling of 255. This counts
ceiling values, not proven vessel saturation or the exact amount clipped.

These 256 by 256 patches establish a reproducible calibration path, not full
model input compatibility. No native 2048-pixel contexts, training patches or
detector predictions were prepared. Larger contexts must pass the geographic
split guard, preserve native geometry and validity, and transfer only defensibly
aligned labels. Resizing or padding a QA patch is not a validated substitute.
The publisher-polynomial/native-TPS displacement remains approximately
2.46 to 115.99 m; no independent absolute geolocation accuracy is claimed.

## Evidence and reproduction

The [aggregate adoption ledger](evidence/allenai_adoption_20261012.json) records
the source, first-pass, second-pass and input-audit hashes, counts and blockers.
It publishes no annotation coordinates or imagery. Local artifacts remain under
`data/model_research/allenai_curation/`:

- `alignment_20261012_01/`: original point template and four inspected sheets.
- `second_pass_20261012_01/`: sealed decisions and 62 class/ignore masks.
- `input_audit_20261012_03/audit.json`: complete input diagnostics.

This ledger preserves the actual pre-optimization run, whose executing verifier
and code hash remain available at submitted commit `a74d852`. The subsequent
compatibility optimization caches verified source paths by exact path and expected
SHA256 within each call. It has synthetic regression coverage only; no new
real-data replay is claimed, and the existing ledger and frozen bytes are unchanged.

Choose fresh outputs when reproducing; existing packets, seals and releases
cannot be overwritten:

```sh
python -m agents.allenai_adoption check --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --first-pass data/model_research/allenai_curation/first_pass_20261012_01 --second-pass data/model_research/allenai_curation/second_pass_20261012_01 --output data/model_research/allenai_curation/NEW_point_check
python -m agents.allenai_adoption inputs --workspace data/model_research/allenai_curation/v0.1.0_20261012_04 --first-pass data/model_research/allenai_curation/first_pass_20261012_01 --output data/model_research/allenai_curation/NEW_input_check
python -m pytest tests/unit/test_allenai_adoption.py tests/unit/test_allenai_curation.py tests/unit/test_allenai_subset.py -q
```

Real pixel checks use authorized local inputs in the project checkout. Isolated
software validation uses synthetic imagery and public metadata. The existing
review guard, quarantine, locked test set and production configuration remain
unchanged. For the C-CORE portfolio, these are data-adoption and preprocessing
checks; no detector improvement or operational qualification is claimed.

Local validation passed 637 tests with two heavy tests deselected, plus Ruff,
Black and strict Mypy across 108 files. The 18 new adoption regressions exercise
calibration replay, input ordering, rehashed point/mask corruption, exposure
declarations, conservative uncertainty, aggregate output and the held-out guard.
Software validation does not establish scientific label correctness.
