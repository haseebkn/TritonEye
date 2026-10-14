# Native NL target noise audit v0.1.0

The [released numerical report](evidence/target_noise_v010.json) replays original
DN, calibration and noise metadata against retained SNAP 14 outputs for the four
previously annotated iceberg candidates. These are four development acquisitions,
one training and three validation; all interpretations remain provisional and
ignored for accuracy. No classifier scores, fitting or new class decisions were
made. The two source-supported vessel examples are outside this numerical scope.

## What happens inside the existing boxes

| Candidate suffix | Role | HV box pixels with nonpositive residual | HV surrounding-context pixels with nonpositive residual | Median HV correction inside box |
| --- | --- | ---: | ---: | ---: |
| 01 | Validation | 10.05% | 28.32% | -1.95 dB |
| 02 | Train | 28.54% | 50.26% | -4.86 dB |
| 03 | Validation | 31.58% | 51.37% | -5.83 dB |
| 04 | Validation | 11.55% | 20.65% | -2.15 dB |

The extreme whole-context median changes previously reported for 02/03 are mostly
driven by background pixels reaching the processor floor. The existing target
boxes are affected too, although their median change is much smaller. A box
includes surrounding pixels; these percentages are not the fraction of a physical
iceberg lost. The annulus is unreviewed surrounding context, not verified pure
water or a negative label. It extends 30 pixels beyond the box and excludes every
known box plus a five-pixel guard. No settings were fitted to class outcomes.

The report retains per-box and annulus quantiles for DN power, metadata noise,
residual power, observed-to-noise ratio, nominal/corrected sigma0 and correction.
It also records target/context median contrast. Very large corrected contrast
when background reaches the floor must not be interpreted as improved detection.

## Reconstruction, precision and remaining failures

Noise is evaluated at the original product row/column offsets as interpolated
range power multiplied by interpolated azimuth factors, in DN-squared units.
The independent reconstruction then subtracts noise before sigma0 calibration.
Nonpositive source DN, missing/overlapping noise coverage, incompatible headers
or an already-corrected source are refused. Original flags are false in all eight
bands. Processor versions are IPF 2.90 and 3.20. Raw SNAP and saved DN are identical;
nominal calibration differs by less than 0.000001 dB in every band.

The upstream [noise operator](https://github.com/senbox-org/s1tbx/blob/master/s1tbx-op-calibration/src/main/java/org/esa/s1tbx/calibration/gpf/Sentinel1RemoveThermalNoiseOp.java)
documents replacing negative residuals with power 1e-5. Its
[metadata reader](https://github.com/senbox-org/s1tbx/blob/master/s1tbx-commons/src/main/java/org/esa/s1tbx/commons/Sentinel1Utils.java)
stores noise LUT nodes as float32. These archived sources explain the mathematical
replay; they are not proof of source/binary equivalence for installed SNAP 14.

Float64 XML diagnostics and float32-node sensitivity are both retained. With
float32 nodes and float64 interpolation, **six of eight corrected bands pass the
unchanged 0.001 dB maximum-error tolerance**. Candidate 01 HV reaches 0.001955 dB;
04 HV reaches 0.002863 dB. Both remain explicit failures, and
`reconstruction_all_passed` is false. Their worst differences occur at almost-zero
positive residuals, about 0.000747 and 0.000125 DN-squared. Across all eight bands,
the maximum power discrepancy divided by original observed power is below 1.8e-7.
This supports numerical sensitivity near cancellation as an explanation, but the
exact remaining implementation-level difference is unresolved. The tolerance was
not relaxed. Float64 versus float32 LUT nodes change no negative-residual masks
in these contexts; their largest corrected dB differences are retained too.

The metadata therefore explains the widespread HV floor without evidence here of
a gross band, coordinate or unit mismatch. It does not prove the physical noise
estimate correct for every pixel or establish object identity. The original SNAP
execution overrun remains recorded in the [earlier audit](SNAP_CORROBORATION.md);
this replay reuses outputs and makes no new bounded-runtime claim.

## Decision and replay

The [native input specification](NATIVE_NL_INPUT.md) freezes nominal uncorrected
sigma0 for an explicitly separate research experiment. The corrected sensitivity
variant is not selected as classifier input. Noise statistics remain sidecars;
no pixels, labels or ignore masks are replaced. Kaggle equivalence, training
admission and operational false-alarm improvement remain unresolved.

```sh
python -m agents.target_noise_audit --verify
python -m agents.target_noise_audit --verify --local
python -m pytest tests/unit/test_target_noise_audit.py -q
```

Public verification checks the release hash and pinned public inputs. Local
verification additionally repeats the numerical reconstruction, original header
and DN checks, development/geographic guard and tensor checks. It verifies the
retained SNAP outputs and sidecars without network access or launching SNAP.
Historical source evidence, admission decisions and hashes are preserved. Future
changes require a new release; this audit cannot admit labels or promote a model.
