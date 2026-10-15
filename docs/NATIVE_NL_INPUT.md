# Frozen native NL HH/HV experimental input v0.1.0

Subsequent [C/D v0.2.0 processing specification and acceptance checks](CD_INPUT_ACCEPTANCE.md)
use a separate consumer. The historical A/B specification below remains unchanged.

The [machine-readable specification](../configs/native_nl/input_v010.json) and
[executable consumer](../agents/native_nl_input.py) define a reproducible native
research input. The consumer pins the specification hash and refuses silent edits.
It produces research tensors or explicit abstentions; it grants no training or
promotion approval. The [label-admission register](LABEL_ADMISSION.md) remains on
hold and independent review remains unavailable.

| Property | Frozen choice |
| --- | --- |
| Sensor / acquisition | Sentinel-1 A/B, IW GRD, native 10 m range and azimuth sampling |
| Channels | HH then HV, calibrated sigma0 dB |
| Calibration | `20*log10(DN/sigmaNought_LUT)` with original offsets and metadata |
| Noise | Preserve existing nominal uncorrected pixels; retain metadata noise diagnostics separately |
| Crop | Complete native 75x75 context centred at integer `((x0+x1)//2, (y0+y1)//2)` of half-open annotation/candidate box |
| Array | Two channels, 75x75, float32, copied from original values |
| Transformations | Identity normalization; no clipping, padding, resizing, speckle filter or terrain correction |
| Metadata | Exact product, acquisition/group/split, window, hashes, spacing, mode and incidence retained by calling audit |
| Invalid inputs | Abstain for missing/incompatible metadata, nonfinite values, oversized target or incomplete crop; test-role requests are rejected |

Sampling is not effective resolution. This context nominally spans 750 m along
each native sampling axis; the bright component does not measure physical hull
or iceberg extent. No physical equivalence to Kaggle has been established.
Incidence must be finite and between 0 and 90 degrees; this is input validity,
not a validated model operating range. The four diagnostic centres span about
32.64–39.67 degrees, not a province-wide or all-condition qualification.

## Why this choice and what it permits

The recipe preserves the existing nominal native pixels, with identity
normalization rather than fitted distribution adjustments. It was not selected
by classifier scores, class accuracy or validation performance. The diagnostic
images and provisional boxes had already been inspected; this history is retained.
The [noise audit](TARGET_NOISE_AUDIT.md) identifies substantial below-noise HV
content and two small maximum-error tolerance failures. Future experiments must
account for that uncertainty; this contract does not make corrected HV reliable
or assert uncorrected values are optimal.

All four existing candidate boxes yield complete research tensors; their hashes
are recorded in the audit. This establishes executable layout and extraction
within that scope. It does not establish class validity, universal preprocessing
compatibility, or permission to train on these objects. The isolated consumer
expects verified sigma0 and metadata: its checks cannot authenticate arbitrary
arrays. The calling audit verifies raw pixels, calibration, source hashes, NL
scope and the existing development/locked-test guard before extracting tensors.

The existing Kaggle classifier retains its original normalization and input
contract. This separate recipe is not passed to that model as a supported input.
Any future fit needs admitted training-role labels and its own versioned run,
separated by acquisition/geography/repeated object. If training normalization,
noise correction, channel substitution or resizing changes, release a new
implemented recipe and freeze it before validation comparisons. Do not move
validation examples into training to balance classes.

No production configuration, detector population, review requirement, quarantine,
locked test, annotation or model selection changed. Vessel detection and SAR–AIS
correlation remain the project objective; this input contract supports the
experimental ship/iceberg confounder study.
