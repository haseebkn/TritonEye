# NL model development status

Step 5 is **partially implemented, not complete**. The project now runs a real,
frozen Sentinel-1 foundation encoder on NL development imagery. It has not
fine-tuned a vessel detector, trained a foundation-feature vessel head, or
demonstrated improved detection. Those experiments require usable vessel labels.
Independent review remains unavailable, as disclosed in the benchmark.

## Experiments and prerequisites

| Approach | Actual status | What the evidence supports |
| --- | --- | --- |
| Existing pretrained xView3 ensemble | Historical bounded replay in [NL_BASELINE.md](NL_BASELINE.md) | Processing and descriptive returns, not measured vessel accuracy |
| Fine-tuned detector | Not run; no trainer implemented | Zero confirmed training vessels in either polarization branch |
| CROMA frozen radar features | Executed on six VV/VH development areas | Checkpoint compatibility, finite features, full selected-pixel coverage and repeatability |
| Frozen-feature vessel head or encoder adaptation | Not run | No usable training vessel labels or validated head to justify adaptation |
| HH/HV model comparison | Not run | Existing models have no validated HH/HV compatibility; labels also remain unresolved |

A blocked fit is not an unsuccessful improvement experiment. The recorded GPU
execution failure below does not establish anything about detection performance.
The existing detector replay and the feature probe are not a controlled detector
comparison: the latter has no vessel head, predictions or operating threshold.

## Foundation encoder and input contract

[CROMA](https://github.com/antofuller/CROMA) is a radar and optical remote-sensing
representation model. Its [paper](https://arxiv.org/abs/2311.00566) identifies
Sentinel-1 VV/VH pretraining inputs. TritonEye uses only the radar encoder and
pooled radar projection; it does not synthesize missing optical inputs. The
adapted radar architecture preserves upstream checkpoint keys and retains the
[MIT attribution](../third_party/CROMA_LICENSE.txt).

The base checkpoint is pinned to Hugging Face revision
`0dd28e3d633bd6715856ae9890e8c49360040598` and SHA256
`0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`.
Its upstream implementation revision is
`59505a6bcadbf36ba20767270154bf9f3067c5e7`. Loading verifies the checkpoint hash
before reading tensor-only state with `weights_only=True`.

The probe verifies released calibrated chip bytes, descriptions and split
eligibility. Inputs remain **VV then VH sigma0 dB**, never renamed HH/HV. Each
512px chip is partitioned into sixteen disjoint 128px windows, with no resampling,
padding or omitted edge pixels. The per-window, per-band mean plus/minus two
sample-standard-deviation stretch follows the upstream float-normalization
example with a batch of one. Statistics are not fitted across acquisitions or
validation data. Constant or nonfinite inputs fail explicitly.

This checks an input contract, not domain equivalence. Native radar-grid SAR
differs from the geocoded land imagery used during pretraining. The 128px window
extrapolates the pretrained 120px field of view. An 8px feature token may obscure
small vessels. These risks require labelled detection experiments, not inference
from embedding appearance.

## Real feature probe results

The successful local probe encoded St. John's harbour, eastern Avalon offshore,
Bonavista coast and offshore, Lewisporte coast and Bay of Exploits water. These
are three Newfoundland acquisitions, not province-wide or Labrador validation.

- Six areas, 96 windows and 1,572,864 selected pixels encoded completely.
- Each area produced token features shaped `16 x 256 x 768` and pooled features
  shaped `16 x 768`. Features were finite.
- Repeating the first window of each area gave a maximum absolute token
  difference of zero on this run.
- Actual model-state digests matched before and after execution; all parameters
  remained frozen. There was no optimizer, gradient update or loss history.
- The run used seed `20261006`, float32, batch size one, deterministic algorithms
  and TF32 disabled, on an NVIDIA GeForce RTX 4070 Laptop GPU.
- The second attempt took approximately 9.92 seconds inside the measured block,
  including checkpoint loading and serialization. Sampled process RSS peaked
  near 1.60 GiB; CUDA allocation peaked near 230 MiB. These are feature-probe
  measurements, not full-swath runtime or training compute.

Four HH/HV areas were explicitly unsupported and not encoded. Two locked-test
areas were not encoded. This does not establish usable offshore HH/HV coverage.
Automatic alerts and production detector configuration remain unchanged.

The first GPU attempt failed at its first forward pass because strict CUDA
determinism required `CUBLAS_WORKSPACE_CONFIG`. It produced no feature files.
That failure was retained with its exact source hashes; its resource telemetry
was unavailable. The retry used `:4096:8`. Subsequent execution failures are now
persisted separately, and existing experiment folders cannot be overwritten.
Repeatability on this machine does not promise cross-device bitwise identity or
identical compressed-file hashes.

## Public annotation screening

The inspected [SARFish annotation mirror](https://github.com/John-J-Tanner/Extract-SARFish-Data)
is maintained by authors of a SARFish-based study. At revision
`9a06750051ab61ff0f8f86cf4317788295c8a909`, its GRD train file contains 64,054
rows and its validation file 19,222. Geographic screening found **zero rows
inside the project's NL study polygon in either file**. This is not a claim
that no suitable public NL dataset exists anywhere.

The CSVs and data terms are saved locally for provenance, not adopted as a
training set. SARFish's [dataset terms](https://github.com/DIUx-xView/SARFish/blob/bfef9694946d192ce7f66c5c5f97d5be364de02c/SARFish_Terms_and_Conditions.md)
state CC BY-SA 3.0 IGO, whereas its Hugging Face card lists Apache 2.0; code
licensing must not be substituted for data terms. Licensing and original-label
provenance would need resolution before adoption. No non-NL imagery was
downloaded or used for model development. No credentials were needed for these
public downloads.

### Additional published NL point candidates

The [AllenAI Sentinel vessel dataset](https://github.com/allenai/vessel-detection-sentinels/tree/60b003d059937a9fcddbf6e1c9b64c00c33c0279)
provides a second, materially different source. Its
[paper](https://arxiv.org/abs/2312.03207) describes expert human annotation of
Sentinel-1 vessel points. Screening its pinned metadata database found **73
whole image windows across 20 acquisitions inside the NL study polygon, with
39 published vessel points in 13 of those acquisitions**. Forty windows have
no point labels. Neither
those empty windows nor unmarked pixels are locally confirmed negatives.

Six acquisitions occur in more than one upstream split. Those original splits
are therefore not adopted. Repeated geographic windows also need grouping;
different acquisition dates alone cannot make overlapping tiles independent.
The screening command reads metadata only and does not bypass the frozen
benchmark's test lock, approve labels, assign a new training split or select a
threshold. All candidate records remain `metric_ready: false`.

Historical VV/VH imagery is being checked against the global Web Mercator point
coordinates, which are not raw SAR pixel offsets. Six small native-grid preview
areas were extracted from the April 21 and May 15, 2022 acquisitions. Initial
visual inspection was mixed: some points were near compact returns, while
others lacked an obvious target under the fixed preview stretch. This is an
alignment investigation, not a local expert re-annotation or proof that the
source labels are wrong. The imagery, source hashes and inspection records
remain under `data/model_research/allenai_audit/` and `data/raw/`.

The repository publishes an Apache-2.0 licence. A separate annotation-data
grant and complete-area annotation protocol have not been verified locally.
These candidates are **not yet adopted for fitting or scoring**. They replace
any assumption that suitable public NL vessel points are universally absent,
but do not resolve the benchmark's current zero-confirmed-vessel limitation.

Reproduce the candidate screen without opening imagery or model weights:

```sh
python -m agents.public_vessel_labels --output data/model_research/new_allenai_nl_screen.json
```

The CLI checks the pinned database hash, validates coordinate and point/window
contracts, excludes incomplete regional windows and annotation-helper points,
and records source, code and regional-boundary hashes. It refuses an existing
output path. Database reads are read-only. Tests exercise conversion, regional
selection, split leakage, hidden records, corrupt metadata and non-adoption.
[Public evidence](evidence/public_label_screen_20261006.json) retains the
screen's counts, hashes and non-adoption scope.

## Reproduce the implemented work

Run from the repository root with the existing project environment. Public
assets require roughly 0.8 GB of storage, excluding local benchmark imagery.

```sh
python -m scripts.download_model_research_assets --include-label-audit
python -m agents.model_research audit-labels --train data/model_research/public_label_audit/GRD_train.csv --validation data/model_research/public_label_audit/GRD_validation.csv --output data/model_research/new_geography_audit.json
python -m agents.model_research readiness --output data/model_research/new_readiness.json
python -m agents.model_research features --weights models/croma/CROMA_base.pt --output data/model_research/new_croma_probe --device cuda
```

Use `--device cpu` when CUDA is unavailable. Reuse verified existing downloaded
assets; the downloader refuses to replace an existing download record. Every
probe needs a new output folder. Local feature archives contain native-window
origins, tokens and pooled vectors. The experiment JSON records code hashes,
dataset and split hashes, checkpoint identity, settings, compute and per-area
results. Large assets stay outside Git; [public evidence](evidence/model_research_20261006.json)
retains the experiment identities and findings.

## Completing the controlled comparison

The following work is still required, not implemented training infrastructure:

1. Resolve training vessels and whole-area annotation decisions in a new
   versioned release; preserve the current frozen release and geographic locks.
   Disclose provisional training labels if independent review remains unavailable.
   Independently reviewed validation labels remain necessary for the current
   operating-point protocol. Clutter context and unlabelled water are not
   automatic negatives.
2. Obtain a reproducible trainable detector checkpoint and audit its pretraining
   overlap with development acquisitions. The exported xView3 inference
   ensemble does not by itself provide a reproducible fine-tuning recipe. Include
   a matching pretrained architecture control if fine-tuning a different model;
   do not attribute an architecture or ensemble change to fine-tuning alone.
3. Implement and run the pretrained control, detector fine-tune and frozen
   CROMA detection head on the same development splits. Record all seeds,
   optimizer settings, augmentations, epochs, selection decisions, weights,
   runtime, memory and failed or non-improving fits. Adapt the encoder only when
   frozen-head evidence justifies a predeclared additional experiment.
4. Use the same evaluated areas, 100 m matching rule, shoreline policy and
   [provisional recall, false-alarm and coverage requirements](NL_BASELINE.md).
   Validation alone selects operating points; keep the locked test untouched
   until a separately approved final evaluation. Report uncertainty and region,
   coastal/open-water and condition breakdowns, including exclusions and misses.
5. Maintain a separate HH/HV experiment with genuine band metadata, an explicit
   compatibility decision and labelled HH/HV train/validation scenes. Do not pool
   it with VV/VH to conceal unsupported coverage.

For a C-CORE portfolio presentation, the defensible claim is **SAR foundation
encoder integration with reproducible experiments and explicit evidence limits**.
The genuine supervised model-development gap remains open.
