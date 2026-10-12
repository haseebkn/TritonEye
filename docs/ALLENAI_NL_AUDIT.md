# AllenAI NL annotation and imagery audit — 12 October 2026

**A usable, versioned alignment-review subset is prepared. Approved supervised
training labels are not delivered.** The explicit CC BY download has no NL
source-window overlap; the older 2022 NL annotations have unresolved
annotation-specific permission, alignment/class questions and completeness.
The existing independent-review guard and locked final test remain intact.

For C-CORE portfolio review, this demonstrates source auditing, geospatial
interoperability, pixel replay and fail-closed dataset handling. It does not
demonstrate vessel precision, false-alarm reduction or model improvement, and
is not compliance with unavailable internal C-CORE protocols.

## Findings

| Question | Evidence and outcome |
| --- | --- |
| Is there an explicit data licence? | **Yes for the separately linked rslearn archive:** AllenAI's pinned [evaluation README](https://github.com/allenai/rslearn_projects/blob/9cfaf6e9a13fcb993c687b591eb3f71df7c144c5/rslp/olmoearth_evals/README.md#sentinel-1-vessel-detection) declares CC BY 4.0. The older [Sentinel vessel repository](https://github.com/allenai/vessel-detection-sentinels/tree/60b003d059937a9fcddbf6e1c9b64c00c33c0279) has Apache-2.0 at its root, but a separate scope statement for the 2022 annotation database was not found. This is an adoption-policy limitation, not a declaration that its annotations are legally unusable. |
| Does that CC BY archive contain the screened NL candidates? | **No overlap in the crosswalked source extents.** All 1,776 archive windows match original window IDs **and** grid origins: 1,598 `nov-2021-point-train` and 178 `nov-2021-point-val`. None intersect the NL study polygon. All screened NL candidates are from later 2022 batches. Do not transfer the newer grant to another distribution by assumption. |
| Is acquisition identity resolved? | Three exact cached SAFE names, acquisition times, VV/VH bands, source file/calibration hashes and catalogue hashes are retained. The old image UUID and current Copernicus product UUID are recorded separately. No date-only image selection or substitute scene is used. |
| Are the coordinates aligned? | Source-grid reconstruction matches the integer image origins for all three products. Corrected positions are mapped to native pixels; all 23 are inside valid calibrated QA patches. The default polynomial publisher warp differs from TritonEye's TPS convention: diagnostic displacement **2.46–115.99 m, median 39.40 m**. This is not surveyed absolute accuracy or proof of the intended object. Original publisher PNGs are unavailable for byte-identical comparison; suggested dimensions differ by at most one edge pixel. |
| Are annotations exhaustive? | **Not established.** The [publication](https://arxiv.org/abs/2312.03207) describes expert-authored point labels. The available database does not supply per-window exhaustive-review histories or resolved non-vessel/ice labels. The old training code's use of unmarked space as background is an assumption, not completeness evidence. A linked dataset-report notebook is absent at the pinned revision. |
| Are these independently reviewed local vessel labels? | **No.** Source expertise is supporting provenance; no qualified independent local reviewer is available. The point-sheet sanity inspection is AI/self QA only. Some points show compact returns; others do not show a clear isolated target at the fixed stretch. No class is silently corrected, confirmed or rejected on that basis. All 23 point records remain unreviewed. |
| Can we adopt the original train/val split? | **No.** Six of the 20 NL native acquisition groups span multiple upstream splits. No original assignments are adopted and no new training/evaluation split is invented. Pretraining overlap remains unaudited. |

The rslearn conversion also changes projection, pixel representation, historical
inputs and split assignment. Its [version notes](https://github.com/allenai/rslearn_projects/blob/9cfaf6e9a13fcb993c687b591eb3f71df7c144c5/data/sentinel1_vessels/README.md)
distinguish `20250521` from the older first-subset `20250602` and discourage the
latter for their production use. The pinned OlmoEarth task config nevertheless
uses `20250602`. Neither source is silently treated as the full 2022 NL release.

### What the archive audit actually checked

The public 9,589,923,840-byte TAR is pinned to GCS generation
`1762098182799064` and ETag `45840224423a0f3a6bd6fd7b61d373fe`.
Its 44,406 member headers were indexed with 5,319 validated range requests,
transferring 348,535,296 bytes. Range read-ahead included incidental raster
bytes that were discarded, never saved or decoded. No archive raster members
were extracted, no full archive MD5 was recomputed and no archive annotation
payload was asserted equal to an older label. Coverage is established by the
exact window-ID/origin crosswalk, not by reading converted label geometries.
The header-sequence hash and index hash are in the frozen manifest.
The old repository's [root licence](https://github.com/allenai/vessel-detection-sentinels/blob/60b003d059937a9fcddbf6e1c9b64c00c33c0279/LICENSE)
snapshot has SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.

The alternative `satlas-explorer-data` source requires requester-pays billing.
The unauthenticated request returned that requirement. No billing project,
credentials, charges or author contact were used. Public access to one source
does not confer permission or complete access to every related dataset.

## Prepared subset

The original pinned database screen has **73 whole-NL windows, 39 published
points, 20 acquisition groups and 40 unmarked windows**. Cached imagery permits
this first alignment package; the remaining 42 windows/16 points are not
represented as processed or usable imagery.

| Exact acquisition date/time, UTC | Publisher windows | Published points | Unmarked windows, not confirmed negatives |
| --- | ---: | ---: | ---: |
| 21 April 2022, 21:22:33 | 20 | 17 | 6 |
| 15 May 2022, 21:22:59 | 2 | 3 | 0 |
| 29 August 2022, 21:38:59 | 9 | 3 | 6 |
| Total across three acquisitions | **31** | **23** | **12** |

These cached examples are **Newfoundland**, not province-wide or Labrador
validation. Whole-NL screening does not mean every NL location has labelled
coverage. No AIS identity labels, detector predictions or HH/HV inputs are
included. Repeated points/windows are not independent vessel confirmations.

All 31 publisher rectangles are rendered in VV and VH. **27 are fully valid;
four contain source-edge nodata:** windows `50191`, `50203`, `29212`, `39770`.
Window `50191` has only 5.99% valid imagery; treating its black border as
negative supervision would be particularly misleading. Every band has an
explicit binary validity mask. Annotation area must be restricted to valid
pixels, not the full rectangular extent or a guessed water denominator.

Each published point has a separate 256×256 native-radar VV/VH sigma0 QA patch.
Point `155812` needs edge-clamped crop context: its label stays at its computed
position, rather than being moved to the crop centre or a bright return.
The full-resolution unmarked windows support an image-first pass; the point
contact sheet is a **second-pass alignment aid**, not a blind first-pass review.

The [dataset card](../datasets/allenai_nl/v0.1.0/DATASET_CARD.md) supplies the
annotation/review guide, attribution and adoption boundary. The
[public manifest](../datasets/allenai_nl/v0.1.0/manifest.json),
[source ledger](../datasets/allenai_nl/v0.1.0/source_evidence.json) and
[release hashes](../datasets/allenai_nl/v0.1.0/release.json) contain aggregate
provenance, not redistributed original point coordinates or imagery.

Local data, retained in the project:

```text
data/model_research/allenai_nl_v010/
  sources/                   pinned publisher source snapshots
  cc_archive_index.json       complete immutable-object TAR member index
  subset_05/manifest.json     local full alignment manifest
  subset_05/windows/         31 complete rectangles, VV/VH and validity masks
  subset_05/points/          23 calibrated native-radar QA patches
  subset_05/point_alignment_sheet.png
  verification_01.json       full pixel replay and integrity result
```

There are **217 derived artifacts**, about 49.8 MB, plus the manifests/source
snapshots. The original cached SAFE products remain in `data/raw/` and were
not downloaded again. No full foreign-image archive or new model weights were
downloaded. Local data remain ignored by Git.

## Validation and retained failures

The builder preserves source hashes, raw pixel positions and all proposed
geographic scopes. It rejects test products/datatakes/geography before their
pixels are read and refuses writes into an existing output or published version.
No frozen `datasets/nl_benchmark` file or production model configuration changed.
Training, evaluation and threshold-selection exports from this release fail
closed; changing a metadata approval boolean is not an adoption workflow.

Three implementation mistakes were caught before release:

- Relative CLI output paths could not form project-relative artifact references.
  Normalization fixes this; the first partial output is retained as `subset/`.
- A centre-based native patch ran beyond the source edge for point `155812`.
  Bounded crop placement preserves the annotation coordinate; `subset_02/` is
  retained as the failed attempt.
- The initial virtual-warp previews were blank. Explicit GCP reprojection fixes
  this, with a real pixel-valued synthetic GCP raster regression. `subset_03/`
  remains the rejected output; `subset_04/` was a valid reconstruction before
  explicit masks were added to the final `subset_05/`.

These directories are not alternative training releases. The public release
points only to the final immutable local manifest. Native GCP files initially
emit Rasterio's no-affine warning while being written; their GCPs are then
assigned and verified. An affine warning is not permission to use an identity
transform or call a blank raster valid.

Repeat the full pixel/integrity verification from the project root, choosing a
fresh output path:

```sh
python -m scripts.verify_allenai_subset --output data/model_research/allenai_nl_v010/verification_new.json
python -m pytest tests/ -q -m "not slow"
python -m ruff check agents tests
python -m black --check --workers 1 agents tests
python -m mypy agents tests
```

The verifier checks every derived artifact hash, all 15 cached source-file
hashes, replays all **62 VV/VH full-window arrays and validity masks**, checks
all 23 native patch dimensions/metadata/validity and exercises blocked adoption
requests. This verifies engineering and reproducibility, not vessel classes or
annotation completeness. Real-data verification runs only in the original
project; isolated CI/validation worktrees use synthetic fixtures and committed
public metadata, not copied private imagery or indirect data access.
The [saved verification ledger](evidence/allenai_nl_verification_20261012.json)
pins that completed replay to the public release and executing verifier hashes.
Final local software validation: **565 passed, two heavyweight tests
deselected**, plus Ruff, Black and strict mypy passing across agents/tests and
the three audit scripts. Those counts are local evidence; data-dependent tests
may skip in a clean CI checkout. CI certifies software, not annotation truth.

## What would make this usable for model training?

Request clarification from AllenAI, without representing this portfolio as
C-CORE or claiming organizational affiliation:

1. Does the Apache-2.0 grant explicitly cover the 2022 vessel point annotations
   in the pinned `data/metadata.sqlite3`, or does another data licence apply?
   Does the newer CC BY declaration cover that full distribution, not only the
   `20250602` first-subset evaluation archive?
2. Were the complete window areas exhaustively annotated, including negative
   windows? Are review records, known missing-label policies and annotation-time
   image tiles available?
3. Can the original image grid/click convention be confirmed for the three
   exact SAFE names and the uncertain point examples in this package?

No message has been sent. Author clarification can resolve provenance and
permission but does not manufacture a local qualified review. If permission
and alignment are established first, a **new version** could support exploratory
positive-only training with explicit missing-label treatment. Complete-area
accuracy, buffer/threshold selection and measured improvement still require
the existing independent-review and held-out evaluation requirements.
