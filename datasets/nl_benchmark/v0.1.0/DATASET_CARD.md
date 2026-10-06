# TritonEye NL SAR review pilot v0.1.0

## Intended use and honest status

An acquisition-independent historical annotation workflow for the Newfoundland
and Labrador portfolio project. **Provisional machine-assisted labels; no
independent reviewer available**, as confirmed by the project owner on
2026-10-05. This release cannot substantiate low false positives or qualify as an
independently reviewed operational benchmark. Accuracy remains unmeasured.

Six real Sentinel-1 acquisitions, twelve complete 512 × 512 selected windows
(3,145,728 valid dual-band pixels), from winter, summer and autumn. Both
Newfoundland and Labrador have coastal and offshore selected areas. These are
purposive small samples; the province, all weather/sea states, vessel sizes,
operating locations and ice conditions are **not** comprehensively represented.
Season is from acquisition date, not a measured sea-state or ice label.

| Split | Acquisition | Geographic groups | Bands |
| --- | --- | --- | --- |
| Train | 2025-01-03 | Southern Avalon / Ferryland | HH/HV |
| Train | 2026-08-29 | Eastern Avalon / St. John's | VV/VH |
| Validation | 2026-08-17 | Bonavista | VV/VH |
| Validation | 2026-09-27 | Bay of Exploits / Lewisporte | VV/VH |
| Validation | 2026-07-06 | Makkovik, Labrador | HH/HV |
| Locked test | 2026-02-09 | Hopedale, Labrador | VV/VH |

Full UUIDs, UTC start/end times, polygons, hashes, source filenames, pixel-window
offsets, polarization and per-band dB summaries are in `manifest.json`.
Selection and the split lock predate annotation. Training/validation were
deliberately separated geographically; the older Newfoundland scenes have
documented development exposure. No untouched-test claim is made for them.

## Sources, licences and preparation

Contains modified Copernicus Sentinel data (2025, 2026). The
[Sentinel legal notice](https://cds.climate.copernicus.eu/licences/ec-sentinel)
governs the imagery; an exact downloaded HTML licence snapshot and its hash are
saved under the local dataset data directory. The catalogue supplies exact
product identities, acquisition times and footprints. The original SAFE
manifest, dual-band measurements and calibration XML hashes are retained.
Catalogue checksums identify original provider objects; local SHA-256 values
identify the actual files used. No synthetic data or Boston imagery is included.

Fixed windows use product LUT-calibrated sigma-nought dB and WGS84 GCP/TPS
geolocation. Native radar geometry is preserved: **no terrain correction,
thermal-noise removal, incidence normalization or independent geolocation
validation** is claimed. Window hashes and calibration offsets support replay.
HH/HV is retained as an annotation domain, not renamed or fed into the current
VV/VH-only pretrained detector. Cross-polarization and incidence differences
must be handled explicitly in future training.

Private AISstream recordings are checked in ±5 minutes around each acquisition
and against swath/ROI polygons, frozen locally, hashed and reported only as
support availability. They are not redistributed or labelled as openly licensed
ground truth. No recording is required to annotate historical imagery. Missing
AIS and out-of-swath observations do not create negative labels. See the final
manifest for exact availability counts; no co-temporal supporting observations
are promised merely because the recorder was running.

## Annotations and limits

The schema supports vessels, fixed structures, clutter, ice and unresolved
objects. First-pass boxes are drawn from whole areas with no detector overlay.
Neither plausible harbour brightness nor a winter date forces a vessel/ice
label. Unresolved clusters and contextual clutter extents are explicitly
retained. The first pass is not expert exhaustive truth; confirmed positive
vessel examples and independent adjudication remain requirements for useful
training and quantitative evaluation. A lack of vessel labels does not establish
that a scene contains no vessels. Review status and class coverage are computed
from the annotations rather than inferred from the schema.

The first-pass reviewer is recorded as `codex_primary_machine_assisted`, not a
human SAR expert. It is not independent of implementation. The absence of an
independent reviewer is in `review_history.json`; no second agent substitutes
for one. Read the annotation guide before editing. Versioned release hashes
detect changed annotations/reviews, and review digests detect stale approval.

## Reuse and verification

From the repository root:

```sh
python -m agents.nl_benchmark
python -m agents.nl_benchmark --verify-files --output data/benchmarks/nl/0.1.0/verification.json
```

Small metadata/annotation/guide/review/lock artifacts are versioned in Git. Large
SAFE products, calibrated chips, previews, licence and AIS snapshots stay in
`data/` in this project folder. They are content-addressed, **not** claimed to
be DVC-managed. The initial builder is `python -m scripts.build_nl_benchmark`;
Copernicus credentials are needed only for uncached imagery and must stay in
`.env`. A released directory is immutable; create a copy/new version before
materializing or revising it. Do not overwrite a released manifest/lock to make
hash checks pass. The review bundle is generated locally without online tiles,
AIS identities or detector predictions.

Independent review and resolved validation objects are required before exporting
one area to coastal-policy comparison. Test export is unavailable. Future work:
expert full-area review, confirmed vessel positives across sizes/conditions,
additional acquisitions/geographic groups, upstream-pretraining overlap audit,
and a predeclared final-test evaluation protocol with confidence intervals.
