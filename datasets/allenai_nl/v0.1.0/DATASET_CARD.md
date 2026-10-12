# AllenAI NL alignment-audit subset v0.1.0

**Status: quarantined alignment/review evidence, not approved training labels or
independent evaluation truth.** This release does not complete roadmap steps
3–5 and does not establish improved vessel detection.

The [aggregate manifest](manifest.json), [source snapshots ledger](source_evidence.json)
and [release hashes](release.json) identify the data, methods and limits. Full
source point coordinates and derived images remain in the ignored project data
folder; the repository does not redistribute the older annotation database.

## Sources and scope

The original AllenAI Sentinel vessel repository is pinned to
`60b003d059937a9fcddbf6e1c9b64c00c33c0279`. Its metadata database checksum is
`d3a8bbd6c6685a6f8bb685f122b37fda613649d6da4c450e3240cace353ca2ac`.
The whole-NL source screen contains 73 windows and 39 published vessel points
across 20 native acquisition groups. Forty windows have no vessel points;
absence of labels is **not** an independently confirmed negative annotation.

The available imagery subset uses exact cached Sentinel-1 VV/VH products from
21 April, 15 May and 29 August **2022**, not the prospective 2026 AIS scenes.
The complete selected publisher windows are included, not only positive crops.
The manifest records the actual prepared counts and any rejected areas.
This release renders 31 windows and 23 point patches: 27 window rectangles are
fully valid and four contain source-edge nodata. Each band has an explicit
validity mask; invalid pixels are not annotation area or negative examples.
No non-NL imagery is adopted. The polygon defines this project's study area,
not an authoritative provincial maritime boundary. HH/HV is not included.

## Licence decision

AllenAI explicitly grants CC BY 4.0 for the
[rslearn Sentinel-1 evaluation download](https://github.com/allenai/rslearn_projects/blob/9cfaf6e9a13fcb993c687b591eb3f71df7c144c5/rslp/olmoearth_evals/README.md#sentinel-1-vessel-detection).
That separately distributed archive is pinned by GCS generation, ETag and
length, and its complete header index is crosswalked to the original database
by window ID **and** column/row. The manifest records regional overlap and
the older batch identities. Archive label contents have not been asserted
identical to the older database; header identity is not label-content identity.

The older repository has an Apache-2.0 root licence. A separate explicit scope
statement for its 2022 annotations was not found. This audit does **not** declare
those annotations unlicensed or legally unusable; it records that the
annotation-specific permission required by this project's adoption policy is
still unverified. The newer CC BY grant is not silently transferred to them.
Sentinel imagery permission is a separate matter under the Copernicus data
terms. No billing account was used for requester-pays storage.

Attribution: Allen Institute for AI, Sentinel vessel identification dataset,
original source repository and associated *Large-Scale Multi-Sensor Remote
Sensing for Maritime Vessel Identification* paper; rslearn Sentinel-1 vessel
evaluation release. Source URLs and exact revisions are retained in the ledger.
Changes: NL-only screening, source-grid reconstruction and local alignment QA;
no publisher endorsement is implied. The CC BY source declaration applies to
that archive, not to this entire mixed-evidence package.

## Alignment method and limits

1. Verify exact cached product names and file hashes, VV/VH metadata and
   calibration LUTs. Retain the old image UUID separately from the current
   Copernicus product UUID; do not pretend those identifier systems are equal.
2. Reconstruct the old zoom-13 Web Mercator image grid with GDAL's default
   polynomial GCP warp and bilinear resampling. Check integer image origins
   against the original metadata. Preserve the fractional origin; do not snap
   points to bright returns. Suggested dimensions may differ by one edge pixel
   from the stored publisher dimensions, which are used for window extents.
3. Render complete publisher-window DN previews, including unmarked windows.
   They reproduce the old clipping convention, **not calibrated sigma0** and
   not byte-identical publisher PNGs. Original publisher tile bytes were not
   available for a direct pixel comparison.
4. Map each published point through the reconstructed polynomial grid into
   native SAR pixels. Prepare separate calibrated native VV/VH QA patches,
   using TritonEye's TPS georeferencing. Mark the computed floating-point
   position, not a hard-coded patch centre. Record transform-convention
   displacement as a diagnostic, **not measured absolute geolocation error**.
   Source-edge crop context may be clamped inside the raster without moving
   the annotation itself; point `155812` uses that rule.
5. Check proposed publisher areas and native patch extents against NL and all
   committed locked-test product, datatake and geographic exclusions **before
   reading those pixels**.

Integer click conventions, GDAL version differences, unresolved absolute SAR
geolocation error and the unavailable original annotation-time tiles remain
limits. A bright point can be a vessel, structure, clutter or something
unresolved. Coordinate consistency is not object-class verification.

## Annotation/review guide and history

First inspect all full-resolution `windows/<window_id>/publisher_dn_vv.png`
and VH counterparts without point coordinates, AIS or detector output. Record
every vessel-like return, fixed structure, clutter/ice and unresolved object,
including possible misses. Do not review only the positive patches. The
point-indexed contact sheet and native patches are **second-pass alignment
checks**, not a detector-blind first-pass or an independent annotation review.

The original publication describes expert-authored point labels. This audit
does not invent reviewer identities, per-window exhaustive-review events,
confirmed non-vessel labels, vessel hull boxes or cross-date AIS identities.
No qualified independent local reviewer is available. Prepared point records
remain `unreviewed_not_confirmed_vessel`; no local AI/self-review event is
promoted to independent truth. Rendered area completeness and annotation
completeness are separate fields.

The old training code treats unmarked space as background and constructs
synthetic boxes around points. That is a modelling assumption, not proof that
all vessels were found. An older linked dataset-report notebook is missing at
the pinned revision. The available evidence cannot certify exhaustive labels.

All areas remain `quarantined_qa`. No train/validation/test assignments are
adopted from the publisher: six of the 20 NL acquisition groups occur across
multiple upstream splits. Any future adoption needs a **new version**, explicit
label permission, alignment decisions, a missing-label strategy and acquisition/
geographic split checks against the existing benchmark. Encoder/detector
pretraining overlap is not established by this audit.

## Using the local package

See [the audit report](../../../docs/ALLENAI_NL_AUDIT.md) for paths and replay
commands. `agents.allenai_subset.load_alignment_subset` verifies public and
local manifests and every derived artifact. It accepts alignment-review use
only; training, evaluation and threshold-selection requests fail closed.
The existing benchmark independent-review gate is unchanged. Locked test
pixels were neither read nor modified for this audit.

Fitting or scoring must not proceed by changing approval booleans, treating
empty windows as negatives, adopting leaked upstream splits or adjusting
labels until the current detector agrees with them.
