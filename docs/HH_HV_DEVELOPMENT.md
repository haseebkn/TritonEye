# Additional HH HV development acquisitions

Eight additional NL acquisitions are reserved **before SAR image inspection or
scoring**. They provide sixteen planned coastal/offshore processing envelopes,
not sixteen downloaded or annotated chips. The original eight exposed workflow
areas and the original locked test set retain their existing roles. This is a
development reservation, not a replacement test set or evidence of improvement.

## Reserved population

| Geographic group | Region | Development split | Winter acquisition UTC | Summer acquisition UTC |
| --- | --- | --- | --- | --- |
| Burin shelf | Newfoundland | Train | 2025-02-25 09:50:22 | 2026-07-03 09:49:24 |
| Cartwright shelf | Labrador | Train | 2025-01-01 09:57:03 | 2026-07-02 09:56:07 |
| Southwest Newfoundland | Newfoundland | Validation | 2025-01-06 10:06:48 | 2026-07-01 10:05:45 |
| St Lewis shelf | Labrador | Validation | 2025-01-13 09:57:02 | 2026-07-08 09:56:02 |

Each acquisition covers both complete envelopes for its group. All eight are
online IW GRDH HH/HV products at reservation time. Exact catalogue UUIDs,
SAFE names, timestamps, footprints, orbit metadata and downloaded metadata
snapshots are in [the sealed reservation](../reservations/hh_hv/v0.1.0/manifest.json).
The [sampling policy](../configs/reservations/hh_hv_development_v010.json)
selects the earliest eligible acquisition in each epoch without using AIS
availability, annotations, detections or classifier scores. Native datatakes
cannot be reused between assignments, including adjacent frames or COG aliases.

Winter and summer are **sampling proxies**, not verified ice or sea-state labels.
All selected passes are descending. Relative orbit numbers are 25, 54 and 127;
winter uses Sentinel-1A, while summer uses Sentinel-1C/1D. Season, year and
platform are therefore confounded. Subsequent filter comparisons must remain
paired within acquisitions and report these strata; they cannot establish
ascending-pass robustness or a causal seasonal effect.

## Geographic separation and exposure

Train and validation share neither geographic groups nor native datatakes.
The policy requires 10 km separation between groups and from recorded exposed
areas, including the original benchmark and quarantined AllenAI windows. In
EPSG:3347, the minimum planned-envelope distances are 120.76 km between new
groups and 33.14 km from prior areas. Satellite swaths can overlap; these distance
checks concern the **reserved processing envelopes**, not whole-scene footprints.

Each envelope is a WGS84 rectangle extending 0.1 degrees in longitude and 0.065
degrees in latitude from its planned centre. Native pixel windows must later fit
inside these envelopes. Existing OSM coastline vectors indicate mixed land/water
in the coastal envelopes and water-dominant offshore envelopes. These are
sampling strata, not new imagery-validated shoreline controls. Physical water
boundaries still require image-first QA; no coastal alert buffer has changed.

The prior-exposure inventory records known local product names and geographic
extents. It conservatively excludes cached native datatakes and the 31 prepared
AllenAI areas without using their labels. It cannot certify access that was never
recorded. The original eight HH/HV workflow areas have already been inspected
and remain workflow checks, not fresh validation evidence. AllenAI's quarantine,
permissions, completeness limits and review restrictions are unchanged.

Natuashish was initially considered as the second Labrador group, but its winter
IW HH/HV catalogue query returned no products. St Lewis was substituted using
metadata availability, before inspecting SAR pixels or scores. The rejected query
and all successful query receipts are retained under
`data/hh_hv_reservations/catalogue_queries/`. No full-scene quicklooks were opened.

## Frozen metadata and consumer checks

The [trusted lock](../configs/reservations/hh_hv_development_v010.lock.json)
pins this seal SHA256:

```text
06272d674f32df5240d5fa77b8ae78b9d9ea1b8b31179b37ffc5ca80a25cd843
```

The seal binds the manifest and eight selected catalogue snapshots. The manifest
also records the selection code hash, frozen generator configuration hash,
prior-exposure metadata hashes and discovery queries. It is an immutable record
of reservation-time exposure, not a live image-access monitor. New processing
records must be added separately rather than editing the original seal.

Offline verification:

```sh
python -m agents.hh_hv_reservation --verify reservations/hh_hv/v0.1.0 --seal-sha256 06272d674f32df5240d5fa77b8ae78b9d9ea1b8b31179b37ffc5ca80a25cd843
```

Once a compatible, versioned development dataset has been materialized, pass
`--reservation reservations/hh_hv/v0.1.0 --reservation-sha256 <trusted seal hash>`
to `python -m agents.hh_hv_candidate_run`, alongside `--dataset` and a fresh
`--output`. The runner rejects unreserved UUIDs, changed SAFE identities, split
or group changes and escaped ROI geometry. It also checks the actual raster's
geographic bounds from its header **before reading SAR arrays**, then records
the reservation hash and accepted bounds in its output. The original benchmark
remains available without this opt-in reservation for exposed workflow checks.
These guards are not access control against a repository owner bypassing them.

## What remains before a comparison

No native imagery was downloaded by this reservation, no new areas were
annotated, and no generator or classifier was run on them. Reservation-time
flags remain `scored: false`, `independent_review: false` and
`promotion_eligible: false`. The reservation is not an NL benchmark dataset
release and cannot itself be used as `--dataset`.

Next, download the exact reserved products, extract bounded native HH/HV windows
inside the frozen envelopes without full-scene previews, and freeze their pixel
identities before scoring. Check calibration, polarization, coastline alignment,
valid pixels and observed conditions. Annotate complete selected areas without
detector or classifier outputs; keep unresolved areas ignored and labels
provisional if no independent reviewer is available. Only then generate the
shared candidate population with the existing frozen settings and run both
comparison arms from that same cache.

If an acquisition proves unavailable or unusable, retain its identity and reason
in an additive exclusion record. Do not replace it because its model results look
unfavourable; any replacement requires a new pre-scoring reservation version.
The independent-review guard and original locked test set still constrain claims
and promotion. Neither precision, recall nor false-alarm reduction is measured
by this reservation.

Catalogue discovery follows the public
[Copernicus Data Space OData interface](https://documentation.dataspace.copernicus.eu/APIs/OData.html).
The project retains the applicable
[Copernicus Data Space terms](https://dataspace.copernicus.eu/terms-and-conditions).
Contains Copernicus Sentinel metadata (2025, 2026); this release does not
redistribute SAR pixels or AllenAI annotations.
