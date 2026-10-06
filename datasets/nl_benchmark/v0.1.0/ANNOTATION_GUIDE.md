# NL SAR annotation guide — version 0.1.0

This is a **provisional, machine-assisted annotation pilot**, not independently
reviewed truth. The user confirmed on 2026-10-05 that no independent reviewer is
available. No annotator is claimed to be a qualified SAR analyst.

## Review unit and procedure

The unit is the complete, preselected 512 × 512 native-radar window, not a
detector-generated candidate crop. Four contiguous 256 × 256 cells account for
every pixel. Both co-polarized and cross-polarized images must be examined at
native resolution; overview-only review is insufficient. All windows contain
valid pixels in both bands. The supplied display stretch is fixed at −30 to +5
dB; inspect the calibrated float32 TIFF at other stretches for expert review.

1. Review every cell, including apparently empty water. Detector predictions and
   AIS are hidden during the image-first pass. Do not label only detector outputs.
2. Mark distinct suspicious compact returns with native pixel boxes, even if
   uncertain. Record context/artefact areas separately as `kind: area`; their
   boxes are contextual extents, not individual target hulls.
3. Use supporting AIS only after the image-first pass. Check acquisition time,
   position uncertainty, actual swath and selected ROI. AIS silence cannot label
   background, establish a non-vessel, or prove deliberate dark activity.
4. Write the decision, evidence, author and uncertainty for every annotation.
   Geographic point coordinates represent the box centre transformed through
   the chip's WGS84 GCPs, not independently surveyed vessel positions.
5. Log all reviewed cell IDs, chip hash and the digest of the exact object set.
   A changed annotation invalidates an old review until reviewed again.

A recorded complete **first pass** means the annotator inspected the full
selected area. It does **not** establish exhaustive vessel ground truth: objects
can still have been missed or misinterpreted. No detector-miss rate can be
inferred until an independent image-first review is complete.

## Classes

| Class | Requirement |
| --- | --- |
| `vessel` | Image morphology and/or corroborating co-temporal evidence supports a vessel; explain alternatives and retain uncertainty. A bright pixel alone is insufficient. |
| `fixed_structure` | Specific stationary structure corroborated by a documented reference or independent temporal evidence. Harbour location alone is insufficient. |
| `clutter` | Diffuse sea/land return or imaging artefact lacking a specific target interpretation. Context regions use `kind: area`, not object-centre negatives. |
| `ice` | Identifiable ice morphology with adequate contextual evidence. Winter acquisition alone is insufficient; an ice chart is not an individual-object label. |
| `unresolved` | Vessel/structure/rock/ice/artefact alternatives cannot be distinguished confidently. This is a useful label, not an error to force into another class. |

Use only `object` or `area` for `kind`; vessels/fixed structures must be objects.
Boxes are `[x_min, y_min, x_max, y_max]` in chip coordinates, half-open at the
right/bottom boundaries. Partial objects on edges are retained and flagged in
their decisions. Clusters that cannot be separated reliably remain one unresolved
cluster, not a fabricated number of vessels. Typical small-target annotation
uncertainty and SAR moving-target displacement must remain visible.

## Independent review and release

Provide the detector-blind two-band bundle to a different reviewer with relevant
SAR expertise. The reviewer must perform a full-area pass, including empty cells
and objects the first annotator missed; reviewing only ambiguous boxes is not
sufficient for a false-positive/miss benchmark. Then compare drafts, resolve or
retain disagreements, and append a timestamped `independent_review` event with
reviewer ID, qualifications/evidence notes, chip hash, reviewed annotation digest,
all cells, independence declaration and acceptance/rejection decision. Preserve
the earlier draft and rejected decisions in history; never silently overwrite
review history. Author identity/independence are auditable declarations, not
cryptographic proof.

In this release independent review is **unavailable**, not silently replaced by
another AI session. Any unresolved object or absent independent full-area review
blocks the coastal evaluator's validation export. Context-area labels are not
exported as target-centre non-vessels. No precision, recall or false-alarm number
is emitted. Future evaluation may introduce independently adjudicated ignore
areas, but must disclose excluded area and targets; v0.1.0 does not implement that.

## Splits and annotation scope

Acquisitions are grouped by platform + absolute orbit + datatake, preventing
adjacent frames/reprocessed COG copies from straddling splits. Geographic groups
are assigned once, with at least 1 km between selected ROI polygons in different
splits. Entire source swaths may overlap; **only selected windows belong to this
dataset**. Never train on other pixels from a held-out geographic group or scene.
The split lock was generated before first-pass inspection. Prior development
exposure is recorded; the older validation scenes are not pristine holdouts.

The Hopedale test acquisition is reserved for final evaluation. Annotation/QA
may inspect it, but detector predictions, buffer tuning, training export and
performance evaluation are prohibited by the current interface. Releasing a
final test requires a separate, versioned, predeclared evaluation protocol.
The lock is a workflow safeguard, not access control over local files. Never
call a test untouched after tuning to its images or labels.

Source bytes, calibration XML, acquisition metadata, licence snapshot, selected
windows, annotations, review events and release files are SHA-256-addressed.
Changes after release require a new version preserving the old record and test
assignments. Annotation revisions are not independent evidence by themselves.
