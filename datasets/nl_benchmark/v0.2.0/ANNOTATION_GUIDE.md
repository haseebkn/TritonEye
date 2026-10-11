# NL SAR annotation guide 0.2.0

This revision follows the [v0.1.0 annotation guide](../v0.1.0/ANNOTATION_GUIDE.md).
Its classes, half-open native pixel boxes, complete-cell review, AIS supporting
role and independent-review requirements remain applicable. No qualified
independent reviewer is currently available. AI/self review remains provisional.

## Review procedure

1. Inspect every 256x256 cell of each complete 512x512 area in both native bands,
   including apparently empty water and objects lacking detector/AIS evidence.
   The offline bundle embeds native images at the fixed -30 to +5 dB stretch.
   Experts can inspect the calibrated TIFF at alternative stretches.
2. Write the complete image-first draft before revealing earlier decisions or
   opening the private `telemetry_second_pass` directory. Selection of the nine
   additions was AIS-informed and the preparer had previous telemetry exposure;
   this limitation cannot be removed by hiding markers.
3. Label `vessel`, `fixed_structure`, `clutter`, `ice` or `unresolved`. A vessel
   interpretation must explain morphology, alternatives and uncertainty. Current
   new vessel interpretations are explicitly provisional. Brightness alone is
   insufficient. Referenced structures and identifiable ice need corroboration.
4. Record clusters with uncertain object counts as unresolved clusters. Context
   labels use `kind: area`; these are not individual negative examples. Include
   edge objects with uncertainty. Explicit empty-area decisions do not establish
   exhaustive absence. Do not erase ambiguous returns to improve a score.
5. Record reviewer identity, qualifications, timestamp, actual chip hash, all
   reviewed cell IDs and the exact annotation digest. Compare a separate full
   image-first pass, including newly found objects, with the earlier draft.
   Preserve disagreements and accepted/rejected decisions in history.
6. Add AIS only as later supporting evidence. Check the whole scene competitor
   pool, acquisition timing, motion and radar-location uncertainty. Proximity
   alone cannot establish correct identity. Preserve unavailable/ambiguous cases.

## Version and test discipline

The 12 parent areas and review histories are inherited without a new inspection
claim. The new packet contains nine added development areas. Existing Hopedale
test images remain reserved for the predeclared final protocol; this extension
does not open, score, tune against or reannotate them. Keep parent acquisition and
geography assignments even when enriching a region with new dates.

Full-area independent review, resolved/explicitly adjudicated uncertainty and a
separate final-test release protocol are still required for performance claims.
The evaluator rejects the present provisional validation export. A second AI
session or a reproducibility check does not establish independent label review.
