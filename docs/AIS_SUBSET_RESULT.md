# Observed NL SAR–AIS proximity result — 3 of 3 across two scenes

## Result and claim boundary

Inspection on 11 October 2026 of two saved, completed real VV/VH evaluations
found **3/3 observed AIS-supported vessel-scene examples within 500 m of a SAR
return: 100% observed AIS-subset proximity recall**. Raw and open-water-eligible
detections both matched all three examples using the same AIS denominator.
The conventional two-sided exact **95% Clopper–Pearson binomial interval is
29.2%–100%**, conditional on treating the three examples as independent trials.
Only **two physical scenes** contribute; this is not three independent scenes
or an independently reviewed vessel benchmark.

| Acquisition start (UTC) | Physical slice (platform/orbit/datatake/start/end) | Observed AIS examples | Raw proximity matches | Eligible proximity matches |
| --- | --- | ---: | ---: | ---: |
| 2026-09-27 21:31:13.883890 | S1D / 004769 / 008EFD / 21:31:13–21:31:38 | 2 | 2 | 2 |
| 2026-10-02 21:38:55.020771 | S1D / 004842 / 00919B / 21:38:55–21:39:20 | 1 | 1 | 1 |
| Combined (n=3 across 2 scenes) | Each physical slice counted once | 3 | 3 | 3 |

The denominator consists of acquisition-aligned AIS positions inside the NL
study region and on valid SAR pixels, not every report in the recorder's time
window. Alignment reduces reports to one position per identity per scene.
Here `n=3` means vessel-scene observations, **not a claim of three distinct
vessel identities across dates**. Repeat executions and original SAFE/COG
catalogue aliases of these slices do not add observations. The evidence ledger
selects one saved execution per slice; it does not claim an exhaustive census
of every acquisition or failed/unsupported attempt in the watcher archive.

## Settings and limitations

The saved runs used the pinned xView3 detector, objectness threshold 0.15 and
300 m coastal policy. The evaluator performed one-to-one geodesic proximity
matching with a fixed 500 m radius. This is **not** the stricter 100 m
selected-area baseline rule, and the two metrics must not be pooled.
All three AIS examples were classified as open water under the cached shoreline.
No harbour/coastal examples contributed; coastal recall is unmeasured.
Registered imagery controls were `outside_scope` for these scene masks, so the
classification is source-relative, not locally verified shoreline truth.

- AIS coverage is explicitly **partial** and self-reported. Silent, unrecorded,
  or misreported vessels can be absent from the denominator. Recorder uptime,
  gaps and selecting scenes with usable AIS can bias the sample.
- A return within 500 m can be accidental, especially amid clutter or competing
  vessels. Proximity does not confirm a vessel label or correct identity.
  No independent reviewer is available. This is not evidence of improved
  correlation or a supervised training label set.
- The binomial interval assumes independent examples and does not correct
  scene/geographic dependence, repeated identities, AIS error, label error,
  selection bias or shoreline/geolocation uncertainty. With only two scenes,
  no credible cluster-robust regional performance interval is claimed.
- Overall vessel recall, precision, false alarms per water km² and association
  correctness remain unknown. Raw returns (264 and 428) and eligible returns
  (16 and 15) are **not confirmed vessels or false-alarm counts**. Unmatched
  returns remain **unassociated candidates**, not confirmed dark vessels.
- No Labrador-specific, HH/HV, seasonal, coastal or province-wide performance
  is established by this result. It cannot select an operating threshold,
  satisfy the minimum reviewed-label/coverage requirements, demonstrate model
  improvement, or justify enabling operational alerts.

For a C-CORE portfolio, the defensible claim is a small, auditable real-scene
SAR–AIS comparison with explicit uncertainty, **not 100% vessel accuracy**,
C-CORE validation, operational readiness or compliance with maritime protocols.

## Evidence and verification

[The public evidence ledger](evidence/ais_subset_20261011.json) contains selected
and processed product UUIDs, exact slice names, evaluation artifact hashes,
saved code/model/configuration provenance, counts and the interval. Source
executions ran on 6 October; 11 October is the inspection/publication date,
**not a fresh inference run on current main**. Saved source-file hashes in the
local payloads supplement their recorded Git commits because working-tree
changes can differ from committed code.

Full evaluations, private AIS snapshots, imagery and detection locations remain
local and ignored by Git. No MMSIs, vessel names, positions or API credentials
are published. A public checkout can recompute the interval and inspect the
summary, but cannot independently reproduce the matches without authorized
local inputs. Hashes establish which artifacts were inspected, not ground truth.

From the project root, verify the retained local payload hashes against the ledger:

```powershell
$ledger = Get-Content docs/evidence/ais_subset_20261011.json -Raw | ConvertFrom-Json
foreach ($scene in $ledger.scenes) {
    $hash = (Get-FileHash -LiteralPath $scene.local_evaluation -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $scene.evaluation_sha256) { throw "Evaluation hash mismatch" }
    $saved = Get-Content -LiteralPath $scene.local_evaluation -Raw | ConvertFrom-Json
    if (-not $saved.processing.complete -or -not $saved.evaluation.scored) { throw "Not measured" }
    if ($saved.selected_product_id -ne $scene.product_id -or
        $saved.processing.sar_product_id -ne $scene.product_id -or
        $saved.evaluation.sar_product_id -ne $scene.product_id) { throw "Product mismatch" }
    if ($saved.evaluation.ais_vessels_in_swath -ne $scene.observed_ais_examples -or
        $saved.evaluation.matched -ne $scene.matched_raw -or
        $saved.evaluation.matched_eligible -ne $scene.matched_eligible -or
        $saved.evaluation.match_radius_m -ne $ledger.match_radius_m) { throw "Metric mismatch" }
}
python -c "from agents.baseline_statistics import binomial_interval; print(binomial_interval(3, 3))"
```

The interval returned is `[0.2924017738212866, 1.0]` (floating-point rounding may
differ). For three successes in three trials, its lower endpoint is
`(0.05 / 2) ** (1 / 3)` and its upper endpoint is 1.

## Relationship to the frozen benchmark

The [v0.1.0 review pilot](NL_BENCHMARK.md) still has **zero confirmed training
vessel labels** and no independent full-area review. Its selected-area AIS
denominators remain zero. The later full-scene observations above do not alter
that immutable release, its split locks or its historical reports. A statement
that the entire project has no measured AIS subset is now stale; a scoped
statement that the frozen pilot has no confirmed vessel labels remains accurate.

See [the baseline protocol](NL_BASELINE.md) for the unfulfilled operating-point
requirements and [target correlation](TARGET_CORRELATION.md) for the distinction
between proximity and reviewed vessel-identity correctness.
