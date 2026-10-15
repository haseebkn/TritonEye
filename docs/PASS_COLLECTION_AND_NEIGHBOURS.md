# Pass-specific AIS collection and neighbouring vessel opportunities

On **15 October 2026**, the existing eight reception priorities were expanded by
two neighbouring 0.1-degree grid rings before any imagery or classifier scores.
The fresh archive contains **224,917 unique AIS reports**, through
**13:45:40.247619 UTC**. **Zero new vessel-training windows qualify.**

The [v0.4.0 manifest](../reservations/cd_vessels/v0.4.0/manifest.json) preserves
the pass diagnostics, cell decisions, source failures and empty annotation
handoff. The [frozen discovery policy](../configs/reservations/neighbour_vessels_v040.json)
and lock bind the release. No new objects were annotated, labels admitted,
models scored or promotion authorized. All labels remain provisional.

## Expanded search

The search constructs **132 distinct whole cells**, including neighbours with no
previously received reports. This allows a moving/interpolated track to qualify
without requiring repeated historical reception inside its cell. It is a bounded
neighbourhood search, not exhaustive Newfoundland and Labrador coverage. Grid
cells have unequal physical area and two rings do not imply a fixed kilometre
radius.

**109 cells pass** the existing whole-cell training-geography checks. Thirteen
are excluded for locked-test/quarantine separation and ten for existing
validation/protected roles. Candidate envelopes and native acquisitions undergo
their own checks after source screening; a passing whole cell alone cannot
authorize a training reservation. Prior identities, exposure and geographic
assignments remain binding. Original eight-cell order is preserved, followed by
neighbour ring, anchor priority and cell ID. No image or model output affects
ranking.

The expanded source search screens **26 aligned supports**, compared with 24
in the earlier eight-cell attempt. Twenty-five fail at least one source check;
the counts below overlap:

| Failed source check | Supports |
| --- | ---: |
| Speed below 1 knot or unavailable | 23 |
| Reference water at least 500 m from shore not established | 23 |
| Bracket over 240 seconds, invalid or missing | 12 |
| Alignment uncertainty over 500 m or invalid | 3 |
| Nearest report over 120 seconds or missing | 1 |

The remaining support is on UUID `01876305-6ca7-4a24-9004-b9bf7a2d9848`,
native group `S1C_009500_012E71`. It was already exposed/reserved, and the frozen
inventory explicitly assigns that acquisition to **validation**. It cannot be
reassigned to training. The 26 supports are neither 26 independently verified
vessels nor newly annotated SAR targets.

The source requirements and four-window cap remain unchanged. Speed and shore
screens are conservative discovery choices; excluded slow/nearshore returns can
still be vessels. Available competing AIS histories remain private, and missing
competitors are possible. Any future qualifying reservation still needs exact
native target-time support, processor/input checks, complete annotation before
scores, uncertainty coverage, association checks and permitted use before an
exploratory training export.

## What the pass audit measures

The audit covers **84 HH/HV product slices in 43 native acquisition groups**
whose footprints intersect eligible search cells. SAFE/COG aliases are handled
by the existing overlap consumer; adjacent slices remain related observations.
These are not 84 independent trials.

Each context extends from acquisition start minus 15 minutes to acquisition end
plus 15 minutes. The manifest records reports and identities in the intersecting
eligible cells/swath, reports during the acquisition itself, missing receipt
timestamps and the median, 95th percentile and maximum receipt-minus-observation
time. Negative values are counted explicitly. Provider/client clocks and source
delays mean this difference is not network latency alone. Legacy timestamps keep
the existing disclosed UTC interpretation.

Only **7 of 84 slices** have received reports in those cells during the context.
Across the historical journals:

- **11 contexts** have complete connected-heartbeat interval support;
- **2** have partial support;
- **71** have no such interval support in the retained snapshot.

These are recorder diagnostics, not estimates of receiver completeness or proof
of actual outages. Some contexts lack historical journal evidence. No received
reports cannot establish that vessels were absent. Overlapping recorder sessions
are merged rather than double-counted. Disconnections, session stops, new
transports and heartbeat gaps over 180 seconds prevent continuity claims.

**Server subscription acknowledgement and negotiated compression are unknown
for every audited context.** Historical `subscribed` events record a local send;
they cannot establish server acceptance. No malformed journal events were found.
Packet loss and historical parsing rejection rates remain unmeasured; the audit
does not attribute sparse reception to a specific failure without evidence.

## Future recorder telemetry

The recorder now explicitly requests `permessage-deflate` and journals the
negotiated extension plus server `SubscriptionConfirmation` receipts, including
their compression field when provided. This makes the library's previous
compression default explicit; it does not demonstrate that past connections
were uncompressed or that changing this option improves reception. The new
events contain only allowlisted status fields, not keys or message bodies.
Class A/Class B position support and the existing CSV schema are retained.

The protocol basis is the [AISStream documentation](https://www.aisstream.io/documentation/),
which describes acknowledgement, compression and potential delivery loss, and
the pinned [websockets 15.0.1 client documentation](https://websockets.readthedocs.io/en/15.0.1/reference/asyncio/client.html).
Local compressed/uncompressed WebSocket servers verify the actual recorder
behaviour, including binary confirmation frames. Those tests are engineering
evidence, not live-provider availability or historical telemetry.

The existing live recorder was not restarted, reconfigured or duplicated during
this work. Its current process/image therefore does not gain this telemetry
until the next normal deployment. No new monitoring task or subscription change
was created, and earlier recordings/journals were preserved.

## Replay and next evidence

Raw snapshots, catalogue responses, complete available competing pools, shoreline
references and frozen priors are retained under
`data/research/neighbour_vessels_v040_01/`. Public outputs omit MMSIs, individual
tracks and object positions; cell geometry and aggregate pass statistics are
published. Snapshots capture complete file prefixes sequentially while collection
continues, rather than asserting an atomic archive transaction.

```sh
python -m agents.neighbour_vessel_batch --verify
python -m agents.neighbour_vessel_batch --verify --local
python -m pytest tests/unit/test_ais_pass_audit.py tests/unit/test_ais_recorder.py -q
```

Retained-data replay reconstructs neighbours, overlaps, shoreline classifications,
pass collection diagnostics and reservation decisions. The positive synthetic
publication regression also checks a nonempty reservation round trip and
overwrite refusal; it caught and now guards a tuple/list coordinate comparison
error that empty real releases could not expose. Synthetic fixtures supply no
new NL observations or scientific review.

Local validation passed **126 focused tests** across the new pass audit, recorder,
reception, provisional admission and C/D export modules. The portable nonempty
publication fixture passed separately after its path adjustment. Retained-data
replay passed; Ruff, Black and mypy passed across **177 Python files**. Automated
review and CI are additional publication gates.

The next useful evidence is a new tightly supported moving-vessel pass in the
eligible cells, or permitted historical tracks from another source. Another
small archive refresh is not itself a sufficient reason to repeat an empty
reservation. Vessel training can proceed separately from iceberg corroboration
when its evidence requirements are met. Independent accuracy and promotion
continue to require the existing scientific review and evaluation safeguards.

For a new release, use a new public version/policy and fresh private directory
with `--capture`, then `--publish`; existing public paths refuse replacement.
