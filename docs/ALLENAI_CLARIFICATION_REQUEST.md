# AllenAI annotation clarification request draft

Draft only, not sent. Choose a verified AllenAI or Skylight contact before sending;
no recipient address or organizational affiliation is assumed.

Subject: Clarification on 2022 Sentinel-1 vessel annotation licence and imagery alignment

Hello AllenAI/Skylight team,

I am an independent portfolio researcher developing TritonEye, a Newfoundland
and Labrador maritime remote-sensing research prototype. It is not a C-CORE
product, and I am not contacting you on C-CORE's behalf.

I am auditing the point annotations in `data/metadata.sqlite3` from
[vessel-detection-sentinels at revision 60b003d](https://github.com/allenai/vessel-detection-sentinels/tree/60b003d059937a9fcddbf6e1c9b64c00c33c0279).
The database SHA256 is
`d3a8bbd6c6685a6f8bb685f122b37fda613649d6da4c450e3240cace353ca2ac`.
My regional screen found 73 source windows and 39 published points; cached
Sentinel-1 imagery currently supports alignment QA for 31 windows and 23 points
from three Newfoundland acquisitions in 2022.

Could you clarify the following?

1. Does the repository's Apache-2.0 licence cover those 2022 vessel annotations,
   or does a separate data licence apply? The
   [rslearn evaluation README](https://github.com/allenai/rslearn_projects/blob/9cfaf6e9a13fcb993c687b591eb3f71df7c144c5/rslp/olmoearth_evals/README.md#sentinel-1-vessel-detection)
   declares CC BY 4.0 for its linked archive, but our ID/origin crosswalk finds
   no Newfoundland/Labrador overlap in that archive's source extents. Does that
   grant also cover the older 2022 database? Please confirm permitted local
   training, derivative annotations/model weights and redistribution, with
   any required attribution or restrictions.
2. Were complete valid image areas exhaustively annotated, including windows
   with no vessel points? Were difficult, coastal, berthed, ice-obscured or
   uncertain targets deliberately omitted? Are annotation guidelines,
   per-window completeness/review records or known exclusions available?
3. Can you confirm the original image warp, pixel-origin and click-coordinate
   conventions? Are the annotation-time tiles available? Our reconstruction of
   the older Web Mercator grid and our native TPS georeferencing differ by
   approximately 2–116 m at these points; this is a transform diagnostic, not
   a claim that your labels are wrong. We would like to establish which image
   object each point was intended to identify, without moving labels to
   detector predictions or nearby bright returns.
4. Is the full 2022 annotation distribution available through a public,
   non-requester-pays download? If not, what access process and costs apply?

The exact source products are:

```text
S1A_IW_GRDH_1SDV_20220421T212233_20220421T212258_042877_051E4D_68C1.SAFE
S1A_IW_GRDH_1SDV_20220515T212259_20220515T212324_043227_0529A7_D334.SAFE
S1A_IW_GRDH_1SDV_20220829T213859_20220829T213924_044773_0558B9_18A6.SAFE
```

I can supply window/point identifiers, reconstruction details and a small
alignment QA bundle through an appropriate channel. The annotations remain
quarantined and provisional in my project. No independent local reviewer is
available, and I will not present them as exhaustive benchmark truth or report
measured detector improvement without adequate reviewed evaluation evidence.

Thank you for your guidance.

Best regards,
Haseeb

