#!/usr/bin/env bash
# Checks for a scorable acquisition and, if one exists, scores it.
#
# The project's one unmeasured quantity is detection performance over
# Newfoundland. Producing it needs a VV/VH pass over open water while the AIS
# recorder was running, which cannot be arranged -- only caught. This is the
# catcher: run it on a schedule and it will process the first qualifying scene
# without anyone watching for it.
#
# SAFE TO RUN REPEATEDLY. agents.scene_watch exits 3 when nothing qualifies, so
# the common case costs one catalogue query and nothing else. A scene is only
# downloaded (~1.7 GB) and processed (~17 min GPU) when all three conditions
# hold: VV/VH, recorded AIS in the +/-5 min window, and those AIS positions in
# alert-eligible water.
#
# Usage:
#   scripts/watch_and_score.sh                 # check and score if possible
#   scripts/watch_and_score.sh --check-only    # never process, just report
#
# Schedule on Windows:
#   schtasks /create /tn TritonEyeWatch /sc hourly ^
#     /tr "C:\Program Files\Git\bin\bash.exe E:\TritonEye\scripts\watch_and_score.sh"

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

AOI="${TRITONEYE_WATCH_AOI:-eastern_newfoundland}"
DAYS="${TRITONEYE_WATCH_DAYS:-12}"
LOG_DIR="$REPO/data/watch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watch.log"

log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

# Allocator tuning: the 126-tile loop fragments the CUDA allocator, which is
# what made earlier full-scene attempts fail on an 8 GB card. The failures were
# host memory, but this costs nothing and removes the other candidate.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITONEYE_DETECTOR="${TRITONEYE_DETECTOR:-xview3}"

log "checking $AOI over the last $DAYS days"
REPORT="$(python -m agents.scene_watch --days "$DAYS" --aoi "$AOI" --json 2>>"$LOG")"
STATUS=$?

if [ $STATUS -eq 3 ]; then
  log "nothing scorable; the recorder must be running BEFORE an acquisition"
  exit 0
fi
if [ $STATUS -ne 0 ]; then
  log "scene_watch failed with status $STATUS"
  exit "$STATUS"
fi

DATE="$(printf '%s' "$REPORT" | python -c '
import json, sys
rows = json.load(sys.stdin)["acquisitions"]
hit = next((r for r in rows if r["scorable"]), None)
print(hit["acquired"][:10] if hit else "")
')"

if [ -z "$DATE" ]; then
  log "scene_watch reported success but named no acquisition; not processing"
  exit 1
fi

log "SCORABLE acquisition found on $DATE"
if [ "${1:-}" = "--check-only" ]; then
  log "--check-only: stopping before download"
  exit 0
fi

# A scene already processed should not be reprocessed on the next tick.
STAMP="$LOG_DIR/scored-$AOI-$DATE.done"
if [ -f "$STAMP" ]; then
  log "$DATE already scored; nothing to do"
  exit 0
fi

log "processing $DATE (downloads ~1.7 GB, then ~17 min GPU)"
if python -m agents.pipeline --date "$DATE" --aoi "$AOI" >>"$LOG" 2>&1; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$STAMP"
  log "SCORED $DATE -- see missions/ for the manifest and evaluation block"
else
  log "pipeline failed for $DATE; leaving unstamped so the next run retries"
  exit 1
fi
