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

# Supervise the recorder before looking for scenes.
#
# `restart: unless-stopped` only holds while the Docker daemon is alive. If
# Docker Desktop is not running -- after a reboot, or because it was quit --
# nothing restarts the recorder, and the gap is invisible until an acquisition
# turns out to be unscorable. That has now cost coverage four separate times,
# most recently ~28 hours.
#
# This task is the reliable component: it is a Windows scheduled task, it
# survives reboots, and it already runs hourly. So it supervises the fragile
# one rather than the other way round.
ensure_recorder() {
  if ! docker info >/dev/null 2>&1; then
    log "docker daemon unreachable; cannot supervise the recorder"
    return 1
  fi
  local state
  state="$(docker inspect -f '{{.State.Running}}' tritoneye-ais-recorder-1 2>/dev/null)"
  if [ "$state" = "true" ]; then
    return 0
  fi
  log "recorder is not running; starting it"
  if docker compose up -d ais-recorder >>"$LOG" 2>&1; then
    log "recorder started"
  else
    log "failed to start the recorder"
    return 1
  fi
}

ensure_recorder

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

# First scorable date NOT already processed. Taking simply the newest scorable
# and stopping if it was stamped would strand any older scorable scene forever
# -- exactly what happens after an outage, when several accumulate at once.
DATE="$(printf '%s' "$REPORT" | python -c '
import json, os, sys
log_dir, aoi = sys.argv[1], sys.argv[2]
rows = json.load(sys.stdin)["acquisitions"]
for r in rows:
    d = r["acquired"][:10]
    if r["scorable"] and not os.path.exists(os.path.join(log_dir, f"processed-{aoi}-{d}.done")):
        print(d)
        break
' "$LOG_DIR" "$AOI")"

if [ -z "$DATE" ]; then
  log "every scorable acquisition has already been processed; nothing to do"
  exit 0
fi

log "SCORABLE acquisition found on $DATE"
if [ "${1:-}" = "--check-only" ]; then
  log "--check-only: stopping before download"
  exit 0
fi

# Two stamps with different meanings, deliberately kept apart:
#   processed-*  the pipeline ran to completion; do not run it again
#   scored-*     the evaluation actually produced a measurement
# A clean pipeline exit is NOT a measurement. Conflating the two once marked
# 2026-09-22 and 2026-09-27 "scored" when both evaluations read
# `scored: False` -- the log claimed a result that did not exist.
PROCESSED="$LOG_DIR/processed-$AOI-$DATE.done"
SCORED_STAMP="$LOG_DIR/scored-$AOI-$DATE.done"
OUT="$LOG_DIR/run-$AOI-$DATE.json"

log "processing $DATE (downloads ~1.7 GB, then ~17 min GPU)"
if python -m agents.pipeline --date "$DATE" --aoi "$AOI" >"$OUT" 2>>"$LOG"; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$PROCESSED"
  RESULT="$(python -c '
import json, sys
try:
    ev = json.load(open(sys.argv[1])).get("evaluation") or {}
except Exception:
    ev = {}
if ev.get("scored") is True:
    print("scored")
else:
    print(ev.get("reason") or ("not scored" if ev else "evaluation missing"))
' "$OUT")"
  if [ "$RESULT" = "scored" ]; then
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$SCORED_STAMP"
    log "SCORED $DATE -- evaluation in $OUT"
  else
    log "processed $DATE but NOT scored ($RESULT) -- see $OUT"
  fi
else
  log "pipeline failed for $DATE; leaving unstamped so the next run retries"
  exit 1
fi
