#!/usr/bin/env bash
# Run one eligible NL acquisition by UUID, recording versioned outcomes.
# Measurements are AIS-subset recall; vessel precision remains unmeasured.
# Usage: scripts/watch_and_score.sh [--check-only]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
AOI="${TRITONEYE_WATCH_AOI:-eastern_newfoundland}"
DAYS="${TRITONEYE_WATCH_DAYS:-12}"
LOG_DIR="$REPO/data/watch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/watch.log"
log() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITONEYE_DETECTOR="${TRITONEYE_DETECTOR:-xview3}"

# Starting a stopped recorder cannot restore historical observations. The
# Python watcher separately checks heartbeat age, observation age and gaps.
ensure_recorder() {
  if ! docker info >/dev/null 2>&1; then
    log "docker daemon unreachable; recorder supervision unavailable"
    return 1
  fi
  local state
  state="$(docker inspect -f '{{.State.Running}}' tritoneye-ais-recorder-1 2>/dev/null)"
  if [ "$state" = "true" ]; then
    return 0
  fi
  log "recorder is not running; starting it"
  docker compose up -d ais-recorder >>"$LOG" 2>&1
}

ensure_recorder
log "checking $AOI over the last $DAYS days"
python -m agents.watch --days "$DAYS" --aoi "$AOI" "$@" 2>>"$LOG" | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
