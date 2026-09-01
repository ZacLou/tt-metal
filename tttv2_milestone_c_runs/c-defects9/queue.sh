#!/usr/bin/env bash
# Durable device queue for job c-defects, attempt 7.
#   queue.sh <queue-file> [<queue-file> ...]
# Line format: <name>|<timeout-seconds>|<pytest node id or file>|<env assignments>
#
# Differences from attempt 3's queue2.sh, both from that attempt's own findings:
#  1. GLOBAL reset budget of 2 for the whole queue (the house rules' cap). Attempt 3
#     made thirteen and the fleet went from 9 bad boards to 25. When the budget is
#     spent the queue STOPS rather than handing run after run to a dead mesh.
#  2. Mesh health is probed with the sysfs heartbeat at the CORRECT path
#     (<node>/tt_heartbeat, not <node>/device/tt_heartbeat) as well as tt-smi.
set -u
REPO=/proj_sw/user_dev/ctr-apbernal/tt-metal
LOGS="${REPO}/tttv2_milestone_c_evidence/defects/logs"
RESULTS="${REPO}/tttv2_milestone_c_evidence/defects/RESULTS.md"
export HF_HOME=/localdev/ctr-apbernal/hf_data
export TT_CACHE_PATH=/localdev/ctr-apbernal/tt_cache
cd "$REPO" || exit 1

LOCK="${REPO}/tttv2_milestone_c_runs/c-defects9/.queue.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[queue] refusing to start: another c-defects9 queue holds ${LOCK}" >&2
  exit 2
fi
mkdir -p "$LOGS" "$TT_CACHE_PATH"
[ -f "$RESULTS" ] || printf '# c-defects device runs, in the order they happened\n\n| when (UTC) | name | node | result | seconds | log |\n| --- | --- | --- | --- | --- | --- |\n' > "$RESULTS"

RESET_BUDGET=2

healthy_boards() {
  local n=0
  for d in /sys/class/tenstorrent/*; do
    hb=$(cat "$d/tt_heartbeat" 2>/dev/null || echo ERR)
    case "$hb" in ERR|4294967295) ;; *) n=$((n+1)) ;; esac
  done
  echo "$n"
}

for QUEUE in "$@"; do
echo "[queue] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${QUEUE}" >&2
while IFS='|' read -r NAME TMO NODE ENVS; do
  case "$NAME" in ''|\#*) continue ;; esac
  LOG="${LOGS}/${NAME}.log"
  if [ -f "$LOG" ]; then echo "[queue] skip ${NAME}: log exists" >&2; continue; fi
  HB=$(healthy_boards)
  if [ "$HB" -lt 32 ]; then
    echo "[queue] HALT before ${NAME}: only ${HB}/32 boards ticking" >&2
    printf '| %s | %s | `%s` | HALTED: %s/32 boards healthy, not run | 0 | - |\n' \
      "$(date -u +%H:%M:%SZ)" "$NAME" "$NODE" "$HB" >> "$RESULTS"
    exit 3
  fi
  START=$(date -u +%s)
  echo "[queue] $(date -u +%Y-%m-%dT%H:%M:%SZ) dequeue ${NAME} -> ${NODE}" >&2
  {
    echo "# name=${NAME}"
    echo "# node=${NODE}"
    echo "# commit=$(git rev-parse HEAD)"
    echo "# started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# HF_HOME=${HF_HOME}"
    echo "# healthy_boards_before=${HB}"
    echo "# env=${ENVS:-}"
  } > "$LOG"
  env ${ENVS:-} timeout --signal=TERM --kill-after=180 "$TMO" \
    python -m pytest -v -rA --color=no -p no:cacheprovider --timeout="$TMO" "$NODE" >> "$LOG" 2>&1
  RC=$?
  END=$(date -u +%s)
  echo "exit=${RC}" >> "$LOG"
  SECS=$((END-START))
  VERDICT=$(grep -Eo '[0-9]+ (passed|failed|skipped|error)[^=]*' "$LOG" | tail -1)
  [ -z "$VERDICT" ] && VERDICT="rc=${RC} (no pytest summary)"
  case "$VERDICT" in *skipped*) VERDICT="$VERDICT  <- SKIPPED IS A FAILED RUN" ;; esac
  printf '| %s | %s | `%s` | %s | %s | `%s` |\n' \
    "$(date -u +%H:%M:%SZ)" "$NAME" "$NODE" "$VERDICT" "$SECS" "logs/${NAME}.log" >> "$RESULTS"

  if [ "$RC" -ne 0 ]; then
    echo "[queue] non-zero rc, draining" >&2
    pgrep -f "python -m pytest .* ${NODE}" | while read -r P; do
      C=$(ps -o comm= -p "$P" 2>/dev/null)
      case "$C" in python|python3|pytest) kill -TERM "$P" ;; esac
    done
    sleep 20
    PROBE="${LOGS}/${NAME}.meshprobe.log"
    HB2=$(healthy_boards)
    echo "healthy_boards_after=${HB2}" > "$PROBE"
    timeout 120 tt-smi -ls >> "$PROBE" 2>&1
    SMI=$?
    if [ "$HB2" -lt 32 ] || [ "$SMI" -ne 0 ]; then
      if [ "$RESET_BUDGET" -gt 0 ]; then
        RESET_BUDGET=$((RESET_BUDGET-1))
        echo "[queue] mesh unhealthy after ${NAME}; reset (budget left ${RESET_BUDGET})" >&2
        timeout 600 tt-smi -glx_reset >> "$PROBE" 2>&1
        echo "reset_rc=$?" >> "$PROBE"
        sleep 30
        echo "healthy_boards_after_reset=$(healthy_boards)" >> "$PROBE"
      else
        echo "[queue] HALT: mesh unhealthy and reset budget spent" >&2
        printf '| %s | (halt) | - | HALTED: mesh unhealthy, reset budget spent | 0 | `%s` |\n' \
          "$(date -u +%H:%M:%SZ)" "logs/${NAME}.meshprobe.log" >> "$RESULTS"
        exit 4
      fi
    fi
  fi
done < "$QUEUE"
done
echo "[queue] $(date -u +%Y-%m-%dT%H:%M:%SZ) queue drained" >&2
