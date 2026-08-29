#!/usr/bin/env bash
# Durable device queue for job c-exec-llama.
#   queue.sh <queue-file>
# Each queue line:  <name>|<timeout-seconds>|<pytest node id or file>
# Results land in RESULTS.md as they finish; logs are never overwritten.
set -u
REPO=/proj_sw/user_dev/ctr-apbernal/tt-metal
LOGS="${REPO}/tttv2_milestone_c_evidence/exec_llama/logs"
RESULTS="${REPO}/tttv2_milestone_c_evidence/exec_llama/RESULTS.md"
QUEUE="$1"
export HF_HOME=/localdev/ctr-apbernal/hf_data
export TT_CACHE_PATH=/localdev/ctr-apbernal/tt_cache
cd "$REPO" || exit 1
# One pytest on the mesh at a time, enforced rather than remembered: a second
# queue started by mistake at 12:56Z put two device runs on the Galaxy for
# thirty seconds and cost both their results.
LOCK="${REPO}/tttv2_milestone_c_runs/c-exec-llama/.queue.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[queue] refusing to start: another c-exec-llama queue holds ${LOCK}" >&2
  exit 2
fi
mkdir -p "$LOGS" "$TT_CACHE_PATH"
[ -f "$RESULTS" ] || printf '# c-exec-llama device runs, in the order they happened\n\n| when (UTC) | name | node | result | seconds | log |\n| --- | --- | --- | --- | --- | --- |\n' > "$RESULTS"

while IFS='|' read -r NAME TMO NODE ENVS; do
  case "$NAME" in ''|\#*) continue ;; esac
  LOG="${LOGS}/${NAME}.log"
  if [ -f "$LOG" ]; then echo "[queue] skip ${NAME}: log exists" >&2; continue; fi
  START=$(date -u +%s)
  echo "[queue] $(date -u +%Y-%m-%dT%H:%M:%SZ) dequeue ${NAME} -> ${NODE}" >&2
  {
    echo "# name=${NAME}"
    echo "# node=${NODE}"
    echo "# commit=$(git rev-parse HEAD)"
    echo "# started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# HF_HOME=${HF_HOME}"
  } > "$LOG"
  # `pytest.ini` caps every test at 300 s (`--timeout=300`), and the Milestone B
  # runs of these cases fitted inside it only because they aborted early at D-C5.
  # With the sampling path fixed they run to completion and the cap terminates
  # them: `c9_q_greedy_run1` reported `Failed: Timeout (>300.0s) from
  # pytest-timeout` at 310 s. The outer `timeout` is the real bound.
  echo "# env=${ENVS:-}" >> "$LOG"
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
  # A TT_FATAL inside a multi-subdevice program leaves the mesh un-drainable.
  if [ "$RC" -ne 0 ] && grep -q "TT_FATAL\|TT_THROW\|Fatal Python error" "$LOG"; then
    echo "[queue] aborted run, draining" >&2
    pgrep -f "python -m pytest .* ${NODE}" | while read -r P; do
      C=$(ps -o comm= -p "$P" 2>/dev/null)
      case "$C" in python|python3|pytest) kill -TERM "$P" ;; esac
    done
    sleep 15
  fi
done < "$QUEUE"
echo "[queue] $(date -u +%Y-%m-%dT%H:%M:%SZ) queue drained" >&2
