#!/usr/bin/env bash
# Run queue files in order, waiting for the flock rather than racing it.
#   chain.sh <queue-file> [<queue-file> ...]
# queue.sh exits 2 when another queue holds the lock; anything else means it ran.
set -u
QUEUE_SH=/proj_sw/user_dev/ctr-apbernal/tt-metal/tttv2_milestone_c_runs/c-exec-llama/queue.sh
for Q in "$@"; do
  while true; do
    "$QUEUE_SH" "$Q"
    RC=$?
    [ "$RC" -eq 2 ] || break
    echo "[chain] $(date -u +%H:%M:%SZ) lock held, waiting before ${Q}" >&2
    sleep 15
  done
  echo "[chain] $(date -u +%H:%M:%SZ) finished ${Q} rc=${RC}" >&2
done
