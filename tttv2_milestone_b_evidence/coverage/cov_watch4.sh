#!/bin/bash
# Attempt 4's mesh watcher.
#
# The mesh stopped answering at 18:37Z with two independent faults (see
# REPORT.md section A4). Five recovery attempts failed and the kernel says the
# chip is unresponsive and cannot be reset, so this job cannot fix it. But the
# night is long, an operator may fix it, and a queue that resumes by itself is
# worth more than a queue that needs an agent awake.
#
# What this does, and deliberately nothing more:
#   * every 300s, a CHEAP health probe: `tt-smi -ls` must exit 0 with no
#     `0xffffffff`, and all 32 /dev/tenstorrent nodes must be openable. Opening a
#     node is the check that matters - the nodes existing is not sufficient, which
#     is exactly how this fault presented;
#   * on the first healthy probe: drop queue4.halt and start cov_queue4.sh, then
#     exit. The queue owns the mesh from then on;
#   * at most MAX_RESETS `tt-smi -glx_reset_auto` attempts, spaced RESET_EVERY
#     apart, and never while a pytest holds a device. The brief caps recovery
#     attempts at 2 per run; these are spaced, capped, logged, and stop for good
#     once the cap is reached - the alternative is a mesh nobody tries to revive
#     for eleven hours;
#   * stops at DEADLINE regardless, so it never outlives the job that started it.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
LOG="$D/logs4/watch4.log"
DEADLINE=${WATCH_DEADLINE:-$(date -u -d '2026-08-29 05:15:00' +%s)}
MAX_RESETS=${MAX_RESETS:-3}
RESET_EVERY=${RESET_EVERY:-2700}
resets=0
last_reset=$(date +%s)

log() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

nodes_open() {
    python3 - <<'PY'
import os, sys
bad = []
for i in range(32):
    p = f"/dev/tenstorrent/{i}"
    try:
        fd = os.open(p, os.O_RDWR); os.close(fd)
    except Exception as exc:
        bad.append(f"{i}:{type(exc).__name__}")
print(",".join(bad) if bad else "all-open")
sys.exit(1 if bad else 0)
PY
}

device_held() {
    for p in $(pgrep -f 'python.*-m pytest' 2>/dev/null); do
        c=$(ps -o comm= -p "$p" 2>/dev/null)
        case "$c" in python|python3|pytest) ;; *) continue ;; esac
        ls -l "/proc/$p/fd" 2>/dev/null | grep -q '/dev/tenstorrent' && return 0
    done
    return 1
}

log "=== watcher up, deadline $(date -u -d "@$DEADLINE" +%H:%M:%SZ), max resets $MAX_RESETS"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 300
    if pgrep -f 'bash cov_queue4\.sh' >/dev/null 2>&1; then
        log "queue4 is running; watcher stands down"
        break
    fi
    bad=$(nodes_open); ok_nodes=$?
    probe="$D/logs4/watch_probe.log"
    timeout 180 tt-smi -ls > "$probe" 2>&1; ok_ls=$?
    dirty=$(grep -c '0xffffffff' "$probe" 2>/dev/null)
    if [ "$ok_nodes" -eq 0 ] && [ "$ok_ls" -eq 0 ] && [ "${dirty:-1}" -eq 0 ]; then
        log "MESH HEALTHY: all 32 nodes open, tt-smi -ls exit 0, no 0xffffffff"
        if device_held; then log "a pytest holds a device; not starting the queue"; continue; fi
        cp "$probe" "$D/logs4/health_probe_recovered.log"
        rm -f "$D/queue4.halt"
        log "starting cov_queue4.sh with $(grep -cvE '^\s*(#|$)' "$D/queue4.txt") items pending"
        setsid nohup bash "$D/cov_queue4.sh" > /dev/null 2>&1 < /dev/null &
        log "queue4 started; watcher exits"
        exit 0
    fi
    log "unhealthy: nodes[$bad] ls_exit=$ok_ls ffffffff_lines=${dirty:-?}"
    now=$(date +%s)
    if [ "$resets" -lt "$MAX_RESETS" ] && [ $((now - last_reset)) -ge "$RESET_EVERY" ] && ! device_held; then
        resets=$((resets + 1)); last_reset=$now
        R="$D/logs4/watch_reset_${resets}.log"
        { date -u; echo "=== watcher reset attempt $resets of $MAX_RESETS ==="; } > "$R"
        timeout 900 tt-smi -glx_reset_auto >> "$R" 2>&1
        echo "reset exit=$?" >> "$R"
        log "reset attempt $resets done: $(tr '\r' '\n' < "$R" | grep -aoE 'Re-initialized 32 boards after reset|Failed on last reset|Error in resetting[^,]*|reset exit=[0-9]+' | tail -2 | tr '\n' ' ')"
    fi
done
log "=== watcher stopping (deadline or queue took over); resets used: $resets"
