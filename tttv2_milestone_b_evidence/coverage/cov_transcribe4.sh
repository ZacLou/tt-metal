#!/bin/bash
# Self-writing run record.
#
# Attempt 2 died with no handoff and attempt 3's agent died four and a half hours
# before its queue did, which cost a later session an evening of transcription by
# hand. This closes that hole: a detached loop that turns every finished log in
# logs4/ into a row of RESULTS_A4_MACHINE.md, exactly once, whether or not any
# agent is awake. Nothing here interprets a result - it reports the commit, the
# wall clock, the pytest summary, the first assertion and the test's own marker
# lines, all grep-ed out of the log.
#
# RESULTS_A4.md stays the human record. This file is the machine one, and a later
# session should merge from it and say that it did.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
OUT="$D/RESULTS_A4_MACHINE.md"
DEADLINE=${TRANSCRIBE_DEADLINE:-$(date -u -d '2026-08-29 05:20:00' +%s)}
if [ ! -f "$OUT" ]; then
    {
        echo "# \`mb-coverage\` attempt 4 — machine-written run record"
        echo
        echo "One row per finished log in \`logs4/\`, appended by \`cov_transcribe4.sh\`"
        echo "as each run ends. **Nothing in this file was typed by a human or by an"
        echo "agent**: every field is \`grep\`-ed out of the log named in the row. It exists"
        echo "so that a night that outlives its agent still writes itself down."
        echo
        echo "| log | commit | wall clock | pytest summary | first assertion | marker lines |"
        echo "| --- | --- | --- | --- | --- | --- |"
    } > "$OUT"
fi
row() {
    local f="$1" n tmp commit dur sum err mark
    n=$(basename "$f" .log)
    tmp=$(mktemp); tr -d '\r' < "$f" > "$tmp"
    commit=$(grep -am1 '^commit:' "$tmp" | cut -d' ' -f2 | cut -c1-11)
    dur=$(grep -aoE 'in [0-9.]+s( \(0:[0-9:]+\))?' "$tmp" | tail -1)
    sum=$(grep -aoE '[0-9]+ (passed|failed|error)[a-z]*(, [0-9]+ [a-z]+)* in [0-9.]+s' "$tmp" | tail -1)
    err=$(grep -aoE '(TT_FATAL|TT_THROW|RuntimeError|AssertionError|ValueError): .{0,140}' "$tmp" | head -1)
    mark=$(grep -aoE '\[(dc8|dc9|pool|temperature|capacity|slots)\] [^"]{0,110}' "$tmp" | grep -av 'flush=True' | head -4 | tr '\n' ' ')
    rm -f "$tmp"
    printf '| `%s` | `%s` | %s | %s | %s | %s |\n' \
        "$n" "${commit:-?}" "${dur:-?}" "${sum:-NO-SUMMARY}" "${err:-—}" "${mark:-—}" >> "$OUT"
}
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    for f in "$D"/logs4/a4_*.log; do
        [ -f "$f" ] || continue
        n=$(basename "$f" .log)
        grep -aq '^exit=' "$f" || continue                  # still running
        grep -qF "\`$n\`" "$OUT" && continue                # already transcribed
        row "$f"
    done
    sleep 120
done
echo "$(date -u +%H:%M:%S) transcriber stopped at its deadline" >> "$D/logs4/watch4.log"
