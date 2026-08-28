#!/bin/bash
# Machine-written verdict line per attempt-4 device log, rewritten into
# VERDICTS_A4.txt. Same shape as VERDICTS_A3.txt: nothing here is typed by hand.
#
# Two things this got wrong on the first pass, both worth keeping in the comment:
# the ttnn logs carry carriage returns mid-line, so a naive `[^\r]` class
# truncates every assertion to its first word; and these logs are megabytes, so
# holding one in a shell variable and piping it with `printf '%s' "$txt"` silently
# loses matches. Both are fixed by normalising into a temp file and grepping that.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
out="$D/VERDICTS_A4.txt"
: > "$out"
for f in "$D"/logs4/a4_*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    tmp=$(mktemp); tr -d '\r' < "$f" > "$tmp"
    rc=$(grep -aoE '^exit=[0-9]+' "$tmp" | tail -1 | cut -d= -f2)
    sum=$(grep -aoE '[0-9]+ (passed|failed|error)[a-z]*(, [0-9]+ [a-z]+)* in [0-9.]+s' "$tmp" | tail -1)
    acc=$(grep -aoE 'top-[0-9] [0-9]+/[0-9]+ = [0-9.]+ \(gate >= [0-9.]+\)' "$tmp" | tr '\n' ' ')
    # Only the test's own printed marker lines. `flush=True` filters out the
    # copies pytest echoes back inside the failure traceback, which are source
    # text and not measurements.
    mark=$(grep -aoE '\[(dc8|dc9|pool|temperature)\] [^"]{0,130}' "$tmp" | grep -av 'flush=True' \
           | grep -aE 'agrees|repeated in|padded ids sampled|all 32 slots|tokens: \[|rows whose|composed' \
           | head -6 | tr '\n' ';')
    err=$(grep -aoE '(TT_FATAL|TT_THROW|RuntimeError|AssertionError): .{0,150}' "$tmp" | head -1)
    rm -f "$tmp"
    printf '%s rc=%s | %s | %s | %s | %s\n' "$n" "${rc:-NONE}" "${sum:-NO-SUMMARY}" "$acc" "$err" "$mark" >> "$out"
done
sed -i 's/[[:space:]]*$//' "$out"
wc -l < "$out"
