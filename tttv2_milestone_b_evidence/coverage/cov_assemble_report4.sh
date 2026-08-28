#!/bin/bash
# Append attempt 4's section to REPORT.md, from the fragments, in order.
# Idempotent: refuses if the marker is already there.
set -eu
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D"
grep -q '^# §A4 — attempt 4, the qualification pass' REPORT.md && { echo "§A4 already present; not appending"; exit 1; }
FRAGS="A4_SECTION_HEAD.md A4_GATE_TABLE.md A4_METHOD.md A4_AREAS.md A4_FINDINGS.md A4_INFRA.md A4_CLOSE.md"
for f in $FRAGS; do
    [ -f "$f" ] || { echo "missing fragment: $f"; exit 2; }
    grep -qE '@@[A-Z_]+@@' "$f" && { echo "unresolved @@PLACEHOLDER@@ in $f - refusing to assemble"; exit 3; }
done
# shellcheck disable=SC2086
cat $FRAGS >> REPORT.md
echo "appended; REPORT.md is now $(wc -l < REPORT.md) lines"
