#!/usr/bin/env python3
"""Parse the TTTv1 Galaxy demo metrics out of a run log.

Uses exactly the regexes documented in
tttv2_milestone_c_evidence/perf/BASELINE_PROCEDURE.md section 1.3.  Both
text_demo.py and text_qwen_demo.py emit textually identical lines, so one
parser serves both arms.

usage: extract_metrics.py <run.log> [<run.log> ...]
Exits non-zero if any log has no metric block, so a run that died before the
summary cannot be mistaken for a zero.
"""
import os
import re
import sys

PATTERNS = {
    "ttft_ms": r"Average Time to First Token \(TTFT\): ([0-9.]+)ms",
    "decode_ms_per_iter": r"Average speed: ([0-9.]+)ms @",
    "decode_tok_s_user": r"Average speed: [0-9.]+ms @ ([0-9.]+) tok/s/user",
    "decode_tok_s": r"Average speed: .*\(([0-9.]+) tok/s throughput\)",
    "tok1_decode_ms": r"1st token decode time: ([0-9.]+)ms",
    "tok128_decode_ms": r"128th token decode time: ([0-9.]+)ms",
    "prefill_compile_s": r"Prefill compile time: ([0-9.]+)s",
    "decode_compile_s": r"Decode compile time: ([0-9.]+)s",
}
COLS = ["label", "exit", "wall_s"] + list(PATTERNS)


def meta_of(path):
    d = os.path.dirname(os.path.abspath(path))
    out = {}
    try:
        for line in open(os.path.join(d, "meta.txt")):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    except OSError:
        pass
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    print("\t".join(COLS))
    missing = 0
    for path in argv[1:]:
        text = open(path, errors="replace").read()
        meta = meta_of(path)
        row = [
            meta.get("label", os.path.basename(os.path.dirname(path))),
            meta.get("exit", "?"),
            meta.get("wall_clock_s", "?"),
        ]
        found_any = False
        for name, pat in PATTERNS.items():
            m = re.search(pat, text)
            row.append(m.group(1) if m else "-")
            found_any = found_any or bool(m)
        if not found_any:
            missing += 1
        print("\t".join(row))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
