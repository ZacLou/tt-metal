#!/usr/bin/env python
"""Machine-write the per-run evidence table for job c-exec-llama.

Reads every log under `tttv2_milestone_c_evidence/exec_llama/logs/`, pulls the
pytest verdict, the wall clock, the `[exec]` / `[kv]` / `[probe...]` measurement
lines and any clash address, and prints one markdown row per run in the order the
runs happened. Prose in REPORT.md is written by hand; this table is not, for the
same reason RESULTS.md is not.

    python tttv2_milestone_c_runs/c-exec-llama/summarize.py
"""

from __future__ import annotations

import re
from pathlib import Path

LOGS = Path("tttv2_milestone_c_evidence/exec_llama/logs")
VERDICT = re.compile(r"(\d+ (?:passed|failed|error)[^=\n]*?in [\d.]+s[^)\n]*\)?)")
CLASH = re.compile(r"clash with L1 buffers on core range (\[[^\]]+\])\. L1 buffer allocated at (\d+)")
MEASURE = re.compile(r"^\[(?:exec|kv|probe[^\]]*)\].*$", re.M)
ERROR = re.compile(r"^E\s+((?:[A-Za-z_]*Error|Failed|assert)[^\n]*)$", re.M)


def summarize(path: Path) -> tuple[str, str, str]:
    text = path.read_text(errors="replace")
    verdicts = VERDICT.findall(text)
    verdict = verdicts[-1].strip() if verdicts else "no pytest summary"
    notes: list[str] = []
    for line in MEASURE.findall(text):
        notes.append(line.strip())
    clash = CLASH.search(text)
    if clash:
        notes.append(f"L1 clash at {clash.group(2)} on {clash.group(1)}")
    if not clash:
        error = ERROR.search(text)
        if error:
            notes.append(error.group(1).strip()[:180])
    return verdict, " · ".join(notes[:8]), path.name


def main() -> None:
    entries = []
    for log in LOGS.glob("*.log"):
        entries.append((log.stat().st_mtime, log))
    entries.sort()
    print("| run | verdict | measured |")
    print("| --- | --- | --- |")
    for _, log in entries:
        verdict, notes, name = summarize(log)
        print(f"| `{name[:-4]}` | {verdict} | {notes or '—'} |")


if __name__ == "__main__":
    main()
