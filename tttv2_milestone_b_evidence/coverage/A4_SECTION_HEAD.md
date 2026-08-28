

# §A4 — attempt 4, the qualification pass

**Written by the attempt-4 agent, on 2026-08-28 from 18:07Z.** Attempt 3's
account (§A3) and the operator addendum (§A3-op) are above and are not amended
here: everything they record was re-verified against the tree and the logs
before this attempt planned a single run, and every claim in them held.

## What this attempt inherited, and what it did about it

The job was **not** in the state a naive reading of "five areas attempted" would
suggest, and it was also not in the state attempt 1 described. Three things were
established from the tree and the machine-written logs, not from the handoffs:

1. **The mesh was alive and free at 18:07Z** — and it did not stay that way.
   `ls /dev/tenstorrent | wc -l` was 32, no pytest held a device, and the `ttmb`
   screen session at 18:06:16Z was this attempt's own driver. Attempt 3's queue
   (`cov_queue.sh`) was not running and `queue.halt` was in place beside a
   `queue.txt` with 28 unconsumed items. **At 18:37Z the mesh broke and five
   recovery attempts failed** — see "The mesh went down at 18:37Z" below before
   planning anything. Thirty-one minutes of healthy Galaxy is all this attempt
   got.
2. **Every attempt-3 measurement applies at HEAD, and so does every attempt-2
   one.** `git diff --name-only b361770f46b..110ba1f0658` — from the commit all 38
   post-agent queue runs are stamped with, to the commit attempt 4 started at —
   is 14 paths and touches **no implementation file at all**: one status `.md`
   under `models/` and thirteen evidence/brief files. Widening to attempt 2's
   gate commit and running it at attempt 4's *final* tree,
   `git diff --name-only 1451b192584..HEAD -- models/` returns four paths:
   `models/common/models/MILESTONE_B_STATUS.md`, the two
   `test_step7_coverage_wh_galaxy.py` files, and the one host test file attempt 4
   added. Filtering that list for anything that is neither a `.md` nor under
   `tests/` returns **nothing**. None of those test files is imported by
   `test_full_model_wh_galaxy.py` or by either `demo.py`, so the nine exit-gate
   rows are measurements of byte-identical implementation code — a claim that is
   re-checkable in one command and is re-checked in
   `logs3/a4_h4_boundary_and_import_gates.log`.
3. **Two specific holes, and only one of them was worth a Galaxy night.**

   * **Never run at all.** `a3_{q,l}_padded_greedy`, `_temperature` and
     `_seeded` had **zero** device runs — the log directory is the proof, not the
     queue file: `logs2/` contains no file with any of those stems. Those are
     three of the four claims the brief's area 4 names, plus
     `test_two_models_in_one_process` on Llama, which is
     repeat-and-cleanup's second bullet. "All five areas attempted and recorded"
     was not yet true.
   * **Observed, not qualified.** Most Llama step-7 claims had exactly one fresh
     process. The house rule is three, and the reason is not pedantry: three of
     Milestone A's four defects presented as intermittent *passes*.

   What attempt 4 deliberately did **not** re-run: the concat-32 ladder (D-C6 is
   byte-identical at four lengths on both models) and the D-C5/D-C8 diagnostics
   (already 3/3 on both models). Re-confirming a deterministic abort is the one
   way to waste this hardware.

## Where to read what

| file | what |
| --- | --- |
| `RESULTS_A4.md` | one row per run, written as it finished, with the log name |
| `VERDICTS_A4.txt` | machine-written: exit code, pytest summary, first assertion and the test's own marker lines, `grep`-ed out of each log by `cov_verdict4.sh` |
| `RESULTS_A4_MACHINE.md` | the same record, written by `cov_transcribe4.sh` **without any agent awake** — one row per finished log, appended as each run ends, so a night that outlives its agent still writes itself down |
| `logs4/` | every attempt-4 device log, its reset log, and the five recovery logs. **`*.log` is in `.gitignore`**, so these are on disk and not in git — the same arrangement attempts 2 and 3 used |
| `logs3/a4_h*.log` | the **seven** host measurements at this tree: `h1` 1D contract gate, `h2` host regression gate, `h3` `llm_runtime`, `h4` boundaries and imports, `h5` the `top_k` contract audit, `h6` all three regression directories in one process, `h7` the D-C9 host corroboration |
| `queue4.txt` | the resume point. Consumed destructively, so what is in it has never run |
| `cov_watch4.sh`, `cov_transcribe4.sh` | the two detached helpers attempt 4 left running: one restarts the queue if the mesh recovers, one writes the record |
| `ENVIRONMENT.md` | the attempt-4 section at the end |
