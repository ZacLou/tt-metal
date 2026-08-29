# Job `c-signoff` — the Milestone C verdict

**No device.** Host-only. This job reads evidence and writes documents; it produces no new
measurements. If you believe you need the Galaxy, you have misread the job — say so and stop.

## Purpose

Decide, on the recorded evidence, whether Milestone C passes its exit gate, and write that verdict
down in a form the next piece of work can act on.

Milestone B's signoff had to correct a predecessor that wrote a status page from dead-mesh evidence
and asserted the opposite of what the tree showed. **That correction was the most valuable artifact
that pass produced.** Do not repeat the original error: every claim you record traces to a log, and
where a page in this tree disagrees with a log, the log wins and you say so.

## Method

For every gate line, find the **raw log** that produced the claim. A number in a `REPORT.md` that you
cannot trace to a log is not evidence; it is a note. Distinguish, in every table you write:

- **qualified** — three fresh processes, byte-identical or within tolerance;
- **observed** — it ran and passed, once or twice;
- **blocked** — it could not run, with the defect that stopped it;
- **not run** — nobody tried.

Milestone B's coverage evidence conflated these until `RESULTS_A3.md` separated them per row, and
that separation is what made the final verdict defensible. Keep it.

**Re-measure nothing, but re-check what changed.** For every claim you quote, confirm the files
behind it have not changed since the run: `git diff --name-only <run-commit>..HEAD -- models/`. If a
gate log predates a change to the thing it measured, say so and downgrade the claim. Milestone B's
signoff could show its gate logs measured a byte-identical tree; do the same or state the gap.

## Scope reminder

Milestone C is **executors, runtime integration, tracing and performance**. vLLM was deferred on
2026-08-28. `tttv2_2d_modules_plan.md`'s "Milestone C" and "Definition of Done" sections still
describe the full scope including vLLM, and **you may not edit them** — this job's licence to write
documents does not extend to rewriting the plan's milestone definitions. Record the deferral as a
decision, with its date, and list precisely what remains undone because of it.

## The exit gate, as it applies here

From the plan's Milestone C exit gate, minus the vLLM lines:

| Gate line | Where the evidence lives |
| --- | --- |
| Direct eager prefill/decode, both models | `exec_llama/REPORT.md`, `exec_qwen/REPORT.md` |
| Eager compilation and warmup | same |
| Traced decode | `trace/REPORT.md` |
| Traced eligible prefill | same |
| Explicit eager handling for trace-ineligible requests | same |
| Eager/traced logits PCC ≥ 0.999 | same |
| Identical deterministic sampled tokens, eager vs traced | same — **this one sits on D-C5/D-C8** |
| Paged-KV late capacity resolution | executor reports |
| Prefix caching and chunked prefill | executor reports — **Llama's was behind the L1 address clash** |
| Async decode read/complete | executor reports |
| Repeated startup, serving and cleanup with no retained TT resources | executor and trace reports — **this one sits on D-C7** |
| All paired performance gates (≤ 3% regression) | `perf/RESULTS.md` |
| All absolute performance targets | same |
| Accuracy gates still met in the performance configuration | executor reports + `perf/RESULTS.md` |
| No legacy model implementation imported by either package | your own grep, recorded |
| No 1D module implementation file changed | `git diff --name-only` against the milestone base |
| Existing 1D executor/runtime integration tests pass with original expectations | run them |
| Every non-config runtime change has a focused reduction and regression test | the jobs' modularity notes |
| Final modularity scorecard shows zero default-runtime behaviour changes | assembled by you |
| Cleanup is repeatable and terminal | executor and trace reports |
| Exact commands, measurements and revisions recorded | all of the above |

Three of these depend on `c-defects` having succeeded. If it deferred D-C6, note that concat-32 is
unavailable and that Milestone C's prefill is sequential by decision — that is not a gate failure.
If it left D-C5/D-C8, D-C7 or the Llama address clash open, the corresponding gate lines **fail**,
and the verdict says so.

## Deliverables

### 1. `models/common/models/MILESTONE_C_STATUS.md`

The verdict page. Model it on `models/common/modules/MILESTONE_A_STATUS.md`, which is the house
standard: a plain verdict up front, a gate table with evidence behind every row, defects with how
each one hid, and limitations stated as limitations. Include:

- the verdict, in the first paragraph, in one sentence;
- the gate table above, one row per line, each with its log and its qualified/observed/blocked/not-run
  label;
- every defect found during Milestone C, with the same "how it hid" treatment Milestone A's page
  gives — that section is what stops the next milestone rediscovering them;
- what carries forward as open, ranked;
- **the vLLM deferral**, its date, and exactly what is undone because of it: the generator, the
  `VLLMAdapter` boundary, plugin model/version routing, the DP=4 logical-lane contract, and the
  server/offline smokes at `--data_parallel_size 4 --max_num_seqs 8` with global capacity 32.

### 2. The modularity scorecard

The plan requires it at each milestone, and passing tests while violating the boundaries does not
count as a successful extension. Record:

- new 2D/model files added;
- existing shared files changed, and why config alone was insufficient for each;
- **1D module implementation files changed — required value: zero**;
- **default runtime behaviours changed — required value: zero**;
- 1D regression suites run and their result;
- any topology assumption discovered in common code;
- whether the extension stayed in module/config/model boundaries or leaked into orchestration.

Assemble it from the modularity notes each job wrote, and verify the zero-value rows yourself with
`git diff --name-only` against the Milestone C base commit
(`tttv2_milestone_c_runs/*/milestone_c_base.txt`). Do not take a job's word for a required-zero row.

### 3. A short handoff into whatever comes next

What works, with the commands that prove it. What is open, ranked, with evidence. The vLLM work as a
defined, deferred body of work rather than a gap. And a **provenance warning** naming any page in
this tree written from evidence that has since been contradicted — the same failure mode that cost
Milestone B's coverage job an hour of archaeology.

### 4. Documentation updates

`models/common/modules/README.md` and `models/common/llm_runtime/README.md` where Milestone C
changed a contract. Surgical edits, not rewrites, and check what is already there before appending —
a status page stops being readable when a second layer lands on top of a stale one.

## Prohibitions

- **Host-only.** No device runs. Note that "host-only" is not self-enforcing here:
  `models/common/tests/models/galaxy/test_plans.py` looks host-only and opens a cluster, because
  `ttnn.SubDevice` constructs the `MetalContext` (finding F-C2).
- **Do not edit `tttv2_2d_modules_plan.md`.**
- **Do not soften a verdict.** If the milestone did not pass, the page says NOT PASSED in its first
  paragraph. Milestone B's did, and that page is the reason this one can be trusted.
- Do not invent a number, and do not carry one forward that you could not trace to a log.

## Finish condition

`MILESTONE_C_STATUS.md` exists and states a verdict; the modularity scorecard is complete with its
required-zero rows verified by `git diff` rather than by assertion; the handoff is written; the
documentation updates are made; and every claim in all four traces to a log on disk.

Print the absolute path of `MILESTONE_C_STATUS.md` as your final line.
