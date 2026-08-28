# Job 4 — Milestone B exit-gate verdict and scorecard

**No device.** Host-only. This job reads evidence and writes documents; it produces no new
measurements and must not attempt any.

## Mission

Decide, on the recorded evidence, whether Milestone B passes its exit gate — and write that verdict
down in a form the next milestone can act on.

The single most important instruction in this brief: **this job is allowed to conclude that Milestone
B does not pass.** Milestone A declared its exit gate passed on 2026-08-19, was wrong, and the
independent re-run that disproved it found two real defects the "passing" evidence had been masking.
That correction is the most valuable artifact Milestone A produced. Do not repeat the original
mistake here.

## Inputs

- `tttv2_milestone_b_briefs/job3_completion_handoff*.md` — start here, at the **newest** one. That
  name is a family, not a file: the un-suffixed document belongs to attempt 1 and each later attempt
  writes `_attempt<k>.md` beside it without overwriting what came before. Resolve it by scanning, never
  by fixed name — attempt 1's asserts a dead mesh and an untested D-B9, both false since. Where two
  disagree the newest wins; it was written last, by an author who had the older ones in hand. If none
  exists, report `BLOCKED (mb-coverage did not complete)` and write the status page from whatever
  evidence does exist, clearly marked as partial.
- `tttv2_milestone_b_evidence/{reconcile,llama,qwen,coverage}/REPORT.md` and their raw logs.
- `tttv2_2d_modules_milestone_b_work_log.md`, all checkpoints.
- `tttv2_2d_modules_plan.md` — "Milestone B exit gate" and "Modularity scorecard".
- `models/common/modules/MILESTONE_A_STATUS.md` — the model for how to write this, including its
  tone about its own errors.

## What to check, and how sceptically

For every exit-gate line, find the **raw log** that produced the claim. A number in a `REPORT.md` that
you cannot trace to a log file is not evidence; mark it `UNSUBSTANTIATED` and say which report asserted
it.

```text
Llama teacher-forced, batch 1, prefill 512 / decode 511    top-1 >= 91%   top-5 >= 99%
Qwen  teacher-forced, batch 1, sequence 512                top-1 >= 89%   top-5 >= 97%
Batch-32 direct demos valid, no cross-slot contamination
Batch-1 4K / 32K / 128K functional smokes pass
Prefix-cached output matches uncached execution
No dependency imports from an existing model-named implementation package
Zero changes to 1D module implementation files
Existing 1D model contract and demo-contract host tests green, expectations unchanged
```

Apply Milestone A's hard-won standards to each:

- **Was it run more than once, in fresh processes?** Three of Milestone A's four defects presented as
  intermittent passes. A single passing run is not a qualification; record it as
  `PASSED (single run — not qualified)`.
- **Was it measured at the final tree?** Evidence collected before a later shared-code change is
  provenance, not current evidence. `mb-coverage` was told to re-measure the accuracy gates for
  exactly this reason; check that it did.
- **Did anything pass because its coverage could not reach the defect?** That is how D4 and D5
  survived — greedy-only sampling could not reach a temperature bug, and uniform memory configs could
  not reach a swapped pair. For each passing line, ask what a defect would have to look like to slip
  through it.

The last three lines are mechanical. Verify them yourself rather than trusting a report:

```sh
git diff --name-only <milestone-a-final>..HEAD | grep '_1d\.py'      # must be empty
git diff --name-only <milestone-a-final>..HEAD | grep 'llm_runtime'  # must be empty
git grep -n "demos.llama3_70b_galaxy\|models.llama33_70b\b\|models.qwen3_32b\b" -- \
    models/common/models/galaxy models/common/models/*_galaxy
python -m pytest -q models/common/tests/modules models/common/tests/models \
                    models/common/tests/llm_runtime
```

## Deliverables

### Before you start: two of your deliverables were deliberately deleted

**`models/common/models/MILESTONE_B_STATUS.md` and `tttv2_milestone_c_brief.md` do not exist, and
their absence is intentional. Write them fresh. Do not `git checkout` the old ones back.**

A signoff pass already ran once, on 2026-08-27, and committed both at `6a3e78a7227`. It was working
from a dead mesh — 21 of 32 boards on the bus, `ttnn` unable to open a cluster at all — and its
verdict reflects that: *no numerical result of any kind has ever been produced on silicon for either
model*, 4 of 9 gate lines `NOT REACHED`, 33 committed device tests never executed. **Every one of
those claims is now false.** The mesh came back, `mb-llama` and `mb-qwen` both declared their finish
conditions met, and `mb-coverage` measured 8 of the 9 lines as passing, including both accuracy
gates. The two documents were deleted at `6983cc52e33` rather than edited, because a missing file
fails loudly where a stale one is believed.

The verdict may well still be **NOT PASSED** — line 9 fails, and areas 2 and 4 are blocked by real
defects. But the *reason* has inverted, from "this milestone was not allowed to be measured" to "this
milestone was measured and these specific defects hold the gate". Say which one you are recording.

### 1. `models/common/models/MILESTONE_B_STATUS.md`

Modelled on `models/common/modules/MILESTONE_A_STATUS.md`, and honest in the same way. It needs:

- a **Current position** table: what is qualified, what is not, and the exit-gate verdict stated
  plainly in the first screen — not buried after the evidence;
- a **Verification status** table, one row per area (Llama block, Llama full model, Qwen block, Qwen
  full model, paged KV, concat-32, prefix cache, device sampling, long context, repeat/cleanup), each
  with its host evidence, its device evidence, and a status that distinguishes *qualified* from
  *passed once*;
- a **Defects found** table in the D1–D5 form — defect, how it hid, fix. "How it hid" is the column
  that teaches something; write it properly;
- **Known limitations, documented and accepted**, each with an anchor and a target milestone, in the
  L1/L2/L3 style;
- **Pending work**, split into blocking and deferrable, each deferrable item naming the milestone it
  belongs to;
- the **exit-gate result table**, requirement by requirement;
- the **modularity scorecard** from the plan: new files added; existing shared files changed and why
  config alone was insufficient; 1D module implementation files changed (required: zero); default
  runtime behaviours changed (required: zero); 1D regression suites run and their result; topology
  assumptions discovered in common code; whether the extension stayed inside module/config/model
  boundaries.

The plan is explicit that the scorecard is project evidence in its own right: *"Passing model tests
while violating these boundaries does not count as a successful TTTv2 extension."* If the boundaries
held, show it. If they did not, say so — that is a finding, not a failure of this job.

### 2. Documentation updates

- `models/common/modules/README.md` — the 2D inventory, the final module contracts as they now stand
  after job 0's amendments, and the Galaxy model packages. Remove any line still saying Milestone A is
  in progress if the Milestone A branch has since closed it; if it has not, leave it and note the
  dependency.
- `models/common/modules/MILESTONE_A_STATUS.md` — **only** the items Milestone B closed or changed:
  the L3 verdict from `mb-llama`, and any Milestone A limitation Milestone B resolved or proved
  worse. Surgical edits, not a rewrite — it is the signed-off Milestone A record, and you are
  appending to it rather than restructuring it.

  **But re-check what is already there before you append.** The 2026-08-27 signoff pass
  (`6a3e78a7227`) already made **+79 lines** of edits to this file and **+24** to
  `modules/README.md`, from the same dead-mesh evidence that produced the two deleted documents. Its
  L3 verdict in particular was written before first silicon settled the question. Run
  `git show 6a3e78a7227 -- models/common/modules/MILESTONE_A_STATUS.md models/common/modules/README.md`
  and check each claim it added against `tttv2_milestone_b_evidence/llama/REPORT.md` and
  `.../coverage/REPORT.md` §A3. Correct what silicon has since contradicted; leave the rest. **Do not
  append a second layer on top of a stale one** — that is how a status page stops being readable.
- A final checkpoint in `tttv2_2d_modules_milestone_b_work_log.md`.

### 3. `tttv2_milestone_c_brief.md`

A short, honest handoff into Milestone C — executors, runtime integration, tracing and performance.

**Scope decision, taken 2026-08-28, after this brief was first written.** Milestone C has been
narrowed: **vLLM is deferred and is not part of it.** The generator/`VLLMAdapter` boundary, the TT
plugin's model/version routing, the DP=4 logical-lane contract and the server/offline smokes are all
out. What remains is the model-owned executors for both models, the common-runtime integration,
tracing, and the paired plus absolute performance gates. Write the handoff for **that** scope: note
the deferral explicitly and once, so the record shows it was a decision and not an omission, and do
not spend a section planning the serving work. The plan's "Milestone C" and "Definition of Done"
sections still describe the full scope including vLLM — say so, and **do not edit them to match**;
recording the split is enough, and this job's licence to edit the plan does not extend to rewriting
its milestone definitions.

Two consequences worth stating in the handoff, because they change what the next job prioritises:

- **Prefill is sequential per row for Milestone C, batch 32 decode.** Batched prefill applies only
  under conditions Milestone C need not meet, so concat-32 is not on its critical path. **D-C6 is
  still routed to Milestone C as a fix to attempt**, because the intent is to get it working — but
  it now has a documented fallback (sequential prefill) rather than being a blocker. Record it that
  way: a fix to attempt with a fallback, not a gate that must pass.
- **Device sampling is on the critical path and has no fallback.** D-C5 and D-C8 stand between this
  tree and any eager-versus-traced sampled-token comparison, which is a Milestone C gate line.

It should carry:

- what Milestone C inherits as working, with the commands that prove it;
- what it inherits as broken or unqualified, with the evidence;
- the items already routed to it by name: **L1** (`Prefetcher2D` global-CB ownership redesign),
  **D-A** (physical-32 real-device trace, which needs a model-owned executor and so genuinely could
  not be done before now), and the **Galaxy CCL / `tt_ccl.py` merge evaluation** the plan defers until
  both models pass;
- **the four defects the deleted brief never knew about**, each with its evidence: **D-C5** and
  **D-C8** (device sampling is blocked two defects deep, both in shared Galaxy code, D-C8 qualified at
  three fresh processes on both models), **D-C7** (a closed model does not return its L1 — one model
  per process), and **D-C6 escalated** (concat-32 does not fit in L1 at any supported length, for
  either model, at byte-identical figures — it is the shared recipe, not per-model tuning, and it is
  a prerequisite for area-2 coverage rather than a finding that coverage produced);
- **a provenance warning naming the documents in this tree that carry dead-mesh evidence.** Milestone
  C's agent will read whatever it is pointed at and plan from it. Say plainly which pages were written
  before first silicon and have not been re-checked — at minimum the `6a3e78a7227` edits to
  `models/common/modules/{MILESTONE_A_STATUS.md,README.md}`, and any of it you did not correct under
  deliverable 2 — so the next job knows which of its inputs to verify against a log before trusting.
  This is the same failure mode that cost `mb-coverage` attempt 3 an hour of archaeology: a brief
  written before the attempt that superseded it;
- the performance-methodology requirements Milestone C will be measured against, so it can set up
  paired TTTv1/TTTv2 measurement from the start rather than retrofitting it: same host, same commit
  and firmware, same checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling and
  KV setup; one unmeasured warmup; three measured runs; compare medians; retain profiler artifacts and
  exact commands.

## Finish condition

`MILESTONE_B_STATUS.md` exists and states a verdict, the documentation updates are made, the Milestone
C brief is written, and every claim in all three traces to a log. Print the absolute path of
`MILESTONE_B_STATUS.md` as your final line.

If the honest verdict is that Milestone B does not pass, say so in the first paragraph and list
exactly what remains. Do not begin Milestone C work; the plan gates it, and the gate is the point.
