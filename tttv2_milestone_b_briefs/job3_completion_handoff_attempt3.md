# Job 3 (`mb-coverage`) attempt 3 → `mb-signoff`: completion handoff

**Written progressively, as each result landed, so it survives a kill.** Attempt 2
died with no handoff of its own and cost the next attempt an hour of
archaeology; this file is rewritten in place at every checkpoint and the
"Last updated" line below is authoritative. Anything marked `IN FLIGHT` was still
running when that line was stamped.

Last updated: **2026-08-28 14:05Z**, and **not by the attempt-3 agent** — that
session died at `09:21:44Z`. Everything from "Read these paragraphs" down to the
areas table was written by the agent at 08:58Z; the sections marked *[operator
update]* were added afterwards by a later session, from the machine-written logs.
Status: **STOPPED, not finished** — no `state/mb-coverage.finished` marker exists,
and none should be written on this evidence.

Branch `apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Full account:
`tttv2_milestone_b_evidence/coverage/REPORT.md` §A3. Run-by-run index, one row
per run written as it finished: `.../coverage/RESULTS_A3.md`. Machine-written
verdicts, extracted from the logs rather than typed:
`.../coverage/VERDICTS_A3.txt`. Environment, costs and the harness:
`.../coverage/ENVIRONMENT.md`.

## Read these four paragraphs before anything else

**1. Attempt 3 ran as two agent invocations inside one driver run.** The first
started `07:37:58Z` at `af589dff4d5` and ended `08:16:43Z`; the driver relaunched
at once. The detached device queue (`cov_queue.sh`, reparented to init) never
stopped — it dequeued the next item one second after the relaunch — so the mesh
was continuously busy and nothing was lost or paid for twice. The second
invocation adopted the running queue instead of killing it. **The Llama build it
would have killed is the run that closed D-C5.**

**2. The mesh is alive and has been all night.** 32 boards on the bus, 32 device
nodes, a real 8×4 cluster opens in ~13 s. Attempt 1's "the mesh never came back"
is two days stale; do not plan from it.

**3. Device sampling does not work on this hardware at this tree, and it is two
defects deep.** This is the single most important thing this job found.
**D-C5**: `GalaxyColumnUserSelector.__call__` is a bare `ttnn.matmul` whose
default program config requires an INTERLEAVED input B, and the *shared* Galaxy
recipe makes both models' decode logits WIDTH_SHARDED — measured on silicon for
Qwen (`a3_q_greedy`) and for Llama (`a3_l_greedy`), same frame, same assertion.
**D-C8**: with that satisfied at the call site, the same line then fails
`TT_FATAL @ program.cpp:2205, Kernel group cores do not match sub device cores` —
the matmul builds its program over cores outside the loaded decode sub-device.
The brief's whole area 4 is behind these two.

**4. L1 has two signatures, and only one of them is the ordering problem the
brief describes.** The address clash is Llama-only at this tree (4/4 Llama, 0/6
Qwen, byte-identical across two commits and three fresh processes). The other —
**D-C7** — is that the L1 a *closed, dereferenced, garbage-collected* model held
is not returned, so the second model in one process cannot create its global
circular buffer: 923776 of 1393472 bytes per bank still allocated. Measured on
**Qwen**, the model that does not clash. No teardown ordering fixes that one.

## Exit-gate verdict

The measured table with a command behind every row is `REPORT.md` §A3, section
"The Milestone B exit gate — final table, measured". Summary:

| Gate line | Verdict |
| --- | --- |
| Llama teacher-forced 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS** — 98.04% / 100.00% |
| Qwen teacher-forced 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS** — 97.46% / 100.00% |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, both models |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, both models, all three geometries |
| Prefix-cached output matches uncached execution | **PASS**, both models |
| No dependency imports from a model-named implementation package | **PASS** for Milestone B; one pre-existing exception, finding F-C3 |
| Zero changes to 1D module implementation files | **PASS** — 0 of 384 changed paths |
| Zero changes to `llm_runtime` | **PASS** — 0 of 384 |
| Existing 1D contract/demo-contract host tests green, expectations unchanged | **FAIL**, 5 of 301, and not owned by Milestone B |

**On "re-measure at this tree, do not quote".** `git diff --name-only
718997518ab..HEAD -- models/` returns exactly one file, and it is a step-7 *test*
file that `test_full_model_wh_galaxy.py` and `demo.py` do not import. So the gate
logs are not older measurements of a changed thing — they are measurements of a
byte-identical thing, and §A3 states the commit for every row.

## Findings, and which need a human

`REPORT.md` §A3 "Findings, attempt 3" has the full write-up for each.

| ID | Needs | What |
| --- | --- | --- |
| **D-C5** | a fix in shared Galaxy code | selector matmul rejects both models' decode logits (WIDTH_SHARDED) |
| **D-C8** | a fix, and it is the harder half | with D-C5 removed, the same matmul violates the loaded decode sub-device's core set. The selector accepts no `program_config` and knows nothing about sub-devices |
| **D-C7** | the Milestone C L1 redesign | a closed model does not return its L1; one model per process |
| **D-C1** | a **decision** | decode's page-table validator cannot separate a prefill-shaped table from a legitimate L1-sharded repeat. Three attempts have declined the fix as a boundary violation |
| **D-C4** | a decision | `paged_attention_config=None` is the default pool, not a contiguous cache, so area 1's gate as *worded* is unreachable. Attempt 3 measured the reachable form instead |
| **D-C2** | a product decision | is a sampling seed per-request or per-(request, slot)? |
| **D-C3** | whoever owns `lazy_weight.py` | weight-cache fingerprint contains `MeshDevice.id()`; 138 GB and 26 min per extra model in a process |
| **F-C3** | `mb-signoff` wording | one pre-existing `models.demos` import under `models/common/tests/modules/moe/` |
| **D-C6, G-C1, G-C2, G-C3, F-C1, F-C2** | as §A2 leaves them | |

## Status of the five areas

*[operator update, 14:05Z]* — the queue ran on for four and a half hours after
the agent died and completed **38 more runs**. No `IN FLIGHT` cells remain. Every
cell below is transcribed from `logs2/<name>.log`; `RESULTS_A3.md` names the log
behind each one.

| Area | Llama | Qwen |
| --- | --- | --- |
| 1 paged KV | **PASS** two-pool PCC (cross-process, 1 run per arm) and late capacity; **BLOCKED** on block-level cross-slot and one-process two-pool — the L1 address clash, not D-C7 | **PASS** two-pool PCC (2 runs per arm), late capacity, block-level cross-slot (2 runs); **FAIL** two pools in one process — **D-C7** |
| 2 concat-32 | **FAIL, every case — D-C6**, not the clash | **FAIL, every case — D-C6** |
| 3 prefix / chunked | **PASS** prefix-vs-uncached, prefix-then-plain, mixed batch; **BLOCKED** on chunked (address clash) | **PASS** all four claims, 2 fresh processes each |
| 4 device sampling | **BLOCKED** by D-C5 then D-C8. D-C8 now **3/3** | **BLOCKED** by D-C5 then D-C8. D-C8 **3/3** |
| 5 long context | **PASS** 4K/32K/128K | **PASS** 4K/32K/128K |
| repeat & cleanup | **FAIL 3/3**, L1 address clash, deterministic | **PASS 3/3** on one live model; **FAIL** on two models in one process (D-C7) |

**The two verdicts that changed after the agent died**, and both change what
Milestone C inherits:

- **D-C6 is not Qwen-only.** Llama's step-7 concat-32 sweep fails with the same
  *capacity* overflow, at **byte-identical** figures at every shared length
  (1 669 312 B at 128, doubling per doubling, against 1 499 136 B of L1). It is a
  property of the shared concat-32 recipe, not of either model's geometry, and
  the smallest supported length is already 11% over. Area 2 has **no reachable
  case at this tree, for either model, at any length or active batch** — so the
  brief's active-16/31/32 isolation question was never asked, in either direction.
- **Area 1's headline claim now passes for both models.** The Llama cross-process
  pool comparison was run host-only from the two recordings already on disk:
  `[pool] all 32 slots agree at PCC >= 0.99 for prefill and decode`,
  `logs3/a3_h14_llama_pool_compare.log`. That row is a **new measurement made by
  the operator session**, not by the agent, and `RESULTS_A3.md` H14 says so.

## If you are attempt 4 rather than `mb-signoff`

Read `RESULTS_A3.md` first, not the report: one row per run, the log name, and how
many fresh processes each claim got. `queue.txt` is the resume point and is
consumed line by line by `cov_queue.sh`; anything still in it has not run.
Re-running what `RESULTS_A3.md` records is the only way to waste a Galaxy night.

### *[operator update, 14:05Z]* — the state you are actually inheriting

**Nothing is running.** Read this before you plan a night.

1. **The agent died at `09:21:44Z`, the queue did not.** The `stream-json` for
   session `d7d4b4ab-…` ends mid-wait with its background monitors killed and no
   `result` event. `cov_queue.sh` (PID 13308, reparented to init) kept running
   unattended and completed 38 further runs. So `RESULTS_A3.md` above row **L3**
   was written by the agent; everything below the `---` in that file was
   transcribed afterwards from the logs. **Both halves are equally real
   measurements** — the difference is only who wrote them down.
2. **The driver is deadlocked and the `ttmb` screen session is wedged.** The
   detached pty stopped draining at `08:16:44Z`; `run_milestone_b_jobs.sh` (PID
   10669) has been blocked ever since inside a `log()` call, in `tee`, writing to
   it. It will never log the job exit, never run its post-job device cleanup and
   never start another pass. **`run_milestone_b_jobs.sh` refuses to launch while a
   screen session named `ttmb` exists**, so that session and PID 10669 must be
   cleared before attempt 4 can start. Neither holds anything: the mesh is free
   and the evidence is on disk.
3. **The queue was halted at `13:48:41Z`** by an operator, at the user's request.
   `queue.halt` exists; delete it and run `nohup bash cov_queue.sh &` from the
   coverage directory to resume. **28 items remain and none of them has run** —
   the file is consumed destructively, so what is in it is exactly what is left.
   Its header records what was dropped and why. Do **not** rebuild the queue from
   gaps in `RESULTS_A3.md`; reconcile against `logs2/queue.out` and
   `VERDICTS_A3.txt`, which are machine-written and complete.
4. **The mesh is clean.** The killed run's wrapper ran its `glx_reset`; 32/32
   boards on the bus, no device-holding pytest.
5. **Perishable input.** The four cross-process pool recordings live in
   `/tmp/tttv2_step7_artifacts/` (~140 MB). `a3_h14` has already consumed the
   Llama pair and `a3_h12` the Qwen pair, so both comparisons are recorded — but
   if you want to re-run either, copy them somewhere durable first.

**What is worth device time, and what is not.** Of the 28 remaining items, six
(`a3_{q,l}_padded_greedy`, `_temperature`, `_seeded`) sit behind D-C5, which is
now qualified at three fresh processes on **both** models; they will spend ~40
minutes re-printing one `TT_FATAL` each. The concat-32 items are behind D-C6, now
established as model-independent at four lengths. What is genuinely unmeasured is
the **run-2/run-3 repeat tail** — most Llama claims have exactly one run, so they
are *observed, not qualified*, and the brief's three-fresh-processes rule is not
satisfied for them. If you get one night, spend it there and on the two defects,
not on re-confirming D-C5 and D-C6.
