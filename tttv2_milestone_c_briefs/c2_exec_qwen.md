# Job `c-exec-qwen` — `Qwen3_32BGalaxyExecutor`

**Device.** Exclusive WH Galaxy `(8, 4)`. One night, 10–12 h; the driver re-attempts you.

## Scope

The same executor contract as `c-exec-llama`, for `models/common/models/qwen3_32b_galaxy/`. Read
`c1_exec_llama.md` in full — its "What the executor composes", "Prefill is sequential",
"Prohibitions" and run procedure apply here unchanged and are not repeated. Read `c-exec-llama`'s
completion handoff before you plan: it is one night old and supersedes this brief wherever they
disagree about the state of the tree.

Add:

```text
models/common/models/qwen3_32b_galaxy/executor.py
models/common/tests/models/qwen3_32b_galaxy/test_executor_wh_galaxy.py
```

No `generator.py`. vLLM is out of this milestone.

## The one thing this job is really testing

`c-exec-llama` built the first Galaxy executor. **This job is the test of whether that was a
reusable contract or a one-off.**

The plan's whole premise is that a new topology and two product models arrive through new 2D
modules, model packages and immutable configuration — not through edits to shared implementation
code. So the measure of success here is not only that Qwen works: it is **how much shared code you
had to change to make it work.**

Record every shared-code change with the reason config alone was insufficient. If `c-exec-llama`
left something Llama-shaped in `models/common/models/galaxy/`, this is where it surfaces, and moving
it into per-model configuration is in scope. If you find yourself adding a model-family branch
anywhere, stop: that is the failure the scorecard exists to catch.

**The ideal diff for this job is `executor.py`, its test, and nothing else.** Say plainly in your
handoff how close you came, and where you did not.

## What differs from Llama

- **Q/K norm.** `Attention2D` is built with per-head Q/K `RMSNorm2D` geometry. It is head-local
  normalisation, not hidden-dimension distributed norm; Milestone A's risk register names confusing
  the two as a specific hazard.
- **Geometry.** hidden 5120, intermediate 25600, head_dim 128, vocab 151936, 64 heads decoupled.
  Milestone B qualified the host adaptor at PCC ≥ 0.9999 against unmodified HF `Qwen3Attention`.
- **Qwen is the clean L1 reference on two shapes and the *only* model that can see D-C7.** The Llama
  address clash is Llama-only at this tree; D-C7's capacity residue was measured on Qwen precisely
  because Qwen does not clash. If `c-defects` fixed both, Qwen is where you verify D-C7 stayed
  fixed.
- **Accuracy gate**: top-1 ≥ 89%, top-5 ≥ 97% at batch 1, sequence 512. Milestone B measured
  97.46% / 100.00% through `GalaxyDirectRunner`; the executor must not lose accuracy.

## Coverage

The same seven items as `c-exec-llama`, on Qwen, three fresh processes each:

1. Eager prefill at 128, 512, 2048, single row — logits PCC ≥ 0.99 against `GalaxyDirectRunner`.
2. Eager decode, batch 1 and batch 32.
3. Paged KV: late capacity resolution, transactional bind/unbind, per-layer metadata, KV PCC ≥ 0.99.
4. Prefix-cached and chunked prefill. **Both already passed for Qwen at Milestone B** (2 fresh
   processes each) — so a failure here is a regression introduced by `c-defects` or by the executor,
   and must be reported as one, not absorbed.
5. Program compilation and warmup.
6. Repeated startup, serving and cleanup with no retained TT resources, three cycles in one process.
   Qwen passed the one-live-model form 3/3 at Milestone B and **failed the two-models-in-one-process
   form (D-C7)**. Both forms are required here.
7. Teacher-forced accuracy through the executor path, meeting the gate above.

Plus one item Llama cannot provide:

8. **Two models in one process**, the D-C7 shape, through the executor: build, use, close, build a
   second, use it. Three fresh processes.

## Deliverables

1. `executor.py` and its device test, committed.
2. `tttv2_milestone_c_evidence/exec_qwen/REPORT.md` — one row per claim, the log behind it, fresh
   process count, and *observed* vs *qualified* stated per row.
3. `tttv2_milestone_c_evidence/exec_qwen/logs/` — every log, never overwritten.
4. **The reuse note**, and it is the point of this job: every shared file changed to make Qwen work,
   why configuration alone was insufficient, and whether the change was Qwen-specific or a genuine
   generalisation. Zero shared-code changes is the target; anything else needs its reason.
5. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`, and your completion handoff.

## Finish condition

Write the finish marker only when all eight coverage items are qualified at three fresh processes
with a log behind each, the reuse note is complete, `pytest models/common/tests/llm_runtime` is green
with unchanged expectations, and there are zero changes to any `*_1d.py`.

**Llama's executor must still pass its own suite afterwards.** Re-run
`models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py` at the end and record it.
A shared-code change that makes Qwen work by moving Llama is not a pass, and this is the one job in
the milestone positioned to catch that.
