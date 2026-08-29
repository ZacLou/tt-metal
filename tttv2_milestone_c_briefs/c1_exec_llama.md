# Job `c-exec-llama` — `Llama33_70BGalaxyExecutor`

**Device.** Exclusive WH Galaxy `(8, 4)`. One night, 10–12 h; the driver re-attempts you.

## Scope

Build the model-owned executor for `models/common/models/llama33_70b_galaxy/`, composing the common
runtime, and qualify it in **eager** execution. Tracing is `c-trace`'s job and is explicitly not
yours: build the executor so a `TraceCompiler`/`TracedExecutor` can be layered over *this exact
eager executor* later, and stop there.

Add:

```text
models/common/models/llama33_70b_galaxy/executor.py
models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py
```

No `generator.py`. **vLLM is out of this milestone** — no `VLLMAdapter`, no lane group, no request
normalisation at a serving boundary.

## What the executor composes

Per the plan and `models/common/llm_runtime/README.md` §"Composition and construction" and
§"Lifecycle":

- one full-mesh `Llama33_70BGalaxyTransformer2D`;
- one `PagedKVCacheManager` (`llm_runtime/paged_kv_cache.py`);
- one `OutputReader` (`llm_runtime/output_reader.py`);
- resolved prefill/decode configs;
- one `ProgramCompiler` (`llm_runtime/program_compiler.py`);
- one `EagerExecutor` (`llm_runtime/execution.py:46`);
- one `WarmupCoordinator` (`llm_runtime/warmup.py`);
- the model's `Prefetcher2D` and Galaxy CCL lifecycle.

**The executor is the resource and cleanup root**, and it is terminal after cleanup: it releases
resources in common-runtime order, with prefetcher/CCL resources released after outstanding work and
before mesh teardown.

**Mode activation is an operation-boundary lifecycle action.** The executor activates the required
prefetcher/CCL sub-device context before delegating to prefill or decode. It is never a static
branch inside a module hot path.

## Two references you may read and must not import

- `models/common/models/llama33_70b/executor.py` — the **1D** executor for the same model family.
  This is the closest thing to a worked example of the contract. Read it. Do not import it, do not
  extract from it, and do not change it.
- `models/common/models/galaxy/direct_runner.py` — `GalaxyDirectRunner`, the ad-hoc runner Milestone
  B used to reach silicon. It is your **behavioural reference**: `prefill_row`, `prefill_chunked`,
  `decode_logits`, `decode_sampled`, `_allocate_kv_cache`, `stage_chunk_page_table`,
  `teacher_forced_decode` encode a great deal of hard-won knowledge about what this model actually
  needs. **Leave it in place and working.** Every claim Milestone B qualified runs through it, and
  `c-signoff` compares against it.

## Prefill is sequential

Milestone C prefills **one row at a time**. `prefill_batched`/concat-32 is out of scope here
regardless of what `c-defects` concluded about D-C6 — read its handoff, and if D-C6 was deferred, do
not build anything that assumes concat-32 exists. Batch 32 applies to **decode**.

This is the single most likely place for a runtime change to look necessary. Before making one, read
the extension-discipline rules in `README.md`. `llm_runtime`'s planner hard-codes supported physical
batches `{1, 2, 4, 8, 16, 32}`; sequential per-row prefill at physical batch 1 is already inside
that set, so **the first implementation attempt must determine whether the Galaxy path works with
existing planner/runtime hooks plus model-owned input reshaping**. Only if that genuinely fails may
you add a frozen topology-neutral config value, and only then the smallest mechanical delegation to
it, preserving the current default exactly.

## Coverage

Qualify on silicon, three fresh processes per claim:

1. **Eager prefill** at 128, 512, 2048, single row — logits PCC ≥ 0.99 against the same request run
   through `GalaxyDirectRunner`, which is already qualified.
2. **Eager decode**, batch 1 and batch 32, first token after prefill.
3. **Paged KV**: late capacity resolution (`README.md` §"Resolve physical KV capacity"), transactional
   bind/unbind, per-layer KV metadata, KV PCC ≥ 0.99.
4. **Prefix-cached and chunked prefill** — both were qualified for Qwen at Milestone B and
   **blocked for Llama by the address clash**, which `c-defects` owns. If it fixed that, these are
   now reachable for the first time on Llama; if it did not, this is a `.blocked`-worthy dependency,
   not something to work around.
5. **Program compilation and warmup**: `WarmupCoordinator` completes; program identity uses physical
   geometry, not active row count.
6. **Repeated startup, serving and cleanup with no retained TT resources** — three cycles in one
   process. This is the gate line D-C7 and the address clash stand in front of; do not declare it
   met on one run.
7. **Teacher-forced accuracy** unchanged from Milestone B through the executor path: top-1 ≥ 91%,
   top-5 ≥ 99% at batch 1, prefill 512 / decode 511. Milestone B measured 98.04% / 100.00% through
   `GalaxyDirectRunner`; the executor must not lose accuracy.

## Prohibitions specific to this job

- **No `generator.py`, no vLLM anything.**
- **No changes to `models/common/modules/**/*_1d.py`.**
- **No Galaxy, Llama, Qwen, 2D-mesh or `(8, 4)` conditional in `llm_runtime`.** Any runtime change
  needs a focused test that fails without it and passes with it, plus
  `pytest models/common/tests/llm_runtime` green with unchanged expectations (1032 passed / 1
  skipped at Milestone B).
- **No imports from a model-named package** — not `models/demos/llama3_70b_galaxy`, not
  `models/common/models/llama33_70b`.
- **Do not delete or rewrite `GalaxyDirectRunner`.** It is the reference the executor is checked
  against, and `c-exec-qwen` and `c-signoff` both still need it.
- Never relax a threshold or `xfail` a case to get past it.

## Run procedure

House rules in `README.md`. Three fresh processes per claim; one pytest at a time, never piped;
`HF_HOME=/localdev/ctr-apbernal/hf_data`; a `skipped` in a run you meant to count is a failure of the
run. Run `test_partition_wh_galaxy.py` (13 s, no checkpoint) first whenever a decode program aborts
on placement.

Write your handoff progressively. Use a durable detached queue for long sweeps and write results as
they land.

## Deliverables

1. `executor.py` and its device test, committed.
2. `tttv2_milestone_c_evidence/exec_llama/REPORT.md` — one row per claim, the log behind it, and how
   many fresh processes it got. Distinguish *observed* (one run) from *qualified* (three).
3. `tttv2_milestone_c_evidence/exec_llama/logs/` — every log, never overwritten.
4. A **modularity note**: new files added, existing shared files changed and why config alone was
   insufficient, 1D module files changed (required: zero), default runtime behaviours changed
   (required: zero). `c-signoff` assembles the scorecard from these.
5. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`, and your completion handoff.

## Finish condition

Write the finish marker only when every one of the seven coverage items above is **qualified at
three fresh processes** with a log on disk, the modularity note shows zero 1D-module and zero
default-runtime-behaviour changes, and `pytest models/common/tests/llm_runtime` is green with
unchanged expectations.

If `c-defects` deferred something you depend on — the address clash in particular — say so
explicitly and mark the affected claims `BLOCKED (D-…)` rather than working around them. A workaround
in the executor for a defect in shared Galaxy code is the wrong place for the fix and will be
rejected at signoff.
