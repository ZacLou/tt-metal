# Job `c-trace` — tracing, parity, and mode transitions

**Device.** Exclusive WH Galaxy `(8, 4)`. One night, 10–12 h, **both models**; the driver
re-attempts you.

## Scope

Layer `TraceCompiler` and `TracedExecutor` (`models/common/llm_runtime/trace_compiler.py`,
`execution.py:261`) over the **exact eager executors** `c-exec-llama` and `c-exec-qwen` built — not
over a second, parallel construction path. Then prove the traced path is indistinguishable from the
eager one where it should be, and explicitly, loudly different where it must be.

Read both upstream handoffs before planning. They are one and two nights old and supersede this
brief wherever they disagree about the tree.

## The rule that shapes this job

**Traced execution never silently falls back to eager execution.** A trace-ineligible request must
be handled by an explicit eager path at the model boundary, with the miss visible — not absorbed
inside `TracedExecutor`. `llm_runtime/execution.py` already defines `TraceCoverageError` and
`PrefillReplayEvidence` for exactly this; use them rather than inventing a parallel mechanism.

The second rule: **trace identity is physical geometry.** Active batch size and runtime slot
assignment must not leak into program or trace identity — that causes trace explosion and stale
replay state. Active-row and slot data are refreshed as trace *input*
(`trace_compiler.py`: `PersistentInputs`, `InputRefreshPolicy`, `RefreshDecision`). Prove it: two
requests with different active row counts at the same physical geometry must hit the **same** trace.

## Coverage, both models, three fresh processes each

1. **Traced decode**, batch 32.
2. **Traced prefill** where eligible. Prefill is **sequential per row** in this milestone; do not
   build or test a concat-32 traced path regardless of what `c-defects` decided about D-C6.
3. **Explicit eager handling for trace-ineligible requests** — the miss is surfaced, not hidden.
4. **Eager vs traced logits PCC ≥ 0.999** for the same prepared request. Note the tighter threshold:
   this is the same graph twice, not a reference comparison.
5. **Identical deterministic sampled tokens** between eager and traced execution. This one sits
   directly on top of D-C5/D-C8 — if `c-defects` did not clear device sampling, this claim is
   `BLOCKED`, and you must say so rather than substituting host sampling and calling it met.
6. **Trace identity**: same physical geometry, different active rows → one trace, not two.
7. **Paged-KV late capacity resolution** under tracing.
8. **Mode transitions**, which the plan names as a required reduction and Milestone A's D3 already
   burned a night on. Compilation, capture, replay and cleanup must each be tested across:
   - decode → prefill;
   - prefill → decode;
   - repeated prefill;
   - repeated decode;
   - **failure during transition**;
   - cleanup from either active mode.

   The hazard is prefill/decode semaphore or stall-group state leaking across capture and replay. A
   hang here is most likely a sub-device/semaphore ownership fault, exactly like D3 — capture a
   traceback before spending a recovery attempt.
9. **Repeated startup, serving and cleanup with no retained TT resources**, with tracing active. The
   traced path allocates more, so this is a stronger test of D-C7's fix than the eager one was.

## Prohibitions specific to this job

- **No vLLM.** No generator, no `VLLMAdapter`, no `LaneGroupExecutor`, no DP work. Do not exercise
  `llm_runtime/lane_group.py`.
- **Do not build a second executor path for tracing.** `TracedExecutor` wraps the eager executor
  that already exists; if that is awkward, the awkwardness is a finding about the eager executor's
  shape, and it belongs in your handoff.
- **No changes to `models/common/modules/**/*_1d.py`**, and no Galaxy/model branch in `llm_runtime`.
  Any runtime change needs a focused failing-then-passing test plus
  `pytest models/common/tests/llm_runtime` green with unchanged expectations.
- Never relax the 0.999 threshold. If eager and traced disagree, that is the result — and it is an
  important one.

## Run procedure

House rules in `README.md`. Three fresh processes per claim; one pytest at a time, never piped;
`HF_HOME=/localdev/ctr-apbernal/hf_data`.

A `TT_FATAL` abort inside a multi-sub-device program leaves the mesh un-drainable — teardown blocks
in `FDMeshCommandQueue::~FDMeshCommandQueue`. This job runs more multi-sub-device programs than any
other, so budget a kill and a `tt-smi -glx_reset` after any such abort, and keep to the two-recovery
-attempt limit before recording `BLOCKED (infra)`.

Write your handoff progressively; use a durable detached queue for the transition matrix, which is
long and mechanical.

## Deliverables

1. Tracing wired into both executors, committed, with device tests under each model's test
   directory.
2. `tttv2_milestone_c_evidence/trace/REPORT.md` — one row per claim per model, log behind each, fresh
   process count, *observed* vs *qualified*.
3. **The transition matrix as a table** — six transitions × four phases (compile, capture, replay,
   cleanup) × two models, each cell with its log. This is the artifact the plan asks for by name.
4. `tttv2_milestone_c_evidence/trace/logs/`, never overwritten.
5. A modularity note in the same form as the executor jobs.
6. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`, and your completion handoff.

## Finish condition

Write the finish marker only when, **for both models**, items 1–9 are qualified at three fresh
processes with a log behind each; the transition matrix has no empty cell; eager-vs-traced logits
PCC ≥ 0.999 and deterministic sampled tokens are identical; and
`pytest models/common/tests/llm_runtime` is green with unchanged expectations.

If device sampling is still blocked by D-C5/D-C8, item 5 cannot be met and **the finish marker must
not be written**. Record it as `BLOCKED (D-C5/D-C8)` with the log, finish everything else, and say in
your handoff exactly which gate line is short. `c-perf-paired` can still proceed on a traced decode
that samples on the host; `c-signoff` cannot pass the milestone with item 5 unmet, and that is the
correct outcome to report rather than to engineer around.
