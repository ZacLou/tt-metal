# Job `c-defects` — clear what Milestone B could not

**Device.** Exclusive WH Galaxy `(8, 4)`. Expect more than one night; the driver re-attempts you.

## Why this job is first

Milestone B's exit gate did **not** pass, and three of its failures are Milestone C's own inputs.
This job owns them. It is not a coverage job and not a measurement job — Milestone B already
measured these to a byte. It is the job that **fixes shared Galaxy code**, and it is the only job in
this milestone whose deliverable is a code change rather than a qualification.

Read `tttv2_milestone_c_brief.md` (Milestone B's signoff) first. Then read
`tttv2_milestone_b_evidence/coverage/REPORT.md` §A3 for the full write-up of each defect below.
Every claim in this brief traces to a log named there. **Re-verify before you plan around any of
it** — but note that these particular findings are unusually well qualified: D-C8 and the Llama
address clash are each three fresh processes, byte-identical, across two commits.

## The five workstreams, in priority order

Work them in this order. They are ordered by what Milestone C cannot proceed without, not by
difficulty. **Each has its own completion marker** under
`tttv2_milestone_c_evidence/defects/` so a re-attempt resumes instead of restarting:

```text
tttv2_milestone_c_evidence/defects/<id>.status      one of: FIXED | DEFERRED | OPEN, plus the log path
```

---

### 1. D-C5 + D-C8 — device sampling, two defects deep. **Critical path, no fallback.**

`GalaxyColumnUserSelector.__call__` — `models/common/models/galaxy/collectives.py:445` — is a bare
`ttnn.matmul(selector, tensor)`, and it fails twice:

**D-C5.** The default multi-core matmul program config requires input B **interleaved**
(`matmul_device_operation.cpp:1233`). What it is handed is `LMHead2D.decode_forward`'s output under
`decode_output_memcfg`, which both models take from the *shared* recipe —
`recipes.py:889`, `lm_head_output_memcfg=width_sharded_memory_config(padded_local_vocab, ring)` —
so it is **WIDTH_SHARDED** for both. Measured on silicon for both models, same frame, same
assertion: Llama `a3_l_greedy`, Qwen `a3_q_greedy`.

**D-C8.** With D-C5 satisfied at the call site (`sharded_to_interleaved` to DRAM), the *same line*
then raises

```text
TT_FATAL @ tt_metal/impl/program/program.cpp:2205: num_intersections == num_cores
Kernel group cores do not match sub device cores for programmable core type TENSIX
```

Decode runs under a loaded sub-device manager (`Prefetcher2D._configure_mode`); a default multi-core
matmul resolves its grid from the tensors and the **full compute grid**, not from the loaded
sub-device. **Qualified at three fresh processes on both models** — `a3_q_dc5{,_run2,_run3}` at
156.06/157.88/154.84 s, `a3_l_dc5{,_run2,_run3}` at 897.12/470.61/435.44 s.

**The reduction, already done for you.** The selector needs *both*:

1. an input B the matmul accepts — interleaved, or a program config that takes width-sharded `in1`;
   **and**
2. a program config whose core grid is inside the decode worker sub-device.

`GalaxyColumnUserSelector.__init__` accepts `memory_config` and `compute_kernel_config` and passes
both to the matmul. It accepts **no `program_config`**, and nothing in it knows which sub-device is
loaded. That is the gap to close.

`collectives._relocate_sharded` (line 122) documents the governing constraint in the same file: a
direct `to_memory_config` between shard specs differing in grid *and* width resolves to
`reshard_program_factory_generic`, which builds over the full compute grid and is **illegal under a
loaded sub-device manager**; `sharded_to_interleaved` runs on its input's own `shard_spec.grid` and
`interleaved_to_sharded` on its output shard's cores, and both are worker-confined. Whatever you do
for (2) must satisfy the same rule.

**Also fix the test that could not see either fault.**
`models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py`, including
`test_column_user_selector_feeds_sampling_2d`, builds its input with
`memory_config=ttnn.DRAM_MEMORY_CONFIG` **and no loaded sub-device manager** — the one layout the
matmul accepts and the one layout the real model never produces. A fix that leaves that test as-is
has not closed the composition gap that hid these two defects. The test must exercise the layout the
LM head actually emits, under a loaded decode sub-device.

**Done when:** both models' device sampling runs end to end, and area 4's five claims — greedy vs
host argmax, the padded-vocabulary claim, seeded slot stability, the near-zero-temperature check for
Milestone A's D4, and per-slot heterogeneous controls — are evaluated on silicon, three fresh
processes each. The tests exist and are committed already; `RESULTS_A3.md` names them.

**Note the D-C2 decision is deferred, not resolved.** Whether a sampling seed is per-request or
per-`(request, slot)` was a product question that mattered because vLLM serving would be built on
it. vLLM is out of Milestone C. Record the current behaviour precisely in your handoff and **do not
change it** to settle the question.

---

### 2. D-C7 — a closed model does not return its L1. **Critical path, no fallback.**

`a3_q_two_pools` builds Qwen twice in one process, each inside `try/finally` with `close()`, `del`
and an explicit `gc.collect()`. The first pool completes a full 32-row prefill and a decode. The
second model **loads** and dies at its first `activate("decode")`:

```text
prefetcher_2d.py:431: in activate -> _ensure_global_cb(context)
TT_FATAL @ bank_manager.cpp:462  Out of Memory: Not enough space to allocate 55444480 B L1
buffer across 70 banks, where each bank needs to store 792064 B, but bank size is 1393472 B
(allocated: 923776 B, free: 469696 B, largest free block: 373824 B)
```

**923776 of 1393472 bytes per L1 bank — 66% — are still allocated after the owner was closed and
collected.** One model alone fits, 6/6. `Prefetcher2D.cleanup()` already does everything Python can:
stops the prefetch, deallocates every retained resource, sets `self._global_cb = None`, clears
`self._contexts`. So this is a **lifetime** problem, not the ordering problem Milestone A's
limitation L1 describes, and no teardown ordering fixes it.

This is the plan's "`Prefetcher2D` global-CB ownership redesign", routed to Milestone C by name. It
is on the critical path because **the executor is the resource and cleanup root** and
"repeated startup, serving and cleanup without retained TT resources" is a Milestone C gate line.

**Done when:** two full models can be built, used and closed in one process, with the second
creating its global circular buffer; three fresh processes; and `Prefetcher2D`'s cleanup contract in
`models/common/modules/README.md` says what is now guaranteed.

---

### 3. The Llama L1 address clash — **critical path for `c-exec-llama`.**

Not one of D-C5..D-C8, and included here deliberately: it is what stands between Llama and the
repeated-startup-and-cleanup gate.

```text
TT_THROW … Statically allocated circular buffers in program 100 clash with L1 buffers on
core range [0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region
ends at 630080
```

**Llama-only at this tree** — 0 of Qwen's 28 attempt-3 device runs match `clash with L1 buffers` —
and **deterministic**: `*_repeated_requests_and_deterministic_cleanup` failed 3/3 in three fresh
processes with byte-identical numbers across two commits, so it is a function of the resolved
placement and nothing else. It costs more than the shape it was found in:

| Claim | Log | Signature |
| --- | --- | --- |
| area 1, block-level cross-slot isolation | `a3_l_cross_slot` | `program 100`, L1 buffer at 544832 |
| area 1, two pools in one process | `a3_l_two_pools` | `program 100`, L1 buffer at 479296 |
| area 3, chunked prefill | `a3_l_chunked` | `program 1546`, L1 buffer at 543360 |
| repeat and cleanup | `a2_g6`, `+run2`, `+run3` | `program 100`, `[0-0 - 0-3]` |

It also **hid D-C5 for Llama** by killing the demo path before it ever reached the sampler, so
expect it to be hiding something else too. Qwen passes all four of those shapes, which makes it a
differential reference: the same code path, one geometry clashing and one not.

**Done when:** Llama passes `*_repeated_requests_and_deterministic_cleanup` three times in fresh
processes, and the three blocked claims above are unblocked and measured.

---

### 4. D-C6 — concat-32 does not fit in L1. **Attempt the fix; sequential prefill is the fallback.**

`validate_circular_buffer_region`, from `direct_runner.py:484` (`prefill_batched`), on core range
`[0-0 - 2-3]`:

| length | Qwen | Llama | L1 available |
| --- | --- | --- | --- |
| 128 | 1 669 312 B | **1 669 312 B** | 1 499 136 B |
| 256 | 3 111 104 B | **3 111 104 B** | 1 499 136 B |
| 512 | 5 994 688 B | **5 994 688 B** | 1 499 136 B |
| 1024 | *not run* | 11 761 856 B | 1 499 136 B |

**Byte-identical between two different model geometries at every shared length.** Llama's
8-KV-head/128256-vocab and Qwen's 64-head/151936-vocab cannot coincidentally need the same
1 669 312 B: the allocation is a property of the **shared concat-32 recipe**, so this is one defect
to fix once, not per-model tuning. The smallest supported length is already 11% over and the
requirement roughly doubles per doubling — 1024 asks for 7.8× the L1 that exists.

**This is the one workstream with an accepted fallback.** Milestone C's prefill is sequential per
row; concat-32 is not on its critical path. We want it working, so attempt it — but if it cannot be
made to fit within this job's nights, write `DEFERRED` in its status file with the measurements
behind that call, and say plainly in your handoff what Milestone C therefore does **not** have.
Deferring D-C6 is a legitimate outcome of this job. Deferring D-C5, D-C8, D-C7 or the address clash
is not.

**If you do fix it**, area 2's real question becomes askable for the first time: do padded rows
change an active row's logits at active batch 16, 31 and 32? All seven of Milestone B's runs died on
the overflow before a single row's logits could be inspected, so **nothing about padding-row
isolation has been measured in either direction.** Do not read those failures as evidence that
padded rows are fine.

---

### 5. Only if 1–4 are done: the decision items

`D-C1` (decode's page-table validator cannot separate a prefill-shaped table from a legitimate
L1-sharded repeat) has had three attempts decline the fix as a boundary violation. `D-C4` (area 1's
gate as *worded* is unreachable because `paged_attention_config=None` is the default pool, not a
contiguous cache). Both need a decision, not a night. **Do not decide them yourself.** State the
options and the consequence of each in your handoff, for a human.

---

## Prohibitions specific to this job

- **You are changing shared Galaxy code that two models depend on.** Every fix must be qualified on
  **both** models before you call it fixed, even when the defect only ever showed on one. The
  address clash looks Llama-only; prove your fix does not move Qwen.
- **Re-run the full Milestone B step-7 host suite after every fix** — `pytest
  models/common/tests/models/galaxy/test_step7_*.py` (162 tests at Milestone B, 3 fresh processes,
  identical). A shared-recipe change that quietly moves a passing claim is worse than the defect.
- **No changes to `models/common/llm_runtime/**` in this job.** None of these five defects is a
  runtime defect; they are all in `models/common/models/galaxy/` or `models/common/modules/`. If you
  believe otherwise, write the reduction and stop.
- **Do not rewrite `GalaxyDirectRunner` into an executor.** That is `c-exec-llama`'s job and it needs
  the runner intact as a behavioural reference.
- Never relax a threshold to turn a failure green, and never `xfail` a blocked case.

## Run procedure

The house rules in `README.md` govern. Three fresh processes per claim, one pytest at a time, never
piped, `HF_HOME=/localdev/ctr-apbernal/hf_data`. Run
`models/common/tests/models/galaxy/test_partition_wh_galaxy.py` (13 s, no checkpoint) first whenever
a decode program aborts on placement.

Use a durable detached queue for long sweeps and write results as they land — Milestone B's coverage
agent died with its queue still running and lost only bookkeeping, not silicon, because it did this.

## Deliverables

1. The fixes, committed, each with the test that fails without it and passes with it.
2. `tttv2_milestone_c_evidence/defects/<id>.status` for all five workstreams.
3. `tttv2_milestone_c_evidence/defects/REPORT.md` — one section per workstream: what the defect was,
   what the fix is, why it is the smallest fix that respects the module/model boundary, and the
   three logs behind each claim.
4. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`.
5. Your completion handoff, written progressively.

## Finish condition

Write the finish marker **only** when all of these hold, with a log on disk behind each:

- **D-C5 and D-C8**: device sampling runs end to end on both models, and all five area-4 claims are
  evaluated on silicon at three fresh processes each.
- **D-C7**: two models built, used and closed in one process, the second creating its global
  circular buffer; three fresh processes.
- **Llama address clash**: `*_repeated_requests_and_deterministic_cleanup` passes 3/3 in fresh
  processes for Llama, and the three claims it blocked are measured.
- **D-C6**: either fixed and qualified at lengths 128–2048 on both models, **or** its status file
  says `DEFERRED` with the measurements behind that call and your handoff states what Milestone C
  does not have as a result.
- The step-7 host suite is green with unchanged expectations, and
  `pytest models/common/tests/llm_runtime` is still 1032 passed / 1 skipped.
- Zero changes to any `*_1d.py` and zero changes under `models/common/llm_runtime/`.

If a workstream other than D-C6 cannot be advanced by any further attempt of this job, that is the
`.blocked` case — and it is held to the same evidence standard. One stuck workstream out of five is
**not** blocked: finish the other four and re-attempt.
