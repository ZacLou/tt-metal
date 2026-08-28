# Milestone C — handoff brief

Written 2026-08-28 by `mb-signoff` (job 4 of the Milestone B job set), at commit `e912a8267bb` on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

> ## This is a handoff, not an authorisation to start
>
> `tttv2_2d_modules_plan.md` says *"Do not begin executor/vLLM integration until both models pass
> Milestone B"*, and **Milestone B does not pass** — see
> [`models/common/models/MILESTONE_B_STATUS.md`](models/common/models/MILESTONE_B_STATUS.md). One
> exit-gate line fails and two items of the milestone's own test list have no reachable case on either
> model. The gate exists so that Milestone C is not built on unqualified ground, and that is exactly
> the situation it is describing. **Close the Milestone B blockers first.** This document exists so
> that when the gate does open, Milestone C starts from the measured state of the tree rather than
> from archaeology.

## Where this sits

There are two Milestone C documents and they are complementary. **This one** is the plan-level
handoff the Milestone B signoff was asked to produce: what the next milestone inherits, measured,
with the log behind each claim. **`tttv2_milestone_c_briefs/`** is the seven-job execution set
(`README.md` plus `c0_defects.md` … `c5_signoff.md`), driven by `run_milestone_c_jobs.sh`, written
2026-08-28 17:42–17:47Z. Read this first for the state of the tree, then the job set for the order
of work.

**One thing the job set cannot know, because it was written first.** It was finished at 17:47Z and
`mb-coverage` attempt 4 started at 18:07Z, so **`tttv2_milestone_c_briefs/` contains no mention of
`D-C9` or `D-S1`, and does not know that `test_column_user_selector_wh_galaxy.py` has since run 3/3
on silicon** — the selector is qualified and is not the defect. Nothing in the job set contradicts
this brief; it is simply twenty minutes older than two of the findings. Take `D-C9` from here and add
it to `c0_defects`'s work before anyone reads a sampled token.

## Scope decision: vLLM is deferred and is not part of Milestone C

Taken 2026-08-28. Out of scope: the generator / `VLLMAdapter` boundary, the TT plugin's exact
model/version routing, the DP=4 logical-lane contract and global-capacity-32 mapping, async decode
output, and the vLLM server and offline smoke tests.

**In scope:** the model-owned executors for both models, integration with `models/common/llm_runtime`,
tracing, and the paired plus absolute performance gates.

`tttv2_2d_modules_plan.md`'s "Milestone C" and "Definition of Done" sections still describe the full
scope including vLLM. **They were deliberately not edited**: recording the split here is enough, and
this job's licence to edit the plan does not extend to rewriting its milestone definitions. Read those
sections with this paragraph beside them.

## Two consequences that change what to prioritise

**1. Prefill is sequential per row for Milestone C, at batch 32 decode.** Batched prefill applies only
under conditions Milestone C need not meet, so **concat-32 is not on the critical path**. `D-C6` is
still routed to Milestone C as *a fix to attempt* — the intent is to get it working — but it now has a
documented fallback, and the fallback is what the demos already do. Record it as a fix to attempt with
a fallback, not as a gate that must pass.

**2. Device sampling is on the critical path and has no fallback.** `D-C5` and `D-C8` stand between
this tree and any eager-versus-traced sampled-token comparison, and *"identical deterministic sampled
tokens between eager and traced execution"* is a Milestone C functional-gate line. `D-C9` has to be
fixed before either of them, because until it is, no sampled token read off this mesh is what the
sampler produced.

---

## What Milestone C inherits as working

Every command below was run on a real WH `(8, 4)` 6U Galaxy and its log is named. `HF_HOME` must be
exported or the real-checkpoint tests turn into silent `SKIPPED`s:

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
```

**One pytest process at a time, one node id per process, never piped.** The one-node-id rule is not
style: `LazyWeight`'s cache fingerprint keys on `MeshDevice.id()`, which changes per test, so every
test after the first in a process re-stages every weight — 138 GB and 26 minutes for Llama (`D-C3`).

| What works | Command that proves it | Evidence |
| --- | --- | --- |
| One Llama block, prefill 128 + decode batch 32, logits and both KV caches, PCC ≥ 0.99 | `pytest models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py::test_llama33_70b_galaxy_one_layer_prefill_and_decode` | `qwen/logs2/a2_40,41,42_llama_step2.log` — **3 fresh, bit-identical**, 0.99958 / 0.99975 |
| One Qwen block, same shape, real 64-head decoupled geometry | `pytest models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py::test_qwen3_32b_galaxy_one_layer_prefill_and_decode_8x4_qwen3_32b_b32_s128` | `qwen/logs2/a2_73,74,75_block.log` — **3 fresh**, 0.99930 / 0.99936 |
| The 80-layer Llama model, prefill + first decode token | `pytest .../llama33_70b_galaxy/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_full_model_prefill_and_first_decode_token` | `qwen/logs2/a2_44_llama_fullmodel.log`, 824.12 s |
| The 64-layer Qwen model, same | the Qwen equivalent | `qwen/logs2/a2_23,28_fullmodel.log` |
| **Llama teacher-forced accuracy**, batch 1, 512/511 | `pytest .../llama33_70b_galaxy/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1` | top-1 **501/511 = 98.04%**, top-5 **100.00%**. `qwen/logs2/a2_45`, `coverage/logs2/a2_g1` |
| **Qwen teacher-forced accuracy**, batch 1, 512 | the Qwen equivalent | top-1 **498/511 = 97.46%**, top-5 **100.00%**. `qwen/logs2/a2_33,34`, `coverage/logs2/a2_g12` |
| Batch-32 direct demos, no cross-slot contamination | `pytest models/common/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/demo.py::..._direct_demo_batch32_has_no_cross_slot_contamination` | `coverage/logs2/a2_g9`, `a2_g21`; `qwen/logs2/a2_22,31,32,47` |
| Batch-1 4K / 32K / 128K functional smokes | `..._long_context_smoke[4k|32k|128k]`, both models | `coverage/logs2/a2_g{3,4,5}`, `a2_g{14,15,16}` — **one run each**, and the assertion is only "the token is in vocabulary" |
| Prefix-cached prefill matches uncached | `..._prefix_cached_prefill_matches_uncached`, both models | `coverage/logs2/a2_g2`, `a2_g13` — one run each |
| Paged KV: late capacity, two pools agreeing, block-level cross-slot (Qwen) | `test_step7_coverage_wh_galaxy.py::..._late_capacity`, `..._two_paged_pools_agree_across_processes`, `..._cross_slot` | `coverage/logs2/a3_q_*`, `a3_l_*` |
| The Galaxy column user selector, and the selector→`Sampling2D` composition | `pytest models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py` | `coverage/logs4/a4_selector{,_run2,_run3}.log` — **3/3**, 49 s of mesh total, no checkpoint needed |
| Milestone A's **L3** — attention decode on the prefetch partition | any of the block gates above | Closed on silicon, at a named cost. See `MILESTONE_B_STATUS.md` |
| Host regression, `llm_runtime` byte-identical | `pytest models/common/tests/llm_runtime` | **1032 passed, 1 skipped, 0 failed**, 0 device opens |

**The operating rules that come with all of this:** one Galaxy model per process; prefill everything
before you decode anything; one node id per pytest process. Every one of them is a defect wearing a
workaround, and all three are listed below.

## What Milestone C inherits as broken or unqualified

### The four defects the earlier Milestone C brief never knew about

A Milestone C brief written on 2026-08-27 (deleted at `6983cc52e33`) was drafted from a dead mesh and
knew none of these. They are the substance of the handoff.

<a id="d-c5"></a>**D-C5 — the column user selector cannot accept either model's decode logits.**
`GalaxyColumnUserSelector.__call__` (`models/common/models/galaxy/collectives.py:445`) is one
`ttnn.matmul`, whose default program config requires input B **INTERLEAVED**. Both models hand it a
**WIDTH_SHARDED** tensor, because both set `decode_output_memcfg` from the shared recipe
`lm_head_output_memcfg` (`recipes.py:889`). Deterministic on both models, three fresh processes each
(`coverage/logs2/a3_q_greedy.log`, `a3_l_greedy.log`). **The fix has precedent 200 lines above it in
the same file**: `collectives._relocate_sharded` already stages through
`ttnn.sharded_to_interleaved(tensor, ttnn.DRAM_MEMORY_CONFIG)` and documents *why* that op and not
`to_memory_config` — it runs on its input's own `shard_spec.grid`, so it stays worker-confined under a
loaded sub-device manager. **Shared Galaxy code; blocking; no fallback.**

<a id="d-c8"></a>**D-C8 — behind D-C5, the same matmul violates the loaded decode sub-device's core
set.** With the logits relocated at the test boundary, the call gets further into the same function
and dies at `TT_FATAL @ program.cpp:2205`: the matmul builds its program over the whole compute grid
while the loaded decode sub-device manager owns only `prefetch_sender_cores() | worker_cores()`.
**Qualified at three fresh processes on both models** (`coverage/logs2/a3_{q,l}_dc5{,_run2,_run3}.log`)
— so it is not geometry-dependent and not a race. **This is a design decision, not a line**: does the
sampling path run inside the decode worker sub-device, or does decode's partition widen?
`recipes.rope_core_grids` already documents this defect class and names `_subgrid_cores` as the
qualified helper. A workaround exists and was measured — load the full-grid *prefill* sub-device
manager around the sampling call — but it is a test-boundary trick, not a design.

<a id="d-c7"></a>**D-C7 — a closed model does not return its L1, so a process gets one model.** After
`close()` **and** an explicit `gc.collect()`, the second model's `activate("decode")` finds
**923 776 of 1 393 472 bytes per L1 bank — 66% — still allocated**, largest free block 373 824 B
against the 792 064 B it needs (`coverage/logs2/a3_q_two_pools.log`). This is the same limitation
Milestone A recorded as **L1**, but Milestone A wrote it as a teardown-*ordering* problem and it is a
**lifetime** problem: no ordering can fix a buffer the destructor of a closed object did not free.
`mb-llama` also implemented and then **refuted on hardware** the obvious fix — releasing the global CB
on `activate("prefill")` left the clashing L1 base address identical, 544832 with and without it. The
flag survives, default off, as a record of the refutation. **Rests on one observation**; its run-2 and
run-3 were queued and the mesh died first. **Not observable on Llama** — its address clash arrives
first — so it is qualified on Qwen alone.

<a id="d-c6"></a>**D-C6 — concat-32 does not fit in L1 at any supported length, for either model, at
byte-identical figures.** 1 669 312 B of static circular buffers at length 128 against 1 499 136 B
available; 3 111 104 at 256; 5 994 688 at 512; 11 761 856 at 1024. Llama's 8-KV-head / 128256-vocab
geometry and Qwen's 64-head / 151936-vocab geometry cannot coincidentally need the same bytes: **it is
the shared concat-32 recipe, not per-model tuning.** The smallest supported length is already 11%
over; 1024 asks for 7.8×. Consequences: the whole of Milestone B's area 2 has **no reachable case**, so
nothing about padded-row isolation at active batch 16/31/32 has been measured **in either direction** —
do not read those seven failures as evidence that padded rows leak. **Per the scope decision above,
this is a fix to attempt with a documented fallback (sequential per-row prefill), not a gate.**

### And a fifth, found after all of the above

<a id="d-c9"></a>**D-C9 — the sampled-token readback composes the wrong mesh axis.**
`GalaxyDirectRunner.decode_sampled` (`direct_runner.py:535`) calls
`to_torch_auto_compose(sampled).reshape(-1)[:32]`. `to_torch_auto_compose` infers its composer from
`tensor.tensor_topology()`, an op's output inherits its *activation's* topology labels rather than the
distribution the weight mapper produced, and `ttnn.sampling`'s output here inherits from an all-gather
over the sampling axis — so the composer concatenates the replicas and the leading 32 values are **one
mesh column's eight users, four times over**. Measured byte-identically in two processes:
`[265, 2631, 1916, 220, 17, 15, 17, 17]` repeated four times.

**The identical trap is documented in the same repo, one op earlier in the same graph.**
`compose_galaxy_logits` exists precisely for it and `_compose_rows` was already fixed; `decode_sampled`
sixty lines below was not. The fix is one line with that precedent:
`ttnn.ConcatMesh2dToTensor(dims=(0, <user axis>))` then mesh row 0.

**Fix D-C9 before reading another device-sampling number.** Every area-4 measurement taken through
`decode_sampled` — including the 7/32 greedy agreement and the reciprocal-temperature reading — is a
readback measurement until it is. **The Milestone B exit gate is untouched**: `decode_logits` goes
through the fixed `_compose_rows`, and the batch-1, batch-32 and concat-32 demo tests all take the
default `GalaxySamplingPolicy(top_k=1, temperature=0.0)` whose `on_device` is `False`, so they sample
on the host. The one demo test that sets `on_device=True` (`demo.py:224-230`) is not a gate line and
is blocked at `D-C5`.

*One ambiguity, stated rather than hidden.* Two hypotheses fit: the readback composed the wrong axis
and the composed tensor held 64 values, or the sampler really produced one column's users and the
composition is innocent. The separating evidence is that element count, and no log prints it. A test
that does — `test_qwen_device_sampling_claims_with_an_explicit_token_composition`, committed, position
1 of `tttv2_milestone_b_evidence/coverage/queue4.txt` — **has never run**. Bet on the first; verify
before fixing.

### Items already routed to Milestone C by name

| ID | What | Where it is written up |
| --- | --- | --- |
| **L1** | `Prefetcher2D` global-CB ownership redesign. Bigger than Milestone A scoped it — see `D-C7`. `global_cb` as a property on the context may not be sufficient; the buffer's lifetime is the problem | `MILESTONE_A_STATUS.md` L1 and D-C; `MILESTONE_B_STATUS.md` L-B1 |
| **D-A** | Physical-32 real-device trace capture/replay at sequence length 128 and up. It needs a model-owned executor with `TraceCompiler`/`TracedExecutor` running a 2D model at batch 32, so it **genuinely could not be done before now** — Milestone C is the first milestone where anything exists to trace | `MILESTONE_A_STATUS.md` D-A; `tttv2_milestone_a_gap_briefs/gap3_batched_prefill_physical32_trace.md` |
| **Galaxy CCL / `tt_ccl.py` merge evaluation** | The plan's follow-up TODO 2 defers this until both models pass their milestones. Both models now *run*; require an API/ownership comparison and regression coverage for 1D and 2D users first. One input is already on record: the **D3** `semaphore_cores` invariant — narrowing a mode's semaphore allocation below its worker subdevice is safe only for a collective that binds its semaphore to a grid it owns | `MILESTONE_A_STATUS.md`, "CCL Follow-Up" |

### The rest of the inheritance, briefly

| ID | What | Status |
| --- | --- | --- |
| **L-B1** | Prefill-after-a-decode fails deterministically on Llama, 3/3 byte-identical, `program 100`, L1 buffer at 544832. **The untried hypothesis is the first thing to try:** make the prefill mode plan the *worker* cores instead of the whole grid, confining every prefill program to the 50 cores the global CB does not occupy. Cost: 20 of 70 cores for prefill, and every prefill number needs re-taking. Oracle: `test_llama33_70b_galaxy_batch32_slots_are_isolated` | open |
| **D-S1** | A device test that **passes** and then holds the mesh: every recorded run of the Qwen `HEAD_LOCAL` Q/K-norm decode test — six, four of them passing — is `SIGTERM`ed in teardown and exits 124, while the block test at the same commit exits 0 three times. Found by `mb-signoff` re-reading the logs; no earlier report mentions it | open, undiagnosed |
| **D-C1** | Decode accepts a prefill-shaped page table. Shape cannot discriminate; `memory_config()` can, and the validator never consults it. **Both the validator and the test that pins 32-row acceptance are Milestone B's own** — it is a contract decision, not a Milestone A expectation. The docstring at `attention_2d.py:678-679` already promises the rejection and is false | decision needed |
| **D-C2** | A sampling seed is keyed on `(seed, slot)`, so a request that migrates slots does not keep its stream — while the step-7 requirement asks for the opposite. The slot mixing is deliberate: it stops 32 slots given one seed from all emitting the same token | **product decision**, not a bug |
| **D-C3** | `LazyWeight._get_fingerprint` keys on `MeshDevice.id()`, not the mesh shape. One line in shared 1D/2D code; until then, one node id per pytest process | open, cheap |
| **D-C4** | `from_pretrained` cannot build a contiguous KV cache, so "paged vs contiguous PCC" has no expressible form at this adaptor API | contract gap |
| **F-C2** | `models/common/tests/models/galaxy/test_plans.py` looks host-only and is not — `ttnn.SubDevice` constructs the `MetalContext`. Its 14 tests should pass on a healthy mesh; if they do not, *that* is a finding | open |
| **Exit-gate line 9** | Five 1D demo-contract/hf-adaptor tests are red and are **not** Milestone B's: their files, their owning packages and `models/demos/utils` are all byte-identical to the Milestone A tip, and Milestone A's own gate never collected them | owner outside this work |
| **The mesh** | Unusable since 2026-08-28T18:37:08Z. Device 21 reads `0xffffffff`; `/dev/tenstorrent/{1,3,5,7}` raise `ENXIO`; the kernel logs `Device is unresponsive, cannot reset`. Five `tt-smi` paths failed. **Needs an operator** | infra |

**Check the mesh by opening nodes, not by counting them.** `ls /dev/tenstorrent | wc -l` was 32
throughout the outage. A healthy mesh opens all 32 nodes *and* `tt-smi -ls` exits 0 with no
`0xffffffff` in its output.

---

## Provenance warning — which documents in this tree carry dead-mesh evidence

Milestone C's agent will plan from whatever it is pointed at, and this tree contains pages written
before first silicon. This warning exists because that failure mode already cost `mb-coverage`
attempt 3 an hour of archaeology.

| Page | Trust |
| --- | --- |
| `models/common/models/MILESTONE_B_STATUS.md` | **Current.** Written 2026-08-28 from the logs |
| `models/common/modules/MILESTONE_A_STATUS.md` | **Corrected 2026-08-28**, surgically. Its 2026-08-27 edits (`6a3e78a7227`, +79 lines) were written from a dead mesh; the L3 verdict, the D-B and D-C deferral rows, the L1 update and the Qwen-geometry paragraph were all wrong and have been rewritten in place. **The Milestone A record proper — the 37-case sweep, D1–D5, the scorecard, P4 — was not touched and stands** |
| `models/common/modules/README.md` | **Corrected 2026-08-28.** Its `6a3e78a7227` edits (+24 lines) asserted that none of Milestone B was qualified on hardware and that the 80-layer model was never built. Both false |
| `tttv2_milestone_b_evidence/{llama,qwen,coverage}/REPORT.md` | **Current, but cumulative.** Each is a stack of dated attempt sections; **read the last section of each first**. Earlier sections are not retracted, only superseded, and each later section names what it supersedes |
| `tttv2_milestone_b_briefs/job3_completion_handoff*.md` | **A family, not a file.** `_attempt4.md` is current. Attempt 1's asserts a dead mesh and an untested `D-B9`; both are false |
| **Six committed source files** | **Stale and knowingly not corrected.** `llama33_70b_galaxy/{test_full_model,test_model}_wh_galaxy.py`, `qwen3_32b_galaxy/test_full_model_wh_galaxy.py`, `galaxy/test_column_user_selector_wh_galaxy.py` and both `demo.py` files still open with **"This file has never been executed."** Every one has since run on silicon; two of them produced the accuracy gates. A seventh: `collectives.py:402` still calls the column user selector *"Unqualified … has never run on a Galaxy mesh"*, and it is qualified 3/3. **`mb-signoff` left them alone deliberately** — the accuracy and demo gate rows are qualified *because* their producing files are byte-identical between the run's commit and `HEAD`, and a docstring edit would have downgraded five gate rows on a mesh that cannot re-measure them. Fix them in the same change that next touches those files on a working mesh |
| `tttv2_2d_modules_plan.md` | **Unedited**, and its Milestone C / Definition of Done sections still include vLLM — see the scope decision at the top of this brief |

---

## Performance methodology — set this up first, not last

Milestone C is measured against paired TTTv1/TTTv2 numbers, and retrofitting a paired harness onto
results already taken does not work. Stand it up before the first executor lands.

The plan requires, for every gated metric:

- the **same** WH Galaxy host;
- the **same** repository commit and firmware/runtime environment;
- the **same** checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling and KV
  setup;
- **one unmeasured warmup**;
- **three measured runs**;
- **compare medians**;
- **retain profiler artifacts and the exact commands.**

Gates: **no gated TTTv2 metric may regress by more than 3%** from its paired TTTv1 median, *and* the
absolute targets must be met:

```text
Llama, batch 32 / sequence 507      TTFT <=  99 ms   decode >= 71.5 tok/s/user   aggregate >= 2288 tok/s
Qwen,  batch 32 / sequence 507      TTFT <= 700 ms   decode >= 60   tok/s/user   aggregate >= 1920 tok/s
```

If an absolute target and the paired baseline disagree materially, the plan says to **stop and
document the environment and baseline discrepancy** rather than weaken either gate.

### Three performance debts already on the books

None of these is a correctness problem; all three will show up in the first paired measurement.

1. **The attention decode matmuls run on three worker columns of seven, and their weights are not
   prefetched.** Both are the cost of closing Milestone A's L3, and both are recovered by moving those
   two matmuls to the 24-core `gather_in0` ring. The ring wiring exists and is exercised by three
   matmuls; `attention_qkv_collective_input_memcfg` is already shaped for those 24 cores.
2. **The head-local Q/K norm relocates four times per call.** Correctness first, and this is the
   decode-latency consequence. The clean fix is a ttnn-level way to pass a `core_range_set` to
   `ttnn.rms_norm` for an interleaved input — the program factory already takes one
   (`layernorm_op_multi_core.cpp:193`) and only the low-level `create_descriptor` binding exposes it.
3. **`D-B7`'s relocation helper added DRAM round trips** to keep every relocation sub-device-legal.
   Recorded at the time as a correctness-first trade to be reclaimed later.

### Two measurement traps this milestone paid for

- **`TTTV2_GALAXY_CCL_TRACE=1` synchronises after each LM-head collective op.** It cannot change
  numerics and it dominates wall clock — a 511-token teacher-forced run took 1356 s with it and ~950 s
  without. **Never leave it on for a timed run.**
- **A bfloat16 cross-device logit sum is order-dependent on ETH ring arrival**, which produces per-row
  logit non-determinism and greedy flips. `fp32_dest_acc=True` on the LM head all-reduce is what buys
  determinism, and three bit-identical runs are the evidence that it is doing its job. If a
  performance change touches that flag, the accuracy gates stop being reproducible before they stop
  being correct.

## The method note worth carrying, in one sentence

Eighteen of Milestone B's twenty-seven fixed bring-up defects were **placement or partition** faults — a tensor, a
program grid or a circular buffer resolved somewhere the loaded sub-device manager does not own — and
**not one of them is visible to a host test, because a host test never loads a partition.** Four more
were in the *measurement apparatus* and all four failed open: wrong logits with no error, an unconsumed
prefetcher entry with no error, a reference loader that turned the accuracy gate into a silent skip.
Milestone C adds tracing, which is another layer of apparatus between the model and the number.
**Make the thing that produces a number prove itself before you believe the number.**
