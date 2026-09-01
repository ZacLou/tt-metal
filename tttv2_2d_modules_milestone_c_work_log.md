# TTTv2 2D Modules Milestone C Work Log

> **Scope reminder.** Milestone C was narrowed on 2026-08-28: model-owned executors for both
> models, common-runtime integration, tracing, and the paired plus absolute performance gates.
> **vLLM is out of scope entirely.** Prefill is sequential per row; batch 32 applies to decode.
> `tttv2_2d_modules_plan.md` still describes the full scope including vLLM; that difference is the
> decision, not drift, and no job here edits the plan.

## Checkpoint C-1 — `c-perf-recon`: the TTTv1 baseline procedure

- Date: **2026-08-29**, 08:25Z – 12:20Z. Job `c-perf-recon`, attempt 1.
- Host: `wh-glx6u-05-special-ctr-apbernal-for-reservation-119144`, WH Galaxy 6U, 32 boards.
- Commit `6af44349413ca6ce2c0d98f5b26dd2898dc1f067`, branch
  `apbernal/tttv2_wh_glx_2d_modules_milestone_c`, tracked working tree clean throughout.
- **Zero Milestone C code touched.** Nothing under `models/common/models/{galaxy,llama33_70b_galaxy,
  qwen3_32b_galaxy}`, `models/common/modules/`, `models/common/llm_runtime/` or
  `models/demos/llama3_70b_galaxy/` was modified. `git status --porcelain --untracked-files=no` is
  empty at the end of the job as it was at the start.

**Answer to the milestone's largest unknown: TTTv1 runs on this host, both models.** Six device
runs, three successful in fresh processes per model, all `exit=0`, all producing coherent text.

Deliverables:

- `tttv2_milestone_c_evidence/perf/BASELINE_PROCEDURE.md` — exact command per model, provisioning,
  metric extraction (file/line/units), TTFT semantics, per-phase and total wall clock, the schedule
  `c-perf-paired` inherits.
- `tttv2_milestone_c_evidence/perf/ENVIRONMENT.md` — host, mesh, firmware (uniform across 32
  boards), host software, checkpoints, cache paths, and the two infrastructure hazards below.
- `tttv2_milestone_c_evidence/perf/recon/` — every log verbatim, including the two failed attempts.
- `tttv2_milestone_c_runs/{run_recon.sh,assert_environment.sh,extract_metrics.py,envs/}` — the
  driver, the environment re-assert/diff, and the metric parser.
- `tttv2_milestone_c_briefs/recon_completion_handoff.md` — the handoff.

Measured, **unpaired reconnaissance at this commit, not a gate result** — medians of three runs:

```text
Llama-3.3-70B   TTFT 18697.89 ms   decode 67.31 tok/s/user   aggregate 2153.93 tok/s
Qwen3-32B       TTFT  1477.88 ms   decode 61.54 tok/s/user   aggregate 1969.29 tok/s
```

Findings that change how later jobs must be run:

1. **TTFT means LAST USER READY.** The timed region wraps one `prefill_forward_text` call covering
   all 32 users and is never divided by batch size (`text_demo.py:1233-1278,1602-1608`;
   `text_qwen_demo.py:796-845,1181-1187`).
2. **At the gated configuration TTTv1's own prefill is sequential per user**, not concat-32:
   `temperature: 0.0` sets `requires_slot_stable_prefill` (`generator.py:478-497`) and the log shows
   `use_batched_prefill: False` for users 0..31. The paired TTFT comparison is therefore
   like-for-like with Milestone C's sequential prefill.
3. **The absolute TTFT targets are unreachable for TTTv1 itself** — 189× for Llama, 2.1× for Qwen —
   and the cause is dated: the `99.0` target landed 2026-06-06 (`1fe9df83b61`), three days before
   `290155969315` (#45532) moved greedy batches onto the sequential prefill path. Nothing
   re-baselined it. Documented, not adjudicated, per the plan.
4. **A green pytest is not a met gate.** `verify_perf` emits a `PerfRegressionWarning` and returns;
   it never asserts. The Qwen demo reports `Perf Check Failed!` on all three runs and still PASSES.
   The Llama demo never even checks the batch-32 entry (`text_demo.py:1690` gates on `repeat2`,
   which is batch 1).
5. **`text_demo.py` cannot run against a HuggingFace checkpoint** (D-R1). It needs a Meta-style
   `LLAMA_DIR`; CI's is `/mnt/MLPerf/…`, which does not exist here. One was assembled from symlinks
   at `/localdev/ctr-apbernal/tttv1_ckpt/Llama-3.3-70B-Instruct`. The Qwen demo has no such
   constraint and uses `HF_MODEL`. The two selectors are mutually exclusive and must be set
   per-process.
6. **`pytest.ini` caps every test at 300 s**; every demo command needs an explicit `--timeout`.
7. **`/proj_sw` filled mid-run and killed a device run** at 10:25:04Z with `ENOSPC` while `df`
   reported 950 GB free. Converted-weight caches now go on `/localdev` via `TT_CACHE_PATH`.
8. **The tt-metal JIT cache decides run length**: first run 2519 s / 1546 s, steady state 1104 s /
   531 s. This is exactly what the methodology's unmeasured warmup exists for.

Schedule handed to `c-perf-paired`: the TTTv1 half of the paired night is **~2.5 h** (2 models ×
1 cold warmup + 3 measured). Whether the full 16-run night fits depends on the TTTv2 arm, which this
job did not measure.

## 2026-08-29 — `c-defects` attempt 1

Four defects at one call site, not two. Milestone B measured D-C5 (the column user selector's
matmul rejects the LM head's WIDTH_SHARDED decode output) and D-C8 (with that satisfied, the
matmul's auto-selected grid leaves the loaded decode sub-device). Fixing them exposed **D-C10**
— three more full-grid programs inside `Sampling2D.decode_forward`: `ttnn.topk`'s
implicit-tile-padding fill, `ttnn.manual_seed` and `ttnn.sampling`. All four are one defect
class, the one `recipes.rope_core_grids` already names: a grid resolved independently of the
partition that has to contain it. **D-C9**, the sampled-token readback composing the wrong mesh
axis, is fixed with it, so all 32 users are read back for the first time on this mesh.

`test_column_user_selector_wh_galaxy.py` now stages the LM head's real placement under a
loaded decode sub-device manager at both models' widths: 6 passed, three fresh processes.

**D-C7** was measured before it was changed. `MeshDevice` exposes no allocator statistics, so
the probe used the symptom — two production-size global CBs cannot coexist in one L1 bank — and
found that the L1 *is* returned when the last Python reference goes. The surviving reference was
in `Prefetcher2D.cleanup()`, which cleared its own map but not the `global_cb` field of the
contexts it had handed to every module.

**D-C6** is a program-config sizing defect, not a capacity wall: `ttnn` defaults
`out_block_h`/`out_block_w` to `per_core_M`/`per_core_N`, so the concat-32 prefill's circular
buffers grow with the whole per-core tile count. `dense_matmul_output_blocks` takes the largest
divisor that fits a budget sitting between the largest currently-qualified config (1 343 488 B)
and the measured ceiling (1 499 136 B), so no qualified program config moves. The
byte-identical overflow on "two different geometries" was never a coincidence: both models have
64 heads, 8 KV heads and head_dim 128, so both resolve `local_qkv_size = 1280`.

**The Llama L1 address clash is not the global circular buffer**, and two comments in this
repository say it is. A probe with a production-size global CB resident placed `ttnn.embedding`
cleanly at both models' prefill row widths, every call a cold compile. That is also why
`release_global_cb_on_prefill` runs and does not help — the clashing address does not move when
the buffer is released. The workstream is OPEN and the next attempt should find out what
actually owns 544832 rather than trying to free the buffer again.

Two facts that cost time and should not be rediscovered: `pytest.ini` caps every test at 300 s,
and the area-4 cases only fitted inside it at Milestone B because they aborted early; and a
program-cache hit skips `validate_circular_buffer_region`, so a placement probe has to force a
cold compile or it measures nothing.

**D-C11 is the largest finding of the attempt, and it was invisible until the four placement
defects above were fixed.** `GalaxyColumnUserSelector` gathers each column's 32 user rows with a
one-hot matmul and the surrounding code calls that an exact row copy. It is not one at ttnn's
default math fidelity: driving a known tensor through the selector changed **4 300 324 of
4 915 200** values, maximum absolute error 0.875 — seven ulp at a decode-logit magnitude of 15,
where bfloat16's ulp is 0.125. A one-hot matmul multiplies by 1.0 and sums zeros, so the only
thing that can move a value is the mantissa truncation in the default fidelity. At `HiFi4` the
same comparison changes **0** values. The fix is `exact_gather_compute_kernel_config()`,
deliberately **without** `fp32_dest_acc_en` — that halves the destination register file and caps
the subblock product at 4, which the qualified `out_subblock_w` of 5 and 6 exceed
(`TT_FATAL @ matmul_device_operation.cpp:567`). HiFi4 alone is what the probe qualified.

It was found by bisection and the three negative results are kept, because each one is a thing
nobody has to re-test: it is not tie-breaking (the disagreeing slots' top-2 gaps were 0.125 to
0.5 and `disagreed-but-tied = 0`), not the `_place_for_topk` reshard this job added, and not the
mesh-row order of the candidate list.

**D-C12 is open and unexplained.** The second `GalaxyDirectRunner` in one process samples
float32 bit patterns as token ids — `1098241487` is `0x41748F4F`, about 15.3 as a float, which
is a decode-logit magnitude — so the readback is landing on memory that holds logits. Note the
shape: "the second runner in one process" is also how `release_global_cb_on_prefill`'s comment
describes the Llama address clash. Two symptoms at one seam.

**Llama area 4 has now been measured for the first time, and its greedy claim passes 32/32.**
Every Milestone B attempt died at D-C5 before reaching the sampler. Since the selector, the
sampler, the program configs and the partition are shared code and identical between the models,
Llama's 32/32 against Qwen's 31/32 means the Qwen residual is not in the shared path — it is
something about Qwen's logits or its vocabulary padding, which is a far smaller search space
than the sampling stack.

**Late findings, and they reframe two workstreams.**

`D-C12` is not "the second `GalaxyDirectRunner` in a process". It is **the second sampling call**,
and the mechanism is the **ttnn program cache**: with the cache cleared, four consecutive sampling
calls on four different inputs are all correct; warm, only the first is (four fresh processes,
16 s each, no checkpoint). Nothing in the tree caught it because every area-4 gate samples once per
model load, and the one case that samples six times feeds identical logits each time, so a result
that lags by a call is bit-identical to a correct one. It is a correctness hazard for anything
that decodes more than one token.

**The Llama L1 clash is not the global circular buffer, and this is now a measurement rather than
an inference.** A one-layer subset reproduces the clash in 141 seconds with allocator dumps at ten
boundaries. With `release_global_cb_on_prefill` on, the 792 064 B buffer is present at the second
runner's open and **absent** by the failing prefill - the release genuinely frees it - and the
abort still names 544 832. Eliminated: the global CB, first-runner residue, the embedding weight
table, the staged token ids (both explicitly DRAM). What remains is a sharded buffer the bank-table
dump does not enumerate, or a runtime allocation.

**Two residuals are numerics, not defects.** Qwen's greedy slot 4 is an exact bfloat16 tie - ids 16
and 17 both at 15.375, so the row has no unique argmax. And concat-32 differs from sequential
prefill by up to **1.06** in logit value, eight ulp at magnitude 15, so the argmax flips only on the
7 of 32 rows whose top-two gap is smaller than that. Neither assertion was moved.

**D-C7 is qualified on silicon**: 792 256 B per L1 bank returned against a `GALAXY_GLOBAL_CB_SIZE`
of 792 064, byte-identical in three fresh processes. Behind it, a new defect `D-C13`: the second
model's L1 is **fragmented, not full** - 1 261 952 B free, largest block 759 488, 32 576 B short,
the same numbers every run.

---

## 2026-08-30 — `c-exec-llama` attempt 1: the Galaxy Llama has an executor, and it runs

`models/common/models/llama33_70b_galaxy/executor.py` composes the common runtime for the 2D
tensor model — one `PagedKVCacheManager` over the model's own KV contract, one `OutputReader`,
resolved prefill/decode configs, one `ProgramCompiler`, one `EagerExecutor`, one
`WarmupCoordinator`, and the trace collaborators built exactly as the 1D executor builds them so
`c-trace` can layer over this same eager instance. **Zero lines of `llm_runtime` changed, zero 1D
module files changed**, and `pytest models/common/tests/llm_runtime` is still 1032 passed / 1
skipped.

**It worked on the first silicon run.** Eager prefill at 128 through the executor agrees with the
qualified `GalaxyDirectRunner` at **PCC 0.99941**, the first decode step at **0.99359**, and
batch-32 decode has all 32 slots taking the reference argmax.

**Two placements the runtime cannot be handed as they are, both model-owned in the executor.**
`DecodeRuntime` maps decode positions and the decode page table *replicated*; the Galaxy decode
graph needs them column-sharded, and a replicated device tensor cannot be resharded on device
because per-device-different slicing is not one SPMD op. And the runtime's logits readers
concatenate mesh *columns* along the vocabulary axis while on Galaxy the vocabulary is sharded over
mesh *rows* — that is D-B23, and `compose_galaxy_logits` already carries the measurement. Both are
adaptation in the model package, which is where the plan's extension discipline puts them.

**The Llama L1 address clash is narrower than "a second allocation cycle".** It is **a prefill
after a decode in one process**, and it now reproduces in **110 seconds** with one executor, one KV
allocation, one compiled decode program and no request at all: `warmup_decode`, then
`warmup_prefill`, 232 ms apart. Three addresses have now been seen — 544 832 (c-defects), 543 488,
542 944 — all on core range `[0-0 - 0-3]` against the same CB region end of 630 080. So the
*region* is fixed and the address is not, and a few-kilobyte buffer there is consistent with
c-defects' dumps finding no live block over 100 kB at the failing prefill. It blocks the
repeated-cycle gate and the decode-first warmup order; both are reported rather than worked around.

**Four things the bring-up found that are not defects and that `c-exec-qwen` will meet.** A cached
request is a chunked request to the planner, so the `PREFIX_CHUNKED` attention recipe must be
registered at construction. The planner's own padding table turns a 512-token prompt into a
1024-token device request, so the registered recipe set is `(128, 1024, 2048)`. A physical KV pool
smaller than the construction ceiling has to reach the *model*, because `Attention2D.bind_kv_cache`
validates against the block count its own metadata declares. And the prefill rotary must use the
model's qualified slice, not a device gather: the gather left the first layer's K at PCC 0.907 while
the same request's logits agreed at 0.9994.

## 2026-08-31 — `c-defects` attempt 3: the global circular buffer has an address

**The Llama L1 address clash is fixed**, and it was never one defect. Three earlier accounts read
the message wrong in the same two ways: `validate_circular_buffer_region` computes one device-wide
lowest-occupied L1 address *before* it loops over circular-buffer core ranges, so the core range in
the message names the **circular buffers** and not the clashing buffer; and the allocator's block
table prints addresses without `offset_bytes_` (105 664 B here) while every allocator message adds
it. Reconciled, the clashing buffer is **32 bytes** — which is why a filter for live blocks over
100 kB never found it.

Then the arithmetic removes an option. 869 056 B of L1 lie above the 630080 that the prefill
embedding's static circular buffers reach; the resident model after a decode is 127 776 B and the
buffer with its config page is 792 256 B. **920 032 does not fit in 869 056**, so the buffer and a
prefill program of this shape cannot coexist on any allocation order. `defer_global_cb` covers only
the first prefill, so `release_global_cb_on_prefill` is *required*, not optional, and is now the
default for both Galaxy models. A serving system interleaves prefill and decode by construction.

**Everything else follows from one allocator rule: `FreeListOpt::allocate` takes the smallest free
block that fits.** L1 is top-down and an address never moves, so the first decode strands 32 bytes
*under* the resident buffer and releasing does not move them — hence `global_cb_headroom`. A
recreated buffer that lands elsewhere is a wrong-address read, not an error, because
`CircularBufferImpl::set_global_circular_buffer` captures `buffer_address()` and `config_address()`
once and `dispatch.cpp` re-sends the captured pair on every launch — on silicon it **hung**. A fixed
headroom cannot put it back; reproducing the free-region top can, and reserving the missing bytes
*once* cannot, because the leftover of the previous headroom is a free gap of exactly that size and
the allocator prefers it: the buffer came back **32 736 B high on both models, to the byte**. And a
global circular buffer is **two** allocations — the data buffer came back at 510816 in two fresh
processes while the 192-byte config page moved to **1367872**, 850 kB away, and chunked prefill
failed on it. So every free gap above the low region is held before the creation, and the check
compares the blocks the creation *adds* rather than a lowest-address proxy that cannot tell that
case from a real move.

Measured, full shape, three fresh processes each: Llama repeat-and-cleanup **3/3 pass** where it
failed 3/3 at 544832; Qwen **3/3 pass**, so the shared change does not move it; block-level
cross-slot isolation **3/3**; chunked prefill **3/3**. The last two had never been evaluated on
Llama in either direction.

**A new defect, D-C14, and it is bigger than the coverage item it blocks.** With the clash gone, two
models in one process get further than ever and then hang at the second model's first decode, in
`FDMeshCommandQueue::wait_for_outstanding_reads`. The ttnn program cache belongs to the *mesh
device*, outlives a model, and keys on op-and-config hashes — so two identical Galaxy models share
compiled decode programs and therefore the first model's buffer addresses. The invariant is per
**process and mesh**, not per owner. `GlobalCBPlacement` is implemented and host-qualified and is
**not** on silicon: the mesh lost board 23 to POST_RESET before it could be, and ten reset attempts
across four mechanisms did not recover it. D-C13 is superseded rather than closed — the allocation
now succeeds and D-C14 is what stands behind it.

Host gates after the change, three fresh processes each and identical to Milestone B:
step-7 34/32/37/18/12/29/8, `test_prefetcher_2d.py` **33 passed** (22 before),
`llm_runtime` **1032 passed / 1 skipped**. Nothing relaxed, no expectation edited, zero changes
under `models/common/llm_runtime/**` or to any `*_1d.py`.

## 2026-08-31 — `c-defects` attempt 4

- Mesh arrived HEALTHY (32/32 ticking, board 23 back). Attempt 3's closing state — 25 boards out,
  "do not reset again" — no longer holds; the host was repaired between 12:26Z and 17:02Z. No reset
  was run by this attempt. Attempt 3's heartbeat recipe reads `<node>/device/tt_heartbeat`, which at
  driver 2.9.0 is the PCI directory and returns `ERR` for a live board; the attribute is
  `<node>/tt_heartbeat`.
- **Corrected the gate ledger against the tree.** Attempt 3 recorded D-C5/D-C8 as MET with "thirty
  runs, every one reaching its assertion". Llama's `a_seeded_slot_repeats_across_runs` aborted on the
  L1 clash 3/3 and never reached the sampler, so the gate is one claim short. Queued as `t7`–`t9`.
- **D-C7 closed to its mechanism.** Attempt 3's placement record reached silicon and turned a
  mesh-wedging 21-minute hang into a 106 s refusal. Behind it: the second model finds 77 extra
  32-byte L1 blocks — 2 464 B — while every larger block matches the first model's in size and count.
  Not capacity, not a bigger pool, not a leaked headroom: fragmentation. `FreeListOpt` takes the
  smallest free block that fits, so 77 holes displaced a 64 kB allocation 109 376 B downward, below
  the recorded free top. D-C7's original 923 776 B/bank leak is 99.7 % gone.
- **Fix `faec6e59938`:** `Prefetcher2DConfig.on_global_cb_released` (default `None`, called once from
  `cleanup`) plus `release_galaxy_global_cb_placement` at the model layer, which clears the mesh
  program cache and forgets the placement record. The module announces, the model layer decides.
  Host 35 passed (33 before); removing the two-line announcement gives 1 failed / 34 passed. On
  silicon the one-layer two-pools case went 1 failed → **1 passed**, with both models' creations in
  the log.
- 19-run full-shape gate queue (`q11`) launched 17:31Z; 25-run host regression (`q12`) prepared to
  follow it.

## 2026-08-31 19:51Z — `c-defects` attempt 5 opens

Arrived to find attempt 4's gate queue **still running** (`queue.sh` pid 19707, 7 of 19 runs
dequeued). Adopted it rather than killing it. Read off `RESULTS.md`: D-C7's gate is **met for
Qwen** at full shape 3/3 (`t1`–`t3`), and **Llama fails 2/2 at the 3600 s bound** (`t4`, `t5`).

The Llama failure is a **DRAM** Out of Memory at `layer53_wqkv` of the *second* model, with DRAM
99.95 % full — so the first model's DRAM was not returned by `close()`. That is a different
resource from D-C7 (L1) and from attempt 4's fragmentation finding, and it is the one thing
standing between this job and the finish condition.

Reordered the running queue's unread tail in place (same inode, reader offset verified at a line
boundary) to put a one-layer DRAM residue probe first, then D-C5/D-C8's last unevaluated claim,
then the step-7 page-table placement file. Deferred the 6-run clash regression until the module is
in its final state.

## 2026-08-31 20:52Z — `c-defects` attempt 6 opens

Attempt 4's queue was **still running** for the third time; adopted again rather than killed.
Reconciled attempt 5's 20:10Z handoff against the tree: `t6` had landed (rc=124, the same DRAM OOM
as `t4`/`t5`, so **3/3 byte-identical**) and attempt 5's probe had already run and **failed in 46 s**
on an API slip — `ttnn.get_memory_view` returns a `MemoryView`, and the blocks are
`view.block_table`. No measurement had been taken. Rewrote the probe and re-queued it.

- **The Llama half of D-C7 is a DRAM retention defect, and capacity is refuted.** Two probes at
  commit `faec6e59938`:
  - 4 layers: **19 931 264 B per DRAM bank still allocated** after `close()`, `del` and
    `gc.collect()`; **12 of 12** prefetcher-registered weights still live;
  - full 80 layers, second arm only: **1 passed in 239 s** — the `max_seq_len=4096`,
    4096-block-pool arm that the two-pools case dies on builds, prefills and decodes **on its own**
    — and **398 617 984 B per bank**, 961 blocks, **240 of 240** weights still live after close.
    That is **37 %** of the 1 070 773 184 B bank, against an OOM reading
    `allocated: 1070239264, free: 533920`.
  The residual histogram names the owner: 240 = 80 × 3 (`w1`/`w2`/`w3`), 80 each of the attention
  shapes including the 208 896 B that is exactly the failing `wqkv_ring`, and 384 × 161 = two RMS
  norms per layer plus the final norm.
- **Fix `299440bb276`.** `MLP2D` had no `release` and no `close` at all; the block's `close()`
  called only `self.attention.close()`, which never touches weights. Added
  `lazy_weight.release_device_weights` (dedupes on the `LazyWeight` *and* on its `_value`, because
  `prefill_wqkv = resolved.prefill_wqkv or wqkv`), `RMSNorm2D.release`, `MLP2D.release`,
  `Attention2D.release`, and `<Block>.release_weights` called per layer from each model's `close()`
  — **after** `Prefetcher2D.cleanup()`, because the attention decode weights are registered with the
  prefetcher and freeing one under a live prefetch is a use-after-free. Zero `llm_runtime` lines,
  zero `*_1d.py` lines.
- **Measured after the fix, same probe, same shape:** residue **19 931 264 B → 0**, blocks 49 → 0,
  registered weights **12 of 12 → 0 of 12**. Host regression pass 1 unchanged on all ten suites.
- **Committed test `d2d6c424030`:** `test_<model>_a_closed_model_returns_its_device_weights`, on
  both models, asserting the allocator returns to baseline with `handle` and `runner` still bound —
  `close()` alone must be sufficient. Threshold zero on principle and zero in measurement.
- Area 4's last unevaluated claim-verdict — Llama's seeded slot — **reached its assertion for the
  first time** (610 s, zero clash lines) and **fails as D-C12**, matching Qwen claim for claim.
- Two of `c-exec-llama`'s three handed-over defects reduce to **not shared-Galaxy-code**:
  `chunk_start` alignment is `llm_runtime/warmup.py:700` assuming a block-aligned prefix is a valid
  chunk start (32 vs Galaxy's 128), and the `page_table width` failure is a stale table after a pool
  shrink. Reductions written; `llm_runtime` not touched, per this brief.

## 2026-09-01 07:07Z — `c-defects` attempt 7 opens

Arrived on an **idle** mesh, 32/32 boards on the bus, with attempt 6's 53-run chain fully drained
at 01:36Z. Zero `tt-smi` resets. First job of this series to start with no inherited queue running.

- **Reconciled attempt 6's 01:14Z handoff against the tree.** Five runs landed after it wrote, and
  one answers its own open question: `test_executor_repeated_startup_and_cleanup` — the direct test
  of whether serving is unblocked, which failed **3/3** on the L1 address clash at `2b463f17fcd` —
  **passes 3/3 in fresh processes at HEAD** (`y4`/`y5`/`y6`, 199.68 / 199.28 / 199.83 s, **0**
  `clash with L1 buffers` lines each, all three logs stamped `# commit=671802f9464`). The received
  claim that the clash blocks serving is a **pre-fix** claim and is now refuted on silicon.
- **Verified the two no-silicon gates by inspection:** zero changes to any `*_1d.py`, zero changes
  under `models/common/llm_runtime/`, and no uncommitted change to `models/` at all.
- **Established that attempt 6's device results are results about HEAD.** Every commit after
  `299440bb276` touches only evidence, one README, and the two step-7 coverage test files — no
  production source. The logs' own `# commit=` headers read `d2d6c424030` / `874a0e9da75` /
  `f61978825cd`, not `299440bb276` as attempt 6's prose says; the prose is loose, the conclusion is
  unaffected, and the runs are *closer* to HEAD than claimed.
- **D-C16 opened and reduced, not fixed.** Attempt 6's §28 called the `chunk_start` alignment defect
  "D-C13", which collides with the superseded global-CB fragmentation OOM in `D-C7.status`. Renamed.
  Qualified 3/3 at HEAD (`y1`/`y2`/`y3`, byte-identical at `attention_2d.py:908`), with every link
  in the chain verified in the source: `warmup.py:700` hard-codes `cached_tokens=layout.block_size`
  (32), `plan.py:163` validates only *block* alignment, `plan.py:381/403` carries it through as
  `chunk_start_idx`, and Galaxy's `chunk_alignment` is 128 — and *is* the flash-SDPA chunk size, so
  lowering it would be relaxing a real constraint. The fix belongs in `llm_runtime`, which this
  brief forbids; the reduction is the deliverable and the job stopped at it.
- **Queued `q16`, 33 runs**, whose only purpose is to bring **the whole of area 4 to one commit on
  both models**. Seven of area 4's ten claim-verdicts were last measured at attempt 3's commits,
  before both teardown fixes; the three already at the fixed commit are not repeated. Plus the two
  clash-blocked claims that are not the D-C7 gate, `test_reference_prefill_and_decode[2048]`, and
  three fresh passes of the one step-7 file that needs a mesh (`u1`/`u2` predate the weight-release
  fix; only `u3` was post-fix).

## c-defects attempt 8 — 2026-09-01T08:25Z checkpoint

Arrived 08:11Z to a **busy** mesh: attempt 7 exited 07:47Z with its 33-run queue `q16` still
running and the driver adopted it. Killed nothing, queued nothing on top of the running test,
and will not exit before it drains and is read.

* Verified at HEAD, not inherited: zero `*_1d.py` diff, zero `llm_runtime/` diff, clean
  `models/`, and no production-code commit after `299440bb276` (one docs file, two test files).
* Inherited as measured, checked against `RESULTS.md` rows rather than prose: D-C7 (`z6`–`z11`,
  6/6), the clash gate (`zc1`–`zc6`), the three claims the clash blocked, step-7 host 170×3,
  `llm_runtime` 1032 passed / 1 skipped.
* **Correction to the inherited account.** `D-C6.status` and attempt 7's handoff say area 2's
  question — do padded rows change an active row's logits — is "answered for Llama". It is
  answered on **neither** model: `test_{qwen,llama}_concat32_padded_rows_change_no_active_rows_
  logits` has zero rows in `RESULTS.md` and zero logs on disk. Six runs (`zp1`–`zp6`) appended
  to the live `q16` to measure it for the first time.
* D-C12 reviewed on the host and deliberately not opened: the two cheap Python hypotheses are
  refuted by reading (`Sampling2D.release` resets every cached handle; `LazyBuffer.update` really
  copies host→device), and attempt 1's cache-vs-nocache bisect points at a ttnn op's program-cache
  handling, whose root fix is C++ — and this job may not rebuild tt-metal.

## c-defects attempt 9 — 2026-09-01T09:57Z checkpoint

Arrived 09:34Z to a **busy** mesh again: attempt 8 exited 09:21:56Z with `q16` still running and
the driver adopted it, inside `zl7_llama_per_slot_controls_r1`. Killed nothing, queued nothing on
top of the running test, and will not exit before it drains and is read.

* **The D-C5/D-C8 gate is MET.** `zl9` landed 09:50:51Z (`1 passed`, 311.74 s) and completes area
  4: ten claim-verdicts, five claims on two models, three fresh processes each, one production
  tree. All thirty logs re-checked directly — `TT_FATAL`, `TT_THROW`,
  `num_intersections == num_cores`, `must be interleaved`, `clash with L1 buffers`, `SKIPPED` —
  **30 logs, 0 with any hit.** Six verdicts pass, four fail, and the four failures are two defects
  that are not D-C5 or D-C8.
* **The driver's standing clash warning is a pre-fix account, and no silicon was spent on it.**
  `c-exec-llama` measured "a prefill after a decode clashes, so the clash blocks serving" at
  `2b463f17fcd`; the two fixes are `32e552bb0b2` and `faec6e59938`, both later. At HEAD,
  `y4`–`y6` (`test_executor_repeated_startup_and_cleanup`, three prefill-then-decode cycles per
  run) pass 3/3 with **0** clash lines, and `y1`–`y3` (the 110-second decode-then-prefill
  reproduction) fail 3/3 on D-C16 with **0** clash lines.
* **Qwen's near-zero-temperature residual is now MEASURED, not hypothesised, and it cost no
  silicon.** `d11_greedy_tie_probe.log` — same `_load`, same rows, same `tokens=[1]*32`,
  `positions=[128]*32`, the gate's exact `T=0.02` policy — measures a top-two bfloat16 gap of
  **zero at exactly three slots: 4, 12 and 21**. The gate misses `[4, 21]`. `torch.argmax` breaks a
  zero gap by lowest index and a sampler does not. The report had already named this as the
  measurement to take; it was on disk and nobody joined it to the gate result.
* **A correction: the `T = 2.0` half of that test proves nothing.** Its call order is
  `decode_logits`, `cold = decode_sampled(T=0.02)`, `hot = decode_sampled(T=2.0)` — so `hot` is the
  **second** device sampling call in the process, which is what D-C12 corrupts. The same structure
  in the tie probe reports `missed=True` in all 32 slots with float32 bit patterns for ids. "D4 is
  confirmed twice" is withdrawn; D4 stands on the `T=0.02` direction alone, which suffices.
* **D-C12's received explanation has a hole, and this repo's own linters widen it.** A stale
  *address* cannot be the mechanism in a probe where every allocation is made and freed in the same
  order each call; and `scripts/detect_smuggled_rta.py` / `scripts/detect_override_rebuild.py`,
  run over **all 2 993** `ttnn/**/device/*.{cpp,hpp}` files, flag eight sites and **none in the
  sampling chain**. The mechanism that fits is a **premature readback** masked on call 0 by the
  compile stall. `tttv2_dc12_scratch/test_dc12_op_bisect.py` asks it directly with three reads of
  the same output per call, and is queued as `q17` behind `q16`.

## `c-defects` attempt 10 — 2026-09-01

**Arrival 10:32Z.** 32/32 boards; the only thing on the mesh was this job's own `q16` (PID 228702,
started by attempt 7, adopted across attempts 8 and 9), on `zr2`. Zero resets this attempt. Nothing
discarded on dead-mesh grounds.

**Attempt 9's handoff reconciled against the tree.** Its 10:02Z status table called four nodes IN
FLIGHT; three had landed. `zm1`-`zm6` all `1 passed` (Llama cross-slot 309.40/333.24/439.24 s,
Llama chunked prefill 303.64/228.93/249.68 s) — so the gate line "the three claims the Llama clash
blocked are measured" is now met with **all six runs at HEAD**, not with older-commit runs plus one.
`u4`/`u5`/`u6` landed `3 passed` each (12.24/12.00/12.07 s), completing the device half of the
step-7 gate at HEAD. Nothing attempt 9 claimed was contradicted.

**All eight of the brief's Finish-condition gates re-derived from the logs, not from status pages** —
each log's own `# commit=`/`# node=` header, its pytest summary, and counts of `clash with L1
buffers`, `SKIPPED`, `TT_FATAL|TT_THROW`. Two things this established that were not on record:

* production code is **byte-identical from `299440bb276` to HEAD** (`git diff --name-only` over that
  range touches, under `models/`, only `modules/README.md` and two device test files that each gain
  one function and modify none) — which is what makes the step-7 host set, taken at two different
  commits, **one** qualification rather than three fragments;
* the brief's "162 tests at Milestone B" is the figure that is off, not an expectation: the seven
  `test_step7_*.py` files are byte-identical to Milestone B and collect **170**.

**D-C17 raised, reduced, and not fixed** — `c-exec-llama`'s third handed-over defect.
`zr1`/`zr2`/`zr3` are `1 failed` in three fresh processes and **do not touch the device**:
`_reference_prefill` caches to disk and all three print `[reference] loading …
llama_prefill2048_layers0.pt`, a file written 2026-08-30 by `c-exec-llama` four fixes back. Read on
the host, that artifact has sane prefill logits and sane KV at 2048 and garbage decode logits in all
32 rows. The cause is visible in source: the test decodes at `positions[0] = length`, which at
`length == _MAX_SEQ_LEN == 2048` is one past the last addressable position, and
`GalaxyDirectRunner.generate` guards that condition (`direct_runner.py:645`) while `decode_logits`,
`decode_sampled` and `_stage_positions` do not. Recorded `OPEN — REDUCED, NOT FIXED` with the
one-check fix and two named owners; not committed, because committing it moves HEAD off the tree all
eight gates are qualified at, and because the fix leaves the test red (it asks for a position that
does not exist) and turning it green means editing another job's test.

**Stale status headers corrected:** `D-C14.status` and `D-C15.status` both still said IN FLIGHT; the
runs behind them have landed and are read (`t1`-`t3`, `z6`-`z11`, `zg1`-`zg6`, `zc1`-`zc6`; 0 `Out
of Memory`, 0 `TT_FATAL`, 0 `TT_THROW` across twelve logs).

**Queued this attempt** (appended to the live `q16`, inode preserved): `zs1`-`zs3` and `zs4`-`zs6`,
`test_reference_prefill_and_decode` at 2048 and 512 with
`LLAMA33_70B_GALAXY_EXECUTOR_REFERENCE=recompute`, to take D-C17's measurement on silicon instead of
off a stale file. The three inherited artifacts are preserved as `*.as-inherited-20260830.pt`.
