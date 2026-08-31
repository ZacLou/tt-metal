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
