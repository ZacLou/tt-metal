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
