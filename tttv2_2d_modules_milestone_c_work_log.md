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
