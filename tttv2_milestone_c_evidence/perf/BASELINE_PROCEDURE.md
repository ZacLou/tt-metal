# `perf/BASELINE_PROCEDURE.md` — how to measure the TTTv1 Galaxy baseline

Written by job `c-perf-recon`, 2026-08-29, on
`wh-glx6u-05-special-ctr-apbernal-for-reservation-119144`, WH Galaxy 6U, 32 boards,
commit `6af44349413ca6ce2c0d98f5b26dd2898dc1f067`.

**Every number in this document is UNPAIRED RECONNAISSANCE at commit
`6af44349413ca6ce2c0d98f5b26dd2898dc1f067`.** It is not a gate result, it is not compared against
any target, and it cannot be paired with a TTTv2 arm measured on another night.
`c-perf-paired` re-runs both arms together. What this document is *for* is the **procedure** and the
**schedule**: the exact commands, the environment they need, where each metric appears, what TTFT
means, and how long one run takes.

Read `ENVIRONMENT.md` alongside this. Every log referenced here is under `perf/recon/`, verbatim,
never overwritten.

---

## 0. The configuration the milestone is measured at

"Batch 32 / sequence length 507" is **not** a token count. `507` is `len(input_prompts[0])` in
**characters** — the key the demos use to look up the centralised targets:

```python
# models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py:1271
if batch_size == 32 and len(input_prompts[0]) == 507:
    resolve_perf_targets(model_name=…, sku=sku, batch_size=batch_size, seq_len=len(input_prompts[0]))
```

The 507-character corpus is
`models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json`
— 32 prompts, every one 507 chars, tokenising to **118 tokens**, padded to a **128**-token prefill.
Output length is 128 decode tokens. It is the corpus the stock `batch-32` parametrisation of both
demos already uses, so **no custom command has to be constructed**.

The absolute targets in the plan are exactly the entries in `models/model_targets.yaml`:

```yaml
llama3.3-70b-galaxy / wh_galaxy_perf / batch_size: 32, seq_len: 507
  prefill_time_to_first_token: 99.0     decode_t/s/u: 71.5   decode_t/s: 2288.0
  prefill_time_to_first_token_tolerance: 0.3   decode_t_s_tolerance: 0.5   decode_t_s_u_tolerance: 0.5
qwen3-32b-galaxy   / wh_galaxy_perf / batch_size: 32, seq_len: 507
  prefill_time_to_first_token: 700.0    decode_t/s/u: 60.0   decode_t/s: 1920.0
  (no per-metric tolerance keys -> DEFAULT_PERF_TOLERANCE = 0.15, models/demos/utils/model_targets.py:36)
```

`wh_galaxy_perf` is what `get_current_device_sku_name()` returns for this cluster
(`models/demos/utils/device_sku.py:20-22`, `ClusterType.GALAXY`/`TG` → `wh_galaxy_perf`).

---

## 1. Llama-3.3-70B — the runnable procedure

### 1.1 Provisioning (do this once; it is already done on this host)

`text_demo.py` **cannot run against a HuggingFace checkpoint.** `text_demo.py:1060` builds the Meta
tiktoken tokenizer unconditionally, while `model_config.py:2737 encode_prompt()` calls
`tokenizer.apply_chat_template` when the checkpoint detects as HuggingFace. The two are only
consistent when `checkpoint_type == CheckpointType.Meta`. Upstream CI uses a Meta-style directory
(`tests/scripts/tg/run_tg_model_perf_tests.sh:20`,
`LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/`) — **`/mnt/MLPerf` does not exist
on this host.** Evidence: `perf/recon/llama_b32_run1_cold/run.log:2565` and
`perf/recon/llama_b32_run2_llamadir/run.log:1737`.

The equivalent directory was assembled from pieces that do exist, **without modifying any package**:

```sh
S=/localdev/ctr-apbernal/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b
D=/localdev/ctr-apbernal/tttv1_ckpt/Llama-3.3-70B-Instruct
mkdir -p "$D"
for f in $S/original/consolidated.0*.pth; do ln -sfn "$f" "$D/$(basename $f)"; done
ln -sfn "$S/original/params.json" "$D/params.json"
ln -sfn "$S/tokenizer.model"      "$D/tokenizer.model"
```

The directory **name matters**: `model_config.py:2194-2202` keys `rope_scaling_factor = 8`,
`is_70b = True` and `max_prefill_chunk_size = 128k` off the substring `3.3-70B` in the path, and
`model_config.py:556` sets `instruct = True` off the substring `instruct`. Keep the leaf name
`Llama-3.3-70B-Instruct`.

Verify before running:

```sh
ls -L /localdev/ctr-apbernal/tttv1_ckpt/Llama-3.3-70B-Instruct
# consolidated.00..07.pth (8 × 17,640,971,024 B), params.json (221 B), tokenizer.model (2,183,982 B)
```

### 1.2 The command

```sh
cd /proj_sw/user_dev/ctr-apbernal/tt-metal
export TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
export PYTHONPATH=/proj_sw/user_dev/ctr-apbernal/tt-metal
export HF_HOME=/localdev/ctr-apbernal/hf_data
export LLAMA_DIR=/localdev/ctr-apbernal/tttv1_ckpt/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/proj_sw/user_dev/ctr-apbernal/tt-metal/model_cache/meta-llama/Llama-3.3-70B-Instruct/TG
unset HF_MODEL MESH_DEVICE TTTV2_GALAXY_CCL_TRACE LINE_RS LINE_AG

timeout --signal=TERM --kill-after=180 5400 \
  python -m pytest -v -rA --color=no -p no:cacheprovider --timeout=5200 \
  'models/demos/llama3_70b_galaxy/demo/text_demo.py::test_demo_text[wormhole_b0-mesh_device0-device_params0-10-performance-batch-32]' \
  > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

Notes that are not optional:

- **`--timeout=5200` is required.** `pytest.ini:2` sets `timeout = 300` repo-wide (upstream commit
  `8cef2551edd`, not a local edit). Without the override the run dies at 300 s with
  `Failed: Timeout >300.0s`, which looks like a model failure and is not one.
- **Use the full node id, not `-k "performance-batch-32"`.** That `-k` expression also selects
  `performance-batch-32-non-uniform-sampling` and `performance-batch-32-log-probs` — three tests,
  three device sessions.
- `LLAMA_DIR` and `HF_MODEL` are mutually exclusive; `model_config.py:511` asserts if both are set.
- The demo creates `models/demos/llama3_70b_galaxy/demo/output/` at runtime (`text_demo.py:939`).
  That is the demo's own behaviour, not a modification of the package.

Driver used here: `tttv2_milestone_c_runs/run_recon.sh` with
`tttv2_milestone_c_runs/envs/llama.sh`. It refuses to overwrite an existing evidence directory,
captures `tt-smi -ls` before and after, and appends one line per run to
`perf/recon/QUEUE_RESULTS.txt`.

### 1.3 Where each metric appears — file, line, units

All three gated metrics are printed by the demo to **stdout/stderr only**. There is no profiler
artifact for a non-CI run: `benchmark_data.save_partial_run_json` is guarded by
`if is_ci_env and repeat_batches > 1` (`text_demo.py:1742`), and neither holds here. The only file
pytest writes is the JUnit XML at `generated/test_reports/most_recent_tests.xml`, which carries the
duration, not the metrics. **Parse the log.**

| Metric | Log line (regex) | Emitting source line | Units | Value expression |
| --- | --- | --- | --- | --- |
| TTFT | `Average Time to First Token \(TTFT\): ([0-9.]+)ms` | `text_demo.py:1685` | **ms** | `avg_time_to_first_token*1000`, `text_demo.py:1608` |
| decode tok/s/user | `Average speed: [0-9.]+ms @ ([0-9.]+) tok/s/user` | `text_demo.py:1686` | **tokens/s/user** | `decode_tok_s_user`, `text_demo.py:1613` |
| aggregate decode tok/s | `Average speed: .* \(([0-9.]+) tok/s throughput\)` | `text_demo.py:1686` | **tokens/s** | `decode_tok_s`, `text_demo.py:1614-1616` |
| decode ms/iteration | `Average speed: ([0-9.]+)ms @` | `text_demo.py:1686` | **ms** | `avg_decode_iteration_time*1000` |
| 1st / 128th token decode | `1st token decode time: ([0-9.]+)ms` / `128th token decode time:` | `text_demo.py:1664,1668` | **ms** | single-iteration durations |
| prefill compile | `Prefill compile time: ([0-9.]+)s` | `text_demo.py:1682` | **s** | warmup, excluded from TTFT |
| decode compile | `Decode compile time: ([0-9.]+)s` | `text_demo.py:1683` | **s** | decode iteration 0, excluded |

A ready-made extractor is `tttv2_milestone_c_runs/extract_metrics.py` (see §4).

### 1.4 TTFT semantics — **LAST USER READY**

This is the question `c4_perf_paired.md` asked to have settled, and the answer is unambiguous.

```text
text_demo.py:1233   profiler.start("inference_prefill")
text_demo.py:1234       toks = generator.prefill_forward_text(...)   # ALL batch_size users
text_demo.py:1278   profiler.end("inference_prefill")
text_demo.py:1602   total_inference_prefill_time = profiler.get_duration("inference_prefill")
text_demo.py:1608   avg_time_to_first_token = total_inference_prefill_time      # no division anywhere
text_demo.py:1685   logger.info(f"Average Time to First Token (TTFT): {…*1000:.2f}ms")
```

The timed region wraps **one** call that prefills **all 32 users**, and the result is **never
divided by `batch_size`**. So the reported TTFT is the wall clock from the start of the batch's
prefill to the moment the **last** user's first token exists. It is **not** first-user-ready and
**not** a per-request mean, despite the identifier `avg_time_to_first_token` and the source comment
"Average prefill time for each user". Qwen is identical
(`text_qwen_demo.py:796 / 845 / 1181 / 1187 / 1265`).

### 1.5 The finding that matters most: at this configuration TTTv1's prefill is **sequential**

`generator.py:486-497` disables the batched (concat-32) prefill whenever the batch needs
slot-stable logits, and `generator.py:478-484` sets that flag for **any** greedy user:

```python
requires_slot_stable_prefill = explicit_seeded_prefill or any(
    float(temp) == 0.0 for temp in temperature_values if temp is not None)
...
use_batched_prefill = (batch >= 16 and len(set(prefill_seq_lens)) == 1
                       and prefill_seq_lens[0] == 128 and (start_pos is None or all(x == 0 …))
                       and not requires_slot_stable_prefill)
```

The gated `batch-32` parametrisation passes `{"temperature": 0.0, "top_p": 0.08}` (argmax), so
`use_batched_prefill` is False and `generator.py:531-534` emits **32 single-user work items**.
Confirmed on silicon — `perf/recon/llama_b32_run3_metackpt/run.log:2408-2474`:

```text
Prefilling User 0,  use_batched_prefill: False, prompt_lens: 118, prefill_seq_len: 128, num_cached_tokens: 0
…
Prefilling User 31, use_batched_prefill: False, prompt_lens: 118, prefill_seq_len: 128, num_cached_tokens: 0
```

**Consequence for the milestone.** Milestone C's scope decision — "prefill is sequential per row,
batched prefill is a fix to attempt and not a gate" — puts TTTv2 on *the same footing TTTv1 is
already on at the gated configuration*. The paired TTFT comparison is like-for-like. The absolute
TTFT target is a different matter; see §3.

### 1.6 Measured — UNPAIRED RECONNAISSANCE, single run, at commit `6af44349413`

**Three runs, three fresh processes**, all `exit=0`, all **PASSED**:

| run | log | TTFT ms | decode ms/iter | tok/s/user | tok/s | 128th tok ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `perf/recon/llama_b32_run3_metackpt/run.log` | 18697.89 | 14.85 | 67.35 | 2155.08 | 15.00 |
| 2 | `perf/recon/llama_b32_run4/run.log` | 18612.90 | 14.86 | 67.28 | 2153.08 | 14.98 |
| 3 | `perf/recon/llama_b32_run5/run.log` | 18990.86 | 14.86 | 67.31 | 2153.93 | 14.97 |
| **median** | | **18697.89** | **14.86** | **67.31** | **2153.93** | **14.98** |
| spread (max-min)/median | | 2.0 % | 0.1 % | 0.1 % | 0.1 % | 0.2 % |

Generated text is coherent — `perf/recon/_samples/llama_generated_text_excerpt.txt`.

Not compared against any gate here. See §3 for what the TTFT number means and what it does not.

### 1.7 Wall clock — one measured run

```text
first run on a host (cold JIT cache):   2519 s   (42 min)     llama_b32_run3_metackpt
steady state (warm JIT cache):          1104 s / 1036 s       llama_b32_run4 / llama_b32_run5
```

**Budget 1104 s (18.4 min) per measured run and 2519 s (42 min) for the warmup.** The difference is
the tt-metal JIT kernel cache, §5. Phase breakdown of the cold run
(`llama_b32_run3_metackpt`, pytest `call` 2468.22 s), from the log's own timestamps:

| Phase | Duration | From → to |
| --- | --- | --- |
| import + cluster open + config | 53 s | 09:19:23 → 09:20:16 |
| Meta checkpoint load and merge (8 × `consolidated.*.pth`) | **271 s** | 09:20:17 → 09:24:48 |
| device weight load from the converted cache + model build | 360 s | 09:24:48 → 09:30:48 |
| **prefill warmup / compile** | **1688 s (28 min)** | 09:30:48 → 09:58:56 |
| **measured prefill (this is the TTFT)** | **18.70 s** | 09:58:56 → 09:59:14.9 |
| decode compile (iteration 0) | 119 s | 09:59:14.9 → 10:01:14.0 |
| 127 measured decode iterations | 1.9 s | 10:01:14.0 → 10:01:15.9 |
| teardown | 14 s | 10:01:15.9 → 10:01:30 |

The prefill warmup is paid **every run** — it compiles and captures a prefill trace for every
supported sequence length, not just 128 — but it shrinks from 1687.81 s to ~394 s once the JIT
cache is warm. The 271 s Meta checkpoint load is also paid every run even with a fully warm
converted-weight cache (`load_checkpoints.py:159` → `Loaded and merged in 270.83s`; 287 s on run 4).
Steady-state phase split (`llama_b32_run4`, 1104 s): 51 s startup, 287 s checkpoint load, 325 s
device weight load and model build, 393 s prefill warmup, **18.61 s measured prefill**, 20 s decode
compile, 1.9 s for 127 decode iterations, 14 s teardown.

**Cold-cache one-off:** the first run on a host must convert weights into
`$TT_CACHE_PATH/tensor_cache_instruct_bfp8` — 111 GB, **33 min**
(`perf/recon/llama_b32_run1_cold/`, `wall_clock_s=2056`). Paid once per model, **not** once per
configuration: run 3 (`LLAMA_DIR`, Meta checkpoint) reused every tensor that run 1 (`HF_MODEL`,
HuggingFace checkpoint) generated, with zero regeneration, because the cache filenames key only on
the meta-style weight name, dtype and layout.

---

## 2. Qwen3-32B — the runnable procedure

### 2.1 Provisioning

**Nothing to provision.** `text_qwen_demo.py:684` builds the tokenizer with
`AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)`, so a HuggingFace repo id works and
`HF_MODEL` is the supported selector. The Llama arm's Meta-checkpoint constraint (§1.1) does **not**
apply here. This is a genuine asymmetry between the two arms and `c-perf-paired` has to satisfy both
in one night: **`LLAMA_DIR` for Llama, `HF_MODEL` for Qwen, and they are mutually exclusive**
(`model_config.py:511` asserts if both are set), so they must be set per-process, never globally.

### 2.2 The command

```sh
cd /proj_sw/user_dev/ctr-apbernal/tt-metal
export TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
export PYTHONPATH=/proj_sw/user_dev/ctr-apbernal/tt-metal
export HF_HOME=/localdev/ctr-apbernal/hf_data
export HF_MODEL=Qwen/Qwen3-32B
export TT_CACHE_PATH=/localdev/ctr-apbernal/tt_cache/Qwen/Qwen3-32B    # becomes .../TG, see 2.5
unset LLAMA_DIR MESH_DEVICE TTTV2_GALAXY_CCL_TRACE LINE_RS LINE_AG

timeout --signal=TERM --kill-after=180 5400 \
  python -m pytest -v -rA --color=no -p no:cacheprovider --timeout=5200 \
  'models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py::test_qwen_demo_text[wormhole_b0-mesh_device0-device_params0-10-performance-batch-32]' \
  > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

`--timeout` is required for the same reason as §1.2. Here `-k "batch-32"` happens to select exactly
one test, but use the node id anyway — the Llama file's does not.

Driver: `tttv2_milestone_c_runs/run_recon.sh` with `tttv2_milestone_c_runs/envs/qwen.sh`.

Benign noise to expect in the log, ~40 times, from the HuggingFace hub layer:

```text
Could not cache non-existence of file. Will ignore error and continue.
  Error: [Errno 13] Permission denied: '/localdev/…/models--Qwen--Qwen3-32B/.no_exist'
```

The Qwen checkpoint directory is owned by another account (see `ENVIRONMENT.md` §6). The message is
self-describing and harmless; do not treat it as a failure.

### 2.3 Where each metric appears

**Identical to §1.3** — `text_qwen_demo.py` emits the same lines from
`text_qwen_demo.py:1244, 1248, 1262, 1263, 1265, 1266`. The same
`tttv2_milestone_c_runs/extract_metrics.py` parses both.

### 2.4 TTFT semantics — LAST USER READY, same as Llama

```text
text_qwen_demo.py:796   profiler.start("inference_prefill")
text_qwen_demo.py:802       toks = generator.prefill_forward_text(...)   # ALL batch_size users
text_qwen_demo.py:845   profiler.end("inference_prefill")
text_qwen_demo.py:1181  total_inference_prefill_time = profiler.get_duration("inference_prefill")
text_qwen_demo.py:1187  avg_time_to_first_token = total_inference_prefill_time    # no division
text_qwen_demo.py:1265  logger.info(f"Average Time to First Token (TTFT): {…*1000:.2f}ms")
```

Prefill is sequential per user here too — the Qwen batch-32 parametrisation also passes
`{"temperature": 0, "top_p": 0.08}`, so `requires_slot_stable_prefill` is set and
`use_batched_prefill: False` appears once per user in the log.

### 2.5 **The Qwen demo asserts its own targets — and it fails them, as a warning only**

Unlike Llama (§3, F4), `text_qwen_demo.py:1271` calls `verify_perf` for exactly the gated
configuration. In `qwen_b32_run2_cold_localcache/run.log:2857-2858`:

```text
WARNING | models.demos.utils.llm_demo_utils:verify_perf:276 -
    prefill_time_to_token (1.477883) is higher than expected 0.7 (tolerance 0.15)
WARNING | models.demos.utils.llm_demo_utils:verify_perf:292 - Perf Check Failed!
```

and the test still reports **PASSED**. `verify_perf` (`llm_demo_utils.py:289-297`) does not assert;
on failure it emits a `PerfRegressionWarning` and returns.

> **`c-perf-paired` must not read a green pytest as a met gate.** For both demos, the pass/fail of
> the test says the demo ran, not that any number met any target. Parse the log.

The other two Qwen metrics were inside their bands: 61.54 vs 60 tok/s/user and 1969.29 vs 1920
tok/s, both with the default 0.15 tolerance.

### 2.6 Measured — UNPAIRED RECONNAISSANCE, single run, at commit `6af44349413`

**Three runs, three fresh processes**, all `exit=0`, all **PASSED** (with the perf warning above):

| run | log | TTFT ms | decode ms/iter | tok/s/user | tok/s | 128th tok ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `perf/recon/qwen_b32_run2_cold_localcache/run.log` | 1477.88 | 16.25 | 61.54 | 1969.29 | 16.34 |
| 2 | `perf/recon/qwen_b32_run3/run.log` | 1477.59 | 16.25 | 61.54 | 1969.20 | 16.32 |
| 3 | `perf/recon/qwen_b32_run4/run.log` | 1477.95 | 16.25 | 61.56 | 1969.79 | 16.33 |
| **median** | | **1477.88** | **16.25** | **61.54** | **1969.29** | **16.33** |
| spread (max-min)/median | | 0.02 % | 0.0 % | 0.03 % | 0.03 % | 0.1 % |

Also emitted: `1st token decode time: 1.08 ms [925.07 t/s/u, 29602.22 t/s]` — see the caution below.
Generated text is coherent — `perf/recon/_samples/qwen_generated_text_excerpt.txt`.

**Caution on "1st token decode time".** 1.08 ms / 925 tok/s/user for a 32 B model on this mesh is not
a real decode rate; iteration 1 is the first iteration after the compile iteration and is measured
host-side without a forced device sync (`text_qwen_demo.py:1041`, "e2e decode inference accounts for
device execution + host post-processing"). It is not one of the gated metrics. **Use
`Average speed`, which is the mean over iterations 1..N-1, and `128th token decode time`.** The
Llama arm's 1st-token figure (10.34 ms, 96.69 t/s/u) is inflated for the same reason, just less so.

### 2.7 Wall clock — one measured run

```text
first run on a host (cold weight conversion + cold JIT cache):  1546 s  (26 min)
steady state (both caches warm):                                 531 s / 576 s   (8.9 / 9.6 min)
```

**Budget 576 s (9.6 min) per measured run and 1546 s (26 min) for the warmup.** Phase breakdown of
the cold run (`qwen_b32_run2_cold_localcache`, pytest `call` 1507.11 s):

| Phase | Duration | From → to |
| --- | --- | --- |
| import + cluster open + config | 56 s | 10:38:35 → 10:39:31 |
| HF weight load + **cold** conversion to `TT_CACHE_PATH` (90 GB) | 504 s | 10:39:31 → 10:47:55 |
| **prefill warmup / compile** | **878 s (14.6 min)** | 10:47:55 → 11:02:33 |
| **measured prefill (this is the TTFT)** | **1.48 s** | 11:02:33.0 → 11:02:34.5 |
| decode compile (iteration 0) | 104 s | 11:02:34.5 → 11:04:18.6 |
| 127 measured decode iterations | 2.1 s | 11:04:18.6 → 11:04:20.7 |
| teardown | 9 s | 11:04:20.7 → 11:04:29 |

Steady state (`qwen_b32_run3`, 531 s) collapses the two compile phases: prefill warmup 381.02 s and
decode compile 27.09 s, and the weight conversion disappears entirely.

### 2.8 The prefill asymmetry between the arms — observed, not explained

At the same corpus, the same batch, the same 118-token prompts and the same sequential path:

```text
Llama-3.3-70B   measured prefill 18.70 s   =  584 ms per user
Qwen3-32B       measured prefill  1.48 s   =   46 ms per user
```

A 12.6× per-user gap between a 70 B and a 32 B model is far more than the parameter ratio explains.
This job did not root-cause it and **must not** — `c-perf-recon` may not modify
`models/demos/llama3_70b_galaxy/`. It is recorded here because it dominates the Llama TTFT number
and because whoever weighs the 99 ms target against the 18,697.89 ms baseline needs to know that the
32 B model on the same host, same path, is 12.6× faster per user.

---

## 3. What the numbers say about the gates — stated, not adjudicated

**Do not read a verdict into this section.** `c-perf-paired` measures and `c-signoff` decides. What
follows is the discrepancy the plan explicitly asks to have documented rather than engineered
around ("If an absolute target and paired baseline disagree materially, stop and document the
environment and baseline discrepancy").

Medians of three runs, against the `models/model_targets.yaml` entries the plan quotes:

| model | metric | target | TTTv1 median (this host, this commit) | |
| --- | --- | --- | --- | --- |
| Llama | TTFT | ≤ 99 ms | **18697.89 ms** | ~**189×** the target |
| Llama | decode | ≥ 71.5 tok/s/user | 67.31 | 5.9 % short |
| Llama | aggregate | ≥ 2288 tok/s | 2153.93 | 5.9 % short |
| Qwen | TTFT | ≤ 700 ms | **1477.88 ms** | **2.1×** the target |
| Qwen | decode | ≥ 60 tok/s/user | 61.54 | meets it |
| Qwen | aggregate | ≥ 1920 tok/s | 1969.29 | meets it |

The decode metrics are in an ordinary regime — a ~6 % baseline gap for Llama, met for Qwen. **TTFT
is not.** Both models miss it, Llama by two orders of magnitude, and the Qwen demo says so itself in
its own log (§2.5).

The TTFT discrepancy has a dated, documented cause, and it is not this host:

| date | event |
| --- | --- |
| 2025-07-25 | `PERF.md` records **TTFT 59.64 ms** at batch 32, ISL 128 — at tt-metal `633160e` |
| 2026-05-07 | `0604086b793` creates the centralised targets YAML |
| **2026-06-06** | `1fe9df83b61` (#45019) adds the `batch_size: 32, seq_len: 507` entries, incl. `prefill_time_to_first_token: 99.0` |
| **2026-06-09** | `290155969315` (#45532, "Llama 70B vLLM determinism fixes") introduces `requires_slot_stable_prefill`, which forces **greedy batches onto the sequential per-user prefill path** |

The 99 ms target was frozen **three days before** the change that took batch-32 greedy off the
concat-32 batched prefill. Nothing has re-baselined it since. On the Llama side nothing would have
noticed: `text_demo.py:1690` only calls `verify_perf` when `"repeat2" in test_id`, and `repeat2` is
a **batch-1** parametrisation — so the `batch 32 / seq 507` entry **is never asserted by the Llama
demo**. Confirmed in the log:

```text
run.log — Test 'wormhole_b0-mesh_device0-device_params0-10-performance-batch-32'
          currently doesn't have performance targets set! Skipping performance checks...
```

The Qwen demo is the opposite: `text_qwen_demo.py:1271` asserts exactly this entry, and it reports
`Perf Check Failed!` on every one of the three runs — as a `PerfRegressionWarning`, with the test
still green (§2.5).

**Practical guidance for `c-perf-paired`.**

1. The paired 3 % gate and the absolute TTFT gate are measuring different things here and will not
   agree. Report both, separately, with the semantics stated, and do not choose between them.
2. **Never relax either.** The right handling of "TTTv1 itself is 189× the absolute TTFT target" is
   to report it with this provenance and let `c-signoff` and a human weigh it — which is what the
   plan means by "stop and document the environment and baseline discrepancy".
3. On the paired axis the picture is ordinary: TTTv2 has to come within 3 % of 18697.89 ms /
   67.31 tok/s/user / 2153.93 tok/s for Llama and 1477.88 ms / 61.54 / 1969.29 for Qwen, at
   whatever commit that night is run.
4. Because prefill is sequential on **both** sides at this configuration (§1.5), the paired TTFT
   comparison is like-for-like and the Milestone C scope decision does not disadvantage TTTv2 here.

---

## 4. Metric extraction helper

`tttv2_milestone_c_runs/extract_metrics.py <run.log> [...]` prints one TSV row per log with the
label, exit code, wall clock and the six parsed metrics, using exactly the regexes in §1.3. It
works for both demos (the emitting lines are textually identical). It exits non-zero if a log it was
handed has no metric block, so a silently-failed run cannot be mistaken for a zero.

---

---

## 5. The JIT kernel cache decides the schedule

`prefill compile` and `decode compile` fall by 2-6× between the first and second run of the same
command at the same commit:

| | prefill compile | decode compile | wall clock |
| --- | --- | --- | --- |
| Llama, first run | 1687.81 s | 119.07 s | 2519 s |
| Llama, runs 2 and 3 | 393.19 / 395.72 s | 19.96 / 20.29 s | 1104 / 1036 s |
| Qwen, first run | 877.75 s | 104.15 s | 1546 s |
| Qwen, runs 2 and 3 | 381.02 / 387.80 s | 27.09 / 27.08 s | 531 / 576 s |

The cause is the persistent tt-metal JIT cache at `/home/ctr-apbernal/.cache/tt-metal-cache`
(11 GB after these runs); `TT_METAL_CACHE` is unset, so that default location is what warms. **This
is what the methodology's "one unmeasured warmup" is protecting against** — run 1 of an arm is
genuinely a different machine from runs 2-4. Hold `HOME` constant across a night and do not clear
that cache between the warmup and the measured runs.

### Schedule for `c-perf-paired` — the TTTv1 half

2 models × (1 unmeasured warmup + 3 measured):

```text
Llama   1 warmup (cold, 42 min)  + 3 × 18.4 min  =  97 min
Qwen    1 warmup (cold, 26 min)  + 3 ×  9.6 min  =  55 min
                                    TTTv1 total  = 152 min   (~2.5 h)
```

If the converted-weight caches and the JIT cache are already warm from a previous night, the two
warmups drop to ~18 and ~10 min and the total falls to ~105 min. **The TTTv1 half fits in a night
comfortably.** Whether the full 16-run night fits depends on the TTTv2 arm, which this job did not
measure. Add margin for: the mesh reset budget the house rules require after any `TT_FATAL`, and the
`/proj_sw` hazard in `ENVIRONMENT.md` §7a.

---

## 6. Summary of all measurements — UNPAIRED RECONNAISSANCE at commit `6af44349413`

Medians of three fresh-process runs per model, all on
`wh-glx6u-05-special-ctr-apbernal-for-reservation-119144`, 2026-08-29, batch 32, the 507-character
`input_data_questions_prefill_128.json` corpus, 118 tokens in / 128 tokens out, greedy
(`temperature 0.0`), paged KV `{page_block_size: 64, page_max_num_blocks: 2048}`, sequential prefill,
traced decode.

| metric | Llama-3.3-70B | Qwen3-32B | units | semantics |
| --- | --- | --- | --- | --- |
| TTFT | **18697.89** | **1477.88** | ms | **last user ready** for all 32 users (§1.4) |
| decode | **67.31** | **61.54** | tokens/s/user | mean over decode iterations 1..N-1 |
| aggregate decode | **2153.93** | **1969.29** | tokens/s | `tok/s/user × 32`, derived not independent |
| decode iteration | 14.86 | 16.25 | ms | mean |
| 128th token decode | 14.98 | 16.33 | ms | single iteration |
| wall clock, warm | 1104 | 576 | s | one measured run |
| wall clock, cold | 2519 | 1546 | s | first run on a host |

**These are not gate results.** They were taken on one night at one commit with no TTTv2 arm beside
them, and the plan's paired methodology requires both arms in the same night. `c-perf-paired`
re-runs everything. What they are good for is sanity-checking that job's TTTv1 numbers when it does:
if its TTTv1 arm lands far from this table at the same commit and firmware, something moved.

---

## STATUS

Complete. Both arms: procedure established, executed three times in fresh processes, metrics
extracted, TTFT semantics stated, wall clock known, environment recorded in `ENVIRONMENT.md`.
