# `c-exec-llama` — completion handoff (attempt 1)

**Last updated:** 2026-08-29T23:38Z — IN FLIGHT
**Base commit:** `67a208db961`. **Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`.
**Job window:** started ~22:50Z, driver PID 7812.

**Finish marker: not written. Blocked marker: not written.**

## Status

| # | Coverage item (brief) | State |
| --- | --- | --- |
| 1 | eager prefill 128/512/2048 vs `GalaxyDirectRunner`, PCC ≥ 0.99 | **observed at 128** (1 layer, PCC 0.99941), run 1 of 3 |
| 2 | eager decode, batch 1 and batch 32 | IN FLIGHT (one-layer shakeout) |
| 3 | paged KV: late capacity, bind/unbind, per-layer metadata, KV PCC | IN FLIGHT (one-layer shakeout) |
| 4 | prefix-cached and chunked prefill | IN FLIGHT; expected BLOCKED (Llama L1 clash, c-defects) |
| 5 | program compilation and `WarmupCoordinator` | IN FLIGHT (one-layer shakeout) |
| 6 | three startup/serve/cleanup cycles in one process | IN FLIGHT; at risk (same clash) |
| 7 | teacher-forced top-1 ≥ 91% / top-5 ≥ 99% | NOT STARTED (needs all 80 layers) |

## Inherited state that shapes this attempt

Read `tttv2_milestone_c_briefs/c0_completion_handoff.md` (c-defects attempt 1, complete 22:45Z,
**no finish marker**) and `tttv2_milestone_c_evidence/defects/REPORT.md` §3.

- **The Llama L1 address clash is OPEN.** `TT_THROW @ program.cpp:1763`, "Statically allocated
  circular buffers in program 100 clash with L1 buffers on core range [0-0 - 0-3]. L1 buffer
  allocated at 544832 and static circular buffer region ends at 630080", raised from
  `Embedding2D._forward` → `ttnn.embedding` on the prefill path. c-defects **refuted** the received
  explanation (it is not the prefetcher global circular buffer: the buffer is freed, measured
  absent, and the clash persists at the same address). No owner named.
- Its trigger shape matters to this job: every reproduction is a **second allocation cycle in one
  process** (a second `GalaxyDirectRunner`, or the `repeated_requests` case) or a
  **chunked/prefix program**. Multiple prefills inside one cycle are fine — c-defects took Llama
  area-4 claims to 3/3 with 32 prefill rows in a single runner. This directly threatens brief
  items 4 and 6, and it constrains the *test design* for item 1: an executor-vs-`GalaxyDirectRunner`
  comparison inside one process is itself a two-cycle process.
- **D-C12 is open**: with a warm ttnn program cache, the second device-sampling call in a process
  returns the previous call's answer. This attempt therefore qualifies the executor with
  **device sampling disabled** — every coverage item in the brief is logits- or KV-shaped — and
  says so rather than producing a sampling number D-C12 would invalidate.

## Design decisions taken (recorded here because they are the reviewable part)

1. **`executor.py` carries a model-owned runtime view.** The common runtime calls a 1D-shaped
   model contract (`prefill_forward(x, rot_mats, user_id=…, get_last_token=…)`,
   `post_process_prefill_output`, `rope_setup.get_rot_mats`, `gather_and_untilize_logits`). The
   Galaxy 2D graph exposes a different, mode-explicit contract. The adaptation is **model-owned
   code in the model package**, so `llm_runtime` needs no Galaxy branch. Details in
   `tttv2_milestone_c_evidence/exec_llama/REPORT.md` §modularity.
2. **Decode positions and page table are staged by the model, not the runtime.**
   `DecodeRuntime._prepare_inputs_host` maps both with `ShardTensor2dMesh(dims=(None, None))`,
   i.e. replicated; the Galaxy decode graph needs them **column-sharded** (`dims=(None, 0)`), which
   is what `GalaxyDirectRunner._stage_positions` / `_stage_page_table(sharded=True)` do and what
   Milestone B qualified. A replicated tensor cannot be resharded on device (per-device-different
   slicing is not expressible in one SPMD op), so the executor stages the Galaxy-placed tensors at
   the operation boundary and the view consumes those. Zero runtime lines changed.
3. **Logits composition is model-owned.** `result_collector.concat_host_output` and
   `decode._concat_host_output` concatenate mesh **columns** along the vocabulary axis; on Galaxy
   the vocabulary is sharded over the eight mesh **rows** and replicated over the four columns —
   that mismatch is D-B23, and `collectives.compose_galaxy_logits` is the qualified composition.
   The view composes with it. Decode returns the composed host tensor (both runtime readers accept
   `torch.Tensor` and pass it through); prefill re-stages the composed row replicated, because the
   runtime untilizes and slices the prefill logits on device before reading them.
4. **Sequential prefill only.** The executor resolves `supports_batched_prefill=False`; nothing is
   built on concat-32.

## Chronology

### 22:56Z — reading complete, implementation starting

Preflight: `ls /sys/class/tenstorrent | wc -l` = **32**, no `pytest|ttnn` holder,
`HF_HOME=/localdev/ctr-apbernal/hf_data` (inherited value already correct, both checkpoints under
`hub/`). The Llama Galaxy weight cache under
`model_cache/meta-llama/Llama-3.3-70B-Instruct/galaxy_8x4` is **warm** (139 GB, 320 attention
entries), so device staging is not this attempt's tax.

Nothing has run on the device yet at this checkpoint.

### 23:22Z — the executor runs on silicon, first attempt, and item 1 is observed

`models/common/models/llama33_70b_galaxy/executor.py` and
`models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py` are written. The
durable queue is `tttv2_milestone_c_runs/c-exec-llama/queue.sh` (flocked, adapted from
c-defects'), results land in `tttv2_milestone_c_evidence/exec_llama/RESULTS.md` as they finish.

| run | log | result |
| --- | --- | --- |
| `p0_partition` | `logs/p0_partition.log` | 5 passed in 17.31s — worker envelope healthy before anything else |
| `i1_ref128_l1` | `logs/i1_ref128_l1.log` | 1 passed in 117.56 s. Reference recorded through `GalaxyDirectRunner`: prefill argmax 115745, decode argmax 20110 |
| `i2_exec128_l1` | `logs/i2_exec128_l1.log` | **1 passed in 119.98 s. `[exec] prefill 128 logits 0.9994114792146915`** |

That is coverage item 1 at length 128, **observed** (one run, one layer): eager prefill through
`Llama33_70BGalaxyExecutor` agrees with the qualified `GalaxyDirectRunner` at PCC 0.99941, and the
two paths pick the same argmax token.

The PCC is 0.9994 rather than 1.0 for two identified reasons, both in the model-owned adaptation
and both deliberate: the executor's prefill rotary cos/sin are **gathered** from the row-major RoPE
table with `ttnn.embedding` rather than sliced from the tilized copy (the runtime hands the position
indices over as a device tensor, and a gather needs no host round trip and is trace-safe), and the
composed logits are re-staged through bfloat16. Neither is a threshold relaxation; the gate is
0.99 and it is met with margin.

**Everything up to here ran with `LLAMA33_70B_GALAXY_TEST_LAYERS=1`** — a one-layer subset of the
real checkpoint, ~120 s per run against six-plus minutes for all eighty. Those runs are a shakeout,
not the gate; the report distinguishes them by name (`_l1`).

### 23:30Z — the address clash reproduces in the executor, and its trigger is narrower than "second cycle"

`i5_warmup_l1` (`logs/i5_warmup_l1.log`) failed, and the failure is the headline of this checkpoint:

```text
2026-08-29 23:27:19.857 | INFO | llm_runtime.warmup:warmup_decode:493 - Compiled decode
2026-08-29 23:27:20.089 | critical | TT_THROW: Statically allocated circular buffers in program 922
clash with L1 buffers on core range [0-0 - 0-3]. L1 buffer allocated at 543488 and static circular
buffer region ends at 630080
```

from `warmup_model_prefill` -> `warmup_prefill` -> `compile_prefill` -> ... -> `embed_prefill` ->
`Embedding2D._forward` -> `ttnn.embedding`. Same op, same core range, same static CB region end
(630080) as every Milestone B and c-defects sighting.

**Three things this adds to c-defects' account of the defect.**

1. **The trigger is "a prefill after a decode in the same process", not "a second allocation
   cycle".** This process had *one* executor, *one* KV allocation, and *one* KV binding. What
   happened in between was a single `warmup_decode` — which compiles one decode program and
   nothing else. The clash landed on the next prefill, 232 ms later. Every earlier reproduction
   (the second `GalaxyDirectRunner`, `repeated_requests`, the demo) also has a decode before the
   failing prefill; this one strips everything else away.
2. **It is reachable in ~110 seconds with no request at all** — a one-layer subset, one
   `warmup_decode`, one `warmup_prefill`. That is cheaper than c-defects'
   `scratch/test_clash_layers_probe.py` (141 s) and much cheaper than the six-minute demo path,
   and it needs no page tables, prompts or sampling.
3. **The clashing address moved: 543 488 here, 544 832 in every c-defects reproduction.** Same
   core range, same CB region end. So the address is not a fixed constant of the defect; what is
   fixed is the *region* — an L1 buffer living inside the range a prefill circular buffer wants.
   This also matters for c-defects' §3 conclusion: their allocator dumps looked for live blocks
   **> 100 kB** and found none at the failing prefill. A buffer of a few kilobytes at 543 488 is
   consistent with everything they measured and would not have appeared in that filter.

**Consequences for this brief's coverage, stated rather than worked around.**

- Item 5 is now measured in **both** warmup orders, by parametrizing the test rather than by
  quietly choosing the order that works: `decode_first` hits the clash, `prefill_first` is the
  order under test for the gate. Both results are reported.
- Item 6 (three startup/serve/cleanup cycles, each of which prefills *and* decodes) is expected to
  clash on cycle 2's prefill. That is exactly the D-C7-adjacent gate line the brief warned about.
- Items 1, 2, 3 and 7 are unaffected: they prefill before they decode and never prefill again.

### 23:38Z — the one-layer shakeout, all ten runs, and two configuration facts worth inheriting

| run | result | what it says |
| --- | --- | --- |
| `i1_ref128_l1` | passed | reference recorded through `GalaxyDirectRunner` |
| `i2_exec128_l1` | **passed** | prefill 128 logits PCC **0.99941** |
| `i3_decode1_l1` | **passed** | decode row 0 logits PCC **0.99359** |
| `i4_pagedkv_l1` | failed | late resolution, metadata and bind/unbind all reached; **KV first-layer K PCC 0.907** |
| `i5_warmup_l1` | failed | L1 address clash, decode-warmup-then-prefill |
| `i6_repeat_l1` | failed | cycle 0 correct (115745, 20110 — identical to the reference); **cycle 1's prefill clashes** |
| `i7_prefix_l1` | failed | `no prefill config for recipe … attention_mode=PREFIX_CHUNKED` |
| `i8_chunked_l1` | failed | `no all_gather resources for axis=1, geometry=(1, 1, 1024, 32), sequence=1024` |

**Two of those are my test's configuration, not defects, and both are worth writing down
because `c-exec-qwen` will meet them.**

1. **A cached request is a chunked request to the planner**
   (`plan.py`: `uses_chunked_prefill = sequence_length > max_prefill_chunk_size or cached > 0`),
   and `Attention2D` resolves one frozen recipe per prefill *shape* — so the
   `PREFIX_CHUNKED` attention mode has to be registered at construction through
   `chunked_prefill_sequence_lengths`. Without it the request is refused on the host, before any
   device work. Fixed by configuring the model, not by relaxing anything.
2. **The runtime's planner, not the model, decides the padded device length.**
   `_padded_prefill_length` pads ≤128 to 128, ≤1024 to **1024**, and anything larger to the next
   power of two. So a **512-token prompt is a 1024-token device request**, and a model built with
   `prefill_sequence_lengths=(128, 512, 2048)` has no 1024 recipe and no 1024 CCL resources —
   which is exactly what `i8` reported. The registered set is now `(128, 1024, 2048)`.
   `GalaxyDirectRunner.padded_prefill_length` resolves against the same registered set, so both
   sides of every comparison pad alike and the 512 claim is a 512-token prompt in a 1024-token
   wave on both paths.

Also: `_max_prefill_chunk_size` refuses any chunk that is not a multiple of 2048, and the model's
runtime config fixes `max_prefill_chunk_size` at 2048 — so a *genuinely* chunked request needs a
context longer than 2048. The chunked test now uses 4096 tokens in two 2048-token chunks with
`max_seq_len=4096`.

**Open at this checkpoint:** the KV K PCC of 0.907 while the same request's logits agree at 0.9994.
K is the one tensor of the pair that passes through RoPE, and this executor gathers its prefill
cos/sin rather than slicing them, so a rope probe (`scratch/test_rope_gather_probe.py`) asks that
question directly. A second probe (`scratch/test_clash_owner_probe.py`) asks who owns the clashing
L1 buffer, with a named candidate: `galaxy_address_memory_config` places the prefetcher's packed
weight-address table **HEIGHT_SHARDED in L1 on `prefetch_sender_cores()`**, and the clash names
core range `[0-0 - 0-3]`.
