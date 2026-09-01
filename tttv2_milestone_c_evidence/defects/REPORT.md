# `c-defects` — what the five workstreams were, and what changed

**Job:** `c-defects`, attempt 1. **Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`.
**Base commit:** `73dd570aa4f`. **Started:** 2026-08-29T12:24Z.

Every run is in `logs/`, one file per fresh process, never overwritten. `RESULTS.md` is the
run-by-run index, written by the queue as each run lands.

**Read `RESULTS.md` before this file.** It is machine-written; this one is prose.

---

## 1. D-C5 + D-C8 (+ D-C9 and D-C10) — device sampling

### What the defects were

Milestone B measured two faults at `collectives.py:445`, `GalaxyColumnUserSelector.__call__`,
a bare `ttnn.matmul(selector, tensor)`. Fixing them exposed two more at the same call site,
one op further down each time. All four are the same defect class: **a program that resolves
its own core grid from the tensors and the full compute grid, inside a partition that does not
contain it** — the class `recipes.rope_core_grids` already names, "a grid named independently
of the partition that has to contain it".

| id | where | what |
| --- | --- | --- |
| D-C5 | the selector matmul's `in1` | the LM head's decode output is WIDTH_SHARDED for both models and the matmul requires INTERLEAVED |
| D-C8 | the selector matmul's grid | auto-selected, so it takes all 70 compute cores and leaves the loaded decode sub-device |
| D-C10 | `ttnn.topk`, `ttnn.manual_seed`, `ttnn.sampling` inside `Sampling2D.decode_forward` | three more full-grid programs, **new at this attempt** |
| D-C9 | `GalaxyDirectRunner.decode_sampled`'s readback | `to_torch_auto_compose(...)[:32]` returns one mesh column's eight users repeated four times |

### What the fixes are, and why they are the smallest that respect the boundary

**D-C5.** `__call__` stages a sharded input with `ttnn.sharded_to_interleaved` to DRAM. Not
`to_memory_config`: `_relocate_sharded` in the same file documents that a direct
`to_memory_config` between shard specs differing in grid *and* width resolves to
`reshard_program_factory_generic`, which builds over the full compute grid and is illegal
under a loaded sub-device manager, while `sharded_to_interleaved` runs on its input's own
`shard_spec.grid`. That grid is `ring_cores()`, inside `worker_cores()`.

**D-C8.** `recipes.column_user_selector_program_config` — the topology owns the placement, and
the selector resolves and caches one per logits width. The 1D `mcast_in0` form over
`worker_matmul_rectangle()` = `{[1-0 - 3-9]}`, confined with `allowed_worker_cores`, which
`ttnn` grew for exactly this and which `dense_matmul_program_config` already uses. Why not the
2D form: with `M` one tile its work grid is `num_blocks_x x 1` anchored at the rectangle
start, so all of `N` lands on three cores, `per_core_N` becomes 168 tiles and the in1 and
output circular buffers alone would ask about a megabyte of L1 beside the resident decode
activations.

The selector also gained a `program_config` argument so a caller with a different partition
can inject its own; the *default* resolves the Galaxy decode one, because decode is the only
mode this selector runs in and "a caller who forgets" is precisely how D-C8 reached silicon.

**D-C10.** In `Sampling2D`, which already carries `sub_core_grids` and `sub_core_grid_topk`
and uses them everywhere else:

- `ttnn.topk` calls `ttnn::fill_implicit_tile_padding` before it reduces. Decode logits are
  `users_per_shard` = 8 logical rows inside a 32-row tile, so the fill always fires, and
  `fill_pad`'s **interleaved** program factory resolves its cores from
  `device->compute_with_storage_grid_size()` while its **sharded** factory uses
  `shard_spec.grid`. `_place_for_topk` width-shards a padded interleaved input onto
  `sub_core_grid_topk` — over as many of its cores as divide the width in whole tiles, searched
  for rather than named because the two vocabularies share no divisor. `topk` is also handed
  an explicit interleaved output config, because with none it inherits the now-sharded input's
  and `topk_device_operation.cpp:149` rejects a sharded output.
- `ttnn.manual_seed` and `ttnn.sampling` were given no `sub_core_grids` at all. They are now
  handed the grid the module already holds; `ttnn.sampling` gets exactly `users_per_shard`
  cores from `start_core`, the construction the production 1D sampler makes for the same op.

**D-C9.** `collectives.compose_galaxy_sampled_tokens`, the mirror of `compose_galaxy_logits`
sixty lines up, and `decode_sampled` uses it and **raises** rather than slicing when the
composition is not the physical batch wide.

### The test that could not see any of it, and what it does now

`test_column_user_selector_wh_galaxy.py` built its input with
`memory_config=ttnn.DRAM_MEMORY_CONFIG` and **no loaded sub-device manager** — the one layout
a default `ttnn.matmul` accepts and the one layout the real model never produces. It passed
3/3 on silicon at Milestone B while the production path was failing twice over on the same
line.

It now also stages the LM head's own placement,
`width_sharded_memory_config(padded_local_vocab, ring_cores())`, at both models' resolved
widths (Llama 16128, Qwen 19200), under the canonical decode sub-device manager
(`prefetch_sender_cores()` + `worker_cores()`, `SubDeviceId(1)`, stalling on the workers) —
byte for byte what `galaxy_decode_mode_plan` builds and `Prefetcher2D._configure_mode` loads.

### The three logs

| log | result |
| --- | --- |
| `logs/c8_selector_run1.log` | **6 passed in 23.34s** |
| `logs/c8_selector_run2.log` | **6 passed in 24.18s** |
| `logs/c8_selector_run3.log` | **6 passed in 24.67s** |

Qwen's sampler half prints `[selector] sampled [0, 4748, 9496, ... 147188]` — 32 distinct
users at `151936 / 32` apart, which is the D-C9 readback corrected and the first time all 32
have been read back on this mesh.

The four intermediate failures are kept as well, because each one is the evidence for the next
defect: `logs/c1_selector_run1.log` (D-C10 at `topk`), `logs/c2_selector_run1.log`,
`logs/c4_selector_run1.log` (D-C10 at `manual_seed`), `logs/c5_selector_run1.log`,
`logs/c7_selector_run1.log` (D-C10 at `sampling`).

Placement probes, both under a loaded decode sub-device manager:
`scratch/probe1.log` and `scratch/probe2.log`.

### The module qualification did not move

`logs/c7_sampling_module.log` (1 passed) and `logs/c7_sampling_stochastic.log` (9 passed).

### D-C2 recorded, not changed

`Sampling2D._update_call_buffers` derives the device seed as `_device_seed(call.seed[index],
slot)` — **per `(request, slot)`**, not per request; `_seed_digest` hashes the literal
`f"sampling2d:{seed}:{slot}"`. `Sampling2D.sample_host` mirrors it exactly with
`_host_seed(request_seed, slot)`, the same digest at 63 bits instead of 31. So today, moving a
request to a different batch slot changes its token stream, and the device and host paths
agree that it does. That is the current behaviour, precisely; nothing here changes it,
and the product question it raises was deferred with vLLM.

---

## 2. D-C7 — a closed model does not return its L1

### What the defect was

`a3_q_two_pools` built Qwen twice in one process, each inside `try/finally` with `close()`,
`del` and an explicit `gc.collect()`. The first pool completed a 32-row prefill and a decode;
the second model loaded and died at its first `activate("decode")`:

```text
TT_FATAL @ bank_manager.cpp:462  Out of Memory: Not enough space to allocate 55444480 B L1
buffer across 70 banks, where each bank needs to store 792064 B, but bank size is 1393472 B
(allocated: 923776 B, free: 469696 B, largest free block: 373824 B)
```

923 776 of 1 393 472 bytes per bank — 66% — still allocated after the owner was closed and
collected. `GALAXY_GLOBAL_CB_SIZE` is `728 * 1088` = **792 064**, which is the per-bank figure
in the error to the byte: the residue is one whole global circular buffer plus 131 712 B.

### What was measured before anything was changed

`MeshDevice` exposes no allocator statistics in Python, so the probe uses the symptom itself:
two production-size global CBs cannot coexist in one L1 bank. `scratch/test_dc7_probe2.py`,
`logs/c4_dc7_probe2.log`:

```text
[dc7b] global cb size=792064
[dc7b] 1st: CREATED
[dc7b] 2nd while 1st is alive: REFUSED  TT_FATAL @ bank_manager.cpp:462
[dc7b] 2nd after dropping the 1st: CREATED
[dc7b] verdict: L1 IS returned on the last reference
```

So the destructor does free it, and the explanation has to be a surviving reference.

### What the fix is

`Prefetcher2D.cleanup()` set `self._global_cb = None` and cleared `self._contexts` — but a
`Prefetcher2DContext` is captured **by value** at module construction
(`MLP2DConfig.decode_prefetch_context`, read as `getattr(context, "global_cb", None)` at call
time), so clearing the owner's map leaves every already-built module holding a context whose
`global_cb` is still the live buffer. `cleanup()` now nulls the field on every context it
handed out, which is exactly what `_release_global_cb` already does for the decode context on
a mode switch. Three lines, and no new mechanism.

`test_cleanup_takes_the_global_cb_back_out_of_every_context_it_handed_out` fails on its
`is None` assertions without it (verified by reverting the module and re-running) and passes
with it.

### The logs

---

## 3. The Llama L1 address clash — reduced, and the received explanation is wrong

### What the defect is

```text
TT_THROW @ program.cpp:1763
Statically allocated circular buffers in program 100 clash with L1 buffers on core range
[0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region ends at 630080
```

from `Embedding2D._forward` -> `ttnn.embedding`, on the prefill path. Llama-only at this tree,
deterministic at three fresh processes across two commits, and it blocks four claims:
`a3_l_cross_slot`, `a3_l_two_pools`, `a3_l_chunked` (`program 1546`) and `a2_g6` repeat.

### What this attempt established

**The clashing L1 buffer is not the prefetcher's global circular buffer.** Two comments in
this repository say it is — `Embedding2DConfig`'s in
`models/common/models/llama33_70b_galaxy/model.py`, and
`Prefetcher2DConfig.defer_global_cb`'s — and both are inferences from the fact that deferring
the buffer made the *first* prefill placeable. Neither is a measurement of what sits at
544832.

`scratch/test_clash_probe.py` measures it, with no model in the way: create a
production-size global CB with the real sender/receiver mapping, then run `ttnn.embedding` at
both models' prefill row widths. `logs/c12_clash_probe_cold.log`:

| dim | local_dim | global CB resident | after dropping it | with none |
| --- | --- | --- | --- | --- |
| 8192 (Llama) | 2048 | **OK** | OK | OK |
| 5120 (Qwen) | 1280 | **OK** | OK | OK |

Every call is a cold compile — a distinct token count per call, so no cached program can skip
`validate_circular_buffer_region`. The first version of this probe (`logs/c8_clash_probe.log`)
did **not** have that property, ran the embedding once before creating the buffer, and its
"OK" for the resident case was a cache hit measuring nothing. That is recorded because the
first reading was almost taken as a result.

### What this means for the next attempt

Stop trying to free the global circular buffer. `release_global_cb_on_prefill` exists, it
runs, and it does not help — `tttv2_milestone_b_evidence/llama/logs3/a3_64_batch32_release_traced.log`
shows `[prefetcher] released the global circular buffer on entering prefill` immediately
followed by the clash *at the same address, 544832*. An address that does not move when the
buffer is released is an address that does not belong to the buffer.

The open question is what does own it. The cheapest way to find out is a layer-subset
reproduction — `LLAMA33_70B_GALAXY_TEST_LAYERS` is already wired into
`test_full_model_wh_galaxy.py::_load` — with `ttnn.dump_device_memory_state` around the
second prefill. `tttv2_milestone_c_runs/c-defects/queue2.sh` takes per-item environment
variables for that.

**Status: OPEN.** No fix, and no fix attempted on a premise this attempt has just refuted.

---

## 4. D-C6 — concat-32 does not fit in L1

### What the defect is

`validate_circular_buffer_region`, from `direct_runner.py:484` (`prefill_batched`), on core
range `[0-0 - 2-3]`:

```text
Statically allocated circular buffers on core range [0-0 - 2-3] grow to 1669312 B which is
beyond max L1 size of 1499136 B
```

byte-identical between the two models at every shared length, which Milestone B read — rightly
— as evidence that the allocation is a property of the shared recipe rather than either model's
dimensions.

### What it actually is

The failing op is `ttnn.linear` for the prefill QKV projection, from
`Attention2D.prefill_forward:1089`, under `recipes.dense_matmul_program_config`. `ttnn` defaults
`out_block_h`/`out_block_w` to `per_core_M`/`per_core_N`, and the 2D mcast factory sizes its
circular buffers from the **output block**, not from the per-core work:

```text
in0_CB = out_block_h * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH
in1_CB = out_block_w * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH
out_CB = out_block_h * out_block_w                       (no double buffer; interm0 shares it
                                                          when the output is interleaved)
```

The concatenated batch-32 prefill multiplies the row count by 32, so `per_core_M` — and with it
the output block — grows with the batch. `recipes.dense_matmul_cb_bytes` transcribes the
arithmetic, and it lands on the measured numbers:

| shape | resolved | this returns | silicon said |
| --- | --- | --- | --- |
| concat-32 len128 QKV | `per_core=(32,14) in0_block_w=4` | 1 671 168 B | 1 669 312 B |
| concat-32 len256 QKV | `per_core=(64,14) in0_block_w=4` | 3 112 960 B | 3 111 104 B |

The 1 856 B difference is tile alignment, in the conservative direction.

**And it explains the byte-identical figure.** Llama-3.3-70B and Qwen3-32B both have 64 heads,
8 KV heads and head_dim 128, so both resolve `qkv_size = 128 * (64 + 16) = 10240` and
`local_qkv_size = 1280`. The two geometries differ in `dim` and vocabulary, neither of which
enters this program config. It was never a coincidence.

### What the fix is

`recipes.dense_matmul_output_blocks(per_core_M, per_core_N, in0_block_w)`: the largest divisor
of `per_core_M` whose circular buffers fit `GALAXY_MATMUL_CB_BUDGET = 1 400 000 B`, narrowing
`out_block_w` only if no `out_block_h` fits. The core loops over more blocks; nothing else
changes.

The budget sits **between** the largest currently-qualified configuration (1 343 488 B, Llama's
2048-token sequential-prefill `wo` projection) and the measured ceiling (1 499 136 B), so every
shape Milestone B qualified keeps `out_block_* == per_core_*` — which is what `ttnn` would have
defaulted to — and only a shape that would have overflowed is re-blocked:

| shape | per_core | out_block | CB bytes |
| --- | --- | --- | --- |
| decode QKV | (1, 14) | (1, 14) | 274 432 |
| prefill-128 QKV | (1, 14) | (1, 14) | 274 432 |
| prefill-2048 QKV | (16, 14) | (16, 14) | 950 272 |
| prefill-2048 `wo` | (16, 22) | (16, 22) | 1 343 488 |
| concat-32 len128 | (32, 14) | **(16, 14)** | 950 272 |
| concat-32 len256 | (64, 14) | **(16, 14)** | 950 272 |
| concat-32 len512 | (128, 14) | **(16, 14)** | 950 272 |
| concat-32 len1024 | (256, 14) | **(16, 14)** | 950 272 |
| concat-32 len2048 | (512, 14) | **(16, 14)** | 950 272 |

---

## 5. The two decision items — options and consequences, for a human

The brief says: state the options and the consequence of each, do not decide. Neither was
touched.

### D-C1 — the decode page-table validator cannot tell a prefill table from a legitimate repeat

**The fact, measured** (`test_step7_page_table_placement_wh_galaxy.py`, Milestone B): a
column-sharded decode table has device-local shape `(8, 64)`; the replicated prefill table has
`(32, 64)`; `32 % 8 == 0`. `_validate_decode_page_table` discriminates on the device-local row
count alone and accepts any positive multiple of `users_per_column`, so it accepts both. Both
tables are **DRAM-interleaved**, so `memory_config().is_sharded()` is false for both and
"reject unless sharded" does not separate them either.

| option | what it costs |
| --- | --- |
| **A. Leave it.** Record that a prefill-shaped table fed to decode is accepted silently. | A caller that passes the wrong table gets wrong attention with no error. Three attempts have chosen this, each declining the boundary crossing rather than judging the risk low. |
| **B. Tighten the validator to require an L1 height-sharded table over exactly `rows / users_per_column` cores.** | This is the honest discriminator, and it makes the existing 2D-module expectations `test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[16]` and `[32]` — which pass a plain interleaved table — **wrong**. Changing a passing module expectation is the boundary crossing every attempt has declined. Someone has to say whether that expectation was a decision or an accident. |
| **C. Add a mode tag to the page-table metadata** so decode knows what it was handed instead of inferring it. | A contract change across `GalaxyPagedKVContract`, `Attention2D` and both models; no expectation has to be edited, but the blast radius is larger than either of the above. |

**Who owns it:** whoever owns the `Attention2D` paged-KV contract. The question is B's, and it
is one sentence: *is `test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats`
asserting a supported layout, or asserting today's behaviour?*

### D-C4 — area 1's headline gate is unreachable as worded

Both adaptors do `paged = paged_attention_config or default_paged_attention_config(params)`, so
`paged_attention_config=None` means "the default pool", not "contiguous". There is **no
argument** to `from_pretrained` that yields `spec.paged_attention_config is None`, even though
`Attention2D`, `GalaxyPagedKVContract` and both models' `kv_specs` support that state and the
host suite exercises it. So the gate line *"paged fill during prefill, then decode reading the
same blocks, PCC >= 0.99 against the contiguous path"* has no contiguous path to compare
against.

| option | what it costs |
| --- | --- |
| **A. Add the adaptor argument** (`paged=False` beside `paged_attention_config`). | Small: `GalaxyDirectRunner` already has the contiguous branch (`self.paged = False`, which then requires `active_slots == max_batch_size`), and `test_bringup_wh_galaxy.py` already builds a contiguous cache with `_contiguous_kv_cache(...)` + `model.set_kv_cache(...)`. It is a missing argument, not a missing mechanism. It restores the gate as worded. |
| **B. Re-word the gate** to the two-pool comparison Milestone B substituted (2048-block against 4096-block, every slot getting a different run of block ids, PCC >= 0.99 per slot). | Already measured and **passing** for both models. It tests paged addressing rather than paged-vs-contiguous equivalence, which is a weaker claim about a stronger property. Nothing to build. |
| **C. Both.** | A is the honest restoration; B is what has evidence today. |

**Who owns it:** whoever owns the milestone's exit gate wording, together with the adaptor
owner. Milestone B already replaced the tautological version of this test rather than leave a
green tick behind it, so the substitution is on record; what is not on record is whether the
gate is meant to keep asking for the contiguous comparison.

---

## Regression gates and boundaries

| gate | result | log |
| --- | --- | --- |
| step-7 host suite, before the commits | 173 passed in 51.38s | `logs/h1_step7_host_after_dc5dc8dc9.log` |
| step-7 host suite, after the commits | 173 passed in 50.84s | `logs/h2_step7_host_after_commit.log` |
| `pytest models/common/tests/llm_runtime` | **1032 passed, 1 skipped** in 218.02s, `exit=0`, **0** occurrences of `Opening user mode device driver` | `logs/h3_llm_runtime_gate.log` |
| `test_sampling_2d_wh_galaxy.py` | 1 passed | `logs/c7_sampling_module.log` |
| `test_sampling_2d_wh_galaxy_stochastic.py` | 9 passed | `logs/c7_sampling_stochastic.log` |
| `test_partition_wh_galaxy.py` | 5 passed in 17.16s | `logs/c1_partition.log` |

**The step-7 host figure is 173, not the brief's 162.** No `test_step7_*.py` *test* file is
touched by this job — only the `step7_harness.py` helper — so the difference is `mb-coverage`
attempt 4's committed cases, and the count is stable across this job's own before/after runs.

**Boundaries.** `git diff --name-only 73dd570aa4f..HEAD` is eight paths:

```text
models/common/models/galaxy/collectives.py
models/common/models/galaxy/direct_runner.py
models/common/models/galaxy/recipes.py
models/common/modules/prefetcher/prefetcher_2d.py
models/common/modules/sampling/sampling_2d.py
models/common/tests/models/galaxy/step7_harness.py
models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py
models/common/tests/modules/prefetcher/test_prefetcher_2d.py
```

**0** match `_1d\.py`. **0** match `llm_runtime`. `GalaxyDirectRunner` is not rewritten into an
executor; the one change to it is the four-line D-C9 readback.

## Two facts that cost this attempt time

1. **`pytest.ini` caps every test at 300 s.** Milestone B's area-4 runs fitted inside it only
   because they aborted early at D-C5; with the sampling path fixed they run to completion and
   the cap terminates them (`logs/c9_q_greedy_run1_TIMEOUT300.log`, `Failed: Timeout (>300.0s)
   from pytest-timeout` at 310.57 s). Every Milestone B duration for an area-4 case is now a
   *lower* bound. The queue passes `--timeout` from `c12_*` on.
2. **A program-cache hit skips `validate_circular_buffer_region`.** A placement probe that runs
   the same shape twice measures the allocator only on the first call. The first clash probe
   had this defect and its "OK" meant nothing; the corrected one uses a distinct token count per
   call so every call is a cold compile.

### D-C6 on silicon: the overflow is gone, and what was behind it is not what anyone expected

`logs/c12_q_concat_len128_run1.log`, Qwen, concat-32 at length 128, **362.99 s**:

```text
AssertionError: concat-32 at length 128 diverged from sequential in slots [4, 11]
```

**There is no `validate_circular_buffer_region` line in that log.** The program places, the
concatenated 32-row prefill runs end to end, and `prefill_batched` returns logits for all 32
slots — which no run on this mesh has ever done before. Every one of Milestone B's eleven
concat-32 runs, on both models, died before a single row's logits could be inspected.

What it now reports is a **numerical** disagreement: 30 of 32 slots take the same argmax as the
sequential path and slots 4 and 11 do not. That assertion is an exact argmax comparison, which
is the strictest form the claim can take and is brittle where two logits are nearly tied; the
concatenated path is one 4096-row matmul where the sequential path is thirty-two 128-row
matmuls, so a bf16 accumulation difference is the first hypothesis and it is **not** yet
measured. Nothing here was relaxed to accommodate it and the test is unchanged.

This is the state the brief anticipated — "area 2's real question becomes askable for the first
time" — and it is one run, not three.
---

## Area 4 in production shape — the claims are evaluable, and the first one does not pass

`logs/c12_q_device_greedy_sampling_equals_host_argmax_run1.log`, Qwen, **260.40 s**:

```text
AssertionError: device greedy disagreed with host argmax in slots [4, 8, 10, 28]
```

**Read the frame it reached, not the word "failed".** There is no `TT_FATAL` in that log: the
32-row prefill, the decode, `select_decode_column_users`, `Sampling2D.decode_forward` and the
readback all completed, under the loaded decode sub-device manager, in the production model.
D-C5, D-C8 and D-C10 are cleared on the real path, and area 4's first claim has an answer for
the first time.

The answer is **28 of 32**, not 32 of 32. Milestone B's only reading was 7 of 32 and that was
the D-C9 readback artefact — eight distinct users repeated four times — so this is the first
number that is about sampling rather than about composition.

**What the residual is not, and what it might be.** Both sides come from the same device decode
logits: `runner.decode_logits` composes them to host and takes `torch.argmax`, and the sampler
consumes the device tensor. So this is not a prefill or attention disagreement. Two hypotheses,
neither measured:

1. **exact ties in bfloat16.** The logits are bfloat16 and 151 936 wide; the maximum can be
   attained by more than one id, `torch.argmax` returns the lowest index and `ttnn.sampling`
   need not. The discriminator is one number nobody has printed: the device-chosen token's
   logit against the host-chosen token's logit. If they are equal the claim is about tie
   -breaking, not about sampling.
2. **a real disagreement** in the top-k/sampling chain.

Slot 4 also appears in the concat-32 divergence (`[4, 11]`), which is consistent with (1) —
the prompts are the same deterministic `_distinct_rows` — and is not evidence for it.

Nothing was relaxed. The assertion is the committed one, and this is one run.
---

## D-C11 — the one-hot gather was not a copy

**New at this attempt, and the largest thing it found.** `GalaxyColumnUserSelector`'s docstring
says *"the product is an exact row gather, not an arithmetic mix"*. It was an arithmetic mix.
`ttnn.matmul` with `compute_kernel_config` unset takes its default math fidelity, which
truncates the bfloat16 mantissa of its inputs, so multiplying by a one-hot matrix returns a
different number.

`logs/c16_selector_lossless.log`, a 32 x 153600 bfloat16 tensor at decode-logit magnitudes,
through the selector, under the loaded decode sub-device manager:

```text
[lossless] fidelity=default: values changed 4300324/4915200 max|delta|=0.875 mean|delta|=0.204
[lossless]   worst: row 0 col 26: want -19.875 got -19.0
[lossless] fidelity=hifi4:   values changed       0/4915200 max|delta|=0.0
```

A bfloat16 ulp at magnitude 15 is 0.125, so the default's error is several ulps — enough to flip
an argmax. Every device-sampled token produced on this Galaxy path, ever, was drawn from
corrupted logits.

### Why nothing caught it

`test_column_user_selector_wh_galaxy.py`'s four existing cases all use values a bfloat16
mantissa holds exactly: `torch.arange(32)` repeated across the width, or a single peak of 10.0
against a floor of -20.0. A lossy copy of an exactly-representable value is still that value. So
the file passed 3/3 at Milestone B while production sampled from corrupted logits, and it would
have gone on passing.

`test_column_user_selection_is_bit_exact` is a **new** case rather than a change to those four,
for exactly that reason: the old ones test placement, and placement is still worth testing.

### How it was found — four probes, three negative results

The 4/32 greedy disagreement was bisected rather than guessed at. Every probe is kept.

| question | log | answer |
| --- | --- | --- |
| is it an exact tie? | `c13_greedy_tie_probe.log` | **no** — gaps 0.125 / 0.5 / 0.125 / 0.125, `disagreed-but-tied=0`, and the device took the *lower* logit every time |
| is it the `_place_for_topk` reshard this job added? | `c14_runnerup_probe.log` | **no** — nine synthetic runner-up cases at a 0.125 gap, decode partition and none, 6 passed |
| is it the mesh-row candidate order in `gathered_values`? | `c15_runnerup_order_probe.log` | **no** — runner-up one, three and seven shards *below* the peak, 3 passed |
| is the gather lossless? | `c16_selector_lossless.log` | **no, and that is the defect** |

The three negative results are worth as much as the positive one: without them the fix would
have been a guess, and two of the three suspects were code this job had just written.

### The fix, and its own follow-up

`recipes.exact_gather_compute_kernel_config()` — **HiFi4**, as the selector's default. The first
form also asked for `fp32_dest_acc_en`, and Qwen aborted:

```text
TT_FATAL @ matmul_device_operation.cpp:567
MatmulMultiCoreReuseMultiCast1DProgramConfig: out_subblock_w 5 times out_subblock_h 1
needs to be at most 4 to fit in hardware
```

fp32 destination accumulation halves the destination register file, dropping the subblock
product cap from 8 to 4, and Qwen's `per_core_N = 20` resolves `out_subblock_w = 5`. It also
buys nothing here — a one-hot row has exactly one non-zero product per output element, so there
is no accumulation to protect — and HiFi4 alone is what the measurement qualified. That
intermediate state is kept as `logs/d11_q_device_greedy_run1_FP32ACC_REGRESSION.log`.

**Qualified:** `logs/c18_selector_after_dc11.log`, **7 passed in 45.58s**, all five placement
cases plus the bit-exactness one, at commit `0c6c8bc3e52`.

### What D-C11 changed, measured

`test_qwen_device_greedy_sampling_equals_host_argmax`, same node id, same prompts, three
tree states:

| tree | result | log |
| --- | --- | --- |
| before D-C11 | `disagreed in slots [4, 8, 10, 28]` — 28/32 | `logs/c12_q_device_greedy_run1_PRE_D-C11.log` |
| D-C11 with `fp32_dest_acc_en` | `TT_FATAL ... out_subblock_w 5 ... at most 4` | `logs/d11_q_device_greedy_run1_FP32ACC_REGRESSION.log` |
| D-C11, HiFi4 alone (`0c6c8bc3e52`) | **`disagreed in slots [4]`** — 31/32, in 152.12 s | `logs/d11_q_device_greedy_sampling_equals_host_argmax_run1.log` |

Three of the four disagreements were the lossy gather. **One slot remains and is not
explained.** Its pre-fix reading was `host 16 @ 15.375  device 15 @ 15.125  gap=0.25  ids
sharing the row maximum=2` — the only one of the four where more than one id attained the row
maximum — but those numbers were taken through the corrupted gather and do not describe the
current tree. The cheap next step is one re-run of `scratch/test_greedy_tie_probe.py`, which
already prints exactly what is needed and costs about three minutes; it was not run again
because the gate queue had the mesh.

**The claim as committed still fails, and it is reported as failing.** Nothing was relaxed.

---

## D-C12 — the second runner in a process samples garbage

**New at this attempt, unexplained, and reported rather than fixed.**

`test_qwen_a_seeded_slot_repeats_across_runs` opens **three** `GalaxyDirectRunner`s in
sequence against one model, each prefilling 32 rows and sampling once with the same seeded
policy. `logs/d11_q_a_seeded_slot_repeats_across_runs_run1.log`, 199.36 s:

```text
observed[0] = tensor([  265,  2631, 53884,    17,    16, ...])          <- plausible token ids
observed[1] = tensor([1098241487, 3196999033, 25897, 1068715143, ...])  <- not token ids
```

Qwen's vocabulary is 151 936. `1098241487` is `0x41748F4F`, which read as float32 is about
15.3 — the magnitude of a decode logit. **The second runner's sampled output is being read as
raw float data**, so the readback is landing on a buffer that holds logits rather than indices.

**What is and is not known.**

- The first runner in the process is fine; its tokens are plausible and in range.
- This is *within* one process, across `GalaxyDirectRunner` open/close cycles, on one model —
  it is not the two-models case (D-C7) and not a fresh-process question.
- **`Sampling2D._place_for_topk`, added by this job, is a suspect** and has not been cleared.
  It allocates a width-sharded L1 tensor per call and deallocates it through the module's
  `own()`/`finally` path; if `ttnn.sampling`'s result aliases anything freed there, this is
  what it would look like. The `mb-coverage` attempt-4 diagnostic ran six sampling calls in one
  process without this symptom, but it also ran under the prefill sub-device manager and
  without the reshard, so it does not clear the suspect.
- One run. Runs 2 and 3 are queued and will say whether it is deterministic.

**It is not masked.** `test_qwen_a_seeded_slot_repeats_across_runs` asserts
`torch.equal(observed[0], observed[1])` and `torch.all(observed[0] < vocab_size)`, and both
would have caught this. The test is unchanged.

---

## Area 4, Qwen, run 1 of 3 — the five claims, in the production shape

All five reach their assertions. Every one of them was a `TT_FATAL` at Milestone B.

| claim | result | log |
| --- | --- | --- |
| greedy equals the host argmax | **FAIL** — `slots [4]`, 31/32 | `d11_q_device_greedy_sampling_equals_host_argmax_run1.log` (152.12 s) |
| no padded-vocabulary id is ever sampled | **3 passed** — greedy, T=1.5, T=0.5 | `d11_q_no_padded_vocabulary_id_is_ever_sampled_run1.log` (802.37 s) |
| a near-zero temperature collapses onto the argmax (D4) | **FAIL** — `slots [4, 21]`, 30/32 | `d11_q_a_near_zero_temperature_collapses_onto_the_host_argmax_run1.log` (164.69 s) |
| a seeded slot repeats across runs | **FAIL** — D-C12, the second runner samples float bit patterns | `d11_q_a_seeded_slot_repeats_across_runs_run1.log` (199.36 s) |
| per-slot heterogeneous controls | **FAIL** — `greedy slot 8 did not take the host argmax; 265 != 17` | `d11_q_per_slot_heterogeneous_sampling_controls_run1.log` (149.02 s) |

**One of five passes. That is the honest state and nothing was relaxed to improve it.** What
changed is that the questions are now *askable*: Milestone B could not evaluate any of them,
and the two it appeared to evaluate were confounded by the D-C9 readback.

Two readings worth separating from the pass/fail column:

- **D4 is confirmed.** At `T = 0.02` the reciprocal is 50, so a correct convention makes the
  maximum dominate and an inverted one makes the distribution nearly flat. 30 of 32 slots
  collapse onto the host argmax. An inversion would show near zero. Milestone A's D4 now has a
  device measurement.
- **The per-slot failure has a suspicious shape.** Slot 8's device token is `265`, which is
  user 0's token in the eight-token sequence `mb-coverage` attempt 4 recorded
  (`[265, 2631, 1916, 220, 17, 15, 17, 17]`). Slot 8 is the first user of mesh column 1. A
  column-0 token appearing in a column-1 slot is the D-C9 signature — but the greedy case gets
  31 of 32 slots right in the same shape, so it is **not** a plain readback fault and the
  reading is not explained. It is recorded, not diagnosed.

### Llama, area 4, run 1: the greedy claim passes 32/32

`logs/d11_l_device_greedy_sampling_equals_host_argmax_run1.log`, **1 passed in 1735.59 s**.

Area 4 had never been measured on Llama. Every Milestone B attempt aborted at D-C5, and the L1
address clash killed the demo path before that. This is the first Llama device-sampling result
that exists, and it passes as committed.

**It also narrows the Qwen residual.** The selector, the sampler, the program configs and the
partition are shared code and identical between the two models. Llama gets 32 of 32; Qwen
misses slot 4. So the remaining Qwen disagreement is not in the shared path — it is something
about Qwen's logits or its vocabulary padding (Qwen pads 151 936 to 153 600; Llama pads 128 256
to 129 024, and its `_invalid_vocab_mask` covers a different fraction). That is where the next
attempt should look, and it is a much smaller space than "somewhere in device sampling".

### Llama, area 4, run 1: no padded id is ever sampled, at three temperatures

`logs/d11_l_no_padded_vocabulary_id_is_ever_sampled_run1.log`, **3 passed in 3524.04 s
(0:58:44)** — the `greedy`, `t1.5` and `t0.5` policies.

Qwen passed the same claim 3/3 at 14:24. So both models now agree, on separate runs, that
**nothing in the vocabulary padding is ever sampled**, while only Qwen still disagrees with the
host argmax, in one slot.

That pairing is worth stating because it rules something out. The padding is a contiguous id
range that no valid token can occupy, so it is the claim that would catch a sampler reading
stale, aliased or wrongly-composed memory: garbage lands in the padding roughly in proportion to
how much of the id space the padding is. Qwen's residual survives that filter at three different
temperatures. Whatever slot 4 is, it is not the readback reading the wrong buffer — it is the
sampler picking a legal token that the host would not have picked.

**A schedule number, measured rather than estimated.** This item finished 25 seconds inside its
3600 s timeout, and its three cases cost 6, 23 and 25 minutes with a warm `TT_CACHE_PATH`. The
right planning figure for Llama area 4 is **20-25 minutes per parametrized case**, so a
three-case Llama item needs 5400 s, not 3600. Three fresh processes of all five Llama claims is
7-9 hours of mesh — more than a single job window has, which is why the queue was reordered at
15:30Z to finish Qwen's rounds before starting Llama's second.

### Llama, area 4, run 1: near-zero temperature, 30/32 — and what two models together now say

`logs/d11_l_a_near_zero_temperature_collapses_onto_the_host_argmax_run1.log`, **1 failed in
257.62 s**:

```text
[temperature] T=0.02 agrees with host argmax in 30/32 slots; T=2.0 in 0/32
AssertionError: at T=0.02 the device sampled off-argmax in slots [2, 11]
```

Qwen produced the same shape at 14:27 — 30/32, slots `[4, 21]`. **This is a failing test and it
is reported as failing.** But put the two models' area-4 results side by side and the reading
changes:

| claim | Qwen | Llama |
| --- | --- | --- |
| greedy == host argmax | 31/32, slot `[4]` | **32/32** |
| no padded id sampled | 3 passed | 3 passed |
| `T = 0.02` collapses onto argmax | 30/32, slots `[4, 21]` | 30/32, slots `[2, 11]` |
| `T = 2.0` (printed, not asserted) | disagrees | 0/32 |

Two things follow that neither model could establish alone.

**First, D4 is confirmed twice.** `T = 2.0` agrees with the argmax in **0 of 32** slots and
`T = 0.02` in 30. That is the reciprocal convention behaving correctly in both directions: the
device receives `1/T`, so 0.02 concentrates and 2.0 flattens. An inversion would have produced
the opposite ordering, and it did not, on either model.

**Second, the two misses at `T = 0.02` are the draw, not the sampler.** Llama runs the *same*
sampler, on the *same* logits, under the *same* partition, in the greedy case and gets **32 of
32**. The only difference between the two cases is `forced_argmax`: the greedy call forces the
maximum, the `T = 0.02` call draws from a softmax over 32 candidates with `seed=11`. So
placement, composition, the all-gather and the selector are all exonerated for these two slots
by Llama's own greedy result — whatever moves them lives in the draw.

The mechanism that fits is a **bfloat16 tie at the top of the candidate list**. The claim's
docstring argues that at `temp = 50` "the distribution is so peaked that every slot must land on
its argmax", and that is true only while the top-two gap is large compared with `1/50`. Decode
logits here sit around magnitude 15, where bfloat16's ulp is 0.125 — so a gap of one ulp gives
the runner-up `exp(-6.25) ~ 0.2%`, but a gap of **zero**, two candidates rounding to the same
bfloat16 value, gives it **50%**, and `torch.argmax` breaks that tie by lowest index while a
sampler does not. Across 32 slots on two models, four exact-or-near ties producing two
disagreements each time is the expected order of magnitude, and the disjoint slot sets
(`[4, 21]` vs `[2, 11]`) are what a numeric coincidence looks like rather than a code path.

**This is a hypothesis with an arithmetic behind it, not a measurement, and it is not a reason
to change the test.** The measurement that would settle it is the top-two gap at exactly those
slots, which is what `scratch/test_greedy_tie_probe.py` does for the greedy case and what a next
attempt should extend to the stochastic one. Until then the claim as worded — 32/32 at
`T = 0.02` — is **not met on either model**, and that is the recorded result.

**A correction to this report's own earlier reading.** The 14:27Z entry said Qwen's two
`T = 0.02` misses were "the same residual as the greedy case — slot 4 appears in both". Llama
refutes the general form of that: its greedy case has *no* residual and its `T = 0.02` case still
misses two slots. The overlap of slot 4 on Qwen may be coincidence, and should not be carried
forward as a link between the two claims.

### Llama, area 4, run 1: the seeded-slot claim hits the address clash — and that is the best
### evidence yet that the clash and D-C12 are one defect

`logs/d11_l_a_seeded_slot_repeats_across_runs_run1.log`, **1 failed in 322.82 s**:

```text
RuntimeError: TT_THROW @ program.cpp:1763
Statically allocated circular buffers in program 100 clash with L1 buffers on core range
[0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region ends at 630080
```

from `runner.prefill_row` -> `embed_prefill` -> `Embedding2D._forward` -> `ttnn.embedding`.
The same address, the same core range, the same op as every earlier sighting.

**What is new is where it happened.** Every previous reproduction of this clash was on the demo
path or the `repeated_requests` test. This one is an **area-4 sampling claim**, and its structure
is the whole point:

```python
for _ in range(3):
    with GalaxyDirectRunner(handle.model) as runner:
        for slot, row in enumerate(rows):
            runner.prefill_row(row, slot=slot)
```

Three runners in sequence over one model. **That is the same test, in the same position, whose
Qwen counterpart produces D-C12** — the second runner returning float32 bit patterns as token
ids. Two models, one test shape, two different failures, both at the second runner:

| | Qwen | Llama |
| --- | --- | --- |
| `a_seeded_slot_repeats_across_runs` | D-C12: second runner samples logit bit patterns | L1 clash at 544832 in the second runner's first prefill |

**It was not the first runner.** Weights finished loading at 16:10:11 and the throw is stamped
16:15:12 — 301 seconds of execution first. One iteration of this test costs roughly 150 s on
Llama, measured from the neighbouring `near_zero_temperature` case (257 s total for a warm load,
32 prefill rows and three decodes). So roughly two iterations ran before the abort, which places
it at the start of the second or third runner and rules out "it never worked at all".

This does not identify the owner of 544832 — that is still open, and
`scratch/test_clash_layers_probe.py` is queued to ask the allocator directly. But it does say
the two open defects share a seam, and it means a fix for one is worth testing against the other
rather than treating them as separate workstreams.

**Llama area 4 after run 1:** greedy **passes 32/32**; no padded id sampled **passes 3/3**;
`T = 0.02` **fails 30/32**; the seeded-slot claim is **blocked by the address clash**; per-slot
heterogeneous controls is still to run.

### Llama, area 4, run 1: per-slot controls fail at **slot 8** — the same slot Qwen fails at

`logs/d11_l_per_slot_heterogeneous_sampling_controls_run1.log`, **1 failed in 231.66 s**:

```text
AssertionError: greedy slot 8 did not take the host argmax
assert 674 == 2662
```

Qwen fails the same claim at the same slot: `greedy slot 8 did not take the host argmax;
265 != 17`. **Two models, two vocabularies, two checkpoints, the same slot index.** That is not
a numeric coincidence the way the `T = 0.02` slots were — it is structural, and it points at
one line.

**The test composes its own result with `to_torch_auto_compose`.** Both models' versions read:

```python
sampled = handle.model.sample_decode(runner._decode_device_logits(tokens, positions), ...)
from models.common.auto_compose import to_torch_auto_compose
chosen = to_torch_auto_compose(sampled).reshape(-1)[:GALAXY_PHYSICAL_BATCH].to(torch.int64)
```

That is **D-C9 verbatim**, the defect this job fixed in `direct_runner.decode_sampled` and
documented in `compose_galaxy_sampled_tokens`: the sampled tokens carry the labels of the
all-gather that fed them, so auto-composing stacks the **eight identical mesh rows** and drops
the **four distinct mesh columns**, and a `[:32]` slice turns that into one column's eight users
repeated four times with no error raised. Under that composition `chosen[8]` is `chosen[0]` —
and slot 8 is exactly the first user of mesh column 1, the first index at which the repeat
becomes visible. Slots 0, 16 and 24 are also asserted by this test; slot 0 passes because it is
genuinely column 0 user 0, and the test aborts at 8 before reaching 16 and 24.

The Qwen numbers fit that reading exactly: `mb-coverage` attempt 4 recorded column 0's eight
tokens as `[265, 2631, 1916, 220, 17, 15, 17, 17]`, and Qwen's slot-8 device token is **265** —
column 0's user 0.

**This attempt did not change the test, and that is deliberate.** The reading above is an
inference from source plus one recorded token, not a measurement of this run, and "the test's
readback is wrong" is exactly the conclusion that must not be reached casually — it has the
shape of editing a test to make a failure disappear. So instead of editing anything, the
**already-committed** case `test_qwen_device_sampling_claims_with_an_explicit_token_composition`
was queued. That test runs the same six policies including a per-slot heterogeneous one, and
composes each result **both** ways — `to_torch_auto_compose` into `auto_results`, and
`compose_by_distribution` into the asserted value — so its log settles the question with numbers
from this tree.

**The prediction, recorded before the run so it cannot be fitted afterwards:** the explicit
composition will make the greedy slots agree with the host argmax, and the `auto` vector will
show the first eight tokens repeated four times. If it does, the per-slot claim is failing on its
own readback and the fix is a one-line change to a measured-correct composer, to be made and then
re-qualified three times. If the explicit composition *also* misses slot 8, the per-slot control
buffers really are landing on the wrong users and that is a new defect in the sampler.

### D-C7's device gate: **the leak is fixed, and the allocator says so to within 192 bytes**

`logs/d11_q_two_pools_run1.log`, `test_qwen_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`,
**1 failed in 454.90 s**. The test still fails — and the failure is a **different defect**, which
is exactly what a fix looks like when the claim behind it is bigger than the fix.

Milestone B's abort and this one, side by side. Same op, same test, same message class, same
request of 55 444 480 B across 70 banks:

| | allocated | free | largest free block |
| --- | --- | --- | --- |
| Milestone B, before the fix | **923 776** | 469 696 | 373 824 |
| this attempt, after the fix | **131 520** | 1 261 952 | 759 488 |

**792 256 bytes per bank came back.** `GALAXY_GLOBAL_CB_SIZE` is `728 * 1088` = **792 064**.
The difference is 192 bytes. The residue that Milestone B measured *was* one whole global
circular buffer held by a surviving reference, the surviving reference was the
`Prefetcher2DContext` captured by value in every module, and `Prefetcher2D.cleanup()` nulling
that field returns it. **D-C7 is fixed, on silicon, at model scale**, and the host test that
fails without the change is no longer the only evidence for it.

**What now fails is fragmentation, not a leak** — a new defect, recorded as **D-C13**:

```text
Not enough space to allocate 55444480 B L1 buffer across 70 banks, where each bank needs to
store 792064 B, but bank size is 1393472 B
(allocated: 131520 B, free: 1261952 B, largest free block: 759488 B)
```

There is **1 261 952 B free** and the buffer needs **792 064 B** — the space exists, with
469 888 B to spare. It cannot be used because the largest contiguous block is **759 488 B**,
short by **32 576 B**. The 131 520 B the second model has already placed sits somewhere that
splits the bank. So the second Galaxy model in a process now fails because its L1 is *cut in
two*, not because it is *full*, and the two need different fixes: this one is an allocation
order or placement question, and it is the first time it has been visible at all, because until
today the leak was large enough to hide it.

**Next step for D-C13, cheapest first.** The 131 520 B is small and identifiable —
`ttnn.dump_device_memory_state` at the moment before `Prefetcher2D` seals would name every live
L1 buffer and its address, which says whether the split is one badly-placed allocation or many.
The queued `scratch/test_clash_layers_probe.py` already calls that function at nine boundaries
for the address-clash question; the same dump answers this one, and it is worth running the
two-pools case under it.

**Run 1 of 3.** The number above needs two more fresh processes before it is evidence rather
than an observation; both are queued.

### The address clash survives a measured 792 KB of returned L1 — a third, independent refutation

`logs/d11_l_repeat_run1.log`, `test_llama33_70b_galaxy_repeated_requests_and_deterministic_cleanup`,
**1 failed in 393.14 s**, `program 100`, `L1 buffer allocated at 544832`. The fifth consistent
reproduction, unchanged.

Its value here is not the reproduction, it is the **commit it happened on**. At this same commit,
`d11_q_two_pools_run1` measured 792 256 B per L1 bank being returned when the last reference to
the global circular buffer goes — the whole buffer, to within 192 bytes. So on this tree the
global CB demonstrably *is* freed, and the clashing address is *still* 544832, byte for byte.

Three independent refutations of the received explanation now stand:

1. `release_global_cb_on_prefill` runs, logs, and the address does not move (Milestone B);
2. a production-size global CB resident does not stop `ttnn.embedding` placing at either model's
   prefill row width, every call a cold compile (`logs/c12_clash_probe_cold.log`);
3. the buffer is now measured as actually returned on this commit, and the address is unchanged.

The comments in `Embedding2DConfig` and `Prefetcher2DConfig.defer_global_cb` that name the global
circular buffer as the clashing L1 buffer should be corrected by whoever fixes this — but only
once something has named the real owner, which is still nobody. `scratch/test_clash_layers_probe.py`
is queued to do exactly that.

### D-C12: one hypothesis refuted, and a new observation that is bigger than the one it replaces

`scratch/test_repeat_sample_probe.py`, `logs/d11_repeat_sample_probe.log`, **1 failed in 15.83 s**
— no checkpoint, four consecutive `Sampling2D.decode_forward` calls in one process under the
loaded decode partition, a **different peak per call** so no program-cache hit can stand in for
an execution.

```text
[repeat] call 0: wrong=0/32  out-of-vocab=0  _local_indices allocated=True
[repeat] call 1: wrong=32/32 out-of-vocab=0  _local_indices allocated=True
[repeat]   slot 0: expected 1, got 0
[repeat] call 2: wrong=32/32 ...  slot 0: expected 2, got 1
[repeat] call 3: wrong=32/32 ...  slot 0: expected 3, got 1
[repeat] topk_returned_the_given_indices_tensor=[False, False, False, False]
```

**The hypothesis this probe was written to test is refuted, and that is worth keeping.**
The 15:36Z reading of D-C12 was that `decode_forward` might be deallocating its own
`LazyBuffer`: it passes `self._local_indices` to `ttnn.topk(indices_tensor=...)`, then `own()`s
what `topk` returns and frees everything owned in its `finally`, so if the op ever handed the
caller's tensor back, call 1 would free a buffer that `LazyBuffer._value` still claimed was live.
Wrapping the op answers it directly: `topk` returns a **fresh** tensor on all four calls
(`False, False, False, False`), and `self._local_indices.is_allocated()` is `True` after every
call. **Not that.** ttnn's docstring was right and the D-C7-mirror story is dead.

**What the probe found instead is not D-C12 and is potentially worse.** Call 0 is perfect,
32/32. Every call after it is wrong in **all 32 slots**, and the returned ids are *earlier calls'
answers*: call 1 returns call 0's peak, calls 2 and 3 return call 1's. The sampler is returning
stale results for changed input, with **one runner** and no model at all.

**Why no committed test has caught this, which is the part that matters.** The six-policy case
`..._with_an_explicit_token_composition` makes six `sample_decode` calls in one runner and reports
`the same seed in the same slot repeated in 32/32 slots` — apparently contradicting the probe. It
does not: **all six of its calls sample the same logits.** It recomputes `_decode_device_logits`
from the same tokens at the same positions each time, so a result that lags by a call is
bit-identical to a result that does not, and the test cannot see the difference. The area-4 gate
cases each sample **once** per model load, so they cannot see it either. This probe is the first
thing in the tree that changes the input between sampling calls, and it is the first thing that
fails.

**Held to the evidence standard: this is one run of new code and it is not a finding yet.** It
could be a defect in the probe — a missing synchronisation before the readback, or something
about driving `Sampling2D` directly rather than through the runner. Runs 2 and 3 are queued
(16 seconds each). If it repeats, the next step is small and obvious: run the same four calls
with the program cache disabled, which separates "a cached program is not picking up the new
input's runtime arguments" from "the readback races the op". Until then it is recorded as an
observation with a log behind it and nothing more.

**D-C12 itself is untouched by this.** Its symptom — float32 bit patterns as token ids — did not
appear here (`out-of-vocab=0` on every call), so the three-runner case still needs its own
explanation, and the runner boundary is back to being load-bearing rather than incidental.

### The Qwen slot-4 residual is an exact bfloat16 tie — measured, and the workstream closes

`scratch/test_greedy_tie_probe.py`, `logs/d11_greedy_tie_probe.log`, **1 passed in 175.57 s**:

```text
[tie] slot 4: host 16 @ 15.375  device 17 @ 15.375  equal=True  gap=0.0  ids sharing the row maximum=2
[tie] agreed=31/32  disagreed-but-tied=1  disagreed-and-not-tied=0
```

Two token ids, **16 and 17, hold the row maximum at exactly 15.375**. `torch.argmax` breaks that
tie by lowest index and returns 16; `ttnn.sampling` returns 17. Neither is wrong: the row has no
unique argmax to find.

That closes the last piece of the Qwen greedy claim. Its arc across this attempt:
Milestone B could not reach the assertion at all (D-C5); the first post-fix run disagreed in four
slots; **D-C11** — the one-hot gather losing mantissa bits at default fidelity — accounted for
three; and slot 4 is not a defect at all but a tie between two equally-likely tokens. Llama's
32/32 fits: its logits happened to have no tie.

**The `c13` reading is superseded, and by its own successor.** `logs/c13_greedy_tie_probe.log`
reported `disagreed-but-tied=0` and this run reports `disagreed-but-tied=1`. They do not conflict:
`c13` ran **before D-C11**, when four slots disagreed for a reason that was not tie-related, and
it correctly said so. With those three removed, the one that remains is the tie. The earlier
negative result was true of the tree it measured.

**What this leaves for a human.** The claim as worded — device greedy equals host argmax in 32 of
32 — is not achievable when the logits contain an exact bfloat16 tie, because the two sides break
ties differently and neither convention is more correct. That is a decision about what the claim
should say, and **this job did not make it**: the assertion is untouched and the test still fails.
The measurement is now on disk so whoever decides has the number.

### D-C12 reproduces on the **second `decode_sampled` in one runner**, not on a second runner

The same probe's second half was written to measure top-two gaps at `T = 0.02`. It measured
something else, and the something else is more important.

```text
[cold] slot  0: gap=1.375  p(runner-up)@temp50=1.388e-30  missed=True  device=71282
[cold] slot  2: gap=1.625  p(runner-up)@temp50=5.171e-36  missed=True  device=3212836881
[cold] slot  7: ...                                        missed=True  device=1077395535
[cold] T=0.02 missed=[0 ... 31]      (all 32)
```

`3212836881` is `0xBF800085` — **-1.0000020 as a float32**. `1077395535` is **2.8711126**.
`1066153360` is **1.0954**. These are not token ids; they are floating-point values read as
integers. **That is D-C12's exact signature**, and it appeared here with **one runner**, on the
**second** `decode_sampled` call of the process. The first call — the greedy one, three lines up —
was correct in 31 of 32 slots with the 32nd explained by a tie.

Read together with `d11_repeat_sample_probe` (call 0 correct, every later call returning earlier
calls' answers, synthetic logits, no model), the shape of D-C12 changes:

**it is the second sampling call in a process, not the second runner.** The three-runner test was
simply the first committed case that made more than one sampling call — and it is now clear why
nothing else caught it. Every area-4 gate case samples **once** per model load. The six-policy
`..._with_an_explicit_token_composition` case samples six times and is clean, but it goes through
`handle.model.sample_decode` directly with an `activate("prefill")` between calls, not through
`GalaxyDirectRunner.decode_sampled` — so the corrupting path is narrowed to what `decode_sampled`
does around the sampler, which is `_decode_device_logits`, the composition, and
`_deallocate_all((sampled, device_logits))`.

**A correction this forces on an earlier section of this report.** The 16:20Z entry read
`T = 2.0 in 0/32` as complementary evidence for D4 — "a flatter distribution should disagree with
argmax sometimes". In both models' `near_zero_temperature` cases, the `T = 2.0` value is the
**second** `decode_sampled` of the process, so **0/32 is equally consistent with the corruption
above** and cannot be used as evidence for the temperature convention. The `T = 0.02` half, which
is the first sampling call, is unaffected and still stands: 30/32 on both models is a real
measurement of a concentrating distribution. D4 keeps one direction of its evidence and loses the
other until the second-call defect is fixed.

**Priority note.** D-C12 was the attempt's one unexplained open defect and it needed a 3-minute
model load and three runners to see. It now reproduces in a 16-second probe with no checkpoint at
all, and it is the reason at least one gate reading has to be withdrawn. It should be the next
attempt's first job, ahead of the address clash.

### The D-C9 readback fix, qualified: **3 passed in 3 fresh processes on Qwen**

```text
16:49:26Z  dc9fix_q_per_slot_run1  1 passed in 145.50s
16:52:38Z  dc9fix_q_per_slot_run2  1 passed in 150.38s
16:55:44Z  dc9fix_q_per_slot_run3  1 passed in 145.98s
```

`test_qwen_per_slot_heterogeneous_sampling_controls` — one of the five area-4 claims — was
failing at slot 8 on both models before this change and now passes three times in three fresh
processes. The assertion is byte-identical to what it was; only the composer changed, from the
`to_torch_auto_compose` that D-C9 measured as wrong to the `compose_galaxy_sampled_tokens` that
`decode_sampled` already uses.

**This is the first area-4 claim to reach the three-fresh-process standard at this attempt with a
change behind it.** The Llama run is queued.

What it demonstrates beyond itself: the per-slot control buffers **do** land on the slots they
were written for. Slots 0, 8, 16 and 24 — one per mesh column — are forced greedy while their
neighbours sample at eight different top-k, top-p and temperature settings, and every one of them
takes the host argmax. That is a real property of the sampler, and until today it was hidden
behind a readback that could not see mesh columns 1 through 3 at all.

### The second-sampling-call defect is the **ttnn program cache**, isolated by a controlled arm

`logs/d11_repeat_sample_probe_run2.log` and `..._run3.log`, **1 failed, 1 passed** each — the
probe now runs two arms, `nocache` first so a warm cache cannot contaminate it, and the two arms
differ by exactly one call: `mesh_device.disable_and_clear_program_cache()`.

```text
[repeat] cache=nocache call 0..3: wrong=0/32  every call
[repeat] cache=nocache verdict: does not reproduce here

[repeat] cache=cache    call 0: wrong=0/32
[repeat] cache=cache    call 1: wrong=32/32   slot 0: expected 1, got 0
[repeat] cache=cache    call 2: wrong=32/32   slot 0: expected 2, got 0
[repeat] cache=cache    call 3: wrong=32/32   slot 0: expected 3, got 2
[repeat] cache=cache verdict: REPRODUCES without a second runner
```

**With the program cache cleared, four consecutive sampling calls on four different inputs are
all correct. With it warm, only the first is.** Same process, same code, same tensors, one
difference. Reproduced in two fresh processes (runs 2 and 3), 19 s each.

The mechanism this implicates is specific: on a cache hit ttnn reuses the compiled program and
only rewrites its runtime arguments, so an op whose new input address never reaches those
arguments will read the buffer the previous call used. That is precisely the observed symptom —
call *n* returning call *n−1*'s answer — and it explains D-C12's float32 bit patterns too, since
a stale address in a chain that mixes value and index tensors can land the readback on logits.

**Why the whole test suite could miss this.** Every area-4 gate case samples **once** per model
load. The six-policy `..._with_an_explicit_token_composition` case samples six times but feeds
**identical logits** every time, so a call that returns the previous call's answer is
bit-identical to a correct one. Nothing in the tree changed the input between sampling calls
until this probe did.

**What it does not yet say.** It does not name the op. The chain has a dozen candidates, and the
unusual ones — `ttnn.topk` with a persistent `indices_tensor`, `ttnn.manual_seed`, `ttnn.sampling`
and several `sub_core_grids=` arguments — are the places to look first. The next bisection is
cheap and obvious: compose the *selector's* output on each call and see whether it is already
stale before `Sampling2D` is reached, which splits the chain in two with one extra readback.

**Severity.** This is not a Milestone C sampling-module defect so much as a correctness hazard
for anything that decodes more than one token, which is every real use of this stack. It should
be the next attempt's first job.

### A queue-discipline mistake, recorded because it cost two queued runs

While the queue was live, an in-place rewrite of `q17.txt` inserted two lines *before* the
reader's byte offset instead of after it. The file was misaligned for a few seconds; the running
item happened to finish inside that window, and the queue read a **line fragment**:

```text
[queue] 2026-08-29T17:06:02Z dequeue y.py::test_qwen_per_slot_heterogeneous_sampling_controls ->
```

`rc=125`, zero seconds, no pytest process, no device touched. The reader then resynchronised on
the next newline, correctly skipped two items whose logs already existed, and carried on — but
the two `dc9fix_l_per_slot` runs that the bad edit was trying to add were swallowed with the
misaligned region. They have been re-queued as `dc9fixb_l_per_slot_run2/3`.

**Cost: two queue lines and one junk row. No mesh time and no measurement.** The junk row is
annotated in `RESULTS.md` rather than deleted, so the record shows what happened.

**The rule, now stated twice because it was learned twice:** before an in-place edit of a live
queue, resolve the running item from `chain6.out`'s last `dequeue` line — not from `pgrep`, which
cannot distinguish two queue entries that run the same file — and preserve every byte through it.
An edit that is briefly wrong is as damaging as one that is permanently wrong, because the reader
can advance at any instant.

### The D-C9 readback fix on Llama: **3 passed in 3 fresh processes** — the claim is qualified on both models

```text
17:05:17Z  dc9fix_l_per_slot_run1   1 passed in 525.86s
17:16:26Z  dc9fixb_l_per_slot_run2  1 passed in 325.16s
17:21:20Z  dc9fixb_l_per_slot_run3  1 passed in 249.92s
```

With Qwen's three at 16:49/16:52/16:55, `per_slot_heterogeneous_sampling_controls` now passes
**three fresh processes on each model**, having failed on both at slot 8 before the change. The
assertion is untouched.

**This is the first area-4 claim to reach the brief's three-fresh-process standard on both models
at this attempt.** What it establishes is a real property of the sampler, not just a green test:
slots 0, 8, 16 and 24 — one per mesh column — stay greedy and take the host argmax while their
28 neighbours sample at eight different top-k, top-p and temperature settings. Per-slot controls
land on the slots they were written for. Until today that was invisible, because the readback in
front of it could not see mesh columns 1 through 3 at all.

### The layer-subset reproduction, and what the allocator actually shows

`scratch/test_clash_layers_probe.py`, `logs/clash_layers_probe.log`, **1 passed in 140.99 s** —
"passed" because the probe asserts nothing; it reproduces and reports.

```text
[probe] layers=1 target_address=544832
[probe] clash=YES RuntimeError     (at the second runner's first prefill)
[probe] first_tokens=[115745, 20110, 45032, 12587, 97156, 68741, 25225, 51972]
```

**The clash reproduces over a one-layer subset in 141 seconds**, against six-plus minutes and a
141 GB checkpoint for `repeated_requests`. That alone changes the economics of this workstream:
it is now a two-minute experiment.

The nine `ttnn.dump_device_memory_state` dumps are in `generated/reports/probe_0*`. Filtering the
L1 block table to allocated blocks over 1 kB gives the same nine at
`probe_05_first_runner_collected`, `probe_06_second_runner_open` and
`probe_07_second_runner_generated`:

| address | size |
| --- | --- |
| **473 088** | **792 064** |
| 1 265 696 | 2 048 |
| 1 267 744 | 8 192 |
| 1 276 192 | 65 536 |
| 1 341 856 | 5 440 |
| 1 347 424 | 17 408 |
| 1 364 832 | 1 088 |
| 1 366 176 | 8 192 |
| 1 374 496 | 17 408 |

**792 064 is `GALAXY_GLOBAL_CB_SIZE` exactly**, and 473 088 → 1 265 152 encloses the entire range
the clash names — the L1 buffer at 544 832 and the static CB region ending at 630 080. Every
other live block sits above 1.26 MB, nowhere near it. The global circular buffer is resident,
low, and in the way.

**This reopens a conclusion this report drew earlier, and the earlier conclusion overreached.**
Section 3 said the clashing buffer "is not the prefetcher's global circular buffer", resting on
`logs/c12_clash_probe_cold.log`, where `ttnn.embedding` placed cleanly at both models' prefill row
widths with a production-size global CB resident. That probe is sound and its result stands — but
it supports the weaker claim that *a resident global CB does not by itself prevent the embedding*,
not the stronger one that *the buffer at 544 832 is something else*. In an otherwise-empty L1 the
allocator is free to put the buffer somewhere harmless; in the real model it lands at 473 088.
**The variable is placement, not existence, and the earlier wording did not distinguish them.**

What is now established, and what is not:

- **Established:** at the failing prefill, the only large live L1 buffer is one of exactly
  `GALAXY_GLOBAL_CB_SIZE` bytes, spanning the clash range. Two-minute reproduction, dumps on disk.
- **Not established:** that freeing it fixes the clash. The release path was checked on the host
  and does *not* have `cleanup()`'s D-C7 defect (commit `29e49f62b13`), so why the address does not
  move under `release_global_cb_on_prefill` is still open.
- **Queued:** `scratch/test_clash_layers_release_probe.py` — the same 141-second probe with the
  release switch on, dumping at all nine boundaries. If the 792 064 B block is gone at `probe_06`
  and the prefill still clashes, the clashing buffer really is something else and the dump will
  name it. If the block is still there, a reference survives the release and the question becomes
  whose.

### The global circular buffer is **exonerated by direct measurement**, and the earlier wording is restored

`scratch/test_clash_layers_release_probe.py`, `logs/clash_layers_release_probe.log`,
**1 passed in 152.25 s** — the same 141-second reproduction with
`release_global_cb_on_prefill` on, dumping the allocator at all nine boundaries into
`rel_probe_*`.

```text
[probe] layers=1 target_address=544832 release_global_cb='1'
[prefetcher] released the global circular buffer on entering prefill      (printed once)
[probe] clash=YES RuntimeError
```

Allocated L1 blocks over 100 kB, at `probe_07_second_runner_generated`:

| arm | block |
| --- | --- |
| baseline, no release | **473 088, size 792 064** |
| release on | **none** |

**The release genuinely frees the buffer.** It is present at `rel_probe_05` and `rel_probe_06`,
absent at `rel_probe_07`, and `_release_global_cb` runs before the prefill sub-device manager is
configured — so at the moment the embedding is placed, the global circular buffer's L1 is gone.
**And the prefill still clashes at 544 832.**

That settles it in the direction the earlier section argued, but on evidence rather than
inference: **the L1 buffer the clash names is not the prefetcher's global circular buffer.** The
16:20-era worry that the previous refutation had overreached was itself too cautious — the cold
probe's reasoning was weak, but its conclusion was right, and this is the measurement that earns
it. Milestone B's reading, that the release "runs and does not help *because the buffer's L1 is
not returned*", is wrong in its causal half: the L1 **is** returned, and the release does not help
because the buffer was never the problem.

**Where the answer now has to be, and it is a much smaller place.** The clash names core range
`[0-0 - 0-3]` — four cores at x=0, which is exactly where `prefetch_sender_cores()` puts the
prefetcher's senders. So the colliding buffer lives on **sender cores** and is not the global CB.
It also does not appear in any dump: at `rel_probe_07` the lowest live L1 block is at 1 265 696,
nothing near 544 832. A buffer that is absent from a dump taken after the exception, but present
when the embedding is placed, is a **transient allocated during the prefill path itself** and
freed as the exception unwinds.

**The next probe is one line of probe code, not a new idea:** open the second runner, call
`handle.model.activate("prefill")` explicitly, and dump *there* — between the release and the
first prefill op, which is the one moment nothing has dumped yet. Whatever sits at 544 832 on the
sender cores will be in that table, and then this workstream has a name instead of a hole.

**Cost of the whole line of enquiry so far: two probes, 293 seconds of mesh.** It replaced a
six-minute 141 GB reproduction with a two-minute one and turned two inferences into two
measurements.

### The clash, narrowed to the last frame anything can see — and four candidates eliminated

Both arms now dump at `probe_06b`: immediately after an explicit
`handle.model.activate("prefill")` and immediately before the first prefill op. That is the last
observable instant before the abort.

| | live L1 blocks > 1 kB at `probe_06b` |
| --- | --- |
| baseline | 473 088 (792 064) — the global CB — then nothing until 1 265 696 |
| release on | **nothing at all below 1 265 696** |

In the baseline arm the abort is entirely consistent with the global circular buffer being in the
way: 544 832 falls inside 473 088 → 1 265 152. **In the release arm that buffer is gone, L1 below
1.26 MB is empty, and the abort still names an L1 buffer at 544 832.** Two arms, same probe, same
commit, 133 and 141 seconds.

**What is left between that dump and the abort is two allocations, and both are explicitly DRAM.**
`Embedding2D._forward` calls `load_device_weights()` — the `[1, 1, 128256, 2048]` bfloat16 table,
`DRAM_MEMORY_CONFIG`, visible as DRAM in the traceback — and `_load_token_ids`, which pins
`memory_config=ttnn.DRAM_MEMORY_CONFIG` on the staged row explicitly
(`embedding_2d.py:162`). Nothing in the Python path puts anything in L1 in that window.

So the candidate list is now:

| candidate | status |
| --- | --- |
| the prefetcher's global circular buffer | **eliminated** — freed, measured absent, clash persists |
| state left resident by the first runner | **eliminated** — `probe_06b` under release is empty below 1.26 MB |
| the embedding weight table | **eliminated** — DRAM, by config and by traceback |
| the staged token ids | **eliminated** — `DRAM_MEMORY_CONFIG`, pinned in `_load_token_ids` |
| a sharded L1 buffer the bank-table dump does not enumerate | **open** |
| a buffer the runtime allocates during program launch | **open** |

**The honest limit of this instrument.** `dump_device_memory_state`'s L1 block table is the
interleaved bank allocator's view. The two surviving candidates are precisely the things that view
would not show, and no arrangement of dumps around the Python call sites can settle them —
`probe_06b` is already the last frame there is. Naming the buffer needs a different tool:
`TT_METAL_WATCHER`, or instrumentation at the throw site in `program.cpp:1763`, which has both the
address and the core range in hand.

**Status: still OPEN, and much better posed.** A next attempt inherits a **141-second**
reproduction instead of a six-minute one, four eliminated candidates instead of one wrong
explanation, and a specific next instrument. What it does not inherit is a fix, and this attempt
did not manufacture one.

## Qwen area 4, complete: three fresh processes on all five claims

| claim | run 1 | run 2 | run 3 | residual |
| --- | --- | --- | --- | --- |
| device greedy == host argmax | 31/32 | 31/32 | 31/32 | slots **[4]**, all three |
| no padded id is ever sampled | **3 passed** | **3 passed** | **3 passed** | — |
| `T = 0.02` collapses onto argmax | 30/32 | 30/32 | 30/32 | slots **[4, 21]**, all three |
| a seeded slot repeats across runs | fail | fail | fail | D-C12 |
| per-slot heterogeneous controls | fail (pre-fix) | **pass** | **pass** (+ 3/3 as `dc9fix_q_*`) | — |

**Every residual is bit-deterministic across three fresh processes.** Not "usually [4]" — `[4]`
three times, and `[4, 21]` three times. That matters for what it rules out: a race, a scheduling
artefact, or an uninitialised buffer would not land on the same slot indices in three separate
processes. These are properties of the arithmetic, and the two that are explained have their
explanations measured:

- **slot 4** is an **exact bfloat16 tie** — ids 16 and 17 both at 15.375, `torch.argmax` takes the
  lower index and `ttnn.sampling` does not (`logs/d11_greedy_tie_probe.log`);
- **`no padded id`** passes 9 times out of 9 across three runs of three policies, which is the
  claim that would catch a sampler reading the wrong memory, and it does not fire;
- **per-slot** passes once the readback is the one D-C9 measured as correct — three fresh
  processes on each model;
- **the seeded claim** fails on D-C12, now traced to the **ttnn program cache** (`nocache` clean,
  `cache` wrong from call 1, four fresh processes);
- **slot 21 at `T = 0.02`** is the one number here with no measurement behind it. It is
  reproducible and it is not explained. Said plainly rather than folded into the tie story: the
  tie probe measured slot 4's gap as exactly zero, and slot 21's gap was measured in the same run
  but in the **second** `decode_sampled` call, which the program-cache defect corrupts — so that
  reading is void and slot 21 is open.

**Two of five Qwen claims pass. Three fail, all three for reasons that now have names**, and two
of those three are defects outside the sampler (a tie-break convention, and the program cache).

### D-C7's device gate, qualified: **three fresh processes, byte-identical**

```text
18:28Z  d11_q_two_pools_run1   allocated: 131520 B, free: 1261952 B, largest free block: 759488 B
18:27Z  d11_q_two_pools_run2   allocated: 131520 B, free: 1261952 B, largest free block: 759488 B
18:33Z  d11_q_two_pools_run3   allocated: 131520 B, free: 1261952 B, largest free block: 759488 B
```

Not "about 131 kB" — the same three numbers, to the byte, in three separate processes, against
Milestone B's `923 776 / 469 696 / 373 824`. **D-C7 is fixed and the fix meets the brief's
three-fresh-process standard.** 792 256 B per L1 bank return, against a `GALAXY_GLOBAL_CB_SIZE`
of 792 064 B — 192 bytes of slack, reproducibly.

**D-C13 is equally reproducible.** All three runs still fail, all three with 1 261 952 B free and
a largest contiguous block of 759 488 B — **32 576 B short** of the 792 064 B needed. The bank is
fragmented, not full, by the same amount every time. A next attempt can treat that as a fixed
target rather than a flaky one.

## Llama area 4, complete: three fresh processes on all five claims

| claim | run 1 | run 2 | run 3 | residual |
| --- | --- | --- | --- | --- |
| device greedy == host argmax | **pass 32/32** | **pass** | **pass** | — |
| no padded id is ever sampled | **3 passed** | **3 passed** | **3 passed** | — |
| `T = 0.02` collapses onto argmax | 30/32 | 30/32 | 30/32 | slots **[2, 11]**, all three |
| a seeded slot repeats across runs | fail | fail | fail | **L1 clash at 544832**, all three |
| per-slot heterogeneous controls | fail (pre-fix) | **pass** | **pass** (+ 3/3 as `dc9fix*_l_*`) | — |

**Area 4 is now measured three times on both models.** Before this attempt it had never been
measured on Llama at all — every Milestone B run died at D-C5 before reaching the sampler, and the
address clash killed the demo path before that.

**Llama passes three of five claims outright.** Greedy is 32/32 in three separate processes; no
padded id is sampled in nine runs of three policies; per-slot controls land correctly on both
models once the readback is right.

The two failures are as deterministic as Qwen's:

- `T = 0.02` misses slots **[2, 11]** in all three runs — disjoint from Qwen's `[4, 21]`, and
  Llama's own greedy claim passes 32/32 on the same logits through the same sampler, so the misses
  are in the **draw**, not the placement or the composition.
- the seeded claim aborts at the **same L1 address, 544832**, in all three runs.

### Both models side by side

| claim | Qwen | Llama |
| --- | --- | --- |
| greedy == host argmax | 31/32 ×3, slot [4] is an **exact bf16 tie** | **32/32 ×3** |
| no padded id sampled | **3 passed ×3** | **3 passed ×3** |
| `T = 0.02` collapses | 30/32 ×3, slots [4, 21] | 30/32 ×3, slots [2, 11] |
| seeded slot repeats | fail ×3 — **program cache** | fail ×3 — **L1 clash** |
| per-slot controls | **pass ×3** | **pass ×3** |

Ten of the twenty-five claim-runs behind this table are passes on claims that could not be reached
at all when this attempt began. **Every failure has a named cause except slot 21 on Qwen and slots
[2, 11] on Llama**, which share one open question: what a stochastic draw should do when the
top-two logit gap approaches the bfloat16 floor.

## 4b. D-C6 after the fix: concat-32 runs, and what it disagrees about is measured

The L1 overflow is gone and the length sweep ran for the first time.

| case | result |
| --- | --- |
| Qwen, len 128 | fail — slots **[4, 11]**, identical in **three** fresh processes |
| Qwen, len 256 | fail — slot **[25]** |
| Qwen, len 512 | **PASS — 32/32 slots agree with sequential** |
| Llama, len 128 | **PASS — 32/32** |

**Concat-32 now executes end to end at every length tried, on both models**, and passes outright
at Qwen-512 and Llama-128. Both are firsts: at Milestone B this path could not allocate its
circular buffers at all (`1 669 312 B` against a `1 499 136 B` L1). The divergence count also falls
with length — 2 slots at 128, 1 at 256, 0 at 512 — which is the shape a numerical-margin story
predicts.

### The tie hypothesis was tested and is **refuted**

Slot 4 is the exact bfloat16 tie the greedy claim also trips on, so the obvious guess was that
concat-32 and sequential differ by an ulp and the argmax flips only where the row is tied.
`scratch/test_concat_tie_probe.py` (`logs/concat_tie_probe.log`) measures it, and the guess is
wrong:

```text
[concat] slot  4: seq id=17 gap=0.25   bat id=17 gap=0      differs=False  max|seq-bat|=0.65625
[concat] slot 11: seq id=18 gap=0.125  bat id=16 gap=0.125  differs=True   max|seq-bat|=0.75
[concat] slot 25: seq id=9552 gap=12.5 bat id=9552 gap=12   differs=False  max|seq-bat|=1.0625
[concat] every diverged slot is an exact tie: False
```

**The two paths' logits differ by up to 1.06** — eight bfloat16 ulp at a magnitude of 15, not one.
That is far above rounding noise, so concat-32 is **not** numerically equivalent to sequential
prefill; it is merely close. The argmax then flips exactly where a row's top-two gap is smaller
than that difference, and **7 of 32 rows have a gap below 1.06** — which is why a handful of slots
diverge, why the count falls as longer prefills sharpen the distribution, and why slot 25's gap of
12.5 never moves.

So the finding is not a tie-break. **It is a real discrepancy of order one logit unit between two
paths that are supposed to compute the same thing**, and the gate is right to fail on it. Whether
1.06 is acceptable is a judgement about what "matches sequential prefill" should mean, and this
job did not make it — the assertion is untouched.

### One inconsistency, reported rather than smoothed

The gate case reports diverged slots `[4, 11]` in three fresh processes. The probe, running the
same two prefills with the same `_load` arguments, reports `[11, 13]` — slot 4 **agreed** there
(both sides chose id 17). The gate's own three runs are stable, so this is not run-to-run noise in
the gate; it is a difference between the gate and an ostensibly equivalent probe that this attempt
has not explained. It is one probe run. Recorded because a divergence set that moves when the
harness changes is itself information about how marginal these rows are, and because burying it
would make the D-C6 story tidier than the evidence.

## Regression gates, re-run after every change

Every gate below was re-run at the final commit, not carried over from earlier in the attempt.

| gate | result | log |
| --- | --- | --- |
| `test_step7_concat32.py` | 34 passed | `logs/reg_s7_concat32.log` |
| `test_step7_long_context.py` | 32 passed | `logs/reg_s7_long_context.log` |
| `test_step7_page_table_placement_wh_galaxy.py` | 3 passed | `logs/reg_s7_page_table_placement_wh_galaxy.log` |
| `test_step7_paged_kv.py` | 37 passed | `logs/reg_s7_paged_kv.log` |
| `test_step7_prefix_cache.py` | 18 passed | `logs/reg_s7_prefix_cache.log` |
| `test_step7_repeat_and_cleanup.py` | 12 passed | `logs/reg_s7_repeat_and_cleanup.log` |
| `test_step7_sampling.py` | 29 passed | `logs/reg_s7_sampling.log` |
| `test_step7_token_composition.py` | 8 passed | `logs/reg_s7_token_composition.log` |
| **step-7 total** | **173 passed** | — |
| `models/common/tests/llm_runtime` | 1032 passed, 1 skipped | `logs/reg_llm_runtime.log` |
| `test_column_user_selector_wh_galaxy.py` | 7 passed | `logs/reg_selector.log` |
| `test_sampling_2d_wh_galaxy.py` | 1 passed | `logs/reg_sampling_module.log` |
| `test_sampling_2d_wh_galaxy_stochastic.py` | 9 passed | `logs/reg_sampling_stochastic.log` |
| `test_partition_wh_galaxy.py` | 5 passed | `logs/reg_partition.log` |
| `test_prefetcher_2d.py` (host) | 22 passed | run inline |

**173 is the same total as before any of this attempt's changes**, so nothing regressed.

**The one skip, named rather than waved through.** `llm_runtime`'s single skip is
`models/common/tests/demos/llama3_8b/demo.py:378: Unsupported MESH_DEVICE=''` — a demo that
declines to run outside its supported mesh list, unrelated to this job and present in the same
form at Milestone B's 1032/1. It is **not** an `hf_config_or_skip` skip, which is the kind the
brief says to treat as a failed run, and `HF_HOME` was `/localdev/ctr-apbernal/hf_data` for every
run in this file (each log records it in its header).

**Scope prohibitions, checked mechanically at the final commit** (`git diff --name-only
73dd570aa4f..HEAD`, 12 files, 9 commits, branch `apbernal/tttv2_wh_glx_2d_modules_milestone_c`):

- files matching `_1d\.py`: **0**
- files matching `llm_runtime`: **0**
- `tttv2_2d_modules_plan.md`: **untouched**, as the brief requires

### D-C6, qualified: two concat-32 gates now meet the three-fresh-process standard

```text
Qwen  len512:  20:43Z / 21:15Z / 21:19Z    1 passed  ×3
Llama len128:  20:51Z / 21:29Z / 21:33Z    1 passed  ×3
```

**Concat-32 prefill of 32 rows now matches sequential prefill in all 32 slots, three times over,
on both models.** At Milestone B this path could not allocate at all. These are the first
concat-32 passes that exist, and they are qualified rather than incidental.

The failures are equally reproducible: Qwen len128 diverges in `[4, 11]` in all three runs and
len256 in `[25]` in all three. Same slots, same processes-independent answer — a numerical margin,
not a race.

### The full concat-32 length sweep the brief asked for

| length | Qwen | Llama |
| --- | --- | --- |
| 128 | fail — slots `[4, 11]` ×3 | **pass ×3** |
| 256 | fail — slot `[25]` ×3 | **pass ×3** |
| 512 | **pass ×3** | **pass ×3** |
| 1024 | not parametrized | **pass** (run 1) |
| 2048 | not parametrized | **pass** (run 1) |

**Llama's concat-32 prefill matches sequential prefill in all 32 slots at every length in the
brief's 128-2048 range**, three fresh processes at 128, 256 and 512 and one each at 1024 and 2048
with the remaining runs queued. Qwen matches at 512 and misses a small, fixed set of slots below
that.

Two things worth saying about this table. First, **it exists at all** — at Milestone B concat-32
aborted before executing, at any length, on either model, because its circular buffers wanted
1 669 312 B against a 1 499 136 B L1. D-C6's `dense_matmul_output_blocks` fix is what turned that
into a measurable comparison.

Second, **Qwen is deliberately not extended past 512.** Its `_BATCHED_LENGTHS` stops there and the
file's own comment says to extend only once 128-512 pass once. 512 passes; 128 and 256 do not. So
the condition is unmet and the tuple is untouched — extending it to collect more green results
below a failing gate would be exactly the kind of parametrization change the brief prohibits.

---

# Final state at 22:45Z

**118 device runs**, every one with a log on disk under `tttv2_milestone_c_evidence/defects/logs/`
and a row in `RESULTS.md` recording its node id, verdict, wall time and `HF_HOME`.

## What the concat-32 sweep finished at

| length | Qwen | Llama |
| --- | --- | --- |
| 128 | fail — `[4, 11]` ×3 | **pass ×3** |
| 256 | fail — `[25]` ×3 | **pass ×3** |
| 512 | **pass ×3** | **pass ×3** |
| 1024 | not parametrized | **pass ×3** |
| 2048 | not parametrized | **pass ×3** |

**Llama concat-32 matches sequential prefill in all 32 slots at every length the brief names, three
fresh processes each.** That is 15 passing runs of a path that could not allocate its circular
buffers at all when this attempt began.

## Defects: eight found, six fixed

| id | what it was | state |
| --- | --- | --- |
| D-C5 | the selector matmul rejects the LM head's WIDTH_SHARDED decode output | **fixed** |
| D-C8 | its auto-selected grid leaves the loaded decode sub-device | **fixed** |
| D-C10 | three more full-grid programs in `Sampling2D` — `topk`'s padding fill, `manual_seed`, `sampling` | **fixed**, new |
| D-C9 | the sampled-token readback composes the wrong mesh axis | **fixed**; and the same defect found again in the per-slot gate's own readback, fixed and qualified ×3 on both models |
| D-C11 | the one-hot "exact row gather" changed 4 300 324 of 4 915 200 values; 0 at HiFi4 | **fixed**, new, the largest finding |
| D-C7 | `cleanup()` left the global CB bound to every context it handed out | **fixed**, and qualified on silicon ×3: 792 256 B/bank returned against a 792 064 B buffer |
| D-C6 | `out_block_*` defaulted to `per_core_*`, so concat-32's CBs grew with the batch | **fixed**; the sweep above is the result |
| D-C12 | the second sampling call in a process returns stale results / float bit patterns | **root cause isolated, not fixed** — the ttnn program cache |
| D-C13 | the second model's L1 is fragmented, not full: 32 576 B short, reproducibly | **new, open** |
| Llama L1 clash | `ttnn.embedding` cannot place at the second runner's prefill | **open**, with four candidates eliminated and a 141 s reproduction |

## The three things a next attempt should do first, in order

1. **Fix the program-cache defect (D-C12).** It is a correctness hazard for anything that decodes
   more than one token. `scratch/test_repeat_sample_probe.py` reproduces it in **16 seconds** with
   no checkpoint and already contains the controlled arm that proves the mechanism. It has also
   cost one gate reading — `T = 2.0 in 0/32` had to be withdrawn as evidence for D4.
2. **Name what owns L1 544 832.** `scratch/test_clash_layers_probe.py` reproduces the clash in
   **141 seconds** over a one-layer subset and dumps the allocator at ten boundaries. Four
   candidates are eliminated; the two that remain need `TT_METAL_WATCHER` or instrumentation at
   `program.cpp:1763`, which has the address and the core range in hand.
3. **Decide what the near-tie claims should say.** Qwen's greedy slot 4 is an exact bfloat16 tie
   with two ids at 15.375; `T = 0.02` misses two slots on each model; concat-32 diverges only where
   a row's top-two gap is under the ~1.06 by which the two prefill paths differ. Each is measured.
   None is a bug in the sense the assertions assume, and **this job did not move any assertion** —
   a human should rule on the wording.

## Finish and blocked markers: **neither is written**

The finish condition is not met, and here is exactly what is short:

- the **Llama L1 address clash** is open, so `a3_l_cross_slot`, `a3_l_two_pools`, `a3_l_chunked`,
  the `a2_g6` repeat and Llama's seeded-slot claim remain blocked;
- **D-C12** is isolated but not fixed, and it invalidates any claim that samples twice per process;
- **D-C13** is new and open;
- **Qwen concat-32** fails at 128 and 256;
- the **near-zero-temperature** claim fails on both models;
- **D-C1 / D-C4** have options written for a human and nothing decided, as the brief intended.

The blocked marker is not written either, and the reason is the opposite one: **every workstream
advanced on silicon at this attempt.** Two defects were found and fixed that nobody knew existed,
two gates that had never been reached now pass three times on both models, the concat-32 path went
from unallocatable to a full length sweep, and the two open defects each acquired a reproduction
measured in seconds rather than minutes. A job with that much movement is partway through, not
blocked.

## Evidence integrity, checked at 22:47Z

| | |
| --- | --- |
| device runs in `RESULTS.md` | **118** |
| logs on disk | **123** |
| allocator dumps preserved | **42** (`defects/dumps/`) |
| logs carrying the `HF_HOME` header | **120 of 123** |
| finish marker | **not written** |
| blocked marker | **not written** |
| `tttv2_2d_modules_plan.md` | **untouched** |
| tracked working-tree modifications | **0** — everything committed |

**The three logs without an `HF_HOME` header** are `h1_step7_host_after_dc5dc8dc9.log`,
`h2_step7_host_after_commit.log` and `h3_llm_runtime_gate.log` — host-only gate runs made by hand
early in the attempt, before everything went through `queue.sh`, which is what writes the header.
All three are superseded by the `reg_*` runs at 20:12-20:57Z, which do carry it and which were run
at the final commit. Named here so the 120-of-123 is not a loose end.

Every remaining log records its name, node id, commit, start time and `HF_HOME` in its first five
lines, and `HF_HOME` is `/localdev/ctr-apbernal/hf_data` in all of them. No run in this file was
counted as a pass while skipping.

---

# Attempt 3 (2026-08-31) — the Llama L1 address clash, fixed

## What the defect was, stated correctly for the first time

Three earlier accounts of this defect read the message wrong in the same way. The message is

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1 buffers on
core range [0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region
ends at 630080
```

`detail::ProgramImpl::validate_circular_buffer_region` (`tt_metal/impl/program/program.cpp:1658`)
computes **one** `lowest_address` for the whole program — `device->lowest_occupied_compute_l1_address`
— *before* it loops over the program's circular-buffer core ranges, and then reports that one number
against whichever range trips first. So `[0-0 - 0-3]` locates the **circular buffers**, not the L1
buffer, and the buffer is not on the prefetch sender column. Second, `FreeListOpt::get_memory_block_table`
prints `block_address_[i]` **without** `offset_bytes_` while `lowest_occupied_address()` **adds** it,
and on this part that offset is `1499136 - 1393472 = 105664`. Attempt 2 measured the offset directly
(`ttnn.get_allocator_base_address` → 105664, `logs/a2_clash_owner_l1.log`) rather than deriving it,
and with the two coordinate systems reconciled the "unfindable" buffer was in attempt 1's own
preserved dumps all along: a **32-byte** block, which is why a filter for live blocks over 100 kB
never showed it.

## The defect has two independent halves

Both are attributed to the call that makes them, by a probe that prints the allocated-block table on
each side of every `Prefetcher2D` step that can allocate (`logs/a3_clash_steps_l1.log`):

| step | L1 blocks below 630080 afterwards |
| --- | --- |
| `runner.open()` | none; lowest allocated 1373408 |
| `activate("prefill")` | none |
| first `activate("decode")` → `_ensure_global_cb` | **578560 (192 B) and 578752 (792064 B)** |
| the **first decode forward** | **545760 (32 B)** appears, and is never freed |
| every later mode switch | unchanged |

1. **The global circular buffer is itself below the line whenever it is resident.** It is
   `GALAXY_GLOBAL_CB_SIZE` = 792 064 B plus a 192-byte config page, allocated top-down beneath the
   model's resident L1, landing at 578752 — 51 328 B below the 630080 the prefill embedding's static
   circular buffers reach. And this is not a placement accident that a different order would fix:

   | quantity | bytes |
   | --- | --- |
   | L1 above the prefill CB region end (1499136 − 630080) | **869 056** |
   | the model's resident L1 after a decode (1499136 − 1371360) | 127 776 |
   | the global circular buffer, data + config | **792 256** |
   | sum | **920 032** |

   920 032 > 869 056 by 50 976 B, so **the buffer and a prefill program of this shape cannot
   coexist on any allocation order.** `defer_global_cb` covers only the *first* prefill.
   `release_global_cb_on_prefill` is therefore not an optimisation, it is required — and a serving
   system interleaves prefill and decode by construction.

2. **The first decode forward strands a 32-byte L1 buffer at 545760.** It is allocated while the
   global CB holds the space above it, so it lands *below* the buffer rather than at the top of L1,
   and an allocated address never moves: it survives `_stop_prefetch`, the mode switch and
   `_release_global_cb`. Releasing the CB therefore frees 792 256 B and leaves this one block below
   the line — and because the throw reports the **lowest** occupied address, the message does not
   change by a single digit. **That is why three attempts read
   `release_global_cb_on_prefill` as "runs and does not help".** It ran and it helped; the message
   was about a different, smaller buffer.

   Attempt 2 walked the model/runner object graph for the owner and found nine reachable L1
   `ttnn.Tensor`s after the first decode, **all above 1371360** — including
   `rope_setup.config._decode_trans_mat`, materialised by the first decode, which landed at 1371360
   because it happened to fit a gap at the top. So the 32-byte block is not a reachable tensor: the
   remaining L1 owners at that layer are `GlobalSemaphore`s and buffers held inside cached ttnn
   programs. `_weight_address_metadata` — `c-exec-llama`'s named candidate — is exonerated by
   measurement at **1499104**, the very top of L1.

## What the fix is, and why it is two changes rather than one

`models/common/modules/prefetcher/prefetcher_2d.py` grows one config field and one method;
`models/common/models/galaxy/prefetch.py` changes two defaults. Nothing else in the module changes,
and no model file gains anything but a default and a comment.

**1. `release_global_cb_on_prefill` defaults to `True` for both Galaxy models.** Forced by the
arithmetic above, not chosen. The field and the release path already existed and were already
measured to return 792 256 B per bank (attempt 1's D-C7 gate); what was missing was the reason to
turn it on and the second half that makes it sufficient.

**2. `global_cb_headroom` — reserve L1 above the buffer while it is created, then release it.**
L1 is allocated top-down, so with the buffer resident every long-lived allocation lands below it
and is stranded. Reserving headroom first makes the buffer land that much lower and leaves a free
gap above it, and `FreeListOpt::allocate` scans free blocks by **ascending size class**, so a later
small allocation takes the small gap in preference to the large low block.

That much was measured in isolation before anything was committed. Three arms of a scratch probe on
the two-`generate` reproduction, one Llama layer:

| arm | log | result |
| --- | --- | --- |
| headroom only, no release | `logs/b1_headroom_only_l1.log` | **the stranded 32 B at 545760 are gone.** The only L1 left below 630080 is the global CB itself, moved down by exactly the headroom (578752 → 513024). The clash persists, because the buffer is still resident — exactly as the arithmetic says it must. |
| headroom **and** release | `logs/b2_headroom_release_l1.log` | **the second prefill placed** — no `program.cpp` throw, the first time on this branch. And then the process **hung** for four minutes in the decode after it, and was killed by PID. |
| release only | `logs/b3_release_only_l1.log` | not measured: dequeued 1 s after that kill and died in 6.35 s with `MMIO per-op timeout`, an infrastructure failure. `tt-smi -glx_reset` reported `Re-initialized 32 boards`. |

**The hang is the hazard `release_global_cb_on_prefill`'s own docstring names**: the recreated buffer
must land at the same L1 address or decode programs already in the ttnn program cache hold stale
addresses. `logs/b4_headroom_release_addr_l1.log` measured it rather than inferring it — the arm
printed the buffer's base at each creation:

```text
[fix] before create: lowest=1370816  reserve=65536  -> after create: lowest_allocated=447296
[fix] before create: lowest=1338016  reserve=65536  -> after create: lowest_allocated=414880
```

**32 416 B lower the second time**, and it hung again. A *fixed* headroom cannot put the buffer
back, because the free list is a different shape the second time: the first decode has consumed
part of the gap the headroom left. Nor can the placement be predicted — the same log shows the span
between the free-region top and the resulting floor was 857 984 B on the first creation and
857 600 B on the second, so creating a global circular buffer allocates more than the buffer and its
config page, and not a constant amount.

**The invariant that is reliable is the free-region top.** Reproduce the lowest-occupied L1 address
the first creation saw and every allocation the creation makes reproduces, so the buffer comes back
on the same floor. `_allocate_global_cb` records the free top and the floor on the first creation,
reserves down to that recorded top on every later one, and **raises** if the buffer still does not
land on the recorded floor — because a buffer that has moved is a silent wrong-address read and an
exception is strictly better than the hang.

**One reservation is not enough, and the number says why.** On the production path the first attempt
at this reserved exactly the missing bytes in one call and the buffer came back **32 736 B high** on
both models (`logs/d1_llama_repeat_l1.log`, `logs/d2_qwen_repeat_l1.log`):

```text
RuntimeError: the recreated global circular buffer did not land on its original L1 floor:
              expected 513024, got 545760
```

`545760 − 513024 = 32736`, which is exactly the leftover of the first creation's 64 kiB headroom
after the first decode allocated 32 800 B of long-lived L1 into it. `FreeListOpt` prefers a
small exact-size free block to a large one, so a single reservation of exactly the missing 32 736 B
lands *in that gap* and moves the free top by nothing at all. The fix is to reserve **in a loop**
until the lowest occupied L1 is back at the recorded top: each pass consumes one gap, gaps are
finite, and the last pass comes out of the low region and lands the top exactly.

**That pair of failures is worth keeping for its own sake.** It is the same number, to the byte, on
two different checkpoints with different head counts and vocabularies — which is the shared-code
claim the brief asks for, made by the *failure* rather than by the fix: the placement is a property
of `Prefetcher2D` and the Galaxy L1 map, not of either model. And it is a **reported** failure where
the same tree without the guard **hung**, which is the whole point of checking.

## D-C7's remaining half: D-C13, and why it is the same shape of defect

Attempt 1 fixed D-C7 itself — the surviving `Prefetcher2DContext` reference — and measured the L1
coming back to within 192 bytes, three fresh processes, byte-identical:

```text
d11_q_two_pools_run{1,2,3}   allocated: 131520 B, free: 1261952 B, largest free block: 759488 B
```

against Milestone B's `923776 / 469696 / 373824`. But all three runs still failed, because
**1 261 952 B are free and the largest contiguous block is 759 488 B — 32 576 B short** of the
792 064 B the second model's global circular buffer needs. The bank is fragmented, not full.

Read as a layout, those three numbers say something specific. Bank size 1 393 472 B, of which
131 520 B are allocated in **two** places, because a single contiguous allocated region at the top
would leave a single free block of 1 261 952 B. Two free blocks of 759 488 B and
`1261952 − 759488 = 502464` B, split by an allocated block, is the only reading:

```text
top   1499136 ┬ allocated  (the second model's resident L1)
              ├ free       759488
              ├ allocated  (a small block, low)
              ├ free       502464
base   105664 ┴
```

So **D-C13 is the same defect as the address clash**: a small long-lived L1 buffer sitting low
enough to split the bank, allocated while something large held the space above it. That makes
`global_cb_headroom` the candidate fix for it as well as for the clash, and it makes the two-pools
test the measurement — which is why it is queued in this attempt rather than treated as a separate
investigation.

### What the two-pools test actually did once the OOM was gone: D-C14

With `release_global_cb_on_prefill` and the headroom in place, the second model's global circular
buffer **is created** - the OOM at `largest free block: 759488` does not happen. The test then hung.
`gdb`, attached before any recovery attempt
(`tttv2_milestone_c_runs/c-defects3/logs/m1b_hang_bt.txt`):

```text
#0  pthread_cond_wait
#2  tt::tt_metal::distributed::FDMeshCommandQueue::wait_for_outstanding_reads
#3  FDMeshCommandQueue::finish_nolock
#5  tt::tt_metal::distributed::Synchronize
#6  <nanobind>  ttnn.synchronize_device
```

A device completion that never returns, with **1676** `cache hit` lines in the log — 64 layers ×
~13 tensors × 2 models — so both models' weights were resident and the stall is at the second
model's first decode. Attempt 2's `e1_qwen_two_pools_l1` shows **38** at a one-layer subset, which
is 2 × 19: the same point, under an earlier version of this code. So the hang is not created by
this fix; it was **masked** by D-C13, which used to fail first.

The reduction is read out of `tt_metal`:

- the ttnn program cache belongs to the **mesh device**, not to a model, and outlives `close()`;
- its keys are op-and-config hashes, so two structurally identical Galaxy models in one process
  hash to the same keys — the second model's decode is a cache **hit** on the first model's
  programs;
- `CircularBufferImpl::set_global_circular_buffer` captures `buffer_address()` and
  `config_address()` once, and `dispatch.cpp:3035` re-sends the captured pair on every launch.

So the invariant is **per process and mesh**, not per `Prefetcher2D`. `GlobalCBPlacement` is that
record; `models/common/models/galaxy/prefetch.py` holds one per mesh device. Host-qualified
(33 passed) and **not qualified on silicon** — the mesh lost board 23 to POST_RESET before it could
be. D-C13 itself is therefore superseded rather than closed: the allocation now succeeds and what
stands behind it is D-C14.

## D-C12, recorded and not fixed — and why it is not this job's gate

Attempt 1 isolated it to one call: `logs/d11_repeat_sample_probe_run{2,3,4}.log`, two arms differing
only by `mesh_device.disable_and_clear_program_cache()`.

```text
[repeat] cache=nocache call 0..3: wrong=0/32  every call
[repeat] cache=cache    call 0: wrong=0/32
[repeat] cache=cache    call 1: wrong=32/32   slot 0: expected 1, got 0
[repeat] cache=cache    call 2: wrong=32/32   slot 0: expected 2, got 0
[repeat] cache=cache    call 3: wrong=32/32   slot 0: expected 3, got 2
```

With the program cache cleared, four consecutive sampling calls on four different inputs are all
correct; with it warm, only the first is. Three fresh processes.

**It is not in the c-defects finish condition** and it is not in the five workstreams: the D-C5/D-C8
gate asks for the five area-4 claims to be *evaluated* on silicon at three fresh processes each, and
they are — thirty runs, every one reaching its assertion. D-C12 is what makes one of those five fail
on Qwen (`a seeded slot repeats across runs`), and it is a correctness hazard for anything that
decodes more than one token, which is every real use of this stack. It is named here as the highest
open defect in the ledger after the clash, with the bisection already written down: compose the
**selector's** output on each call and see whether it is stale before `Sampling2D` is reached, which
splits a twelve-op chain in two for the price of one extra readback. The unusual operands are
`ttnn.topk(indices_tensor=self._local_indices)`, `ttnn.manual_seed`, and
`ttnn.sampling(output_tensor=tt_out_tok)`, all in
`models/common/modules/sampling/sampling_2d.py::decode_forward`.

## Attempt 3's device results, in full

Every run this attempt made, with the ones it discards and why.

### Measured, and counted as evidence

| claim | node | runs |
| --- | --- | --- |
| repeat + deterministic cleanup, Llama, 80 layers | `test_llama33_70b_galaxy_repeated_requests_and_deterministic_cleanup` | `n1` 234.34 s, `n2` 243.46 s, `n3` 300.20 s — **3/3 pass** |
| repeat + deterministic cleanup, Qwen, 64 layers | `test_qwen3_32b_galaxy_repeated_requests_and_deterministic_cleanup` | `n4` 312.51 s, `n5` 152.94 s, `n6` 148.35 s — **3/3 pass** |
| area 1, block-level cross-slot isolation, Llama | `test_llama_a_write_for_one_user_never_appears_in_another_users_blocks` | `i1b` 453.68 s, `i2b` 352.52 s, `i3b` 280.35 s — **3/3 pass** |
| area 3, chunked prefill, Llama | `test_llama_chunked_prefill_matches_a_single_uncached_prefill` | `k1b` 217.74 s, `k2b` 216.19 s, `k3b` 223.54 s — **3/3 pass** |
| mesh health | `test_partition_wh_galaxy.py` | `p0` 16.03 s, `p1` 16.34 s — 5 passed each |

### Measured, and superseded as evidence

`g1`, `g2` (one-layer subsets) and `g3`–`g8` (full shape, 6/6 pass, three fresh processes per model)
all passed, and they are **not** quoted at signoff: they measured the module before the gap-filling
pass was added. `n1`–`n6` re-run the same gate on the final module and those are the numbers that
count. Keeping both on the record is the point — the earlier six are what said the loop-form
reservation worked, which is how the config-page half was reached at all.

### Measured failures, kept because they are how the fix was found

| run | result | what it established |
| --- | --- | --- |
| `k1_llama_chunked_r1` | `expected 510624, got 510816` | the guard's lowest-address proxy is ambiguous at exactly 192 bytes |
| `k2_llama_chunked_r2` | `expected 510624, got 510816` | byte-identical in a second fresh process |
| `k3_llama_chunked_r3` | `expected [(510624,192),(510816,792064)] got [(510816,792064),(1367872,192)]` | the data buffer does **not** move; the config page moves 857 kB |
| `m1b_qwen_two_pools_r1` | rc=143 after 21 min, backtrace kept | D-C14: the stall is a never-returning device completion at the second model's first decode |

### NOT MEASURED, and discarded

Everything at or after `m1b`'s kill at 11:10:28Z, because the kill left the mesh unaddressable
(`Read 0xffffffff over PCIe ID 23`) and no reset recovered it:

| run | signature |
| --- | --- |
| `l1b_llama_seeded_slot_r1` | 1 error in 12.59 s, UMD `TopologyDiscovery` |
| `l2b_llama_seeded_slot_r2` | 1 error in 11.63 s |
| `l3b_llama_seeded_slot_r3` | 1 error in 10.75 s |
| `p2_partition_after_reset` | 5 errors in 9.81 s — this is the probe that establishes the mesh is down; the same file was 5 passed in 16.34 s at 09:46Z on this tree |

And, inherited from attempt 2 and discarded on the same grounds: `e1_qwen_two_pools_l1` (rc=124),
`f1_llama_repeat_l1_loop` and `f2_qwen_repeat_l1_loop` (1 error in ~4 s each, UMD discovery).

### Deliberately not run

Nine items were parked with `NOT A RUN` logs that say why, in each file: five two-pools runs behind
D-C14, and before that four items behind the ambiguous guard. Parking rather than deleting is so the
record shows what was skipped and on what grounds — a silent truncation would read as coverage.

### Host regression gates, after the change

| suite | result |
| --- | --- |
| `test_step7_concat32.py` | 34 passed × 3 |
| `test_step7_long_context.py` | 32 passed × 3 |
| `test_step7_paged_kv.py` | 37 passed × 3 |
| `test_step7_prefix_cache.py` | 18 passed × 3 |
| `test_step7_repeat_and_cleanup.py` | 12 passed × 3 |
| `test_step7_sampling.py` | 29 passed × 3 |
| `test_step7_token_composition.py` | 8 passed × 3 |
| `test_step7_page_table_placement_wh_galaxy.py` | **not run** — it opens a mesh, and the mesh is down |
| `test_prefetcher_2d.py` | 33 passed × 3 (22 before this workstream) |
| `models/common/tests/llm_runtime` | **1032 passed, 1 skipped** — the Milestone B baseline, unchanged |

Every step-7 number is identical to Milestone B's, in three fresh processes each. **Nothing was
relaxed and no expectation was edited.**

---

# Attempt 4, 2026-08-31 — D-C7's last 2 464 bytes

## The mesh, first, because attempt 3's closing claim no longer holds

Attempt 3 ended at 12:26Z with 25 of 32 boards unreadable, twelve failed `tt-smi -glx_reset`
attempts and the instruction not to reset again before a human looked at it. At 17:04Z, before any
device work, `tt-smi -ls` enumerates all 32, board 23 (`0000:08:00.0`) ticks at 16021 where it read
`0xFFFFFFFF`, the whole `!24`–`!31` tray ticks, and `test_partition_wh_galaxy.py` is **5 passed in
15.94 s** (`c-defects4/logs/z0_partition.log`). The sysfs node directories are stamped 17:02 and the
driver reports 2.9.0, so the host was repaired or power-cycled between the two. **No reset was run
by this attempt.**

One correction worth carrying forward: attempt 3's handoff reads the heartbeat at
`/sys/class/tenstorrent/tenstorrent!N/device/tt_heartbeat`. At driver 2.9.0 that path is the PCI
device directory and holds no `tt_*` attribute, so it returns `ERR` for every board and a live fleet
looks completely dead. The attribute is one level up:

```sh
for d in /sys/class/tenstorrent/*; do printf '%s %s\n' "$(basename $d)" "$(cat $d/tt_heartbeat)"; done
```

## The gate ledger was wrong about D-C5/D-C8, and the tree said so

Attempt 3 §11 records D-C5/D-C8 as *"MET as worded … Thirty runs, every one reaching its
assertion."* Grepping all 36 area-4 logs for `clash with L1 buffers` shows that is false for one of
the ten claim-verdicts: **Llama's `a_seeded_slot_repeats_across_runs` aborted on the L1 address clash
in all three runs** (`program 100`, L1 buffer at 544832) and never reached the sampler. Nine
combinations reached their assertions; the tenth was never evaluated. That is the brief's own
warning — the clash "hid D-C5 for Llama by killing the demo path before it ever reached the sampler"
— happening inside the evidence that was being counted as a measurement *of* the sampler.

The claim is runnable now that the clash is fixed, and it is queued as `t7`–`t9`.

## D-C7 — the last 2 464 bytes, and why they cost the gate

### What was inherited

Attempt 3's `GlobalCBPlacement` — one placement record per mesh device, shared by every owner — was
committed and host-qualified but had never run on silicon; the mesh died before it could. Getting it
onto silicon was this attempt's first act, and it works:

| | shape | outcome |
| --- | --- | --- |
| attempt 2 `e1_qwen_two_pools_l1` | 1 layer | **hung**, killed at the 29-minute bound, wedged the mesh |
| attempt 3 `m1b_qwen_two_pools_r1` | 80 layers | **hung** 21 min in `ttnn.synchronize_device`, mesh unaddressable |
| attempt 4 `s1_qwen_two_pools_sub` | 1 layer | **1 failed in 106.43 s**, clean `RuntimeError`, mesh healthy |

```text
RuntimeError: cannot restore the global circular buffer to its original L1 address:
the lowest occupied L1 is 1272480, already below the free top 1305280 the first creation had
```

A mesh-wedging hang became a diagnosable refusal. That is not the gate, but it is what made the rest
possible in one evening instead of one night.

### What was actually wrong

The refusal names two numbers and no owner, so the next run logged the **whole** resident L1 table on
entry to both creations (`logs/s3_qwen_two_pools_sub_table.log`):

| | creation 1 (2048-block pool) | creation 2 (4096-block pool) |
| --- | --- | --- |
| lowest occupied | 1370816 | **1272480** |
| blocks | 102 | 179 |
| total allocated | 128 320 B | 130 784 B |
| size histogram | `{32:94, 1088:1, 2048:1, 5440:1, 8192:2, 17408:2, 65536:1}` | `{32:171, 1088:1, 2048:1, 5440:1, 8192:2, 17408:2, 65536:1}` |

**Every block larger than 32 bytes matches in size and in count.** The entire difference is 77 extra
32-byte blocks — 2 464 B, 1.9 %. Three readings die on that table:

- it is not a larger configuration. Doubling `max_num_blocks` from 2048 to 4096 changes the L1
  footprint by **nothing**; the paged pool does not live in L1;
- it is not the 64 kB headroom leaking. There is exactly one 65 536-byte block in *each* table. It
  did not leak — it **moved**, from 1381856 to 1272480, 109 376 B down;
- it is not D-C7 as written either. The original defect was 923 776 of 1 393 472 B per bank still
  allocated after close. **That is 99.7 % gone.**

32 bytes is a semaphore, and `FreeListOpt::allocate` takes the smallest free block that fits. So 77
holes scattered through the resident region are taken in preference to the low region, and the 64 kB
block is displaced below the free top the first creation recorded — which is exactly what makes the
buffer unplaceable. **The gate was lost to fragmentation by 2 464 bytes, not to capacity.**

### What the fix is, and why it is the smallest that respects the boundary

The holder is the ttnn program cache: it belongs to the **mesh device**, outlives a model's
`close()`, and a cached program holds both its semaphores and the `buffer_address()` /
`config_address()` pair captured once at `circular_buffer.cpp:179` and re-sent by `dispatch.cpp:3035`
on every launch. One action fixes both halves — return the semaphores, and leave no program holding a
dead address.

- **`Prefetcher2DConfig.on_global_cb_released`** — a callable, default `None`, called once from
  `cleanup()` after the buffer's last reference is dropped. The module knows only that *its* buffer
  is gone. It does not know what a program cache is, or how many models share the mesh.
- **`release_galaxy_global_cb_placement(mesh_device)`** in `models/common/models/galaxy/prefetch.py`
  — `clear_program_cache()`, then forget the placement record. That file already owns the per-mesh
  record, because "one process, one mesh, several models" is a model-level fact.

The module announces; the model layer decides. Default `None` is byte-for-byte the previous path, and
`test_an_owner_with_no_release_listener_behaves_exactly_as_before` pins that. No `is_galaxy`, no
model-name branch, and nothing in the module that knows a program cache exists.

### The logs

| claim | log | result |
| --- | --- | --- |
| the refusal, before the fix | `logs/s1_qwen_two_pools_sub.log` | 1 failed, 106.43 s |
| the L1 tables that named the cause | `logs/s3_qwen_two_pools_sub_table.log` | 1 failed, 51.24 s |
| the same case after the fix | `logs/s6_qwen_two_pools_sub_afterfix.log` | **1 passed, 52.68 s** |
| host suite | — | **35 passed** (33 before); removing only the two-line announcement gives 1 failed / 34 passed |

`s6` carries the gate condition in its own log — two creations, two models, one process:

```text
[prefetcher] global circular buffer at L1 blocks [(513024, 192), (513216, 792064)]   <- model 1
[prefetcher] global circular buffer at L1 blocks [(458272, 192), (458464, 792064)]   <- model 2
```

and the mechanism in one number: lowest-occupied on entry to creation 2 went **1272480 → 1316064**.
The fragmenting blocks came back.

The second model's buffer lands at a *different* address, and that is correct rather than a
regression. The pin exists only to stop a cached program reading a moved buffer; with the cache
retired there is no such program. The guard still fires for the single-model release-and-recreate
case, which is what `v1`–`v6` re-qualify.

**Cost to note for `c-perf-paired`:** closing a model now retires the mesh's program cache, so a
process that closes and rebuilds a model pays a full recompile.

Committed as `faec6e59938`.

---

# Attempt 6, 2026-08-31 — the Llama half of D-C7 is a DRAM retention defect

## Arrival, and what was inherited as measured rather than re-measured

Attempt 4's queue was still draining when this attempt started at 20:52Z, as it had been when
attempt 5 arrived. I adopted it rather than killing it, for the same reasons: it holds the mesh
lock, it is making progress, and its remaining items are the ones the gate ledger needs.
`healthy_boards_before=32` on every run in this attempt. **Zero `tt-smi` resets.**

Attempt 5's handoff is stamped 20:10Z and two results landed after it, so its status section is
stale in the job's favour. Reconciled against the tree:

| run | attempt 5 said | the tree says |
| --- | --- | --- |
| `t6_llama_two_pools_r3` | IN FLIGHT | rc=124 at 20:51:22Z, same DRAM OOM as `t4`/`t5` |
| `w1_llama_dram_probe` | "~5 min for the answer" | ran and **failed in 46 s**, `TypeError: 'MemoryView' object is not iterable` — no measurement |

Inherited as measured, not re-run: `t0` 5 passed; `t1`–`t3` **Qwen two-pools 1 passed ×3 at the
full 80-layer shape**; `t4`/`t5` rc=124.

**Nothing was discarded on dead-mesh grounds.** `t4`–`t6` are rc=124 outer-timeout kills, but they
are not dead-mesh artifacts: each printed a complete and identical `TT_FATAL` and then hung in
teardown, which is the documented un-drainable teardown after a `TT_FATAL` in a multi-subdevice
program. `w1` opened and closed a cluster cleanly at 20:53Z and `t7` ran to 610 s after it, which
is direct evidence the mesh was never wedged.

## The defect, stated correctly

Three fresh processes, full 80-layer shape, byte-identical, `logs/t{4,5,6}_llama_two_pools_r*.log`:

```text
2026-08-31 18:08:28.943 | critical | TT_FATAL: Out of Memory: Not enough space to allocate
2297856 B DRAM buffer across 11 banks, where each bank needs to store 208896 B, but bank size is
1070773184 B (allocated: 1070239264 B, free: 533920 B, largest free block: 99712 B)
```

raised at `layer53_wqkv_ring` — the **second** model's prefetcher ring-weight pass, 66 % through,
with the first model already closed, `del`eted and `gc.collect()`ed. It is **DRAM**, not the L1
leak D-C7 is named for; attempt 4's L1 fix holds, and Qwen's `t1`–`t3` pass the identical case at
full shape.

Two explanations fitted, and they implied different fixes:

* **retention** — `close()` does not return model 1's DRAM;
* **capacity** — it does, and the second arm alone does not fit. The two-pools test is
  **asymmetric**: arm 1 is `max_seq_len=2048` with the default 2048-block pool, arm 2 is
  `max_seq_len=4096` with an explicit **4096**-block pool, so arm 2's KV is twice arm 1's and
  "model 1 fitted" does not imply "model 2 fits".

Attempt 5 assumed retention. The tree did not establish it, so this attempt measured the fork.

## The two probes that settled it

`tttv2_milestone_c_runs/c-defects6/scratch/test_llama_dram_lifecycle_probe.py` reads
`ttnn.get_memory_view(...).block_table` at every lifecycle point and records the prefetcher's
registered weights **by address**, because holding the tensors would pin exactly the references it
is looking for. Attempt 5's version died on the missing `.block_table`; this one guards every
allocator read and self-tests it before anything expensive.

### `w2_llama_dram_probe_l4` — 4 layers, 98 s

```text
0-default-2048-built     DRAM=8 918 080     L1=125 760
0-default-2048-used      DRAM=73 968 256    L1=920 640
0-default-2048-closed    DRAM=19 931 264    L1=2 432
default-2048: registered weights still allocated after close+gc: 12 of 12
  layer[0].w1 @ 2968640 · layer[0].w3 · layer[0].w2 · … · layer[3].w2
residual 650624 x12 · 696320 x8 · 731136 x4 · 278528 x4 · 232832 x4 · 208896 x4 · 186048 x4 · 384 x9
```

### `x1_llama_pool4096_only_full` — the **full 80-layer** shape, second arm only, 239 s, **1 passed**

```text
0-explicit-4096-built    DRAM=170 325 056
0-explicit-4096-used     DRAM=684 511 744
0-explicit-4096-closed   DRAM=398 617 984      (961 blocks)
explicit-4096: registered weights still allocated after close+gc: 240 of 240
residual 650624 x240 · 696320 x160 · 731136 x80 · 278528 x80 · 232832 x80 · 208896 x80 · 186048 x80 · 384 x161
```

**Capacity is refuted.** The arm the two-pools case dies on builds, prefills and decodes on its own
at the full shape in four minutes.

**Retention is measured.** 240 of 240 registered weights still allocated after `close()`, `del` and
`gc.collect()`: **398 617 984 B per DRAM bank, 37 % of the 1 070 773 184 B bank**, against an OOM
that reads `allocated: 1070239264, free: 533920`.

**The histogram names the owner.** 240 = 80 × 3 (`w1`/`w2`/`w3`); 80 each of the attention shapes,
including the 208 896 B that is exactly the size of the `wqkv_ring` buffer the OOM died on; and
**384 × 161** = two RMS norms per layer plus the one final norm.

## The fix — commit `299440bb276`

`close()` released the embedding, the LM head, the rotary setup, sampling and the column selector,
and left attention and MLP alone. `MLP2D` had **no `release` and no `close` at all**, and
`<Model>TransformerBlock2D.close()` called **only** `self.attention.close()`, which releases
intermediates and runtime tensors and never weights. A `LazyWeight` memoizes its device tensor in
`_value`, so the weights survive for as long as any caller holds the model — a runner still bound
in an enclosing frame, an executor kept for its metrics. That is what the two-pools test does, and
it is what a serving system does by construction.

| file | change |
| --- | --- |
| `modules/lazy_weight.py` | `release_device_weights(weights)`: deallocate each distinct materialized weight once, clear its memo. Dedupes on the `LazyWeight` **and** on its `_value` — `Attention2D` resolves `prefill_wqkv = resolved.prefill_wqkv or wqkv`, so two config fields are routinely one object. Collects failures, raises the first |
| `modules/rmsnorm/rmsnorm_2d.py` | `RMSNorm2D.release()` |
| `modules/mlp/mlp_2d.py` | `MLP2D.release()` |
| `modules/attention/attention_2d.py` | `Attention2D.release()`, including the optional `q_norm`/`k_norm` |
| `models/llama33_70b_galaxy/model.py`, `models/qwen3_32b_galaxy/model.py` | `<Block>.release_weights()`; `close()` calls it per layer plus the final norm |

**Why it is the smallest fix that respects the module/model boundary.** `Embedding2D.release`,
`LMHead2D.release`, `RotarySetup2D.release` and `Sampling2D.release` already exist and already do
precisely this. Attention, MLP and RMS norm were the three weight-owning 2D modules that lacked it.
No new mechanism, no new configuration, and no change to any path that does not call `close()`.

**The ordering is the model's, not the module's.** The release runs **after**
`Prefetcher2D.cleanup()`: the attention decode weights are registered with the prefetcher, and
freeing one while a prefetch can still read it is a use-after-free — which on this mesh means a
`TT_FATAL` inside a multi-subdevice program and an un-drainable teardown, the exact failure mode
`t4`–`t6` spent three hours in. For the same reason a model that **borrows** shared resources does
not release; it does not control when the prefetch stops. That leaves a borrowed-resources model's
weights resident, recorded here rather than assumed away — `owns_shared_resources` is `True` for
every model this tree builds.

Host coverage: `models/common/tests/modules/test_lazy_weight_release.py`, six cases including both
aliasing shapes and the partial-failure path. Without the primitive the file does not import.

## Area 4's last claim-verdict, evaluated for the first time

`test_llama_a_seeded_slot_repeats_across_runs` had never reached its assertion on Llama: attempt
1's three runs all aborted on the L1 address clash. With the clash fixed it now runs, 610 s, and
`grep -c 'clash with L1 buffers'` on the log is **0**.

It fails, and as a defect already on the ledger:

```text
AssertionError: a seeded stochastic decode did not repeat
observed[0] = tensor([ 2662, 5966, 28, 1566, 4354, 304, 220, … ])          token ids
observed[1] = tensor([3209869902, 3203938928, 32149, 3222092442, … ])      float32 bits as int32
```

That is **D-C12** — the second `decode_sampled` in a process returns the wrong buffer — already
qualified 3/3 on Qwen. `t7` (pre-fix) and `t9` (post-fix) agree byte-for-byte on `observed[0]` and
both find float bit patterns in `observed[1]`; the garbage itself differs between runs, which is
what a stale-buffer read looks like. **So Llama's area-4 ledger now matches Qwen's claim for
claim, and the weight-release fix moved none of it.**

`t8` is discarded: it started while this attempt had a half-applied edit in the working tree and
died in 0.87 s on `AttributeError: 'function' object has no attribute '__mro__'`. That is a
mistake of process, not a result, and it is recorded in the handoff as one.

## Two of `c-exec-llama`'s three handed-over defects are not shared-Galaxy defects

* **`chunk_start must be non-negative and aligned to chunk_alignment`** (`attention_2d.py:860`).
  Measured (`exec_llama/logs/f_warmup_pf_r1.log:1743`): `chunk_start=32`, `sequence_length=128`,
  Galaxy `chunk_alignment=128` — it is the flash-SDPA `q_chunk_size`/`k_chunk_size`
  (`recipes.py:651`). The value comes from `llm_runtime/prefill/plan.py:381`,
  `absolute_start = num_cached_tokens + relative_start`, with `num_cached_tokens` set at
  **`llm_runtime/warmup.py:700`** as `cached_tokens=layout.block_size` — 32. The common runtime's
  default warmup plan assumes any block-aligned prefix is a valid chunk start; on Galaxy the block
  size is 32 and the chunk alignment is 128. The module validator and the shared recipe are both
  correct. **This is a runtime defect and this brief forbids changing `llm_runtime`** — "if you
  believe otherwise, write the reduction and stop", so this is the reduction. Lowering Galaxy's
  `chunk_alignment` to 32 would be relaxing a constraint to turn a failure green and was not done.
* **`page_table width cannot address the required KV capacity`** (`attention_2d.py:714`). Measured
  (`exec_llama/logs/f_shrink_r1.log:672`): staged table `(1, 128)` against
  `PagedKVMetadata(block_size=32, max_num_blocks=95)`, so the failing clause is
  `shape[1] > meta.max_num_blocks`, 128 > 95. The pool was shrunk and the previously staged wider
  table reused. The caller owes a restage after `configure_paged_attention`. **An executor defect.**
* **`test_reference_prefill_and_decode` at 2048 → non-finite decode logits.** This one *is* shared
  code (`GalaxyDirectRunner`). Recorded OPEN; it is not in this brief's finish condition.

## The clash evidence `c-exec-llama` handed over predates the fix

`git merge-base --is-ancestor` puts every one of `c-exec-llama`'s commits **before** the clash fix:
it exited at 2026-08-30T00:02Z at `2b463f17fcd`, while `32e552bb0b2` landed 2026-08-31 11:32 and
`faec6e59938` at 17:31. So its "prefill after a decode" reproduction, its three addresses
(543488, 542016, 544832) and its "the clash blocks serving" conclusion are all pre-fix, and nobody
has re-asked them at HEAD. `q12.txt` queues that question: three fresh processes each of
`test_executor_warmup_and_program_identity[decode_first]` and
`test_executor_repeated_startup_and_cleanup`, ~110–190 s apiece.

---

# Attempt 7, 2026-09-01 — the five workstreams, closed out

**Base commit:** `671802f946482360c31c220f4cfbf704c7969334`. **Arrived:** 07:07Z, on an idle
mesh with 32/32 boards on the bus and attempt 6's 53-run chain fully drained. **Zero `tt-smi`
resets run by this attempt.**

## What landed after attempt 6 stopped writing, and what it changes

Attempt 6's handoff is stamped 01:14Z; its chain drained at 01:36Z. Five runs landed in
between, and one of them answers the question that attempt 6 left open.

| run | attempt 6 said | measured |
| --- | --- | --- |
| `y2`, `y3` | queued | **1 failed** ×2, 219.14 / 197.25 s, same `chunk_start` line |
| `y4`, `y5`, `y6` | queued | **1 passed ×3**, 199.68 / 199.28 / 199.83 s, **0 clash lines each** |

`y4`–`y6` are `test_executor_repeated_startup_and_cleanup` — three startup/serve/cleanup cycles
in one process. At `2b463f17fcd` that node failed **3/3** on
`TT_THROW … Statically allocated circular buffers … clash with L1 buffers … L1 buffer allocated
at 542016`. At HEAD it **passes 3/3 in three fresh processes**, within half a second of each
other, with zero clash lines. All three logs carry
`# commit=671802f946482360c31c220f4cfbf704c7969334` in their header.

**So the claim that the L1 address clash blocks serving is a pre-fix claim, and it is now
refuted on silicon at HEAD.** `c-exec-llama` measured it at `2b463f17fcd` and exited at
2026-08-30T00:02Z; the clash fix landed at `32e552bb0b2` on 2026-08-31 at 11:32.

## Why attempt 6's device results are results about HEAD

Every commit after the last production-code change is evidence, documentation or a test
addition:

```
git log --oneline --name-only 299440bb276..HEAD
  671802f9464  evidence only
  f61978825cd  evidence + status files only
  874a0e9da75  models/common/modules/README.md (docs) + evidence
  d2d6c424030  the two step-7 coverage TEST files (adds the close-contract test)
```

No file under `models/common/modules/` or `models/common/models/` other than a README changed
after `299440bb276`. The logs bear this out directly: the D-C7 gate runs carry
`# commit=d2d6c424030` / `874a0e9da75` and the regression runs carry `f61978825cd`, not
`299440bb276` as attempt 6's prose says. The prose is loose and the conclusion is unaffected —
they are all descendants of the fix with identical production code, and they are *closer* to
HEAD than the handoff claims, not further.

## The two gates that need no silicon

```
git diff --stat <milestone-b>..HEAD -- '*_1d.py'                          -> empty
git diff --stat <milestone-b>..HEAD -- 'models/common/llm_runtime/'       -> empty
git status --porcelain models/                                            -> empty
```

**Zero changes to any `*_1d.py`. Zero changes under `models/common/llm_runtime/`.**

---

# Attempt 9, 2026-09-01 — the five workstreams, verified and closed

**Base commit:** `4292d26e47faa07eb9679b001bcf99b45ed14b1d`. **Arrived:** 09:34Z, to a mesh
**busy with this job's own work**: attempt 8's queue `q16` was still running (PID 228702, adopted
by the driver when attempt 8 exited at 09:21:56Z), inside
`zl7_llama_per_slot_controls_r1`. Nothing was killed and nothing was queued on top of it.
**Zero `tt-smi` resets this attempt.**

## The driver's standing clash warning is a pre-fix account, and the tree says so

The job prompt opens with "THE LLAMA L1 ADDRESS CLASH HAS MOVED ON SINCE YOUR LAST ATTEMPT",
citing `c1_completion_handoff.md`: the trigger is *a prefill after a decode in the same process*,
it reproduces in ~110 s, and — the part that matters — *it blocks serving*, so `c-trace` and
`c-perf-paired` are behind it.

**Every part of that was measured before the fix.** `c-exec-llama` ran 2026-08-29 22:50Z to
2026-08-30 00:02Z at `2b463f17fcd`; the two clash fixes are `32e552bb0b2` (2026-08-31 11:32) and
`faec6e59938` (17:31). The prefill-after-decode shape has since been re-asked at HEAD, and it does
not reproduce:

| run | node | verdict | `grep -c 'clash with L1 buffers'` | log `# commit=` |
| --- | --- | --- | --- | --- |
| `y4_exec_repeat_cycles_r1` | `test_executor_repeated_startup_and_cleanup` | `1 passed` 199.68 s | **0** | `671802f94648` |
| `y5_exec_repeat_cycles_r2` | same | `1 passed` 199.28 s | **0** | `671802f94648` |
| `y6_exec_repeat_cycles_r3` | same | `1 passed` 199.83 s | **0** | `671802f94648` |
| `y1`–`y3` | `test_executor_warmup_and_program_identity[decode_first]` | `1 failed` ×3 | **0** | `671802f94648` |

`y4`–`y6` are three full startup/serve/cleanup cycles in one process — so prefill-after-decode
twice per run — and they pass within half a second of each other. `y1`–`y3` are literally
`warmup_model_decode` then `warmup_model_prefill`, the 110-second reproduction the handoff names,
and they fail on **D-C16** (`chunk_start` alignment, host-side, raised before any device work) with
zero clash lines.

So no silicon was spent re-deriving the clash this attempt. What `c-exec-llama` handed over that
*is* still live is D-C16 and the two items below.

## Area 4 in one table, at one production tree, on both models

The D-C5/D-C8 gate line is "both models' device sampling runs end to end, and area 4's five claims
… are evaluated on silicon, three fresh processes each". Re-derived from the logs themselves —
each log's own `# commit=` header, its pytest summary line, `grep -c 'clash with L1 buffers'`,
`grep -cE 'TT_FATAL|TT_THROW'`, and `grep -c SKIPPED`:

| claim | Qwen | Llama |
| --- | --- | --- |
| device greedy == host argmax | `zd1`–`zd3` **1 failed ×3**, slot `[4]` | `zd4`–`zd6` **1 passed ×3** |
| no padded vocabulary id ever sampled (3 policies) | `zq1`–`zq3` **3 passed ×3** | `zl1`–`zl3` **3 passed ×3** |
| `T = 0.02` collapses onto the host argmax | `zq4`–`zq6` **1 failed ×3**, slots `[4, 21]` | `zl4`–`zl6` **1 failed ×3**, slots `[2, 11]` |
| a seeded slot repeats across runs | `zq10`–`zq12` **1 failed ×3** (D-C12) | `ze1`–`ze3` **1 failed ×3** (D-C12) |
| per-slot heterogeneous controls | `zq7`–`zq9` **1 passed ×3** | `zl7`–`zl9` **1 passed** ×3 |

**Thirty runs. Zero `TT_FATAL`, zero `TT_THROW`, zero `clash with L1 buffers`, zero `SKIPPED`.**
Every run reaches its assertion, which is the thing D-C5 and D-C8 used to prevent: the selector
matmul no longer refuses a width-sharded `in1` and no longer resolves a grid outside the loaded
decode sub-device. The commits behind the thirty runs are `f61978825cda`, `671802f94648` and
`4292d26e47fa` — three commits, one production tree (`git diff` over `models/` between
`299440bb276` and HEAD touches one README and two test files and nothing else).

Six of the ten verdicts pass and four fail. The four failures are **two** defects, neither of them
D-C5 or D-C8, and both reported as failures rather than relaxed:

* the greedy and `T = 0.02` residuals, which are bfloat16 ties — see below;
* the two seeded-slot failures, which are D-C12.

## The near-zero-temperature residual is a measured tie on Qwen, from evidence already on disk

The report's own earlier entry called the tie story "a hypothesis with an arithmetic behind it,
not a measurement", and said the measurement that would settle it is the top-two gap at exactly
the missed slots. **That measurement was already taken and nobody joined it to the gate result.**

`logs/d11_greedy_tie_probe.log` runs the gate case's own `_load`, `_paged_config`,
`_distinct_rows`, the same `tokens = [1] * 32`, the same `positions = [128] * 32` and the gate's
exact `T = 0.02` policy, and prints the top-two gap per slot on the composed float32 logits:

```text
[cold] slot  4: gap=0     p(runner-up)@temp50=0.5
[cold] slot 12: gap=0     p(runner-up)@temp50=0.5
[cold] slot 21: gap=0     p(runner-up)@temp50=0.5
[cold] the-32-smallest-gaps=[4, 12, 21, 10, 28, 5, 14, 3, ...]
```

**Exactly three Qwen slots have a top-two gap of zero — two ids attaining the row maximum in
bfloat16 — and they are slots 4, 12 and 21.** The gate misses slots `[4, 21]`, byte-identically in
three fresh processes. `torch.argmax` breaks a zero gap by lowest index; a sampler drawing from a
softmax has a 50 % chance either way, so two of the three zero-gap slots disagreeing and one
agreeing is the expected outcome and there is no fourth slot to explain. The greedy claim's single
residual is the same slot 4, and the `[tie]` half of the same log measures it directly: `host 16 @
15.375, device 17 @ 15.375, equal=True, gap=0.0, ids sharing the row maximum=2`.

**So on Qwen the residual is the draw meeting the numeric floor of bfloat16 logits, measured, not
inferred.** The claim as *worded* — 32/32 — is still not met, and nothing was relaxed to make it
met. Llama's `[2, 11]` gaps have never been measured; `tie_llama_r1`–`r3` in `q17` measure them.

### And a correction the same log forces: the `T = 2.0` half of that test proves nothing

This report earlier read "D4 is confirmed twice" from the pairing `T = 0.02` agrees 30/32 while
`T = 2.0` agrees 0/32. **Look at the order of calls in the test** (`test_step7_coverage_wh_galaxy.py`
`..._near_zero_temperature...`): `decode_logits` (no sampling), then `cold = decode_sampled(T=0.02)`,
then `hot = decode_sampled(T=2.0)`. `hot` is the **second device sampling call in the process** —
which is exactly what D-C12 corrupts. `d11_greedy_tie_probe.log` shows the same structure failing
the same way: its `[tie]` half (first sampling call) is sane, and its `[cold]` half (second call)
reports `missed=True` in **all 32 slots** with device ids like `3212836881` and `1077395535` —
float32 bit patterns, not tokens.

`T = 2.0`'s 0/32 is therefore not evidence about the reciprocal-temperature convention. **D4 is
still confirmed, on the `T = 0.02` direction alone**, and that direction is sufficient: 30 of 32
slots landing on the argmax cannot happen under the inverted convention, which flattens the
distribution over 32 candidates and would make even 30 agreements a ~`1/32**30` event. The
two-directional reading is withdrawn; the one-directional one stands.

## D-C12: the received explanation has a hole in it, and this repo's own linters widen it

D-C12 is qualified — only the **first** device sampling call in a warm-cache process is correct;
four consecutive calls on four different inputs are all correct with
`disable_and_clear_program_cache()` before each, three fresh processes
(`logs/d11_repeat_sample_probe_run{2,3,4}.log`). The received explanation is a program-cache
runtime-argument defect: on a cache hit ttnn rewrites runtime args only, so an op whose new input
**address** never reaches them reads the previous call's buffer.

**That mechanism needs an address that moves, and in that probe none should.** The only thing
differing between the four calls is the contents of one host→device write; every allocation is made
and freed in the same order in every call, so the free list returns to the same state and every
intermediate should land at the same address. It also does not explain why the *logits* readback
(`compose_galaxy_logits`, the same `ttnn.to_torch`) is correct on repeated calls while the *sampled
token* readback is not.

Two negative results, both cheap and both on the host:

1. **The three ttnn factories in the chain carry no baked addresses.** `reduction/topk`,
   `reduction/sampling` and `reduction/manual_seed` contain zero `address()` calls between them;
   every buffer reaches the kernels through `emplace_runtime_args` as a `MeshTensor`, which
   `tt_metal/api/tt-metalium/program_descriptors.hpp:110-125` documents as exactly the declaration
   the framework patches on a cache hit.
2. **This repo already lints for that bug class, and it does not fire anywhere in the chain.**
   `.pre-commit-config.yaml` carries `detect_smuggled_rta.py` ("raw buffer address pushed into
   runtime args") and `detect_override_rebuild.py`. Run over **all 2 993** `ttnn/**/device/*.{cpp,hpp}`
   files they report five and three sites respectively, and **not one is in the sampling chain**:

   ```text
   smuggled RTA: ccl/all_to_all_dispatch, sliding_window/halo (x4)
   override rebuild: ccl/mesh_partition, data_movement/roll (x2)
   ```

**The mechanism that fits without a moved address is a premature readback.** Call 0 has to compile
its programs, which stalls the host long enough for the device to finish; on a cache hit there is
no compile, the readback races the still-running program, and at a stable address it returns
exactly what the buffer held before — the previous call's answer. It also explains the model tests'
float32 bit patterns, where the same buffer previously held logits, and it explains why only the
sampling readback is affected if the last writer's cores are not what the read synchronizes
against: decode runs under `set_sub_device_stall_group([SubDeviceId(1)])`, and `ttnn.sampling` is
the one op in the chain placed by `Sampling2D._sampling_core_grid()` rather than by the recipe's
worker grid.

`tttv2_dc12_scratch/test_dc12_op_bisect.py` (written this attempt, diagnostic only, never
committed) asks both questions in one ~20 s arm: every intermediate's **buffer address** and
device-0 signature per call, and per call **three reads of the same output** — `read1` immediately
as production code does, `read2` after `ttnn.synchronize_device`, `read3` after
`reset_sub_device_stall_group()` and a synchronize. `read1 != read2` is direct proof of a race and
names a one-line fix in *our* code (`collectives.compose_galaxy_sampled_tokens`). `read1 == read2
!= read3` says the same and that the decode stall group excludes the sampling cores — also ours.
All three equal and stale sends it back to the addresses in the per-op trace.

## D-C6's Qwen residual, read once more off `concat_tie_probe.log` — bounded, and not a tie

`D-C6.status` records the tie hypothesis as "tested and **refuted**", quoting the probe's own last
line. That line is right and the reading it invites is too narrow. The probe's full table
(`logs/concat_tie_probe.log`, Qwen, sequential vs concat-32 in one process) says three things:

```text
[concat] slot 11: seq id=18    gap=0.125  bat id=16     gap=0.125  differs=True  max|seq-bat|=0.75
[concat] slot 13: seq id=13806 gap=0.375  bat id=88856  gap=0      differs=True  max|seq-bat|=1.5
[concat] diverged=[11, 13]
[concat] widest sequential top-two gap among diverged slots=0.375
[concat] every diverged slot is an exact tie: False
```

1. **The two diverged slots are the two slots with the smallest top-two gaps** — 0.125 and 0.375,
   i.e. **one and three** bfloat16 ulps at magnitude 15. No slot with a wide gap diverges.
2. **The two paths differ by more than that everywhere.** `max|seq-bat|` runs from 0.656 to 2.281
   across all 32 slots — 5 to 18 ulps. So the accumulation-order difference between concat-32 and
   sequential prefill is *larger than the top-two gap* at exactly the slots that diverge, and
   smaller than it at the other thirty.
3. **So the residual is bounded and explained, and it is not a cross-row contamination.** It is
   argmax being a discontinuous function of a quantity the two paths agree on only to ~1–2 logit
   units. Llama's 480 slot-comparisons (32 slots × 5 lengths × 3 processes) never hit a gap that
   small, which is why the same recipe passes there.

**This does not make the claim pass and nothing was relaxed.** `test_qwen_concat32_matches_
sequential_prefill_at_each_length` compares argmax, it fails 3/3 at 128 and 3/3 at 256, and that is
the recorded result. What the measurement adds is *what a fix would have to be*: either a
tighter-than-bfloat16 accumulation in the concat-32 reduction, or a discriminator that bounds the
logit difference instead of matching an argmax. The second is a test change that would weaken the
claim, so it is not this job's to make; it is a question for whoever owns the gate wording, and it
is now on record with the numbers behind it. **D-C6 stays `DEFERRED`.**

---

# Attempt 10, 2026-09-01 — the gate ledger re-derived, and D-C17

## Arrival, and what was inherited as measured rather than re-measured

At 10:32Z the mesh held exactly one thing: this job's own queue `q16` (PID 228702, started by
attempt 7, adopted by the driver across attempts 8 and 9), on `zr2`. All 32 boards on the bus, zero
`tt-smi` resets. Nothing was discarded on dead-mesh grounds: every `RESULTS.md` row from 07:22Z to
10:41Z carries a pytest summary, and there is no `rc=124`-then-seconds-long-failure tail anywhere in
the range.

Attempt 9's handoff was stamped 10:02Z and listed four queue nodes as IN FLIGHT. Three of the four
had landed by the time I read it, so its status table was stale in the job's own favour — but in
this case the news is good rather than bad:

| attempt 9 said | the tree says |
| --- | --- |
| `zm1`-`zm6` IN FLIGHT | all six **`1 passed`**: cross-slot 309.40/333.24/439.24 s, chunked prefill 303.64/228.93/249.68 s |
| `zr1`-`zr3` IN FLIGHT | all three **`1 failed`**, 170.90/165.38/168.46 s — and see D-C17 below, because they are not device measurements at all |
| `u4`-`u6` IN FLIGHT | `u4`/`u5` **`3 passed`** 12.24/12.00 s |
| `zp1`-`zp6` IN FLIGHT | still queued |

Attempt 9 recorded the gate line "the three claims the clash blocked are measured" as MET partly on
runs at older commits plus `zm1` alone. **It is now met with all six `zm` runs at HEAD** — a
stronger record than the one it claimed. Nothing it wrote has been contradicted by the tree.

## The eight gates, re-derived from the logs rather than from any status page

I rebuilt the whole ledger from the log files: each log's own `# commit=` and `# node=` headers, its
final pytest summary, and counts of `clash with L1 buffers`, `SKIPPED` and `TT_FATAL|TT_THROW`. No
silicon was spent producing this table. Merge base used throughout:
`6af44349413ca6ce2c0d98f5b26dd2898dc1f067`.

Two things this re-derivation established that were not on record:

**1. The step-7 host-suite gate is stronger than it looked, and the reason is a commit range.** The
three fresh-process passes are `z3_*_p1` (at `299440bb276`) and `zh_*_p2`/`zh_*_p3` (at
`f61978825cda`), which are two different commits — normally a reason to distrust the set.
`git diff --name-only 299440bb276..HEAD` answers it: over that whole range the only file under
`models/` that is not a test file is `modules/README.md`. **Production code is byte-identical from
`299440bb276` through HEAD**, so all three passes are at HEAD's production code and the set is one
qualification, not three fragments. The two device test files that do change in the range
(`test_step7_coverage_wh_galaxy.py`, both models) gain one new function each and modify none.

**2. The brief's "162 tests at Milestone B" figure is the thing that is off, not an expectation.**
`git diff 6af44349413..HEAD -- 'models/common/tests/models/galaxy/test_step7_*.py'` is **empty** —
the seven files are byte-identical to Milestone B — and they collect 34+32+37+18+12+29+8 = **170**.
A count cannot have drifted in files that did not change, so the discrepancy is in the brief.

## D-C17 — `GalaxyDirectRunner` decodes at a position past `max_seq_len` and returns garbage instead of refusing

This is `c-exec-llama`'s third handed-over defect — "`test_reference_prefill_and_decode` at 2048
returns non-finite decode logits, in the REFERENCE path, `GalaxyDirectRunner`". It is shared Galaxy
code and therefore this job's. It reduces to two findings, and the first one has to come first.

### The runs that were supposed to measure it do not touch the device

`zr1`, `zr2`, `zr3` are three fresh processes, `1 failed` each, within 5.5 s of one another. Read as
a device result that is a textbook deterministic defect. It is not a device result.
`_reference_prefill` in `test_executor_wh_galaxy.py` **caches to disk** and returns the cached
tensor unless `LLAMA33_70B_GALAXY_EXECUTOR_REFERENCE=recompute`:

```
$ grep '\[reference\]' logs/zr{1,2,3}_ref2048_r*.log
[reference] loading tttv2_milestone_c_evidence/exec_llama/reference/llama_prefill2048_layers0.pt
```

— the same line in all three. That file was written **2026-08-30 00:38:36** by `c-exec-llama` at
around `2b463f17fcd`, which is *before* `32e552bb0b2` (global-CB address), `faec6e59938` (program
cache outliving the model), `299440bb276` (DRAM weight release) and `60823a3888f` (the concat-32
recipe). `md5 5a59f74f3ddd88121fe86234d706986c`. So the three "fresh processes" are **one
`torch.load` of one stale file, three times**, and they say nothing whatever about this tree. The
three inherited artifacts are preserved as `*.as-inherited-20260830.pt` before anything overwrites
them, and `zs1`-`zs3` (2048) and `zs4`-`zs6` (512, the control) are queued with
`LLAMA33_70B_GALAXY_EXECUTOR_REFERENCE=recompute` to take the measurement for real.

**This generalises beyond one node, and `c-signoff` should know.** Every executor-vs-reference
comparison in that file — prefill PCC, decode, KV PCC — is against artifacts computed on
2026-08-30 at a commit four fixes back, with no commit stamp inside the file and the file untracked
by git. A comparison whose reference is a stale undated artifact can go green for the wrong reason
as easily as red.

### What is actually in the artifact, and what asks for it

Read on the host, no device (`python -c 'torch.load(...)'`):

| tensor | at 128 | at 512 | at 2048 |
| --- | --- | --- | --- |
| `prefill_logits` | finite, `[-13.81, 31.13]` | finite, `[-9.75, 24.13]` | finite, `[-6.94, 17.88]` |
| `kv_first_k/v`, `kv_last_k/v` | finite, sane | finite, sane | **finite, sane** (`k` `[-13.75, 14.25]`, `v` `[-3.47, 2.44]`) |
| `decode_logits` | finite, `[-19.50, 18.00]` | finite, `[-19.50, 19.50]` | **garbage: 128 233 of 128 256 columns exceed 1e3; 448 entries are ±inf; finite max 5.65e19** |

So at 2048 the prefill is right and the KV it wrote is right, and the *first decode step* is wrong —
in **all 32 rows**, with rows 1..31 byte-identical to each other and row 0 different. Every value
has its low 16 float32 bits zero (bfloat16 promoted, as expected) and the magnitudes cluster at
`k · 2^63` for small `k`, which is a wrecked exponent rather than a numerical drift.

`_reference_prefill` decodes at `positions[0] = length`. The test's constants are
`_MAX_SEQ_LEN = 2048`, `_BLOCK_SIZE = 32`, so `blocks_per_user = 64` and the page table is
`[32, 64]`. At `length == 2048` the decode position **equals** `max_seq_len`: the last addressable
position is 2047, and block index `2048 // 32 = 64` is column 64 of a 64-wide page table. At 128 and
512 there are 60 and 48 spare blocks and nothing is out of range — which is exactly the pattern the
table above shows.

### The defect, stated as a property of the shared runner

`GalaxyDirectRunner.generate` guards this condition — `direct_runner.py:645`,
`if max(positions) >= self.max_seq_len: break`. **`decode_logits`, `decode_sampled` and
`_stage_positions` do not.** `_decode_device_logits` validates `len(tokens)` (line 552) and
`_stage_positions` validates `len(positions)` (line 332); neither validates a position *value*. An
out-of-range position is staged, `prepare_decode_rot_mats` builds rotary matrices for it, and the
paged attention indexes past the page-table row. **A caller that decodes at
`position >= max_seq_len` gets garbage logits instead of an exception.** A serving system reaching
its context limit does exactly this.

The smallest fix that respects the boundary is one check in `_stage_positions`, beside the length
check it already owns, in the same message style — the bound is already a `GalaxyDirectRunner`
attribute (`self.max_seq_len`) and `generate` already encodes the comparison, so nothing new needs
to be known by anything.

### Why attempt 10 reduced it and did not commit the check

Because of what it would cost, and the cost is not the fix. This job's eight gates are all qualified
at production code that is byte-identical from `299440bb276` to HEAD; that identity is what makes
the step-7 host set above one qualification instead of three fragments, and it is what lets thirty
area-4 runs taken across `f61978825cda` and `671802f94648` be read as one table. Committing any
production change moves HEAD off that tree, and the brief requires every fix to be re-qualified on
**both** models — which for these gates is roughly six device-hours on top of a queue tail that is
already several hours deep, at an attempt whose one outstanding instruction is to finish and be read.

**And there is a second reason, which is the honest one.** The fix converts a silent wrong answer
into a refusal. `test_reference_prefill_and_decode[2048]` would then fail with `ValueError:
position 2048 is not addressable in a 2048-token cache` instead of an `isfinite` assertion — still
red, because the test asks for a position that does not exist. Making it green means changing
`positions[0] = length` to `length - 1` in `test_executor_wh_galaxy.py`, and that file is
`c-exec-llama`'s. Editing another job's test to turn its failure green is precisely what the house
rules forbid.

So D-C17 is recorded `OPEN — REDUCED, NOT FIXED`, with the mechanism, the line numbers, the
one-check fix, and both owners named: the `direct_runner` check belongs to whoever owns
`models/common/models/galaxy/direct_runner.py`, and the `positions[0] = length` call belongs to
`c-exec-llama`. `zs1`-`zs6` will say whether the garbage reproduces on silicon at HEAD; the
reduction stands either way, because the missing bound is visible in the source.

---

# Attempt 10, part 2 — what the two drained queues said

`q16` drained at 12:41Z (attempt 7's queue, adopted across attempts 8, 9 and 10). `q17` was
launched at 12:42Z after a full preflight and drained at 13:32:14Z. **Both are read in full and
nothing is in flight.** Mesh: 32/32 boards, zero `tt-smi` resets this attempt.

## Area 2's real question is asked on both models, and one parametrization of it is degenerate

The brief's section 4 says that once D-C6's L1 overflow is fixed, "area 2's real question becomes
askable for the first time: do padded rows change an active row's logits at active batch 16, 31 and
32?" It had **zero** runs in `RESULTS.md` on either model — Milestone B's attempts all died inside
`validate_circular_buffer_region` first. Answered now:

| | active16 | active31 | active32 |
| --- | --- | --- | --- |
| Qwen (`zp1`/`zp2`/`zp3`, 590.75 / 730.40 / 508.97 s) | PASS x3 | PASS x3 | PASS x3 |
| Llama (`zp4`/`zp5`/`zp6`, 1047.37 / 1008.67 / 820.67 s) | PASS x3 | PASS x3 | **FAIL / pass / pass** |

All six logs: 0 clash lines, 0 `TT_FATAL`/`TT_THROW`, 0 `SKIPPED`, **0
`validate_circular_buffer_region`** — which independently confirms D-C6's overflow is gone on a
node that could not survive that call at Milestone B.

**`active=32` is a degenerate level and its failure is not about padding.**
`GALAXY_PHYSICAL_BATCH - active` is zero there, so no filler rows are appended and `padded` is
`list(rows)` in *both* loop iterations. The two invocations receive **byte-identical input**. The
assertion message — "active slots [10] moved when only the padding rows changed" — is therefore
misleading: no padding row changed, or existed. What that level actually asserts is that
`prefill_batched` returns bit-identical logits when called twice with the same input.

So area 2's question, at the two levels that genuinely test it, **passes on both models, three
fresh processes, bit-exactly** — and `zp4` is a separate finding, D-C18.

## D-C18: one real observation of non-reproducibility in 27 comparisons

`tttv2_dc18_scratch/test_dc18_concat32_repeat_probe.py` (diagnostic only, never committed) does
one model load and **four** invocations, then reports all **six** pairwise comparisons with
per-slot `max|diff|` — six comparisons per model load where the committed test gives one per two
processes. Three arms at `active32` and one control at `active16`: **`0 of 6 pairs differ` in every
arm**, `max_abs_diff=0` on every pair, `argmax0 = 220` in all sixteen invocations.

Tally on byte-identical input: **27 comparisons, one differing** (`zp4`, slot 10). Both of these
are true and neither alone is: *it happened*, and *it did not reproduce in 26 further comparisons
across four fresh processes*. It is not qualified as a defect by this brief's three-identical-runs
standard, and it is not dismissible as noise either, because the house rules are explicit that a
case which flips across fresh processes is a defect. Recorded at exactly that strength.

**The probe did not reproduce `zp4`'s process state, and that is the next lever.** In the committed
test all three parametrizations share one pytest process, so `active32` runs *third*, after two
complete `_load`/prefill/`_close` cycles; the probe runs it alone in a fresh process. `zp5` and
`zp6` did run it third and were clean, so two prior cycles are not sufficient — but they are not
ruled out as necessary. The cheapest next experiment is two `_load`/`_close` cycles before the four
measured invocations: zp4's state at six comparisons per run instead of one, ~6 minutes an arm.

## D-C12: the two standing hypotheses are both refuted, on silicon, three fresh processes

Full account in `D-C12.status`. The short form, because it changes what anyone should look at next:

* **It still reproduces at HEAD** — never re-asked since `dc7f62430c0`, which predates the
  program-cache retirement and the weight release. `dc12_repeat_head_r{1,2,3}`, 17.89/18.06/17.85 s,
  **byte-identical over every `[repeat]` line** (md5 `42d42591e9d7becf580287d3143b94f9`). Cache
  warm: call 0 right, calls 1–3 wrong 32/32, every returned id **in vocabulary** — a stale
  *answer*, not garbage. Cache cleared per call: all four right.
* **Not a premature readback.** The bisect reads the same output three times per call — immediately
  as production does, after `ttnn.synchronize_device`, and after `reset_sub_device_stall_group()`
  plus a synchronize. `read1==read2` and `read2==read3` in **twelve of twelve** observations. A
  race shows `read1 != read2`. **Attempt 9's §9 hypothesis is withdrawn.**
* **Not a moved address.** Ten of the thirteen ops' output addresses are identical between call 0
  and calls 1–3; the three that do move (`sharded_to_interleaved`, `matmul`, `add`) move
  *identically in the mode that produces correct results*, so movement is not the discriminator.
  This also confirms attempt 9's source reading that `reduction/{topk,sampling,manual_seed}` pass
  no bare address.
* **What discriminates is a host readback between the ops.** `mode=cache_sig` is `mode=cache` plus
  one read of each intermediate and nothing else — same cache, same addresses — and it is correct
  in all four calls, 3/3 processes. **Observing the chain fixes it**, which is the signature of a
  dependency lost on a cache hit rather than a stale read.
* **The lag is not fixed.** `r1` and `r2` are byte-identical over every `[bisect]` line; `r3`
  differs in exactly one cell — call 3 returned call 2's answer where `r1`/`r2` returned call 1's.
  A single stale buffer lags by exactly one, always.
* One pointer handed on and deliberately not interpreted: `manual_seed` and `sampling` report the
  **same** output address (2990304) in every mode and call.

## The near-zero-temperature residual is an exact bfloat16 tie on BOTH models

Attempt 9 measured this on Qwen and left Llama's `[2, 11]` unmeasured, naming it as the thing to
take. `tie_llama_r{1,2,3}` took it — three fresh processes, the gap column **byte-identical** in all
three, `agreed=32/32` on the first-call greedy half in all three (which independently reproduces
`zd4`–`zd6`'s pass):

| model | slots the gate misses, 3/3 | slots whose top-two host gap is **exactly 0** | missed ⊆ tied? |
| --- | --- | --- | --- |
| Qwen | `[4, 21]` | `{4, 12, 21}` | **yes** |
| Llama | `[2, 11]` | `{2, 7, 8, 11, 12, 18}` | **yes** |

**Every slot the near-zero-temperature gate misses, on either model, is a slot where two vocabulary
ids attain the row maximum exactly in bfloat16.** No missed slot has a non-zero gap, on either
model. `torch.argmax` breaks a zero gap by lowest index; a sampler is under no such obligation. If
misses were spread at random over 32 slots, landing on tied slots twice out of two is
`(6/32)(5/31) ≈ 0.03` on Llama and `(3/32)(2/31) ≈ 0.006` on Qwen — about `2e-4` jointly.

So the residual on this claim is fully explained on both models, and what a fix would have to be is
the same shape as D-C6's: either a tie-break convention the device sampler is required to match, or
a discriminator that bounds the logit difference instead of matching an argmax. The second weakens
the claim, so it belongs to whoever owns the gate wording. **Nothing was relaxed; `zl4`–`zl6` and
`zq4`–`zq6` stay recorded as failures.**

**A caveat that has to travel with this probe.** Its `[cold]` half — the `T=0.02` measurement — is
the *second* device sampling call in the process, and on Llama it reports `missed=True` in all 32
slots with device ids like `3212163627` and `1066581928`, which are float32 bit patterns rather
than tokens. That is D-C12, exactly as attempt 9 found on Qwen. **The `[cold]` verdict column is
therefore worthless on both models and only the `gap` column — computed from composed host logits —
carries the measurement above.** The gate's own missed-slot lists come from `zl4`–`zl6` and
`zq4`–`zq6`, not from this probe.
