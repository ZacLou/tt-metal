# `c-exec-llama` — `Llama33_70BGalaxyExecutor`, evidence report

**Job:** `c-exec-llama`, attempt 1. **Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`.
**Base commit:** `67a208db961`. **Device:** exclusive WH Galaxy `(8, 4)`, 32 boards.
**`HF_HOME`:** `/localdev/ctr-apbernal/hf_data` for every run; each log's header records it.
**Last updated:** 2026-08-29T23:47Z — IN FLIGHT.

Run-by-run index, machine-written as runs land: [`RESULTS.md`](RESULTS.md). Every log is under
[`logs/`](logs/) and no log is ever overwritten.

*observed* means one run. *qualified* means three fresh processes with the same result, which is the
brief's standard for a claim. `_l1` in a run name means the run used
`LLAMA33_70B_GALAXY_TEST_LAYERS=1`, a one-layer subset of the real checkpoint: those runs are the
implementation shakeout, ~120 s each, and they are **not** the gate.

## Coverage scorecard

| # | Claim | State | Fresh processes | Evidence |
| --- | --- | --- | --- | --- |
| 1 | eager prefill 128 / 512 / 2048, single row, logits PCC ≥ 0.99 vs `GalaxyDirectRunner` | *(pending)* | | |
| 2 | eager decode, batch 1 and batch 32, first token after prefill | *(pending)* | | |
| 3 | paged KV: late capacity resolution, transactional bind/unbind, per-layer metadata, KV PCC ≥ 0.99 | *(pending)* | | |
| 4 | prefix-cached and chunked prefill | *(pending)* | | |
| 5 | program compilation and `WarmupCoordinator`; program identity on physical geometry | *(pending)* | | |
| 6 | three startup / serving / cleanup cycles in one process, nothing retained | *(pending)* | | |
| 7 | teacher-forced accuracy: top-1 ≥ 91%, top-5 ≥ 99% at prefill 512 / decode 511 | *(pending)* | | |

## The one-layer shakeout, in full

Ten runs, ~2 min each, `LLAMA33_70B_GALAXY_TEST_LAYERS=1`. **Not the gate** — this is the record of
how the executor was brought up and what it found on the way. Every log is in `logs/`.

| run | result | what it established |
| --- | --- | --- |
| `p0_partition` | 5 passed | the worker envelope is healthy before anything else ran |
| `i1_ref128_l1` | passed | reference recorded through `GalaxyDirectRunner`: prefill argmax 115745, decode argmax 20110 |
| `i2_exec128_l1` | **passed** | prefill 128 logits PCC **0.99941** vs the runner, same argmax |
| `i3_decode1_l1` | **passed** | first decode step, row 0 logits PCC **0.99359** vs the runner |
| `i10_decode32_l1` | **passed** | 32 sequential prefills then one decode; all 32 slots take the reference argmax |
| `i4_pagedkv_l1` | failed | late resolution, per-layer metadata and bind/unbind all reached; **first-layer K PCC 0.907** |
| `i5_warmup_l1` | failed | **L1 address clash**, one compiled decode program then the next prefill |
| `i6_repeat_l1` | failed | cycle 0 correct and identical to the reference; **cycle 1's prefill clashes** |
| `i7_prefix_l1` | failed | `no prefill config for recipe … PREFIX_CHUNKED` — model configuration |
| `i8_chunked_l1` | failed | `no all_gather resources … sequence=1024` — model configuration |
| `i9_shrink_l1` | failed | `paged KV cache shape must be (2048, 1, 32, 128), got (95, 1, 32, 128)` |

### Four things the shakeout changed, and why none of them is a relaxation

1. **`i9` — a smaller physical KV pool has to reach the model.** The runtime documents late
   resolution as "only `num_blocks` becomes final". `Attention2D.bind_kv_cache` validates a bound
   cache against the block count its **own** metadata declares, so on Galaxy the model must be told
   the physical count too. `configure_paged_kv_cache` now does that, refuses an expansion, and
   narrows the construction ceiling to the physical pool so the bound cache, the module metadata
   and the page-table geometry describe one geometry.
2. **`i7` — a cached request is a chunked request to the planner**
   (`uses_chunked_prefill = sequence_length > max_prefill_chunk_size or cached > 0`), and the 2D
   attention module resolves one frozen recipe per prefill *shape*. The `PREFIX_CHUNKED` recipe has
   to be registered at construction through `chunked_prefill_sequence_lengths`, or the request is
   refused on the host before any device work. Fixed by configuring the model.
3. **`i8` — the runtime's planner decides the padded device length, not the model.**
   `_padded_prefill_length` pads ≤128 to 128, ≤1024 to **1024**, and larger to the next power of
   two, so a **512-token prompt is a 1024-token device request**. A model built with
   `(128, 512, 2048)` has no 1024 recipe and no 1024 CCL resources. The registered set is now
   `(128, 1024, 2048)`; `GalaxyDirectRunner.padded_prefill_length` resolves against the same set,
   so both sides of every comparison pad alike. Separately, `_max_prefill_chunk_size` refuses any
   chunk that is not a multiple of 2048 and the model fixes `max_prefill_chunk_size` at 2048, so a
   genuinely chunked request needs a context longer than 2048 — the chunked case is 4096 tokens in
   two 2048-token chunks.
4. **`i4` — the prefill rotary went back to the qualified slice.** The first implementation
   *gathered* cos/sin from the RoPE table with `ttnn.embedding`, because the runtime hands the
   position indices over as a device tensor and a gather needs no host round trip and is
   trace-compatible. It is not equivalent: with the gather the first layer's K came back at PCC
   0.907 while the same request's logits agreed at 0.9994, and K is the one tensor of the pair that
   passes through RoPE. The view now reads the start position out of the staged tensor — four bytes,
   and the runtime builds it as `arange(start_pos, start_pos + seq_len)` — and calls the model's own
   `prepare_prefill_rot_mats`, which is what `GalaxyDirectRunner` calls. `scratch/test_rope_gather_probe.py`
   measures the two directly and stays queued, because whether the gather is exact is what decides
   if the readback can go away again.

## Modularity note

Checked mechanically against the job's base commit `67a208db961`:

```sh
git diff --name-only 67a208db961..HEAD | grep -vE '^tttv2_'
#   models/common/models/llama33_70b_galaxy/executor.py
#   models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py
git diff --name-only 67a208db961..HEAD | grep -cE '_1d\.py$'                     # 0
git diff --name-only 67a208db961..HEAD | grep -cE '^models/common/llm_runtime/'   # 0
git diff --name-only 67a208db961..HEAD | grep -c  'tttv2_2d_modules_plan.md'      # 0
```

| Question `c-signoff` asks | Answer |
| --- | --- |
| new files added | **2** — `models/common/models/llama33_70b_galaxy/executor.py`, `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py` |
| existing shared files changed, and why config alone was insufficient | **0.** No shared file was changed at all, so the question does not arise. |
| 1D module files changed (required: zero) | **0** |
| default runtime behaviours changed (required: zero) | **0.** `models/common/llm_runtime/**` is byte-identical to Milestone B. |
| new config values added to the runtime | **0** |
| imports from a model-named package | **0** — checked for `models.demos`, `models.common.models.llama33_70b.`, `models.common.models.qwen3_32b`, `vllm`, `lane_group`; none present. |
| `GalaxyDirectRunner` | untouched, and now also the reference this executor is checked against |

### What the adaptation cost, and where it sits

Four things the common runtime does in a 1D-shaped way had to be met by
model-owned code rather than by a runtime change. Each is named here because the
alternative — a Galaxy branch in `llm_runtime` — is what the extension discipline
forbids, and because `c-exec-qwen` will meet the same four.

1. **The model contract.** `_GalaxyRuntimeModelView` presents
   `embed_prefill` / `embed_decode` / `prefill_forward` / `post_process_prefill_output` /
   `decode_forward` / `gather_and_untilize_logits` / `rope_setup` over the 2D graph's
   mode-explicit contract. Pure delegation plus reshaping; no numerics.
2. **Prefill rotary.** The runtime hands the position indices over as a *device*
   tensor; `RotarySetup2D.prefill_forward` wants a host `start_pos`/`seq_len` pair.
   The view gathers cos/sin from the same row-major RoPE table with
   `ttnn.embedding`, which is the op the decode rotary path already runs on this
   mesh and which emits the TILE layout `rotary_embedding_llama` requires. No host
   round trip, so it stays trace-compatible.
3. **Decode positions and the decode page table.** The runtime maps both
   replicated (`ShardTensor2dMesh(dims=(None, None))`); the Galaxy decode graph
   needs them column-sharded (`dims=(None, 0)`), and a replicated device tensor
   cannot be resharded on device because per-device-different slicing is not one
   SPMD op. The executor stages the Galaxy-placed pair at the operation boundary
   and releases it after the call. **Cost:** the runtime still allocates and frees
   its own two small tensors, which this path ignores — two DRAM allocations of
   32 int32 and 32 x 64 int32 per decode step, unused.
4. **Logits composition.** `prefill/result_collector.concat_host_output` and
   `decode._concat_host_output` concatenate mesh **columns** along the vocabulary
   axis. On Galaxy the vocabulary is sharded over the eight mesh **rows** and
   replicated over the four columns; composing along the wrong axis is D-B23, and
   it fails loudly here rather than silently (the composed width would be 64512
   against a 128256 destination). The view composes with the qualified
   `collectives.compose_galaxy_logits`. Decode returns the composed host tensor —
   both runtime readers accept `torch.Tensor` and pass it through — and prefill
   re-stages one composed row as a replicated `[1, 1, 32, vocab]` TILE tensor,
   because the runtime untilizes and slices prefill logits on device before
   reading them. **Cost:** one host round trip per prefill call (~8 MB staged
   replicated) and one composition per decode step. **The trace-compatible
   successor is a device all-gather over the mesh-row axis**, which needs a new
   persistent CCL resource in `models/common/models/galaxy/plans.py`; it is named
   here for `c-trace` and was not added by this job.

### One latent defect found in shared Galaxy code, not fixed here

`GalaxyPagedKVContract` (`models/common/models/galaxy/kv_contract.py`) **snapshots**
the model's per-layer KV specs at construction, so a `configure_paged_attention`
that moves `block_size` or `max_num_blocks` leaves the contract describing the old
geometry and `PagedKVCacheManager.configure` then validates a replacement against
stale metadata. The documented late-resolution step only sets `num_blocks`, which
is unaffected, so this executor takes that path and rebuilds the manager against a
fresh contract in the one case where the ceiling really moves. Recorded rather than
patched: `kv_contract.py` is shared with `c-exec-qwen` and `c-defects`.
