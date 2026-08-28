# `mb-coverage` — Milestone B step 7: paged KV, concat-32, prefix cache, device sampling, long context

Written 2026-08-27, unattended, at commit `0c1ccd8557c7cb25cd1ca300d522eab1ed5db733`
on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.
Environment and mesh facts: `ENVIRONMENT.md`. Raw logs: `logs/`.

---

## Read this first

**The mesh is still down, in exactly the state `mb-qwen` left it.** Eleven of 32
boards are off the PCIe bus (`0 1 2 3 4 5 6 7 10 11 14`). `ttnn` cannot open a
cluster at all: every attempt dies at
`TTDevice::is_pcie_hung — Read 0xffffffff over PCIe ID 17`. **No recovery
attempt was spent**, because two jobs have now proved that neither
`tt-smi -glx_reset` nor `tt-smi -r` can bring back a board that is not on the
bus, and `mb-qwen`'s handoff says so explicitly.

So: **every device line of the Milestone B exit gate is `NOT REACHED`, and this
job measured none of them.** There is still no numerical result from silicon for
either model, of any kind. That has now been true for three consecutive jobs.

**Qwen is additionally blocked upstream.** Its weights are not on this machine —
`config.json` only, ~65 GB still to fetch. Per this job's brief, the Qwen half
of every area is recorded `BLOCKED (upstream)` and the scope was not quietly
halved: the Qwen device coverage was written, and it is marked never-executed
like the Llama half.

**What this job did instead.** Most of what makes step 7 *correct* is decided on
the host before a single TTNN call: which blocks a slot owns, which of the two
page-table layouts a call stages and how it is mapped, which tokens and source
rows a concatenated prefill plans, and what values `Sampling2D` writes into its
per-slot buffers. All five areas were attacked at that level, with **162 new
host tests that pass identically in three fresh processes**, plus 33 new device
tests that are written, collectible, and honestly labelled as never executed.

That produced **three defects/gaps and two corrections to the brief's own
premises** — including one, D-C1, that says a gate the brief asks for cannot be
met by the current contract at all.

---

## Summary by area

| # | Area | Host-decidable half | Device half | Findings |
| --- | --- | --- | --- | --- |
| 1 | Paged KV | **PASS** — 39 tests | `NOT REACHED` (paged-vs-contiguous PCC) | **D-C1** |
| 2 | Concat-32 physical prefill | **PASS** — 34 tests, lengths 128→2048 ascending | `NOT REACHED` | **G-C1**, **G-C2** |
| 3 | Prefix-cached / chunked prefill | **PASS** — 19 tests | `NOT REACHED` (the numerical gate) | **G-C3** |
| 4 | Device sampling | **PASS** — 26 tests | `NOT REACHED` | **D-C2**, **F-C1** |
| 5 | Long context 4K/32K/128K | **PASS** — 32 tests (capacity accounting) | `NOT REACHED` (functional smokes) | capacity table below |
| — | Repeat and cleanup | **PASS** — 12 tests | `NOT REACHED` (the L1 OOM itself) | L1 confirmed on host |
| — | Regression gates | **PASS**, boundaries clean | n/a | **F-C2** |

Per-area detail follows the exit-gate table.

---

## The Milestone B exit gate, measured at this tree

Every row carries the command that produced it. Nothing here is quoted from
`mb-llama` or `mb-qwen`.

| Gate line | Result | Measured value | Command / log |
| --- | --- | --- | --- |
| Llama teacher-forced, batch 1, prefill 512 / decode 511, top-1 ≥ 91%, top-5 ≥ 99% | **NOT REACHED** | none — cluster open fails | `pytest models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1` → `1 error in 7.72s`, `logs/12_device_attempt_*teacher_forced*.log` |
| Qwen teacher-forced, batch 1, sequence 512, top-1 ≥ 89%, top-5 ≥ 97% | **BLOCKED (upstream)** + NOT REACHED | none — weights absent *and* no mesh | `logs/12_device_attempt_*qwen*.log`; `ENVIRONMENT.md` §Checkpoints |
| Batch-32 direct demos valid, no cross-slot contamination | **PARTIAL** | mechanism proved on host; no device demo output | host: `test_step7_paged_kv.py::test_no_two_slots_can_address_the_same_block` (5 params) + `..._sink_never_lands_in_an_active_slots_run` — 8 passed ×3 processes. device: `models/common/models/llama33_70b_galaxy/demo.py` never executed |
| Batch-1 4K / 32K / 128K functional smokes | **NOT REACHED** | capacity accounting only (table below) | host: `test_step7_long_context.py` — 32 passed ×3. device: `test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke` errors at cluster open |
| Prefix-cached output matches uncached execution | **NOT REACHED** | addressing proved on host; no PCC | host: `test_step7_prefix_cache.py` — 19 passed ×3. device: new `test_step7_coverage_wh_galaxy.py::test_llama_chunked_prefill_matches_a_single_uncached_prefill`, never executed |
| No dependency imports from an existing model-named implementation package | **PASS** | 0 matches | grep below |
| Zero changes to 1D module implementation files | **PASS** | 0 files | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → empty |
| Zero changes to `llm_runtime` | **PASS** | 0 files | `git diff --name-only bc6ad03bfc2..HEAD \| grep 'llm_runtime'` → empty |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL** (inherited, not caused here) | 5 failures | see below |

### The `re-measure, do not quote` instruction

The brief asks that the two accuracy numbers be re-measured at this tree rather
than quoted from `mb-llama` and `mb-qwen`. **Neither job ever measured one.**
Both recorded `BLOCKED (infra)` with no accuracy number at all. There is
therefore nothing to re-measure and nothing to disagree with; the honest
statement is that the two accuracy gates have never been measured, by anybody,
at any tree.

### The 1D demo-contract line

Re-measured here, and it is the same set `reconcile` recorded as its finding O2:

```text
FAILED models/common/tests/models/deepseek_r1_distill_qwen_14b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/qwen2_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/qwen25_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/llama33_70b/test_demo_contract.py::test_demo_resolves_central_trace_region_size_for_each_supported_sku
FAILED models/common/tests/models/llama32_3b/test_hf_adaptor.py::test_generator_downgrades_n150_all_trace_to_decode_only
```

Proved independent of Milestone B, mechanically rather than by assertion: the
complete set of files changed between the Milestone A tip `bc6ad03bfc2` and HEAD
is `models/common/models/{galaxy,llama33_70b_galaxy,qwen3_32b_galaxy}`, four 2D
module files, their tests, and markdown. Nothing under `models/common/llm_runtime`,
nothing in any 1D model package, and none of those five test files.
`_plan_prefill_requests` — the function three of the five failures land in — is
`llm_runtime` code and is byte-identical to Milestone A's.

```sh
git diff --name-only bc6ad03bfc2..HEAD | sed 's|/[^/]*$||' | sort | uniq -c
git diff --name-only bc6ad03bfc2..HEAD | grep -v '^models/common/\(models\|modules\|tests\)/' | grep -v '^tttv2'   # empty
```

So the gate line is **FAIL as measured** and **not Milestone B's to fix**. It
belongs to whoever owns those packages, and `mb-signoff` should record it that
way rather than as a Milestone B regression.

### Model-named import gate

```sh
grep -rnE 'from models\.(demos|common\.models\.(llama33_70b|qwen3_32b))[. ]' \
     --include='*.py' models/common/models/galaxy models/common/models/llama33_70b_galaxy \
     models/common/models/qwen3_32b_galaxy
```

`logs/15_import_boundary_20260827T030156Z.log` — 0 matches.

---

## Area 1 — Paged KV

`models/common/tests/models/galaxy/test_step7_paged_kv.py`, 39 tests.

### Proved on the host

| Claim | How |
| --- | --- |
| No two slots can address the same block, at active batch 1, 8, 16, 31, 32 | pairwise set intersection over all 32 rows of the real `_page_table_rows()` |
| No idle slot's sink block lands inside an active slot's run | same, split active/idle |
| Every addressed block is inside the allocated pool | `min >= 0`, `max < max_num_blocks` |
| Prefill replicates a **padded** table; decode shards an **unpadded** one | staged mapper and width captured at `ttnn.from_torch` |
| The decode table's device-local view is `[8, blocks]`; prefill's is `[32, blocks]` | shard arithmetic modelled from `TensorToMesh::Impl::create_tensor` |
| Late capacity resolution reaches every layer, for **both** model classes | `configure_paged_attention` on a detached model, then `local_cache_shape()` |
| Capacity cannot be re-resolved while a cache is bound | `RuntimeError("cannot be reconfigured")` |
| A bind that fails part-way leaves **no** layer bound | layer 2's cache given the wrong dtype; layers 0–1 already bound |
| A malformed layer entry unwinds every earlier layer | one tensor instead of two |
| Unbind is transactional, idempotent, and owner-only | `PermissionError` for a second owner |
| Rebinding replaces rather than stacks | binding identity compared |

Both model classes (`Llama33_70BGalaxyTransformer2D`, `Qwen3_32BGalaxyTransformer2D`)
are parametrized: their `set_kv_cache` and `configure_paged_attention` are
character-identical, and pinning both keeps them that way.

### `NOT REACHED`

Paged fill during prefill, then decode reading the same blocks, at PCC ≥ 0.99
against the contiguous path. Written as
`test_step7_coverage_wh_galaxy.py::test_{llama,qwen}_paged_and_contiguous_caches_agree`,
never executed. **Nothing in this tree has ever compared the two cache layouts.**

### D-C1 — a prefill-shaped page table fed to decode is accepted, not rejected

**Severity: correctness. This is the one gate in the brief that the current
contract cannot meet.**

The brief asks: *"feed decode a prefill-shaped table and assert it is rejected,
not silently accepted."* It is not rejected.

`Attention2D._validate_decode_page_table` discriminates on **row count alone**:

```python
per_column = self.config.users_per_column          # 8
if shape[0] < per_column or shape[0] % per_column:
    raise ValueError(...)
```

The modulo is deliberate — an L1-sharded decode table legitimately repeats the
device-local batch once per core. But the replicated prefill table's
device-local view is **32 rows**, and `32 == 4 * 8`, so it passes the row check.
The width check then passes too, because the prefill table is stick-aligned to
eight int32 entries and is therefore *wider* than the decode table, never
narrower. The dtype matches. The table reaches `paged_update_cache` and the
paged decode SDPA with the wrong layout.

**Why shape cannot fix it.** `ttnn` reports a distributed tensor's `.shape` as
the *shard* shape: `TensorToMesh::Impl::create_tensor` builds the output
`Tensor` from `compute_tensor_spec_for_shards`, for both the host-tensor and the
raw-buffer entry points. So the correct decode table presents `[8, W]` and the
prefill table presents `[32, W_padded]` — and `[32, W]` is *also* the legal
4-core L1-sharded form. Two different things with the same rank-2 shape.

**The discriminator that would work, and is never consulted:** placement. The
prefill table is DRAM-interleaved and replicated; a legitimate repeat is L1
height-sharded over exactly `rows / users_per_column` cores.
`_validate_decode_page_table` never calls `memory_config()`.

**Not fixed here, on purpose.** An existing 2D module test,
`test_attention_2d.py::test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[32]`,
asserts that a 32-row table *is* accepted. Making decode reject it requires
changing that expectation, and the brief is explicit that changing an existing
expectation to accommodate this work is a boundary violation to report rather
than to commit. It also cannot be validated without a mesh.

**Proposed fix, for `mb-signoff` and Milestone C.** In
`_validate_decode_page_table`, require:

* `shape[0] == users_per_column` when the table's memory config is interleaved; or
* `shape[0] == users_per_column * n_cores` when it is L1 height-sharded, with
  `n_cores` read from the shard spec.

Then update the module test's `rows=32` case to supply an L1-sharded table, and
add the interleaved-32 case as a rejection. That is a coherent, testable change;
it is just not this job's to make unilaterally.

**Pinned, not papered over.** `test_step7_paged_kv.py::test_decode_cannot_tell_the_prefill_layout_from_a_four_core_l1_repeat`
records the behaviour that exists, and its docstring says in full why it is not
the behaviour the gate asks for. The reverse direction *does* fail closed and is
also pinned: a decode-shaped 8-row table handed to a prefill that fills user 8
raises `"page_table must have one row for every addressed user"`.

---

## Area 2 — Concat-32 physical prefill

`models/common/tests/models/galaxy/test_step7_concat32.py`, 34 tests.

The plan's risk is *padding inactive rows must not write KV or return logits for
inactive slots*. All three artefacts the brief names were inspected directly.

### The planned tokens

At lengths **128, 256, 512, 1024, 2048** — ascending, never jumping to 2048 —
the flat stream `prefill_batched` builds gives row *r* exactly
`[r * length, (r + 1) * length)` and nothing else. With active batches **16, 31
and 32**, every padded position in a row's span is token id `0`; no row's
padding ever carries a neighbour's token.

### The page table

The concatenated call passes the **replicated** prefill table, names
`user_ids == tuple(range(32))`, and passes no chunk table, no chunk start and no
prefix user. Verified at all five lengths.

### The source rows

With a deliberately non-identity user order (`reversed(range(32))`),
`_fill_prefill_cache` issues exactly one K and one V `paged_fill_cache` per row,
each with `batch_idx=0` against a **one-row slice** of the table, and the slices
come out in row order naming that row's user:
`[(slice(31, 32), …), (slice(30, 31), …), …]`. A row cannot address a user it was
not assigned.

### Logit isolation

`token_indices` addresses each row's last **real** token — `len(row) - 1`, not
`sequence_length - 1` — at active 16, 31 and 32. A padded row's logit is
computed from its own single real token, never from a zero.

### G-C1 — active batches 16 and 31 are not expressible as a smaller allocation

Recorded limitation, not a defect.

Two isolation mechanisms exist and **they do not compose**:

* `GalaxyDirectRunner(active_slots=k)` gives each idle slot its own sink block;
* `prefill_batched` refuses any runner with `active_slots != 32`
  (`"concatenated prefill needs exactly 32 active rows"`), and
  `Attention2D._recipe_identity` resolves only `SINGLE_ROW` or `CONCAT_32` —
  a 16- or 31-row prefill raises `"prefill recipes support exactly one row or
  concat-32 users"`.

So "active batch 16" through the concat path means *32 physical rows of which 16
carry real prompts*, which is what this suite measures. A 16-slot paged
allocation and a concatenated prefill cannot be used together. Both facts are
pinned by tests.

### G-C2 — an empty row is caught one call too late

`generate` refuses an empty prompt outright. `prefill_batched` called directly
does not: it plans `token_indices[r] == -1` and leaves the rejection to
`project_prefill_logits`. The rejection *does* happen, so no padded logit can be
returned — but only after the whole concatenated prefill graph has run. Minor;
worth an early check in the runner.

### `NOT REACHED`

Device KV and logit isolation at active 16/31/32, and concat-32 agreeing with
sequential prefill at each length. Written as
`test_step7_coverage_wh_galaxy.py::test_{llama,qwen}_concat32_*`, never executed.

---

## Area 3 — Prefix-cached and chunked prefill

`models/common/tests/models/galaxy/test_step7_prefix_cache.py`, 19 tests.

### Proved on the host

* A chunk table staged for chunk *c* starts at block `c * chunk / block_size` for
  every slot, is stick-aligned to eight entries, and pads with zeros only —
  checked against the real `_page_table_rows()` at chunks 1, 2 and 7.
* A chunk table never shares a block between two slots.
* An unaligned `chunk_start`, and a chunk past a slot's allocation, both fail
  closed.
* The chunked plan is right: chunk 0 is an ordinary prefill; every later chunk
  carries `chunk_start`, its own chunk table, and `prefix_user_id == slot`.
* Every chunk table is deallocated before the next chunk is staged — a chunk
  table that outlived its chunk would leak once per chunk across a long context.
* **Interaction, prefix-cached then normal**: after a chunked prefill on slot 0,
  a plain `prefill_row` on slot 1 plans with no prefix user, no chunk start, no
  chunk table, and the full replicated table.
* **Interaction, a mix across slots**: interleaving chunked and plain requests
  leaves every call addressing exactly one slot; no request widens another's
  batch.
* The single-row slice chunked SDPA needs is taken, and it follows
  `prefix_user_id` (`table[27:28, :]`), falling back to `user_ids[0]`
  (`table[4:5, :]`) when it is absent. An already-single-row table passes
  through; a concat-32 call keeps the full table because Q carries every row.

### A contract fact worth knowing

`_validate_prefill` requires `prefix_user_id in user_ids`. For a single-row
prefill that forces `prefix_user_id == user_ids[0]`, so the two branches in
`_sdpa_page_table` can only differ on a call that bypassed validation. The
branch is defensive, not load-bearing. Pinned both ways.

### G-C3 — the `chunk_page_table` guard is unreachable

`_recipe_identity` treats a non-`None` `chunk_page_table` as one of the four
signals that select `PREFIX_CHUNKED`. By the time `_validate_prefill` reaches

```python
if metadata.chunk_page_table is not None:
    raise ValueError("chunk_page_table requires a prefix/chunked recipe")
```

the recipe is *already* `PREFIX_CHUNKED`, so the branch can never fire. Passing a
chunk table with no chunk start silently runs the chunked recipe from token 0
instead of being refused. Dead code plus a missing check; pinned by
`test_a_chunk_page_table_alone_selects_the_prefix_chunked_recipe`.

### `NOT REACHED`

The gate itself — prefix-cached output matching uncached execution under the
model's numerical acceptance. Written for both models, never executed.

---

## Area 4 — Device sampling

`models/common/tests/models/galaxy/test_step7_sampling.py`, 26 tests.

### D-C2 — "moving a request to a different slot does not change its stream" is false

**Severity: contract conflict. Measured, and deliberately not "fixed".**

Both the device seed and the host seed are

```python
_seed_digest(seed, slot) = blake2b(f"sampling2d:{seed}:{slot}")
```

so the slot is part of the key. `_device_seed(1234, 3) != _device_seed(1234, 7)`,
and a request with one seed and one set of logits samples a different token in
slot 3 than in slot 7.

The brief's other clause **does** hold and is proved: the same seed in the same
slot gives the same token across runs, across three freshly constructed sampler
objects, and a request's *row position within a call* does not change its stream
(slot 25 at row 0 == slot 25 at row 2).

The slot mixing is not an accident — it is what stops 32 slots given one seed by
a serving front end from all emitting the same token, which is also proved here
(`test_one_seed_across_every_slot_does_not_collapse_the_batch`). The step-7
requirement and the module's design are in direct conflict. Resolving it is a
product decision — *is a seed per-request or per-(request, slot)?* — not a bug
fix, so this job measured it and left the module alone. **`mb-signoff` should
put this in front of whoever owns the serving contract.**

### F-C1 — Llama has no vocabulary padding, so its padded-vocab gate is vacuous

The brief says *"Llama's 128256 and Qwen's 151936 both pad"*. Llama does not.

```text
Galaxy alignment = 8 vocab shards * 32 = 256
128256 / 256 = 501 exactly  ->  padded_vocab_size == vocab_size == 128256
151936 / 256 = 593.5        ->  padded_vocab_size == 152064, 128 invalid ids
```

`build_invalid_vocab_mask(128256, 128256, 32)` returns `None` and
`Sampling2D(...).config.invalid_vocab_mask is None`. There is nothing to mask
for Llama and nothing that can be sampled. **A Llama pass on this gate would be
evidence of nothing**, which is why the device version of the case lives only in
the Qwen file. Asserted explicitly so the premise cannot quietly return.

For Qwen the gate is real and passes on host: the mask is `finfo(bfloat16).min`
on exactly the 128 padded ids, additively below every real logit, and no padded
id is sampled at temperature 0.0, 0.7, 1.0 or 2.0 even when every padded entry
carries a logit of `1e4`.

### Also proved on the host

| Claim | Detail |
| --- | --- |
| Greedy equals host argmax exactly | both vocabularies, 8 rows, `torch.equal` |
| `forced_argmax` and `temperature == 0` agree | same tokens with identical seeds |
| Per-slot heterogeneous top-k / top-p / temperature | slots 0, 5, 8, 17, 31 given five different triples; buffers read back per global slot; unnamed slots keep the greedy defaults `(1, 0.0, 1.0)` |
| **The temperature reciprocal pairing (defect D4)** | at T ∈ {0.25, 0.5, 0.8, 2.0, 4.0}: the runner hands the module the **raw** T, and the module writes **1/T** into the buffer. Never checked at T = 1.0, which is its own reciprocal and is what hid D4 |
| The runner's own host reference divides by T | low T concentrates on the largest logit; high T spreads across every candidate |
| The runner forces argmax for a greedy policy | `forced_argmax is True` reaches `sample_decode` |

`top_k > 32` was **not** tested and the contract was not extended, as the brief
requires.

### The composition property no single module can check

The sampler's slot→column map, the column selector's row gather, and the
runner's decode position sharding must all put global slot *s* on mesh column
`s // 8`. If any two disagree, a user samples from another user's logits — a
cross-slot contamination bug invisible to every per-module test.

Verified: `Sampling2D.slot_placement(s) == divmod(s, 8)`;
`GalaxyColumnUserSelector` stages `I(32)` sharded `(None, 2)` so column *c* owns
rows `8c..8c+7`, each an exact one-hot on its global slot;
`GalaxyDirectRunner._stage_positions` shards `[32]` with `(None, 0)`, shard width
8. All three agree.

### `NOT REACHED`

Every device half. Written for both models, never executed.

---

## Area 5 — Long context

`models/common/tests/models/galaxy/test_step7_long_context.py`, 32 tests.

The smokes are functional and the brief expects capacity, not numerics, to be
the limit — and asks for a record of where each one spends it. That record is
arithmetic over the resolved geometry, so it was produced and checked on the
host. Configuration mirrors
`test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke`:
batch 1, one served slot, 2048-token chunks, one chunk of headroom, one sink
block per idle slot.

| Context | Served | Blocks/user | Pool | KV per device, Llama (80 layers) | KV per device, Qwen (64 layers) | RoPE tables per device | Chunks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4K | 6 144 | 192 | 223 | **0.14 GiB** | 0.12 GiB | 6 MiB | 2 |
| 32K | 34 816 | 1 088 | 1 119 | **0.73 GiB** | 0.58 GiB | 34 MiB | 16 |
| 128K | 133 120 | 4 160 | 4 191 | **2.72 GiB** | 2.17 GiB | 130 MiB | 64 |

Method, all of it checked by a test rather than asserted in prose:

* the paged pool is **replicated** — every device owns the whole block pool and
  writes only the users its column serves, which is what makes one page table
  valid on every device. So the KV figure is per device and **does not shrink
  with the mesh**;
* bfloat8_b is 1088 bytes per 32×32 tile, i.e. 1.0625 B/element;
* elements per device = `pool * (n_kv_heads / 8) * 32 * head_dim`, times 2 for
  K and V, times the layer count;
* RoPE cos/sin are replicated bf16 `[1, 1, table_len, 128]` with
  `table_len = max(2 * served, 8192)` rounded up to 128 → 266 240 at 128K;
* page tables stay sub-megabyte even at 128K (`32 * 4 168 * 4 B` ≈ 534 kB).

**Where the capacity goes at 128K, Llama:** ~2.3 GiB of weights per device
(70 B params at bfloat8_b over 32 devices) + **2.72 GiB** of KV + 0.13 GiB of
RoPE tables ≈ 5.2 GiB against a 12 GB device. It should fit; the risk is
fragmentation and the 64 sequential chunked-prefill graphs, not the total.

Also proved: each long-context geometry resolves and needs **one** prefill
recipe, not one per length; a pool one block short of the served context fails
closed with `"cannot hold max_seq_len"`; the chunked plan walks the context
without revisiting a block; and the headroom chunk really does leave the decode
after a full prefill a block to write into.

### `NOT REACHED`

The three functional smokes themselves.

---

## Repeat and cleanup

`models/common/tests/models/galaxy/test_step7_repeat_and_cleanup.py`, 12 tests.

### Repeated requests against one live model

Two identical `generate` calls produce identical tokens *and* identical plans:
same prefill and decode call counts, and the staged token rows compare equal
element for element. A repeated request rebinds nothing and restages nothing —
the KV binding happens once, at `open`.

### Runner teardown

`close` unbinds the cache, deallocates both page tables and every K/V tensor, and
is idempotent; a closed runner refuses further graph calls; reopening allocates a
genuinely fresh cache; `open` on an already-open runner is a no-op.

**A failed `open` leaves nothing bound.** Injecting a failure while staging the
*decode* table — after the cache has been allocated and bound — leaves
`bind_calls[-1] is None`, an empty `_kv_cache`, and both tables `None`. The
ordering comment in `direct_runner.open` ("Recorded before the page tables so a
staging failure still unbinds") is correct and is now pinned.

### L1, confirmed on the host

`Prefetcher2D.cleanup()` clears `self._global_cb` **without adding it to the
resources it deallocates** — ttnn exposes no free for a global circular buffer.
Measured with the module suite's injectable `create_global_cb`/`deallocate`:
after `cleanup()` the owner reports `owned_resources == ()` while the CB it
created was never handed to `deallocate`. Two owners in one process allocate two
CBs and neither is freed.

That gap is the whole of L1. The **OOM** it causes needs real L1 and was not
reproduced. The honest reading of a clean `cleanup()` is "nothing this object
still owns", not "nothing is left on the device", and the tests now say so.

**Is the ordering contract workable at model scale?** Unknown, and it should not
be guessed at. The 80-layer model has never been built, so no one has ever
observed a second Galaxy model construction in one process. `test_two_models_in_one_process`
exists in `llama33_70b_galaxy/test_bringup_wh_galaxy.py` and has never run. This
stays as Milestone C input, unchanged from `reconcile`'s O5.

---

## Regression gates

### The brief's command, before and after

```sh
python -m pytest -q models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime
```

| | Before this job's changes | After |
| --- | --- | --- |
| Log | `logs/03_baseline_full_gate_BEFORE_20260827T020908Z.log.gz` | `logs/14_full_gate_AFTER_20260827T025719Z.log.gz` |
| Result | `18 failed, 1959 passed, 2059 skipped, 3276 deselected, 318 errors in 987.13s` | `18 failed, 2121 passed, 2059 skipped, 3276 deselected, 351 errors in 1048.36s` |
| Delta | — | **+162 passed** (exactly this job's host tests), **+33 errors** (exactly this job's two device files, both at cluster open), **failure set byte-identical** |

`logs/16_regression_delta_20260827T032039Z.log` holds the diff of the two
`FAILED` sets: **empty**. The three largest logs are stored gzipped.

The 18 failures are the same 18 before and after: 13 are F-C2 below, 5 are the
pre-existing 1D demo-contract set. **This job introduced no failure and fixed
none** — it changed no implementation file.

### F-C2 — `models/common/tests/models/galaxy/test_plans.py` is not a host-only suite

13 of the 18 baseline failures are in `test_plans.py`, and every one of them is
**device-induced**, not a real defect:

```text
galaxy_prefill_mode_plan -> ttnn.SubDevice([cores])
  -> SubDeviceImpl::SubDeviceImpl -> MetalContext::instance()
  -> Cluster::open_driver -> RuntimeError: Read 0xffffffff over PCIe ID 17
```

`ttnn.SubDevice` implicitly constructs the `MetalContext`, so a suite that looks
host-only — no `mesh_device` fixture, a `MagicMock` mesh, no `_wh_galaxy` in its
name — cannot run without a cluster. `mb-qwen`'s filtered host command missed
this because it does not include `models/common/tests/models/galaxy`. Worth
knowing for `mb-signoff`: on a healthy mesh these 13 should pass, and if they do
not, *that* is a finding.

The 318 errors are all `*_wh_galaxy*` device suites plus the three `moe` device
suites, all at cluster open.

### Boundaries

```sh
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'      # empty
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'  # empty
```

Both empty, over all 190 changed paths. This job changed **no implementation
file at all** — only tests and evidence. That is deliberate: every defect it
found either needs a mesh to validate a fix, or needs a product decision first.

---

## Defects and gaps, collected

| ID | Severity | Where | One line |
| --- | --- | --- | --- |
| **D-C1** | correctness | `attention_2d.py::_validate_decode_page_table` | The replicated prefill page table is accepted by decode, because its device-local 32 rows are indistinguishable by shape from a 4-core L1-sharded repeat. The step-7 rejection gate cannot be met without consulting `memory_config()`. |
| **D-C2** | contract conflict | `sampling_2d.py::_seed_digest` | Moving a seeded request to another slot changes its stream, because the slot is part of the seed digest. Deliberate decorrelation; directly contradicts the step-7 slot-stability gate. Needs a product decision. |
| **G-C1** | limitation | `direct_runner.prefill_batched`, `attention_2d._recipe_identity` | Concat-32 requires all 32 slots active and exactly 1 or 32 prefill rows; the sink-block mechanism for `active_slots < 32` cannot be combined with it. |
| **G-C2** | minor | `direct_runner.prefill_batched` | An empty row plans `token_indices == -1`; rejection happens one call later, after the whole concatenated graph has run. |
| **G-C3** | dead code + missing check | `attention_2d._validate_prefill` | `"chunk_page_table requires a prefix/chunked recipe"` is unreachable, because a chunk table alone already selects `PREFIX_CHUNKED`. |
| **F-C1** | premise correction | `recipes.galaxy_padded_vocab_size` | Llama-3.3-70B has **no** vocabulary padding; its padded-vocab gate is vacuous. Only Qwen pads (128 ids). |
| **F-C2** | test-infra | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster: `ttnn.SubDevice` constructs the `MetalContext`. |

Inherited and untouched: **L1** (global-CB ownership — confirmed on host here),
**L3**, **D-B9**, and the five pre-existing 1D demo-contract failures.

---

## What was committed

**No implementation file changed.** New tests only:

```text
models/common/tests/models/galaxy/step7_harness.py                     (helper, not collected)
models/common/tests/models/galaxy/test_step7_paged_kv.py               39 tests
models/common/tests/models/galaxy/test_step7_concat32.py               34 tests
models/common/tests/models/galaxy/test_step7_prefix_cache.py           19 tests
models/common/tests/models/galaxy/test_step7_sampling.py               26 tests
models/common/tests/models/galaxy/test_step7_long_context.py           32 tests
models/common/tests/models/galaxy/test_step7_repeat_and_cleanup.py     12 tests
                                                                      --- 162 host tests, 3 fresh processes, identical

models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   17 tests, NEVER EXECUTED
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     16 tests, NEVER EXECUTED
```

### On committing device tests that have never run

`mb-qwen` deliberately wrote none, arguing that an unexecuted device test invites
you to trust it. That argument is right, and this job took the other side of it
for one reason: the step-7 device gaps are now *specific* — paged-vs-contiguous,
late capacity, concat-32 at 16/31/32 across five lengths, the two prefix-cache
interactions, four sampling claims — and leaving them as prose in a report means
the next person with a mesh has to re-derive them under time pressure.

The mitigation is to be loud rather than to abstain. Both files say **"This file
has never been executed"** in their module docstring, both name the date and the
reason, and both say to treat a first run as bringup rather than as a
regression. Both were verified to *collect* (17 and 16 node ids), which proves
the imports and fixtures resolve — and nothing more than that.

### The `step7_harness.py` shard-shape model

The harness reproduces one non-obvious `ttnn` fact: a distributed tensor's
`.shape` is the **shard** shape, not the global one. That was read out of
`ttnn/core/distributed/distributed_tensor.cpp` (`TensorToMesh::Impl::create_tensor`
builds the output `Tensor` from `compute_tensor_spec_for_shards`), not measured
on silicon. Every host conclusion that depends on a device-local shape — D-C1
most of all — rests on it. **First person with a mesh: check it.** One line:

```python
t = ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), device=mesh,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(8, 4)),
                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
assert tuple(t.shape) == (8, 64)   # if this is (32, 64), D-C1 is worse than described
```

If `.shape` turns out to be the *global* shape, then the decode validator's
`shape[0] % users_per_column` check never sees 8 at all, the "device-local rows"
branch is unreachable for a correctly-mapped table, and D-C1 is not a loophole
but a total absence of validation.

---

## Two housekeeping notes

Commit produced: **`1cd451cd965`**.

**Pre-commit reformatted four test files.** `black` and `isort` rewrote
`test_step7_{long_context,paged_kv,sampling,concat32,prefix_cache,repeat_and_cleanup}.py`
and the Llama device file on the first commit attempt. The host suites were
re-run afterwards in **three more fresh processes** — `162 passed` each time,
`logs/17_step7_after_precommit_format_*.log` — and both device files re-collected
at 17 and 16 node ids. The committed content is the post-format content, and the
three-fresh-processes rule is satisfied against *it*, not only against the
pre-format text.

**Pre-commit's `trailing-whitespace` hook also rewrote eleven of the raw logs.**
It strips trailing spaces from line ends; no line was removed and no content
changed. Verified after the fact: the mesh-state log still has its 34 lines and
the device-attempt log still contains `Read 0xffffffff over PCIe ID 17`. Noted
because "never overwrite a log" is a house rule, and a hook did it rather than
this job — but the evidence is intact and this is what it looks like when it
happens.

---

# §A2 — attempt 2, on a live mesh

Written 2026-08-27/28 by `mb-coverage` **attempt 2**, unattended, at commit
`b1e824537a4` (`mb-qwen` attempt 2's tip) on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Everything above this line is attempt 1's report and is left untouched. It was
written with the mesh down and is a host-only document; where it and this section
disagree about the machine, this section was measured on silicon and is later.

## The three premises of attempt 1 that are false at this tree

| Attempt 1 said | At this tree |
| --- | --- |
| "The mesh never came back … `ttnn` cannot open a cluster at all." | **Alive.** `ls /sys/class/tenstorrent \| wc -l` = 32, and `test_partition_wh_galaxy.py` opens a real 8×4 cluster: `5 passed in 12.32s` (`logs2/a2_00_mesh_health.log`). Established before planning anything. |
| "Three consecutive device jobs have produced zero numerical results from silicon, for either model." | **False.** `mb-qwen` attempt 2 (17:53–22:51 UTC, after attempt 1) qualified both models end to end: Llama 501/511 top-1, Qwen 498/511, PCC 0.999+ per block for both. Its handoff is `job2_completion_handoff_attempt2.md`. |
| "Qwen's weights are not on this machine." | **Present**, under `HF_HOME=/localdev/ctr-apbernal/hf_data` — *not* `/proj_sw/user_dev/hf_data`, which reaches Llama only. |

Attempt 1 was not wrong about what it saw at 03:00 UTC. It was superseded by a
mesh repair at ~17:00 and by a job that ran after it. This is the same failure
mode its own handoff warned about ("evidence collected at a tree that has since
moved is not evidence") applied to the *machine* rather than the tree.

## F-C1 is superseded: Llama does pad its vocabulary, by 768 ids

Attempt 1's finding F-C1 reads: *"**Llama has no vocabulary padding.** 128256 is
already a multiple of `8 * 32`. Its padded-vocab gate is vacuous; only Qwen pads
(128 ids)."* Both halves are false at this tree, and the tree already knew:

```python
>>> from models.common.models.galaxy.recipes import galaxy_padded_vocab_size
>>> galaxy_padded_vocab_size(128256), galaxy_padded_vocab_size(151936)
(129024, 153600)
```

The width is not rounded to `8 * 32`; it is rounded so that the **per-device**
width is a whole number of 24-core ring rows — `(padded // 8) % (24 * 32) == 0` —
which is the invariant D-B19 was named for. `128256 // 8 = 16032` is 501 tiles,
which no usable core count divides, so Llama pads to 129024 and carries **768**
invalid ids; Qwen pads to 153600 and carries **1664**, not 128.

`test_step7_sampling.py` was corrected for this in `60fdec0c09e` (after attempt
1's commit), so the host suite is right. What was left wrong was the *device*
coverage: `test_step7_coverage_wh_galaxy.py` for Llama said in its module
docstring that the padded-vocabulary case is "not applicable" and omitted it.
Attempt 2 added `test_llama_no_padded_vocabulary_id_is_ever_sampled` at three
policies (greedy, T=1.5, T=0.5 — never T=1.0, which is its own reciprocal and
hides D4) and corrected both files' docstrings.

**Why it matters beyond bookkeeping:** an invalid id winning is a correctness
bug, and for Llama the gate was recorded as vacuous — i.e. nobody would ever
measure it.

## The Milestone B exit gate, measured at this tree

Commit measured: `1451b192584` for runs 01/01b/02/03/g1 and `718997518ab` for every run from `g2` onward. `mb-coverage` attempt 3 established that `git diff 718997518ab..HEAD -- models/` is **empty**, so every `718997518ab` row below was produced against source identical to `af589dff4d5`; the two commits between `1451b192584` and `718997518ab` touched only the two `test_step7_coverage_wh_galaxy.py` files, which `test_full_model_wh_galaxy.py` does not import. See §A3 for the final table, which supersedes this one.

Every value below was produced by a command in this section — none is quoted from `mb-llama`,
`mb-qwen` or attempt 1. Where a number *does* agree with an earlier job's, that
agreement is stated as a result of re-measurement, which is what the brief asked
for.

| Gate line | Verdict | Measured |
| --- | --- | --- |
| Llama teacher-forced, batch 1, 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS**, 2 runs | top-1 **501/511 = 98.04%** (gate ≥ 91%), top-5 **511/511 = 100.00%** (gate ≥ 99%). `a2_01_llama_full_model_file.log` and `a2_g1_llama_tf.log`, character-identical |
| Qwen teacher-forced, batch 1, 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS**, 1 run | top-1 **498/511 = 97.46%** (gate ≥ 89%), top-5 **511/511 = 100.00%** (gate ≥ 97%). `a2_g12_qwen_tf.log` |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, 1 run per model | Llama `a2_g9`, Qwen `a2_g21`: 32 slots, each answering its own prompt; Llama slot 0 character-identical to the batch-1 demo. The *test* `*_batch32_slots_are_isolated` is a different shape and FAILED for Llama on L1 (`a2_g7`), PASSED for Qwen 3/3 |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, 1 run per geometry per model | Llama 4K/32K/128K `a2_g3`/`a2_g4`/`a2_g5` (7/11/13 min); Qwen `a2_g14`/`a2_g15`/`a2_g16` (3/3/5 min). Qwen 128K exceeds its own `max_position_embeddings` (40960) and nothing enforces it: a capacity-and-plumbing result, not a quality one |
| Prefix-cached output matches uncached execution | **PASS**, 1 run per model | Llama `a2_g2`, Qwen `a2_g13`: two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99 |
| No dependency imports from a model-named implementation package | **PASS** | 0 matches, over `models/common/{models/galaxy,modules,models/llama33_70b_galaxy,models/qwen3_32b_galaxy}` |
| Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → 0 of 338 changed paths |
| Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → 0 |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and not owned by Milestone B | **5 failed, 296 passed** (`a2_h1_1d_contract_gate.log`). The same five ids attempt 1 recorded. None of the five packages appears in `bc6ad03bfc2..HEAD` at all, so Milestone B cannot be their cause. Expectations unchanged — nothing was edited to accommodate this work |

## How attempt 2 ran, and what "recorded" means here

One pytest process on the mesh at a time, never piped, driven by
`cov_seq2.sh` over a manifest; each cycle reaps only the PID it started, refuses
to signal anything whose `comm` is not python, and runs `tt-smi -glx_reset`
after any non-clean exit. Logs are `logs2/a2_*.log`, one per cycle, never
overwritten. `RESULTS_A2.md` is the run-by-run index, written as each cycle
finished.

**The cost that shaped the night.** Every test in these files builds its own
model, and a Llama 80-layer build from the *warm* device weight cache is ~5.5
minutes; a *cold* recipe is far worse — `a2_01` spent 26 minutes staging 723
weights because the 512-token prefill recipe had never been resolved at this
commit. So a 17-node-id file is a three-hour run, and the house rule "three runs
in fresh processes before any device claim" cannot be applied to all 36 step-7
device cases in one night. What attempt 2 did instead, and states per row:

* the **exit-gate** lines and the **headline** step-7 mechanisms get three fresh
  processes;
* the remaining step-7 cases get **one** process, and are recorded as
  *observed, not qualified*. A single pass is bringup, and this project's own
  history says a case that passes once has proved nothing.

Nothing was recorded as evidence at a run count it did not get. Where a row says
`1 run` that is a statement about how much you may lean on it.

### One node id per process, and the 55 minutes it took to learn that

Attempt 2 began by running whole files in one process, on the reasoning that a
mesh open costs 25 s and 8 node ids in one process saves seven of them. That is
wrong on this stack, and expensively so — see **D-C3**: the device weight cache
fingerprint contains `MeshDevice.id()`, which increments per test, so test 2 of a
file re-stages all 965 weight tensors (138 GB, 26 min) and test 3 does it again.
The first cycle was stopped at 00:18 for that reason, its two completed tests
kept, and everything after it re-queued **one node id per process**. In that
shape every run is 100% cache hits.

The queue runner (`cov_queue.sh`) also grew a disk guard when this was found: it
prunes only the `.tensorbin` files this job wrote, and halts rather than
continue, if `/proj_sw` falls below 300 GB / 150 GB free.

## Which device case covers which of the brief's five areas

`L` = `models/common/tests/models/llama33_70b_galaxy/`,
`Q` = `models/common/tests/models/qwen3_32b_galaxy/`,
`G` = `models/common/tests/models/galaxy/`,
`step7` = `test_step7_coverage_wh_galaxy.py`,
`full` = `test_full_model_wh_galaxy.py`.

| Brief area | Claim it asks for | Device case |
| --- | --- | --- |
| 1 paged KV | paged fill then decode, PCC ≥ 0.99 vs contiguous | `{L,Q}/step7::*_paged_and_contiguous_caches_agree` |
| 1 | late capacity resolution | `{L,Q}/step7::*_paged_capacity_resolved_after_construction_serves_a_request` |
| 1 | transactional bind/unbind, failed bind leaves no partial state | host only (`G/test_step7_paged_kv.py`) — no device case needs one, the unwind is pure Python |
| 1 | no cross-slot contamination | `{L,Q}/step7::*_a_write_for_one_user_never_appears_in_another_users_blocks`, and `{L,Q}/demo.py::*_batch32_has_no_cross_slot_contamination` |
| 1 | a prefill-shaped table fed to decode is **rejected** | **not satisfiable at this contract** — D-C1. Pinned on the host and now on silicon: `G/test_step7_page_table_placement_wh_galaxy.py` |
| 2 concat-32 | concat-32 agrees with sequential prefill, 128 → 2048 ascending | `L/step7::*_concat32_matches_sequential_prefill_at_each_length[len128..len2048]`, `Q/…[len128..len512]` |
| 2 | padded rows change no active row's logits, active 16/31/32 | `{L,Q}/step7::*_concat32_padded_rows_change_no_active_rows_logits[active16,31,32]` |
| 3 prefix cache | prefix-cached output matches uncached | `{L,Q}/full::*_prefix_cached_prefill_matches_uncached` and `{L,Q}/step7::*_chunked_prefill_matches_a_single_uncached_prefill` (the second also decodes, so the cache the chunks *wrote* is read) |
| 3 | a prefix-cached request then a normal one | `{L,Q}/step7::*_a_prefix_cached_request_then_a_normal_one` |
| 3 | a mix of both in one batch | `L/step7::test_llama_prefix_cached_and_plain_requests_mixed_across_slots` (Llama only) |
| 4 sampling | greedy equals host argmax, every slot | `{L,Q}/step7::*_device_greedy_sampling_equals_host_argmax`, `{L,Q}/demo.py::*_device_sampling_matches_host_greedy` |
| 4 | seeded slot stability across runs | `{L,Q}/step7::*_a_seeded_slot_repeats_across_runs` |
| 4 | a padded id can never be sampled | `Q/step7::test_qwen_no_padded_vocabulary_id_is_ever_sampled`, and **new in attempt 2** `L/step7::test_llama_no_padded_vocabulary_id_is_ever_sampled` |
| 4 | per-slot heterogeneous top-k/top-p/temperature | `L/step7::test_llama_per_slot_heterogeneous_sampling_controls` (Llama only) |
| 5 long context | batch-1 4K / 32K / 128K functional smokes | `{L,Q}/full::*_long_context_smoke[4k,32k,128k]` |
| repeat/cleanup | repeated requests, deterministic | `{L,Q}/full::*_repeated_requests_and_deterministic_cleanup` |
| repeat/cleanup | two model constructions in one process | `G/test_step7_repeat_and_cleanup.py` on host; **no device case** — see L1 |

## Area by area, on silicon

Each row names the log. `runs` is how many fresh processes the claim got; a claim
with one run is *observed*, not qualified, and says so.

### Area 1 — paged KV

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefill and decode page tables have the layouts D-C1 assumes | `a2_01b_page_table_placement`, `a2_s34_placement_run2`, `a2_s35_placement_run3` | **3** | **PASS.** decode global `(32, 64)` → device-local `(8, 64)`; prefill global `(32, 64)` → device-local `(32, 64)`; ratio 4; **both DRAM-interleaved**. Identical output all three runs |
| A cache bound after construction serves a request | `a2_02_llama_late_capacity` | 1 | **FAIL** on `assert all(spec.paged_attention_config is None …)`. Not a model defect — **D-C4**: `from_pretrained` substitutes the default pool for `None`. Test rewritten to the reachable claim and re-queued |
| Paged fill then decode, PCC ≥ 0.99 against the contiguous path | `a2_03_llama_paged_vs_contig` | 0 | **STOPPED at 4 min, deliberately** (`rc=143`). D-C4 makes both arms the same 2048-block pool, so the case was a tautology. Rewritten as `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` and re-queued. The gate line as written is **not expressible at this adaptor API** |
| No cross-slot contamination in the blocks | — | 0 | **NOT REACHED** |
| Transactional unbind, failed bind leaves no partial state | host suite only (attempt 1, 39 tests) | — | host PASS; no device case was reached |

### Area 2 — concat-32 physical prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Concat-32 prefill agrees with sequential prefill, Llama | `a2_g10_llama_demo_concat32` | 1 | **FAIL — L1 address clash, and a new detail.** `program 1552` clashes on `[0-0 - 6-9]` — the **whole 7×10 grid**, not the four sender cores of the other L1 failures. The test runs `run_direct_demo` twice, so the second prefill follows a decode |
| Concat-32 prefill agrees with sequential prefill, Qwen | `a2_g22_qwen_demo_concat32` | 1 | **FAIL, and not the Llama failure.** `Statically allocated circular buffers on core range [0-0 - 2-3] grow to 1669312 B which is beyond max L1 size of 1499136 B`, from `validate_circular_buffer_region` at `direct_runner.py:484` (`prefill_batched`). A **capacity** overflow, not an address clash. **Finding D-C6** |
| Active batches 16, 31, 32 write no KV and return no logits for inactive slots | — | 0 | **NOT REACHED** |
| Lengths 128 → 2048 in the padded lengths the policy supports | — | 0 | **NOT REACHED** on device. The host recipe suite covers all five Llama lengths |

### Area 3 — prefix-cached and chunked prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefix-cached prefill matches uncached, Llama | `a2_g2_llama_prefix` | 1 | **PASS** — two 128-token chunks vs one 256-token prefill, same argmax and PCC ≥ 0.99 |
| Prefix-cached prefill matches uncached, Qwen | `a2_g13_qwen_prefix` | 1 | **PASS** |
| Chunked prefill matches a single uncached prefill | — | 0 | **NOT REACHED** |
| A prefix-cached request then a normal one | — | 0 | **NOT REACHED** |
| A mix of both in one batch | — | 0 | **NOT REACHED** (and the Qwen test did not exist; attempt 3 wrote it) |

### Area 4 — device sampling

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Device greedy sampling equals host argmax, Llama, through the demo | `a2_g11_llama_demo_sampling` | 1 | **FAIL — L1, `program 100`.** The demo runs twice (host policy, then device policy), so the second prefill follows a decode and never reaches the sampler. The claim itself is untested by this log |
| Device greedy sampling equals host argmax, Qwen, through the demo | `a2_g23_qwen_demo_sampling` | 1 | **FAIL, and not L1 at all.** `MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED, got: TensorMemoryLayout::WIDTH_SHARDED` at `collectives.py:445`, `GalaxyColumnUserSelector.__call__`, reached from `model.sample_decode` → `select_decode_column_users`. The host-sampling half ran first and passed. **Finding D-C5** |
| Seeded slot stability, padded vocabulary, near-zero temperature (D4), per-slot heterogeneous controls | — | 0 | **NOT REACHED.** All four cases were written (the padded-vocabulary and temperature cases *by* attempt 2) and queued, and the host was withdrawn before they ran |

### Area 5 — long context

| Geometry | Llama | Qwen |
| --- | --- | --- |
| 4K | **PASS** (`a2_g3`, ~7 min, 2 chunks of 2048) | **PASS** (`a2_g14`, ~3 min) |
| 32K | **PASS** (`a2_g4`, ~11 min, 16 chunks) | **PASS** (`a2_g15`, ~3 min) |
| 128K | **PASS** (`a2_g5`, ~13 min, 64 chunks, then a decode at position 131072) | **PASS** (`a2_g16`, ~5 min) |

One run each. Attempt 1's accounting predicted ~5.2 GiB per device for Llama at
128K against 12 GB and named fragmentation as the risk; it fits. **Qwen3-32B's
`max_position_embeddings` is 40960**, so its 128K smoke runs three times past the
trained context and nothing in the stack refuses it — `max_context_len` is carried
on the runtime config and never checked against `max_seq_len`. Functional, as the
brief defines it; not a quality statement.

Attempt 1's capacity accounting for these three geometries (blocks per user,
pool size, KV bytes per device, RoPE table size, chunk count) is in area 5 above
this section and was not re-derived; what attempt 2 adds is whether each one
actually runs.

### Repeat and cleanup

| Shape | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` — the same request twice through two runners on one live model | **FAIL 2/2**, deterministic (`a2_g6`, `a2_L1_llama_repeat_run2`): `program 100` clashes on `[0-0 - 0-3]`, L1 buffer at 544832, static CB region ends at 630080 | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2/3`) |
| `*_batch32_slots_are_isolated` — slot 0 alone, then slot 0 inside a full batch | **FAIL 1/1**, same signature (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2/3`) |
| Repeated model construction and teardown in one process (`test_two_models_in_one_process`) | **NOT REACHED** | not applicable — the bringup file is Llama-only |

The two Qwen run-3 logs (`a2_L1_qwen_repeat_run3`, `a2_L1_qwen_batch32_run3`,
both `exit=0`) landed after `RESULTS_A2.md`'s last row was written and are
recorded here for the first time; attempt 3 re-read them off disk to confirm it.
`a2_L1_llama_repeat_run3` was in flight when the host went away and has no
verdict.

## L1, and why four step-7 cases cannot be measured behind it

`mb-llama` attempt 3 named the shape of Milestone A limitation **L1** precisely:
`Prefetcher2D` allocates a `global_circular_buffer` on `activate("decode")`,
there is no `deallocate` for that type, and a *prefill* program afterwards cannot
place its circular buffers on the four sender cores the CB still occupies:

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1
             buffers on core range [0-0 - 0-3]
```

So **prefill-before-any-decode is fine; prefill-after-a-decode is not**, in one
process. Attempt 3 implemented the obvious fix
(`Prefetcher2DConfig.release_global_cb_on_prefill`) and *refuted it on hardware* —
the L1 base address is identical with the flag on, because dropping the last
Python reference does not return the L1.

Every step-7 case whose shape is *(prefill, decode) then (prefill, …)* in one
process inherits that, and there are five of them:

| Case | Why it has two phases |
| --- | --- |
| `{L,Q}/full::*_batch32_slots_are_isolated` | slot 0 alone, then all 32, and each `generate` decodes |
| `{L,Q}/full::*_repeated_requests_and_deterministic_cleanup` | the repeat *is* the second phase |
| `{L,Q}/step7::*_paged_and_contiguous_caches_agree` | two models, each prefilling and decoding |
| `{L,Q}/step7::*_a_write_for_one_user_never_appears_in_another_users_blocks` | two runners, decode after each |
| `{L,Q}/step7::*_a_seeded_slot_repeats_across_runs` | three runners, decode in each |
| `{L,Q}/step7::*_chunked_prefill_matches_a_single_uncached_prefill` | uncached then cached, decode after each |

**This is not a reason to restructure them.** The two-phase shape is the *claim*:
"a repeated identical request produces the same tokens" is not testable in one
phase, and neither is "slot 0's continuation does not depend on the other 31".
A single-phase rewrite would pass while proving nothing, which is the failure
mode this project distrusts most. They are recorded against L1 with their logs,
and they are the concrete cost of L1 to Milestone B's step-7 gate — which is
worth more to `mb-signoff` than a green tick would be.

The one open hypothesis, from attempt 3 and untried: confine the prefill mode
plan to the **worker** cores (`galaxy_prefill_mode_plan_cores` currently returns
the whole compute grid) so no prefill program can be placed on the sender
columns at all. Attempt 2 did not try it — it changes the grid of every prefill
program, so prefill 128, prefill 2048, the 80-layer prefill and both accuracy
gates all have to be re-taken behind it, and this job's own gate evidence would
have gone with it. One fact attempt 2 can add: the prefill matmuls are *already*
worker-confined (`dense_matmul_program_config` sets `allowed_worker_cores`), so
whatever program 100 is, it is not one of those two — narrowing the search to the
collectives and the MLP ring form.

## Findings, attempt 2

Attempt 1's seven (D-C1, D-C2, G-C1, G-C2, G-C3, F-C1, F-C2) plus what a live
mesh added. Only the changes are written out here; the unchanged ones keep
attempt 1's text above.

### F-C1 — **superseded, and it was the wrong way round**

See §A2's opening. Llama pads by 768 ids, Qwen by 1664. Attempt 1 recorded
Llama's padded-vocabulary gate as *vacuous*; it is live, and now has a device
case (`test_llama_no_padded_vocabulary_id_is_ever_sampled`, three policies).

### D-C1 — premise confirmed on silicon, verdict unchanged

Attempt 1 derived D-C1 from a host model of one `ttnn` fact and asked for one
line on a live mesh to settle it. That line is now a committed test,
`models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py`,
and it says attempt 1 read the fact correctly:

* a column-sharded decode table (`ShardTensor2dMesh(dims=(None, 0))`, mesh
  `(8, 4)`) has device-local shape **(8, 64)** — the shard shape, one mesh
  column's users;
* the replicated prefill table has device-local shape **(32, 64)**, and
  `32 % 8 == 0`.

So `_validate_decode_page_table`, which discriminates on the device-local row
count alone and accepts any positive multiple of `users_per_column`, cannot tell
the prefill layout from a legitimate four-core L1 repeat. **D-C1 stands exactly
as attempt 1 wrote it, and the worse variant it feared is ruled out.**

The test also records what attempt 1 could not check: **both tables are
DRAM-interleaved**, so `memory_config().is_sharded()` is false for both. A fix
therefore cannot be "reject unless sharded" applied to the 32-row case alone
without also deciding what a 32-row *interleaved* table means; the honest
discriminator is that a repeat is only legitimate when the tensor is L1
height-sharded over exactly `rows / users_per_column` cores, which makes the
existing 2D-module expectation
`test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[16]`
and `[32]` — which pass a plain interleaved table — the thing that has to change.
That is the boundary attempt 1 declined to cross, and attempt 2 declines it for
the same reason: the brief says report it, do not edit the expectation.

### D-C2 — unchanged, and still a product decision

`_device_seed`/`_host_seed` are `blake2b("sampling2d:{seed}:{slot}")`, so a
request that migrates slots does not keep its stream. The step-7 gate asks for
the opposite. Attempt 2 measured only the half that holds — same seed, same slot,
same token across fresh runs — and did not assert the half that does not.

### D-C3 — the device weight cache is keyed by `MeshDevice.id()`, so every test after the first in a process re-stages every weight

**New, severity: test-infrastructure, and it costs hours and hundreds of GB.**

`LazyWeight._get_fingerprint` ends with

```python
device_id = self.device.id() if hasattr(self.device, "id") else "single"
parts.append(f"device_{device_id}")
```

`self.device` is the **`MeshDevice`**, and the `mesh_device` fixture builds a new
one per test, so its `.id()` is 0 for the first test in a pytest process, 1 for
the second, 2 for the third. The cache path therefore changes per test, and every
test after the first misses on **every** weight.

Measured, on this mesh, at this commit:

| | |
| --- | --- |
| whole-file run, `test_full_model_wh_galaxy.py` (8 node ids) | test 1: 240 cache hits, model built in ~6 min. Test 2: **965 misses**, 26 min of staging, **138 GB** written. Test 3: staging device_2's set again |
| the same test alone in its own process | **240 hits, 0 misses**, whole test 237 s |

A complete cache set is 138.5 GB for Llama-3.3-70B, so an 8-node-id file needs
**1.1 TB** of cache to run — on a filesystem that started this night with 1.0 TB
free and 95% used. This attempt paid 55 minutes and 277 GB of it before reading
the fingerprint, and then pruned the two duplicate sets.

**Consequence for anyone scheduling this hardware: one node id per pytest
process, always.** Every earlier job's harness happens to do that — `mb-qwen`'s
manifest format is one node id per line — but nothing in the tree says why, and
the cost of not knowing is a whole night.

The fix is a one-line change in shared 1D/2D code (`models/common/modules/lazy_weight.py`),
which is outside this job's mandate: a mesh of the same shape and mapper produces
the same tensor, so the fingerprint wants the mesh **shape**, not the instance id.
Reported, not changed.

### D-C4 — `from_pretrained` cannot build a contiguous KV cache, so area 1's headline gate is not expressible through the adaptor

**New, severity: contract gap. It also made one committed test a tautology.**

Both adaptors do

```python
paged = paged_attention_config or default_paged_attention_config(params)
```

so `paged_attention_config=None` does not mean "contiguous" - it means "give me
the default pool", `ceil(max_seq_len / 32) * max_batch_size` blocks. There is no
argument that yields `spec.paged_attention_config is None`, even though
`Attention2D`, `GalaxyPagedKVContract` and the model's own `kv_specs` all support
that state and the host suite exercises it.

Two consequences, both measured:

1. `test_*_paged_capacity_resolved_after_construction_serves_a_request` **failed**
   on `assert all(spec.paged_attention_config is None ...)` (`a2_02`). That is a
   true report of the gap, not a broken model.
2. `test_*_paged_and_contiguous_caches_agree` compared the default pool against
   an explicitly-constructed pool of **exactly the same geometry** - at
   `max_seq_len=2048`, batch 32, block 32, both are 2048 blocks. It would have
   passed at PCC 1.0 while proving nothing about paged addressing.

Attempt 2 rewrote both rather than leaving a green tautology:

* `test_*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` runs the
  same 32 requests through a 2048-block and a 4096-block pool - which gives every
  slot a different run of block ids - and compares prefill and decode logits per
  slot at PCC ≥ 0.99. It asserts `resolved is not None` with a message telling a
  future reader to restore the original comparison once D-C4 is fixed;
* the late-capacity case now asserts the *reachable* claim: the geometry
  installed at construction can still be replaced before anything is bound, is
  refused while bound, and can be replaced again after unbind.

**The gate line "paged fill during prefill, then decode reading the same blocks,
PCC ≥ 0.99 against the contiguous path" therefore cannot be met at this API**, and
that is the honest verdict rather than a green tick from a tautology.

**Where the contiguous path does exist**, for whoever fixes D-C4:
`models/common/tests/models/llama33_70b_galaxy/test_bringup_wh_galaxy.py` builds
one with `_contiguous_kv_cache(...)` and `model.set_kv_cache(...)` directly, and
`GalaxyDirectRunner` has a contiguous branch (`self.paged = False`, which then
requires `active_slots == max_batch_size`). So the missing piece is only an
adaptor argument — something like `paged=False` alongside
`paged_attention_config` — not a new mechanism.

### D-C5 — the column user selector cannot accept Qwen's decode logits: its matmul requires an INTERLEAVED input B

**New, severity: correctness-blocking for device sampling on Qwen.** Added by
`mb-coverage` attempt 3 from attempt 2's `a2_g23_qwen_demo_sampling.log`; attempt
2 measured it and was cut off before writing it up.

`GalaxyColumnUserSelector.__call__` (`models/common/models/galaxy/collectives.py:445`)
is a single `ttnn.matmul(self.selector(), tensor, …)`: an identity-matrix selector
against the decode logits. `ttnn.matmul` with the default (multi-core) program
config requires **input B interleaved**
(`matmul_device_operation.cpp:1233`), and Qwen's decode logits arrive
**WIDTH_SHARDED**:

```
TT_FATAL: MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED,
          got: TensorMemoryLayout::WIDTH_SHARDED
  models/common/models/qwen3_32b_galaxy/model.py:1810  in sample_decode
  models/common/models/qwen3_32b_galaxy/model.py:1793  in select_decode_column_users
  models/common/models/galaxy/collectives.py:445       in __call__
```

The selector's own `memory_config` default is `DRAM_MEMORY_CONFIG`, so the
constraint is on the *incoming* tensor, which the selector neither checks nor
converts. Its only guard is a shape check (`[1, 1, max_batch_size, W]`); memory
layout is unvalidated, so the failure surfaces as a `TT_FATAL` from inside
`ttnn` rather than as a contract error naming the caller.

Two things make this a *2D-module* finding rather than a Qwen one:

* the selector is shared Galaxy code (`collectives.py`), not model code, and its
  contract is silent about the layout it accepts;
* it is reached only through `model.sample_decode`, so **every** device-sampling
  claim for Qwen is behind it — greedy-vs-host-argmax, the padded vocabulary, the
  seeded slots and the heterogeneous controls alike.

The host-sampling half of the same test ran first and passed, which localises the
fault to the device path.

**What it needs**, for whoever owns `collectives.py`: either the selector accepts
a sharded input B (an `interleaved_to_sharded`/`sharded_to_interleaved` at the
boundary, or a matmul program config that takes it), or `sample_decode` states the
layout it requires and the model converts before the call. Both are runtime
changes, so attempt 3 reports rather than makes them.

### D-C6 — Qwen's concat-32 prefill program does not fit in L1 at all

**New, severity: limitation, and it is a capacity result rather than an
ownership one.** Added by attempt 3 from attempt 2's
`a2_g22_qwen_demo_concat32.log`.

```
TT_THROW: Statically allocated circular buffers on core range [0-0 - 2-3]
          grow to 1669312 B which is beyond max L1 size of 1499136 B
  tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region
  models/common/models/galaxy/direct_runner.py:484  in prefill_batched
  models/common/models/galaxy/direct_demo.py:69     in run_direct_demo
```

This is **not** the L1 limitation L1/G-C\* family. Those are address collisions —
"static circular buffers … *clash with* L1 buffers … L1 buffer allocated at
544832" — which depend on what a previous phase left allocated. This one is the
sum of the program's own static circular buffers exceeding the whole 1499136 B of
L1 on a 3×4 core range, **by 170176 B (11%)**, which is a property of the resolved
concat-32 prefill recipe alone and cannot be fixed by teardown ordering.

The distinction matters for scheduling Milestone C: L1's ownership redesign will
not make this case pass. Qwen's concat-32 prefill needs a smaller resolved recipe
(fewer or smaller CBs, or a narrower core range per stream) before it can run at
all, at any length.

**One thing this leaves open**, and attempt 3 queued it: the failure was observed
in the *second* `run_direct_demo` of the demo test, so it has not yet been
separated from "after a decode". If `test_qwen_concat32_matches_sequential_prefill_at_each_length`
fails the same way in a fresh model with no preceding decode, the finding is
unconditional; if it passes, the capacity overflow is history-dependent after all
and D-C6 collapses into the L1 family.

### L1's **address clash** is Llama-specific at this tree; Qwen fails the same two demo shapes for two unrelated reasons

**New, and it contradicts an inherited claim.** `mb-qwen` attempt 2's handoff
says of L1's remaining half — prefill after a decode — *"Untouched, inherited,
**identical for both models**."* Measured here, it is not:

| Test shape (two prefill phases with a decode between them) | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` | **FAIL**, `program 100` clashes on `[0-0 - 0-3]` (`a2_g6`) | **PASS**, no clash (`a2_g17`) |
| `*_batch32_slots_are_isolated` | **FAIL**, same signature (`a2_g7`) | **PASS**, no clash (`a2_g18`) |
| `demo.py::*_concat32_prefill_matches_sequential` | **FAIL**, `program 1552` clashes on `[0-0 - 6-9]` — the whole grid (`a2_g10`) | **FAIL**, but *not* an address clash: static CBs on `[0-0 - 2-3]` **grow to 1669312 B against a 1499136 B L1** (`a2_g22`) — a capacity overflow, **D-C6** |
| `demo.py::*_device_sampling_matches_host_greedy` | **FAIL**, `program 100` (`a2_g11`) | **FAIL**, and not L1 at all: the column user selector matmul refuses a WIDTH_SHARDED input B (`a2_g23`) — **D-C5** |

Both Qwen results were taken in fresh single-node-id processes and re-run to
three (`a2_L1_qwen_*_run2/3`); the Llama failures are four independent
reproductions in four different tests.

**Read the last two rows before the first two.** This section was written at
02:49 UTC against the first two rows only, when the heading said *"L1's remaining
half is Llama-specific"*. The Qwen cells then completed, and both are failures —
so the claim as first written is too strong and attempt 3 narrowed it, heading
included. What survives is precise and still useful:

* the **L1 address clash** — the `clash with L1 buffers on core range …, L1 buffer
  allocated at 544832` signature — is **Llama-only at this tree**: 4 reproductions
  in 4 Llama tests, 0 in 6 Qwen runs of the two shapes that reproduce it for
  Llama;
* but **Qwen is not clean on the two demo shapes**. It fails
  `*_concat32_prefill_matches_sequential` on an L1 **capacity** overflow (D-C6) and
  `*_device_sampling_matches_host_greedy` on a **matmul layout contract** (D-C5).
  Neither is an address collision, neither depends on a preceding decode as far as
  the evidence goes, and neither would be fixed by the teardown-ordering work L1
  points at.

So the honest one-line version is: *the address clash is a property of Llama's
resolved geometry, and the two-prefill-phase demo shapes are unreliable on both
models for three distinct reasons.* Qwen is still the differential reference L1
needs — it runs the two `*_repeated_requests*` / `*_batch32_slots*` shapes clean
3/3 — but it is not a clean bill of health for the concat-32 or sampling paths.

**Why this matters more than a green tick.** The clash is an address collision —
`L1 buffer allocated at 544832 and static circular buffer region ends at …` — and
Qwen's decode placements are narrower than Llama's (residual on 10 cores against
16, `local_dim` 1280 against 2048, and a 40-core LM-head reduction against 42).
So the failure is a function of *how much L1 the decode mode leaves below the
prefill program's static CB region*, not of the mechanism being absent. That
gives Milestone C something it did not have: **a working reference configuration
on the same silicon**, which turns "why does prefill-after-decode clash" from a
one-sided debugging problem into a differential one.

It also means the limitation cannot be stated as a property of the 2D modules. It
is a property of a *resolved geometry*, and the next model added to this stack may
land on either side of it with nothing in the contract to warn it.
### The command behind each exit-gate line

All of them under `HF_HOME=/localdev/ctr-apbernal/hf_data`, one pytest process at
a time, through `cov_run3.sh`:

```sh
L=models/common/tests/models/llama33_70b_galaxy
Q=models/common/tests/models/qwen3_32b_galaxy

# Llama teacher-forced, batch 1, prefill 512 / decode 511
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# Qwen teacher-forced, batch 1, 512
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# batch-32 direct demos, no cross-slot contamination
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
# batch-1 4K / 32K / 128K functional smokes
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke   # 4k, 32k, 128k
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_long_context_smoke     # 4k, 32k, 128k
# prefix-cached output matches uncached execution
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_prefix_cached_prefill_matches_uncached
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached
$L/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches      # and the decode after it
$Q/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches
```

Host, device-free:

```sh
# no dependency imports from a model-named implementation package
grep -rnE '^\s*(from|import)\s+models\.(demos\.llama3_70b_galaxy|common\.models\.(llama33_70b|qwen3_32b)([^_]|$))' \
    models/common/models/galaxy models/common/modules models/common/models/*_galaxy
# zero changes to 1D module implementation files, and to llm_runtime
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'
# existing 1D model contract and demo-contract host tests
bash tttv2_milestone_b_evidence/coverage/cov_1d_contract_gate.sh <log>
```

## What attempt 2 committed

Tests, evidence and two docstring corrections. **No implementation file, in any
package.** Both boundary greps stay empty and the model-named import gate stays
at zero.

```text
models/common/tests/models/galaxy/test_step7_page_table_placement_wh_galaxy.py   new, 3 device cases
models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   +1 case (x3 policies), docstring
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     docstring, `_distinct_rows` fallback
tttv2_milestone_b_evidence/coverage/                                            logs2/, RESULTS_A2.md, this section
```

Three test-level changes, and the reason for each:

1. **`test_llama_no_padded_vocabulary_id_is_ever_sampled`** — the case F-C1 said
   was vacuous. It is not; Llama pads 768 ids.
2. **`test_step7_page_table_placement_wh_galaxy.py`** — the one host assumption
   attempt 1 flagged as needing a mesh, as a test rather than a one-off script,
   because D-C1's write-up depends on it.
3. **`_distinct_rows` cyclic fallback** — the reference file holds 1024 tokens, so
   the straight window walk *skipped* every concat-32 length ≥ 1024, which are
   exactly the lengths the brief asks for last. A skip is not a result. The
   exact-window path is untouched, so results taken before the change are
   comparable.

None of these relaxes a threshold, a tolerance or a parametrization; (3) widens
one.

## What Milestone C inherits from this job

* **L1's remaining half — prefill after a decode — is now costed.** Five step-7
  cases cannot be measured behind it, and the list is in §A2's L1 section with
  the one untried hypothesis (confine the prefill mode plan to worker cores) and
  the one new fact that narrows it (the prefill matmuls are already
  worker-confined, so the clashing program is a collective or the MLP ring form).
* **D-C1** — decode's page-table validator cannot separate the prefill layout
  from a legitimate L1 repeat, and the premise is now confirmed on silicon. The
  fix requires changing a 2D-module expectation, which two attempts have now
  declined as a boundary violation. It needs a decision, not a patch.
* **D-C2** — is a sampling seed per-request or per-(request, slot)? A product
  decision about the serving contract.
* **G-C1, G-C2, G-C3, F-C2** — unchanged from attempt 1.
* **The device weight cache is unbounded.** Staging Llama's full interleaved and
  ring weight sets at this commit wrote **138 GB** in 26 minutes, on a filesystem
  with 1.0 TB free and 95% used. A step-7 sweep that resolves many recipes is a
  disk-capacity question as much as a device-time one.

---

# §A3 — attempt 3, the completing pass

Written 2026-08-28 by `mb-coverage` **attempt 3**, unattended, on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Run directory
`tttv2_milestone_b_runs/20260828T073724Z`.

**Attempt 3 ran as two agent invocations inside one driver run, and this section
is written by the second.** The first started at `07:37:58Z` at commit
`af589dff4d5`, committed `2061c126743` at `07:59Z`, and ended at `08:16:43Z`; the
driver relaunched immediately. The device queue it had started
(`cov_queue.sh`, PID 13308, reparented to init) **never stopped** — it dequeued
`a3_l_greedy` at `08:16:44`, one second after the relaunch — so the mesh was never
idle and no run was lost or repeated. The second invocation adopted that queue
rather than restarting it, re-prioritised `queue.txt` under it, and added commits
`6df3c4a14a3` and `152d4c49efb`. Every run below names the commit its log is
stamped with, and the two runs whose stamp does not match their source say so
explicitly.

Everything above this line is attempts 1 and 2 and is left untouched, except for
the `@@…@@` cells in §A2 that attempt 2 was cut off before it could fill: those
were resolved from the logs they were waiting on, and §A2 says so. **This section
is the final verdict and supersedes both where they disagree.**

## What attempt 3 inherited, and what it verified before planning

Attempt 2's handoff was a hand-written bridge, not the job's own account, so the
first thing this attempt did was check its claims against the tree.

| Inherited claim | Verified at `af589dff4d5` | How |
| --- | --- | --- |
| The mesh is alive | **True.** `ls /sys/class/tenstorrent \| wc -l` = 32, `/dev/tenstorrent` = 32, and `test_partition_wh_galaxy.py` opened a real 8×4 cluster: **5 passed in 13.66s** | `logs2/a3_00_mesh_health.log` |
| `HF_HOME` must be exported as `/localdev/ctr-apbernal/hf_data`; the inherited value is empty | **True.** `echo "[$HF_HOME]"` → `[]` in this job's own environment. Every harness script exports it | `cov_run3.sh`, `cov_device_run.sh` |
| 51 logs and 33 machine verdicts from attempt 2 are on disk | **True**, and re-derived rather than trusted: a watcher re-read every `logs2/*.log` and extracted its own pytest summary line. 38 rows agree with `RESULTS_A2.md` | `VERDICTS_A3.txt` |
| `queue.txt` holds 40 pending items, none of them run | **True** for 38 of them. **Two were already done**: `a2_L1_qwen_repeat_run3` (`1 passed in 458.26s`, `exit=0`) and `a2_L1_qwen_batch32_run3` (`1 passed in 175.50s`, `exit=0`) both completed at 03:38 and 03:42 and were never written down anywhere. Attempt 3 dropped them from its queue rather than pay for a fourth run | `logs2/a2_L1_qwen_*_run3.log` |
| `a2_L1_llama_repeat_run3` was in flight when the host went away | **True.** Its log stops inside `Loading weights: 100%` with no verdict and no `exit=` line | `logs2/a2_L1_llama_repeat_run3.log` |

**One inherited claim was wrong and it is worth naming**, because it is the kind
of error that costs a night: the bridge said the two Qwen run-3 verdicts did not
exist. They did. `RESULTS_A2.md` — written one row at a time precisely so it
would survive a kill — stops one row before the end, and the bridge was written
from `RESULTS_A2.md`. The logs are the record; the index of the logs is not.

## The one thing that makes attempt 2's numbers usable

The brief says: *re-measure the accuracy numbers at this tree, do not quote
them*, because "evidence collected at a tree that has since moved is not
evidence". That applies to attempt 2's own numbers as much as to `mb-llama`'s, so
attempt 3 established exactly how far the tree has moved:

```sh
git diff --stat 718997518ab..HEAD -- models/     # empty
git diff --stat 1451b192584..HEAD -- models/     # only the two test_step7_coverage_wh_galaxy.py files
```

* **every attempt-2 log stamped `718997518ab`** — which is all of `g2`…`g23`, the
  `L1_*` re-runs and the placement re-runs — was produced against source
  **byte-identical to `HEAD`** under `models/`. There is nothing to re-measure for
  those rows; they *are* measurements at this tree;
* the four logs stamped `1451b192584` (`a2_01`, `a2_01b`, `a2_02`, `a2_03`) sit two
  commits back, and both commits touched only
  `models/common/tests/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/test_step7_coverage_wh_galaxy.py`
  — a file `test_full_model_wh_galaxy.py` neither imports nor shares a fixture
  with. The Llama accuracy figure in `a2_01` is therefore also unaffected, and
  attempt 3 re-ran it anyway.

This is the difference between *quoting* an earlier number and *inheriting a
measurement whose tree you have proved identical*. Every row in the gate table
below says which of the two it is.

## The Milestone B exit gate — final table, measured

**This table supersedes §A2's.** Every number in it was produced by a command in
"Gate commands, §A3" below, on this machine, against source identical to `HEAD`
under `models/`. Nothing is quoted from `mb-llama`, `mb-qwen` or attempt 1.

### What "measured at this tree" means for a log stamped with an older commit

The brief's instruction is *re-measure at this tree, do not quote*, and its reason
is Milestone A's lesson that evidence from a tree that has moved is not evidence.
Attempt 3 discharged that instruction by proving the tree has **not** moved under
the code any of these gates exercise:

```sh
git diff --name-only 718997518ab..HEAD -- models/
# models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py
git diff --name-only 1451b192584..HEAD -- models/
# models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py
# models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py
```

`models/` has exactly **one changed file** between the commit every gate log was
produced at and `HEAD` — and it is a *step-7 test file*, which
`test_full_model_wh_galaxy.py`, `demo.py` and every module under
`models/common/{models,modules}` neither import nor share a fixture with. So a
`718997518ab` gate log is not an older measurement of a changed thing; it is a
measurement of a byte-identical thing. That is the distinction the instruction
turns on, and every row below states which commit produced its number.

### The nine lines

| # | Gate line | Verdict | Measured value, and the log |
| --- | --- | --- | --- |
| 1 | Llama teacher-forced, batch 1, prefill 512 / decode 511 — top-1 ≥ 91%, top-5 ≥ 99% | **PASS** | top-1 **501/511 = 98.04%**, top-5 **511/511 = 100.00%**. `1 passed in 1029.52s (0:17:09)`. `logs2/a2_g1_llama_tf.log`, commit `1451b192584`; character-identical to `logs2/a2_01_llama_full_model_file.log`. 2 fresh processes |
| 2 | Qwen teacher-forced, batch 1, sequence 512 — top-1 ≥ 89%, top-5 ≥ 97% | **PASS** | top-1 **498/511 = 97.46%**, top-5 **511/511 = 100.00%**. `1 passed in 915.10s (0:15:15)`. `logs2/a2_g12_qwen_tf.log`, commit `718997518ab`. 1 fresh process |
| 3 | Batch-32 direct demos valid, no cross-slot contamination | **PASS** | Llama `logs2/a2_g9_llama_demo_batch32.log`, `1 passed in 277.69s`; Qwen `logs2/a2_g21_qwen_demo_batch32.log`, `1 passed in 153.47s`. 32 slots, each answering its own prompt, slot texts printed per slot; Llama slot 0 character-identical to the batch-1 demo (`a2_g8`). Commit `718997518ab` |
| 4 | Batch-1 4K / 32K / 128K functional smokes pass | **PASS** | Llama `a2_g3` 4K `357.81s`, `a2_g4` 32K `641.17s`, `a2_g5` 128K `721.70s`; Qwen `a2_g14` 4K `117.91s`, `a2_g15` 32K `136.29s`, `a2_g16` 128K `245.76s`. All `1 passed`, commit `718997518ab`, 1 run per geometry per model |
| 5 | Prefix-cached output matches uncached execution | **PASS** | Llama `a2_g2_llama_prefix.log` `1 passed in 424.35s`, Qwen `a2_g13_qwen_prefix.log` `1 passed in 158.58s` — two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99. Commit `718997518ab` |
| 6 | No dependency imports from an existing model-named implementation package | **PASS** for Milestone B, with one pre-existing exception named below | **0 matches** at `HEAD` for `models\.common\.models\.(llama33_70b\|qwen3_32b)` and **0** for `models\.common\.llm_runtime`, over all eight of Milestone B's own source and test directories. `models\.demos\.` matches **once**, and attempt 3 widened §A2's grep to find it: `models/common/tests/modules/moe/test_tt_moe_decode.py:33` imports three helpers from `models.demos.deepseek_v3`. It **exists unchanged at the job-0 base** `bc6ad03bfc2` (added upstream by `b705bc150e5`, "MoE: (towards) a configurable e2e decode module (#45041)"), is a *test*, and is nowhere on any Galaxy import path. Milestone B did not introduce it and does not depend on it — but the gate as written is not literally 0 over `models/common`, and `mb-signoff` should say so rather than assert a clean zero. Finding **F-C3** |
| 7 | Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → **0** of **384** changed paths, at `HEAD` |
| 8 | Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → **0** of **384**, at `HEAD` |
| 9 | Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and demonstrably not owned by Milestone B | **5 failed, 296 passed in 108.67s** (`logs3/a3_h1_1d_contract_gate.log`, commit `af589dff4d5`). The same five node ids attempt 1 and attempt 2 recorded, now at three different commits. **No expectation was edited.** Attribution: none of the five packages (`deepseek_r1_distill_qwen_14b`, `llama32_3b`, `llama33_70b`, `qwen25_7b`, `qwen2_7b`) appears anywhere in `bc6ad03bfc2..HEAD`, so Milestone B cannot be their cause |

Plus the host regression gate the brief's "Regression gates" section names, which
is not one of the nine but is the thing the nine sit on:

| Gate | Verdict | Measured |
| --- | --- | --- |
| `models/common/tests/modules` + `models/common/tests/models`, host selection | **PASS** | **553 passed, 0 failed in 139.00s** (`logs3/a3_h2_host_gate.log`, commit `af589dff4d5`). Host selection: the 2D module host suites, all of `tests/models/galaxy`, and the Llama host suite, with `--ignore-glob='*_wh_galaxy*.py'`. `test_plans.py` is excluded — finding **F-C2**, it needs a live cluster and this job holds it |
| `models/common/tests/llm_runtime` — the third directory of the brief's regression command | **PASS** | **1032 passed, 1 skipped, 0 failed in 223.39s** (`logs3/a3_h11_llm_runtime_host_gate.log`, commit `f319763439a`). **Never run by attempt 2 or by attempt 3's first invocation.** Verified device-free before running it beside the queue: 0 occurrences of `Opening user mode device driver` in the log |
| the brief's regression command **as literally written** | **not run, deliberately** | `pytest -q models/common/tests/modules models/common/tests/models models/common/tests/llm_runtime` collects `*_wh_galaxy*.py` and the 1D module device suites (`test_rope_1d.py`, `test_lm_head_1d.py`, `test_mlp_1d.py`, `test_attention_1d.py`, `test_rmsnorm_1d.py`, `test_embedding_1d.py`, the MoE suites), all of which **open a mesh**. Run whole, it would have taken the Galaxy out from under the step-7 queue. The three rows above are its host-safe partition plus `tests/models/galaxy`; what is left out is the 1D module *device* suites, and they are not one of the nine gate lines |

## Method, and what a "run" costs here

One serial device queue (`cov_queue.sh`), one pytest process at a time, never
piped. Each item is **one node id in its own fresh process** — not a choice of
style but a hard requirement of this stack, per finding D-C3: the device weight
cache is fingerprinted with `MeshDevice.id()`, so the second test in a pytest
process misses on all 965 Llama weights and pays 26 minutes and 138 GB to
re-stage them.

Between items the harness reaps any process still holding `/dev/tenstorrent`
(`cov_ensure_mesh_free.sh`, comm restricted to `python`/`python3`/`pytest` and
gated on the fd actually being open) and runs `tt-smi -glx_reset` after any
non-clean exit (`cov_after_device_run.sh`, 900 s cap — 600 s was measured too
tight for a wedged ARC controller).

Measured wall clock at this tree, warm disk cache:

| | |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer build, warm | ~5.5 min |
| Qwen 64-layer build, warm | ~2 min |
| Llama teacher-forced 512/511, whole process | ~18 min |
| Qwen teacher-forced 512, whole process | ~16 min |
| a `tt-smi -glx_reset` after a failure | ~2–4 min |

Two consequences that shaped the night's ordering:

1. a Qwen case costs roughly a third of the Llama case with the same shape, so
   Qwen coverage was run first — it buys more distinct claims per hour;
2. an item that fails costs its own wall clock *plus* a reset, so a block of
   expected failures is more expensive than a block of expected passes.

## Corrections to §A2's area map

Four rows of §A2's "which device case covers which area" table are wrong at this
commit and attempt 3 corrects them rather than reprinting the table:

| §A2 row | At `af589dff4d5` |
| --- | --- |
| area 1, "paged fill then decode, PCC ≥ 0.99 vs contiguous" → `*_paged_and_contiguous_caches_agree` | that test no longer exists. D-C4 made it a tautology and attempt 2 replaced it with `*_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`. **The brief's claim as written has no device case, because the contiguous path is unreachable through `from_pretrained`** |
| area 3, "a mix of both in one batch" → Llama only | **both models.** Attempt 3 wrote `test_qwen_prefix_cached_and_plain_requests_mixed_across_slots` |
| area 4, "per-slot heterogeneous top-k/top-p/temperature" → Llama only | **both models.** Attempt 3 wrote `test_qwen_per_slot_heterogeneous_sampling_controls` |
| repeat/cleanup, "two model constructions in one process" → "no device case" | there is one: `L/test_bringup_wh_galaxy.py::test_two_models_in_one_process`. Attempt 3 queued it |

### Three device cases attempt 3's second invocation added to the map

| Brief area | Claim it asks for | Device case, and why it is new |
| --- | --- | --- |
| 1 paged KV | paged fill then decode, PCC ≥ 0.99 vs the contiguous path | `{L,Q}/step7::*_paged_pool_logits_are_recorded_for_cross_process_comparison[default2048\|explicit4096]` plus the host-only `{L,Q}/step7::*_two_paged_pools_agree_across_processes`. **The claim as worded has no device case (D-C4) and its nearest reachable form has no *single-process* device case (D-C7)**, so the recording and the comparison are separate node ids. Same PCC threshold, same claim, one model per process |
| 4 sampling | all five area-4 claims at once, with D-C5 removed at the call site | `{L,Q}/step7::*_device_sampling_claims_behind_dc5_with_interleaved_logits`. A **diagnostic**, not a substitute gate: area 4 stays BLOCKED whatever it reports. It is what distinguishes "one memory-layout precondition" from "a memory-layout precondition and a sub-device core-set violation behind it" (D-C8) |
| regression | `models/common/tests/llm_runtime`, the third directory of the brief's regression command | no new test; the directory had simply never been run. Host-only, verified device-free first. See the gate table |

## Area by area, on silicon — attempt 3

`runs` is how many **fresh processes** the claim got *at this tree*, counting
attempt 2's logs where §A3's section head proved their source byte-identical to
`HEAD`. A claim with one run is **observed**, not qualified, and says so. A claim
with three identical *failures* is qualified in the other direction: three of
Milestone A's four defects presented as intermittent passes, so a failure that
repeats to the byte across fresh processes is not a race.

*This table was written as results landed. The agent session died at `09:21:44Z`
with several rows still `IN FLIGHT`; the queue ran on unattended until an
operator session halted it at `13:48:41Z`. **Every former `IN FLIGHT` row below
has since been filled from the logs by that operator session** — see the
provenance note in `RESULTS_A3.md`, which names the log behind each figure. No
row here is `IN FLIGHT` any more; rows that were never run say `NOT RUN` and name
where they still sit in `queue.txt`.*

### Area 1 — paged KV

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefill and decode page tables have the layouts D-C1 assumes | `a2_01b`, `a2_s34_placement_run2`, `a2_s35_placement_run3` | **3** | **PASS.** decode global `(32, 64)` → device-local `(8, 64)`; prefill `(32, 64)` → `(32, 64)`; ratio 4; both DRAM-interleaved. Identical all three runs |
| A prefill-shaped page table fed to decode is **rejected** | as above | **3** | **FAIL by design — D-C1.** `32 % 8 == 0` and both tables are interleaved, so `_validate_decode_page_table` cannot separate the prefill layout from a legitimate L1-sharded repeat. Needs a 2D-module expectation changed, so it needs a decision, not a patch |
| Paged fill then decode, PCC ≥ 0.99 **against the contiguous path** | — | — | **NOT EXPRESSIBLE — D-C4.** `from_pretrained(paged_attention_config=None)` installs the default 2048-block pool, not a contiguous cache. The brief's wording has no reachable form at this adaptor API |
| …its nearest reachable form: two *different* paged pools agree | `a3_q_pool_default`, `a3_q_pool_default_run2`, `a3_q_pool_explicit`, `logs3/a3_h12_pool_compare_committed_tree` | **2** per pool arm | **PASS for Qwen.** 2048-block against 4096-block, `[pool] all 32 slots agree at PCC >= 0.99 for prefill and decode`. Guard exercised: with either recording absent the comparison **fails** (`logs3/a3_h10_pool_compare_missing_guard`), so the pass is a comparison and not a no-op |
| …the same, in **one** process | `a3_q_two_pools` | 1 | **FAIL — D-C7.** The second model's `activate("decode")` cannot create its global circular buffer: 923776 of 1393472 B per L1 bank still allocated after the first model's `close()` and an explicit `gc.collect()`. This is what forced the cross-process split above |
| …the same two-pool comparison, **Llama** | `a3_l_pool_default`, `a3_l_pool_explicit`, `logs3/a3_h14_llama_pool_compare` | **1** per pool arm | **PASS for Llama.** Both recordings passed (888.58s, 693.56s, prefill/decode `(32, 128256)`) and the host comparison agrees: `[pool] all 32 slots agree at PCC >= 0.99 for prefill and decode`, `1 passed in 7.85s`. **Area 1's headline claim now passes for both models.** One recording process per arm, against Qwen's two — observed, not qualified |
| …the same, in one process, **Llama** | `a3_l_two_pools` | 1 | **FAIL, and not D-C7.** The Llama **address clash** arrives first — `program 100`, L1 buffer at 479296, CB region ends 630080 — so D-C7's capacity residue is *not observable on Llama*. Two defects, one shape, and only Qwen can see the second |
| Late capacity resolution — a cache bound after construction | `a2_02` (superseded), `a3_q_late_capacity`, `a3_q_late_capacity_run2`, `a3_l_late_capacity` | **2** Qwen, **1** Llama | **PASS both models.** Qwen 414.58s and 124.32s, Llama 543.91s; `[pool] as constructed: GalaxyPagedAttentionConfig(block_size=32, max_num_blocks=2048)`. `a2_02`'s earlier failure was **D-C4**, not the model: it asserted `paged_attention_config is None` after construction, which the adaptor never leaves true. The case was rewritten to the reachable claim and re-run |
| No cross-slot contamination in the blocks | `a3_q_cross_slot`, `a3_q_cross_slot_run2`, `a3_l_cross_slot`; and both demos' `*_batch32_has_no_cross_slot_contamination` | **2** Qwen, 1 per model for the demo | Demo form **PASS** both models (`a2_g9`, `a2_g21`). Block-level form: **PASS for Qwen**, 222.38s and 184.09s, two fresh processes. **BLOCKED for Llama** — `a3_l_cross_slot` died at 611.25s on the address clash (`program 100`, 544832) before any slot data was compared. Blocked, not contradicted |
| Transactional unbind, and a failed bind leaves no partial state | host suite (`G/test_step7_paged_kv.py`) | — | host **PASS**. The unwind is pure Python; no device case is needed and none was written |

### Area 2 — concat-32 physical prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Concat-32 agrees with sequential prefill, Llama, through the demo | `a2_g10` | 1 | **FAIL — L1 address clash**, `program 1552` on `[0-0 - 6-9]`, the whole 7×10 grid. The demo prefills, decodes, then prefills again |
| Concat-32 agrees with sequential prefill, Qwen, through the demo | `a2_g22` | 1 | **FAIL — D-C6**, and not the clash: static circular buffers on `[0-0 - 2-3]` sum to 1669312 B against 1499136 B of L1. A **capacity** overflow, 11% over, raised by `validate_circular_buffer_region` from `direct_runner.py:484` |
| Concat-32 agrees with sequential prefill, step-7 form, lengths 128 → 2048 | `a3_q_concat_len128`, `_len128_run2`, `_len256`, `_len512`; `a3_l_concat_len128`, `_len256`, `_len512`, `_len1024` | **2** Qwen at 128, 1 elsewhere | **FAIL, every length, both models — D-C6.** The step-7 form builds a model and prefills **once**, with no preceding decode, so it is the case that separates D-C6 from the L1 clash. It fires anyway, and it fires for **Llama** too: `1669312 B` at 128, `3111104 B` at 256, `5994688 B` at 512, `11761856 B` at 1024, against 1499136 B of L1 — **byte-identical between the two models at every shared length**. Length 2048 was dequeued and terminated by the operator at 13:48:41Z, un-measured and deliberately not re-queued |
| Padded rows change no active row's logits, active 16 / 31 / 32 | `a3_q_concat_active{32,16,31}`, `a3_q_concat_active32_run2`, `a3_l_concat_active{32,16,31}` | **2** Qwen at 32, 1 elsewhere | **NOT REACHABLE — D-C6.** All seven runs die with the identical `1669312 B` overflow before a single row's logits can be inspected. The brief's three active batches are not a dimension this hardware can distinguish at this tree: the program does not fit at any of them |
| Active batches 16 and 31 are not expressible as a smaller allocation | — | — | **G-C1**, host, unchanged from attempt 1 |

**Area 2 has no reachable case at this tree, for either model, at any supported
length or active batch.** D-C6 was recorded in §A2 as a Qwen-only capacity
overflow that Llama merely hid behind its address clash; the step-7 sweep shows
that reading was wrong in an important way. Llama produces the *same byte counts*
as Qwen — 1669312 B at length 128, doubling with length — which points at the
shared concat-32 recipe rather than either model's dimensions, and means the
smallest length the batched-prefill policy supports is already **11% over L1**
before any model-specific geometry enters.

### Area 3 — prefix-cached and chunked prefill

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Prefix-cached prefill matches uncached, Llama | `a2_g2` | 1 | **PASS** — two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99 |
| Prefix-cached prefill matches uncached, Qwen | `a2_g13` | 1 | **PASS** |
| Chunked prefill matches a single uncached prefill, and the decode after it reads what the chunks wrote | `a3_q_chunked`, `a3_q_chunked_run2`, `a3_l_chunked` | **2** Qwen | **PASS for Qwen**, 141.01s and 138.22s. **BLOCKED for Llama** — `a3_l_chunked` died at 353.38s on the address clash (`program 1546`, 543360). This is the chunk-aligned SDPA path that reads the paged cache, so the single-row page-table slicing the brief names is qualified on Qwen only |
| A prefix-cached request then a normal one | `a3_q_prefix_then_plain`, `_run2`, `a3_l_prefix_then_plain` | **2** Qwen, **1** Llama | **PASS both models.** Qwen 125.55s and 124.77s, Llama 320.22s |
| A mix of both in one batch | `a3_q_mixed_slots`, `_run2`, `a3_l_mixed_slots` | **2** Qwen, **1** Llama | **PASS both models.** Qwen 170.29s and 166.89s, Llama 386.08s. The Qwen case did not exist before attempt 3 wrote it |
| The `chunk_page_table` guard is unreachable | — | — | **G-C3**, host, unchanged |

### Area 4 — device sampling

**BLOCKED for both models, and measured rather than unmeasured.** Two stacked
defects in shared Galaxy code, the second only visible once the first is removed:

| Claim | Log(s) | Runs | Result |
| --- | --- | --- | --- |
| Device greedy sampling equals the host argmax, Qwen | `a2_g23` (demo), `a3_q_greedy` (step-7) | 2 | **FAIL — D-C5.** `collectives.py:445`, `Input B memory layout must be INTERLEAVED, got WIDTH_SHARDED` |
| Device greedy sampling equals the host argmax, Llama | `a2_g11` (demo, died earlier on L1), `a3_l_greedy` (step-7) | 1 for the sampler | **FAIL — D-C5, same frame, same assertion.** So the defect is not Qwen-specific and not an artefact of the demo path |
| …with D-C5 removed at the call site: greedy, padded vocabulary, D4's near-zero reciprocal temperature, seed repetition, per-slot heterogeneous controls | `a3_q_dc5`, `a3_q_dc5_run2`, `a3_q_dc5_run3` | **3** | **FAIL — D-C8.** The relocation works (`WIDTH_SHARDED → INTERLEAVED`, width 19200) and the same line then raises `Kernel group cores do not match sub device cores`. **None of the five claims could be evaluated**, because all five are behind the selector |
| The same diagnostic, Llama | `a3_l_dc5`, `a3_l_dc5_run2`, `a3_l_dc5_run3` | **3** | **FAIL — D-C8, identical, and now qualified.** `WIDTH_SHARDED width 16128 → INTERLEAVED` in all three (897.12s, 470.61s, 435.44s), then the same `TT_FATAL @ program.cpp:2205` from the same line. **D-C8 is deterministic at three fresh processes on both models**, so neither D-C5 nor D-C8 is geometry-dependent, unlike the L1 address clash |
| Per-slot heterogeneous top-k / top-p / temperature, since serving mixes them | `a3_q_heterogeneous`, `a3_l_heterogeneous` | 1 per model | **FAIL — D-C5**, 159.96s and 423.39s, the same `Input B memory layout must be INTERLEAVED, got WIDTH_SHARDED`. Both cases were written by attempt 3 for this brief line; neither can be evaluated until the selector works |
| Seeded slot **stability across slots** | host (`G/test_step7_sampling.py`) | — | **FAIL by design — D-C2.** `_seed_digest` mixes the slot in, so moving a request changes its stream. A product decision |
| Llama pads its vocabulary, so the padded-vocab gate is live | host, `recipes.galaxy_padded_vocab_size` | — | **F-C1 superseded.** 128256 → 129024 (768 ids); Qwen 151936 → 153600 (1664) |
| D4's reciprocal-temperature pairing, **on the host, by inspection**, since the device cannot reach it | source read at `HEAD` | — | **CORRECT.** `sampling_2d.py:213` writes `1.0 / call.temperature[index]` into the buffer and passes it as `temp=self._temperature` (line 384), so the module performs the inversion exactly once. Both host references divide: `sampling_2d.py:260` and `direct_runner.py:570` compute `torch.topk(row / T, k=k)`. And `direct_runner.py:531` hands the module the **raw** `policy.temperature`. Raw T in, one inversion inside, division on the host reference — the pairing the brief asked to be verified rather than assumed. This is a code reading, **not** the device measurement the brief wanted; that one is behind D-C5 and D-C8, and `test_*_a_near_zero_temperature_collapses_onto_the_host_argmax` at `T = 0.02` is written, committed and queued for the day the selector works |
| The composition has a device test that cannot see either defect | `G/test_column_user_selector_wh_galaxy.py` | — | It builds its input `DRAM_MEMORY_CONFIG` — the one layout the real model never produces — and loads no sub-device manager. Every module in the chain is green in its own suite; the chain does not run |

### Area 5 — long context

| Geometry | Llama | Qwen |
| --- | --- | --- |
| 4K | **PASS** `a2_g3`, 357.81s | **PASS** `a2_g14`, 117.91s |
| 32K | **PASS** `a2_g4`, 641.17s | **PASS** `a2_g15`, 136.29s |
| 128K | **PASS** `a2_g5`, 721.70s | **PASS** `a2_g16`, 245.76s |

One run each, commit `718997518ab`, which §A3's head proves is byte-identical to
`HEAD` under `models/`. Where the capacity goes: attempt 1's accounting (blocks
per user, pool size, KV bytes per device, RoPE table size, chunk count) predicted
~5.2 GiB per device for Llama at 128K against 12 GB and named fragmentation as
the risk; it fits, at 64 chunks of 2048 followed by a decode at position 131072.
**Qwen3-32B's `max_position_embeddings` is 40960**, so its 128K smoke runs three
times past the trained context and nothing in the stack refuses it —
`max_context_len` rides on the runtime config and is never checked against
`max_seq_len`. Functional, as the brief defines it; not a quality statement.

### Repeat and cleanup

| Shape | Llama | Qwen |
| --- | --- | --- |
| repeated requests, two runners, one live model | **FAIL 3/3**, byte-identical (`a2_g6`, `a2_L1_llama_repeat_run2`, `a3_L1_llama_repeat_run3`) — L1 address clash | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2/3`) |
| `*_batch32_slots_are_isolated` | **FAIL 1/1**, same signature (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2/3`) |
| **two model constructions in one process** | **FAIL** (`a3_l_two_pools`) — but on the **address clash** (`program 100`, 479296), not D-C7. `a3_l_two_models` **NOT RUN**; it is still in `queue.txt` | **FAIL** (`a3_q_two_pools`) — **D-C7**, and this is the shape the brief warned about |

See "L1, corrected" below: the address clash is Llama-only at this tree, the
capacity residue is not, and only the first of the two could yield to the
teardown ordering the brief suggests.

## Limitation L1, and repeat-and-cleanup — now qualified

The brief asks for repeated requests against one live model and repeated model
construction/teardown in one process, and warns that this is where L1 bites.

### The Llama address clash is deterministic, at three fresh processes

`test_llama33_70b_galaxy_repeated_requests_and_deterministic_cleanup` — the same
request twice through two `GalaxyDirectRunner`s on one live model — has now failed
**3/3 in three fresh processes** with a byte-identical message:

```
TT_THROW … Statically allocated circular buffers in program 100 clash with L1
buffers on core range [0-0 - 0-3]. L1 buffer allocated at 544832 and static
circular buffer region ends at 630080
```

| run | log | commit | verdict |
| --- | --- | --- | --- |
| 1 | `a2_g6_llama_repeat.log` | `718997518ab` | FAILED |
| 2 | `a2_L1_llama_repeat_run2.log` | `718997518ab` | FAILED |
| 3 | `a3_L1_llama_repeat_run3.log` | `af589dff4d5` | FAILED, 891.51 s |

That matters more than it looks. **Three of Milestone A's four defects presented
as intermittent passes**, which is why this project runs everything three times;
the same rule applied to a failure tells you the opposite thing — this is not a
race, not aliased L1, and not sensitive to what the mesh did before. The numbers
are the same to the byte across two commits and three processes, so the clash is a
function of the *resolved placement* and nothing else.

### The same rule says Qwen is genuinely clean on these two shapes

> **Read "L1, corrected" at the end of this section before quoting anything
> below.** This subsection was written at 08:02Z, before `a3_q_two_pools` ran.
> Its measurements stand — six Qwen runs of these two shapes, zero clashes —
> but its *generalisation*, that Qwen is a clean reference for L1, does not.
> Qwen fails L1 in the third shape the brief names.

| shape | Llama | Qwen |
| --- | --- | --- |
| `*_repeated_requests_and_deterministic_cleanup` | **FAIL 3/3** | **PASS 3/3** (`a2_g17`, `a2_L1_qwen_repeat_run2`, `a2_L1_qwen_repeat_run3`) |
| `*_batch32_slots_are_isolated` | **FAIL** (`a2_g7`) | **PASS 3/3** (`a2_g18`, `a2_L1_qwen_batch32_run2`, `a2_L1_qwen_batch32_run3`) |

Six Qwen runs of the two shapes that reproduce for Llama four times over, zero
clashes. **This is the single most useful thing this job hands Milestone C**: a
working reference configuration on the same silicon, in the same tree, through the
same shared modules. L1's "prefill after decode cannot recover the global circular
buffer" is not a property of `Prefetcher2D` as such — it is a property of how much
L1 a *particular* resolved decode geometry leaves free below the prefill program's
static circular-buffer region. Qwen's decode placements are narrower than Llama's
(residual on 10 cores against 16, `local_dim` 1280 against 2048, a 40-core LM-head
reduction against 42), and it fits where Llama does not.

Two things follow, and they are for the redesign rather than for this job:

1. **the debugging problem is differential, not one-sided.** The question is no
   longer "why does prefill-after-decode clash" but "what does Llama's decode leave
   at 544832 that Qwen's does not";
2. **nothing in the module contract warns a new model which side it will land on.**
   The next geometry added to this stack picks up L1 or does not, silently, and the
   only way to find out is to run two prefill phases on real silicon. That is a
   contract gap, not just a bug — and it is the input the brief asked for towards
   the Milestone B/C L1 ownership redesign.

### What the brief's suggested fix does *not* do

The brief says "if you hit it, the fix is teardown ordering". Two candidate fixes
have now been refuted on hardware rather than argued about:

* **`Prefetcher2DConfig.release_global_cb_on_prefill`** — implemented and refuted
  by `mb-llama` attempt 3. Dropping the last Python reference to a
  `global_circular_buffer` does not return its L1; the type has no `deallocate`;
* **"confine the prefill mode plan to the worker cores"** — refuted by
  `a2_g10_llama_demo_concat32`, where the clash is in `program 1552` on core range
  `[0-0 - 6-9]`, the **whole 7×10 grid**. The concat-32 prefill program spans the
  full grid, so a worker-core-only confinement cannot cover it.

Tearing the consumers down *before* the owner is still untested as a fix, and it
is not expressible through `GalaxyDirectRunner`'s context manager at this tree:
the runner owns the decode plan and the model owns the prefetcher, so a test
cannot order them without reaching inside the model. **That is the reduction the
redesign has to answer**, and this job records it rather than working around it.


## L1, corrected: Qwen is not a clean reference, it fails a different shape

`a3_q_two_pools` (finding **D-C7**) changes the conclusion this section reached
an hour before it ran, and the corrected version is narrower and more useful.

**What is still true.** The *address clash* — `Statically allocated circular
buffers in program N clash with L1 buffers on core range …, L1 buffer allocated
at 544832 and static circular buffer region ends at 630080` — reproduced 4/4 in
Llama runs and 0/6 in Qwen runs of the same two shapes, byte-identical across two
commits and three fresh processes. That asymmetry is real and it is still the
most useful differential this job hands Milestone C.

**What is not true.** "Qwen fits where Llama does not" was a statement about L1
in general, and L1 in general does not respect it. The brief names three
repeat-and-cleanup shapes; here is all three, per model, as measured:

| Shape | Llama | Qwen |
| --- | --- | --- |
| repeated requests, two runners, **one** live model | **FAIL 3/3** — address clash, `program 100`, `[0-0 - 0-3]` | **PASS 3/3** |
| `batch32_slots_are_isolated` — one live model | **FAIL 1/1** — same signature | **PASS 3/3** |
| **two model constructions in one process** | queued as `a3_l_two_models` / `a3_l_two_pools` | **FAIL** — `a3_q_two_pools`: OOM in `CreateGlobalCircularBuffer`, 923776 of 1393472 B per bank still allocated after the first model's `close()` and an explicit `gc.collect()` |

So the honest statement is: **L1 has two signatures, not one.**

1. *the address clash* — a prefill program cannot place its static circular
   buffers because a still-resident global CB occupies the sender cores. Depends
   on the resolved decode geometry, so it is Llama-only at this tree. Ordering is
   the plausible fix and it is the one the brief suggests;
2. *the capacity residue* — the L1 a closed model held is not returned to the
   allocator, so the **second** model in a process cannot create its global CB at
   all. Model-independent by construction: measured on Qwen, and Qwen is the
   model that does not clash. **No teardown ordering can fix this one** — the
   owner was not merely torn down in the wrong order, it was torn down
   completely, garbage-collected, and its L1 still did not come back.

Signature 2 is the one that matters for the redesign, because it is the one the
brief's suggested fix cannot reach and because it puts a hard "one model per
process" bound on this stack — which is exactly the bound finding **D-C3** puts
on it for a different reason (the weight cache is fingerprinted with
`MeshDevice.id()`). Two independent mechanisms, same operational consequence, and
between them they are why every device run in this job is one node id in one fresh
process.

## Findings, attempt 3

Attempt 1's seven and attempt 2's three stand as §A2 leaves them, except where
this section says otherwise. Attempt 3 escalates one, adds one, and closes one.

### D-C5 — **escalated**: the column user selector cannot accept *either* model's decode logits

§A2 records D-C5 as a Qwen failure. It is not model-specific, and the reason is
visible on the host without opening the mesh.

`GalaxyColumnUserSelector.__call__` (`models/common/models/galaxy/collectives.py:445`)
is one `ttnn.matmul(selector, tensor)`. The default multi-core matmul program
config requires **input B interleaved** (`matmul_device_operation.cpp:1233`). The
tensor it is handed is whatever `model.decode_forward` returned, and that comes
from `LMHead2D.decode_forward` with `decode_output_memcfg`, which both models set
from the *shared* Galaxy recipe:

```python
# models/common/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/model.py
decode_output_memcfg=decode.lm_head_output_memcfg
# models/common/models/galaxy/recipes.py:889
lm_head_output_memcfg=width_sharded_memory_config(padded_local_vocab, ring)
```

Resolved on the host for both geometries (`logs3/a3_h6_decode_placements_probe.log`):

| | Llama-3.3-70B | Qwen3-32B |
| --- | --- | --- |
| `lm_head_output_memcfg` layout | **WIDTH_SHARDED**, L1, 24 cores, shard `(32, 672)` | **WIDTH_SHARDED**, L1, 24 cores, shard `(32, 800)` |
| `residual_memcfg` cores | 16 | 10 |
| `local_dim` | 2048 | 1280 |
| LM-head all-reduce cores | 42 | 40 |

So the selector is fed a width-sharded tensor for **both** models, and the
`TT_FATAL` attempt 2 saw for Qwen is reachable for Llama by exactly the same route.
The only reason no log showed it for Llama is that Llama's demo path dies of the
L1 address clash at its second prefill, *before* it ever reaches the sampler
(`a2_g11`). Two independent faults, one hiding the other.

**Measured, not inferred.** The paragraph above was written from the host probe
at 08:06Z; `a3_l_greedy` then ran the Llama step-7 greedy case directly and
closed it on silicon (`logs2/a3_l_greedy.log`, `1 failed in 886.53s`):

```text
models/common/models/galaxy/direct_runner.py:527: in decode_sampled
models/common/models/galaxy/collectives.py:445: in __call__
E   RuntimeError: TT_FATAL @ matmul_device_operation.cpp:1233
E   MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED,
E   got: TensorMemoryLayout::WIDTH_SHARDED
```

Same frame, same assertion, same line as Qwen's `a3_q_greedy`. Both models, and
on the **step-7** path rather than a demo, so the fault is not an artefact of
the demo's two-phase shape either. D-C5 is a two-model, two-entry-point,
shared-code defect.

**And the fix has a precedent in the same file.** `collectives._relocate_sharded`
(line 122) already stages through `ttnn.sharded_to_interleaved(tensor,
ttnn.DRAM_MEMORY_CONFIG)` and documents *why* that op and not
`to_memory_config`: it runs on its input's own `shard_spec.grid`, so it stays
worker-confined under a loaded sub-device manager, whereas a generic reshard
builds over the full compute grid and is illegal there. So the one-line fix for
the selector is not a guess — it is the op two hundred lines above it, chosen
for exactly this constraint. Attempt 3 tested that claim on hardware rather
than asserting it; see `test_{qwen,llama}_device_sampling_claims_behind_dc5_with_interleaved_logits` in the area-4 table.

**Why this is a 2D-module finding and not a model one.** Both the selector
(`collectives.py`) and the LM head placement (`recipes.py`) are shared Galaxy code.
The selector's only guard is a shape check:

```python
if len(shape) != 4 or shape[-2] != self.max_batch_size:
    raise ValueError(f"column user selection expects [1, 1, {self.max_batch_size}, W], got {shape}")
```

Memory layout is unvalidated, so the incompatibility surfaces as a `TT_FATAL`
thrown from inside `ttnn` rather than as a contract error naming the caller — and
it surfaces only when someone composes the LM head with the sampler on a real
model.

**And that is the composition gap.** The selector *does* have a device test,
`models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py`,
including one called `test_column_user_selector_feeds_sampling_2d`. It builds its
input with

```python
memory_config=ttnn.DRAM_MEMORY_CONFIG        # interleaved
```

which is the one layout the matmul accepts and the one layout the real model never
produces. Every module in the chain is green in its own suite; the chain is broken.
This is precisely the class of defect the plan's per-module contracts cannot catch
and the reason step 7 exists.

**Consequence for the exit gate.** Everything in the brief's area 4 —
greedy-vs-host-argmax, the padded-vocabulary claim, seeded slot stability, the
near-zero-temperature check for defect D4, per-slot heterogeneous controls — is
behind `sample_decode`, hence behind this one matmul, for both models. See the
area-4 table for what that measured out as.

**What it needs**, and none of it is this job's to do: either the selector accepts
a sharded input B (a `sharded_to_interleaved` at the boundary, or a matmul program
config that takes width-sharded in1), or `sample_decode` declares the layout it
requires and each model relocates before calling. Both are runtime changes to
shared code. Reported, not made.

### D-C7 — **new**: closing a model does not return its L1, and the second model in a process cannot start

This is the finding attempt 3's second half was told to look for, and it is the
one that changes §A3's L1 story.

`a3_q_two_pools` (`logs2/a3_q_two_pools.log`, commit `2061c126743`, `1 failed,
2 warnings in 571.29s`) builds **Qwen** twice in one process, once per paged
pool, each inside its own `try/finally` that runs

```python
def _close(handle):
    try:
        handle.close()
    finally:
        del handle
        gc.collect()
```

The first pool completed — `[pool] default-2048: block_size=32
max_num_blocks=2048` at log line 331, a full prefill of 32 rows and a decode,
then `close()` and an explicit `gc.collect()`. The second model then **loaded
successfully** (`[pool] explicit-4096: block_size=32 max_num_blocks=4096`, line
11798) and died at its first decode:

```text
models/common/models/galaxy/direct_runner.py:543: in _decode_device_logits
    self.model.activate("decode")
models/common/models/galaxy/resources.py:363: in activate
    self._prefetcher.activate(mode)
models/common/modules/prefetcher/prefetcher_2d.py:431: in activate
    self._ensure_global_cb(context)
...
E   RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:462
E   Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70
E   banks, where each bank needs to store 792064 B, but bank size is 1393472 B
E   (allocated: 923776 B, free: 469696 B, largest free block: 373824 B)
```

**Read the numbers.** At the moment the second model asks for its global
circular buffer, **923776 of 1393472 bytes per L1 bank — 66% — are still
allocated**, with a largest free block of 373824 B against the 792064 B needed.
The first model had been closed *and* garbage-collected. One model alone fits:
`a2_g17`, `a2_g18`, `a2_L1_qwen_repeat_run2/3` and `a2_L1_qwen_batch32_run2/3`
all create exactly one Qwen model and all create the global CB without
complaint, 6/6.

**Why it is a finding and not a restatement of L1.** `Prefetcher2D.cleanup()`
already does everything Python can do: it stops the prefetch, deallocates every
retained resource, sets `self._global_cb = None` and clears `self._contexts`.
`mb-llama` attempt 3 showed that dropping the *last* reference does not return
the buffer's L1 mid-process. This measures the stronger statement — **the L1 is
not returned by full model teardown either**, and quantifies what is left behind.
Milestone A's limitation L1 is written as a prefill-after-decode ordering
problem; this says the residue outlives the owner entirely, which is a lifetime
problem, not an ordering one. No teardown ordering the brief suggests can fix a
buffer that the destructor of a closed object did not free.

**Why it matters more than the Llama clash.** §A3's L1 section, written earlier
in this attempt, concluded that "the address clash is Llama-only at this tree"
and offered Qwen as "a working reference configuration". That is still true of
the two shapes it was measured on, and it is **not** true of L1 in general:
Qwen hits L1 too, in the shape the brief names third — *repeated model
construction and teardown in one process* — with a capacity signature instead of
an address one. The corrected statement is in "L1, corrected" below.

**Consequence for the exit gate.** Area 1's headline claim, "paged fill during
prefill then decode reading the same blocks, PCC ≥ 0.99 against the contiguous
path", was already **not expressible** through the adaptor (D-C4). Its nearest
reachable substitute — two *different* paged pools compared against each other —
needs two models, and D-C7 says a process gets one. Attempt 3's answer is to
compare across processes; see "Area 1" below.

### F-C3 — the model-named import gate is not literally zero over `models/common`

§A2 reported "0 matches" for the brief's "no dependency imports from an existing
model-named implementation package" line. Attempt 3 widened the grep to
`models/common/tests/modules` and found one:

```text
models/common/tests/modules/moe/test_tt_moe_decode.py:33
    from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.test_optimized_moe_decode_block import (
        create_torch_dispatch_input_expert_scores_tensor,
        create_torch_dispatch_input_tensor,
        verify_output,
    )
```

It is **not Milestone B's**: it exists byte-identically at the job-0 base
`bc6ad03bfc2`, was added upstream by `b705bc150e5` ("MoE: (towards) a
configurable e2e decode module (#45041)"), is a *test* importing test helpers,
and is on no Galaxy import path — `git diff --name-only bc6ad03bfc2..HEAD` does
not contain the file at all. Milestone B's own verdict on this gate is a clean
**PASS**. But `mb-signoff` should state the exception rather than assert a bare
zero over `models/common`, because the next person to run the grep will find it
and will not know it is pre-existing.

### D-C8 — **new**: behind D-C5 the selector matmul violates the loaded decode sub-device's core set

This is why the diagnostic was worth a Galaxy quarter-hour.

`a3_q_dc5` (`logs2/a3_q_dc5.log`, commit `152d4c49efb`, `1 failed, 2 warnings in
156.06s`) relocated the decode logits exactly as D-C5's proposed one-line fix
would. **Three fresh processes, byte-identical**: `a3_q_dc5` 156.06s,
`a3_q_dc5_run2` 157.88s, `a3_q_dc5_run3` 154.84s, each printing the same
relocation line and raising the same `TT_FATAL`. On this hardware a passing run
proves nothing, and the same rule applied to a failure says the opposite: this
is not a race and not aliased L1, it is a function of the resolved placement.

```text
[dc5] greedy: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200;
      relocated to TensorMemoryLayout.INTERLEAVED
```

The `INTERLEAVED` assertion is gone; the call gets **further into the same
function** and then dies:

```text
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py:822: in sample
models/common/models/qwen3_32b_galaxy/model.py:1810: in sample_decode
models/common/models/qwen3_32b_galaxy/model.py:1793: in select_decode_column_users
models/common/models/galaxy/collectives.py:445: in __call__
E   RuntimeError: TT_FATAL @ tt_metal/impl/program/program.cpp:2205:
E                 num_intersections == num_cores
E   info:
E   Kernel group cores do not match sub device cores for programmable core type TENSIX
```

`collectives.py:445` is the same line as D-C5 — the bare `ttnn.matmul` — and the
new failure is one layer down: the program the matmul builds spans cores that are
**not in the loaded decode sub-device's core set**. Decode runs under a
sub-device manager (`Prefetcher2D._configure_mode`); a default multi-core matmul
program config resolves its grid from the tensors and the full compute grid, not
from the loaded sub-device, so the two disagree and `program.cpp` refuses the
program.

**So D-C5's fix is not one line, and the file already says why.** Two hundred
lines above the selector, `collectives._relocate_sharded` documents this exact
hazard for a *different* op:

> a direct `to_memory_config` between two shard specs that differ in grid **and**
> width resolves to `reshard_program_factory_generic`, which builds over the full
> compute grid and is illegal under a loaded sub-device manager.
> `sharded_to_interleaved` runs on its input's `shard_spec.grid` and
> `interleaved_to_sharded` on its output shard's cores, and both of those are
> worker-confined here.

The relocation was chosen for worker-confinement and it *is* worker-confined —
that part of the fix works, and `a3_q_dc5` is the hardware evidence. What is not
worker-confined is the **matmul that consumes it**. Making the selector's input
interleaved satisfies the matmul's memory-layout precondition and simultaneously
hands it a placement decision it makes over the wrong grid.

**The reduction, for whoever owns this.** `GalaxyColumnUserSelector` needs *both*:

1. an input B the matmul accepts — interleaved, or a program config that takes
   width-sharded in1; **and**
2. a program config whose core grid is inside the decode worker sub-device, the
   way every other decode-time op in `collectives.py` is.

Neither is expressible from a test, and (2) is the one no amount of relocation
reaches. `GalaxyColumnUserSelector.__init__` already accepts a
`compute_kernel_config` and a `memory_config` and passes both to the matmul; it
accepts no `program_config`, and nothing in it knows which sub-device is loaded.

**Both models, and that distinguishes D-C8 from the L1 clash.** `a3_l_dc5`
(`logs2/a3_l_dc5.log`, commit `75f47d1228e`, `1 failed in 897.12s`) relocated
Llama's decode logits — `WIDTH_SHARDED, width 16128`, which is Llama's per-device
share of its 129024 padded vocabulary — and raised the same `TT_FATAL @
program.cpp:2205` from the same line of `collectives.py`. Four device runs, two
geometries: the *only* thing that differs is the shard width. The L1 address
clash is a function of the resolved decode placement and reproduces for one model
and not the other; D-C5 and D-C8 are functions of the code and reproduce for
both.

**And this is the third fault in one stack of three.** The L1 address clash hid
D-C5 for Llama; D-C5 hid D-C8 for both models. The class's own docstring predicted
it in as many words —

> **Unqualified.** This composition has never run on a Galaxy mesh. Qualify it
> with the focused selector test before trusting a device sampling path built on
> it; the alternative is composing the logits to host and calling
> `Sampling2D.sample_host`.

— and the focused selector test it points at
(`test_column_user_selector_wh_galaxy.py`) builds its input with
`memory_config=ttnn.DRAM_MEMORY_CONFIG` **and no loaded sub-device manager**, so
it cannot see either fault. The alternative the docstring offers — compose to host
and call `sample_host` — is what both demos' passing half actually does, and it is
the only sampling path this tree has that works on a Galaxy.

**Consequence for the exit gate.** Area 4 is **BLOCKED**, for both models, by two
stacked defects in shared Galaxy code. Not "unmeasured": measured, twice, with the
first blocker removed at the call site to reach the second. Milestone B's device
sampling does not work end to end on this hardware at this tree, and the report
should say so in those words.

---

### D-C6 — **escalated**: the concat-32 L1 overflow is not Qwen's, it is the shared recipe's

*Recorded after the agent session ended, by the operator session that halted the
queue at `13:48:41Z`. Transcribed from `logs2/a3_{q,l}_concat_*.log`, all stamped
`commit: b361770f46b`; every byte count below is `grep`-ed from the log named.*

§A2 recorded D-C6 as a **Qwen-only** capacity overflow: Qwen's concat-32 demo
(`a2_g22`) could not fit its static circular buffers, while Llama's (`a2_g10`)
failed earlier with the L1 *address clash* and so never reached the question.
That reading was reasonable on the evidence available and it is **wrong**.

The step-7 sweep — a model built once and prefilled once, with no decode before
it, which is exactly the shape that cannot raise the address clash — gives:

| length | Qwen | Llama | L1 available |
| --- | --- | --- | --- |
| 128 | 1 669 312 B | **1 669 312 B** | 1 499 136 B |
| 256 | 3 111 104 B | **3 111 104 B** | 1 499 136 B |
| 512 | 5 994 688 B | **5 994 688 B** | 1 499 136 B |
| 1024 | *not run* | 11 761 856 B | 1 499 136 B |
| active 16 / 31 / 32, length 128 | 1 669 312 B | 1 669 312 B | 1 499 136 B |

all on core range `[0-0 - 2-3]`, all raised by `validate_circular_buffer_region`.

Three things follow, and each changes what Milestone C inherits:

1. **The requirement is identical between the two models, to the byte, at every
   shared length.** Llama's 8-KV-head 128256-vocab geometry and Qwen's 64-head
   151936-vocab geometry cannot both coincidentally need 1 669 312 B. The
   allocation is a property of the **shared concat-32 recipe**, not of either
   model's dimensions, so this is one defect to fix once — not a per-model tuning
   exercise.
2. **The smallest supported length is already 11% over.** The brief says "qualify
   sequence length 128 first, then expand through 2048". There is no first step:
   128 does not fit. And the requirement roughly **doubles per length doubling**
   (1.67 → 3.11 → 5.99 → 11.76 MB), so nothing above 128 is a near miss either —
   1024 asks for **7.8×** the L1 that exists.
3. **The active-batch dimension is unreachable, not passing.** The brief's whole
   area 2 turns on active batches 16, 31 and 32 behaving differently. All three
   produce the identical 1 669 312 B and die before a single row's logits can be
   compared. Nothing about padding-row isolation was measured — in either
   direction. Do not read these seven failures as evidence that padded rows leak;
   read them as evidence that the question cannot be asked at this tree.

Length 2048 was dequeued at `13:44:49Z` and terminated at `13:48:41Z` by the
operator, un-measured and deliberately not re-queued: the scaling and the
model-independence were both already established, and the run had no answer left
to give.

### The Llama address clash blocks three more claims than §A2 knew

Same provenance as D-C6 above. The clash (§A2's limitation **L1**, `program 100`,
core range `[0-0 - 0-3]`) is still **Llama-only at this tree** — `grep -l 'clash
with L1 buffers' logs2/a3_q_*.log` matches **0 of Qwen's 28 attempt-3 device
runs**, and 0 of its attempt-2 runs. But the Llama half of the step-7 sweep
shows it costs more coverage than the repeat/teardown shapes it was found in:

| Claim | Log | Signature |
| --- | --- | --- |
| area 1, block-level cross-slot isolation | `a3_l_cross_slot` | `program 100`, L1 buffer at 544832 |
| area 1, two pools in one process | `a3_l_two_pools` | `program 100`, L1 buffer at 479296 |
| area 3, chunked prefill | `a3_l_chunked` | `program 1546`, L1 buffer at 543360 |

The second of these matters for **D-C7**: Qwen's two-pool run failed with the
capacity residue (923776 of 1393472 B per bank still held after `close()` and
`gc.collect()`), which is the finding. Llama's fails *earlier*, on the clash, so
**D-C7 is not observable on Llama** and the "one model per process" limitation is
qualified on Qwen alone. Two different defects produce one symptom in the same
shape, and a fix for either one will not silence the other.

All three claims are **blocked, not contradicted**: no slot data, no pool
comparison and no chunk comparison was ever performed in those runs.

### Gate commands, §A3

Every device row above was produced by one line of `queue.txt` through
`cov_queue.sh` → `cov_run3.sh` → `cov_device_run.sh`, which is

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
timeout --signal=TERM --kill-after=180 "$MB_DEADLINE" \
  python -u -m pytest -v -rA --color=no -p no:cacheprovider \
    --timeout="$MB_PYTEST_TIMEOUT" "<one node id>" -o faulthandler_timeout=900 > "$LOG" 2>&1
```

never piped, one process at a time, with `cov_ensure_mesh_free.sh` before and
`cov_after_device_run.sh` (which runs `tt-smi -glx_reset` after any non-clean
exit) behind. The node ids, per gate line:

```sh
L=models/common/tests/models/llama33_70b_galaxy
Q=models/common/tests/models/qwen3_32b_galaxy

# 1  Llama teacher-forced, batch 1, prefill 512 / decode 511
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# 2  Qwen teacher-forced, batch 1, 512
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# 3  batch-32 direct demos, no cross-slot contamination
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
# 4  batch-1 4K / 32K / 128K functional smokes
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke[4k|32k|128k]
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_long_context_smoke[4k|32k|128k]
# 5  prefix-cached output matches uncached execution
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_prefix_cached_prefill_matches_uncached
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached
$L/test_step7_coverage_wh_galaxy.py::test_llama_chunked_prefill_matches_a_single_uncached_prefill
$Q/test_step7_coverage_wh_galaxy.py::test_qwen_chunked_prefill_matches_a_single_uncached_prefill
```

Host, device-free, and re-run at `HEAD` by attempt 3's second invocation:

```sh
# 6  no dependency imports from a model-named implementation package.
#    NOTE the directory list: it is wider than §A2's, which is how F-C3 was found.
DIRS="models/common/models/galaxy models/common/modules \
      models/common/models/llama33_70b_galaxy models/common/models/qwen3_32b_galaxy \
      models/common/tests/models/galaxy models/common/tests/models/llama33_70b_galaxy \
      models/common/tests/models/qwen3_32b_galaxy models/common/tests/modules"
grep -rnE '^\s*(from|import)\s+(models\.demos|models\.common\.models\.(llama33_70b|qwen3_32b)([^_]|$)|models\.common\.llm_runtime)' $DIRS
#   -> 1 match, models/common/tests/modules/moe/test_tt_moe_decode.py:33, pre-existing

# 7, 8  boundaries
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'      # 0 of 384
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'  # 0 of 384

# 9  existing 1D model contract and demo-contract host tests
bash tttv2_milestone_b_evidence/coverage/cov_1d_contract_gate.sh logs3/a3_h1_1d_contract_gate.log
#   and, for "expectations unchanged", the check that matters more than the run:
git diff --name-only bc6ad03bfc2..HEAD -- models/common/tests/models/ | grep -v galaxy   # empty

# the host regression gate the brief's "Regression gates" section names
python -m pytest -q models/common/tests/modules models/common/tests/models \
                   models/common/tests/llm_runtime \
                   --ignore=models/common/tests/models/galaxy/test_plans.py   # F-C2

# area 1's cross-process comparison, host only
python -m pytest -v $Q/test_step7_coverage_wh_galaxy.py::test_qwen_two_paged_pools_agree_across_processes
```

## What attempt 3 committed

Tests and evidence. **No implementation file, in any package**, either invocation.

```text
models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   +3 cases
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     +5 cases
tttv2_milestone_b_evidence/coverage/                                             logs2/, logs3/, this section
```

The five test-level changes, and why each is a measurement rather than an
accommodation:

1. **`test_qwen_prefix_cached_and_plain_requests_mixed_across_slots`** and
   **`test_qwen_per_slot_heterogeneous_sampling_controls`** — two cases the brief
   asks for by name that existed only for Llama. Written by attempt 3's first
   invocation.
2. **`test_{qwen,llama}_paged_pool_logits_are_recorded_for_cross_process_comparison`**
   and **`test_{qwen,llama}_two_paged_pools_agree_across_processes`** — area 1's
   headline claim, split across processes because D-C7 says a process gets one
   model. Same PCC threshold, same claim, one fewer model per process. The
   comparison **fails** rather than skips when a recording is absent, and refuses
   to run if both recordings report the same `max_num_blocks` — the exact
   tautology D-C4 created the first time this case was written. Both guards were
   exercised: `logs3/a3_h10_pool_compare_missing_guard.log`.
3. **`test_{qwen,llama}_device_sampling_claims_behind_dc5_with_interleaved_logits`**
   — the diagnostic that found D-C8. It removes D-C5 *at the call site, in the
   test*, and does not touch the product: area 4 stays reported as BLOCKED
   whatever it says. It is the reason this job can distinguish "device sampling is
   blocked by one memory-layout precondition" from "device sampling is blocked by
   a memory-layout precondition **and** a sub-device core-set violation behind
   it", which is a different conversation with whoever fixes it.

Nothing here relaxes a threshold, a tolerance or a parametrization, and no test
was deleted or `xfail`ed. Two tests were **added that fail**, on purpose, because
the thing they measure is broken.

## What Milestone C inherits from this job

Ranked by what a human has to decide, not by severity.

1. **Device sampling does not work end to end on this hardware, and it is two
   defects deep.** D-C5 (the selector matmul requires an interleaved input B;
   both models' decode logits are width-sharded, from the *shared* recipe) and
   D-C8 (with that satisfied, the matmul builds a program over cores outside the
   loaded decode sub-device). Both are in `models/common/models/galaxy/`, both are
   shared code, and the fix needs a program config the selector currently has no
   way to accept. Every claim in the brief's area 4 is behind them.
2. **L1 has two signatures and only one of them is an ordering problem.** The
   address clash is Llama-only at this tree and might yield to teardown ordering.
   The capacity residue (**D-C7**) will not: the owner was closed, dereferenced
   and garbage-collected and its L1 did not come back, so the second model in a
   process cannot create its global circular buffer. That puts a hard *one model
   per process* bound on the stack — the same bound **D-C3** puts on it from the
   other direction, via a weight cache fingerprinted with `MeshDevice.id()`.
3. **D-C1** — decode's page-table validator cannot separate a prefill-shaped
   table from a legitimate L1-sharded repeat. Premise confirmed on silicon by
   three fresh processes. The fix changes a 2D-module expectation, which three
   attempts have now declined as a boundary violation. It needs a decision.
4. **D-C4** — `paged_attention_config=None` installs the default pool, not a
   contiguous cache. Either the adaptor grows a way to ask for a contiguous cache
   or the plan's area-1 wording changes to the two-pool form this job measured.
5. **D-C2** — is a sampling seed per-request or per-(request, slot)? A product
   decision about the serving contract, not a bug.
6. **F-C3** — one pre-existing `models.demos` import sits in
   `models/common/tests/modules/moe/`. Not Milestone B's, and `mb-signoff` should
   name it rather than assert a bare zero.
7. **G-C1, G-C2, G-C3, F-C2** — unchanged from attempt 1.
8. **Scheduling reality, for whoever plans Milestone C's device nights.** One node
   id per process is mandatory (D-C3), a warm Llama build is ~5.5 min and a cold
   weight set is 26 min and 138 GB, and a failing run costs its own wall clock
   plus a `tt-smi -glx_reset`. A 17-node-id file is a three-hour run. Plan around
   builds, not around tests.

---

# §A3-op — what landed after the agent died, and who wrote it down

**Appended 2026-08-28 14:05Z by an operator session, not by `mb-coverage`.** It is
outside `cov_assemble_report3.sh`'s fragment list on purpose: the assembler's nine
fragments are attempt 3's own account, and this is not.

The attempt-3 agent session ended at **09:21:44Z** — its `stream-json` ends
mid-wait with its background monitors killed and no `result` event. Its detached
device queue (`cov_queue.sh`, reparented to init) kept running **unattended for
four and a half hours** and completed **38 further runs**, until the operator
halted it at **13:48:41Z** at the user's request. `ENVIRONMENT.md` §"Operator
intervention" has the timeline, the deadlock that caused it, and the two harness
fixes it implies. `RESULTS_A3.md` has one row per run, below its own provenance
note, with the log behind every figure.

## The exit gate did not move

**None of the nine gate lines is affected.** No gate line depends on concat-32,
block-level cross-slot isolation, or chunked prefill, and every gate log was
already on disk and stamped before the agent died. The table in "§A3 — the nine
lines" stands as written: eight pass, and line 9 fails 5 of 301 at node ids
Milestone B provably does not own. Nothing below changes a gate verdict; what
changes is what Milestone C inherits.

## Two verdicts changed

1. **D-C6 escalates from a Qwen quirk to a shared-recipe defect.** §A2 read it as
   Qwen-only, with Llama merely dying earlier on its L1 address clash. The step-7
   sweep — model built once, prefilled once, no decode before it, so the clash
   cannot fire — gives Llama the *same capacity overflow* at **byte-identical**
   figures at every shared length: 1 669 312 B at 128, 3 111 104 B at 256,
   5 994 688 B at 512, 11 761 856 B at 1024, against 1 499 136 B of L1. Two
   different model geometries cannot coincidentally need the same bytes: the
   allocation belongs to the shared concat-32 recipe. **Area 2 has no reachable
   case at this tree, for either model, at any supported length or active batch**,
   and the plan's padding-isolation risk was therefore never tested in either
   direction. Full write-up in §A3 "D-C6 — escalated".
2. **Area 1's headline claim now passes for both models.** The Llama
   cross-process pool comparison was run host-only from the two recordings the
   queue had already produced: `all 32 slots agree at PCC >= 0.99 for prefill and
   decode`, `logs3/a3_h14_llama_pool_compare.log`, `1 passed in 7.85s`, no device
   opened. That row is a **new measurement by the operator session**;
   `RESULTS_A3.md` H14 labels it as such. One recording process per arm for Llama
   against two for Qwen, so it is observed rather than qualified.

Also: the Llama address clash costs three more claims than §A2 knew
(`a3_l_cross_slot`, `a3_l_two_pools`, `a3_l_chunked`), and because it arrives
*before* the capacity residue, **D-C7 is not observable on Llama at all** — one
symptom, two defects, and fixing either will not silence the other.

## What this adds to Milestone C's inheritance

Insert between items 1 and 2 of "What Milestone C inherits from this job":

> **1b. Concat-32 physical prefill does not fit in L1 at any supported length —
> D-C6, and it is shared code, not per-model tuning.** The static circular buffer
> requirement is identical for both models and roughly doubles per length
> doubling, so the smallest length the batched-prefill policy supports is already
> 11% over and length 1024 asks for 7.8× the L1 that exists. Whoever owns the
> concat-32 recipe needs a smaller resolved allocation before any of the brief's
> area-2 questions — padded-row KV isolation, padded-row logit isolation, active
> batches 16/31/32 — can be asked at all. This is a prerequisite for the coverage,
> not a finding the coverage produced.

## The queue is stopped, and 28 items have never run

`queue.halt` is present, `cov_queue.sh` has exited, the mesh is free and reset
(32/32 boards, `Re-initialized 32 boards after reset`). `queue.txt` is consumed
destructively, so its remaining 28 lines are exactly what is left; its header
records what was dropped and why. `a3_l_concat_len2048` was terminated in flight
and deliberately not re-queued.

Of those 28, six are behind D-C5 and several more behind D-C6 — both now
qualified at three fresh processes. **The genuinely unmeasured work is the
repeat tail**: most Llama claims have exactly one run, so they are *observed, not
qualified*, and the brief's three-fresh-processes rule is unsatisfied for them.
That, and the two defects, is where a Milestone C night pays.
