
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
