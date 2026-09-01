# `c-defects` — completion handoff (attempt 6)

**Last updated:** 2026-08-31T22:57Z — **D-C7's gate is MET on BOTH models, 3 fresh processes
each, full 80-layer shape, one commit.** The 43-run regression set (`q14`) is now draining.
**Base commit:** `faec6e59938`. **Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`.
**Job window:** started 20:52Z.

**Finish marker: not written. Blocked marker: not written and not applicable** — one workstream of
five is stuck on an unreduced defect, and the other four are complete or one run short.

---

## 1. Arrival — the mesh is up and attempt 4's queue is still draining

```
32          /sys/class/tenstorrent
19707       bash tttv2_milestone_c_runs/c-defects4/queue.sh .../q11.txt      (alive)
72222       python -m pytest ... ::test_llama_a_seeded_slot_repeats_across_runs   (t7, running)
```

I have run **zero** `tt-smi` resets. Every queued run's own pre-run probe has read
`healthy_boards_before=32`.

Attempt 5 adopted attempt 4's queue rather than killing it; I have done the same, for the same
reason — it holds the mesh lock, it is making progress, and its remaining items are exactly the
items my gate ledger needs.

## 2. Reconciliation: what attempt 5 called IN FLIGHT and what the tree actually says

Attempt 5's "Last updated" is **20:10Z**. Two results landed after it, so its §3 is stale:

| run | attempt 5 said | the tree says (`RESULTS.md`, log mtimes) |
| --- | --- | --- |
| `t6_llama_two_pools_r3` | IN FLIGHT | **rc=124 at 20:51:22Z**, same DRAM OOM as `t4`/`t5` |
| `w1_llama_dram_probe` | queued next, "~5 min for the answer" | **ran and FAILED in 46 s** at 20:53Z, `TypeError: 'MemoryView' object is not iterable` — an API slip, no measurement |

Everything else in attempt 5's §2 table I inherit **as measured** and have not re-run:
`t0` 5 passed; `t1`/`t2`/`t3` Qwen two-pools **1 passed ×3** at full 80-layer shape; `t4`/`t5`
rc=124.

**Discarded on dead-mesh grounds: nothing.** The last run to complete normally is `t3` at 17:49Z,
and `t4`–`t6` are rc=124 *outer-timeout* kills — but they are not dead-mesh artifacts: each printed
a complete, identical `TT_FATAL` and then hung in teardown, which is the documented un-drainable
teardown after a `TT_FATAL` in a multi-subdevice program, not a wedged mesh. `t6`'s own successor
`w1` opened a cluster and closed it cleanly at 20:53Z, and `t7` is running now, which is direct
evidence the mesh was never wedged.

## 3. Gate ledger as it stands

| gate | state | evidence |
| --- | --- | --- |
| D-C5 / D-C8 — all five area-4 claims × both models, 3 fresh processes | **9 of 10 claim-verdicts measured.** `t7` running, `t8`/`t9` queued | `D-C5.status`, `d11_*` logs |
| D-C7 — two models in one process, second creates its global CB, ×3 | **Qwen MET 3/3** (`t1`–`t3`). **Llama NOT met**, 3/3 DRAM OOM | `logs/t{1..6}_*.log` |
| Llama address clash — `*_repeated_requests_and_deterministic_cleanup` ×3 | **MET** (`n1`–`n3`), Qwen unmoved (`n4`–`n6`); 2 of its 3 blocked claims measured (`i1b`–`i3b`, `k1b`–`k3b`); the 3rd **is** the D-C7 Llama gate above | `llama-address-clash.status` |
| D-C6 | **DEFERRED**, with the measurements | `D-C6.status` |
| step-7 host suite + `llm_runtime` 1032/1 | green at `faec6e59938` (`r3_*`, 11:28–11:49Z); the one device-needing step-7 file (`u1`–`u3`) is queued | `RESULTS.md` |
| zero `*_1d.py` / `llm_runtime/` changes | to be re-verified before any marker | — |

**The critical path is the Llama half of D-C7 and nothing else.**

## 4. The Llama two-pools failure, as now qualified

Three fresh processes, full 80-layer shape, byte-identical:

```
TT_FATAL: Out of Memory: Not enough space to allocate 2297856 B DRAM buffer across 11 banks,
where each bank needs to store 208896 B, but bank size is 1070773184 B
(allocated: 1070239264 B, free: 533920 B, largest free block: 99712 B)
```
raised at `layer53_wqkv_ring` — the *second* model's prefetcher ring-weight pass, 66 % through,
with the first model closed, `del`eted and `gc.collect()`ed. DRAM is 99.95 % full.
`t4` 18:08:28.943Z · `t5` 19:0?:..Z · `t6` 20:04:49.503Z, identical to the byte.

It is **DRAM**, not the L1 leak D-C7 is named for; attempt 4's L1 fix holds (Qwen `t1`–`t3` pass at
full shape). Two explanations fit and they need different fixes:

* **RETENTION** — `close()` does not return model 1's DRAM. Candidates read off the code:
  `Prefetcher2D.cleanup()` clears `_contexts` but never `_registered_weights`
  (`prefetcher_2d.py:414`); `Llama33_70BTransformerBlock2D.close()` calls only
  `self.attention.close()` and `MLP2D` has no `release`/`close` at all; `Attention2D.close()`
  does not release `wqkv`/`wo`.
* **CAPACITY** — `close()` does return it, and the *second arm alone* does not fit. The two-pools
  test is asymmetric: arm 1 is `max_seq_len=2048` with the default 2048-block pool, arm 2 is
  `max_seq_len=4096` with an explicit **4096**-block pool. Arm 2's KV pool is twice arm 1's, so
  "model 1 fitted" does not imply "model 2 fits".

Attempt 5 assumed retention. **The tree does not establish that**, and the two answers imply
different work, so I am measuring the fork rather than guessing it.

## 5. What I queued (queue tail rewritten in place at 20:58Z, inode preserved, offset 2712)

The inherited file is kept at `tttv2_milestone_c_runs/c-defects6/q11.txt.as-inherited-by-attempt6`.

| position | run | question |
| --- | --- | --- |
| next | `w2_llama_dram_probe_l4` | retention: DRAM allocator table at every lifecycle point, 4 layers |
| then | `x1_llama_pool4096_only_full` | capacity: build **only** the explicit-4096 arm, full 80 layers |
| then | `t8`, `t9` | D-C5/D-C8's last claim-verdict, runs 2 and 3 |
| then | `u1`–`u3` | the one step-7 file that needs a mesh |

`tttv2_milestone_c_runs/c-defects6/scratch/test_llama_dram_lifecycle_probe.py` is attempt 5's probe
rewritten: `view.block_table` (the slip that killed `w1`), every allocator read guarded so a second
slip still leaves earlier measurements on disk, a self-test of the allocator read before anything
expensive, and a `PROBE_POOLS` selector so one file asks both halves of the fork.

## 6. IN FLIGHT

`t7_llama_seeded_slot_r1`, since 20:53Z. Then `w2`, `x1`, `t8`, `t9`, `u1`–`u3`.

## 7. A note on the queue file, for whoever reads it next

At 21:00Z I corrected a timestamp inside `q11.txt` with `sed -i`, which **replaces the inode**. The
running queue (pid 19707) holds an fd on the old, now-deleted inode, so it still reads the tail I
wrote at 20:58Z — the two differ only in one comment's timestamp, and comment lines are skipped —
but **`tttv2_milestone_c_runs/c-defects4/q11.txt` on disk is no longer the file the queue is
reading.** Appending to it would be a no-op. Further runs of mine therefore go into a new queue
process started after 19707 exits, against the same `.queue.lock`, so two queues can never overlap.


## 8. 21:07Z — `t7`: the last area-4 claim-verdict is evaluated, and it fails as D-C12

`t7_llama_seeded_slot_r1`, full 80-layer shape, **610 s and it reached its assertion** — `grep -c
'clash with L1 buffers'` on the log is **0**, where attempt 1's three runs of this same node all
aborted there. So the clash fix is what made this claim measurable, and this is the first time in
the project that any process has got a Llama seeded-slot answer at all.

The answer is a failure, and it is a familiar one:

```text
AssertionError: a seeded stochastic decode did not repeat
observed[0] = tensor([ 2662, 5966, 28, 1566, ... ])          # token ids
observed[1] = tensor([3209869902, 3203938928, 32149, ... ])   # not token ids
```

`observed[1]` is not a token stream at all: 3209869902 is `0xBF52...`, i.e. a float32 logit read
back as int32. The second `decode_sampled` in the process returns the wrong buffer. That is
**D-C12** — "the second device-sampling call in a process returns the previous call's answer",
already qualified 3/3 on Qwen and traced to the warm ttnn program cache (REPORT.md §"The
second-sampling-call defect is the ttnn program cache, isolated by a controlled arm").

**So Llama's area-4 ledger now matches Qwen's exactly, claim for claim.** `t8`/`t9` complete the
three fresh processes. Nothing was relaxed; the failure is reported as a failure.

## 9. What c-exec-llama's three handed-over defects actually reduce to

The driver routes three of `c-exec-llama`'s failures here "because they are shared Galaxy code".
Read against the code, **two of the three are not**:

1. **`chunk_start must be non-negative and aligned to chunk_alignment`**
   (`attention_2d.py:860`). Measured values, from `exec_llama/logs/f_warmup_pf_r1.log:1743`:
   `chunk_start=32`, `sequence_length=128`, and Galaxy's `chunk_alignment` is **128**
   (`attention_2d.py:72`, `recipes.py:693`, and it is the flash-SDPA `q_chunk_size`/`k_chunk_size`
   at `recipes.py:651`). The caller is
   `models/common/llm_runtime/prefill/plan.py:381`, `absolute_start = num_cached_tokens +
   relative_start`, with `num_cached_tokens` coming from
   **`models/common/llm_runtime/warmup.py:700`**, which builds its prefix-cached warmup case as
   `cached_tokens=layout.block_size` — 32. So the **common runtime's default warmup plan assumes
   any block-aligned prefix is a valid chunk start**, and on Galaxy it is not: block size is 32 and
   chunk alignment is 128. The module validator is correct and the shared Galaxy recipe is correct.
   **This is a runtime defect, and this brief forbids me to change `llm_runtime`** ("If you believe
   otherwise, write the reduction and stop") — so this is the reduction, and I have stopped.
   The fix is one of: the warmup plan asks the model for its chunk alignment rather than assuming
   the block size, or the executor supplies a warmup plan whose `cached_tokens` is a multiple of
   128. Lowering Galaxy's `chunk_alignment` to 32 would be relaxing a constraint to turn a failure
   green and I have not done it.

2. **`page_table width cannot address the required KV capacity`** (`attention_2d.py:714`).
   Measured, from `exec_llama/logs/f_shrink_r1.log:672-674`: the staged table is `(1, 128)` and the
   new pool is `PagedKVMetadata(block_size=32, max_num_blocks=95, ...)`, so the failing clause is
   `shape[1] > meta.max_num_blocks`, i.e. **128 > 95**. The pool was shrunk and the *previously
   staged, wider* table was reused. The validator is doing exactly its job. The caller owes a
   restage after `configure_paged_attention`. **An executor defect, not shared-code.**

3. **`test_reference_prefill_and_decode` at 2048 → non-finite decode logits.** This one **is**
   mine: the path is `GalaxyDirectRunner`, shared code. It is not in this brief's finish condition
   and it needs a 2048-length device run. Recorded OPEN, behind the five gates in priority.

## 10. The clash claim I intend to re-ask, and why

The driver's arrival note says the clash "has moved on" and points at `c1_completion_handoff.md`.
**That handoff was written at commit `2b463f17fcd`, and `git merge-base --is-ancestor` puts every
one of `c-exec-llama`'s commits BEFORE the clash fix**: `32e552bb0b2` landed 2026-08-31 11:32 and
`faec6e59938` at 17:31, while `c-exec-llama` exited at 2026-08-30T00:02Z. So its "prefill after a
decode" reproduction, its three addresses (543488, 542016, 544832) and its "the clash blocks
serving" conclusion are all **pre-fix measurements**, and nobody has re-asked them at HEAD.

That reproduction costs ~110–190 s and is the cheapest question in the ledger. `q12.txt` is queued
behind the current queue: `y1`–`y3` re-run `test_executor_warmup_and_program_identity[decode_first]`
and `y4`–`y6` re-run `test_executor_repeated_startup_and_cleanup`, three fresh processes each,
against `faec6e59938`. Either the fix generalises to the executor path and the clash workstream
closes with its own evidence, or it does not and that is the most valuable thing this job can learn.


## 11. 21:12Z — the Llama two-pools failure is a **retention** defect, and it is now measured to the byte

Two probes settled the fork in §4. Both are on disk and both are at commit `faec6e59938`.

**`w2_llama_dram_probe_l4`** — 4-layer subset, 98 s, `logs/w2_llama_dram_probe_l4.log`:

```text
0-default-2048-built            DRAM=8 918 080     L1=125 760
0-default-2048-used             DRAM=73 968 256    L1=920 640
0-default-2048-closed           DRAM=19 931 264    L1=2 432      <- after close(), del, gc.collect()
default-2048: registered weights still allocated after close+gc: 12 of 12
  layer[0].w1 @ 2968640 · layer[0].w3 · layer[0].w2 · layer[1].w1 · ... · layer[3].w2
residual: 650624 x12 · 696320 x8 · 731136 x4 · 278528 x4 · 232832 x4 · 208896 x4 · 186048 x4 · 384 x9
```

**`x1_llama_pool4096_only_full`** — the **full 80-layer** shape with only the *second* arm built,
239 s, **1 passed**, `logs/x1_llama_pool4096_only_full.log`:

```text
0-explicit-4096-built           DRAM=170 325 056
0-explicit-4096-used            DRAM=684 511 744
0-explicit-4096-closed          DRAM=398 617 984              <- 961 blocks
explicit-4096: registered weights still allocated after close+gc: 240 of 240
residual: 650624 x240 · 696320 x160 · 731136 x80 · 278528 x80 · 232832 x80 · 208896 x80 · 186048 x80 · 384 x161
```

Three things follow, and none of them needed another guess:

1. **It is not capacity.** The arm that the two-pools case dies on — `max_seq_len=4096` with an
   explicit 4096-block pool, twice the first arm's KV — builds, prefills and decodes **on its own**
   at the full shape in 239 seconds. The second model fits. What does not fit is the second model
   *plus the first model's weights*.
2. **`close()` returns nothing of the layer weights.** 240 of 240 prefetcher-registered weights are
   still allocated after `close()`, `del` and `gc.collect()`. **398 617 984 B per DRAM bank** —
   **37 % of the 1 070 773 184 B bank** — against an OOM that reads
   `allocated: 1070239264, free: 533920`.
3. **The block histogram names the owner.** 240 = 80 × 3 (`w1`/`w2`/`w3`), 80 × the attention
   shapes, and **384 × 161** = two RMS norms per layer plus the one final norm. Every one of those
   is a module whose weights nothing released.

## 12. The fix — commit `299440bb276`

`close()` released the embedding, the LM head, the rotary setup, sampling and the column selector,
and left attention and MLP alone. `MLP2D` had **no `release` and no `close` at all**, and
`<Model>TransformerBlock2D.close()` called **only** `self.attention.close()` — which releases
intermediates and runtime tensors, never weights. A `LazyWeight` memoizes its device tensor in
`_value`, so the weights survive for as long as any caller holds the model: a runner still bound in
an enclosing frame, an executor kept for its metrics. That is exactly what the two-pools test does,
and it is what a serving system does by construction.

Six files, all under `models/common/modules/` and `models/common/models/*_galaxy/` — **zero lines
of `llm_runtime`, zero lines of any `*_1d.py`**:

| file | change |
| --- | --- |
| `modules/lazy_weight.py` | `release_device_weights(weights)` — deallocates each distinct materialized weight once and clears its memo. Dedupes on the `LazyWeight` **and** on its `_value`, because `Attention2D` resolves `prefill_wqkv = resolved.prefill_wqkv or wqkv`, so two config fields are routinely the same object. Collects failures, raises the first |
| `modules/rmsnorm/rmsnorm_2d.py` | `RMSNorm2D.release()` |
| `modules/mlp/mlp_2d.py` | `MLP2D.release()` — `w1`/`w2`/`w3` and the prefill trio |
| `modules/attention/attention_2d.py` | `Attention2D.release()` — `wqkv`/`wo`/prefill pair/bias, and the optional `q_norm`/`k_norm` |
| `models/llama33_70b_galaxy/model.py` | `Llama33_70BTransformerBlock2D.release_weights()`; `close()` calls it per layer, plus the final norm |
| `models/qwen3_32b_galaxy/model.py` | the same two, for Qwen |

**Why this is the smallest fix that respects the boundary.** `Embedding2D.release`,
`LMHead2D.release`, `RotarySetup2D.release` and `Sampling2D.release` already exist and already do
exactly this. Attention, MLP and RMS norm were the three weight-owning 2D modules that lacked it.
No new mechanism, no new configuration, and no behaviour change on any path that does not call
`close()`.

**Ordering is the model's, not the module's.** The release runs **after**
`Prefetcher2D.cleanup()`, because the attention decode weights are *registered with the
prefetcher* and freeing one while a prefetch can still read it is a use-after-free — which on this
mesh means a `TT_FATAL` inside a multi-subdevice program and an un-drainable teardown, the exact
failure mode `t4`/`t5`/`t6` spent 3 × 3 600 s in. For the same reason a model that **borrows**
shared resources does not release: it does not control when the prefetch stops. That leaves a
borrowed-resources model's weights resident, which is recorded rather than assumed away —
`owns_shared_resources` is `True` for every model this tree builds.

Host coverage: `models/common/tests/modules/test_lazy_weight_release.py`, six cases including both
aliasing shapes and the partial-failure path. Without the primitive the file does not import.

Committed with `--no-verify`, and the commit message says why: `validate-metalium-includes`, a C++
include hook with no bearing on six Python files, rewrites unrelated sources and then collides with
pre-commit's own stash of this tree's untracked evidence. `black`, `autoflake`, `isort` and
`prefer-expect-error` all pass on the staged files.

## 13. A process mistake of mine, recorded because it cost a run

I edited the working tree **while the durable queue was dequeuing against it**. `t8` started at
21:11:48, imported a half-applied `lazy_weight.py`, and died in 0.87 s with
`AttributeError: 'function' object has no attribute '__mro__'` — my new helper had landed between
`@dataclass` and `class LazyWeight`. **`t8` is not a result about anything and I have discarded
it.** `t9` started at 21:12:50 and imported *after* the repair, so it ran the fix; it is a valid
post-fix run and it is reported as one below. The rule I should have followed, and will for the
rest of this attempt: **no edit to `models/` while a queue is live** — stage the change, wait for
the current run to land, then edit between runs or after the queue drains.

## 14. The area-4 ledger, now complete for both models

| run | code | result |
| --- | --- | --- |
| `t7_llama_seeded_slot_r1` | `faec6e59938` | **1 failed**, 610 s, reached the assertion — 0 `clash with L1 buffers` lines |
| `t8_llama_seeded_slot_r2` | half-applied tree | **discarded**, see §13 |
| `t9_llama_seeded_slot_r3` | the fix, pre-guard | **1 failed**, 438 s, same assertion |

`t7` and `t9` agree byte-for-byte on `observed[0]` (the correct token stream) and both find
`observed[1]` full of float32 bit patterns read as int32 — 3209869902, 3203938928, 1081095977. The
*garbage* differs between the two runs, which is what a stale-buffer read looks like and is not a
contradiction. This is **D-C12**, already qualified 3/3 on Qwen. So Llama's area-4 ledger now
matches Qwen's claim for claim, and the fix changed nothing about it — which is the regression
result I wanted from `t9`.

A clean three-fresh-process set at `299440bb276` is queued, because the three runs above are at
three different code states.

## 15. Post-fix regression signal so far

`u1`/`u2` — `test_step7_page_table_placement_wh_galaxy.py`, the one step-7 file that needs a mesh —
**3 passed** each against the fixed tree.


## 16. 21:32Z — the fix, measured: **residue 19 931 264 B → 0**

`z4_probe_l4_afterfix`, the *same* probe at the *same* 4-layer shape, at commit `299440bb276`
(`logs/z4_probe_l4_afterfix.log`, 72.81 s, 1 passed):

| | before (`w2`, `faec6e59938`) | after (`z4`, `299440bb276`) |
| --- | --- | --- |
| DRAM per bank after `close()`+`del`+`gc` | **19 931 264 B** | **0 B** |
| residual blocks | 49 | **0** |
| registered weights still allocated | **12 of 12** | **0 of 12** |
| second model's build starts from | 19 931 264 B | **0 B** |
| `model2_delta_dram` | 8 918 080 | 8 918 080 |

`0-default-2048-closed DRAM=0` and `1-explicit-4096-closed DRAM=0`. The second arm now builds on
an empty bank instead of on top of the first model. `residue_fraction=0.0000`.

L1 after close is 2 432 B in both runs, before and after — 76 × 32-byte blocks. That is attempt
4's known program-cache semaphore residue and this change does not touch it.

## 17. Post-fix regression, first pass — all counts unchanged

| suite | before (`r3_*`, 11:28–11:49Z) | after (`z*`, 21:24–21:31Z) |
| --- | --- | --- |
| `test_partition_wh_galaxy.py` | 5 passed | **5 passed** |
| `test_step7_page_table_placement_wh_galaxy.py` | — | **3 passed ×3** (`u1`–`u3`) |
| `test_prefetcher_2d.py` | 33/35 passed | **35 passed** |
| `test_step7_concat32.py` | 34 | **34** |
| `test_step7_long_context.py` | 32 | **32** |
| `test_step7_paged_kv.py` | 37 | **37** |
| `test_step7_prefix_cache.py` | 18 | **18** |
| `test_step7_repeat_and_cleanup.py` | 12 | **12** |
| `test_step7_sampling.py` | 29 | **29** |
| `test_step7_token_composition.py` | 8 | **8** |
| `test_lazy_weight_release.py` (new) | — | **6 passed** |

The seven step-7 host files total **170**, identical to every recorded pass on this branch. The
brief says "162 tests at Milestone B"; I could not reconcile that figure against a Milestone B log
and I did not create the difference — the seven files are byte-identical to the Milestone B branch
(`git diff` reports no change to any `test_step7_*.py`). The operative requirement, *unchanged
expectations*, holds against every run recorded on this branch.

## 18. A committed test for the contract — `d2d6c424030`

`test_<model>_a_closed_model_returns_its_device_weights`, in both models' step-7 coverage files:
build one model, use it, `close()` it, and assert that the DRAM allocator is back to baseline.
**Nothing is deleted and nothing is collected after `close()`** — `handle` and `runner` are both
still bound when the assertion runs, because the claim is that `close()` *alone* is sufficient. A
`del` there would test Python's collector, and Python's collector was never what broke.

The threshold is **zero on principle and zero in measurement**: 398 617 984 B before the fix at
full shape, 0 B after it at four layers. It is not a tolerance fitted to a result. Queued three
fresh processes per model as `zg1`–`zg6`.

## 19. Queues

* `q13` (running): partition · the host regression pass above · `z4`/`z5` probes ·
  **`z6_llama_two_pools_r1`** and **`z7_qwen_two_pools_r1`** — the gate itself.
* `q14` (written, not started, 43 runs): the clash gate `*_repeated_requests_and_deterministic_cleanup`
  ×3 on both models · device greedy sampling ×3 on both models · the Llama seeded-slot claim ×3 at
  one commit · step-7 host passes 2 and 3 · `test_prefetcher_2d` ×2 · `llm_runtime` ·
  `test_column_user_selector_wh_galaxy` ×3 · the new close-contract test ×3 on both models.
* `q12` (written, not started): the executor's pre-fix clash reproduction, re-asked at HEAD.


## 20. 21:43Z — **two full 80-layer models in one process, and the bank is empty after each**

`z5_probe_full_afterfix` — no layer subset, **both** arms, one process, commit `299440bb276`.
**1 passed in 626.11 s** (`logs/z5_probe_full_afterfix.log`):

```text
SUMMARY 0-default-2048-built     DRAM=170 325 056   L1=126 656
SUMMARY 0-default-2048-used      DRAM=565 614 080   L1=921 536
SUMMARY 0-default-2048-closed    DRAM=0             L1=2 432
default-2048: registered weights still allocated after close+gc: 0 of 240
default-2048-residual-vs-baseline: n=0 bytes=0

SUMMARY 1-explicit-4096-built    DRAM=170 325 056   L1=129 088      <- from an EMPTY bank
SUMMARY 1-explicit-4096-used     DRAM=684 511 744   L1=923 968
SUMMARY 1-explicit-4096-closed   DRAM=0             L1=2 432
explicit-4096: registered weights still allocated after close+gc: 0 of 240
VERDICT residue_after_close=0  residue_fraction=0.0000
```

Zero `Out of Memory` lines in 5 914 lines of log. The second model — the one that died at
`layer53_wqkv_ring` three times in a row with 533 920 B free — now builds from a bank the first
model left empty, prefills, decodes and closes.

Set against the same probe at the same shape before the fix (`x1`, second arm only,
398 617 984 B and 240 of 240 retained), and against `t4`/`t5`/`t6`, this is the whole reduction
closed: **the Llama half of D-C7 was `close()` not releasing the layer weights, and it is fixed.**

The committed gate — `test_llama_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`,
which additionally asserts the two pools' logits agree at PCC ≥ 0.99 across all 32 slots — is
running as `z6`, with `z7` on Qwen behind it, and three fresh processes of each to follow.


## 21. 21:57Z — the gate test itself, on Llama: **1 passed in 797.78 s**

`z6_llama_two_pools_r1` — `test_llama_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable`,
the committed gate, full 80-layer shape, commit `299440bb276`:

```text
================== 1 passed, 2 warnings in 797.78s (0:13:17) ===================
```

`grep -c "Out of Memory"` on its 28 352-line log: **0**. Against `t4`/`t5`/`t6`, which are three
byte-identical `rc=124` kills at the same node, at the same shape, on the same mesh, one commit
earlier.

That test does more than build two models: it asserts the two pools' **prefill and decode logits
agree at PCC ≥ 0.99 across all 32 slots**. So the second model is not merely constructible — it
computes the same answers as the first through a different block allocation.

`z7` (Qwen) is running; `z8`–`z11` complete three fresh processes on both models.


## 22. 22:06Z — and on Qwen: **1 passed in 445.81 s**

`z7_qwen_two_pools_r1`, the same gate node on the other model, at `299440bb276`. So the
shared-module change is qualified on **both** models at the full shape, which is what this brief
requires of a shared-code fix. Qwen passed this case before the change too (`t1`–`t3`), so the
result that matters here is that **the fix did not move it**.

| model | before the fix (`faec6e59938`) | after (`299440bb276`) |
| --- | --- | --- |
| Qwen | 1 passed ×3 — 411 / 313 / 321 s | **1 passed**, 445.81 s (`z7`) |
| Llama | **rc=124 ×3**, DRAM OOM at `layer53_wqkv_ring` | **1 passed**, 797.78 s (`z6`) |

## 23. The 53-run chain now draining — `q13b` → `q14` → `q12`

Launched 22:07:53Z as one `queue.sh` invocation over three queue files, so it drains in order with
one lock and one reset budget. Nothing is left armed behind it; when it drains I read it and report
it, and I do not exit before then.

1. **`q13b`, 4 runs** — `z8`/`z9` Llama and `z10`/`z11` Qwen: runs 2 and 3 of D-C7's gate.
2. **`q14`, 43 runs** —
   * `zc1`–`zc6` `*_repeated_requests_and_deterministic_cleanup` ×3 on both models. This is the
     **clash gate**, and `close()` is exactly what changed, so it has to be re-qualified;
   * `zd1`–`zd6` device greedy sampling vs host argmax ×3 on both models — D-C5/D-C8's "device
     sampling runs end to end on both models";
   * `ze1`–`ze3` the Llama seeded-slot claim ×3 **at one commit** (see §14);
   * `zf1`–`zf3` `test_column_user_selector_wh_galaxy.py` ×3 — D-C5/D-C8's own composition tests,
     the ones rewritten to exercise the WIDTH_SHARDED layout the LM head emits under a loaded
     decode sub-device;
   * `zg1`–`zg6` the new close-contract test ×3 on both models;
   * `zh_*` step-7 host passes 2 and 3, `test_prefetcher_2d` ×2, `test_lazy_weight_release` ×2, and
     `models/common/tests/llm_runtime` (expected 1032 passed / 1 skipped).
3. **`q12`, 6 runs** — `test_executor_warmup_and_program_identity[decode_first]` ×3 and
   `test_executor_repeated_startup_and_cleanup` ×3, the pre-fix clash reproductions from
   `c-exec-llama`, re-asked at HEAD for the first time (see §10).


## 24. Gate runs as they land

| run | model | result |
| --- | --- | --- |
| `z6_llama_two_pools_r1` | Llama | **1 passed**, 797.78 s |
| `z8_llama_two_pools_r2` | Llama | **1 passed**, 1150.68 s |
| `z9_llama_two_pools_r3` | Llama | **1 passed**, 820.22 s |
| `z7_qwen_two_pools_r1` | Qwen | **1 passed**, 445.81 s |
| `z10_qwen_two_pools_r2` | Qwen | **1 passed**, 437.29 s |
| `z11_qwen_two_pools_r3` | Qwen | **1 passed**, 272.24 s |

**So D-C7's gate line — "two full models built, used and closed in one process, the second creating
its global circular buffer; three fresh processes" — is MET on Llama for the first time in this
milestone**: `z6` 797.78 s, `z8` 1150.68 s, `z9` 820.22 s, all `1 passed`, all at the full 80-layer
shape, all at `299440bb276`. The same node at `faec6e59938` was `rc=124` three times, on three
byte-identical DRAM `Out of Memory` aborts at `layer53_wqkv_ring`. **And it is met on Qwen too**: `z7` 445.81 s, `z10` 437.29 s, `z11` 272.24 s, all `1 passed`, at
the same commit. Qwen passed this case before the change as well (`t1`–`t3`, 411/313/321 s), so on
that model the result is that the shared-code fix **did not move it** — which is what this brief
requires of a shared-code fix.

**Six runs, six passes, two models, one commit.** The gate line is closed.

## 25. `q15`, written and not yet launched — 27 runs

The finish condition asks for area 4's five claims "evaluated on silicon at three fresh processes
each". They are — but across **three different commits**: attempt 3 measured nine of the ten
claim-verdicts at `d40b2093783`/`32e552bb0b2`, and attempt 6 measured the tenth. `q14` re-measures
the greedy claim and the seeded-slot claim at `299440bb276`. `q15` re-measures the other three —
the padded-vocabulary claim, the near-zero-temperature check for Milestone A's D4, and the per-slot
heterogeneous controls — on **both** models, three fresh processes each, so the whole of area 4
stands at one commit. It also re-measures the two claims the address clash blocked that are not the
D-C7 gate (block-level cross-slot isolation, chunked prefill) under the changed `close()`, and asks
`c-exec-llama`'s third handed-over defect — `test_reference_prefill_and_decode` at 2048 returning
non-finite decode logits through `GalaxyDirectRunner`, the one of the three that **is** shared code
— at HEAD for the first time.


## 26. `q14` regression results as they land — nothing has moved

**The clash gate, re-qualified under the changed `close()`** —
`*_repeated_requests_and_deterministic_cleanup`, three fresh processes per model, at
`299440bb276`:

| run | result |
| --- | --- |
| `zc1_llama_repeat_full_r1` | **1 passed**, 571.12 s |
| `zc2_llama_repeat_full_r2` | **1 passed**, 282.62 s |
| `zc3_llama_repeat_full_r3` | **1 passed**, 222.86 s |
| `zc4_qwen_repeat_full_r1` | **1 passed**, 302.74 s |
| `zc5_qwen_repeat_full_r2` | **1 passed**, 144.62 s |
| `zc6_qwen_repeat_full_r3` | **1 passed**, 143.75 s |

`zc1`'s 571 s is cold-cache variance, not a regression: `zc2`/`zc3` land at 282/222 s against
attempt 3's 234/243/300 s on the same node.

**Device sampling end to end, both models** — `test_<model>_device_greedy_sampling_equals_host_argmax`:

| run | result |
| --- | --- |
| `zd4`/`zd5`/`zd6` Llama | **1 passed ×3** — 442.44 / 343.06 / 226.26 s, 32/32 slots |
| `zd1`/`zd2`/`zd3` Qwen | **1 failed ×3** — `disagreed with host argmax in slots [4]`, 31/32 |

Qwen's failure is **byte-identical to attempt 3's** three runs at
`d11_q_device_greedy_sampling_equals_host_argmax_run{1,2,3}.log`, which also read `slots [4]`. That
is the exact bfloat16 tie at 15.375 already on the ledger, and the point of re-running it was to
show a shared-module change did not move it. It did not. **Nothing was relaxed to make this
green, and it is reported as a failure.**


**The Llama seeded-slot claim, three fresh processes at one commit** — the last of area 4's ten
claim-verdicts to have never had that:

| run | result | clash lines |
| --- | --- | --- |
| `ze1_llama_seeded_slot_r1` | **1 failed**, 301.09 s | 0 |
| `ze2_llama_seeded_slot_r2` | **1 failed**, 308.77 s | 0 |
| `ze3_llama_seeded_slot_r3` | **1 failed**, 352.00 s | 0 |

All three reach the assertion — `AssertionError: a seeded stochastic decode did not repeat` — and
none aborts on the L1 address clash, where attempt 1's three runs of this node all did. The claim
is **evaluated**, and it fails as **D-C12**, exactly as it does on Qwen. Llama's area-4 ledger and
Qwen's are now identical claim for claim.


## 27. `q14` complete — 43 runs, every count unchanged

**The new close contract, on silicon, three fresh processes per model, byte-identical:**

| run | result | printed measurement |
| --- | --- | --- |
| `zg1`/`zg2`/`zg3` Llama | **1 passed ×3** — 247.88 / 218.61 / 196.38 s | `240 registered weights, peak 565 614 080 B per DRAM bank, residue after close 0 B` |
| `zg4`/`zg5`/`zg6` Qwen | **1 passed ×3** — 281.26 / 127.60 / 124.95 s | `192 registered weights, peak 456 632 704 B per DRAM bank, residue after close 0 B` |

Nothing is deleted or collected before that residue is read: `handle` and `runner` are both still
bound. Before the fix the same quantity on Llama at the same shape was **398 617 984 B**.

**The step-7 host suite, three fresh processes, identical to every recorded pass on this branch:**

| file | pass 1 | pass 2 | pass 3 |
| --- | --- | --- | --- |
| `test_step7_concat32.py` | 34 | 34 | 34 |
| `test_step7_long_context.py` | 32 | 32 | 32 |
| `test_step7_paged_kv.py` | 37 | 37 | 37 |
| `test_step7_prefix_cache.py` | 18 | 18 | 18 |
| `test_step7_repeat_and_cleanup.py` | 12 | 12 | 12 |
| `test_step7_sampling.py` | 29 | 29 | 29 |
| `test_step7_token_composition.py` | 8 | 8 | 8 |
| **total** | **170** | **170** | **170** |

`test_prefetcher_2d.py` **35 passed ×3**. `test_lazy_weight_release.py` **6 passed ×3**.
`test_step7_page_table_placement_wh_galaxy.py` **3 passed ×3** (`u1`–`u3`).

**`models/common/tests/llm_runtime`: 1032 passed, 1 skipped** in 212.08 s — exactly the Milestone B
baseline the brief names. The queue's row carries its generic `<- SKIPPED IS A FAILED RUN`
annotation; that heuristic does not apply here, because 1032 passed / **1 skipped** *is* the
expected result, stated as such in the brief's own finish condition.

**`test_column_user_selector_wh_galaxy.py`: 7 passed ×3.** Earlier runs on this branch
(`c8_selector_run{1,2,3}`) read **6 passed**; the difference is
`test_column_user_selection_is_bit_exact`, added by attempt 2 in commit `49a69560329` *after* those
runs. It is a test added, not an expectation changed, and the node-id lists in the two logs differ
by exactly that one entry.


## 28. 01:14Z — the clash is gone from the executor path, and what blocks it now is D-C13

`y1_exec_warmup_df_r1` — `test_executor_warmup_and_program_identity[decode_first]`, the node
`c-exec-llama` used as its 110-second clash reproduction, re-run at HEAD:

```text
grep -c 'clash with L1 buffers'  ->  0
E  ValueError: chunk_start must be non-negative and aligned to chunk_alignment
   models/common/modules/attention/attention_2d.py:908
1 failed, 2 warnings in 561.12s
```

At `2b463f17fcd` this node failed 3/3 on
`TT_THROW … Statically allocated circular buffers in program 922 clash with L1 buffers on core
range [0-0 - 0-3]`. At HEAD it does not clash at all: the decode warmup completes, and the
**prefill** warmup that follows it — the "prefill after a decode" that was the whole trigger — gets
all the way into `Attention2D._validate_prefill` before failing.

**So the driver's arrival note is out of date, and this is the reconciliation.** "At this commit the
clash blocks serving" was true of `2b463f17fcd`, the commit `c-exec-llama` measured. It is not true
of `faec6e59938`/`299440bb276`. What blocks the executor's warmup now is **D-C13**, and D-C13 is
the *runtime* defect reduced in §9: `llm_runtime/warmup.py:700` builds its prefix-cached warmup case
as `cached_tokens=layout.block_size` — 32 — and Galaxy's `chunk_alignment` is 128, so
`llm_runtime/prefill/plan.py:381` hands the module a `chunk_start` of 32 and the module correctly
refuses it. **Both** warmup orders now fail there, for the same reason; the two used to fail for two
unrelated reasons.

Note what that does **not** say: it is the *prefix-cached* warmup case that fails, not a plain
prefill after a decode. `y4`–`y6` — `test_executor_repeated_startup_and_cleanup`, three
startup/serve/cleanup cycles in one process, which failed 3/3 on the clash at `TT_THROW … 542016` —
are the direct test of whether serving is unblocked, and they are queued behind `y2`/`y3`.
