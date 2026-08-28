# Milestone A 2D Module Status

Status of Milestone A of `tttv2_2d_modules_plan.md`, as of **2026-08-26**, at commit `bf403d93fed`
on `gongyu/tttv2_wh_glx_2d_modules`.

Sources: `tttv2_2d_modules_work_log.md`, `tttv2_2d_modules_galaxy_prefetcher_work_log.md`, and the
evidence packages `tttv2_milestone_a_device_evidence/`, `tttv2_milestone_a_gap1_evidence/`,
`tttv2_milestone_a_gap2_evidence/` and `tttv2_milestone_a_final_evidence/`.

> **This page previously declared the exit gate passed, on 2026-08-19. That was wrong**, and the
> correction is the most important thing on this page. An independent re-run on 2026-08-24 could not
> reproduce two of its claims, and root-causing them found **two real defects that the recorded
> "passing" evidence had been masking**. Two further defects were found afterwards by closing
> coverage gaps this page had listed as caveats. All four are fixed or documented; the details are in
> [Defects found after the premature sign-off](#defects-found-after-the-premature-sign-off).
>
> The lesson is recorded because it should change how the remaining milestones are signed off: on this
> hardware, a test that passes once has not proved anything. Three of the four defects presented as
> **intermittent passes**, not failures, because they read aliased or uninitialised L1. Repeat runs in
> fresh processes are the minimum bar for a device claim here.

## Scope

Milestone A adds reusable modules for the canonical Wormhole Galaxy logical mesh `(8, 4)`:

- `Embedding2D`, `RotarySetup2D`, `RMSNorm2D`, `Attention2D`, `MLP2D`, `LMHead2D`, and
  `Sampling2D`;
- shared Galaxy CCL/resource bindings and `Prefetcher2D`; and
- an immutable, topology-neutral batched-prefill policy in the common runtime.

There is no `Penalties2D`. The 2D modules expose explicit config construction and have no
`from_model_args` compatibility API.

## Current position

**Every module is qualified on real `(8, 4)` hardware, and the whole matrix was re-run green as one
sweep at the committed tree.** One blocking item remains, and it is being run elsewhere.

| | |
| --- | --- |
| Modules qualified on WH `(8, 4)` | 7 of 7, plus Galaxy CCL/resources and `Prefetcher2D` |
| Final device sweep | **37 of 37 cases passed** in 15 m 55 s, clean teardown in all 9 groups, **no reset** |
| Integrated host gate | **`1263 passed, 1 skipped` in 265.06 s** at the committed tree |
| Exit gate | **One item outstanding** — the 1D regression matrix ([P4](#p4)), running on separate hardware |
| Deferrable items | 7, each with a stated target milestone |
| Working tree | Committed and pushed: `cf803f23647` (code) and `bf403d93fed` (evidence and docs) |

### Final device sweep — 2026-08-26T14:39:07Z → 14:55:02Z, commit `bf403d93fed`

One pytest process at a time, one group per process, logs in
`tttv2_milestone_a_final_evidence/logs/`. Every group logged
`Closing user mode device drivers` → `Cluster destructor completed`; all 32 devices closed normally
in each, and no `tt-smi -glx_reset` was needed at any point in the sweep.

| Group | Cases | Result | Duration |
| --- | --- | --- | --- |
| `20_embedding` | 2 | passed | 57.32 s |
| `21_rope` | 2 | passed | 7.31 s |
| `22_rmsnorm` | 8 | passed | 33.27 s |
| `23_mlp` | 4 | passed | 116.40 s |
| `24_lm_head` | 2 | passed | 65.68 s |
| `25_sampling_greedy` | 1 | passed | 7.46 s |
| `26_sampling_stochastic` | 9 | passed | 29.96 s |
| `27_attention` | 2 | passed | 74.61 s |
| `28_prefetcher` | 7 | passed (1 deselected) | 227.64 s |
| **Total** | **37** | **37 passed, 0 failed, 0 blocked** | **15 m 55 s** |

The deselected case is `attention_decode_with_active_prefetch`, excluded deliberately: it is terminal
FAILED by construction ([L3](#l3)) and its `TT_FATAL` abort leaves the mesh un-drainable, so including
it would cost a reset and contaminate the sweep. Its diagnosis is recorded in
`tttv2_milestone_a_gap2_evidence/REPORT.md` §6.

This is the artifact the 2026-08-24 run could not produce: one coherent green matrix at one tree,
covering the original 21 cases plus the 9 `Sampling2D` stochastic and 7 `Prefetcher2D` cases added
since.

## Verification Status

| Area | Host evidence | WH `(8, 4)` device evidence | Status |
| --- | --- | --- | --- |
| Embedding2D | Focused suite: 11 passed | Llama and Qwen decode batch 32 plus prefill 128/2048, each repeated, PCC >= 0.99 | Qualified. Independently reproduced 2026-08-24 |
| RotarySetup2D | Focused suite: 13 passed | Llama and Qwen decode plus prefill 128/2048, each repeated, PCC >= 0.99 | Qualified. Independently reproduced 2026-08-24 |
| LMHead2D | Focused suite: 19 passed | Llama and Qwen decode/prefill final-token batches repeated, PCC >= 0.99; Qwen padding mask checked exactly | Qualified. Independently reproduced 2026-08-24 |
| MLP2D | Focused MLP/Galaxy/prefetch suite: 73 passed | Llama and Qwen decode plus prefill 128/2048, each repeated, PCC validated; complete file `4 passed`, re-confirmed 2026-08-25 after the shared-helper split | Qualified. Independently reproduced 2026-08-24 |
| RMSNorm2D | Focused contracts: 19 passed (three new: fused stats-placement rejection, head-local staying interleaved, stats on the norm sender core) | Llama/Qwen batch-32 fused residual decode repeated; distributed prefill 128/2048 repeated; head-local Q/K repeated, all PCC >= 0.99. Complete file `8 passed` over four consecutive runs plus both fused decode node IDs alone in fresh processes | Qualified **after fixing D1** (fused-stats L1 aliasing) and **D2** (head-local shard recipe). The 2026-08-24 Llama-8192 "pass" was reading aliased L1 |
| Attention2D | Host suite: 64 passed; `test_attention_2d.py` + `models/common/tests/models/galaxy` `90 passed` | Llama-70B and Qwen3-32B repeated decode plus prefill 128/2048; output and K/V cache PCC >= 0.99; complete file `2 passed` over four whole-file runs in fresh processes, re-confirmed `2 passed` 2026-08-26. **The Qwen row does not cover the product geometry — see the correction below** | Qualified **after fixing D3** (CCL semaphore/subdevice mismatch). The recorded `2 passed in 53.93s` had been luck; four consecutive full-bound hangs followed it. ~~The decoupled-head-dim path is **unqualified**~~ — **closed by Milestone B, 2026-08-28**: Qwen3-32B's real 64-head geometry (`attention_dim 8192 != dim 5120`) is qualified on silicon at decode PCC 0.99936, three fresh processes; see the correction below |
| Sampling2D | Final 1259-test host gate, plus `test_sampling_2d.py` `27 passed` including a regression test pinning the device temperature buffer to `1/T` | Forced argmax with exact tokens and padded-vocabulary exclusion; **stochastic path qualified** — top-k/top-p containment over 5 parametrizations, padded-vocabulary exclusion under stochastic sampling, seeded repeatability with per-slot seed stability, unseeded freshness, per-slot heterogeneous k/p/temperature. `9 passed` in each of three fresh processes, zero boundary violations | Qualified for both the forced-argmax and stochastic cases, **after fixing D4** (reciprocal temperature). Not covered: RNG distributional correctness, `top_k > 32`, trace/capture |
| Galaxy CCL/resources | Concrete CCL/resource/composition host contracts in the final gate; prefetcher/galaxy/MLP host suites `78 passed` after the shared-helper split | Repeated MLP/RMS paths and fused Attention axis-1 decode pass with clean teardown; the resource owner's own `activate`/`synchronize`/`cleanup` lifecycle now qualified directly across a 12-step prefill↔decode matrix, three fresh processes | Qualified. Non-fused Attention decode is not required or qualified. Attention decode on the *prefetch* subdevice partition is qualified as **incompatible** — see [L3](#l3) |
| Prefetcher2D | Concrete composition regression: 29 passed | Own hardware suite, 7 cases, `7 passed` in each of three fresh processes with identical output: sealed weight addresses read back off all 12 sender cores on all 32 devices; the full transition matrix (`decode→prefill`, `prefill→decode`, `decode→decode`, `prefill→prefill`) with PCC at all 12 steps; failed-transition rollback; cleanup from either active mode proven by a second owner sealing and computing on the same mesh in the same process | Lifecycle qualified for the MLP2D consumer shape. Two documented limits: [L1](#l1) global-CB ownership, [L2](#l2) undersized `global_cb_size` accepted |
| Batched-prefill policy | Integrated host gate re-run at the committed tree: **`1263 passed, 1 skipped, 9 warnings in 265.06s`** | **None.** No test under `models/common/tests/llm_runtime/` touches silicon: no `indirect=True` fixture anywhere in that directory, and `test_llama3_8b_integration.py` runs on a `_Mesh` mock | Host lifecycle qualified; no device evidence at all. See [D-A](#d-a) |

The integrated host gate covers the runtime, Galaxy CCL/resources, `Prefetcher2D`, the concrete
prefetcher composition regression, and every 2D module host suite. It was re-run on 2026-08-26 at
commit `bf403d93fed` — `1263 passed`, up from the `1259 passed` recorded on 2026-08-19, the extra
tests being the new `RMSNorm2D` and `Sampling2D` contracts added by the defect fixes. Log:
`tttv2_milestone_a_final_evidence/logs/host01_integrated_gate.log`.

Host mocks establish config, validation, ownership, and failure-path behaviour; they do not
substitute for real-device numerical, cache, repeat-invocation, or teardown evidence — a point this
page previously made and then failed to apply.

## Defects found after the premature sign-off

| | Defect | How it hid | Fix |
| --- | --- | --- | --- |
| **D1** | `RMSNorm2D` fused decode: `fused_rms_minimal` binds `cb_stats` to the stats tensor's L1 address on the **norm grid's first core**, but the resolved config placed stats on `x=1` while the grid starts at `x=2`. The kernel reduced whatever the allocator had left there | Outcome depended on residual L1: PCC 0.0977 / 0.1394 / 0.1555 / 0.1701 in some processes, ~0.9999 in others. The recorded Llama-8192 pass was aliased L1 | `decode_stats_memcfg` defaults to the norm grid's first core; `_require_fused_stats_placement` turns silent corruption into a `ValueError` |
| **D2** | `RMSNorm2D` head-local Q/K built a 128-wide width shard over a hardcoded two-core range, declaring `2 × 128 = 256` padded width for a 128-wide tensor | Aborted in op validation before any kernel, so it had never produced a numerical result at all despite being recorded as qualified | Head-local decode defaults to interleaved DRAM like prefill; the distributed norm grid derives from `grid_width`. **Milestone B update, 2026-08-28 — the *decode* half was not actually closed by that fix, and now is.** Interleaved DRAM is correct for prefill and **unplaceable** for decode on a partitioned mesh: an interleaved `ttnn.rms_norm` resolves `LayerNormDefaultProgramConfig`, which spreads its tile rows over the whole compute grid, including the sender columns the loaded decode sub-device manager does not own. Milestone B's Qwen bring-up measured it (**D-B26**) and closed it with `RMSNorm2DConfig.decode_compute_cores`. Qwen3-32B's per-head Q/K norm is now qualified in **both** modes at PCC >= 0.99998 on all 32 devices, three fresh processes — but see `MILESTONE_B_STATUS.md` defect **D-S1**: all three of those processes hung in teardown after passing |
| **D3** | `Attention2D` decode: the worker subdevice spanned the whole compute grid while CCL global semaphores were allocated on a narrower core set, so `all_reduce_create_qkv_heads` placed a sender on a core whose semaphore address was never reserved or zeroed | Polled uninitialised L1 forever. Passed in one process, hung indefinitely in another — the recorded `2 passed in 53.93s` versus four consecutive 2700 s timeouts | Decode plan no longer narrows `semaphore_cores`; the invariant is recorded on `galaxy_mode_plan` |
| **D4** | `Sampling2D`: `ttnn.sampling`'s `temp` argument is the **reciprocal** temperature (the kernel multiplies by it), but the module wrote raw `T`. Every request at `T != 1.0` sampled from a distribution warped the wrong way | `1.0` is its own reciprocal, and the greedy path forces `temp = 1.0`. The only hardware test was greedy, so the defect was **structurally unreachable** by existing coverage | One line at `sampling_2d.py:213`; `sample_host` was already correct. Pinned by a host regression test |
| **D5** | `Attention2D`: `resolve_attention2d_config` passed `wo_weight_memory_config` to the `wqkv` lazy-weight resolution and `weight_memory_config` to `wo` — each projection was handed the other's placement field | **Unreachable, and that is the finding.** `_require_exact_weight_policy` (`attention_2d.py:488-491`) runs *first* and rejects any weight whose `memory_config` is not already equal to its own config field, and `resolve_lazy_weight` only fills fields that are `None`. So the swapped arguments could never overwrite anything. Confirmed by running the same two-different-configs probe against both orderings and getting identical results | Two arguments swapped back, isolated in `Fix three WH Galaxy 2D module contract defects found during Milestone B`. Dead code, not a live defect — but it becomes live the moment the exact-policy gate is relaxed, so the gate itself is now pinned by `test_a_projection_placed_against_the_other_configs_value_is_rejected` |

D1, D2 and D3 were found by an independent re-run that was told to reproduce this page's claims rather
than trust them. D4 was found by closing a coverage gap this page had listed as an accepted caveat.
D5 was found by Milestone B, which was the first caller to set the two placement fields to genuinely
different values — and then re-derived, during the Milestone A/B reconciliation, to be latent rather
than live. The reconciliation analysis had predicted the D4 masking pattern here (the only hardware
test that sets both configs builds both `LazyWeight`s through a helper whose `memory_config`
parameter defaults to `ttnn.DRAM_MEMORY_CONFIG`, so the two values are equal and the swap is a
no-op). That is true but not the operative reason: the earlier exact-policy gate makes the swap
unreachable for *any* caller, equal configs or not.

**So the sentence "No known functional defect stands between here and the gate" still holds.** D5 is
recorded because the code was wrong, not because it produced a wrong result.

## Post-record module contract corrections

Three `Attention2D` / `LMHead2D` contract amendments landed *after* the evidence above was recorded,
in the Milestone A/B reconciliation (commit
`Fix three WH Galaxy 2D module contract defects found during Milestone B`). They are listed here so
the Milestone A audit sees them rather than finding them inside a Milestone B model diff. *Updated
2026-08-28: two of the three have since been exercised on real `(8, 4)` hardware by Milestone B; the
third (D5) is dead code by construction and cannot be.*

**The recorded Qwen attention qualification used a geometry no product has.**
`test_attention_2d_wh_galaxy.py:86` builds `_ModelSpec("qwen3-32b", dim=5120, n_heads=40, ...)`.
40 heads x 128 = 5120 = `dim`, chosen so that `n_heads * head_dim == dim` and the square `wo`
contract holds. Real Qwen3-32B has **64** attention heads, giving `attention_dim = 8192 != dim =
5120`. **The decoupled-head-dim path therefore has no hardware evidence of any kind**, and the
`Attention2D` row of the evidence matrix should be read accordingly. `mb-qwen` (Milestone B, plan
steps 4-6) is the job that gets it some; until then this is an open gap, not a covered one.

*Milestone B update, 2026-08-28 — **CLOSED, on silicon**.* `mb-qwen` qualified the real decoupled
geometry (`dim 5120`, `n_heads 64`, `head_dim 128`, `attention_dim 8192`, `wo [8192, 5120]`) on a real
`(8, 4)` mesh, not only on the host. One Qwen block: prefill 128 logits **0.999303669584255**, decode
**0.999360219056066**, KV cache K **0.9998897994661545** / V **0.9998944730661905** on all four
column-local users, prefill 2048 **0.9990203192392576** — three fresh processes, bit-identical, at a
tree byte-identical to the current Milestone B commit
(`tttv2_milestone_b_evidence/qwen/logs2/a2_73,74,75_block.log`). The 64-layer model, the batch-1 and
batch-32 demos and the teacher-forced accuracy gate (top-1 **97.46%**, top-5 **100.00%**) all run on
that geometry too. Host side, attention rebuilt from the converted tensors alone reproduces unmodified
HF `Qwen3Attention` at PCC >= 0.9999
(`models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py`, 13 tests).

*An earlier version of this paragraph, written 2026-08-27, said the geometry remained unqualified on
silicon because `mb-qwen` had no working mesh. That was true when written and is now false; the mesh
came back the same evening.* One trap worth carrying forward regardless: `local_qkv_size ==
local_dim == 1280` for this model, so a fused-QKV-vs-residual width confusion is **shape-invisible**;
`local_attention_dim` (1024) is the width that differs, and so the one a shape check can catch.

- **`wo` source shape** is now `(n_heads * head_dim, dim)` rather than `(dim, dim)`, which is the
  only way to express that geometry. The two coincide for every case the recorded evidence covers,
  so no recorded numerical result changes. Pinned host-side by both the decoupled case
  (`8192 x 5120`) and the square case (`8192 x 8192`), plus the rejection message for a wrong shape.
  **Qualified on hardware** by Milestone B's Qwen3-32B bring-up, as above.
- **`LMHead2D` activation width** now also accepts a column-local width (`dim / 4`) alongside the
  full `dim` (`lm_head_2d.py:507-511`), because a device activation off the column-sharded residual
  stream carries its column shard; the recorded qualification only ever passed host `LazyWeight`
  inputs. A strict superset, so no Milestone A test changes behaviour. **Exercised on hardware**: it
  is the width both Milestone B models actually present at decode — 2048 for Llama, 1280 for Qwen —
  so every decode logit produced on the Galaxy goes through the widened branch.
- **D5**, above. **Host-tested only, and it cannot be otherwise**: `_require_exact_weight_policy`
  makes the swap unreachable for any caller, so no device run can distinguish the two orderings.

## Known limitations, documented and accepted

<a id="l1"></a>**L1 — `Prefetcher2D.cleanup()` cannot free the global circular buffer.** ttnn exposes
no free for one, so its L1 is reclaimed only when the last handle dies — and every module holding a
`Prefetcher2DContext` holds one. After cleanup the owner truthfully reports `owned_resources == ()`
while ~55 MB of L1 stays resident, and the next owner's `seal()` fails with an L1 OOM. **Consumers
must be torn down before, or together with, the owner.** Recommended design fix (Milestone B/C): make
`global_cb` a property on the context rather than a stored handle.

*Milestone B update, 2026-08-28 — **measured at model scale on silicon, and it is worse than this
paragraph says.** L1 is a **lifetime** problem, not the ordering problem it is written as.* Three
results, each with a log:

- **the mechanism, on host.** `cleanup()` clears `self._global_cb` without ever handing it to
  `deallocate`, so the owner's truthful `owned_resources == ()` and the CB's continued residency are
  the same event. Two owners in one process allocate two CBs and free neither;
- **dropping the last Python reference does not return the L1.** `mb-llama` implemented the obvious
  fix — release on `activate("prefill")`, recreate on the next `activate("decode")` — behind
  `Prefetcher2DConfig.release_global_cb_on_prefill`, default off. The release ran and the clashing L1
  base address was **identical** with and without it (544832 in both). The flag and its tests are left
  in the tree, default off, with the refutation recorded against them;
- **nor does closing the whole model** (Milestone B defect `D-C7`). After `close()` *and* an explicit
  `gc.collect()`, **923 776 of 1 393 472 bytes per L1 bank — 66% — are still allocated**, largest free
  block 373 824 B against the 792 064 B the next model needs. **No teardown ordering can fix a buffer
  the destructor of a closed object did not free.**

So read a clean `cleanup()` as "nothing this object still owns", not "nothing is left on the device",
and take two operating rules from it: **one Galaxy model per process**, and **prefill everything
before you decode anything** — a second runner that prefills after the first has decoded fails
deterministically, 3/3 in three fresh processes, on Llama. Redesign re-routed to Milestone C; full
account in `models/common/models/MILESTONE_B_STATUS.md`, limitation `L-B1`.

<a id="l2"></a>**L2 — an undersized `global_cb_size` is silently accepted.** The rejection contract
does not exist. Low severity (the qualified configuration is correct) but it is a missing guard, not
a working one.

<a id="l3"></a>**L3 — Attention2D decode is incompatible with the prefetch subdevice partition
*as Milestone A configured it*.** The decode QKV `ttnn.linear` at `attention_2d.py:851` uses a
`(7,1)` grid that normalizes to `CoreRange((0,0),(6,0))` — 2 sender cores plus 5 worker cores — and
tt-metal rejects programs straddling subdevices. It cannot be narrowed: `allowed_worker_cores` must
be a dense rectangle, the worker subdevice is not one, and every origin-anchored rectangle includes
sender column `x=0`.

~~**Wired, not qualified (updated during the Milestone A/B reconciliation).** This is a
program-config choice, not a module limit, and Milestone B now makes the partition-compatible choice:
the decode QKV and `wo` projections resolve to the 24-core ring form built by
`models/common/models/galaxy/recipes.py::ring_matmul_program_config`.~~

> **CLOSED on silicon, 2026-08-28, at a named cost.** Two things this limitation says are now known
> to be wrong, and one of them was disproved by hardware.
>
> **It can be narrowed.** "`allowed_worker_cores` must be a dense rectangle, the worker subdevice is
> not one, and every origin-anchored rectangle includes sender column `x=0`" is outdated: `ttnn` has
> since grown `allowed_worker_cores` for exactly this case, deprecating
> `compute_with_storage_grid_size`, and populating it **does** make the program legal. Milestone B
> confines both attention decode matmuls to `dense_matmul_worker_rectangle` — the largest rectangle
> anchored at the *worker envelope's* origin, three columns wide on `(8, 4)` — and with
> `in0_block_w = gcd(k_tiles, 4)` their circular buffers fit (Milestone B defect **D-B9**, closed).
> Re-verified for this record at `models/common/models/galaxy/recipes.py:373-425,850,853`.
>
> **And it is numerically qualified**, on both product models, three fresh processes each: Llama
> decode attention output PCC **0.99975**, decode logits **0.99975**, KV K **0.99993** / V
> **0.99975**; Qwen decode logits **0.99936**. Logs
> `tttv2_milestone_b_evidence/qwen/logs2/a2_40,41,42_llama_step2.log` and `a2_73,74,75_block.log`.
>
> **Two costs, recorded rather than absorbed.** Three worker columns instead of seven; and the
> attention weights lose their prefetching, because the global circular buffer is received only by the
> 24 ring cores, so a matmul on the confined rectangle cannot take its weight from it. Registering
> them with the prefetcher anyway is a **correctness** defect and not a performance one — the
> unconsumed entries shift every later consumer, which is how the MLP came to score PCC 0.096
> (**D-B25a**). Moving the two matmuls to the 24-core `gather_in0` ring recovers both at once and is
> **Milestone C performance work**; the recipes already anticipate it
> (`attention_qkv_collective_input_memcfg` is shaped for exactly those 24 cores).
>
> *Two superseded readings, kept so the record shows how it moved.* The struck paragraph above,
> written during the A/B reconciliation, claimed Milestone B had put attention on the ring form: it
> had not — only the MLP had, and no host test could see the difference (`mb-llama` defect **D-B5**).
> A 2026-08-27 signoff pass then recorded "L3 is therefore STILL OPEN" with **D-B9** open behind it;
> that was written from `mb-llama` attempt 1 and was superseded by attempts 2 and 3 the same week.
>
> Full account: `models/common/models/MILESTONE_B_STATUS.md`, "L3 — CLOSED, with a named cost".

A related operational note: a `TT_FATAL` abort inside a multi-subdevice program leaves the mesh
un-drainable — teardown blocks in `FDMeshCommandQueue::~FDMeshCommandQueue → wait_for_outstanding_reads`.
Budget a kill and a `tt-smi -glx_reset` after any such abort.

## Pending work

### Blocking — must close before Milestone A can be signed off

**One item remains.**

<a id="p4"></a>**P4 — Re-run the 1D regression matrix.** `test_attention_1d.py` (−228 lines),
`test_mlp_1d.py` (−27) and `test_sampling_1d.py` (−31) now import shared reference plumbing from
`_hf_reference.py`. Only `test_sampling_1d.py` has been re-run on hardware (`140 passed, 50
deselected`); the attention and MLP 1D hardware matrices have not. No 1D implementation file changed,
so this should be a formality — but the exit gate requires the evidence and it is currently absent.
**In progress on separate hardware**, deliberately not run on the Galaxy host.

### Closed

<a id="p1"></a>**P1 — Commit the change set. Done.** `cf803f23647` carries the module fixes to
`rmsnorm_2d.py` and `sampling_2d.py`, the `_hf_reference.py` / `_mlp_2d_galaxy.py` shared test
plumbing, the `_wh_galaxy_hardware.py` refactor, the two new device suites and the edits to six
existing test files. `bf403d93fed` carries the evidence packages, gap briefs and work-log
checkpoints. Both pushed to `origin/gongyu/tttv2_wh_glx_2d_modules`. No
`models/common/modules/**/*_1d.py` implementation file is touched.

*Two things surfaced while committing, both recorded rather than worked around.* The repo's
`prefer-expect-error` hook blocked the commit on twelve `pytest.raises` blocks in the two touched host
suites — mostly pre-existing, one of them new. They were converted to the sanctioned `expect_error`
fixture rather than suppressed; both suites still pass (`46 passed`). And the repo's `.gitignore`
excludes `*.log`, so the raw pytest logs behind the evidence packages stay on the host that produced
them; each `REPORT.md` names the log behind every claim.

<a id="p2"></a>**P2 — Re-run the full device matrix at the final commit. Done.** 37 of 37 cases
passed; see [Final device sweep](#final-device-sweep--2026-08-26t143907z--145502z-commit-bf403d93fed).

<a id="p3"></a>**P3 — Re-run the integrated host gate. Done.** `1263 passed, 1 skipped, 9 warnings in
265.06s` at commit `bf403d93fed`.

*A false start worth recording,* because it is a trap the next person will hit. The first attempt
passed `models/common/tests/modules/prefetcher` to pytest as a **directory**, which collected that
module's `*_wh_galaxy.py` **device** suite alongside its host suite — including the terminal-failing
`attention_decode_with_active_prefetch`. The run aborted with the [L3](#l3) `TT_FATAL` and hung in the
un-drainable teardown; it needed a kill and a `tt-smi -glx_reset`
(`logs/host01_integrated_gate_ABORTED_bad_selection.log`, `logs/reset01_after_bad_selection.log`).
Host-only selections must filter `--ignore-glob="*_wh_galaxy*.py"`, which is now documented in
`modules/README.md`.

<a id="p5"></a>**P5 — Update `modules/README.md` and re-audit the modularity scorecard. Done.** The
README's 1D-vs-2D section now records the hardware qualification, both ownership limits ([L1](#l1),
[L3](#l3)), the host-only-vs-device suite split and the shared reference plumbing. The scorecard below
is re-audited against the committed diff.

### Deferrable — with the milestone each belongs to

<a id="d-a"></a>**D-A — Physical-32 real-device trace → Milestone C.** The plan asks to qualify
physical-32 capture/replay at sequence length 128 and up. This needs a model-owned executor with
`TraceCompiler`/`TracedExecutor` running a 2D model at batch 32 on `(8, 4)`; executors are Milestone C
and the 2D models are Milestone B, so **there is nothing on the Galaxy to trace yet**. Milestone A's
own sequence, item 7, scopes this area to "the generic batched-prefill policy and host/runtime tests",
which is done. *Achievable sooner and worth separating:* the delegation itself has no device evidence
of any kind, and could be exercised on N150/T3K with an existing 1D model to prove the default is
byte-for-byte preserved. See `tttv2_milestone_a_gap_briefs/gap3_batched_prefill_physical32_trace.md`.

**D-B — Attention2D on the prefetch subdevice partition → CLOSED by Milestone B; its residual is a
performance item for Milestone C.** [L3](#l3). Milestone B confined the dense grid with
`allowed_worker_cores` and closed the circular-buffer clash it opened (**D-B9**); both attention
decode matmuls now execute and are numerically qualified on both models. What is left is not a
correctness deferral: four of seven worker columns, and the attention weights read from DRAM instead
of the prefetcher. Both are recovered by moving those two matmuls to the 24-core `gather_in0` ring.
**Routed to Milestone C as performance work.**

**D-C — `Prefetcher2D` global-CB ownership redesign → Milestone C, and bigger than this row assumed.**
[L1](#l1). Milestone B built both full models, ran a second Galaxy model construction in one process,
and found that the L1 is not returned by full model teardown either (**D-C7**: 66% of every bank still
allocated after `close()` and `gc.collect()`). The redesign therefore has to address the buffer's
**lifetime**, not only the teardown order, and `global_cb` as a property on the context may not be
sufficient on its own. The API change should still land with the executor work that exercises repeated
owner lifecycles. Two measurement gaps remain and are named in the Milestone B status page: D-C7 rests
on **one** observation, and `test_two_models_in_one_process` has **never run on Llama**.

**D-D — `global_cb_size` validation → any time.** [L2](#l2). Small and self-contained; no dependency
on later milestones.

**D-E — 1D reference-temperature follow-up → separate, 1D test surface.** The 1D containment
tolerances are asymmetric in exactly the direction a multiply-vs-divide mismatch predicts (`t=0.5`
allows 6, `t=2.0` allows 2, against a `t=1.0` baseline of 3). `Sampling1D` passes `temp` straight
through, so *its* argument is already the op's `1/T`; the mismatch is in the test's HF reference.
Testable prediction: passing `1/temp` to `hf_valid_token_set` should remove the temperature-dependent
excess. Not the module, not this milestone.

**D-F — Sampling2D RNG distributional correctness and `top_k > 32` → not required.** Containment
proves support membership, not probabilities. `top_k > 32` is outside the config contract and is where
the containment argument stops being sound.

**D-G — WH Galaxy CI registration → separate PR.** Explicitly out of scope per the plan.

## Exit-Gate Result

**Eleven of twelve exit-gate lines are met. The twelfth ([P4](#p4)) is running on separate
hardware.** Against the plan's checklist:

| Exit-gate requirement | State |
| --- | --- |
| Host-only config validation tests | **Met** — `1263 passed, 1 skipped` at the committed tree |
| Real WH `(8, 4)` decode and prefill tests | **Met** for all 7 modules; 37/37 in one sweep |
| Representative Llama and Qwen geometry | **Met** |
| PCC >= 0.99 against an independent PyTorch/HF reference | **Met** — and strengthened: the 2D suites now compare against the same HF references the 1D suites use, instead of hand-written re-implementations |
| KV-cache PCC >= 0.99 where applicable | **Met** (Attention2D) |
| Ownership/cleanup and repeat-invocation tests | **Met**, including the `Prefetcher2D` transition matrix, subject to [L1](#l1) |
| Prefetch/CCL/static strategy resolved before the hot path | **Met** |
| No `from_model_args` dependency | **Met** |
| Every 1D module implementation file unchanged | **Met** — no changed `models/common/modules/**/*_1d.py` in either commit |
| Existing 1D module test suite passes | **Outstanding — [P4](#p4)**, in progress on separate hardware |
| Pre-existing default-runtime tests and expectations preserved | **Met** — integrated gate re-run green at `bf403d93fed` |
| Runtime execution-code change limited to tested topology-neutral config delegation | **Met on host**; no device evidence ([D-A](#d-a)) |

No known functional defect stands between here and the gate. When [P4](#p4) lands green, Milestone A
is complete and Milestone B may begin.

One caveat on the last line, carried deliberately rather than waived: the runtime delegation is
qualified by host tests only. That is consistent with Milestone A's own sequence (item 7 scopes this
area to "the generic batched-prefill policy and host/runtime tests"), but it means the claim
"behaviourally identical for every existing 1D model" has never been checked against silicon. See
[D-A](#d-a).

## Modularity Scorecard

Re-audited 2026-08-26 against the committed diff (`de4c8f4e659..bf403d93fed`).

| Required item | Evidence | Assessment |
| --- | --- | --- |
| New 2D/model files | Five new functional module implementations, `Prefetcher2D`, Galaxy `ccl.py`/`resources.py`, package exports, and focused tests; MLP2D and RMSNorm2D completed in their existing files. Since: two new device suites (`test_sampling_2d_wh_galaxy_stochastic.py`, `test_prefetcher_2d_wh_galaxy.py`) and two shared test-plumbing modules (`_hf_reference.py`, `_mlp_2d_galaxy.py`) | Within Milestone A boundaries |
| Existing shared files changed | `llm_runtime/prefill/config.py`, `plan.py`, `runtime.py` add and consume a generic immutable batching policy; `modules/README.md` documents the inventory; MLP2D/RMSNorm2D files are milestone-owned. Test-side only: `_wh_galaxy_hardware.py` refactored to expose the prefetch geometry, `test_mlp_2d_wh_galaxy.py` reduced to imports from `_mlp_2d_galaxy.py` | Runtime changes remain generic policy delegation; the test-side churn touches no product code |
| Why config alone was insufficient | Eligibility was encoded directly in planner/runtime decisions, so the resolved policy had to be threaded through the planner call and consumed mechanically | Narrow shared plumbing with focused tests |
| 1D module implementation files changed | **Zero.** `git diff --stat de4c8f4e659..HEAD -- 'models/common/modules/**/*_1d.py'` is empty. Three 1D **test** files import shared reference plumbing instead of defining their own copies; no 1D behaviour changed | Required value met |
| Default runtime behavior changed | **Zero intentional.** The default policy preserves prior values, and the integrated host gate re-ran green at the committed tree (`1263 passed`) | Confirmed at [P3](#p3) |
| 1D regressions | `test_sampling_1d.py` `140 passed, 50 deselected` post-refactor; attention and MLP 1D hardware matrices in progress | **Incomplete — [P4](#p4)** |
| Common-code topology assumptions | Batched-prefill eligibility and physical batch selection were fixed in common planning code | Moved behind a topology-neutral immutable policy; no Galaxy/model branch added |
| Boundary leakage | Static topology and model differences live in 2D configs or injected Galaxy collaborators; the runtime diff contains no Galaxy, Llama, Qwen, Wormhole, 2D, or `(8, 4)` execution branch. The four defect fixes are all inside 2D modules or their test resource plans — none required a runtime or 1D change | Boundary preserved in the committed diff |

## CCL Follow-Up

Galaxy CCL remains separate from `models/common/modules/tt_ccl.py`. After both reconstructed models
pass their later milestones, evaluate whether their APIs can share an owner. The overlap includes
collective topology, semaphores, persistent buffers, and subdevice identity; Galaxy additionally
requires mode-specific resource keys, exact tensor/sequence plans, adjacent semaphore windows, and
explicit sender/worker subdevice lifecycle.

One input to that evaluation is now on record: the `semaphore_cores` invariant from [D3](#defects-found-after-the-premature-sign-off).
Narrowing a mode's semaphore allocation below its worker subdevice is safe only for a collective that
binds its semaphore to a grid it owns (as the fused RMS all-gather does). The generic async CCLs
choose senders from the subdevice and must keep the default, or they hang on uninitialised L1.

## Reference — evidence packages

| Package | Contents |
| --- | --- |
| `tttv2_milestone_a_device_evidence/` | The 2026-08-24 independent re-run. Its results table is superseded for RMSNorm2D and Attention2D; its header records which rows |
| `tttv2_milestone_a_gap1_evidence/` | `Sampling2D` stochastic qualification and defect [D4](#defects-found-after-the-premature-sign-off) |
| `tttv2_milestone_a_gap2_evidence/` | `Prefetcher2D` / Galaxy resource qualification, findings [L1](#l1), [L2](#l2), [L3](#l3) |
| `tttv2_milestone_a_gap_briefs/` | The agent briefs and completion handoffs for the three coverage gaps |
| `tttv2_milestone_a_final_evidence/` | The 2026-08-26 sweep at the committed tree: the integrated host gate, the 37-case device matrix, and `run_device_matrix.sh` which reproduces it |

Raw pytest logs are excluded from git by the repository's `*.log` ignore rule and remain on the host
that produced them (`wh-glx6u-05-…`); every `REPORT.md` names the log behind each claim. The device
sweep is reproducible with `bash tttv2_milestone_a_final_evidence/run_device_matrix.sh`.
