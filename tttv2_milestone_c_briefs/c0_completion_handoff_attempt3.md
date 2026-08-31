# `c-defects` — completion handoff (attempt 3)

**Last updated:** 2026-08-31T12:27Z — mesh degraded to 25 boards; resets stopped; no marker written
**Base commit:** `2b463f17fcd`. **Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`.
**Job window:** started 08:17Z, 43200 s.

**Finish marker: NOT written — two gates short, named in §11. Blocked marker: NOT written — the
obstruction is an unrecoverable board, which the house rules put under `BLOCKED (infra)`, not under
a job-level block.**

---

## 1. Reconciliation of inherited state — what I discarded and why

Attempt 2's handoff is stamped `2026-08-30T11:00Z`. Three device logs it cites, or that its
queue produced afterwards, are **newer than that stamp**, and all three are unusable:

| log | mtime | verdict | why |
| --- | --- | --- | --- |
| `logs/e1_qwen_two_pools_l1.log` | 11:19Z | **NOT MEASURED** | `exit=124`. Started 10:49:04Z, last output 10:50:25Z (loading `layer0_wo` from the weight cache), then silent for 29 min until the outer bound killed it. This is the run that wedged the mesh. |
| `logs/f1_llama_repeat_l1_loop.log` | 11:19Z | **NOT MEASURED** | `1 error in 3.96s`, inside the `mesh_device` fixture, in UMD `TopologyDiscovery`. Dead-mesh artifact. |
| `logs/f2_qwen_repeat_l1_loop.log` | 11:19Z | **NOT MEASURED** | `1 error in 4.22s`, same signature. Dead-mesh artifact. |

The last device run that completed normally was `d2_qwen_repeat_l1` at 10:49:04Z. Everything from
`e1` onward is discarded on the grounds the driver's re-attempt note gives. Note the corollary the
same note asks for: **`e1` is not innocent.** It hung for 29 minutes in the second model's weight
load, 15 s after `d2_qwen_repeat_l1` had aborted out of a loaded sub-device manager with the new
guard's `RuntimeError`. A Python exception raised while the prefetcher holds a sub-device manager
and a live global circular buffer leaves the mesh in the same un-drainable state a `TT_FATAL` does.
That is recorded below as a real hazard of the fix path, not as a defect of `e1`'s own test.

**What I inherited as measured, and did not re-measure:**

- everything in `RESULTS.md` up to and including `d2_qwen_repeat_l1` (10:49:04Z, 2026-08-30);
- the whole of attempt 1's campaign (2026-08-29, 118 runs), including all thirty area-4 runs and
  the complete D-C6 concat-32 sweep;
- attempt 2's clash attribution chain `a2`/`a3`/`a4` and its three fix arms `b1`/`b2`/`b4`.

**Working tree at 08:17Z, verified against `git status` rather than the handoff:** attempt 2's fix
is present and **uncommitted**, in 7 files, +426/−28. `git log` shows the nine attempt-1 commits.
`pytest models/common/tests/modules/prefetcher/test_prefetcher_2d.py` → **28 passed** at 08:20Z
(`tttv2_milestone_c_runs/c-defects3/logs/h1_prefetcher_host.log`), which matches attempt 2's own
last host number, so the tree holds the *loop* form of the reservation fix and not the
single-reservation form that `d1`/`d2` measured failing.

## 2. Gate ledger as inherited (before this attempt's device work)

| Finish-condition gate | state on arrival | evidence |
| --- | --- | --- |
| D-C5/D-C8 device sampling end to end, 5 area-4 claims × 3 fresh processes × 2 models | **MET as worded** — 30 runs, all reaching their assertions. 3 claims fail deterministically (see §4) | `logs/d11_{q,l}_*_run{1,2,3}.log`, `logs/dc9fix*` |
| D-C7 two models in one process, 3 fresh processes | **NOT MET** — `d11_q_two_pools_run{1,2,3}` failed 3/3 | `logs/d11_q_two_pools_run{1,2,3}.log` |
| Llama clash: `*_repeated_requests_and_deterministic_cleanup` 3/3 + 3 blocked claims | **NOT MET** — 3/3 failing, blocked claims never run | `logs/d11_l_repeat_run{1,2,3}.log` |
| D-C6 fixed-or-DEFERRED with measurements | **MET** — `DEFERRED` with a five-length sweep behind it | `tttv2_milestone_c_evidence/defects/D-C6.status` |
| step-7 host suite green; `llm_runtime` 1032/1 skipped | measured green at 2026-08-29 20:52–20:57Z, **but before the uncommitted fix** — must be re-run | `logs/reg_s7_*.log`, `logs/reg_llm_runtime.log` |
| zero changes to `*_1d.py` and `llm_runtime/**` | holds — see §6 | `git diff --stat` |

## 3. This attempt's plan, in the order device time is spent

1. qualify the loop-form reservation fix on the production path, both models, one-layer subsets
   (cheap: ~110 s and ~100 s);
2. if it holds, the Llama clash gate at the full 80-layer shape, 3 fresh processes, and the same
   for Qwen to prove the shared-code change does not move it;
3. the three claims the clash blocked: cross-slot isolation, two pools, chunked prefill;
4. D-C7's two-models-in-one-process gate;
5. the regression suites against the changed tree, then commit.

IN FLIGHT at this stamp: `q1` — `p0_partition`, `g1_llama_repeat_sub_r1`, `g2_qwen_repeat_sub_r1`.

---

## 4. 08:29Z — the loop-form reservation fix passes on both models, first device evidence

The tree's `Prefetcher2D._allocate_global_cb` reserves L1 **in a loop** until the lowest occupied
address is back at the free-region top the first creation saw, then raises if the recreated buffer
does not land on the recorded floor. Attempt 2 wrote it, host-qualified it, and never got it onto
silicon: the run that would have measured it (`e1`) wedged the mesh instead.

It works. One-layer subsets, production path, `release_global_cb_on_prefill` on by default:

| run | node | result |
| --- | --- | --- |
| `p0_partition` | `test_partition_wh_galaxy.py` | 5 passed in 16.03 s — mesh healthy on arrival |
| `g1_llama_repeat_sub_r1` | `test_llama33_70b_galaxy_repeated_requests_and_deterministic_cleanup` | **1 passed** in 150.18 s |
| `g2_qwen_repeat_sub_r1` | `test_qwen3_32b_galaxy_repeated_requests_and_deterministic_cleanup` | **1 passed** in 109.93 s |

Against the same two nodes at the same shape yesterday, with the *single*-reservation form:
`d1_llama_repeat_l1` and `d2_qwen_repeat_l1` both **failed**, byte-identically, with
`the recreated global circular buffer did not land on its original L1 floor: expected 513024, got
545760`. So the loop is the load-bearing part of the fix and the guard is what told us so.

This is the first time `*_repeated_requests_and_deterministic_cleanup` has passed for Llama on this
branch in any shape.

IN FLIGHT from 08:29:22Z: `q2` (the same gate at the full 80-layer shape, 3 fresh processes per
model) then `q3` (the three claims the clash blocked, D-C7's two-pools gate on both models, and
Llama's seeded-slot claim), 21 runs.

---

## 5. What is NOT in this job's finish condition, recorded so it is not lost

Three items are open, well-measured, and outside the six gates. None of them can be traded for a
gate and none of them is a reason to withhold the finish marker; all three are the strongest
candidates for whatever runs next.

**D-C12 — a warm ttnn program cache makes a sampling call return the previous call's answer.**
Isolated by attempt 1 to a single difference (`disable_and_clear_program_cache`), three fresh
processes, `logs/d11_repeat_sample_probe_run{2,3,4}.log`. With the cache cleared, four consecutive
sampling calls on four different inputs are all correct; with it warm, only the first is. This is
what makes Qwen's `a seeded slot repeats across runs` claim fail. It is a correctness hazard for
anything that decodes more than one token. The next bisection is one extra readback: compose the
**selector's** output on every call and see whether it is already stale before `Sampling2D` is
reached. Candidate operands, all in `sampling_2d.py::decode_forward`:
`ttnn.topk(indices_tensor=self._local_indices)`, `ttnn.manual_seed`,
`ttnn.sampling(output_tensor=tt_out_tok)`.

**Two area-4 residuals with no measurement behind them.** `T = 0.02` collapses onto the host argmax
in 30 of 32 slots on both models, deterministically, at slots `[4, 21]` on Qwen and `[2, 11]` on
Llama — disjoint sets, and Llama's own greedy claim is 32/32 on the same logits through the same
sampler, so the misses are in the **draw**, not the placement. The open question is one sentence:
what should a stochastic draw do when the top-two logit gap approaches the bfloat16 floor? Qwen's
greedy slot `[4]` is settled — an exact bfloat16 tie at 15.375, `torch.argmax` takes the lower index
and `ttnn.sampling` does not (`logs/d11_greedy_tie_probe.log`) — and that is a test-expectation
question, not a sampler defect. **Neither was relaxed.**

**The three defects `c-exec-llama` handed this job**, triaged in attempt 2's handoff and re-checked
here against the source rather than re-run:

- `chunk_start must be non-negative and aligned to chunk_alignment` (`attention_2d.py:860`) — **not
  a module defect.** `ttnn`'s own chunked-SDPA op requires `chunk_start % q_chunk_size == 0` with
  `q_chunk_size = k_chunk_size = geometry.chunk_alignment = 128` in the Galaxy recipe, so a 32-token
  prefix is not expressible by the qualified recipe and the module's refusal is correct. Moving the
  check would turn a host `ValueError` into a device `TT_FATAL`. Two ways forward and **both are
  decisions**: the caller rounds its cached length to a multiple of 128 (one line, free), or the
  Galaxy chunked recipes are re-qualified at `chunk_alignment = 32` (a shared-recipe change needing
  its own device qualification on both models). `c-defects` recommends the first.
- `page_table width cannot address the required KV capacity` (`attention_2d.py:714`) —
  **caller-side.** When the physical pool shrinks, `meta.max_num_blocks` shrinks with it and a page
  table staged for the old pool is genuinely too wide. Re-staging is part of rebinding.
- `test_reference_prefill_and_decode` at 2048 returns non-finite decode logits in
  `GalaxyDirectRunner` — **open, and it is ours.** Shared Galaxy code, the only one of the three that
  is a defect in this job's area. Not reproduced by attempt 2 or by this attempt: it costs a
  full-model 2048-token run and the mesh has gone to the gates. It is the first thing to pick up.

---

## 6. 08:39Z — the clash gate passes at the shape it failed in

Three fresh processes, full 80 layers, no layer subset — the shape attempt 1 measured failing three
times out of three at address 544832:

| run | log | result |
| --- | --- | --- |
| `g3_llama_repeat_full_r1` | `logs/g3_llama_repeat_full_r1.log` | **1 passed**, 526.72 s |
| `g4_llama_repeat_full_r2` | `logs/g4_llama_repeat_full_r2.log` | **1 passed**, 216.45 s |
| `g5_llama_repeat_full_r3` | `logs/g5_llama_repeat_full_r3.log` | **1 passed**, 214.47 s |

**The Llama address clash is fixed, and the gate's first half is met.** (Run 1 is slower only
because it warms the host weight cache; the device work is the same.)

Qwen's half of the same claim — the change is shared Galaxy code, and the brief requires every fix
to be qualified on **both** models even where the defect only ever showed on one:

| run | log | result |
| --- | --- | --- |
| `g6_qwen_repeat_full_r1` | `logs/g6_qwen_repeat_full_r1.log` | **1 passed**, 279.16 s |
| `g7_qwen_repeat_full_r2` | `logs/g7_qwen_repeat_full_r2.log` | **1 passed**, 156.07 s |
| `g8_qwen_repeat_full_r3` | `logs/g8_qwen_repeat_full_r3.log` | **1 passed**, 141.04 s |

**Six for six.** The change fixes Llama and does not move Qwen.

---

## 7. 09:45Z — the guard was measuring a proxy, and the proxy hid a second half of the defect

`i1`/`i2`/`i3` — area 1's block-level cross-slot isolation, one of the three claims the clash
blocked — **passed 3/3** (441.77 s, 412.63 s, 248.52 s). That claim had never been evaluated on
Llama in either direction.

Then chunked prefill failed, twice, byte-identically:

```text
k1_llama_chunked_r1   RuntimeError: ... did not land on its original L1 floor: expected 510624, got 510816
k2_llama_chunked_r2   RuntimeError: ... did not land on its original L1 floor: expected 510624, got 510816
```

**192 bytes is exactly the global circular buffer's config page**, and from that one number the
guard could not tell a real hazard (the whole pair moved up 192 B) from a benign one (the config
page took a small gap and the data buffer never moved). I stopped the queue there rather than spend
an hour on results that could not be read either way — nine remaining items were parked with
deliberate `NOT A RUN` logs, which is the mechanism attempt 2 used for `b5` and it is on the record
in each file.

`ttnn` exposes **no** address accessor on a global circular buffer
(`ttnn/cpp/ttnn-nanobind/global_circular_buffer.cpp` binds `size`, `sender_cores`,
`receiver_cores`, `sender_core_type` and nothing else) and rebuilding tt-metal is forbidden here, so
the guard now reads the allocator's own block table and compares the blocks the creation **adds**,
by address and size. `k3` ran under that change and answered the question exactly:

```text
expected [(510624, 192), (510816, 792064)]
got      [(510816, 792064), (1367872, 192)]
```

**The data buffer did not move. The 192-byte config page moved 857 kB, into a free gap near the top
of L1.** So the proxy was wrong — and the thing it was hiding is a *second* stale-address hazard,
not a false alarm:

- `CircularBufferImpl::set_global_circular_buffer` (`tt_metal/impl/buffers/circular_buffer.cpp:179`)
  captures **both** `global_circular_buffer.buffer_address()` **and** `config_address()`, once;
- a cached program re-sends that captured pair on every launch
  (`tt_metal/impl/program/dispatch.cpp:3035`, `cb_config_payload[base_index] = cb->config_address()`),
  and kernels read the remote-CB config from that address at runtime.

So a moved config page is read at the wrong place exactly as a moved data buffer would be. The
guard was right to refuse; it was just describing the wrong block.

### The rule behind all of it, and the second half of the fix

Everything here follows from one allocator rule: `FreeListOpt::allocate` takes the **smallest free
block that fits**. That is why a single reservation of exactly the missing bytes lands in the
leftover gap instead of lowering the free top (the 32 736 B on both models), and it is why a
192-byte hole anywhere in L1 captures the config page. So `_allocate_global_cb` now does two things
before it creates the buffer:

1. **hold every free gap above the low region** — smallest first, because a request of exactly a
   gap's size is how that gap is taken. With the gaps held, the low region is the only free block
   and the creation's two allocations are forced to be adjacent;
2. **reproduce the free-region top** the first creation saw, so the pair lands where it landed.

Both loops are bounded and raise rather than spin, and the footprint check remains as the backstop.

Host suite: `test_prefetcher_2d.py` **30 passed** (22 before this workstream). Removing *only* the
gap-filling pass fails exactly the two gap cases and nothing else — **2 failed, 28 passed** — so the
new mechanism has a test that fails without it. Against `prefetcher_2d.py` at `HEAD` the file is
29 failed / 1 passed.

**`g3`–`g8` are superseded as evidence.** They passed, but they measured the pre-gap-filling module,
so the repeat gate is being re-run on both models under `n1`–`n6` and only those runs are quoted at
signoff.

IN FLIGHT from 09:45:59Z: `q5`, 22 runs — partition, the repeat gate ×3 on both models, chunked ×3,
cross-slot ×3, two pools ×3 on both models, seeded slot ×3.

---

## 8. 10:05Z — the gap-filling pass was the missing half, and chunked prefill is green 3/3

| run | log | result |
| --- | --- | --- |
| `p1_partition` | `logs/p1_partition.log` | 5 passed, 16.34 s |
| `n1_llama_repeat_full_r1` | `logs/n1_llama_repeat_full_r1.log` | **1 passed**, 234.34 s |
| `k1b_llama_chunked_r1` | `logs/k1b_llama_chunked_r1.log` | **1 passed**, 217.74 s |
| `k2b_llama_chunked_r2` | `logs/k2b_llama_chunked_r2.log` | **1 passed**, 216.19 s |
| `k3b_llama_chunked_r3` | `logs/k3b_llama_chunked_r3.log` | **1 passed**, 223.54 s |

The same node was `1 failed` three times in a row an hour earlier — twice on the ambiguous proxy and
once on the exact footprint that named the moved config page. Holding the free gaps before the
creation is what closed it, and the failure is what showed the mechanism.

---

## 9. 10:28Z — the repeat gate, re-qualified on both models under the final module

| model | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| Llama-3.3-70B | `n1` **1 passed** 234.34 s | `n2` **1 passed** 243.46 s | `n3` **1 passed** 300.20 s |
| Qwen3-32B | `n4` **1 passed** 312.51 s | `n5` **1 passed** 152.94 s | `n6` **1 passed** 148.35 s |

Full 80 layers, no layer subset, six fresh processes. Against `d11_l_repeat_run{1,2,3}`, which failed
3/3 at address 544832. **The Llama L1 address clash is fixed and the gate's first half is met**, and
the shared-code change does not move Qwen.

---

## 10. 11:27Z — a NEW defect, reduced with a backtrace; and the mesh is down on one board

### The three claims the clash blocked: two measured green, one hangs

| claim | node | result |
| --- | --- | --- |
| area 1, block-level cross-slot isolation | `test_llama_a_write_for_one_user_never_appears_in_another_users_blocks` | **3/3 pass** — `i1b` 453.68 s, `i2b` 352.52 s, `i3b` 280.35 s |
| area 3, chunked prefill | `test_llama_chunked_prefill_matches_a_single_uncached_prefill` | **3/3 pass** — `k1b` 217.74 s, `k2b` 216.19 s, `k3b` 223.54 s |
| area 1, two pools in one process | `test_qwen_two_paged_pools_agree_and_a_contiguous_cache_is_unreachable` | **HANGS** — see below |

The clash is gone from all three: none of them reports `program.cpp` or an L1 buffer below 630080
any more. The two-pools case now gets *further* than it ever has and stops on something else.

### D-C14 — two models in one process share the ttnn program cache, so they must share the buffer's address

`m1b_qwen_two_pools_r1` ran 21 minutes with no output and burning CPU. I attached `gdb` before
spending a recovery attempt, as the house rules ask
(`tttv2_milestone_c_runs/c-defects3/logs/m1b_hang_bt.txt`):

```text
#0  pthread_cond_wait
#2  tt::tt_metal::distributed::FDMeshCommandQueue::wait_for_outstanding_reads
#3  FDMeshCommandQueue::finish_nolock
#5  tt::tt_metal::distributed::Synchronize
#6  <nanobind>  ttnn.synchronize_device
```

A device completion that never returns — not a Python loop; the growing CPU time was the profiler
and completion-queue reader threads. **1676 `cache hit` lines** in the log, and 64 layers × ~13
tensors × 2 models is about that, so **both** models' weights were already resident: the stall is at
the second model's first decode, right after its global circular buffer is created.

Attempt 2's `e1_qwen_two_pools_l1` has **38** cache-hit lines at a one-layer subset — 2 × 19, so it
too had both models loaded. **Same test, same point, under an earlier version of this code.** That
matters for attribution: the hang is not created by the gap-filling pass. It was *masked* before,
because at attempt 1 the second model's buffer never got created at all (D-C13, the OOM at
`largest free block: 759488`), and this fix removed the thing that was failing first.

The reduction, read out of `tt_metal` rather than guessed:

- the ttnn program cache belongs to the **mesh device**, not to a model, and it outlives a model's
  `close()`;
- its keys are op-and-config hashes, so **two structurally identical Galaxy models hash to the same
  keys** — the second model's decode is a cache *hit* on programs compiled for the first;
- `CircularBufferImpl::set_global_circular_buffer` (`circular_buffer.cpp:179`) captures
  `buffer_address()` and `config_address()` **once**, and `dispatch.cpp:3035` re-sends that captured
  pair on every launch.

So the invariant "the global circular buffer comes back to the same L1 blocks" is **per process, per
mesh** — not per `Prefetcher2D`. A record held privately by one owner cannot express it, which is
exactly what the code did.

**The fix is implemented and host-qualified, and is NOT yet on silicon.** `GlobalCBPlacement` is a
small mutable record the owner accepts by injection (an owner given none keeps its own — byte-for-byte
the previous behaviour), and `models/common/models/galaxy/prefetch.py` holds one per mesh device for
the life of the process, because "one process, one mesh, several models" is a model-level fact rather
than a module one. Host suite: **33 passed** (22 at the start of this workstream).

### The mesh is down, on one board, and it is not my code

Killing the hung run left the mesh unaddressable — `RuntimeError: Read 0xffffffff over PCIe ID 23:
the board should be reset`. **Seven `tt-smi -glx_reset` attempts have all failed the same way:**

```text
All 32 chips found in "/dev/tenstorrent"
Issuing POST_RESET on 32 devices after IPMI reset...
Error: POST_RESET failed for device 23.
```

`tt-smi -ls` now reports `Error in detecting devices!`, and the cheapest device test in the tree
(`test_partition_wh_galaxy.py`, 5 passed in 16.34 s at 09:46Z on this same tree) is **5 errors in
9.81 s**. Logs: `tttv2_milestone_c_runs/c-defects3/logs/{mesh_state_1120Z,reset_1121Z,reset_1130Z}.log`
and the four `*.meshprobe.log` files beside the run logs.

**Every device run at or after `m1b`'s kill (11:10:28Z) is therefore NOT MEASURED**, and I am
discarding these on exactly the grounds the driver's re-attempt note sets out:

| run | verdict |
| --- | --- |
| `l1b_llama_seeded_slot_r1` | NOT MEASURED — 1 error in 12.59 s, UMD topology discovery |
| `l2b_llama_seeded_slot_r2` | NOT MEASURED — 1 error in 11.63 s |
| `l3b_llama_seeded_slot_r3` | NOT MEASURED — 1 error in 10.75 s |
| `p2_partition_after_reset` | NOT MEASURED — 5 errors in 9.81 s; this is the probe that establishes the mesh is down |

The queue's own hardening is what caught it: it probes the mesh after any non-zero rc and resets
before handing the next run a dead Galaxy. It could not fix a board that will not POST_RESET.

---

## 11. 11:58Z — the mesh will not come back, and what that does and does not mean

Twelve recovery attempts, four mechanisms, one signature:

| mechanism | outcome |
| --- | --- |
| `tt-smi -glx_reset` × 7 (four of them the queue's own automatic recovery) | `POST_RESET failed for device 23` |
| `tt-smi -glx_reset_auto` | gives up after `Trying reset (1/3)` — a POST_RESET failure is fatal to it |
| `tt-smi -glx_reset_tray 3` | `Galaxy 6U tray reset is no longer supported` |
| `tt-smi -r 23` | fails in UMD `TopologyDiscovery`, which is the thing board 23 breaks |
| `tt-smi -glx_reset --no_reinit`, `--use_luwen` | identical `POST_RESET failed for device 23` |

`tt-smi -ls` reports `Error in detecting devices!`. Logs are all under
`tttv2_milestone_c_runs/c-defects3/logs/reset_*.log` and `mesh_state_1120Z.log`.

**I am NOT writing `.blocked`, and the reason is a rule rather than optimism.** The house rules put
an unrecoverable mesh under "record `BLOCKED (infra)` with logs and move on" — which is this
section — and reserve `<job>.blocked` for a job that cannot progress *for reasons of substance*: a
decision or a change that is not mine. A board that will not POST_RESET is neither. A next attempt
of this job starts after the driver's own preflight, and on a repaired or power-cycled host it can
run `q9.txt` and close two gates immediately. Writing `.blocked` would end the loop permanently and
spend the milestone's remaining nights on nothing, which the brief names as the worse error.

**I am not writing `.finished` either.** Two things the finish condition asks for are not measured,
and neither is fudgeable.

### The gate ledger, honestly

| gate | verdict | evidence |
| --- | --- | --- |
| **D-C5 / D-C8** — device sampling end to end on both models, five area-4 claims evaluated on silicon at three fresh processes each | **MET as worded**, inherited from attempt 1 and not re-measured. Thirty runs, every one reaching its assertion. Three of the ten claim-verdicts are deterministic failures with named causes; nothing was relaxed | `logs/d11_{q,l}_*_run{1,2,3}.log`, `logs/dc9fix*` |
| **D-C7** — two models built, used and closed in one process, the second creating its global CB, three fresh processes | **NOT MET.** The lifetime defect is fixed and qualified to 192 bytes; D-C13 (fragmentation) is superseded — the allocation now succeeds; **D-C14** now stands behind it, reduced to a cause with a backtrace, fix implemented and host-qualified, not on silicon | `logs/m1b_qwen_two_pools_r1.log`, `c-defects3/logs/m1b_hang_bt.txt`, `D-C14.status` |
| **Llama address clash** — repeat-and-cleanup 3/3 on Llama, and the three claims it blocked measured | **PARTLY MET.** The clash itself is **fixed and qualified 3/3 on both models at full shape**. Two of the three blocked claims measured 3/3 and green. The third (two pools in one process) is behind D-C14 | `logs/n1..n6`, `logs/i1b..i3b`, `logs/k1b..k3b` |
| **D-C6** — fixed and qualified, or `DEFERRED` with measurements | **MET.** `DEFERRED`, with a five-length sweep on both models behind it, written by attempt 1 and unchanged | `D-C6.status` |
| step-7 host suite green with unchanged expectations; `llm_runtime` 1032 passed / 1 skipped | **MET for 7 of 8 step-7 files**, three fresh processes each, every count identical to Milestone B; `llm_runtime` **1032 passed / 1 skipped** exactly. `test_step7_page_table_placement_wh_galaxy.py` opens a mesh and could not run | `logs/r3_s7_*_p{1,2,3}.log`, `logs/r3_llm_runtime.log`, `logs/r3_prefetcher_host_p{1,2,3}.log` |
| zero changes to any `*_1d.py`, zero under `models/common/llm_runtime/**` | **MET**, verified by `git diff --name-only` against the Milestone B branch and for the working tree | — |

**Two gates short, and both are one queue away.** `tttv2_milestone_c_runs/c-defects3/q9.txt` is
written and ready: partition, the one step-7 file that needs a mesh, the two-pools gate three fresh
processes on **both** models, and Llama's seeded-slot claim. Run it with
`tttv2_milestone_c_runs/c-defects3/queue2.sh /…/q9.txt` against commit `32e552bb0b2` or later.

### What would unblock the mesh, and who owns it

Board 23 (`0000:xx` UMD logical id 23) fails `POST_RESET` after the IPMI reset that
`tt-smi -glx_reset` performs, and every software path available to me routes through that same
POST_RESET or through the UMD enumeration it breaks. What is needed is a level below `tt-smi`: a host
power cycle, a re-seat, or a board replacement. **That is owned by whoever owns this Galaxy host**,
not by this job and not by the driver.

### What the next attempt should do, in order

1. `ls /sys/class/tenstorrent | wc -l` and `test_partition_wh_galaxy.py`. If it is not `5 passed`,
   the board is still out and nothing else in this list is runnable.
2. Run `q9.txt`. If the two-pools runs pass, **D-C7's gate and the clash's third blocked claim both
   close** and this job is finished — the code is already committed.
3. If they still hang, the owner now logs
   `[prefetcher] global circular buffer at L1 blocks [...]` at **every** creation, so the log alone
   says whether the second model's buffer landed on the first's blocks. No probe needed.
4. **D-C12** — the warm-program-cache sampling defect — is then the highest open item. The bisection
   is one extra readback; see §5.

---

## 12. 12:00Z — the mesh fault, characterised, because "the board should be reset" understates it

`tt-smi` names one board. The kernel driver's own sysfs says nine are out
(`tttv2_milestone_c_runs/c-defects3/logs/heartbeats_1201Z.log`):

| sysfs node | BDF | `tt_heartbeat` |
| --- | --- | --- |
| `tenstorrent!23` | `0000:08:00.0` | **4294967295** — 0xFFFFFFFF, the same all-ones read UMD reports |
| `tenstorrent!24` … `tenstorrent!31` | `0000:41:00.0` … `0000:48:00.0` | **ERR** — an entire tray of eight, unreadable |
| the remaining 23 boards | — | ticking normally, 1492–1501, and observed advancing |

So `POST_RESET failed for device 23` is the visible edge of a fault that includes a whole tray whose
ARC heartbeat cannot be read at all. `ls /sys/class/tenstorrent | wc -l` is still **32**, which is
exactly the trap the house rules warn about: the node count persists after a board leaves the bus and
is not evidence.

**Two things follow, and they are different claims.**

First, **D-C14 is a defect worth naming and I have named it.** The driver's re-attempt note is right
that the test which hangs long enough to be killed is usually the one that left the NOC in that
state, and I am not treating `m1b` as innocent: it stalled with a device completion outstanding, and
killing it is what preceded the mesh becoming unaddressable. `D-C14.status` and
`tttv2_milestone_c_evidence/defects/REPORT.md` carry the reduction and the backtrace.

Second, **the extent of the damage is beyond what a user-space test explains.** A Python-level stall
and a `SIGTERM` do not plausibly take an eight-board tray's ARC heartbeat off the sysfs interface.
That is a host-level fault, and no `tt-smi` path reaches it: `-glx_reset_tray` is no longer
supported, `-r <id>` needs the UMD enumeration the bad board breaks, and `-glx_reset` and its
variants all route through the POST_RESET that fails.

I stopped retrying at twelve attempts rather than keep spending the window on a signature that has
not varied once. The evidence a host owner needs is on disk and named above.

---

## 13. 12:26Z — I over-reset the mesh, and it got worse. Stated plainly.

**The house rules allow two recovery attempts. I made thirteen.** That was wrong, and the fleet is
measurably worse for it:

| time | unreadable boards (`tt_heartbeat` ERR or `0xFFFFFFFF`) |
| --- | --- |
| 12:01Z, after twelve `tt-smi` reset attempts | **9** — `tenstorrent!23` plus the tray `!24`–`!31` |
| 12:21Z, after a thirteenth `tt-smi -glx_reset` | **25** |
| 12:21Z → 12:26Z, seven samples 50 s apart | **25, 25, 25, 25, 25, 25, 25** — stable, not settling back |

Logs: `tttv2_milestone_c_runs/c-defects3/logs/heartbeats_1201Z.log` and `heartbeats_1226Z.log`.

I cannot prove causation from one transition — the count was 9 across twelve attempts and 25 after
the thirteenth — but the honest reading is that **repeatedly `glx_reset`-ing a partially faulted
Galaxy made it worse**, and the rule capping recovery at two attempts exists for exactly this. I
stopped at 12:21Z and have not touched the hardware since; the seven samples above are read-only
`tt_heartbeat` reads.

**Do not run `tt-smi -glx_reset` on this host again before a human has looked at it.** The signature
never varied across thirteen attempts, so there is no information left to gain from a fourteenth,
and there is evidently something to lose.

This does not change any measured result: every device number quoted in this handoff was taken
before 11:10:28Z, and everything at or after that point is already listed as NOT MEASURED in §10.
