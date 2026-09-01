# `c-defects` — completion handoff (attempt 9)

**Last updated:** 2026-09-01T09:57Z — checkpoint 3. **The D-C5/D-C8 gate is now MET** (`zl9` landed 09:50:51Z, `1 passed`). **Attempt 8's queue `q16` is STILL
RUNNING on the mesh** (PID 228702, adopted by the driver when attempt 8 exited at 09:21:56Z).
I have not killed it and I will not exit before it drains, is read, and is written into this
file, `RESULTS.md`, `REPORT.md` and the status files.

**Base commit:** `4292d26e47faa07eb9679b001bcf99b45ed14b1d`.
**Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. **Job window:** started 09:34Z.

**Finish marker: not written yet. Blocked marker: not written and not applicable.**

---

## 1. Arrival state, 09:35Z

```
32                       /sys/class/tenstorrent            (all boards on the bus)
284659/284660            python -m pytest ... test_llama_per_slot_heterogeneous_sampling_controls
228702                   bash tttv2_milestone_c_runs/c-defects7/queue.sh .../q16.txt
```

The mesh is busy with **my own job's** work. Zero `tt-smi` resets this attempt so far.

**Discarded on dead-mesh grounds: nothing.** Every row in `RESULTS.md` from 07:22Z onwards carries
a pytest summary line; there is no `rc=124`-then-seconds-long-failure tail, and 32/32 boards are
healthy. The last run that completed normally is the most recent row.

## 2. What I verified on disk rather than re-measuring

Independently re-derived from the logs (not from prose), 09:40Z. Each row: the log's own
`# commit=` header, its pytest summary, and `grep -c 'clash with L1 buffers'`.

| gate | evidence | verified |
| --- | --- | --- |
| Llama clash — `*_repeated_requests_and_deterministic_cleanup` 3/3 | `zc1`/`zc2`/`zc3` 571.12/282.62/222.86 s, `1 passed`, **0 clash lines each** | **MET** |
| same node on Qwen, to show the shared fix does not move Qwen | `zc4`/`zc5`/`zc6` 302.74/144.62/143.75 s, `1 passed`, 0 clash | **MET** |
| D-C7 — two models one process, second creates its global CB, x3, both models | `z6`/`z8`/`z9` Llama 797.78/1150.68/820.22 s; `z7`/`z10`/`z11` Qwen 445.81/437.29/272.24 s; all `1 passed`, 0 clash | **MET** |
| the clash "blocks serving" claim, at HEAD | `y4`/`y5`/`y6` `test_executor_repeated_startup_and_cleanup` (3 startup/serve/cleanup cycles, each prefilling *and* decoding) `1 passed` x3 at 199.68/199.28/199.83 s, `# commit=671802f94648`, 0 clash lines | **REFUTED at HEAD** |

## 3. The driver's clash warning is superseded by the tree, and here is how I established it

The job prompt says the clash "HAS MOVED ON", citing `c1_completion_handoff.md`: trigger is
*a prefill after a decode in the same process*, reproducible in ~110 s, and *it blocks serving*.

**That is a pre-fix account.** `c-exec-llama` measured it at `2b463f17fcd` and exited
2026-08-30T00:02Z; the two clash fixes landed afterwards (`32e552bb0b2`, 2026-08-31 11:32, and
`faec6e59938`, 17:31). The prefill-after-decode shape has since been re-asked at HEAD by this
job's own attempt 7 and it does **not** reproduce:

* `y4`-`y6`, `test_executor_repeated_startup_and_cleanup` — three full startup/serve/cleanup
  cycles in one process, so prefill-after-decode twice per run — `1 passed` x3, 0 clash lines,
  within half a second of each other, all three logs stamped `# commit=671802f94648`.
* `y1`-`y3`, `test_executor_warmup_and_program_identity[decode_first]` — literally
  `warmup_model_decode` then `warmup_model_prefill`, the 110-second reproduction the c-exec-llama
  handoff names — `1 failed` x3 with **0 clash lines**. What they fail on is a different, host-side
  defect (D-C16, `chunk_start` alignment), raised before any device work.

So the clash is fixed in the shape the driver warned about, and I am not spending silicon
re-deriving it. What `c-exec-llama` handed over that is still open is D-C16 and the two items in §6.

## 4. IN FLIGHT

`q16` (`tttv2_milestone_c_runs/c-defects7/q16.txt`). Position at 09:35Z:
`zl7_llama_per_slot_controls_r1`. Remaining: `zl7`-`zl9`, `zm1`-`zm6`, `zr1`-`zr3`, `u4`-`u6`,
`zp1`-`zp6` — 24 runs, estimated ~2.5-3 h. Results land in
`tttv2_milestone_c_evidence/defects/RESULTS.md` as they complete and are read into §5 here.

## 5. `q16` verdicts (rewritten at every checkpoint)

| runs | node | verdict |
| --- | --- | --- |
| `p0_partition` | partition envelope | see RESULTS.md |
| `zq1`-`zq3` | Qwen padded vocabulary | **3 passed x3** (656.18 / 510.17 / 620.36 s) |
| `zq4`-`zq6` | Qwen near-zero temperature | **1 failed x3** (182.73 / 172.23 / 167.22 s) |
| `zq7`-`zq9` | Qwen per-slot controls | **1 passed x3** (159.54 / 167.56 / 164.92 s) |
| `zq10`-`zq12` | Qwen seeded slot | **1 failed x3** (237.45 / 225.75 / 231.25 s) |
| `zl1`-`zl3` | Llama padded vocabulary | **3 passed x3** (1234.84 / 813.37 / 1072.36 s) |
| `zl4`-`zl6` | Llama near-zero temperature | **1 failed x3** (304.95 / 310.33 / 282.99 s) |
| `zl7`-`zl9` | Llama per-slot controls | **1 passed x3** (248.44 / 248.30 / 311.74 s) |
| `zm1`-`zm6` | Llama cross-slot, Llama chunked prefill | IN FLIGHT |
| `zr1`-`zr3` | `test_reference_prefill_and_decode[2048]` | IN FLIGHT |
| `u4`-`u6` | step-7 page-table placement (device) | IN FLIGHT |
| `zp1`-`zp6` | concat-32 padded-row isolation, both models | IN FLIGHT |

## 6. Still open, triaged as work proceeds

* **D-C16** — `chunk_start must be non-negative and aligned to chunk_alignment`. Status file says
  `OPEN — REDUCED, NOT FIXED`; the fix is in `llm_runtime/warmup.py`, which this brief forbids.
  Reduction is written. This is the brief's own "write the reduction and stop" outcome.
* **`page_table width cannot address the required KV capacity`** — attempt 7 reduced it to a
  missing restage after `configure_paged_attention`, i.e. an executor defect, not shared code.
* **`test_reference_prefill_and_decode` at 2048 -> non-finite decode logits** — shared code
  (`GalaxyDirectRunner`), mine. `zr1`-`zr3` are measuring it now.
* **D-C12** — second device-sampling call in a warm process returns stale tokens. Not in this
  brief's finish condition, but it is what makes four of the ten area-4 claim-verdicts fail.
  Investigating host-side while the mesh is busy; see §7.

## 7. Area 4, re-derived from the logs and complete on both models

The D-C5/D-C8 gate wants all five area-4 claims evaluated on silicon, three fresh processes each,
both models. I rebuilt the table from the logs themselves (each log's own `# commit=` header, its
pytest summary, `grep -c 'clash with L1 buffers'`, `grep -cE 'TT_FATAL|TT_THROW'`,
`grep -c SKIPPED`) rather than from any prose:

| claim | Qwen | Llama |
| --- | --- | --- |
| device greedy == host argmax | `zd1`-`zd3` **1 failed x3**, slot `[4]` | `zd4`-`zd6` **1 passed x3** |
| no padded vocabulary id ever sampled | `zq1`-`zq3` **3 passed x3** | `zl1`-`zl3` **3 passed x3** |
| `T = 0.02` collapses onto host argmax | `zq4`-`zq6` **1 failed x3**, slots `[4, 21]` | `zl4`-`zl6` **1 failed x3**, slots `[2, 11]` |
| a seeded slot repeats across runs | `zq10`-`zq12` **1 failed x3** (D-C12) | `ze1`-`ze3` **1 failed x3** (D-C12) |
| per-slot heterogeneous controls | `zq7`-`zq9` **1 passed x3** | `zl7`-`zl9` **1 passed x3** |

**Thirty runs, all read. Zero `TT_FATAL`, zero `TT_THROW`, zero `num_intersections == num_cores`,
zero `must be interleaved`, zero `clash with L1 buffers`, zero `SKIPPED` — 30 logs checked, 0 with
any hit.** Every run reaches its assertion, which is exactly what D-C5 and D-C8 used to
prevent. Six verdicts pass, four fail, and the four failures are two defects — bfloat16 ties and
D-C12 — neither of which is D-C5 or D-C8, and both reported as failures rather than relaxed.

## 8. Two findings this attempt, from evidence already on disk and costing no silicon

**(a) Qwen's near-zero-temperature residual is a MEASURED tie, not a hypothesis.**
`logs/d11_greedy_tie_probe.log` runs the gate case's own `_load`/`_paged_config`/`_distinct_rows`,
the same `tokens=[1]*32`, `positions=[128]*32` and the gate's exact `T=0.02` policy, and prints the
top-two gap per slot. **Exactly three Qwen slots have gap = 0** — two ids attaining the row maximum
in bfloat16 — and they are **slots 4, 12 and 21**. The gate misses `[4, 21]`, byte-identically in
three fresh processes. `torch.argmax` breaks a zero gap by lowest index; a sampler has a 50 %
chance either way. Two of the three zero-gap slots disagreeing, and no fourth slot to explain, is
what a tie looks like. The greedy claim's single residual is the same slot 4, measured directly in
the same log: `host 16 @ 15.375, device 17 @ 15.375, equal=True, gap=0.0, ids sharing the row
maximum=2`. The report's own earlier entry called this "a hypothesis with an arithmetic behind it,
not a measurement" and named exactly this measurement as the thing to take; it had already been
taken and nobody joined it to the gate result. Llama's `[2, 11]` gaps are still unmeasured.

**(b) A correction: the `T = 2.0` half of the near-zero-temperature test proves nothing.**
The test's call order is `decode_logits` (no sampling), then `cold = decode_sampled(T=0.02)`, then
`hot = decode_sampled(T=2.0)`. `hot` is the **second device sampling call in the process** — which
is precisely what D-C12 corrupts. The same structure in `d11_greedy_tie_probe.log` fails the same
way: its first-call `[tie]` half is sane and its second-call `[cold]` half reports `missed=True` in
**all 32 slots**, with device ids like `3212836881` and `1077395535` — float32 bit patterns, not
tokens. So `T = 2.0` agreeing 0/32 is **not** evidence about the reciprocal-temperature convention,
and this report's earlier "D4 is confirmed twice" reading is withdrawn. **D4 is still confirmed on
the `T = 0.02` direction alone**, and that direction suffices: under the inverted convention the
distribution is near-uniform over 32 candidates, so 30 of 32 slots landing on the argmax is a
~`1/32**30` event.

## 9. D-C12: the stale-address story has a hole, and a sharper experiment is queued

D-C12 is qualified (only the FIRST device sampling call in a warm-cache process is correct; four
consecutive calls are all correct with `disable_and_clear_program_cache()` before each,
3 fresh processes, `logs/d11_repeat_sample_probe_run{2,3,4}.log`), and the received explanation is
a program-cache runtime-argument defect: on a cache hit ttnn only rewrites runtime args, so an op
whose new input **address** never reaches them reads the previous call's buffer.

**That mechanism needs an address that moves, and in this probe none should.** The only thing that
differs between the four calls is the contents of one host->device write; every allocation is made
and freed in the same order in every call, so the free list returns to the same state.

The mechanism that fits without a moved address is a **premature readback**: call 0 has to compile
its programs, which stalls the host long enough for the device to finish; on a cache hit there is
no compile, the readback races the still-running program, and at a stable address it returns
exactly what the buffer held before — the previous call's answer. It also explains the model
tests' float32 bit patterns, where the same buffer previously held logits.

I read the three ttnn factories in the chain for the received story's own signature — an address
baked as a plain `uint32_t` instead of a `Buffer*`/`MeshTensor`, which
`tt_metal/api/tt-metalium/program_descriptors.hpp:110-125` documents as the thing the framework
patches on cache hits — and `reduction/{topk,sampling,manual_seed}` have **no** `address()` call
between them; every buffer goes through `emplace_runtime_args` as a `MeshTensor`. That is a
negative result for the received story, not a proof against it.

So `tttv2_dc12_scratch/test_dc12_op_bisect.py` (written this attempt, diagnostic only, never
committed) asks both questions in one 20-second arm:
* every intermediate's **buffer address** and device-0 signature, per call, so a moved address can
  be told from a stale value at a stable address;
* per call, **three reads of the same output**: `read1` immediately as production code does,
  `read2` after `ttnn.synchronize_device`, `read3` after `reset_sub_device_stall_group()` and a
  synchronize. `read1 != read2` is direct proof of a race and names a fix in *our* code. `read1 ==
  read2 != read3` says the same, and that the decode stall group excludes the sampling cores —
  also our recipe's choice. All three equal and stale sends it back to the addresses.

Queued as `dc12_bisect_r1`-`r3` in `tttv2_milestone_c_runs/c-defects9/q17.txt`, to run **after**
`q16` drains. Also queued: `tie_llama_r1`-`r3`, the same tie measurement on Llama.

## 10. Work this attempt

* This file, rewritten at every checkpoint.
* `tttv2_milestone_c_evidence/defects/REPORT.md` — a new "Attempt 9" section carrying §3, §7, §8
  above.
* `tttv2_dc12_scratch/test_dc12_op_bisect.py` — the D-C12 op/race bisect, written not yet run.
* `tttv2_milestone_c_runs/c-defects9/{queue.sh,q17.txt}` — the durable queue for after `q16`.
