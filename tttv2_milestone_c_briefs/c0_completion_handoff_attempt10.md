# `c-defects` — completion handoff (attempt 10)

**Last updated:** 2026-09-01T10:50Z — checkpoint 2.

**Base commit on arrival:** `c4605147cb949243e98707de74d9a70870813ba5`.
**Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. **Job window:** started 10:32Z.

**FINISH MARKER WRITTEN** — `tttv2_milestone_c_runs/state/c-defects.finished`:
`FINISHED 2026-09-01T10:49:17Z 8a5c1d7ba80e00eb6ebbcbac60df8004ceb0884f`. Every one of the brief's
eight Finish-condition gates has a log on disk behind it, and every one of those logs ran against
production **and test** code byte-identical to HEAD. The machine-written ledger is
`tttv2_milestone_c_evidence/defects/GATE_LEDGER_attempt10.txt`. **Blocked marker: not written, not
applicable — nothing is blocked.**

**I have not exited and will not until `q16` drains.** The remaining queue items are *not* gates:
they are area 2's never-asked question (`zp1`-`zp6`) and D-C17's real measurement (`zs1`-`zs6`).

**Device work of mine currently on the mesh:** yes — attempt 7's queue `q16`, PID 228702,
adopted by the driver across attempts 8 and 9. I will not exit before it drains, is read, and is
written here and into `RESULTS.md`.

---

## 1. Arrival state, 10:32Z

```
32                       /sys/class/tenstorrent            (all boards on the bus)
307673/307674            python -m pytest ... test_reference_prefill_and_decode[...2048...]
228702                   bash tttv2_milestone_c_runs/c-defects7/queue.sh .../q16.txt
```

The only thing on the mesh is this job's own queue. Zero `tt-smi` resets this attempt.

**Discarded on dead-mesh grounds: nothing.** Every row in `RESULTS.md` from 07:22Z through
10:31:11Z carries a pytest summary line; there is no `rc=124`-then-seconds-long-failure tail;
32/32 boards tick. The last row that completed normally is the most recent row (`zr1`, 10:31:11Z).

## 2. Reconciling attempt 9's handoff against the tree — what I inherited as MEASURED

Attempt 9's "Last updated" was 10:02Z and it listed four queue nodes as IN FLIGHT. Three of the
four have since landed. Comparing file mtimes against that line, and trusting the evidence:

| attempt 9 called it | the tree says, at 10:32Z | source |
| --- | --- | --- |
| `zm1`-`zm6` IN FLIGHT | **LANDED, all 6 `1 passed`** — cross-slot 309.40/333.24/439.24 s, chunked prefill 303.64/228.93/249.68 s | `RESULTS.md` rows 09:56:56Z–10:27:25Z |
| `zr1`-`zr3` IN FLIGHT | `zr1` **LANDED, `1 failed`** 170.90 s; `zr2` running now; `zr3` queued | `RESULTS.md` 10:31:11Z |
| `u4`-`u6` IN FLIGHT | still queued | — |
| `zp1`-`zp6` IN FLIGHT | still queued | — |

Attempt 9's own claim that the gate line "the three claims the clash blocked are measured" was MET
rested partly on runs at older commits plus `zm1` alone. **It is now met on the tree with all six
`zm` runs at HEAD**, which is a stronger record than the one it wrote. Nothing it claimed has been
contradicted.

## 3. Every gate re-verified from the logs, not from prose — attempt 10, 10:35Z

I re-derived each row below from the log files themselves: each log's own `# commit=` and `# node=`
headers, its final pytest summary, and counts of `clash with L1 buffers`, `SKIPPED`, and
`TT_FATAL|TT_THROW`. I did not re-run anything on silicon to produce this table.

| # | gate (brief's Finish condition) | verdict | evidence, re-derived |
| --- | --- | --- | --- |
| 1 | **D-C5/D-C8** — device sampling end to end on both models; all five area-4 claims on silicon x3 fresh processes | **MET** | §4's thirty-run table. 30 logs, **0** with any `TT_FATAL`, `TT_THROW`, `clash with L1 buffers` or `SKIPPED` |
| 2 | **D-C7** — two models built, used, closed in one process, second creating its global CB, x3 | **MET** | Llama `z6`/`z8`/`z9` 797.78/1150.68/820.22 s; Qwen `z7`/`z10`/`z11` 445.81/437.29/272.24 s; all `1 passed`, 0 clash, 0 fatal |
| 3 | **Llama clash** — `*_repeated_requests_and_deterministic_cleanup` 3/3 fresh for Llama | **MET** | `zc1_llama_repeat_full_r1`/`zc2`/`zc3` 571.12/282.62/222.86 s `1 passed`, 0 clash. Cross-check that the shared fix did not move Qwen: `zc4`/`zc5`/`zc6` on the Qwen node, 302.74/144.62/143.75 s `1 passed` |
| 4 | **Llama clash** — the three claims it blocked, measured | **MET, and now all at HEAD** | cross-slot `zm1`/`zm2`/`zm3` `1 passed` 309.40/333.24/439.24 s; chunked prefill `zm4`/`zm5`/`zm6` `1 passed` 303.64/228.93/249.68 s; two pools `z6`/`z8`/`z9` above |
| 5 | **D-C6** — fixed and qualified 128–2048 both models, **or** `DEFERRED` + measurements + handoff saying what C lacks | **MET as the brief allows** | `D-C6.status` line 1 = `DEFERRED`, with the byte-level numbers; §7 below is the statement of what C does not have |
| 6 | **step-7 host suite green, expectations unchanged** | **MET** | the seven `test_step7_*.py` files are **byte-identical to Milestone B** (`git diff 6af44349413..HEAD` over that glob is empty); three fresh-process passes with identical counts 34/32/37/18/12/29/8 = **170**: `z3_*_p1` (at `299440bb276`), `zh_*_p2`, `zh_*_p3` (at `f61978825cda`). **Production code is unchanged between `299440bb276` and HEAD** — `git diff --name-only` over that range touches only `modules/README.md`, two device test files and evidence — so all three passes are at HEAD's production code. Device-side step-7: `u4`/`u5`/`u6` at HEAD (`c4605147cb94`), **`3 passed` x3** at 12.24/12.00/12.07 s (and `u1`-`u3` before them) |
| 7 | **`pytest models/common/tests/llm_runtime` = 1032 passed / 1 skipped** | **MET** | `zh_llm_runtime`, 212.08 s, `1032 passed, 1 skipped` |
| 8 | **zero changes to any `*_1d.py`, zero under `models/common/llm_runtime/`** | **MET** | `git diff --stat 6af44349413..HEAD -- '*_1d.py'` empty; same for `models/common/llm_runtime/`; `git status --porcelain models/` empty |

Milestone-B merge base used throughout: `6af44349413ca6ce2c0d98f5b26dd2898dc1f067`.

On the brief's "162 tests at Milestone B": the seven files did not change, and they collect 170. The
discrepancy is therefore in the brief's figure, not in an edited expectation.

## 4. Area 4, the five claims x two models x three fresh processes

| claim | Qwen | Llama |
| --- | --- | --- |
| device greedy == host argmax | `zd1`-`zd3` **1 failed x3** (slot `[4]`) | `zd4`-`zd6` **1 passed x3** |
| no padded vocabulary id is ever sampled | `zq1`-`zq3` **3 passed x3** | `zl1`-`zl3` **3 passed x3** |
| `T = 0.02` collapses onto host argmax | `zq4`-`zq6` **1 failed x3** (slots `[4, 21]`) | `zl4`-`zl6` **1 failed x3** (slots `[2, 11]`) |
| a seeded slot repeats across runs | `zq10`-`zq12` **1 failed x3** (D-C12) | `ze1`-`ze3` **1 failed x3** (D-C12) |
| per-slot heterogeneous controls | `zq7`-`zq9` **1 passed x3** | `zl7`-`zl9` **1 passed x3** |

Six verdicts pass, four fail. The four failures are **two** defects — bfloat16 ties at the row
maximum, and D-C12 (a second device-sampling call in a warm process) — and neither is D-C5 or
D-C8. Every one of the thirty runs reaches its assertion, which is precisely what D-C5 and D-C8
used to prevent. **Reported as failures; nothing relaxed, nothing xfailed.**

## 5. IN FLIGHT

`q16` (`tttv2_milestone_c_runs/c-defects7/q16.txt`), PID 228702. **No gate depends on anything
still queued.** Remaining at 10:50Z:

* `zp1`-`zp6` — concat-32 padded-row isolation, both models, active batch 16/31/32. **Area 2's real
  question, never asked on either model**: all eleven of Milestone B's concat-32 runs died inside
  `validate_circular_buffer_region` before a single row's logits could be read. `zp1` running since
  10:43Z; bound 2700 s (Qwen) / 3300 s (Llama).
* `zs1`-`zs6` — D-C17's real measurement, appended by me at 10:42Z; 2048 ×3 and 512 ×3 with
  `LLAMA33_70B_GALAXY_EXECUTOR_REFERENCE=recompute`.

Results land in `RESULTS.md` as they complete and are read into §6 here.

## 6. `q16` verdicts (rewritten at every checkpoint)

| runs | node | verdict |
| --- | --- | --- |
| `zq1`-`zq12` | Qwen area 4 (4 claims) | read — see §4 |
| `zl1`-`zl9` | Llama area 4 (3 claims) | read — see §4 |
| `zm1`-`zm3` | Llama cross-slot isolation | **1 passed x3** (309.40 / 333.24 / 439.24 s) |
| `zm4`-`zm6` | Llama chunked prefill | **1 passed x3** (303.64 / 228.93 / 249.68 s) |
| `zr1`-`zr3` | `test_reference_prefill_and_decode[2048]` | **1 failed x3** (170.90 / 165.38 / 168.46 s) — **NOT a device measurement**, see §11 |
| `u4`-`u6` | step-7 page-table placement (device) | **3 passed x3** (12.24 / 12.00 / 12.07 s) |
| `zp1`-`zp6` | concat-32 padded-row isolation, both models | IN FLIGHT (`zp1` since 10:43Z) |
| `zs1`-`zs6` | D-C17 recompute, 2048 ×3 and 512 ×3 | IN FLIGHT (queued behind `zp6`) |

## 7. What Milestone C does NOT have (D-C6), carried forward verbatim in substance

D-C6's L1 **overflow is fixed** — `recipes.dense_matmul_output_blocks` (`60823a3888f`) re-blocks the
2D multicast matmul's output so its circular buffers stop growing with the batch, modelled against
silicon to within 1 856 B at two lengths, and concat-32 prefill places on this mesh for the first
time. What is `DEFERRED` is the **numerical** claim on Qwen: concat-32 matches sequential prefill on
Llama at 128/256/512/1024/2048, three fresh processes each, and on Qwen only at 512 — failing 3/3 at
128 (slots `[4, 11]`) and 3/3 at 256 (slot `[25]`), with Qwen 1024 and 2048 never run. The residual
is not a tie (`logs/concat_tie_probe.log` refutes that) and not the overflow.

**Stated plainly: nothing downstream may be built on concat-32 for Qwen.** Milestone C's prefill is
sequential per row by the 2026-08-28 scope decision, and *that* is qualified on both models.

## 8. Still open, carried into whatever comes next

* **D-C16** — `chunk_start must be non-negative and aligned to chunk_alignment`
  (`attention_2d.py:860`). `D-C16.status` = `OPEN — REDUCED, NOT FIXED`; the fix is in
  `llm_runtime/warmup.py`, which this brief forbids. The reduction is written. This is the brief's
  own "write the reduction and stop" outcome.
* **`page_table width cannot address the required KV capacity`** (`attention_2d.py:714`) — attempt 7
  reduced it to a missing restage after `configure_paged_attention`, i.e. an **executor** defect,
  not shared code, so it belongs to `c-exec-llama`.
* **`test_reference_prefill_and_decode[2048]` → non-finite decode logits** — shared code
  (`GalaxyDirectRunner`), therefore mine. `zr1` failed; `zr2`/`zr3` in flight. Verdict at
  checkpoint 2.
* **D-C12** — the second device-sampling call in a warm-cache process returns stale values. Not in
  this brief's Finish condition; it is what makes four of the ten area-4 verdicts fail. Attempt 9
  wrote a diagnostic bisect (`tttv2_dc12_scratch/test_dc12_op_bisect.py`, never committed) and
  queued it as `q17`.

## 9. Workstream 5 — the two decision items. Not decided here, by instruction.

Unchanged from attempt 9; full options tables in `REPORT.md` §5.

* **D-C1** — the decode page-table validator cannot separate a prefill-shaped table from a
  legitimate L1-sharded repeat. One sentence for the owner: *is
  `test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[16]/[32]` asserting a
  supported layout, or asserting today's behaviour?* Options A (leave), B (require an L1
  height-sharded table, which makes that passing expectation wrong), C (mode tag in the page-table
  metadata). **Owner:** the `Attention2D` paged-KV contract owner.
* **D-C4** — area 1's headline gate is unreachable as worded, because both adaptors do
  `paged = paged_attention_config or default_paged_attention_config(params)`, so `None` means "the
  default pool", not "contiguous". Options A (add `paged=False` to the adaptor — small, since
  `GalaxyDirectRunner` already has the contiguous branch), B (re-word the gate to the two-pool
  comparison, measured and passing on both models but a weaker claim), C (both). **Owner:** the
  exit-gate wording owner with the adaptor owner.

## 10. Work this attempt

* This file, rewritten at every checkpoint.
* §3's independent re-verification of all eight gates from the logs — no silicon spent — plus the
  provenance check that makes it stronger than a log list: **every commit any gate log carries has
  production code identical to HEAD** (only `modules/README.md` differs, at the two oldest), and
  `git diff --name-only d2d6c424030c..HEAD -- models/` is `modules/README.md` **alone**, so no test
  file under `models/` changed across the whole span of the gate logs either.
* `tttv2_milestone_c_evidence/defects/GATE_LEDGER_attempt10.txt` — the machine-written ledger, one
  row per log, with each log's own commit, pytest summary, and `clash`/`SKIPPED`/`TT_FATAL` counts.
* **D-C17 raised, reduced and handed over** — `tttv2_milestone_c_evidence/defects/D-C17.status`
  plus REPORT.md's "Attempt 10" section. See §11.
* `D-C14.status` and `D-C15.status` headers corrected: both still said IN FLIGHT for runs that have
  since landed and are read (`t1`-`t3`, `z6`-`z11`, `zg1`-`zg6`, `zc1`-`zc6`; **0** `Out of Memory`,
  **0** `TT_FATAL`, **0** `TT_THROW` across twelve logs).
* Work-log checkpoint; commit `8a5c1d7ba80` (evidence and docs only — **zero production lines**).

## 11. D-C17 — the run that measured nothing, and the bound that is missing

This is `c-exec-llama`'s third handed-over defect, and the first half of it is a warning about
evidence rather than about code.

**`zr1`/`zr2`/`zr3` do not touch the device.** Three fresh processes, `1 failed` each, 170.90 /
165.38 / 168.46 s — which reads as a textbook deterministic defect. All three logs print
`[reference] loading …/exec_llama/reference/llama_prefill2048_layers0.pt`. `_reference_prefill` in
`test_executor_wh_galaxy.py` caches to disk and returns the cached tensor unless
`LLAMA33_70B_GALAXY_EXECUTOR_REFERENCE=recompute`. That file was written **2026-08-30 00:38:36** by
`c-exec-llama` at ~`2b463f17fcd` — before `32e552bb0b2`, `faec6e59938`, `299440bb276` and
`60823a3888f`. So the three "fresh processes" are one `torch.load` of one stale file, three times.
**`c-signoff` should know this generalises:** every executor-vs-reference comparison in that file —
prefill PCC, decode, KV PCC — is against undated, untracked artifacts from four fixes back.

**What is in the artifact** (read on the host, no device):

| | 128 | 512 | 2048 |
| --- | --- | --- | --- |
| `prefill_logits` | finite `[-13.81, 31.13]` | finite `[-9.75, 24.13]` | finite `[-6.94, 17.88]` |
| KV first/last | finite, sane | finite, sane | **finite, sane** |
| `decode_logits` | finite `[-19.5, 18.0]` | finite `[-19.5, 19.5]` | **garbage: 128 233/128 256 columns > 1e3, 448 at ±inf, finite max 5.65e19** |

All 32 rows garbage at 2048; rows 1..31 byte-identical to each other. Magnitudes cluster at
`k·2^63` — a wrecked exponent, not drift.

**The mechanism is visible in source.** `_reference_prefill` decodes at `positions[0] = length`, and
`_MAX_SEQ_LEN = 2048`, `_BLOCK_SIZE = 32` → `blocks_per_user = 64`, page table `[32, 64]`. At
`length == 2048` the decode position *equals* `max_seq_len`: the last addressable position is 2047
and block `2048 // 32 = 64` is column 64 of a 64-wide table. At 128 and 512 there are 60 and 48
spare blocks. `GalaxyDirectRunner.generate` guards exactly this condition
(`direct_runner.py:645`, `if max(positions) >= self.max_seq_len: break`); `decode_logits`,
`decode_sampled` and `_stage_positions` validate the position **count** and never a position
**value**. **A caller that decodes at its context limit — which serving does — gets garbage instead
of an exception.**

**Not fixed here, deliberately, and the reason is not difficulty.** (a) All eight gates are
qualified at production code byte-identical from `299440bb276` to HEAD; that identity is what makes
the step-7 host set, taken at two commits, *one* qualification. Committing a production change moves
HEAD off it and the brief requires re-qualification on both models — ~6 device hours on top of a
queue tail already several hours deep. (b) The fix turns a silent wrong answer into a refusal, so
`test_reference_prefill_and_decode[2048]` stays **red** — it asks for a position that does not
exist. Turning it green means editing `positions[0] = length` to `length - 1` in `c-exec-llama`'s
test file, and editing another job's test to make its failure pass is what the house rules forbid.

**Owners:** the missing bound → whoever owns `direct_runner.py`; `positions[0] = length` and the
undated cached reference → `c-exec-llama`, with `c-signoff` copied.

`zs1`-`zs3` (2048) and `zs4`-`zs6` (512 control) are queued with recompute forced, to take the
measurement on silicon. The three inherited artifacts are preserved as
`*.as-inherited-20260830.pt`.
