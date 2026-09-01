# `c-defects` — completion handoff (attempt 10)

**Last updated:** 2026-09-01T10:36Z — checkpoint 1.

**Base commit on arrival:** `c4605147cb949243e98707de74d9a70870813ba5`.
**Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. **Job window:** started 10:32Z.

**Finish marker: not written yet.** **Blocked marker: not written, not applicable.**

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
| 6 | **step-7 host suite green, expectations unchanged** | **MET** | the seven `test_step7_*.py` files are **byte-identical to Milestone B** (`git diff 6af44349413..HEAD` over that glob is empty); three fresh-process passes with identical counts 34/32/37/18/12/29/8 = **170**: `z3_*_p1` (at `299440bb276`), `zh_*_p2`, `zh_*_p3` (at `f61978825cda`). **Production code is unchanged between `299440bb276` and HEAD** — `git diff --name-only` over that range touches only `modules/README.md`, two device test files and evidence — so all three passes are at HEAD's production code. Device-side step-7: `u1`/`u2`/`u3` `3 passed` x3; `u4`-`u6` re-asking at HEAD, queued |
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

`q16` (`tttv2_milestone_c_runs/c-defects7/q16.txt`), PID 228702. Remaining at 10:36Z:
`zr2`, `zr3` (~3 min each), `u4`-`u6` (~1 min each), `zp1`-`zp6` (concat-32 padded-row isolation,
both models; never asked on either model before; bound 2700/3300 s each). Results land in
`RESULTS.md` as they complete and are read into §6 here.

## 6. `q16` verdicts (rewritten at every checkpoint)

| runs | node | verdict |
| --- | --- | --- |
| `zq1`-`zq12` | Qwen area 4 (4 claims) | read — see §4 |
| `zl1`-`zl9` | Llama area 4 (3 claims) | read — see §4 |
| `zm1`-`zm3` | Llama cross-slot isolation | **1 passed x3** (309.40 / 333.24 / 439.24 s) |
| `zm4`-`zm6` | Llama chunked prefill | **1 passed x3** (303.64 / 228.93 / 249.68 s) |
| `zr1`-`zr3` | `test_reference_prefill_and_decode[2048]` | `zr1` **1 failed** (170.90 s); `zr2`/`zr3` IN FLIGHT |
| `u4`-`u6` | step-7 page-table placement (device) | IN FLIGHT |
| `zp1`-`zp6` | concat-32 padded-row isolation, both models | IN FLIGHT |

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
* §3's independent re-verification of all eight gates from the logs (no silicon spent).
