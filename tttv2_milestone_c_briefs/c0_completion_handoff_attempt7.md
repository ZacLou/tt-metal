# `c-defects` — completion handoff (attempt 7)

**Last updated:** 2026-09-01T07:25Z — arrived on an idle, healthy mesh with attempt 6's whole
53-run chain already drained. Reconciled it against the tree, verified the tree hygiene gates
by inspection, and launched a 30-run queue (`q16`) whose only purpose is to bring **the whole
of area 4 to one commit on both models**. Nothing else in the finish condition is short.

**Base commit:** `671802f946482360c31c220f4cfbf704c7969334`.
**Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. **Job window:** started 07:07Z.

**Finish marker: not written yet. Blocked marker: not written and not applicable.**

---

## 1. Arrival state — and it is better than attempt 6's handoff says

```
32          /sys/class/tenstorrent
(none)      python.*pytest
(none)      queue.sh
```

The mesh is idle and all 32 boards are on the bus. **Zero `tt-smi` resets run by this attempt.**

Attempt 6's "Last updated" is **2026-09-01T01:14Z**, and its chain drained at **01:36Z**. Five
device runs landed after it wrote, so its §28 is stale in exactly the way the driver warned about
— and this time the stale part is stale *against* the job, not for it.

### Reconciliation: what landed after attempt 6 stopped writing

| run | attempt 6 said | the tree says (`RESULTS.md`, log mtimes) |
| --- | --- | --- |
| `y1_exec_warmup_df_r1` | reported: 1 failed, chunk-alignment | confirmed, 01:14:18Z, 561.12 s |
| `y2_exec_warmup_df_r2` | "queued" | **1 failed**, 219.14 s, 01:19:16Z |
| `y3_exec_warmup_df_r3` | "queued" | **1 failed**, 197.25 s, 01:23:45Z |
| `y4_exec_repeat_cycles_r1` | "the direct test of whether serving is unblocked … queued" | **1 passed**, 199.68 s, 01:28:18Z |
| `y5_exec_repeat_cycles_r2` | "queued" | **1 passed**, 199.28 s, 01:32:26Z |
| `y6_exec_repeat_cycles_r3` | "queued" | **1 passed**, 199.83 s, 01:36:33Z |

**`y4`–`y6` answer attempt 6's own open question, and the answer is yes.**
`test_executor_repeated_startup_and_cleanup` — three startup/serve/cleanup cycles in one process,
the node that failed **3/3** on the L1 address clash at `2b463f17fcd` with
`TT_THROW … clash with L1 buffers … L1 buffer allocated at 542016` — now **passes 3/3 in three
fresh processes** at HEAD, 199.68 / 199.28 / 199.83 s, within half a second of each other.

I inherit all six as measured and have re-run none of them.

**Discarded on dead-mesh grounds: nothing.** The last run of attempt 6's chain (`y6`) completed
normally with a pytest summary, `rc=0`, and the mesh was found idle with 32/32 boards this
morning. There is no post-wedge tail in this results table.

## 2. What that does to the driver's arrival note about the clash

The arrival note says "at this commit the clash blocks serving, and `c-trace` and `c-perf-paired`
are behind it too". **That is no longer true of this tree, and there is now silicon behind saying
so.** It was true of `2b463f17fcd`, the commit `c-exec-llama` measured; the clash fix landed at
`32e552bb0b2` on 2026-08-31, after that job exited.

| `c-exec-llama`'s reproduction | at `2b463f17fcd` | at HEAD |
| --- | --- | --- |
| `test_executor_warmup_and_program_identity[decode_first]` | 3/3 `clash with L1 buffers` | `grep -c` → **0**; fails later, on host, at `chunk_start` alignment (`y1`–`y3`) |
| `test_executor_repeated_startup_and_cleanup` | 3/3 clash at 542016 | **1 passed ×3** (`y4`–`y6`) |

So the "prefill after a decode in the same process" trigger is gone: the decode warmup completes
and the following prefill warmup gets all the way into `Attention2D._validate_prefill`. What
stops the *warmup-identity* node now is a different, host-side defect, reduced in §5.

## 3. Tree hygiene — the two gates that need no silicon, verified this morning

```
git diff --stat apbernal/tttv2_wh_glx_2d_modules_milestone_b..HEAD -- '*_1d.py'                 -> empty
git diff --stat apbernal/tttv2_wh_glx_2d_modules_milestone_b..HEAD -- 'models/common/llm_runtime/' -> empty
git status --porcelain models/                                                                   -> empty
```

**Zero changes to any `*_1d.py`. Zero changes under `models/common/llm_runtime/`.** No
uncommitted change to `models/` at all.

And the reason attempt 6's device results are valid at HEAD rather than only at `299440bb276`:

```
git log --oneline --name-only 299440bb276..HEAD
  671802f9464  evidence only
  f61978825cd  evidence + status files only
  874a0e9da75  models/common/modules/README.md (docs) + evidence
  d2d6c424030  the two step-7 coverage TEST files (adds the close-contract test)
```

**No production code changed after `299440bb276`.** The one source-tree edit is a documentation
file; the one test edit adds a new test function and modifies no existing one. So every run
attempt 6 recorded at `299440bb276` measures the code that is at HEAD.

## 4. Gate ledger

| gate | state | evidence |
| --- | --- | --- |
| **D-C5 / D-C8** — device sampling end to end on both models; all five area-4 claims × 3 fresh processes | **all ten claim-verdicts evaluated**, but across three commits. `q16` (running) brings the remaining seven to HEAD | `D-C5.status`, `d11_*`, `zd*`, `ze*` |
| **D-C7** — two models in one process, second creates its global CB, ×3 | **MET, both models** | `z6`/`z8`/`z9` Llama, `z7`/`z10`/`z11` Qwen |
| **Llama address clash** — `*_repeated_requests_and_deterministic_cleanup` 3/3 + three blocked claims measured | **MET**; `q16` re-measures two of the three blocked claims under the changed `close()` | `llama-address-clash.status`, `zc1`–`zc6` |
| **D-C6** | **DEFERRED with the measurements**, and the handoff says what C does not have (§7) | `D-C6.status` |
| **step-7 host suite green, unchanged expectations; `llm_runtime` 1032/1** | **MET** — 170 ×3, `llm_runtime` 1032 passed / 1 skipped | `z3_*`, `zh_*` |
| **zero `*_1d.py` / `llm_runtime/` changes** | **MET**, verified §3 | `git diff` above |

**The only thing between this job and its finish marker is `q16`.**

## 5. Defects handed over by `c-exec-llama`, triaged against the code (attempt 6's §9, re-checked)

1. **`chunk_start must be non-negative and aligned to chunk_alignment`** — `attention_2d.py`.
   `llm_runtime/warmup.py` builds its prefix-cached warmup case with
   `cached_tokens=layout.block_size` (32); Galaxy's `chunk_alignment` is 128. The module
   validator is right, the Galaxy recipe is right, and the assumption "any block-aligned prefix
   is a valid chunk start" is the runtime's. **This brief forbids changing `llm_runtime`, so the
   reduction is the deliverable and I have stopped at it.** Qualified 3/3 at HEAD (`y1`–`y3`).
   *Naming note:* attempt 6's §28 calls this "D-C13", but `D-C7.status` already uses D-C13 for the
   superseded global-CB fragmentation OOM. To stop that collision spreading I am giving this
   defect **D-C16** and will say so in its status file.
2. **`page_table width cannot address the required KV capacity`** — a staged `(1, 128)` table
   reused against a pool shrunk to `max_num_blocks=95`. The validator is doing its job; the caller
   owes a restage after `configure_paged_attention`. **An executor defect, not shared code.**
3. **`test_reference_prefill_and_decode` at 2048 → non-finite decode logits** — `GalaxyDirectRunner`,
   which *is* shared code and *is* mine. Not in this brief's finish condition. Queued as
   `zr1`–`zr3` because it is cheap and unmeasured at HEAD.

## 6. `q16` — 30 runs, launched 07:10:13Z

`tttv2_milestone_c_runs/c-defects7/q16.txt`, drained by `c-defects7/queue.sh` (attempt 4's
script, its own `flock`, global reset budget 2, halts rather than feeding a sick mesh).

The finish condition asks for area 4's five claims "evaluated on silicon at three fresh processes
each". They are — but across **three commits**, and two of those commits predate both the
program-cache retirement (`faec6e59938`) and the `close()`-releases-weights fix (`299440bb276`),
each of which changes teardown on every one of these paths. `q16` puts the whole of area 4 at one
commit on both models.

| runs | what |
| --- | --- |
| `zq1`–`zq12` | Qwen: padded vocabulary, near-zero temperature, per-slot controls, seeded slot — ×3 each |
| `zl1`–`zl9` | Llama: padded vocabulary, near-zero temperature, per-slot controls — ×3 each |
| `zm1`–`zm6` | the two clash-blocked claims that are not the D-C7 gate: cross-slot isolation, chunked prefill — ×3 each |
| `zr1`–`zr3` | `test_reference_prefill_and_decode[2048]` ×3 (§5.3) |

Not repeated, because they are already at `299440bb276`: greedy vs host argmax on both models
(`zd1`–`zd6`) and the Llama seeded-slot claim (`ze1`–`ze3`).

## 7. What Milestone C does not have (D-C6, stated plainly as the brief requires)

D-C6's L1 overflow **is fixed** — concat-32 places on this mesh, which no Milestone B run ever
achieved. What is `DEFERRED` is the numerical claim on Qwen:

* batched (concat-32) prefill is **not qualified on Qwen**. Llama passes at 128/256/512/1024/2048,
  three fresh processes each; Qwen passes only at 512, fails 3/3 at 128 (slots [4, 11]) and 3/3 at
  256 (slot [25]), and has never been run at 1024 or 2048;
* area 2's real question — do padded rows change an active row's logits at active batch 16, 31, 32
  — is therefore answered for Llama and **not** for Qwen;
* nothing downstream may be built on concat-32 for Qwen. Milestone C's prefill is sequential per
  row by the 2026-08-28 scope decision; sequential prefill is qualified on both models and this
  changes none of that.

The residual is not the overflow and not a bfloat16 tie: concatenated and sequential logits differ
by up to **1.06**, eight ulp at magnitude 15 (`logs/concat_tie_probe.log`). Whether that is a
different accumulation order or a real cross-row effect is **not established**, and no attempt of
this job has separated them.

## 8. IN FLIGHT

`q16`, from 07:10:13Z. I do not exit before it drains, before I read it, and before its results
are in this file and in the evidence pages.

## 9. Deliverables written so far this attempt

| deliverable | state |
| --- | --- |
| `tttv2_milestone_c_evidence/defects/D-C16.status` | **new** — the `chunk_start` alignment defect, reduced to the line, 3/3 at HEAD, and the D-C13 name collision corrected |
| `tttv2_milestone_c_evidence/defects/D-C5.status` / `D-C8.status` | rewritten: they said "NOT MET as of attempt 4 arrival"; all ten claim-verdicts are now evaluated. Earlier text kept below the line |
| `tttv2_milestone_c_evidence/defects/REPORT.md` | attempt-7 section appended: what landed after attempt 6 stopped writing, why its results are results about HEAD, and the two no-silicon gates |
| `tttv2_2d_modules_milestone_c_work_log.md` | attempt-7 checkpoint appended |
| `tttv2_milestone_c_evidence/defects/D-C6.status`, `D-C7.status`, `D-C14.status`, `D-C15.status`, `llama-address-clash.status` | inherited, verified against the logs, unchanged |

## 10. The two decision items — for a human, not for this job (brief §5)

Workstreams 1–4 are done (D-C6 `DEFERRED`, which the brief names as a legitimate outcome), so the
brief's §5 applies. Neither item was touched. Both are set out in full in `REPORT.md` §5; the
one-line versions:

**D-C1 — decode's page-table validator cannot separate a prefill-shaped table from a legitimate
L1-sharded repeat.** A column-sharded decode table is device-locally `(8, 64)`, the replicated
prefill table `(32, 64)`, and `32 % 8 == 0`; the validator discriminates on the row count alone,
and both tables are DRAM-interleaved so `is_sharded()` separates nothing either. **A**: leave it,
and a caller who passes the wrong table gets wrong attention silently. **B**: require an L1
height-sharded table over exactly `rows / users_per_column` cores — the honest discriminator,
which makes two *currently passing* 2D-module expectations wrong. **C**: put a mode tag in the
page-table metadata, a contract change across `GalaxyPagedKVContract`, `Attention2D` and both
models. Four attempts have now declined B as a boundary crossing rather than judged its risk.
**The question is one sentence, and it belongs to whoever owns the `Attention2D` paged-KV
contract:** is `test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats`
asserting a supported layout, or today's behaviour?

**D-C4 — area 1's gate is unreachable as worded.** Both adaptors do
`paged = paged_attention_config or default_paged_attention_config(params)`, so
`paged_attention_config=None` means *the default pool*, not *contiguous*. No argument to
`from_pretrained` produces `spec.paged_attention_config is None`, so the gate's "PCC ≥ 0.99
against the contiguous path" has no contiguous path. **A**: add the missing adaptor argument —
small, because `GalaxyDirectRunner` already has the contiguous branch and `test_bringup_wh_galaxy`
already builds a contiguous cache; it restores the gate as worded. **B**: re-word the gate to the
two-pool comparison Milestone B substituted, which is measured and passing on both models — a
weaker claim about a stronger property, nothing to build. **C**: both. **Owner:** whoever owns
the exit-gate wording, with the adaptor owner.
