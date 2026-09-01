# `c-defects` — completion handoff (attempt 8)

**Last updated:** 2026-09-01T08:22Z — arrived to find attempt 7's queue `q16` **still running** on
the mesh (the driver adopted it when attempt 7 exited at 07:47Z). I did not kill it and I will not
exit before it drains, is read, and is written into this file and the evidence pages.

**Base commit:** `671802f946482360c31c220f4cfbf704c7969334`.
**Branch:** `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. **Job window:** started 08:11Z.

**Finish marker: not written yet. Blocked marker: not written and not applicable.**

---

## 1. Arrival state

```
2026-09-01T08:11:15Z
32                       /sys/class/tenstorrent           (all boards on the bus)
254823/254824            python -m pytest ... test_qwen_a_seeded_slot_repeats_across_runs
228702                   bash tttv2_milestone_c_runs/c-defects7/queue.sh .../q16.txt
```

**The mesh is busy with my own job's work, not idle.** Attempt 7 launched `q16` (30 runs, later 33)
at 07:10:13Z and exited at 07:47:00Z with it running; the driver adopted it. At 08:11Z it had
completed `p0`+`zq1`–`zq10` and was inside `zq11`. Zero `tt-smi` resets this attempt.

**Discarded on dead-mesh grounds: nothing.** Every run in `RESULTS.md` since 2026-08-31T17:xx has a
pytest summary line; there is no rc=124-then-seconds-long-failure tail anywhere in the current
table, and 32/32 boards are healthy right now.

## 2. What I verified rather than re-measured

Both no-silicon gates, re-run this morning at HEAD:

```
git diff --stat <milestone_b>..HEAD -- '*_1d.py'                    -> empty
git diff --stat <milestone_b>..HEAD -- 'models/common/llm_runtime/' -> empty
git status --porcelain models/                                      -> empty
git log --oneline --name-only 299440bb276..HEAD | grep '^models/'
    models/common/modules/README.md                                  (docs)
    models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py
    models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py
```

**Zero `*_1d.py` changes, zero `llm_runtime/` changes, and no production-code change after
`299440bb276`** — so attempt 6's device results, taken at that commit, are results about HEAD.

Gate evidence I inherit as measured (checked against `RESULTS.md` rows and log mtimes, not prose):

| gate | evidence | state |
| --- | --- | --- |
| D-C7 — two models one process, second creates its global CB, ×3, both models | `z6`/`z8`/`z9` Llama 797.78/1150.68/820.22 s, `z7`/`z10`/`z11` Qwen 445.81/437.29/272.24 s, all `1 passed` | **MET** |
| Llama clash — `*_repeated_requests_and_deterministic_cleanup` 3/3 | `zc1`–`zc3` Llama 571.12/282.62/222.86 s `1 passed`; `zc4`–`zc6` Qwen, to show the shared fix does not move Qwen | **MET** |
| Llama clash — the three claims it blocked, measured | cross-slot `i1b`/`i2b`/`i3b`; chunked prefill `k1b`/`k2b`/`k3b`; two-pools `z6`/`z8`/`z9` | **MET** (`q16` re-asks the first two at HEAD) |
| step-7 host suite green, expectations unchanged | `z3_*` and `zh_*`, 170 tests over 7 files, ×3 passes | **MET** |
| `llm_runtime` 1032 passed / 1 skipped | `zh_llm_runtime`, 212.08 s | **MET** |
| D-C6 | `D-C6.status` = `DEFERRED` with the measurements; §5 below states what C does not have | **MET as the brief allows** |
| D-C5 / D-C8 — all five area-4 claims ×3 on both models | all ten claim-verdicts evaluated, but across three commits | **`q16` IN FLIGHT** — brings the whole of area 4 to one commit |

## 3. IN FLIGHT

`q16` (`tttv2_milestone_c_runs/c-defects7/q16.txt`, 33 runs). Results land in
`tttv2_milestone_c_evidence/defects/RESULTS.md` as they complete. Read this file's §4 for the
verdicts; it is rewritten at every checkpoint.

## 4. `q16` results so far

| run | verdict |
| --- | --- |
| `p0_partition` | see RESULTS.md |
| `zq1`–`zq3` Qwen padded vocabulary | **3 passed ×3** (656.18 / 510.17 / 620.36 s) |
| `zq4`–`zq6` Qwen near-zero temperature | **1 failed ×3** (182.73 / 172.23 / 167.22 s) |
| `zq7`–`zq9` Qwen per-slot controls | **1 passed ×3** (159.54 / 167.56 / 164.92 s) |
| `zq10` Qwen seeded slot | **1 failed** (237.45 s) |
| the rest | IN FLIGHT |

## 5. What Milestone C does not have (D-C6), carried forward unchanged

D-C6's L1 overflow **is fixed** — concat-32 places on this mesh. What is `DEFERRED` is the
numerical claim on Qwen: batched (concat-32) prefill passes on Llama at 128/256/512/1024/2048 ×3,
and on Qwen passes only at 512, failing 3/3 at 128 (slots [4, 11]) and 3/3 at 256 (slot [25]), with
1024 and 2048 never run. Area 2's question — do padded rows change an active row's logits at active
batch 16, 31, 32 — is therefore answered for Llama and not for Qwen. Nothing downstream may be
built on concat-32 for Qwen; Milestone C's prefill is sequential per row by the 2026-08-28 scope
decision, and that is qualified on both models.

## 6. What I added to the queue, and why — area 2's question has never been asked

The brief's §4 says that if D-C6's overflow is fixed, *"area 2's real question becomes askable for
the first time: do padded rows change an active row's logits at active batch 16, 31 and 32? All
seven of Milestone B's runs died on the overflow before a single row's logits could be inspected,
so **nothing about padding-row isolation has been measured in either direction.**"*

**The overflow is fixed (`60823a3888f`) and the question has still never been asked — on either
model.** `test_{qwen,llama}_concat32_padded_rows_change_no_active_rows_logits` has **zero rows in
`RESULTS.md`** and zero logs in `logs/`. Milestone B's six attempts at it
(`tttv2_milestone_b_evidence/coverage/logs2/a3_{q,l}_concat_active{16,31,32}.log`) all died inside
`validate_circular_buffer_region`, exactly as the brief says.

`D-C6.status` and attempt 7's handoff both say area 2's question is *"answered for Llama and not
for Qwen"*. **That is wrong, and I am not carrying it forward.** What is answered for Llama is a
different test — `test_llama_concat32_matches_sequential_prefill_at_each_length`, concat-32 versus
sequential at five lengths. Padding-row isolation is a separate node and it is unmeasured on both.

So I appended six runs to the live `q16` (`>>` preserves the inode; the queue's open fd sees them):

| runs | node | processes |
| --- | --- | --- |
| `zp1`–`zp3` | `test_qwen_concat32_padded_rows_change_no_active_rows_logits`, all three `active` levels | 3 fresh |
| `zp4`–`zp6` | `test_llama_concat32_padded_rows_change_no_active_rows_logits`, all three `active` levels | 3 fresh |

This is not a finish-condition gate. It is the one measurement the brief names that the fix made
possible and nobody has taken, and it is also the sharpest discriminator available for D-C6's open
question: if 16 padding rows can be changed completely without moving one bit of an active row's
logits, cross-row contamination is ruled out and the concat-versus-sequential residual is an
accumulation-order effect. If they do move, that is a real cross-row defect and a far more serious
finding than the argmax disagreement it was hiding behind.

## 7. D-C12, reviewed on the host and deliberately NOT opened

D-C12 — the second sampling call in a warm process returns stale or garbage tokens — is, after the
clash, the highest open defect in the ledger, and it is what makes `a seeded slot repeats across
runs` fail on **both** models (`zq10`–`zq12` Qwen, `ze1`–`ze3` Llama). It is not one of this brief's
five workstreams and not in its finish condition. I read the code rather than spend silicon on it,
and I am recording why I stopped:

* **The obvious Python-level cause is already handled.** `Sampling2D.release()`
  (`models/common/modules/sampling/sampling_2d.py:502`) releases all eight `LazyBuffer`s, resets
  `_device_buffers_loaded` **and** `delattr`s every cached handle, so a second runner cannot be
  holding a dangling `_local_indices`/`_index_offsets`/`_seeds` handle. `LazyBuffer.update()`
  (`lazy_buffer.py:121`) really does `copy_host_to_device_tensor` into the same handle, so per-call
  `top_k`/`top_p`/`temperature`/`seed` writes really do reach the device. Neither of the two
  cheapest hypotheses survives reading.
* **`ttnn.sampling`'s `output_tensor=` is not implicated on this path.** Both models'
  `sample_decode` call `Sampling2D.decode_forward` without `tt_out_tok`, so the op allocates its own
  output; the preallocated-output-with-a-cached-program shape is not what is happening here.
* **What is left is attempt 1's own bisect**, and it points outside this job. `logs/d11_repeat_
  sample_probe_run{2,3,4}.log`: four consecutive sampling calls on four different inputs are all
  correct with `disable_and_clear_program_cache()` between them and only the first is correct
  without it, three fresh processes. A cached ttnn program serving a later call with earlier
  addresses is a **ttnn op** defect, and the root fix for one is C++. **This job's fixed parameters
  forbid rebuilding tt-metal**, so `c-defects` cannot land that fix, and the only Python-level
  workaround in reach — clearing the program cache around sampling — trades a correctness defect for
  a performance one that `c-perf-paired` would then own. I have not made that trade and I do not
  recommend it silently.

The reduction that would settle it is still attempt 3's and still costs one model load: compose
`select_decode_column_users`' output on each of four calls and see whether staleness is already
present before `Sampling2D` is entered. That splits a twelve-op chain in two for one extra readback.

## 8. Deliverables this attempt

Nothing beyond this file yet. Checkpointing continues as `q16` lands.
