
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
