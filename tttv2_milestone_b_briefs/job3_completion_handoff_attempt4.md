# Job 3 (`mb-coverage`) attempt 4 → `mb-signoff` (or attempt 5): completion handoff

**Written progressively, as each result landed.** Last updated: **2026-08-28
18:55Z**, by the attempt-4 agent itself.

Status: **STOPPED, not finished.** No `tttv2_milestone_b_runs/state/mb-coverage.finished`
marker was written and none should be written on this evidence. What is short is
listed precisely in "What is still missing" below, and in `REPORT.md` §A4 "What is
short of the finish condition".

Branch `apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Started at
`110ba1f0658`, committed `aff4e95dbf6`, `54b9fadb3ff`, `0e2c0dc50b4` — **test
files and run scripts only, no implementation file touched**.

| file | what |
| --- | --- |
| `tttv2_milestone_b_evidence/coverage/REPORT.md` §A4 | the full account |
| `.../coverage/RESULTS_A4.md` | one row per run, written as it finished |
| `.../coverage/VERDICTS_A4.txt` | machine-written, `grep`-ed out of the logs |
| `.../coverage/logs4/` | every device log, reset log and recovery log |
| `.../coverage/logs3/a4_h*.log` | the four host gates at this tree |
| `.../coverage/queue4.txt` | **the resume point**, consumed destructively |

---

## READ THIS FIRST: the mesh is broken and this job cannot fix it

**At 18:37Z the Galaxy stopped answering, and five recovery attempts failed.**
Do not plan a night around a healthy mesh without checking.

* the wrapper's `tt-smi -glx_reset` after `a4_q_dc8_run2` printed
  **`Error: POST_RESET failed for device 21`** (`logs4/reset_a4_q_dc8_run2.log`);
* PCIe ID **21** now reads `0xffffffff` — every subsequent mesh open dies at
  setup in ~7s with `RuntimeError: Read 0xffffffff over PCIe ID 21: the board
  should be reset`;
* device nodes **1, 3, 5 and 7 cannot be opened at all**: `os.open` raises
  `OSError [Errno 6] No such device or address`. They were fine at 18:33Z (two
  earlier resets that night printed `Re-initialized 32 boards after reset`), so
  this is new damage from the failed `POST_RESET`;
* **`ls /dev/tenstorrent | wc -l` is still 32 and means nothing.** All 32 nodes
  exist. The node existing and the chip answering are different things. Check by
  *opening* each node — `cov_watch4.sh` shows how — and by `tt-smi -ls` exiting 0
  with no `0xffffffff` in its output;
* every `tt-smi` reset path aborts at `USER_RESET` on device 7's `ENXIO` before it
  ever reaches device 21. `-glx_reset` ×2, `-glx_reset_auto` (3 internal
  retries), `-glx_reset_tray 1` (refused: no longer supported), `tt-smi -r all`
  (reset the 28 PCI-visible chips, then failed re-initialising on 21). All five
  logged in `logs4/recovery{1..5}*.log`;
* the kernel agrees and says why (`logs4/recovery_dmesg.log`, driver 2.4.1):
  `tenstorrent: Skipping message 00000011 due to FW not running`,
  `tenstorrent 0000:01:00.0: Device is unresponsive, cannot reset`, and a kernel
  stack trace through `tt_hwmon_read+0x45/0xa0 [tenstorrent]`.

**And attempt 4's own test is a suspect for having caused it.** `dmesg`, dated to
UTC in `logs4/dmesg_dated.log`: **all 7234
`tenstorrent: pin_user_pages_longterm failed: -14` messages in this 29-day boot
fall inside one minute — 18:34Z — which is inside `a4_q_dc8_run2`**, a run of a
test this attempt wrote 40 minutes earlier. A kernel stack trace through
`tt_hwmon_read` follows 23 seconds later, and only then does the reset fail.
`-14` is `EFAULT` pinning user pages for DMA, and the `dc8` case does something
no earlier case did: **six full sub-device-manager cycles in one process**
(`activate("prefill")` around each sampling call, `activate("decode")` for each
new logits fetch), each stopping and restarting the `Prefetcher2D`'s DRAM
prefetch. Against that: the *first* run of the same case survived, and pinned
pages are freed at process exit. **Not established either way** — the two runs
that would settle it are in `REPORT.md` §A4 "What actually broke the mesh".

**The queue now guards against it.** `cov_queue4.sh` counts
`pin_user_pages_longterm failed` and `Device is unresponsive` in `dmesg` before
and after every run and halts, with a `dmesg` tail dumped to
`logs4/kernel_guard_<name>.log`, if either grows. **Note that
`a4_q_dc9_explicit` is position 1 and performs the same six cycles**, so if this
is real the guard will fire on the most valuable run in the queue. That is the
intended behaviour. If it fires, do not just delete `queue4.halt`: read the
`dmesg` first, and consider a one-cycle variant of the case.

**This needs an operator: an IPMI power cycle of the trays, or a host reboot.**
Neither is something an unattended job should do to shared hardware unasked, so
attempt 4 did not.

**A watcher is running and will resume the queue by itself if the mesh comes
back.** `cov_watch4.sh`, detached, logging to `logs4/watch4.log`: every 300s it
opens all 32 nodes and runs `tt-smi -ls`; on the first healthy probe it deletes
`queue4.halt`, starts `cov_queue4.sh` and exits. It will also try
`tt-smi -glx_reset_auto` at most **3** times, 45 minutes apart, never while a
pytest holds a device. It stops at **05:15Z** regardless, so it cannot outlive
this job. **If `logs4/` holds logs newer than the last row of `RESULTS_A4.md`,
the watcher produced them after the agent stopped** — transcribe them from
`logs4/queue4.out` and `VERDICTS_A4.txt` the way §A3-op did, and say who wrote
each row down.

---

## What attempt 4 established in the 31 minutes of healthy mesh it had

### 1. The Galaxy column user selector is qualified on silicon. 3/3. It is not the defect.

`models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py` began
with the sentence **"This file has never been executed"**, and `logs2/` agreed —
no attempt had a log with that stem. It is the file the
`GalaxyColumnUserSelector` docstring names as the qualification for *"the only
unqualified step in the Milestone B device sampling path"*.

**2 passed, 2 passed, 2 passed** — 31.32s, 8.66s, 8.89s, three fresh processes,
`logs4/a4_selector{,_run2,_run3}.log`. Total cost: **49 seconds of mesh**, no
checkpoint, no weights. Column `c` receives exactly users `8c..8c+7` in order,
and selector-plus-`Sampling2D` reproduces a per-user argmax for all 32 users.

The `GalaxyColumnUserSelector` docstring's "**Unqualified.** This composition has
never run on a Galaxy mesh" is **now out of date and should be corrected** —
that is a one-line doc change for whoever owns `collectives.py`.

### 2. Area 4 has device measurements for the first time in four attempts.

Attempt 3 established that every area-4 claim dies in one shared matmul, twice
over: **D-C5** (the decode logits are WIDTH_SHARDED and the bare `ttnn.matmul`
needs INTERLEAVED) and, behind it, **D-C8** (the same matmul builds its program
over the whole compute grid while the loaded decode sub-device manager owns only
`prefetch_sender_cores() | worker_cores()`).

Attempt 4 removed the second obstacle the way attempt 3 removed the first — **at
the test boundary, public model API only** — by loading the full-grid *prefill*
sub-device manager (`model.activate("prefill")`) around the sampling call.
`Sampling2D`'s own grids are inside that envelope, so every program is legal.
It works: `test_qwen_device_sampling_claims_behind_dc5_and_dc8` ran all six
policies to completion, in two fresh processes, byte-identically:

| brief claim | measured |
| --- | --- |
| padded-vocabulary entries can never be sampled | **PASS on Qwen, for the eight users the readback surfaces.** `padded ids sampled in slots []` under **six** policies (greedy, T=0.02, T=2.0, two seeded passes, per-slot heterogeneous). vocab 151936, padded width 19200/device |
| a seeded request in a slot repeats across runs | **PASS on Qwen, same eight users.** `the same seed in the same slot repeated in 32/32 slots` |
| greedy matches the host argmax exactly | **7/32 slots** — and see finding D-C9: that is the readback, not the sampler |

`logs4/a4_q_dc8.log` (185.57s), `logs4/a4_q_dc8_run2.log` (160.59s).

**Read the two PASSes with D-C9 in hand.** The composed vector has 32 entries but they are one mesh column's **eight** users repeated four times, so both positive claims are measured for 8 distinct users and trivially for their 24 duplicates. That is the first device measurement either claim has ever had, and it is not yet the 32-user statement the brief asks for. The Qwen case that would give the 32-user statement is `a4_q_dc9_explicit`, position 1 in `queue4.txt`, never run.

### 3. New finding D-C9, and it changes how every sampling number should be read.

The 32 greedy tokens were `[265, 2631, 1916, 220, 17, 15, 17, 17]` **repeated
four times** — one mesh column's eight users standing in for all four —
byte-identically in both processes. Since the selector is qualified (above), the
fault is the **readback**.

`models/common/models/galaxy/collectives.py::compose_galaxy_logits` already
documents this exact trap for the *logits* tensor one op earlier in the same
graph: `to_torch_auto_compose` infers its composer from `tensor.tensor_topology()`,
an op's output inherits its **activation's** topology labels rather than the
distribution the weight mapper produced, the composer concatenates the wrong mesh
axis, and *"a caller that slices `[:, :vocab_size]` gets no error at all"*.
`auto_compose.py` says the same from the other side: for replicated dimensions
*"the composer will concatenate all replicas, resulting in duplicated data"*.

`GalaxyDirectRunner._compose_rows` was fixed for it and calls
`compose_galaxy_logits`. **`GalaxyDirectRunner.decode_sampled`, sixty lines
further down the same file, still calls `to_torch_auto_compose(sampled)` and then
`.reshape(-1)[:32]`.** `ttnn.sampling`'s output inherits from `gathered_values`,
an `all_gather` over the sampling axis; the eight devices of a mesh column hold
identical tokens and the four columns hold different users, so concatenating the
replicas first and taking the leading 32 values yields exactly eight users four
times over.

**Consequences, and they cut both ways:**

* **every device-sampling number ever taken through `decode_sampled` is a
  readback measurement.** That includes attempt 4's 7/32 and the D4
  reciprocal-temperature reading. Fix D-C9 before anyone reads another one;
* **the exit gate is untouched.** `decode_logits` goes through `_compose_rows`,
  the fixed path — so both teacher-forced accuracy numbers, the batch-32
  slot-isolation cases and every PCC in areas 1, 2, 3 and 5 are unaffected.

**One ambiguity, stated rather than hidden.** Two hypotheses fit everything
measured: (1) the readback composed the wrong axis, so the composed tensor held
**64** values and `[:32]` took eight users four times; or (2) the sampler really
produced only one column's users and the composition is innocent, in which case
the composed tensor held **32**. The separating evidence is that element count,
and `a4_q_dc8` does not print it — a real gap in the case as written. The queued
`a4_q_dc9_explicit` prints it, plus the per-device shape, how many mesh rows are
byte-identical, and the `tensor_topology()` placements of both tensors. Bet on
(1) — the precedent is exact and in the same file, and the host test reproduces
the pattern arithmetically — but it is a bet until that log exists.

The fix is one line with precedent in the same file: compose by distribution,
`ttnn.ConcatMesh2dToTensor(dims=(0, <user axis>))` then mesh row 0 — the mirror
of `compose_galaxy_logits(dims=(3, 0))`, axes swapped because there it is the rows
that carry the vocabulary. Attempt 4 committed a **test** that does this
(`test_qwen_device_sampling_claims_with_an_explicit_token_composition`, commit
`0e2c0dc50b4`, position 1 in `queue4.txt`) and did not change
`direct_runner.py`. **That test has never run** — the mesh died before it was
dequeued. Running it is the single highest-value item on the queue.

### 4. Four exit-gate lines and two host gates re-measured at this tree.

| gate | result | log |
| --- | --- | --- |
| 1D model contract / demo-contract host tests | **5 failed, 296 passed in 89.32s** — the same five node ids attempts 1-3 saw, now at a **fourth** commit. No expectation edited | `logs3/a4_h1_1d_contract_gate.log` |
| host regression gate (2D modules + galaxy + Llama host suite) | **553 passed, 0 failed**, 0 device opens | `logs3/a4_h2_host_gate.log` |
| `models/common/tests/llm_runtime` | **1032 passed, 1 skipped, 0 failed**, 0 device opens | `logs3/a4_h3_llm_runtime_host_gate.log` |
| boundaries + model-named imports | **0** of 432 changed paths match `_1d\.py`; **0** match `llm_runtime`; **0** `models.demos` and **0** non-galaxy model-package imports in any of Milestone B's seven directories | `logs3/a4_h4_boundary_and_import_gates.log` |

**F-C3 is wider than §A3 recorded.** The wide sweep over `models/common/tests`
finds **24** pre-existing `models.demos` imports, not one: 23 are
`models.demos.utils.*` (`llm_demo_utils`, `model_targets`,
`trace_region_sizes`) in 1D demo and demo-contract files, and one is §A3's
`models.demos.deepseek_v3` test-helper import. **None** of the owning files
appears in `bc6ad03bfc2..HEAD` — checked file by file. `mb-signoff` should word
the gate line as "no Milestone B code imports a model-named implementation
package", which is measured and true, and footnote the pre-existing class.

---

## The exit gate, as attempt 4 leaves it

Unchanged from §A3 in every verdict. The measured table with a command behind
every row is `REPORT.md` §A4 "The Milestone B exit gate at this tree".

| Gate line | Verdict |
| --- | --- |
| Llama teacher-forced 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS** — 98.04% / 100.00% |
| Qwen teacher-forced 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS** — 97.46% / 100.00% |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, both models |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, both models, all three |
| Prefix-cached output matches uncached execution | **PASS**, both models |
| No dependency imports from a model-named implementation package | **PASS** for Milestone B; pre-existing exceptions, F-C3, now a class of 24 |
| Zero changes to 1D module implementation files | **PASS** — 0 of 432 |
| Zero changes to `llm_runtime` | **PASS** — 0 of 432 |
| Existing 1D contract/demo-contract host tests green, expectations unchanged | **FAIL**, 5 of 301, and not owned by Milestone B |

**On the two accuracy rows.** The brief asks for them to be re-measured at this
tree rather than quoted. Attempt 4 queued exactly that (`a4_l_tf`, `a4_q_tf`,
positions 6 and 7) and the mesh died first. What is on disk instead is the
argument that they need no re-measurement: `git diff --name-only 1451b192584..HEAD
-- models/` — from the commit both logs are stamped with, to HEAD — returns three
paths, one status `.md` and the two `test_step7_coverage_wh_galaxy.py` files, and
neither test file is imported by `test_full_model_wh_galaxy.py` or by either
`demo.py`. **That is an argument, not a measurement, and it is recorded as one.**

---

## What is still missing, and it is the whole reason no marker was written

1. **Area 4's focused cases have never run on Llama, and two of three have never
   run on Qwen.** `a4_{q,l}_temperature`, `a4_{q,l}_seeded`, `a4_l_padded_greedy`
   — positions 3, 4, 12, 13, 14 of `queue4.txt`. Qwen has all three claims
   measured *inside* the `dc8` diagnostic, which is real, but the focused
   temperature case (D4's reciprocal — the one the brief singles out, and warns
   `T = 1.0` cannot test) has never run on either model, and the `dc8` reading of
   it is confounded by D-C9.
2. **Repeat-and-cleanup's second bullet has never run at all.**
   `test_two_models_in_one_process` (Llama), zero device runs across four
   attempts, position 5.
3. **The three-fresh-processes rule is unsatisfied for most Llama step-7 claims.**
   `a4_l_late_capacity`, `a4_l_prefix_then_plain`, `a4_l_mixed_slots` are at
   **one** passing process each. Their run-2 and run-3 are queued.
4. **D-C7 rests on one observation.** "A closed model does not return its L1" is
   one of the findings this job hands Milestone C and it has exactly one run
   (`a3_q_two_pools`). `a4_q_two_pools_run2/run3` are queued.
5. **Neither test that would measure area 4's arithmetic has run.**
   `a4_q_dc9_bisect_retry` (one activate cycle, decisive on the composition
   question) is position 1; `a4_q_dc9_explicit` (six cycles, the full claim set)
   is position 8.

`queue4.txt` holds **54** items and is consumed destructively, so what is in it is
exactly what has never run. Its header explains the order. **Do not rebuild it from gaps in
`RESULTS_A4.md`**; reconcile against `logs4/queue4.out` and `VERDICTS_A4.txt`,
which are machine-written.

---

## If you are attempt 5

1. **Check the mesh before anything else, by opening nodes, not by counting
   them.** If `tt-smi -ls` exits non-zero or any of `/dev/tenstorrent/{1,3,5,7}`
   raises `ENXIO`, the hardware is still broken and no amount of pytest will
   change that. Say so, record `BLOCKED (infra)`, and spend the night on the
   write-up rather than on 7-second setup failures.
2. **If it is healthy**: `rm -f queue4.halt` and `nohup bash cov_queue4.sh &` from
   the coverage directory. `queue4.txt` is at **revision 4** and its order is not
   simply "highest value first": it leads with the **one-cycle** cases and banks
   the safe never-run claims before any **six-cycle** case, because six
   sub-device-manager cycles in one process is the mechanism suspected of costing
   the mesh. `a4_q_dc9_bisect_retry` is first and is decisive on its own — it
   prints the element count of both compositions, which is the one number that
   separates D-C9's two hypotheses. The reasoning is in the file's header. Do not
   reorder it back without reading that.
3. **Do not re-run** the concat-32 ladder (D-C6 byte-identical at four lengths on
   both models), the `dc5` diagnostics (3/3 both models), or the selector
   qualification (3/3). Those are done.
4. **Do not fix D-C5, D-C8 or D-C9 in implementation code** unless you are
   prepared to re-measure all nine gate rows: `direct_runner.py` is imported by
   `test_full_model_wh_galaxy.py` and both `demo.py` files, so editing it
   invalidates the byte-identity argument the two accuracy rows rest on. §A4
   "Why attempt 4 did not repair D-C5, D-C8 or D-C9" has the full reasoning.

## What Milestone C inherits, added to §A3's list

| ID | Needs | What |
| --- | --- | --- |
| **D-C9** | a one-line fix in `direct_runner.py`, with precedent in the same repo | `decode_sampled` composes sampled tokens by topology labels instead of by distribution and returns one mesh column's eight users four times. Fix it before reading another device-sampling number |
| **D-C8** | a design decision, not a line | the selector matmul's grid is named independently of the sub-device partition that must contain it. `recipes.rope_core_grids` already documents this defect class and names `_subgrid_cores` as the qualified helper. The decision: does the sampling path run inside the decode worker sub-device, or does decode's partition widen? |
| **the selector** | a doc correction | it is qualified now; its docstring still says it never ran |
| **the mesh** | an operator | see the top of this file |
