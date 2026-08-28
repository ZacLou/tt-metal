# `mb-signoff` attempt 2 — Milestone B exit-gate verdict

Run 2026-08-28, unattended, from `tttv2_milestone_b_briefs/job4_signoff.md`. **Host-only. No device
was taken and none was needed** — the Galaxy has been unusable since 18:37:08Z, an operator problem
this job cannot fix and did not try to.

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit read | `e912a8267bb` |
| Commit written | `f94b37f9f93` |
| Milestone A final (`$A`) | `bc6ad03bfc2` — re-derived at run time as `git merge-base HEAD gongyu/tttv2_wh_glx_2d_modules`, which is also that branch's tip |
| Evidence | this directory; eighteen checks in `logs2/s2_*.log` |
| Window | 19:24Z → 20:15Z |

## Verdict

**Milestone B does not pass its exit gate.** Eight of the nine lines pass; line 9 — existing 1D model
contract and demo-contract host tests — **fails**, 5 of 301. And the gate table is not the whole
milestone: **three of the plan's ten *Milestone B tests* items are red or unreachable** and a fourth
is partial.

The full account is `models/common/models/MILESTONE_B_STATUS.md`, with the verdict in the first
screen. This report records what *this job* did, so the verdict can be audited rather than believed.

**The reason has inverted since the last time the verdict was written, and that is the headline.** The
2026-08-27 signoff pass recorded `NOT PASSED` because *no numerical result of any kind had ever been
produced on silicon for either model*. Every one of those claims is now false. Milestone B is held by
named defects, not by an absence of measurement.

## What this job ran

Eighteen checks, all host-only, all logged.

| Log | What |
| --- | --- |
| `s2_01_boundaries.log` | gate 7, 8 and the brief's literal gate-6 `git grep` |
| `s2_02_imports.log` | gate 6 as import statements, over Milestone B's seven directories and the wider `models/common/tests` sweep |
| `s2_03_1d_contract_gate.log` | **gate 9** — 21 non-galaxy `test_demo_contract.py`/`test_hf_adaptor.py` files |
| `s2_04_host_regression_filtered.log` | the brief's regression command with device suites filtered out |
| `s2_05_host_regression_literal.log` | the brief's regression command **literally**, unfiltered |
| `s2_06_gate_log_trace.log` | every exit-gate number, read out of its raw log by this job |
| `s2_07_run_counts.log` | how many fresh processes each accuracy/demo gate actually got |
| `s2_08_provenance.log` | per-commit `git diff … -- models/` for every gate log's commit |
| `s2_09_qualification.log` | the per-gate qualification computation (see below) |
| `s2_10_expectations_unchanged.log` | attribution of gate 9's five failures |
| `s2_11_changed_impl.log` | every implementation file Milestone B changed |
| `s2_12_shared_module_diffs.log` | what changed in each pre-existing shared file |
| `s2_13_l3_recipes.log` | L3 / D-B5 / D-B9 re-verified in the source at `HEAD` |
| `s2_14_milestone_a_gate_collection.log` | what Milestone A's 1263-test gate actually collected |
| `s2_15_scorecard.log` | the modularity scorecard, `git diff --numstat` |
| `s2_16_block_gates.log` | provenance for the block, full-model and Q/K-norm gates |
| `s2_17_teardown_hang.log` | **the new finding, D-S1** |
| `s2_18_test_counts.log` | static test-function counts per area |
| `s2_18_host_collect_counts.log` | an abandoned attempt to get those counts by collection — every device suite opens a cluster during collection and the mesh is down. Kept so the abandonment is on the record |

`s2_host_gates.sh` is the script behind `s2_03`, `s2_04` and `s2_05`; `logs2/s2_host_gates.done`
records when it finished.

## The three things this job added that were not in its inputs

### 1. Per-row provenance, which is stronger evidence than the reports had

The brief asks whether each claim was *measured at the final tree*. `mb-coverage` answered that for
the two accuracy rows with a byte-identity argument and said, correctly, that an argument is not a
measurement. This job made it a measurement instead: for **every** gate log, `git diff <its
commit>..HEAD` decides whether any implementation file **or its own test file** changed since. A run
counts only if neither did.

| Gate | Runs at a HEAD-identical tree | Was recorded as |
| --- | --- | --- |
| Llama teacher-forced accuracy | **2** (`a2_45`, `a2_g1`) | one run + an argument |
| Qwen teacher-forced accuracy | **3** — qualified (`a2_33`, `a2_34`, `a2_g12`) | one run + an argument |
| Llama block gate | **3** — qualified (`a2_40,41,42`) | 3 fresh, but at `mb-llama`'s tree, where 12 implementation files have changed since |
| Qwen block gate | **3** — qualified (`a2_73,74,75`) | 3 fresh |
| Llama batch-32 demo | **2** | 1 |
| Qwen batch-32 demo | **4** — qualified | 1 |
| Llama full model | **1** | 1 |
| Qwen full model | **2** | 3 |
| Long context, prefix cache | **1 each** | 1 each |

Four rows come out **stronger** than any report claimed and none comes out weaker. The three Llama
accuracy runs `mb-llama` recorded are *not* at this tree — eleven implementation files and the test
file itself have changed since — but they report the same 501/511, which is corroboration across a
change rather than qualification at the tree, and the page says so in those words.

### 2. D-S1 — a device test that passes and then will not release the mesh

`grep -rl 'TEARDOWN HANG'` over every evidence log directory returns ten logs, and **no `REPORT.md` in
this tree mentions any of them.** Nine are Qwen, one is a Llama bisection probe.

The signal is in one node id. **Every recorded run of
`test_qwen3_32b_galaxy_qk_norm_head_local_..._decode_and_prefill` — six runs, five distinct commits,
*four of them passing* — writes its verdict and then holds all `/dev/tenstorrent` descriptors past the
harness's 90 s grace, is `SIGTERM`ed at the deadline and exits 124.** Three of those four passes are
`a2_70/71/72`, the runs that re-qualified the Q/K-norm claim at the final tree.

The harness's own comment explains hung teardowns after a *failure* — an aborted multi-sub-device
program leaves the mesh un-drainable — so a hang after a pass reads as the same known thing and was
never separated out. **The differential is decisive:** at the same commit, on the same night, through
the same wrapper, `a2_73/74/75_block` exit 0 three times.

The numerical claim stands (PCC ≥ 0.99998 on all 32 devices, 3 fresh processes). What does not stand
is the idea that this test is usable in a suite: it costs a kill and a reset every time.

### 3. D-C1's framing was wrong, and the correction makes it cheaper

`mb-coverage` declined to fix D-C1 because doing so "requires changing an existing expectation", which
its brief forbids. **Both the validator and the expectation are Milestone B's own.** At `bc6ad03bfc2`
there is no `_validate_decode_page_table` and no
`test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats`; there is one
`_validate_page_table` that required `shape[0] > max(users)` — **at least 32 rows** — for decode too.
Milestone B split the validator and corrected decode to the device-local batch, which is right, and in
doing so admitted `32 == 4 × 8`.

So it is a Milestone B contract decision, not a Milestone A expectation to negotiate. And
`attention_2d.py:678-679` already states the guarantee the code does not provide: *"A table sized to
the full physical batch is the prefill layout and is rejected here rather than at the first op."*

## Line 9, and why the failures are not Milestone B's

`5 failed, 296 passed in 83.02s` at `e912a8267bb` — the same five node ids at a **fifth** distinct
commit. Proved unattributable mechanically, not argued:

- `git diff --name-only bc6ad03bfc2..HEAD -- models/common/tests/models/ | grep -v galaxy` → **empty**;
- all five owning test packages, all five owning model packages and `models/demos/utils` → **0**
  changed paths;
- everything Milestone B changed outside `models/common/{models,modules,tests}` and the evidence
  directories → **nothing**.

**And Milestone A's own 1263-test integrated gate never collected them.** Read out of Milestone A's own
log: it collected `tests/llm_runtime/`, `tests/modules/*` and `tests/models/galaxy/` only. Milestone B
is the first milestone to measure this line, and it was red the first time anyone looked. The 102
apparent hits for those package names in that log are parametrization ids in `llm_runtime` tests, not
the files.

## What was written

| File | What |
| --- | --- |
| `models/common/models/MILESTONE_B_STATUS.md` | **New, 634 lines.** Current position, verification status by area, D-B and D-C defect tables, limitations, pending work, the exit-gate table, the modularity scorecard, and a provenance section |
| `models/common/modules/MILESTONE_A_STATUS.md` | **Surgical, +85/−49.** L3 rewritten as CLOSED with a named cost; L1 rewritten as a lifetime problem; the Qwen decoupled-geometry paragraph now closed on silicon; D2's decode half closed; the Attention2D verification row corrected; the "host-tested only" claim on the post-record amendments corrected. **The Milestone A record proper — the 37-case sweep, D1–D5, the scorecard, P4 — is untouched.** Stale paragraphs were **replaced**, not layered over |
| `models/common/modules/README.md` | **+12/−8.** L1 and L3 paragraphs corrected; the D-C1 gap note extended with the provenance correction; the Milestone B section rewritten — it had said "none of it is qualified on hardware" and "never built a model large enough" |
| `tttv2_milestone_c_brief.md` | **New, 285 lines.** vLLM deferral recorded once; what C inherits working with the command behind each; the five defects the deleted brief never knew about; the items routed by name; a provenance warning; the paired performance methodology |
| `tttv2_2d_modules_milestone_b_work_log.md` | Final checkpoint |

Committed as `f94b37f9f93`. **No implementation file changed.** No test was written, deleted, skipped,
`xfail`ed or relaxed; no threshold, tolerance or parametrization was touched.

## Decisions taken without being able to ask

1. **Ran the brief's regression command both ways.** The literal, unfiltered form attempts ~380 cluster
   opens against a mesh an operator has to fix. It was run anyway, because the brief names it
   literally, and a filtered form was run beside it because that is the one that carries signal. Both
   logs are kept. Neither took the mesh in any meaningful sense: every device test fails at cluster
   open in ~7 s.
2. **Did not correct six stale "This file has never been executed" docstrings**, though every one is
   now false. Editing them would break the byte-identity that qualifies five gate rows, on a mesh that
   cannot re-measure them. **A comment fix is not worth a gate row.** Recorded as doc debt in both new
   documents, with the reasoning, and routed to whoever next touches those files on a working mesh.
3. **Left `mb-coverage` attempt 4's detached watcher and transcriber running.** They belong to job 3,
   they stop by themselves at 05:15Z and 05:20Z, and killing another job's recovery machinery
   unattended is not this job's call. They probe every 300 s and touch only `mb-coverage`'s own files.
   **Consequence, stated rather than hidden:** if an operator revives the mesh before 05:15Z the
   watcher will restart `cov_queue4.sh`, and coverage evidence newer than this report's 20:15Z cutoff
   will exist. This page is a snapshot at that cutoff. The mesh was still down at 19:48Z
   (`coverage/logs4/watch4.log`).
4. **Counted the exit gate as nine lines, not eight.** The plan lists eight bullets; every job in this
   set has split "zero changes to 1D module implementation files" into the `_1d.py` and `llm_runtime`
   checks. Keeping nine keeps this page comparable with the coverage evidence.
5. **Recorded the verdict as `NOT PASSED` for two independent reasons**, not one. A literal reading of
   the eight gate bullets fails only on line 9. The milestone's own test list also names concat-32 and
   device sampling, and both are red at every case. Reporting only the first would have been true and
   misleading.

## What this job did not do

- **No device work of any kind.** The brief is host-only and the mesh is broken; both were reasons.
- **Did not attempt mesh recovery.** Five `tt-smi` paths were already spent by `mb-coverage`, the
  brief caps recovery at two attempts, and `rmmod`/IPMI/reboot are operator actions on shared
  hardware.
- **Did not fix D-C1, D-C5, D-C7, D-C8, D-C9 or D-S1.** Every one needs either a product decision or a
  mesh to validate on, and `direct_runner.py` in particular is imported by both `demo.py` files and
  both full-model suites, so editing it would invalidate the provenance this page rests on.
- **Did not edit `tttv2_2d_modules_plan.md`.** Its Milestone C and Definition-of-Done sections still
  include vLLM; the split is recorded in the Milestone C brief instead, as the brief directs.
- **Did not begin any Milestone C work.** The plan gates it and the gate is the point.

## Finish condition

The brief's finish condition is: *`MILESTONE_B_STATUS.md` exists and states a verdict, the
documentation updates are made, the Milestone C brief is written, and every claim in all three traces
to a log.*

| Gate | State |
| --- | --- |
| `MILESTONE_B_STATUS.md` exists and states a verdict | **Met** — `NOT PASSED`, in the first screen |
| `models/common/modules/README.md` updated | **Met** |
| `models/common/modules/MILESTONE_A_STATUS.md` updated, surgically | **Met** — 6 corrections, no restructuring, stale text replaced rather than layered |
| Work-log checkpoint | **Met** |
| `tttv2_milestone_c_brief.md` written | **Met** |
| Every claim traces to a log | **Met** — 18 logs written here, and every figure taken from another job's evidence names the log it came from and was re-read out of that log by this job |

`tttv2_milestone_b_runs/state/mb-signoff.finished` written.
