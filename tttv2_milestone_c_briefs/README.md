# Milestone C — executors, runtime integration, tracing, performance

Seven unattended Claude Code jobs that take the two Milestone B tensor models and turn them into
model-owned executors with tracing and measured performance. Driven by `run_milestone_c_jobs.sh`.

| Job | Brief | Device | Typical | What it delivers |
| --- | --- | --- | --- | --- |
| `bootstrap` | *(none — script only)* | no | minutes | Milestone C branch cut from Milestone B, build + venv verified |
| `c-perf-recon` | `recon_perf_baseline.md` | yes | 4–8 h | The TTTv1 baseline **procedure**: it runs on this host, with these commands, in this long |
| `c-defects` | `c0_defects.md` | yes | 10–12 h × n | D-C5, D-C8, D-C7, the Llama L1 address clash, and D-C6 |
| `c-exec-llama` | `c1_exec_llama.md` | yes | 10–12 h | `Llama33_70BGalaxyExecutor`: eager prefill/decode, paged KV, warmup, cleanup |
| `c-exec-qwen` | `c2_exec_qwen.md` | yes | 10–12 h | `Qwen3_32BGalaxyExecutor`, same contracts, no new shared code |
| `c-trace` | `c3_trace.md` | yes | 10–12 h | `TraceCompiler`/`TracedExecutor`, eager-vs-traced parity, mode transitions |
| `c-perf-paired` | `c4_perf_paired.md` | yes | 10–12 h | Paired TTTv1/TTTv2 medians and the absolute targets |
| `c-signoff` | `c5_signoff.md` | no | 2–3 h | Exit-gate verdict, modularity scorecard, `MILESTONE_C_STATUS.md` |

## Scope — read this before anything else

Milestone C was **narrowed on 2026-08-28**, after `tttv2_2d_modules_plan.md` was written.

**In scope:** model-owned executors for both models, common-runtime integration, tracing, and the
paired plus absolute performance gates.

**Out of scope — vLLM, entirely.** No generator, no `VLLMAdapter`, no TT-plugin model/version
routing, no DP=4 logical-lane work, no server or offline smokes. Do not build it, do not test it,
do not write plans for it. `models/common/llm_runtime/lane_group.py` and `vllm_adapter.py` are not
yours to touch or to exercise.

**Prefill is sequential per row. Batch 32 applies to decode.** Batched prefill (concat-32) is
applied only under conditions Milestone C does not need to meet. This has one important
consequence: **D-C6 is a fix to attempt, not a gate.** We want it working and `c-defects` tries to
make it work — but if it cannot be made to fit, sequential prefill is the accepted fallback and the
milestone proceeds. Nothing downstream may be built on concat-32 being available.

The plan's "Milestone C" and "Definition of Done" sections still describe the full scope including
vLLM. That difference is this decision, recorded — **not drift, and not something to fix by editing
the plan.** No job here may edit `tttv2_2d_modules_plan.md`.

## Where this work happens

**In place, in the working checkout:**

```text
/proj_sw/user_dev/ctr-apbernal/tt-metal     branch apbernal/tttv2_wh_glx_2d_modules_milestone_c
```

`bootstrap` cuts that branch from `apbernal/tttv2_wh_glx_2d_modules_milestone_b` and reuses the
existing `build/` and `python_env/`. **The Milestone B branch is read-only from here on**: it is the
signed-off record of Milestone B, and `c-defects` is about to change shared Galaxy code that
Milestone B's evidence describes in byte-level detail. Diff against it, never write to it.

## One lane, always

Every job is strictly sequential, including the host-only ones. There is no parallel host lane, and
that is deliberate: finding **F-C2** records that `models/common/tests/models/galaxy/test_plans.py`
*looks* host-only and opens a cluster, because `ttnn.SubDevice` constructs the `MetalContext`. A
host lane would have had to be trusted not to trip over that. It isn't worth the risk to a device
night.

The driver also refuses to start while a Milestone B run (`ttmb`) is live, because a second driver's
preflight would see the other's pytest as a "holder", wait 900 s, and then run `tt-smi -glx_reset`
**on a live test**.

## Shared context every agent must read

1. `tttv2_milestone_c_brief.md` — **Milestone B's signoff handoff into this milestone.** The
   exit-gate verdict, what C inherits as working with the commands that prove it, the defect ledger,
   and a provenance warning naming which documents in this tree carry dead-mesh evidence. This is
   your primary input; the driver refuses to start `c-defects` without it.
2. `tttv2_2d_modules_plan.md` — "Milestone C", the per-module contracts, "Authoritative Design
   Constraints", and "Extension discipline". Read it for the contracts, not for the scope; the
   scope is the section above.
3. `models/common/llm_runtime/README.md` — the runtime contract. The model owns graph orchestration;
   the model-owned executor is the resource and cleanup root; the common runtime owns planning,
   staging, compilation, tracing, KV allocation, output reads and warmup mechanics.
4. `models/common/modules/README.md` and `models/common/modules/MILESTONE_A_STATUS.md`.
5. `tttv2_milestone_b_evidence/coverage/REPORT.md` §A3 — the measured state of every step-7 claim,
   and the full write-up of D-C5 through D-C8. `RESULTS_A3.md` is the run-by-run index.
6. `models/common/models/MILESTONE_B_STATUS.md` — Milestone B's own verdict page.

**On provenance.** Several pages in this tree were written from dead-mesh evidence and later
contradicted by silicon. Milestone B's signoff names them. Before you plan around any claim in a
status page, check it against a log — and prefer `RESULTS_A3.md` and `VERDICTS_A3.txt`, which are
machine-written, over prose.

## House rules, common to all seven

- **One pytest process on the device at a time.** Never pipe pytest; a pipeline can hand back
  control while the nested process still holds the mesh.
- **Three runs in fresh processes before any device claim.** Three of Milestone A's four defects
  presented as intermittent *passes*, because they read aliased or uninitialised L1. A case that
  flips across processes is a defect, not noise. The same rule applied to a **failure** says the
  opposite thing: three byte-identical failures are a defect, not flakiness — that is how D-C8 was
  qualified.
- **A failing test is a result, not a bug to patch.** Never relax a threshold, tolerance or
  parametrization to turn a failure green. Never delete or `xfail` a test to get past it.
- **Never fabricate.** An honest `BLOCKED` with logs beats an invented pass. If you did not run it,
  say you did not run it.
- **`export HF_HOME=/localdev/ctr-apbernal/hf_data`** — this exact value — before anything that
  loads weights. Verified 2026-08-28: its `hub/` holds 62 GB of `models--Qwen--Qwen3-32B` and
  `models--meta-llama--Llama-3.3-70B-Instruct`. **`/proj_sw/user_dev/hf_data` reaches Llama only**,
  and the value inherited from the shell (`.../hf_data/hub`) is one directory too deep and holds
  only Mistral. Under any wrong value `hf_config_or_skip` turns every real-checkpoint test into a
  **`SKIPPED`** and the run looks green having measured nothing. `mb-qwen` lost a night to this.
  **Treat a `skipped` in a run you meant to count as a failure of the run.**
- **Zero changes to `models/common/modules/**/*_1d.py`.** Sharing a *test* helper across the 1D and
  2D suites is fine and has precedent (`models/common/tests/modules/_hf_reference.py`).
- **`models/common/llm_runtime/**` is now in play, but under the plan's extension discipline.**
  Milestone B changed zero lines of it and that is the baseline to preserve. Before changing an
  execution path: express it with an existing config field or injected collaborator; failing that,
  add a frozen topology-neutral config value; failing that, make the smallest mechanical delegation
  to the resolved config and preserve the previous default **exactly**. If more than config plus
  mechanical delegation is required, stop and write the focused reduction the plan demands before
  changing runtime. **No `is_galaxy`, model-name, architecture or mesh-shape branch may enter a
  common runtime path**, and no runtime file may import a 2D module or a reconstructed model package.
- **Every runtime change needs a focused test that fails without it and passes with it**, plus the
  full 1D runtime regression set green with unchanged expectations
  (`pytest models/common/tests/llm_runtime`, 1032 passed / 1 skipped at Milestone B).
- **No imports from an existing model-named package** — not `models/demos/llama3_70b_galaxy`, not
  `models/common/models/llama33_70b`, not `models/common/models/qwen3_32b`. They are behavioural
  references you may *read*, never dependencies you may import. `c-perf-recon` is the one job that
  *runs* TTTv1 — as an external binary, not as an import into our code.
- **Git**: commit on `apbernal/tttv2_wh_glx_2d_modules_milestone_c`. Never `push`, never `checkout`
  another branch, never `reset --hard` or `stash` work you did not create in this session.
- Every job appends a terse checkpoint to `tttv2_2d_modules_milestone_c_work_log.md` and writes its
  completion handoff progressively, not at the end.

## Do not kill your own process tree

You run inside `timeout … claude -p`, launched by `run_milestone_c_jobs.sh`. The driver's PID is in
`$MC_JOB_DRIVER_PID`. A previous run in this project read its own wrapper in `pgrep -af pytest`,
mistook it for a stuck test, and killed itself 16 minutes in.

```sh
ps -o pid=,ppid=,comm=,args= -p <pid>       # comm must be python/python3/pytest
```

Never signal a PID whose `comm` is `claude`, `timeout`, `bash` or `screen`. Prefer targeting the
exact file: `pkill -f 'python.*pytest.*<the test file you launched>'`, confirmed with `pgrep` first.

## Device run procedure

```sh
ls /sys/class/tenstorrent | wc -l           # must be 32 — this is the authoritative count
ls /dev/tenstorrent | wc -l                 # persists after a board leaves the bus; NOT evidence
tt-smi -ls
pgrep -af 'pytest|ttnn' | grep -v grep      # must be empty (ignore the claude/timeout wrapper)

timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The harness caps a foreground tool call at 600 s. Anything longer goes out as a tracked background
process that you block on before starting the next one. Re-check `pgrep` between runs.

**Run a durable queue for long device sweeps.** Milestone B's coverage job died at 09:21:44Z while
its detached queue ran on for four and a half hours and completed 38 more runs — the device time was
not lost, only the bookkeeping was. Write the queue and its results to disk as they land, so a
session that dies costs transcription and not silicon.

`models/common/tests/models/galaxy/test_partition_wh_galaxy.py` is the cheapest useful thing on this
mesh: no checkpoint, ~13 s, and it tells you the worker envelope is not contiguous and that
sender ∪ worker does not cover the compute grid. **Run it first whenever a decode program aborts on
placement** — most of Milestone B's nine silicon defects were that fact, rediscovered nine times.

On a hang or crash: kill the tree, confirm the device is free, `tt-smi -glx_reset`, confirm
`Re-initialized 32 boards`, retry. **Maximum 2 recovery attempts**, then record `BLOCKED (infra)`
with logs and move on. Before spending a recovery attempt on a hang, capture a traceback — a
repeating `faulthandler.dump_traceback_later` pytest plugin (diagnostic only, never committed)
located Milestone A's D3 stall in two dumps 90 s apart.

A `TT_FATAL` abort inside a multi-subdevice program leaves the mesh un-drainable — teardown blocks in
`FDMeshCommandQueue::~FDMeshCommandQueue`. Budget a kill and a `tt-smi -glx_reset` after any such
abort.

Keep every log from every attempt. Never overwrite a log.

## Three ways a job ends

| Marker | Meaning |
| --- | --- |
| `<state>/<job>.finished` | every gate in the brief's Finish condition is met, with a log behind each |
| `<state>/<job>.blocked` | no further attempt of this job can progress without a change or decision that is not yours |
| neither | it is re-attempted, up to `--attempts` |

`rc=0` is none of them. Milestone B's `mb-llama` attempt 2 exited 0 having fixed nine real defects
and met no gate; that was a good night's work and it was not done.

`.blocked` is held to the **same** evidence standard as `.finished`: three fresh processes, logs on
disk, and a statement of what would unblock it and who owns that. "Blocked" means no further attempt
of this job can progress — not that it is hard, slow, or that one workstream of several is stuck. If
any part of your brief is still advanceable, you are not blocked; you are partway through.
