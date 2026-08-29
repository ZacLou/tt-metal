# Job `c-perf-recon` — establish the TTTv1 baseline procedure

**Device.** Exclusive WH Galaxy `(8, 4)`. One night, 4–8 h. **This job touches no Milestone C code.**

## Why this job exists, and why it is early

Half of Milestone C's exit gate is a *paired* comparison against TTTv1, and **nobody has ever
measured TTTv1 on this host.** Confirmed 2026-08-28. That makes it the largest unknown in the
milestone and the only work that depends on nothing we are building — so it runs early, on its own
night, before four defect nights and three bring-up nights have been spent.

If TTTv1 turns out not to run here, we need to know now. If it runs, `c-perf-paired` inherits a
known procedure instead of discovering one inside the same night it is supposed to produce twelve
measured runs.

## What this job is NOT

**It is not the gate measurement.** The paired methodology requires the same host, the same commit,
and the same firmware/runtime environment for both arms. Numbers you take tonight are at tonight's
commit and cannot be paired with a TTTv2 arm measured weeks later. `c-perf-paired` re-runs both arms
together.

So do not tune, do not optimise, and do not report a verdict against the absolute targets. **Your
deliverable is a procedure and a feasibility answer**, plus whatever numbers fall out of
establishing it — clearly labelled as unpaired reconnaissance.

## Scope

The existing TTTv1 Galaxy stack is `models/demos/llama3_70b_galaxy/`. It has its own `PERF.md`,
`README.md`, `conftest.py`, and under `demo/`: `demo_decode.py`, `demo_qwen_decode.py`,
`text_demo.py`, `demo_performance.py`, `prefix_caching_benchmark.py`, `text_qwen_demo.py`. Both
models are served from this one package — note `demo_qwen_decode.py` and `text_qwen_demo.py`.

Answer these, for **both** Llama-3.3-70B and Qwen3-32B:

1. **Does it run at all on this host, at this commit?** Cluster opens, weights load, a demo produces
   output.
2. **What is the exact command** for the performance configuration Milestone C is measured against —
   **batch 32, sequence length 507**, with the same precision recipe, prompt corpus, trace, sampling
   and KV setup the TTTv2 arm will use. Read `models/demos/llama3_70b_galaxy/PERF.md` first; it may
   already name it.
3. **Which metrics does it emit, in what units, from which artifact?** TTFT, decode tokens/s/user,
   aggregate decode tokens/s. Name the file and the line. If TTFT is reported, **record its exact
   semantics** — first user ready, last user ready, or per-request mean. This matters: it determines
   whether sequential prefill can meet `TTFT <= 99 ms`, and the answer feeds directly back into
   whether D-C6 needs to be fixed.
4. **What does one measured run cost in wall clock?** The paired job needs `1 unmeasured warmup + 3
   measured runs` per arm per model — twelve runs plus four warmups. If one run is 40 minutes, that
   job does not fit in one night and we need to know before we schedule it.
5. **What is the environment that must be held constant?** Firmware versions, `tt-smi` output,
   driver version, host, checkpoint revision/hash, commit. Record it in the exact form
   `c-perf-paired` can re-assert and compare.
6. **Does it need anything we do not have?** A checkpoint at a different path, a different
   `HF_HOME`, an extra requirement from `models/demos/llama3_70b_galaxy/requirements.txt`, a
   different `MESH_DEVICE` env, a tokenizer, a prompt file. Record precisely.

## Prohibitions specific to this job

- **Do not modify anything under `models/common/models/{galaxy,llama33_70b_galaxy,qwen3_32b_galaxy}`
  or `models/common/modules/` or `models/common/llm_runtime/`.** This job's whole value is that it
  is independent of Milestone C's tree.
- **Do not modify `models/demos/llama3_70b_galaxy/` either.** If TTTv1 needs a fix to run at all,
  **record what it needs and stop** — a patched baseline is not a baseline, and the change would
  have to be justified to whoever owns that package.
- **Do not import anything from `models/demos/` into our code.** You are running TTTv1 as an external
  program, not depending on it. The no-model-named-imports rule is unchanged.
- Do not tune for performance, and do not report pass/fail against the absolute targets.

## Run procedure

House rules in `README.md` govern: one pytest/demo process at a time, never piped,
`HF_HOME=/localdev/ctr-apbernal/hf_data` (verified to reach both models — `/proj_sw/user_dev/hf_data`
reaches Llama only, and a wrong value turns a checkpoint test into a silent `SKIPPED`).

TTTv1 may want a different `HF_HOME` or checkpoint layout than TTTv2 does. **If it does, that is a
finding, not an obstacle** — record both values and how they differ, because `c-perf-paired` has to
satisfy both in one night.

Capture, for every run: the full command line, the environment, `tt-smi -ls` before and after, the
complete stdout/stderr log, and any profiler artifact the tooling produces. Never overwrite a log.

## Deliverables

1. `tttv2_milestone_c_evidence/perf/BASELINE_PROCEDURE.md` — the runnable procedure, one section per
   model: exact command, environment, expected wall clock, where each metric appears, and the TTFT
   semantics. Written so `c-perf-paired` can execute it without rediscovering anything.
2. `tttv2_milestone_c_evidence/perf/ENVIRONMENT.md` — the environment that must be held constant
   between the two arms, in a form that can be re-asserted and diffed.
3. `tttv2_milestone_c_evidence/perf/recon/` — every log, verbatim.
4. Whatever numbers you obtained, in `BASELINE_PROCEDURE.md`, **explicitly labelled unpaired
   reconnaissance at commit `<sha>`** and not compared against any gate.
5. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`, and your completion handoff.

## Finish condition

Write the finish marker only when, for **both** models:

- TTTv1 has been run on this host at this commit and either produced output or failed with a
  recorded, understood reason;
- the exact performance command for batch 32 / sequence 507 is written down, and either executed or
  documented as unreachable with the reason;
- the metric extraction is named — file, line, units — and the **TTFT semantics are stated
  explicitly**;
- one measured run's wall clock is known, so `c-perf-paired` can be scheduled;
- `ENVIRONMENT.md` records everything that must be held constant.

**A clean negative is a complete result.** "TTTv1 cannot run on this host, here is the failure, here
is what it needs" satisfies this job's purpose fully — it converts the milestone's biggest unknown
into a known, which is what the night was for. Write the finish marker for that outcome too, and say
so plainly in the handoff. Do not spend nights trying to make someone else's package work.
