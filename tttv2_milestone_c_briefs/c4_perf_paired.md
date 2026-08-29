# Job `c-perf-paired` — paired TTTv1/TTTv2 measurement

**Device.** Exclusive WH Galaxy `(8, 4)`. One night, 10–12 h. **This is a measurement job, not an
optimisation job.**

## Read first

`tttv2_milestone_c_evidence/perf/BASELINE_PROCEDURE.md` and `ENVIRONMENT.md`, written by
`c-perf-recon`, plus its completion handoff. They contain the exact TTTv1 command per model, the
metric extraction, the TTFT semantics, the per-run wall clock, and the environment that must be held
constant. **If they do not exist, say so at the top of your handoff and expect not to finish** —
discovering the baseline procedure and producing sixteen runs in one night is not a plan.

Then read `c-trace`'s handoff: the TTTv2 arm is measured through the traced executors it qualified,
and its handoff says which claims are qualified and which are merely observed.

## Methodology, which is not negotiable

Both arms, both models, in **one night, on one host, at one commit, with one firmware**:

- same WH Galaxy;
- same repository commit and firmware/runtime environment;
- same checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling and KV setup;
- **one unmeasured warmup, then three measured runs**;
- compare **medians**;
- retain profiler artifacts and the exact commands.

That is 2 arms × 2 models × (1 warmup + 3 measured) = **16 runs**. Schedule from
`BASELINE_PROCEDURE.md`'s measured wall clock. If they do not fit in one night, run **Llama's two
arms first, completely** — a complete paired result for one model is worth far more than four half
-measured ones, because a half-measured pair cannot be compared at all.

**Assert the environment, do not assume it.** Re-read `ENVIRONMENT.md` and check every constant
before the first run and after the last. If firmware, commit or checkpoint changed between the arms,
the pairing is void and the runs are reconnaissance — say so rather than reporting a comparison that
does not hold.

## The gates

**Paired:** no gated TTTv2 metric may regress by more than **3%** from its paired TTTv1 median.

**Absolute**, batch 32 / sequence length 507:

```text
Llama    TTFT <= 99 ms     decode >= 71.5 tokens/s/user    aggregate decode >= 2288 tokens/s
Qwen     TTFT <= 700 ms    decode >= 60   tokens/s/user    aggregate decode >= 1920 tokens/s
```

**Prefill is sequential per row in this milestone.** This matters most for TTFT, and
`c-perf-recon` was asked to record TTFT's exact semantics for precisely this reason. If TTFT means
*last user ready* at batch 32, sequential prefill of 32 rows is unlikely to reach 99 ms, and that is
a **finding about the scope decision**, not a failure to engineer around. Report the number, state
the semantics under which it was measured, and let `c-signoff` and a human weigh it.

**If an absolute target and the paired baseline disagree materially, stop and document the
environment and baseline discrepancy** — the plan says so explicitly. Do not weaken either gate
silently, and do not pick whichever one the numbers happen to satisfy.

## Prohibitions specific to this job

- **Do not tune, optimise, or change any model, module or runtime code to improve a number.** If
  TTTv2 misses a gate, that is the measurement. Record it, characterise it, and hand it on. A
  milestone that passes because its measuring job changed the thing being measured has measured
  nothing.
- **Do not modify `models/demos/llama3_70b_galaxy/`.** A patched baseline is not a baseline.
- **Do not substitute a different configuration** — a smaller batch, a shorter sequence, an
  untraced path — to get a comparable pair. If the configured measurement cannot be taken, that is
  the result.
- **Production acceptance always uses the configured `Prefetcher2D` path.** A no-prefetch fallback
  may aid diagnosis and can never satisfy a gate. If you measure one, label it as diagnosis.
- No vLLM: no server, no offline throughput harness, no DP.

## Run procedure

House rules in `README.md`. One process on the device at a time, never piped,
`HF_HOME=/localdev/ctr-apbernal/hf_data` for the TTTv2 arm and whatever `BASELINE_PROCEDURE.md`
records for TTTv1 — `c-perf-recon` was asked to record both, because they may differ.

Interleave the arms per model rather than running all of one arm then all of the other, so a drift
in host or mesh state hits both arms rather than biasing one. Record `tt-smi -ls` before and after
every run. Never overwrite a log or a profiler artifact.

## Deliverables

1. `tttv2_milestone_c_evidence/perf/RESULTS.md` — for each model and each metric: three TTTv1 runs,
   three TTTv2 runs, both medians, the percentage delta, the paired verdict, and the absolute
   verdict. One row per metric, with the log and profiler artifact named for every individual run —
   not just the medians.
2. `tttv2_milestone_c_evidence/perf/COMMANDS.md` — every command executed, verbatim, with its
   environment, in the order run.
3. `tttv2_milestone_c_evidence/perf/artifacts/` and `.../logs/` — every profiler artifact and log.
4. An explicit statement of the **TTFT semantics** the numbers were taken under.
5. A checkpoint in `tttv2_2d_modules_milestone_c_work_log.md`, and your completion handoff.

## Finish condition

Write the finish marker when, for **both** models, both arms have one warmup and three measured runs
at one commit and one firmware, the medians and deltas are recorded with a log behind every
individual run, and both the paired and absolute verdicts are stated.

**The finish marker records that the measurement was properly taken — not that the gates passed.**
A complete, correctly paired measurement showing TTTv2 6% behind TTTv1 is a finished job and a failed
gate, and both facts belong in the handoff, stated separately and plainly. `c-signoff` decides the
milestone; you decide only whether the numbers are trustworthy.

Do not write the marker if any arm is incomplete, if the environment drifted between arms, or if any
number came from a configuration other than the one specified.
