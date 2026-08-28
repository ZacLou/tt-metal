
## 2026-08-28 — `mb-coverage` attempt 4 (device, Galaxy (8, 4))

Ran 18:07Z onwards at `110ba1f0658`, committing `aff4e95dbf6`, `54b9fadb3ff`,
`0e2c0dc50b4`. Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md` §A4.
Run-by-run: `.../coverage/RESULTS_A4.md`. Machine-written verdicts:
`.../coverage/VERDICTS_A4.txt`. **Not finished** — no
`tttv2_milestone_b_runs/state/mb-coverage.finished` marker was written, and §A4
"What is short of the finish condition" says exactly why.

What it established, in the 31 minutes of healthy mesh it had:

* **the Galaxy column user selector is qualified on silicon, 3/3 fresh
  processes, for 49 seconds of mesh.**
  `models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py`
  opened with "This file has never been executed" and `logs2/` agreed. It is the
  file the `GalaxyColumnUserSelector` docstring names as the qualification for
  *"the only unqualified step in the Milestone B device sampling path"*. Both
  cases pass: column `c` gets users `8c..8c+7`, and selector-plus-`Sampling2D`
  reproduces a per-user argmax for all 32 users. That docstring's "**Unqualified.**
  This composition has never run on a Galaxy mesh" is now out of date.
* **area 4 has device measurements for the first time.** With D-C5 and D-C8 both
  removed at the test boundary (public model API only: relocate the logits to
  interleaved DRAM; load the full-grid prefill sub-device manager around the
  sampling call), Qwen's whole area-4 claim set ran. Measured, twice, byte-
  identically: **no padded vocabulary id under any of six policies**; **the same
  seed in the same slot repeats in 32/32 slots**; greedy agrees with the host
  argmax in **7/32**.
* **new finding D-C9, and it explains the 7/32.**
  `GalaxyDirectRunner.decode_sampled` composes the sampled tokens with
  `to_torch_auto_compose`, which infers its composer from topology labels an op
  inherits from its activation rather than from the distribution the mapper
  produced — so it concatenates the eight identical devices of a mesh column
  before it reaches the next column, and `.reshape(-1)[:32]` returns one column's
  eight users four times over. `collectives.compose_galaxy_logits` documents this
  exact trap for the logits tensor one op earlier, and `_compose_rows` was already
  fixed for it; `decode_sampled`, sixty lines below, was not. **Every device
  sampling number taken through `decode_sampled` is a readback measurement until
  this is fixed.** The logits path — and therefore both accuracy numbers and every
  PCC in areas 1, 2, 3 and 5 — is unaffected.
* **four exit-gate lines and two supporting host gates re-measured at this tree**:
  the 1D contract gate (5 failed / 296 passed, the same five ids at a fourth
  commit), the host regression gate (553 passed), `llm_runtime` (1032 passed), and
  the boundary and import greps (0 `_1d.py`, 0 `llm_runtime`, 0 model-named
  imports in any Milestone B directory).

**The mesh went down at 18:37Z and did not come back.** A wrapper `glx_reset`
failed with `POST_RESET failed for device 21`; device 21 then reads `0xffffffff`
and device 7 cannot be opened at all (`ENXIO`), which makes every `tt-smi` reset
path abort at `USER_RESET` before it reaches device 21. The kernel says `Device is
unresponsive, cannot reset` and `FW not running`. Five recovery attempts are
logged in `logs4/recovery*.log`. It needs an operator: an IPMI power cycle or a
host reboot.
