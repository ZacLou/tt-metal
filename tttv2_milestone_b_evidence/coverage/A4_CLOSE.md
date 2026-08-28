
## What attempt 4 committed

| commit | what |
| --- | --- |
| `aff4e95dbf6` | `test_{qwen,llama}_device_sampling_claims_behind_dc5_and_dc8` — area 4's whole claim set with **both** known program-construction faults removed at the test boundary, public model API only: the decode logits relocated to interleaved DRAM (D-C5) and the full-grid prefill sub-device manager loaded around the sampling call (D-C8). Plus `cov_run4.sh`, `cov_queue4.sh` and `queue4.txt` |
| `54b9fadb3ff` | `test_qwen_the_selected_column_users_are_the_users_they_claim` — the D-C9 bisection: `select_decode_column_users` only, composed to host, each row's argmax against the host argmax of the same decode step, with the logits' placement printed |
| `0e2c0dc50b4` | `test_qwen_device_sampling_claims_with_an_explicit_token_composition` — area 4's claims again, with the sampled tokens composed by the distribution (`ConcatMesh2dToTensor(dims=(0, user axis))` then mesh row 0) instead of by their topology labels, both compositions printed side by side in one log |

No implementation file was touched. `git diff --name-only bc6ad03bfc2..HEAD` is
432 paths, **0** matching `_1d\.py` and **0** matching `llm_runtime`, and the only
non-test, non-evidence paths in it are the ones attempts 0-3 already owned.

## What is short of the finish condition, precisely

The brief's finish condition is: *all five areas attempted and recorded, the
exit-gate table filled in with measured values rather than quoted ones, repeat
and cleanup exercised, and the handoff written.* Four of those are met. **One is
not**, and no `state/mb-coverage.finished` marker was written:

1. **Five areas attempted and recorded — NOT fully met.** Area 4's focused
   temperature case (D4's reciprocal, the one the brief singles out and warns
   `T = 1.0` cannot test) and its focused seeded case have **never run on
   Llama**, and the padded-vocabulary case has never run on Llama either.
   `a4_l_padded_greedy`, `a4_l_temperature` and `a4_l_seeded` were positions
   12-14 of revision 3's queue when the mesh stopped answering. Qwen's
   `a4_q_temperature` and `a4_q_seeded` — positions 3 and 4 — did not run either;
   what Qwen has instead is the same three claims measured inside the `dc8`
   diagnostic, which is a real measurement of the claims but not the focused
   case the brief describes, and its temperature reading is confounded by D-C9.
2. **Repeat and cleanup — one bullet still never exercised.** The brief's second
   bullet is *"repeated model construction and teardown in one process"*.
   `test_two_models_in_one_process` (Llama) has **zero** device runs across all
   four attempts. It was position 5 in revision 3's queue.
3. **The exit-gate table is filled in and every row has a log**, but the two
   accuracy rows are §A2 measurements defended by a byte-identity argument rather
   than re-measured in a fresh process at HEAD. The brief asks for the
   re-measurement; `a4_l_tf` and `a4_q_tf` were positions 6 and 7. This is a
   weaker gap than 1 and 2 — the argument is checkable and is stated in full — but
   it is a gap, and it is stated as one rather than papered over.
4. **The three-fresh-processes rule** is still unsatisfied for most Llama step-7
   claims: `a4_l_late_capacity`, `a4_l_prefix_then_plain` and `a4_l_mixed_slots`
   remain at **one** passing process each. Their run-2 and run-3 were queued.

`queue4.txt` is the resume point and is consumed destructively, so what is in it
is exactly what has never run. **Do not rebuild it from gaps in this file**;
reconcile against `logs4/queue4.out` and `VERDICTS_A4.txt`, both machine-written.

## What Milestone C inherits from attempt 4, on top of §A3's list

| ID | Needs | What |
| --- | --- | --- |
| **D-C9** | a fix in `direct_runner.py`, and it is a one-liner with precedent | `GalaxyDirectRunner.decode_sampled` composes the sampled tokens with `to_torch_auto_compose`, which follows topology labels rather than the distribution and therefore returns one mesh column's eight users four times over. `compose_galaxy_logits` in the same repo documents the identical trap for the logits tensor one op earlier, and `_compose_rows` was already fixed for it. **Fix D-C9 before anyone reads another device-sampling number**: every area-4 measurement taken through `decode_sampled` is a readback measurement until it is fixed |
| **D-C8** | a design decision, not a line | the selector matmul builds its program over the whole compute grid while the loaded decode sub-device manager owns only `prefetch_sender_cores() \| worker_cores()`. `recipes.rope_core_grids`' docstring already names this defect class and names `_subgrid_cores` as the qualified helper; the decision is whether the sampling path runs inside the decode worker sub-device or whether decode's partition widens |
| **the selector is not the problem** | nothing — it is now qualified | `test_column_user_selector_wh_galaxy.py` passes 3/3 on silicon for 49 seconds of mesh. Its subject is what the `GalaxyColumnUserSelector` docstring calls *"the only unqualified step in the Milestone B device sampling path"*. That docstring's "**Unqualified.** This composition has never run on a Galaxy mesh" is now **out of date and should be corrected** |
| **the mesh** | an operator | device 7 cannot be opened and device 21 reads `0xffffffff`; the kernel says `Device is unresponsive, cannot reset` and `FW not running`. Every `tt-smi` reset path aborts on device 7 before reaching device 21. This needs an IPMI power cycle or a host reboot — see §A4's infra section |
