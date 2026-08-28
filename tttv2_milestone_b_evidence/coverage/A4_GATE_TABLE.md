
## The Milestone B exit gate at this tree, §A4

Every row below has a log in `logs3/` or `logs4/` (attempt 4) or in `logs2/`
(attempts 2-3, at a commit whose implementation code is byte-identical to HEAD —
established in §A4 point 2, not assumed). Nothing here is quoted from a handoff.

| Gate line | Verdict | Measured | Command / log |
| --- | --- | --- | --- |
| Llama teacher-forced, batch 1, prefill 512 / decode 511: top-1 ≥ 91%, top-5 ≥ 99% | **PASS** | top-1 501/511 = **98.04%**, top-5 511/511 = **100.00%** | `logs2/a2_g1_llama_tf.log`, 1029.52s. A fresh re-measurement at HEAD was queued as `a4_l_tf` and **did not run** — see §A4's infra section |
| Qwen teacher-forced, batch 1, sequence 512: top-1 ≥ 89%, top-5 ≥ 97% | **PASS** | top-1 498/511 = **97.46%**, top-5 511/511 = **100.00%** | `logs2/a2_g12_qwen_tf.log`, 915.10s. `a4_q_tf` queued, did not run |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, both models | Qwen `logs2/a2_g21_qwen_demo_batch32.log` 153.47s; Llama `logs2/a2_g9_llama_demo_batch32.log` 277.69s | `models/common/models/*_galaxy/demo.py::..._direct_demo_batch32_has_no_cross_slot_contamination` |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, both models, all three geometries | Qwen 117.91s / 136.29s / 245.76s; Llama 357.81s / 641.17s / 721.70s | `logs2/a2_g{3,4,5}`, `a2_g{14,15,16}` |
| Prefix-cached output matches uncached execution | **PASS**, both models | Llama 424.35s, Qwen 158.58s | `logs2/a2_g2_llama_prefix.log`, `a2_g13_qwen_prefix.log` |
| No dependency imports from a model-named implementation package | **PASS for Milestone B**; the pre-existing exception is now a class, not a single case | Milestone B's seven directories: `models.demos` = 0, non-galaxy model package = 0, each. Wider sweep over `models/common/tests`: **24** pre-existing `models.demos` imports, 23 of them `models.demos.utils.*`, none in a file changed since the job-0 base | `logs3/a4_h4_boundary_and_import_gates.log` |
| Zero changes to 1D module implementation files | **PASS** | 0 of **432** changed paths match `_1d\.py` | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` |
| Zero changes to `llm_runtime` | **PASS** | 0 of 432 | same log |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, 5 of 301 — and not owned by Milestone B | **5 failed, 296 passed in 89.32s** at `aff4e95dbf6`; the same five node ids at a **fourth** distinct commit. No expectation was edited. §A3 H5 checked all five owning packages against `bc6ad03bfc2..HEAD`: 0 each | `logs3/a4_h1_1d_contract_gate.log` |

Supporting host gates at this tree, neither of which is one of the nine lines:

| Gate | Result | Log |
| --- | --- | --- |
| the brief's host regression command, 2D modules + galaxy + Llama host suite | **553 passed, 0 failed**, 0 device opens | `logs3/a4_h2_host_gate.log` |
| the same command's third directory, `models/common/tests/llm_runtime` | **1032 passed, 1 skipped, 0 failed**, 0 device opens | `logs3/a4_h3_llm_runtime_host_gate.log` |

**The two accuracy rows are the ones to read carefully.** The brief asks for them
to be re-measured at this tree rather than quoted, and attempt 4 queued exactly
that (`a4_l_tf`, `a4_q_tf`, positions 6 and 7 of revision 3's queue). They did not
run, because the mesh stopped answering at 18:37Z. What is on disk instead is the
argument that they need no re-measurement: `git diff --name-only 1451b192584..HEAD
-- models/` — from the commit both logs are stamped with, to HEAD — returns three
paths, one status `.md` and the two `test_step7_coverage_wh_galaxy.py` files, and
neither test file is imported by `test_full_model_wh_galaxy.py` or by either
`demo.py`. That is a strong argument and it is **not** a measurement. It is
recorded here as an argument.
