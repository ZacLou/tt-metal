# Milestone B Galaxy Model Status

Status of Milestone B of `tttv2_2d_modules_plan.md`, as of **2026-08-28**, at commit `e912a8267bb`
on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Sources: the evidence packages `tttv2_milestone_b_evidence/{reconcile,llama,qwen,coverage,signoff}/`
and their raw logs, `tttv2_2d_modules_milestone_b_work_log.md`, and the four job handoffs in
`tttv2_milestone_b_briefs/`. Every number on this page was read out of a named log file by the job
that wrote the page; where a report asserted a figure this job could not find a log for, the page
says so.

> ## The verdict: Milestone B does not pass its exit gate
>
> **Eight of the nine exit-gate lines pass. The ninth fails**, and it fails on tests Milestone B did
> not write, did not change and cannot break — but "remain green" is what the line says, and they are
> not green ([line 9](#exit-gate-result)).
>
> **And the gate is not the whole milestone.** The plan's own *Milestone B tests* list names
> physical-32 concatenated prefill and device sampling. Both are **red at every case on both models**,
> behind four defects in shared Galaxy code
> ([D-C6](#d-c6), [D-C5](#d-c5), [D-C8](#d-c8), [D-C9](#d-c9)). Neither is one of the eight gate
> bullets, so a literal reading of the gate table would let them through. They are recorded here as
> blocking anyway, because the milestone's test list is part of the milestone.
>
> **The reason for the verdict has inverted since the last time it was written, and that matters more
> than the verdict.** A signoff pass on 2026-08-27 recorded `NOT PASSED` because *no numerical result
> of any kind had ever been produced on silicon for either model* — the mesh was down to 21 of 32
> boards and four gate lines were `NOT REACHED`. **Every one of those claims is now false.** Both
> models run on real `(8, 4)` hardware, both accuracy gates pass with margin, and the milestone is
> held by named defects rather than by an absence of measurement. The two documents that recorded the
> old verdict were deleted at `6983cc52e33` rather than edited, and this page is written fresh.
>
> **What this page is careful about, because Milestone A was not.** Milestone A declared its exit gate
> passed on 2026-08-19, was wrong, and the re-run that disproved it found two real defects the
> "passing" evidence had been masking — three of its four defects presented as intermittent *passes*.
> So every device row below carries how many **fresh processes** produced it *and* whether those
> processes ran at a tree byte-identical to this commit. A row that passed once says
> `passed once — not qualified` in those words. The measured provenance is in
> `tttv2_milestone_b_evidence/signoff/logs2/s2_09_qualification.log`; it is not a quotation from any
> report.

## Scope

Milestone B builds two clean TTTv2 tensor models on the Milestone A 2D module set, for the canonical
Wormhole Galaxy logical mesh `(8, 4)`:

- `models/common/models/llama33_70b_galaxy` — Llama-3.3-70B-Instruct, 80 layers;
- `models/common/models/qwen3_32b_galaxy` — Qwen3-32B, 64 layers, 64 decoupled attention heads
  (`attention_dim 8192 != dim 5120`);
- `models/common/models/galaxy` — everything Galaxy-specific but model-neutral: placement recipes,
  collective-resource plans, the `Attention2D`/`LMHead2D` collective adapters, the prefetch
  construction policy, the paged-KV metadata view, and the direct prefill/decode runner the tests and
  demos drive.

No executor, no `llm_runtime` integration, no tracing and no vLLM: those are Milestone C, and the plan
gates them on this milestone passing.

## Current position

| | |
| --- | --- |
| **Exit gate** | **NOT PASSED.** 8 of 9 lines pass; line 9 (existing 1D contract/demo-contract host tests) **fails**, 5 of 301, and not by Milestone B's doing |
| **Also blocking, though not a gate bullet** | concat-32 physical prefill and device sampling have **no passing case on either model**, behind [D-C6](#d-c6), [D-C5](#d-c5), [D-C8](#d-c8) |
| Both models on real WH `(8, 4)` silicon | **Yes.** Llama-3.3-70B 80 layers and Qwen3-32B 64 layers both prefill, decode, teacher-force and emit coherent text |
| Llama teacher-forced, batch 1, 512/511 | **top-1 501/511 = 98.04%** (gate ≥ 91%), **top-5 511/511 = 100.00%** (gate ≥ 99%) |
| Qwen teacher-forced, batch 1, 512 | **top-1 498/511 = 97.46%** (gate ≥ 89%), **top-5 511/511 = 100.00%** (gate ≥ 97%) |
| Block-level PCC, both models | Llama 0.99958 prefill / 0.99975 decode; Qwen 0.99930 / 0.99936; KV caches ≥ 0.99974 on all four column-local users. **3 fresh processes each, at this tree, bit-identical** |
| Defects found | **27 fixed** on the bring-up path ([D-B1 … D-B27](#defects-found-on-the-bring-up-path), 28 entries with D-B25 split in two, one struck), plus 2 unnumbered; **9 open or routed to Milestone C** ([D-C1 … D-C9](#defects-found-by-step-7-coverage)); **1 new on this page** ([D-S1](#d-s1)) |
| Milestone A limitations | **L3 closed on silicon**, at a named cost. **L1 is worse than recorded**, not better — see [L-B1](#l-b1) |
| Three-fresh-processes rule | Met for both block gates, the Qwen accuracy gate, the Qwen batch-32 demo and every *failure* verdict. **Not met** for the Llama accuracy gate (2), the Llama batch-32 demo (2), the long-context smokes (1 each) and the prefix-cache gate (1 each) |
| Mesh | **DOWN since 2026-08-28T18:37:08Z, the first failed mesh open.** Device 21 reads `0xffffffff`; `/dev/tenstorrent/{1,3,5,7}` raise `ENXIO`; the kernel says `Device is unresponsive, cannot reset`. Five `tt-smi` recovery paths failed. **Needs an operator** — IPMI power cycle or host reboot |
| Working tree | Committed, not pushed. 84 commits over Milestone A final `bc6ad03bfc2`; 432 changed paths |
| Modularity scorecard | **Boundaries held.** 0 `*_1d.py`, 0 `llm_runtime`, 0 model-named imports, nothing outside `models/` and the evidence directories. [Full scorecard](#modularity-scorecard) |

### The five sentences that matter most

1. **Both accuracy gates pass with margin, and neither is the weak link.** Llama clears top-1 by
   seven points; Qwen by eight.
2. **The exit gate fails on a line Milestone B cannot affect.** The five failing 1D demo-contract
   tests are byte-identical to the Milestone A tip, their owning packages are byte-identical, and
   **Milestone A's own 1263-test gate never collected them** — verified here out of Milestone A's own
   log, not accepted from a report. Milestone B is the first milestone to measure that line, and it
   was red the first time anyone looked.
3. **Two whole areas of the milestone's test list have never produced a single reachable case.**
   Concat-32 does not fit in L1 at any supported length, on either model, at byte-identical figures —
   so it is the shared recipe, not per-model tuning. Device sampling is blocked two defects deep, and
   a third defect means every sampled-token number ever read on this mesh is a readback artifact.
4. **The limitation Milestone A called L1 is a lifetime problem, not the ordering problem it was
   written as.** A closed, garbage-collected model still holds 66% of every L1 bank. One Galaxy model
   per process is the operating rule until that is fixed.
5. **Nothing here was made green.** No threshold was relaxed, no test deleted, skipped or `xfail`ed,
   no parametrization narrowed. Two tests that fail deterministically at three fresh processes are
   recorded as failures.

## Verification status

`fresh` counts **`python -m pytest` processes at a tree byte-identical to this commit** — no
implementation file and no owning test file changed between that run's own commit and `HEAD`. Runs at
superseded trees are named where they add corroboration, but they do not count towards the bar. The
computation is `tttv2_milestone_b_evidence/signoff/logs2/s2_09_qualification.log` and
`s2_16_block_gates.log`; both are `git diff` output, not judgement.

Host figures are static test-function counts where marked (`s2_18_test_counts.log`) and pytest
summaries otherwise.

| Area | Host evidence | WH `(8, 4)` device evidence | Status |
| --- | --- | --- | --- |
| **Llama adaptor and configs** | `test_model_host.py` 25 defs + `test_hf_conversion_host.py` 9 defs; converted tensors reproduce unmodified HF `LlamaAttention`/`LlamaMLP` at PCC ≥ 0.9999 | Covered by the block and full-model rows below | **Qualified** |
| **Llama block** — prefill 128 + decode batch 32, logits and both KV caches | in the Llama host selection, **570 passed** (`qwen/logs2/a2_60_host_gate.log`) | prefill 0.999584002863212, decode 0.9997463458407887, K 0.9999347766610057, V 0.9997498179150203 on all four column-local users; prefill 2048 0.9996201066107949. **3 fresh at this tree** (`qwen/logs2/a2_40,41,42_llama_step2.log`), bit-identical; 3 more at a superseded tree (`llama/logs3/a3_32,33,34`) with the same digits | **Qualified.** Closes Milestone A [L3](#l3-closed) at a named cost |
| **Llama full model** — 80 layers, prefill + first decode | as above | `qwen/logs2/a2_44_llama_fullmodel.log`, 1 passed in 824.12s: prefill token in the reference top-5, decode-at-128 token in the reference top-5. **1 fresh at this tree** (+1 superseded, `llama/logs3/a3_43`) | **Passed. One run at this tree, one at a superseded tree — not qualified** |
| **Llama teacher-forced accuracy** — batch 1, prefill 512 / decode 511 | `test_hf_conversion_host.py` 9 defs, 3 fresh processes; the reference loader pinned after it was found returning a length-1 sequence (the unnumbered fix below) | **top-1 501/511 = 98.04%** (gate ≥ 91%), **top-5 511/511 = 100.00%** (gate ≥ 99%). **2 fresh at this tree** — `qwen/logs2/a2_45_llama_accuracy.log` (1293.62s), `coverage/logs2/a2_g1_llama_tf.log` (1029.52s) — plus 3 at superseded trees with **identical counts** | **Gate met. One process short of qualified** |
| **Qwen adaptor and 64-head geometry** | `test_model_host.py` 32 defs + `test_hf_conversion_host.py` 13 defs, **50 passed** (`qwen/logs2/a2_61_qwen_host.log`); rebuilt attention reproduces HF `Qwen3Attention` at PCC ≥ 0.9999 in the real decoupled geometry | `qwen/logs2/a2_01_geometry` on the mesh; then every row below | **Qualified on silicon.** This closes the Milestone A gap that its Attention2D row was recorded against a 40-head geometry no product has |
| **Qwen per-head Q/K norm, alone, both modes** | in the 50 above | prefill q 0.9999821268225385 / k 0.9999833417066442; decode q 0.999988294981757 / k 0.9999879678611943, **identical on all 32 devices**. **3 fresh at this tree** (`qwen/logs2/a2_70,71,72_qknorm.log`) | **Numerically qualified** — and see [D-S1](#d-s1): all three of those processes hung in teardown after passing and were killed |
| **Qwen block** — prefill 128 + decode batch 32 | as above | prefill 0.999303669584255, decode 0.999360219056066, K 0.9998897994661545, V 0.9998944730661905; prefill 2048 0.9990203192392576. **3 fresh at this tree** (`qwen/logs2/a2_73,74,75_block.log`), bit-identical | **Qualified** |
| **Qwen full model** — 64 layers, prefill + first decode | as above | `qwen/logs2/a2_23,28_fullmodel.log`. **2 fresh at this tree** (+1 superseded) | **Passed — one process short of qualified** |
| **Qwen teacher-forced accuracy** — batch 1, 512 | as above | **top-1 498/511 = 97.46%** (gate ≥ 89%), **top-5 511/511 = 100.00%** (gate ≥ 97%). **3 fresh at this tree** — `qwen/logs2/a2_33,34_accuracy.log`, `coverage/logs2/a2_g12_qwen_tf.log` | **Gate met and qualified** |
| **Batch-32 direct demos** — no cross-slot contamination | — | Llama `qwen/logs2/a2_47`, `coverage/logs2/a2_g9` — **2 fresh**. Qwen `qwen/logs2/a2_22,31,32`, `coverage/logs2/a2_g21` — **4 fresh**. 32 slots, 8 distinct prompts × 4; duplicates agree exactly, distinct prompts diverge | **Qwen qualified; Llama one short.** Read the coverage caveat in [what a defect would have to look like](#what-would-have-slipped-through) |
| **Paged KV** | `test_step7_paged_kv.py` 20 defs, all passing | Late capacity resolution **PASS** (Qwen 2 fresh, Llama 1). Two different paged pools agree at PCC ≥ 0.99 on all 32 slots, **across processes** — Qwen 2 arms, Llama 1 arm. Block-level cross-slot isolation **PASS Qwen** (2 fresh), **BLOCKED Llama** ([L-B1](#l-b1) address clash). Both pools in **one** process **FAIL** ([D-C7](#d-c7) on Qwen, the address clash on Llama) | **Partial.** The brief's headline form is not expressible at all ([D-C4](#d-c4)); the reachable substitute passes on both models but only cross-process |
| **Concat-32 physical prefill** | `test_step7_concat32.py` 14 defs, all passing — planning, page tables, source rows | **FAIL, every case, both models, every length.** 1 669 312 B of static circular buffers at length 128 against 1 499 136 B of L1; 3 111 104 at 256; 5 994 688 at 512; 11 761 856 at 1024 — **byte-identical between the two models** | **BLOCKED — [D-C6](#d-c6).** Zero reachable cases. Active batches 16/31/32 cannot be distinguished because none of them fits |
| **Prefix cache / chunked prefill** | `test_step7_prefix_cache.py` 16 defs, all passing | Prefix-cached matches uncached: **PASS** both models, **1 fresh each** (`coverage/logs2/a2_g2`, `a2_g13`). Prefix-then-plain and mixed-slot batches **PASS** both (Qwen 2 fresh, Llama 1). Chunked prefill **PASS Qwen** (2 fresh), **BLOCKED Llama** (address clash) | **Passed — not qualified.** The chunk-aligned SDPA path is qualified on Qwen only |
| **Device sampling** | `test_step7_sampling.py` 20 defs, all passing, including the pinned reciprocal-temperature contract | **BLOCKED in the production shape, on both models, at 3 fresh processes each.** [D-C5](#d-c5) then, behind it, [D-C8](#d-c8). With both bypassed at the test boundary, Qwen ran all six policies and no padded id was sampled and seeded slots repeated — but only **8 distinct users** are visible, because [D-C9](#d-c9) makes the readback return one mesh column four times. The `GalaxyColumnUserSelector` itself is **qualified, 3/3** and is not the defect | **BLOCKED.** No claim in this area has a 32-user measurement |
| **Long context** — batch-1 4K / 32K / 128K | `test_step7_long_context.py` 12 defs, all passing | **PASS** all six, **1 run each** (`coverage/logs2/a2_g{3,4,5}`, `a2_g{14,15,16}`) | **Passed once — not qualified**, and the assertion is only "the token is in vocabulary" |
| **Repeat and cleanup** | `test_step7_repeat_and_cleanup.py` 12 defs, all passing | Repeated requests against one live model: **Qwen PASS 3/3**; **Llama FAIL 3/3**, byte-identical ([L-B1](#l-b1)). Two model constructions in one process: **Qwen FAIL** ([D-C7](#d-c7), **1 observation**); **Llama never run** | **FAIL for Llama, and one bullet of the brief has zero runs** |
| **Galaxy shared layer** (recipes, collectives, plans, direct runner) | 45 host defs across `test_collectives.py`, `test_recipes.py`, `test_direct_runner.py`; `test_plans.py`'s 14 need a cluster despite looking host-only ([F-C2](#f-c2)) | Exercised by every device row above; the mesh-partition suite passes **5/5** (`qwen/logs2/a2_00_partition.log`) | **Qualified for what the models drive**; the sampling path is not |

### Host gates at this commit, run by this job

| Gate | Command | Result | Log |
| --- | --- | --- | --- |
| Exit-gate line 9 — 1D model contract and demo-contract host tests | 21 non-galaxy `test_demo_contract.py` / `test_hf_adaptor.py` files | **5 failed, 296 passed in 83.02s** | `signoff/logs2/s2_03_1d_contract_gate.log` |
| Boundaries and model-named imports | `git diff` + `grep`, both forms | **0 / 0 / 0** for Milestone B | `signoff/logs2/s2_01_boundaries.log`, `s2_02_imports.log` |
| Evidence provenance | per-log `git diff <log commit>..HEAD` | see the `fresh` column above | `signoff/logs2/s2_09_qualification.log`, `s2_16_block_gates.log` |
| Modularity scorecard | `git diff --numstat` | see [scorecard](#modularity-scorecard) | `signoff/logs2/s2_15_scorecard.log` |
| The brief's regression command, **literally**, unfiltered | `pytest -q models/common/tests/{modules,models,llm_runtime}` | **18 failed, 2146 passed, 2058 skipped, 3276 deselected, 379 errors in 1058.34 s.** 18 = 13 [F-C2](#f-c2) (`test_plans.py` needs a cluster) + the 5 line-9 ids. **All 379 errors are cluster opens** on the dead mesh — 392 `Read 0xffffffff over PCIe ID 21` occurrences, 251 of them in the untouched `tests/modules/moe/` suites | `signoff/logs2/s2_05_host_regression_literal.log` |
| The same, with device suites filtered out | `… --ignore-glob="*_wh_galaxy*.py" --ignore=…/test_plans.py` | **5 failed, 2140 passed, 2058 skipped, 251 errors in 873.64 s** — the 251 are the `moe` device suites, which are not named `*_wh_galaxy*` | `signoff/logs2/s2_04_host_regression_filtered.log` |

## Defects found on the bring-up path

Twenty-eight entries were numbered — D-B1 through D-B27, with D-B25 split into `a` and `b`. **One
was struck as not a defect**, twenty-seven were fixed, and two more fixes below carry no ID. All are
in 2D modules, in the shared Galaxy layer, in a model package or in the measurement apparatus. **None
required a change to a `*_1d.py` file or to `llm_runtime`.** Full accounts with quoted aborts:
`tttv2_milestone_b_evidence/llama/REPORT.md` §3, §A2.2, §A3.2 and
`tttv2_milestone_b_evidence/qwen/REPORT.md` §5, §6.

The `How it hid` column is the one worth reading. **Three of these produced a wrong number with no
error anywhere**, and a fourth made the accuracy gate report a *skip*: on this mesh the measurement
apparatus fails quietly more often than the graph does.

| | Defect | How it hid | Fix |
| --- | --- | --- | --- |
| **D-B1** | `RotarySetup2D` built its prefill tables with an on-device clone of the decode table | Host tests build tables from a `LazyWeight`, never from a device tensor, so the clone path had no host coverage at all | Prefill tables materialise from the host source |
| **D-B2** | The decode RoPE shards were placed on prefetch sender cores | A placement is legal until a sub-device manager is loaded over it. Every host test resolves the placement and none loads a partition | Shards resolve inside `worker_cores()` |
| **D-B3** | The Llama embedding decode output was L1-interleaved | Interleaved is a legal memory config; the next op simply cannot be placed. The error surfaces two ops later, in the norm | Sharded to the decode residual placement |
| **D-B4** | `RMSNorm2D` deallocated its own return value | **Latent until Milestone B, then live.** The Milestone A callers all discarded the tensor; the first caller to *use* it read freed memory | The module no longer frees what it returns |
| **D-B5** | Milestone A's [L3](#l3-closed) was recorded as closed by a ring-form decode matmul. It was not: both attention decode matmuls were still `dense_matmul_program_config` on the `(7,1)` grid | **A documentation defect, and no host test could see it.** The reconciliation read `ring_matmul_program_config` in `recipes.py` and inferred that attention used it; only the MLP did | The dense config is confined with `allowed_worker_cores=dense_matmul_worker_rectangle(...)` |
| **D-B6** | The attention all-reduce used an op that cannot run on this partition | The op is correct; it is the *partition* it is illegal on, and no partition is loaded on the host | Partition-compatible collective |
| **D-B7** | `_relocate` reached a full-grid program factory three different ways | Three call shapes, one of which is `ttnn::prim::copy`, which is not sub-device aware and says so nowhere | One relocation helper, sharded-to-interleaved staging |
| **D-B8** | The shared all-reduce buffer was allocated at the wrong dtype | A dtype mismatch on a persistent buffer is a size error that only appears when the L1 is tight | Allocated at the consumer's dtype |
| **D-B9** | Confining the attention matmul (D-B5) made its circular buffers no longer fit — ~20 kB over, clashing with the decode activations resident on `x=1..3` | Opened by D-B5's own fix; visible only on hardware, and only after the placement was legal | `in0_block_w = gcd(k_tiles, 4)` (`recipes.py:414`). **Closed on silicon**, and both models' decode blocks now pass 3/3 |
| **D-B10** | An interleaved, non-DRAM relocation target fell through to `to_memory_config`, i.e. `ttnn::prim::copy` on the full grid | Same class as D-B7, on the fall-through branch nothing had exercised. Latent for prefill too | Explicit staging |
| **D-B11** | The LM head's `decode_program_configs` resolved to `(None,)`, so ttnn auto-selected the full seven-column grid | `None` is a legal value meaning "choose for me". Nothing validates that the choice lands inside the loaded sub-device | 24-core `gather_in0` ring |
| **D-B12** | `ring_cores()` and `ring_receiver_cores()` are the same 24 cores **in a different order**, and `gather_in0` with a DRAM in1 needs in0 and output on the same cores. `decode_weights_memcfgs` was dead config | Two sets that compare equal as sets and differ as sequences. A host test that checks membership passes | Order made explicit; the dead field removed |
| **D-B13** | `ttnn.linear` was never given a `sub_device_id`, and the `gather_in0` factory intersects its ring with sub-device **0** — the prefetch senders — producing an empty core set | Defaulting to sub-device 0 is silent, and an empty intersection is not an error until the program is enqueued | `sub_device_id` threaded through |
| **D-B14** | `ttnn.all_reduce` forwards to the buffer-less `all_reduce_async` overload, which falls back to a composite all-gather with **no** `sub_core_grids` — i.e. the full grid. `subdevice_id` is honoured by the fused path and ignored by the fallback | The same argument is respected on one code path and dropped on the other, with no diagnostic | Keyed persistent buffer, so the fused path is taken |
| **D-B15** | That persistent buffer was allocated resident in **L1** — 129 kB per core at bfloat16 | "Persistent" was read as "L1-resident". It means the resource owner holds it across calls | DRAM-resident, L1 view per call |
| ~~D-B16~~ | ~~A ring matmul output reports its padded width~~ | **Struck — not a defect.** Hardware says a matmul output keeps its **logical** width; only a reduce-scatter output takes the padded one | Reverted, with the distinction recorded in the code |
| **D-B17** | The reduce staging used the production's literal `num_cores_after_lm_head = 32`, and 32 does not divide Llama's 504 tiles | A copied constant that happens to be right for the model it was copied from | Core count derived from the tile width |
| **D-B18** | The decode logits, and so the reduction buffer, were bfloat16 — ~96 kB/core, clashing with the ring matmul's circular buffers | Borrowed `decode_activation_dtype` rather than declaring its own | `lm_head_output_dtype = bfloat8_b`, **the production value**, under its own field. The accumulation stays fp32 (`fp32_dest_acc=True`) |
| **D-B19** | `galaxy_padded_vocab_size` left 501 tiles per device in a 42-core × 12-tile spec, so the 42nd core's shard was never full and `all_reduce_async`'s `cb_in.wait_front(ring_size * block_num_tiles)` waited on every output core for tiles the fabric would never send | **No abort, no traceback — an indefinite hang and a mesh reset.** The arithmetic is exact everywhere except the last core | Vocabulary padded to a ring-exact width |
| **D-B20** | `seal()` allocated the global circular buffer at model build. 774 kB of unfreeable L1 per sender/receiver core made every prefill program needing static CBs there unplaceable, starting with `ttnn.embedding` | Prefill never *reads* the buffer, so nothing connected "the prefetcher is sealed" to "prefill cannot be placed" | `Prefetcher2DConfig.defer_global_cb`; the first `activate("decode")` allocates |
| **D-B21** | The prefill RoPE table copy inherited decode's **row-major** layout; `rotary_embedding_llama` requires TILE and `ttnn.embedding` requires row-major | One legal layout per consumer, and they differ. A single stored table cannot satisfy both, and nothing said so | The prefill copy tilizes |
| **D-B22** | The prefill transformation matrix was `head_dim × head_dim`; the op applies it one tile at a time and validates `[-1] == TILE_WIDTH` | **The module's own host test encoded the wrong shape**, so module and test agreed and both disagreed with the hardware | `TILE_SIZE × TILE_SIZE` |
| **D-B23** | The logits composed along the **wrong mesh axis**. A matmul output carries its *activation's* topology labels, not its weight's, so `to_torch_auto_compose` concatenated the four columns along the vocabulary axis; the runner then sliced `[:, :vocab_size]`, which narrows without raising | **Failed open, silently.** It would have produced a plausible top-1 in the sixties and sent the next session after norm epsilons | `compose_galaxy_logits`, composing by distribution |
| **D-B24** | The step-2 test's KV reference was in the wrong RoPE convention — device holds post-RoPE K in Meta interleaved order, HF in split order | **The two conventions cancel inside `Q·Kᵀ`**, so the logits agreed at 0.99958 while the caches scored 0.0386. A logits-only test would have called this qualified | Reference K permuted |
| **D-B25a** | `wqkv` and `wo` were registered with the prefetcher, but their *confined* matmuls (D-B9) cannot read the global CB, so two entries per layer went unconsumed and the MLP's `w1` read the entry meant for `wqkv` | **Failed open.** MLP PCC 0.096 with no error anywhere. Registering a weight that is never consumed is a **correctness** defect, not a performance one | Only the MLP's three projections are registered |
| **D-B25b** | The **non-fused** decode RoPE pair wrote a K of `|max| = inf` into the cache, while V — which skips RoPE — was exact | An infinity in K only; every V check passed. The non-fused pair is the Blackhole fallback and wants a different cos/sin layout | `use_qk_fused_rotary` defaults True, as production selects |
| **D-B26** | Qwen's per-head Q/K **decode** norm was unplaceable three ways: interleaved DRAM resolves `LayerNormDefaultProgramConfig` over the whole compute grid; height-sharded input is rejected outright; any single sharded placement breaks a property the rotary depends on | **This is the unresolved half of Milestone A's D2**, and it is Qwen-only — Llama has no per-head Q/K norm, so no `HEAD_LOCAL` norm had ever executed a decode step. **Prefill is correct on exactly the same config**, in the same run, because prefill's mode plan is one sub-device over the full grid. The host test asserted that the model's config *agreed with the module default* — and agreement is what carried the defect | `RMSNorm2DConfig.decode_compute_cores`; qualified at PCC ≥ 0.99998 on all 32 devices |
| **D-B27** | The decode LM head's all-reduce was left **no** worker cores and segmentation-faulted: `lm_head_reduce_core_count` returned the whole 50-core envelope for Qwen's 600 tiles | **Llama survived by luck**: its 504 tiles have no divisor between 43 and 50, so 42 leaves eight cores free. Nothing reserved them | `GALAXY_CCL_RESERVED_WORKER_CORES = 4`. Llama still resolves 42, bit-identically; Qwen resolves 40 |
| *(unnumbered)* | `galaxy_hardware.load_reference_tokens` returned a `(1, 1024)` tensor raw while every consumer treats the sequence as flat, so `len()` was **1** | **Failed open as a SKIP.** A caller asking for a 512-token prompt saw "reference sequence has 1 tokens" and skipped; the accuracy gate could not have run and would have reported green | Squeezed once, in the loader |
| *(unnumbered)* | `GalaxyColumnAllReduce` never passed `subdevice_id` to `ttnn.all_reduce` | Same silent-drop mechanism as D-B14 | Passed |

### What the shape of this list says

Eighteen of the twenty-seven are **placement, partition or core-count** faults: a tensor, a program grid or a
circular buffer resolved somewhere the loaded sub-device manager does not own. Every one of them is
invisible to a host test, because a host test never loads a partition. That is the single most
transferable fact Milestone B produced, and it is why the plan's per-module contracts could not catch
them: **the modules are individually correct and the composition is where the topology lives.**

Four more — D-B23, D-B24, D-B25a and the reference loader — were in the **measurement apparatus**, and
all four failed open. The rule the Llama job drew from that is worth carrying into Milestone C:
*make the thing that produces a number prove itself before you believe the number.*

## Defects found by step-7 coverage

These are the ones that are **still open**. None was fixed, and that was deliberate: every one either
needs a product decision, or needs a mesh to validate a fix on, or sits in `direct_runner.py` — which
`test_full_model_wh_galaxy.py` and both `demo.py` files import, so editing it would invalidate the
byte-identity that every accuracy and demo row above rests on. Full accounts:
`tttv2_milestone_b_evidence/coverage/REPORT.md`, findings sections of §A2, §A3 and §A4.

| | Defect | How it hid | What it needs |
| --- | --- | --- | --- |
| <a id="d-c1"></a>**D-C1** | A prefill-shaped page table handed to **decode** is accepted, not rejected. `Attention2D._validate_decode_page_table` discriminates on row count alone and takes any positive multiple of `users_per_column` | **Shape cannot separate the two cases.** The modulo is deliberate — an L1-sharded decode table legitimately repeats the device-local batch once per core — but the replicated prefill table's device-local view is 32 rows and `32 == 4 × 8`, so it passes; the width check passes too, because the prefill table is stick-aligned and therefore *wider*, never narrower. Both tables are DRAM-interleaved, so `is_sharded()` cannot separate them either. Confirmed on silicon: `test_step7_page_table_placement_wh_galaxy.py`, 3 fresh | A **contract decision**. The discriminator that would work is `memory_config()`, which the validator never consults. See the correction below — this is cheaper to fix than the coverage report thought |
| <a id="d-c2"></a>**D-C2** | "Moving a request to a different slot does not change its stream" is **false**. `_seed_digest(seed, slot) = blake2b("sampling2d:{seed}:{slot}")` mixes the slot into the key | Not hidden — measured, and deliberately not "fixed". The slot mixing is what stops 32 slots given one seed by a serving front end from all emitting the same token, which is *also* proved. The step-7 requirement and the module's design are in direct conflict | A **product decision**: is a seed per-request or per-(request, slot)? Put it in front of whoever owns the serving contract before Milestone C builds on it |
| <a id="d-c3"></a>**D-C3** | `LazyWeight._get_fingerprint` ends with `device_{MeshDevice.id()}`, and the `mesh_device` fixture builds a new mesh per test — so the cache path changes per test and **every test after the first in a process misses on every weight** | Costs time and disk, never correctness, so nothing failed. Measured: test 1 got 240 hits and built in ~6 min; test 2 took **965 misses**, 26 min of staging and **138 GB** written | A one-line change in shared 1D/2D code (`models/common/modules/lazy_weight.py`): the fingerprint wants the mesh **shape**, not the instance id. **Operationally, until then: one node id per pytest process, always.** An 8-node-id file needs 1.1 TB of cache for Llama |
| <a id="d-c4"></a>**D-C4** | `from_pretrained(paged_attention_config=None)` installs the default 2048-block pool rather than a contiguous cache, so the brief's headline area-1 gate — "paged fill then decode, PCC ≥ 0.99 **against the contiguous path**" — has no reachable form at this adaptor API | It also made a committed test a tautology: it asserted `paged_attention_config is None` after construction, which the adaptor never leaves true | A **contract gap**. Either the adaptor grows a contiguous option or the gate is restated. The reachable substitute (two different paged pools agreeing) is what was measured instead |
| <a id="d-c5"></a>**D-C5** | `GalaxyColumnUserSelector.__call__` is one `ttnn.matmul`, whose default program config requires input B **INTERLEAVED**. Both models' decode logits arrive **WIDTH_SHARDED** from the shared recipe `lm_head_output_memcfg` | **Every module in the chain is green in its own suite; the chain does not run.** The selector's only guard is a shape check — memory layout is unvalidated, so it surfaces as a `TT_FATAL` from inside ttnn rather than a contract error naming the caller. And the selector's own device test builds its input `DRAM_MEMORY_CONFIG`: the one layout the matmul accepts and the one layout the real model never produces | A shared-code change: either the selector accepts a sharded input B, or `sample_decode` declares the layout it needs and each model relocates. **Precedent is 200 lines above it** — `_relocate_sharded` already stages through `ttnn.sharded_to_interleaved(..., DRAM)` for exactly this constraint. **Deterministic on both models, 3 fresh processes each** |
| <a id="d-c6"></a>**D-C6** | Concat-32 physical prefill does not fit in L1 **at any supported length, on either model**: 1 669 312 B of static circular buffers at length 128 against 1 499 136 B available, doubling per length doubling | §A2 recorded this as Qwen-only, because Llama's concat-32 demo died earlier on the [L-B1](#l-b1) address clash and never reached the question. The step-7 form — build once, prefill once, no preceding decode — cannot raise the clash, and it fires for Llama too **at byte-identical figures**. Two different geometries cannot coincidentally need the same 1 669 312 B: it is the **shared recipe**, not either model's dimensions | A **recipe fix**, once, not per model. The smallest supported length is already 11% over; 1024 asks for 7.8×. **Nothing about padded-row isolation was measured in either direction** — active batches 16/31/32 all die before a row's logits can be inspected |
| <a id="d-c7"></a>**D-C7** | Closing a model does not return its L1. After `close()` **and** an explicit `gc.collect()`, the second model's `activate("decode")` finds **923 776 of 1 393 472 bytes per L1 bank — 66% — still allocated**, with a largest free block of 373 824 B against the 792 064 B it needs | `Prefetcher2D.cleanup()` already does everything Python can do; the owner truthfully reports `owned_resources == ()`. Milestone A's [L1](#l-b1) is written as a prefill-after-decode **ordering** problem; this is a **lifetime** problem, and no teardown ordering can fix a buffer the destructor of a closed object did not free. **Not observable on Llama** — its address clash arrives first — so it is qualified on Qwen alone | The `Prefetcher2D` global-CB ownership redesign, routed to Milestone C. **Rests on one observation**; its run-2 and run-3 were queued and the mesh died first |
| <a id="d-c8"></a>**D-C8** | Behind D-C5: with the logits relocated to INTERLEAVED at the test boundary, the same matmul builds its program over the **whole compute grid** while the loaded decode sub-device manager owns only `prefetch_sender_cores() \| worker_cores()`. `TT_FATAL @ program.cpp:2205` | Only reachable once D-C5 is removed — one defect stacked behind another, which is why the diagnostic was worth the mesh time. **Deterministic at 3 fresh processes on both models**, so neither is geometry-dependent | A **design decision**, not a line: does the sampling path run inside the decode worker sub-device, or does decode's partition widen? `recipes.rope_core_grids` already documents this defect class and names `_subgrid_cores` as the qualified helper |
| <a id="d-c9"></a>**D-C9** | `GalaxyDirectRunner.decode_sampled` (`direct_runner.py:535`) composes the sampled tokens with `to_torch_auto_compose(...).reshape(-1)[:32]`. That follows the tensor's **topology labels** rather than its distribution, so it concatenates the replicas of one mesh column and returns **eight users four times over** | **The same trap is documented in the same repo, one op earlier in the same graph.** `compose_galaxy_logits` exists precisely because a matmul output inherits its *activation's* topology labels; `_compose_rows` was already fixed for it, sixty lines above. `decode_sampled` was not. The 32 greedy tokens were `[265, 2631, 1916, 220, 17, 15, 17, 17]` repeated four times, byte-identically in two processes | A one-line fix with precedent in the same file: compose by distribution, `ttnn.ConcatMesh2dToTensor(dims=(0, <user axis>))` then mesh row 0. **Fix it before anyone reads another device-sampling number.** The exit gate is untouched: `decode_logits` goes through the fixed `_compose_rows`, and the batch-1, batch-32 and concat-32 demo tests all take the default `GalaxySamplingPolicy(top_k=1, temperature=0.0)`, whose `on_device` is `False`, so they sample on the host. Re-verified here at `direct_runner.py:48-55,507-514,636-639` and `{llama33_70b_galaxy,qwen3_32b_galaxy}/demo.py:141,168,192`. The one demo test that *does* set `on_device=True` (`demo.py:224-230`) is not a gate line and is blocked at [D-C5](#d-c5). *`mb-llama`'s attempt-3 narrative describes its batch-1 demo run as "with device sampling enabled"; the source says otherwise, and the source is what ran.* |
| <a id="d-s1"></a>**D-S1** | *New on this page.* A device test that **passes** and then will not release the mesh. Every recorded run of `test_qwen3_32b_galaxy_qk_norm_head_local_..._decode_and_prefill` — six, at five distinct commits, four of them passing — writes its verdict and then holds all `/dev/tenstorrent` descriptors past a 90 s grace, is `SIGTERM`ed at the deadline, and exits 124 | **No `REPORT.md` in this tree mentions it.** The harness comment explains hung teardowns after a *failure* (an aborted multi-sub-device program leaves the mesh un-drainable), so a hang after a pass reads as the same known thing and was not separated out. It is not: at the same commit, on the same night, `a2_73/74/75_block` exit 0 three times. **The three runs that re-qualified this claim at the final tree are three of the four passing hangs** | Diagnosis, then a fix in the Qwen decode `HEAD_LOCAL` norm teardown path — most likely the same resource-lifetime family as [D-C7](#d-c7) and [L-B1](#l-b1). Evidence: `signoff/logs2/s2_17_teardown_hang.log`, and `qwen/logs2/a2_{05,08,09,14,70,71,72}_qknorm.log` |

### Correction to D-C1, established here

The coverage report declines to fix D-C1 on the grounds that making decode reject a 32-row table
"requires changing an existing expectation, and the brief is explicit that changing an existing
expectation to accommodate this work is a boundary violation". **That premise is wrong, and it makes
D-C1 cheaper than recorded.**

Both the validator and the expectation are **Milestone B's own**. At the Milestone A tip
`bc6ad03bfc2` there is no `_validate_decode_page_table` and no
`test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats`; there is a single
`_validate_page_table` that required `shape[0] > max(users)` — i.e. **at least 32 rows**, the prefill
layout — for decode as well. Milestone B split the validator and corrected the decode side to the
device-local batch, which is right, and in doing so admitted `32 == 4 × 8`.

Two consequences:

- **it is a Milestone B contract decision, not a Milestone A expectation to negotiate.** Nobody
  outside this milestone depends on 32-row acceptance;
- **the module's own docstring already claims the guarantee the code does not provide.**
  `attention_2d.py:678-679` says *"A table sized to the full physical batch is the prefill layout and
  is rejected here rather than at the first op."* It is not. Whoever fixes D-C1 should fix that
  sentence in the same change, and until then it is a false contract statement in the source.

Verified at `signoff/logs2/s2_11_changed_impl.log` and by
`git show bc6ad03bfc2:models/common/modules/attention/attention_2d.py`.

### Recorded limitations that are not defects

| | | |
| --- | --- | --- |
| <a id="g-c1"></a>**G-C1** | Active batches 16 and 31 are not expressible as a smaller allocation | `prefill_batched` refuses any runner with `active_slots != 32` and `_recipe_identity` resolves only `SINGLE_ROW` or `CONCAT_32`. "Active batch 16" means 32 physical rows of which 16 carry prompts. Both facts are pinned by tests |
| <a id="g-c2"></a>**G-C2** | An empty row is caught one call too late | `generate` refuses an empty prompt; `prefill_batched` called directly plans `token_indices[r] == -1` and leaves the rejection to `project_prefill_logits`. The rejection *does* happen, so no padded logit is ever returned — but only after the whole concatenated prefill graph has run |
| <a id="g-c3"></a>**G-C3** | The `chunk_page_table` guard is unreachable | `_recipe_identity` treats a non-`None` `chunk_page_table` as one of the signals selecting `PREFIX_CHUNKED`, so by the time `_validate_prefill` checks for it the recipe is already chunked. Dead code plus a missing check |
| <a id="f-c1"></a>**F-C1** | Both models pad their vocabulary — the reverse of what was first recorded | Llama pads by 768 ids and Qwen by 1664 under the ring-exact rule ([D-B19](#defects-found-on-the-bring-up-path)). The first coverage attempt computed the pre-D-B19 alignment and concluded Llama's padded-vocab gate was vacuous. It is live for both |
| <a id="f-c2"></a>**F-C2** | `models/common/tests/models/galaxy/test_plans.py` is **not** a host-only suite | It has no `mesh_device` fixture, a `MagicMock` mesh and no `_wh_galaxy` in its name — but `ttnn.SubDevice` implicitly constructs the `MetalContext`, which opens a cluster. Its 14 tests cannot run without a mesh, and on a healthy mesh they should pass; if they do not, *that* is a finding |
| <a id="f-c3"></a>**F-C3** | The model-named import gate is not literally zero over `models/common` — but every exception is pre-existing | Measured here, not quoted: **33** `models.demos` import lines in **16** files under `models/common/tests`; 32 are `models.demos.utils.{llm_demo_utils,model_targets,trace_region_sizes}` in 1D demo and demo-contract files and one is a DeepSeek test-helper import in `tests/modules/moe/`. **None of the 16 files appears in `git diff --name-only bc6ad03bfc2..HEAD`.** Milestone B's own seven directories are 0/0. Word the gate line as *"no Milestone B code imports a model-named implementation package"*, which is measured and true, and footnote the class. (`signoff/logs2/s2_02_imports.log`. The coverage report counted 24 over a narrower sweep; the difference is scope, not disagreement — both find zero attributable to Milestone B) |

## Milestone A limitations, as Milestone B leaves them

### <a id="l3-closed"></a>L3 — attention decode on the prefetch subdevice partition: **CLOSED**, with a named cost

Milestone A recorded the decode QKV `ttnn.linear` as terminal because its `(7,1)` grid straddled the
sender/worker sub-device split, and said it "cannot be narrowed". Both halves of that are now settled
on silicon.

- **It can be narrowed.** `ttnn` has since grown `allowed_worker_cores`, deprecating
  `compute_with_storage_grid_size`. Milestone B confines both attention decode matmuls to the largest
  worker rectangle anchored at the worker envelope's origin — `dense_matmul_worker_rectangle`, three
  columns wide on `(8, 4)` — and with `in0_block_w = gcd(k_tiles, 4)` their circular buffers fit
  ([D-B9](#defects-found-on-the-bring-up-path)).
- **It executes and it is numerically right**, on both models. At this tree, three fresh processes
  each: Llama decode logits **0.99975**, KV K **0.99993** / V **0.99975**; Qwen decode logits
  **0.99936**, KV **0.99989**. The sub-module bisection that attributes 0.99975 to the *attention
  output* specifically was a single run at a superseded tree (`llama/logs3/a3_31`), and is cited as
  attribution rather than as a qualified figure.

**Two costs, recorded rather than absorbed:**

1. **three worker columns instead of seven**;
2. **the attention weights lose their prefetching.** The global circular buffer is received by the 24
   ring cores, so a matmul confined to the worker rectangle cannot take its weight from it.
   Registering those weights anyway is a *correctness* defect, not a performance one — the unconsumed
   entries shift every later consumer, which is how the MLP came to score PCC 0.096
   ([D-B25a](#defects-found-on-the-bring-up-path)).

Moving the two matmuls to the 24-core `gather_in0` ring recovers both at once, and is now a smaller
job than it was: the ring wiring exists and is exercised by three matmuls, and
`attention_qkv_collective_input_memcfg` is already shaped for those 24 cores. **That is a Milestone C
performance item, not an open correctness gap.**

*Re-verified for this page at `models/common/models/galaxy/recipes.py:850,853` (both attention decode
matmuls are `dense_matmul_program_config`) and `:373-425` (that config confines with
`allowed_worker_cores` and sets `in0_block_w = math.gcd(k_tiles, 4)`); log
`signoff/logs2/s2_13_l3_recipes.log`.* The previous signoff pass recorded "L3 is therefore STILL
OPEN" and "D-B9, still open"; both were written from `mb-llama` attempt 1 and were superseded by
attempts 2 and 3 the same week.

### <a id="l-b1"></a>L-B1 — L1 is not returned. This is worse than Milestone A's L1, not better

Milestone A's L1 says `Prefetcher2D.cleanup()` cannot free the global circular buffer, and prescribes
an **ordering** rule: tear consumers down before, or together with, the owner. Milestone B measured
three things that say the problem is **lifetime**, not ordering.

1. **Dropping the last Python reference does not return the L1.** `mb-llama` implemented the obvious
   fix — release the global CB on `activate("prefill")`, recreate it on the next
   `activate("decode")` — behind `Prefetcher2DConfig.release_global_cb_on_prefill`, default off, with
   an opt-in env var so it could be measured without disturbing the qualified path. The release ran
   (its print is unconditional) and the clashing L1 base address was **identical** to the run without
   it: 544832 in both (`llama/logs3/a3_64` against `a3_46`). There is no `deallocate` on the type and
   the allocation does not go back when the last reference dies. **The flag and its tests are left in
   the tree, default off, with this result recorded against them, so nobody spends the run again.**
2. **Nor does closing the whole model.** [D-C7](#d-c7): after `close()` and `gc.collect()`, 66% of
   every L1 bank is still held.
3. **The remaining symptom is prefill-after-a-decode**, and it is deterministic. Llama's
   `..._repeated_requests_and_deterministic_cleanup` fails **3/3 in three fresh processes** with a
   byte-identical `TT_THROW … Statically allocated circular buffers in program 100 clash with L1
   buffers on core range [0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region
   ends at 630080`. Production has the same property.

**The operating rules that follow, and they are rules, not advice:**

- **one Galaxy model per process**;
- **prefill everything before you decode anything.** Batch 32 itself is fine — all 32 slots prefill
  before any decodes, which is what the demos do. A second runner that prefills after the first has
  decoded is not.

**Qwen is not a clean reference.** It survives the two shapes where Llama's address clash bites, and
it fails the third — repeated model construction — with a capacity signature instead of an address
one. Two different defects produce one symptom; a fix for either will not silence the other.

**The untried hypothesis, and it is the first thing to try.** The clash is that a *full-grid* prefill
program cannot place its circular buffers on the sender columns. The prefill mode plan is currently
one sub-device covering the whole grid (`galaxy_prefill_mode_plan_cores`). Making it the **worker**
cores instead confines every prefill program to the 50 cores the global CB does not occupy, and the
problem disappears without touching the buffer's lifetime at all. The cost is 20 of 70 cores for
prefill, and the reason `mb-llama` did not do it is honest and worth repeating: it changes the grid of
every prefill program, so the prefill 128 and 2048 numbers, the 80-layer prefill and the accuracy gate
would all need re-taking. `test_llama33_70b_galaxy_batch32_slots_are_isolated` is the oracle.

**Routed to Milestone C**, as the `Prefetcher2D` global-CB ownership redesign (Milestone A's D-C).

### <a id="l-b2"></a>L-B2 — Qwen's 128K smoke runs three times past its trained context, and nothing refuses it

`Qwen3-32B`'s `max_position_embeddings` is **40960**. The 128K functional smoke runs anyway:
`max_context_len` rides on the runtime config and is never checked against `max_seq_len`. The smoke is
functional by definition — it asserts only that the emitted token is in vocabulary — so this is not a
wrong result, but it is a missing guard and the 128K Qwen row should not be read as a quality
statement. **Target: Milestone C**, with the rest of the capacity-validation surface.

### <a id="l-b3"></a>L-B3 — the device weight cache forces one node id per pytest process

[D-C3](#d-c3). Not a correctness limit; a scheduling one, and it costs a night if it is not known.
Every Milestone B harness happens to run one node id per process, and nothing in the tree said why
until now. **Target: any time** — it is one line in `lazy_weight.py`.

### <a id="l-b4"></a>L-B4 — what the accuracy gates do and do not cover

Both teacher-forced numbers are batch 1, paged KV, greedy argmax over host-composed logits, with the
reference's own tokens fed back at every step. **Teacher forcing bounds error accumulation by
construction**: a defect that drifts over a long free-running generation cannot show up in this
measurement, and nothing in Milestone B measures free-running quality beyond 16 demo tokens.
**Target: Milestone C**, where the paired TTTv1/TTTv2 comparison gives a natural place for it.

## Pending work

### Blocking — must close before Milestone B can be signed off

<a id="p1"></a>**P1 — Exit-gate line 9: five 1D demo-contract/hf-adaptor tests are red.**
`5 failed, 296 passed in 83.02s` at this commit (`signoff/logs2/s2_03_1d_contract_gate.log`), the
same five node ids at a **fifth** distinct commit. **They are not Milestone B's**, and this page
proves that mechanically rather than asserting it:

- `git diff --name-only bc6ad03bfc2..HEAD -- models/common/tests/models/ | grep -v galaxy` is
  **empty** — no 1D expectation was edited;
- all five owning test packages and all five owning model packages are **0 changed paths** since the
  Milestone A tip; so is `models/demos/utils`, which three of them import;
- **everything** Milestone B changed outside `models/common/{models,modules,tests}` and the evidence
  directories is: nothing.

**And Milestone A's own integrated gate never collected them.** Verified here out of Milestone A's own
log (`tttv2_milestone_a_final_evidence/logs/host01_integrated_gate.log`): its 1263 collected items
come from `tests/llm_runtime/`, `tests/modules/*` and `tests/models/galaxy/` only, and none of the
five files appears in it. **Milestone B is the first milestone to measure this exit-gate line, and it
was red the first time anyone looked.** Whoever owns those five tests has to fix them or restate the
expectation; the line cannot be closed from inside this milestone.

<a id="p2"></a>**P2 — Concat-32 has no reachable case, on either model, at any supported length.**
[D-C6](#d-c6). This is one of the plan's named *Milestone B tests* ("physical-32 prefill at sequence
length 128, then through 2048") and the whole of the brief's area 2. Not a near miss: length 128 is
11% over L1 and 1024 asks for 7.8×. **The fix is in the shared concat-32 recipe**, once, and the
byte-identity between the two models is the evidence for that. Until it lands, nothing about
padded-row isolation at active batch 16/31/32 has been measured **in either direction**.

<a id="p3"></a>**P3 — Device sampling is blocked two defects deep, and the third makes the readings
unreadable.** [D-C5](#d-c5) → [D-C8](#d-c8) → [D-C9](#d-c9). Also a named *Milestone B test*
("greedy and stochastic device sampling"). The selector itself is qualified 3/3 and is **not** the
problem. **Fix D-C9 first** — it is a one-liner with precedent in the same file, and until it lands
every sampled-token number produced on this mesh is a readback artifact rather than a sampler
measurement.

<a id="p4"></a>**P4 — Repeated requests fail deterministically on Llama, and one bullet of
repeat-and-cleanup has never run.** [L-B1](#l-b1). `..._repeated_requests_and_deterministic_cleanup`
is **FAIL 3/3** byte-identical. `test_bringup_wh_galaxy.py::test_two_models_in_one_process` (Llama) has **zero device
runs across four attempts** — the brief's second repeat-and-cleanup bullet is unmeasured on Llama, and
on Qwen the equivalent shape is a single observation ([D-C7](#d-c7)).

<a id="p5"></a>**P5 — Four qualification tails.** The three-fresh-processes rule is unmet for the
**Llama teacher-forced accuracy gate** (2 at this tree), the **Llama batch-32 demo** (2), the **six
long-context smokes** (1 each) and the **prefix-cache gate** (1 each). Each is one or two runs, not a
night. They were queued as `a4_l_tf`, `a4_q_tf` and the `_run2/_run3` tails of `queue4.txt`, and the
mesh died before they were dequeued.

<a id="p6"></a>**P6 — The mesh needs an operator.** Device 21 reads `0xffffffff`;
`/dev/tenstorrent/{1,3,5,7}` raise `ENXIO`; the kernel logs `Device is unresponsive, cannot reset` and
`Skipping message 00000011 due to FW not running`, with a stack trace through
`tt_hwmon_read+0x45/0xa0 [tenstorrent]`. Five `tt-smi` recovery paths were tried and logged
(`coverage/logs4/recovery{1..5}*.log`). **`ls /dev/tenstorrent | wc -l` is still 32 and means
nothing** — the nodes exist and the chips do not answer. Check by *opening* each node. Recovery not
tried, in ascending severity: `rmmod tenstorrent && modprobe tenstorrent` then `tt-smi -glx_reset`;
an IPMI power cycle of the trays; a host reboot.

**A suspect for what broke it, stated because it is actionable and not because it is proven.** All
7234 `tenstorrent: pin_user_pages_longterm failed: -14` kernel messages in a 29-day boot fall inside
**one minute — 18:34Z** — which is inside `a4_q_dc8_run2`, a run of a test written 40 minutes earlier
whose case performs **six full sub-device-manager cycles in one process**. A kernel stack trace
follows 23 s later and only then does the reset fail. Against it: the *first* run of the same case
survived, and pinned pages are freed at process exit. **Not established either way.** The resumed
queue leads with a **one-cycle** variant for exactly this reason, and `cov_queue4.sh` now counts those
kernel messages before and after every run and halts if either grows.

### Deferrable — with the milestone each belongs to

| | Item | Target |
| --- | --- | --- |
| **D-B/L3 residual** | Move both attention decode matmuls to the 24-core `gather_in0` ring. Recovers four worker columns **and** the attention weights' prefetching at once ([L3](#l3-closed)) | **Milestone C** — performance |
| **D-C (A's)** | `Prefetcher2D` global-CB ownership redesign. Now bigger than Milestone A scoped it: [D-C7](#d-c7) says the residue outlives the owner entirely | **Milestone C** |
| **D-A (A's)** | Physical-32 real-device trace. Needs a model-owned executor with `TraceCompiler`/`TracedExecutor`; **this is the first milestone where anything exists to trace** | **Milestone C** |
| **D-C1** | Decide whether a decode page table is discriminated by shape or by placement, and fix the docstring that already promises the rejection ([correction](#correction-to-d-c1-established-here)) | **Milestone C**, before serving is built on it |
| **D-C2** | Decide whether a sampling seed is per-request or per-(request, slot) | **Milestone C** — a product decision, not a bug |
| **D-C3 / L-B3** | Key `LazyWeight`'s cache fingerprint on the mesh **shape**, not the `MeshDevice` instance id | **Any time** — one line, in shared 1D/2D code |
| **D-C4** | Give the adaptor a contiguous-KV option, or restate area 1's headline gate | **Milestone C** |
| **D-S1** | Diagnose the Qwen Q/K-norm teardown hang ([D-S1](#d-s1)) | **Milestone C**, with the rest of the L1 lifetime work |
| **G-C2** | Reject an empty prefill row in the runner rather than after the concatenated graph has run | **Any time** |
| **G-C3** | The `chunk_page_table` guard is unreachable; add the check that is missing | **Any time** |
| **F-C2** | `test_plans.py` opens a cluster despite looking host-only. Either mark it or make it mock the `SubDevice` construction | **Any time** |
| **L-B2** | Check `max_context_len` against the model's `max_position_embeddings` | **Milestone C** |
| **CCL merge** | Evaluate merging Galaxy CCL with `models/common/modules/tt_ccl.py`. The plan defers this until both models pass; both models now *run*, and the D3 `semaphore_cores` invariant is one input to the evaluation | **Milestone C**, per the plan's follow-up TODO 2 |
| **Doc debt** | Six committed source files still open with **"This file has never been executed."** Every one has now run on silicon, repeatedly. Not corrected here **on purpose** — see [why](#why-this-page-did-not-fix-the-stale-docstrings) | **Any time**, and before Milestone C reads them |

## Exit-gate result

Against `tttv2_2d_modules_plan.md` → "Milestone B exit gate". The plan lists eight bullets; the
"zero changes to 1D module implementation files" bullet is split here into the two mechanical checks
every job in this set ran, giving nine lines — the numbering the coverage evidence uses.

**Every row below has a raw log, and this job read the number out of that log rather than out of a
report.** The extraction is `signoff/logs2/s2_06_gate_log_trace.log`; the fresh-process accounting is
`s2_09_qualification.log`.

| # | Exit-gate requirement | Verdict | Measured | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Llama teacher-forced, batch 1, prefill 512 / decode 511: top-1 ≥ 91%, top-5 ≥ 99% | **PASS** — *2 fresh processes at this tree, not the 3 this project requires* | top-1 **501/511 = 98.04%**, top-5 **511/511 = 100.00%** | `qwen/logs2/a2_45_llama_accuracy.log` (1293.62 s), `coverage/logs2/a2_g1_llama_tf.log` (1029.52 s); 3 further runs at superseded trees, identical counts |
| 2 | Qwen teacher-forced, batch 1, sequence 512: top-1 ≥ 89%, top-5 ≥ 97% | **PASS — qualified**, 3 fresh at this tree | top-1 **498/511 = 97.46%**, top-5 **511/511 = 100.00%** | `qwen/logs2/a2_33_accuracy.log`, `a2_34_accuracy.log`, `coverage/logs2/a2_g12_qwen_tf.log` |
| 3 | Batch-32 direct demos produce valid output with no cross-slot contamination | **PASS** both models — Qwen qualified (4 fresh), **Llama 2 fresh** | 32 slots, 8 distinct prompts × 4; duplicates byte-identical, distinct prompts diverge; slot 0 identical served alone or alongside 31 others | Llama `qwen/logs2/a2_47`, `coverage/logs2/a2_g9`; Qwen `qwen/logs2/a2_22,31,32`, `coverage/logs2/a2_g21` |
| 4 | Batch-1 4K / 32K / 128K functional smokes pass | **PASS**, all six — **single run each, not qualified** | Llama 357.81 / 641.17 / 721.70 s; Qwen 117.91 / 136.29 / 245.76 s | `coverage/logs2/a2_g{3,4,5}`, `a2_g{14,15,16}` |
| 5 | Prefix-cached output matches uncached execution under the model's numerical acceptance | **PASS** both — **single run each, not qualified** | two 128-token chunks against one 256-token prefill: same argmax, PCC ≥ 0.99. Llama 424.35 s, Qwen 158.58 s | `coverage/logs2/a2_g2_llama_prefix.log`, `a2_g13_qwen_prefix.log` |
| 6 | No dependency imports come from an existing model-named implementation package | **PASS for Milestone B.** Pre-existing exceptions exist elsewhere and are named | Milestone B's seven directories: `models.demos` = **0**, non-galaxy model package = **0**, each. Wider sweep over `models/common/tests`: **33** pre-existing import lines in 16 files, **none** in a file Milestone B changed ([F-C3](#f-c3)) | `signoff/logs2/s2_02_imports.log`. The brief's literal `git grep` returns 2 hits, both **comments** citing a production file path in `collectives.py:361` and `recipes.py:801` — not imports |
| 7 | The Milestone B diff still contains zero changes to 1D module implementation files | **PASS** | **0** of 432 changed paths match `_1d\.py`; `git diff -- 'models/common/modules/**/*_1d.py'` is empty | `signoff/logs2/s2_01_boundaries.log` |
| 8 | Zero changes to `models/common/llm_runtime` | **PASS** | **0** of 432 | same log |
| 9 | Existing 1D model contract and demo-contract host tests remain green **without expectation changes** | **FAIL** — 5 of 301. Expectations **unchanged** (0 files touched); the failures are not Milestone B's | **5 failed, 296 passed in 83.02 s** at `e912a8267bb` | `signoff/logs2/s2_03_1d_contract_gate.log`; attribution in `s2_10_expectations_unchanged.log`, `s2_14_milestone_a_gate_collection.log` |

**Eight of nine pass. The ninth fails, so the gate is not met.**

### And the gate table is not the whole test

The plan's *Milestone B tests* section names **ten** items, for both models. Measured against that list:

| Milestone B test | State |
| --- | --- |
| host-only adaptor and config tests | **Met**, both models |
| one-layer decode and prefill PCC | **Met and qualified**, both models, 3 fresh each |
| full-model prefill plus first decode token | **Met**; Llama 1 fresh at this tree, Qwen 2 |
| teacher-forced decode | **Met**; gate lines 1–2 |
| batch 1 and batch 32 | **Met**; gate line 3 |
| paged KV | **Partial** — headline form not expressible ([D-C4](#d-c4)); substitute passes cross-process only; Llama block-level cross-slot **BLOCKED** |
| prefix-cached / chunked prefill | **Met for prefix-cached**; chunked qualified on **Qwen only** — Llama blocked by [L-B1](#l-b1) |
| **physical-32 prefill at 128, then through 2048** | **FAILED — no reachable case at any length on either model** ([D-C6](#d-c6)) |
| **greedy and stochastic device sampling** | **BLOCKED — no case reaches the sampler in the production shape** ([D-C5](#d-c5), [D-C8](#d-c8)); what was measured behind them is a readback ([D-C9](#d-c9)) |
| **repeated requests and deterministic cleanup** | **FAILED on Llama, 3/3**; the second bullet has zero runs ([L-B1](#l-b1), [P4](#p4)) |

So the honest statement is not "one line short". **Three of the ten test items are red or
unreachable, and one more is partial.** Line 9 would have failed the gate on its own; these would
too, if the gate bullets named them, and Milestone C should treat them as the real work rather than
as footnotes.

One item *outside* that list is worth recording on the credit side: the plan's "representative Llama
and Qwen geometry" requirement is now met on silicon for Qwen's **real** 64-head decoupled geometry,
which closes the Milestone A gap where its `Attention2D` row had been recorded against a 40-head
fixture no product has.

### <a id="what-would-have-slipped-through"></a>What would have slipped through each passing line

The brief's third scepticism test: *for each passing line, what would a defect have to look like to
slip through it?* This is how D4 and D5 survived Milestone A — greedy-only sampling could not reach a
temperature bug, and uniform memory configs could not reach a swapped pair.

- **Lines 1–2, teacher-forced accuracy.** Batch 1, greedy argmax over host-composed logits, with the
  reference's own tokens fed back every step. **Teacher forcing bounds error accumulation by
  construction**, so a drift that only appears in free-running generation cannot show up here. Neither
  gate touches device sampling, concat-32, batch 32, or any slot but slot 0. And they are measured
  through `_compose_rows`, the composition that was **already fixed** for the D-B23/D-C9 trap — a
  regression *into* that trap would produce ~60% top-1, which reads as a precision problem.
- **Line 3, batch-32 demos.** 32 slots but only **8 distinct prompts**, each repeated four times. A
  leak from slot `k` into slot `k mod 8` is invisible **by construction**, because the test asserts
  those slots agree. The divergence check is `len(unique) > 1` over the eight — two differing prompts
  satisfy it. And the demo prefills all 32 slots before any decode, so it cannot reach the
  prefill-after-decode failure ([L-B1](#l-b1)) that the repeat test does reach.
- **Line 4, long-context smokes.** The assertion is literally `0 <= token < vocab_size`
  (`test_full_model_wh_galaxy.py:295,302`). **Any defect that produces garbage in range passes** —
  wrong KV blocks at long positions, a mis-sliced chunk table, a silently truncated context. It proves
  the chunked prefill completes at 128K and nothing else. Qwen's 128K case additionally runs past its
  trained context ([L-B2](#l-b2)).
- **Line 5, prefix cache.** Two chunks against one prefill, at 256 tokens. A defect that needs three
  or more chunks, a non-chunk-aligned prefix, or a longer context is out of reach. The chunk-aligned
  SDPA path — the one that reads the paged cache — is qualified on **Qwen only**.
- **Lines 6–8, boundaries.** These are `git diff` and `grep`; they cannot be wrong about the past, but
  they say nothing about *quality* of what was added. A boundary can hold while the design inside it
  is poor, which is exactly why the plan asks for the scorecard as well.
- **Line 9.** It failed, so there is nothing to slip through — but note the inverse: it had never been
  *collected* before, which is how a red line survives two milestones.

## Modularity scorecard

Measured at `bc6ad03bfc2..HEAD` — Milestone A final to this commit — by
`signoff/logs2/s2_15_scorecard.log` and `s2_11_changed_impl.log`. Every number is `git diff --numstat`
output.

The plan is explicit that this is project evidence in its own right: *"Passing model tests while
violating these boundaries does not count as a successful TTTv2 extension."* The converse also has to
be said, and this page says it in both directions: **the boundaries held, and the model tests did not
all pass. Those are two separate results.**

| Required item | Evidence | Assessment |
| --- | --- | --- |
| **New 2D/model files added** | **17 new implementation files, +8892 lines.** 7 shared-Galaxy (`collectives.py`, `direct_demo.py`, `direct_runner.py`, `kv_contract.py`, `plans.py`, `prefetch.py`, `recipes.py`), 5 per model package (`__init__`, `demo`, `hf_adaptor`, `model`, `weight_utils`). Plus **28 new test files, +13093 lines, 306 test functions** | Within Milestone B boundaries. **More test code than implementation code** |
| **Existing shared files changed, and why config alone was insufficient** | **8 files, +754 / −48** — 7.8% of the implementation insertions. Six 2D modules and two shared-Galaxy files. Reasons below, one per file | Every one is a **contract correction required to issue a valid TTNN call**, or a new frozen config value with the previous default preserved. None is a behavioural change to an existing caller |
| **1D module implementation files changed** | **Zero.** `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` is empty over all 432 paths; so is the explicit pathspec form | **Required value met** |
| **Default runtime behaviours changed** | **Zero.** `models/common/llm_runtime/**` is **byte-identical** — 0 changed paths, not merely behaviour-preserving. `models/common/tests/llm_runtime` reports **1032 passed, 1 skipped, 0 failed** with 0 device opens | **Required value met**, and by the strongest available reading |
| **1D regression suites run, and their result** | 1D contract/demo-contract host gate: **5 failed, 296 passed** — the [P1](#p1) failures, proved unattributable. The brief's literal command, unfiltered: **18 failed, 2146 passed, 2058 skipped, 379 errors**, decomposing exactly as 13 [F-C2](#f-c2) + 5 [P1](#p1) + 379 cluster opens on the dead mesh. Filtered to host-only: **5 failed, 2140 passed** | **Red, and not by this milestone's doing** — every failing and erroring file is byte-identical to the Milestone A tip or needs hardware. Stated as a failure anyway |
| **Topology assumptions discovered in common code** | **Four, all found on silicon and all now derived or explicit:** (a) the LM-head reduce core count was a copied literal 32 ([D-B17](#defects-found-on-the-bring-up-path)); (b) the reduce envelope reserved no cores for the CCL's fabric links ([D-B27](#defects-found-on-the-bring-up-path)); (c) the vocabulary padding had to be ring-exact, not merely tile-aligned ([D-B19](#defects-found-on-the-bring-up-path)); (d) `HEAD_LOCAL` decode norm placement is a function of the loaded partition, not of the tensor ([D-B26](#defects-found-on-the-bring-up-path)). A fifth is **open**: the selector matmul's grid is named independently of the sub-device that must contain it ([D-C8](#d-c8)) | Recorded and corrected, except D-C8 which needs a design decision |
| **Did the extension stay inside module / config / model boundaries?** | **Yes.** Nothing outside `models/common/{models,modules,tests}` and the evidence directories changed — the check is `git diff --name-only bc6ad03bfc2..HEAD \| grep -v '^models/' \| grep -v '^tttv2_'`, which is **empty**. No `is_galaxy`, model-name, architecture or mesh-shape branch was added to a common runtime hot path; there is no common runtime diff at all. Concat-32 needed no `Attention2D` change and device sampling needed no module change — the selector is a model-layer collaborator | **Boundary preserved** |

### The eight shared files, and why configuration alone could not do it

| File | Change | Why config alone was insufficient |
| --- | --- | --- |
| `modules/prefetcher/prefetcher_2d.py` (+118/−5) | `defer_global_cb` and `release_global_cb_on_prefill`, both defaulting **False** | It *is* a config value — rung 1 of the extension discipline — but the value has to change **when** an allocation happens, and no existing field could. The obvious alternative, injecting a lazy proxy through the existing `create_global_cb` hook, cannot work: `ttnn.dram_prefetcher(global_cb=...)` is a nanobind boundary that needs the real object. Default `False` keeps Milestone A's qualification bit-for-bit |
| `modules/rmsnorm/rmsnorm_2d.py` (+219/−14) | `decode_compute_cores` for the `HEAD_LOCAL` geometry, plus a placement helper that refuses an interleaved-to-interleaved move | [D-B26](#defects-found-on-the-bring-up-path). An interleaved `ttnn.rms_norm` resolves a program config over the **whole compute grid**, which no configuration of the tensor can change. The grid has to be nameable |
| `modules/lm_head/lm_head_2d.py` (+96/−10) | Vocabulary-padding validation loosened from "exactly the minimal multiple" to "a multiple, at least the minimum, at most one extra shard per row"; explicit `sub_device_id`; stage masks | **The old rule forbade the only width the decode chain can run.** `all_reduce_async`'s kernel waits for a full shard on every output core, and Llama's minimal padding leaves 501 tiles, which no usable core count divides. A validation that rejected a legal geometry — loosened in the direction hardware requires, with a new upper bound so it still fails closed |
| `modules/attention/attention_2d.py` (+77/−10) | `wo` source shape `(dim, dim)` → `(n_heads * head_dim, dim)`; decode/prefill page-table validation split | The square `wo` cannot express Qwen3-32B's decoupled geometry at all. The page-table split is a **correction**: Milestone A's single validator required ≥ 32 rows for decode, which is the *prefill* layout — see the [D-C1 correction](#correction-to-d-c1-established-here) |
| `modules/rope/rope_2d.py` (+83/−3) | Prefill table copy tilizes; transformation matrix is `TILE_SIZE`-square | Two corrections, not extensions. Two consumers require **different** layouts of one stored table, so there is nothing to configure; and the op validates `[-1] == TILE_WIDTH`, which the module and its own host test both got wrong |
| `modules/sampling/sampling_2d.py` (+19/−3) | Accepts the ring-exact vocabulary padding alongside the minimal one | Follows [D-B19](#defects-found-on-the-bring-up-path); a strict widening of an input contract |
| `models/galaxy/resources.py` (+46) | `allow_narrow_semaphore_cores`, defaulting **False**, and a validator that the semaphore cores cover the workers | Promotes Milestone A's **D3** invariant from test plumbing into production validation. The opt-out exists because the fused RMS all-gather legitimately binds its semaphore to a grid it owns |
| `models/galaxy/__init__.py` (+101/−3) | Exports | Mechanical |

### What the scorecard does not say

It does not say the design is right — only that it stayed where it was supposed to. Two things inside
the boundary are worth an architect's attention before Milestone C builds on them:

1. **grids are named in three places and reconciled in none.** [D-C8](#d-c8), [D-B13](#defects-found-on-the-bring-up-path),
   [D-B26](#defects-found-on-the-bring-up-path) and Milestone A's [L3](#l3-closed) are all the same
   shape: a program's core grid is chosen by a recipe, the sub-device partition is chosen by a mode
   plan, and nothing checks that the first is inside the second until ttnn raises a `TT_FATAL`.
   `recipes.rope_core_grids` already documents the class and names `_subgrid_cores` as the qualified
   helper. A validator at the plan boundary would have caught four defects.
2. **the resource layer has no free.** [L-B1](#l-b1), [D-C7](#d-c7), [D-B20](#defects-found-on-the-bring-up-path)
   and [D-S1](#d-s1) are one problem wearing four faces.

## Provenance — which pages in this tree can be trusted

Milestone C's agent will read whatever it is pointed at. Two of the four inputs to this page were
written before first silicon and one of them was never corrected, so this section names what is safe.

| Document | Status |
| --- | --- |
| `tttv2_milestone_b_evidence/{llama,qwen,coverage}/REPORT.md` | **Current.** Each is written in dated attempt sections, newest last, and each later section names what it supersedes. Read the **last** section of each first |
| `tttv2_milestone_b_briefs/job3_completion_handoff*.md` | **A family, not a file.** `_attempt4.md` is current; attempt 1's asserts a dead mesh and an untested D-B9, both false since |
| `models/common/modules/MILESTONE_A_STATUS.md` | **Corrected by this page's deliverable 2.** Its 2026-08-27 edits (`6a3e78a7227`, +79 lines) were written from dead-mesh evidence; the L3, D-B, D-C and Qwen-geometry claims among them were wrong and have been rewritten. The Milestone A record itself — the 37-case sweep, D1–D5, the scorecard — is untouched and stands |
| `models/common/modules/README.md` | **Corrected by this page's deliverable 2.** Its `6a3e78a7227` edits (+24 lines) asserted that Milestone B was unqualified on hardware and that the 80-layer model was never built. Both false |
| **Six committed source files** | **Stale and NOT corrected here** — see below |
| `tttv2_milestone_c_briefs/` (the seven-job Milestone C execution set) | **Written 17:42–17:47Z on 2026-08-28, twenty minutes before `mb-coverage` attempt 4 began.** It therefore contains no mention of [D-C9](#d-c9) or [D-S1](#d-s1), and does not know that the column user selector has since been qualified 3/3 on silicon. Nothing in it is contradicted by this page; it is simply older than two findings. `tttv2_milestone_c_brief.md` says so at the top |
| `tttv2_2d_modules_plan.md` | **Unchanged by this job**, and its Milestone C section still describes the full scope including vLLM, which has since been deferred. `tttv2_milestone_c_brief.md` records the split; the plan was deliberately not edited to match |

### <a id="why-this-page-did-not-fix-the-stale-docstrings"></a>Six source files still say "This file has never been executed", and this page did not fix them

```text
models/common/tests/models/llama33_70b_galaxy/test_full_model_wh_galaxy.py:22
models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py:9
models/common/tests/models/qwen3_32b_galaxy/test_full_model_wh_galaxy.py:22
models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py:19
models/common/models/llama33_70b_galaxy/demo.py:24
models/common/models/qwen3_32b_galaxy/demo.py:24
```

Every one has now run on silicon, most of them many times; two of them produced the accuracy gates.
A seventh stale claim is in `models/common/models/galaxy/collectives.py:402`, where
`GalaxyColumnUserSelector`'s docstring still says *"**Unqualified.** This composition has never run on
a Galaxy mesh"* — it ran 3/3 on 2026-08-28 for 49 seconds of mesh and is qualified.

**They were left alone on purpose, and the reason is the same reason this page can qualify anything at
all.** The two teacher-forced accuracy rows, the batch-32 demo rows and the block gates are qualified
*because* their producing test files are byte-identical between the run's commit and `HEAD`. Editing a
docstring in `test_full_model_wh_galaxy.py` or in either `demo.py` breaks that identity and downgrades
five gate rows from "measured at this tree" to "measured at a superseded tree", on a mesh that cannot
re-measure them. **A one-line comment fix is not worth a gate row.**

They should be corrected in the same change that next touches those files on a working mesh — which is
also when the numbers can be re-taken. Until then, treat "never been executed" in this tree as
**false**, and check the evidence packages instead.

## Reference — evidence packages

| Package | Contents |
| --- | --- |
| `tttv2_milestone_b_evidence/reconcile/` | Job 0. The rebase onto the final Milestone A tree, C1–C10 disposition, the C1 fix deviation and the subgrid-overlap audit |
| `tttv2_milestone_b_evidence/llama/` | `mb-llama`, three attempts. `REPORT.md` is cumulative: §1–§9 attempt 1, §A2 attempt 2, §A3 attempt 3. Logs in `logs/`, `logs2/`, `logs3/` |
| `tttv2_milestone_b_evidence/qwen/` | `mb-qwen`, two attempts. Attempt 1 host-only; **§A2 supersedes its verdict**. Logs in `logs2/` |
| `tttv2_milestone_b_evidence/coverage/` | `mb-coverage`, four attempts, §A2/§A3/§A3-op/§A4. `RESULTS_A*.md` are the per-run indexes; `VERDICTS_A4.txt` and `RESULTS_A4_MACHINE.md` are machine-written and were produced with no agent awake. Logs in `logs2/`, `logs3/`, `logs4/`. `queue4.txt` is the unconsumed resume point |
| `tttv2_milestone_b_evidence/signoff/` | This page. `logs2/s2_*.log` are the eighteen checks it ran, and `REPORT.md` records what the job did, what it decided without being able to ask, and what it deliberately did not do |

Raw pytest logs are excluded from git by the repository's `*.log` ignore rule and remain on the host
that produced them (`wh-glx6u-05`). Every claim above names the log behind it.

---

*Written by `mb-signoff` attempt 2, 2026-08-28, unattended, host-only. No device was taken: the mesh
has been unusable since 18:37Z and this job needed none. No implementation file was changed. No test
was written, deleted, skipped, `xfail`ed or relaxed, and no threshold was touched.*
