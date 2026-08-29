# `perf/ENVIRONMENT.md` — the environment that must be held constant between the two arms

Written by job `c-perf-recon`, 2026-08-29, commit `6af44349413ca6ce2c0d98f5b26dd2898dc1f067`.

This page exists so `c-perf-paired` can **re-assert and diff** the environment rather than assume
it. Every value below was captured by machine, not typed. The raw captures are in
`tttv2_milestone_c_evidence/perf/recon/_env0/`:

- `host.txt` — date, hostname, `uname -a`, commit, `git status --porcelain` (tracked), `tt-smi --version`
- `tt-smi-snapshot.json` — the full 32-board `tt-smi -s` snapshot

A re-assert script is provided: `tttv2_milestone_c_runs/assert_environment.sh`. It re-captures the
same values and diffs them against the frozen expectations in
`tttv2_milestone_c_evidence/perf/recon/_env0/EXPECTED.txt`. **Run it before the first paired run and
after the last.**

## 1. Host

```text
hostname   wh-glx6u-05-special-ctr-apbernal-for-reservation-119144
distro     Ubuntu 22.04.5 LTS
kernel     6.8.0-83-generic  (#83~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC)
platform   x86_64
memory     566.12 GB
driver     TT-KMD 2.9.0
```

## 2. Mesh

```text
boards under /sys/class/tenstorrent   32      <- authoritative count
board type                            tt-galaxy-wh L
board_id (all 32)                     0100035100000000  (BOARD_ID_HIGH 0x1000351 on all 32)
cluster                               WH Galaxy, logical mesh (8, 4)
galaxy_type fixture                   "6U"   (conftest.py:59 -> is_6u())
```

`galaxy_type == "6U"` matters: `models/demos/llama3_70b_galaxy/conftest.py` turns
`fabric_config: True` into **`ttnn.FabricConfig.FABRIC_1D_RING`** on 6U and `FABRIC_1D` on 4U. Both
arms must see the same value. Do not set `MESH_DEVICE`; leave it unset (see §5).

## 3. Firmware — identical on all 32 boards

```text
fw_bundle_version   19.12.0.0        (FW_BUNDLE_VERSION 0x130c0000)
cm_fw               2.43.0.0         (2026-05-07)
eth_fw              7.5.0            (ETH_FW_VERSION 0x75000)
dm_bl_fw            129.2.0.0        (DM_BL_FW_VERSION 0x81020000)
dm_app_fw           5.4.0.0          (DM_APP_FW_VERSION 0x5040000)
tt_flash_version    0.3.10.0
ARC0/1/3 FW         0x22b0000        (uniform across all 32)
SPIBOOTROM FW       0x30e0000        (uniform across all 32)
gddr_fw             N/A
```

## 4. Host software

```text
tt-smi     5.2.0
pyluwen    0.8.5
tt-umd     0.9.5
python     3.10.21  (python_env/bin/python; tt-smi's own interpreter reports 3.10.20)
pytest     9.0.3    (plugins: timeout 2.4.0, repeat 0.9.4, split 0.11.0, benchmark 5.2.3, cov 7.0.0)
```

## 5. Repository and process environment

```text
repo        /proj_sw/user_dev/ctr-apbernal/tt-metal
branch      apbernal/tttv2_wh_glx_2d_modules_milestone_c
commit      6af44349413ca6ce2c0d98f5b26dd2898dc1f067
tracked working tree at capture time: clean (`git status --porcelain --untracked-files=no` empty)

export TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
export PYTHONPATH=/proj_sw/user_dev/ctr-apbernal/tt-metal
export HF_HOME=/localdev/ctr-apbernal/hf_data
unset  MESH_DEVICE
unset  LLAMA_DIR
unset  TT_CACHE_PATH
unset  TTTV2_GALAXY_CCL_TRACE          # see the measurement trap below
unset  LINE_RS LINE_AG                 # debug-only line-topology CCL overrides
```

`MESH_DEVICE` must stay **unset**. `models/demos/llama3_70b_galaxy/demo/text_demo.py:930-932` skips
any batch other than 1 or 32 when `MESH_DEVICE == "TG"`; the mesh shape itself comes from the
`mesh_device` fixture parametrised `(8, 4)` indirect, not from the env var. `PERF.md` only sets
`MESH_DEVICE=TG` for the vLLM launch lines, which are out of scope for Milestone C.

**`TTTV2_GALAXY_CCL_TRACE=1` must never be set for a timed run.** It synchronises after each LM-head
collective and dominates wall clock (`tttv2_milestone_c_brief.md`: 1356 s with, ~950 s without, for
the same 511-token run). It cannot change numerics — only the number you are trying to measure.

## 6. Checkpoints — the arm-specific difference

**Both arms share `HF_HOME`.** That is the important answer to the brief's question 6: TTTv1 does
**not** need a different `HF_HOME` or a different checkpoint layout from TTTv2. It reads the same
HuggingFace hub cache.

```text
HF_HOME=/localdev/ctr-apbernal/hf_data     <- this exact value, for BOTH arms

hub/models--meta-llama--Llama-3.3-70B-Instruct   refs/main = 6f6073b423013f6a7d4d9f39144961bfbfbc386b
hub/models--Qwen--Qwen3-32B                      refs/main = 9216db5781bf21249d130ec9da846c4624c16137
```

`/proj_sw/user_dev/hf_data` reaches **Llama only**, and the value inherited from an interactive
shell (`.../hf_data/hub`) is one directory too deep and holds only Mistral. Under either, weight
loading fails or `hf_config_or_skip` turns a real-checkpoint test into a silent `SKIPPED`.

**`$HF_HOME/hub` is a hand-assembled symlink farm, not a real HuggingFace cache.** It was created
2026-08-29 08:14 UTC and holds exactly two entries:

```text
models--meta-llama--Llama-3.3-70B-Instruct -> /proj_sw/user_dev/hf_data/hub/models--meta-llama--Llama-3.3-70B-Instruct
models--Qwen--Qwen3-32B                    -> /proj_sw/user_dev/Qwen/models--Qwen--Qwen3-32B
```

That is *why* `/proj_sw/user_dev/hf_data` reaches Llama only — the Qwen checkpoint is not under
`hf_data` at all, it lives in `/proj_sw/user_dev/Qwen/`. `run_milestone_c_jobs.sh:432-439` only
**checks** for the two directories; **no job recreates them.** If the farm is lost, every
checkpoint test goes back to skipping and the run looks green. Assert the two symlinks resolve
before the first paired run.

Both checkpoint directories are owned by other accounts and are **read-only** to this one. Two
consequences, both benign but both worth recognising in a log:

- the HF hub layer prints, ~40 times per Qwen run,
  `Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 13]
  Permission denied: '…/models--Qwen--Qwen3-32B/.no_exist'`;
- the Llama snapshot's own `TG/` converted-weight cache (105 GB, dated 2026-06-08) **cannot be
  written to**, so `TT_CACHE_PATH` must point somewhere this account owns.

**TTTv1 additionally requires `HF_MODEL` to be exported**, per model. TTTv2 does not use it.
`models/demos/llama3_70b_galaxy/tt/model_config.py:509-534` asserts on startup if neither `LLAMA_DIR`
nor `HF_MODEL` is set:

```text
Llama arm (TTTv1)   export HF_MODEL=meta-llama/Llama-3.3-70B-Instruct
Qwen  arm (TTTv1)   export HF_MODEL=Qwen/Qwen3-32B
```

`LLAMA_DIR` and `HF_MODEL` are mutually exclusive (`model_config.py:511` asserts). `LLAMA_DIR` must
stay unset.

## 7. Converted-weight caches — no collision between the arms

TTTv1 derives its cache directory as `model_cache/<HF_MODEL>/<device_name>` where `device_name` is
`"TG"` for 32 devices (`model_config.py:470,526`):

```text
TTTv1 Llama   model_cache/meta-llama/Llama-3.3-70B-Instruct/TG
TTTv1 Qwen    model_cache/Qwen/Qwen3-32B/TG
TTTv2         model_cache/<org>/<model>/galaxy_8x4/{attn,mlp,norm,rope,lm_head,embedding}
```

Different leaf directories, so the two arms **cannot** overwrite each other's converted weights.
Confirmed on disk. `model_cache/` already held 139 G (meta-llama) + 109 G (Qwen) of TTTv2 caches
before this job; `/proj_sw` had 1.2 T free. Budget for TTTv1's `TG/` caches on top.

`TT_CACHE_PATH`, if set, overrides the derivation above (`model_config.py:517,524`). **This job sets
it deliberately for both arms** — see `BASELINE_PROCEDURE.md` §1.2 and §2.2 — because the default
derivation lands on `/proj_sw`, and `/proj_sw` is not safe for a 100 GB write (§7a). Note the two
branches differ: under `LLAMA_DIR` the value is used verbatim, under `HF_MODEL` the device name is
appended (`model_config.py:527`).

Sizes actually observed: Llama `tensor_cache_instruct_bfp8` **111 GB** / 1316 files, Qwen
`tensor_cache_bfp8` **90 GB** / ~1000 files.

## 7a. `/proj_sw` is shared and filled mid-run — this is a scheduling hazard

At **2026-08-29T10:25:04Z** the shared 30 TB weka mount returned `ENOSPC (-28)` while a device run
was writing its converted-weight cache, and killed it outright:

```text
dmesg: wekafsio: … N[layers.60.feed_forward.w3_inter] … Truncated dirty-pages
       total-errors(ENOSPC(2), OTHER(0)) … => -28
dmesg: wekafsio: … N[run.log] … total-sync-errs(ENOSPC(1), OTHER(0)) sync => -28
```

`df` reported 950 GB free the whole time. Even `run.log` failed to sync, so the run left **no exit
line and no metrics**. Two minutes later a 5-byte write to `/proj_sw` still failed; by 10:38 it
succeeded again with 1.1 TB free. Evidence: `perf/recon/qwen_b32_run1_cold/` (truncated log, no
`exit=` line, no `wall_clock_s` in `meta.txt`).

**Mitigation, now part of the procedure:** converted-weight caches go on `/localdev` (local disk,
1.9 TB free) via `TT_CACHE_PATH`. Check `/proj_sw` has headroom before a night starts, and treat a
run that ends with no `exit=` line in its log as an infrastructure kill, not a model failure.

## 7b. The tt-metal JIT kernel cache decides how long a run takes

```text
/home/ctr-apbernal/.cache/tt-metal-cache      11 GB after these runs      TT_METAL_CACHE is unset
```

First run on a host vs steady state, same command, same commit:

| | prefill compile | decode compile | wall clock |
| --- | --- | --- | --- |
| Llama, first run | 1687.81 s | 119.07 s | 2519 s |
| Llama, steady state | 393.19 s | 19.96 s | 1104 s / 1036 s |
| Qwen, first run | 877.75 s | 104.15 s | 1546 s |
| Qwen, steady state | 381.02 s | 27.09 s | 531 s |

This is what the paired methodology's **unmeasured warmup** is protecting against. Hold `HOME`
constant across all runs of a night, and do not clear this cache between the warmup and the measured
runs.

## 8. What "held constant" means operationally

`c-perf-paired` must assert, before its first run and after its last:

| Constant | How to check |
| --- | --- |
| commit | `git rev-parse HEAD` == `6af44349413…` (or its own commit, asserted identical across all 16 runs) |
| tracked tree clean | `git status --porcelain --untracked-files=no` empty |
| board count | `ls /sys/class/tenstorrent \| wc -l` == 32 |
| firmware | `tt-smi -s` → every board's `firmwares` block matches §3 |
| driver / tt-smi / umd | §1, §4 |
| galaxy_type | `6U` — otherwise `fabric_config` silently changes topology |
| checkpoint revision | `cat $HF_HOME/hub/models--*/refs/main` matches §6 |
| env | `HF_HOME` exact; `MESH_DEVICE`, `TTTV2_GALAXY_CCL_TRACE`, `LINE_RS`, `LINE_AG` unset. `LLAMA_DIR`/`HF_MODEL`/`TT_CACHE_PATH` are set **per process, per arm** (§6, §7) and are never both set at once |
| HF symlink farm | both entries under `$HF_HOME/hub` resolve (§6) |
| JIT kernel cache | `HOME` unchanged, `/home/ctr-apbernal/.cache/tt-metal-cache` not cleared between the warmup and the measured runs (§7b) |
| `/proj_sw` headroom | checked before the night; nothing large written there (§7a) |

If any of these differ between the two arms, **the pairing is void** and the runs are
reconnaissance, not a paired measurement.
