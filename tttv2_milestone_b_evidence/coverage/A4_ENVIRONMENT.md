

# Attempt 4 — environment, verified rather than inherited

Appended 2026-08-28 by `mb-coverage` attempt 4. The attempt-3 environment
section above is unchanged; this records what attempt 4 checked for itself
before planning, because the brief it was handed had been wrong about the
environment once already (attempt 1 declared the mesh dead; attempt 2 found it
alive).

| Fact | How it was established | Value at 18:07Z |
| --- | --- | --- |
| device nodes | `ls /dev/tenstorrent \| wc -l` | **32** |
| device holders | `pgrep -af 'pytest\|ttnn'` | none but this session's own wrapper |
| attempt-3 queue | `pgrep -af cov_queue`, `queue.halt` | not running, halt file present, 28 items unconsumed |
| screen session | `screen -ls` | one, `ttmb`, created **18:06:16Z** — this attempt's own driver, not the wedged one attempt 3 reported |
| driver PID | `$MB_JOB_DRIVER_PID` | 234017, and never signalled |
| repo state | `git status --porcelain` | no uncommitted **tracked** changes at start |
| branch | `git rev-parse --abbrev-ref HEAD` | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| start commit | `git rev-parse HEAD` | `110ba1f0658d3485d778db0da7fefcd5223998e5` |
| `/proj_sw` free | `df --output=avail -B1G /proj_sw` | **1152 G** — above `cov_queue4.sh`'s 300 G prune threshold, so the disk guard never fired |
| Python env | reused, never rebuilt | `python_env/bin/python`, 3.10.21, pytest 9.0.3 |
| `HF_HOME` | exported by `cov_run4.sh` | `/localdev/ctr-apbernal/hf_data` |

**The wedged driver attempt 3 reported is gone.** `run_milestone_b_jobs.sh` PID
10669 and its `ttmb` screen session are not present; the session that exists was
created at 18:06:16Z, one minute before this attempt started, which is this
attempt's own. Nothing was killed to achieve that — it was already clear.

**Perishable input, re-checked.** `/tmp/tttv2_step7_artifacts/` still holds the
four cross-process pool recordings attempt 3 wrote, so `a3_h14` and `a3_h12`
remain reproducible from disk. Attempt 4 copied nothing out of `/tmp`: both
comparisons are already recorded, and their logs are in `logs3/`.
