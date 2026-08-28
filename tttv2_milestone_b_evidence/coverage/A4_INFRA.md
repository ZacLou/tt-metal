
## The mesh went down at 18:37Z, and what was done about it

**This section exists because the night's plan changed here, and because the next
attempt needs to know the state of the hardware, not just of the evidence.**

### The sequence, from the machine-written logs

| time (UTC) | what | log |
| --- | --- | --- |
| 18:35:20 | `a4_q_dc8_run2` exits `rc=1` (an `AssertionError`, no `TT_FATAL`); the wrapper resets as it does after any non-clean exit | `logs4/a4_q_dc8_run2.log` |
| ~18:36:50 | the reset reaches `Issuing POST_RESET on 32 devices after IPMI reset` and prints **`Error: POST_RESET failed for device 21`**, `reset exit=1`. All 32 chips were found in `/dev/tenstorrent` before that step | `logs4/reset_a4_q_dc8_run2.log` |
| 18:37:08 | `a4_q_dc9_bisect` errors **at setup** in 7.51s: `RuntimeError: Read 0xffffffff over PCIe ID 21: the board should be reset` (`ttnn/ttnn/distributed/distributed.py:631`) | `logs4/a4_q_dc9_bisect.log` |
| 18:37:54 | `a4_q_dc8_run3` — same setup error, 7.41s | `logs4/a4_q_dc8_run3.log` |
| 18:38:24 | **queue halted** (`queue4.halt`) | — |
| 18:38:40 | `a4_l_dc8` — same setup error, 7.40s; the runner honours the halt and stops cleanly | `logs4/a4_l_dc8.log`, `logs4/queue4.out` |

PCIe ID 21 is the board whose `POST_RESET` failed. This is the failure mode
`cov_after_device_run.sh` was already commented for — *"which left a chip's ARC
firmware half-initialised … and cost a full recovery cycle"* — except that this
time the recovery cycle did not succeed.

### Recovery attempts, all logged

| # | command | result | log |
| --- | --- | --- | --- |
| 1 | `tt-smi -glx_reset` (900s cap) | fails **earlier** than the wrapper's did — at `Issuing USER_RESET`, with `[Errno 6] No such device or address: '/dev/tenstorrent/7'` | `logs4/recovery1_glx_reset.log` |
| 2 | `tt-smi -glx_reset` again | identical failure | `logs4/recovery2_glx_reset.log` |
| 3 | `tt-smi -glx_reset_auto` (the tool's own 3 internal retries) | `Trying reset (1/3)`, `(2/3)`, `(3/3)`, then `Failed on last reset...exiting with error code 1` — each attempt stops at `USER_RESET` on device 7 | `logs4/recovery3_glx_reset_auto.log` |
| 4 | `tt-smi -glx_reset_tray 1` | refused: *"Galaxy 6U tray reset is no longer supported. Please use tt-smi -glx_reset to reset all chips or tt-smi -r."* | `logs4/recovery4_tray1.log` |
| 5 | `tt-smi -r all` | see below | `logs4/recovery5_r_all.log` |

### Two independent faults, not one

* **device 21** reads `0xffffffff` — the symptom the ttnn error names, and the one
  a reset is supposed to clear;
* **device 7** cannot be *opened at all*. `os.open('/dev/tenstorrent/7', O_RDWR)`
  raises `OSError [Errno 6] No such device or address` while 0, 21 and 31 open
  fine. All 32 nodes are present in `/dev/tenstorrent`; the node existing and the
  chip answering are different things, which is why `ls /dev/tenstorrent | wc -l`
  is a necessary and **not** a sufficient mesh-health check.

Device 7 is what makes this unrecoverable from inside the job: every `glx_reset`
path enumerates all 32 chips at `USER_RESET` and aborts on the first `ENXIO`, so
it never gets as far as the chip that actually needs the reset.

### The kernel agrees, and says why

`dmesg` (captured in `logs4/recovery_dmesg.log`):

```text
tenstorrent: Skipping message 00000011 due to FW not running.
tenstorrent 0000:01:00.0: Device is unresponsive, cannot reset.
tenstorrent: Skipping message 00000011 due to FW not running.
```

and, earlier in the same window, a kernel stack trace through
`tt_hwmon_read+0x45/0xa0 [tenstorrent]`. Driver version 2.4.1. A chip whose
firmware is not running, that the driver itself declines to reset, is not
recoverable by any user-space command available to this job: it needs an IPMI
power cycle of the tray or a host reboot, and both are outside what an unattended
job may do to shared hardware without being asked.


## What actually broke the mesh, dated to the second — and attempt 4's own test is a suspect

`logs4/dmesg_dated.log` converts the kernel's boot-relative timestamps to UTC.
The result changes the story from "a reset failed" to something more specific,
and it does not let this attempt off the hook.

| time (UTC) | kernel | what was running |
| --- | --- | --- |
| 18:34:42 | **7234 × `tenstorrent: pin_user_pages_longterm failed: -14`** | `a4_q_dc8_run2`, 18:32:26-18:35:07 |
| 18:35:05 | a kernel stack trace through `tt_hwmon_read+0x45/0xa0 [tenstorrent]` | the same run's teardown |
| ~18:36:50 | — | its wrapper `glx_reset`: **`Error: POST_RESET failed for device 21`** |
| 18:37:08 onwards | — | three runs error at setup: `Read 0xffffffff over PCIe ID 21` |
| 18:41:33-18:41:45 | `Skipping message 00000011 due to FW not running`, `0000:01:00.0: Device is unresponsive, cannot reset` | recovery 3, `glx_reset_auto` |
| 18:46:34-18:46:58 | the same pair for `0000:c6:00.0`, `0000:c7:00.0`, `0000:c8:00.0` | recovery 5, `tt-smi -r all` |

**All 7234 pin failures fall inside one minute, and there is not one other
occurrence in this boot** — the host has been up 29 days. So this is not
background noise, it is an event, and it happened *inside* a run of a test this
attempt wrote 40 minutes earlier, not inside the reset that followed.

**The honest position on cause.** `-14` is `EFAULT` on pinning user pages for
DMA. Attempt 4's `dc8` case does something no earlier case did: it calls
`model.activate("prefill")` and then, via `runner._decode_device_logits`,
`model.activate("decode")` — **six full sub-device-manager cycles in one
process**, each of which stops and restarts the `Prefetcher2D`'s DRAM prefetch.
That is a plausible way to exhaust or corrupt pinned host pages, and it is the
one genuinely new thing in the window. Against that: the *first* run of the same
case, `a4_q_dc8`, completed at 18:31 and its `glx_reset` printed
`Re-initialized 32 boards after reset`, so one process of six cycles was
survivable; and pinned pages are released at process exit, which makes a
cross-process accumulation story harder to tell.

**It is not established either way, and it should not be guessed at.** What
would establish it, in one run each, on a repaired mesh:

1. run `a4_q_dc8` once with `dmesg -w` recording, and see whether the pin-failure
   burst reappears. If it does, the case is the cause;
2. if it does, run a variant that does **one** activate cycle instead of six —
   sample once rather than six times — and see whether the burst goes away.

**Until that is known, the queue guards against it.** `cov_queue4.sh` now counts
`pin_user_pages_longterm failed` and `Device is unresponsive` in `dmesg` before
and after every run, and if either grows it dumps the tail of `dmesg` to
`logs4/kernel_guard_<name>.log`, writes `queue4.halt` and stops. **`a4_q_dc9_explicit`
is position 1 in the queue and performs the same six activate cycles**, so the
guard will fire on the very run that matters most if this is real — which is the
correct outcome: one halted queue is cheaper than another lost night of Galaxy.
