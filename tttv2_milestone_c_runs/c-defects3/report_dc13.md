
## D-C7's remaining half: D-C13, and why it is the same shape of defect

Attempt 1 fixed D-C7 itself — the surviving `Prefetcher2DContext` reference — and measured the L1
coming back to within 192 bytes, three fresh processes, byte-identical:

```text
d11_q_two_pools_run{1,2,3}   allocated: 131520 B, free: 1261952 B, largest free block: 759488 B
```

against Milestone B's `923776 / 469696 / 373824`. But all three runs still failed, because
**1 261 952 B are free and the largest contiguous block is 759 488 B — 32 576 B short** of the
792 064 B the second model's global circular buffer needs. The bank is fragmented, not full.

Read as a layout, those three numbers say something specific. Bank size 1 393 472 B, of which
131 520 B are allocated in **two** places, because a single contiguous allocated region at the top
would leave a single free block of 1 261 952 B. Two free blocks of 759 488 B and
`1261952 − 759488 = 502464` B, split by an allocated block, is the only reading:

```text
top   1499136 ┬ allocated  (the second model's resident L1)
              ├ free       759488
              ├ allocated  (a small block, low)
              ├ free       502464
base   105664 ┴
```

So **D-C13 is the same defect as the address clash**: a small long-lived L1 buffer sitting low
enough to split the bank, allocated while something large held the space above it. That makes
`global_cb_headroom` the candidate fix for it as well as for the clash, and it makes the two-pools
test the measurement — which is why it is queued in this attempt rather than treated as a separate
investigation.

### What the two-pools test actually did once the OOM was gone: D-C14

With `release_global_cb_on_prefill` and the headroom in place, the second model's global circular
buffer **is created** - the OOM at `largest free block: 759488` does not happen. The test then hung.
`gdb`, attached before any recovery attempt
(`tttv2_milestone_c_runs/c-defects3/logs/m1b_hang_bt.txt`):

```text
#0  pthread_cond_wait
#2  tt::tt_metal::distributed::FDMeshCommandQueue::wait_for_outstanding_reads
#3  FDMeshCommandQueue::finish_nolock
#5  tt::tt_metal::distributed::Synchronize
#6  <nanobind>  ttnn.synchronize_device
```

A device completion that never returns, with **1676** `cache hit` lines in the log — 64 layers ×
~13 tensors × 2 models — so both models' weights were resident and the stall is at the second
model's first decode. Attempt 2's `e1_qwen_two_pools_l1` shows **38** at a one-layer subset, which
is 2 × 19: the same point, under an earlier version of this code. So the hang is not created by
this fix; it was **masked** by D-C13, which used to fail first.

The reduction is read out of `tt_metal`:

- the ttnn program cache belongs to the **mesh device**, not to a model, and outlives `close()`;
- its keys are op-and-config hashes, so two structurally identical Galaxy models in one process
  hash to the same keys — the second model's decode is a cache **hit** on the first model's
  programs;
- `CircularBufferImpl::set_global_circular_buffer` captures `buffer_address()` and
  `config_address()` once, and `dispatch.cpp:3035` re-sends the captured pair on every launch.

So the invariant is **per process and mesh**, not per `Prefetcher2D`. `GlobalCBPlacement` is that
record; `models/common/models/galaxy/prefetch.py` holds one per mesh device. Host-qualified
(33 passed) and **not qualified on silicon** — the mesh lost board 23 to POST_RESET before it could
be. D-C13 itself is therefore superseded rather than closed: the allocation now succeeds and what
stands behind it is D-C14.
