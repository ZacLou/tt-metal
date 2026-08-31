
---

# Attempt 3 (2026-08-31) — the Llama L1 address clash, fixed

## What the defect was, stated correctly for the first time

Three earlier accounts of this defect read the message wrong in the same way. The message is

```text
TT_THROW ... Statically allocated circular buffers in program 100 clash with L1 buffers on
core range [0-0 - 0-3]. L1 buffer allocated at 544832 and static circular buffer region
ends at 630080
```

`detail::ProgramImpl::validate_circular_buffer_region` (`tt_metal/impl/program/program.cpp:1658`)
computes **one** `lowest_address` for the whole program — `device->lowest_occupied_compute_l1_address`
— *before* it loops over the program's circular-buffer core ranges, and then reports that one number
against whichever range trips first. So `[0-0 - 0-3]` locates the **circular buffers**, not the L1
buffer, and the buffer is not on the prefetch sender column. Second, `FreeListOpt::get_memory_block_table`
prints `block_address_[i]` **without** `offset_bytes_` while `lowest_occupied_address()` **adds** it,
and on this part that offset is `1499136 - 1393472 = 105664`. Attempt 2 measured the offset directly
(`ttnn.get_allocator_base_address` → 105664, `logs/a2_clash_owner_l1.log`) rather than deriving it,
and with the two coordinate systems reconciled the "unfindable" buffer was in attempt 1's own
preserved dumps all along: a **32-byte** block, which is why a filter for live blocks over 100 kB
never showed it.

## The defect has two independent halves

Both are attributed to the call that makes them, by a probe that prints the allocated-block table on
each side of every `Prefetcher2D` step that can allocate (`logs/a3_clash_steps_l1.log`):

| step | L1 blocks below 630080 afterwards |
| --- | --- |
| `runner.open()` | none; lowest allocated 1373408 |
| `activate("prefill")` | none |
| first `activate("decode")` → `_ensure_global_cb` | **578560 (192 B) and 578752 (792064 B)** |
| the **first decode forward** | **545760 (32 B)** appears, and is never freed |
| every later mode switch | unchanged |

1. **The global circular buffer is itself below the line whenever it is resident.** It is
   `GALAXY_GLOBAL_CB_SIZE` = 792 064 B plus a 192-byte config page, allocated top-down beneath the
   model's resident L1, landing at 578752 — 51 328 B below the 630080 the prefill embedding's static
   circular buffers reach. And this is not a placement accident that a different order would fix:

   | quantity | bytes |
   | --- | --- |
   | L1 above the prefill CB region end (1499136 − 630080) | **869 056** |
   | the model's resident L1 after a decode (1499136 − 1371360) | 127 776 |
   | the global circular buffer, data + config | **792 256** |
   | sum | **920 032** |

   920 032 > 869 056 by 50 976 B, so **the buffer and a prefill program of this shape cannot
   coexist on any allocation order.** `defer_global_cb` covers only the *first* prefill.
   `release_global_cb_on_prefill` is therefore not an optimisation, it is required — and a serving
   system interleaves prefill and decode by construction.

2. **The first decode forward strands a 32-byte L1 buffer at 545760.** It is allocated while the
   global CB holds the space above it, so it lands *below* the buffer rather than at the top of L1,
   and an allocated address never moves: it survives `_stop_prefetch`, the mode switch and
   `_release_global_cb`. Releasing the CB therefore frees 792 256 B and leaves this one block below
   the line — and because the throw reports the **lowest** occupied address, the message does not
   change by a single digit. **That is why three attempts read
   `release_global_cb_on_prefill` as "runs and does not help".** It ran and it helped; the message
   was about a different, smaller buffer.

   Attempt 2 walked the model/runner object graph for the owner and found nine reachable L1
   `ttnn.Tensor`s after the first decode, **all above 1371360** — including
   `rope_setup.config._decode_trans_mat`, materialised by the first decode, which landed at 1371360
   because it happened to fit a gap at the top. So the 32-byte block is not a reachable tensor: the
   remaining L1 owners at that layer are `GlobalSemaphore`s and buffers held inside cached ttnn
   programs. `_weight_address_metadata` — `c-exec-llama`'s named candidate — is exonerated by
   measurement at **1499104**, the very top of L1.

## What the fix is, and why it is two changes rather than one

`models/common/modules/prefetcher/prefetcher_2d.py` grows one config field and one method;
`models/common/models/galaxy/prefetch.py` changes two defaults. Nothing else in the module changes,
and no model file gains anything but a default and a comment.

**1. `release_global_cb_on_prefill` defaults to `True` for both Galaxy models.** Forced by the
arithmetic above, not chosen. The field and the release path already existed and were already
measured to return 792 256 B per bank (attempt 1's D-C7 gate); what was missing was the reason to
turn it on and the second half that makes it sufficient.

**2. `global_cb_headroom` — reserve L1 above the buffer while it is created, then release it.**
L1 is allocated top-down, so with the buffer resident every long-lived allocation lands below it
and is stranded. Reserving headroom first makes the buffer land that much lower and leaves a free
gap above it, and `FreeListOpt::allocate` scans free blocks by **ascending size class**, so a later
small allocation takes the small gap in preference to the large low block.

That much was measured in isolation before anything was committed. Three arms of a scratch probe on
the two-`generate` reproduction, one Llama layer:

| arm | log | result |
| --- | --- | --- |
| headroom only, no release | `logs/b1_headroom_only_l1.log` | **the stranded 32 B at 545760 are gone.** The only L1 left below 630080 is the global CB itself, moved down by exactly the headroom (578752 → 513024). The clash persists, because the buffer is still resident — exactly as the arithmetic says it must. |
| headroom **and** release | `logs/b2_headroom_release_l1.log` | **the second prefill placed** — no `program.cpp` throw, the first time on this branch. And then the process **hung** for four minutes in the decode after it, and was killed by PID. |
| release only | `logs/b3_release_only_l1.log` | not measured: dequeued 1 s after that kill and died in 6.35 s with `MMIO per-op timeout`, an infrastructure failure. `tt-smi -glx_reset` reported `Re-initialized 32 boards`. |

**The hang is the hazard `release_global_cb_on_prefill`'s own docstring names**: the recreated buffer
must land at the same L1 address or decode programs already in the ttnn program cache hold stale
addresses. `logs/b4_headroom_release_addr_l1.log` measured it rather than inferring it — the arm
printed the buffer's base at each creation:

```text
[fix] before create: lowest=1370816  reserve=65536  -> after create: lowest_allocated=447296
[fix] before create: lowest=1338016  reserve=65536  -> after create: lowest_allocated=414880
```

**32 416 B lower the second time**, and it hung again. A *fixed* headroom cannot put the buffer
back, because the free list is a different shape the second time: the first decode has consumed
part of the gap the headroom left. Nor can the placement be predicted — the same log shows the span
between the free-region top and the resulting floor was 857 984 B on the first creation and
857 600 B on the second, so creating a global circular buffer allocates more than the buffer and its
config page, and not a constant amount.

**The invariant that is reliable is the free-region top.** Reproduce the lowest-occupied L1 address
the first creation saw and every allocation the creation makes reproduces, so the buffer comes back
on the same floor. `_allocate_global_cb` records the free top and the floor on the first creation,
reserves down to that recorded top on every later one, and **raises** if the buffer still does not
land on the recorded floor — because a buffer that has moved is a silent wrong-address read and an
exception is strictly better than the hang.

**One reservation is not enough, and the number says why.** On the production path the first attempt
at this reserved exactly the missing bytes in one call and the buffer came back **32 736 B high** on
both models (`logs/d1_llama_repeat_l1.log`, `logs/d2_qwen_repeat_l1.log`):

```text
RuntimeError: the recreated global circular buffer did not land on its original L1 floor:
              expected 513024, got 545760
```

`545760 − 513024 = 32736`, which is exactly the leftover of the first creation's 64 kiB headroom
after the first decode allocated 32 800 B of long-lived L1 into it. `FreeListOpt` prefers a
small exact-size free block to a large one, so a single reservation of exactly the missing 32 736 B
lands *in that gap* and moves the free top by nothing at all. The fix is to reserve **in a loop**
until the lowest occupied L1 is back at the recorded top: each pass consumes one gap, gaps are
finite, and the last pass comes out of the low region and lands the top exactly.

**That pair of failures is worth keeping for its own sake.** It is the same number, to the byte, on
two different checkpoints with different head counts and vocabularies — which is the shared-code
claim the brief asks for, made by the *failure* rather than by the fix: the placement is a property
of `Prefetcher2D` and the Galaxy L1 map, not of either model. And it is a **reported** failure where
the same tree without the guard **hung**, which is the whole point of checking.
