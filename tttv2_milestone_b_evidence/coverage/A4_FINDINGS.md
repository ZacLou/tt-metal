
## Findings, attempt 4

### The selector is qualified, and it is not the defect — `test_column_user_selector_wh_galaxy.py`, 3/3

`models/common/tests/models/galaxy/test_column_user_selector_wh_galaxy.py` opened
with the sentence **"This file has never been executed."** That was true at
18:29Z on 2026-08-28 and `logs2/` proves it: no attempt has a log with that stem.
It is also the file the `GalaxyColumnUserSelector` docstring points at — *"That
composition is the only unqualified step in the Milestone B device sampling path,
so it is worth qualifying on its own — a failure here is a placement problem,
whereas the same failure inside a 70B demo is a needle in a haystack."*

It cost **49 seconds of mesh in total** — no checkpoint, no weights, a 256-wide
synthetic tensor — and it passes at three fresh processes:

| log | result |
| --- | --- |
| `logs4/a4_selector.log` | 2 passed in 31.32s |
| `logs4/a4_selector_run2.log` | 2 passed in 8.66s |
| `logs4/a4_selector_run3.log` | 2 passed in 8.89s |

Both cases: column `c` receives exactly users `8c .. 8c + 7` in order, twice per
run so the cached selector is exercised on re-entry, **and** selector-plus-
`Sampling2D` reproduces a per-user argmax for all 32 users.

That is the load-bearing negative result of the night. Everything area 4 has
failed with — D-C5, D-C8, D-C9 — is *around* this composition, not in it.

### D-C9 — **new**: the sampled-token readback composes the wrong mesh axis

**What was measured.** `a4_q_dc8` and `a4_q_dc8_run2`, two fresh processes,
byte-identical:

```text
[dc8] greedy tokens: [265, 2631, 1916, 220, 17, 15, 17, 17,
                      265, 2631, 1916, 220, 17, 15, 17, 17,
                      265, 2631, 1916, 220, 17, 15, 17, 17,
                      265, 2631, 1916, 220, 17, 15, 17, 17]
[dc8] greedy agrees with host argmax in 7/32 slots
[dc8] the same seed in the same slot repeated in 32/32 slots
[dc8] <every policy>: padded ids sampled in slots []
```

Thirty-two slots, eight distinct tokens, repeated four times — one mesh column's
users standing in for all four. Slots 0-3 and 5-7 agree with the host argmax;
slots 8-31 are copies of slots 0-7.

**This is also the limit of what area 4's two positive readings prove.** The
padded-vocabulary guarantee and the seed-stability claim were asserted over all
32 composed entries and held — but 24 of those entries are duplicates of the
other 8, so what is measured is those claims for **eight distinct users**.

**Root cause, and it is already written down in this repository.**
`models/common/models/galaxy/collectives.py::compose_galaxy_logits` carries the
full diagnosis for the *logits* tensor one op earlier in the same graph:

> **`to_torch_auto_compose` cannot be used here, and gets it wrong silently.** It
> infers a composer from the tensor's own `tensor_topology()`, and a matmul output
> inherits its *activation's* topology, not its weight's. […] Auto-composing
> therefore concatenates the four *columns* along the vocabulary axis and takes
> one row […] **A caller that slices `[:, :vocab_size]` gets no error at all**,
> just a truncated tensor of the wrong tokens.

`models/common/auto_compose.py` says the same thing from the other side: *"For ND
meshes with replicated dimensions, the composer will concatenate all replicas,
resulting in duplicated data. Callers may want to slice the result if only one
copy is desired."*

`GalaxyDirectRunner._compose_rows` was fixed — it calls `compose_galaxy_logits`
and validates the composed width. **`GalaxyDirectRunner.decode_sampled`, sixty
lines further down the same file, still calls `to_torch_auto_compose(sampled)`
and then `.reshape(-1)[:32]`** — the exact "slice and get no error" the
docstring warns about. `ttnn.sampling`'s output inherits from `gathered_values`,
an `all_gather` over the sampling axis, so it carries activation labels for a
distribution it does not have: the eight devices of a mesh column hold identical
tokens (they all-gathered the whole vocabulary between them) and the four columns
hold different users. Concatenating the replicas first and taking the leading 32
values yields exactly eight users repeated four times.

**Where it bites.** `decode_sampled` is the single entry point every area-4 case
uses, and the two sampling diagnostics (attempt 3's `dc5`, attempt 4's `dc8`)
copied its composition. So the 7/32 number is a **readback** measurement, not a
statement about `ttnn.sampling`.

**What is *not* affected.** `decode_logits` — and therefore both teacher-forced
accuracy numbers, the batch-32 slot-isolation tests and every PCC comparison in
areas 1, 2, 3 and 5 — goes through `_compose_rows`, which is the fixed path. This
finding does not touch the exit gate.

**The fix, in one line, and it has precedent in the same file:** compose the
sampled tokens by their distribution rather than their labels —
`ttnn.ConcatMesh2dToTensor(dims=(0, <user axis>))` then mesh row 0, the mirror of
`compose_galaxy_logits(dims=(3, 0))` with the axes swapped because there it is
the rows that carry the vocabulary. Attempt 4 committed a test that does exactly
this (`test_qwen_device_sampling_claims_with_an_explicit_token_composition`,
commit `0e2c0dc50b4`) so that the arithmetic can be measured; it did **not**
change `direct_runner.py`, for the reason in §A4's method note.

### Why attempt 4 did not repair D-C5, D-C8 or D-C9

Three defects at one call site, all three with a one-line shape of fix, and the
job still did not apply them. That is a decision, not an omission:

* **the exit gate's evidence would stop being evidence.** Nine gate rows rest on
  the argument that implementation code is byte-identical to the tree they were
  measured at (§A4, point 2). `direct_runner.py` is imported by
  `test_full_model_wh_galaxy.py` and by both `demo.py` files; editing it
  invalidates every row and costs a full re-measurement — around two and a half
  hours of mesh — to end up where the job already is;
* **the brief is explicit that this job measures.** "A `FAIL` with a diagnosis is
  a complete result for this job. Milestone B's gate is a fact to be measured,
  not a target to be reached by adjusting the measurement";
* **D-C8's real fix is a design decision, not a line.** The selector accepts no
  `program_config` and no core grid; giving it one means deciding whether the
  Galaxy sampling path runs inside the decode worker sub-device (and if so, on
  which of `worker_cores()`' cores) or whether decode's sub-device partition
  should be widened. `recipes.rope_core_grids`' docstring already names this
  defect class — *"a grid named independently of the partition that has to
  contain it"* — and names `_subgrid_cores` as the qualified helper. That is
  Milestone C's L1/ownership work, not a night's patch.
