
## The five areas after attempt 4

**This table is §A3's table plus attempt 4's deltas, and only the deltas are
attempt 4's work.** Where a cell is unchanged from §A3 it says so; the logs
behind the unchanged cells are in `RESULTS_A3.md`, not repeated here.

| Area | Llama | Qwen | attempt 4's delta |
| --- | --- | --- | --- |
| 1 paged KV | as §A3: **PASS** two-pool PCC (cross-process) and late capacity; **BLOCKED** on block-level cross-slot and one-process two-pool by the L1 address clash | as §A3: **PASS** two-pool PCC, late capacity, block-level cross-slot; **FAIL** two pools in one process (D-C7) | none. Queued (`a4_q_two_pools_run2/run3`, `a4_l_*_run2`), did not run |
| 2 concat-32 | as §A3: **FAIL, every case, D-C6** | as §A3: **FAIL, every case, D-C6** | none. Deliberately not re-run: D-C6 is byte-identical at four lengths on both models |
| 3 prefix / chunked | as §A3: **PASS** prefix-vs-uncached, prefix-then-plain, mixed batch, each at **one** process; **BLOCKED** on chunked | as §A3: **PASS** all four claims at two processes each | none. The third-process tail was queued and did not run |
| 4 device sampling | still **BLOCKED** in production shape by D-C5 then D-C8; the Llama half of the new measurements did not run | **the area changed**: three of its four brief-named claims now have device measurements for the first time, all from behind D-C5 and D-C8 | see below |
| 5 long context | as §A3: **PASS** 4K/32K/128K | as §A3: **PASS** 4K/32K/128K | none |
| repeat & cleanup | as §A3: repeated requests **FAIL 3/3** (L1 address clash, deterministic); **two model constructions in one process: still never run** | as §A3: repeated requests **PASS 3/3**; two models in one process **FAIL** (D-C7, one observation) | none — `a4_l_two_models`, the one brief-named claim in this row with zero runs, was position 5 in revision 3's queue and did not run |

### Area 4, in detail, because this is where attempt 4 moved

| Brief claim | Before attempt 4 | After |
| --- | --- | --- |
| greedy matches the host argmax exactly | never measured on either model; `TT_FATAL` at D-C5 | **measured, and it does not**: 7/32 slots on Qwen, twice, byte-identically — **but the disagreement is D-C9, a readback defect, not the sampler**. Not a verdict on the claim |
| padded-vocabulary entries can never be sampled | never measured; `a4_q_padded_greedy` re-confirmed the D-C5 abort at 423s | **PASS for the eight users the readback surfaces.** `padded ids sampled in slots []` under **six** policies (greedy, T=0.02, T=2.0, two seeded passes, per-slot heterogeneous), vocab 151936, padded width 19200/device, two fresh processes. **Read this with D-C9 in hand**: the composed vector has 32 entries but they are one mesh column's eight users repeated four times, so the guarantee is measured for **8 distinct users**, not 32. It is the first time it has been measured for any. Llama: not run |
| deterministic seeded requests stay slot-stable (same seed, same slot, same token) | never measured | **measured and PASSING, for the same eight users**: `the same seed in the same slot repeated in 32/32 slots`, two fresh processes — but 24 of those 32 entries are copies, so the claim is established for **8 distinct users** and trivially for their duplicates. The second sense — moving a request to another slot — remains D-C2, a product decision |
| per-slot heterogeneous top-k / top-p / temperature | `TT_FATAL` at D-C5 | ran to completion; its assertion is subsumed by the greedy one and is therefore also D-C9-confounded |
| D4's reciprocal temperature, verified on device | never measured | **still not cleanly measured.** `T=0.02` agreed with the host argmax in 7/32 and `T=2.0` in 6/32 — the same 7/32 the greedy case shows, which is the readback signature, not a temperature signature. `a4_q_temperature` and `a4_l_temperature`, the focused cases, have still never run |
| `top_k > 32` | prohibited by the brief | **respected.** Maximum `top_k` anywhere in either step-7 file, including the three cases attempt 4 added: **32**. Nothing widens `GalaxySamplingPolicy`'s contract |
