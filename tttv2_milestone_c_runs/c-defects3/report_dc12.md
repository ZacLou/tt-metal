
## D-C12, recorded and not fixed — and why it is not this job's gate

Attempt 1 isolated it to one call: `logs/d11_repeat_sample_probe_run{2,3,4}.log`, two arms differing
only by `mesh_device.disable_and_clear_program_cache()`.

```text
[repeat] cache=nocache call 0..3: wrong=0/32  every call
[repeat] cache=cache    call 0: wrong=0/32
[repeat] cache=cache    call 1: wrong=32/32   slot 0: expected 1, got 0
[repeat] cache=cache    call 2: wrong=32/32   slot 0: expected 2, got 0
[repeat] cache=cache    call 3: wrong=32/32   slot 0: expected 3, got 2
```

With the program cache cleared, four consecutive sampling calls on four different inputs are all
correct; with it warm, only the first is. Three fresh processes.

**It is not in the c-defects finish condition** and it is not in the five workstreams: the D-C5/D-C8
gate asks for the five area-4 claims to be *evaluated* on silicon at three fresh processes each, and
they are — thirty runs, every one reaching its assertion. D-C12 is what makes one of those five fail
on Qwen (`a seeded slot repeats across runs`), and it is a correctness hazard for anything that
decodes more than one token, which is every real use of this stack. It is named here as the highest
open defect in the ledger after the clash, with the bisection already written down: compose the
**selector's** output on each call and see whether it is stale before `Sampling2D` is reached, which
splits a twelve-op chain in two for the price of one extra readback. The unusual operands are
`ttnn.topk(indices_tensor=self._local_indices)`, `ttnn.manual_seed`, and
`ttnn.sampling(output_tensor=tt_out_tok)`, all in
`models/common/modules/sampling/sampling_2d.py::decode_forward`.
