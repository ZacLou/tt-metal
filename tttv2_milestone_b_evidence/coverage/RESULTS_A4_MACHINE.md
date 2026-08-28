# `mb-coverage` attempt 4 — machine-written run record

One row per finished log in `logs4/`, appended by `cov_transcribe4.sh`
as each run ends. **Nothing in this file was typed by a human or by an
agent**: every field is `grep`-ed out of the log named in the row. It exists
so that a night that outlives its agent still writes itself down.

| log | commit | wall clock | pytest summary | first assertion | marker lines |
| --- | --- | --- | --- | --- | --- |
| `a4_l_dc8` | `54b9fadb3ff` | in 7.40s | 1 error in 7.40s | RuntimeError: Read 0xffffffff over PCIe ID 21: the board should be reset. | — |
| `a4_q_dc8` | `aff4e95dbf6` | in 185.57s (0:03:05) | 1 failed, 2 warnings in 185.57s | AssertionError: device greedy disagreed with the host argmax in slots [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,  | [dc8] greedy: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTE [dc8] T=0.02: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTE [dc8] T=2.0: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTER [dc8] seeded pass 1: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayo  |
| `a4_q_dc8_run2` | `aff4e95dbf6` | in 160.59s (0:02:40) | 1 failed, 2 warnings in 160.59s | AssertionError: device greedy disagreed with the host argmax in slots [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,  | [dc8] greedy: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTE [dc8] T=0.02: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTE [dc8] T=2.0: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayout.INTER [dc8] seeded pass 1: decode logits were TensorMemoryLayout.WIDTH_SHARDED, width 19200; relocated to TensorMemoryLayo  |
| `a4_q_dc8_run3` | `54b9fadb3ff` | in 7.41s | 1 error in 7.41s | RuntimeError: Read 0xffffffff over PCIe ID 21: the board should be reset. | — |
| `a4_q_dc9_bisect` | `54b9fadb3ff` | in 7.51s | 1 error in 7.51s | RuntimeError: Read 0xffffffff over PCIe ID 21: the board should be reset. | — |
| `a4_q_padded_greedy` | `110ba1f0658` | in 423.12s (0:07:03) | 1 failed, 2 warnings in 423.12s | TT_FATAL: MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED, got: TensorMemoryLayout::WIDTH_SHARDED (assert.hpp:104) | — |
| `a4_selector` | `aff4e95dbf6` | in 31.32s | 2 passed in 31.32s | — | — |
| `a4_selector_run2` | `aff4e95dbf6` | in 8.66s | 2 passed in 8.66s | — | — |
| `a4_selector_run3` | `aff4e95dbf6` | in 8.89s | 2 passed in 8.89s | — | — |
