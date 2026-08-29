# c-exec-llama device runs, in the order they happened

| when (UTC) | name | node | result | seconds | log |
| --- | --- | --- | --- | --- | --- |
| 23:14:41Z | i1_ref128_l1 | `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py::test_reference_prefill_and_decode[wormhole_b0-128-device_params0-mesh_device0]` | 1 passed, 2 warnings in 117.56s (0:01:57)  | 153 | `logs/i1_ref128_l1.log` |
| 23:17:20Z | i2_exec128_l1 | `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py::test_executor_prefill_matches_reference[wormhole_b0-128-device_params0-mesh_device0]` | 1 passed, 2 warnings in 119.98s (0:01:59)  | 159 | `logs/i2_exec128_l1.log` |
| 23:22:24Z | i3_decode1_l1 | `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py::test_executor_decode_first_token[wormhole_b0-1-device_params0-mesh_device0]` | 1 passed, 2 warnings in 119.54s (0:01:59)  | 156 | `logs/i3_decode1_l1.log` |
| 23:24:56Z | i4_pagedkv_l1 | `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py::test_executor_paged_kv_contract[wormhole_b0-device_params0-mesh_device0]` | 1 failed, 2 warnings in 114.85s (0:01:54)  | 152 | `logs/i4_pagedkv_l1.log` |
| 23:27:26Z | i5_warmup_l1 | `models/common/tests/models/llama33_70b_galaxy/test_executor_wh_galaxy.py::test_executor_warmup_and_program_identity[wormhole_b0-device_params0-mesh_device0]` | 1 failed, 2 warnings in 113.04s (0:01:53)  | 150 | `logs/i5_warmup_l1.log` |
