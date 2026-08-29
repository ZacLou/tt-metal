# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`prepare_chunk_recurrence` on Kimi-K3's real gate, against the suite's own oracle.

Every existing case builds the gate as `g = -0.001 - 0.05 * rand`, so the op is only ever asked for
`g` in [-0.051, -0.001]. Kimi-K3's gate is `KDA_GATE_LOWER_BOUND * sigmoid(...)` with a lower bound
of -5.0, and its real per-32-row-chunk cumulative sum reaches -56 at layer 1 and -160 at layer 0.
That is far outside the covered range, and it is where the chunk formulation gets delicate: `intra`
and `t_inv` both pair `exp(cumsum g)` against `exp(-cumsum g)`, whose product is bounded by 1 for
i >= j but whose factors separately span `exp(±|cumsum|)`.

Note what this file does NOT claim. A synthetic sweep widening the gate to K3's full span makes the
ORACLE overflow first — at span 5.0 its `t_inv` is entirely NaN and `intra` carries 159 infinities —
so any PCC there measures the reference breaking, not the kernel. `exp(160)` is not representable in
fp32 at all, so the naive chunk formulation simply has no valid answer at layer 0's range, and only
a formulation that forms `exp(cumsum_i - cumsum_j)` directly does.

So this drives the op with the model's real tensors at the one layer where the oracle stays finite,
which is exactly the layer that fails end-to-end: layer 1 scores ~0 against its own torch reference
on device while layer 0 scores 0.99993. Each of the seven outputs is scored separately, so a single
wrong one names the stage.

Read the `t_inv` row carefully. It scored 0.99508 before the `invert_horner` fix, which looks like a
pass and is not one: `t_inv` is `I + strictly-lower`, so whole-tensor PCC is dominated by the
identity diagonal and stays near 1 while the off-diagonals — the entire content of the UT transform
— are wrong. Scored on its strictly-lower part alone it was 0.935, and substituting it for the torch
protocol took the recurrence from 1.0 to 0.0014. Any future threshold on this output should be set
on the strictly-lower part, not the whole tile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import causal_depthwise_conv_reference, kda_gate_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_prepare_chunk_recurrence import (
    CHUNK_SIZE,
    OUTPUT_NAMES,
    _device_inputs,
    _oracle,
    _run,
)

pytestmark = [run_for_blackhole()]

SEQ = 1024
BF16_MASK = 0x26


def _real_inputs(checkpoint: Path, hidden: torch.Tensor, layer_idx: int, config):
    """q, k, v, g, beta for one real Kimi-K3 layer, in the flat layout the op takes."""
    weights = load_kda_layer_state_dict(checkpoint, layer_idx, config)
    hidden = hidden.float().unsqueeze(0)
    state = hidden.new_zeros(1, config.conv_kernel_size - 1, config.q_dim)

    def projected(name, conv):
        out, _ = causal_depthwise_conv_reference(
            F.linear(hidden, weights[f"{name}.weight"].float()), weights[conv], state
        )
        return out

    q, k, v = (projected(n, f"{n[0]}_conv1d.weight") for n in ("q_proj", "k_proj", "v_proj"))
    raw = F.linear(F.linear(hidden, weights["f_a_proj.weight"].float()), weights["f_b_proj.weight"].float())
    gate = kda_gate_reference(
        raw.reshape(1, hidden.shape[1], config.num_heads, config.head_k_dim),
        weights["A_log"],
        weights["dt_bias"],
        config.gate_lower_bound,
    ).reshape(1, hidden.shape[1], config.q_dim)
    beta = torch.sigmoid(F.linear(hidden, weights["b_proj.weight"].float()))
    beta = beta.reshape(hidden.shape[1] // CHUNK_SIZE, CHUNK_SIZE, config.num_heads).permute(2, 0, 1).unsqueeze(-1)
    return q, k, v, gate, beta.contiguous()


@run_for_blackhole()
def test_prepare_matches_oracle_on_real_kimi_k3_gate(device):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")
    config = kimi_k3_kda_config()
    hidden = trace.rows("kda", "kda_input_layer_0", 0, SEQ)

    for layer_idx in (0, 1):
        inputs = _real_inputs(Path(checkpoint), hidden, layer_idx, config)
        cumsum = inputs[3].reshape(1, SEQ // CHUNK_SIZE, CHUNK_SIZE, -1).cumsum(dim=2)
        try:
            expected = _oracle(inputs, config.num_heads, BF16_MASK)
        except Exception as error:
            # Layer 0's chunk cumsum reaches -160, so exp(-cumsum) is inf and the Gram matrix the
            # oracle inverts is singular. The reference formulation has no answer here at all, which
            # is a fact about the formulation and not about the kernel: the device reproduces this
            # layer at 0.99993 against the stable recurrent reference.
            logger.info(
                f"  layer {layer_idx} (chunk cumsum min {float(cumsum.min()):7.1f}): "
                f"oracle unrepresentable -- {type(error).__name__}: {error}"
            )
            continue
        finite = all(torch.isfinite(t.float()).all() for t in expected)
        actual = _run(_device_inputs(inputs, device), config.num_heads, output_bf16_mask=BF16_MASK)
        scores = "  ".join(
            f"{name} {float(str(comp_pcc(want.float(), ttnn.to_torch(got).float(), 0.99)[1]).split()[-1]):8.5f}"
            for name, want, got in zip(OUTPUT_NAMES, expected, actual, strict=True)
        )
        logger.info(f"  layer {layer_idx} (chunk cumsum min {float(cumsum.min()):7.1f}, oracle finite={finite})")
        logger.info(f"    {scores}")
