# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance gate for multi-turn resume on KDA (#54962).

Multi-turn prefill resumes at the KV offset the previous turn ended on. That offset decides which
chip holds the chunk's first token, and both KDA sequence-parallel combinators
(`convolution_halo`, `_distributed_affine_prefix`) compose their carries walking chips in device
index order. When the two disagree, KDA integrates the tokens in the wrong order.

The test feeds the SAME 5120 tokens for every offset. Only the chip layout changes, taken from
`rotated_chip_positions`, which is what the MLA KV writer requires once a chunk starts mid-slab.
The reference is those tokens in true sequence order. So any PCC drop here is ordering and nothing
else: same weights, same maths, same tokens, same starting carry.

Three offsets, one per row of the ladder:

    0      chunk-aligned    chip order 0..7            control, must pass today
    640    block-aligned    chips rotated, no split    needs sequence-order composition
    2592   mid-block        chips rotated, one split   needs split-chip composition

The two non-zero offsets are `xfail(strict=True)`, so when a fix lands they XPASS and the marker
has to be removed deliberately. Run it with:

    pytest models/demos/deepseek_v3_d_p/tests/kda/model/test_multiturn_resume.py -k fabric2d -v

Passing all three rows without the xfail markers is what "KDA supports multi-turn" means.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KDA_PLACEMENTS,
    random_weights,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
    sp_sequence,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device, device_params", KDA_PLACEMENTS, indirect=True),
]

SP_AXIS, TP_AXIS = 0, 1
PCC = 0.999

OFFSETS = [
    pytest.param(0, id="offset0_chunk_aligned"),
    pytest.param(
        640,
        marks=pytest.mark.xfail(strict=True, reason="#54962: chips rotated, composed in device order"),
        id="offset640_block_aligned",
    ),
    pytest.param(
        2592,
        marks=pytest.mark.xfail(strict=True, reason="#54962: split chip, two disjoint ranges on one chip"),
        id="offset2592_mid_block",
    ),
]


def _config() -> KDAConfig:
    return KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )


def _rotate(window: torch.Tensor, offset: int, sp: int, chunk_local: int) -> torch.Tensor:
    """Lay the window out the way a chunk resuming at `offset` reaches the chips.

    `rotated_chip_positions(offset)[c][r]` is the global position chip `c`'s row `r` carries. The
    mesh mapper splits the sequence axis evenly, so chip `c` receives rows
    `[c*chunk_local, (c+1)*chunk_local)`; writing each chip's positions into that band puts every
    token on the chip that would really hold it.
    """
    positions = rotated_chip_positions(offset, sp, chunk_local)
    index = torch.tensor([p - offset for c in range(sp) for p in positions[c]], dtype=torch.long)
    return window[:, index, :]


def _unrotate(rotated: torch.Tensor, offset: int, sp: int, chunk_local: int) -> torch.Tensor:
    """Inverse of `_rotate`, so device output can be compared against a sequence-order reference."""
    positions = rotated_chip_positions(offset, sp, chunk_local)
    out = torch.empty_like(rotated)
    row = 0
    for c in range(sp):
        for p in positions[c]:
            out[:, p - offset, :] = rotated[:, row, :]
            row += 1
    return out


def _pcc(want: torch.Tensor, got: torch.Tensor) -> float:
    a, b = want.flatten().double(), got.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.parametrize("offset", OFFSETS)
def test_kda_resumes_at_offset(mesh_device: ttnn.MeshDevice, device_params, offset: int) -> None:
    config = _config()
    weights = random_weights(config)
    sp = tuple(mesh_device.shape)[SP_AXIS]
    sequence = sp_sequence(mesh_device, SP_AXIS)
    chunk_local = sequence // sp

    window = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(4962)).to(
        torch.bfloat16
    )
    expected_output, expected_state = kda_forward_reference(window, weights, config)

    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(summary_group_chunks=8),
            gated_rms_output_dtype=ttnn.bfloat16,
            output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
    )

    mesh_dims = [None, None]
    mesh_dims[SP_AXIS] = 1
    hidden_tt = ttnn.from_torch(
        _rotate(window, offset, sp, chunk_local),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=tuple(mesh_device.shape)),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, state = layer.forward(hidden_tt, layer.allocate_state(batch_size=1))

    actual_output = _unrotate(
        reconstruct_sp_tp_tensor(output_tt, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1),
        offset,
        sp,
        chunk_local,
    )
    actual_carry = reconstruct_state_at_sp_rank(state.recurrent, mesh_device, SP_AXIS, TP_AXIS, sp - 1)

    output_pcc = _pcc(expected_output.float(), actual_output.float())
    carry_pcc = _pcc(expected_state.recurrent.float(), actual_carry.float())
    starts_on = min(range(sp), key=lambda c: min(rotated_chip_positions(offset, sp, chunk_local)[c]))
    logger.info(
        f"resume offset {offset} (% {chunk_local} = {offset % chunk_local}, starts on chip {starts_on}): "
        f"output pcc {output_pcc:.6f}  carry pcc {carry_pcc:.6f}  (bar {PCC})"
    )

    assert output_pcc >= PCC, f"offset {offset}: output pcc {output_pcc:.6f} < {PCC}"
    assert carry_pcc >= PCC, f"offset {offset}: carry pcc {carry_pcc:.6f} < {PCC}"
