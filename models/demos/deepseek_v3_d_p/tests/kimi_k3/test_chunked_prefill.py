# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 prefilled in 5120-token chunks, carried across all 102400 tokens of the 100k trace.

The depth ladder proves one chunk. This proves the thing a chunk cannot: that the KDA recurrent
carry survives the boundary between chunks and still equals the model's own, twenty times in a row.
That is the part of Kimi-K3 with genuine cross-chunk state — AttnRes has none, since every one of its
reductions is over the hidden dimension and the token axis is a free batch axis, so each chunk opens
a fresh walk while the KDA carries advance.

5120 is not a chosen number. `ttnn.TILE_SIZE(32) * KDA_SUMMARY_GROUP_CHUNKS(20) * SP(8)` is both the
chunk size and the shortest sequence the KDA recurrence accepts on this mesh, so it is the only
chunk size available here.

Two oracles, and the second is the one that matters:

  * each chunk's residual stream against `decoder_output_layer_0` over that window;
  * the KDA recurrent carry at every boundary against `kda_recurrent_state_layer_0`, which the trace
    snapshots every 640 tokens — so a 5120-token boundary is snapshot row `8k - 1`. A run that
    silently restarted its recurrence each chunk would still pass the first oracle on chunk 0 and
    drift afterwards; this one fails immediately at the first boundary.

The golden stores the carry as `[heads, v_dim, k_dim]` and the layer produces `[heads, k_dim,
v_dim]`. Both are `[96, 128, 128]`, so omitting the transpose reports PCC ~0.01 and looks exactly
like a broken recurrence. `test_kda_golden.py` pins that convention; this file just obeys it.

Twenty chunks also covers the standing memory gate: device DRAM is sampled after every chunk and
must not grow, which is what would catch the KDA carries or the walk's per-block batches leaking.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    PLACEMENTS,
    SP_AXIS,
    TP_AXIS,
    _compose,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root, mark_layer_cached
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor

CHUNK = 5120
NUM_CHUNKS = 20
NUM_LAYERS = 1
SNAPSHOT_STRIDE = 640

# Layer 0 one-shot scores 0.9998628 on the ladder. Chunking must not cost anything structural, so
# the bar sits just under that rather than at the package's usual 0.98.
OUTPUT_PCC = 0.999
# The carry is a 5120-step bf16 recurrence per chunk compounded across chunks, and it is compared
# against a snapshot the model wrote in fp32; the ladder's own KDA output sits at 0.9999.
CARRY_PCC = 0.99


def _compose_carry(mesh_device, state):
    """The global `[heads, k_dim, v_dim]` carry from its TP shards.

    The carry is TP-sharded on heads and SP-replicated, so one SP row holds the whole thing and the
    other seven are duplicates. Take row 0 and concatenate the TP shards on the head axis.
    """
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(state)]
    rows, columns = tuple(mesh_device.shape)
    tp_size = (rows, columns)[TP_AXIS]
    head_shards = []
    for tp_rank in range(tp_size):
        row, column = (0, tp_rank) if SP_AXIS == 0 else (tp_rank, 0)
        head_shards.append(
            shards[row * columns + column].reshape(-1, KimiK3Config.KDA_HEAD_DIM, KimiK3Config.KDA_HEAD_DIM)
        )
    return torch.cat(head_shards, dim=0).float()


def _dram_bytes(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_chunked_prefill_carries_kda_state(mesh_device, device_params):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=CHUNK)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, NUM_LAYERS, root),
            None,
            num_layers=NUM_LAYERS,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden):
        # A fresh walk per chunk. AttnRes state is per token — every reduction is over the hidden
        # dimension — so there is nothing to carry, and `finish()` frees the stream each time.
        return TtAttnResResidual(
            TtAttnResWalk(
                attn_res,
                hidden,
                list(attn_res.weights.pre),
                list(attn_res.weights.post),
                attn_res.weights.output,
                NUM_LAYERS,
            )
        )

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, NUM_LAYERS, root, cache),
        num_layers=NUM_LAYERS,
        seq_len=CHUNK,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=CHUNK,
        is_chunked=True,
        weight_cache_path=cache,
    )
    for layer_idx in range(NUM_LAYERS):
        mark_layer_cached(cache, layer_idx)

    golden_carry = trace.rows("kda", "kda_recurrent_state_layer_0")
    footprints = []
    failures = []

    for chunk in range(NUM_CHUNKS):
        start = chunk * CHUNK
        if chunk == 0:
            # Only at the head of a request: a carry summarizes the prefix behind it, so zeroing it
            # between chunks is precisely the bug this test exists to catch.
            model.reset_streams()

        tokens_tt = prepare_prefill_input_tensor(
            trace.token_ids(CHUNK, start)[0].tolist(),
            mesh_device,
            tuple(mesh_device.shape)[SP_AXIS],
            False,
            tuple(mesh_device.shape),
            SP_AXIS,
        )
        # The comparison is against the LIVE running sum, which is what `decoder_output_layer_i`
        # records — not `forward`'s return, which has passed through the final norm. `layer_tap` is
        # the same seam the depth ladder uses.
        captured = {}
        out = model.forward(
            tokens_tt,
            kvpe_cache=None,
            actual_start=start,
            layer_tap=lambda idx, h: captured.__setitem__(idx, _compose(mesh_device, h)),
        )
        if out is not None:
            ttnn.deallocate(out)
        got = captured[NUM_LAYERS - 1]

        want = trace.decoder_output(0, start, start + CHUNK)
        output_pcc = float(str(comp_pcc(want, got, OUTPUT_PCC)[1]).split()[-1])

        # Snapshots land every 640 tokens, so the boundary after chunk k is row 8(k+1) - 1. The
        # golden's [heads, v_dim, k_dim] needs transposing into the layer's [heads, k_dim, v_dim].
        row = (start + CHUNK) // SNAPSHOT_STRIDE - 1
        want_carry = golden_carry[row].transpose(-1, -2)
        got_carry = _compose_carry(mesh_device, model.kda_states.read(0, 0).recurrent)
        carry_pcc = float(str(comp_pcc(want_carry, got_carry, CARRY_PCC)[1]).split()[-1])

        footprints.append(_dram_bytes(mesh_device))
        logger.info(
            f"  chunk {chunk:2d} [{start:6d}:{start + CHUNK:6d}]  output {output_pcc:.6f}  "
            f"carry(row {row:3d}) {carry_pcc:.6f}  dram {footprints[-1] / 2**20:8.1f} MiB"
        )
        if output_pcc < OUTPUT_PCC:
            failures.append(f"chunk {chunk} output {output_pcc}")
        if carry_pcc < CARRY_PCC:
            failures.append(f"chunk {chunk} carry {carry_pcc}")

    # The carries and the walk's per-block batches are the two new allocation surfaces; both are
    # supposed to be steady-state after the first chunk warms the pools.
    steady = footprints[1:]
    growth = max(steady) - min(steady)
    logger.info(f"  DRAM after chunk 1: {min(steady) / 2**20:.1f} MiB, drift over 19 chunks: {growth / 2**20:.1f} MiB")
    assert growth == 0, f"device DRAM grew {growth} bytes across chunks 1..{NUM_CHUNKS - 1}: {footprints}"
    assert not failures, "chunked prefill diverged from the model: " + "; ".join(failures)
