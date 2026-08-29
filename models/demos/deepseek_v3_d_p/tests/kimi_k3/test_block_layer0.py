# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's layer 0 on device: KDA attention, a dense SiTU FFN, and the two norms.

Layer 0 is the cheapest complete K3 layer and the only one that needs no MoE: `first_k_dense_replace`
is 1, so it runs the checkpoint's dense MLP, and it is a KDA layer, so it writes no KV. That makes it
the right first end-to-end gate — everything K3-specific about a block except the MoE, with none of
the 59 GB of routed experts a MoE layer would drag in.

**The residual is deliberately the plain one here.** `PlainResidualStream` reproduces
`TtPrefillBlock`'s `ttnn.add` semantics exactly, so this test measures the KDA bridge, the norms and
the SiTU FFN against a torch reference that does the same, and nothing else. AttnRes is scored
separately, against the golden trace, in `test_golden_contract.py` (host) and the `attn_res/model/`
suite (device). Bisecting the two is the entire reason `PlainResidualStream` exists: a failure here
is KDA, a norm, or the FFN, and cannot be AttnRes.

The input is not random. It is `decoder_input_layer_0` from the vLLM trace — the real embedding of a
real prompt — so the activation statistics the KDA recurrence and the SiTU caps see are the ones the
model actually produces, which random normals do not reproduce (SiTU's tanh caps are the reason
that matters).
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict

SP_AXIS, TP_AXIS = 0, 1

# 5120 is not a round number picked for convenience: TILE_SIZE(32) x KDA_SUMMARY_GROUP_CHUNKS(20) x
# SP(8) is the SHORTEST sequence K3's KDA recurrence accepts on an 8x4 mesh, and it is also exactly
# the prefill chunk size. The two coincide, so a chunk is the minimum unit of work.
SEQ_LEN = 5120

# bf16 activations through a 5120-step recurrence and two matmul stacks. The MLA output bar in this
# package is 0.98 and the teacher-forced block bar is 0.98; KDA's own device tests hold 0.999 at
# small shapes, so this sits between them and will be tightened once the first number is in.
BLOCK_PCC = 0.98

PLACEMENTS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(l1_small_size=1152),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    )
]


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    scale = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS)
    return hidden * scale * weight


def _torch_layer0(hidden: torch.Tensor, state_dict: dict) -> torch.Tensor:
    """Layer 0 with a PLAIN residual, so it matches what `PlainResidualStream` does on device."""
    hf_config = KimiLinearConfig(**{k: v for k, v in vars(kimi_k3_hf_config(max_seq=SEQ_LEN)).items()})
    mlp = KimiMLP(hf_config).to(torch.bfloat16).eval()
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(state_dict["ffn_weights"]["gate_proj"])
        mlp.up_proj.weight.copy_(state_dict["ffn_weights"]["up_proj"])
        mlp.down_proj.weight.copy_(state_dict["ffn_weights"]["down_proj"])

    attn_in = _rms_norm(hidden.float(), state_dict["attn_norm_weight"].float())
    attn_out, _ = kda_forward_reference(attn_in.unsqueeze(0), state_dict["kda_weights"], kimi_k3_kda_config())
    residual = hidden.float() + attn_out.squeeze(0).float()

    ffn_in = _rms_norm(residual, state_dict["ffn_norm_weight"].float())
    with torch.no_grad():
        ffn_out = mlp(ffn_in.to(torch.bfloat16)).float()
    return residual + ffn_out


def _shard(mesh_device, hidden: torch.Tensor) -> ttnn.Tensor:
    """`[1, 1, T, d]` with the sequence on SP and the embedding on TP — the block's own layout."""
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.from_torch(
        hidden.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
def test_layer0_plain_residual_matches_torch(mesh_device, device_params, tmp_path):
    """The K3 block's KDA + dense-FFN path, with AttnRes deliberately not in the picture."""
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("no Kimi-K3 checkpoint on this host; set KIMI_K3_HF_MODEL")
    trace = resolve_trace(TRACE_100K)
    if trace is None:
        pytest.skip("no Kimi-K3 100k golden trace on this host; set KIMI_K3_GOLDEN_TRACE")

    state_dict = load_layer_state_dict(Path(checkpoint), 0)
    hidden = trace.decoder_input(0, SEQ_LEN)

    schedule = KimiK3LayerSchedule.build(KimiK3Config, 0, 1)
    assert not schedule.local_is_mla(0), "layer 0 must be a KDA layer for this test to mean anything"

    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    states = None
    try:
        attention = build_attention(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            schedule=schedule,
            seq_len=SEQ_LEN,
            state_cache=None,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )
        # The cache needs the built layer, and the layer needs no cache to be built — so the two are
        # wired after construction rather than in one call.
        states = KdaStateCache({0: attention._kda})
        attention._states = states

        block = TtKimiK3Block(
            mesh_device,
            config,
            KimiK3Config,
            state_dict,
            layer_idx=0,
            local_idx=0,
            attention=attention,
            seq_len=SEQ_LEN,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )

        residual = PlainResidualStream(_shard(mesh_device, hidden))
        block.forward(residual, K3AttnContext())
        got = ttnn.to_torch(
            residual.finish(),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 2), mesh_shape=tuple(mesh_device.shape)),
        ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]
    finally:
        if states is not None:
            states.deallocate()

    want = _torch_layer0(hidden, state_dict)
    passed, message = comp_pcc(want, got, BLOCK_PCC)
    logger.info(f"K3 layer 0 (KDA + dense SiTU FFN), T={SEQ_LEN}: {message}")
    assert passed, f"K3 layer 0 block != torch reference: {message}"
