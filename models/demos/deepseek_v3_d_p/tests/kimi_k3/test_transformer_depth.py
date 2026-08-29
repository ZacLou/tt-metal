# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's first N layers against the model's own per-layer outputs.

The depth ladder. `test_block_layer0*.py` gate one layer in isolation; this runs the stack the way
prefill does — embedding, then N layers threading one AttnRes walk — and scores **every** layer
against `decoder_output_layer_i` from the vLLM trace. The per-layer curve is what says whether error
accumulates or stays flat, which a single end-of-stack number cannot.

The curve comes from a tap, not a second forward: `TtKimiK3Transformer.forward(layer_tap=...)` fires
after each layer with the live residual, and under AttnRes the live residual **is** what the trace
records (pinned host-side in `test_golden_contract.py`).

**Cost is why the ladder starts at 1 and 2.** Layer 0 is dense; every layer after it is LatentMoE
with 896 routed experts at 33 M parameters each — 59 GB of bf16 per layer, read from the 5.5 TB
dequantized store. One MoE layer is affordable; the 5/12/24-layer rungs want a built TTNN cache
rather than a fresh conversion each run.

**Fabric2D**, because `attn_res_gather_softmax` hangs on the wrapped fabrics
(`tests/attn_res/model/harness.py::mesh_topology`).
"""

from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict, load_routed_expert_weights
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120

# Per-layer, against the model itself. The package's chunked per-layer bar is 0.88 at depth 61-78;
# at depths 1-2 the accumulated error should be far smaller, so this starts strict and the ladder's
# deeper rungs will say where it has to relax.
LAYER_PCC = 0.99

DEPTHS = [1, 2]

PLACEMENTS = [
    pytest.param(
        (8, 4),
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 1152},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-8x4",
    )
]


def _model_state_dict(checkpoint: Path, num_layers: int, root: str) -> dict:
    """The transformer's state dict: embedding, final norm, and one entry per layer.

    Routed experts are fetched per layer rather than up front — 59 GB each — so a caller that only
    wants layer 0 never pays for them.
    """
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_tensors

    model = load_tensors(
        checkpoint, {"embed_weight": f"{root}embed_tokens.weight", "norm_weight": f"{root}norm.weight"}
    )
    layers = []
    for layer_idx in range(num_layers):
        layer = load_layer_state_dict(checkpoint, layer_idx)
        if layer_idx >= KimiK3Config.NUM_DENSE_LAYERS:
            logger.info(f"layer {layer_idx}: reading {KimiK3Config.NUM_ROUTED_EXPERTS} routed experts (~59 GB)")
            layer["routed_expert_weights"] = load_routed_expert_weights(
                checkpoint, layer_idx, KimiK3Config.NUM_ROUTED_EXPERTS
            )
        layers.append(layer)
    return {"embed_weight": model["embed_weight"].float(), "norm_weight": model["norm_weight"], "layers": layers}


def _compose(mesh_device, tensor):
    dims = [0, 0]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    ).reshape(-1, KimiK3Config.EMB_SIZE)[:SEQ_LEN]


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
def test_depth_ladder_matches_golden(mesh_device, device_params, num_layers):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    state_dict = _model_state_dict(checkpoint, num_layers, root)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden):
        walk = TtAttnResWalk(
            attn_res,
            hidden,
            list(attn_res.weights.pre),
            list(attn_res.weights.post),
            attn_res.weights.output,
            num_layers,
        )
        return TtAttnResResidual(walk)

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        state_dict,
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )

    # The repo's own placement, not a hand-rolled mapper: tokens shard on the SEQUENCE axis, and
    # `prepare_prefill_input_tensor` is what produces the [sp_factor, 1, isl_per_chip] uint32
    # ROW_MAJOR layout the embedding reads. `is_balanced=False` is chunked prefill's block-cyclic
    # order, which is the order the golden trace's tokens are in too.
    tokens_tt = prepare_prefill_input_tensor(
        trace.token_ids(SEQ_LEN)[0].tolist(),
        mesh_device,
        tuple(mesh_device.shape)[SP_AXIS],
        False,
        tuple(mesh_device.shape),
        SP_AXIS,
    )

    per_layer = {}

    def tap(local_idx, hidden):
        per_layer[local_idx] = _compose(mesh_device, hidden)

    try:
        model.forward(tokens_tt, layer_tap=tap)
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    worst = 1.0
    for local_idx in range(num_layers):
        want = trace.decoder_output(local_idx, 0, SEQ_LEN)
        passed, message = comp_pcc(want, per_layer[local_idx], LAYER_PCC)
        logger.info(f"L{num_layers} layer {local_idx} vs decoder_output_layer_{local_idx}: {message}")
        assert passed, f"layer {local_idx} diverged from the model: {message}"
