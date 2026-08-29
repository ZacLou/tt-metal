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
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# Aliased: the local `attn_res` in this module is the TtAttnRes instance, and the collision is
# silent until the reference is called and LightweightModule.__call__ looks for a `forward`.
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import attn_res as attn_res_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import fold_query
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import (
    TRACE_1M,
    TRACE_100K,
    load_checkpoint_tensors,
    resolve_checkpoint,
    resolve_trace,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import load_layer_state_dict, load_routed_expert_weights
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

SP_AXIS, TP_AXIS = 0, 1
SEQ_LEN = 5120

# Per-layer, against the model itself. The package's chunked per-layer bar is 0.88 at depth 61-78;
# at depths 1-2 the accumulated error should be far smaller, so this starts strict and the ladder's
# deeper rungs will say where it has to relax.
LAYER_PCC = 0.99

# 1 and 2 are cheap-ish; 5 is the first rung with a full-attention layer (layer 3) and so the first
# that needs a KV cache at all. 12 and 24 follow the same shape and are gated on a built TTNN weight
# cache rather than a per-run conversion.
DEPTHS = [1, 2, 5, 12, 24]

# The 100k trace instruments the inside of a layer — kda_*, moe_io, mla_io — but only records
# decoder_output for layers 0..4, so it can only score depths up to 5. The 1M trace records
# decoder_output for layers 0..24 and the 24 MLA layers' KV, and nothing else; it is the only oracle
# for the deeper rungs, and the inner taps below fall silent there because `trace.has` says so.
DEEP_TRACE_FROM = 12

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
    trace = resolve_trace(TRACE_1M if num_layers >= DEEP_TRACE_FROM else TRACE_100K)
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
        max_seq_len=SEQ_LEN,
    )

    # One KV slot per FULL-ATTENTION layer, not per layer. Depths 1 and 2 hold none — layers 0 and 1
    # are both KDA — so there is nothing to allocate and `kvpe_cache=None` is the honest argument.
    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=SEQ_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
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
    inner = {}

    def tap(local_idx, hidden):
        per_layer[local_idx] = _compose(mesh_device, hidden)

    # Wrap each layer's two norms so their outputs are recorded without adding debug plumbing to the
    # block. The FFN norm's output is the model's own `moe_input_layer_i`, and the attention norm's
    # is `kda_input_layer_i` where the trace records it — so a divergence can be placed on one side
    # of the layer or the other instead of only at its output.
    def _record(layer, name, fn):
        def wrapped(x):
            out = fn(x)
            inner[(layer, name)] = _compose(mesh_device, out)
            return out

        return wrapped

    for local_idx, layer in enumerate(model.layers):
        layer.attn_norm = _record(local_idx, "attn_norm", layer.attn_norm)
        if not layer.kv_only:
            layer.ffn_norm = _record(local_idx, "ffn_norm", layer.ffn_norm)

        # And the attention output itself. The trace records it directly only for layer 0
        # (`kda_output_layer_0`), but for any later layer it is derivable from the schedule:
        # `out_i = out_{i-1} + attn_i + mlp_i` with no seal, so
        # `attn_i = decoder_output_i - decoder_output_{i-1} - moe_output_i`.
        attention = layer.attention
        inner_fn = attention.forward

        def _attn(normed, ctx, _idx=local_idx, _fn=inner_fn):
            out = _fn(normed, ctx)
            inner[(_idx, "attn_out")] = _compose(mesh_device, out)
            return out

        attention.forward = _attn

    try:
        # No rope tensors: K3 is NoPE, so `ttMLA` binds `_apply_rope_none` at construction and the
        # rope dict is only ever indexed inside the two rotating paths. Passing None is correct
        # rather than lazy.
        model.forward(tokens_tt, kvpe_cache=kvpe, layer_tap=tap)
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    # Report the inner taps first: when a layer's output is wrong, these say which half.
    for local_idx in range(num_layers):
        got = inner.get((local_idx, "attn_norm"))
        if trace.has("kda", f"kda_input_layer_{local_idx}"):
            want = trace.rows("kda", f"kda_input_layer_{local_idx}", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer {local_idx} attn_norm vs kda_input: {comp_pcc(want, got, 0.99)[1]}")
        elif got is not None and local_idx == 1:
            # The trace records kda_input only for layer 0, but layer 1's is derivable and is the
            # single most diagnostic number in this test: it is the first PRE-ATTENTION AttnRes read
            # the walk issues (layer 0's is skipped, nothing being sealed yet), and the first read
            # where the sealed candidate carries real weight — 27% of the softmax mass against the
            # 4% layer 0's post-read gave it. So it separates "the read is wrong" from "the
            # recurrence is wrong" in one comparison.
            #     read_1   = attn_res(running_sum=out_0, block_residual=[embed], q_pre[1])
            #     kda_in_1 = input_layernorm_1(read_1)
            names = [
                f"{root}layers.1.{k}"
                for k in ("self_attention_res_norm.weight", "self_attention_res_proj.weight", "input_layernorm.weight")
            ]
            w = {k: v.float() for k, v in load_checkpoint_tensors(checkpoint, names).items()}
            read1 = attn_res_reference(
                trace.decoder_output(0, 0, SEQ_LEN),
                trace.decoder_input(0, SEQ_LEN).unsqueeze(1),
                fold_query(w[names[0]], w[names[1]]),
                eps=KimiK3Config.RMS_NORM_EPS,
            )
            want = read1 * torch.rsqrt(read1.pow(2).mean(-1, keepdim=True) + KimiK3Config.RMS_NORM_EPS) * w[names[2]]
            logger.info(f"  L{num_layers} layer 1 attn_norm vs DERIVED kda_input: {comp_pcc(want, got, 0.99)[1]}")
        got = inner.get((local_idx, "attn_out"))
        if got is not None and local_idx == 0 and trace.has("kda", "kda_output_layer_0"):
            want = trace.rows("kda", "kda_output_layer_0", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer 0 attn_out vs kda_output: {comp_pcc(want, got, 0.99)[1]}")
        elif (
            got is not None
            and local_idx > 0
            and local_idx % KimiK3Config.ATTN_RES_BLOCK_SIZE
            and trace.has("moe_io", f"moe_output_layer_{local_idx}")
        ):
            # `attn_i = out_i - out_{i-1} - moe_i` holds only while the running sum is continuous
            # across the boundary. At a seal layer (`i % 12 == 0`) the stream restarts, so out_i
            # carries none of out_{i-1} and the subtraction is meaningless. Skip those rather than
            # print a number that looks like a failure.
            want = (
                trace.decoder_output(local_idx, 0, SEQ_LEN)
                - trace.decoder_output(local_idx - 1, 0, SEQ_LEN)
                - trace.rows("moe_io", f"moe_output_layer_{local_idx}", 0, SEQ_LEN)
            )
            logger.info(f"  L{num_layers} layer {local_idx} attn_out vs derived attn: {comp_pcc(want, got, 0.99)[1]}")
        if trace.has("moe_io", f"moe_input_layer_{local_idx}"):
            got = inner.get((local_idx, "ffn_norm"))
            want = trace.rows("moe_io", f"moe_input_layer_{local_idx}", 0, SEQ_LEN)
            logger.info(f"  L{num_layers} layer {local_idx} ffn_norm vs moe_input: {comp_pcc(want, got, 0.99)[1]}")

    for local_idx in range(num_layers):
        want = trace.decoder_output(local_idx, 0, SEQ_LEN)
        passed, message = comp_pcc(want, per_layer[local_idx], LAYER_PCC)
        logger.info(f"L{num_layers} layer {local_idx} vs decoder_output_layer_{local_idx}: {message}")
        assert passed, f"layer {local_idx} diverged from the model: {message}"
