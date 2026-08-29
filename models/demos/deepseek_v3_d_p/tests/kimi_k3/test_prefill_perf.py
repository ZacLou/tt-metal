# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How fast is Kimi-K3 prefill at 1, 5, 12 and 24 layers?

Two numbers per depth, because on this mesh they answer different questions.

**Device kernel time** is the sum over programs of that program's critical path across the 32 chips.
The real-time profiler reports one record per program per chip, and the chips run a program
concurrently, so the program costs what its slowest chip costs; programs then execute in sequence,
so the sum is the time the device spends inside kernels. It excludes the dispatch gaps between
programs, which is exactly what makes it the right number to compare against another model's kernels
and the wrong number to quote as throughput.

**Eager wall-clock** is a synchronized multi-iteration measurement of `forward` as the tests
actually call it, host dispatch included. The gap between the two is the host overhead that trace
capture removes, so reporting both says how much there is to win from tracing rather than leaving it
implicit.

Both are reported per token as well, since a depth comparison in milliseconds says little when the
layer counts differ by 24x.

The weight cache matters here more than anywhere: without it each depth spends over an hour reading
routed experts before the first token moves, and a perf sweep re-reads them at every rung.
"""

import time
from collections import defaultdict
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    PLACEMENTS,
    SEQ_LEN,
    SP_AXIS,
    TP_AXIS,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root, mark_layer_cached
from models.demos.deepseek_v3_d_p.tt.mla.kv_cache import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

DEPTHS = [1, 5, 12, 24]
ITERATIONS = 10


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
def test_prefill_cost(mesh_device, device_params, num_layers):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=SEQ_LEN)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

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
        return TtAttnResResidual(
            TtAttnResWalk(
                attn_res,
                hidden,
                list(attn_res.weights.pre),
                list(attn_res.weights.post),
                attn_res.weights.output,
                num_layers,
            )
        )

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, num_layers, root, cache),
        num_layers=num_layers,
        seq_len=SEQ_LEN,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=SEQ_LEN,
        weight_cache_path=cache,
    )
    for layer_idx in range(num_layers):
        mark_layer_cached(cache, layer_idx)

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
    tokens_tt = prepare_prefill_input_tensor(
        trace.token_ids(SEQ_LEN)[0].tolist(),
        mesh_device,
        tuple(mesh_device.shape)[SP_AXIS],
        False,
        tuple(mesh_device.shape),
        SP_AXIS,
    )

    def once():
        out = model.forward(tokens_tt, kvpe_cache=kvpe)
        if out is not None:
            ttnn.deallocate(out)

    try:
        for _ in range(3):  # warm the program cache; the first pass compiles
            once()
        ttnn.synchronize_device(mesh_device)

        start = time.perf_counter()
        for _ in range(ITERATIONS):
            once()
        ttnn.synchronize_device(mesh_device)
        eager_ms = (time.perf_counter() - start) / ITERATIONS * 1e3

        device_ms = None
        try:
            _, records = profile_realtime_program(mesh_device, once, collect_all=True)
            # One record per (program, chip). The chips run a program concurrently, so the program
            # costs its slowest chip; programs then run in sequence, so the sum is device kernel time.
            critical_path = defaultdict(float)
            for record in records:
                critical_path[record["runtime_id"]] = max(critical_path[record["runtime_id"]], record["duration_ns"])
            device_ms = sum(critical_path.values()) / 1e6
            programs = len(critical_path)
        except RuntimeError as error:
            # Dropping records makes the set partial, and a partial sum under-reports. Say so rather
            # than quote a number that looks like a measurement.
            logger.warning(f"  L{num_layers}: device kernel time unavailable — {error}")
            programs = 0
    finally:
        if model.kda_states is not None:
            model.kda_states.deallocate()

    logger.info(
        f"L{num_layers:2d} @ {SEQ_LEN} tokens, 8x4: eager {eager_ms:8.2f} ms "
        f"({SEQ_LEN / eager_ms * 1e3:8.0f} tok/s, {eager_ms / num_layers:6.2f} ms/layer)"
    )
    if device_ms is not None:
        logger.info(
            f"        device kernel {device_ms:8.2f} ms over {programs} programs "
            f"({SEQ_LEN / device_ms * 1e3:8.0f} tok/s, {device_ms / num_layers:6.2f} ms/layer) "
            f"-- host overhead {eager_ms - device_ms:.2f} ms ({(1 - device_ms / eager_ms) * 100:.0f}%)"
        )
