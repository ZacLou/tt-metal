# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Kimi-K3 layer stack.

Owns the three things a hybrid AttnRes model needs and `TtPrefillTransformer` has no place for: the
schedule that says which layers are MLA, the KDA carries, and the residual the blocks talk to. Every
leaf below it is the shared one — `TtParallelEmbedding`, `TtDistributedRmsNorm`, and the blocks' own
`ttMLA` / `TtMoe` / `TtFfn`. The LM head is deliberately absent so far: the bring-up gates compare
per-layer residuals and the KV cache against the golden traces, and logits only enter at the very
end.

The residual is injected rather than chosen here. `PlainResidualStream` is the bring-up arm: it
reproduces `ttnn.add` exactly, so a K3 run can be scored layer by layer with AttnRes out of the
picture, and a failure is then a KDA, norm, MoE or MLA failure and nothing else. The AttnRes arm
swaps in a walk. Nothing in the layer loop changes between them, which is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import CHECKPOINT_PREFIX, load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import mark_layer_cached
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding


class TtKimiK3Transformer(LightweightModule):
    """A slice of Kimi-K3's layers, plus the tail when this rank holds it."""

    @staticmethod
    def check_cache_complete(
        cache_path,
        num_layers: int,
        experts_per_chip: int = 8,
        first_k_dense: int = 1,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        kv_only_last_layer: bool = False,
        model_cfg: type | None = None,
        routed_expert_weights_dtype=None,
    ) -> bool:
        """Whether this rank's whole slice is on disk, in the signature the runtime calls.

        Deliberately conservative: it answers from the per-layer completion markers the cache
        generator writes, and a marker goes down only after that layer was built end to end from real
        weights. Composing each component's own `check_cache_complete` is the tempting alternative
        and the wrong one — those are easy to get subtly wrong, and being wrong here is not a failure
        but a silently wrong model, because `ttnn.as_tensor` writes whatever tensor it is handed when
        a file is absent and `TtDistributedRmsNorm` hands it `torch.empty` (#54841).
        """
        from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import layer_is_cached

        if cache_path is None:
            return False
        return all(layer_is_cached(Path(cache_path), first_layer_idx + i) for i in range(num_layers))

    def __init__(
        self,
        mesh_device,
        config,
        model_cfg: type,
        state_dict: dict,
        num_layers: int,
        seq_len: int,
        *,
        first_layer_idx: int = 0,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        kv_only_last_layer: bool = False,
        residual_factory: Optional[Callable] = None,
        num_links: int = 1,
        topology=None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        weight_cache_path=None,
        max_seq_len: Optional[int] = None,
        is_chunked: bool = False,
        slot_num: int = 1,
        num_users: int = 1,
        gate_fallback_mode=None,
        is_balanced: bool = False,
        build_tail: bool = True,
        # Model-level knobs the shared runtime passes to every transformer. Named here rather than
        # left to `**block_kwargs`, which would forward them to `TtKimiK3Block` and raise.
        lm_head_is_column_parallel: bool = False,
        padding_side: str = "right",
        sparse_kv_cache_format=None,
        **block_kwargs,
    ):
        super().__init__()
        # Kimi-K3's MLA cache is dense: `zero_padded_kv_cache` asserts TILE layout, which a sparse
        # kvpe cache (bf16/fp8 ROW_MAJOR, read natively by sparse_sdpa) does not satisfy. Accepting a
        # sparse format silently would produce a cache the pad-zero path cannot touch, so refuse it.
        if sparse_kv_cache_format is not None:
            raise ValueError(
                f"Kimi-K3 uses a dense MLA KV cache; got sparse_kv_cache_format={sparse_kv_cache_format!r}"
            )
        self.lm_head_is_column_parallel = lm_head_is_column_parallel
        self.padding_side = padding_side
        self.mesh_device = mesh_device
        # Resolve once, here, rather than letting `None` reach a collective. The MoE's
        # `reduce_scatter_minimal_async` validates `topology == Ring or Linear` and a None is a
        # TT_FATAL eight frames deep in `all_reduce_async`. Everything below wants the per-axis pair
        # the fabric actually opened, which is what `MLAPrefillAdapter.build_runtime` passes too.
        topology = topology if topology is not None else per_axis_topology()
        self.topology = topology
        self.schedule = KimiK3LayerSchedule.build(model_cfg, first_layer_idx, num_layers)
        self.first_layer_idx = first_layer_idx
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.kv_only_last_layer = kv_only_last_layer
        # AttnRes is the model, not an option. Defaulting to a plain running sum would run, produce
        # 0.447 against the golden where AttnRes gives 0.99985, and issue none of the 2N reads — so it
        # would also under-report performance. `PlainResidualStream` exists for bisecting an AttnRes
        # bug from a KDA/MoE one and must be asked for explicitly.
        self._attn_res = None
        if residual_factory is None:
            self._attn_res = TtAttnRes(
                mesh_device,
                hidden_size=model_cfg.EMB_SIZE,
                eps=model_cfg.RMS_NORM_EPS,
                tp_axis=tp_axis,
                weights=load_attn_res_weights(
                    mesh_device,
                    state_dict.get("attn_res_weights"),
                    weight_cache_path,
                    num_layers=num_layers,
                    tensor_parallel_axis=tp_axis,
                    prefix=state_dict.get("attn_res_prefix", CHECKPOINT_PREFIX),
                ),
            )
            attn_res = self._attn_res

            def residual_factory(hidden, _n=num_layers):
                # A fresh walk per forward: AttnRes state is per token — every reduction is over the
                # hidden dimension and the token axis is a free batch axis — so there is nothing to
                # carry between chunks, unlike the KDA recurrence.
                return TtAttnResResidual(
                    TtAttnResWalk(
                        attn_res,
                        hidden,
                        list(attn_res.weights.pre),
                        list(attn_res.weights.post),
                        attn_res.weights.output,
                        _n,
                    )
                )

        self._residual_factory = residual_factory

        if kv_only_last_layer:
            # Rejects a KDA last layer, which would compute a recurrence and write nothing.
            self.schedule.validate_kv_only_last_layer()

        if is_first_rank:
            self.embed = TtParallelEmbedding(
                mesh_device=mesh_device,
                vocab_size=config.vocab_size,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("embed_weight"),
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                weight_cache_path=weight_cache_path,
            )

        # Every KDA layer's carry, allocated once for the run so a capture can bake in its address.
        # Built after the layers, since it needs the ttKDA instances; the attentions are wired to it
        # as they are created.
        self.kda_states: KdaStateCache | None = None
        kda_layers: dict[int, object] = {}

        self.layers = []
        for local_idx in range(num_layers):
            layer_idx = first_layer_idx + local_idx
            is_last = local_idx == num_layers - 1
            layer_state = state_dict["layers"][local_idx] if state_dict.get("layers") else {}
            attention = build_attention(
                mesh_device,
                config,
                model_cfg,
                layer_state,
                layer_idx=layer_idx,
                schedule=self.schedule,
                seq_len=seq_len,
                state_cache=None,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                num_links=num_links,
                topology=topology,
                weight_cache_path=weight_cache_path,
                max_seq_len=max_seq_len,
                is_chunked=is_chunked,
                slot_num=slot_num,
                kv_only=kv_only_last_layer and is_last,
                is_balanced=is_balanced,
                first_layer_idx=first_layer_idx,
            )
            if not self.schedule.local_is_mla(local_idx):
                kda_layers[layer_idx] = attention.kda

            self.layers.append(
                TtKimiK3Block(
                    mesh_device,
                    config,
                    model_cfg,
                    layer_state,
                    layer_idx=layer_idx,
                    local_idx=local_idx,
                    attention=attention,
                    seq_len=seq_len,
                    num_links=num_links,
                    topology=topology,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    is_balanced=is_balanced,
                    gate_fallback_mode=gate_fallback_mode,
                    weight_cache_path=weight_cache_path,
                    kv_only=kv_only_last_layer and is_last,
                    **block_kwargs,
                )
            )
            # This layer's tensorbins are now all written, so its cache is known complete. Marking
            # here rather than after the whole stack means an interrupted build keeps the layers it
            # finished: a 24-layer run that dies at layer 22 would otherwise leave 22 layers of
            # tensorbins on disk with nothing recording that they are usable, and the next run would
            # rebuild every one of them.
            mark_layer_cached(weight_cache_path, layer_idx)

        if kda_layers:
            self.kda_states = KdaStateCache(kda_layers, num_slots=num_users)
            for layer in self.layers:
                if not layer.attention.writes_kv:
                    layer.attention.bind_state_cache(self.kda_states)

        self.norm = None
        if is_last_rank and build_tail and not kv_only_last_layer:
            self.norm = TtDistributedRmsNorm(
                mesh_device=mesh_device,
                emb_dim=config.hidden_size,
                torch_weight=state_dict.get("norm_weight"),
                epsilon=config.rms_norm_eps,
                cluster_axis=tp_axis,
                num_links=num_links,
                topology=topology[tp_axis] if isinstance(topology, (tuple, list)) else topology,
                weight_cache_path=weight_cache_path,
                cache_name_prefix="norm",
            )

    def set_trace_controller(self, controller):
        """Attach (or clear with None) a SubDeviceTraceController on every layer.

        Required by `TtPrefillRuntime._prepare_trace`, which calls this unconditionally when
        PREFILL_USE_TRACE=1; without it the runner dies with AttributeError during compile().

        Kimi-K3 is trace-eligible for the reason `TtPrefillTransformer.set_trace_controller` gives:
        its attention is KDA or dense MLA, never a sparse/DSA indexer. The KDA carries are already
        address-stable (`KdaStateCache` commits with `ttnn.copy` into persistent buffers), which is
        what a capture requires.
        """
        for layer in self.layers:
            layer.set_trace_controller(controller)

    def release_sub_device_managers(self):
        """Remove every MoE-created overlap sub-device manager before closing the mesh device.
        Leaving them registered at mesh close has been observed to segfault teardown. Idempotent."""
        self.mesh_device.clear_loaded_sub_device_manager()
        for layer in self.layers:
            layer.release_sub_device_managers()

    def reset_streams(self, slot: int = 0) -> None:
        """Zero the KDA carries for a new request. Call outside any captured region."""
        if self.kda_states is not None:
            self.kda_states.reset(slot)

    def forward(
        self,
        token_ids,
        kvpe_cache=None,
        actual_isl: Optional[int] = None,
        return_intermediates: bool = False,
        read_profiler: bool = False,
        temperature: Optional[float] = None,
        d2h_service=None,
        record_dev=None,
        on_layer_complete=None,
        on_layer_hidden: Optional[Callable] = None,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        cache_user_id: int = 0,
        index_kv_cache=None,
        metadata: Optional[tuple] = None,
        *,
        rope_tensors=None,
        padding_side: str = "right",
        layer_tap: Optional[Callable] = None,
    ):
        """Run this rank's layers. Returns the post-norm hidden state, or the raw one mid-pipeline.

        `layer_tap(local_idx, hidden)` fires after each layer with the LIVE residual — which under
        AttnRes is the running sum, and is exactly what the vLLM traces record as
        `decoder_output_layer_i` (pinned by `tests/kimi_k3/test_golden_contract.py`). That makes the
        per-layer PCC curve a tap rather than a separate forward.
        """
        if on_layer_hidden is not None:
            # DFlash's per-layer tap. Under AttnRes a tap is a READ SITE, and a site needs its own
            # folded query in the walk order — so exposing one is a schedule change, not a callback.
            raise NotImplementedError(
                "Kimi-K3 does not support on_layer_hidden: under AttnRes every read needs its own "
                "query and site, so a per-layer tap is a walk-schedule change rather than a hook"
            )
        if index_kv_cache is not None:
            raise ValueError("Kimi-K3 has no DSA indexer; index_kv_cache must be None")
        if return_intermediates:
            raise NotImplementedError("Kimi-K3 does not implement return_intermediates")

        hidden = ttnn.unsqueeze_to_4D(self.embed(token_ids)) if self.is_first_rank else token_ids
        residual = self._residual_factory(hidden)

        for local_idx, layer in enumerate(self.layers):
            ctx = K3AttnContext(
                rope_tensors=rope_tensors,
                kvpe_cache=kvpe_cache,
                cache_layer_idx=self.schedule.kv_slot(local_idx),
                cache_user_id=cache_user_id,
                actual_start=actual_start,
                metadata=metadata,
            )
            layer.forward(
                residual,
                ctx,
                d2h_service=d2h_service,
                record_dev=record_dev,
                on_layer_complete=on_layer_complete,
                actual_end=actual_end,
                actual_isl=actual_isl,
                padding_side=padding_side,
            )
            if layer_tap is not None and not layer.kv_only:
                layer_tap(local_idx, residual.current())

        if self.kv_only_last_layer:
            # Nothing downstream reads the output; the walk's remaining sites go unconsumed.
            residual.discard()
            return None

        hidden = residual.finish()
        return self.norm(hidden) if self.norm is not None else hidden
