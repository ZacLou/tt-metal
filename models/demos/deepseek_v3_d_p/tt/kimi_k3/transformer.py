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

from typing import Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_state import KdaStateCache
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding


class TtKimiK3Transformer(LightweightModule):
    """A slice of Kimi-K3's layers, plus the tail when this rank holds it."""

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
        **block_kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.schedule = KimiK3LayerSchedule.build(model_cfg, first_layer_idx, num_layers)
        self.first_layer_idx = first_layer_idx
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.kv_only_last_layer = kv_only_last_layer
        self._residual_factory = residual_factory or PlainResidualStream

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

    def reset_streams(self, slot: int = 0) -> None:
        """Zero the KDA carries for a new request. Call outside any captured region."""
        if self.kda_states is not None:
            self.kda_states.reset(slot)

    def forward(
        self,
        token_ids,
        *,
        rope_tensors=None,
        kvpe_cache=None,
        cache_user_id: int = 0,
        actual_start: Optional[int] = None,
        actual_end: Optional[int] = None,
        actual_isl: Optional[int] = None,
        metadata: Optional[tuple] = None,
        padding_side: str = "right",
        d2h_service=None,
        record_dev=None,
        on_layer_complete=None,
        layer_tap: Optional[Callable] = None,
    ):
        """Run this rank's layers. Returns the post-norm hidden state, or the raw one mid-pipeline.

        `layer_tap(local_idx, hidden)` fires after each layer with the LIVE residual — which under
        AttnRes is the running sum, and is exactly what the vLLM traces record as
        `decoder_output_layer_i` (pinned by `tests/kimi_k3/test_golden_contract.py`). That makes the
        per-layer PCC curve a tap rather than a separate forward.
        """
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
