# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Llama-3.3-70B Galaxy `(8, 4)` execution composition and cleanup root.

This is the Milestone C executor for the reconstructed 2D tensor model. It composes
the common runtime (`models/common/llm_runtime`) exactly as
`models/common/models/llama33_70b/executor.py` composes it for the 1D model, and it
is the resource and cleanup root for one execution lane.

Two things make this executor longer than its 1D sibling, and both of them are
*model-owned adaptation* rather than new mechanics:

**1. The runtime's model contract is 1D-shaped, and the Galaxy graph is not.**
`PrefillRuntime` and `DecodeRuntime` call

```text
model.embed_prefill(tokens)                       model.embed_decode(tokens)
model.prefill_forward(x, rot_mats, user_id=…,     model.decode_forward(x, positions,
                      get_last_token=-1, …)                            rot_mats, page_table=…)
model.post_process_prefill_output(hidden, last)   model.gather_and_untilize_logits(logits)
model.rope_setup.{load_device_weights, cos_matrix, get_rot_idxs, get_rot_mats}
model.prepare_prefill_rot_mats(position_indices)
```

`Llama33_70BGalaxyTransformer2D` exposes a different, mode-explicit contract
(`activate(mode)`, `prefill_forward(..., sequence_length=…, user_ids=…)`,
`project_prefill_logits`, `prepare_prefill_rot_mats(start_pos, seq_len)`). The
adaptation lives here, in the model package, so that `llm_runtime` keeps zero
Galaxy, Llama, 2D-mesh or `(8, 4)` knowledge. See `_GalaxyRuntimeModelView`.

**2. Two placements differ from the 1D convention, and both were qualified at
Milestone B in `GalaxyDirectRunner`.**

*Decode positions and the decode page table.* `DecodeRuntime._prepare_inputs_host`
maps both with `ShardTensor2dMesh(dims=(None, None))` — replicated. The Galaxy
decode graph attends to one mesh column's users on each device, so
`paged_update_cache` and the paged decode SDPA need the device-local table to carry
exactly `users_per_column` rows and the positions to carry that column's users:
`dims=(None, 0)`. A replicated device tensor cannot be turned into a
column-sharded one on device, because slicing a different range per device is not
expressible in one SPMD op. So the executor stages the Galaxy-placed pair at the
operation boundary, from the host request it was handed, and the view consumes
those instead of the runtime's. The runtime's own two small tensors are still
allocated and released by the runtime; nothing in `llm_runtime` changes.

*Logits composition.* `result_collector.concat_host_output` and
`decode._concat_host_output` concatenate mesh **columns** along the vocabulary
axis. On Galaxy the vocabulary is sharded over the eight mesh **rows** and
replicated over the four columns; composing it along the wrong axis is finding
D-B23, and `collectives.compose_galaxy_logits` is the qualified composition that
carries the measurement. This view composes with it. Decode returns the composed
host tensor — both runtime readers accept `torch.Tensor` and pass it straight
through — while prefill re-stages the composed row as a replicated device tensor,
because the runtime untilizes and slices prefill logits on device before reading
them. A device-side all-gather over the mesh-row axis is the trace-compatible
successor and needs a new persistent CCL resource in `galaxy/plans.py`; that is
not this job's to add.

**Out of scope here, deliberately.** No `generator.py`, no vLLM adapter, no lane
group. Batched (concat-32) prefill is out of Milestone C: this executor resolves
`supports_batched_prefill=False` and `post_process_batched_prefill_output` raises.
Tracing is `c-trace`'s job — the trace collaborators are constructed exactly as the
1D executor constructs them, over this same eager executor, and nothing here reads
`trace_mode` in a hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator, Sequence

import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeRuntime, DecodeRuntimeConfig
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.paged_kv_cache import PagedKVCacheManager
from models.common.llm_runtime.prefill.config import PrefillRuntimeConfig
from models.common.llm_runtime.prefill.runtime import PrefillRuntime
from models.common.llm_runtime.program_compiler import ProgramCompiler
from models.common.llm_runtime.tensor_resources import attach_cleanup_failures
from models.common.llm_runtime.trace_compiler import TraceCompiler
from models.common.llm_runtime.warmup import WarmupCoordinator, WarmupCoordinatorConfig
from models.common.models.galaxy.collectives import compose_galaxy_logits, deallocate_if_allocated
from models.common.models.galaxy.recipes import GALAXY_MESH_SHAPE
from models.common.models.llama33_70b_galaxy.model import Llama33_70BGalaxyTransformer2D

#: One tile row block. The runtime's prefill readback addresses the last token by
#: its row inside this block, so a projected prefill result must present 32 rows.
_TILE_SIZE = 32


@dataclass(frozen=True)
class Llama33_70BGalaxyExecutorConfig:
    """Immutable aggregate policy paired with one model-owned Galaxy executor."""

    trace: TraceConfig
    warmup: WarmupConfig
    paged_kv_cache: PagedKVCacheConfig
    device_sampling_enabled: bool = False
    #: Milestone C prefills one row at a time. This resolves the runtime's
    #: batched-prefill capability to False, so the planner never buckets rows
    #: into a concat-32 wave. It is a policy value, not a fallback switch.
    sequential_prefill_only: bool = True

    def __post_init__(self) -> None:
        nested_configs = (
            ("trace", self.trace, TraceConfig),
            ("warmup", self.warmup, WarmupConfig),
            ("paged_kv_cache", self.paged_kv_cache, PagedKVCacheConfig),
        )
        for name, value, expected_type in nested_configs:
            if type(value) is not expected_type:
                raise TypeError(f"{name} must be exactly {expected_type.__name__}")
        for name in ("device_sampling_enabled", "sequential_prefill_only"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


class _GalaxyRopeView:
    """Present `RotarySetup2D` through the runtime's rope-setup contract.

    The runtime reads `cos_matrix.shape[2]` for the rotary capacity check, stages
    decode rotary indices with `get_rot_idxs`, and turns them into cos/sin with
    `get_rot_mats`. `RotarySetup2D` already owns the first two under those exact
    names; only `get_rot_mats` differs, and it is `decode_forward` there.
    """

    def __init__(self, rope: Any):
        self._rope = rope

    @property
    def cos_matrix(self) -> Any:
        return self._rope.cos_matrix

    @property
    def sin_matrix(self) -> Any:
        return self._rope.sin_matrix

    @property
    def config(self) -> Any:
        return self._rope.config

    def load_device_weights(self) -> None:
        self._rope.load_device_weights()

    def get_rot_idxs(self, position_idxs: Any, on_host: bool = False) -> Any:
        return self._rope.get_rot_idxs(position_idxs, on_host=on_host)

    def get_rot_mats(self, rot_idxs: Any) -> list[Any]:
        return self._rope.decode_forward(rot_idxs)


class _GalaxyRuntimeModelView:
    """Adapt the Galaxy 2D graph to the common runtime's model contract.

    Every method here delegates to `Llama33_70BGalaxyTransformer2D`; the view owns
    no device state except the transients it creates and releases inside one call.
    Mode activation is **not** done here — it is an operation-boundary action the
    executor performs before delegating, so no module hot path carries a mode
    branch.
    """

    def __init__(self, model: Llama33_70BGalaxyTransformer2D, *, device_sampling_enabled: bool):
        self._model = model
        self.rope_setup = _GalaxyRopeView(model.rope_setup)
        self._device_sampling_enabled = bool(device_sampling_enabled)
        #: Galaxy-placed decode inputs staged by the executor for one call.
        self._decode_positions: Any = None
        self._decode_page_table: Any = None
        #: Row inside the final tile block that a cached/chunked request's last
        #: token occupies, bound by the executor from the public request.
        self._prefill_last_row: int | None = None

    # -- identity the runtime validates -----------------------------------

    @property
    def model(self) -> Llama33_70BGalaxyTransformer2D:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    @property
    def vocab_size(self) -> int:
        return int(self._model.vocab_size)

    @property
    def num_devices(self) -> int:
        return int(self._model.num_devices)

    @property
    def mesh_device(self) -> Any:
        return self._model.mesh_device

    @property
    def sampling(self) -> Any:
        # Reported only when this lane resolved device sampling. The runtime
        # validates `allow_force_argmax` against this attribute, so presenting a
        # sampler that the executor will not use would misdescribe the lane.
        return self._model.sampling if self._device_sampling_enabled else None

    def iter_executor_named_modules(self) -> Iterator[tuple[str, Any]]:
        return self._model.iter_executor_named_modules()

    # -- decode input staging owned by the model --------------------------

    def bind_decode_inputs(self, *, positions: Any, page_table: Any) -> None:
        """Install the Galaxy-placed decode pair for the next decode body."""

        self._decode_positions = positions
        self._decode_page_table = page_table

    def clear_decode_inputs(self) -> None:
        self._decode_positions = None
        self._decode_page_table = None

    def bind_prefill_last_row(self, row: int | None) -> None:
        """Install the last token's row inside its tile block for this call.

        The runtime hands the cached/chunked prefill body the tile-block bounds as
        device tensors and the row index only when device sampling is on, but the
        readback always addresses row ``(last_token - cached_tokens) % 32``. That
        value is derivable on the host from the public request, which the executor
        has, so the executor binds it at the operation boundary rather than the
        view reading a device tensor back.
        """

        self._prefill_last_row = None if row is None else int(row)

    def clear_prefill_last_row(self) -> None:
        self._prefill_last_row = None

    # -- staging helpers the runtime calls --------------------------------

    def embed_prefill(self, tokens: Any) -> Any:
        return self._model.embed_prefill(_as_token_row(tokens))

    def embed_decode(self, tokens: Any) -> Any:
        return self._model.embed_decode(_as_token_row(tokens))

    def prepare_prefill_rot_mats(self, position_indices: Any) -> list[Any]:
        """Gather prefill cos/sin for an arbitrary position-index tensor.

        `RotarySetup2D.prefill_forward(start_pos, seq_len)` slices its tilized
        table copy, which needs the host start position; the runtime supplies the
        positions as a *device* tensor instead. Gathering them with
        `ttnn.embedding` reads the same table rows, produces the tilized cos/sin
        that `rotary_embedding_llama` requires (`ttnn.embedding` takes a row-major
        table and emits TILE), needs no host round trip, and is the same op the
        decode rotary path already runs on this mesh.
        """

        rope = self._model.rope_setup
        rope.load_device_weights()
        memory_config = rope.config.prefill_cos_sin_memcfg
        gathered: list[Any] = []
        try:
            for table in (rope.cos_matrix, rope.sin_matrix):
                values = ttnn.embedding(
                    _as_token_row(position_indices),
                    table,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                )
                gathered.append(ttnn.reshape(values, ttnn.Shape((1, 1, int(values.shape[-2]), int(values.shape[-1])))))
        except BaseException:
            for value in gathered:
                deallocate_if_allocated(value)
            raise
        return gathered

    # -- graph bodies -----------------------------------------------------

    def prefill_forward(
        self,
        x_embed: Any,
        rot_mats: Any,
        *,
        user_id: Any = 0,
        page_table: Any = None,
        chunk_page_table: Any = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
        batch_size: int | None = None,
        chunk_start_idx_tensor: Any = None,
        last_token_slice: Any = None,
        last_token_index: Any = None,
    ) -> Any:
        """Run one prefill invocation and return hidden state or a logits block.

        The runtime uses two shapes of prefill body. A regular single request asks
        for the hidden state and post-processes it in a separate call; a
        cached/chunked request passes `last_token_slice` and expects the body to
        return the last-token logits itself. Both are served here.
        """

        if int(get_last_token) != -1:
            raise ValueError("the Galaxy prefill body extracts its last token after the graph, not inside it")
        user_ids = tuple(int(value) for value in user_id) if isinstance(user_id, (list, tuple)) else (int(user_id),)
        rows = int(batch_size) if batch_size is not None else len(user_ids)
        if rows != 1 or len(user_ids) != 1:
            raise NotImplementedError(
                "batched (concat-32) prefill is out of Milestone C scope; this executor prefills one row at a time"
            )
        tokens = int(x_embed.shape[-2])
        if tokens % rows:
            raise ValueError(f"{tokens} prefill tokens do not divide into {rows} rows")
        sequence_length = tokens // rows
        chunked = chunk_start_idx is not None
        hidden = self._model.prefill_forward(
            x_embed,
            list(rot_mats),
            sequence_length=sequence_length,
            user_ids=user_ids,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            # The module accepts either the host start or its device tensor, never
            # both. The host value is the one the recipe's chunk alignment is
            # validated against, so it is the one passed.
            chunk_start=chunk_start_idx,
            prefix_user_id=user_ids[0] if chunked else None,
            return_hidden_state=True,
        )
        if last_token_slice is None:
            return hidden
        if self._prefill_last_row is None:
            raise RuntimeError(
                "a cached or chunked Galaxy prefill needs its last-token row bound at the operation boundary"
            )
        try:
            return self._project_tile_block(
                hidden,
                _tile_block_start(last_token_slice),
                sequence_length,
                row=self._prefill_last_row,
            )
        finally:
            deallocate_if_allocated(hidden)

    def post_process_prefill_output(
        self,
        hidden: Any,
        last_token: int,
        *,
        last_token_slice: Any = None,
        last_token_index: Any = None,
    ) -> Any:
        """Normalize, project and present one prefill row's logits.

        Returns a replicated `[1, 1, 32, vocab_size]` TILE tensor whose row
        `last_token % 32` holds the requested token's logits. The runtime
        untilizes it, slices that row, reads it, and composes it with the
        column-concatenating reader — which is correct for a replicated
        full-vocabulary tensor and is not correct for the row-sharded LM head
        output. Composition therefore happens here, with the qualified
        `compose_galaxy_logits`.
        """

        last_token = int(last_token)
        sequence_length = int(hidden.shape[-2])
        if not 0 <= last_token < sequence_length:
            raise ValueError(f"prefill last token {last_token} is outside {sequence_length} tokens")
        row = last_token % _TILE_SIZE
        logits = None
        try:
            (logits,) = self._model.project_prefill_logits(
                hidden,
                rows=1,
                sequence_length=sequence_length,
                token_indices=(last_token,),
            )
            composed = compose_galaxy_logits(
                logits,
                mesh_device=self._model.mesh_device,
                vocab_size=self.vocab_size,
            )
        finally:
            deallocate_if_allocated(logits)
        return self._present_logits_block(composed[:1, :], row)

    def post_process_batched_prefill_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "batched (concat-32) prefill extraction is out of Milestone C scope for the Galaxy executor"
        )

    def decode_forward(self, x_embed: Any, current_pos: Any, rot_mats: Any, page_table: Any = None) -> Any:
        """Run one physical-batch-32 decode step on the Galaxy-placed inputs.

        `current_pos` and `page_table` arrive replicated from the runtime's
        staging and are ignored in favour of the column-sharded pair the executor
        staged; see this module's docstring.
        """

        if self._decode_positions is None or self._decode_page_table is None:
            raise RuntimeError("Galaxy decode inputs were not staged for this call")
        return self._model.decode_forward(
            x_embed,
            self._decode_positions,
            rot_mats,
            self._decode_page_table,
        )

    def gather_and_untilize_logits(self, logits: Any) -> torch.Tensor:
        """Compose row-sharded decode logits into `[1, 1, 32, vocab]` on host.

        The runtime's logits reader accepts a `torch.Tensor` and passes it through
        unchanged, which is what makes the model-owned composition possible
        without a runtime change. The device-side successor is an all-gather over
        the mesh-row axis; it needs a new persistent CCL resource and is left to
        the tracing job that will require it.
        """

        composed = compose_galaxy_logits(
            logits,
            mesh_device=self._model.mesh_device,
            vocab_size=self.vocab_size,
        )
        rows = int(self._model.config.max_batch_size)
        if int(composed.shape[0]) < rows:
            raise ValueError(f"composed {composed.shape[0]} decode rows, expected {rows}")
        return composed[:rows, :].reshape(1, 1, rows, -1)

    # -- private ----------------------------------------------------------

    def _project_tile_block(self, hidden: Any, block_start: int, sequence_length: int, *, row: int) -> Any:
        """Project a chunk's last token and present it in its tile block row."""

        block_start = int(block_start)
        row = int(row)
        if not 0 <= row < _TILE_SIZE:
            raise ValueError(f"prefill last-token row {row} is outside one tile block")
        last_token = block_start + row
        if block_start < 0 or last_token >= sequence_length:
            raise ValueError(f"tile block row {last_token} is outside {sequence_length} prefill tokens")
        outputs: tuple[Any, ...] = ()
        try:
            outputs = self._model.project_prefill_logits(
                hidden,
                rows=1,
                sequence_length=sequence_length,
                token_indices=(last_token,),
            )
            composed = compose_galaxy_logits(
                outputs[0],
                mesh_device=self._model.mesh_device,
                vocab_size=self.vocab_size,
            )
        finally:
            for value in outputs:
                deallocate_if_allocated(value)
        return self._present_logits_block(composed[:1, :], row)

    def _present_logits_block(self, row_logits: torch.Tensor, row: int | None) -> Any:
        """Stage `[1, 1, 32, vocab]` replicated TILE logits from one host row."""

        vocab = int(row_logits.shape[-1])
        block = torch.zeros((1, 1, _TILE_SIZE, vocab), dtype=torch.bfloat16)
        if row is None:
            block[0, 0, :, :] = row_logits[0].to(torch.bfloat16)
        else:
            block[0, 0, int(row), :] = row_logits[0].to(torch.bfloat16)
        return ttnn.from_torch(
            block,
            device=self._model.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._model.mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class Llama33_70BGalaxyExecutor:
    """Compose every runtime owner for one Galaxy Llama-3.3-70B execution lane.

    Construction wires one `PrefillRuntime`, `DecodeRuntime`, `ProgramCompiler`
    and `EagerExecutor` over one full-mesh `Llama33_70BGalaxyTransformer2D`, one
    `PagedKVCacheManager`, one `OutputReader` and one `WarmupCoordinator`.
    Trace-enabled configurations add one `TraceCompiler` and one `TracedExecutor`
    over that exact eager instance.

    The caller resolves and allocates the paged KV cache, warms or compiles
    programs, then calls `prefill_forward` and `decode_forward`. `cleanup` is the
    deterministic release root and makes the executor terminal. The model, its
    `Prefetcher2D` and its Galaxy CCL resources are **borrowed**: the executor
    activates the mode context each operation needs and synchronizes the active
    mode during cleanup, but the loader handle that built the model closes it.
    That is what makes repeated startup/serve/cleanup cycles over one model
    possible.
    """

    requires_prefill_trace_warmup = True

    def __init__(
        self,
        model: Llama33_70BGalaxyTransformer2D,
        runtime_config: Any,
        config: Llama33_70BGalaxyExecutorConfig,
    ) -> None:
        if not isinstance(config, Llama33_70BGalaxyExecutorConfig):
            raise TypeError("config must be a Llama33_70BGalaxyExecutorConfig")
        if not callable(getattr(model, "iter_executor_named_modules", None)):
            raise TypeError("model must provide iter_executor_named_modules()")
        if not callable(getattr(model, "paged_kv_contract", None)):
            raise TypeError("model must provide paged_kv_contract()")
        if not callable(getattr(model, "activate", None)):
            raise TypeError("model must provide activate(mode)")
        if not callable(getattr(runtime_config, "can_enable_trace", None)):
            raise TypeError("runtime_config must provide can_enable_trace()")
        mesh_device = getattr(getattr(model, "config", None), "mesh_device", None)
        if mesh_device is None:
            raise ValueError("model.config.mesh_device is required")
        if tuple(int(value) for value in mesh_device.shape) != GALAXY_MESH_SHAPE:
            raise ValueError(f"the Galaxy executor requires a {GALAXY_MESH_SHAPE} mesh")
        if config.device_sampling_enabled and model.sampling is None:
            raise ValueError("device sampling requires a resolved Sampling2D on the model")
        if config.sequential_prefill_only and tuple(config.warmup.prefill_batch_sizes) != (1,):
            # A warmup batch size denotes a physical padded prefill wave. With
            # batching resolved off, any value above one would plan that many
            # identical single-row requests and compile the same program again,
            # which reads as coverage it is not.
            raise ValueError("sequential prefill requires warmup.prefill_batch_sizes == (1,)")

        self.model = model
        self.runtime_config = runtime_config
        self.model_args = runtime_config
        self.config = config
        self.mesh_device = mesh_device
        self.cache_path = getattr(runtime_config, "model_cache_path", None)
        self.geometry = model.geometry
        self._terminal = False
        self._cleaned_up = False
        self._sampling_buffers_loaded = False
        self._runtime_configuration_sealed = False
        self._active_mode: str | None = None

        self.runtime_model = _GalaxyRuntimeModelView(
            model,
            device_sampling_enabled=config.device_sampling_enabled,
        )
        self.kv_cache_manager = PagedKVCacheManager(model.paged_kv_contract(), config.paged_kv_cache)
        self.page_table_layout = self._resolve_page_table_layout()
        self.output_reader = OutputReader(mesh_device)
        self.prefill_runtime = PrefillRuntime(self._resolve_prefill_config(self.page_table_layout))
        self.decode_runtime = DecodeRuntime(self._resolve_decode_config(self.page_table_layout))
        self.program_compiler = ProgramCompiler(mesh_device, lambda: self.kv_cache_manager.bound_context)
        self.eager_executor = EagerExecutor(
            prefill=self.prefill_runtime,
            decode=self.decode_runtime,
            program_compiler=self.program_compiler,
        )
        self.trace_compiler: TraceCompiler | None = None
        self.traced_executor: TracedExecutor | None = None
        if config.trace.mode != "none":
            self.trace_compiler = TraceCompiler(self.program_compiler)
            self.traced_executor = TracedExecutor(
                eager=self.eager_executor,
                trace_compiler=self.trace_compiler,
                trace_mode=config.trace.mode,
            )
        self.eager_execution = self.eager_executor
        self.traced_prefill_execution = (
            self.traced_executor if config.trace.prefill_enabled and self.traced_executor is not None else None
        )
        self.traced_decode_execution = (
            self.traced_executor if config.trace.decode_enabled and self.traced_executor is not None else None
        )
        self._prefill_execution = self.traced_prefill_execution or self.eager_executor
        self._decode_execution = self.traced_decode_execution or self.eager_executor

        prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_warmup_seq_lens", ())
        if not prefill_sequence_lengths:
            prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
        self.warmup = WarmupCoordinator(
            config=WarmupCoordinatorConfig.resolve(
                warmup=config.warmup,
                trace=config.trace,
                prefill=self.prefill_runtime.config,
                decode=self.decode_runtime.config,
                prefill_sequence_lengths=tuple(int(value) for value in prefill_sequence_lengths),
            ),
            execution=self.traced_executor or self.eager_executor,
            ensure_sampling_buffers=self._ensure_sampling_buffers,
            validate_bound_cache=self._validate_bound_cache,
        )

    # Public model execution API

    @property
    def model_config(self) -> Any:
        return self.model.config

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        return self.kv_cache_manager.config

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def active_mode(self) -> str | None:
        """Return the prefetcher/CCL mode this executor last activated."""

        return self._active_mode

    @property
    def already_warmed_up_prefill(self) -> bool:
        return self.warmup.already_warmed_up_prefill

    def configure_paged_kv_cache(self, config: PagedKVCacheConfig) -> None:
        """Resolve the physical KV geometry before the first allocation.

        The runtime's late-resolution step is documented as "only ``num_blocks``
        becomes final", and on a 1D model that is all it takes. On Galaxy it is
        not: `Attention2D.bind_kv_cache` validates a bound cache against the
        block count its **own** metadata declares, so a physical pool smaller
        than the construction ceiling is refused at binding —

            ValueError: paged KV cache shape must be (2048, 1, 32, 128),
                        got (95, 1, 32, 128)                (`logs/i9_shrink_l1.log`)

        The model therefore has to be told the physical count, not just the
        ceiling. That is the model-owned half of this step and it lives here.
        Resolving may only shrink: the ceiling was a construction-time capacity
        bound, and narrowing it to the physical pool is what makes the bound
        cache, the module metadata and the page-table geometry describe one
        geometry.
        """

        self._ensure_active()
        if self._runtime_configuration_sealed:
            raise RuntimeError("runtime configuration is sealed")
        if not isinstance(config, PagedKVCacheConfig):
            raise TypeError("config must be a PagedKVCacheConfig")
        current = self.kv_cache_manager.config
        if current.is_resolved():
            raise RuntimeError("paged KV cache configuration is already resolved")
        if config.dtype != current.dtype:
            raise ValueError("resolved paged KV cache cannot change dtype")
        if config.memory_config != current.memory_config:
            raise ValueError("resolved paged KV cache cannot change memory_config")
        if config.block_size != current.block_size:
            raise ValueError("resolved paged KV cache cannot change block_size")
        if not config.is_resolved():
            raise ValueError("resolved paged KV cache must contain num_blocks")
        physical = int(config.num_blocks)
        if physical > current.max_num_blocks:
            raise ValueError(
                f"resolved paged KV capacity {physical} exceeds the construction ceiling {current.max_num_blocks}"
            )
        resolved = PagedKVCacheConfig(
            block_size=int(config.block_size),
            max_num_blocks=physical,
            dtype=config.dtype,
            memory_config=config.memory_config,
            num_blocks=physical,
        )
        if physical == current.max_num_blocks:
            # Nothing about the model's per-layer metadata moves, so the manager
            # keeps its identity and its already-validated model contract.
            self.kv_cache_manager.configure(resolved)
        else:
            # The model's paged metadata moves, and `GalaxyPagedKVContract`
            # snapshots that metadata at construction. The manager owns no device
            # resource before `allocate()`, so the honest move is to rebuild it
            # against the model's updated contract rather than to let a stale
            # snapshot validate the replacement.
            self.model.configure_paged_attention(block_size=resolved.block_size, max_num_blocks=physical)
            self.kv_cache_manager = PagedKVCacheManager(self.model.paged_kv_contract(), resolved)
        self.config = replace(self.config, paged_kv_cache=resolved)
        self._refresh_page_table_layout()

    def allocate_kv_cache(self) -> list[list[Any]]:
        """Allocate and bind the model-owned paged KV cache."""

        self._ensure_active()
        if not self.kv_cache_manager.config.is_resolved():
            raise RuntimeError("Paged KV cache capacity must be resolved before allocation")
        self._seal_runtime_configuration()
        return self.kv_cache_manager.allocate()

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Compile prefill on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("prefill")
        self.runtime_model.bind_prefill_last_row(_prefill_last_row(tokens, prompt_lens, start_pos))
        try:
            return (execution or self._prefill_execution).compile_prefill(
                tokens=tokens,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Compile decode on the supplied eager or traced execution target."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("decode")
        with self._staged_decode_inputs(start_pos=start_pos, page_table=page_table):
            return (execution or self._decode_execution).compile_decode(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
            )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Validate ownership, activate the prefill context, and run one call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("prefill")
        self.runtime_model.bind_prefill_last_row(_prefill_last_row(tokens, prompt_lens, start_pos))
        try:
            return (execution or self._prefill_execution).prefill_forward(
                tokens=tokens,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        read_from_device: bool = True,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        """Validate ownership, activate the decode context, and run one call."""

        self._ensure_active()
        self._validate_bound_cache(kv_cache)
        self._ensure_sampling_for(sampling_params)
        self.activate("decode")
        with self._staged_decode_inputs(start_pos=start_pos, page_table=page_table):
            return (execution or self._decode_execution).decode_forward(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
                read_from_device=read_from_device,
            )

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
    ) -> bool:
        """Classify whether the prefill request can use this lane's trace."""

        if self.traced_executor is None or not self.config.trace.prefill_enabled:
            return False
        return self.prefill_runtime.can_trace(
            tokens=tokens,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
        )

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        self._ensure_active()
        return self.decode_runtime.read_decode_output(tt_out=tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
        self._ensure_active()
        return self.decode_runtime.process_decode_output_host(tt_out=tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool = False,
        enable_trace: bool = False,
    ) -> None:
        self._ensure_active()
        self.activate("prefill")
        # Every warmup plan includes a cached (prefix) prefill case, and a cached
        # request's body extracts its own last token. Warmup builds each case with
        # its whole padded length present and no partial prompt, so the last
        # token's row inside its tile block is `(length - 1) % 32` for every case.
        self.runtime_model.bind_prefill_last_row(self._warmup_prefill_last_row())
        try:
            return self.warmup.warmup_prefill(
                kv_cache=kv_cache,
                can_sample_on_device=can_sample_on_device,
                enable_trace=enable_trace,
            )
        finally:
            self.runtime_model.clear_prefill_last_row()

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,
        max_batch_size: int | None = None,
        num_blocks: int | None = None,
        can_sample_on_device: bool = False,
        enable_trace: bool = False,
    ) -> None:
        self._ensure_active()
        lane_capacity = int(self.decode_runtime.config.lane_capacity)
        blocks = int(num_blocks) if num_blocks is not None else int(self.page_table_layout.decode_width)
        self.activate("decode")
        page_table = torch.zeros((lane_capacity, blocks), dtype=torch.int32)
        with self._staged_decode_inputs(
            start_pos=torch.zeros(lane_capacity, dtype=torch.long),
            page_table=page_table,
        ):
            return self.warmup.warmup_decode(
                kv_cache=kv_cache,
                max_batch_size=lane_capacity if max_batch_size is None else int(max_batch_size),
                num_blocks=blocks,
                can_sample_on_device=can_sample_on_device,
                enable_trace=enable_trace,
            )

    def activate(self, mode: str) -> Any:
        """Activate one operation's prefetcher/CCL context at its boundary."""

        if mode not in ("prefill", "decode"):
            raise ValueError(f"unsupported Galaxy execution mode: {mode!r}")
        context = self.model.activate(mode)
        self._active_mode = mode
        return context

    def synchronize(self) -> None:
        """Wait for the active mode's outstanding device work."""

        if self._active_mode is not None:
            self.model.synchronize(self._active_mode)

    def cleanup(self) -> None:
        """Release runtime, trace, program and KV resources in order.

        Outstanding device work is drained first, so nothing the runtime is about
        to deallocate is still referenced by a running program. The borrowed
        prefetcher/CCL context is left as the model found it; the model's own
        owner closes it.
        """

        self._terminal = True
        if self._cleaned_up:
            return

        failures: list[BaseException] = []
        actions = [
            self.synchronize,
            self.runtime_model.clear_decode_inputs,
            self.runtime_model.clear_prefill_last_row,
            self.decode_runtime.drain_external_outputs,
            self.output_reader.drain,
            self.prefill_runtime.cleanup,
            self.decode_runtime.cleanup_transients,
        ]
        if self.trace_compiler is not None:
            actions.append(self.trace_compiler.cleanup)
        actions.append(self.program_compiler.cleanup)
        if self.config.device_sampling_enabled:
            actions.append(self.model.sampling.release)
        actions.append(self.kv_cache_manager.release)

        for action in actions:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - collect, then raise the first
                failures.append(error)
        if failures:
            _raise_cleanup_failures(failures, "Llama33_70BGalaxyExecutor")
        self._cleaned_up = True

    # Private implementation

    class _StagedDecodeInputs:
        """Own the Galaxy-placed decode pair for the span of one call."""

        def __init__(self, executor: "Llama33_70BGalaxyExecutor", *, positions: Any, page_table: Any):
            self._executor = executor
            self._positions = positions
            self._page_table = page_table

        def __enter__(self) -> "Llama33_70BGalaxyExecutor._StagedDecodeInputs":
            self._executor.runtime_model.bind_decode_inputs(
                positions=self._positions,
                page_table=self._page_table,
            )
            return self

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            self._executor.runtime_model.clear_decode_inputs()
            deallocate_if_allocated(self._positions)
            deallocate_if_allocated(self._page_table)

    def _staged_decode_inputs(self, *, start_pos: torch.Tensor, page_table: torch.Tensor) -> Any:
        """Stage the column-sharded decode positions and page table."""

        lane_capacity = int(self.decode_runtime.config.lane_capacity)
        positions = _require_row(start_pos, lane_capacity, "decode start_pos")
        table = _require_decode_page_table(page_table, lane_capacity, int(self.page_table_layout.decode_width))
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE)
        staged_positions = ttnn.from_torch(
            positions.to(torch.int32),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        try:
            staged_table = ttnn.from_torch(
                table,
                device=self.mesh_device,
                mesh_mapper=mapper,
                dtype=self.model.kv_specs[0].page_table_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except BaseException:
            deallocate_if_allocated(staged_positions)
            raise
        return self._StagedDecodeInputs(self, positions=staged_positions, page_table=staged_table)

    def _warmup_prefill_last_row(self) -> int | None:
        """Return the tile-block row every configured warmup case's last token has."""

        lengths = set(int(value) for value in self.warmup.config.prefill_sequence_lengths)
        # The Q128 sampled path primes the four tile ends of a 128-token prefill.
        if 128 in lengths and self.config.device_sampling_enabled:
            lengths.update((32, 64, 96, 128))
        rows = {(length - 1) % _TILE_SIZE for length in lengths}
        return rows.pop() if len(rows) == 1 else None

    def _resolve_prefill_config(self, layout: PageTableLayout) -> PrefillRuntimeConfig:
        runtime_config = self.runtime_config
        supports_batched = (
            False
            if self.config.sequential_prefill_only
            else bool(getattr(runtime_config, "supports_batched_prefill", False))
        )
        return PrefillRuntimeConfig.resolve(
            model=self.runtime_model,
            output_reader=self.output_reader,
            page_table_layout=layout,
            max_batch_size=int(self.model.config.max_batch_size),
            max_prefill_chunk_size=int(runtime_config.max_prefill_chunk_size),
            device_sampling_enabled=self.config.device_sampling_enabled,
            can_enable_trace=runtime_config.can_enable_trace,
            supports_batched_prefill=supports_batched,
            disable_batched_prefill=self.config.sequential_prefill_only
            or bool(getattr(runtime_config, "disable_batched_prefill", False)),
            max_prefill_batch_size=1
            if self.config.sequential_prefill_only
            else int(getattr(runtime_config, "max_prefill_batch_size", 1)),
            batched_prefill_batched_extract=False
            if self.config.sequential_prefill_only
            else bool(getattr(runtime_config, "batched_prefill_batched_extract", False)),
        )

    def _resolve_decode_config(self, layout: PageTableLayout) -> DecodeRuntimeConfig:
        return DecodeRuntimeConfig.resolve(
            model=self.runtime_model,
            output_reader=self.output_reader,
            lane_capacity=int(self.model.config.max_batch_size),
            page_table_layout=layout,
            device_sampling_enabled=self.config.device_sampling_enabled,
            force_greedy_top_k=self.config.warmup.include_decode_top_k,
        )

    def _resolve_page_table_layout(self) -> PageTableLayout:
        kv_config = self.kv_cache_manager.config
        physical_num_blocks = kv_config.num_blocks or kv_config.max_num_blocks
        return PageTableLayout.resolve(
            block_size=int(kv_config.block_size),
            model_max_sequence_length=int(self.model.config.max_seq_len),
            physical_num_blocks=int(physical_num_blocks),
            max_prefill_chunk_size=min(
                int(self.runtime_config.max_prefill_chunk_size),
                int(self.model.config.max_seq_len),
            ),
        )

    def _refresh_page_table_layout(self) -> None:
        """Install the final physical geometry across every resolved config."""

        layout = self._resolve_page_table_layout()
        prefill_config = self._resolve_prefill_config(layout)
        decode_config = self._resolve_decode_config(layout)
        warmup_config = WarmupCoordinatorConfig.resolve(
            warmup=self.warmup.config.warmup,
            trace=self.config.trace,
            prefill=prefill_config,
            decode=decode_config,
            prefill_sequence_lengths=self.warmup.config.prefill_sequence_lengths,
        )
        self.prefill_runtime.config = prefill_config
        self.decode_runtime.config = decode_config
        self.warmup.config = warmup_config
        self.page_table_layout = layout

    def _seal_runtime_configuration(self) -> None:
        self.warmup.seal_configuration()
        self._runtime_configuration_sealed = True

    def _ensure_sampling_for(self, sampling_params: Any) -> None:
        if sampling_params is None:
            return
        if not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")
        self._ensure_sampling_buffers()

    def _ensure_sampling_buffers(self) -> None:
        if self._sampling_buffers_loaded or not self.config.device_sampling_enabled:
            return
        if self.trace_compiler is not None and self.trace_compiler.trace_active:
            raise RuntimeError("cannot materialize sampling buffers after trace activation")
        self.model.sampling.load_device_buffers()
        self._sampling_buffers_loaded = True

    def _validate_bound_cache(self, kv_cache: Any) -> None:
        if self.kv_cache_manager.bound_context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before execution")
        if kv_cache is not None:
            self.kv_cache_manager.validate_borrowed_handle(kv_cache)

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("Llama33_70BGalaxyExecutor is terminal; construct a new executor")
        if self.prefill_runtime.transient_orphan_count or self.decode_runtime.transient_orphan_count:
            raise RuntimeError("Llama33_70BGalaxyExecutor has unreleased transient resources; clean up this executor")


def build_llama33_70b_galaxy_executor(
    llm: Any,
    config: Llama33_70BGalaxyExecutorConfig,
) -> Llama33_70BGalaxyExecutor:
    """Build one executor around an already-loaded Galaxy Llama handle."""

    return Llama33_70BGalaxyExecutor(llm.model, llm.runtime_config, config)


def default_galaxy_paged_kv_cache_config(model: Any, dtype: Any = None) -> PagedKVCacheConfig:
    """Return the `PagedKVCacheConfig` matching a built model's paged geometry."""

    spec = model.kv_specs[0]
    paged = spec.paged_attention_config
    if paged is None:
        raise ValueError("the Galaxy executor requires a paged KV model")
    return PagedKVCacheConfig(
        block_size=int(paged.block_size),
        max_num_blocks=int(paged.max_num_blocks),
        dtype=spec.kv_cache_dtype if dtype is None else dtype,
        num_blocks=int(paged.max_num_blocks),
    )


def _prefill_last_row(
    tokens: torch.Tensor,
    prompt_lens: torch.Tensor | None,
    start_pos: torch.Tensor | None,
) -> int | None:
    """Return the shared tile-block row of every request row's last token.

    The runtime addresses a cached or chunked prefill result at row
    ``(last_token_index - cached_tokens) % 32``, and derives the last token from
    ``prompt_lens`` and the cached prefix from ``start_pos`` exactly this way.
    Returns ``None`` when the public call's rows disagree, which leaves the view
    to refuse rather than to guess.
    """

    rows = int(tokens.shape[0])
    width = int(tokens.shape[-1])
    lengths = [width] * rows if prompt_lens is None else [int(value) for value in prompt_lens.reshape(-1)]
    cached = [0] * rows if start_pos is None else [int(value) for value in start_pos.reshape(-1)]
    if len(lengths) != rows or len(cached) != rows:
        return None
    candidates = {(length - 1 - num_cached) % _TILE_SIZE for length, num_cached in zip(lengths, cached)}
    return candidates.pop() if len(candidates) == 1 else None


def _as_token_row(tokens: Any) -> Any:
    """Return a rank-2 `[1, n]` view of a runtime-staged index tensor."""

    shape = tuple(int(value) for value in tokens.shape)
    if len(shape) == 2:
        return tokens
    return ttnn.reshape(tokens, ttnn.Shape((1, shape[-1])))


def _tile_block_start(last_token_slice: Any) -> int:
    """Read the tile-block start out of the runtime's slice-bound tensor."""

    start = last_token_slice[0] if isinstance(last_token_slice, (tuple, list)) else last_token_slice
    values = ttnn.to_torch(ttnn.get_device_tensors(start)[0]).reshape(-1)
    return int(values[2])


def _require_row(values: torch.Tensor, expected: int, name: str) -> torch.Tensor:
    row = values.reshape(-1)
    if int(row.numel()) != expected:
        raise ValueError(f"{name} must hold {expected} entries, got {int(row.numel())}")
    return row


def _require_decode_page_table(page_table: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    """Return the `[rows, width]` int32 decode table the Galaxy graph reads.

    The Galaxy decode SDPA derives each slot's KV length from the table's row
    width, so the table is presented at the resolved decode width: narrower
    request tables are zero-extended and wider ones are refused rather than
    silently narrowed.
    """

    if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
        raise ValueError("decode page_table must be a rank-2 torch.Tensor")
    if int(page_table.shape[0]) != rows:
        raise ValueError(f"decode page_table must have {rows} rows, got {int(page_table.shape[0])}")
    actual = int(page_table.shape[1])
    if actual > width:
        raise ValueError(f"decode page_table width {actual} exceeds the resolved decode width {width}")
    table = torch.zeros((rows, width), dtype=torch.int32)
    table[:, :actual] = page_table.to(torch.int32)
    return table


def _raise_cleanup_failures(failures: list[BaseException], owner: str) -> None:
    primary, *additional = failures
    attach_cleanup_failures(
        primary,
        additional,
        note=f"{owner} cleanup also encountered {{count}} failure(s)",
    )
    raise primary


__all__ = [
    "Llama33_70BGalaxyExecutor",
    "Llama33_70BGalaxyExecutorConfig",
    "build_llama33_70b_galaxy_executor",
    "default_galaxy_paged_kv_cache_config",
]
