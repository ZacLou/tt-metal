"""Diagnostic probe, never committed: where does the Llama two-pools DRAM go?

`t4`/`t5`/`t6` (attempt 4's gate queue, full 80-layer shape, THREE fresh
processes, byte-identical) show the Llama two-pools case dying while loading
the SECOND model's ring weights:

    TT_FATAL: Out of Memory: Not enough space to allocate 2297856 B DRAM buffer
    across 11 banks, ... (allocated: 1070239264 B, free: 533920 B)

at `layer53_wqkv_ring`, i.e. 66 % of the way through the second model's ring
pass. The first model had already been closed, deleted and gc-collected.

Two explanations fit, and they need different fixes:

  RETENTION - `close()` does not return model 1's DRAM, so model 2 is loading
      on top of it. Candidates: `Prefetcher2D.cleanup()` never clears
      `self._registered_weights`; `Llama33_70BTransformerBlock2D.close()` calls
      only `self.attention.close()` and `MLP2D` has no `release` at all;
      `Attention2D.close()` does not release `wqkv`/`wo`.
  CAPACITY - `close()` does return it, and model 2 alone does not fit, because
      the second arm asks for a 4096-block pool where the first asked for 2048.

This separates them by measuring the DRAM allocator directly at each lifecycle
point, on a layer subset, using the REAL test's `_load`/`_paged_config` so the
path is the one that fails. Run it at two different layer counts and the slope
of `model_dram` per layer extrapolates each arm to the full 80-layer shape;
the residue after close answers the retention question on its own.

Attempt 5's version of this probe died in 46 s on an API slip:
`ttnn.get_memory_view` returns a `MemoryView`, and the block list is
`view.block_table` (see `prefetcher_2d._default_l1_block_table`). Every
allocator read below is guarded so a second such slip still leaves the earlier
measurements on disk.
"""

from __future__ import annotations

import collections
import gc
import os
from typing import Any

import pytest
import torch

import ttnn

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner
from models.common.models.galaxy.kv_contract import GalaxyPagedAttentionConfig
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    hf_config_or_skip,
    load_reference_tokens,
)

#: Verbatim from `test_step7_coverage_wh_galaxy.py` - replicated rather than
#: imported, because that module is also collected by pytest and importing it
#: under a second dotted name is a needless way to lose a device slot.
_BLOCK_SIZE = 32
_REFERENCE_NAME = "Llama-3.3-70B-Instruct"


def _paged_config(*, context: int, active_slots: int) -> GalaxyPagedAttentionConfig:
    blocks_per_user = -(-context // _BLOCK_SIZE)
    sinks = GALAXY_PHYSICAL_BATCH - active_slots
    return GalaxyPagedAttentionConfig(block_size=_BLOCK_SIZE, max_num_blocks=blocks_per_user * active_slots + sinks)


def _layers() -> int | None:
    value = os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS")
    return int(value) if value else None


def _load(mesh_device: Any, **overrides: Any):
    hf_model = os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)
    hf_config_or_skip(hf_model)
    kwargs: dict[str, Any] = dict(
        hf_model=hf_model,
        max_seq_len=2048,
        prefill_sequence_lengths=(128,),
        n_layers=_layers(),
    )
    kwargs.update(overrides)
    return from_pretrained(mesh_device, **kwargs)


def _close(handle: Any) -> None:
    try:
        handle.close()
    finally:
        del handle
        gc.collect()


def _distinct_rows(length: int, count: int) -> list[list[int]]:
    reference_tokens, _ = load_reference_tokens(_REFERENCE_NAME)
    source = [int(value) for value in reference_tokens]
    if len(source) < length + count:
        return [[source[(offset + index) % len(source)] for index in range(length)] for offset in range(count)]
    return [source[offset : offset + length] for offset in range(count)]


def _table(mesh_device: Any, buffer_type: Any) -> tuple[tuple[int, int, bool], ...]:
    view = ttnn.get_memory_view(mesh_device, buffer_type)
    offset = ttnn.get_allocator_base_address(mesh_device, buffer_type)
    return tuple(
        (int(block["address"]) + offset, int(block["size"]), block.get("allocated") == "yes")
        for block in view.block_table
    )


def _live(mesh_device: Any, buffer_type: Any) -> dict[int, int]:
    try:
        return {a: s for a, s, allocated in _table(mesh_device, buffer_type) if allocated}
    except BaseException as exc:  # noqa: BLE001
        print(f"[probe] allocator view unavailable: {type(exc).__name__}: {exc}", flush=True)
        return {}


def _report(mesh_device: Any, label: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, buffer_type in (("DRAM", ttnn.BufferType.DRAM), ("L1", ttnn.BufferType.L1)):
        live = _live(mesh_device, buffer_type)
        total = sum(live.values())
        biggest = sorted(live.items(), key=lambda kv: -kv[1])[:5]
        print(
            f"[probe] {label} {name}: blocks={len(live)} total={total} "
            f"lowest={min(live, default=-1)} biggest={biggest}",
            flush=True,
        )
        out[name] = total
    try:
        print(f"[probe] {label} program_cache_entries={mesh_device.num_program_cache_entries()}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"[probe] {label} program_cache_entries unavailable: {exc}", flush=True)
    return out


def _hist(label: str, blocks: dict[int, int]) -> None:
    hist = collections.Counter(blocks.values())
    print(f"[probe] {label}: n={len(blocks)} bytes={sum(blocks.values())}", flush=True)
    for size, count in sorted(hist.items(), key=lambda kv: -kv[0] * kv[1])[:12]:
        print(f"[probe]   {label} size {size} x{count} = {size * count}", flush=True)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_llama_dram_lifecycle(mesh_device: ttnn.MeshDevice):
    layers = os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS")
    print(f"[probe] LLAMA33_70B_GALAXY_TEST_LAYERS={layers}", flush=True)

    # Self-test the allocator read before anything expensive happens.
    probe_table = _live(mesh_device, ttnn.BufferType.DRAM)
    print(f"[probe] allocator self-test: {len(probe_table)} live DRAM blocks", flush=True)

    baseline = _report(mesh_device, "0-before-any-model")
    base_live = _live(mesh_device, ttnn.BufferType.DRAM)

    rows = _distinct_rows(128, GALAXY_PHYSICAL_BATCH)
    all_pools = {
        "default-2048": None,
        "explicit-4096": _paged_config(context=4096, active_slots=GALAXY_PHYSICAL_BATCH),
    }
    # `PROBE_POOLS` selects which arms to build, so the same file can ask two
    # different questions: both arms answers "is model 1's DRAM returned", and
    # `explicit-4096` alone answers "does the second arm's larger pool fit at
    # all", which is the capacity half of the same fork.
    wanted = os.getenv("PROBE_POOLS", ",".join(all_pools))
    pools = {k: v for k, v in all_pools.items() if k in {w.strip() for w in wanted.split(",")}}
    assert pools, f"PROBE_POOLS={wanted!r} selected no arm of {sorted(all_pools)}"
    use_model = os.getenv("PROBE_SKIP_USE", "") not in {"1", "true", "yes"}
    print(f"[probe] pools={sorted(pools)} use_model={use_model}", flush=True)

    marks: dict[str, dict[str, int]] = {"baseline": baseline}
    for index, (name, paged) in enumerate(pools.items()):
        handle = _load(mesh_device, max_seq_len=4096 if paged else 2048, paged_attention_config=paged)
        marks[f"{index}-{name}-built"] = _report(mesh_device, f"{index}-{name}-built")

        # Record the prefetcher's registered weights by ADDRESS only. Holding the
        # tensors would pin exactly the references this is trying to detect; an
        # address is inert and can be looked up in the table after the model is gone.
        borrowed: list[tuple[str, int]] = []
        try:
            registered = handle.model.prefetcher._registered_weights
            for weight_name, tensor in registered.items():
                try:
                    borrowed.append((weight_name, int(tensor.buffer_address())))
                except BaseException as exc:  # noqa: BLE001
                    print(f"[probe] cannot address registered weight {weight_name}: {exc}", flush=True)
            del registered
        except BaseException as exc:  # noqa: BLE001
            print(f"[probe] no registered-weight map: {type(exc).__name__}: {exc}", flush=True)
        print(f"[probe] {name}: recorded {len(borrowed)} registered weight addresses", flush=True)

        try:
            if not use_model:
                raise RuntimeError("PROBE_SKIP_USE set: model not exercised, by request")
            with GalaxyDirectRunner(handle.model) as runner:
                for slot, row in enumerate(rows[:4]):
                    runner.prefill_row(row, slot=slot)
                runner.decode_logits([1] * GALAXY_PHYSICAL_BATCH, [128] * GALAXY_PHYSICAL_BATCH)
                marks[f"{index}-{name}-used"] = _report(mesh_device, f"{index}-{name}-used")
        except BaseException as exc:  # noqa: BLE001
            print(f"[probe] {name}: use failed: {type(exc).__name__}: {exc}", flush=True)
        marks[f"{index}-{name}-runner-closed"] = _report(mesh_device, f"{index}-{name}-runner-closed")

        _close(handle)
        del handle
        gc.collect()
        after = _report(mesh_device, f"{index}-{name}-closed-and-collected")
        marks[f"{index}-{name}-closed"] = after

        live_now = _live(mesh_device, ttnn.BufferType.DRAM)
        still = [(n, a) for n, a in borrowed if a in live_now]
        print(
            f"[probe] {name}: registered weights still allocated after close+gc: "
            f"{len(still)} of {len(borrowed)}",
            flush=True,
        )
        for weight_name, address in still[:15]:
            print(f"[probe]   still live: {weight_name} @ {address}", flush=True)

        residual = {a: s for a, s in live_now.items() if a not in base_live}
        _hist(f"{name}-residual-vs-baseline", residual)

    print("[probe] ---- summary ----", flush=True)
    for label, value in marks.items():
        print(f"[probe] SUMMARY {label} DRAM={value['DRAM']} L1={value['L1']}", flush=True)
    names = list(pools)
    first = names[0]
    m1 = marks.get(f"0-{first}-built", {}).get("DRAM", 0) - baseline["DRAM"]
    r1 = marks.get(f"0-{first}-closed", {}).get("DRAM", 0) - baseline["DRAM"]
    m2 = 0
    if len(names) > 1:
        m2 = marks.get(f"1-{names[1]}-built", {}).get("DRAM", 0) - marks.get(f"0-{first}-closed", {}).get("DRAM", 0)
    print(f"[probe] VERDICT layers={layers} pools={names} model1_dram={m1} residue_after_close={r1} model2_delta_dram={m2}", flush=True)
    print(f"[probe] VERDICT residue_fraction={r1 / m1 if m1 else float('nan'):.4f}", flush=True)
    # No assertion: this is a measurement. The numbers above are the result.
