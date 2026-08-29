# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic only — never committed as a gate.

**Who owns the L1 buffer the Llama prefill circular buffers clash with?**

c-defects attempt 1 refuted the received answer (the prefetcher global circular
buffer: freed, measured absent, clash persists) and left the owner unnamed, asking
for `TT_METAL_WATCHER` or instrumentation at `program.cpp:1763`. This job's
`i5_warmup_l1` narrowed the trigger to *a prefill after a decode in one process*
and reproduced it in 110 s with no request at all, which makes a direct question
affordable.

There is a candidate with the right shape. `galaxy_address_memory_config` places
the prefetcher's packed weight-address table **HEIGHT_SHARDED in L1 on
`prefetch_sender_cores()`** — and the clash names core range `[0-0 - 0-3]`, four
cores at x=0. Its shard is `[1, weight_count]` uint32, i.e. a few kilobytes, which
is consistent with c-defects' allocator dumps finding no live block over 100 kB at
the failing prefill.

So this probe asks three things in one process:

1. what address the packed weight-address table actually occupies, and what the
   global circular buffer occupies, printed from the tensors themselves;
2. whether the minimal trigger is `activate("decode")` alone, or whether a decode
   program has to run first;
3. the clashing address the throw reports, in the same process, so the two
   numbers can be compared rather than matched across logs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

import ttnn
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm
from models.common.tests.models.galaxy.galaxy_hardware import GALAXY_DEVICE_PARAMS, GALAXY_MESH_SHAPE


def _report_prefetcher(prefetcher: Any, tag: str) -> None:
    metadata = getattr(prefetcher, "_weight_address_metadata", None)
    global_cb = getattr(prefetcher, "_global_cb", None)
    size = getattr(prefetcher, "_resolved_global_cb_size", None)
    print(f"[probe:{tag}] resolved_global_cb_size={size} global_cb_present={global_cb is not None}", flush=True)
    if metadata is None:
        print(f"[probe:{tag}] weight address metadata: absent", flush=True)
        return
    try:
        address = metadata.buffer_address()
    except BaseException as error:  # noqa: BLE001 - a diagnostic, report and continue
        address = f"unavailable ({error})"
    print(
        f"[probe:{tag}] weight address metadata: address={address} shape={tuple(metadata.shape)} "
        f"dtype={metadata.dtype} memcfg={metadata.memory_config()}",
        flush=True,
    )


def _tiny_prefill(model: Any, tokens: int = 128) -> None:
    row = torch.zeros((1, tokens), dtype=torch.int32)
    embedded = model.embed_prefill(LazyWeight(source=row, device=model.mesh_device))
    ttnn.deallocate(embedded)


@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_who_owns_the_clashing_l1_buffer(mesh_device: ttnn.MeshDevice) -> None:
    layers = int(os.getenv("LLAMA33_70B_GALAXY_TEST_LAYERS", "1"))
    handle = from_pretrained(
        mesh_device,
        hf_model=DEFAULT_HF_MODEL,
        max_seq_len=2048,
        prefill_sequence_lengths=(128,),
        n_layers=layers,
        enable_device_sampling=False,
        load_hf_model=lambda: load_layer_subset_causal_lm(DEFAULT_HF_MODEL, layer_indices=tuple(range(layers))),
    )
    try:
        model = handle.model
        prefetcher = model.prefetcher
        _report_prefetcher(prefetcher, "sealed")

        # Arm 1: prefill first, with nothing before it. This is the path every
        # passing run of this job took, so it must place cleanly.
        model.activate("prefill")
        try:
            _tiny_prefill(model)
            print("[probe:arm1] prefill embedding before any decode: OK", flush=True)
        except BaseException as error:
            print(
                f"[probe:arm1] prefill embedding before any decode: CLASH {type(error).__name__}: {error}", flush=True
            )

        # Arm 2: activate decode, run no decode op at all, then prefill again.
        model.activate("decode")
        _report_prefetcher(prefetcher, "decode-active")
        model.activate("prefill")
        _report_prefetcher(prefetcher, "prefill-after-decode")
        try:
            _tiny_prefill(model)
            print("[probe:arm2] prefill after activate(decode) only: OK", flush=True)
        except BaseException as error:
            print(
                f"[probe:arm2] prefill after activate(decode) only: CLASH {type(error).__name__}: {error}", flush=True
            )

        # Arm 3: a prefill at a *different* token count, so no cached program can
        # skip validate_circular_buffer_region. c-defects lost a reading to
        # exactly that.
        try:
            _tiny_prefill(model, tokens=96)
            print("[probe:arm3] prefill at a cold token count: OK", flush=True)
        except BaseException as error:
            print(f"[probe:arm3] prefill at a cold token count: CLASH {type(error).__name__}: {error}", flush=True)
    finally:
        handle.close()
