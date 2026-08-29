# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic only — never committed as a gate.

Is the executor's prefill rotary **gather** numerically the same as
`RotarySetup2D.prefill_forward`'s **slice**? `i4_pagedkv_l1` reported the first
layer's K at PCC 0.907 while the prefill logits agreed at 0.9994, and K is the one
tensor of the pair that passes through RoPE. This asks the question directly, over
a one-layer subset of the real checkpoint, with no executor in the way.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

import ttnn
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, from_pretrained
from models.common.tests.models.galaxy.galaxy_checkpoint import load_layer_subset_causal_lm
from models.common.tests.models.galaxy.galaxy_hardware import GALAXY_DEVICE_PARAMS, GALAXY_MESH_SHAPE

_LENGTH = 128


def _compose(tensor: Any, mesh_device: Any) -> torch.Tensor:
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=GALAXY_MESH_SHAPE),
    ).float()


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    from models.common.utility_functions import comp_pcc

    _, message = comp_pcc(expected.unsqueeze(0).float(), actual.unsqueeze(0).float(), 0.0)
    return message


@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
def test_rope_gather_matches_the_slice(mesh_device: ttnn.MeshDevice) -> None:
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
        model.activate("prefill")
        rope = model.rope_setup
        rope.load_device_weights()
        print(f"[probe] cos_matrix shape={tuple(rope.cos_matrix.shape)} dtype={rope.cos_matrix.dtype}", flush=True)
        print(
            f"[probe] cos_matrix_prefill shape={tuple(rope.cos_matrix_prefill.shape)} "
            f"dtype={rope.cos_matrix_prefill.dtype} layout={rope.cos_matrix_prefill.layout}",
            flush=True,
        )

        sliced = rope.prefill_forward(start_pos=0, seq_len=_LENGTH)
        indices = ttnn.from_torch(
            torch.arange(0, _LENGTH, dtype=torch.long).reshape(1, -1),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered = []
        for table in (rope.cos_matrix, rope.sin_matrix):
            values = ttnn.embedding(
                indices,
                table,
                layout=ttnn.TILE_LAYOUT,
                memory_config=rope.config.prefill_cos_sin_memcfg,
            )
            gathered.append(ttnn.reshape(values, ttnn.Shape((1, 1, int(values.shape[-2]), int(values.shape[-1])))))

        for name, expected, actual in (("cos", sliced[0], gathered[0]), ("sin", sliced[1], gathered[1])):
            print(
                f"[probe] {name}: sliced shape={tuple(expected.shape)} layout={expected.layout} "
                f"memcfg={expected.memory_config()} | gathered shape={tuple(actual.shape)} "
                f"layout={actual.layout} memcfg={actual.memory_config()}",
                flush=True,
            )
            host_expected = _compose(expected, mesh_device)
            host_actual = _compose(actual, mesh_device)
            print(f"[probe] {name}: composed {tuple(host_expected.shape)} vs {tuple(host_actual.shape)}", flush=True)
            if host_expected.shape == host_actual.shape:
                diff = (host_expected - host_actual).abs()
                print(
                    f"[probe] {name}: pcc={_pcc(host_expected, host_actual)} maxabsdiff={float(diff.max()):.6g} "
                    f"mismatched={int((diff > 0).sum())}/{diff.numel()}",
                    flush=True,
                )
                # Where does it diverge — early positions, late positions, or a stride?
                per_position = diff.reshape(-1, host_expected.shape[-1]).max(dim=-1).values
                bad = (per_position > 0).nonzero().reshape(-1)[:24].tolist()
                print(f"[probe] {name}: first divergent rows {bad}", flush=True)
            print(
                f"[probe] {name}: device0 sliced[0,0,:2,:4]="
                f"{_compose(expected, mesh_device).reshape(-1, host_expected.shape[-1])[:2, :4].tolist()}",
                flush=True,
            )
            print(
                f"[probe] {name}: device0 gathered[0,0,:2,:4]="
                f"{_compose(actual, mesh_device).reshape(-1, host_actual.shape[-1])[:2, :4].tolist()}",
                flush=True,
            )
    finally:
        handle.close()
