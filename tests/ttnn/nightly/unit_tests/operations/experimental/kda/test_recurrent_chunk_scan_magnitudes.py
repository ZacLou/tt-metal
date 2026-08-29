# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does `recurrent_chunk_scan` hold up outside the magnitudes its suite generates?

`host_protocol` pins every tensor to one scale — `final_decay` to [0.86, 0.94], the strict-lower
block feeding `t_inv` to 0.015, `kd`/`k_dec_t` to 0.025 — and every existing test draws from it. So
the op is covered at exactly one point in a seven-dimensional magnitude space.

That matters because Kimi-K3's real weights land somewhere else in that space, and there the KDA
layer disagrees with its own torch reference completely: layer 0 scores 0.99993 against the model's
recorded output while all 68 other KDA layers score ~0. The failing variable is the decay magnitude
and nothing else — shifting layer 0's `dt_bias` down by 1.0 takes it from 0.99994 to -0.005, and
shifting layer 1's up by 3.0 takes it from 0.004 to 0.99905. Dtype is not the cause (forcing every
preparation output to fp32 changes nothing), nor is math fidelity, chunk grouping, or sequence
length: it is already wrong with a single chunk per rank.

This sweeps one protocol scale at a time against the suite's own oracle, so a failure here is a
reproduction with no model, no checkpoint and no mesh in it.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    CHUNK_SIZE,
    device_protocol,
    host_protocol,
    initial_state,
    recurrent_oracle,
    run_recurrent,
    to_device,
)

pytestmark = [run_for_blackhole()]

BATCH_HEADS, NUM_CHUNKS, KEY_DIM, VALUE_DIM = 8, 4, 128, 128


def _scaled(protocol, index, factor):
    out = list(protocol)
    out[index] = out[index] * factor
    return tuple(out)


def _with_final_decay(protocol, base):
    """Replace the [0.86, 0.94] band with one centred on `base`, keeping the same spread shape."""
    out = list(protocol)
    generator = torch.Generator().manual_seed(99)
    out[5] = (base + 0.08 * torch.rand(*out[5].shape, generator=generator)).float()
    return tuple(out)


@run_for_blackhole()
def test_scan_matches_oracle_across_magnitudes(device):
    state = initial_state(BATCH_HEADS, KEY_DIM, VALUE_DIM)
    base_protocol = host_protocol(BATCH_HEADS, NUM_CHUNKS, KEY_DIM, VALUE_DIM)

    def check(label, protocol):
        want_out, want_state = recurrent_oracle(protocol, state)
        got = run_recurrent(device_protocol(protocol, device), to_device(state, device))
        pcc_out = float(str(comp_pcc(want_out.float(), ttnn.to_torch(got[0]).float(), 0.99)[1]).split()[-1])
        pcc_state = float(str(comp_pcc(want_state, ttnn.to_torch(got[1]).float(), 0.99)[1]).split()[-1])
        logger.info(f"  {label:38s} output {pcc_out:9.5f}   state {pcc_state:9.5f}")

    check("suite default", base_protocol)
    for base in (0.86, 0.6, 0.4, 0.2, 0.05, 0.98):
        check(f"final_decay base {base:.2f}", _with_final_decay(base_protocol, base))
    for name, index in (("v_beta", 0), ("kd", 1), ("q_decay", 2), ("intra", 3), ("k_dec_t", 4), ("t_inv", 6)):
        for factor in (4.0, 16.0):
            check(f"{name} x{factor:g}", _scaled(base_protocol, index, factor))
