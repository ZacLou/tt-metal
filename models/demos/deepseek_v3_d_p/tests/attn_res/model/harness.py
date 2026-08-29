# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement, composition and one block's reads — what every device gate here needs
before it can measure anything.

The shape is fixed rather than parametrized. `d = 7168` at 640 rows per chip is what
prefill runs — it chunks 5120 tokens across an 8-deep sequence axis on the Galaxy — and
the op's cost and its collective's choice of algorithm both turn on that row count, so a
gate parametrized at 64 tokens exercises a reduction the model never issues.

Inputs are drawn from a caller-supplied generator rather than a seed, so a file that
draws several things from one stream keeps its own draw order and stays reproducible
across changes here.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

HIDDEN_SIZE = 7168
PER_CHIP_TOKENS = 640

# On a unit-RMS stream the scores are `N(0, ‖q‖₂²)`, so the folded query's norm is the
# softmax's temperature and the only thing about the weights this op can be sensitive to.
# This lands at `‖q‖₂ ≈ 1.7`; K3's own query weights run 0.07 to 0.23 over a block, which
# is a near-uniform softmax and a milder shift for the online rescale to carry. The scale
# here is kept above the checkpoint's deliberately — it is the harder of the two.
PROJ_STD = 0.02

# `ttnn.all_reduce` and the gather kernel's own fabric writes both need an initialized
# fabric context on a real mesh; without it they die in the control plane rather than
# returning wrong numbers. 2D is what the rest of this model runs on and what the op's
# own unit test pins, and the two must agree — the op picks its route from the config.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}

# Kimi-K3's own profile. This is NOT the `(2, 4)` arm at a wider sequence axis, which the op
# genuinely is indifferent to: TorusXY physically wraps both axes, so `per_axis_topology()`
# returns `(Ring, Ring)` and the read's exchange runs as a ring rather than a line. That is a
# different collective on the *tensor* axis, which is the axis this op is built around. It is also
# the only placement a Blackhole Galaxy can open at all — a sub-mesh request there does not skip,
# it dies in `Fabric Router Sync` after ten seconds — so without this arm AttnRes has no coverage
# whatsoever on the box Kimi-K3 runs on.
# `fabric_payload_size` is not a detail. Left unset, `torus_xy_device_params` opens the fabric at
# `get_max_payload_size()`; every model test in this package passes its own
# `Config.FABRIC_PAYLOAD_SIZE` instead, and Kimi-K3's is 7168 (`kimi_k3_config.py`, kept in sync with
# the migration code). Opening at a different payload than the model does is a different fabric, and
# the op writes to it directly.
TORUS_XY = torus_xy_device_params(fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE)
TORUS_XY_TRACED = torus_xy_device_params(
    fabric_payload_size=KimiK3Config.FABRIC_PAYLOAD_SIZE, trace_region_size=23887872
)

# `mesh_device` skips a placement asking for more chips than the host has, so a box holding
# neither shape collects these and skips rather than failing.
GALAXY_MARK = pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4")

PLACEMENTS = [
    pytest.param((2, 4), FABRIC, id="mesh-2x4"),
    # Plain Fabric2D at Galaxy width. Held alongside the torus arm to separate the two variables the
    # Galaxy changes at once — mesh width and fabric wrap — because they fail differently.
    pytest.param((8, 4), FABRIC, marks=GALAXY_MARK, id="fabric2d-8x4"),
    pytest.param((8, 4), TORUS_XY, marks=GALAXY_MARK, id="torus-xy-8x4"),
]


def mesh_topology(mesh_device):
    """The per-axis CCL topology of the fabric that is actually open, as `TtAttnRes` wants it.

    `TtAttnRes` takes one `ttnn.Topology` per mesh axis and defaults to all-Linear. On TorusXY that
    default is wrong in a way nothing downstream reports: the collective would be issued as a line
    on an axis the fabric physically wraps. Querying the live fabric keeps the op on the same route
    every other module in this model takes — `MLAPrefillAdapter.build_runtime` does the same.
    """
    return list(per_axis_topology())[: len(tuple(mesh_device.shape))]


# The op was brought up and measured only on Blackhole, and its mixture runs on
# `ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`, which has no Wormhole coverage.
blackhole_only = pytest.mark.skipif(not is_blackhole(), reason="Kimi K3 AttnRes is brought up on Blackhole only")


def generator(seed=0):
    return torch.Generator().manual_seed(seed)


def random_hidden(rng, num_tokens):
    return torch.randn(num_tokens, HIDDEN_SIZE, generator=rng)


def random_case(rng, num_tokens, num_sealed):
    """One read's inputs: the live stream and `num_sealed` frozen snapshots."""
    return random_hidden(rng, num_tokens), torch.randn(num_tokens, num_sealed, HIDDEN_SIZE, generator=rng)


def random_queries(rng, count):
    """`count` folded queries, each a norm weight times a projection row."""
    randn = lambda: torch.randn(HIDDEN_SIZE, generator=rng)
    return [(1.0 + 0.1 * randn()) * (PROJ_STD * randn()) for _ in range(count)]


def reference_block_reads(running_sum, block_residual, queries, eps=EPS):
    """Every read site of one block on host, materializing the candidate set once.

    Algebraically identical to calling `attn_res` per site — `test_attn_res.py` still scores against
    `attn_res` itself, so the two cannot drift — but it hoists the two loop-invariant parts out of
    the block: `attn_res` rebuilds `cat(block_residual, running_sum).float()` on every call, and
    `(v * q).sum(-1)` materializes a second `[N, S+1, d]` fp32 tensor to reduce it away. Neither
    depends on the query.

    Measured at the Galaxy arm's shape (N=5120, S=8, 24 sites): 2.0 s here against ~10 s for the
    per-site form, and no 1.3 GB temporaries. `candidates @ query` produces the `[N, S+1]` scores
    directly. Small in absolute terms — the first run of this test is dominated by JIT-linking the
    gather-softmax kernel, not by host arithmetic — but it is per-test-run forever after, and the
    per-site form scales with sites x tokens for no reason.
    """
    candidates = torch.cat((block_residual, running_sum.unsqueeze(1)), dim=1).float()
    rms_inv = torch.rsqrt(candidates.pow(2).mean(-1) + eps)
    for query in queries:
        scores = torch.matmul(candidates, query.float()) * rms_inv
        probs = scores.softmax(-1)
        yield torch.matmul(probs.unsqueeze(1), candidates).squeeze(1).to(running_sum.dtype)


def place(op, tensor, mesh_mapper=None):
    """One host tensor onto the mesh, in the stream's dtype and layout."""
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=op.mesh_device,
        mesh_mapper=op.stream_mapper if mesh_mapper is None else mesh_mapper,
    )


def place_case(op, running_sum, block_residual, mesh_mapper=None):
    """The live stream as `[1, 1, N, d]` and the sealed set as `[1, S, N, d]`.

    The reference holds the sealed set `[N, S, d]` and the read batches over `S`, so the
    two leading axes swap on the way across. An empty sealed set has no device tensor at
    all — `merge` takes None there rather than a zero-wide operand.
    """
    stream = place(op, running_sum.unsqueeze(0).unsqueeze(0), mesh_mapper)
    sealed = place(op, block_residual.permute(1, 0, 2).unsqueeze(0), mesh_mapper) if block_residual.shape[1] else None
    return stream, sealed


def compose(op, tensor):
    """A device tensor back on host as `[rows, d]`, its shards joined."""
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, HIDDEN_SIZE)


def read_block(op, tt_block, tt_prefix, tt_queries):
    """Every read site of one block on host, in the order the walk issues them.

    One `inter_block` over the sealed set, then a `merge` per site: the sealed half is
    loop-invariant across a block and only the live half is not. Each site is freed as
    soon as it is composed, so the block holds one output at a time — a caller that needs
    them together builds its own list.
    """
    partials, shifts, masses = op.inter_block(tt_block, tt_queries)
    try:
        for site, tt_query in enumerate(tt_queries):
            merged = op.merge(partials, shifts, masses, tt_prefix, tt_query, site)
            yield compose(op, merged)
            ttnn.deallocate(merged)
    finally:
        for tensor in (partials, shifts, masses):
            ttnn.deallocate(tensor)
