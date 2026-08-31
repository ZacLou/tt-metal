# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive int32 true-division sweep (companion to test_div_fp32.py).

ttnn.div on int32 inputs computes float(a) * reciprocal(float(b)) and returns a
float32 true-division result. The SFPU reciprocal is only ~1 ulp accurate, so
without the residual (remainder) correction in calculate_div_int32 exact
quotients land 1 ulp off and disagree across architectures (28/14 -> 1.9999998
on WH, 2.0000001 on BH). This sweep quantifies the quotient quality over a wide
(a, b) grid and specifically measures bit-exactness of exact integer quotients.

Two references are used:
  * "vs_fp32in": round_fp32( fp64(fp32(a)) / fp64(fp32(b)) ) -- isolates the
    division itself from the unavoidable int32->fp32 input rounding, so it is the
    fair ceiling for the kernel (which divides fp32-converted operands).
  * "vs_true":   round_fp32( fp64(a) / fp64(b) ) -- the exact int ratio rounded
    to fp32; for |operand| > 2**24 this also includes input-conversion error.
"""

import time

import torch

import ttnn


def compute_ulp_distance_f32(tensor_a, tensor_b):
    """ULP distance between two float32 tensors via signed-magnitude ordinals."""
    a_f32 = tensor_a.to(torch.float32).contiguous()
    b_f32 = tensor_b.to(torch.float32).contiguous()

    a_bits = a_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    b_bits = b_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_a = (a_bits >> 31) & 1
    sign_b = (b_bits >> 31) & 1
    a_ord = torch.where(sign_a == 0, a_bits + 0x80000000, 0x80000000 - a_bits)
    b_ord = torch.where(sign_b == 0, b_bits + 0x80000000, 0x80000000 - b_bits)
    ulp_dist = (a_ord - b_ord).abs()
    ulp_dist = torch.where((a_f32 == 0) & (b_f32 == 0), torch.zeros_like(ulp_dist), ulp_dist)
    return ulp_dist


def _build_int32_value_set():
    INT_MAX = 2147483647
    vals = []

    # Powers of two and their +/-1 neighbors, both signs.
    for e in range(0, 31):
        p = 1 << e
        for v in (p - 1, p, p + 1):
            vals.extend([v, -v])

    # fp32 exact-integer boundary (2**24) and other notable boundaries.
    for v in (16777215, 16777216, 16777217, 1 << 30, INT_MAX, INT_MAX - 1):
        vals.extend([v, -v])

    # Dense small integers (exercises many exact quotients).
    vals.extend(range(-300, 301))

    # Random values across the full int32 range, both signs.
    g = torch.Generator().manual_seed(0)
    rand = torch.randint(1, INT_MAX, (1500,), generator=g, dtype=torch.int64)
    vals.extend(rand.tolist())
    vals.extend((-rand).tolist())

    t = torch.tensor(vals, dtype=torch.int64)
    t = torch.unique(t)  # sorted, deduped
    return t


def test_div_int32_exhaustive_sweep(device):
    value_set = _build_int32_value_set()
    N = value_set.numel()

    # Divisor set excludes 0 (int32 has no inf/nan; div-by-zero handled elsewhere).
    divisor_set = value_set[value_set != 0].contiguous()
    ND = divisor_set.numel()
    total_pairs = N * ND
    print(f"\nint32 div sweep: {N} numerators x {ND} divisors = {total_pairs:,} pairs")

    refs = ("vs_fp32in", "vs_true")
    max_ulp = {r: 0 for r in refs}
    ulp_hist = {r: {"0": 0, "1": 0, "2": 0, "3_10": 0, "11_100": 0, ">100": 0} for r in refs}
    worst = {r: None for r in refs}  # (ulp, a, b, tt, ref)

    # Exact-quotient bit-exactness tracking -- this is what the fix targets.
    # "strict" = the quotient AND both operands are fp32-representable, so the correctly-rounded
    # result is the exact integer and the kernel MUST return it bit-for-bit. Quotients whose true
    # value exceeds 2**24 (or operands that aren't fp32-exact) can't be represented and are tracked
    # separately as informational only.
    strict_total = 0
    strict_bitexact = 0
    strict_worst = None  # (ulp, a, b, tt, ref)
    unrep_total = 0
    unrep_bitexact = 0

    def bucket(hist, u):
        hist["0"] += int((u == 0).sum())
        hist["1"] += int((u == 1).sum())
        hist["2"] += int((u == 2).sum())
        hist["3_10"] += int(((u >= 3) & (u <= 10)).sum())
        hist["11_100"] += int(((u >= 11) & (u <= 100)).sum())
        hist[">100"] += int((u > 100).sum())

    y_row = divisor_set.unsqueeze(0)  # [1, ND]
    y_f32_row = divisor_set.to(torch.float32).unsqueeze(0)
    y_f64_row = divisor_set.to(torch.float64).unsqueeze(0)
    y_f32in_row = y_f32_row.to(torch.float64)

    batch_rows = 64
    num_batches = (N + batch_rows - 1) // batch_rows
    start = time.time()

    for bi in range(num_batches):
        s = bi * batch_rows
        e = min(s + batch_rows, N)
        a_col = value_set[s:e].unsqueeze(1)  # [B, 1]
        B = e - s

        a_grid = a_col.expand(B, ND).contiguous()
        b_grid = y_row.expand(B, ND).contiguous()

        # References
        a_f32 = a_col.to(torch.float32)
        ref_fp32in = (a_f32.to(torch.float64) / y_f32in_row).to(torch.float32).expand(B, ND).contiguous()
        ref_true = (a_col.to(torch.float64) / y_f64_row).to(torch.float32).expand(B, ND).contiguous()

        # Device compute (int32 -> float32 true division)
        a_tt = ttnn.from_torch(a_grid.to(torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b_grid.to(torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        tt = ttnn.to_torch(ttnn.div(a_tt, b_tt)).to(torch.float32)[:B, :ND].contiguous()

        for r, ref in (("vs_fp32in", ref_fp32in), ("vs_true", ref_true)):
            u = compute_ulp_distance_f32(ref, tt)
            bucket(ulp_hist[r], u)
            m = int(u.max())
            if m > max_ulp[r]:
                idx = (u == m).nonzero()[0].tolist()
                max_ulp[r] = m
                worst[r] = (
                    m,
                    int(a_grid[idx[0], idx[1]]),
                    int(b_grid[idx[0], idx[1]]),
                    float(tt[idx[0], idx[1]]),
                    float(ref[idx[0], idx[1]]),
                )

        # Exact integer quotient subset. Split into "strict" (operands + quotient fp32-representable,
        # so tt MUST be bit-exact) and "unrepresentable" (true quotient not fp32-exact -> informational).
        a64 = a_grid.to(torch.int64)
        b64 = b_grid.to(torch.int64)
        divisible = (a64 % b64) == 0
        q64 = torch.where(divisible, a64 // b64, torch.zeros_like(a64))
        a_rep = a_grid.to(torch.float32).to(torch.int64) == a64
        b_rep = b_grid.to(torch.float32).to(torch.int64) == b64
        q_rep = q64.to(torch.float32).to(torch.int64) == q64
        u_exact = compute_ulp_distance_f32(ref_true, tt)

        strict_mask = divisible & a_rep & b_rep & q_rep
        strict_total += int(strict_mask.sum())
        strict_bitexact += int((strict_mask & (u_exact == 0)).sum())
        miss = strict_mask & (u_exact != 0)
        if bool(miss.any()):
            mu = int(u_exact[miss].max())
            if strict_worst is None or mu > strict_worst[0]:
                midx = (miss & (u_exact == mu)).nonzero()[0].tolist()
                strict_worst = (
                    mu,
                    int(a_grid[midx[0], midx[1]]),
                    int(b_grid[midx[0], midx[1]]),
                    float(tt[midx[0], midx[1]]),
                    float(ref_true[midx[0], midx[1]]),
                )

        unrep_mask = divisible & ~strict_mask
        unrep_total += int(unrep_mask.sum())
        unrep_bitexact += int((unrep_mask & (u_exact == 0)).sum())

        el = time.time() - start
        last = bi == num_batches - 1
        print(
            f"[{100.0*e/N:5.1f}%] batch {bi+1}/{num_batches} | {el:.1f}s | "
            f"max_ulp(vs_fp32in)={max_ulp['vs_fp32in']} vs_true={max_ulp['vs_true']}",
            end="\n" if last else "\r",
            flush=True,
        )

    print(f"\nPairs compared: {total_pairs:,}")
    for r in refs:
        h = ulp_hist[r]
        denom = max(sum(h.values()), 1)
        print(f"\n--- ULP {r} ---  max={max_ulp[r]}")
        for k in ("0", "1", "2", "3_10", "11_100", ">100"):
            print(f"  ULP {k:>6}: {h[k]:>14,} ({100.0*h[k]/denom:7.4f}%)")
        if worst[r]:
            u, a, b, ttv, refv = worst[r]
            print(f"  worst: {a} / {b}  tt={ttv!r}  ref={refv!r}  ({u} ulp)")

    sbx_pct = 100.0 * strict_bitexact / max(strict_total, 1)
    print(f"\n--- exact integer quotients ---")
    print(f"  strict (operands+quotient fp32-representable): {strict_total:,}")
    print(f"    bit-exact: {strict_bitexact:,} ({sbx_pct:.4f}%)")
    if strict_worst:
        u, a, b, ttv, refv = strict_worst
        print(f"    worst miss: {a} / {b}  tt={ttv!r}  ref={refv!r}  ({u} ulp)")
    ubx_pct = 100.0 * unrep_bitexact / max(unrep_total, 1)
    print(f"  unrepresentable quotient/operands (informational): {unrep_total:,}")
    print(f"    bit-exact: {unrep_bitexact:,} ({ubx_pct:.4f}%)  <- fp32 can't hold these exactly")

    # Guarantee 1: division quality (isolated from int->fp32 input rounding) is within 1 ulp of
    # the correctly-rounded quotient everywhere. This is the property the residual correction buys.
    assert max_ulp["vs_fp32in"] <= 1, f"max ULP vs fp32-input reference = {max_ulp['vs_fp32in']} (> 1)"
    # Guarantee 2: representable exact quotients are overwhelmingly bit-exact. A few (~0.03%) remain
    # 1 ulp off because the fp32 residual (a - q*b) isn't perfectly exact -- the same limitation as
    # the fp32 reciprocal-refine path. Pre-fix, essentially none of these were bit-exact.
    assert sbx_pct >= 99.0, f"exact-quotient bit-exact rate {sbx_pct:.4f}% < 99%; worst={strict_worst}"
    assert strict_worst is None or strict_worst[0] <= 1, f"an exact quotient was off by >1 ulp: {strict_worst}"
