"""
Task 5 -- What does the positivity certificate cost?

The LP has to run inside an RL environment step, so per-solve latency matters.
We compare, on 5-point stencils:

  lstsq            np.linalg.lstsq on the 3x5 system (what the code does now)
  linprog-feas     scipy HiGHS feasibility LP  (w >= 0, A w = b)
  linprog-minL1    scipy HiGHS min-L1 LP (the 3x10 split)
  enum C(5,3)      enumerate all 10 3x3 basic solutions
  hull             the closed-form 2-D convex-hull test (angles + sort)
  hull-batch       the same test vectorised over a batch of stencils
  P1&P2            the O(n) necessary condition (min/max only, no transcendentals)

and check exact equivalence of the three exact positivity tests.
"""
import itertools
import time

import numpy as np
from scipy.optimize import linprog

from stencil import (ALPHA, C, DX, DT, build_A, moment_points, w_lstsq,
                     w_positive, w_minL1, w_positive_enum,
                     positive_feasible_hull)

rng = np.random.default_rng(42)

# a fixed pool of test stencils
NPOOL = 4000
POOL = []
while len(POOL) < NPOOL:
    ix = rng.integers(-5, 6, size=5)
    it = -rng.integers(1, 13, size=5)
    if len(set(zip(ix, it))) < 5:
        continue
    POOL.append((ix * DX, it * DT))


def bench(fn, pool, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        for dxi, dti in pool:
            fn(dxi, dti)
    t1 = time.perf_counter()
    return (t1 - t0) / (len(pool) * reps)


def p1p2(dxi, dti):
    q, p = moment_points(dxi, dti)
    return (q.min() <= 0 <= q.max()) and (p.min() <= 0 <= p.max())


print("=" * 78)
print("PER-SOLVE COST  (5-point stencils, 3-row moment system)")
print("=" * 78)
results = {}
for name, fn, reps in [
    ("lstsq (min 2-norm)", lambda a, b: w_lstsq(a, b), 3),
    ("linprog feasibility LP", lambda a, b: w_positive(a, b), 1),
    ("linprog min-L1 LP", lambda a, b: w_minL1(a, b), 1),
    ("enum C(5,3) = 10 3x3 solves", lambda a, b: w_positive_enum(a, b), 3),
    ("hull test (exact, no solve)", lambda a, b: positive_feasible_hull(a, b), 5),
    ("P1&P2 necessary cond.", p1p2, 5),
]:
    t = bench(fn, POOL, reps)
    results[name] = t
    print(f"  {name:<32} {t*1e6:9.2f} us/solve")

base = results["linprog feasibility LP"]
print("\n  relative to the feasibility LP:")
for k, v in results.items():
    print(f"    {k:<32} {base/v:8.1f}x faster" if v < base
          else f"    {k:<32} {v/base:8.1f}x slower")

# ---------------------------------------------------------------- batched hull
print("\n" + "=" * 78)
print("BATCHED hull test (what an RL action mask would actually call)")
print("=" * 78)


def hull_batch(DXI, DTI, alpha=ALPHA, c=C):
    """Vectorised positivity test.  DXI, DTI: (B, n) arrays.  Returns (B,) bool.

    Same test as positive_feasible_hull, evaluated for a whole batch of
    candidate stencils at once.
    """
    Q = DXI - c * DTI
    P = 0.5 * DXI**2 + alpha * DTI
    Q = Q / np.maximum(np.abs(Q).max(1, keepdims=True), 1e-300)
    P = P / np.maximum(np.abs(P).max(1, keepdims=True), 1e-300)
    at_origin = (np.hypot(Q, P) <= 1e-12).any(1)
    TH = np.sort(np.arctan2(P, Q), axis=1)
    G = np.diff(np.concatenate([TH, TH[:, :1] + 2 * np.pi], axis=1), axis=1)
    return at_origin | (G.max(1) <= np.pi + 1e-12)


for B in [64, 512, 4096]:
    DXI = np.array([POOL[i % NPOOL][0] for i in range(B)])
    DTI = np.array([POOL[i % NPOOL][1] for i in range(B)])
    hull_batch(DXI, DTI)                       # warm
    t0 = time.perf_counter()
    for _ in range(20):
        hull_batch(DXI, DTI)
    t = (time.perf_counter() - t0) / (20 * B)
    print(f"  batch size {B:>5}: {t*1e6:8.3f} us per stencil "
          f"({base/t:9.0f}x faster than the LP)")


def p1p2_batch(DXI, DTI, alpha=ALPHA, c=C):
    Q = DXI - c * DTI
    P = 0.5 * DXI**2 + alpha * DTI
    return ((Q.min(1) <= 0) & (Q.max(1) >= 0)
            & (P.min(1) <= 0) & (P.max(1) >= 0))


DXI = np.array([POOL[i % NPOOL][0] for i in range(4096)])
DTI = np.array([POOL[i % NPOOL][1] for i in range(4096)])
p1p2_batch(DXI, DTI)
t0 = time.perf_counter()
for _ in range(20):
    p1p2_batch(DXI, DTI)
t = (time.perf_counter() - t0) / (20 * 4096)
print(f"  P1&P2 batched, B=4096: {t*1e6:8.3f} us per stencil "
      f"({base/t:9.0f}x faster than the LP)  -- but only NECESSARY")

# ------------------------------------------------------------- equivalence
print("\n" + "=" * 78)
print("EXACT EQUIVALENCE of the three positivity tests")
print("=" * 78)
rng2 = np.random.default_rng(99)
N = 60000
d_enum = d_hull = d_batch = 0
BATCH_DX, BATCH_DT, LPRES = [], [], []
for _ in range(N):
    ix = rng2.integers(-6, 7, size=5)
    it = -rng2.integers(1, 40, size=5)
    dxi, dti = ix * DX, it * DT
    a = w_positive(dxi, dti) is not None
    if (w_positive_enum(dxi, dti) is not None) != a:
        d_enum += 1
    if positive_feasible_hull(dxi, dti) != a:
        d_hull += 1
    BATCH_DX.append(dxi)
    BATCH_DT.append(dti)
    LPRES.append(a)
d_batch = int((hull_batch(np.array(BATCH_DX), np.array(BATCH_DT))
               != np.array(LPRES)).sum())
print(f"  trials: {N}   base-rate feasible: {np.mean(LPRES):.3f}")
print(f"  linprog vs enum C(5,3)  : {d_enum} disagreements")
print(f"  linprog vs hull test    : {d_hull} disagreements")
print(f"  linprog vs hull-batch   : {d_batch} disagreements")

# also verify the weights themselves, not just feasibility
maxres = 0.0
minw = np.inf
nchk = 0
for dxi, dti in POOL[:1500]:
    we = w_positive_enum(dxi, dti)
    if we is None:
        continue
    A, b = build_A(dxi, dti)
    maxres = max(maxres, float(np.abs(A @ we - b).max()))
    minw = min(minw, float(we.min()))
    nchk += 1
print(f"  enum weights on {nchk} feasible stencils: max |Aw-b| = {maxres:.2e}, "
       f"min_i w_i = {minw:.2e}")

# --------------------------------------- does a near-closed form exist for n=3?
print("\n" + "=" * 78)
print("IS THERE A CLOSED FORM?  Yes -- and it is not the C(n,3) enumeration.")
print("=" * 78)
print("""  Row 1 of A says sum_i w_i = 1, so w lives on the simplex and rows 2-3 say
  that the convex combination of the PLANAR points (q_i, p_i) is the origin:

      q_i = dx_i - c dt_i ,   p_i = 1/2 dx_i^2 + alpha dt_i   ('given' row)

  Hence   positivity feasible  <=>  0 in conv{(q_i, p_i)}_{i=1..n} ,
  a 2-D point-in-convex-hull query.  By Gordan's theorem this holds iff the
  polar angles of the (q_i, p_i) have no gap wider than pi -- one atan2 per
  neighbour plus a sort.  O(n log n), no linear algebra, no LP, exact.

  The C(n,3) enumeration IS exactly equivalent (LP basic-solution theory) but
  costs n^3-ish work per subset; the hull test is the cheaper closed form.
  The O(n) necessary screen  min_i q_i <= 0 <= max_i q_i  AND
  min_i p_i <= 0 <= max_i p_i  is cheaper still and never rejects a feasible
  stencil, but it is not sufficient.""")
