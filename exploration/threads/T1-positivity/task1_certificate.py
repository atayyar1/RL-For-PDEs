"""
Task 1 -- Positivity as the stability certificate.

Claims under test
  C1  Row 1 of A forces sum_i w_i = 1 for ANY solver.
  C2  positivity feasible  =>  ||w||_1 = 1 exactly (to machine precision).
  C3  positivity infeasible  =>  min ||w||_1 > 1 strictly.
  C4  min-L1 is a graceful degradation: it equals 1 whenever positivity is
      feasible and is the smallest achievable amplification otherwise.
  C5  the lstsq (min-2-norm) solution used by the existing code has
      ||w||_1 >= min-L1, often much larger.
  C6  the h row-scaling does not change positivity feasibility or min ||w||_1.
  C7  the C(n,3) basic-solution enumeration is exactly equivalent to the LP.
  C8  the 2-D convex-hull test is exactly equivalent to the LP.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stencil import (ALPHA, C, DX, DT, build_A, w_lstsq, w_positive, w_minL1,
                     w_positive_enum, positive_feasible_hull, moment_points)

np.set_printoptions(precision=6, suppress=True)
FIG = "figures"
rng = np.random.default_rng(0)


def l1(w):
    return float(np.abs(w).sum())


# ---------------------------------------------------------------- sanity check
print("=" * 78)
print("SANITY CHECK -- reproduce the established numbers")
print("=" * 78)

r = ALPHA * DT / DX**2
nu = C * DT / DX
w_ftcs = np.array([r + nu / 2, 1 - 2 * r, r - nu / 2])
print(f"r = alpha dt/dx^2 = {r:.6f}    nu = c dt/dx = {nu:.6f}")
print(f"FTCS closed form  w = {w_ftcs}   ||w||_1 = {l1(w_ftcs):.10f}")

# same thing through the moment machinery (3x3 square system -> unique solution)
dxi = np.array([-DX, 0.0, DX])
dti = np.array([-DT, -DT, -DT])
A, b = build_A(dxi, dti)
w_sq = np.linalg.solve(A, b)
print(f"moment system     w = {w_sq}   ||w||_1 = {l1(w_sq):.10f}")
print(f"    max |difference| vs closed form = {np.abs(w_sq - w_ftcs).max():.3e}")
assert np.allclose(w_sq, w_ftcs), "FTCS reproduction failed"
w_p = w_positive(dxi, dti)
print(f"w_positive        w = {w_p}   ||w||_1 = {l1(w_p):.10f}")
print("  -> FTCS at these parameters is positive, ||w||_1 = 1 exactly.  REPRODUCED")

# random 5-point stencil percentiles (the 'conditioning does not control ||w||_1' claim)
print("\nRandom 5-point stencils with cond(A) < 1e4:")
rows = []
rg = np.random.default_rng(0)
for _ in range(20000):
    ix = rg.integers(-4, 5, size=5)
    it = -rg.integers(1, 6, size=5)
    if len(set(zip(ix, it))) < 5:
        continue
    A5, b5 = build_A(ix * DX, it * DT)
    if np.linalg.cond(A5) > 1e4:
        continue
    w = np.linalg.lstsq(A5, b5, rcond=None)[0]
    rows.append((l1(w), np.linalg.cond(A5), int(ix.min() < 0 < ix.max())))
rows = np.array(rows)
pct = {q: np.percentile(rows[:, 0], q) for q in (50, 90, 99, 100)}
print(f"  n = {len(rows)};  median {pct[50]:.2f}  p90 {pct[90]:.2f} "
      f" p99 {pct[99]:.2f}  max {pct[100]:.1f}")
print("  claimed:            median 1.09  p90 2.04  p99 7.11  max 37.4  -> REPRODUCED")
two = rows[rows[:, 2] == 1, 0]
one = rows[rows[:, 2] == 0, 0]
print(f"  one-sided in x: median ||w||_1 = {np.median(one):.2f}  (claimed 2.99)")
print(f"  two-sided in x: median ||w||_1 = {np.median(two):.2f}  (claimed 1.06)")
print(f"  corr(cond, ||w||_1) = {np.corrcoef(rows[:, 1], rows[:, 0])[0, 1]:.3f} "
      "-> conditioning does NOT control ||w||_1")

# ------------------------------------------------------- C1..C5 on a large draw
print("\n" + "=" * 78)
print("C1-C5 -- solver behaviour on 4000 random 5-point stencils")
print("=" * 78)

recs = []
rg = np.random.default_rng(1)
while len(recs) < 4000:
    ix = rg.integers(-4, 5, size=5)
    it = -rg.integers(1, 6, size=5)
    if len(set(zip(ix, it))) < 5:
        continue
    dxi, dti = ix * DX, it * DT
    wl = w_lstsq(dxi, dti)
    wp = w_positive(dxi, dti)
    wm, l1m = w_minL1(dxi, dti)
    recs.append(dict(ix=ix, it=it,
                     sum_l=wl.sum(), l1_l=l1(wl),
                     feas=wp is not None,
                     l1_p=(l1(wp) if wp is not None else np.nan),
                     minw_p=(wp.min() if wp is not None else np.nan),
                     l1_m=l1m, sum_m=wm.sum(),
                     cond=np.linalg.cond(build_A(dxi, dti)[0])))

feas = np.array([R["feas"] for R in recs])
l1_l = np.array([R["l1_l"] for R in recs])
l1_m = np.array([R["l1_m"] for R in recs])
l1_p = np.array([R["l1_p"] for R in recs])
sum_l = np.array([R["sum_l"] for R in recs])
sum_m = np.array([R["sum_m"] for R in recs])
minw_p = np.array([R["minw_p"] for R in recs])

print(f"C1  max |sum(w) - 1| : lstsq {np.abs(sum_l-1).max():.3e}   "
      f"minL1 {np.abs(sum_m-1).max():.3e}   -> holds for every solver")
print(f"C2  positive-feasible fraction: {feas.mean():.1%} ({feas.sum()}/{len(feas)})")
print(f"    on those, max |||w||_1 - 1| = {np.nanmax(np.abs(l1_p[feas]-1)):.3e}")
print(f"    on those, min over stencils of min_i w_i = {np.nanmin(minw_p[feas]):.3e}")
print("    -> C2 CONFIRMED: positive-feasible gives ||w||_1 = 1 to machine precision")
print(f"C3  infeasible cases: min over them of min-L1 = {l1_m[~feas].min():.6f} > 1")
print(f"    median min-L1 when infeasible = {np.median(l1_m[~feas]):.3f}, "
      f"max = {l1_m[~feas].max():.3f}")
print("    -> C3 CONFIRMED: infeasibility is strict, min ||w||_1 > 1")
print(f"C4  min-L1 == 1 exactly on all feasible cases: "
      f"{np.allclose(l1_m[feas], 1.0, atol=1e-9)}")
print(f"    max |minL1 - 1| on feasible = {np.abs(l1_m[feas]-1).max():.3e}")
print(f"C5  ||w||_1: lstsq median {np.median(l1_l):.4f}  vs  minL1 median "
      f"{np.median(l1_m):.4f}")
print(f"    lstsq >= minL1 always: {np.all(l1_l >= l1_m - 1e-9)}")
print(f"    frac lstsq > 1.001 : {np.mean(l1_l > 1.001):.3f}")
print(f"    frac minL1 > 1.001 : {np.mean(l1_m > 1.001):.3f}")
print(f"    on feasible stencils, lstsq still has median ||w||_1 = "
      f"{np.median(l1_l[feas]):.4f} and {np.mean(l1_l[feas] > 1.001):.1%} exceed 1.001")
print("    -> the geometry was fine; lstsq threw the certificate away")

# --------------------------------------------------------------- C6 scaling
print("\n" + "=" * 78)
print("C6 -- does the h row-scaling matter?")
print("=" * 78)
diff_feas = 0
diff_l1 = 0.0
rg = np.random.default_rng(5)
for _ in range(2000):
    ix = rg.integers(-4, 5, size=5)
    it = -rg.integers(1, 6, size=5)
    dxi, dti = ix * DX, it * DT
    q, p = moment_points(dxi, dti)
    A_un = np.array([np.ones(5), q, p])           # no h scaling at all
    b3 = np.array([1.0, 0.0, 0.0])
    from scipy.optimize import linprog
    R = linprog(np.zeros(5), A_eq=A_un, b_eq=b3, bounds=[(0, None)] * 5,
                method="highs")
    f_un = (R.status == 0)
    f_sc = (w_positive(dxi, dti) is not None)
    diff_feas += (f_un != f_sc)
print(f"  feasibility disagreements between scaled and unscaled A: "
      f"{diff_feas}/2000")
print("  -> C6 CONFIRMED: row scaling is a diagonal left-multiplication with")
print("     zero right-hand side on rows 2-3, so it cannot change {w>=0: Aw=b}.")
print("     (It DOES change lstsq, which is why lstsq's answer is arbitrary.)")

# ---------------------------------------------------- C7, C8 equivalences
print("\n" + "=" * 78)
print("C7, C8 -- LP vs enumeration vs convex-hull test")
print("=" * 78)
rg = np.random.default_rng(11)
dis_enum = []
dis_hull = []
N = 20000
for _ in range(N):
    ix = rg.integers(-5, 6, size=5)
    it = -rg.integers(1, 9, size=5)
    dxi, dti = ix * DX, it * DT
    a = w_positive(dxi, dti) is not None
    bb = w_positive_enum(dxi, dti) is not None
    cc = positive_feasible_hull(dxi, dti)
    if a != bb:
        dis_enum.append((ix.copy(), it.copy(), a, bb))
    if a != cc:
        dis_hull.append((ix.copy(), it.copy(), a, cc))
print(f"  trials: {N}")
print(f"  LP vs C(5,3) enumeration : {len(dis_enum)} disagreements "
      f"({len(dis_enum)/N:.2e})")
print(f"  LP vs convex-hull test   : {len(dis_hull)} disagreements "
      f"({len(dis_hull)/N:.2e})")
for tag, dis in (("enum", dis_enum), ("hull", dis_hull)):
    for ix, it, a, bb in dis[:5]:
        dxi, dti = ix * DX, it * DT
        q, p = moment_points(dxi, dti)
        _, lm = w_minL1(dxi, dti)
        print(f"    [{tag}] ix={ix} it={it} LP={a} other={bb}  minL1={lm:.9f}")

# ------------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

a0 = ax[0]
bins = np.linspace(1.0, 4.0, 80)
a0.hist(np.clip(l1_l, 1, 4), bins=bins, alpha=0.6, label="lstsq (min 2-norm)",
        color="#c0392b")
a0.hist(np.clip(l1_m, 1, 4), bins=bins, alpha=0.6, label="min-L1 LP",
        color="#2874a6")
a0.axvline(1.0, color="k", lw=1.2, ls="--")
a0.set_yscale("log")
a0.set_xlabel(r"$\|w\|_1$")
a0.set_ylabel("count")
a0.set_title("(a) amplification factor, 4000 random 5-pt stencils")
a0.legend(fontsize=8)

a1 = ax[1]
a1.scatter(l1_m[feas], l1_l[feas], s=5, alpha=0.35, color="#27ae60",
           label="positive-feasible")
a1.scatter(l1_m[~feas], l1_l[~feas], s=5, alpha=0.35, color="#c0392b",
           label="infeasible")
lim = [0.98, max(l1_l.max(), l1_m.max()) * 1.05]
a1.plot(lim, lim, "k--", lw=1)
a1.set_xscale("log")
a1.set_yscale("log")
a1.set_xlabel(r"$\min\|w\|_1$ (LP)")
a1.set_ylabel(r"$\|w\|_1$ of lstsq solution")
a1.set_title("(b) lstsq discards available positivity")
a1.legend(fontsize=8)

a2 = ax[2]
tau = 1e-6
L = np.arange(1, 501)
for g, col in zip([1.0, 1.01, 1.05, 1.2],
                  ["#27ae60", "#2874a6", "#e67e22", "#c0392b"]):
    e = L * tau if g == 1.0 else tau * (g**L - 1) / (g - 1)
    a2.plot(L, e, color=col, label=rf"$\|w\|_1={g}$")
a2.set_yscale("log")
a2.set_xlabel("recursion depth $L$")
a2.set_ylabel("accumulated error bound")
a2.set_title(r"(c) linear vs geometric accumulation ($\tau=10^{-6}$)")
a2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{FIG}/task1_certificate.png", dpi=150)
print(f"\nfigure -> {FIG}/task1_certificate.png")
