"""
Task 3 -- Where in stencil-geometry space is positivity feasible?

Three studies:
  3A  symmetric stencil, half-width m, all neighbours at one time level -k*Dt.
      Map feasibility in (m, k); compare with the claimed k_max = (m Dx)^2/(2 alpha Dt).
  3B  the same but allowing neighbours spread over several time levels.
  3C  random 5-point stencils drawn from a past window of shape (M, K):
      feasible fraction vs window shape, and cheap geometric predictors of
      feasibility with measured precision/recall (candidate RL action masks).

All feasibility calls use the convex-hull test, which task1 verified to be
exactly equivalent to the LP over 20000 random stencils; 3A re-verifies the
equivalence on the structured geometries used here.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stencil import (ALPHA, C, DX, DT, moment_points, positive_feasible_hull,
                     w_positive, w_minL1, build_A)

FIG = "figures"
R_NUM = ALPHA * DT / DX**2          # 0.45
NU = C * DT / DX                    # 0.0454545
PE = C * DX / ALPHA                 # 0.10101

print(f"r = {R_NUM:.6f}   nu = {NU:.6f}   cell-Pe = {PE:.6f}")
print(f"2/Pe = 2 alpha/(c Dx) = {2/PE:.3f}  (this will turn out to matter)")


def feas(ix, it, variant="given"):
    return positive_feasible_hull(np.asarray(ix) * DX, np.asarray(it) * DT,
                                  variant=variant)


# ======================================================================= 3A
print("\n" + "=" * 78)
print("3A -- symmetric stencil, half-width m, single time level t = -k Dt")
print("=" * 78)

ms = np.arange(1, 41)
ks = np.arange(1, 1201)
kmax_num = np.zeros(len(ms), dtype=int)
disagree = 0
for a, m in enumerate(ms):
    ix = np.arange(-m, m + 1)
    best = 0
    for k in ks:
        ok = feas(ix, -k * np.ones(len(ix), dtype=int))
        if ok:
            best = k
        elif k > best + 60:          # feasible set is a lower set here; stop early
            break
    kmax_num[a] = best

# spot-check hull == LP on these structured stencils
for m in [1, 3, 8, 20, 35]:
    ix = np.arange(-m, m + 1)
    for k in [1, kmax_num[m - 1], kmax_num[m - 1] + 1, 2 * kmax_num[m - 1] + 5]:
        k = max(int(k), 1)
        it = -k * np.ones(len(ix), dtype=int)
        if feas(ix, it) != (w_positive(ix * DX, it * DT) is not None):
            disagree += 1
print(f"  hull-vs-LP disagreements on structured spot checks: {disagree}")

k_diff = ms**2 / (2 * R_NUM)                 # (m Dx)^2 / (2 alpha Dt)
k_adv = ms / NU                              # m Dx / (c Dt)
k_vtx = 2 * ALPHA / (C**2 * DT) * np.ones_like(ms, dtype=float)
k_pred = np.minimum(np.minimum(k_diff, k_adv), k_vtx)

print(f"\n  {'m':>3} {'k_max (LP)':>11} {'(mDx)^2/(2aDt)':>15} {'mDx/(cDt)':>11} "
      f"{'2a/(c^2 Dt)':>12} {'min of three':>13}")
for a, m in enumerate(ms):
    if m in (1, 2, 3, 5, 8, 12, 16, 19, 20, 22, 25, 30, 40):
        print(f"  {m:>3} {kmax_num[a]:>11} {k_diff[a]:>15.1f} {k_adv[a]:>11.1f} "
              f"{k_vtx[a]:>12.1f} {k_pred[a]:>13.1f}")

sel = kmax_num > 0
print(f"\n  ratio k_max(LP) / (mDx)^2/(2 alpha Dt) : "
      f"m<=12 mean {np.mean(kmax_num[:12]/k_diff[:12]):.3f}, "
      f"m>=25 mean {np.mean(kmax_num[24:]/k_diff[24:]):.3f}")
print(f"  ratio k_max(LP) / min(three bounds)    : "
      f"m<=12 mean {np.mean(kmax_num[:12]/k_pred[:12]):.3f}, "
      f"m>=25 mean {np.mean(kmax_num[24:]/k_pred[24:]):.3f}")
print("\n  VERDICT: the claimed k_max = (m Dx)^2/(2 alpha Dt) is right only for")
print("  small m.  It is a DIFFUSIVE bound; for larger m the ADVECTIVE bound")
print(f"  k <= m Dx/(c Dt) and the vertex bound k <= 2 alpha/(c^2 Dt) = {k_vtx[0]:.0f}")
print("  take over, and the achievable time-step speedup SATURATES.")
print(f"  Crossover where k_diff = k_adv:  m = 2 alpha/(c Dx) = 2/Pe = {2/PE:.1f}")

# ======================================================================= 3B
print("\n" + "=" * 78)
print("3B -- neighbours spread over SEVERAL time levels")
print("=" * 78)
print("  FIRST ATTEMPT (and why it is vacuous):  ask for the largest K such that")
print("  the full past block {|j| <= m} x {1 <= k <= K} is feasible.  The answer")
print("  is 'unbounded' for every m >= 1 -- but only because that block CONTAINS")
print("  the 3-point FTCS stencil (j in {-1,0,1}, k = 1), which is feasible here")
print(f"  (r = {R_NUM:.2f} <= 1/2, cell-Pe = {PE:.3f} <= 2).  The question measures")
print("  nothing about depth.  Recorded as a null result.")
for m in [1, 5, 20, 30]:
    pts = [(j, -k) for j in range(-m, m + 1) for k in range(1, 5001)]
    jx = np.array([q_[0] for q_ in pts]); jt = np.array([q_[1] for q_ in pts])
    print(f"    m={m:>3}  block with K=5000 feasible: {feas(jx, jt)}")

print("\n  REFORMULATED.  The quantity that matters for a solver is how far BACK")
print("  IN TIME the whole stencil sits -- i.e. how big an effective step it")
print("  takes.  So constrain every neighbour to depth k >= k_min and allow a")
print("  band of D extra levels: k in [k_min, k_min + D], |j| <= m.  Largest k_min?")
print(f"  {'m':>3} " + "".join(f"{'D=' + str(D):>9}" for D in [0, 1, 3, 10, 50]) +
      f"{'m^2/(2r)':>11}")
band_tab = {}
for m in [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]:
    row = []
    for D in [0, 1, 3, 10, 50]:
        best = 0
        lo_, hi_ = 1, 4000
        # feasible k_min set is a lower set here; bisect
        def f_kmin(kmin, m=m, D=D):
            pts = [(j, -k) for j in range(-m, m + 1)
                   for k in range(int(kmin), int(kmin) + D + 1)]
            jx = np.array([q_[0] for q_ in pts]); jt = np.array([q_[1] for q_ in pts])
            return feas(jx, jt)
        if not f_kmin(1):
            row.append(0); continue
        lo_, hi_ = 1, 2
        while hi_ < 4000 and f_kmin(hi_):
            lo_, hi_ = hi_, hi_ * 2
        while hi_ - lo_ > 1:
            mid = (lo_ + hi_) // 2
            if f_kmin(mid):
                lo_ = mid
            else:
                hi_ = mid
        row.append(lo_)
    band_tab[m] = row
    print(f"  {m:>3} " + "".join(f"{v:>9}" for v in row) +
          f"{m**2/(2*R_NUM):>11.1f}")

print("\n  RESULT: widening the time band does NOT increase the reachable depth.")
print("  Every column is the same.  Reason: feasibility needs at least one")
print("  neighbour with p_i >= 0, i.e. 1/2 dx_i^2 >= alpha |dt_i|, and the best")
print("  candidate is always the WIDEST neighbour at the SHALLOWEST time, so the")
print("  bound is set by k_min alone:")
print("      1/2 (m Dx)^2 >= alpha k_min Dt   <=>   m >= sqrt(2 alpha k_min Dt)/Dx")
print("  i.e. THE STENCIL HALF-WIDTH MUST BE AT LEAST THE DIFFUSION LENGTH OF")
print("  THE JUMP.  Extra time levels add nothing; only extra x-width does.")

# ======================================================================= 3C
print("\n" + "=" * 78)
print("3C -- random 5-point stencils from a past window of shape (M, K)")
print("=" * 78)

rng = np.random.default_rng(3)
Ms = [1, 2, 3, 4, 6, 8, 12]
Ks = [1, 2, 3, 5, 8, 12, 20, 40]
frac = np.zeros((len(Ks), len(Ms)))
NTRIAL = 3000
for a, K in enumerate(Ks):
    for b, M in enumerate(Ms):
        cells = [(j, -k) for j in range(-M, M + 1) for k in range(1, K + 1)]
        if len(cells) < 5:
            frac[a, b] = np.nan
            continue
        cnt = 0
        for _ in range(NTRIAL):
            sel = [cells[i] for i in rng.choice(len(cells), 5, replace=False)]
            cnt += feas([p[0] for p in sel], [p[1] for p in sel])
        frac[a, b] = cnt / NTRIAL
print(f"  feasible fraction (rows K = time depth, cols M = x half-width)")
print("      " + "".join(f"{m:>8}" for m in Ms))
for a, K in enumerate(Ks):
    print(f"  K={K:<3}" + "".join(
        ("     nan" if np.isnan(v) else f"{v:>8.3f}") for v in frac[a]))

# -------------------------------------------------- geometric predictors
print("\n  Cheap geometric predictors of positivity (candidate RL action masks)")
rng = np.random.default_rng(17)
rows = []
for _ in range(40000):
    M = rng.integers(1, 9)
    K = rng.integers(1, 25)
    cells = [(j, -k) for j in range(-M, M + 1) for k in range(1, K + 1)]
    if len(cells) < 5:
        continue
    sel = [cells[i] for i in rng.choice(len(cells), 5, replace=False)]
    ix = np.array([p[0] for p in sel])
    it = np.array([p[1] for p in sel])
    dxi, dti = ix * DX, it * DT
    q, p = moment_points(dxi, dti)
    y = positive_feasible_hull(dxi, dti)
    rows.append(dict(
        y=y,
        brackets_char=(q.min() <= 0 <= q.max()),
        straddles_diff=(p.min() <= 0 <= p.max()),
        two_sided_x=(ix.min() < 0 < ix.max()),
        xspread=ix.max() - ix.min(),
        tdepth=-it.min(),
        tspread=it.max() - it.min(),
        cond=np.linalg.cond(build_A(dxi, dti)[0]),
    ))

y = np.array([R["y"] for R in rows])
print(f"    sample: {len(y)} stencils, base rate feasible = {y.mean():.3f}")


def pr(name, mask):
    mask = np.asarray(mask, bool)
    tp = (mask & y).sum()
    prec = tp / max(mask.sum(), 1)
    rec = tp / max(y.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    print(f"    {name:<44} precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  "
          f"keeps {mask.mean():.3f}")
    return prec, rec


bc = np.array([R["brackets_char"] for R in rows])
sd = np.array([R["straddles_diff"] for R in rows])
ts = np.array([R["two_sided_x"] for R in rows])
xs = np.array([R["xspread"] for R in rows])
td = np.array([R["tdepth"] for R in rows])
cd = np.array([R["cond"] for R in rows])

pr("P0  accept everything (baseline)", np.ones_like(y))
pr("P1  brackets characteristic foot (min q <= 0 <= max q)", bc)
pr("P2  straddles diffusion length (min p <= 0 <= max p)", sd)
pr("P1 AND P2   <- NECESSARY condition, O(n), no solve", bc & sd)
pr("P3  two-sided in x", ts)
pr("P4  x-spread >= 2 cells", xs >= 2)
pr("P3 AND P4", ts & (xs >= 2))
pr("P5  cond(A) < 1e2", cd < 1e2)
pr("P6  cond(A) < 1e4", cd < 1e4)
pr("P1 AND P2 AND P3", bc & sd & ts)
pr("P1 AND P2 AND time depth <= 8", bc & sd & (td <= 8))

print("\n    -> P1 AND P2 is necessary for feasibility, so recall = 1 by")
print("       construction; the measurement is its PRECISION, i.e. how much of")
print("       the remaining infeasible mass it fails to remove.")
print("    -> conditioning-based masks are much worse on both axes, confirming")
print("       that cond(A) is not a proxy for positivity.")

# ======================================================================= FIG
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

a = ax[0]
a.plot(ms, kmax_num, "o-", color="#1b4f72", ms=4, label="$k_{max}$ (LP / hull)")
a.plot(ms, k_diff, "--", color="#c0392b", label=r"$(m\Delta x)^2/(2\alpha\Delta t)$")
a.plot(ms, k_adv, "-.", color="#e67e22", label=r"$m\Delta x/(c\Delta t)$")
a.axhline(k_vtx[0], color="#8e44ad", ls=":", label=r"$2\alpha/(c^2\Delta t)$")
a.axvline(2 / PE, color="gray", lw=1)
a.text(2 / PE + 0.4, 3, r"$m=2\alpha/(c\Delta x)$", fontsize=8, rotation=90)
a.set_xlabel("stencil half-width $m$ (cells)")
a.set_ylabel("max feasible time depth $k$ (steps)")
a.set_yscale("log")
a.set_title("3A  single-time-level symmetric stencil")
a.legend(fontsize=8, loc="lower right")

a = ax[1]
im = a.imshow(frac, origin="lower", aspect="auto", cmap="viridis",
              vmin=0, vmax=1,
              extent=[-0.5, len(Ms) - 0.5, -0.5, len(Ks) - 0.5])
a.set_xticks(range(len(Ms)))
a.set_xticklabels(Ms)
a.set_yticks(range(len(Ks)))
a.set_yticklabels(Ks)
a.set_xlabel("window x half-width $M$ (cells)")
a.set_ylabel("window time depth $K$ (steps)")
a.set_title("3C  fraction of random 5-pt stencils\nthat are positivity-feasible")
for i in range(len(Ks)):
    for j in range(len(Ms)):
        if not np.isnan(frac[i, j]):
            a.text(j, i, f"{frac[i,j]:.2f}", ha="center", va="center",
                   fontsize=7, color="w" if frac[i, j] < 0.6 else "k")
fig.colorbar(im, ax=a)

a = ax[2]
names, precs, recs = [], [], []
for nm, mk in [("all", np.ones_like(y)), ("P1", bc), ("P2", sd),
               ("P1&P2", bc & sd), ("2-sided x", ts), ("xspread>=2", xs >= 2),
               ("cond<1e2", cd < 1e2), ("cond<1e4", cd < 1e4),
               ("P1&P2&2sided", bc & sd & ts)]:
    mk = np.asarray(mk, bool)
    tp = (mk & y).sum()
    names.append(nm)
    precs.append(tp / max(mk.sum(), 1))
    recs.append(tp / max(y.sum(), 1))
xpos = np.arange(len(names))
a.bar(xpos - 0.2, precs, 0.4, label="precision", color="#2874a6")
a.bar(xpos + 0.2, recs, 0.4, label="recall", color="#e67e22")
a.set_xticks(xpos)
a.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
a.axhline(y.mean(), color="k", ls=":", lw=1)
a.text(0.05, y.mean() + 0.02, "base rate", fontsize=8)
a.set_ylim(0, 1.05)
a.set_title("3C  cheap predictors of positivity")
a.legend(fontsize=8, loc="lower left")

fig.tight_layout()
fig.savefig(f"{FIG}/task3_geography.png", dpi=150)
print(f"\nfigure -> {FIG}/task3_geography.png")
