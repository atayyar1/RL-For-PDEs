"""
Task 4 -- Does the positivity certificate actually fix error accumulation?

Marching rollout on u_t + c u_x = alpha u_xx.  Level 0 is the exact initial
condition; at every level each interior point is predicted from n = 5
already-computed neighbours chosen from a local past window.  Boundaries are
held exact (u = 0).

POLICIES
  lstsq   minimum 2-norm weights (what the existing code does)
  minL1   minimise ||w||_1 (identical to the positive solution when feasible)
  mask    resample the neighbour set until positivity is feasible (budget R),
          then use the nonnegative weights; fall back to minL1 if the budget
          runs out.  This is the RL-action-mask policy.

lstsq and minL1 are driven by the SAME geometry sequence (same RNG stream), so
the comparison isolates the solver.  mask necessarily changes the geometry --
that is the intervention being tested.

GEOMETRY SELECTORS
  benign   |dx| <= 2 cells, 1-3 levels back      (the original setting)
  wide     |dx| <= 4 cells, 1-5 levels back
  sprint   ONE-SIDED in x, biased to deep time levels: a time-pressured agent
           racing toward the target, which is exactly the regime where
           one-sided stencils and large ||w||_1 appear.

Because the exact solution decays, we report error RELATIVE to max|u_true| at
that level as well as the raw max error; otherwise a decaying solution makes a
growing scheme look stable.
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stencil import (ALPHA, C, DX, DT, NX, X_FD, u_true, w_lstsq, w_minL1,
                     w_positive_enum, positive_feasible_hull, moments_solvable)

FIG = "figures"
NB = 5                                   # neighbours per prediction


def hull_batch(DXI, DTI, alpha=ALPHA, c=C):
    """Vectorised exact positivity test over a batch of candidate stencils."""
    Q = DXI - c * DTI
    P = 0.5 * DXI**2 + alpha * DTI
    Q = Q / np.maximum(np.abs(Q).max(1, keepdims=True), 1e-300)
    P = P / np.maximum(np.abs(P).max(1, keepdims=True), 1e-300)
    at0 = (np.hypot(Q, P) <= 1e-12).any(1)
    TH = np.sort(np.arctan2(P, Q), axis=1)
    G = np.diff(np.concatenate([TH, TH[:, :1] + 2 * np.pi], axis=1), axis=1)
    return at0 | (G.max(1) <= np.pi + 1e-12)


# --------------------------------------------------------------- selectors
def cand_cells(sel, i, lev, rg):
    """Candidate (index, level) neighbours for target (i, lev)."""
    if sel == "benign":
        dxs, kmax = range(-2, 3), 3
        off = [(d, k) for k in range(1, kmax + 1) for d in dxs]
    elif sel == "wide":
        dxs, kmax = range(-4, 5), 5
        off = [(d, k) for k in range(1, kmax + 1) for d in dxs]
    elif sel == "sprint":
        s = 1 if rg.random() < 0.5 else -1          # commit to ONE side
        off = [(s * d, k) for k in range(1, 6) for d in range(0, 4)]
    else:
        raise ValueError(sel)
    out = [(i + d, lev - k) for d, k in off
           if 0 <= i + d < NX and lev - k >= 0]
    return out


def draw(sel, i, lev, rg, tries=12):
    """Draw a neighbour set whose moment system is SOLVABLE at all.

    Vertical stencils (all neighbours at one x) make A rank-deficient with b
    outside its range -- no weights exist, for any solver.  That is a property
    of the geometry, so it is screened out identically for every policy and
    does not bias the comparison.  lstsq would silently return a vector with
    residual ~0.5 here; the existing code's cond(A) > 1e4 guard catches the
    same cases.
    """
    for _ in range(tries):
        pick = _draw_raw(sel, i, lev, rg)
        if pick is None:
            return None
        dxi = np.array([X_FD[q[0]] - X_FD[i] for q in pick])
        dti = np.array([(q[1] - lev) * DT for q in pick])
        if moments_solvable(dxi, dti):
            return pick
    return None


def _draw_raw(sel, i, lev, rg):
    cells = cand_cells(sel, i, lev, rg)
    if len(cells) < NB:
        return None
    if sel == "sprint":
        # a time-pressured agent prefers the DEEPEST available levels
        depth = np.array([lev - cc[1] for cc in cells], float)
        pr = depth**2
        pr = pr / pr.sum()
        idx = rg.choice(len(cells), NB, replace=False, p=pr)
    else:
        idx = rg.choice(len(cells), NB, replace=False)
    return [cells[k] for k in idx]


# ------------------------------------------------------------------ rollout
def march(policy, sel, L=1000, seed=0, R=8):
    rg = np.random.default_rng(seed)
    U = np.full((L + 1, NX), np.nan)
    U[0] = u_true(X_FD, 0.0)
    l1_all = []
    err_abs = np.zeros(L)
    err_rel = np.zeros(L)
    n_feas = n_tot = n_exhaust = n_degen = 0

    for lev in range(1, L + 1):
        U[lev, 0] = 0.0
        U[lev, NX - 1] = 0.0
        for i in range(1, NX - 1):
            pick = draw(sel, i, lev, rg)
            if pick is None:
                U[lev, i] = u_true(X_FD[i], lev * DT)[0]
                continue
            n_tot += 1

            if policy == "mask":
                # propose R candidate stencils, batch-test, take the first
                # positivity-feasible one
                props = [pick] + [draw(sel, i, lev, rg) for _ in range(R - 1)]
                props = [p for p in props if p is not None]
                DXI = np.array([[X_FD[q[0]] - X_FD[i] for q in p] for p in props])
                DTI = np.array([[(q[1] - lev) * DT for q in p] for p in props])
                ok = hull_batch(DXI, DTI)
                if ok.any():
                    j = int(np.argmax(ok))
                    pick = props[j]
                    dxi, dti = DXI[j], DTI[j]
                    w = w_positive_enum(dxi, dti)
                    if w is None:                       # numerical edge case
                        w, _ = w_minL1(dxi, dti)
                    n_feas += 1
                    if w is None:
                        w = w_lstsq(dxi, dti); n_degen += 1
                else:
                    n_exhaust += 1
                    pick = props[0]
                    dxi, dti = DXI[0], DTI[0]
                    w, _ = w_minL1(dxi, dti)
                    if w is None:
                        w = w_lstsq(dxi, dti); n_degen += 1
            else:
                dxi = np.array([X_FD[q[0]] - X_FD[i] for q in pick])
                dti = np.array([(q[1] - lev) * DT for q in pick])
                if policy == "lstsq":
                    w = w_lstsq(dxi, dti)
                elif policy == "minL1":
                    w = w_positive_enum(dxi, dti)
                    if w is None:
                        w, _ = w_minL1(dxi, dti)
                        if w is None:
                            w = w_lstsq(dxi, dti); n_degen += 1
                    else:
                        n_feas += 1
                else:
                    raise ValueError(policy)
                if policy == "lstsq" and positive_feasible_hull(dxi, dti):
                    n_feas += 1

            unb = np.array([U[q[1], q[0]] for q in pick])
            l1_all.append(float(np.abs(w).sum()))
            U[lev, i] = float(w @ unb)

        ut = u_true(X_FD, lev * DT)
        e = np.abs(U[lev] - ut)
        err_abs[lev - 1] = np.nanmax(e)
        err_rel[lev - 1] = np.nanmax(e) / max(np.abs(ut).max(), 1e-300)

    return dict(err_abs=err_abs, err_rel=err_rel, l1=np.array(l1_all),
                feas_frac=n_feas / max(n_tot, 1),
                degen_frac=n_degen / max(n_tot, 1),
                exhaust_frac=n_exhaust / max(n_tot, 1), U=U)


def growth(err, lo, hi):
    """Geometric-mean per-level growth factor of err over [lo, hi)."""
    seg = err[lo:hi]
    seg = seg[np.isfinite(seg) & (seg > 0)]
    if len(seg) < 3:
        return np.nan
    return float(np.exp(np.polyfit(np.arange(len(seg)), np.log(seg), 1)[0]))


# --------------------------------------------------------------------- run
L = 1000
SELECTORS = ["benign", "wide", "sprint"]
POLICIES = ["lstsq", "minL1", "mask"]
res = {}
print(f"marching L = {L} levels, {NX-2} interior points, n = {NB} neighbours")
print(f"final time = {L*DT:.4f}  (T = 0.5);  max|u(.,0)| = "
      f"{np.abs(u_true(X_FD,0.0)).max():.4f}, max|u(.,{L*DT:.3f})| = "
      f"{np.abs(u_true(X_FD,L*DT)).max():.4e}")
for sel in SELECTORS:
    for pol in POLICIES:
        t0 = time.perf_counter()
        res[(sel, pol)] = march(pol, sel, L=L, seed=3)
        print(f"  {sel:>7} / {pol:<6}  {time.perf_counter()-t0:6.1f}s")

print("\n" + "=" * 96)
print("MAX RELATIVE ERROR vs LEVEL")
print("=" * 96)
levels = [10, 50, 100, 200, 300, 500, 750, 1000]
for sel in SELECTORS:
    print(f"\n  selector = {sel}")
    print("    " + f"{'level':>7}" + "".join(f"{p:>14}" for p in POLICIES)
          + f"{'lstsq/mask':>13}{'lstsq/minL1':>13}")
    for lv in levels:
        row = [res[(sel, p)]["err_rel"][lv - 1] for p in POLICIES]
        print("    " + f"{lv:>7}" + "".join(f"{v:>14.3e}" for v in row)
              + f"{row[0]/row[2]:>13.1f}{row[0]/row[1]:>13.1f}")

print("\n" + "=" * 96)
print("||w||_1 STATISTICS and POSITIVITY RATES")
print("=" * 96)
print(f"  {'selector':>8} {'policy':>7} {'median':>9} {'mean':>9} {'p99':>9} "
      f"{'max':>10} {'frac>1.001':>11} {'positive-feasible':>18}")
for sel in SELECTORS:
    for pol in POLICIES:
        R_ = res[(sel, pol)]
        l1 = R_["l1"]
        print(f"  {sel:>8} {pol:>7} {np.median(l1):>9.4f} {l1.mean():>9.4f} "
              f"{np.percentile(l1,99):>9.4f} {l1.max():>10.3f} "
              f"{np.mean(l1>1.001):>11.3f} {R_['feas_frac']:>18.3f}")
    print()

print("=" * 96)
print("EMPIRICAL PER-LEVEL GROWTH FACTOR  (geometric fit of max rel. error)")
print("=" * 96)
print(f"  {'selector':>8} {'policy':>7} {'lev 20-200':>12} {'lev 200-600':>13} "
      f"{'lev 600-1000':>14} {'mean ||w||_1':>14}")
for sel in SELECTORS:
    for pol in POLICIES:
        R_ = res[(sel, pol)]
        e = R_["err_rel"]
        print(f"  {sel:>8} {pol:>7} {growth(e,20,200):>12.5f} "
              f"{growth(e,200,600):>13.5f} {growth(e,600,1000):>14.5f} "
              f"{R_['l1'].mean():>14.5f}")
    print()

print("=" * 96)
print("EFFECT SIZE -- honest summary")
print("=" * 96)
for sel in SELECTORS:
    a = res[(sel, "lstsq")]["err_rel"]
    b = res[(sel, "minL1")]["err_rel"]
    cc = res[(sel, "mask")]["err_rel"]
    print(f"  {sel:>7}: rel.err at L=300  lstsq {a[299]:.3e}  minL1 {b[299]:.3e} "
          f"({a[299]/b[299]:6.1f}x)  mask {cc[299]:.3e} ({a[299]/cc[299]:8.1f}x)")
    print(f"  {'':>7}  rel.err at L=1000 lstsq {a[999]:.3e}  minL1 {b[999]:.3e} "
          f"({a[999]/b[999]:6.1f}x)  mask {cc[999]:.3e} ({a[999]/cc[999]:8.1f}x)")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
cols = {"lstsq": "#c0392b", "minL1": "#2874a6", "mask": "#1e8449"}
LV = np.arange(1, L + 1)
for j, sel in enumerate(SELECTORS):
    a = ax[0, j]
    for pol in POLICIES:
        a.semilogy(LV, res[(sel, pol)]["err_rel"], color=cols[pol], lw=1.4,
                   label=pol)
    a.set_xlabel("level $L$")
    a.set_ylabel("max relative error" if j == 0 else "")
    a.set_title(f"selector = {sel}")
    a.legend(fontsize=8)
    a.grid(alpha=0.25)

    a = ax[1, j]
    for pol in POLICIES:
        l1 = res[(sel, pol)]["l1"]
        a.hist(np.clip(l1, 1, 3), bins=np.linspace(1, 3, 90), alpha=0.55,
               color=cols[pol], label=f"{pol} (mean {l1.mean():.3f})")
    a.set_yscale("log")
    a.axvline(1.0, color="k", ls="--", lw=1)
    a.set_xlabel(r"$\|w\|_1$")
    a.set_ylabel("count" if j == 0 else "")
    a.legend(fontsize=8)
fig.suptitle("Task 4 -- error accumulation over 1000 recursion levels, "
             "advection-diffusion", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}/task4_rollout.png", dpi=150)
print(f"\nfigure -> {FIG}/task4_rollout.png")
print("\n  degenerate-draw fallbacks (should be ~0): " + ", ".join(
    f"{s_}/{p_}={res[(s_,p_)]['degen_frac']:.2e}"
    for s_ in SELECTORS for p_ in POLICIES))
np.save(f"{FIG}/../task4_err.npy",
        {k: dict(err_rel=v["err_rel"], err_abs=v["err_abs"], l1=v["l1"],
                 feas=v["feas_frac"]) for k, v in res.items()},
        allow_pickle=True)
