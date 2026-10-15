"""
task3_relations.py -- Step 2: find the linear relations among the recovered jets.

Everything below consumes ONE shared noisy grid, so the jet route and weak-form
SINDy see identical data.

Jet route:  local MLS jet at N targets -> library of jet monomials -> STLSQ
Weak route: WSINDy on the same grid (no data derivatives at all)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J
from griddata import Grid
from weakform import WeakSINDy

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
np.set_printoptions(precision=5, suppress=True)

def ht_cells_of(grid, m):
    return max(1, min(m ** 2, grid.nt - 2))


# ---------------------------------------------------------------- jet extraction
def extract_jets(grid, Un, targets, m, K=4, constrained=False, sub=None, seed=0):
    """Local jet at each target.  constrained=True reproduces the sum(w)=1
    moment formulation, which pins u* to its NOISY measured value."""
    qs = J.monomials(K, "parabolic")
    hx = m * grid.dx
    # parabolic box: ht = hx^2/alpha.  The grid is built with dt = dx^2/alpha,
    # so ht/dt = m^2 exactly.  (In general the time bandwidth is a second free
    # hyperparameter of the method, since alpha is what we are trying to find.)
    ht_cells = ht_cells_of(grid, m)
    ht = ht_cells * grid.dt
    rng = np.random.default_rng(seed)
    out = {"u": [], "u_x": [], "u_xx": [], "u_xxx": [], "u_t": [], "x": [], "t": []}
    idx = {q: k for k, q in enumerate(qs)}
    for (i, j) in targets:
        dxi, dti, un = grid.cloud(i, j, m, ht_cells, Un, subsample=sub, rng=rng)
        if len(dxi) < len(qs) + 4:
            continue
        Phi = J.design_matrix(dxi, dti, qs, hx, ht)
        if constrained:
            d = J.wls_jet(Phi, un - Un[i, j],
                          np.eye(len(un)) - np.ones((len(un),) * 2) / len(un))
            u0 = Un[i, j]
        else:
            d, u0 = J.mls_jet_unconstrained(dxi, dti, un, qs, hx, ht)
        d = d / J.jet_scaling(qs, hx, ht)
        out["u"].append(u0); out["x"].append(grid.x[i]); out["t"].append(grid.t[j])
        out["u_x"].append(d[idx[(1, 0)]])
        out["u_xx"].append(2 * d[idx[(2, 0)]])
        out["u_xxx"].append(6 * d[idx[(3, 0)]] if (3, 0) in idx else 0.0)
        out["u_t"].append(d[idx[(0, 1)]])
    return {k: np.array(v) for k, v in out.items()}


LIB = ["1", "u", "u^2", "u_x", "u*u_x", "u_xx", "u*u_xx", "u_x^2", "u_xxx"]

def build_library(jt):
    u, ux, uxx, uxxx = jt["u"], jt["u_x"], jt["u_xx"], jt["u_xxx"]
    return np.column_stack([np.ones_like(u), u, u ** 2, ux, u * ux,
                            uxx, u * uxx, ux ** 2, uxxx])

def report(xi, truth, tag, tol=1e-12):
    act = {LIB[k]: xi[k] for k in range(len(LIB)) if abs(xi[k]) > tol}
    ok = set(act) == set(truth)
    s = "  ".join(f"{k}={v:+.5f}" for k, v in act.items())
    err = max((abs(act.get(k, 0) - v) / abs(v) for k, v in truth.items()), default=np.nan)
    print(f"    {tag:<26} {'CORRECT ' if ok else 'WRONG   '} {s}")
    return ok, err


if __name__ == "__main__":
    sol = J.AdvDiff(alpha=0.1, c=1.0)
    TRUTH = {"u_x": -sol.c, "u_xx": sol.alpha}
    g = Grid(sol, nx=200, t0=0.10, t1=0.40, tag="advdiff")
    print("=" * 86)
    print("TASK 3 -- RELATION FINDING (Step 2)")
    print("=" * 86)
    print(f"shared grid: nx={g.nx} dx={g.dx:.4g}, nt={g.nt} dt={g.dt:.4g}, "
          f"RMS(u)={g.urms:.4f}")
    print(f"truth: u_t = {-sol.c:+g} u_x {sol.alpha:+g} u_xx")
    print(f"library ({len(LIB)} terms): {LIB}")

    rng = np.random.default_rng(3)
    def pick(N, m, ht_cells):
        ii = rng.integers(m + 1, g.nx - m - 1, N)
        jj = rng.integers(ht_cells + 1, g.nt, N)
        return list(zip(ii, jj))

    # ------------------------------------------------ A. noise sweep, tuned m
    print("\n--- A. coefficient recovery vs noise (m tuned per noise level) ---")
    ETAS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    MS = [3, 4, 6, 8, 12, 16, 20]
    N = 400
    best_rows = {}
    for eta in ETAS:
        Un = g.noisy(eta, seed=11)
        best = None
        for m in MS:
            ht_cells = ht_cells_of(g, m)
            tg = pick(N, m, ht_cells)
            jt = extract_jets(g, Un, tg, m)
            if len(jt["u"]) < 20:
                continue
            Th = build_library(jt)
            xi, _ = J.stlsq(Th, jt["u_t"], thresh=0.02)
            e = max(abs(xi[LIB.index("u_x")] + sol.c) / sol.c,
                    abs(xi[LIB.index("u_xx")] - sol.alpha) / sol.alpha)
            sel = set(LIB[k] for k in range(len(LIB)) if abs(xi[k]) > 1e-12)
            if best is None or e < best[0]:
                best = (e, m, xi, sel)
        e, m, xi, sel = best
        best_rows[eta] = best
        cc = -xi[LIB.index("u_x")]; aa = xi[LIB.index("u_xx")]
        print(f"  eta={eta:<7.0e} m*={m:>3}  c={cc:+.6f} ({abs(cc-sol.c)/sol.c:.2e})  "
              f"alpha={aa:+.6f} ({abs(aa-sol.alpha)/sol.alpha:.2e})  "
              f"terms={'CORRECT' if sel == set(TRUTH) else sorted(sel)}")

    # ------------------------------------------------ B. number of targets
    print("\n--- B. accuracy vs number of targets (eta = 1e-3, m = 8) ---")
    Un = g.noisy(1e-3, seed=11); m = 8
    ht_cells = ht_cells_of(g, m)
    Ns = [10, 25, 50, 100, 200, 400, 800, 1600]
    curveN = []
    for N_ in Ns:
        ee = []
        for rep in range(6):
            tg = pick(N_, m, ht_cells)
            jt = extract_jets(g, Un, tg, m)
            Th = build_library(jt)
            xi, _ = J.stlsq(Th, jt["u_t"], thresh=0.02)
            ee.append(max(abs(xi[LIB.index("u_x")] + sol.c) / sol.c,
                          abs(xi[LIB.index("u_xx")] - sol.alpha) / sol.alpha))
        curveN.append(np.median(ee))
        print(f"  N={N_:>5}  max rel coeff err = {curveN[-1]:.3e}")

    # ------------------------------------------------ C. spurious-term rejection
    print("\n--- C. does STLSQ reject the spurious terms? (N=400) ---")
    print(f"  {'eta':>8}  selected terms")
    for eta in ETAS:
        e, m, xi, sel = best_rows[eta]
        mark = "OK " if sel == set(TRUTH) else "BAD"
        print(f"  {eta:>8.0e}  {mark} {sorted(sel)}")
    print("\n  sensitivity to the STLSQ threshold (eta=1e-3, m=8, N=400):")
    tg = pick(400, 8, ht_cells); jt = extract_jets(g, Un, tg, 8)
    Th = build_library(jt)
    for th in [0.001, 0.005, 0.02, 0.05, 0.1, 0.2]:
        xi, _ = J.stlsq(Th, jt["u_t"], thresh=th)
        sel = sorted(LIB[k] for k in range(len(LIB)) if abs(xi[k]) > 1e-12)
        print(f"    thresh={th:<6} -> {sel}")

    # ------------------------------------------------ D. weak-form SINDy, same data
    print("\n--- D. weak-form SINDy on the SAME grid (closest competitor) ---")
    WTERMS = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3)]
    WNAME = ["u", "u^2", "u_x", "(u^2)_x", "u_xx", "(u^2)_xx", "u_xxx"]
    ws = WeakSINDy(mx=10, mt=10, px=6, pt=6)
    print(f"  weak library: {WNAME}   (truth: u_x -> {-sol.c}, u_xx -> {sol.alpha})")
    weak_err = {}
    for eta in ETAS:
        Un = g.noisy(eta, seed=11)
        best = None
        for (Hx, Ht) in [(12, 40), (20, 100), (30, 200), (40, 400)]:
            if Ht >= g.nt - 2 or Hx >= g.nx // 2:
                continue
            ci = rng.integers(Hx + 1, g.nx - Hx - 1, 200)
            cj = rng.integers(Ht + 1, g.nt - Ht - 1, 200)
            Th, b = ws.assemble(g, Un, list(zip(ci, cj)), WTERMS, Hx, Ht)
            if len(b) < 20:
                continue
            xi, _ = J.stlsq(Th, b, thresh=0.02)
            cc, aa = -xi[2], xi[4]
            e = max(abs(cc - sol.c) / sol.c, abs(aa - sol.alpha) / sol.alpha)
            sel = set(WNAME[k] for k in range(len(WNAME)) if abs(xi[k]) > 1e-12)
            if best is None or e < best[0]:
                best = (e, (Hx, Ht), cc, aa, sel)
        e, HH, cc, aa, sel = best
        weak_err[eta] = e
        print(f"  eta={eta:<7.0e} patch(Hx,Ht)={HH}  c={cc:+.6f} "
              f"({abs(cc-sol.c)/sol.c:.2e})  alpha={aa:+.6f} "
              f"({abs(aa-sol.alpha)/sol.alpha:.2e})  "
              f"terms={'CORRECT' if sel == {'u_x','u_xx'} else sorted(sel)}")

    print("\n  HEAD TO HEAD (max rel error in c, alpha):")
    print(f"  {'eta':>9} {'jet route':>13} {'weak SINDy':>13} {'winner':>10}")
    jetc, wkc = [], []
    for eta in ETAS:
        je = best_rows[eta][0]; we = weak_err[eta]
        jetc.append(je); wkc.append(we)
        print(f"  {eta:>9.0e} {je:>13.3e} {we:>13.3e} "
              f"{'jet' if je < we else 'weak':>10}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    xs = np.array([1e-7 if e == 0 else e for e in ETAS])
    axes[0].loglog(xs, jetc, "o-", label="jet route (MLS + STLSQ)")
    axes[0].loglog(xs, wkc, "s--", label="weak-form SINDy")
    axes[0].set_xlabel("relative noise $\\eta$"); axes[0].set_ylabel("max rel error in $c,\\alpha$")
    axes[0].set_title("Coefficient recovery vs noise"); axes[0].grid(alpha=.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].loglog(Ns, curveN, "o-", color="tab:green")
    axes[1].set_xlabel("number of targets $N$"); axes[1].set_ylabel("max rel error in $c,\\alpha$")
    axes[1].set_title("Jet route: accuracy vs $N$ ($\\eta=10^{-3}$)"); axes[1].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task3_relations.png"), dpi=150)
    print("\n  figure -> figures/fig_task3_relations.png")
