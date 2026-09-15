"""
task6_propagator.py -- the propagator-moment (Kramers-Moyal) reformulation.

A. the moment hierarchy, verified independently
B. the jet <-> moment correspondence (Task 1 is reusable, not wasted)
C. estimating the propagator moments FROM DATA, and a fair head-to-head
   against the jet route and weak-form SINDy on one shared grid
D. the Pawula / nonlocality detector, with a predicted exponent
E. Hankel realisability -- does positivity actually regularise the fit?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import jetlib as J, moments as MO
from griddata import Grid
from weakform import WeakSINDy
from task3_relations import extract_jets, build_library, LIB, ht_cells_of

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
sol = J.AdvDiff(0.1, 1.0); C, AL = sol.c, sol.alpha

# =========================================================== A. the hierarchy
print("=" * 88); print("TASK 6 -- PROPAGATOR-MOMENT (KRAMERS-MOYAL) REFORMULATION")
print("=" * 88)
print("""
A. THE HIERARCHY.  With u(x,t+tau) = INT K(y,tau) u(x-y,t) dy, the stencil
   moments Mfrak_q = sum_i w_i dx_i^q (offsets) relate to kernel moments by
   Mfrak_q = (-1)^q M_q, and

       sum_q Mfrak_q s^q / q!  =  exp( tau P(s) ),    P(s) = alpha s^2 - c s

   the MGF of the stencil is the exponential of the SYMBOL of the generator.
   Cumulants are kappa_q = tau q! [s^q] P(s), so a second-order operator has
   only kappa_1 = -c tau, kappa_2 = 2 alpha tau: the propagator is Gaussian.
   This is the semigroup-symbol / Levy-Khintchine relation.""")
dt_c = 4.591e-4
print(f"  {'tau':>10} {'|ODE - closed form|':>21} {'|series - (-1)^q M_q|':>23} "
      f"{'c read-off':>12} {'alpha read-off':>15}")
for k in [10, 50, 400]:
    tau = k * dt_c; Q = 8
    ex = MO.gauss_moments(C, AL, tau, Q)
    od = MO.solve_hierarchy({1: C, 2: AL}, tau, Q)
    fr = MO.moments_from_symbol({1: -C, 2: AL}, tau, Q)
    ch, ah = MO.coeffs_from_moments(fr, tau)
    sg = np.array([(-1) ** q for q in range(Q + 1)])
    print(f"  {tau:>10.5f} {np.max(np.abs(od-ex)):>21.2e} "
          f"{np.max(np.abs(fr-sg*ex)):>23.2e} {ch:>12.9f} {ah:>15.9f}")
kk = MO.cumulants(MO.moments_from_symbol({1: -C, 2: AL}, 50 * dt_c, 8), 8)
print(f"  cumulants kappa_3..kappa_8 at tau=50dt: {np.abs(kk[3:]).max():.2e}  (exactly 0)")

# ================================================== B. jet <-> moment mapping
print("""
B. CORRESPONDENCE WITH THE TAYLOR ROWS OF RESULTS.md.
   Row 2  sum w (dx - c dt) = 0  with dt = -tau  reads  Mfrak_1 = -c tau: the
   EXACT first-moment condition.  Row 3  sum w (dx^2/2 + alpha dt) = 0 reads
   Mfrak_2 = 2 alpha tau, but the exact second moment is 2 alpha tau + c^2 tau^2.
   The Taylor rows are the LOW-ORDER TRUNCATION of the moment conditions.""")
print(f"  {'tau':>10} {'Mfrak_2 exact':>15} {'Taylor row target':>19} {'dropped c^2 tau^2':>19} {'rel':>9}")
for k in [1, 10, 50, 400]:
    tau = k * dt_c
    fr = MO.moments_from_symbol({1: -C, 2: AL}, tau, 4)
    print(f"  {tau:>10.5f} {fr[2]:>15.6e} {2*AL*tau:>19.6e} {C**2*tau**2:>19.6e} "
          f"{C**2*tau**2/fr[2]:>9.4f}")
print("  -> at the parabolic scale tau ~ h^2/alpha the dropped term is O(h^4);")
print("     at large tau it dominates, which is exactly why the Taylor rows fail")
print("     for a big step and the moment rows do not.")

# ============================== C. estimating the moments from data + head-to-head
print("\n" + "=" * 88)
print("C. ESTIMATING THE PROPAGATOR FROM DATA -- fair head-to-head, one shared grid")
print("=" * 88)
g = Grid(sol, nx=400, t0=0.10, t1=0.40, dt=5e-4, tag="p400")
print(f"  shared grid: nx={g.nx} dx={g.dx:.4g} nt={g.nt} dt={g.dt:.4g} RMS(u)={g.urms:.4f}")
rng = np.random.default_rng(0)

def fit_propagator(k, R, Un, n_tg=8000, nonneg=False, lam=0.0):
    tau = k * g.dt; offs = np.arange(-R, R + 1)
    ii = rng.integers(R, g.nx - R, n_tg); jj = rng.integers(k, g.nt, n_tg)
    D = np.stack([Un[ii + o, jj - k] for o in offs], axis=1); y = Un[ii, jj]
    W = 1e3 * np.sqrt(len(y))
    A = [D, W * np.ones((1, len(offs)))]; b = [y, [W]]
    if lam > 0:
        A.append(lam * np.eye(len(offs))); b.append(np.zeros(len(offs)))
    A = np.vstack(A); b = np.concatenate(b)
    w = (lsq_linear(A, b, bounds=(0, np.inf), tol=1e-13, max_iter=400).x if nonneg
         else np.linalg.lstsq(A, b, rcond=None)[0])
    Mf = np.array([np.sum(w * (offs * g.dx) ** q) for q in range(9)])
    return w, Mf, tau, offs

ETAS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2]
k_lag, R_lag = 25, 85
print(f"\n  propagator fit at lag tau={k_lag*g.dt:.4g}, stencil half-width R={R_lag} "
      f"({2*R_lag+1} weights), 8000 targets")
prop_err = {}
for eta in ETAS:
    Un = g.noisy(eta, seed=11)
    w, Mf, tau, offs = fit_propagator(k_lag, R_lag, Un)
    ch, ah = MO.coeffs_from_moments(Mf, tau)
    prop_err[eta] = max(abs(ch - C) / C, abs(ah - AL) / AL)
    print(f"  eta={eta:<7.0e} c={ch:>10.6f} alpha={ah:>10.6f}  "
          f"max rel err {prop_err[eta]:.3e}   min w={w.min():+.2e}")

print("\n  same grid, jet route (Task 3 pipeline, m tuned):")
jet_err = {}
for eta in ETAS:
    Un = g.noisy(eta, seed=11); best = None
    for m in [3, 4, 6, 8, 12, 16]:
        hc = ht_cells_of(g, m)
        if hc + 2 >= g.nt: continue
        tg = list(zip(rng.integers(m + 1, g.nx - m - 1, 600),
                      rng.integers(hc + 1, g.nt, 600)))
        jt = extract_jets(g, Un, tg, m)
        if len(jt["u"]) < 20: continue
        xi, _ = J.stlsq(build_library(jt), jt["u_t"], thresh=0.02)
        e = max(abs(xi[LIB.index("u_x")] + C) / C, abs(xi[LIB.index("u_xx")] - AL) / AL)
        if best is None or e < best: best = e
    jet_err[eta] = best
    print(f"  eta={eta:<7.0e} max rel err {best:.3e}")

print("\n  same grid, weak-form SINDy (patch tuned):")
WT = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2), (1, 3)]
ws = WeakSINDy(mx=10, mt=10, px=6, pt=6); weak_err = {}
for eta in ETAS:
    Un = g.noisy(eta, seed=11); best = None
    for (Hx, Ht) in [(20, 20), (40, 50), (80, 100), (120, 200)]:
        if Ht >= g.nt - 2 or Hx >= g.nx // 2: continue
        ci = rng.integers(Hx + 1, g.nx - Hx - 1, 200)
        cj = rng.integers(Ht + 1, g.nt - Ht - 1, 200)
        Th, bb = ws.assemble(g, Un, list(zip(ci, cj)), WT, Hx, Ht)
        if len(bb) < 20: continue
        xi, _ = J.stlsq(Th, bb, thresh=0.02)
        e = max(abs(-xi[2] - C) / C, abs(xi[4] - AL) / AL)
        if best is None or e < best: best = e
    weak_err[eta] = best
    print(f"  eta={eta:<7.0e} max rel err {best:.3e}")

print(f"\n  HEAD TO HEAD (max rel error in c, alpha; same bytes for all three)")
print(f"  {'eta':>9} {'propagator':>13} {'jet route':>13} {'weak SINDy':>13} {'winner':>12}")
for eta in ETAS:
    tri = {"propagator": prop_err[eta], "jet": jet_err[eta], "weak": weak_err[eta]}
    print(f"  {eta:>9.0e} {prop_err[eta]:>13.3e} {jet_err[eta]:>13.3e} "
          f"{weak_err[eta]:>13.3e} {min(tri, key=tri.get):>12}")

# ======================================================= D. the Pawula detector
print("\n" + "=" * 88)
print("D. THE NONLOCALITY DETECTOR -- with a predicted exponent")
print("=" * 88)
print("""  For a stable propagator of index mu = 2s, G(z,tau) = tau^(-1/mu) g(z tau^(-1/mu))
  with tails g ~ |y|^(-1-mu).  Truncating the moment at stencil half-width R:
      M_2(R,tau) ~ R^(2-mu) * tau   =>   alpha_eff(R) ~ R^(2-2s).
  (M_2 is LINEAR in tau; the signature is the R-dependence, not the tau-dependence.)
  So the drift does not merely flag nonlocality -- it MEASURES s = 1 - slope/2.""")

NX = 512; dxp = 1.0 / NX; xp = np.arange(NX) * dxp
def pfield(alpha, s, K=200, decay=0.5, seed=0):
    r = np.random.default_rng(seed); kk = np.arange(1, K + 1)
    a = kk ** (-float(decay)); ph = r.uniform(0, 2 * np.pi, K)
    lam = alpha * (2 * np.pi * kk) ** (2 * s)
    return lambda t: (np.cos(2 * np.pi * np.outer(xp, kk) + ph)
                      * (a * np.exp(-lam * t))).sum(-1)

def alpha_eff(u, tau, ts, R, eta=0.0, seed=1):
    r = np.random.default_rng(seed); offs = np.arange(-R, R + 1); D = []; y = []
    for t in ts:
        a0, a1 = u(t), u(t + tau)
        if eta:
            rms = np.sqrt(np.mean(a0 ** 2))
            a0 = a0 + eta * rms * r.normal(size=NX); a1 = a1 + eta * rms * r.normal(size=NX)
        D.append(np.stack([np.roll(a0, -o) for o in offs], axis=1)); y.append(a1)
    D = np.vstack(D); y = np.concatenate(y); W = 1e3 * np.sqrt(len(y))
    A = np.vstack([D, W * np.ones((1, len(offs)))]); b = np.concatenate([y, [W]])
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    Mf = np.array([np.sum(w * (offs * dxp) ** q) for q in range(5)])
    return (Mf[2] - Mf[1] ** 2) / (2 * tau), Mf, w

AP, TAU = 3e-5, 0.02; TS = np.linspace(0.0, 0.30, 60); RS = [4, 8, 16, 32, 64]
print(f"\n  data-driven, clean.  nx={NX}, tau={TAU}")
print(f"  {'s':>6} " + " ".join(f"{'R='+str(r):>11}" for r in RS) +
      f" {'slope':>8} {'s_est':>8} {'|err|':>8}")
det = {}
for s_ in [1.0, 0.9, 0.75, 0.5]:
    u = pfield(AP, s_); v = [alpha_eff(u, TAU, TS, R)[0] for R in RS]
    sl = np.polyfit(np.log(RS), np.log(np.abs(v)), 1)[0]
    det[s_] = (np.array(v), sl)
    print(f"  {s_:>6} " + " ".join(f"{x:>11.3e}" for x in v) +
          f" {sl:>8.3f} {1-sl/2:>8.3f} {abs(1-sl/2-s_):>8.3f}")

print(f"\n  robustness to noise (s recovered from the drift):")
print(f"  {'eta':>9} " + " ".join(f"{'s='+str(s):>10}" for s in [1.0, 0.9, 0.75, 0.5]))
for eta in [0.0, 1e-4, 1e-3, 1e-2]:
    row = []
    for s_ in [1.0, 0.9, 0.75, 0.5]:
        u = pfield(AP, s_)
        v = [alpha_eff(u, TAU, TS, R, eta=eta)[0] for R in RS]
        sl = np.polyfit(np.log(RS), np.log(np.abs(v)), 1)[0]
        row.append(1 - sl / 2)
    print(f"  {eta:>9.0e} " + " ".join(f"{x:>10.3f}" for x in row))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for s_ in [1.0, 0.9, 0.75, 0.5]:
    v, sl = det[s_]
    axes[0].loglog(RS, v / v[0], "o-", ms=5,
                   label=f"$s$={s_} (est {1-sl/2:.2f})")
    axes[0].loglog(RS, (np.array(RS) / RS[0]) ** (2 - 2 * s_), "k:", lw=.8)
axes[0].set_xlabel("stencil half-width $R$ (cells)")
axes[0].set_ylabel("$\\alpha_{\\rm eff}(R)/\\alpha_{\\rm eff}(4)$")
axes[0].set_title("Nonlocality detector: $\\alpha_{\\rm eff}\\sim R^{2-2s}$\n"
                  "(dotted = predicted slope)", fontsize=10)
axes[0].grid(alpha=.3, which="both"); axes[0].legend(fontsize=8)
xs = np.array([1e-7 if e == 0 else e for e in ETAS])
axes[1].loglog(xs, [prop_err[e] for e in ETAS], "o-", label="propagator moments")
axes[1].loglog(xs, [jet_err[e] for e in ETAS], "s--", label="jet route")
axes[1].loglog(xs, [weak_err[e] for e in ETAS], "d-.", label="weak-form SINDy")
axes[1].set_xlabel("relative noise $\\eta$"); axes[1].set_ylabel("max rel error in $c,\\alpha$")
axes[1].set_title("Head to head on one shared grid", fontsize=10)
axes[1].grid(alpha=.3, which="both"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task6_propagator.png"), dpi=150)

# ============================================= E. Hankel / positivity as regulariser
print("\n" + "=" * 88)
print("E. DOES POSITIVITY REGULARISE THE FIT?  (claim (c), tested)")
print("=" * 88)
print(f"  {'eta':>9} {'plain LS':>22} {'ridge 1e-2':>22} {'NNLS':>22}")
print(f"  {'':>9} {'err':>10} {'Hankel':>11} {'err':>10} {'Hankel':>11} {'err':>10} {'Hankel':>11}")
for eta in ETAS:
    Un = g.noisy(eta, seed=11); cells = []
    for nm, kw in [("ls", {}), ("ridge", {"lam": 1e-2}), ("nnls", {"nonneg": True})]:
        w, Mf, tau, offs = fit_propagator(k_lag, R_lag, Un, **kw)
        ch, ah = MO.coeffs_from_moments(Mf, tau)
        e = max(abs(ch - C) / C, abs(ah - AL) / AL)
        hk, _ = MO.hankel_psd_margin(Mf, 3)
        cells.append((e, hk))
    print(f"  {eta:>9.0e} " + " ".join(f"{e:>10.2e} {h:>11.2e}" for e, h in cells))
print("""
  VERDICT ON CLAIM (c): NOT SUPPORTED as an estimator regulariser.  NNLS is
  1e5x WORSE than plain least squares on clean data and never wins outright.
  The reason is structural: NNLS returns a SPARSE solution (most weights exactly
  zero), which is a poor approximation to a smooth Gaussian kernel.  The Hankel
  margin stays positive for every fit here, including the bad ones, so it does
  not discriminate either.  Positivity earns its keep when CONSTRUCTING the
  integrator (task 5 / task 7), not when estimating the coefficients.""")
print(f"\n  figure -> figures/fig_task6_propagator.png")
