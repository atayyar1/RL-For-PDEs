"""
task4_generalise.py -- does the route generalise?

(a) pure diffusion (c=0): must NOT hallucinate advection.
(b) viscous Burgers.  Two solutions are used on purpose:
      - the Taylor shock, a TRAVELLING wave, where u_t = -s u_x identically.
        The Burgers relation is then structurally unidentifiable from the data,
        and the method (correctly) cannot recover it.  This is a data failure,
        not a method failure, and it is worth recording.
      - a multi-mode Cole-Hopf solution, which is identifiable.  Here we ask
        whether the recovered advection coefficient tracks the local u.
(c) spectral fractional diffusion with MANY modes: no local PDE exists.
    (With only a handful of Fourier modes any operator is exactly matched by a
    finite-order local one, so a multi-mode field is required for this test to
    mean anything.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J
from griddata import Grid
from task3_relations import extract_jets, build_library, LIB, ht_cells_of

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
rng = np.random.default_rng(5)


def targets(g, N, m):
    hc = ht_cells_of(g, m)
    return list(zip(rng.integers(m + 1, g.nx - m - 1, N),
                    rng.integers(hc + 1, g.nt, N)))


def discover(g, eta, m, N=600, thresh=0.02, seed=11, K=4):
    Un = g.noisy(eta, seed=seed)
    jt = extract_jets(g, Un, targets(g, N, m), m, K=K)
    Th = build_library(jt)
    xi, _ = J.stlsq(Th, jt["u_t"], thresh=thresh)
    sel = [LIB[k] for k in range(len(LIB)) if abs(xi[k]) > 1e-12]
    res = np.linalg.norm(Th @ xi - jt["u_t"]) / np.linalg.norm(jt["u_t"])
    return xi, sel, res, jt


print("=" * 90); print("TASK 4 -- GENERALISATION"); print("=" * 90)

# ================================================================== (a) diffusion
print("\n" + "=" * 90)
print("(a) PURE DIFFUSION  u_t = alpha u_xx  (c = 0: must not hallucinate advection)")
print("=" * 90)
sd = J.AdvDiff(alpha=0.1, c=0.0)
gd = Grid(sd, nx=200, t0=0.10, t1=0.40, tag="diff")
print(f"  {'eta':>8} {'m':>4} {'c_hat':>13} {'alpha_hat':>11} {'rel resid':>11}  selected")
for eta, m in [(0.0, 3), (1e-5, 4), (1e-4, 8), (1e-3, 12), (1e-2, 16)]:
    xi, sel, r, _ = discover(gd, eta, m)
    print(f"  {eta:>8.0e} {m:>4} {-xi[LIB.index('u_x')]:>13.3e} "
          f"{xi[LIB.index('u_xx')]:>11.6f} {r:>11.2e}  {sel}")
print("  -> advection is never invented: c_hat is thresholded to exactly 0 in")
print("     every clean and low-noise case.  alpha is recovered to <1%.")

# ================================================================== (b) Burgers
print("\n" + "=" * 90)
print("(b1) BURGERS, TRAVELLING-WAVE DATA (Taylor shock) -- an identifiability trap")
print("=" * 90)
sb1 = J.BurgersTanh(nu=0.05, s=1.0, a=0.8)
gb1 = Grid(sb1, nx=200, t0=0.10, t1=0.40, alpha_scale=sb1.nu, tag="burgers")
r = [sb1.deriv(x, 0.2, 0, 1) / sb1.deriv(x, 0.2, 1, 0) for x in (0.25, 0.5, 0.75)]
print(f"  u = s - a tanh(a(x-st)/2nu) is a travelling wave: u_t/u_x = {np.round(r,6)}")
print(f"  so u_t = -s u_x EXACTLY, and the library columns obey the exact")
print(f"  dependence (s-u) u_x + nu u_xx = 0.  Burgers is NOT identifiable here.")
xi, sel, res, jt = discover(gb1, 0.0, 4, N=800)
print(f"\n  clean-data STLSQ result: "
      + "  ".join(f"{LIB[k]}={xi[k]:+.4f}" for k in range(len(LIB)) if abs(xi[k]) > 1e-12))
A = np.column_stack([jt["u"] * jt["u_x"], jt["u_xx"]])
print(f"  forcing the true support {{u*u_x, u_xx}}: "
      f"{np.linalg.lstsq(A, jt['u_t'], rcond=None)[0]}   (truth [-1, 0.05])")
print(f"  condition number of [u_x, u*u_x, u_xx] = "
      f"{np.linalg.cond(np.column_stack([jt['u_x'], jt['u']*jt['u_x'], jt['u_xx']])):.3e}")
print("  -> the method returns u_t = -u_x, which is TRUE for this data and is")
print("     the parsimonious answer.  No PDE-discovery method can do better on")
print("     a single travelling wave.  Report this as a data requirement.")

print("\n" + "=" * 90)
print("(b2) BURGERS, MULTI-MODE COLE-HOPF DATA -- the real test")
print("=" * 90)
sb = J.BurgersColeHopf(nu=0.05, an=(0.8, 0.15))
gb = Grid(sb, nx=200, t0=0.02, t1=0.30, alpha_scale=sb.nu, tag="ch")
print(f"  phi = 1 + 0.8 cos(pi x) e^(-nu pi^2 t) + 0.15 cos(2pi x) e^(-4 nu pi^2 t)")
print(f"  grid dx={gb.dx:.4g} dt={gb.dt:.4g} nt={gb.nt}; truth: u*u_x -> -1, u_xx -> {sb.nu}")
print(f"\n  {'eta':>8} {'m':>4} {'coef(u*u_x)':>13} {'coef(u_xx)':>12} {'resid':>10}  selected")
for eta, m in [(0.0, 3), (1e-6, 4), (1e-5, 4), (1e-4, 6), (1e-3, 8)]:
    xi, sel, res, _ = discover(gb, eta, m, N=800)
    print(f"  {eta:>8.0e} {m:>4} {xi[LIB.index('u*u_x')]:>13.5f} "
          f"{xi[LIB.index('u_xx')]:>12.5f} {res:>10.2e}  {sel}")

print("\n  --- does the LOCAL advection coefficient track u?  (bin targets by u) ---")
Un = gb.noisy(0.0, seed=11)
jt = extract_jets(gb, Un, targets(gb, 6000, 4), 4)
edges = np.quantile(jt["u"], np.linspace(0, 1, 11))
ub, ae, be, nb = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (jt["u"] >= lo) & (jt["u"] < hi)
    if k.sum() < 40:
        continue
    A = np.column_stack([jt["u_x"][k], jt["u_xx"][k]])
    co = np.linalg.lstsq(A, jt["u_t"][k], rcond=None)[0]
    ub.append(jt["u"][k].mean()); ae.append(co[0]); be.append(co[1]); nb.append(k.sum())
ub, ae, be = map(np.array, (ub, ae, be))
print(f"  {'<u> in bin':>11} {'n':>6} {'a_eff':>10} {'-u (truth)':>11} {'b_eff':>10} "
      f"{'nu (truth)':>11}")
for uu, a_, b_, n_ in zip(ub, ae, be, nb):
    print(f"  {uu:>11.4f} {n_:>6} {a_:>10.4f} {-uu:>11.4f} {b_:>10.5f} {sb.nu:>11.3f}")
sl, ic = np.polyfit(ub, ae, 1)
print(f"\n  fit a_eff = {sl:+.4f}*<u> {ic:+.4f}      (truth: slope -1, intercept 0)")
print(f"  b_eff: mean {be.mean():.5f}, spread {be.std():.5f}   (truth nu = {sb.nu})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(ub, ae, "o", label="recovered $a_{\\rm eff}$")
axes[0].plot(ub, -ub, "k--", label="truth $-u$")
axes[0].set_xlabel("local $u$"); axes[0].set_ylabel("$a_{\\rm eff}$")
axes[0].set_title("Burgers (Cole-Hopf): advection coefficient\ntracks the local state",
                  fontsize=10)
axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].plot(ub, be, "o", color="tab:green", label="recovered $b_{\\rm eff}$")
axes[1].axhline(sb.nu, color="k", ls="--", label=f"truth $\\nu={sb.nu}$")
axes[1].set_xlabel("local $u$"); axes[1].set_ylabel("$b_{\\rm eff}$")
axes[1].set_ylim(0, 2 * sb.nu)
axes[1].set_title("Diffusion coefficient stays constant", fontsize=10)
axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task4_burgers.png"), dpi=150)

# ================================================================== (c) nonlocal
print("\n" + "=" * 90)
print("(c) SPECTRAL FRACTIONAL DIFFUSION  u_t = -alpha(-Lap)^s u  (NO local PDE)")
print("=" * 90)
bn = tuple(np.arange(1, 21) ** -1.5)
print("  initial data has 20 Fourier modes, so no finite-order local operator")
print("  can match the symbol alpha k^{2s} unless s = 1.")
print(f"\n  {'s':>6} {'m':>4} {'alpha_eff':>11} {'rel resid (CLEAN data)':>24}")
MS = [3, 4, 6, 8, 12]
store = {}
for s_ in [1.0, 0.9, 0.75, 0.5]:
    sf = J.FracDiff(alpha=0.1, s=s_, bn=bn)
    gf = Grid(sf, nx=200, t0=0.01, t1=0.08, alpha_scale=0.1, tag=f"fr2_{s_}")
    est, rs = [], []
    for m in MS:
        tg = targets(gf, 500, m)
        jt = extract_jets(gf, gf.U, tg, m, K=4)
        A = jt["u_xx"][:, None]
        co = np.linalg.lstsq(A, jt["u_t"], rcond=None)[0]
        est.append(co[0])
        rs.append(np.linalg.norm(A @ co - jt["u_t"]) / np.linalg.norm(jt["u_t"]))
        print(f"  {s_:>6} {m:>4} {est[-1]:>11.5f} {rs[-1]:>24.3e}")
    store[s_] = (np.array(est), np.array(rs))
    print(f"  {'':>6} {'-->':>4} drift max/min = {max(est)/min(est):>6.3f}   "
          f"residual floor = {min(rs):.3e}")

print("\n  TWO ALARMS, both fire only for s < 1:")
print(f"  {'s':>6} {'residual floor (clean)':>24} {'alpha_eff drift over m':>24}")
for s_ in [1.0, 0.9, 0.75, 0.5]:
    e, r = store[s_]
    print(f"  {s_:>6} {r.min():>24.3e} {e.max()/e.min():>24.3f}")
print("\n  -> On CLEAN data a genuine local PDE leaves a residual of ~1e-3 that")
print("     shrinks with the bandwidth.  The nonlocal cases leave 35-90% and it")
print("     does not shrink.  The method does fail loudly, PROVIDED the residual")
print("     is reported.  STLSQ's selected term list alone looks innocent.")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for s_ in [1.0, 0.9, 0.75, 0.5]:
    e, r = store[s_]
    lab = f"$s$={s_}" + (" (local)" if s_ == 1.0 else "")
    axes[0].loglog(MS, r, "o-", ms=4, label=lab)
    axes[1].semilogx(MS, e / e[0], "o-", ms=4, label=lab)
axes[0].set_xlabel("bandwidth $m$"); axes[0].set_ylabel("relative residual (clean data)")
axes[0].set_title("Alarm 1: irreducible residual floor", fontsize=10)
axes[0].grid(alpha=.3, which="both"); axes[0].legend(fontsize=8)
axes[1].axhline(1, color="k", lw=.8)
axes[1].set_xlabel("bandwidth $m$"); axes[1].set_ylabel("$\\alpha_{\\rm eff}(m)/\\alpha_{\\rm eff}(3)$")
axes[1].set_title("Alarm 2: coefficient drifts with scale", fontsize=10)
axes[1].grid(alpha=.3, which="both"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task4_nonlocal.png"), dpi=150)
print("\n  figures -> fig_task4_burgers.png, fig_task4_nonlocal.png")
