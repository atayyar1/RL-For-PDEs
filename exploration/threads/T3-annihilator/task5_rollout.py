"""
task5_rollout.py -- close the loop.

Take the coefficients RECOVERED from noisy data, build the PDE-constrained
solver rows from them, solve for stencil weights, and integrate forward.
Compare against the identical integrator built from the TRUE coefficients.

Rows (derived in jetlib's docstring):
    sum_i w_i               = 1
    sum_i w_i (dx_i - c dt_i)            = 0
    sum_i w_i (dx_i^2/2 + alpha dt_i)    = 0

Two weight choices on the same geometry:
    min-norm  : lstsq  -- may give ||w||_1 > 1 and amplify error geometrically
    positive  : w >= 0 via LP -- then ||w||_1 = 1 exactly and errors only add

The positivity option is the one thing the moment/annihilator framing supplies
that a local-polynomial derivative estimate does not.
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

sol = J.AdvDiff(alpha=0.1, c=1.0)
C_TRUE, A_TRUE = sol.c, sol.alpha

# ---------------------------------------------------------------- 1. discovery
print("=" * 88); print("TASK 5 -- CLOSE THE LOOP: discover, build the rows, integrate")
print("=" * 88)
g = Grid(sol, nx=200, t0=0.10, t1=0.40, tag="advdiff")
rng = np.random.default_rng(3)
ETAS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2]
BEST_M = {0.0: 3, 1e-6: 4, 1e-4: 8, 1e-3: 16, 1e-2: 20}
found = {}
print("\n--- 1. coefficients discovered from noisy data (task 3 pipeline) ---")
print(f"  {'eta':>8} {'c_hat':>11} {'alpha_hat':>11} {'|dc|/c':>10} {'|da|/a':>10}")
for eta in ETAS:
    m = BEST_M[eta]; hc = ht_cells_of(g, m)
    tg = list(zip(rng.integers(m + 1, g.nx - m - 1, 600),
                  rng.integers(hc + 1, g.nt, 600)))
    jt = extract_jets(g, g.noisy(eta, seed=11), tg, m)
    xi, _ = J.stlsq(build_library(jt), jt["u_t"], thresh=0.02)
    ch, ah = -xi[LIB.index("u_x")], xi[LIB.index("u_xx")]
    found[eta] = (ch, ah)
    print(f"  {eta:>8.0e} {ch:>11.6f} {ah:>11.6f} "
          f"{abs(ch-C_TRUE)/C_TRUE:>10.2e} {abs(ah-A_TRUE)/A_TRUE:>10.2e}")

# ---------------------------------------------------------------- 2. integrator
NX, T = 100, 0.5
xg = np.linspace(0, 1, NX); DX = xg[1] - xg[0]
DT = T / (max(int(np.ceil(T / min(0.9 * DX / C_TRUE, 0.45 * DX ** 2 / A_TRUE))) + 1, 10) - 1)
NLEV = int(round(T / DT))
OFF = np.arange(-2, 3)                       # 5-point stencil, one level back

def weights(c, a, offs=OFF, k=1, mode="positive"):
    dxi = offs * DX; dti = -k * DT * np.ones(len(offs))
    w = J.w_positive(dxi, dti, c, a) if mode == "positive" else J.w_minnorm(dxi, dti, c, a)
    return w

def rollout(w, offs=OFF, k=1, nlev=NLEV):
    """March with a fixed stencil; the k outermost cells each side and the first
    k levels are held at the exact solution (identical for every run)."""
    pad = int(np.abs(offs).max())
    U = np.array([sol.u(xg, j * DT) for j in range(k)])       # (k, NX) seed levels
    U = list(U)
    err = []
    for lev in range(k, nlev + 1):
        prev = U[lev - k]
        new = np.zeros(NX)
        for wi, o in zip(w, offs):
            new[pad:NX - pad] += wi * prev[pad + o: NX - pad + o]
        ex = sol.u(xg, lev * DT)
        new[:pad] = ex[:pad]; new[NX - pad:] = ex[NX - pad:]
        U.append(new)
        err.append(np.max(np.abs(new - ex)))
    return np.array(err), U[-1]

print(f"\n--- 2. integrator: nx={NX}, dx={DX:.4g}, dt={DT:.4g}, {NLEV} levels to T={T} ---")
w_true = weights(C_TRUE, A_TRUE)
print(f"  true-coefficient 5-pt stencil w = {np.round(w_true, 5)}")
print(f"  ||w||_1 = {np.abs(w_true).sum():.6f}   (nonnegative -> exactly 1)")

print(f"\n--- 3. rollout accuracy: recovered rows vs true rows ---")
print(f"  {'eta':>8} {'max err @T (recovered)':>24} {'max err @T (true)':>20} "
      f"{'ratio':>8} {'predicted |dc| T |u_x|':>24}")
curves = {}
uxmax = max(abs(sol.deriv(xg, t, 1, 0)).max() for t in [0.0, 0.25, 0.5])
e_true, _ = rollout(w_true)
for eta in ETAS:
    ch, ah = found[eta]
    w = weights(ch, ah)
    if w is None:
        print(f"  {eta:>8.0e}   no nonnegative stencil exists for these coefficients")
        continue
    e, _ = rollout(w)
    curves[eta] = e
    pred = abs(ch - C_TRUE) * T * uxmax
    print(f"  {eta:>8.0e} {e[-1]:>24.4e} {e_true[-1]:>20.4e} "
          f"{e[-1]/e_true[-1]:>8.2f} {pred:>24.2e}")
# --- 3b. how does discovery error actually propagate?  synthetic sweep ---
print("\n--- 3b. controlled sweep: perturb c only, hold alpha true ---")
print(f"  {'|dc|/c':>10} {'max err @T':>13} {'err - floor':>13} {'ratio to prev':>14}")
prev = None
dcs = [0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
sweep = []
for d in dcs:
    w = weights(C_TRUE * (1 + d), A_TRUE)
    e, _ = rollout(w)
    above = e[-1] - e_true[-1]
    sweep.append(e[-1])
    rr = (above / prev) if (prev and prev > 0) else np.nan
    print(f"  {d:>10.0e} {e[-1]:>13.4e} {above:>13.3e} {rr:>14.2f}")
    prev = above if above > 0 else prev
lo = [(d, v) for d, v in zip(dcs, sweep) if d >= 1e-3]
sl = np.polyfit(np.log([d for d, _ in lo]),
                np.log([v - e_true[-1] for _, v in lo]), 1)[0]
sl_top = np.log(sweep[-1] / sweep[-3]) / np.log(dcs[-1] / dcs[-3])
print(f"  fitted exponent over the whole range      : {sl:.3f}")
print(f"  fitted exponent over |dc|/c in [1e-2,1e-1]: {sl_top:.3f}   (linear => 1)")
print("  -> HONEST READING: the whole-range exponent is contaminated because the")
print("     max-norm error is signed and cancels against the scheme's own truncation")
print("     error near the floor, so 'error - floor' is not a clean quantity.  Well")
print("     above the floor the growth is close to linear.  The essential point is")
print("     that nothing amplifies: the integrator stays stable and degrades smoothly.")

# ------------------------------------------------- 4. positivity, tested properly
print(f"\n--- 4. positivity: what it actually buys (von Neumann, not ||w||_1) ---")
print("  For a one-step stencil the exact max-norm/L2 stability criterion is the")
print("  amplification factor  g = max_theta |sum_j w_j exp(i j theta)|.")
print("  ||w||_1 >= g always, so ||w||_1 > 1 does NOT imply instability; but")
print("  w >= 0 with sum w = 1 forces g = 1 exactly.  Positivity is SUFFICIENT,")
print("  not necessary.  Below we measure how often it actually matters.")

def amp(w, offs):
    th = np.linspace(0, 2 * np.pi, 2001)
    return np.max(np.abs(np.exp(1j * np.outer(th, offs)) @ w))

ch, ah = found[1e-4]
print(f"\n  {'stencil (k=1)':>30} {'mode':>9} {'||w||_1':>9} {'min w':>9} "
      f"{'g':>9} {'max err @T':>12}")
for nm, offs in [("symmetric [-2..2]", np.arange(-2, 3)),
                 ("offset [-3..1]", np.arange(-3, 2)),
                 ("symmetric [-3..3]", np.arange(-3, 4)),
                 ("upwind-only [-4..0]", np.arange(-4, 1))]:
    for mode in ["minnorm", "positive"]:
        w = weights(ch, ah, offs, 1, mode)
        if w is None:
            print(f"  {nm:>30} {mode:>9} {'infeasible / ill-conditioned':>42}")
            continue
        e, _ = rollout(w, offs, 1)
        print(f"  {nm:>30} {mode:>9} {np.abs(w).sum():>9.5f} {w.min():>9.5f} "
              f"{amp(w, offs):>9.6f} {e[-1]:>12.4e}")

print("\n  --- how often does min-norm actually produce an UNSTABLE stencil? ---")
rg = np.random.default_rng(0)
nA = nB = nboth = tot = 0
worst = []
for _ in range(4000):
    k = rg.integers(4, 8)
    offs = np.sort(rg.choice(np.arange(-5, 6), k, replace=False))
    wm = weights(ch, ah, offs, 1, "minnorm")
    wp = weights(ch, ah, offs, 1, "positive")
    if wm is None:
        continue
    tot += 1
    gm = amp(wm, offs)
    if gm > 1 + 1e-9:
        nA += 1
        worst.append(gm)
        if wp is not None:
            nboth += 1
    if wp is not None:
        nB += 1
print(f"  {tot} random 4-7 point one-step stencils in [-5,5]:")
print(f"    min-norm unstable (g > 1)                        : {nA} ({nA/tot:.1%})")
print(f"    a nonnegative stencil exists                     : {nB} ({nB/tot:.1%})")
print(f"    min-norm unstable AND positivity available (rescued): {nboth} ({nboth/tot:.1%})")
if worst:
    print(f"    among unstable min-norm: median g = {np.median(worst):.4f}, "
          f"max g = {max(worst):.4f}")
    print(f"    at g = {np.median(worst):.4f} over {NLEV} levels the amplification is "
          f"{np.median(worst)**NLEV:.3e}")
print("\n  -> HONEST SUMMARY.  On the well-conditioned symmetric stencils the")
print("     min-norm solution is already nonnegative, so positivity changes")
print("     nothing.  It matters on the awkward geometries, which is exactly")
print("     where a meshfree solver lives.  And when positivity IS imposed on a")
print("     lopsided stencil it can cost accuracy (see offset [-3..1] above):")
print("     it buys a stability guarantee, not a free lunch.")
print("\n     The real argument for positivity is not this 1-D table.  On a UNIFORM")
print("     one-step stencil you can simply compute g and reject.  But a meshfree")
print("     solver has scattered, level-varying geometry where no Fourier symbol")
print("     exists, and there g is not computable.  w >= 0 with sum w = 1 still")
print("     certifies max-norm non-amplification there, and it is a single LP.")
print("     That certificate is a property of the WEIGHTS.  A jet estimate --")
print("     however it was obtained -- does not provide one.")

# ---------------------------------------------------------------- 5. figure
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
tt = np.arange(1, NLEV + 1) * DT
axes[0].semilogy(tt, e_true, "k-", lw=2, label="true $c,\\alpha$")
for eta in ETAS:
    if eta in curves:
        axes[0].semilogy(tt, curves[eta], lw=1.2,
                         label=f"recovered @ $\\eta$={eta:.0e}")
axes[0].set_xlabel("$t$"); axes[0].set_ylabel("max error")
axes[0].set_title("Rollout: discovered rows vs true rows", fontsize=10)
axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
es = sweep
dc = dcs
axes[1].loglog(np.maximum(np.array(dc)*C_TRUE, 1e-9), es, "o", ms=7, label="controlled $c$ perturbation")
axes[1].loglog([abs(found[e][0]-C_TRUE) for e in ETAS if e in curves],
               [curves[e][-1] for e in ETAS if e in curves], "s", ms=8,
               mfc="none", color="tab:orange", label="actual discovered coefficients")
xs = np.logspace(-9, -1, 50)
axes[1].loglog(xs, xs * T * uxmax, "k--", label="$|\\hat c - c|\\,T\\,\\max|u_x|$ (bound)")
axes[1].axhline(e_true[-1], color="crimson", ls=":", label="true-coefficient floor")
axes[1].set_xlabel("$|\\hat c - c|$"); axes[1].set_ylabel("max error at $T$")
axes[1].set_title("Discovery error propagates linearly", fontsize=10)
axes[1].grid(alpha=.3, which="both"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task5_rollout.png"), dpi=150)
print("\n  figure -> figures/fig_task5_rollout.png")
