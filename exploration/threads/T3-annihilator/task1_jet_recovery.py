"""
task1_jet_recovery.py -- Step 1 characterised.

Implements the moment estimator exactly as specified, verifies once per sweep
that it coincides with the equivalent weighted local-polynomial fit (task2a),
then uses the closed form for the sweeps because it is the same estimator and
~1000x cheaper.

Reports
  (1) how many (w,G) samples are needed,
  (2) accuracy vs neighbourhood radius h  (bias-variance),
  (3) which jet components are recoverable and which are hopeless,
  (4) the truncation bias: measured exponent vs the predicted h^(K+1-p),
      p = parabolic degree a + 2b of the component.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
np.set_printoptions(precision=4, suppress=False)
sol = J.AdvDiff(alpha=0.1, c=1.0)
ALPHA = sol.alpha
x0, t0 = 0.40, 0.25
DXG = 1 / 99.0                      # companion-project grid scale

def sample_cloud(hx, n_pool, rng, one_sided=True):
    ht = hx ** 2 / ALPHA
    dxi = hx * rng.uniform(-1, 1, n_pool)
    dti = ht * (rng.uniform(-1, 0, n_pool) if one_sided else rng.uniform(-1, 1, n_pool))
    return dxi, dti, ht

def recover(hx, K, grading, n_pool=60, sigma=0.0, seed=0, one_sided=True,
            method="wls", S=4000, n_nb=6):
    rng = np.random.default_rng(seed)
    qs = J.monomials(K, grading)
    dxi, dti, ht = sample_cloud(hx, n_pool, rng, one_sided)
    u_nb = np.array([sol.u(x0 + a, t0 + b) for a, b in zip(dxi, dti)])
    u_st = sol.u(x0, t0)
    if sigma:
        u_nb = u_nb + sigma * rng.normal(size=n_pool)
        u_st = u_st + sigma * rng.normal()
    Phi = J.design_matrix(dxi, dti, qs, hx, ht)
    y = u_nb - u_st
    if method == "moment":
        Wf = np.zeros((S, n_pool))
        for s in range(S):
            idx = rng.choice(n_pool, n_nb, replace=False)
            Wf[s, idx] = J.random_weights(1, n_nb, rng, "gauss", tau=1.0)[0]
        dh = J.moment_regression(Phi, y, Wf)
    else:
        dh = J.wls_jet(Phi, y, np.eye(n_pool) - np.ones((n_pool, n_pool)) / n_pool)
    scale = J.jet_scaling(qs, hx, ht)
    return qs, dh / scale, J.true_jet(sol, x0, t0, qs), np.linalg.cond(Phi)

print("=" * 78); print("TASK 1 -- JET RECOVERY"); print("=" * 78)
print(f"target z* = ({x0}, {t0}),  alpha={ALPHA}, c={sol.c}")
print(f"grid scales of companion project: dx={DXG:.4g}, parabolic dt=dx^2/alpha={DXG**2/ALPHA:.3g}")

# ---------------------------------------------------------------- 1. cross-check
print("\n--- 1. moment estimator vs its closed form (spot check) ---")
for K in [2, 4]:
    qs, dm, dt_, _ = recover(3 * DXG, K, "parabolic", method="moment", S=6000)
    _, dw, _, _ = recover(3 * DXG, K, "parabolic", method="wls")
    print(f"  K={K}: max rel diff = {np.max(np.abs(dm - dw) / np.abs(dt_)):.2e}"
          f"   (Monte-Carlo residue only)")
print("  minimum S for identifiability = Q = number of monomials; beyond that S")
print("  only reduces Monte-Carlo noise about the closed form.")

# ------------------------------------------------------- 2. grading comparison
print("\n--- 2. total-degree vs parabolic grading (clean data, hx=3dx) ---")
print(f"{'grading':>10} {'K':>3} {'Q':>3} {'cond(Phi)':>11} {'u_x relerr':>12} "
      f"{'u_t relerr':>12} {'u_xx relerr':>12}")
for grading in ["total", "parabolic"]:
    for K in [2, 3, 4]:
        qs, dh, dtr, cond = recover(3 * DXG, K, grading)
        idx = {q: i for i, q in enumerate(qs)}
        def re(q):
            i = idx.get(q)
            return np.nan if i is None else abs(dh[i] - dtr[i]) / abs(dtr[i])
        print(f"{grading:>10} {K:>3} {len(qs):>3} {cond:>11.3g} {re((1,0)):>12.2e} "
              f"{re((0,1)):>12.2e} {re((2,0)):>12.2e}")

# ------------------------------------------------ 3. h sweep: bias and its slope
print("\n--- 3. truncation bias vs neighbourhood radius h (clean data) ---")
ms = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12])
hs = ms * DXG
comps = [(1, 0), (0, 1), (2, 0), (3, 0), (1, 1), (0, 2)]
bias = {}
for K in [2, 3, 4]:
    qs = J.monomials(K, "parabolic"); idx = {q: i for i, q in enumerate(qs)}
    tab = {q: [] for q in comps}
    for h in hs:
        _, dh, dtr, _ = recover(h, K, "parabolic")
        for q in comps:
            i = idx.get(q)
            tab[q].append(np.nan if i is None else abs(dh[i] - dtr[i]) / abs(dtr[i]))
    bias[K] = {q: np.array(v) for q, v in tab.items()}

for K in [2, 3, 4]:
    print(f"\n  K={K} (parabolic).  rel. bias, and fitted slope in h vs predicted K+1-p:")
    for q in comps:
        v = bias[K][q]
        if np.all(np.isnan(v)):
            continue
        p = q[0] + 2 * q[1]
        good = np.isfinite(v) & (v > 1e-13)
        sl = np.polyfit(np.log(hs[good][:6]), np.log(v[good][:6]), 1)[0] if good.sum() > 3 else np.nan
        print(f"    d_{q} (p={p}): h=dx {v[2]:>9.2e}  h=3dx {v[5]:>9.2e}  "
              f"h=8dx {v[8]:>9.2e} | slope {sl:>5.2f} vs predicted {K + 1 - p}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
for ax, K in zip(axes, [2, 3, 4]):
    for q in comps:
        v = bias[K][q]
        if np.all(np.isnan(v)):
            continue
        ax.loglog(ms, v, "o-", ms=3.5, label=f"$d_{{{q[0]}{q[1]}}}$")
    ax.set_title(f"$K={K}$ (parabolic)"); ax.set_xlabel("$h/\\Delta x$")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
axes[0].set_ylabel("relative truncation bias")
fig.suptitle("Truncation bias grows with $h$ at the predicted parabolic rate "
             "$h^{K+1-p}$ (clean data)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task1_bias.png"), dpi=150)

# ------------------------------------------- 4. bias-variance: noisy h sweep
print("\n--- 4. bias-variance tradeoff in h, with noise ---")
uref = J.noise_scale(sol, x0, t0, 3 * DXG, (3 * DXG) ** 2 / ALPHA, rng=np.random.default_rng(1))
print(f"  noise reference RMS(u) over neighbourhood = {uref:.4f}")
K = 4
res = {}
for eta in [0.0, 1e-6, 1e-4, 1e-2]:
    tab = {q: [] for q in [(1, 0), (0, 1), (2, 0)]}
    for h in hs:
        errs = {q: [] for q in tab}
        for seed in range(24):
            qs, dh, dtr, _ = recover(h, K, "parabolic", sigma=eta * uref, seed=seed)
            idx = {q: i for i, q in enumerate(qs)}
            for q in tab:
                errs[q].append(abs(dh[idx[q]] - dtr[idx[q]]) / abs(dtr[idx[q]]))
        for q in tab:
            tab[q].append(np.median(errs[q]))
    res[eta] = {q: np.array(v) for q, v in tab.items()}
    best = {q: ms[np.argmin(res[eta][q])] for q in tab}
    print(f"  noise {eta:>7.0e}:  best h/dx  u_x={best[(1,0)]:>5g} "
          f"u_t={best[(0,1)]:>5g} u_xx={best[(2,0)]:>5g}   "
          f"min relerr u_xx = {res[eta][(2,0)].min():.2e}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
for ax, q, nm in zip(axes, [(1, 0), (0, 1), (2, 0)], ["$u_x$", "$u_t$", "$u_{xx}$"]):
    for eta in res:
        ax.loglog(ms, res[eta][q], "o-", ms=3.5, label=f"noise {eta:.0e}")
    ax.set_title(nm); ax.set_xlabel("$h/\\Delta x$"); ax.grid(alpha=.3, which="both")
axes[0].set_ylabel("relative error"); axes[0].legend(fontsize=8)
fig.suptitle("Bias-variance in the neighbourhood radius ($K=4$ parabolic, 60 points)",
             fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task1_biasvar.png"), dpi=150)

# ------------------------------------- 5. which components are recoverable at all
print("\n--- 5. recoverability of every component (K=4 parabolic, hx=3dx, 60 pts) ---")
print(f"{'q=(a,b)':>10} {'p':>2} {'true d_q':>13} {'clean relerr':>13} "
      f"{'relerr @1e-4':>13} {'relerr @1e-2':>13}  verdict")
qs = J.monomials(4, "parabolic")
rows = []
for eta in [0.0, 1e-4, 1e-2]:
    acc = []
    for seed in range(24):
        _, dh, dtr, _ = recover(3 * DXG, 4, "parabolic", sigma=eta * uref, seed=seed)
        acc.append(np.abs(dh - dtr) / np.abs(dtr))
    rows.append(np.median(acc, axis=0))
_, _, dtr, _ = recover(3 * DXG, 4, "parabolic")
for i, q in enumerate(qs):
    v = [r[i] for r in rows]
    verdict = ("solid" if v[2] < 0.1 else "usable to 1e-4" if v[1] < 0.1
               else "clean-data only" if v[0] < 0.1 else "HOPELESS")
    print(f"{str(q):>10} {q[0]+2*q[1]:>2} {dtr[i]:>13.4g} {v[0]:>13.2e} "
          f"{v[1]:>13.2e} {v[2]:>13.2e}  {verdict}")
print("\n  figures -> figures/fig_task1_bias.png, figures/fig_task1_biasvar.png")
