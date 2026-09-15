"""Figures from the sweep CSVs (150 dpi). No interpretation printed.

  fig1_growth.png      growth envelope max_n ||M^n|| per arm, by geometry (Dirichlet + periodic, safety 0.5)
  fig2_error_vs_pe.png L_inf(T) per arm vs cell Peclet, stable runs only, medians over geometry/seed
  fig3_march.png       time series max|u| and L_inf error for one geometry, four arms
  fig4_feasibility.png positivity-infeasible node fraction vs Peclet and Courant number
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import g1lib as g

ARMS = ["minnorm", "projgauss", "spectral", "spectralg", "molfe", "molrk3", "molfilt", "maxent", "lp", "lprand", "maxentw"]
COL = {"minnorm": "C0", "projgauss": "C9", "spectral": "C1", "spectralg": "C8", "molfe": "C3", "molrk3": "C6",
       "maxent": "C2", "lp": "C4", "lprand": "C5", "maxentw": "C7", "molfilt": "#8B0000"}


def load(fn):
    rows = []
    with open(fn) as f:
        for r in csv.DictReader(f):
            for k in r:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
            r["geom"] = f"{r['kind']}:{r['disorder']:.2f}"
            rows.append(r)
    return rows


dir05 = load("results/sweep_dir_s05.csv") + load("results/sweep_filt_dir_s05.csv")
per05 = load("results/sweep_per_s05.csv") + load("results/sweep_filt_per_s05.csv")
geoms = ["jitter:0.00", "jitter:0.10", "jitter:0.25", "jitter:0.45", "random:0.00"]

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, rows, title in [(axes[0], dir05, "Dirichlet walls, nx=101"), (axes[1], per05, "periodic, nx=100")]:
    for ai, arm in enumerate(ARMS):
        for gi, geom in enumerate(geoms):
            gs = np.array([min(r["growth"], 1e300) for r in rows if r["arm"] == arm and r["geom"] == geom])
            if len(gs) == 0:
                continue
            xj = gi + (ai - 4.5) * 0.08
            ax.scatter(np.full(len(gs), xj), np.log10(np.maximum(gs, 1e-2)), s=9, color=COL[arm], label=arm if gi == 0 else None, alpha=0.7)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(2, color="gray", lw=0.6, ls="--")
    ax.set_xticks(range(len(geoms))); ax.set_xticklabels(geoms)
    ax.set_title(f"growth envelope max_n ||M^n||_inf  ({title}; all alpha, K, seeds; safety 0.5)", fontsize=10)
    ax.set_ylabel("log10 growth  (0 = certified bound; dashed = 100)")
axes[0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("figures/fig1_growth.png", dpi=150); plt.close()

# ---------------------------------------------------------------- fig 2
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
for ax, rows, title in [(axes[0], dir05, "Dirichlet"), (axes[1], per05, "periodic")]:
    pes = sorted(set(r["Pe"] for r in rows))
    for arm in ARMS:
        med, lo, hi = [], [], []
        for pe in pes:
            es = np.array([r["errT"] for r in rows if r["arm"] == arm and r["Pe"] == pe and r["growth"] <= 100 and np.isfinite(r["errT"])])
            n_all = sum(1 for r in rows if r["arm"] == arm and r["Pe"] == pe)
            if len(es) < max(3, 0.5 * n_all):          # fewer than half the runs stable: no point
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            med.append(np.median(es)); lo.append(np.percentile(es, 25)); hi.append(np.percentile(es, 75))
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(pes, med, yerr=[med - lo, hi - med], fmt="o-", ms=4, color=COL[arm], label=arm, capsize=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("cell Peclet  c h / alpha"); ax.set_ylabel("L_inf error at T=0.5 (median, IQR)")
    ax.set_title(f"{title}: L_inf(T) of stable runs only (growth <= 100); gap = fewer than half stable", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("figures/fig2_error_vs_pe.png", dpi=150); plt.close()

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
p = g.Problem(alpha=0.1, c=1.0)
for ci, (kind, dis) in enumerate([("jitter", 0.25), ("random", 0.0)]):
    x = g.make_points(101, dis, seed=0, kind=kind)
    dt = g.time_step(x, p, 0.5); nt = int(np.ceil(0.5 / dt)); m = max(1, nt // 100)
    samp = np.arange(m, nt + 1, m); samp = samp if samp[-1] == nt else np.append(samp, nt)
    ref = g.Reference(p, dt * samp, x); k_of = {int(s): k for k, s in enumerate(samp)}
    u0 = p.u_true(x, 0.0)
    for arm in ["molfe", "molfilt", "minnorm", "spectral", "lp", "maxent", "maxentw"]:
        op = g.Operator(x, p, dt, arm, 5, nt=nt)
        res = op.march(u0, None, nt, ref_fn=lambda k: ref.at(k_of[k]), sample_every=m)
        t = res["steps"] * dt
        axes[0, ci].semilogy(t, np.maximum(res["maxu"], 1e-16), color=COL[arm], label=f"{arm} (fb={op.fallback.sum()})" + (f" eps={op.eps:.3g}" if arm == "molfilt" else ""))
        axes[1, ci].semilogy(t, np.maximum(res["err"], 1e-16), color=COL[arm], label=arm)
    axes[0, ci].set_title(f"{kind}:{dis}, K=5, alpha=0.1 (Pe=0.1), nt={nt}: max|u|"); axes[0, ci].set_ylim(1e-3, 1e6)
    axes[1, ci].set_title("L_inf error vs reference"); axes[1, ci].set_ylim(1e-6, 1e6); axes[1, ci].set_xlabel("t")
    axes[0, ci].legend(fontsize=7); axes[0, ci].axhline(1.3, color="gray", lw=0.5)
plt.tight_layout(); plt.savefig("figures/fig3_march.png", dpi=150); plt.close()

# ---------------------------------------------------------------- fig 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, safety in [(axes[0], 0.5), (axes[1], 0.9)]:
    rows = load(f"results/sweep_dir_s0{int(safety*10)}.csv")
    for gi, geom in enumerate(geoms):
        pes, fr = [], []
        for pe in sorted(set(r["Pe"] for r in rows)):
            rs = [r for r in rows if r["arm"] == "maxent" and r["geom"] == geom and r["Pe"] == pe and r["K"] == 5]
            if rs:
                pes.append(pe); fr.append(np.mean([r["n_fallback"] / 99 for r in rs]))
        ax.plot(pes, fr, "o-", label=geom)
    ax.set_xscale("log"); ax.set_xlabel("cell Peclet"); ax.set_ylabel("fraction of interior nodes positivity-infeasible (K=5)")
    ax.set_title(f"safety {safety}: nu_mean = {safety} * hmin/h at Pe>2, r = {safety}/2 * (hmin/h)^2 at Pe<2"); ax.legend(fontsize=8)
    ax.axvline(2, color="gray", lw=0.6, ls="--")
plt.tight_layout(); plt.savefig("figures/fig4_feasibility.png", dpi=150); plt.close()
print("figures written")
