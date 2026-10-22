"""Where the maps put their points, against the exact Green's function band (mean +- 2 std of the OU foot).
    python explore/fig_footprint.py  -> footprint.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from evaluate import *

QS = [(12, 40), (25, 55), (37, 25)]
RULES = [("LW map", lw), ("LW + stop(c=1)", stop_cost(lw, 1.0)), ("greedy", greedy_rule(1.0)), ("greedy + myopic stop", greedy_stop(1.0))]
fig, axes = plt.subplots(len(QS), len(RULES), figsize=(4 * len(RULES), 3.3 * len(QS)), sharex=True)
for i, z in enumerate(QS):
    its = np.arange(1, z[1] + 1)
    lo, hi = np.array([green_band(z, it) for it in its]).T
    for j, (name, rule) in enumerate(RULES):
        m, ax = build_map_s(z, rule), axes[i, j]
        r = report_s(m)
        P = np.array(list(m["stencil"]))
        sc = ax.scatter(P[:, 0], P[:, 1], c=[abs(m["beta"][tuple(p)]) for p in P], s=5, cmap="viridis", norm=LogNorm(1e-6, 1))
        if m["stopped"]:
            S = np.array(list(m["stopped"]))
            ax.scatter(S[:, 0], S[:, 1], marker="x", s=12, color="r", lw=0.8, label="stopped")
        ax.fill_betweenx(its, np.clip(lo, 0, 1) / dx, np.clip(hi, 0, 1) / dx, color="orange", alpha=0.15, label="exact Green band (+-2 std)")
        ax.plot(*z, "r*", ms=12)
        ax.set_xlim(-1, nx); ax.set_ylim(0, z[1] + 2)
        ax.set_title(f"{name} {z}\n{r['points']} pts, {r['stops']} stops, err {r['error']:.1e}, in band {in_band_fraction(m):.0%}", fontsize=8)
        if j == 0:
            ax.set_ylabel("it")
        if i == len(QS) - 1:
            ax.set_xlabel("ix")
axes[0, 1].legend(fontsize=6, loc="lower right")
fig.colorbar(sc, ax=axes, label="|beta|", shrink=0.6)
fig.savefig(os.path.join(os.path.dirname(__file__), "footprint.png"), dpi=100, bbox_inches="tight")
print("saved footprint.png")
