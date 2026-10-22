"""The shaped agent's maps at unseen queries, against the exact Green's function band.
    python explore/fig_agent.py [run_name]  -> agent_footprint.png"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rl_stop import *
from stop import green_band

name = sys.argv[1] if len(sys.argv) > 1 else "stop_shaped_25-45"
model = MaskablePPO.load(os.path.join(RUNS, name, "model"), device="cpu")
QS = [(12, 40), (25, 55), (37, 25), (25, 45)]
fig, axes = plt.subplots(1, len(QS), figsize=(4.2 * len(QS), 3.6))
for ax, z in zip(axes, QS):
    m = run_policy(model, z)
    r = report_s(m)
    its = np.arange(1, z[1] + 1)
    lo, hi = np.array([green_band(z, it) for it in its]).T
    ax.fill_betweenx(its, np.clip(lo, 0, 1) / dx, np.clip(hi, 0, 1) / dx, color="orange", alpha=0.25, label="exact Green band (+-2 std)")
    P = np.array(list(m["stencil"]))
    sc = ax.scatter(P[:, 0], P[:, 1], c=[abs(m["beta"][tuple(p)]) for p in P], s=6, cmap="viridis", norm=LogNorm(1e-6, 1))
    if m["stopped"]:
        S = np.array(list(m["stopped"]))
        ax.scatter(S[:, 0], S[:, 1], marker="x", s=12, color="r", lw=0.8, label="stopped")
    ax.plot(*z, "r*", ms=12)
    ax.set_xlim(-1, nx); ax.set_ylim(0, z[1] + 2); ax.set_xlabel("ix")
    ax.set_title(f"{z}{' (train)' if z == (25, 45) else ''}: {r['points']} pts, {r['stops']} stops\nerr {r['error']:.1e}, "
                 f"LW map {n_lw(z)} pts", fontsize=8)
axes[0].set_ylabel("it"); axes[0].legend(fontsize=6, loc="lower right")
fig.colorbar(sc, ax=axes, label="|beta|", shrink=0.8)
fig.suptitle(f"PPO + stop + shaping ({name}), advection", fontsize=9)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_footprint.png"), dpi=100, bbox_inches="tight")
print("saved agent_footprint.png")
