"""Points vs true error on the unseen queries, sweeping the price of a point, for each family of rules.
    python explore/pareto.py              -> pareto_advection.png / .json
    PDE=burgers python explore/pareto.py  -> pareto_inflow_front.png / .json
Every point of a curve = geometric mean over the unseen queries of (method / LW map of the same query)."""
import json
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import *
from rollout import policy_iteration

LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)


def families():
    F = {"LW + stop(c)": [(c, lambda z, c=c: build_map_s(z, stop_cost(lw, c))) for c in (0.03, 0.1, 0.3, 1, 3, 10)],
         "greedy(lam)": [(l, lambda z, l=l: build_map_s(z, greedy_rule(l))) for l in LAMS],
         "greedy(lam) + myopic stop": [(l, lambda z, l=l: build_map_s(z, greedy_stop(l))) for l in LAMS],
         "greedy(lam) + rollout stop": [(l, lambda z, l=l: policy_iteration(z, greedy_rule(l), l, iters=1)[-1]) for l in LAMS],
         "LW + rollout stop(lam)": [(l, lambda z, l=l: policy_iteration(z, lw, l, iters=1)[-1]) for l in LAMS],
         "lowest score + stop(c)": [(c, lambda z, c=c: build_map_s(z, stop_cost(lowest_score, c))) for c in (0.1, 1, 10)]}
    return F


def run(z):
    lw_rep = report_s(build_map_s(z, lw))
    out = {}
    for fam, members in families().items():
        out[fam] = []
        for v, make in members:
            r = report_s(make(z))
            out[fam].append(dict(v=v, **{k: r[k] / lw_rep[k] for k in ("points", "error", "bound")}, stops=r["stops"]))
    return str(z), out


if __name__ == "__main__":
    with Pool(4) as pool:
        res = dict(pool.map(run, Q_UNSEEN))
    curves = {}
    for fam in families():
        curves[fam] = [dict(v=res[str(Q_UNSEEN[0])][fam][i]["v"],
                            **{k: gm([res[str(z)][fam][i][k] for z in Q_UNSEEN]) for k in ("points", "error", "bound")})
                       for i in range(len(res[str(Q_UNSEEN[0])][fam]))]
    json.dump({"per_query": res, "curves": curves}, open(os.path.join(os.path.dirname(__file__), f"pareto_{PROBLEM}.json"), "w"), indent=1, default=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (fam, c) in enumerate(curves.items()):
        for ax, k in zip(axes, ("error", "bound")):
            ax.loglog([d["points"] for d in c], [d[k] for d in c], "o-", color=f"C{i}", label=fam, ms=4)
            for d in c:
                ax.annotate(f"{d['v']:g}", (d["points"], d[k]), fontsize=6, color=f"C{i}", xytext=(3, 2), textcoords="offset points")
    for ax, k in zip(axes, ("true error", "exact bound")):
        ax.plot(1, 1, "kD", ms=7, label="LW map")
        ax.set_xlabel("points / LW map (geo. mean, 9 unseen queries)"); ax.set_ylabel(f"{k} / LW map")
        ax.grid(True, which="both", lw=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{PROBLEM}: points vs accuracy, price of a point swept (labels = lam or c)")
    plt.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), f"pareto_{PROBLEM}.png"), dpi=110)
    for fam, c in curves.items():
        print(fam, " | ".join(f"{d['v']:g}: pts x{d['points']:.2f} err x{d['error']:.3f} bnd x{d['bound']:.3f}" for d in c))
