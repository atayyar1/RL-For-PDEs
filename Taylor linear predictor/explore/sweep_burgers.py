"""Burgers: the price of a stop swept for the best recipe, and semi-Lagrangian deep stencils, under converged Picard.
    PDE=burgers GUESS=fd5:2 PICARD=3 python explore/sweep_burgers.py  -> sweep_<tag>.json / .png
Everything against the notebook's LW map (LW guess, picard 1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from semilag import sl_rule
from evaluate import *

CS = (0.1, 0.3, 1, 3, 10, 30, 100)
FAM = {"LW + stop(c)": lambda c: stop_cost(lw, c), "SL k=2 + stop(c)": lambda c: stop_cost(sl_rule(2), c),
       "SL k=3 + stop(c)": lambda c: stop_cost(sl_rule(3), c)}


def run(z):
    return str(z), {f: [report_s(build_map_s(z, mk(c))) for c in CS] for f, mk in FAM.items()}


if __name__ == "__main__":
    with Pool(4) as pool:
        R = dict(pool.map(run, Q_UNSEEN))
    B = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_inflow_front.json")))["R"]
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweep_{TAG}.json"), "w"), default=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, f in enumerate(FAM):
        pts = [gm([R[str(z)][f][j]["points"] / B[str(z)]["LW map"]["points"] for z in Q_UNSEEN]) for j in range(len(CS))]
        err = [gm([R[str(z)][f][j]["error"] / B[str(z)]["LW map"]["error"] for z in Q_UNSEEN]) for j in range(len(CS))]
        ax.loglog(pts, err, "o-", color=f"C{i}", label=f, ms=4)
        for c, x, y in zip(CS, pts, err):
            ax.annotate(f"{c:g}", (x, y), fontsize=6, color=f"C{i}", xytext=(3, 2), textcoords="offset points")
        print(f, " | ".join(f"c={c:g}: x{x:.2f} pts, x{y:.2f} err" for c, x, y in zip(CS, pts, err)))
    ax.plot(1, 1, "kD", ms=7, label="notebook LW map")
    ax.set_xlabel("points / notebook LW map (geo. mean, 9 unseen)"); ax.set_ylabel("true error / notebook LW map")
    ax.grid(True, which="both", lw=0.3); ax.legend(fontsize=7)
    ax.set_title(f"Burgers front, {TAG}: price of a stop swept", fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweep_{TAG}.png"), dpi=110)
