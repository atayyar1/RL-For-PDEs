"""Burgers: combine what helped, every result against ONE baseline (the notebook's LW map: LW guess, picard 1).
    PDE=burgers [GUESS=fd5:2] [PICARD=3] python explore/combo.py   -> combo_<tag>.json / table
Rules: LW stencils + reuse stops (existing points, no guess value) + guess stops (stop_cost)."""
from reuse import *

BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_inflow_front.json")


def rules():
    return {"LW map": lw, "LW+stop(c=1)": stop_cost(lw, 1.0), "LW+reuse(c=1)": with_reuse(lw, 1.0),
            "LW+reuse(c=1)+stop(c=1)": with_reuse(stop_cost(lw, 1.0), 1.0),
            "LW+reuse(c=10)+stop(c=1)": with_reuse(stop_cost(lw, 1.0), 10.0),
            "LW+reuse(c=1)+stop(c=10)": with_reuse(stop_cost(lw, 10.0), 1.0)}


def run(z):
    return str(z), {n: report_s(build_map_s(z, r)) for n, r in rules().items()}


if __name__ == "__main__":
    with Pool(4) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    tag = TAG if "TAG" in globals() else PROBLEM
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"combo_{tag}_picard{PICARD}.json"), "w"), indent=1, default=float)
    B = json.load(open(BASE_FILE))["R"] if PROBLEM != "advection" else {z: {"LW map": R[z]["LW map"]} for z in R}
    print(f"{tag}, picard {PICARD}: against the notebook's LW map (LW guess, picard 1)")
    for q, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{lab}:\n| method | points | stops | exact bound | true error | wins |\n|---|---|---|---|---|---|")
        for n in R[str(q[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / B[str(z)]["LW map"][k] for z in q])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in q]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {sum(R[str(z)][n]['error'] < B[str(z)]['LW map']['error'] for z in q)}/{len(q)} |")
    print("max identity gap", max(r["identity_gap"] for v in R.values() for r in v.values()))
