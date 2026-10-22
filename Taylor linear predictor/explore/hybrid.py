"""Burgers: front-aware hybrid, no score. Deep semi-Lagrangian stencils (k levels) where the guess is smooth, LW where
the guess says the front is near (|u_x| dx above a threshold, i.e. u changes by more than th per cell).
    PDE=burgers GUESS=fd5:2 PICARD=3 python explore/hybrid.py"""
from semilag import sl_rule
from evaluate import *


def hybrid(k, th):
    deep = sl_rule(k)
    return lambda m, p: lw(m, p) if abs(GUESS[1, p[1], p[0]]) * dx > th else deep(m, p)


def rules():
    R = {"LW map": lw, "SL k=2+stop(c=3)": stop_cost(sl_rule(2), 3.0), "LW+stop(c=1)": stop_cost(lw, 1.0)}
    for k in (2, 3):
        for th in (0.01, 0.03, 0.1):
            R[f"hybrid k={k} th={th:g}+stop(c=3)"] = stop_cost(hybrid(k, th), 3.0)
    return R


def run(z):
    return str(z), {n: report_s(build_map_s(z, r)) for n, r in rules().items()}


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 3))) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    B = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_inflow_front.json")))["R"]
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hybrid_{TAG}.json"), "w"), default=float)
    print(f"{TAG}: against the notebook's LW map")
    for q, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{lab}:\n| method | points | stops | exact bound | true error | wins |\n|---|---|---|---|---|---|")
        for n in R[str(q[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / B[str(z)]["LW map"][k] for z in q])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in q]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {sum(R[str(z)][n]['error'] < B[str(z)]['LW map']['error'] for z in q)}/{len(q)} |")
