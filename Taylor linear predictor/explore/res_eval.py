"""Rules with the residual score vs the baselines, on the unseen + new queries.
    PDE=burgers python explore/res_eval.py   (also: python explore/res_eval.py for advection)"""
import json
from multiprocessing import Pool
from residual import *
from evaluate import Q_UNSEEN, Q_NEW, gm, greedy_stop, stop_cost

TABS = {"lw2": guess_function(2, "lw"), "lw4": guess_function(4, "lw"), "fd52": guess_function(2, "fd5")}


def rules():
    return {"LW map": lw, "greedy": greedy_rule(1.0), "greedy+stop(myopic)": greedy_stop(1.0),
            "greedy res[lw2]": greedy_res(1.0, [TABS["lw2"]]),
            "greedy res[max lw2,lw4]": greedy_res(1.0, [TABS["lw2"], TABS["lw4"]]),
            "greedy res[fd52]": greedy_res(1.0, [TABS["fd52"]]),
            "lowest res[fd52]": lowest_res([TABS["fd52"]]),
            "greedy res[fd52]+stop(myopic)": greedy_res(1.0, [TABS["fd52"]], stop=True),
            "LW map+stop(c=1)": stop_cost(lw, 1.0)}


def run(z):
    out = {}
    for n, r in rules().items():
        m = build_map_s(z, r)
        out[n] = report_s(m)
        out[n]["est_res"] = (sum(abs(m["beta"][p]) * residual(TABS["fd52"], p, *m["stencil"][p]) for p in m["stencil"])
                             + sum(abs(m["beta"][q]) * guess_err(q) for q in m["stopped"]))
    return str(z), out


if __name__ == "__main__":
    with Pool(4) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"res_eval_{PROBLEM}.json"), "w"), indent=1, default=float)
    for qs, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{PROBLEM} {lab}:\n| method | points | stops | exact bound | true error | est/bound | est_res/bound | wins |\n|---|---|---|---|---|---|---|---|")
        for n in R[str(qs[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in qs])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in qs]):.0f} | x{rel('bound'):.2f} | x{rel('error'):.2f} | "
                  f"{gm([R[str(z)][n]['est'] / R[str(z)][n]['bound'] for z in qs]):.2f} | {gm([R[str(z)][n]['est_res'] / R[str(z)][n]['bound'] for z in qs]):.2f} | "
                  f"{sum(R[str(z)][n]['error'] < R[str(z)]['LW map']['error'] for z in qs)}/{len(qs)} |")
