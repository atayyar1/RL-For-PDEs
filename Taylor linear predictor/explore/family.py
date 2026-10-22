"""Burgers: greedy over a FEW physical stencil families instead of ~18k stencils, so a wrong score has little to exploit.
Families at p: LW (3 cells, 1 level down) and semi-Lagrangian k = 1, 2, 3 (5 cells nearest the characteristic foot).
Cost of a family = |beta_p| s_p / EPS + lam new / N_LW (cell 13's one-step cost); + the myopic stop.
    PDE=burgers GUESS=fd5:2 PICARD=3 python explore/family.py"""
from semilag import sl_rule
from evaluate import *

FAMS = [lw, sl_rule(1), sl_rule(2), sl_rule(3)]


def family_greedy(lam, stop=False, use_score=True):
    def choose(m, p):
        b = abs(m["beta"][p])
        best, best_cost = None, np.inf
        for f in FAMS:
            nbs = f(m, p)
            w = weights(p, nbs)
            if w is None:
                continue
            new = sum(1 for q in set(nbs) if not is_known(q) and q not in m["beta"])
            cost = (b * point_score(p, nbs, w) / EPS if use_score else 0.0) + lam * new / n_lw(m["z_star"])
            if cost < best_cost:
                best, best_cost = nbs, cost
        if stop and p != m["z_star"] and b * guess_err(p) / EPS < best_cost:
            return None
        return best
    return choose


def rules():
    return {"LW map": lw, "SL k=2+stop(c=3)": stop_cost(sl_rule(2), 3.0),
            "family greedy": family_greedy(1.0), "family greedy+myopic stop": family_greedy(1.0, stop=True),
            "family greedy+stop(c=3)": stop_cost(family_greedy(1.0), 3.0),
            "family greedy lam=0.1+stop(c=3)": stop_cost(family_greedy(0.1), 3.0)}


def run(z):
    return str(z), {n: report_s(build_map_s(z, r)) for n, r in rules().items()}


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 2))) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    B = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_inflow_front.json")))["R"]
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"family_{TAG}.json"), "w"), default=float)
    print(f"{TAG}: against the notebook's LW map")
    for q, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{lab}:\n| method | points | stops | exact bound | true error | wins |\n|---|---|---|---|---|---|")
        for n in R[str(q[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / B[str(z)]["LW map"][k] for z in q])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in q]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {sum(R[str(z)][n]['error'] < B[str(z)]['LW map']['error'] for z in q)}/{len(q)} |")
