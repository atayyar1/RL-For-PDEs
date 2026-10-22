"""Semi-Lagrangian deep stencils: no score at all.
Every point jumps k levels and takes the 5 cells nearest to the foot of its characteristic, x_p - a_p k dt (with the
map's speed a_build), inside the window. Falls back to shallower k, then LW, when not admissible.
Greedy saves points mostly by deep stencils; this does the same from the physics alone.
    python explore/semilag.py [PDE=burgers]
"""
from evaluate import *


def sl_rule(k):
    def choose(m, p):
        allowed = {s for s in allowed_sets(m, p)[0]}
        for kk in range(min(k, K_T), 0, -1):
            foot = -a_build(p) * kk * dt / dx                       # foot offset in cells
            row = [c for c in range(N_CELLS) if WINDOW[c][1] == -kk]
            pick = tuple(sorted(sorted(row, key=lambda c: abs(WINDOW[c][0] - foot))[:N_NB]))
            if pick in allowed:
                return [place(p, c) for c in pick]
        return lw(m, p)
    return choose


def run(z):
    out = {}
    for n, r in {"LW map": lw, "SL k=1": sl_rule(1), "SL k=2": sl_rule(2), "SL k=3": sl_rule(3),
                 "SL k=3+stop(c=1)": stop_cost(sl_rule(3), 1.0), "SL k=2+stop(c=1)": stop_cost(sl_rule(2), 1.0),
                 "greedy": greedy_rule(1.0)}.items():
        out[n] = report_s(build_map_s(z, r))
    return str(z), out


if __name__ == "__main__":
    import sys
    qs = Q_UNSEEN + Q_NEW
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        R = dict(pool.map(run, qs))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"semilag_{PROBLEM}.json"), "w"), indent=1, default=float)
    for q, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{PROBLEM} {lab}:\n| method | points | stops | exact bound | true error | wins |\n|---|---|---|---|---|---|")
        for n in R[str(q[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in q])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in q]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {sum(R[str(z)][n]['error'] < R[str(z)]['LW map']['error'] for z in q)}/{len(q)} |")
