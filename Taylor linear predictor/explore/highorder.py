"""More Taylor conditions: higher-order weights (a formulation change).

Cell 4 imposes 3 conditions (value, u_x, u_xx) on 5 neighbours, so every stencil is at most 2nd order and the other
2 degrees of freedom only minimise ||w||. For a frozen speed a the PDE gives EVERY Taylor row exactly:
    u(x + dx, t + dt) = exp(xi d/dx + alpha dt d^2/dx^2) u,   xi = dx - a dt
so the coefficient of the n-th x-derivative is  c_n = sum_j xi^(n-2j) (alpha dt)^j / ((n-2j)! j!).
NROWS = 4 or 5 imposes rows n = 0 .. NROWS-1 (with 5 rows the 5 weights are unique: a 4th-order stencil).
For advection row 1 keeps cell 4's a' correction; the higher rows use the frozen speed (a' terms neglected there).
NROWS is set by the env var NROWS (default 3 = the notebook). Installs itself into the core before anything is built.
    NROWS=5 PDE=burgers python explore/highorder.py
"""
import os
from math import factorial
from evaluate import *

NROWS = int(os.environ.get("NROWS", 3))
_solve3 = CORE.solve_weights_many


def solve_weights_many_R(DX, DT, a_loc):
    if NROWS == 3:
        return _solve3(DX, DT, a_loc)
    xi = DX - a_loc * DT
    h = np.maximum(np.abs(DX).max(1), np.sqrt(alpha * np.abs(DT).max(1)))[:, None]
    rows = []
    for n in range(min(NROWS, DX.shape[1])):                                 # never more rows than neighbours
        c = sum(xi**(n - 2 * j) * (alpha * DT)**j / (factorial(n - 2 * j) * factorial(j)) for j in range(n // 2 + 1))
        if n == 1 and PROBLEM == "advection":
            c = xi - C_OU * DT * (DX - 0.5 * a_loc * DT)
        rows.append(c / h**n)
    A = np.stack(rows, axis=1)                                               # (M, NROWS, n)
    s = np.linalg.svd(A, compute_uv=False)
    ok = (s[:, -1] > s[:, 0] * DX.shape[1] * np.finfo(float).eps) & (s[:, 0] <= COND_MAX * s[:, -1])
    e0 = np.zeros(len(rows)); e0[0] = 1.0
    W = np.linalg.pinv(A) @ e0
    return W, ok & (np.abs(W).sum(1) <= W_MAX)


CORE.solve_weights_many = solve_weights_many_R                               # core's weights / admissible / solve_weights
CORE._w_cache.clear(); CORE._sets_cache.clear()

W5 = lambda m, p: [(p[0] + d, p[1] - 1) for d in (-2, -1, 0, 1, 2)]           # 5 cells one level down
W5c = lambda m, p: [(min(max(p[0] + d, 0), nx - 1), p[1] - 1) for d in (-2, -1, 0, 1, 2)]


def w5_rule(m, p):
    """5 cells one level down (moved onto the wall near it; LW where the 5 cells collapse or are not admissible)."""
    nbs = W5c(m, p)
    if len(set(nbs)) == N_NB and weights(p, nbs) is not None:
        return nbs
    return lw(m, p)


def run(z):
    out = {}
    for n, r in {"LW map": lw, "W5": w5_rule, "W5+stop(c=1)": stop_cost(w5_rule, 1.0),
                 "greedy": greedy_rule(1.0), "greedy+stop(myopic)": greedy_stop(1.0)}.items():
        out[n] = report_s(build_map_s(z, r))
    return str(z), out


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"highorder_{PROBLEM}_rows{NROWS}.json"), "w"), indent=1, default=float)
    print(f"{PROBLEM}, NROWS = {NROWS}: ratios to the notebook's LW map (3 points, 3 rows)")
    for q, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{lab}:\n| method | points | stops | exact bound | true error | median true error | wins |\n|---|---|---|---|---|---|---|")
        for n in R[str(q[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in q])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in q]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {np.median([R[str(z)][n]['error'] for z in q]):.1e} | {sum(R[str(z)][n]['error'] < R[str(z)]['LW map']['error'] for z in q)}/{len(q)} |")
