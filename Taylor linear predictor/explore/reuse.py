"""Reuse stops (Ali's idea): a point stops by taking a stencil made ONLY of points the map already has (or data),
found by a nearest-neighbour search beyond the 21-cell window. No new point, no guess value in u_hat.

Only points on LOWER levels are candidates: they are still pending (or data), so they are expanded after p and their
beta receives p's share; the identity stays exact (it is just another stencil with Taylor weights).
Neighbours are searched in the frame of the characteristic: distance of (x_q - (x_p - a_p * dT), dT) scaled by dx and
dt, so the search looks along the foot of the characteristic first. Among the K nearest, every 5-subset with
admissible weights is scored (Taylor score at p, any offsets), and the lowest is the reuse stencil.
    python explore/reuse.py [PDE=burgers]
"""
from scipy.spatial import cKDTree
from evaluate import *

R_FAR, KT_FAR, K_NN = 8, 6, 8          # search box |dix| <= 8, 1..6 levels down; the 8 nearest are tried


def reuse_stencil(m, p):
    """(neighbours, weights, score) of the best stencil of existing points / data below p, or None."""
    ap = a_build(p)
    cand = set()
    for q in m["beta"]:
        if q[1] < p[1] and p[1] - q[1] <= KT_FAR and abs(q[0] - p[0]) <= R_FAR and q not in m.get("stopped", ()):
            cand.add(q)
    for dit in range(1, KT_FAR + 1):                       # data in the box: IC row and the walls
        for dix in range(-R_FAR, R_FAR + 1):
            q = (min(max(p[0] + dix, 0), nx - 1), max(p[1] - dit, 0))
            if is_known(q):
                cand.add(q)
    cand = sorted(cand)
    if len(cand) < N_NB:
        return None
    C = np.array(cand)
    DT = (C[:, 1] - p[1]) * dt
    feat = np.c_[(C[:, 0] * dx - (p[0] * dx + ap * DT)) / dx, DT / dt]
    _, idx = cKDTree(feat).query([0.0, 0.0], k=min(K_NN, len(cand)))
    near = [cand[i] for i in np.atleast_1d(idx)]
    subsets = list(itertools.combinations(near, N_NB))
    off = np.array([[(q[0] - p[0], q[1] - p[1]) for q in s] for s in subsets], dtype=float)
    W, ok = solve_weights_many(off[..., 0] * dx, off[..., 1] * dt, ap)
    if not ok.any():
        return None
    sc = np.array([point_score(p, list(s), w) if o else np.inf for s, w, o in zip(subsets, W, ok)])
    i = int(np.argmin(sc))
    return list(subsets[i]), sc[i]


def with_reuse(rule, c=None):
    """rule, but p takes its reuse stencil when that is cheaper:
    c = None: when |beta| s_reuse / EPS beats the rule's own stencil cost |beta| s / EPS + lam new / N_LW (lam = 1)
    c given : when |beta| s_reuse < c EPS / N_LW (a price per point, as stop_cost)."""
    def choose(m, p):
        nbs = rule(m, p)
        if p == m["z_star"] or nbs is None:
            return nbs
        r = reuse_stencil(m, p)
        if r is None:
            return nbs
        b = abs(m["beta"][p])
        if c is None:
            w = weights(p, nbs)
            new = sum(1 for q in set(nbs) if not is_known(q) and q not in m["beta"])
            own = b * point_score(p, nbs, w) / EPS + new / n_lw(m["z_star"])
            take = b * r[1] / EPS < own
        else:
            take = b * r[1] < c * EPS / n_lw(m["z_star"])
        return r[0] if take else nbs
    return choose


def run(z):
    out = {}
    for n, r in {"LW map": lw, "LW+stop(c=1)": stop_cost(lw, 1.0), "LW+reuse(c=1)": with_reuse(lw, 1.0),
                 "LW+reuse(c=10)": with_reuse(lw, 10.0), "greedy": greedy_rule(1.0), "greedy+reuse": with_reuse(greedy_rule(1.0)),
                 "greedy+stop(myopic)": greedy_stop(1.0), "greedy+stop(myopic)+reuse": with_reuse(greedy_stop(1.0))}.items():
        out[n] = report_s(build_map_s(z, r))
    return str(z), out


if __name__ == "__main__":
    with Pool(4) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"reuse_{PROBLEM}.json"), "w"), indent=1, default=float)
    for qs, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{PROBLEM} {lab}:\n| method | points | stops | exact bound | true error | wins |\n|---|---|---|---|---|---|")
        for n in R[str(qs[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in qs])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in qs]):.0f} | x{rel('bound'):.2f} | "
                  f"x{rel('error'):.2f} | {sum(R[str(z)][n]['error'] < R[str(z)]['LW map']['error'] for z in qs)}/{len(qs)} |")
    print("max identity gap", max(r["identity_gap"] for v in R.values() for r in v.values()))
