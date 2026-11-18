"""Burgers front: a score robust to the speed error.

In Burgers the map's weights use the GUESSED speed a_g = u_guess(p), but the forward pass re-solves them with the live
speed (picard 1). A stencil whose guessed Taylor terms cancel at a_g can be bad at the speed actually used, so the score
of the chosen stencil is not the score of the used one. Robust score: the worst score over the speeds the forward pass
may use, a in {a_g - D, a_g, a_g + D}, D = the estimated guess error at p (two-guess Richardson, stop.py), at least one
A_STEP. A stencil not admissible at one of these speeds is not allowed (its weights may jump).
    PDE=burgers python explore/burgers_robust.py
"""
import os
os.environ["PDE"] = "burgers"
from evaluate import *

_at_cache = {}
def admissible_at(p, a_loc):
    """Cell 12's admissible(), at a given speed: {stencil: moments}."""
    key = (tuple((place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS)), a_loc)
    if key not in _at_cache:
        cells, seen = [], set()
        for k in range(N_CELLS):
            if place(p, k) not in seen:
                seen.add(place(p, k)); cells.append(k)
        sets = list(itertools.combinations(cells, min(N_NB, len(cells))))
        off = np.array([[(place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in s] for s in sets])
        DX, DT = off[..., 0] * dx, off[..., 1] * dt
        W, ok = solve_weights_many(DX, DT, a_loc)
        mom = np.stack([(W[ok] * DX[ok]**i * DT[ok]**j).sum(1) for i, j in TERMS], axis=1)
        _at_cache[key] = dict(zip([s for s, o in zip(sets, ok) if o], mom))
    return _at_cache[key]


def robust_scores(p):
    """(sets, uses, robust score) for p's admissible stencils (as allowed_sets orders them)."""
    sets, uses, sc = admissible(p)
    c = taylor_coeffs(guess_derivs(p))
    c = np.array([c[t] for t in TERMS])
    D = A_STEP * max(1, int(np.ceil(guess_err(p) / A_STEP)))
    worst = sc.copy()
    for a_loc in (a_build(p) - D, a_build(p) + D):
        tab = admissible_at(p, round(a_loc / A_STEP) * A_STEP)
        s = np.array([abs(tab[st] @ c) if st in tab else np.inf for st in sets])
        worst = np.maximum(worst, s)
    return sets, uses, worst


def greedy_robust(lam, stop=False):
    def choose(m, p):
        sets, uses, sc = robust_scores(p)
        if not np.isfinite(sc).any():
            return lw(m, p)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)])
        cost = abs(m["beta"][p]) * sc / EPS + lam * (uses.astype(float) @ new) / n_lw(m["z_star"])
        if stop and p != m["z_star"] and abs(m["beta"][p]) * guess_err(p) / EPS < cost.min():
            return None
        return [place(p, k) for k in sets[int(np.argmin(cost))]]
    return choose


def lowest_robust(m, p):
    sets, _, sc = robust_scores(p)
    return [place(p, k) for k in sets[int(np.argmin(sc))]] if np.isfinite(sc).any() else lw(m, p)


def run(z):
    out = {}
    for name, rule in [("LW map", lw), ("greedy", greedy_rule(LAM)), ("lowest robust", lowest_robust),
                       ("greedy robust", greedy_robust(LAM)), ("greedy robust + myopic stop", greedy_robust(LAM, stop=True)),
                       ("LW + stop(c=1)", stop_cost(lw, 1.0))]:
        out[name] = report_s(build_map_s(z, rule))
    return str(z), out


if __name__ == "__main__":
    with Pool(4) as pool:
        R = dict(pool.map(run, Q_UNSEEN + Q_NEW))
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "burgers_robust.json"), "w"), indent=1, default=float)
    for qs, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
        print(f"\n{lab}:\n| method | points | stops | exact bound | true error | est/bound | wins |\n|---|---|---|---|---|---|---|")
        for n in R[str(qs[0])]:
            rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in qs])
            print(f"| {n} | x{rel('points'):.2f} | {np.mean([R[str(z)][n]['stops'] for z in qs]):.0f} | x{rel('bound'):.2f} | x{rel('error'):.2f} | "
                  f"{gm([R[str(z)][n]['est'] / R[str(z)][n]['bound'] for z in qs]):.2f} | "
                  f"{sum(R[str(z)][n]['error'] < R[str(z)]['LW map']['error'] for z in qs)}/{len(qs)} |")
