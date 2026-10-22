"""Every rule-based method (with and without stops) on the 9 unseen queries + new ones. Writes results.json.
    python explore/evaluate.py              (advection, ~3 min on 4 cores)
    PDE=burgers python explore/evaluate.py  (Burgers front)
Ratios are geometric means over queries of (method / LW map of the same query)."""
import json, sys
from collections import Counter
from multiprocessing import Pool
from robust import *

LAM = 1.0
if PROBLEM == "advection":
    Q_UNSEEN = [(ix, it) for it in (25, 40, 55) for ix in (12, 25, 37)]
    Q_NEW    = [(25, 45), (6, 50), (44, 30), (25, 15), (18, 52), (33, 20)]     # (25, 45) = the agent's training query
else:                                                                        # Burgers front: queries of 12.3.x (it <= 136)
    Q_UNSEEN = [(ix, it) for it in (50, 75, 100) for ix in (12, 25, 38)]
    Q_NEW    = [(25, 130), (12, 130), (38, 130), (6, 90), (44, 60), (30, 115)]


def stencil_cells(p, nbs):
    offs = [(q[0] - p[0], q[1] - p[1]) for q in nbs]
    return tuple(sorted(WINDOW.index(o) for o in offs)) if all(o in WINDOW for o in offs) else None


def fixed_rule(cells):
    def choose(m, p):
        sets, _, _ = allowed_sets(m, p)
        return [place(p, k) for k in cells] if cells in sets else lw(m, p)
    return choose


def greedy_stop(lam):
    """Greedy, plus a myopic stop: stop when |beta| g_err / EPS beats the best one-step stencil cost."""
    def choose(m, p):
        sets, uses, sc = allowed_sets(m, p)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)])
        cost = abs(m["beta"][p]) * sc / EPS + lam * (uses.astype(float) @ new) / n_lw(m["z_star"])
        if abs(m["beta"][p]) * guess_err(p) / EPS < cost.min():
            return None
        return [place(p, k) for k in sets[int(np.argmin(cost))]]
    return choose


def stop_cost(rule, c):
    """Stop when the estimated stop cost |beta_p| g_err(p) is below c * EPS / N_LW (a price per point, like lam)."""
    return lambda m, p: None if abs(m["beta"][p]) * guess_err(p) < c * EPS / n_lw(m["z_star"]) else rule(m, p)


BASE = {"LW map": lw, "lowest score": lowest_score, "greedy": greedy_rule(LAM)}


def methods():
    M = dict(BASE)
    M["greedy+stop(myopic)"] = greedy_stop(LAM)
    for kind in ("abs", "2g"):
        M[f"lowest score[{kind}]"] = lowest_kind(kind)
        M[f"greedy[{kind}]"] = greedy_kind(LAM, kind)
        M[f"greedy[{kind}]+stop(myopic)"] = greedy_kind(LAM, kind, stop=True)
    for c in (0.1, 1.0):
        for b in ("LW map", "lowest score", "greedy"):
            M[f"{b}+stop(c={c:g})"] = stop_cost(BASE[b], c)
    return M


def run_query(z):
    out, used = {}, Counter()
    for name, rule in methods().items():
        t0 = time.time()
        m = build_map_s(z, rule)
        r = report_s(m)
        r.update(band=in_band_fraction(m), sec=time.time() - t0, est_abs=est_kind(m, "abs"), est_2g=est_kind(m, "2g"))
        out[name] = r
        if name in ("greedy", "lowest score"):
            used.update(c for p, (nbs, w) in m["stencil"].items() if (c := stencil_cells(p, nbs)) is not None)
    return z, out, used


def run_fixed(args):
    z, cands = args
    return z, {str(c): report_s(build_map_s(z, fixed_rule(c))) for c in cands}


gm = lambda v: float(np.exp(np.mean(np.log(v))))


def table(R, qs, names):
    rows = ["| method | points | stops | exact bound | true error | est/bound | est_abs/bound | est_2g/bound | in Green band | wins (err < LW) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        rel = lambda k: gm([R[str(z)][n][k] / R[str(z)]["LW map"][k] for z in qs])
        wins = sum(R[str(z)][n]["error"] < R[str(z)]["LW map"]["error"] for z in qs)
        stops = np.mean([R[str(z)][n]["stops"] for z in qs])
        cal = gm([R[str(z)][n]["est"] / R[str(z)][n]["bound"] for z in qs])
        cal2 = [gm([R[str(z)][n].get(k, np.nan) / R[str(z)][n]["bound"] for z in qs]) for k in ("est_abs", "est_2g")]
        band = np.mean([R[str(z)][n]["band"] for z in qs]) if "band" in R[str(qs[0])][n] else float("nan")
        rows.append(f"| {n} | x{rel('points'):.2f} | {stops:.0f} | x{rel('bound'):.2f} | x{rel('error'):.2f} | {cal:.2f} | {cal2[0]:.2f} | {cal2[1]:.2f} | {band:.2f} | {wins}/{len(qs)} |")
    return "\n".join(rows)


if __name__ == "__main__":
    Q = Q_UNSEEN + Q_NEW
    with Pool(4) as pool:
        res = pool.map(run_query, Q)
        R = {str(z): out for z, out, _ in res}
        used = sum((u for _, _, u in res), Counter())
        cands = [c for c, _ in used.most_common(12)]
        fix = dict(pool.map(run_fixed, [(z, cands) for z in Q]))
    # best fixed stencil: lowest exact bound over the unseen queries (an oracle choice, uses u_true)
    best = min(cands, key=lambda c: gm([fix[z][str(c)]["bound"] / R[str(z)]["LW map"]["bound"] for z in Q_UNSEEN]))
    for z in Q:
        R[str(z)]["best fixed"] = dict(fix[z][str(best)], band=float("nan"))
    json.dump({"R": R, "best_fixed": best, "G_COARSE": G_COARSE, "G_COARSE4": G_COARSE4}, open(os.path.join(os.path.dirname(__file__), f"results_{PROBLEM}.json"), "w"), indent=1, default=float)
    names = list(methods()) + ["best fixed"]
    print(f"{PROBLEM}: guess cost {G_COARSE} + {G_COARSE4} + {G_COARSE8} (2nd, 3rd guess) grid points, shared by all queries; R_OBS = {R_OBS:.2f}\n")
    print("unseen 9:\n" + table(R, Q_UNSEEN, names))
    print("\nnew 6:\n" + table(R, Q_NEW, names))
    print("\nmax identity gap:", max(r["identity_gap"] for z in R for r in R[z].values() if "identity_gap" in r))
