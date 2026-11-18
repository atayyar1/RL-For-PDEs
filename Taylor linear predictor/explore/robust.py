"""Scores that cannot be gamed by cancellation between Taylor terms.

The score s_p = |sum_terms mom_t c_t| (cell 12) can be made tiny by a stencil whose guessed Taylor terms cancel.
The guess is only ~1-10% accurate, so the cancellation is not real. Two fixes, both from estimates only:
  abs : s_abs = sum_terms |mom_t c_t|  (no cancellation across terms; an upper bound of the Taylor estimate)
  2g  : s_2g  = max(s with the 2x guess, s with the 4x guess)  (a cancellation must hold for both guesses)
"""
from stop import *
assert SAME_LEVEL == 0, "these rules use admissible() directly (no forbidden cells)"


def _moments(p):
    """(sets, uses, moments) of p's admissible stencils (fills the cache of cell 12)."""
    admissible(p)
    key = (tuple((place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS)), a_build(p))
    return CORE._sets_cache[key]


def _coeffs(G, p):
    d = G[:, p[1], p[0]]
    c = taylor_coeffs(d, a_build(p)) if PROBLEM == "advection" else taylor_coeffs(d)     # Burgers: the speed is u
    return np.array([c[t] for t in TERMS])


def scores(p, kind):
    sets, uses, mom = _moments(p)
    if kind == "abs":
        return sets, uses, np.abs(mom * _coeffs(GUESS, p)).sum(1)
    if kind == "2g":
        return sets, uses, np.maximum(np.abs(mom @ _coeffs(GUESS, p)), np.abs(mom @ _coeffs(GUESS4, p)))
    return sets, uses, np.abs(mom @ _coeffs(GUESS, p))


def point_score_kind(p, nbs, w, kind):
    """Score of one placed stencil with the given kind (for the est/bound calibration of any map)."""
    DX, DT = np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt
    mom = np.array([(w * DX**i * DT**j).sum() for i, j in TERMS])
    if kind == "abs":
        return float(np.abs(mom * _coeffs(GUESS, p)).sum())
    if kind == "2g":
        return float(max(abs(mom @ _coeffs(GUESS, p)), abs(mom @ _coeffs(GUESS4, p))))
    return float(abs(mom @ _coeffs(GUESS, p)))


def greedy_kind(lam, kind, stop=False):
    """Greedy (cell 13) with the score of the given kind; stop = the myopic stop of evaluate.greedy_stop."""
    def choose(m, p):
        sets, uses, sc = scores(p, kind)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)])
        cost = abs(m["beta"][p]) * sc / EPS + lam * (uses.astype(float) @ new) / n_lw(m["z_star"])
        if stop and abs(m["beta"][p]) * guess_err(p) / EPS < cost.min():
            return None
        return [place(p, k) for k in sets[int(np.argmin(cost))]]
    return choose


def lowest_kind(kind):
    def choose(m, p):
        sets, _, sc = scores(p, kind)
        return [place(p, k) for k in sets[int(np.argmin(sc))]]
    return choose


def est_kind(m, kind):
    """Estimated bound of a finished map with this score kind (stops cost |beta| g_err)."""
    return (sum(abs(m["beta"][p]) * point_score_kind(p, *m["stencil"][p], kind) for p in m["stencil"])
            + sum(abs(m["beta"][q]) * guess_err(q) for q in m.get("stopped", ())))
