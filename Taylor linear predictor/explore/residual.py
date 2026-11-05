"""Residual score: apply the stencil to the guess FUNCTION instead of to its degree-4 Taylor polynomial.

    s_res(p) = | G(p) - sum_q w_q G(q) |,   G = a smooth spline (quintic in x, cubic in t) through the coarse solve.

The Taylor score needs u_xxx, u_xxxx of the guess, which are garbage near a front (85% / 370% median error for Burgers),
and it truncates at degree 4. The residual uses every order of a real function, so its cancellations are those of a
function close to u. Check: random admissible stencils at random points, s vs the true local error (u_true only here).
    PDE=burgers python explore/residual.py
"""
import os
from scipy.interpolate import RectBivariateSpline
from stop import *


def coarse_solution(K, scheme):
    """(xc, tc, U[t, x]) of a coarse solve: 'lw' = cell 8, 'fd5' = burgers_guess.coarse_fd5's solver."""
    xc = np.linspace(0.0, 1.0, (nx - 1) // K + 1)
    h, n = xc[1] - xc[0], len(xc)
    if scheme == "lw":
        c2, c1 = (A_MAX / h)**2, 2 * alpha / h**2
        tau = T / int(np.ceil(T / ((-c1 + np.sqrt(c1**2 + 4 * c2 * 0.9)) / (2 * c2))))
    else:
        tau = T / int(np.ceil(T / min(0.5 * h / A_MAX, 0.25 * h**2 / alpha)))
    tc = np.arange(0.0, T + tau / 2, tau)
    sp_ = (lambda x: speed(x)) if PROBLEM == "advection" else None
    U, u = [u0(xc)], u0(xc)
    ht = 1e-5
    wall_rate = lambda t: (np.array([g_left(t + ht), g_right(t + ht)]) - np.array([g_left(max(t - ht, 0)), g_right(max(t - ht, 0))])) / (t + ht - max(t - ht, 0))

    def rhs(v, t):
        u2, u1 = np.zeros(n), np.zeros(n)
        u2[1:-1], u1[1:-1] = v[2:] - 2 * v[1:-1] + v[:-2], (v[2:] - v[:-2]) / 2
        u2[2:-2] = (-v[4:] + 16 * v[3:-1] - 30 * v[2:-2] + 16 * v[1:-3] - v[:-4]) / 12
        u1[2:-2] = (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / 12
        a_ = v if sp_ is None else sp_(xc)
        du = alpha * u2 / h**2 - a_ * u1 / h
        du[0], du[-1] = wall_rate(t)
        return du

    for j in range(len(tc) - 1):
        if scheme == "lw":
            ui = u[1:-1]
            a_ = ui if sp_ is None else sp_(xc[1:-1])
            r, nu = alpha * tau / h**2, a_ * tau / h
            new = u.copy()
            fac = (1 - C_OU * tau / 2) if PROBLEM == "advection" else 1.0
            new[1:-1] = ui + (r + nu**2 / 2) * (u[2:] - 2 * ui + u[:-2]) - nu / 2 * fac * (u[2:] - u[:-2])
            u = new
        else:
            t = tc[j]
            k1 = rhs(u, t); k2 = rhs(u + tau * k1 / 2, t + tau / 2)
            k3 = rhs(u + tau * k2 / 2, t + tau / 2); k4 = rhs(u + tau * k3, t + tau)
            u = u + tau * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u[0], u[-1] = g_left(tc[j + 1]), g_right(tc[j + 1])
        U.append(u.copy())
    return xc, tc, np.array(U)


def guess_function(K, scheme):
    """G(ix, it) on the main grid: a quintic-in-x, cubic-in-t spline through the coarse solve (tabulated once)."""
    xc, tc, U = coarse_solution(K, scheme)
    sp = RectBivariateSpline(tc, xc, U, kx=3, ky=min(5, len(xc) - 1))
    return sp(np.arange(nt) * dt, x_grid)                     # TAB[it, ix]


def residual(TAB, p, nbs, w):
    return abs(TAB[p[1], p[0]] - w @ np.array([TAB[q[1], q[0]] for q in nbs]))


if __name__ == "__main__":
    tabs = {f"{s} {K}x": guess_function(K, s) for s, K in [("lw", 2), ("fd5", 4), ("fd5", 2)]}
    tabs["exact"] = np.array([U_row(it) for it in range(nt)])
    rng = np.random.default_rng(0)
    cells = [(i, j) for j in (-1, -2, -3) for i in range(-3, 4)]
    rows = []
    while len(rows) < 400:
        p = (int(rng.integers(3, nx - 3)), int(rng.integers(3, nt)))
        nb = [(p[0] + c[0], p[1] + c[1]) for c in (cells[k] for k in rng.choice(len(cells), 5, replace=False))]
        w = weights(p, nb)
        if w is None:
            continue
        true = abs(Ug(p) - w @ np.array([Ug(q) for q in nb]))
        rows.append([true, point_score(p, nb, w)] + [residual(t, p, nb, w) for t in tabs.values()])
    A = np.array(rows) + 1e-15
    names = ["Taylor score (cell 6)"] + [f"residual on {k}" for k in tabs]
    print(f"{PROBLEM}: 400 random stencils, weights with the guessed speed, true local error with those weights")
    for i, nme in enumerate(names):
        print(f"  {nme:26s} corr(log, log true) {np.corrcoef(np.log(A[:, 0]), np.log(A[:, i + 1]))[0, 1]:.2f} | "
              f"median est/true {np.median(A[:, i + 1] / A[:, 0]):.2f}")


# ── Rules with the residual score (maps) ──
_wsets = {}
def admissible_w(p):
    """(sets, uses, W, offsets) of p's admissible stencils (as cell 12's admissible, but keeping the weights)."""
    key = (tuple((place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS)), a_build(p))
    if key not in _wsets:
        sets, uses, _ = admissible(p)
        off = np.array([[(place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in s] for s in sets])
        W, ok = solve_weights_many(off[..., 0] * dx, off[..., 1] * dt, a_build(p))
        assert ok.all()
        _wsets[key] = (sets, uses, W, off)
    return _wsets[key]


def res_scores(p, tabs):
    """Residual score of every admissible stencil at p; with several guess tables, the max over them."""
    sets, uses, W, off = admissible_w(p)
    s = 0.0
    for TAB in tabs:
        s = np.maximum(s, np.abs(TAB[p[1], p[0]] - (W * TAB[p[1] + off[..., 1], p[0] + off[..., 0]]).sum(1)))
    return sets, uses, s


def greedy_res(lam, tabs, stop=False):
    """Greedy (cell 13) with the residual score; stop = the myopic stop."""
    def choose(m, p):
        sets, uses, sc = res_scores(p, tabs)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)])
        cost = abs(m["beta"][p]) * sc / EPS + lam * (uses.astype(float) @ new) / n_lw(m["z_star"])
        if stop and p != m["z_star"] and abs(m["beta"][p]) * guess_err(p) / EPS < cost.min():
            return None
        return [place(p, k) for k in sets[int(np.argmin(cost))]]
    return choose


def lowest_res(tabs):
    def choose(m, p):
        sets, _, sc = res_scores(p, tabs)
        return [place(p, k) for k in sets[int(np.argmin(sc))]]
    return choose
