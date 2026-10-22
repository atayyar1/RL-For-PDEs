"""The "stop" action: a point may skip its stencil and take the coarse-guess value instead.

A stopped point p is a leaf like the IC, but its value is the guess g_p, so its error u_p - g_p enters the identity:
    e(z*) = sum_{expanded p} beta_p delta_p + sum_{stopped p} beta_p (u_p - g_p).
With SAME_LEVEL = 0 every user of p sits on a higher level, so beta_p is final when p is popped: the rule can use it.
Estimate of the guess error (no u_true): two LW guesses, 2x and 4x coarser; LW is 2nd order, so
    u - g2 ~ (g2 - g4) / (R - 1)   (Richardson), with the ratio R observed from a third guess 8x coarser.
"""
import os
if os.environ.get("PDE", "advection") == "burgers":      # PDE=burgers python ...  -> the Burgers front problem (12.3)
    from core_burgers import *
    import core_burgers as CORE
else:
    from core import *
    import core as CORE                                    # for the private caches (not exported by *)
from scipy.interpolate import make_interp_spline


def make_guess(K):
    """Cell 8 with K_COARSE = K: GUESS[k, it, ix] = k-th x-derivative of a K-times coarser LW solve."""
    xc = np.linspace(0.0, 1.0, (nx - 1) // K + 1)
    hc = xc[1] - xc[0]
    c2, c1 = (A_MAX / hc)**2, 2 * alpha / hc**2
    tau = T / int(np.ceil(T / ((-c1 + np.sqrt(c1**2 + 4 * c2 * 0.9)) / (2 * c2))))
    tc = np.arange(0.0, T + tau / 2, tau)
    Uc = [u0(xc)]
    for j in range(1, len(tc)):
        u, ui = Uc[-1], Uc[-1][1:-1]
        n = u.copy()
        if PROBLEM == "advection":                      # cell 8 of 12.4: speed a(x), with the a a' u_x dt^2/2 term
            r, nu = alpha * tau / hc**2, speed(xc[1:-1]) * tau / hc
            n[1:-1] = ui + (r + nu**2 / 2) * (u[2:] - 2 * ui + u[:-2]) - nu / 2 * (1 - C_OU * tau / 2) * (u[2:] - u[:-2])
        else:                                           # cell 8 of 12.3: speed u
            r, nu = alpha * tau / hc**2, a(ui) * tau / hc
            n[1:-1] = ui + (r + nu**2 / 2) * (u[2:] - 2 * ui + u[:-2]) - nu / 2 * (u[2:] - u[:-2])
        n[0], n[-1] = g_left(tc[j]), g_right(tc[j])
        Uc.append(n)
    Dc = np.zeros((5, len(tc), nx))
    for j, u in enumerate(Uc):
        sp = make_interp_spline(xc, u, k=5 if len(xc) > 6 else 3)
        for k in range(5):
            Dc[k, j] = sp.derivative(k)(x_grid) if k else sp(x_grid)
    tg = np.arange(nt) * dt
    jj = np.minimum((tg // tau).astype(int), len(tc) - 2)
    f = ((tg - tc[jj]) / tau)[None, :, None]
    return (1 - f) * Dc[:, jj] + f * Dc[:, jj + 1], len(xc) * len(tc)


GUESS4, G_COARSE4 = make_guess(2 * K_COARSE)
GUESS8, G_COARSE8 = make_guess(4 * K_COARSE)
guess_value = lambda q: GUESS[0, q[1], q[0]]
# observed convergence ratio of the guesses, from the guesses only: |g4 - g8| ~ R |g2 - g4|  (R = 4 for a clean 2nd order;
# near a front LW is pre-asymptotic and R is smaller, so dividing by 3 would underestimate the error)
_d24, _d48 = np.abs(GUESS[0] - GUESS4[0])[1:, 1:-1], np.abs(GUESS4[0] - GUESS8[0])[1:, 1:-1]
R_OBS = float(np.median(_d48) / np.median(_d24))
guess_err = lambda q: abs(GUESS[0, q[1], q[0]] - GUESS4[0, q[1], q[0]]) / (R_OBS - 1)    # estimate of |u - guess| at q


# ── Map with stops: choose(m, p) returns the neighbours, or None to stop p ──
def stop_point(m):
    """Pop the next point and make it a leaf with the guess value."""
    it, ix = heapq.heappop(m["pending"])
    m.setdefault("stopped", set()).add((ix, -it))


def build_map_s(z_star, choose):
    m = start_map(z_star)
    m["stopped"] = set()
    while m["pending"]:
        p = next_point(m)
        nbs = choose(m, p)
        if nbs is None and p != z_star:
            stop_point(m)
            if m["level"] and (not m["pending"] or -m["pending"][0][0] != p[1]):
                close_level(m)                      # the stop was the last point of its level
        else:
            expand(m, nbs if nbs is not None else lw(m, p))
    if m["level"]:
        close_level(m)
    return m


PICARD = 0 if PROBLEM == "advection" else 1             # as in the notebooks' evaluate (exact speed needs none)


def evaluate_s(m, picard=PICARD):
    """Forward pass (the notebooks' evaluate), stopped points take the guess value."""
    u_hat, m["w_used"] = {q: guess_value(q) for q in m["stopped"]}, {}
    value = lambda q: known_value(q) if is_known(q) else u_hat[q]
    for p in forward_order(m):
        nbs, w = m["stencil"][p]
        v = np.array([value(q) for q in nbs])
        DX, DT = np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt
        for _ in range(picard):
            w_new = solve_weights(DX, DT, a(w @ v))     # speed from the value just computed (Burgers)
            if w_new is None:
                break
            w = w_new
        m["w_used"][p], u_hat[p] = w, w @ v
    return u_hat


def report_s(m):
    """Points (expanded), stops, true error, exact bound (stopped points add |beta (u - g)|), and the identity check."""
    z = m["z_star"]
    u_hat, W = evaluate_s(m), m["w_used"]
    beta = {z: 1.0}                                       # recomputed with the weights actually used (as in report)
    for p in forward_order(m)[::-1]:
        for q, wq in zip(m["stencil"][p][0], W[p]):
            if not is_known(q):
                beta[q] = beta.get(q, 0.0) + beta[p] * wq
    delta = {p: Ug(p) - W[p] @ np.array([Ug(q) for q in m["stencil"][p][0]]) for p in m["stencil"]}
    delta.update({q: Ug(q) - guess_value(q) for q in m["stopped"]})
    e = Ug(z) - u_hat[z]
    ident = sum(beta.get(p, 0.0) * delta[p] for p in delta)
    return {"points": len(m["stencil"]), "stops": len(m["stopped"]), "error": abs(e),
            "bound": sum(abs(beta.get(p, 0.0) * delta[p]) for p in delta), "identity_gap": abs(e - ident),
            "est": sum(abs(m["beta"][p]) * point_score(p, *m["stencil"][p]) for p in m["stencil"])
                   + sum(abs(m["beta"][q]) * guess_err(q) for q in m["stopped"])}


# ── Stop rules: wrap any rule ──
def stop_beta(rule, thr):
    """Stop when |beta_p| < thr."""
    return lambda m, p: None if abs(m["beta"][p]) < thr else rule(m, p)


def stop_budget(rule, budget):
    """Stop while the estimated cost |beta_p| g_err(p) of all stops so far stays under budget (cheapest first is
    impossible in one pass, so: a point stops if its own cost is below budget / N_LW and the running sum allows it)."""
    def choose(m, p):
        c = abs(m["beta"][p]) * guess_err(p)
        spent = m.setdefault("_spent", 0.0)
        if c < budget / n_lw(m["z_star"]) and spent + c < budget:
            m["_spent"] = spent + c
            return None
        return rule(m, p)
    return choose


def green_band(z, it, k=2.0):
    """Exact mean +- k std of the OU foot, at level it, for the query z (the Green's function support)."""
    s = (z[1] - it) * dt
    mu = X0 + (z[0] * dx - X0) * np.exp(-C_OU * s)
    sd = np.sqrt(alpha * (1 - np.exp(-2 * C_OU * s)) / C_OU)
    return mu - k * sd, mu + k * sd


def in_band_fraction(m, k=2.0):
    """Fraction of expanded points inside the exact Green band (band clipped to [0, 1]; a band past a wall
    ends on the wall, so the wall column counts as inside)."""
    if PROBLEM != "advection":                            # closed-form band only for the OU problem
        return float("nan")
    inside = 0
    for p in m["stencil"]:
        lo, hi = green_band(m["z_star"], p[1], k)
        inside += (min(lo, 1.0) - dx <= p[0] * dx <= max(hi, 0.0) + dx)
    return inside / max(len(m["stencil"]), 1)
