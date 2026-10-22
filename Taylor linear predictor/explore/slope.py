"""Burgers: the variable-speed (a') term in the u_x row, as cell 4 of the advection notebook has it.
    u_q ~ u + [xi - s DT (DX - a DT)] u_x + [xi^2 / 2 + alpha DT] u_xx,   a = u_p,  s = u_x(p)
(for Burgers u_xt = -u_x^2 - ..., u_tt = 2 u u_x^2 + ...: the advection form has DX - a DT / 2, Burgers DX - a DT)
12.3's weights drop the a' term (frozen speed). In the forward pass we know a = u_p (Picard) but not u_x: take it
from a guess (SLOPE=lw: the map's guess, SLOPE=fd5: an FD-5 2x guess), or from the computed neighbours (SLOPE=nb:
least-squares slope of the stencil's computed values, re-estimated each Picard step).
    PDE=burgers PICARD=3 python explore/slope.py
"""
import os
os.environ.setdefault("PDE", "burgers")
from evaluate import *
from burgers_guess import coarse_fd5

GF = coarse_fd5(2, nx, nt, T, alpha, u0, g_left, g_right, A_MAX, x_grid, dt)[0]


def weights_slope(DX, DT, a_loc, s):
    xi = DX - a_loc * DT
    h = max(np.abs(DX).max(), np.sqrt(alpha * np.abs(DT).max()))
    A = np.stack([np.ones_like(DX), (xi - s * DT * (DX - a_loc * DT)) / h, (0.5 * xi**2 + alpha * DT) / h**2])   # Burgers: u_tt = 2 u u_x^2 + ...  -> (DX - a DT)
    return np.linalg.pinv(A) @ np.array([1.0, 0.0, 0.0])


def evaluate_slope(m, mode, picard=3):
    """Forward pass with the a' term; returns u_hat and the weights used (so the identity can be checked)."""
    u_hat, W = {q: guess_value(q) for q in m["stopped"]}, {}
    value = lambda q: known_value(q) if is_known(q) else u_hat[q]
    for p in forward_order(m):
        nbs, w = m["stencil"][p]
        v = np.array([value(q) for q in nbs])
        DX, DT = np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt
        for _ in range(picard):
            a_ = a(w @ v)
            if mode == "lw":
                s = GUESS[1, p[1], p[0]]
            elif mode == "fd5":
                s = GF[1, p[1], p[0]]
            elif mode == "nb":                                   # slope of the computed values along the foot
                xs = DX - a_ * DT
                s = np.polyfit(xs, v, 1)[0] if np.ptp(xs) > 1e-12 else 0.0
            else:
                s = 0.0
            w = weights_slope(DX, DT, a_, s)
        W[p], u_hat[p] = w, w @ v
    return u_hat, W


if __name__ == "__main__":
    out = {}
    for z in Q_UNSEEN:
        m = build_map_s(z, lw)
        out[str(z)] = {mode: abs(Ug(z) - evaluate_slope(m, mode)[0][z]) for mode in ("none", "lw", "fd5", "nb")}
        print(z, " | ".join(f"{k} {v:.2e}" for k, v in out[str(z)].items()), flush=True)
    for mode in ("lw", "fd5", "nb"):
        print(f"slope from {mode}: geo-mean error / no slope term x{gm([out[str(z)][mode] / out[str(z)]['none'] for z in Q_UNSEEN]):.2f}, "
              f"wins {sum(out[str(z)][mode] < out[str(z)]['none'] for z in Q_UNSEEN)}/9")
