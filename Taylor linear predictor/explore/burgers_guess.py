"""Better guesses for Burgers: a higher-order coarse solve instead of coarse LW.
GUESS=fd5 (env) -> stop.py installs these guesses (2x, 4x, 8x coarser) in place of the LW ones.
    FD-5 + RK4 : 4th-order central differences (2nd order next to the walls), classical RK4, IC + wall data only.
Cost (counted honestly): nodes x steps x 4 stages.
    PDE=burgers python explore/burgers_guess.py     -> guess accuracy, LW guess vs FD-5 guess
"""
import numpy as np
from scipy.interpolate import make_interp_spline


def coarse_fd5(K, nx, nt, T, alpha, u0, g_left, g_right, A_MAX, x_grid, dt, gd=None, h_t=1e-5):
    """GUESS[k, it, ix] from an FD-5 + RK4 solve on a grid K times coarser, and its cost."""
    xc = np.linspace(0.0, 1.0, (nx - 1) // K + 1)
    h, n = xc[1] - xc[0], len(xc)
    tau = T / int(np.ceil(T / min(0.5 * h / A_MAX, 0.25 * h**2 / alpha)))
    tc = np.arange(0.0, T + tau / 2, tau)
    wall_rate = lambda t: (np.array([g_left(t + h_t), g_right(t + h_t)]) - np.array([g_left(t - h_t), g_right(t - h_t)])) / (2 * h_t)

    def rhs(u, t):
        u2, u1 = np.zeros(n), np.zeros(n)
        u2[1:-1], u1[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2], (u[2:] - u[:-2]) / 2
        u2[2:-2] = (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]) / 12
        u1[2:-2] = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / 12
        du = alpha * u2 / h**2 - u * u1 / h
        du[0], du[-1] = wall_rate(max(t, h_t))
        return du

    Uc, u = [u0(xc)], u0(xc)
    for j in range(len(tc) - 1):
        t = tc[j]
        k1 = rhs(u, t); k2 = rhs(u + tau * k1 / 2, t + tau / 2)
        k3 = rhs(u + tau * k2 / 2, t + tau / 2); k4 = rhs(u + tau * k3, t + tau)
        u = u + tau * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u[0], u[-1] = g_left(tc[j + 1]), g_right(tc[j + 1])
        Uc.append(u.copy())
    # same fit as cell 8: quintic spline in x, linear in t onto the main grid
    Dc = np.zeros((5, len(tc), len(x_grid)))
    for j, v in enumerate(Uc):
        sp = make_interp_spline(xc, v, k=min(5, n - 1 - (n % 2 == 1 and n < 7)))   # odd order, fewer nodes -> lower
        for k in range(min(5, sp.k + 1)):                                            # derivatives above the order: 0
            Dc[k, j] = sp.derivative(k)(x_grid) if k else sp(x_grid)
    tg = np.arange(nt) * dt
    jj = np.minimum((tg // tau).astype(int), len(tc) - 2)
    f = ((tg - tc[jj]) / tau)[None, :, None]
    return (1 - f) * Dc[:, jj] + f * Dc[:, jj + 1], n * (len(tc) - 1) * 4


if __name__ == "__main__":
    import os
    os.environ["PDE"] = "burgers"
    from stop import *
    rng = np.random.default_rng(0)
    P = [(int(rng.integers(1, nx - 1)), int(rng.integers(1, nt))) for _ in range(400)]
    ex = np.array([x_derivs(lambda x: u_true(x, q[1] * dt), q[0] * dx) for q in P])
    steep = np.abs(ex[:, 1]) >= np.quantile(np.abs(ex[:, 1]), 0.9)
    for name, G, cost in [("LW 2x", GUESS, G_COARSE)] + [
            (f"FD-5 {K}x", *coarse_fd5(K, nx, nt, T, alpha, u0, g_left, g_right, A_MAX, x_grid, dt)) for K in (1, 2, 4)]:
        gs = np.array([G[:, q[1], q[0]] for q in P])
        rel = np.abs(gs - ex) / (np.abs(ex) + 1e-12)
        print(f"{name:8s} cost {cost:7d} | median rel err u..u_xxxx: " + " ".join(f"{v:.3f}" for v in np.median(rel, 0))
              + " | steepest 10%: " + " ".join(f"{v:.3f}" for v in np.median(rel[steep], 0)))
