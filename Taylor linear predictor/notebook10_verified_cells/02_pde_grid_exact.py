import numpy as np
# ── PDE:  u_t + c u_x = alpha u_xx   on x in [0, 1],  u(0,t) = u(1,t) = 0 ──
alpha = 0.1          # diffusion coefficient
c     = 1.0          # advection speed

# ── Grid: every point the map can use sits on this grid ──
nx = 100
T  = 0.5
dx = 1.0 / (nx - 1)
dt_cfl = min(0.9 * dx / c, 0.45 * dx**2 / alpha)     # advective and diffusive limits
nt = max(int(np.ceil(T / dt_cfl)) + 1, 10)
dt = T / (nt - 1)

# ── Exact solution ──
# u = exp(b x - alpha b^2 t) v  with  b = c / (2 alpha)  removes the advection term;
# v solves the heat equation, so it is a sine series.
_xq = np.linspace(0.0, 1.0, 80001)
_b  = c / (2 * alpha)
u0  = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)
_v0 = u0(_xq) * np.exp(-_b * _xq)
_n  = np.arange(1, 801)
_bn = np.array([2 * np.trapezoid(_v0 * np.sin(k * np.pi * _xq), _xq) for k in _n])

def u_true(x, t):
    """Exact u at positions x (scalar or array) and one time t. Always returns an array."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    v = (np.sin(np.pi * np.outer(x, _n)) * (_bn * np.exp(-alpha * (_n * np.pi)**2 * t))).sum(axis=1)
    return np.exp(_b * x - alpha * _b**2 * t) * v

print(f"nx = {nx}, nt = {nt}, dx = {dx:.5f}, dt = {dt:.3e}")
print(f"check at t = 0: max |u_true - u0| = {np.abs(u_true(np.arange(nx) * dx, 0.0) - u0(np.arange(nx) * dx)).max():.1e}")
