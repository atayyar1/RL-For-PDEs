"""Solvers for u_t + c u_x = alpha u_xx on [0,1], u(0)=u(1)=0, with FLOP accounting.

Cost convention: a P-term dot product costs 2P-1 flops (P mults, P-1 adds).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import core as C


# ----------------------------------------------------------------- (a) FTCS
def solve_ftcs(N, T_end, alpha, c, ic, cfl_safety=0.9):
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    dt_lim = min(0.5 * dx ** 2 / alpha, 2 * alpha / max(c ** 2, 1e-300))
    dt = cfl_safety * dt_lim
    ns = max(int(np.ceil(T_end / dt)), 1); dt = T_end / ns
    nu = alpha * dt / dx ** 2; co = c * dt / dx
    wl, w0, wr = nu + co / 2, 1 - 2 * nu, nu - co / 2
    assert min(wl, w0, wr) >= -1e-14, "FTCS not positive"
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0
    for _ in range(ns):
        un = u.copy()
        un[1:-1] = wl * u[:-2] + w0 * u[1:-1] + wr * u[2:]
        un[0] = un[-1] = 0.0
        u = un
    flops = ns * (N - 2) * 5
    return x, u, dict(flops=flops, steps=ns, dt=dt, nu=nu, stencil=3, N=N)


# ----------------------------------------------------------------- (c) Crank-Nicolson
def solve_cn(N, T_end, alpha, c, ic, nsteps):
    from scipy.linalg import lu_factor, lu_solve
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    dt = T_end / nsteps
    nu = alpha * dt / dx ** 2; co = c * dt / dx
    n = N - 2
    lo, di, up = nu / 2 + co / 4, -nu, nu / 2 - co / 4      # (dt/2) L
    import scipy.linalg as sla
    ab_L = np.zeros((3, n)); ab_L[0, 1:] = -up; ab_L[1, :] = 1 - di; ab_L[2, :-1] = -lo
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0
    for _ in range(nsteps):
        r = u[1:-1] + lo * u[:-2] + di * u[1:-1] + up * u[2:]
        u[1:-1] = sla.solve_banded((1, 1), ab_L, r)
        u[0] = u[-1] = 0.0
    # 5 flops RHS + Thomas (2 fwd + 3 back) = 10 flops/point/step
    flops = nsteps * n * 10
    return x, u, dict(flops=flops, steps=nsteps, dt=dt, nu=nu, stencil=3, N=N)


# ----------------------------------------------------------------- (b) wide positive
def wide_weights(dt_big, dx, alpha, c, s, mcap=None, mode="exact"):
    """Positive weights approximating exp(dt_big * L) as a lattice measure.

    s = m/sigma safety factor. Returns (w, m) or (None, m) if infeasible.
    """
    sig = np.sqrt(2 * alpha * dt_big) / dx           # kernel sd, cells
    mu = -c * dt_big / dx                            # departure shift, cells
    m = int(np.ceil(s * sig + abs(mu))) + 1
    if mcap is not None:
        m = min(m, mcap)
    m = max(m, 1)
    M2 = sig ** 2 + mu ** 2 if mode == "exact" else sig ** 2
    w = C.kernel_moment_matched(m, mu, M2)
    if w is None:                                    # fall back: extremal 3-point
        p = M2 / m ** 2
        if p > 1:
            return None, m
        w = np.zeros(2 * m + 1); w[0] = w[-1] = p / 2; w[m] = 1 - p
        # first moment via a shift of mass between 0 and +-1
        j = np.arange(-m, m + 1)
        need = mu - (w * j).sum()
        if abs(need) <= w[m]:
            w[m] -= abs(need); w[m + int(np.sign(need))] += abs(need)
        else:
            return None, m
    return w, m


def apply_wide_dirichlet(u, w, m, mode="images"):
    """One big step on a Dirichlet grid.  'images' = odd reflection (exact for u=0)."""
    N = len(u)
    if mode == "images":
        L = 2 * (N - 1)
        U = np.zeros(L)
        U[:N] = u
        U[N:] = -u[-2:0:-1]
        idx = np.arange(N)
        out = np.zeros(N)
        for l, wl in zip(range(-m, m + 1), w):
            if wl != 0.0:
                out += wl * U[(idx + l) % L]
        out[0] = out[-1] = 0.0
        return out
    elif mode == "zeropad":
        U = np.zeros(N + 2 * m); U[m:m + N] = u
        out = np.zeros(N)
        for l, wl in zip(range(-m, m + 1), w):
            if wl != 0.0:
                out += wl * U[m + l: m + l + N]
        out[0] = out[-1] = 0.0
        return out
    raise ValueError(mode)


def solve_wide(N, T_end, alpha, c, ic, nsteps, s, bmode="images", mcap=None):
    """Wide positive explicit stencil.  Advection handled exactly by the
    Cole-Hopf-like change of variable V = u exp(-beta x), beta = c/(2 alpha),
    which turns the problem into pure diffusion with V(0)=V(1)=0 so that the
    method of images gives the EXACT Dirichlet Green's function."""
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    dt_big = T_end / nsteps
    beta = c / (2 * alpha)
    w, m = wide_weights(dt_big, dx, alpha, 0.0, s, mcap=mcap)   # pure diffusion kernel
    if w is None:
        return x, None, dict(flops=np.inf, steps=nsteps, m=m, fail=True)
    P = int((w > 1e-16).sum())
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0
    ef, eb = np.exp(-beta * x), np.exp(beta * x)
    decay = np.exp(-alpha * beta ** 2 * dt_big)
    use_transform = abs(c) > 0
    for _ in range(nsteps):
        v = u * ef if use_transform else u
        v = apply_wide_dirichlet(v, w, m, bmode)
        u = (v * eb) * decay if use_transform else v
        u[0] = u[-1] = 0.0
    flops = nsteps * (N - 2) * (2 * P - 1)
    if use_transform:
        flops += nsteps * N * 3                    # 2 mults + 1 mult for decay
    return x, u, dict(flops=flops, steps=nsteps, m=m, P=P, dt=dt_big,
                      sigma_cells=np.sqrt(2 * alpha * dt_big) / dx,
                      wmin=float(w.min()), N=N, fail=False)


# ----------------------------------------------------------------- reference: FFT sine
def solve_spectral(N, T_end, alpha, c, ic):
    """Exact evolution of the grid data in the sine basis (DST) - reference cost."""
    from scipy.fft import dst, idst
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    beta = c / (2 * alpha)
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0
    v = u * np.exp(-beta * x)
    a = dst(v[1:-1], type=1)
    n = np.arange(1, N - 1)
    a = a * np.exp(-alpha * (n * np.pi) ** 2 * T_end)
    v2 = np.zeros(N); v2[1:-1] = idst(a, type=1)
    u2 = v2 * np.exp(beta * x) * np.exp(-alpha * beta ** 2 * T_end)
    u2[0] = u2[-1] = 0.0
    flops = 2 * 5 * N * np.log2(max(N, 2)) + 6 * N
    return x, u2, dict(flops=flops, steps=1, N=N)


# ----------------------------------------------------------------- RKC (the real competitor)
def solve_rkc(N, T_end, alpha, c, ic, nsteps, s, order=1, eps_damp=2.0 / 13.0):
    """Runge-Kutta-Chebyshev with s stages.

    order=1: R_s(z) = T_s(1 + z/s^2), stability z in [-2 s^2, 0].
    order=2: damped RKC2, stability boundary ~0.653 s^2 (standard Verwer et al. form).
    Stencil footprint is s cells; cost is s applications of the 3-point operator.
    """
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    dt = T_end / nsteps
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0

    def L(v):
        o = np.zeros_like(v)
        o[1:-1] = (alpha * (v[:-2] - 2 * v[1:-1] + v[2:]) / dx ** 2
                   - c * (v[2:] - v[:-2]) / (2 * dx))
        return o

    if order == 1:
        for _ in range(nsteps):
            Y0 = u.copy()
            Y1 = u + (dt / s ** 2) * L(u)
            Tm2, Tm1 = 1.0, 1.0
            for j in range(2, s + 1):
                Tj = 2 * Tm1 - Tm2
                mu, nuj = 2 * Tm1 / Tj, -Tm2 / Tj
                Y2 = mu * Y1 + nuj * Y0 + (mu / s ** 2) * dt * L(Y1)
                Y0, Y1 = Y1, Y2
                Tm2, Tm1 = Tm1, Tj
            u = Y1; u[0] = u[-1] = 0.0
    else:
        w0 = 1 + eps_damp / s ** 2
        # Chebyshev values at w0
        T = [1.0, w0]
        Tp = [0.0, 1.0]
        Tpp = [0.0, 0.0]
        for j in range(2, s + 1):
            T.append(2 * w0 * T[j - 1] - T[j - 2])
            Tp.append(2 * T[j - 1] + 2 * w0 * Tp[j - 1] - Tp[j - 2])
            Tpp.append(4 * Tp[j - 1] + 2 * w0 * Tpp[j - 1] - Tpp[j - 2])
        w1 = Tp[s] / Tpp[s]
        b = [Tpp[j] / Tp[j] ** 2 if Tp[j] != 0 else 1.0 / (2 * w0) for j in range(s + 1)]
        b[0] = b[2] if s >= 2 else b[1]; b[1] = b[2] if s >= 2 else b[1]
        for _ in range(nsteps):
            F0 = L(u)
            Y0 = u.copy()
            Y1 = u + (b[1] * w1) * dt * F0
            for j in range(2, s + 1):
                mu = 2 * b[j] * w0 / b[j - 1]
                nuj = -b[j] / b[j - 2]
                mut = 2 * b[j] * w1 / b[j - 1]
                gt = -(1 - b[j - 1] * T[j - 1]) * mut
                Y2 = (1 - mu - nuj) * u + mu * Y1 + nuj * Y0 + mut * dt * L(Y1) + gt * dt * F0
                Y0, Y1 = Y1, Y2
            u = Y1; u[0] = u[-1] = 0.0
    flops = nsteps * s * (N - 2) * 5 + nsteps * s * (N - 2) * 4   # L + combination
    return x, u, dict(flops=flops, steps=nsteps, s=s, dt=dt, N=N)
