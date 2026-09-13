"""G1 -- time-marching learned/projected scattered-point operators.

Eulerian scattered points on [0,1], u = 0 at both walls, u_t + c u_x = alpha u_xx.
Every arm produces, for each interior node, a weight vector over a local stencil
drawn from the previous one or two time levels. Because the points do not move,
each arm is a FIXED linear one-step map M (or a two-level pair M1, M2), so
stability is a property of that matrix and is measured directly (spectral radius,
||M^n||) as well as by marching.

Arms
  minnorm     min-2-norm consistent propagator weights  == SpeND projection with w~ = 0
  projgauss   SpeND projection of a Gaussian-kernel raw guess w~
  spectral    consistent weights minimising a dispersion+dissipation LS loss over k
  spectralg   same, plus SpeND's one-sided penalty on |g(k)| > 1 (gamma = 10)
  molfe       spatial D1, D2 by min-norm projection (p=2), forward Euler in time
  molrk3      same spatial operators, 3-stage RK in time
  maxent      max-entropy point of {w >= 0 : A w = b}          (certified)
  lp          an LP vertex of the same set, zero objective     (certified, control)
  lprand      an LP vertex with a random objective             (check 3)
  maxentw     maxent with the stencil widened until feasible   (certified, adaptive)
Infeasible nodes fall back to minnorm and are counted.
"""
import os, sys
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from core.pde import Problem                                   # noqa: E402
from core import stencil as st                                 # noqa: E402

ARMS_CONSISTENT = ["minnorm", "projgauss", "spectral", "spectralg", "molfe", "molrk3"]
ARMS_POSITIVE = ["maxent", "lp", "lprand", "maxentw"]
ARMS = ARMS_CONSISTENT + ARMS_POSITIVE


# ----------------------------------------------------------------- points
def make_points(nx, disorder, seed=0, kind="jitter"):
    """nx points on [0,1] with both walls fixed. Mean spacing h = 1/(nx-1).

    jitter: uniform grid + disorder*h*U(-1,1) on interior nodes (disorder < 0.5 keeps order).
    random: gaps = 0.3 + Exp(0.7), rescaled so mean gap = h  (min gap >= 0.3 h).
    """
    rng = np.random.default_rng(seed)
    h = 1.0 / (nx - 1)
    if kind == "jitter":
        x = np.linspace(0.0, 1.0, nx)
        x[1:-1] += disorder * h * rng.uniform(-1, 1, nx - 2)
    elif kind == "random":
        gaps = 0.3 + rng.exponential(0.7, nx - 1)
        x = np.concatenate([[0.0], np.cumsum(gaps)])
        x /= x[-1]
    else:
        raise ValueError(kind)
    assert np.all(np.diff(x) > 0)
    return x


def time_step(x, prob, safety=0.5):
    """dt = safety * min(h_min/c, h_min^2/(2 alpha)) on the ACTUAL minimum spacing."""
    hmin = np.diff(x).min()
    adv = hmin / abs(prob.c) if prob.c else np.inf
    dif = hmin**2 / (2 * prob.alpha) if prob.alpha else np.inf
    return safety * min(adv, dif)


def neighbours(x, i, K):
    """Indices of the K nearest nodes to node i (including i)."""
    return np.argsort(np.abs(x - x[i]), kind="stable")[:K]


# ------------------------------------------------------- weight builders
def _rows(dxi, dti, prob):
    return st.rows_taylor(dxi, dti, prob)


def w_minnorm(A, b, **k):
    return st.solve_minnorm(A, b)


def w_projgauss(A, b, dxi, dti, prob, h, **k):
    """SpeND Eq. 5 with a Gaussian raw guess in the characteristic offset."""
    xi = dxi - prob.c * dti
    wt = np.exp(-0.5 * (xi / h) ** 2)
    wt /= wt.sum()
    return wt - np.linalg.pinv(A) @ (A @ wt - b)


def _nullspace(A):
    u, s, vt = np.linalg.svd(A)
    r = (s > 1e-12 * s[0]).sum()
    return vt[r:].T                      # (n, n-r)


def _symbol_parts(dxi, dti, prob, h, nk=64):
    """Rows for g(k) = sum_j w_j exp(i k dx_j) and the exact one-step symbol.

    Time offsets: a point at level n-1 sits at dti = -2dt; the exact symbol relative
    to the target is exp(-i k c (-dti) - alpha k^2 (-dti)) applied to u at that level,
    so for mixed levels the target is per-point. Real and imaginary parts stacked.
    """
    ks = np.linspace(0, np.pi / h, nk + 1)[1:]
    E = np.exp(1j * np.outer(ks, dxi))                            # (nk, n)
    tau = -dti                                                    # (n,) positive
    # exact: u_hat(t*) = exp(-(i k c + alpha k^2) tau_j) u_hat(t_j). For a
    # single level every tau_j is equal, and the target is a scalar per k.
    tau0 = tau.min()
    g_ex = np.exp(-(1j * ks * prob.c + prob.alpha * ks**2) * tau0)
    # points at older levels carry the extra propagation inside their column
    E = E * np.exp(-(1j * ks[:, None] * prob.c + prob.alpha * ks[:, None]**2) * (tau - tau0)[None, :])
    return ks, E, g_ex


def w_spectral(A, b, dxi, dti, prob, h, gamma=0.0, **k):
    """Consistent w minimising sum_k |g(k) - g_exact(k)|^2 (+ gamma * hinge on |g|>1)."""
    w0 = st.solve_minnorm(A, b)
    if w0 is None:
        return None
    N = _nullspace(A)
    if N.shape[1] == 0:
        return w0
    ks, E, g_ex = _symbol_parts(dxi, dti, prob, h)
    B = np.vstack([np.real(E @ N), np.imag(E @ N)])
    r = np.concatenate([np.real(g_ex - E @ w0), np.imag(g_ex - E @ w0)])
    z = np.linalg.lstsq(B, r, rcond=None)[0]
    if gamma > 0:
        def f(z):
            g = E @ (w0 + N @ z)
            d = g - g_ex
            hinge = np.maximum(np.abs(g) - 1.0, 0.0)
            return np.sum(np.abs(d) ** 2) + gamma * np.sum(hinge**2)
        z = minimize(f, z, method="L-BFGS-B").x
    return w0 + N @ z


def w_spectralg(A, b, dxi, dti, prob, h, **k):
    return w_spectral(A, b, dxi, dti, prob, h, gamma=10.0)


def w_maxent(A, b, **k):
    return st.solve_maxent(A, b)


def w_lp(A, b, **k):
    return st.solve_positive(A, b)


def w_lprand(A, b, rng=None, **k):
    rng = rng or np.random.default_rng(0)
    r = linprog(rng.uniform(0, 1, A.shape[1]), A_eq=A, b_eq=b,
                bounds=[(0, None)] * A.shape[1], method="highs")
    return r.x if r.status == 0 else None


BUILDERS = {"minnorm": w_minnorm, "projgauss": w_projgauss, "spectral": w_spectral,
            "spectralg": w_spectralg, "maxent": w_maxent, "lp": w_lp, "lprand": w_lprand}


# ------------------------------------------------- spatial (MOL) operators
def spatial_ops(x, K, p=2):
    """Global D1, D2 (N x N) from min-norm p-th order consistent rows on K nearest nodes.

    Rows: sum_j w_j dx_j^q / q! = delta_{q,m} for q = 0..p, m = 1 (D1) or 2 (D2).
    Wall rows are zero (u = 0 held there).
    """
    N = len(x)
    D1, D2 = np.zeros((N, N)), np.zeros((N, N))
    for i in range(1, N - 1):
        nb = neighbours(x, i, K)
        dxi = x[nb] - x[i]
        hloc = np.abs(dxi).max()
        A = np.vstack([(dxi / hloc) ** q / np.math.factorial(q) for q in range(p + 1)])
        e1 = np.zeros(p + 1); e1[1] = 1.0 / hloc
        e2 = np.zeros(p + 1); e2[2] = 1.0 / hloc**2
        D1[i, nb] = np.linalg.lstsq(A, e1, rcond=None)[0]
        D2[i, nb] = np.linalg.lstsq(A, e2, rcond=None)[0]
    return D1, D2


# ------------------------------------------------------ operator assembly
class Operator:
    """One-step linear map. levels=1: u^{n+1} = M u^n. levels=2: u^{n+1} = M1 u^n + M2 u^{n-1}."""

    def __init__(self, x, prob, dt, arm, K, levels=1, Kmax_widen=None, seed=0):
        self.x, self.prob, self.dt, self.arm, self.K, self.levels = x, prob, dt, arm, K, levels
        N = len(x)
        h = 1.0 / (N - 1)
        rng = np.random.default_rng(seed)
        self.fallback = np.zeros(N, bool)
        self.K_used = np.full(N, K)
        self.min_w = np.zeros(N)
        self.l1 = np.ones(N)
        self.M1, self.M2 = np.zeros((N, N)), np.zeros((N, N))

        if arm in ("molfe", "molrk3"):
            assert levels == 1
            D1, D2 = spatial_ops(x, K)
            L = -prob.c * D1 + prob.alpha * D2
            I = np.eye(N)
            if arm == "molfe":
                M = I + dt * L
            else:                                             # classical 3-stage RK (linear => Taylor poly)
                M = I + dt * L + (dt * L) @ (dt * L) / 2 + (dt * L) @ (dt * L) @ (dt * L) / 6
            M[0, :] = 0; M[-1, :] = 0
            self.M1 = M
            self.min_w = M.min(1)
            self.l1 = np.abs(M).sum(1)
            self.l1[[0, -1]] = 1.0
            return

        for i in range(1, N - 1):
            w, nb, lev = None, None, None
            Ks = [K] if arm != "maxentw" else list(range(K, (Kmax_widen or K + 6) + 1, 2))
            for Kk in Ks:
                nb, dxi, dti, lev = self._stencil(i, Kk)
                A, b = _rows(dxi, dti, prob)
                builder = BUILDERS["maxent" if arm == "maxentw" else arm]
                w = builder(A, b, dxi=dxi, dti=dti, prob=prob, h=h, rng=rng)
                if w is not None:
                    self.K_used[i] = Kk
                    break
            if w is None:                                     # fall back to min-norm at K
                nb, dxi, dti, lev = self._stencil(i, K)
                A, b = _rows(dxi, dti, prob)
                w = st.solve_minnorm(A, b, guard=False)
                self.fallback[i] = True
            self.min_w[i] = w.min()
            self.l1[i] = np.abs(w).sum()
            for wj, j, l in zip(w, nb, lev):
                (self.M1 if l == 1 else self.M2)[i, j] += wj

    def _stencil(self, i, Kk):
        x, dt = self.x, self.dt
        nb = neighbours(x, i, Kk)
        dxi = x[nb] - x[i]
        dti = -dt * np.ones(Kk)
        lev = np.ones(Kk, int)
        if self.levels == 2:
            nb = np.concatenate([nb, nb])
            dxi = np.concatenate([dxi, dxi])
            dti = np.concatenate([dti, -2 * dt * np.ones(Kk)])
            lev = np.concatenate([lev, 2 * np.ones(Kk, int)])
        return nb, dxi, dti, lev

    # ---- linear-algebra diagnostics
    def companion(self):
        if self.levels == 1:
            return self.M1
        N = len(self.x)
        return np.block([[self.M1, self.M2], [np.eye(N), np.zeros((N, N))]])

    def spectral_radius(self):
        return float(np.abs(np.linalg.eigvals(self.companion())).max())

    def power_norm(self, n):
        """||C^n||_inf, by repeated squaring."""
        C = self.companion()
        R, P, e = np.eye(len(C)), C.copy(), n
        while e:
            if e & 1:
                R = R @ P
            P = P @ P
            e >>= 1
        return float(np.abs(R).sum(1).max())

    # ---- marching
    def march(self, u0, u1, nt, ref_fn=None, sample_every=1):
        """March nt steps. Returns dict with max|u| history, error history (if ref_fn), final u.

        ref_fn(step) -> reference u at that step (only called on sampled steps).
        """
        u_prev, u = u0.copy(), (u0.copy() if self.levels == 1 else u1.copy())
        start = 0 if self.levels == 1 else 1
        maxu, err, steps = [], [], []
        for n in range(start, nt):
            un = self.M1 @ u + (self.M2 @ u_prev if self.levels == 2 else 0.0)
            u_prev, u = u, un
            if (n + 1) % sample_every == 0 or n + 1 == nt:
                m = float(np.abs(u).max())
                maxu.append(m); steps.append(n + 1)
                if ref_fn is not None:
                    err.append(float(np.abs(u - ref_fn(n + 1)).max()) if np.isfinite(m) else np.inf)
                if not np.isfinite(m) or m > 1e6:
                    break
        return dict(steps=np.array(steps), maxu=np.array(maxu), err=np.array(err), u=u)


# --------------------------------------------------------------- reference
def crank_nicolson(prob, times, nx_ref=8001, dt_max=1e-4):
    """CN on a uniform fine grid, landing exactly on each requested time. Returns (x_ref, U)."""
    xr = np.linspace(0, 1, nx_ref)
    h = xr[1] - xr[0]
    u = prob.u_true(xr, 0.0)
    a, c = prob.alpha, prob.c
    # L u = alpha u_xx - c u_x (central), tridiagonal
    lo = a / h**2 + c / (2 * h)
    di = -2 * a / h**2
    up = a / h**2 - c / (2 * h)
    out, t = [u.copy()], 0.0
    for T in times:
        if T <= t + 1e-15:
            out.append(u.copy()); continue
        m = int(np.ceil((T - t) / dt_max))
        dt = (T - t) / m
        ab = np.zeros((3, nx_ref))
        ab[0, 2:] = -0.5 * dt * up
        ab[1, :] = 1 - 0.5 * dt * di
        ab[2, :-2] = -0.5 * dt * lo
        ab[1, 0] = ab[1, -1] = 1.0
        ab[0, 1] = 0.0; ab[2, -2] = 0.0
        for _ in range(m):
            Lu = np.zeros_like(u)
            Lu[1:-1] = lo * u[:-2] + di * u[1:-1] + up * u[2:]
            rhs = u + 0.5 * dt * Lu
            rhs[0] = rhs[-1] = 0.0
            u = solve_banded((1, 1), ab, rhs)
        t = T
        out.append(u.copy())
    return xr, np.array(out[1:])


class Reference:
    """u_ref(x, t_k) at requested times, from the exact series where it is trustworthy
    (alpha >= 0.05: self-consistency 1e-14 measured) and from CN otherwise. `floor`
    is the measured discrepancy between the two resolutions of the reference."""

    def __init__(self, prob, times, x_eval, force_cn=False, nx_ref=8001, dt_max=1e-4):
        self.times = np.asarray(times)
        self.exact = prob.alpha >= 0.05 and not force_cn
        if self.exact:
            self.U = np.array([prob.u_true(x_eval, t) for t in self.times])
            U2 = np.array([prob.u_true(x_eval, t, n_series=800, nq=80001) for t in self.times])
            self.floor = float(np.abs(self.U - U2).max())
        else:
            xr, Ur = crank_nicolson(prob, self.times, nx_ref, dt_max)
            xr2, Ur2 = crank_nicolson(prob, self.times, 2 * nx_ref - 1, dt_max / 2)
            self.U = np.array([CubicSpline(xr, u)(x_eval) for u in Ur])
            U2 = np.array([CubicSpline(xr2, u)(x_eval) for u in Ur2])
            self.floor = float(np.abs(self.U - U2).max())
        self.U[:, 0] = 0.0; self.U[:, -1] = 0.0

    def at(self, k):
        return self.U[k]


# ------------------------------------------------------------ uniform refs
def ftcs_weights(prob, dx, dt):
    r, nu = prob.alpha * dt / dx**2, prob.c * dt / dx
    return np.array([r + nu / 2, 1 - 2 * r, r - nu / 2])          # (left, centre, right)


def lw_weights(prob, dx, dt):
    r, nu = prob.alpha * dt / dx**2, prob.c * dt / dx
    return np.array([r + nu / 2 + nu**2 / 2, 1 - 2 * r - nu**2, r - nu / 2 + nu**2 / 2])
