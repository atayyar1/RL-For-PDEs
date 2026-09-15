"""
jetlib.py -- core library for the "moment / annihilator" route to PDE discovery.

The object of study
-------------------
A meshfree local predictor at a target z* = (x*, t*) is

    uhat* = sum_i w_i u_i     over neighbours at offsets (dx_i, dt_i)

Taylor expanding each u_i about z* and writing the (scaled) jet coordinates

    d_q = (d_x^a d_t^b u)(z*) / (a! b!)          q = (a, b)

gives, provided sum_i w_i = 1,

    e(w, G) = uhat* - u* = sum_{q != (0,0)} M_q(w, G) d_q(z*)
    M_q(w, G) = sum_i w_i dx_i^a dt_i^b                      ("moment vector")

which is LINEAR in the jet.  Step 1 regresses observed e on M to recover d.
Step 2 looks for linear relations among the recovered jets across targets;
those relations are the PDE, and their orthogonal complement in moment space
is exactly the set of solver rows used by a PDE-constrained meshfree stencil.

Bookkeeping, worked out and verified in tests/test_bookkeeping:
    u_t + c u_x = alpha u_xx
      => d_01 = 2 alpha d_20 - c d_10
      => rho = (c, 1, -2 alpha) on (d_10, d_01, d_20) annihilates the jet subspace
      => zero-error weights need M in span{rho}, i.e. M perp rho^perp,
         and a basis of rho^perp gives the two familiar solver rows
             sum w (dx - c dt) = 0      and      sum w (dx^2/2 + alpha dt) = 0.

Grading
-------
Because dt ~ dx^2/alpha on the solution manifold, the natural truncation is by
PARABOLIC degree a + 2b, not total degree a + b.  Both are supported.

numpy 1.26 (np.trapz), scipy 1.15.
"""
import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# monomial bookkeeping
# ----------------------------------------------------------------------------

def monomials(K, grading="parabolic", include_const=False):
    """Monomial exponents (a,b) with degree <= K, sorted by degree then by b.

    grading='total'     -> deg = a + b
    grading='parabolic' -> deg = a + 2b   (correct scaling for parabolic PDEs)
    """
    deg = (lambda a, b: a + b) if grading == "total" else (lambda a, b: a + 2 * b)
    qs = [(a, b) for a in range(K + 1) for b in range(K + 1) if deg(a, b) <= K]
    if not include_const:
        qs = [q for q in qs if q != (0, 0)]
    qs.sort(key=lambda q: (deg(*q), q[1], q[0]))
    return qs


def design_matrix(dxi, dti, qs, hx=1.0, ht=1.0):
    """Phi[i, j] = (dx_i/hx)^a_j (dt_i/ht)^b_j  -- non-dimensionalised monomials."""
    X = np.asarray(dxi, float) / hx
    T = np.asarray(dti, float) / ht
    return np.column_stack([X ** a * T ** b for (a, b) in qs])


def moments(wsamp, Phi):
    """M[s, j] = sum_i w[s,i] Phi[i,j].  wsamp (S,n) or (n,)."""
    return np.atleast_2d(wsamp) @ Phi


def jet_scaling(qs, hx, ht):
    """Multiply a physical jet d_q by this to get the non-dimensional jet."""
    return np.array([hx ** a * ht ** b for (a, b) in qs])


def true_jet(sol, x0, t0, qs):
    """Physical jet d_q = (d_x^a d_t^b u)(z*) / (a! b!) from an exact solution."""
    from math import factorial
    return np.array([sol.deriv(x0, t0, a, b) / (factorial(a) * factorial(b))
                     for (a, b) in qs])


# ----------------------------------------------------------------------------
# exact solutions (each carries an analytic mixed-derivative method)
# ----------------------------------------------------------------------------

class AdvDiff:
    """u_t + c u_x = alpha u_xx on [0,1], u(0)=u(1)=0.

    Written as a single complex Fourier sum so that ALL mixed derivatives are
    exact and cheap:
        u = sum_n b_n exp(-lam_n t) Im( exp(k_n x) ),  k_n = beta + i n pi,
        beta = c/(2 alpha),  lam_n = alpha (n pi)^2 + alpha beta^2.
    c = 0 reduces to pure diffusion (beta = 0, b_n = [1, 0.5, 0, ...]).
    """

    name = "advection-diffusion"

    def __init__(self, alpha=0.1, c=1.0, N=400, nq=40001):
        self.alpha, self.c, self.N = alpha, c, N
        self.beta = beta = c / (2 * alpha)
        xq = np.linspace(0, 1, nq)
        v0 = (np.sin(np.pi * xq) + 0.5 * np.sin(2 * np.pi * xq)) * np.exp(-beta * xq)
        self.n = np.arange(1, N + 1)
        self.bn = np.array([2 * np.trapz(v0 * np.sin(nn * np.pi * xq), xq) for nn in self.n])
        self.k = beta + 1j * self.n * np.pi
        self.lam = alpha * (self.n * np.pi) ** 2 + alpha * beta ** 2

    def deriv(self, x, t, a=0, b=0):
        """d_x^a d_t^b u, exact."""
        x = np.atleast_1d(np.asarray(x, float))
        t = float(t)
        ex = np.where(-self.lam * t < 700, -self.lam * t, 700.0)   # guard t<0
        amp = self.bn * (-self.lam) ** b * np.exp(ex)                # (N,)
        keep = np.isfinite(amp) & (np.abs(amp) > 1e-300)
        if not keep.any():
            return np.zeros_like(x) if x.size > 1 else 0.0
        amp, k = amp[keep], self.k[keep]
        v = (np.exp(np.outer(x, k)) * (amp * k ** a)).sum(-1).imag
        return v if x.size > 1 else float(v[0])

    def u(self, x, t):
        return self.deriv(x, t, 0, 0)

    # true PDE written as u_t = sum_j coef_j * term_j over the SPATIAL library
    def truth(self):
        return {"u_x": -self.c, "u_xx": self.alpha}


class BurgersTanh:
    """Exact Taylor-shock solution of viscous Burgers u_t + u u_x = nu u_xx:

        u = s - a tanh(theta),  theta = a (x - s t) / (2 nu)

    All mixed derivatives follow from d_theta^m tanh, computed exactly with a
    polynomial-in-tanh recursion (d/dth T^j = j T^{j-1} - j T^{j+1}).
    """

    name = "viscous Burgers (Taylor shock)"

    def __init__(self, nu=0.05, s=1.0, a=0.8, maxord=8):
        self.nu, self.s, self.a = nu, s, a
        self.k = a / (2 * nu)
        # polys[m] = coefficient array of d_theta^m tanh as a polynomial in T
        polys = [np.array([0.0, 1.0])]                      # tanh = T
        for _ in range(maxord):
            p = polys[-1]
            q = np.zeros(len(p) + 2)
            for j, cj in enumerate(p):
                if cj == 0 or j == 0:
                    continue
                q[j - 1] += j * cj
                q[j + 1] -= j * cj
            polys.append(q)
        self.polys = polys

    def _dtanh(self, m, T):
        return np.polyval(self.polys[m][::-1], T)

    def deriv(self, x, t, a=0, b=0):
        x = np.atleast_1d(np.asarray(x, float))
        th = self.k * (x - self.s * t)
        T = np.tanh(th)
        if (a, b) == (0, 0):
            v = self.s - self.a * T
        else:
            v = -self.a * self.k ** a * (-self.s * self.k) ** b * self._dtanh(a + b, T)
        return v if x.size > 1 else float(v[0])

    def u(self, x, t):
        return self.deriv(x, t, 0, 0)

    def truth(self):
        return {"u*u_x": -1.0, "u_xx": self.nu}


class FracDiff:
    """Spectral fractional diffusion u_t = -alpha (-Delta)^s u on [0,1].

        u = sum_n b_n sin(n pi x) exp(-alpha (n pi)^{2s} t)

    For s != 1 this has NO finite-order local PDE: it is the negative control.
    """

    name = "fractional diffusion (nonlocal)"

    def __init__(self, alpha=0.1, s=0.75, bn=(1.0, 0.5, 0.25), nmax=None):
        self.alpha, self.s = alpha, s
        self.n = np.arange(1, len(bn) + 1) if nmax is None else np.arange(1, nmax + 1)
        b = np.zeros(len(self.n)); b[:len(bn)] = bn
        self.bn = b
        self.lam = alpha * (self.n * np.pi) ** (2 * s)

    def deriv(self, x, t, a=0, b=0):
        x = np.atleast_1d(np.asarray(x, float))
        amp = self.bn * (-self.lam) ** b * np.exp(-self.lam * t)
        kn = self.n * np.pi
        # d_x^a sin(kx) = k^a sin(kx + a pi/2)
        v = (np.sin(np.outer(x, kn) + a * np.pi / 2) * (amp * kn ** a)).sum(-1)
        return v if x.size > 1 else float(v[0])

    def u(self, x, t):
        return self.deriv(x, t, 0, 0)

    def truth(self):
        return {}


# ----------------------------------------------------------------------------
# noise
# ----------------------------------------------------------------------------

def noise_scale(sol, x0, t0, hx, ht, nsamp=400, rng=None):
    """sigma reference = RMS of u over the sampling neighbourhood."""
    rng = np.random.default_rng(0) if rng is None else rng
    xs = x0 + hx * rng.uniform(-1, 1, nsamp)
    us = np.array([sol.u(xx, t0 + ht * rng.uniform(-1, 0)) for xx in xs])
    return float(np.sqrt(np.mean(us ** 2)))


# ----------------------------------------------------------------------------
# Step 1 estimators
# ----------------------------------------------------------------------------

def random_weights(S, n, rng, kind="gauss", tau=1.0):
    """S weight vectors of length n with sum = 1."""
    if kind == "gauss":
        z = rng.normal(0, tau, size=(S, n))
        z -= z.mean(1, keepdims=True)            # project onto sum = 0
        return 1.0 / n + z
    if kind == "dirichlet":
        return rng.dirichlet(np.ones(n), size=S)
    raise ValueError(kind)


def moment_regression(Phi, y, wsamp, rcond=None):
    """Step 1 as specified: regress e(w) on M(w) over random weight draws.

    Phi  (n,Q)  design matrix of scaled monomials at the neighbours
    y    (n,)   u_i - u*   (possibly noisy)
    wsamp(S,n)  weight vectors, rows sum to 1
    returns dhat (Q,) in SCALED jet coordinates
    """
    M = wsamp @ Phi
    e = wsamp @ y
    return np.linalg.lstsq(M, e, rcond=rcond)[0]


def wls_jet(Phi, y, W=None, rcond=None):
    """Weighted local-polynomial fit CONSTRAINED to interpolate u* exactly:
        min (Phi d - y)^T W (Phi d - y)      with  y = u_i - u*
    This is the closed form that moment_regression provably reduces to.
    """
    if W is None:
        return np.linalg.lstsq(Phi, y, rcond=rcond)[0]
    if W.ndim == 1:
        Ws = np.sqrt(W)
        return np.linalg.lstsq(Ws[:, None] * Phi, Ws * y, rcond=rcond)[0]
    A = Phi.T @ W @ Phi
    return np.linalg.lstsq(A, Phi.T @ W @ y, rcond=rcond)[0]


def mls_jet_unconstrained(dxi, dti, ui, qs, hx, ht, kernel=None):
    """Standard moving-least-squares: fit the constant too, no interpolation
    constraint.  Returns the jet for qs (constant dropped)."""
    qs_full = [(0, 0)] + list(qs)
    Phi = design_matrix(dxi, dti, qs_full, hx, ht)
    w = np.ones(len(ui)) if kernel is None else kernel
    Ws = np.sqrt(w)
    d = np.linalg.lstsq(Ws[:, None] * Phi, Ws * np.asarray(ui, float), rcond=None)[0]
    return d[1:], d[0]


def tricube(dxi, dti, hx, ht):
    r = np.sqrt((np.asarray(dxi) / hx) ** 2 + (np.asarray(dti) / ht) ** 2)
    r = np.clip(r, 0, 1)
    return (1 - r ** 3) ** 3


# ----------------------------------------------------------------------------
# finite-difference baseline
# ----------------------------------------------------------------------------

def fd_jet(sol, x0, t0, dx, dt, sigma=0.0, rng=None, backward_t=True):
    """Central FD for u_x, u_xx on a 5-point x-stencil; u_t by a 2nd-order
    backward (or central) formula.  Returns dict of DERIVATIVES (not d_q)."""
    rng = np.random.default_rng(0) if rng is None else rng
    xs = x0 + dx * np.arange(-2, 3)
    u0 = np.array([sol.u(xx, t0) for xx in xs]) + sigma * rng.normal(size=5)
    ux = (u0[0] - 8 * u0[1] + 8 * u0[3] - u0[4]) / (12 * dx)
    uxx = (-u0[0] + 16 * u0[1] - 30 * u0[2] + 16 * u0[3] - u0[4]) / (12 * dx ** 2)
    if backward_t:
        tv = [t0, t0 - dt, t0 - 2 * dt]
        uu = np.array([sol.u(x0, tt) for tt in tv]) + sigma * rng.normal(size=3)
        ut = (3 * uu[0] - 4 * uu[1] + uu[2]) / (2 * dt)
    else:
        uu = np.array([sol.u(x0, t0 - dt), sol.u(x0, t0 + dt)]) + sigma * rng.normal(size=2)
        ut = (uu[1] - uu[0]) / (2 * dt)
    return {"u": u0[2], "u_x": ux, "u_xx": uxx, "u_t": ut}


# ----------------------------------------------------------------------------
# solver rows / stencil weights (the companion project's integrator)
# ----------------------------------------------------------------------------

def build_A(dxi, dti, c, alpha):
    dxi = np.asarray(dxi, float); dti = np.asarray(dti, float)
    h = max(np.max(np.abs(dxi)), np.sqrt(max(alpha, 1e-12) * np.max(np.abs(dti))), 1e-12)
    A = np.array([np.ones(len(dxi)),
                  (dxi - c * dti) / h,
                  (0.5 * dxi ** 2 + alpha * dti) / h ** 2])
    return A, np.array([1.0, 0.0, 0.0])


def w_minnorm(dxi, dti, c, alpha, condmax=1e4):
    A, b = build_A(dxi, dti, c, alpha)
    if np.linalg.cond(A) > condmax:
        return None
    return np.linalg.lstsq(A, b, rcond=None)[0]


def w_positive(dxi, dti, c, alpha):
    """Nonnegative weights with A w = b  =>  ||w||_1 = 1 exactly (stable)."""
    A, b = build_A(dxi, dti, c, alpha)
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b,
                bounds=[(0, None)] * A.shape[1], method="highs")
    return r.x if r.status == 0 else None


# ----------------------------------------------------------------------------
# sparse regression (STLSQ)
# ----------------------------------------------------------------------------

def stlsq(Theta, y, thresh=0.05, n_iter=20, normalize=True):
    """Sequentially thresholded least squares on COLUMN-NORMALISED data."""
    Theta = np.asarray(Theta, float); y = np.asarray(y, float)
    scale = np.linalg.norm(Theta, axis=0) if normalize else np.ones(Theta.shape[1])
    scale[scale == 0] = 1.0
    Tn = Theta / scale
    xi = np.linalg.lstsq(Tn, y, rcond=None)[0]
    active = np.ones(Theta.shape[1], bool)
    for _ in range(n_iter):
        small = np.abs(xi) < thresh
        if not small.any() or (~small).sum() == 0:
            break
        new_active = active.copy()
        new_active[small] = False
        if new_active.sum() == 0:
            break
        xi = np.zeros(Theta.shape[1])
        xi[new_active] = np.linalg.lstsq(Tn[:, new_active], y, rcond=None)[0]
        if (new_active == active).all():
            break
        active = new_active
    return xi / scale, active


# ----------------------------------------------------------------------------
# truncated 2-variable Taylor (jet) arithmetic -- used for Cole-Hopf Burgers
# ----------------------------------------------------------------------------

def jet_mul(A, B, D):
    C = np.zeros_like(A)
    for a in range(D + 1):
        for b in range(D + 1 - a):
            s = 0.0
            for i in range(a + 1):
                for j in range(b + 1):
                    s += A[i, j] * B[a - i, b - j]
            C[a, b] = s
    return C


def jet_div(N, Dn, D):
    """Truncated series division N / Dn (Dn[0,0] != 0)."""
    Q = np.zeros_like(N)
    for a in range(D + 1):
        for b in range(D + 1 - a):
            s = N[a, b]
            for i in range(a + 1):
                for j in range(b + 1):
                    if (i, j) == (a, b):
                        continue
                    s -= Q[i, j] * Dn[a - i, b - j]
            Q[a, b] = s / Dn[0, 0]
    return Q


class BurgersColeHopf:
    """u_t + u u_x = nu u_xx via Cole-Hopf, u = -2 nu (log phi)_x, with

        phi = 1 + sum_n a_n cos(n pi x) exp(-nu (n pi)^2 t)   (solves the heat eqn)

    Unlike the Taylor shock this is NOT a travelling wave, so the Burgers
    relation is actually identifiable from it.  Mixed derivatives are obtained
    exactly by truncated Taylor arithmetic on phi.
    """

    name = "viscous Burgers (Cole-Hopf, multi-mode)"

    def __init__(self, nu=0.05, an=(0.5, 0.25), D=8):
        self.nu, self.D = nu, D
        self.an = np.asarray(an, float)
        self.n = np.arange(1, len(self.an) + 1)
        self.lam = nu * (self.n * np.pi) ** 2
        assert 1 - np.abs(self.an).sum() > 0, "phi must stay positive"

    def _phi_jet(self, x, t, D):
        from math import factorial
        C = np.zeros((D + 1, D + 1))
        kn = self.n * np.pi
        for a in range(D + 1):
            for b in range(D + 1 - a):
                v = (self.an * kn ** a * np.cos(kn * x + a * np.pi / 2)
                     * (-self.lam) ** b * np.exp(-self.lam * t)).sum()
                C[a, b] = v / (factorial(a) * factorial(b))
        C[0, 0] += 1.0
        return C

    def _u_jet(self, x, t, D):
        P = self._phi_jet(x, t, D + 1)
        Px = np.zeros_like(P)
        Px[:-1, :] = P[1:, :] * np.arange(1, P.shape[0])[:, None]
        return -2 * self.nu * jet_div(Px[:D + 1, :D + 1], P[:D + 1, :D + 1], D)

    def deriv(self, x, t, a=0, b=0):
        from math import factorial
        x = np.atleast_1d(np.asarray(x, float))
        out = np.array([self._u_jet(float(xx), float(t), max(a + b, 1))[a, b]
                        * factorial(a) * factorial(b) for xx in x])
        return out if x.size > 1 else float(out[0])

    def u(self, x, t):
        x = np.atleast_1d(np.asarray(x, float))
        kn = self.n * np.pi
        E = self.an * np.exp(-self.lam * t)
        phi = 1 + (np.cos(np.outer(x, kn)) * E).sum(-1)
        phix = -(np.sin(np.outer(x, kn)) * (E * kn)).sum(-1)
        v = -2 * self.nu * phix / phi
        return v if x.size > 1 else float(v[0])

    def truth(self):
        return {"u*u_x": -1.0, "u_xx": self.nu}
