"""
moments.py -- the propagator-moment (Kramers-Moyal) reformulation.

Setup and sign conventions (these matter and are easy to get wrong)
-------------------------------------------------------------------
For u_t = L u with L = alpha d_x^2 - c d_x  (i.e. u_t + c u_x = alpha u_xx),

    u(x, t+tau) = INT K(y, tau) u(x - y, t) dy

so K is the forward kernel: a Gaussian with MEAN c*tau and VARIANCE 2*alpha*tau.
(Check: alpha=0 gives u(x,t+tau) = u(x-c tau, t), a right-moving wave.)

A meshfree predictor at target z* = (x*, t*) using neighbours at OFFSETS
(dx_i, -tau) is uhat* = sum_i w_i u(x* + dx_i, t* - tau), so w_i sits at
y = -dx_i.  Define the STENCIL moments in the offset variable

    Mfrak_q(w) := sum_i w_i dx_i^q          (what the code can actually form)

and the KERNEL moments M_q(tau) := INT K(y,tau) y^q dy.  Then

    Mfrak_q = (-1)^q M_q .

THE CENTRAL IDENTITY
--------------------
Taylor-expanding the predictor about (x*, t*-tau) and demanding it equal
exp(tau L) applied at the same point matches powers of d_x, giving

    sum_q Mfrak_q s^q / q!  =  exp( tau * P(s) ),     P(s) = alpha s^2 - c s

i.e. *the moment generating function of the stencil is the exponential of the
symbol of the generator*, with d_x -> s.  Equivalently the CUMULANTS are

    kappa_q = tau * q! * [s^q] P(s),

so for a second-order operator only kappa_1 = -c tau and kappa_2 = 2 alpha tau
are non-zero: the propagator is exactly Gaussian.  This is the Levy-Khintchine
/ semigroup-symbol relation, and it is what Pawula's theorem is about.

Consequences used in this thread:
  * closed triangular ODE hierarchy (below), driven only by the coefficients;
  * c = M_1/tau and alpha = (M_2 - M_1^2)/(2 tau), EXACTLY, at every tau;
  * for a non-polynomial symbol (fractional Laplacian) the expansion cannot
    terminate and the high moments diverge -- the nonlocality detector.

Relation to the Taylor/annihilator rows of RESULTS.md
-----------------------------------------------------
Row 2  sum w (dx - c dt) = 0  with dt = -tau  reads  Mfrak_1 = -c tau  -- the
EXACT first moment condition.  Row 3  sum w (dx^2/2 + alpha dt) = 0 reads
Mfrak_2 = 2 alpha tau, whereas the exact second moment is 2 alpha tau + c^2
tau^2.  The Taylor rows are therefore the LOW-ORDER TRUNCATION of the moment
conditions: they drop c^2 tau^2, which is O(h^4) under parabolic scaling
tau ~ h^2/alpha.  Nothing in Task 1 is wasted -- it is the same expansion
re-summed.
"""
import numpy as np
from math import factorial, comb
from scipy.integrate import solve_ivp
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# exact kernel moments from the closed form (advection-diffusion only)
# ---------------------------------------------------------------------------

def gauss_moments(c, alpha, tau, Q):
    """Raw moments of N(mean = c tau, var = 2 alpha tau), q = 0..Q."""
    mu, var = c * tau, 2 * alpha * tau
    M = np.zeros(Q + 1)
    M[0] = 1.0
    if Q >= 1:
        M[1] = mu
    for q in range(2, Q + 1):                      # M_q = mu M_{q-1} + (q-1) var M_{q-2}
        M[q] = mu * M[q - 1] + (q - 1) * var * M[q - 2]
    return M


# ---------------------------------------------------------------------------
# the hierarchy: dM_q/dtau = alpha q(q-1) M_{q-2} + c q M_{q-1},  M_q(0)=delta_q0
# ---------------------------------------------------------------------------

def hierarchy_rhs(coeffs, Q):
    """coeffs: dict {order: value} for the generator L = sum_n coeffs[n] d_x^n.
    For u_t + c u_x = alpha u_xx  ->  {1: -c, 2: alpha}.
    Returns f(tau, M) for dM/dtau, with M_q = INT K y^q.
    Derivation: dM_q/dtau = INT y^q L^dagger K.  For L = sum a_n d_x^n acting on
    the kernel as K_tau = sum a_n (-1)^n d_y^n K ... but it is cleaner to use the
    symbol: d/dtau of the MGF is P(s) * MGF, so with m_q = M_q/q!,
        dm_q/dtau = sum_n p_n m_{q-n},   p_n = [s^n] P(s).
    In raw moments: dM_q/dtau = sum_n p_n * q!/(q-n)! * M_{q-n}."""
    p = {n: v for n, v in coeffs.items()}

    def f(tau, M):
        d = np.zeros_like(M)
        for q in range(len(M)):
            s = 0.0
            for n, pn in p.items():
                if n <= q and n >= 1:
                    s += pn * (factorial(q) / factorial(q - n)) * M[q - n]
            d[q] = s
        return d
    return f


def solve_hierarchy(coeffs, tau, Q, rtol=1e-12, atol=1e-14):
    M0 = np.zeros(Q + 1); M0[0] = 1.0
    sol = solve_ivp(hierarchy_rhs(coeffs, Q), (0.0, tau), M0,
                    rtol=rtol, atol=atol, dense_output=True, method="DOP853")
    return sol.y[:, -1]


def moments_from_symbol(coeffs, tau, Q):
    """Closed form: Mfrak_q/q! = [s^q] exp(tau P(s)) by series exponentiation.
    Returns the OFFSET moments Mfrak_q (note the sign convention)."""
    P = np.zeros(Q + 1)
    for n, v in coeffs.items():
        if n <= Q:
            P[n] = v
    E = np.zeros(Q + 1); E[0] = 1.0        # exp of a series with zero constant term
    term = np.zeros(Q + 1); term[0] = 1.0
    for k in range(1, Q + 1):
        new = np.zeros(Q + 1)
        for i in range(Q + 1):
            if term[i] == 0:
                continue
            for j in range(1, Q + 1 - i):
                new[i + j] += term[i] * tau * P[j]
        term = new / k
        E += term
    return E * np.array([factorial(q) for q in range(Q + 1)])


# ---------------------------------------------------------------------------
# coefficient read-off (the Kramers-Moyal estimator)
# ---------------------------------------------------------------------------

def coeffs_from_moments(Mfrak, tau):
    """c and alpha from the OFFSET moments. Mfrak_1 = -c tau, and the second
    CUMULANT is kappa_2 = Mfrak_2 - Mfrak_1^2 = 2 alpha tau."""
    c = -Mfrak[1] / tau
    alpha = (Mfrak[2] - Mfrak[1] ** 2) / (2 * tau)
    return c, alpha


def cumulants(Mfrak, Q):
    """Raw moments -> cumulants, kappa_q = tau q! [s^q] P(s)."""
    k = np.zeros(Q + 1)
    for q in range(1, Q + 1):
        k[q] = Mfrak[q] - sum(comb(q - 1, j - 1) * k[j] * Mfrak[q - j]
                              for j in range(1, q))
    return k


# ---------------------------------------------------------------------------
# moment-matched stencils (the integrator rows)
# ---------------------------------------------------------------------------

def moment_rows(offs, Mfrak_target, p, scale=None):
    """Rows sum_i w_i dx_i^q = Mfrak_q for q = 0..p.

    NON-DIMENSIONALISED: offsets are divided by `scale` (default: the kernel
    width implied by the second cumulant) before raising to the q-th power.
    Without this the rows span many orders of magnitude and the LP declares
    perfectly feasible problems infeasible -- the same lesson as the jet design
    matrix in jetlib.design_matrix."""
    offs = np.asarray(offs, float)
    M = np.asarray(Mfrak_target[:p + 1], float)
    if scale is None:
        # scale by the STENCIL RADIUS, not the kernel width: this keeps every
        # entry of every row inside [-1,1].  Scaling by sigma instead leaves the
        # q-th row spanning (R/sigma)^q, which at R=6 sigma, q=8 is 2e6 and makes
        # the solver report feasible problems as infeasible.
        scale = max(np.max(np.abs(offs)), 1e-12)
    A = np.array([(offs / scale) ** q for q in range(p + 1)])
    b = np.array([M[q] / scale ** q for q in range(p + 1)])
    return A, b


def w_moment_positive(offs, Mfrak_target, p, scale=None):
    A, b = moment_rows(offs, Mfrak_target, p, scale)
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b,
                bounds=[(0, None)] * A.shape[1], method="highs")
    return r.x if r.status == 0 else None


def w_moment_minnorm(offs, Mfrak_target, p, condmax=1e12, scale=None):
    A, b = moment_rows(offs, Mfrak_target, p, scale)
    if np.linalg.cond(A) > condmax:
        return None
    return np.linalg.lstsq(A, b, rcond=None)[0]


def hankel_psd_margin(Mfrak, n):
    """Smallest eigenvalue of the (n+1)x(n+1) Hankel matrix H_ij = Mfrak_{i+j}.
    A sequence is the moment sequence of a NONNEGATIVE measure only if every
    such Hankel matrix is PSD (Hamburger).  Negative margin => no nonnegative
    stencil can reproduce these moments, at any geometry."""
    H = np.array([[Mfrak[i + j] for j in range(n + 1)] for i in range(n + 1)])
    return float(np.linalg.eigvalsh(H)[0]), H


# ---------------------------------------------------------------------------
# periodic test fields: translation-invariant propagator, exactly known symbol
# ---------------------------------------------------------------------------

class PeriodicField:
    """u(x,t) = sum_k a_k cos(2 pi k x + phi_k) exp(-lam_k t) on the unit torus,
    with lam_k = alpha (2 pi k)^(2s) + i-free.  Optional advection shifts phase.

    The propagator over a lag tau is translation invariant with symbol
        ghat(k) = exp(-lam_k tau) * exp(-i 2 pi k c tau),
    so its moments/cumulants are exactly defined and, for s < 1, the symbol is
    NOT a polynomial in k: no finite-order local PDE exists.
    """

    def __init__(self, alpha=0.1, s=1.0, c=0.0, K=40, decay=1.0, seed=0):
        self.alpha, self.s, self.c, self.K = alpha, s, c, K
        rng = np.random.default_rng(seed)
        self.k = np.arange(1, K + 1)
        self.a = self.k ** (-float(decay))
        self.phi = rng.uniform(0, 2 * np.pi, K)
        self.lam = alpha * (2 * np.pi * self.k) ** (2 * s)

    def u(self, x, t):
        x = np.atleast_1d(np.asarray(x, float))
        arg = 2 * np.pi * np.outer(x - self.c * t, self.k) + self.phi
        v = (np.cos(arg) * (self.a * np.exp(-self.lam * t))).sum(-1)
        return v if x.size > 1 else float(v[0])

    def symbol_moments(self, tau, Q, L=1.0, n=4096):
        """Exact moments of the periodic propagator over one period, by
        assembling the kernel from its symbol."""
        z = (np.arange(n) / n - 0.5) * L
        G = np.ones(n) / n
        kk = self.k
        for j, kj in enumerate(kk):
            G += (2.0 / n) * np.cos(2 * np.pi * kj * (z - self.c * tau)) \
                 * np.exp(-self.lam[j] * tau)
        return np.array([np.sum(G * z ** q) for q in range(Q + 1)]), z, G


def w_moment_positive_smooth(offs, Mfrak_target, p, scale=None, big=1e6):
    """Minimum-norm NONNEGATIVE weights matching moments 0..p.

    The LP in w_moment_positive returns a vertex of the feasible polytope, which
    has at most p+1 non-zeros -- a Gauss-quadrature-like rule that reproduces the
    moments but approximates a smooth kernel badly.  For an integrator we want the
    SMOOTHEST non-negative representer, so solve
        min ||w||_2  s.t.  A w = b,  w >= 0
    as a bound-constrained least-squares problem with the equalities weighted up.
    """
    from scipy.optimize import lsq_linear
    A, b = moment_rows(offs, Mfrak_target, p, scale)
    n = A.shape[1]
    M = np.vstack([big * A, np.eye(n)])
    r = np.concatenate([big * b, np.zeros(n)])
    res = lsq_linear(M, r, bounds=(0, np.inf), tol=1e-15, max_iter=5000)
    w = res.x
    # per-row RELATIVE residual: the high-q targets are tiny after scaling, so an
    # absolute tolerance would silently accept garbage there.
    rel = np.abs(A @ w - b) / np.maximum(np.abs(b), 1e-13)
    if np.max(rel) > 1e-6:
        return None
    return w
