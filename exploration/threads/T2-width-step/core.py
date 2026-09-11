"""
T2 -- width vs time-step frontier for wide POSITIVE explicit stencils.

PDE:   u_t + c u_x = alpha u_xx    on [0,1], u(0)=u(1)=0.

A "wide positive stencil of half-width m reaching back k time steps" is a set of
weights w_{-m..m} >= 0 with

    u(x, t)  ~=  sum_j w_j u(x + j*dx, t - k*dt).

Consistency rows used by the team-lead spec (build_A) are, after substituting
u_t = -c u_x + alpha u_xx into the Taylor expansion and keeping
O(1), O(dx), O(dt), O(dx^2):

    (R0)  sum_j w_j                      = 1
    (R1)  sum_j w_j (dx_j - c dt_j)      = 0
    (R2)  sum_j w_j (dx_j^2/2 + alpha dt_j) = 0

with dx_j = j*dx and dt_j = -k*dt for every j.  In moment language, with
    nu  = alpha*dt/dx^2      (diffusion number)
    Co  = c*dt/dx            (Courant number)
these read
    M0 = 1,   M1 = -k*Co,   M2 = 2*k*nu,        M_p := sum_j w_j j^p .

KEY STRUCTURAL FACT (proved in RESULTS.md, verified numerically in task1):
w is a probability measure on the integer lattice {-m..m}; the largest possible
second moment of such a measure is m^2 (all mass at +-m), and the smallest
possible second moment given mean mu is the lower convex envelope of j^2.
Hence feasibility of the LP is *exactly*

    psi(|M1|) <= M2 <= m^2 ,        psi = lower convex envelope of j^2 on Z.

which gives the two closed-form bounds
    k <= m^2/(2 nu)        = (m dx)^2 / (2 alpha dt)     [diffusion / width]
    k <= 2 nu / Co^2       = 2 alpha / (c^2 dt)          [advection / variance]
(the second up to a <=1/4 lattice correction from psi).
"""
import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------- canonical setup
ALPHA, CVEL = 0.1, 1.0
NX, T_REF = 100, 0.5
X_FD = np.linspace(0, 1, NX)
DX = X_FD[1] - X_FD[0]
DT_CFL = min(0.9 * DX / abs(CVEL), 0.45 * DX ** 2 / ALPHA)
NT = max(int(np.ceil(T_REF / DT_CFL)) + 1, 10)
T_FD = np.linspace(0, T_REF, NT)
DT = T_FD[1] - T_FD[0]

NU_REF = ALPHA * DT / DX ** 2
CO_REF = CVEL * DT / DX


# ---------------------------------------------------------------- LP feasibility
def build_A(dxi, dti, alpha=ALPHA, c=CVEL):
    """Team-lead spec, verbatim (alpha, c injectable)."""
    h = max(np.max(np.abs(dxi)), np.sqrt(alpha * np.max(np.abs(dti))), 1e-12)
    A = np.array([np.ones(len(dxi)),
                  (dxi - c * dti) / h,
                  (0.5 * dxi ** 2 + alpha * dti) / h ** 2])
    return A, np.array([1.0, 0.0, 0.0])


def w_positive(dxi, dti, alpha=ALPHA, c=CVEL):
    """Return (feasible, w).  Solve for exists w >= 0 with A w = b."""
    A, b = build_A(np.asarray(dxi, float), np.asarray(dti, float), alpha, c)
    res = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b,
                  bounds=(0, None), method="highs")
    if res.status == 0:
        return True, res.x
    return False, None


def feasible_mk(m, k, nu=NU_REF, co=CO_REF, dx=DX, dt=DT, alpha=ALPHA, c=CVEL,
                use_lp=True):
    """Symmetric half-width m, all neighbours at time offset -k*dt."""
    j = np.arange(-m, m + 1)
    if use_lp:
        ok, _ = w_positive(j * dx, -k * dt * np.ones(2 * m + 1), alpha, c)
        return ok
    return feasible_analytic(m, k, nu, co)


# ---------------------------------------------------------------- analytic frontier
def psi(mu):
    """Lower convex envelope of j^2 on the integers, evaluated at mu.

    = minimum second moment of a probability measure on Z with mean mu.
    """
    mu = np.abs(np.asarray(mu, float))
    n = np.floor(mu)
    f = mu - n
    return n ** 2 + f * (2 * n + 1)


def feasible_analytic(m, k, nu=NU_REF, co=CO_REF):
    """Exact feasibility test of the 3-row system with w >= 0 on {-m..m}."""
    M1 = -k * co
    M2 = 2.0 * k * nu
    if abs(M1) > m:
        return False
    return (M2 <= m ** 2 + 1e-12) and (M2 >= psi(M1) - 1e-12)


def kmax_analytic(m, nu=NU_REF, co=CO_REF, kcap=100000):
    """Largest k >= 1 with the 3-row system feasible (scan; set may be an interval)."""
    best = 0
    # diffusion bound is a hard ceiling
    kdiff = int(np.floor(m ** 2 / (2 * nu))) if nu > 0 else kcap
    for k in range(1, min(kdiff, kcap) + 1):
        if feasible_analytic(m, k, nu, co):
            best = k
    return best


def kmax_closed_form(m, nu=NU_REF, co=CO_REF):
    """Closed form ignoring the <=1/4 lattice correction in psi."""
    kdiff = m ** 2 / (2 * nu)
    kadv = np.inf if co == 0 else 2 * nu / co ** 2
    return min(kdiff, kadv)


# ---------------------------------------------------------------- exact solution
class Exact:
    """u_t + c u_x = alpha u_xx, u(0)=u(1)=0, via v = u exp(-beta x), beta=c/2alpha."""

    def __init__(self, f, alpha=ALPHA, c=CVEL, nmodes=400, nq=40001):
        self.alpha, self.c = alpha, c
        self.beta = c / (2 * alpha)
        xq = np.linspace(0, 1, nq)
        v0 = np.asarray(f(xq), float) * np.exp(-self.beta * xq)
        self.n = np.arange(1, nmodes + 1)
        self.bn = np.array([2 * np.trapz(v0 * np.sin(nn * np.pi * xq), xq)
                            for nn in self.n])

    def __call__(self, x, t):
        x = np.atleast_1d(np.asarray(x, float))
        v = (np.sin(np.pi * np.outer(x, self.n))
             * (self.bn * np.exp(-self.alpha * (self.n * np.pi) ** 2 * t))).sum(-1)
        return np.exp(self.beta * x - self.alpha * self.beta ** 2 * t) * v


def ic_sine(x):
    return np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)


def ic_gauss(x, x0=0.35, s=0.03):
    return np.exp(-((x - x0) ** 2) / (2 * s ** 2))


def ic_square(x, a=0.3, b=0.5):
    return ((x >= a) & (x <= b)).astype(float)


# reference exact solution for the canonical problem (team-lead spec)
_EX = None
def u_true(x, t):
    global _EX
    if _EX is None:
        _EX = Exact(ic_sine, ALPHA, CVEL, nmodes=400)
    return _EX(x, t)


# ---------------------------------------------------------------- kernel designs
def kernel_two_point(m):
    """Frontier weights: the UNIQUE measure on {-m..m} with M2 = m^2 (c=0)."""
    w = np.zeros(2 * m + 1)
    w[0] = w[-1] = 0.5
    return w


def discrete_gauss(m, mu, var, tol=0.0):
    """Sampled, truncated, renormalised Gaussian on {-m..m}."""
    j = np.arange(-m, m + 1)
    if var <= 0:
        w = np.zeros(2 * m + 1)
        w[np.argmin(np.abs(j - mu))] = 1.0
        return w
    lg = -(j - mu) ** 2 / (2 * var)
    w = np.exp(lg - lg.max())
    w[w < tol] = 0.0
    return w / w.sum()


def kernel_moment_matched(m, mu_t, m2_t, tol=1e-14, itmax=60):
    """Positive weights on {-m..m} with EXACT first moment mu_t and second moment m2_t.

    Two-parameter Newton/secant on (mu, var) of a truncated sampled Gaussian.
    Positivity is automatic.  Returns None if the target is infeasible.
    """
    if m2_t > m ** 2 or m2_t < psi(mu_t):
        return None
    j = np.arange(-m, m + 1)
    mu, var = mu_t, max(m2_t - mu_t ** 2, 1e-9)
    for _ in range(itmax):
        w = discrete_gauss(m, mu, var)
        f1 = (w * j).sum() - mu_t
        f2 = (w * j ** 2).sum() - m2_t
        if abs(f1) < tol and abs(f2) < tol * max(1.0, abs(m2_t)):
            return w
        # numerical Jacobian
        h1, h2 = 1e-6 * max(1.0, abs(mu)), 1e-6 * max(1.0, var)
        wa = discrete_gauss(m, mu + h1, var)
        wb = discrete_gauss(m, mu, var + h2)
        J = np.array([[((wa * j).sum() - (w * j).sum()) / h1,
                       ((wb * j).sum() - (w * j).sum()) / h2],
                      [((wa * j ** 2).sum() - (w * j ** 2).sum()) / h1,
                       ((wb * j ** 2).sum() - (w * j ** 2).sum()) / h2]])
        try:
            d = np.linalg.solve(J, -np.array([f1, f2]))
        except np.linalg.LinAlgError:
            return None
        # damped update, keep var > 0
        lam = 1.0
        for _ in range(40):
            mn, vn = mu + lam * d[0], var + lam * d[1]
            if vn > 1e-12 and abs(mn) < m:
                mu, var = mn, vn
                break
            lam *= 0.5
        else:
            return None
    return None


def kernel_exact_semigroup(m, k, nu, co, order="exact"):
    """Weights approximating the true solution operator over k*dt.

    The exact operator is convolution with a Gaussian of
        mean  = -k*Co   (departure point, semi-Lagrangian shift)
        var   =  2*k*nu
    in lattice units.  order='spec' reproduces the 3-row spec instead
    (M2 = 2*k*nu, i.e. dropping the u_tt term).
    """
    mu = -k * co
    var = 2.0 * k * nu
    if order == "spec":
        m2 = var                      # team-lead 3-row system
    else:
        m2 = var + mu ** 2            # true second moment of the heat kernel
    return kernel_moment_matched(m, mu, m2)


def symbol_error(w, k, nu, co, theta):
    """|W(theta) - exp(-i k Co theta - k nu theta^2)| : one-big-step symbol error."""
    j = np.arange(-(len(w) // 2), len(w) // 2 + 1)
    W = (w[None, :] * np.exp(1j * np.outer(theta, j))).sum(-1)
    G = np.exp(-1j * k * co * theta - k * nu * theta ** 2)
    return W - G
