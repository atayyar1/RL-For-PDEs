"""Consistency rows, weight solvers, and the positivity certificate.

Two ways to build the constraint rows A w = b for a local predictor
u(z*) ~ sum_i w_i u(z_i), with offsets dx_i = x_i - x*, dt_i = t_i - t*:

  rows_taylor  -- Taylor expansion with the PDE substituted RECURSIVELY.
                  Exact through 2nd order.  [F8: the original code dropped
                  -c dx dt and 1/2 c^2 dt^2 from the u_xx row.]

  rows_moment  -- match the moments of the exact propagator exp(tau L), which
                  are computable from the PDE COEFFICIENTS ALONE via a closed
                  triangular ODE hierarchy (no Green's function). Extends to
                  arbitrary order p, which the Taylor rows cannot. [F6]

`rows_taylor` is the p=2 case of `rows_moment` restricted to one time level.

Positivity: since sum_i w_i = 1 always, w >= 0 <=> ||w||_1 = 1 <=> the predictor
is a convex combination <=> a discrete maximum principle holds <=> errors
accumulate linearly rather than geometrically under recursion.
"""
import numpy as np
from scipy.optimize import linprog
from .pde import DEFAULT


# ------------------------------------------------------------------ rows
def rows_taylor(dxi, dti, prob=DEFAULT):
    """Exact 2nd-order PDE-constrained rows. xi = dx - c dt is the characteristic offset.

    In the frame moving at speed c the equation is pure diffusion, so the rows
    are simply  [1, xi, 1/2 xi^2 + alpha dt].
    """
    dxi, dti = np.asarray(dxi, float), np.asarray(dti, float)
    h = max(np.max(np.abs(dxi)), np.sqrt(prob.alpha * np.max(np.abs(dti))), 1e-12)
    xi = dxi - prob.c * dti
    A = np.array([np.ones(len(dxi)), xi / h, (0.5 * xi**2 + prob.alpha * dti) / h**2])
    return A, np.array([1.0, 0.0, 0.0])


def propagator_moments(tau, p, prob=DEFAULT):
    """Raw moments 0..p of the propagator over tau, in the offset dx = x_i - x*.

    The Green's function obeys the same PDE, so integrating by parts gives the
    closed triangular hierarchy (in z = x* - x_i)

        dM_q/dtau = alpha q(q-1) M_{q-2} + c q M_{q-1},   M_q(0) = delta_q0

    which integrates exactly. Moments in dx = -z carry a factor (-1)^q.
    Uses ONLY the PDE coefficients -- this is what makes the construction general.
    """
    M = np.zeros(p + 1)
    M[0] = 1.0
    # exact solution of the triangular hierarchy: the propagator is Gaussian with
    # mean c*tau and variance 2*alpha*tau in z, so use the moment recurrence.
    mu, var = prob.c * tau, 2.0 * prob.alpha * tau
    if p >= 1:
        M[1] = mu
    for q in range(2, p + 1):
        M[q] = mu * M[q - 1] + (q - 1) * var * M[q - 2]
    return M * (-1.0) ** np.arange(p + 1)      # convert z -> dx


def rows_moment(dxi, dti, p, prob=DEFAULT):
    """Match propagator moments 0..p. Requires a single time level (all dti equal)."""
    dxi, dti = np.asarray(dxi, float), np.asarray(dti, float)
    if np.ptp(dti) > 1e-14:
        raise ValueError("rows_moment needs one time level; use rows_taylor otherwise")
    tau = -float(dti[0])
    h = max(np.max(np.abs(dxi)), 1e-12)
    A = np.vstack([(dxi / h) ** q for q in range(p + 1)])
    b = propagator_moments(tau, p, prob) / h ** np.arange(p + 1)
    return A, b


# --------------------------------------------------------------- solvers
def solve_minnorm(A, b, guard=True):
    """Minimum 2-norm solution (what the original code used). None if inconsistent.

    `guard` catches the rank-deficient case (e.g. every neighbour at the same x),
    where lstsq silently returns a large-residual vector with sum(w) != 1. [F9]
    """
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    if guard and abs(w.sum() - 1.0) > 1e-6:
        return None
    return w


def solve_positive(A, b):
    """Any w >= 0 with A w = b, or None. Feasible => ||w||_1 = 1 exactly."""
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b,
                bounds=[(0, None)] * A.shape[1], method="highs")
    return r.x if r.status == 0 else None


def solve_maxent(A, b, tol=1e-13, iters=200):
    """The MAXIMUM-ENTROPY point of {w >= 0 : A w = b}. Prefer this to solve_positive.

    `solve_positive` runs an LP, and linear programming returns VERTICES: with 3
    equality rows it hands back a weight vector with at most 3 non-zeros, which is
    the extremal two-point measure that F26 showed is the WORST member of the
    feasible set (consistent, positive, stable, and 1e10 times less accurate than
    an interior point).

    Maximising -sum w log w subject to the same constraints returns the interior
    point instead. Since w ~ exp(A[1:]^T lam) and the rows are 1, xi, xi^2, the
    answer is a discrete Gaussian in the characteristic offset xi -- i.e. the
    propagator itself, recovered without being told it. (Credit: T5 proposed
    max-entropy; T2 showed it coincides with their two-parameter construction to
    1e-16.)

    Newton on the convex dual with backtracking. Returns None if not converged.
    """
    R, lam, tgt = A[1:], np.zeros(A.shape[0] - 1), b[1:]

    def dual(l):
        z = R.T @ l
        mx = z.max()
        return np.log(np.exp(z - mx).sum()) + mx - l @ tgt

    for _ in range(iters):
        z = R.T @ lam
        w = np.exp(z - z.max())
        w /= w.sum()
        g = R @ w - tgt
        if np.abs(g).max() < tol:
            return w
        Rw = R @ w
        H = (R * w) @ R.T - np.outer(Rw, Rw)
        try:
            step = np.linalg.solve(H + 1e-14 * np.eye(len(lam)), g)
        except np.linalg.LinAlgError:
            return None
        f0, t_ = dual(lam), 1.0
        for _ in range(60):
            if dual(lam - t_ * step) <= f0 - 1e-4 * t_ * (g @ step):
                break
            t_ *= 0.5
        lam = lam - t_ * step
    return None


def solve_min_l1(A, b):
    """Minimum ||w||_1 solution (graceful degradation when positivity is infeasible)."""
    n = A.shape[1]
    r = linprog(np.ones(2 * n), A_eq=np.hstack([A, -A]), b_eq=b,
                bounds=[(0, None)] * 2 * n, method="highs")
    return (r.x[:n] - r.x[n:]) if r.status == 0 else None


# -------------------------------------------- fast positivity certificate
def positive_feasible(A, b, tol=1e-12):
    """Exact O(n log n) positivity test for the 3-row system, no LP.

    With b = [1,0,0], a non-negative solution is a convex combination of the
    planar points p_i = (A[1,i], A[2,i]) that lands on the origin. So feasibility
    <=> the origin lies in conv{p_i} <=> the points are not all strictly inside
    an open half-plane through the origin.
    Credit: T1.
    """
    if A.shape[0] != 3 or not (abs(b[0] - 1) < tol and abs(b[1]) < tol and abs(b[2]) < tol):
        return solve_positive(A, b) is not None          # fall back for other shapes
    P = A[1:3].T                                          # (n,2)
    rad = np.hypot(P[:, 0], P[:, 1])
    if np.any(rad <= tol):                                # a point at the origin
        return True
    ang = np.sort(np.arctan2(P[:, 1], P[:, 0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return bool(np.max(gaps) <= np.pi + 1e-12)


def k_max(m, prob=DEFAULT):
    """Largest k for which a symmetric half-width-m stencil at -k*dt is positive-feasible.

    Sharp closed form (credit: T5), valid for the CORRECTED rows. The consistency
    conditions fix the RAW second moment 2*alpha*k*dt + (c*k*dt)^2 -- variance PLUS
    mean squared -- and a probability measure on {-m..m} can realise it only while
    it stays below m^2*dx^2. Solving the quadratic:

        k_max = ( -r + sqrt(r^2 + nu^2 m^2) ) / nu^2,   r = alpha dt/dx^2, nu = c dt/dx

    Limits: nu -> 0 gives m^2/(2r) (diffusive); m -> inf gives m/nu (advective).
    It interpolates both smoothly and matches the LP at every m tested (m <= 76).

    NOTE the uncorrected row instead saturates at 2*alpha/(c^2 dt) regardless of m
    (435 steps at the benchmark). That saturation is an artefact of the dropped
    u_tt terms and disappears with the corrected row.
    """
    r, nu = prob.r, prob.nu
    if nu == 0:
        return int(m**2 / (2 * r))
    return int((-r + np.sqrt(r**2 + nu**2 * m**2)) / nu**2)


def amplification(w):
    """||w||_1 -- a SUFFICIENT, conservative, always-computable stability certificate.

    ||w||_1 = 1 (equivalently w >= 0) gives linear error accumulation at every
    composition depth, with no transient and no translation invariance required.

    It is NOT necessary. The bound |e| <= ||w||_1^L is a worst-case over adversarial
    error patterns and is generically nowhere near attained: Lax-Wendroff at nu=0.4
    has ||w||_1 = 1.24 yet max|symbol| = 1 exactly, and 512-fold composition gives
    ||w||_1 = 1.59 against a bound of 6.8e47. (Credit: T5.)

    The sharp criterion is von Neumann, max|symbol| <= 1 -- but the symbol requires
    TRANSLATION INVARIANCE, which scattered meshfree geometry does not have. So in
    this setting ||w||_1 is the only certificate available, at the cost of rejecting
    schemes (like Lax-Wendroff) that are perfectly stable.
    """
    return float(np.abs(w).sum())


def symbol_max(w, offsets):
    """max |g(theta)| -- the sharp von Neumann factor. Uniform-grid stencils only."""
    th = np.linspace(-np.pi, np.pi, 2049)
    g = sum(wi * np.exp(-1j * th * oi) for wi, oi in zip(w, offsets))
    return float(np.abs(g).max())
