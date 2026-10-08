"""
stencil.py -- weight solvers for PDE-constrained meshfree stencils, and the
positivity certificate.

Setting
-------
A meshfree solver predicts u at a target point z* = (x*, t*) from n scattered
neighbour values u_i at offsets (dx_i, dt_i) = (x_i - x*, t_i - t*):

    u_hat* = sum_i w_i u_i .

The weights come from PDE-constrained Taylor moment conditions for

    u_t + c u_x = alpha u_xx .

Substituting the PDE into the Taylor expansion of u(x*+dx, t*+dt) about z*
turns time derivatives into space derivatives.  Keeping terms through second
order in the characteristic variable gives, per neighbour,

    u_i = u* + q_i u_x + p_i u_xx + O(3),        where

    q_i = dx_i - c dt_i                                  (row 2)
    p_i = 1/2 (dx_i - c dt_i)^2 + alpha dt_i             (row 3, EXACT)

Requiring sum_i w_i * (1, q_i, p_i) = (1, 0, 0) annihilates u_x and u_xx and
leaves a third-order local truncation error.

Two variants of row 3 are supported:

  variant='given'  ->  p_i = 1/2 dx_i^2 + alpha dt_i
       the row used by the existing neural-integrators code.  It drops the
       cross term -c dx dt (from dx dt u_xt) and the 1/2 c^2 dt^2 term (from
       1/2 dt^2 u_tt).  It is exact only when |c dt| << |dx|.

  variant='exact'  ->  p_i = 1/2 (dx_i - c dt_i)^2 + alpha dt_i
       the true second-order PDE-constrained moment.  With this row the
       classical schemes drop out exactly (see task2_cfl.py): the 3-point
       centred stencil reproduces Lax-Wendroff, whereas the 'given' row
       reproduces the unconditionally unstable FTCS-central advection scheme.

Row scaling
-----------
build_A divides row 2 by h and row 3 by h^2 with h a local length scale.  This
is a diagonal row rescaling of the equality system; since b = (1, 0, 0) and the
two rescaled rows have right-hand side 0, it does NOT change the feasible set
{w >= 0 : A w = b}, nor the min-L1 solution.  It DOES change the min-2-norm
(lstsq) solution and the condition number.

Key identity
------------
Row 1 forces sum_i w_i = 1, hence ||w||_1 >= |sum_i w_i| = 1 always, with
equality iff every w_i >= 0.  So

    a nonnegative stencil exists  <=>  min ||w||_1 = 1  <=>  the stencil is a
    convex combination of its neighbours  <=>  a discrete maximum principle
    holds locally.

See THEORY.md.
"""

import itertools

import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------------------
# Default problem parameters (advection-diffusion benchmark)
# ----------------------------------------------------------------------------
ALPHA = 0.1
C = 1.0
NX = 100
T_FINAL = 0.5

X_FD = np.linspace(0.0, 1.0, NX)
DX = X_FD[1] - X_FD[0]                      # 0.010101010101
_DT_CFL = min(0.9 * DX / abs(C), 0.45 * DX**2 / ALPHA)
NT = max(int(np.ceil(T_FINAL / _DT_CFL)) + 1, 10)    # 1090
T_FD = np.linspace(0.0, T_FINAL, NT)
DT = T_FD[1] - T_FD[0]                      # 4.591368e-04


# ----------------------------------------------------------------------------
# Moment system
# ----------------------------------------------------------------------------
def moment_points(dxi, dti, alpha=ALPHA, c=C, variant="given"):
    """Return the unscaled 2-D moment coordinates (q_i, p_i) of each neighbour.

    q_i annihilates u_x, p_i annihilates u_xx.  Positivity feasibility is
    exactly the statement that the origin lies in the convex hull of the points
    (q_i, p_i) -- see positive_feasible_hull.
    """
    dxi = np.asarray(dxi, float)
    dti = np.asarray(dti, float)
    q = dxi - c * dti
    if variant == "given":
        p = 0.5 * dxi**2 + alpha * dti
    elif variant == "exact":
        p = 0.5 * q**2 + alpha * dti
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return q, p


def build_A(dxi, dti, alpha=ALPHA, c=C, variant="given", order=2):
    """Build the moment matrix A and right-hand side b.

    order=2 -> 3 rows (consistency, u_x, u_xx).  Second-order accurate.
    order=1 -> 2 rows (consistency, u_x).        First-order accurate; the
               right choice for pure advection (alpha = 0), where the u_xx row
               degenerates to sum_i w_i dx_i^2 = 0 and admits no nonnegative
               solution other than a point mass at dx = 0.
    """
    dxi = np.asarray(dxi, float)
    dti = np.asarray(dti, float)
    q, p = moment_points(dxi, dti, alpha, c, variant)
    h = max(np.max(np.abs(dxi)), np.sqrt(alpha * np.max(np.abs(dti))), 1e-12)
    rows = [np.ones(len(dxi)), q / h]
    if order >= 2:
        rows.append(p / h**2)
    A = np.array(rows)
    b = np.zeros(A.shape[0])
    b[0] = 1.0
    return A, b


# ----------------------------------------------------------------------------
# Solvability of the equality system itself (prior to any positivity question)
# ----------------------------------------------------------------------------
def moments_solvable(dxi, dti, tol=1e-8, **kw):
    """Is A w = b solvable AT ALL, for any w (signs unrestricted)?

    It is not always.  The clearest failure is a VERTICAL stencil: all
    neighbours at the same x, stacked over several time levels.  Then dx_i is
    constant, so both q_i = dx - c dt_i and p_i = 1/2 dx^2 + alpha dt_i are
    affine in the single varying quantity dt_i.  Rows 2 and 3 therefore lie in
    span{row of ones, row of dt_i}, A drops to rank 2, and b = (1,0,0) is not
    in its range.  Physically: you cannot recover two spatial derivatives from
    data at one spatial location.

    np.linalg.lstsq does NOT signal this -- it returns the least-squares point,
    whose residual is ~0.5, and which satisfies none of the moment conditions.
    A caller that trusts lstsq WITHOUT a guard will silently use a stencil that
    is not consistent with the PDE.

    Severity: latent, not active.  A cond(A) > 1e4 gate catches every such case
    incidentally and cleanly (rejected stencils have cond ~1e16 and residual
    ~0.47; every stencil kept by the gate has residual < 1.5e-14).  The rate is
    2 in 15948 random 5-point draws.  Prefer this explicit residual check, or an
    assert on |sum(w) - 1|, over relying on conditioning to imply consistency.
    """
    A, b = build_A(dxi, dti, **kw)
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(np.abs(A @ w - b).max()) <= tol


# ----------------------------------------------------------------------------
# Solver 1: minimum 2-norm (what the existing code does)
# ----------------------------------------------------------------------------
def w_lstsq(dxi, dti, **kw):
    """Minimum-2-norm solution of A w = b.  Always returns something."""
    A, b = build_A(dxi, dti, **kw)
    return np.linalg.lstsq(A, b, rcond=None)[0]


# ----------------------------------------------------------------------------
# Solver 2: positivity certificate via LP feasibility
# ----------------------------------------------------------------------------
def w_positive(dxi, dti, tol=1e-9, **kw):
    """Find w >= 0 with A w = b, or return None if none exists.

    When a solution exists, sum_i w_i = 1 and w >= 0, so ||w||_1 = 1 exactly.
    The LP objective is zero: any feasible vertex will do.
    """
    A, b = build_A(dxi, dti, **kw)
    n = A.shape[1]
    res = linprog(np.zeros(n), A_eq=A, b_eq=b,
                  bounds=[(0, None)] * n, method="highs")
    if res.status != 0:
        return None
    w = res.x
    if w.min() < -tol:
        return None
    return np.maximum(w, 0.0)


def positive_feasible(dxi, dti, **kw):
    """Boolean wrapper around w_positive."""
    return w_positive(dxi, dti, **kw) is not None


# ----------------------------------------------------------------------------
# Solver 3: minimum L1 (graceful degradation when positivity is infeasible)
# ----------------------------------------------------------------------------
def w_minL1(dxi, dti, **kw):
    """min ||w||_1 subject to A w = b, via the split w = p - n, p, n >= 0.

    Returns (w, ||w||_1), or (None, None) if the equality system itself is
    infeasible -- see moments_solvable for when that happens (vertical
    stencils).  This is a property of the GEOMETRY, not of the L1 objective.
    The split is exact at the optimum: p_i n_i = 0 for a basic optimal
    solution, so sum(p + n) = ||w||_1.
    """
    A, b = build_A(dxi, dti, **kw)
    n = A.shape[1]
    Aeq = np.hstack([A, -A])
    res = linprog(np.ones(2 * n), A_eq=Aeq, b_eq=b,
                  bounds=[(0, None)] * (2 * n), method="highs")
    if res.status != 0:
        return None, None
    w = res.x[:n] - res.x[n:]
    return w, float(np.abs(w).sum())


# ----------------------------------------------------------------------------
# Solver 4: basic-solution enumeration (LP-free, exact)
# ----------------------------------------------------------------------------
def w_positive_enum(dxi, dti, tol=1e-10, **kw):
    """Find w >= 0 with A w = b by enumerating m-column subsets.

    LP basic-solution theory: if {w >= 0 : A w = b} is nonempty and A has m
    rows, then it contains a basic feasible solution supported on at most m
    columns.  So for n neighbours and m rows it suffices to try all C(n, m)
    square subsystems.  For n = 5, m = 3 that is 10 3x3 solves.

    Exactly equivalent to the LP (no tolerance heuristics beyond the
    singularity guard).  Returns None if no subset yields w >= 0.
    """
    A, b = build_A(dxi, dti, **kw)
    m, n = A.shape
    for cols in itertools.combinations(range(n), m):
        M = A[:, cols]
        try:
            ws = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(ws)):
            continue
        if ws.min() >= -tol:
            # verify: a near-singular M can produce a large spurious solution
            if np.abs(M @ ws - b).max() > 1e-8 * max(1.0, np.abs(ws).max()):
                continue
            w = np.zeros(n)
            w[list(cols)] = np.maximum(ws, 0.0)
            return w
    return None


# ----------------------------------------------------------------------------
# Solver 5: the cheap closed-form test -- origin in a 2-D convex hull
# ----------------------------------------------------------------------------
def positive_feasible_hull(dxi, dti, alpha=ALPHA, c=C, variant="given",
                           tol=1e-12):
    """Exact O(n log n) positivity test, no linear algebra at all.

    Row 1 of A forces sum_i w_i = 1, so w ranges over the simplex.  Rows 2-3
    then say that the convex combination of the 2-D points (q_i, p_i) is the
    origin.  Hence

        positivity feasible  <=>  0 in conv{(q_i, p_i)}.

    By Gordan's theorem, 0 is in the convex hull of a finite planar point set
    iff no open half-plane through the origin contains every point, i.e. iff
    the points' polar angles have no angular gap wider than pi.

    Geometric reading of the two coordinates:
        q_i = dx_i - c dt_i  is the neighbour's offset from the foot of the
              characteristic through z*.  Needing q's of both signs means the
              stencil must BRACKET THE CHARACTERISTIC FOOT.
        p_i >= 0 iff |q_i| >= sqrt(2 alpha |dt_i|) ('exact' variant).  Needing
              p's of both signs means the stencil must contain at least one
              neighbour FARTHER than the diffusion length and at least one
              NEARER than it.
    """
    q, p = moment_points(dxi, dti, alpha, c, variant)
    sq = np.max(np.abs(q))
    sp = np.max(np.abs(p))
    if sq > 0:
        q = q / sq
    if sp > 0:
        p = p / sp
    r = np.hypot(q, p)
    if np.any(r <= tol):
        return True                       # a neighbour sits exactly at the origin
    th = np.sort(np.arctan2(p, q))
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    return bool(gaps.max() <= np.pi + tol)


# ----------------------------------------------------------------------------
# Exact solution of the benchmark advection-diffusion problem
# ----------------------------------------------------------------------------
_BETA = C / (2 * ALPHA)
_XQ = np.linspace(0.0, 1.0, 40001)
_V0 = (np.sin(np.pi * _XQ) + 0.5 * np.sin(2 * np.pi * _XQ)) * np.exp(-_BETA * _XQ)
_NARR = np.arange(1, 401)
_BN = np.array([2 * np.trapz(_V0 * np.sin(n * np.pi * _XQ), _XQ) for n in _NARR])


def u_true(x, t):
    """Exact solution: u_t + c u_x = alpha u_xx, u(x,0)=sin(pi x)+0.5 sin(2 pi x),
    u(0,t)=u(1,t)=0.  Via the Cole-Hopf-like substitution u = exp(beta x -
    alpha beta^2 t) v with beta = c/(2 alpha), v solving the pure heat equation.
    """
    x = np.atleast_1d(np.asarray(x, float))
    v = (np.sin(np.pi * np.outer(x, _NARR))
         * (_BN * np.exp(-ALPHA * (_NARR * np.pi) ** 2 * t))).sum(-1)
    return np.exp(_BETA * x - ALPHA * _BETA**2 * t) * v


if __name__ == "__main__":
    print(f"dx = {DX:.10g}   dt = {DT:.10g}   nt = {NT}")
    print(f"r  = alpha dt/dx^2 = {ALPHA*DT/DX**2:.6f}")
    print(f"nu = c dt/dx       = {C*DT/DX:.6f}")
    print(f"cell Peclet = c dx/alpha = {C*DX/ALPHA:.6f}")


# ============================================================================
# The full PDE-constrained moment hierarchy (added for Task 6)
# ============================================================================
# Because d_x and L = -c d_x + alpha d_x^2 commute,
#
#     u(x+dx, t+dt) = exp(dx d_x) exp(dt L) u
#                   = exp( (dx - c dt) d_x + (alpha dt) d_x^2 ) u
#                   = exp( q d_x + a d_x^2 ) u ,     q = dx - c dt, a = alpha dt.
#
# So EVERY time derivative is absorbed into the two numbers (q, a), and the
# coefficient of d_x^s u for neighbour i is the s-th Taylor coefficient of the
# generating function exp(q xi + a xi^2):
#
#     C_s(q, a) = sum_{j=0}^{floor(s/2)}  q^(s-2j) a^j / ((s-2j)! j!)
#
#     C_0 = 1
#     C_1 = q
#     C_2 = q^2/2 + a                    <- the 'exact' row of build_A
#     C_3 = q^3/6 + q a
#     C_4 = q^4/24 + q^2 a/2 + a^2/2
#
# Requiring sum_i w_i C_s(q_i, a_i) = delta_{s0} for s = 0..p annihilates
# u_x ... d_x^p u and leaves a local truncation error O(h^{p+1}).
#
# NOTE what this means for the "u_tt cannot be cancelled" argument: there is no
# free-standing u_tt term.  1/2 dt^2 u_tt is redistributed by the PDE into
# 1/2 c^2 dt^2 u_xx  -  c alpha dt^2 u_xxx  +  1/2 alpha^2 dt^2 u_xxxx, and the
# u_xx part is already inside C_2, which nonnegative weights cancel routinely
# (FTCS does exactly that).  sum_i w_i dt_i^2 is NOT one of the conditions.
from math import factorial as _fact


def moment_coeff(q, a, s):
    """[xi^s] exp(q xi + a xi^2), vectorised over q and a."""
    q = np.asarray(q, float)
    a = np.asarray(a, float)
    tot = np.zeros(np.broadcast(q, a).shape)
    for j in range(s // 2 + 1):
        tot = tot + q ** (s - 2 * j) * a ** j / (_fact(s - 2 * j) * _fact(j))
    return tot


def build_A_order(dxi, dti, alpha=ALPHA, c=C, order=2, scale=True):
    """Moment matrix for rows s = 0..order of the exact hierarchy.

    scale=True divides row s by h^s, which is a diagonal row rescaling with
    zero right-hand side on rows s >= 1: it cannot change {w >= 0 : A w = b}
    (only the lstsq solution and the condition number).
    """
    dxi = np.asarray(dxi, float)
    dti = np.asarray(dti, float)
    q = dxi - c * dti
    a = alpha * dti
    h = max(np.max(np.abs(dxi)), np.sqrt(abs(alpha) * np.max(np.abs(dti))), 1e-12)
    rows = [moment_coeff(q, a, s) / (h**s if scale else 1.0)
            for s in range(order + 1)]
    A = np.array(rows)
    b = np.zeros(order + 1)
    b[0] = 1.0
    return A, b


def w_positive_order(dxi, dti, order=2, tol=1e-9, **kw):
    """Nonnegative weights satisfying the hierarchy through `order`, or None."""
    A, b = build_A_order(dxi, dti, order=order, **kw)
    res = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b,
                  bounds=[(0, None)] * A.shape[1], method="highs")
    if res.status != 0 or res.x.min() < -tol:
        return None
    return np.maximum(res.x, 0.0)


def w_lstsq_order(dxi, dti, order=2, **kw):
    """Minimum-2-norm signed weights for the hierarchy through `order`."""
    A, b = build_A_order(dxi, dti, order=order, **kw)
    return np.linalg.lstsq(A, b, rcond=None)[0]
