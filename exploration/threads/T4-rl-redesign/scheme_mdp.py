"""
MDP-B: scheme selection as a Markov decision process, with a computable optimum.

The point of this module is VALIDATION. Ali's query-driven walker MDP has no known
optimal policy, so "the agent learned something" is unfalsifiable. Here the optimal
policy is computable by exact dynamic programming, so we can ask the sharp question:
does RL find the optimum on a problem where we know it?

Why DP is exact here -- and this is the interesting part:

    With positive weights, ||w||_1 = 1, so errors accumulate LINEARLY:
        e_total  ~  sum_j tau(m_j, k_j)
    An additive objective over a 1-D time axis is exactly a shortest-path problem.
    Positivity is what makes planning tractable.

    Without positivity, ||w||_1 > 1 makes the accumulation multiplicative and
    history-dependent, DP no longer factorises, and search is genuinely needed.
    That is the honest argument for using RL at all.

PDE:  u_t + c u_x = alpha u_xx  on [0,1], u(0,t)=u(1,t)=0
"""
import numpy as np
from scipy.optimize import linprog

# ----------------------------------------------------------------- setup
ALPHA, C = 0.1, 1.0
NX, T_DOMAIN = 201, 0.5
X = np.linspace(0.0, 1.0, NX)
DX = X[1] - X[0]
DT = min(0.9 * DX / abs(C), 0.45 * DX**2 / ALPHA)      # explicit CFL step

_BETA = C / (2 * ALPHA)
_xq = np.linspace(0.0, 1.0, 40001)
_v0 = (np.sin(np.pi * _xq) + 0.5 * np.sin(2 * np.pi * _xq)) * np.exp(-_BETA * _xq)
_N = np.arange(1, 401)
_BN = np.array([2 * np.trapz(_v0 * np.sin(n * np.pi * _xq), _xq) for n in _N])


def u_true(x, t):
    x = np.atleast_1d(np.asarray(x, float))
    v = (np.sin(np.pi * np.outer(x, _N)) * (_BN * np.exp(-ALPHA * (_N * np.pi) ** 2 * t))).sum(-1)
    return np.exp(_BETA * x - ALPHA * _BETA**2 * t) * v


# ------------------------------------------------- the stencil under study
def build_A(dxi, dti):
    """PDE-constrained Taylor rows. Same system as the RL-For-PDEs code."""
    h = max(np.max(np.abs(dxi)), np.sqrt(ALPHA * np.max(np.abs(dti))), 1e-12)
    xi = dxi - C * dti                       # characteristic offset
    A = np.array([np.ones(len(dxi)),
                  xi / h,
                  (0.5 * xi**2 + ALPHA * dti) / h**2])   # CORRECTED (F8)
    return A, np.array([1.0, 0.0, 0.0])


_WCACHE = {}


def weights(m, k, offsets=None):
    """Positive consistent weights for a stencil of half-width m at time offset -k*DT.

    Among all feasible w >= 0 satisfying the 3 consistency rows, pick the one
    minimising the 4th spatial moment -- the leading uncancelled error term.
    Returns None if no non-negative solution exists (the positivity frontier).
    """
    key = (m, k) if offsets is None else (m, k, tuple(offsets))
    if key in _WCACHE:
        return _WCACHE[key]
    idx = np.arange(-m, m + 1) if offsets is None else np.asarray(offsets)
    dxi = idx * DX
    dti = -k * DT * np.ones(len(idx))
    A, b = build_A(dxi, dti)
    obj = (dxi / DX) ** 4                       # minimise leading uncancelled moment
    res = linprog(obj, A_eq=A, b_eq=b, bounds=[(0, None)] * len(idx), method="highs")
    w = res.x if res.status == 0 else None
    if w is not None and abs(w.sum() - 1.0) > 1e-6:       # F9 guard: rank-deficient A
        w = None
    _WCACHE[key] = w
    return w


def k_max(m, kcap=4000):
    """Largest k for which a non-negative consistent stencil of half-width m exists."""
    lo, hi = 1, 1
    while hi < kcap and weights(m, hi) is not None:
        lo, hi = hi, hi * 2
    if hi >= kcap:
        return kcap
    while lo + 1 < hi:                                   # bisect
        mid = (lo + hi) // 2
        if weights(m, mid) is not None:
            lo = mid
        else:
            hi = mid
    return lo


# ------------------------------------------------------------ the dynamics
def _window(j, m):
    """2m+1 consecutive nodes containing j, shifted to stay inside the domain.

    Near a wall we keep the stencil WIDTH (which is what buys the time step) and
    make the stencil one-sided, rather than shrinking m -- a narrow stencil cannot
    take the large step and would force the whole march back to the CFL limit.
    """
    lo = int(np.clip(j - m, 0, NX - 1 - 2 * m))
    return lo, np.arange(lo, lo + 2 * m + 1) - j


def advance(u, m, k):
    """Apply the wide positive stencil once, one-sided near the walls."""
    if 2 * m + 1 > NX:
        return None
    w = weights(m, k)
    if w is None:
        return None
    out = np.zeros_like(u)
    conv = np.convolve(u, w[::-1], mode="same")
    out[m:NX - m] = conv[m:NX - m]
    for j in list(range(1, min(m, NX - 1))) + list(range(max(1, NX - 1 - m), NX - 1)):
        lo, off = _window(j, m)
        wj = weights(m, k, offsets=off)
        if wj is None:
            return None
        out[j] = wj @ u[lo: lo + 2 * m + 1]
    out[0] = out[-1] = 0.0
    return out


_UCACHE = {}


def _u_at(j):
    if j not in _UCACHE:
        _UCACHE[j] = u_true(X, j * DT)
    return _UCACHE[j]


_TCACHE = {}


def tau(m, k, j_to):
    """One-step truncation error of (m,k) landing at time index j_to, on exact data."""
    key = (m, k, j_to)
    if key in _TCACHE:
        return _TCACHE[key]
    if weights(m, k) is None:
        _TCACHE[key] = np.inf
        return np.inf
    pred = advance(_u_at(j_to - k), m, k)
    val = np.inf if pred is None else float(np.max(np.abs(pred - _u_at(j_to))))
    _TCACHE[key] = val
    return val


def cost(m):
    """Compute cost of one macro-step: one FMA per stencil point per interior node."""
    return (NX - 2) * (2 * m + 1)


# ------------------------------------------------------- the exact optimum
def dp_optimal(J, menu, lam):
    """Exact DP for  min  sum_steps [ cost(m) + lam * tau(m,k) ]  reaching index J.

    Valid precisely because positivity makes the error additive.
    Returns (value, policy) with policy a list of (m,k) from t=0.
    """
    V = np.full(J + 1, np.inf)
    arg = [None] * (J + 1)
    V[0] = 0.0
    for j in range(1, J + 1):
        for (m, k) in menu:
            if k > j:
                continue
            prev = V[j - k]
            if not np.isfinite(prev):
                continue
            cand = prev + cost(m) + lam * tau(m, k, j)
            if cand < V[j]:
                V[j] = cand
                arg[j] = (m, k)
    pol, j = [], J
    while j > 0 and arg[j] is not None:
        m, k = arg[j]
        pol.append((m, k))
        j -= k
    return V[J], pol[::-1]


def rollout(policy, J):
    """Run a policy and report (final L-inf error, total cost, steps)."""
    u = u_true(X, 0.0).copy()
    j, tot = 0, 0
    for (m, k) in policy:
        if j + k > J:
            break
        nxt = advance(u, m, k)
        if nxt is None:
            return np.inf, tot, 0
        u, j, tot = nxt, j + k, tot + cost(m)
    return float(np.max(np.abs(u - u_true(X, j * DT)))), tot, j
