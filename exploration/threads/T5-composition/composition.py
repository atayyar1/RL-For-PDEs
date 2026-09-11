"""
T5-composition: core library.

A *stencil* (agent) is a finitely supported signed measure on a displacement
lattice.  We represent it as

    w      : array of weights,        shape (n,)
    offx   : integer x-offsets (cells) shape (n,)
    offt   : integer t-offsets (steps) shape (n,)   (negative = into the past)

so that   uhat(x_j, t_m) = sum_i w_i * u(x_{j+offx_i}, t_{m+offt_i}).

For the single-time-level stencils used throughout Tasks 2-4, offt is
constant (= -1) and the object reduces to a 1-D kernel in x, for which
composition is *exactly* discrete convolution.
"""
import os
for _v in ("OMP","OPENBLAS","MKL","VECLIB_MAXIMUM","NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")  # tiny matrices: threading is pure overhead
import numpy as np

# ---------------------------------------------------------------- problem setup
ALPHA, C = 0.1, 1.0
NX, T_END = 100, 0.5
X_FD = np.linspace(0.0, 1.0, NX)
DX = X_FD[1] - X_FD[0]
_dt_cfl = min(0.9 * DX / abs(C), 0.45 * DX**2 / ALPHA)
NT = max(int(np.ceil(T_END / _dt_cfl)) + 1, 10)
DT = T_END / (NT - 1)

R = ALPHA * DT / DX**2          # 0.45
RA = C * DT / DX                # 0.0455...

W_FTCS = np.array([R + RA / 2, 1 - 2 * R, R - RA / 2])   # offsets (-1, 0, +1)
OFF_FTCS = np.array([-1, 0, 1])


# ---------------------------------------------------------------- moments
def raw_moments(w, offx, offt, n_max=6, dx=DX, dt=DT):
    """Raw space-time moments M_{p,q} = sum_i w_i xi_i^p tau_i^q, physical units.

    Returns dict {(p, q): value} for p + q <= n_max.
    """
    xi = np.asarray(offx, float) * dx
    ta = np.asarray(offt, float) * dt
    out = {}
    for p in range(n_max + 1):
        for q in range(n_max + 1 - p):
            out[(p, q)] = float(np.sum(w * xi**p * ta**q))
    return out


def spatial_raw_moments(w, offx, n_max=8, dx=DX):
    """M_p = sum_i w_i xi_i^p  (single time level; physical units)."""
    xi = np.asarray(offx, float) * dx
    return np.array([np.sum(w * xi**p) for p in range(n_max + 1)])


def cumulants_from_raw(M, n_max=None):
    """Cumulants kappa_1..kappa_n from raw moments M_0..M_n.

    Requires M_0 != 0; normalises by M_0 first (so kappa_n are the cumulants of
    the *normalised* measure).  Uses the standard recursion
        kappa_n = m_n - sum_{k=1}^{n-1} C(n-1, k-1) kappa_k m_{n-k}.
    Only meaningful (as a probability statement) when w >= 0, but the algebra
    and the additivity theorem hold for any signed measure with M_0 = 1.
    """
    from math import comb
    M = np.asarray(M, float)
    n = (len(M) - 1) if n_max is None else n_max
    m = M[: n + 1] / M[0]
    k = np.zeros(n + 1)
    for j in range(1, n + 1):
        k[j] = m[j] - sum(comb(j - 1, i - 1) * k[i] * m[j - i] for i in range(1, j))
    return k


def exact_cumulants(tau, n_max=6, alpha=ALPHA, c=C):
    """Cumulants of the exact advection-diffusion propagator over time tau>0,
    as a kernel in the *displacement* xi = x_source - x_target.

    G_tau(xi) = N(mean = -c tau, var = 2 alpha tau)  =>  all cumulants linear in tau.
    """
    k = np.zeros(n_max + 1)
    if n_max >= 1:
        k[1] = -c * tau
    if n_max >= 2:
        k[2] = 2.0 * alpha * tau
    return k


def cumulant_defect(w, offx, tau, n_max=6, dx=DX):
    """eps_n = kappa_n(w) - kappa_n^exact(tau).  THE object that adds under composition."""
    M = spatial_raw_moments(w, offx, n_max=n_max, dx=dx)
    return cumulants_from_raw(M, n_max) - exact_cumulants(tau, n_max)


# ---------------------------------------------------------------- composition
def compose_uniform(w_outer, off_outer, w_inner, off_inner):
    """Compose when every sub-agent is the SAME stencil v relative to its own centre.

    Support adds, weights multiply  ->  discrete convolution.
    Returns (weights, offsets) on the merged offset lattice.
    """
    off_outer = np.asarray(off_outer); off_inner = np.asarray(off_inner)
    lo = off_outer.min() + off_inner.min()
    hi = off_outer.max() + off_inner.max()
    # fast path: both supports are contiguous integer ranges -> plain convolution
    contig = (off_outer.size == off_outer.max() - off_outer.min() + 1
              and off_inner.size == off_inner.max() - off_inner.min() + 1
              and np.all(np.diff(off_outer) == 1) and np.all(np.diff(off_inner) == 1))
    if contig:
        return np.convolve(w_outer, w_inner), np.arange(lo, hi + 1)
    off = np.arange(lo, hi + 1)
    w = np.zeros(off.size)
    for wi, oi in zip(w_outer, off_outer):
        for vj, oj in zip(w_inner, off_inner):
            w[oi + oj - lo] += wi * vj
    return w, off


def compose_general(w_outer, off_outer, subs):
    """Compose with sub-agents that MAY DIFFER per neighbour.

    subs[i] = (v_i weights, e_i offsets) for the agent sitting at off_outer[i],
    expressed relative to ITS OWN centre.
    """
    pieces = [(off_outer[i] + subs[i][1], w_outer[i] * subs[i][0])
              for i in range(len(w_outer))]
    lo = min(p[0].min() for p in pieces)
    hi = max(p[0].max() for p in pieces)
    off = np.arange(lo, hi + 1)
    w = np.zeros(off.size)
    for o, ww in pieces:
        np.add.at(w, o - lo, ww)
    return w, off


def compose_power(w, off, L):
    """L-fold self-composition (L-fold convolution power)."""
    ww, oo = w.copy(), off.copy()
    for _ in range(L - 1):
        ww, oo = compose_uniform(ww, oo, w, off)
    return ww, oo


def compose_power_fast(w, off, L):
    """L-fold convolution power by binary exponentiation (FFT-free, exact-ish)."""
    assert L >= 1
    res_w, res_o = np.array([1.0]), np.array([0])
    base_w, base_o = w.copy(), off.copy()
    n = L
    while n:
        if n & 1:
            res_w, res_o = compose_uniform(res_w, res_o, base_w, base_o)
        n >>= 1
        if n:
            base_w, base_o = compose_uniform(base_w, base_o, base_w, base_o)
    return res_w, res_o


# ---------------------------------------------------------------- symbols
def symbol(w, off, theta):
    """Von Neumann symbol g(theta) = sum_k w_k exp(i k theta) (k in cells)."""
    theta = np.atleast_1d(np.asarray(theta, float))
    return (w[None, :] * np.exp(1j * np.outer(theta, off))).sum(axis=1)


def mgf(w, off, s, dx=DX):
    """Moment generating function W(s) = sum_i w_i exp(s xi_i), xi in physical units."""
    s = np.atleast_1d(np.asarray(s, float))
    return (w[None, :] * np.exp(np.outer(s, np.asarray(off, float) * dx))).sum(axis=1)


# ---------------------------------------------------------------- exact solution
def exact_kernel_on_lattice(tau, off, dx=DX, alpha=ALPHA, c=C):
    """Point-sampled (not cell-averaged) exact Gaussian propagator, renormalised."""
    xi = np.asarray(off, float) * dx
    g = np.exp(-(xi + c * tau) ** 2 / (4 * alpha * tau)) / np.sqrt(4 * np.pi * alpha * tau)
    return g * dx


def l1(w):
    return float(np.abs(w).sum())
