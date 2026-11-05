"""De-risk check: is p* the spectral gap, or something else?

Coarse-grain the periodic FTCS heat stencil by decimation factor M.
Fine symbol at mode k:  g_k = 1 - 2r(1 - cos(2 pi k/N)).
Coarse mode q aliases the M fine modes k = q + j*Nc, j = 0..M-1.

A coarse law of memory depth p (lags 1..p) is EXACT iff, at every coarse mode q,
its characteristic polynomial annihilates all M aliased symbols. A degree-M monic
polynomial with exactly those roots always exists, so:

    exact depth = M - 1 extra lags, ALWAYS, purely algebraically (Cayley-Hamilton).

That part is classical and carries no information about the spectrum. The live
question is the POSITIVITY-constrained depth: how deep must the memory be before
the lag coefficients are realisable as NON-NEGATIVE compact spatial weights?
If that exceeds M-1, the gap is the actual object of study.
"""
import numpy as np
from scipy.optimize import nnls
np.set_printoptions(precision=4, suppress=True)

N, r = 120, 0.45

def aliased_symbols(M):
    """For each coarse mode q, the M fine symbols that alias onto it."""
    Nc = N // M
    g = lambda k: 1 - 2*r*(1 - np.cos(2*np.pi*k/N))
    return np.array([[g(q + j*Nc) for j in range(M)] for q in range(Nc)])   # (Nc, M)

def exact_lag_coeffs(M):
    """Coefficients c_1..c_M of the recurrence whose roots are the aliased symbols.
    u^{n+1} = c_1 u^n + ... + c_M u^{n-M+1}, per coarse mode."""
    G = aliased_symbols(M)
    out = []
    for row in G:
        poly = np.poly(row)             # monic, roots = aliased symbols
        out.append(-poly[1:])           # u^{n+1} = -(a1 u^n + a2 u^{n-1} + ...)
    return np.array(out)                # (Nc, M)

def min_positive_depth(M, s_max=4, p_max=10, tol=1e-10):
    """Smallest p admitting NON-NEGATIVE compact spatial lag operators of half-width <= s."""
    Nc = N // M
    G = aliased_symbols(M)
    qs = 2*np.pi*np.arange(Nc)/Nc
    for p in range(1, p_max+1):
        for s in range(0, s_max+1):
            width = 2*s+1
            # unknowns: b_j[-s..s] >= 0 for j=1..p.  Symbol B_j(q) = sum_l b_j[l] e^{-i q l}
            # exactness: for each aliased symbol g, sum_j B_j(q) g^{p-j} = g^p
            rows, rhs = [], []
            for qi, q in enumerate(qs):
                basis = np.array([np.cos(q*np.arange(-s, s+1)),
                                  np.sin(q*np.arange(-s, s+1))])
                for g in G[qi]:
                    for part in range(2):                     # real, imag
                        rows.append(np.concatenate([basis[part]*g**(p-j) for j in range(1, p+1)]))
                        rhs.append(g**p if part == 0 else 0.0)
            A, b = np.array(rows), np.array(rhs)
            x, res = nnls(A, b, maxiter=20000)
            if res < tol*np.linalg.norm(b):
                return p, s, res
    return None, None, None

print(f"periodic FTCS, N={N}, r={r}\n")
print(f"{'M':>3} {'exact depth (=M-1?)':>20} {'min POSITIVE depth p*':>24} {'half-width s':>13}")
for M in [2, 3, 4, 5, 6]:
    ex = exact_lag_coeffs(M)
    # sanity: does the exact recurrence really reproduce the dynamics?
    G = aliased_symbols(M)
    err = max(abs(sum(ex[qi][j]*g**(M-1-j) for j in range(M)) - g**M)
              for qi in range(len(G)) for g in G[qi])
    p_star, s_star, res = min_positive_depth(M)
    tag = f"{p_star} (s={s_star})" if p_star else "> 10"
    print(f"{M:>3} {M-1:>13} (resid {err:.0e}) {str(tag):>24} {str(s_star):>13}")

print("\n=> exact depth is M-1 by Cayley-Hamilton: purely algebraic, spectrum-independent.")
print("   The live quantity is the gap between that and the POSITIVE depth.")
