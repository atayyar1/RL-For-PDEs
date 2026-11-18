"""
positioning_check.py -- three small facts that decide how this thread is positioned.

Not the thread's experiments. Only the checks needed so POSITIONING.md asserts
nothing it has not verified.

(1) Minimal exact memory depth of a linear projection P of a linear map A is the
    Krylov/observability index of (P, A) minus one.  p* = 0 iff rowspace(P) is
    A-invariant on the right (equivalently range(P^T) is A^T-invariant).
(2) THEREFORE the unconstrained "minimise memory" problem is DEGENERATE: every
    A-invariant subspace of dimension K attains p* = 0, fast ones included.
    Memory alone does not select the slow subspace. VAMP does.
(3) Decimation by M on a circulant FTCS ring gives p* = M-1, matching T5/F14's
    Cayley-Hamilton count, by a completely different route (rank of a Krylov stack).
"""
import os
for _v in ("OMP", "OPENBLAS", "MKL", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")
import sys
import numpy as np

sys.path.insert(0, "/Users/josephbakarji/Documents/00-projects/01-tests/RL-For-PDEs/exploration")
from core import DEFAULT

RNG = np.random.default_rng(0)
SEP = "=" * 78


def ftcs_ring(N, prob=DEFAULT):
    """Circulant FTCS matrix for u_t + c u_x = alpha u_xx on a periodic ring."""
    r, nu = prob.r, prob.nu
    row = np.zeros(N)
    row[0] = 1 - 2 * r
    row[1] = r - nu / 2          # coefficient of u_{j+1}
    row[-1] = r + nu / 2         # coefficient of u_{j-1}
    return np.array([np.roll(row, k) for k in range(N)])


def memory_depth(P, A, qmax=None, tol=1e-9):
    """Minimal q with P A^q in rowspan{P, PA, ..., PA^{q-1}}; memory depth p = q-1.

    q = 1 (p = 0) is a Markov coarse law.  Returns (p, ranks).
    """
    N = A.shape[0]
    qmax = qmax or N
    S = P.copy()
    ranks = [np.linalg.matrix_rank(S, tol=tol)]
    B = P.copy()
    for q in range(1, qmax + 1):
        B = B @ A
        S2 = np.vstack([S, B])
        rk = np.linalg.matrix_rank(S2, tol=tol)
        if rk == ranks[-1]:
            return q - 1, ranks
        ranks.append(rk)
        S = S2
    return np.inf, ranks


def ar_fit_residual(P, A, p, n_ic=None, seed=1):
    """Independent check, by REGRESSION ON DATA rather than by rank.

    Draw many random initial conditions, run each p+2 steps, and least-squares fit
    y_{n+1} = sum_{j=1..p+1} B_j y_{n+1-j} on half of them; report max abs error on
    the other half.  Overdetermined by construction (n_ic >> K(p+1)), so a small
    residual means the depth-p law really exists, not that the fit was underdetermined.
    """
    if p < 0:
        return np.nan
    rng = np.random.default_rng(seed)
    N, K, q = A.shape[0], P.shape[0], p + 1
    n_ic = n_ic or max(400, 8 * K * q)
    U0 = rng.standard_normal((N, n_ic))
    Y = []                                   # Y[n] = (n_ic, K)
    U = U0
    for _ in range(q + 1):
        Y.append((P @ U).T)
        U = A @ U
    X = np.hstack([Y[q - 1 - j] for j in range(q)])        # lags 1..q, (n_ic, Kq)
    T = Y[q]
    h = n_ic // 2
    B, *_ = np.linalg.lstsq(X[:h], T[:h], rcond=None)
    return float(np.abs(X[h:] @ B - T[h:]).max() / max(np.abs(T[h:]).max(), 1e-300))


print(SEP + "\n(1)+(3)  minimal exact memory depth = Krylov index - 1\n" + SEP)
N = 60
A = ftcs_ring(N)
print(f"   ring N={N}, r={DEFAULT.r}, nu={DEFAULT.nu}")
print(f"\n   {'projection P':<42} {'K':>4} {'p* (rank)':>10} {'AR resid at p*':>15} {'at p*-1':>10}")
for M in (2, 3, 4, 5, 6):
    K = N // M
    P = np.zeros((K, N))
    for i in range(K):
        P[i, i * M] = 1.0                                  # pure decimation
    p, _ = memory_depth(P, A)
    r1 = ar_fit_residual(P, A, p)
    r0 = ar_fit_residual(P, A, p - 1)
    print(f"   {'decimation by M=%d' % M:<42} {K:>4} {p:>10} {r1:>15.2e} {r0:>10.2e}")

for M in (2, 3, 4):
    K = N // M
    P = np.zeros((K, N))
    for i in range(K):
        P[i, i * M:(i + 1) * M] = 1.0 / M                  # block average
    p, _ = memory_depth(P, A)
    print(f"   {'block average, M=%d' % M:<42} {K:>4} {p:>10} {ar_fit_residual(P, A, p):>15.2e}")

P = RNG.standard_normal((20, N))
p, _ = memory_depth(P, A)
print(f"   {'random dense rank-20':<42} {20:>4} {p:>10} {ar_fit_residual(P, A, p):>15.2e}")
print(f"\n   trivial upper bound: rank([P;PA;...]) saturates at N, so p* <= ceil(N/K)-1")
print(f"   N/K for the decimations above = M, so DECIMATION SITS EXACTLY ON THE")
print(f"   TRIVIAL CEILING -- 'M-1 lags' is the worst case, not a special structure.")


print("\n" + SEP + "\n(2)  the unconstrained problem is DEGENERATE\n" + SEP)
lam, V = np.linalg.eig(A)          # circulant: normal, left evecs = right evecs
order = np.argsort(-np.abs(lam))   # slow (|lam| near 1) first
K = 20


def real_span(idx):
    """Real K-dim invariant subspace from a conjugate-closed index set."""
    B = V[:, idx]
    R = np.hstack([B.real, B.imag])
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    return U[:, s > 1e-10 * s[0]].T   # rows span the subspace


def conj_closed(idx_sorted, K):
    """Take modes off a sorted list until a conjugate-closed real span of dim K."""
    take = []
    for i in idx_sorted:
        if len(real_span(take + [i])) > K:
            break
        take.append(i)
        if len(real_span(take)) == K:
            break
    return take


def vamp2(Prows, A, lam):
    """VAMP-2 score of a subspace for a normal A: sum of squared eigenvalues of the
    restriction of A to the subspace (the retained relaxation content)."""
    Q = np.linalg.qr(Prows.T)[0]
    Ahat = Q.T @ A @ Q
    return float(np.sum(np.abs(np.linalg.eigvals(Ahat)) ** 2))


slow = conj_closed(list(order), K)
fast = conj_closed(list(order[::-1]), K)
mid = conj_closed(list(order[len(order) // 2:]) + list(order[:len(order) // 2]), K)

print(f"   {'subspace (all A-invariant, dim %d)' % K:<42} {'p*':>4} {'VAMP-2':>12} {'slowest |lam|':>14}")
for name, idx in (("K SLOWEST eigen-directions", slow),
                  ("K FASTEST eigen-directions", fast),
                  ("K middle eigen-directions", mid)):
    Prows = real_span(idx)
    p, _ = memory_depth(Prows, A)
    print(f"   {name:<42} {p:>4} {vamp2(Prows, A, lam):>12.4f} {np.abs(lam[idx]).max():>14.6f}")

Prows = real_span(slow)[:K]
Pdec = np.zeros((K, N))
for i in range(K):
    Pdec[i, i * 3] = 1.0
print(f"   {'decimation by M=3 (not invariant)':<42} "
      f"{memory_depth(Pdec, A)[0]:>4} {vamp2(Pdec, A, lam):>12.4f}")

print("""
   READ: every A-invariant subspace has p* = 0, including the FASTEST one.
   Memory depth alone cannot distinguish them -- it is a rank condition, blind to
   the magnitude of the eigenvalues.  What separates them is the VAMP-2 column.
   So "minimise memory" is NOT "maximise VAMP"; it is the strictly weaker, and
   massively degenerate, condition that VAMP's optimum happens to satisfy.
""")
