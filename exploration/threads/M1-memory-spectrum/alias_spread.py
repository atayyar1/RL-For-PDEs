"""Memory-free coarse-graining requires the fine operator to be BLIND to unresolved modes.

T5's finding: the frontier scheme 1/2(delta_-m + delta_+m) has symbol cos(m theta).
Coarse-grained by M = m, every aliased mode gets the SAME eigenvalue, so the alias
subspace is an eigenspace, the minimal polynomial has degree 1, and the coarse law
is exactly Markov -- ZERO memory. And that same degeneracy is why it is useless.

The general statement that should follow, and which is testable:

    exact memory depth  =  (number of DISTINCT eigenvalues on the alias class) - 1

so zero memory <=> the symbol is constant on alias classes <=> the fine operator
cannot tell apart modes the coarse grid cannot represent. An ACCURATE operator
evolves those modes differently, so it MUST pay memory.

    Memory is the price of the fine operator being able to distinguish
    what the coarse grid cannot.
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
N = 120

def alias_eigs(symbol, M, q):
    Nc = N // M
    return np.array([symbol(2*np.pi*(q + j*Nc)/N) for j in range(M)])

def exact_depth(symbol, M, tol=1e-12):
    """Max over coarse modes of (distinct aliased eigenvalues - 1)."""
    Nc = N // M
    d = 0
    for q in range(Nc):
        g = alias_eigs(symbol, M, q)
        # count distinct up to tol
        u = []
        for x in g:
            if not any(abs(x - y) < tol for y in u): u.append(x)
        d = max(d, len(u) - 1)
    return d

def alias_spread(symbol, M):
    Nc = N // M
    return max(np.ptp(alias_eigs(symbol, M, q)) for q in range(Nc))

r = 0.45
ftcs   = lambda th: 1 - 2*r*(1 - np.cos(th))
print("=== 1. T5's claim: the frontier scheme has ZERO memory under M = m ===")
print(f"{'m = M':>6} {'operator':>28} {'alias spread':>14} {'exact memory depth':>20}")
for m in [3, 4, 6]:
    frontier = lambda th, m=m: np.cos(m*th)
    print(f"{m:>6} {'cos(m theta)  [frontier]':>28} {alias_spread(frontier,m):>14.2e}"
          f" {exact_depth(frontier,m):>20}")
    print(f"{'':>6} {'FTCS r=0.45':>28} {alias_spread(ftcs,m):>14.2e} {exact_depth(ftcs,m):>20}")
print("\n=> confirmed: the frontier operator has zero memory at M = m; FTCS needs M-1.")

print("\n=== 2. The general law: depth = distinct aliased eigenvalues - 1 ===")
print("    Test on a family interpolating between the two, u -> (1-a) FTCS + a frontier")
print(f"\n{'a':>6} {'alias spread (M=3)':>20} {'exact depth':>13} {'max |g| (stability)':>21}")
for a in [0.0, 0.25, 0.5, 0.75, 0.99, 1.0]:
    mix = lambda th, a=a: (1-a)*ftcs(th) + a*np.cos(3*th)
    th = np.linspace(-np.pi, np.pi, 2001)
    print(f"{a:>6.2f} {alias_spread(mix,3):>20.4f} {exact_depth(mix,3):>13} "
          f"{np.abs(mix(th)).max():>21.4f}")
print("\n=> depth is a RANK statistic (distinct eigenvalues), so it jumps rather than")
print("   scaling -- it is 2 for every non-degenerate mix and 0 only at exact degeneracy.")
print("   The spread is the continuous quantity; the depth is its rank shadow.")
print("   So the right question for p* is not the exact depth but the POSITIVE depth,")
print("   which is continuous in the spread. That is what Phase 2 should measure.")
