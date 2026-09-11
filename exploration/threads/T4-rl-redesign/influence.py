"""Is the optimal allocation of effort a LOCAL formula, or does it need the target?

This is the decisive test for the query-driven framing (Ali's state includes z*).

Setup: predict u(x*,t*). At each grid point and time level choose a cheap stencil
(3-point FTCS, positive) or an expensive one (5-point, moments 0..4, positive).
Spend a budget. Where?

Linearised, the error at the query is

    E  =  sum_{j,n}  G(j,n) * tau_s(j,n)

with tau_s the LOCAL truncation of the chosen scheme and G the INFLUENCE FUNCTION
-- how much a perturbation at (j,n) moves the answer at (x*,t*). G is the adjoint
solution: one backward sweep of A^T from a delta at x*.

Three policies, all greedy on the same budget:
  uniform        ignores everything
  local          greedy on  d(tau)/d(cost)          -- the obvious error indicator
  target-aware   greedy on  G * d(tau)/d(cost)      -- needs to know the query

The gap between `local` and `target-aware` IS the value of conditioning on z*.
If it is ~0, the query-driven framing buys nothing and a formula suffices.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from core.pde import Problem
from core import stencil as st

p = Problem(nx=161)
NT = 120                                   # time levels marched
X, DX, DT = p.x, p.dx, p.dt

w_cheap = st.solve_positive(*st.rows_taylor(np.array([-DX, 0, DX]), np.full(3, -DT), p))
w_exp = st.solve_positive(*st.rows_moment(np.arange(-2, 3) * DX, np.full(5, -DT), 4, p))
assert w_cheap is not None and w_exp is not None
COST_CHEAP, COST_EXP = 3, 5
print(f"cheap  m=1 : w={np.round(w_cheap,5)}  ||w||_1={np.abs(w_cheap).sum():.6f}")
print(f"expens m=2 : w={np.round(w_exp,5)}  ||w||_1={np.abs(w_exp).sum():.6f}")

def apply_stencil(u, w, m):
    out = np.zeros_like(u)
    out[m:len(u)-m] = np.convolve(u, w[::-1], "same")[m:len(u)-m]
    for j in list(range(1, m)) + list(range(len(u)-m, len(u)-1)):
        out[j] = w_cheap @ u[j-1:j+2]          # fall back near walls
    return out

# ---- local truncation of each scheme, on exact data -------------------------
tau_c = np.zeros((NT, p.nx)); tau_e = np.zeros((NT, p.nx))
for n in range(NT):
    u0, u1 = p.u_true(X, n*DT), p.u_true(X, (n+1)*DT)
    tau_c[n] = apply_stencil(u0, w_cheap, 1) - u1
    tau_e[n] = apply_stencil(u0, w_exp, 2) - u1
    tau_c[n, [0, -1]] = tau_e[n, [0, -1]] = 0.0

# ---- influence function: adjoint sweep backward from the query --------------
def influence(jstar):
    G = np.zeros((NT, p.nx))
    g = np.zeros(p.nx); g[jstar] = 1.0
    for n in range(NT-1, -1, -1):
        G[n] = g
        g = np.convolve(g, w_cheap, "same")     # A^T for a symmetric-support stencil
        g[[0, -1]] = 0.0
    return G

def allocate(score, budget_extra):
    """Greedy: upgrade the cells with the best score until the extra budget is gone."""
    flat = np.argsort(-score.ravel())
    k = min(budget_extra // (COST_EXP - COST_CHEAP), len(flat))
    mask = np.zeros(score.size, bool); mask[flat[:k]] = True
    return mask.reshape(score.shape)

def run(mask):
    """March with the per-cell scheme choice; return error at the query and cost."""
    u = p.u_true(X, 0.0).copy()
    cost = 0
    for n in range(NT):
        uc, ue = apply_stencil(u, w_cheap, 1), apply_stencil(u, w_exp, 2)
        u = np.where(mask[n], ue, uc)
        u[[0, -1]] = 0.0
        cost += int(mask[n].sum()) * COST_EXP + int((~mask[n]).sum()) * COST_CHEAP
    return u, cost

print(f"\ngrid {p.nx} x {NT} levels, t* = {NT*DT:.4g}, sigma = {np.sqrt(2*p.alpha*NT*DT)/DX:.1f} cells")
print(f"{'x*':>6} {'budget':>8} | {'uniform':>10} {'local':>10} {'target-aware':>13} | {'gain vs local':>14}")

gains = []
for xs in [0.25, 0.5, 0.75]:
    jstar = int(round(xs / DX))
    G = influence(jstar)
    dtau = np.abs(tau_c) - np.abs(tau_e)            # truncation reduction from upgrading
    base = p.nx * NT * COST_CHEAP
    for frac in [0.05, 0.15, 0.35]:
        extra = int(frac * p.nx * NT * (COST_EXP - COST_CHEAP))
        res = {}
        for name, score in [("uniform", np.random.default_rng(0).random(dtau.shape)),
                            ("local", dtau / (COST_EXP - COST_CHEAP)),
                            ("target", G * dtau / (COST_EXP - COST_CHEAP))]:
            u, cost = run(allocate(score, extra))
            res[name] = abs(u[jstar] - p.u_true(xs, NT*DT))
        g = res["local"] / res["target"]
        gains.append(g)
        print(f"{xs:>6.2f} {frac:>7.0%} | {res['uniform']:>10.3e} {res['local']:>10.3e}"
              f" {res['target']:>13.3e} | {g:>13.2f}x")
print(f"\nmedian gain of target-awareness over the local indicator: {np.median(gains):.2f}x")
