"""The 3-row system is only enough for a SMALL step. For a giant step you need
to match more moments of the propagator.

For u_t = L u with constant-coefficient L, the exact propagator over tau is
exp(tau L). Its moments against monomials are computable from the PDE ALONE by
applying the operator repeatedly to polynomials -- a finite computation, no
fundamental solution required. For u_t + c u_x = alpha u_xx the propagator is a
Gaussian in Dx with mean -c*tau and variance 2*alpha*tau, so

    M_q := sum_i w_i Dx_i^q  =  q-th moment of N(-c tau, 2 alpha tau)

Matching q = 0..p gives order p+1. Positivity is then exactly the classical
truncated MOMENT PROBLEM: a non-negative w exists iff the target moment sequence
is realisable on the given support (Hankel condition).
"""
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
import scheme_mdp as S

def gauss_moments(mu, var, p):
    """Raw moments 0..p of N(mu, var), via the recurrence m_q = mu m_{q-1} + (q-1) var m_{q-2}."""
    m = [1.0, mu]
    for q in range(2, p + 1):
        m.append(mu * m[q - 1] + (q - 1) * var * m[q - 2])
    return np.array(m[: p + 1])

def weights_moment(m_half, tau, p, positive=True):
    """Positive weights matching propagator moments 0..p on offsets -m..m."""
    idx = np.arange(-m_half, m_half + 1)
    dxi = idx * S.DX
    h = m_half * S.DX
    A = np.vstack([(dxi / h) ** q for q in range(p + 1)])
    b = gauss_moments(-S.C * tau, 2 * S.ALPHA * tau, p) / h ** np.arange(p + 1)
    n = len(idx)
    if positive:
        r = linprog(np.zeros(n), A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs")
        return r.x if r.status == 0 else None
    r = linprog(np.ones(2 * n), A_eq=np.hstack([A, -A]), b_eq=b,
                bounds=[(0, None)] * 2 * n, method="highs")
    return (r.x[:n] - r.x[n:]) if r.status == 0 else None

x_star = 0.5; jstar = int(round(x_star / S.DX))
print("ONE giant step from the exact initial condition to t*, varying moment order p.")
print("(gamma = stencil half-width in units of the kernel sigma)\n")
for J in [100, 400, 1600]:
    tau = J * S.DT
    sig = np.sqrt(2 * S.ALPHA * tau)
    u_ref = float(S.u_true(x_star, tau)[0])
    cost_fd = sum(min(S.NX - 2, 2 * (J - q) + 1) for q in range(J))
    uf = S.u_true(S.X, 0.0).copy()
    for _ in range(J):
        uf = S.advance(uf, 1, 1)
    err_fd = abs(uf[jstar] - u_ref)
    print(f"=== t*={J} steps ({tau:.4g}), sigma={sig/S.DX:.1f} cells | "
          f"FD: cost={cost_fd}, err={err_fd:.2e} ===")
    print(f"  {'gamma':>5} {'m':>4} {'cost':>6} | " + " ".join(f"p={p:<10}" for p in [2,4,6,8]))
    for g in [2, 3, 4]:
        mh = int(np.ceil(g * sig / S.DX))
        if jstar - mh < 0 or jstar + mh > S.NX - 1:
            continue
        u0 = S.u_true(S.X[jstar - mh: jstar + mh + 1], 0.0)
        cells = []
        for p in [2, 4, 6, 8]:
            w = weights_moment(mh, tau, p)
            if w is None:
                cells.append("infeasible ")
            else:
                cells.append(f"{abs(float(w @ u0) - u_ref):.3e}  ")
        print(f"  {g:>5} {mh:>4} {2*mh+1:>6} | " + " ".join(cells))
    print()
