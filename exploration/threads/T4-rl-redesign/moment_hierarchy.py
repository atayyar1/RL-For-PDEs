"""Are the propagator moments derivable from the PDE alone?  (If not, the method
is 'sample the known Green's function' and is not general.)

For u_t = alpha u_xx - c u_x the Green's function G(z,tau) satisfies the same PDE,
so its raw moments M_q(tau) = int G z^q dz obey a CLOSED, TRIANGULAR ODE hierarchy
obtained by integrating by parts:

    dM_q/dtau = alpha q(q-1) M_{q-2}  +  c q M_{q-1},      M_q(0) = delta_{q0}

No fundamental solution is used -- only the PDE coefficients. The hierarchy is
triangular, so it integrates exactly in closed form, and it exists for ANY linear
PDE (and, with state-dependent coefficients, for nonlinear ones locally).
This is what makes the moment-row construction general.
"""
import numpy as np
from scipy.integrate import solve_ivp
import scheme_mdp as S

P = 8

def moments_from_pde(tau, alpha=S.ALPHA, c=S.C):
    """Integrate the hierarchy. Uses ONLY the PDE coefficients."""
    def rhs(t, M):
        d = np.zeros_like(M)
        for q in range(1, P + 1):
            d[q] = c * q * M[q - 1] + (alpha * q * (q - 1) * M[q - 2] if q >= 2 else 0.0)
        return d
    M0 = np.zeros(P + 1); M0[0] = 1.0
    sol = solve_ivp(rhs, (0.0, tau), M0, rtol=1e-12, atol=1e-14, dense_output=True)
    return sol.y[:, -1]

def moments_from_gaussian(tau, alpha=S.ALPHA, c=S.C):
    """Closed form, using the known Green's function (the thing we must NOT need)."""
    mu, var = c * tau, 2 * alpha * tau
    m = [1.0, mu]
    for q in range(2, P + 1):
        m.append(mu * m[q - 1] + (q - 1) * var * m[q - 2])
    return np.array(m)

print("Moments of the propagator in z = x - y, at tau = 0.045")
print(f"{'q':>3} {'from PDE hierarchy':>22} {'from Gaussian form':>22} {'rel diff':>12}")
tau = 400 * S.DT
a, b = moments_from_pde(tau), moments_from_gaussian(tau)
for q in range(P + 1):
    rel = abs(a[q] - b[q]) / max(abs(b[q]), 1e-300)
    print(f"{q:>3} {a[q]:>22.12e} {b[q]:>22.12e} {rel:>12.2e}")
print(f"\nmax relative difference: {np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300)):.2e}")
print("\n=> the moment rows are computable from the PDE COEFFICIENTS ALONE.")
print("   No Green's function, no fundamental solution. The construction generalises")
print("   to any linear PDE, and locally to variable/state-dependent coefficients.")

print("\n--- and the hierarchy is what T3 should be learning ---")
print("Recovering (alpha, c) from moments alone, at small tau:")
for tau_s in [10*S.DT, 50*S.DT]:
    m = moments_from_pde(tau_s)
    c_hat = m[1] / tau_s
    a_hat = (m[2] - m[1]**2) / (2 * tau_s)
    print(f"  tau={tau_s:.3e}:  c_hat={c_hat:.6f} (true {S.C})   alpha_hat={a_hat:.6f} (true {S.ALPHA})")
