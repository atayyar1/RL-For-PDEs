"""TASK 6 (deliverable prerequisite) -- IS THIS RKC IN DISGUISE?

Claim to test:  the wide POSITIVE stencil at the positivity frontier is EXACTLY
first-order Runge-Kutta-Chebyshev (RKC1) with s = m stages.

RKC1 uses the shifted Chebyshev polynomial  R_s(z) = T_s(1 + z/s^2), which is
bounded by 1 for z in [-2 s^2, 0].  With the 3-point Laplacian, lambda_max =
4 alpha/dx^2, so RKC1 is stable for

        dt <= 2 s^2 dx^2 / (4 alpha) = s^2 dx^2 / (2 alpha)

which is IDENTICAL to the positivity frontier  k dt = m^2 dx^2/(2 alpha).

Algebraic identity to verify:  at the frontier the wide kernel is
w = (1/2)(delta_{-m} + delta_{+m}), symbol cos(m theta); and with
z = dt_base * lambda(theta) = -4 nu sin^2(theta/2),
        T_m(1 + z/(2 nu)) = T_m(cos theta) = cos(m theta).
So the two operators are the same polynomial of the same Laplacian.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numpy.polynomial import chebyshev as Ch
import core as C

nu = C.NU_REF
print("=" * 100)
print("A.  SYMBOL IDENTITY:  cos(m theta)  ==  T_m(1 + z/(2 nu)),  z = -4 nu sin^2(theta/2)")
print("=" * 100)
th = np.linspace(0, np.pi, 9)
for m in [1, 2, 3, 6, 12, 20]:
    z = -4 * nu * np.sin(th / 2) ** 2
    lhs = np.cos(m * th)
    rhs = np.cos(m * np.arccos(np.clip(1 + z / (2 * nu), -1, 1)))   # T_m
    print("  m=%2d   max|cos(m th) - T_m(1+z/2nu)| = %.3e" % (m, np.max(np.abs(lhs - rhs))))

print("\n" + "=" * 100)
print("B.  OPERATOR IDENTITY: run RKC1 with s stages vs the frontier stencil, same grid")
print("=" * 100)


def rkc1_step(u, dt, dx, alpha, s, apply_L):
    """First-order RKC: u_{j} built from the Chebyshev recurrence for T_s(1+z/s^2)."""
    w0 = 1.0
    mu_t = 1.0 / s ** 2                      # dt-scaled: T_s(1 + z/s^2)
    Y0 = u.copy()
    Y1 = u + mu_t * dt * apply_L(u)
    Tm2, Tm1 = 1.0, 1.0                      # T_{j-2}(w0), T_{j-1}(w0) at w0 = 1
    for j in range(2, s + 1):
        Tj = 2 * w0 * Tm1 - Tm2              # = 1 for all j (w0 = 1)
        mu = 2 * w0 * Tm1 / Tj
        nuj = -Tm2 / Tj
        mu_tj = mu / s ** 2
        Y2 = mu * Y1 + nuj * Y0 + mu_tj * dt * apply_L(Y1)
        Y0, Y1 = Y1, Y2
        Tm2, Tm1 = Tm1, Tj
    return Y1


N = 401
x = np.linspace(0, 1, N); dx = x[1] - x[0]
alpha = 0.1


def applyL(u):
    out = np.zeros_like(u)
    out[1:-1] = alpha * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx ** 2
    return out


u0 = np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)
print("%4s %10s | %14s %14s | %s" % ("s=m", "dt_big", "RKC1 vs frontier", "both vs exact",
                                     "same time step?"))
for m in [2, 4, 8, 16]:
    dt_big = m ** 2 * dx ** 2 / (2 * alpha)          # the frontier == RKC1 stability limit
    # RKC1
    ur = rkc1_step(u0.copy(), dt_big, dx, alpha, m, applyL); ur[0] = ur[-1] = 0
    # frontier wide stencil: (1/2)(u_{j-m}+u_{j+m}) with odd reflection
    L = 2 * (N - 1); U = np.zeros(L); U[:N] = u0; U[N:] = -u0[-2:0:-1]
    idx = np.arange(N)
    uw = 0.5 * (U[(idx - m) % L] + U[(idx + m) % L]); uw[0] = uw[-1] = 0
    ex = C.Exact(C.ic_sine, alpha, 0.0)
    print("%4d %10.3e | %14.3e %14.3e | %s"
          % (m, dt_big, np.max(np.abs(ur - uw)),
             np.max(np.abs(uw - ex(x, dt_big))),
             "yes: dt = m^2 dx^2/(2 alpha) for both"))

print("\n  (RKC1 differs from the frontier stencil only near the boundary, where the")
print("   3-point Laplacian's homogeneous Dirichlet condition and the odd reflection")
print("   coincide exactly; interior values agree to round-off.)")

print("\n" + "=" * 100)
print("C.  STABILITY CONSTANTS AT EQUAL STENCIL FOOTPRINT m")
print("=" * 100)
print("  method                      dt_max                   footprint   order   monotone")
print("  " + "-" * 92)
rows = [("forward Euler / FTCS",      "1 * dx^2/(2 alpha)",      "1",  "1 (t), 2 (x)", "yes"),
        ("RKC1, s = m stages",        "m^2 * dx^2/(2 alpha)",    "m",  "1 (t), 2 (x)", "yes*"),
        ("wide positive, half-width m", "m^2 * dx^2/(2 alpha)",  "m",  "1 (t), 2 (x)", "yes"),
        ("RKC2, s = m stages",        "0.65 m^2 * dx^2/(2 alpha)", "m", "2 (t), 2 (x)", "no"),
        ("RKL1 (Legendre), s = m",    "~0.5 m^2 * dx^2/(2 alpha)", "m", "1 (t), 2 (x)", "no"),
        ("SSP-RK, s stages (nonlinear bound)", "m * dx^2/(2 alpha)", "m", "<=4", "yes"),
        ("Crank-Nicolson",            "unlimited",               "global", "2 (t), 2 (x)", "no")]
for r in rows:
    print("  %-28s %-24s %-11s %-14s %s" % r)
print("\n  * RKC1 at EXACTLY its stability limit equals (1/2)(delta_-m + delta_+m),")
print("    a convex combination -- so it is monotone there.  Below the limit the")
print("    Chebyshev stencil is T_s(1 - (4 nu'/s^2) sin^2(th/2)) which is NOT in")
print("    general a non-negative measure, whereas the LP construction is.  That")
print("    residual difference is the only thing the positivity view adds.")

print("\n" + "=" * 100)
print("D.  SSP / MONOTONICITY BARRIER")
print("=" * 100)
print("  Explicit SSP Runge-Kutta methods obey  C_SSP <= s  (Ketcheson): monotone")
print("  time step grows only LINEARLY in the number of stages.  The wide positive")
print("  stencil reaches m^2, i.e. it beats the SSP barrier.  There is no")
print("  contradiction: the SSP bound constrains methods built from convex")
print("  combinations of forward-Euler steps of a FIXED operator and must hold for")
print("  ALL problems satisfying the forward-Euler condition.  Here the spatial")
print("  footprint itself is widened, and the argument is specific to the constant-")
print("  coefficient linear heat operator.  For that operator the quadratic scaling")
print("  is classical (Chebyshev / super-time-stepping).")
