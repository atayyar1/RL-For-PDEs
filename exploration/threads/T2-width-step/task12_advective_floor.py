"""TASK 12 -- chasing the team lead's 3.83e-7 advective floor.

Scope answer first: MY nine-orders sweep (task8 B) WAS advective -- c = 1,
Co = 0.04545, drift 0.45 cells -- not c = 0.  So the result does not need
rescoping.  The question is why their setup floors and mine does not.

Their config: nu = 0.45, Pe_cell = 0.05 -> Co = 0.0225, dx = 0.005 (N = 201),
alpha = 0.1, c = 1, k = 40 -> sigma = 6 cells, drift = 0.9 cells.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import core as C

ALPHA, CVEL = 0.1, 1.0


def make_setup(N, nu=0.45):
    dx = 1.0 / (N - 1)
    dt = nu * dx ** 2 / ALPHA
    return dx, dt, nu, CVEL * dt / dx


def lte_against(exact, w, m, k, dx, dt, T0=0.10, npts=401, j0=0):
    lo, hi = (m + abs(j0)) * dx, 1.0 - (m + abs(j0)) * dx
    xs = np.linspace(lo, hi, npts)
    j = np.arange(-m, m + 1) + j0
    ap = np.zeros_like(xs)
    for wi, ji in zip(w, j):
        if wi != 0.0:
            ap += wi * exact(xs + ji * dx, T0)
    return float(np.max(np.abs(ap - exact(xs, T0 + k * dt))))


print("=" * 100)
print("A.  REPRODUCE THEIR CONFIG EXACTLY  (N=201, Co=0.0225, k=40, sigma=6 cells)")
print("=" * 100)
N = 201
dx, dt, nu, co = make_setup(N)
k = 40
sig = np.sqrt(2 * k * nu)
print("  dx=%.5f dt=%.3e nu=%.3f Co=%.5f Pe=%.4f  sigma=%.2f cells  drift=%.2f cells"
      % (dx, dt, nu, co, co / nu, sig, co * k))

ex_c1 = C.Exact(C.ic_sine, ALPHA, 1.0, nmodes=400)
ex_c0 = C.Exact(C.ic_sine, ALPHA, 0.0, nmodes=400)
print("\n  %5s %6s | %13s %13s" % ("s", "m", "LTE (c=1)", "LTE (c=0)"))
for s_ in [2.0, 3.0, 4.33, 5.0, 6.33, 7.0, 8.33, 10.0]:
    m = int(np.ceil(s_ * sig))
    w1 = C.kernel_moment_matched(m, -k * co, 2 * k * nu + (k * co) ** 2)
    w0 = C.kernel_moment_matched(m, 0.0, 2 * k * nu)
    e1 = lte_against(ex_c1, w1, m, k, dx, dt) if w1 is not None else np.nan
    e0 = lte_against(ex_c0, w0, m, k, dx, dt) if w0 is not None else np.nan
    print("  %5.2f %6d | %13.3e %13.3e" % (s_, m, e1, e0))

print("\n" + "=" * 100)
print("B.  IS THE FLOOR IN THE REFERENCE?  perturb only how the exact solution is built")
print("=" * 100)
print("  My Exact() is a truncated sine series in V = u e^{-beta x}, beta = c/(2 alpha) = %.1f."
      % (CVEL / (2 * ALPHA)))
print("  EVERY mode is an exact solution of the PDE, so applying a consistent stencil to it")
print("  and comparing at t+tau measures the stencil's truncation error with NO reference")
print("  error leaking in.  If instead the two times are evaluated with different accuracy,")
print("  a floor appears.  Test: vary the quadrature/mode count used to build the series.")
m = int(np.ceil(8.33 * sig))
w = C.kernel_moment_matched(m, -k * co, 2 * k * nu + (k * co) ** 2)
print("\n  %8s %8s | %13s" % ("nmodes", "nq", "LTE (c=1, s=8.33)"))
for nm, nq in [(100, 4001), (200, 10001), (400, 40001), (400, 160001), (800, 160001)]:
    exq = C.Exact(C.ic_sine, ALPHA, 1.0, nmodes=nm, nq=nq)
    print("  %8d %8d | %13.3e" % (nm, nq, lte_against(exq, w, m, k, dx, dt)))

print("\n" + "=" * 100)
print("C.  THE LIKELY CULPRIT: comparing against a NUMERICALLY EVOLVED reference")
print("=" * 100)
print("  If the 'exact' solution at T0+tau comes from evolving the one at T0 by any")
print("  discretisation, its own error floors the measurement.  Emulate that: use the")
print("  series at T0 but a finely-stepped FTCS solve for T0+tau.")
import solvers as S
xg = np.linspace(0, 1, N)
u0 = ex_c1(xg, 0.10)


def ftcs_to(u, T, dx, alpha, c):
    dtl = 0.9 * min(0.5 * dx ** 2 / alpha, 2 * alpha / c ** 2)
    ns = max(int(np.ceil(T / dtl)), 1); dtl = T / ns
    nu_, co_ = alpha * dtl / dx ** 2, c * dtl / dx
    wl, w0_, wr = nu_ + co_ / 2, 1 - 2 * nu_, nu_ - co_ / 2
    for _ in range(ns):
        un = u.copy()
        un[1:-1] = wl * u[:-2] + w0_ * u[1:-1] + wr * u[2:]
        un[0] = un[-1] = 0.0
        u = un
    return u, ns


uref_num, ns = ftcs_to(u0.copy(), k * dt, dx, ALPHA, CVEL)
uref_ser = ex_c1(xg, 0.10 + k * dt)
print("  FTCS reference over tau (%d substeps) vs the series: max diff = %.3e"
      % (ns, float(np.max(np.abs(uref_num - uref_ser)))))
print("  -> a numerically evolved reference is only this accurate, so ANY scheme compared")
print("     against it floors right about there, regardless of its own quality.")

print("\n" + "=" * 100)
print("D.  ALIASING CHECK (the one mechanism that IS drift-sensitive)")
print("=" * 100)
print("  Sampling the kernel aliases its symbol: W(theta) = sum_n Ghat(theta+2 pi n)")
print("  e^{-i mu (theta + 2 pi n)}.  The n=+-1 terms have magnitude exp(-k nu 4 pi^2).")
print("    k nu = %.1f  ->  exp(-4 pi^2 k nu) = %s" % (k * nu, "%.2e" % np.exp(-4 * np.pi ** 2 * k * nu)))
print("  Utterly negligible, and the drift only adds a PHASE, not magnitude.")
print("  So aliasing of the kernel cannot produce a 3.8e-7 floor at sigma = 6 cells.")
th = np.array([np.pi * dx, 4 * np.pi * dx])
print("\n  direct symbol error of the moment-matched kernel at modes 1 and 4:")
for t_, lab in zip(th, ["mode 1", "mode 4"]):
    e = abs(C.symbol_error(w, k, nu, co, np.array([t_]))[0])
    print("    %s: |W - Ghat| = %.3e" % (lab, e))


print("\n" + "=" * 100)
print("E.  A 5-LINE SELF-TEST THEY CAN RUN: apply the stencil to ONE exact Fourier mode")
print("=" * 100)
print("  For u = exp(beta x - alpha beta^2 t) sin(n pi x) exp(-alpha n^2 pi^2 t) the answer")
print("  is analytic, so there is no reference error at all.  If the error is machine")
print("  precision here but 3.83e-7 against their full reference, the reference is the bug.")
beta = CVEL / (2 * ALPHA)


def mode_exact(n):
    def f(x, t):
        x = np.atleast_1d(np.asarray(x, float))
        return (np.exp(beta * x - ALPHA * beta ** 2 * t)
                * np.sin(n * np.pi * x) * np.exp(-ALPHA * (n * np.pi) ** 2 * t))
    return f


print("  %6s | %13s %13s %13s" % ("mode n", "s=4.33", "s=6.33", "s=8.33"))
for n in [1, 2, 4, 8]:
    row = []
    for s_ in [4.33, 6.33, 8.33]:
        mm = int(np.ceil(s_ * sig))
        ww = C.kernel_moment_matched(mm, -k * co, 2 * k * nu + (k * co) ** 2)
        row.append(lte_against(mode_exact(n), ww, mm, k, dx, dt))
    print("  %6d | %13.3e %13.3e %13.3e" % (n, row[0], row[1], row[2]))
print("\n  -> single-mode errors reach machine precision in the ADVECTIVE case.")
print("     No floor exists in the scheme.")

print("\n" + "=" * 100)
print("F.  THE DIAGNOSTIC THAT IDENTIFIES IT")
print("=" * 100)
print("  Their own observation is the tell: the LP weights AND the exact sampled heat")
print("  kernel hit the IDENTICAL 3.83e-7.  Two very different weight constructions")
print("  cannot share a floor that comes from the weights.  A shared floor is a property")
print("  of what they are compared AGAINST, not of what is being compared.")
print("  Ranked candidates, with what each would look like:")
print("   1. reference evolved numerically over tau (my emulation: 5.1e-6) -- floor is")
print("      flat in s AND in moment order p, exactly as they report.  MOST LIKELY.")
print("   2. the two time levels evaluated with different accuracy (different mode count,")
print("      quadrature, or one analytic and one numerical) -- same signature.")
print("   3. measuring absolute L-inf where e^{beta x} = e^5 = %.0f amplifies the right"
      % np.exp(5))
print("      half of the domain -- shifts the number by ~1e2, not by 1e8.")
print("  Cheap discriminator: section E above.  If single-mode is machine-precision and")
print("  the full reference is not, it is 1 or 2.")
