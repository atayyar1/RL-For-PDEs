"""TASK 8 -- (i) the 'irreducible O((k dt)^2) error floor' claim, and
             (ii) does the 3-row LP recover the heat kernel?

CLAIM UNDER TEST (coordinating thread):
  "With all neighbours on a single past time level, sum w_i dt_i^2 = (k dt)^2
   identically, so the u_tt error term is irreducible and grows like k^2.
   And it cannot be fixed by adding time levels: for any w >= 0, Jensen gives
   sum w_i dt_i^2 >= (sum w_i dt_i)^2 > 0."

WHY THIS IS WRONG.  sum w_i dt_i^2 is not a quantity that has to vanish.  Once
u_t = L u is substituted RECURSIVELY (as the corrected build_A does), there is no
free-standing u_tt term left to cancel: every time derivative collapses into
x-derivatives and the conditions become moment conditions on the characteristic
offset xi = dx - c dt ALONE.  Explicitly, since shift and propagator commute,

    u(x+d, t+tau) = exp[ xi d_x + alpha tau d_x^2 ] u         (EXACT)

so matching u(x,t) requires the measure w to have, in xi,
    E[xi] = 0,  E[xi^2] = 2 alpha |tau|,  E[xi^3] = 0,  E[xi^4] = 3 E[xi^2]^2, ...
    i.e.  E[xi^(2p)] = (2p-1)!! (2 alpha |tau|)^p ,  E[xi^odd] = 0.
These are exactly the moments of a GAUSSIAN, which is a POSITIVE measure.  So all
orders are simultaneously satisfiable with w >= 0, to every order, and there is
no error floor of any kind.  The u_tt term is not an error: it is already
accounted for inside exp[xi d_x + alpha tau d_x^2].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import core as C

ALPHA, CVEL, DX, DT = C.ALPHA, C.CVEL, C.DX, C.DT
nu, co = C.NU_REF, C.CO_REF

print("=" * 104)
print("A.  HOW WELL DOES A FINITE POSITIVE STENCIL MATCH THE xi-MOMENTS?")
print("    target: E[xi^p] = 0 (p odd), (p-1)!! sigma^p (p even),  sigma^2 = 2 alpha |tau|")
print("=" * 104)
print("  An UNTRUNCATED Gaussian satisfies every one of these exactly and is positive,")
print("  so there is no obstruction in principle.  On a finite stencil the moments match")
print("  up to an order set by the safety factor s = m/sigma, then degrade:")
print("")
print("%4s %6s %7s | %s" % ("m", "k", "s", "relative residual of E[xi^p], p = 2,4,6,8,10"))
for m, k in [(8, 10), (12, 10), (20, 10), (30, 10), (45, 10)]:
    sig = np.sqrt(2 * k * nu)
    w = C.kernel_moment_matched(m, -k * co, 2 * k * nu + (k * co) ** 2)
    if w is None:
        continue
    j = np.arange(-m, m + 1)
    xi = j + k * co
    out = []
    for p in [2, 4, 6, 8, 10]:
        tgt = np.prod(np.arange(1, p, 2)) * sig ** p
        out.append("%9.2e" % (abs((w * xi ** p).sum() - tgt) / tgt))
    odd = max(abs((w * xi ** p).sum()) / sig ** p for p in [1, 3, 5])
    print("%4d %6d %7.2f | %s   (odd moments <= %.1e, min w = %.1e)"
          % (m, k, m / sig, " ".join(out), odd, w.min()))
print("  -> at fixed k, widening the stencil (raising s) matches ever more moments.")
print("     Nothing is irreducible; the truncation order is a free design parameter.")

print("\n" + "=" * 104)
print("B.  DIRECT REFUTATION OF THE 'IRREDUCIBLE (k dt)^2 FLOOR'")
print("    Hold k dt FIXED and vary s.  A floor would not move.  It moves.")
print("=" * 104)
ex = C.Exact(C.ic_sine, ALPHA, CVEL)
T0 = 0.10


def lte(w, m, k):
    lo, hi = m * DX, 1.0 - m * DX
    if hi <= lo:
        return np.nan
    xs = np.linspace(lo, hi, 401)
    j = np.arange(-m, m + 1)
    ap = np.zeros_like(xs)
    for wi, ji in zip(w, j):
        if wi != 0.0:
            ap += wi * ex(xs + ji * DX, T0)
    return float(np.max(np.abs(ap - ex(xs, T0 + k * DT))))


for k in [10, 25]:
    print("\n  k = %d  (k dt = %.5f, claimed floor (k dt)^2 = %.3e)"
          % (k, k * DT, (k * DT) ** 2))
    print("  %6s %5s | %12s | %14s | %s"
          % ("m", "s", "measured LTE", "LTE/(k dt)^2", "all w >= 0?"))
    sig = np.sqrt(2 * k * nu)
    for s_ in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        m = int(np.ceil(s_ * sig)) + 1
        w = C.kernel_moment_matched(m, -k * co, 2 * k * nu + (k * co) ** 2)
        if w is None:
            continue
        e = lte(w, m, k)
        if not np.isfinite(e):
            continue
        print("  %6d %5.1f | %12.3e | %14.3e | %s"
              % (m, m / sig, e, e / (k * DT) ** 2, bool((w >= -1e-15).all())))
print("\n  -> At FIXED k dt the error falls by 5-7 orders of magnitude as s grows,")
print("     with non-negative weights throughout.  An irreducible floor would be flat.")
print("     What IS true: at FIXED s the error scales as (k dt)^2 (my Task 2 result,")
print("     LTE/(k dt) = C (m dx)^2 and (m dx)^2 = 2 alpha s^2 k dt).  So the")
print("     coordinating thread measured a real (k dt)^2 scaling, but its CONSTANT is")
print("     ~exp(-s^2/2), not a fixed number -- it is a design choice, not a floor.")
print("\n  WHERE THE JENSEN ARGUMENT FAILS: it requires sum w dt^2 = 0, which is not a")
print("  consistency condition.  After recursive substitution of u_t = L u -- which the")
print("  corrected build_A already does -- no free-standing u_tt term remains; the")
print("  conditions are moments of xi = dx - c dt alone, and the Gaussian in xi (a")
print("  positive measure) satisfies all of them.  0/600 stencils cancelling sum w dt^2")
print("  is a correct computation of the wrong constraint.")

print("\n" + "=" * 104)
print("C.  DOES THE 3-ROW LP RECOVER THE HEAT KERNEL?   (team lead's follow-up 1)")
print("=" * 104)


def lp_any(m, k):
    """Feasible vertex of the 3 corrected rows."""
    j = np.arange(-m, m + 1).astype(float)
    xi = j + k * co
    s = max(m, 1.0)
    A = np.array([np.ones_like(j), xi / s,
                  (0.5 * xi ** 2 - ALPHA * k * DT / DX ** 2) / s ** 2])
    b = np.array([1.0, 0.0, 0.0])
    r = linprog(np.zeros(len(j)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return r.x if r.status == 0 else None


def lp_maxent(m, k):
    """Most spread-out feasible point: maximise -sum w^2 proxy by minimising max w."""
    j = np.arange(-m, m + 1).astype(float)
    xi = j + k * co
    s = max(m, 1.0)
    n = len(j)
    # variables [w, t]; minimise t s.t. w_i <= t
    A_eq = np.hstack([np.array([np.ones_like(j), xi / s,
                                (0.5 * xi ** 2 - ALPHA * k * DT / DX ** 2) / s ** 2]),
                      np.zeros((3, 1))])
    A_ub = np.hstack([np.eye(n), -np.ones((n, 1))])
    r = linprog(np.r_[np.zeros(n), 1.0], A_eq=A_eq, b_eq=np.array([1.0, 0.0, 0.0]),
                A_ub=A_ub, b_ub=np.zeros(n),
                bounds=[(0, None)] * n + [(0, None)], method="highs")
    return r.x[:n] if r.status == 0 else None


print("%4s %6s | %8s %8s %10s | %10s %10s | %s"
      % ("m", "k", "nnz LP", "nnz max", "nnz kernel", "TV(LP,K)", "TV(max,K)",
         "symbol err: LP / maxent / kernel"))
for m, k in [(8, 4), (12, 10), (20, 25), (30, 50)]:
    wl, wm = lp_any(m, k), lp_maxent(m, k)
    wk = C.kernel_moment_matched(m, -k * co, 2 * k * nu + (k * co) ** 2)
    if wl is None or wk is None:
        continue
    th = np.linspace(1e-3, 1.0, 60)
    ee = lambda w: np.max(np.abs(C.symbol_error(w, k, nu, co, th)))
    tv = lambda a, b: 0.5 * np.abs(a - b).sum()
    print("%4d %6d | %8d %8d %10d | %10.4f %10.4f | %.2e / %.2e / %.2e"
          % (m, k, (wl > 1e-14).sum(), (wm > 1e-14).sum(), (wk > 1e-14).sum(),
             tv(wl, wk), tv(wm, wk), ee(wl), ee(wm), ee(wk)))
print("  -> NO.  The 3-row LP returns a VERTEX with <= 3 non-zeros: total-variation")
print("     distance ~1 from the heat kernel, and a symbol error 4-8 orders of magnitude")
print("     worse.  Even the most-spread-out feasible point is not the kernel.")
print("  -> Three moment conditions plus positivity do NOT recover the Green's function.")
print("     The kernel is the UNIQUE measure matching ALL moments; pinning three of them")
print("     leaves a polytope whose vertices are 3-point measures.  Recovering the kernel")
print("     requires imposing the higher moments, i.e. knowing the propagator already.")
