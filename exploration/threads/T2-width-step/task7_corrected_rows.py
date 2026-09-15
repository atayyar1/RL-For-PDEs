"""TASK 7 -- team lead's build_A correction: verify, then redo what depends on it.

Claim: row 3 should be  1/2 (dx - c dt)^2 + alpha dt,  not  1/2 dx^2 + alpha dt.

DERIVATION (confirms the claim).  For the constant-coefficient operator
L = -c d_x + alpha d_x^2, the shift and the propagator commute, so EXACTLY

    u(x+d, t+tau) = exp[ d d_x ] exp[ tau L ] u = exp[ (d - c tau) d_x + alpha tau d_x^2 ] u

Writing xi = d - c tau (the characteristic offset), the expansion is

    1 + xi d_x + (alpha tau + xi^2/2) d_x^2 + (xi^3/6 + alpha tau xi) d_x^3
      + (xi^4/24 + alpha tau xi^2/2 + (alpha tau)^2/2) d_x^4 + ...

so the u_xx row is  alpha tau + xi^2/2 = alpha dt + (dx - c dt)^2/2.  CONFIRMED.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import core as C

ALPHA, CVEL = C.ALPHA, C.CVEL
DX, DT = C.DX, C.DT
nu, co = C.NU_REF, C.CO_REF


def build_A_corrected(dxi, dti, alpha=ALPHA, c=CVEL):
    h = max(np.max(np.abs(dxi)), np.sqrt(alpha * np.max(np.abs(dti))), 1e-12)
    xi = dxi - c * dti
    A = np.array([np.ones(len(dxi)), xi / h, (0.5 * xi ** 2 + alpha * dti) / h ** 2])
    return A, np.array([1.0, 0.0, 0.0])


def w_pos_corrected(dxi, dti, alpha=ALPHA, c=CVEL):
    A, b = build_A_corrected(np.asarray(dxi, float), np.asarray(dti, float), alpha, c)
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if r.status != 0:
        return False, None
    w = r.x
    assert abs(w.sum() - 1) < 1e-6, "rank-deficient row space: sum(w) = %.6f" % w.sum()
    return True, w


print("=" * 100)
print("A.  CORRECTED ROWS == the moment condition I already used (M2 = 2k nu + (k Co)^2)")
print("=" * 100)
print("  Row 3 with dt_i = -k dt for all i, and M1 = -k Co (row 2):")
print("    1/2[dx^2 M2 + 2 dx c k dt M1 + (c k dt)^2] = alpha k dt")
print("    => M2 = 2 k nu + (k Co)^2 .  This is exactly core.kernel_exact_semigroup(order='exact').")
print("\n  numerical check (m, k) -> does the corrected LP agree with the analytic rule?")
ok_all = True
for m in [3, 6, 10, 15, 20, 25, 30]:
    for k in [1, 5, 20, 100, 300, 600]:
        lp, _ = w_pos_corrected(np.arange(-m, m + 1) * DX, -k * DT * np.ones(2 * m + 1))
        M1, M2 = -k * co, 2 * k * nu + (k * co) ** 2
        an = (abs(M1) <= m) and (M2 <= m ** 2 + 1e-12) and (M2 >= C.psi(M1) - 1e-12)
        if lp != an:
            ok_all = False
            print("    MISMATCH m=%d k=%d  LP=%s analytic=%s" % (m, k, lp, an))
print("  LP == analytic for all probed (m,k): %s" % ok_all)

print("\n" + "=" * 100)
print("B.  CORRECTED FRONTIER: closed form and how it differs from the buggy one")
print("=" * 100)
print("  2 k nu + (k Co)^2 <= m^2   =>   k_max = floor( [sqrt(nu^2 + Co^2 m^2) - nu] / Co^2 )")
print("  (-> m^2/(2 nu) as Co -> 0).  Physically: sigma^2 + mu^2 <= (m dx)^2, i.e. the")
print("  stencil's second moment must cover diffusive variance PLUS advective displacement^2.")


def kmax_corr(m, nu=nu, co=co):
    if co == 0:
        return int(np.floor(m ** 2 / (2 * nu)))
    k = (np.sqrt(nu ** 2 + co ** 2 * m ** 2) - nu) / co ** 2
    kk = int(np.floor(k))
    while kk > 0 and not (2 * kk * nu + (kk * co) ** 2 <= m ** 2):
        kk -= 1
    return kk


print("\n%4s | %12s %12s %12s | %10s" % ("m", "buggy rows", "corrected", "closed form", "buggy/corr"))
for m in [3, 5, 8, 12, 16, 20, 25, 30, 40]:
    kb = C.kmax_analytic(m, nu, co)
    kc = kmax_corr(m)
    lp, _ = w_pos_corrected(np.arange(-m, m + 1) * DX, -kc * DT * np.ones(2 * m + 1))
    lp2, _ = w_pos_corrected(np.arange(-m, m + 1) * DX, -(kc + 1) * DT * np.ones(2 * m + 1))
    tag = "ok" if (lp and not lp2) else "CHECK"
    print("%4d | %12d %12d %12d | %10.2f  (LP %s)"
          % (m, kb, kc, kc, kb / max(kc, 1), tag))
print("\n  NOTE: the buggy rows OVERSTATE k_max at small m (they ask for too little")
print("  second moment) and then CAP it at 2 nu/Co^2 = %.0f for large m.  The" % (2 * nu / co ** 2))
print("  saturation the coordinating thread saw near k~435 is an artefact of the bug;")
print("  the corrected frontier does NOT saturate.")

print("\n" + "=" * 100)
print("C.  PECLET SWEEP REDONE WITH CORRECTED ROWS")
print("=" * 100)
print("  Buggy rows gave a spurious 'no positive stencil exists for nu*Pe^2 > 2'.")
print("  Corrected lower bound:  2 k nu + (k Co)^2 >= psi(k Co) = (k Co)^2 + f(1-f)")
print("                      =>  2 k nu >= f(1-f),  f = frac(k Co),  f(1-f) <= 1/4")
print("  So for nu >= 1/8 the lower bound NEVER binds: no cell-Peclet barrier at all.\n")
print("%7s %9s | %s" % ("Pe", "Co", "k_max(m = 1, 5, 12, 25, 40), corrected rows"))
for pe in [0.0, 0.101, 0.5, 1.0, 2.0, 5.0, 20.0]:
    cc = pe * ALPHA / DX
    cod = cc * DT / DX
    ks = []
    for m in [1, 5, 12, 25, 40]:
        kc = kmax_corr(m, nu, cod)
        lp, _ = w_pos_corrected(np.arange(-m, m + 1) * DX,
                                -max(kc, 1) * DT * np.ones(2 * m + 1), ALPHA, cc)
        ks.append("%d%s" % (kc, "" if (lp or kc == 0) else "!"))
    print("%7.3f %9.5f | %s" % (pe, cod, ", ".join(ks)))
print("\n  -> with the corrected rows a positive stencil exists at EVERY cell Peclet")
print("     tested (up to Pe=20).  The 'Pe > sqrt(2/nu) barrier' I reported earlier was")
print("     a consequence of the dropped terms and is WITHDRAWN.")
print("  -> at high Pe the frontier becomes k_max ~ m/Co (advection-limited, LINEAR in m),")
print("     not m^2/(2 nu):  the stencil must reach the departure point m dx >= c k dt.")

print("\n" + "=" * 100)
print("D.  3-POINT STENCIL WITH CORRECTED ROWS = LAX-WENDROFF (team lead's claim)")
print("=" * 100)
for cod in [0.2, 0.5, 0.9]:
    m, k = 1, 1
    M1, M2 = -cod, 2 * 0.0 + cod ** 2          # alpha = 0, pure advection
    w_lw = np.array([cod / 2 + cod ** 2 / 2, 1 - cod ** 2, -cod / 2 + cod ** 2 / 2])
    j = np.array([-1.0, 0, 1])
    print("  Co=%.1f  LW weights %s   M1=%.4f (target %.4f)  M2=%.4f (target %.4f)  min w=%+.4f"
          % (cod, np.array2string(w_lw, precision=4), (w_lw * j).sum(), M1,
             (w_lw * j ** 2).sum(), M2, w_lw.min()))
print("  -> confirmed: the corrected rows at m=k=1, alpha=0 give exactly Lax-Wendroff,")
print("     which has a NEGATIVE weight for 0 < Co < 1 (Godunov's barrier).  The buggy")
print("     rows give FTCS-central, unconditionally unstable for pure advection.")
