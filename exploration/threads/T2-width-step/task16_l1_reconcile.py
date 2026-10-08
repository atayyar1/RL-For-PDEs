"""TASK 16 -- reconciling the ||w||_1 compounding claim.

The team lead cannot reproduce my positive-vs-signed gap, and T5 shows the l1
bound is astronomically loose for a FIXED UNIFORM stencil (Lax-Wendroff: ||w||_1
= 1.24, symbol modulus exactly 1).  Before sending my config I re-audit my own
experiment, because two things about it were weak:

  (i) my task14 (b-why) run used a fine-FTCS reference whose OWN error may sit at
      the same level as the positive arm's error -- if so the 78.1x is unreliable;
 (ii) the "signed" arm was np.linalg.lstsq, i.e. the minimum-2-norm point of an
      underdetermined system.  That is an arbitrary representative, not the best
      signed stencil, so the comparison may be rigged by the choice.

Tests below: coverage fraction, accurate reference, FAIRER signed arms, a DIRECT
measurement of operator amplification ||A^nb||_inf, and the team lead's
translation-invariance hypothesis (constant alpha as the control).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import task13_harden as H

A0, T_END = H.A0, H.T_END
u0f = H.u0f
N = 128
xg = np.arange(N) / N
dx = 1.0 / N


def rows_at(x0, tau, offs, P, Q, dfun):
    d = np.asarray(offs, float) * dx
    tg = H.targets(x0, tau, P, Q, dfun)
    sc = np.array([max(abs(np.max(np.abs(d))) ** p, 1e-300) for p in range(P + 1)])
    return np.array([d ** p / sc[p] for p in range(P + 1)]), tg / sc


def solve_w(A, b, mode):
    n = A.shape[1]
    if mode == "pos":
        r = linprog(np.zeros(n), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        return r.x if (r.status == 0 and abs(r.x.sum() - 1) < 1e-6) else None
    if mode == "minl2":
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        return w if abs(w.sum() - 1) < 1e-6 else None
    if mode == "minl1":                       # fairest signed arm: smallest ||w||_1
        r = linprog(np.r_[np.ones(n), np.ones(n)],
                    A_eq=np.hstack([A, -A]), b_eq=b,
                    bounds=[(0, None)] * (2 * n), method="highs")
        if r.status != 0:
            return None
        w = r.x[:n] - r.x[n:]
        return w if abs(w.sum() - 1) < 1e-6 else None
    raise ValueError(mode)


def build_operator(nb, m, P, Q, dfun, mode):
    """Returns (A_matrix, coverage) or (None, coverage) if any point unsolvable."""
    tau = T_END / nb
    offs = np.arange(-m, m + 1)
    A = np.zeros((N, N))
    ok = 0
    for j in range(N):
        Amat, b = rows_at(xg[j], tau, offs, P, Q, dfun)
        w = solve_w(Amat, b, mode)
        if w is None:
            return None, ok / N
        ok += 1
        A[j, (j + offs) % N] += w
    return A, 1.0


def ref_spectral(afun, Nref=1024, nsub=60000):
    x = np.arange(Nref) / Nref
    u = u0f(x); al = afun(x)
    kk = 2 * np.pi * np.fft.fftfreq(Nref, d=1.0 / Nref)
    dal = np.real(np.fft.ifft(1j * kk * np.fft.fft(al)))

    def rhs(v):
        vh = np.fft.fft(v)
        return al * np.real(np.fft.ifft(-(kk ** 2) * vh)) + dal * np.real(np.fft.ifft(1j * kk * vh))
    dt = T_END / nsub
    for _ in range(nsub):
        k1 = rhs(u); k2 = rhs(u + dt / 2 * k1); k3 = rhs(u + dt / 2 * k2); k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    uh = np.fft.fft(u) / Nref; k2_ = np.fft.fftfreq(Nref, d=1.0 / Nref)
    return lambda xq: np.real(np.exp(2j * np.pi * np.outer(np.atleast_1d(xq), k2_)) @ uh)


print("=" * 104)
print("REFERENCE AUDIT: what is my reference actually worth?")
print("=" * 104)
afv, adv = H.make_alpha("smooth", 0.8, 1)
r1 = ref_spectral(afv, 1024, 60000)
r2 = ref_spectral(afv, 1536, 90000)
print("  spectral Nref=1024/nsub=60000 vs 1536/90000 : %.3e   <- usable floor"
      % float(np.max(np.abs(r1(xg) - r2(xg)))))
xr4 = np.arange(4096) / 4096


def ref_ftcs(afun, Nf=4096):
    x = np.arange(Nf) / Nf; dxf = 1.0 / Nf
    ah = afun(x + dxf / 2)
    dt = 0.2 * dxf ** 2 / (2 * afun(x).max())
    ns = int(np.ceil(T_END / dt)); dt = T_END / ns
    u = u0f(x)
    for _ in range(ns):
        fl = ah * (np.roll(u, -1) - u) / dxf ** 2
        u = u + dt * (fl - np.roll(fl, 1))
    return np.interp(xg, np.r_[x, 1.0], np.r_[u, u[0]])


print("  fine-FTCS Nref=4096 (what task14 used) vs spectral: %.3e" % float(np.max(np.abs(ref_ftcs(afv) - r1(xg)))))
print("  -> the task14 (b-why) reference was only good to ~1e-6, i.e. AT or ABOVE the")
print("     positive arm's error.  That 78.1x was not measurable with it.  Redone below.")
ref = r1

print("\n" + "=" * 104)
print("A.  COVERAGE + FAIRER SIGNED ARMS, against the spectral reference")
print("=" * 104)
print("  %5s %3s %3s | %9s | %11s %11s %11s | %8s %8s"
      % ("nb", "m", "P", "coverage", "err pos", "err minL2", "err minL1",
         "L1 pos", "L1 mL1"))
for nb, m, P in [(4, 8, 6), (8, 8, 6), (16, 8, 6), (32, 8, 6)]:
    out = {}
    for mode in ["pos", "minl2", "minl1"]:
        Amat, cov = build_operator(nb, m, P, 6, adv, mode)
        if Amat is None:
            out[mode] = (np.nan, np.nan, cov); continue
        u = u0f(xg).copy()
        for _ in range(nb):
            u = Amat @ u
        out[mode] = (float(np.max(np.abs(u - ref(xg)))),
                     float(np.abs(Amat).sum(1).max()), cov)
    print("  %5d %3d %3d | %8.0f%% | %11.3e %11.3e %11.3e | %8.3f %8.3f"
          % (nb, m, P, 100 * out["pos"][2], out["pos"][0], out["minl2"][0],
             out["minl1"][0], out["pos"][1], out["minl1"][1]))
print("\n  coverage = fraction of the 128 points where that arm's solve succeeded;")
print("  my solver DISCARDS the whole configuration if any point fails, so every")
print("  number I have ever reported is at 100% coverage.  (The lead's 57% run is")
print("  therefore not the same experiment -- 43% of their domain fell back to a")
print("  shared 3-point scheme that dominated both arms.)")

print("\n" + "=" * 104)
print("B.  DIRECT TEST OF THE MECHANISM: is it amplification, or just worse rows?")
print("     amplification = ||A^nb||_inf  (1.0 => non-amplifying)")
print("=" * 104)
print("  %5s | %10s %12s | %10s %12s | %s"
      % ("nb", "L1 pos", "||A^nb|| pos", "L1 minL2", "||A^nb|| minL2", "err ratio"))
for nb in [4, 8, 16, 32]:
    Ap, _ = build_operator(nb, 8, 6, 6, adv, "pos")
    As, _ = build_operator(nb, 8, 6, 6, adv, "minl2")
    if Ap is None or As is None:
        continue
    Pp = np.linalg.matrix_power(Ap, nb)
    Ps = np.linalg.matrix_power(As, nb)
    up = Ap.copy(); us = As.copy()
    u1 = u0f(xg); a = u1.copy(); b_ = u1.copy()
    for _ in range(nb):
        a = Ap @ a; b_ = As @ b_
    ep = float(np.max(np.abs(a - ref(xg)))); es = float(np.max(np.abs(b_ - ref(xg))))
    print("  %5d | %10.4f %12.4f | %10.4f %12.4f | %8.1fx"
          % (nb, np.abs(Ap).sum(1).max(), np.abs(Pp).sum(1).max(),
             np.abs(As).sum(1).max(), np.abs(Ps).sum(1).max(), es / ep))

print("\n" + "=" * 104)
print("C.  THE LEAD'S HYPOTHESIS: does the effect need VARIABLE coefficients?")
print("     control = constant alpha (translation invariant, von Neumann applies)")
print("=" * 104)
afc = lambda x: np.full_like(np.asarray(x, float), A0 * 1.0)
adc = lambda x, n: (A0 if n == 0 else 0.0 * np.asarray(x, float))
refc = ref_spectral(afc)
print("  %-12s %5s | %11s %11s | %10s | %12s"
      % ("alpha", "nb", "err pos", "err minL2", "err ratio", "||A^nb|| mL2"))
for lab, af_, ad_, rf in [("constant", afc, adc, refc), ("variable", afv, adv, ref)]:
    for nb in [8, 16, 32]:
        Ap, _ = build_operator(nb, 8, 6, 6, ad_, "pos")
        As, _ = build_operator(nb, 8, 6, 6, ad_, "minl2")
        if Ap is None or As is None:
            print("  %-12s %5d | %s" % (lab, nb, "infeasible")); continue
        a = u0f(xg).copy(); b_ = u0f(xg).copy()
        for _ in range(nb):
            a = Ap @ a; b_ = As @ b_
        ep = float(np.max(np.abs(a - rf(xg)))); es = float(np.max(np.abs(b_ - rf(xg))))
        print("  %-12s %5d | %11.3e %11.3e | %9.1fx | %12.4f"
              % (lab, nb, ep, es, es / ep,
                 np.abs(np.linalg.matrix_power(As, nb)).sum(1).max()))
