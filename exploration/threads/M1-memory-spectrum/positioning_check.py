"""
positioning_check.py -- analytic checks supporting POSITIONING.md (Task 0).

Nothing here is a new experiment.  It exists only to decide whether the M1 thread
is vacuous, by (a) reproducing T5's coarse-memory numbers from a closed form,
(b) testing an analytic OBSTRUCTION to non-negative coarse laws that I derived
while reading T5, and (c) re-testing T5's Observation 4.

Notation follows T5: fine grid N, FTCS symbol g(theta) = 1-2r+2r cos th - i ra sin th,
keep every M-th point, Nc = N/M, coarse wavenumber th_c = 2 pi q / Nc.
For coarse mode q the M aliased fine modes are th = 2 pi (q + l Nc)/N, l = 0..M-1.
"""
import os
for _v in ("OMP", "OPENBLAS", "MKL", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import linprog

N = 120
R_DEF, C_DEF, ALPHA = 0.45, 1.0, 0.1
DX = 1.0 / 99


def ra_of(r, c=C_DEF):
    """Courant number consistent with a given diffusion number r (T5's convention)."""
    return r * c * DX / ALPHA


def g_sym(theta, r, ra):
    return (1 - 2 * r) + 2 * r * np.cos(theta) - 1j * ra * np.sin(theta)


def alias_gs(M, r, ra, K=1, n=N):
    """(Nc, M): the M fine amplification factors, to the K-th power, aliasing onto each coarse mode."""
    Nc = n // M
    q = np.arange(Nc)[:, None]
    l = np.arange(M)[None, :]
    return g_sym(2 * np.pi * (q + Nc * l) / n, r, ra) ** K


def cayley_hamilton(M, r, ra, K=1, n=N):
    """T5's exact coarse law: B_p symbol = (-1)^{p+1} e_p(g_0..g_{M-1}).  Returns (B, S)."""
    G = alias_gs(M, r, ra, K, n)
    S = np.array([-np.poly(G[qi])[1:] for qi in range(G.shape[0])]).T   # (M, Nc)
    return np.real(np.fft.ifft(S, axis=1)), S


SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)

# --------------------------------------------------------------------------- A
hdr("A.  Closed form for M=2 vs T5's Cayley-Hamilton numbers (pure diffusion, c=0)")
print("""   Hand derivation.  With c=0, phi = 2 pi q / N and th_c = 2 phi:
     g_0 = 1-4r sin^2(phi/2),  g_1 = 1-4r cos^2(phi/2)
     B1hat = e_1 = 2-4r                        -> stencil (0, 2-4r, 0),   half-width 0
     B2hat = -e_2 = -(1-4r+2r^2) + 2r^2 cos th_c -> stencil (r^2, -(1-4r+2r^2), r^2), half-width 1
   so B_2 >= 0  <=>  1-4r+2r^2 <= 0  <=>  r >= 1 - 1/sqrt(2) = 0.292893.""")
for r in (0.45, 0.35, 0.30, 0.292893, 0.25, 0.15):
    B, _ = cayley_hamilton(2, r, 0.0)
    Nc = N // 2
    b1c, b2c, b2n = B[0][0], B[1][0], B[1][1]
    print(f"   r={r:<9.6f} B1 centre {b1c:>10.6f} (2-4r={2-4*r:>9.6f})   "
          f"B2 centre {b2c:>10.6f} (pred {-(1-4*r+2*r**2):>9.6f})   "
          f"B2 nbr {b2n:>9.6f} (pred {r**2:>8.6f})   min B {min(B[0].min(), B[1].min()):>10.2e}")

# --------------------------------------------------------------------------- B
hdr("B.  THE OBSTRUCTION.  A non-negative coarse law of ANY depth and ANY width")
print("""   requires every UNRESOLVED alias at the constant coarse mode to be <= 0.

   Proof.  Write the law as U^{n+1} = sum_{j,k} b_{j,k} U^{n+1-j}(x + k dx_c), b >= 0.
   Exactness on a fine mode with amplification g and coarse wavenumber th_c reads
        sum_{j,k} b_{j,k} e^{i k th_c} g^{-j} = 1.
   At q = 0 (constant coarse mode) th_c = 0, so with beta_j := sum_k b_{j,k} >= 0 this is
        phi(y) := sum_{j>=1} beta_j y^j = 1   at   y = 1/g.
   (q,l)=(0,0) has g=1 and gives sum_j beta_j = 1 -- unit mass, automatically.
   phi has non-negative coefficients, phi(0)=0, phi(1)=1, and is STRICTLY INCREASING on
   [0, inf).  So y=1 is its ONLY positive root of phi(y)=1.  Hence no unresolved alias
   may have g in (0,1).  Since |g| <= 1 for a stable scheme: g_l(0) <= 0 for all l != 0.
   Depth and width are irrelevant -- more memory cannot help.  [analytic, mine]

   For pure diffusion g_l(0) = 1 - 4 r sin^2(pi l / M), so the condition is
        r >= 1 / (4 sin^2(pi/M))   (the l=1 alias binds)
   which needs r >= 1/4 (M=2), 1/3 (M=3), 1/2 (M=4), 0.724 (M=5), 1.0 (M=6), ...
   FTCS itself needs r <= 1/2.  ==> p*(M) = INFINITY for every M >= 5, and M=4 only at
   the single point r = 1/2.""")
print(f"\n   unresolved aliases g_l(q=0) = 1-4r sin^2(pi l/M), pure diffusion, r = {R_DEF}")
print(f"   {'M':>3} {'r_min needed':>13} {'feasible at r=0.45?':>21}   aliases l=1..M-1")
for M in range(2, 13):
    rmin = 1.0 / (4 * np.sin(np.pi / M) ** 2)
    gl = [1 - 4 * R_DEF * np.sin(np.pi * l / M) ** 2 for l in range(1, M)]
    ok = "YES" if all(x <= 1e-14 for x in gl) else "NO  (g>0 present)"
    print(f"   {M:>3} {rmin:>13.4f} {ok:>21}   " + " ".join(f"{x:+7.4f}" for x in gl[:6])
          + (" ..." if M > 7 else ""))

# --------------------------------------------------------------------------- C
hdr("C.  LP test of the obstruction: minimal ||b||_1 over EXACT compact laws")
print("""   delta(P,s) := min ||b||_1  s.t.  the law of depth P and half-width s is exact on
   every (mode, alias) pair.  Since exactness at (0,0) forces sum b = 1, delta >= 1 always,
   and delta = 1 exactly when a NON-NEGATIVE exact law exists.  delta - 1 is therefore a
   scale-free positivity defect that CANNOT decay by renormalisation.  Exactness is imposed
   as  sum_j Bhat_j(q) g_l^{P-j} = g_l^P  (the numerically safe form).""")


def defect(M, P, s, r, ra, n=N, tol=1e-9):
    """min ||b||_1 over exact compact laws of depth P, half-width s.  None if inexact-infeasible."""
    Nc = n // M
    G = alias_gs(M, r, ra, 1, n)                       # (Nc, M)
    ks = np.arange(-s, s + 1)
    nb = P * len(ks)
    rows, rhs = [], []
    for q in range(Nc):
        e = np.exp(1j * 2 * np.pi * q * ks / Nc)       # (2s+1,) coarse-stencil symbol
        for l in range(M):
            g = G[q, l]
            coef = np.outer(g ** (P - np.arange(1, P + 1)), e).ravel()   # (P*(2s+1),)
            rows.append(coef)
            rhs.append(g ** P)
    A = np.array(rows); y = np.array(rhs)
    Ar = np.vstack([np.hstack([A.real, -A.real]), np.hstack([A.imag, -A.imag])])
    br = np.concatenate([y.real, y.imag])
    res = linprog(np.ones(2 * nb), A_eq=Ar, b_eq=br, bounds=[(0, None)] * 2 * nb, method="highs")
    if res.status != 0:
        return None
    b = res.x[:nb] - res.x[nb:]
    return float(np.abs(b).sum()), b.reshape(P, len(ks))


for (r, ra, lab) in [(R_DEF, 0.0, "pure diffusion, r=0.45"),
                     (R_DEF, ra_of(R_DEF), "advection-diffusion, r=0.45 (T5's operating point)")]:
    print(f"\n   --- {lab} ---")
    print(f"   {'M':>3} {'s':>2} " + "".join(f"{'P=%d' % P:>12}" for P in range(1, 11)))
    for M in [2, 3, 4, 5, 6]:
        for s in [1, 2, 3]:
            row = []
            for P in range(1, 11):
                out = defect(M, P, s, r, ra)
                row.append("  infeasible" if out is None else f"{out[0] - 1:>12.3e}")
            print(f"   {M:>3} {s:>2} " + "".join(f"{v:>12}" if not v.startswith(" ") else v for v in row))

# --------------------------------------------------------------------------- D
hdr("D.  Re-test of T5 Observation 4: 'diffusive M^2 time coarsening restores positivity'")
print("""   Under K = M^2 the aliases become g_l^K.  For M=2 the unresolved alias is
   g = 1-4r = -0.8, and g^4 = +0.4096 -- POSITIVE, hence forbidden by B above.
   Prediction: the coarse law under diffusive scaling is NOT non-negative, and T5's
   'YES' column is an artefact.  Direct check of T5's own B_p:""")
print(f"\n   {'M':>3} {'K':>5} {'min_p min B_p':>15} {'sum_p ||B_p||_1':>16} {'T5 verdict':>12} {'alias g^K at q=0':>22}")
for M in [2, 3, 4, 5, 6]:
    K = M * M
    B, _ = cayley_hamilton(M, R_DEF, ra_of(R_DEF), K=K)
    mn = min(B[p].min() for p in range(M))
    l1 = sum(np.abs(B[p]).sum() for p in range(M))
    gl = alias_gs(M, R_DEF, ra_of(R_DEF), K=K)[0]
    print(f"   {M:>3} {K:>5} {mn:>15.3e} {l1:>16.9f} {('YES' if mn >= -1e-12 else 'no'):>12}   "
          + " ".join(f"{x.real:+7.4f}" for x in gl[1:4]))
print("""
   NOTE the tolerance.  T5's test was  min B_p >= -1e-12.  Read the column above: the
   violation is real but TINY, because g^{M^2} is tiny -- the law is Markov to round-off
   and B_2..B_M are ~0 with a small negative part.  So T5's Observation 4 is right in
   spirit (memory is crushed) but the word 'positive' is doing work it cannot do: the
   coarse law is non-negative only up to the size of the memory it neglects.""")
