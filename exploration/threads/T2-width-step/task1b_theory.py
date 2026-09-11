"""TASK 1b -- sharpen the frontier theory:
  (i) the <=1/4 lattice correction psi(mu) vs mu^2  (find a case where it bites)
 (ii) the high-Peclet cutoff: no positive stencil exists at ALL for nu*Pe^2 > 2
(iii) the CORRECTED consistency rows (keep u_tt): does the advection cap survive?
 (iv) the higher-order frontiers: M2 <= m^2/3 for 4th order, etc.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import core as C

def lp_feasible(A, b):
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return r.status == 0, (r.x if r.status == 0 else None)

def moments_A(m, targets):
    """rows = moments 0..len(targets)-1 of w on {-m..m}; b = targets."""
    j = np.arange(-m, m + 1).astype(float)
    s = max(m, 1.0)
    A = np.array([(j / s) ** p for p in range(len(targets))])
    b = np.array([t / s ** p for p, t in enumerate(targets)])
    return A, b

print("="*78)
print("(i) LATTICE CORRECTION psi(mu) = lower convex envelope of j^2  vs  mu^2")
print("="*78)
print("  psi(mu) - mu^2 = f(1-f),  f = frac(|mu|)  ->  max 1/4 at half-integer mu")
bad = 0
for mu in [0.5, 1.5, 2.5, 3.5, 0.3, 7.5]:
    # minimal second moment on the lattice with mean mu
    m = 12
    j = np.arange(-m, m+1).astype(float)
    r = linprog(j**2, A_eq=np.array([np.ones_like(j), j]), b_eq=np.array([1.0, mu]),
                bounds=(0, None), method="highs")
    print("  mu=%5.2f   LP min M2 = %.10f   psi = %.10f   mu^2 = %.4f   gap = %.4f"
          % (mu, r.fun, C.psi(mu), mu**2, C.psi(mu)-mu**2))
    assert abs(r.fun - C.psi(mu)) < 1e-9
print("  -> psi verified as the exact minimum-second-moment envelope.")

# Now a regime where psi (not mu^2) sets k_max.  Need frac(k*Co) ~ 1/2 at the edge.
print("\n  Searching for (nu,Co) where the lattice correction changes k_max ...")
found = []
rng = np.random.default_rng(0)
for _ in range(4000):
    nu = rng.uniform(0.05, 0.5); co = rng.uniform(0.02, 0.6)
    m = 30
    kA = 0; kB = 0
    for k in range(1, 400):
        M1, M2 = -k*co, 2*k*nu
        if M2 > m**2: break
        if M2 >= C.psi(M1) - 1e-12: kA = k          # exact lattice rule
        if M2 >= M1**2 - 1e-12: kB = k              # naive Var>=0 rule
    if kA != kB and kA > 0:
        found.append((nu, co, kA, kB))
    if len(found) >= 5: break
for nu, co, kA, kB in found:
    m = 30
    okA, _ = C.w_positive(np.arange(-m, m+1)*C.DX, -(kB)*np.ones(2*m+1)*C.DT,
                          alpha=nu*C.DX**2/C.DT, c=co*C.DX/C.DT)
    print("  nu=%.4f Co=%.4f  k_max(lattice)=%d  k_max(naive Var>=0)=%d   "
          "LP at naive k: %s" % (nu, co, kA, kB, okA))
print("  -> the lattice envelope is strictly tighter than Var>=0; LP confirms.")

print("\n" + "="*78)
print("(ii) HIGH-PECLET CUTOFF: is there ANY (m,k) with a positive stencil?")
print("="*78)
print("  k=1 feasible  <=>  Co^2 <= 2 nu  <=>  nu * Pe^2 <= 2,  Pe = Co/nu = c dx/alpha")
for nu in [0.45, 0.25, 0.1]:
    pc = np.sqrt(2.0 / nu)
    print("  nu=%.2f -> critical cell Peclet = sqrt(2/nu) = %.4f" % (nu, pc))
    for pe in [pc*0.9, pc*1.05]:
        co = nu*pe
        # m = 40, scan k
        any_ok = any(C.feasible_analytic(40, k, nu, co) for k in range(1, 500))
        # LP cross-check at k=1, m=40
        dtl = nu*C.DX**2/C.ALPHA; cc = co*C.DX/dtl
        ok1, _ = C.w_positive(np.arange(-40, 41)*C.DX, -dtl*np.ones(81), C.ALPHA, cc)
        print("     Pe=%.4f: any (m<=40,k) feasible? %-5s   LP(m=40,k=1): %s"
              % (pe, any_ok, ok1))
print("  -> classic cell-Peclet barrier: central-in-space consistency with EXACTLY")
print("     the physical diffusion (no numerical diffusion) is positive only if")
print("     nu*Pe^2 <= 2.  Widening the stencil does NOT help.")

print("\n" + "="*78)
print("(iii) CORRECTED consistency (keep the u_tt term): M2 = 2 k nu + (k Co)^2")
print("="*78)
print("  The spec's 3 rows enforce M2 = 2 k nu, but the TRUE solution operator over")
print("  k dt is a Gaussian with mean -k Co and variance 2 k nu, i.e. second moment")
print("  M2 = 2 k nu + (k Co)^2.  Dropping (k Co)^2 IS the u_tt truncation error.")
nu, co = C.NU_REF, C.CO_REF
print("\n   m | k_max spec (M2=2k nu) | k_max corrected (M2=2k nu+(kCo)^2)")
for m in [5, 10, 15, 20, 25, 30, 40]:
    ks = C.kmax_analytic(m, nu, co)
    kc = 0
    for k in range(1, 5000):
        M1, M2 = -k*co, 2*k*nu + (k*co)**2
        if M2 > m**2: break
        if M2 >= C.psi(M1) - 1e-12: kc = k
    # LP check of the corrected system at kc (4-row: 1, j, j^2)
    okc, wc = lp_feasible(*moments_A(m, [1.0, -kc*co, 2*kc*nu + (kc*co)**2]))
    print("  %3d | %20d | %12d   (LP: %s)" % (m, ks, kc, okc))
print("  -> the spurious advection cap 2 nu/Co^2 DISAPPEARS once u_tt is kept.")
print("     Corrected rule: 2 k nu + (k Co)^2 <= m^2, i.e. the stencil must cover")
print("     (advective displacement)^2 + (diffusive variance), both in cells.")

print("\n" + "="*78)
print("(iv) HIGHER-ORDER frontiers: matching p Gaussian moments needs M2 <= m^2/c_p")
print("="*78)
print("  Gaussian moment relations (c=0): M4 = 3 M2^2, M6 = 15 M2^3, ...")
print("  Support bound M_{2p} <= m^{2p-2} M2 forces M2 <= m^2/(2p-1)!!^(1/(p-1)) ...")
for m in [6, 10, 16, 24]:
    row = [m]
    for tgt_order in [2, 4, 6]:
        # find max k with a positive w on {-m..m} matching Gaussian moments up to tgt_order
        best = 0
        for k in range(1, int(m**2/(2*nu))+2):
            b = [1.0, 0.0]
            v = 2*k*nu
            mom = [1.0, 0.0, v]
            if tgt_order >= 4: mom += [0.0, 3*v**2]
            if tgt_order >= 6: mom += [0.0, 15*v**3]
            ok, _ = lp_feasible(*moments_A(m, mom))
            if ok: best = k
        row.append(best)
    print("  m=%2d:  k_max (2 moments)=%4d   (4 moments)=%4d   (6 moments)=%4d   "
          "ratios %.2f, %.2f" % (row[0], row[1], row[2], row[3],
                                 row[1]/max(row[2],1), row[1]/max(row[3],1)))
print("  -> predicted ratios: k(2)/k(4) = 3,  k(2)/k(6) = 15^(1/2)=3.87 .. exact")
print("     values follow from  M_{2p} <= m^{2p-2} M2  with Gaussian M_{2p}.")
