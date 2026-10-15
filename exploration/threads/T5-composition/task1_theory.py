"""
Task 1 -- composition law, the two elementary theorems, counterexample hunt.
Everything printed here is either a symbolic identity (sympy) or a numerical
verification of a stated claim.  Nothing is asserted without a check.
"""
import os
for _v in ("OMP","OPENBLAS","MKL","VECLIB_MAXIMUM","NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")  # tiny matrices: threading is pure overhead
import numpy as np, sympy as sp, itertools
from math import comb
import composition as cp

np.set_printoptions(precision=6, suppress=True)
SEP = "=" * 78


def hdr(s):
    print("\n" + SEP + "\n" + s + "\n" + SEP)


# ============================================================ 1A. support law
hdr("1A.  Support / weight law:  composition = pushforward of product under +")
rng = np.random.default_rng(0)
w = rng.normal(size=4); ow = np.array([-2, -1, 1, 3])
subs = []
for i in range(4):
    n = rng.integers(2, 5)
    subs.append((rng.normal(size=n), np.sort(rng.choice(np.arange(-3, 4), n, replace=False))))
wc, oc = cp.compose_general(w, ow, subs)
# brute force
brute = {}
for i in range(4):
    for vj, ej in zip(*subs[i]):
        brute[ow[i] + ej] = brute.get(ow[i] + ej, 0.0) + w[i] * vj
err = max(abs(brute.get(o, 0.0) - wv) for o, wv in zip(oc, wc))
print(f"  general composition matches brute-force sum_ij w_i v_ij delta_{{d_i+e_ij}} :  err = {err:.2e}")
w_c = rng.normal(size=4); o_c = np.arange(-1, 3)          # contiguous supports
v_c = rng.normal(size=3); e_c = np.arange(-1, 2)
wc2, oc2 = cp.compose_uniform(w_c, o_c, v_c, e_c)
print(f"  uniform sub-agents (contiguous) == np.convolve                   :  err = "
      f"{np.abs(wc2 - np.convolve(w_c, v_c)).max():.2e}   offsets "
      f"[{oc2[0]},{oc2[-1]}] = [{o_c[0]+e_c[0]},{o_c[-1]+e_c[-1]}]")

# ============================================================ 1B. moment law
hdr("1B.  Moment composition law (binomial / shift convolution)")
print("""  Symbol:   W(s) = sum_i w_i exp(<s, d_i>),   M_{p,q}(w) = d_s^p d_t^q W(0).
  General composition:      W_comp(s) = sum_i w_i e^{<s,d_i>} V_i(s)
  Moment-homogeneous case:  W_comp(s) = W(s) V(s)   =>  moments convolve binomially.""")
p_, q_ = 4, 3
print("\n  (i) Homogeneous sub-agents:  M_{p,q}(w*v) = sum_{a,b} C(p,a)C(q,b) M_{a,b}(w) M_{p-a,q-b}(v)")
wA = rng.normal(size=5); oAx = np.array([-2, -1, 0, 1, 2]); oAt = -np.ones(5, int)
wB = rng.normal(size=4); oBx = np.array([-1, 0, 1, 2]);      oBt = -np.array([1, 1, 2, 2])
# compose in full 2-D (x,t)
pieces_x, pieces_t, pieces_w = [], [], []
for i in range(5):
    pieces_x.append(oAx[i] + oBx); pieces_t.append(oAt[i] + oBt); pieces_w.append(wA[i] * wB)
cx = np.concatenate(pieces_x); ct = np.concatenate(pieces_t); cw = np.concatenate(pieces_w)
MA = cp.raw_moments(wA, oAx, oAt, n_max=p_ + q_)
MB = cp.raw_moments(wB, oBx, oBt, n_max=p_ + q_)
MC = cp.raw_moments(cw, cx, ct, n_max=p_ + q_)
worst = 0.0
for p in range(p_ + 1):
    for q in range(q_ + 1):
        pred = sum(comb(p, a) * comb(q, b) * MA[(a, b)] * MB[(p - a, q - b)]
                   for a in range(p + 1) for b in range(q + 1))
        worst = max(worst, abs(pred - MC[(p, q)]) / max(abs(MC[(p, q)]), 1e-300))
print(f"      max RELATIVE error over all (p,q), p<={p_}, q<={q_}:  {worst:.3e}")

print("\n  (ii) Heterogeneous sub-agents:  M_{p,q}(comp) = sum_i w_i sum_{a,b} C(p,a)C(q,b) xi_i^a tau_i^b M_{p-a,q-b}(v_i)")
subs2 = []
for i in range(5):
    n = int(rng.integers(3, 6))
    subs2.append((rng.normal(size=n),
                  np.sort(rng.choice(np.arange(-3, 4), n, replace=False)),
                  -rng.integers(1, 4, size=n)))
px, pt, pw = [], [], []
for i in range(5):
    v, ex, et = subs2[i]
    px.append(oAx[i] + ex); pt.append(oAt[i] + et); pw.append(wA[i] * v)
cx2 = np.concatenate(px); ct2 = np.concatenate(pt); cw2 = np.concatenate(pw)
MC2 = cp.raw_moments(cw2, cx2, ct2, n_max=p_ + q_)
Msub = [cp.raw_moments(v, ex, et, n_max=p_ + q_) for (v, ex, et) in subs2]
worst2 = 0.0
for p in range(p_ + 1):
    for q in range(q_ + 1):
        pred = 0.0
        for i in range(5):
            xi = oAx[i] * cp.DX; ta = oAt[i] * cp.DT
            pred += wA[i] * sum(comb(p, a) * comb(q, b) * xi**a * ta**b * Msub[i][(p - a, q - b)]
                                for a in range(p + 1) for b in range(q + 1))
        worst2 = max(worst2, abs(pred - MC2[(p, q)]) / max(abs(MC2[(p, q)]), 1e-300))
print(f"      max RELATIVE error over all (p,q):  {worst2:.3e}")

# symbolic confirmation of the shift structure
hdr("1C.  Symbolic check of the shift structure (sympy)")
s, d1, d2, e1, e2 = sp.symbols('s d1 d2 e1 e2')
a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2')
W = a1 * sp.exp(s * d1) + a2 * sp.exp(s * d2)
V = b1 * sp.exp(s * e1) + b2 * sp.exp(s * e2)
comp = a1 * sp.exp(s * d1) * V + a2 * sp.exp(s * d2) * V
print("  W_comp - W*V  (homogeneous sub-agents)  simplifies to:",
      sp.simplify(sp.expand(comp - W * V)))
print("  log W_comp = log W + log V  =>  CUMULANTS ADD.  Check kappa_2 additivity symbolically:")
K = sp.log(W); Kv = sp.log(V); Kc = sp.log(sp.expand(W * V))
k2_sum = sp.simplify(sp.diff(K, s, 2).subs(s, 0) + sp.diff(Kv, s, 2).subs(s, 0)
                     - sp.diff(Kc, s, 2).subs(s, 0))
print("    kappa_2(w)+kappa_2(v)-kappa_2(w*v) =", sp.simplify(k2_sum))

# ==================================================== 1D. Theorem 1 (closure)
hdr("1D.  THEOREM 1 (closure).  Exactness on a translation-invariant class.")
print("""  Statement.  Let S be any set of functions on space-time that is invariant under
  the translations used (here: any linear space of exact solutions of a
  constant-coefficient PDE).  If w reproduces every u in S at its own centre, and
  EVERY sub-agent v_i reproduces every u in S at its own centre, then the composite
  reproduces every u in S at z*.  One line:
        sum_i w_i sum_j v_ij u(z*+d_i+e_ij) = sum_i w_i u(z*+d_i) = u(z*).
  No hypothesis on supports, weights, homogeneity, or positivity.  Two instances:""")

print("""
  (a) S = Pi_r, space-time polynomials of total degree <= r.
      OBSTRUCTION: a SINGLE-TIME-LEVEL stencil (all tau_i = tau) can never be
      Pi_r-exact for r >= 1, because M_{0,1} = tau * sum_i w_i = tau != 0.
      So Pi_r is the wrong class for evolution agents.  Verify:""")
for name, ox, ot in [("FTCS (1 level)", cp.OFF_FTCS, -np.ones(3, int)),
                     ("2 time levels ", np.array([-1, 0, 1, -1, 0, 1]), -np.array([1, 1, 1, 2, 2, 2]))]:
    A = np.array([(np.asarray(ox, float))**p * (np.asarray(ot, float))**q
                  for p in range(2) for q in range(2 - p)])
    b = np.zeros(A.shape[0]); b[0] = 1.0
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"      {name}: min ||A w - b||_2 over all w for Pi_1-exactness = {np.linalg.norm(A@w-b):.3e}")

print("""
  (b) S = Theta_r, the PDE-adapted (heat/drift) polynomials for u_t + c u_x = alpha u_xx:
          Theta_n(x,t) = H_n(x - c t, t),  H_n(y,t) = n! sum_k (alpha t)^k y^(n-2k) / (k!(n-2k)!)
      Theta_0=1, Theta_1=y, Theta_2=y^2+2 alpha t, Theta_3=y^3+6 alpha t y, ...
      These ARE reachable at one time level.  This is the right exactness class.""")


def theta(n, xi, tau, alpha=cp.ALPHA, c=cp.C):
    """Theta_n(xi, tau) -- PDE-adapted polynomial solution, evaluated at a displacement."""
    from math import factorial
    y = np.asarray(xi, float) - c * np.asarray(tau, float)
    t = np.asarray(tau, float)
    return factorial(n) * sum((alpha * t)**k * y**(n - 2 * k)
                              / (factorial(k) * factorial(n - 2 * k))
                              for k in range(n // 2 + 1))


# sanity: Theta_n really solves the PDE
xs, ts = sp.symbols('x t')
for n in range(5):
    from sympy import factorial as sfac
    y = xs - cp.C * ts
    Th = sfac(n) * sum((cp.ALPHA * ts)**k * y**(n - 2 * k) / (sfac(k) * sfac(n - 2 * k))
                       for k in range(n // 2 + 1))
    res = sp.simplify(sp.diff(Th, ts) + cp.C * sp.diff(Th, xs) - cp.ALPHA * sp.diff(Th, xs, 2))
    print(f"      PDE residual of Theta_{n}: {res}")

print("\n  Theta_n-exactness <=> the n-th CUMULANT DEFECT eps_n vanishes.  Check on FTCS:")
xi_f = cp.OFF_FTCS * cp.DX; ta_f = -cp.DT * np.ones(3)
eps = cp.cumulant_defect(cp.W_FTCS, cp.OFF_FTCS, cp.DT, n_max=4)
for n in range(5):
    thd = float(np.sum(cp.W_FTCS * theta(n, xi_f, ta_f)) - theta(n, 0.0, 0.0))
    print(f"      n={n}:  Theta_n defect = {thd:+.6e}   eps_n = {eps[n]:+.6e}   "
          f"{'MATCH' if abs(thd-eps[n]) <= 1e-12*max(abs(thd),1e-12)+1e-18 else 'differ'}")
print(f"      (eps_2 = -c^2 dt^2 = {-cp.C**2*cp.DT**2:+.6e}  <- FTCS's negative numerical diffusion)")

print("\n  Theorem 1 test on S = Theta_2, 200 random HETEROGENEOUS compositions:")


def theta_exact_stencil(ox, ot, r, rng):
    A = np.array([theta(n, np.asarray(ox, float) * cp.DX, np.asarray(ot, float) * cp.DT)
                  for n in range(r + 1)])
    # column-scale to lattice units for conditioning
    sc = np.array([max(np.abs(A[n]).max(), 1e-300) for n in range(r + 1)])[:, None]
    A = A / sc
    b = np.array([theta(n, 0.0, 0.0) for n in range(r + 1)]) / sc[:, 0]
    w0 = np.linalg.lstsq(A, b, rcond=None)[0]
    _, sv, vt = np.linalg.svd(A)
    ns = vt[int(np.sum(sv > 1e-10 * sv[0])):]
    if ns.shape[0]:
        w0 = w0 + ns.T @ rng.normal(size=ns.shape[0]) * 0.3
    return w0, float(np.linalg.norm(A @ w0 - b))


def theta_defect(w, ox, ot, r):
    d = []
    for n in range(r + 1):
        v = theta(n, np.asarray(ox, float) * cp.DX, np.asarray(ot, float) * cp.DT)
        sc = max(float(np.sum(np.abs(w) * np.abs(v))), 1e-300)
        d.append(abs(float(np.sum(w * v)) - theta(n, 0.0, 0.0)) / sc)
    return max(d)


rng = np.random.default_rng(11)
for r in [1, 2, 3]:
    win, wout, nskip = 0.0, 0.0, 0
    for _ in range(200):
        ox = np.arange(-3, 4); ot = -np.ones(7, int)
        wA, res = theta_exact_stencil(ox, ot, r, rng)
        if res > 1e-10:
            nskip += 1; continue
        subs, good = [], True
        for i in range(7):
            n = int(rng.integers(r + 3, r + 8))
            ex = rng.integers(-4, 5, size=n); et = -rng.integers(1, 5, size=n)
            v, rs = theta_exact_stencil(ex, et, r, rng)
            if rs > 1e-10:
                good = False; break
            subs.append((v, ex, et))
        if not good:
            nskip += 1; continue
        px, pt, pw = [], [], []
        for i in range(7):
            v, ex, et = subs[i]
            px.append(ox[i] + ex); pt.append(ot[i] + et); pw.append(wA[i] * v)
        CX, CT, CW = np.concatenate(px), np.concatenate(pt), np.concatenate(pw)
        win = max(win, theta_defect(CW, CX, CT, r))
        wout = max(wout, theta_defect(CW, CX, CT, r + 1))
    print(f"   r={r} ({200-nskip} valid draws): max rel. Theta_{{<=r}} defect of composite = {win:.2e}"
          f"   |  Theta_{{r+1}} defect = {wout:.2e}")

print("\n   => order is INHERITED, and it takes the MIN; it does not add.")

# ============================== 1E. Theorem 2 (submultiplicativity) ==========
hdr("1E.  THEOREM 2 (submultiplicativity of the l1 norm).")
rng = np.random.default_rng(3)
viol = 0; tight = 0; N = 4000
gaps = []
for _ in range(N):
    n = int(rng.integers(2, 6))
    w = rng.normal(size=n); ox = np.sort(rng.choice(np.arange(-4, 5), n, replace=False))
    subs = []
    for i in range(n):
        m = int(rng.integers(2, 6))
        subs.append((rng.normal(size=m), np.sort(rng.choice(np.arange(-4, 5), m, replace=False))))
    CW, CO = cp.compose_general(w, ox, subs)
    lhs = cp.l1(CW); rhs = cp.l1(w) * max(cp.l1(v) for v, _ in subs)
    if lhs > rhs * (1 + 1e-9):
        viol += 1
    gaps.append(lhs / rhs)
print(f"   ||w_comp||_1 <= ||w||_1 * max_i ||v_i||_1 :  violations {viol}/{N}")
print(f"   ratio lhs/rhs  -- median {np.median(gaps):.4f}, max {np.max(gaps):.6f}, min {np.min(gaps):.4f}")

print("\n   Equality case: nonneg weights & disjoint-or-sign-aligned supports.")
for L in [1, 2, 4, 8, 64, 512]:
    wl, _ = cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L)
    print(f"     FTCS (w>=0): L={L:>4}  ||w^{{*L}}||_1 = {cp.l1(wl):.12f}   min w = {wl.min():.3e}")

print("\n   Geometric compounding when ||w||_1 > 1:")
w_bad = np.array([-0.25, 1.5, -0.25])   # sum 1, ||.||_1 = 2
for L in [1, 2, 4, 8, 16]:
    wl, _ = cp.compose_power_fast(w_bad, cp.OFF_FTCS, L)
    print(f"     w=(-.25,1.5,-.25): L={L:>3}  ||w^{{*L}}||_1 = {cp.l1(wl):>12.4f}   2^L = {2.0**L:>12.4f}")

print("\n   Positivity is preserved under composition (products & sums of nonnegatives):")
rng = np.random.default_rng(5)
ok = True
for _ in range(2000):
    n = int(rng.integers(2, 6)); m = int(rng.integers(2, 6))
    w = rng.random(n); w /= w.sum(); v = rng.random(m); v /= v.sum()
    cw = np.convolve(w, v)
    ok &= bool(cw.min() >= -1e-16 and abs(cw.sum() - 1) < 1e-12)
print(f"     2000 random positive x positive compositions all positive & mass-1:  {ok}")

# ================================ 1F. Theorem 1' : cumulant defects ADD =======
hdr("1F.  THEOREM 1' -- cumulant defects are EXACTLY additive under composition.")
print("""  The exact propagator over time tau is a Levy (infinitely divisible) kernel:
      kappa_1 = -c tau,  kappa_2 = 2 alpha tau,  kappa_n = 0 (n>=3)  -- ALL LINEAR IN tau.
  Cumulants add under convolution; the exact target also adds.  Hence for stencils
  advancing the same tau,
      eps_n(w * v) = eps_n(w) + eps_n(v)      EXACTLY (no O(.) remainder),
  and after L self-compositions   eps_n(w^{*L}) = L eps_n(w).
  Consequence: the modified-equation coefficients are L-INDEPENDENT:
      alpha_eff = kappa_2(w^{*L}) / (2 L dt) = alpha + eps_2/(2 dt),  for every L.""")
eps1 = cp.cumulant_defect(cp.W_FTCS, cp.OFF_FTCS, cp.DT, n_max=5)
print(f"\n   {'L':>5} {'eps_2(w^*L)':>16} {'L*eps_2(w)':>16} {'rel err':>10} "
      f"{'eps_3(w^*L)':>16} {'L*eps_3(w)':>16} {'alpha_eff':>12}")
for L in [1, 2, 4, 8, 32, 128, 512]:
    wl, ol = cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L)
    e = cp.cumulant_defect(wl, ol, L * cp.DT, n_max=5)
    k2 = cp.cumulants_from_raw(cp.spatial_raw_moments(wl, ol, 5))[2]
    print(f"   {L:>5} {e[2]:>16.8e} {L*eps1[2]:>16.8e} {abs(e[2]-L*eps1[2])/abs(L*eps1[2]):>10.2e} "
          f"{e[3]:>16.8e} {L*eps1[3]:>16.8e} {k2/(2*L*cp.DT):>12.8f}")
print(f"   alpha = {cp.ALPHA},  alpha + eps_2/(2 dt) = {cp.ALPHA + eps1[2]/(2*cp.DT):.8f}"
      f"  = alpha - c^2 dt/2 = {cp.ALPHA - cp.C**2*cp.DT/2:.8f}")
print("   -> FTCS's well-known negative numerical diffusion, recovered as an additive defect.")

hdr("1G.  COUNTEREXAMPLE HUNT.  What actually degrades under composition?")
print("""  (i) NOT the order.  Thm 1 forbids the composite from dropping below min(r_w, r_v).
      A search for a violation is therefore pointless; we searched anyway (1D): none.
  (ii) The INHOMOGENEOUS one-step conditions are NOT inherited as identities.
      w and v each exact for step dt  =/=>  w*v exact for step 2dt.""")
w2, o2 = cp.compose_uniform(cp.W_FTCS, cp.OFF_FTCS, cp.W_FTCS, cp.OFF_FTCS)
M2c = cp.spatial_raw_moments(w2, o2, 2)
print(f"      M_2(w*w)       = {M2c[2]:.8e}")
print(f"      required for 2dt (= 2 alpha (2dt) + (c 2dt)^2) = {2*cp.ALPHA*2*cp.DT + (cp.C*2*cp.DT)**2:.8e}")
print(f"      shortfall      = {M2c[2] - (2*cp.ALPHA*2*cp.DT + (cp.C*2*cp.DT)**2):.8e}"
      f"   = 2 * single-step shortfall ({2*eps1[2]:.8e})")

print("""
  (iii) HETEROGENEOUS sub-agents: defects are AVERAGED WITH SIGNED WEIGHTS.
        Theta_n is an Appell/binomial-type sequence: Theta_n(z+z') = sum_k C(n,k) Theta_k(z) Theta_{n-k}(z').
        Feeding that through the composition gives the EXACT law
            D_n(comp) = D_n(w) + sum_i w_i sum_k C(n,k) Theta_k(d_i) d^{(i)}_{n-k},
        which, when every sub-agent is Theta_{n-1}-exact, collapses to
            D_n(comp) = D_n(w) + sum_i w_i d^{(i)}_n,
        hence   |D_n(comp) - D_n(w)| <= ||w||_1 max_i |d^{(i)}_n|,  ATTAINED when
        sign(d^{(i)}_n) = sign(w_i).  ||w||_1 > 1 amplifies sub-agent defects.""")


def theta_defect_n(w, ox, ot, n):
    v = theta(n, np.asarray(ox, float) * cp.DX, np.asarray(ot, float) * cp.DT)
    return float(np.sum(w * v)) - float(theta(n, 0.0, 0.0))


def outer_theta2_exact(Lam):
    """5-point, tau=0 spatial agent, exact on Theta_0,Theta_1,Theta_2 (=1, xi, xi^2), ||w||_1 = Lam."""
    ox = np.arange(-2, 3); xi = ox * cp.DX
    A = np.array([np.ones(5), xi, xi**2]); bb = np.array([1.0, 0.0, 0.0])
    w0 = np.linalg.lstsq(A, bb, rcond=None)[0]
    _, sv, vt = np.linalg.svd(A); ns = vt[int((sv > 1e-10 * sv[0]).sum()):]
    if Lam <= cp.l1(w0) + 1e-12:
        return w0, ox
    from scipy.optimize import brentq
    d = ns[0] / np.linalg.norm(ns[0])
    f = lambda t: cp.l1(w0 + t * d) - Lam
    t = brentq(f, 0, 10 * Lam)
    return w0 + t * d, ox


print(f"      {'||w||_1':>9} {'D_2(w)':>13} {'D_2(comp)':>14} {'D_2(w)+sum w_i d_i':>20} "
      f"{'||w||_1 max|d_i|':>18}")
for Lam in [1.0, 1.5, 3.0, 8.0]:
    wo, oo = outer_theta2_exact(Lam)
    sdel = 1e-3
    subs, dsub = [], []
    for i in range(len(wo)):
        pert = np.array([1.0, -2.0, 1.0]) * sdel * np.sign(wo[i])    # keeps Theta_0,Theta_1 exactness
        v = cp.W_FTCS + pert
        subs.append((v, cp.OFF_FTCS))
        dsub.append(theta_defect_n(v, cp.OFF_FTCS, -np.ones(3, int), 2))
    CW, CO = cp.compose_general(wo, oo, subs)
    Dc = theta_defect_n(CW, CO, -np.ones(len(CO), int), 2)
    Dw = theta_defect_n(wo, oo, np.zeros(len(oo), int), 2)
    print(f"      {cp.l1(wo):>9.4f} {Dw:>13.3e} {Dc:>14.6e} {Dw + float(np.dot(wo, dsub)):>20.6e} "
          f"{cp.l1(wo)*max(abs(np.array(dsub))):>18.6e}")
print("      (columns 3 and 4 agree to round-off: the law is exact, and col 5 is attained.)")

print("""
  (iv) THE SHARP ONE: individually consistent, composite ILL-POSED.
       Consistency constrains eps; it does not constrain its SIGN.  A stencil with
       kappa_2 < 0 is a perfectly good first-order-consistent one-step scheme but its
       composite is not a kernel at all (negative variance) and amplifies.""")
print(f"   {'scheme':>34} {'kappa_2':>14} {'||w||_1':>9} {'max|g|':>9} {'max|g|^1089':>13}")
th = np.linspace(-np.pi, np.pi, 20001)
cases = [("FTCS adv-diff (r=0.45)", cp.W_FTCS),
         ("FTCS pure advection (alpha=0)", np.array([cp.RA / 2, 1.0, -cp.RA / 2])),
         ("central adv + tiny diff r=0.0005", np.array([0.0005 + cp.RA / 2, 1 - 0.001, 0.0005 - cp.RA / 2]))]
for nm, ww in cases:
    k2 = cp.cumulants_from_raw(cp.spatial_raw_moments(ww, cp.OFF_FTCS, 4))[2]
    mg = float(np.abs(cp.symbol(ww, cp.OFF_FTCS, th)).max())
    print(f"   {nm:>34} {k2:>14.5e} {cp.l1(ww):>9.5f} {mg:>9.6f} {mg**1089:>13.4g}")
print("   Note the middle two are 1st-order CONSISTENT (M_0=1, M_1=-c dt) yet kappa_2 < 0.")
print("   Lax equivalence in cumulant language: consistency fixes eps_n; stability fixes sign(kappa_2).")

hdr("1H.  The honest counterexample: HIGHER ORDER LOSES TO POSITIVITY AT LARGE L.")
print("""  Build the unique 5-point one-level stencil exact on Theta_0..Theta_4 (2nd order in
  time, 4th in space) and race it against 3-point positive FTCS under composition.""")
ox5 = np.arange(-2, 3); ot5 = -np.ones(5, int)
A5 = np.array([theta(n, ox5 * cp.DX, ot5 * cp.DT) for n in range(5)])
sc5 = np.abs(A5).max(axis=1)[:, None]
b5 = np.array([theta(n, 0.0, 0.0) for n in range(5)])[:, None] / sc5
w5 = np.linalg.solve(A5 / sc5, b5[:, 0])
print(f"   w5 = {w5}   sum={w5.sum():.12f}   ||w5||_1={cp.l1(w5):.8f}   min={w5.min():.6f}")
e5 = cp.cumulant_defect(w5, ox5, cp.DT, n_max=5)
print(f"   eps(w5)   = {e5}")
print(f"   eps(ftcs) = {cp.cumulant_defect(cp.W_FTCS, cp.OFF_FTCS, cp.DT, n_max=5)}")


def band_error(w, off, L, band=np.pi / 4, nth=4001):
    """max over well-resolved modes |theta|<=band of |g(theta)^L - exact propagator|."""
    t = np.linspace(-band, band, nth)
    g = cp.symbol(w, off, t)
    ex = np.exp(-1j * cp.C * t / cp.DX * L * cp.DT - cp.ALPHA * (t / cp.DX)**2 * L * cp.DT)
    return float(np.abs(g**L - ex).max())


print(f"\n   {'L':>6} {'FTCS (pos, 1st ord)':>22} {'w5 (4th ord, indef)':>22} {'ratio':>10}")
prev = None; cross = None
for L in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1089]:
    a = band_error(cp.W_FTCS, cp.OFF_FTCS, L); b_ = band_error(w5, ox5, L)
    if prev is not None and prev[1] < prev[0] and b_ > a and cross is None:
        cross = (prev[2], L)
    prev = (a, b_, L)
    print(f"   {L:>6} {a:>22.6e} {b_:>22.6e} {b_/a:>10.4f}")
print(f"   crossover bracket L* in {cross}" if cross else "   no crossover in this range")
print(f"   ||w5||_1 = {cp.l1(w5):.6f}  ->  worst-case growth ||w5||_1^L at L=1089: "
      f"{cp.l1(w5)**1089:.3e}; actual max|g|^L = "
      f"{float(np.abs(cp.symbol(w5, ox5, th)).max())**1089:.3e}")
