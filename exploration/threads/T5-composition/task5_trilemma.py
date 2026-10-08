"""
Task 5 -- positivity / locality / accuracy: the decisive experiment.

Ask directly: at fixed LOCALITY (compact coarse stencil of half-width s) does extra
MEMORY (depth p) buy back POSITIVITY?

Exact formulation, no trajectory fitting.  A coarse law
    U^{n+1} = B_1 U^n + ... + B_{p+1} U^{n-p}
with compact coarse stencils B_j (half-width s) reproduces the fine dynamics exactly iff,
for every coarse mode q and every one of its M aliased amplification factors g_l(q)^K,

    sum_{j=1..p+1}  Bhat_j(q) * g_l^{p+1-j}  =  g_l^{p+1},      Bhat_j(q) = sum_k (B_j)_k e^{i k phi_q}

-- M*Nc complex equations in the (p+1)(2s+1) REAL unknowns (B_j)_k, shared across all q.
The q=0, l=0 row is g=1, i.e. sum_{j,k}(B_j)_k = 1: mass conservation is already imposed.

We solve it twice: unconstrained (R_free) and with (B_j)_k >= 0 (R_pos).  The gap is the
PRICE OF POSITIVITY at that (locality, memory) pair -- a continuous price, where the
classical results give a yes/no.
"""
import os
for _v in ("OMP", "OPENBLAS", "MKL", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import composition as cp

FIG = "figures"; SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)

N = 120
DIVS = [M for M in range(2, 25) if N % M == 0]


def g_fine(theta, w=cp.W_FTCS):
    return (w[None, :] * np.exp(1j * np.outer(theta, cp.OFF_FTCS))).sum(axis=1)


def alias_gs(M, w=cp.W_FTCS, K=1, n=N):
    assert n % M == 0
    Nc = n // M
    q = np.arange(Nc)[:, None]; l = np.arange(M)[None, :]
    return g_fine(2 * np.pi * (q + Nc * l).ravel() / n, w).reshape(Nc, M)**K


def build_system(M, s, p, K=1, w=cp.W_FTCS, n=N):
    """Real linear system A x = b for the compact coarse law.  x = vec((B_j)_k)."""
    Nc = n // M
    G = alias_gs(M, w, K, n)                              # (Nc, M)
    ks = np.arange(-s, s + 1)
    phi = 2 * np.pi * np.arange(Nc) / Nc                  # coarse-mode angles
    E = np.exp(1j * np.outer(phi, ks))                    # (Nc, 2s+1)
    rows, rhs = [], []
    for l in range(M):
        gl = G[:, l]                                      # (Nc,)
        # powers g^{p+1-j}, j = 1..p+1  ->  exponents p, p-1, ..., 0
        P = np.stack([gl**(p - (j - 1)) for j in range(1, p + 2)], axis=1)   # (Nc, p+1)
        # coefficient of (B_j)_k in mode q:  E[q,k] * P[q,j]
        blk = (P[:, :, None] * E[:, None, :]).reshape(Nc, (p + 1) * (2 * s + 1))
        rows.append(blk); rhs.append(gl**(p + 1))
    Ac = np.vstack(rows); bc = np.concatenate(rhs)
    A = np.vstack([Ac.real, Ac.imag]); b = np.concatenate([bc.real, bc.imag])
    return A, b


def solve_pair(M, s, p, K=1, w=cp.W_FTCS):
    A, b = build_system(M, s, p, K, w)
    nb = max(np.linalg.norm(b), 1e-300)
    xf, *_ = np.linalg.lstsq(A, b, rcond=None)
    rf = np.linalg.norm(A @ xf - b) / nb
    xp, _ = nnls(A, b, maxiter=20000)          # exact active-set NNLS; x >= 0
    rp = np.linalg.norm(A @ xp - b) / nb
    return rf, rp, xf, xp


hdr("5A.  Sanity: the exact law is recovered at full support and p = M-1")
print(f"   {'M':>4} {'s':>6} {'p':>4} {'R_free':>12} {'R_pos':>12} {'min free wt':>13}")
for M in [2, 3, 4, 6]:
    Nc = N // M; s_full = Nc // 2
    for p in [M - 2, M - 1, M]:
        if p < 0: continue
        rf, rp, xf, xp = solve_pair(M, s_full, p)
        print(f"   {M:>4} {s_full:>6} {p:>4} {rf:>12.3e} {rp:>12.3e} {xf.min():>13.3e}")
print("   -> R_free hits round-off exactly at p = M-1 (Thm 3), and R_pos does NOT for M>=3:")
print("      the exact coarse law is not representable with nonnegative weights.")

hdr("5B.  THE TRADE CURVE: does memory buy back positivity at fixed locality?")
print("""   For each coarsening M, sweep locality s (compact half-width, in COARSE cells) against
   memory depth p (extra lags beyond Markov).  R_free = best achievable; R_pos = best
   achievable with all weights >= 0.  Both are worst-case-free least squares over all
   Nc modes x M aliases, so they are initial-condition independent.""")
RES = {}
for M in [3, 4, 6]:
    Nc = N // M
    ss = [0, 1, 2, 3, 4, 6, Nc // 2]
    ps = list(range(0, M + 3))
    print(f"\n   M = {M}  (Nc = {Nc}, exact law needs p = {M-1} and full support)")
    print(f"   R_free / R_pos      " + "".join(f"{'p=%d' % p:>19}" for p in ps))
    for s in ss:
        cells = "full" if s == Nc // 2 else str(s)
        row = []
        for p in ps:
            rf, rp, xf, xp = solve_pair(M, s, p)
            RES[(M, s, p)] = (rf, rp)
            row.append(f"{rf:>8.2e}/{rp:<10.2e}")
        print(f"   s={cells:>4} ({2*s+1:>3} wts)   " + "".join(row))
print("""
   Read the rows: within a row (locality fixed) does R_pos fall as p grows?""")

hdr("5C.  The price of positivity, isolated")
print("""   price(s,p) = R_pos / R_free  -- how much accuracy positivity costs at that
   (locality, memory).  price = 1 means positivity is free.""")
for M in [3, 4, 6]:
    Nc = N // M; ps = list(range(0, M + 3))
    print(f"\n   M = {M}:   price = R_pos / R_free")
    print(f"   {'s':>8} " + "".join(f"{'p=%d' % p:>12}" for p in ps))
    for s in [0, 1, 2, 3, 4, 6, Nc // 2]:
        cells = "full" if s == Nc // 2 else str(s)
        row = []
        for p in ps:
            rf, rp = RES[(M, s, p)]
            row.append(f"{rp/max(rf,1e-16):>12.3g}" if rf > 1e-14 else f"{'inf':>12}")
        print(f"   {cells:>8} " + "".join(row))

hdr("5D.  Best nonnegative scheme at each accuracy target")
print("""   Turn the table around: for a target worst-case residual, what is the CHEAPEST
   (s, p) that a NONNEGATIVE compact coarse law can reach it with?  Cost = (p+1)(2s+1)
   weights, i.e. work per coarse point per coarse step.""")
for M in [3, 4, 6]:
    Nc = N // M
    print(f"\n   M = {M}:")
    print(f"   {'target R_pos':>14} {'best (s,p)':>13} {'weights':>9} {'R_pos':>11} {'R_free there':>13}")
    for tgt in [1e-1, 1e-2, 1e-3, 1e-4]:
        best = None
        for s in [0, 1, 2, 3, 4, 6, Nc // 2]:
            for p in range(0, M + 3):
                rf, rp = RES[(M, s, p)]
                if rp <= tgt:
                    c = (p + 1) * (2 * s + 1)
                    if best is None or c < best[0]:
                        best = (c, s, p, rp, rf)
        if best is None:
            print(f"   {tgt:>14.0e} {'unreachable':>13}")
        else:
            c, s, p, rp, rf = best
            cells = "full" if s == Nc // 2 else str(s)
            print(f"   {tgt:>14.0e} {'s=%s,p=%d' % (cells, p):>13} {c:>9} {rp:>11.2e} {rf:>13.2e}")

# ------------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
M = 3; Nc = N // M
ps = np.arange(0, 9)
for jj, s_ in enumerate([0, 1, 2, Nc // 2]):
    lab = "full ($N_c/2$)" if s_ == Nc // 2 else f"$s={s_}$ ({2*s_+1} wts/lag)"
    rp = [max(solve_pair(M, s_, int(p))[1], 1e-17) for p in ps]
    ax.semilogy(ps, rp, 'os^v'[jj] + '-', color=f"C{jj}", ms=5, label=lab)
ax.axvline(M - 1, color='k', ls=':', lw=1.2)
ax.text(M - 1 + 0.1, 2e-16, "$p=M-1$", fontsize=7.5, rotation=90)
ax.set_xlabel("memory depth $p$"); ax.set_ylabel(r"$R_{\rm pos}$ (best with $w\geq0$)")
ax.set_title("$M=3$: locality is irrelevant until $p=5$,\nthen decisive", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which='both'); ax.set_ylim(1e-17, 3)

fig.tight_layout()
fig.savefig(f"{FIG}/task5_trilemma.png", dpi=150, bbox_inches="tight")

hdr("5E.  Does memory buy back positivity at MINIMAL locality?  Deep sweep, s = 1.")
print("""   Fix locality at the smallest useful value -- 3 coarse weights per lag -- and push the
   memory depth p well past M-1.  R_pos is the best achievable with ALL weights >= 0.""")
RATE = {}
for M in [2, 3, 4, 6]:
    ps = list(range(0, 23))
    vals = [solve_pair(M, 1, p) for p in ps]
    print(f"\n   M = {M}   (the unconstrained law is exact at p = {M-1})")
    for lo, hi in [(0, 12), (12, 23)]:
        print("      p    " + "".join(f"{p:>10}" for p in ps[lo:hi]))
        print("  R_pos    " + "".join(f"{v[1]:>10.2e}" for v in vals[lo:hi]))
    tail = [(p, v[1]) for p, v in zip(ps, vals) if p >= M + 1 and v[1] > 1e-13]
    if len(tail) >= 4:
        a = np.polyfit([t[0] for t in tail], np.log([t[1] for t in tail]), 1)[0]
        RATE[M] = (np.exp(a), np.log(10) / -a)
        print(f"   -> R_pos ~ {np.exp(a):.4f}^p ;  extra lags per decade of accuracy = {np.log(10)/-a:.2f}")
    xp = vals[20][3]
    print(f"   at p=20: min weight = {xp.min():.2e}, sum = {xp.sum():.10f}  -> convex combination: "
          f"{bool(xp.min() >= -1e-12 and abs(xp.sum()-1) < 1e-8)}")
print("""
   ANSWER: YES.  At fixed minimal locality, extra memory buys back positivity GEOMETRICALLY.

   Every solution has all weights >= 0 by construction, and total mass S <= 1 (mass is the
   q=0, l=0 row of the system, so it is met only to within the residual: S = 1 to 1e-8 for
   M = 2,3,4 at p = 20, and S = 0.99995 for M = 6 where R_pos is still 1.1e-3).  Since the
   weights are non-negative with S <= 1, each scheme satisfies
        max |U^{n+1}| <= S * max_j max |U^{n+1-j}| <= max_j max |U^{n+1-j}|,
   a discrete maximum principle: unconditionally stable in l_inf, with no transient.
   The trade curve is real and it is quantitative:""")
print(f"\n   {'M':>4} {'decay per lag':>15} {'extra lags per decade':>23}")
for M in sorted(RATE):
    print(f"   {M:>4} {RATE[M][0]:>15.4f} {RATE[M][1]:>23.2f}")
print("""   The rate degrades sharply with the coarsening factor: memory buys back positivity, but
   the exchange rate gets worse the harder you coarse-grain.""")

hdr("5F.  Pawula: what it does and does not forbid")
print("""   PAWULA'S THEOREM (Phys. Rev. 162, 186, 1967).  For a Markov process with Kramers-Moyal
   expansion  d_t P = sum_n (-d_x)^n [ D^(n) P ],  positivity of P forces the generalised
   Cauchy-Schwarz inequality  [D^(m+n)]^2 <= D^(2m) D^(2n).  Hence if ANY even coefficient
   D^(2j), j >= 2, vanishes then D^(n) = 0 for all n >= 3.  The expansion therefore either
   stops at n = 2 (Fokker-Planck) or has INFINITELY many non-zero terms: a truncation at any
   finite order N with 3 <= N < infinity is inconsistent with a non-negative density.

   Translating this as 'a local positive propagator is necessarily second-order' is WRONG,
   and the distinction matters.  Pawula constrains the stencil's OWN cumulant sequence -- it
   may not TERMINATE beyond order 2.  Accuracy only requires MATCHING finitely many moments
   of the target kernel, which is a different thing entirely.  Demonstration:""")
from math import factorial


def theta_p(n, xi, tau):
    y = np.asarray(xi, float) - cp.C * np.asarray(tau, float); t = np.asarray(tau, float)
    return factorial(n) * sum((cp.ALPHA * t)**k * y**(n - 2 * k)
                              / (factorial(k) * factorial(n - 2 * k)) for k in range(n // 2 + 1))


def wide_exact(m, r):
    dt = r * cp.DX**2 / cp.ALPHA; ox = np.arange(-m, m + 1); nn = 2 * m + 1
    A = np.array([theta_p(k, ox * cp.DX, -dt * np.ones(nn)) for k in range(nn)])
    sc = np.abs(A).max(axis=1)[:, None]
    return (np.linalg.solve(A / sc, np.array([theta_p(k, 0., 0.) for k in range(nn)]) / sc[:, 0]),
            ox, np.linalg.cond(A / sc))


w5, o5, _ = wide_exact(2, cp.R)
kk = cp.cumulants_from_raw(cp.spatial_raw_moments(w5, o5, 10), 10)
print(f"\n   The 5-point stencil exact on Theta_0..Theta_4 at r={cp.R}: min w = {w5.min():.5f} (POSITIVE),"
      f" ||w||_1 = {cp.l1(w5):.6f}")
print("   Its own cumulants:")
for n in range(1, 11):
    tag = "matched to the exact kernel" if n <= 4 else "NON-ZERO -- as Pawula requires"
    print(f"     kappa_{n:>2} = {kk[n]:+.6e}   <- {tag}")
mc = np.array([np.sum(w5 * ((o5 - np.sum(w5 * o5)) * cp.DX)**q) for q in range(11)])
bad = sum(1 for a_ in range(1, 5) for b_ in range(1, 5)
          if mc[a_ + b_]**2 > mc[2 * a_] * mc[2 * b_] * (1 + 1e-12))
print(f"   Cauchy-Schwarz [M_(m+n)]^2 <= M_2m M_2n over 1<=m,n<=4: {bad} violations (0 expected).")

print("""
   So there IS an order cap on positive local stencils -- but it is set by r, the per-step
   mixing, not by a universal 'second order'.  Largest m for which the (2m+1)-point
   Theta_0..Theta_2m-exact stencil stays non-negative:""")
print(f"\n   {'r':>7} {'sigma=sqrt(2r) cells':>21} {'max m positive':>16} {'spatial order':>15}")
for r in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
    best = 0
    for m in range(1, 16):
        w, ox, cond = wide_exact(m, r)
        if cond > 1e12: break
        if w.min() >= 0: best = m
        else: break
    print(f"   {r:>7.2f} {np.sqrt(2*r):>21.3f} {best:>16} {2*best:>15}")
print("""
   The cap tracks the kernel width: when sigma << 1 cell the exact kernel is nearly a delta
   and its high moments cannot be reproduced by a non-negative measure on a wide lattice
   support.  At r = 0.45 a POSITIVE LOCAL stencil reaches TENTH order.  Note this is the same
   condition that appeared in Task 4D (coarsening by M=2 stays positive iff r >= 1-1/sqrt2):
   positivity needs enough mixing per step, in both settings.

   The classical result that DOES cap order for positive schemes is GODUNOV'S THEOREM --
   a linear, constant-coefficient, monotonicity-preserving scheme for the ADVECTION equation
   is at most first-order accurate.  That is a hyperbolic statement.  Here, with r held fixed
   and real diffusion present, the problem is parabolic and Godunov's bound does not bite,
   which is exactly why high-order positive stencils exist at large r.""")


# -------------------------------------------------- FIGURE (completed after 5E/5F)
ax = axes[1]
for ii, M in enumerate([2, 3, 4, 6]):
    ps = np.arange(0, 23)
    rp = [max(solve_pair(M, 1, p)[1], 1e-17) for p in ps]
    ax.semilogy(ps, rp, 'os^v'[ii] + '-', color=f"C{ii}", ms=4, label=f"M={M}")
    ax.axvline(M - 1, color=f"C{ii}", ls=':', lw=1, alpha=0.5)
ax.set_xlabel("memory depth $p$ (at fixed locality $s=1$)")
ax.set_ylabel(r"$R_{\rm pos}$")
ax.set_title("memory buys back positivity, geometrically\n(dotted: $p=M-1$, where the free law is exact)",
             fontsize=9.5)
ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both'); ax.set_ylim(1e-17, 3)

ax = axes[2]
rs = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
orders = []
for r in rs:
    best = 0
    for m in range(1, 16):
        w, ox, cond = wide_exact(m, r)
        if cond > 1e12: break
        if w.min() >= 0: best = m
        else: break
    orders.append(2 * best)
ax.plot(rs, orders, 'o-', ms=7, color='C0', label="max spatial order, $w\\geq0$")
ax.axvline(1 - 1 / np.sqrt(2), color='C3', ls='--', lw=1.4)
ax.annotate("$r=1-1/\\sqrt{2}$\n($M{=}2$ coarse\npositivity threshold)",
            xy=(1 - 1 / np.sqrt(2), 7), xytext=(0.33, 3.2), fontsize=7, color='C3',
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
ax.set_xlabel(r"$r=\alpha\Delta t/\Delta x^2$  (per-step mixing)")
ax.set_ylabel("highest spatial order reachable positively")
ax.set_title("the order cap on positive local stencils\nis set by $r$, not by '2'", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_ylim(0, 11)

fig.tight_layout()
fig.savefig(f"{FIG}/task5_trilemma.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task5_trilemma.png")

hdr("5G.  Independent validation of the exact nonnegative compact coarse laws")
print("""   The two exact nonnegative laws found in 5B/5E are the strongest claims in this thread,
   so validate them OUTSIDE the Fourier system they were fitted in: evolve the fine scheme
   from random data, restrict to the coarse grid, and (a) test the one-step recurrence,
   (b) roll the coarse law forward 400 steps with NO re-injection of fine data.""")


def fine_op(w=cp.W_FTCS, n=N):
    A = np.zeros((n, n))
    for k, wk in zip(cp.OFF_FTCS, w):
        A += wk * np.roll(np.eye(n), k, axis=1)
    return A


Afine = fine_op()
for M, s_, p_ in [(2, 1, 1), (3, 2, 5)]:
    rf, rp, xf, xp = solve_pair(M, s_, p_)
    B = xp.reshape(p_ + 1, 2 * s_ + 1)
    Nc = N // M; ks = np.arange(-s_, s_ + 1)
    print(f"\n   === M={M}, s={s_}, p={p_}  ({(p_+1)*(2*s_+1)} weights) ===")
    print(f"   rows = lag 0..{p_}, cols = coarse offsets {list(ks)}:")
    for j in range(p_ + 1):
        print(f"     B_{j+1} = " + "  ".join(f"{v:+.6f}" for v in B[j]))
    print(f"   min = {xp.min():.3e}   mass = {xp.sum():.14f}   all >= 0: {xp.min() >= -1e-14}")
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(5):
        u = rng.normal(size=N); tr = [u.copy()]
        for _ in range(80):
            u = Afine @ u; tr.append(u.copy())
        Uc = np.array(tr)[:, ::M]
        for n_ in range(p_, Uc.shape[0] - 1):
            pred = np.zeros(Nc)
            for j in range(p_ + 1):
                for kk, wk in zip(ks, B[j]):
                    pred += wk * np.roll(Uc[n_ - j], -kk)
            worst = max(worst, np.abs(pred - Uc[n_ + 1]).max() / np.abs(Uc[n_ + 1]).max())
    print(f"   max relative one-step error, 5 trajectories x 75 steps : {worst:.3e}")
    u = rng.normal(size=N); tr = [u.copy()]
    for _ in range(400):
        u = Afine @ u; tr.append(u.copy())
    Uc = np.array(tr)[:, ::M]
    hist = [Uc[i].copy() for i in range(p_ + 1)]; err = []
    for n_ in range(p_, Uc.shape[0] - 1):
        pred = np.zeros(Nc)
        for j in range(p_ + 1):
            for kk, wk in zip(ks, B[j]):
                pred += wk * np.roll(hist[-1 - j], -kk)
        hist.append(pred); err.append(np.abs(pred - Uc[n_ + 1]).max())
    print(f"   400-step self-rollout, no re-injection: max abs err {max(err):.3e}, final {err[-1]:.3e}")
print("""
   Both are exact to round-off and stable over a long unforced rollout.

   Two things worth noting about the M=3 solution:
     - B_1 = B_2 = 0 EXACTLY.  The non-negative representation uses no dependence at all on
       the two most recent coarse levels; it is a PURE-DELAY scheme, starting at lag 2.  So
       "buying back positivity with memory" does not mean adding a small correction to a
       Markov law -- it means moving the whole law backwards in time.
     - The M=2 solution B_1 = 0.2 I, B_2 = (0.2235, 0.3940, 0.1826) reproduces, to all digits
       shown, the law derived independently in Task 4D from Cayley-Hamilton on the alias
       block.  Two unrelated routes -- elementary symmetric functions of the aliased
       amplification factors, and NNLS on the Fourier system -- agree exactly.""")


hdr("5H.  Does an exact nonnegative compact law exist for M >= 4?")
print("""   For M = 2 and 3 the price of positivity is exactly zero (5G).  For M = 4 it appears not
   to be: widening the support barely helps and the residual only falls geometrically.""")
print(f"\n   M = 4:  R_pos")
print(f"   {'s':>5} " + "".join(f"{'p=%d' % p:>11}" for p in [5, 8, 11, 14, 17]))
for s_ in [2, 3, 4]:
    print(f"   {s_:>5} " + "".join(f"{solve_pair(4, s_, p)[1]:>11.2e}" for p in [5, 8, 11, 14, 17]))
print("""
   Going from 3 to 5 weights per lag took M=3 from 6e-5 to machine zero at p=5.  Nothing
   comparable happens at M=4: s=2 -> s=4 moves p=17 only from 1.0e-8 to 1.5e-9.  So M=4
   looks qualitatively different -- geometric convergence rather than exact representability.
   This is a FAILURE TO FIND within the budget searched (s <= 4, p <= 17), NOT a proof of
   non-existence.""")
