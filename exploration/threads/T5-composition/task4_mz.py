"""
Task 4 -- partial observation and memory (Mori-Zwanzig) for a numerical scheme.

Retain every M-th grid point and ask for an evolution law in the retained variables.
Periodic BCs => the fine operator is circulant, the aliasing structure is exact, and
the Mori-Zwanzig memory kernel turns out to be EXACT and FINITE.
"""
import os
for _v in ("OMP","OPENBLAS","MKL","VECLIB_MAXIMUM","NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")  # tiny matrices: threading is pure overhead
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import composition as cp

FIG = "figures"; SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)

N = 120                                    # fine periodic grid
DIVS = [M for M in range(2, 25) if N % M == 0]   # aliasing is only exact if M | N


def g_fine(theta, w=cp.W_FTCS):
    return (w[None, :] * np.exp(1j * np.outer(theta, cp.OFF_FTCS))).sum(axis=1)


def fine_operator(w=cp.W_FTCS, n=N):
    A = np.zeros((n, n))
    for k, wk in zip(cp.OFF_FTCS, w):
        A += wk * np.roll(np.eye(n), k, axis=1)
    return A


def alias_gs(M, w=cp.W_FTCS, K=1, n=N):
    assert n % M == 0, f"coarsening factor {M} must divide N={n} for exact aliasing"
    """(Nc, M) array: the M fine amplification factors, raised to K fine steps,
    that alias onto each coarse mode."""
    Nc = n // M
    q = np.arange(Nc)[:, None]; l = np.arange(M)[None, :]
    return g_fine(2 * np.pi * (q + Nc * l).ravel() / n, w).reshape(Nc, M)**K


def coarse_memory(M, w=cp.W_FTCS, K=1, n=N):
    """Exact coarse memory operators B_1..B_M (real coarse-grid convolution stencils)
    for  U^{n+1} = B_1 U^n + ... + B_M U^{n-M+1},  one coarse step = K fine steps."""
    G = alias_gs(M, w, K, n)                                    # (Nc, M)
    S = np.array([-np.poly(G[qi])[1:] for qi in range(G.shape[0])]).T   # (M, Nc) symbols
    return np.real(np.fft.ifft(S, axis=1)), S


hdr("4A.  THEOREM: the coarse law is EXACT with exactly M time levels.")
print(f"""   Periodic fine grid N={N}; keep every M-th point, Nc = N/M.  The fine operator is
   circulant with symbol g(theta_j)= sum_k w_k e^{{i k theta_j}}, theta_j = 2 pi j/N.
   Restriction to every M-th point ALIASES the M fine modes j, j+Nc, ..., j+(M-1)Nc onto
   one coarse mode, so for each coarse mode q the retained observable is
        U^n_q = sum_{{l=0}}^{{M-1}} a_l g_l^n ,
   a sum of exactly M geometric sequences.  Operator form: the alias subspace is an
   M-dimensional INVARIANT subspace of the fine operator; the coarse observable is one
   linear functional on it; Cayley-Hamilton on that M x M block gives an exact order-M
   recurrence with characteristic polynomial prod_l (z-g_l).  Hence
        B_p has symbol (-1)^{{p+1}} e_p(g_0,...,g_{{M-1}})   (elementary symmetric functions),
   the memory is EXACTLY M-1 lags, and it TERMINATES.  This is not an approximation, and
   it is unusual: MZ memory kernels are normally infinite.  Finiteness here comes from the
   fine operator being finite-dimensional AND block-diagonal over alias classes.""")

rng = np.random.default_rng(0)
A = fine_operator()
print(f"\n   Direct check: evolve the fine scheme, restrict, test the recurrence.")
print(f"   {'M':>4} {'Nc':>5} {'resid, M-1 lags':>18} {'resid, M-2 lags':>18} {'resid, Markov only':>20}")
for M in DIVS:
    Nc = N // M
    u = rng.normal(size=N); traj = [u.copy()]
    for _ in range(60):
        u = A @ u; traj.append(u.copy())
    Uc = np.array(traj)[:, ::M]
    B, _ = coarse_memory(M)

    def resid(p):
        e = 0.0
        for nn in range(M, len(Uc) - 1):
            pred = np.zeros(Nc)
            for j in range(1, min(p + 1, M) + 1):
                pred += np.real(np.fft.ifft(np.fft.fft(Uc[nn + 1 - j]) * np.fft.fft(B[j - 1])))
            e = max(e, np.abs(Uc[nn + 1] - pred).max() / np.abs(Uc[nn + 1]).max())
        return e
    print(f"   {M:>4} {Nc:>5} {resid(M-1):>18.3e} {resid(M-2):>18.3e} {resid(0):>20.3e}")

hdr("4B.  Fitted memory: the irreducible residual as a function of p")
print("""   Forget the theorem and FIT, as one would in practice.  Because every operator in
   sight is circulant, an unrestricted coarse-convolution fit decouples per coarse mode:
   for each q, least-squares for (b_1..b_p) in  Uhat^{n+1}_q = sum_j b_j Uhat^{n+1-j}_q.
   That is exactly equivalent to the full physical-space fit and is cheap.""")


def fit_residual(M, p, K=1, w=cp.W_FTCS, nt=200, ntr=4, seed=0):
    """Unrestricted coarse-convolution fit with p lags.  Circulant => decouples per coarse
    mode; batched complex normal equations over all modes at once."""
    Nc = N // M
    Ak = np.linalg.matrix_power(fine_operator(w), K)
    rg = np.random.default_rng(seed)
    trajs = []
    for _ in range(ntr):
        u = rg.normal(size=N); tr = [u.copy()]
        for _ in range(nt):
            u = Ak @ u; tr.append(u.copy())
        trajs.append(np.fft.fft(np.array(tr)[:, ::M], axis=1))      # (nt+1, Nc)
    Uh = np.concatenate([t[None] for t in trajs], axis=0)           # (ntr, nt+1, Nc)
    ns = Uh.shape[1] - 1 - p
    # X[:, s, j] = Uhat at time (p+s) - j   for j = 0..p-1  ;  y[:, s] = Uhat at time p+s+1
    X = np.stack([Uh[:, p - j: p - j + ns, :] for j in range(p)], axis=-1)  # (ntr,ns,Nc,p)
    y = Uh[:, p + 1: p + 1 + ns, :]                                        # (ntr,ns,Nc)
    X = X.transpose(2, 0, 1, 3).reshape(Nc, ntr * ns, p)
    y = y.transpose(2, 0, 1).reshape(Nc, ntr * ns)
    Xh = np.conj(X).transpose(0, 2, 1)
    G = Xh @ X                                                       # (Nc,p,p)
    rhs = np.einsum('qjs,qs->qj', Xh, y)
    lam = 1e-13 * np.maximum(np.abs(np.trace(G, axis1=1, axis2=2)).real, 1.0)
    G = G + lam[:, None, None] * np.eye(p)[None]
    c = np.linalg.solve(G, rhs)
    r_ = np.einsum('qsj,qj->qs', X, c) - y
    num = float(np.sum(np.abs(r_)**2)); den = float(np.sum(np.abs(y)**2))
    return np.sqrt(num / max(den, 1e-300))


print(f"\n   {'M':>4} " + "".join(f"{'p=%d' % p:>12}" for p in range(1, 8)))
fitres = {}
for M in [2, 3, 4, 6, 8]:
    row = []
    for p in range(1, 8):
        v = fit_residual(M, p); fitres[(M, p)] = v; row.append(v)
    print(f"   {M:>4} " + "".join(f"{v:>12.3e}" for v in row))
print("   -> the residual collapses to round-off exactly at p = M, and not one lag before.")
print("   -> MEMORY REQUIRED GROWS LINEARLY WITH THE COARSENING FACTOR M.")

hdr("4C.  Does the kernel decay?  No -- and the Markov term is not even dominant.")
print(f"   {'M':>4} " + "".join(f"{'||B_%d||_1' % p:>12}" for p in range(1, 7)) + f"{'sum_p||B_p||_1':>16}")
for M in DIVS:
    B, _ = coarse_memory(M)
    row = "".join(f"{np.abs(B[p]).sum():>12.4f}" if p < M else f"{'-':>12}" for p in range(6))
    print(f"   {M:>4} " + row + f"{sum(np.abs(B[p]).sum() for p in range(M)):>16.4f}")
print("""
   The kernel does not decay in p.  ||B_2||_1 > ||B_1||_1 for every M from 2 to 20 (it
   finally dips below at M=24), and at large M the LARGEST operator is not the Markov one
   at all: at M=20 the biggest is ||B_5||_1 = 7.96 against ||B_1||_1 = 2.00.  Space-only
   coarse-graining at the fine time step therefore does not add a small memory correction
   to a Markov law -- memory dominates it, with no exponential decay to truncate and hence
   no cheap approximate closure in this variable.  (Section 4E shows this is an artefact of
   coarsening space without coarsening time.)""")

hdr("4D.  *** Is the coarse scheme still POSITIVE? ***   (space-only, K=1)")
print("""   B_p symbol = (-1)^{p+1} e_p(g_0..g_{M-1}): the signs ALTERNATE with the lag.  Whether
   that lands on a nonnegative stencil depends on the signs of the aliased g_l, hence on r.
   'POS' = every entry of every B_p is >= 0 (a monotone / maximum-principle scheme);
   the bracketed number is sum_p ||B_p||_1, which equals 1 exactly when POS.""")
print(f"\n   fine symbol g(theta) = 1-2r+2r cos(theta) - i ra sin(theta);  Re g spans [1-4r, 1].")
print(f"   at r = {cp.R}: [{1-4*cp.R:.3f}, 1.000]  -- negative for theta near pi.\n")
rs = [0.45, 0.35, 0.30, 0.25, 0.15, 0.05]
print(f"   {'M':>4} " + "".join(f"{'r=%.2f' % r:>14}" for r in rs))
for M in DIVS:
    row = []
    for r in rs:
        ra = r * cp.C * cp.DX / cp.ALPHA
        wv = np.array([r + ra / 2, 1 - 2 * r, r - ra / 2])
        B, _ = coarse_memory(M, w=wv)
        mn = min(B[p].min() for p in range(M)); l1 = sum(np.abs(B[p]).sum() for p in range(M))
        row.append(f"{('POS' if mn >= -1e-12 else 'neg') + '(%.2f)' % l1:>14}")
    print(f"   {M:>4} " + "".join(row))

# locate the M=2 threshold
def minw2(r):
    ra = r * cp.C * cp.DX / cp.ALPHA
    B, _ = coarse_memory(2, w=np.array([r + ra / 2, 1 - 2 * r, r - ra / 2]))
    return min(B[p].min() for p in range(2))


lo, hi = 0.20, 0.49
for _ in range(60):
    mid = (lo + hi) / 2
    if minw2(mid) >= -1e-14: hi = mid
    else: lo = mid
print(f"""
   ANSWER (corrected -- an earlier draft claimed M=2 was positive at every r; it is not).

   M = 2 is positive exactly when r is LARGE ENOUGH.  The centre weight of B_2 is
        (B_2)_0 = -(1-2r)^2 + 2r^2 - ra^2/2 = -(4r^2 - 8r + 2 + ra^2)/2   (sympy),
   which is >= 0 iff  r >= 1 - sqrt(2 - ra^2)/2  ->  1 - 1/sqrt(2) = {1-1/np.sqrt(2):.6f} as ra -> 0.
        measured threshold by bisection : r* = {hi:.6f}
        analytic 1 - 1/sqrt(2)          : {1-1/np.sqrt(2):.6f}
   At r = {cp.R} the coarse M=2 law is the positive, mass-1, three-term scheme
        U^{{n+1}}_j = {2-4*cp.R:.1f} U^n_j + B_2 * U^{{n-1}},   B_2 ~ (0.2235, 0.3940, 0.1825),
   so the maximum principle survives coarsening by 2 -- but only for r >= 0.293.

   M >= 3 is NOT positive at ANY r tested, and sum_p ||B_p||_1 grows fast with M.
   So stability and coarse-graining ARE in tension, with a threshold: the tension begins
   at M = 3, and for M = 2 only once r drops below 1 - 1/sqrt(2).
   Since FTCS itself needs r <= 1/2, the window where coarsening by 2 is safe is
   r in [0.293, 0.5] -- i.e. ONLY when the fine scheme is run near its stability limit,
   so that one fine step already mixes a full cell.""")
Ms = np.array(DIVS)
l1s = np.array([sum(np.abs(coarse_memory(M)[0][p]).sum() for p in range(M)) for M in Ms])
print(f"\n   {'M':>4} " + "".join(f"{m:>9}" for m in Ms))
print(f"   {'l1':>4} " + "".join(f"{v:>9.3f}" for v in l1s))
sl = np.polyfit(Ms[2:], np.log(l1s[2:]), 1)
print(f"   fit: sum_p ||B_p||_1 ~ exp({sl[0]:.4f} M): GEOMETRIC growth in the coarsening factor.")
print("   That l1 norm is the l_inf amplification bound per coarse step, so space-only")
print("   coarse-graining loses the maximum principle by a factor compounding with M.")

hdr("4E.  RG-CONSISTENT (diffusive) coarsening: memory is crushed to an M-independent floor")
print("""   Space-only coarsening is not the RG-natural move.  Diffusive scaling pairs
   dx -> M dx with dt -> M^2 dt, holding r = alpha dt/dx^2 FIXED (the scheme sits at a
   fixed point of the scaling).  One coarse step = K = M^2 fine steps.""")
print(f"\n   {'M':>4} {'K':>6} {'||B_1||_1':>11} {'min B_1':>11} {'sum B_1':>10} {'B_1>=0?':>8} "
      f"{'||B_2||/||B_1||':>16} {'sum||B_p||_1-1':>16} {'min all B_p':>13}")
for M in DIVS:
    K = M * M; B, _ = coarse_memory(M, K=K)
    print(f"   {M:>4} {K:>6} {np.abs(B[0]).sum():>11.6f} {B[0].min():>11.2e} {B[0].sum():>10.6f} "
          f"{str(B[0].min() >= -1e-12):>8} {np.abs(B[1]).sum()/np.abs(B[0]).sum():>16.3e} "
          f"{sum(np.abs(B[p]).sum() for p in range(M))-1:>16.3e} "
          f"{min(B[p].min() for p in range(M)):>13.2e}")
print(f"""
   Three findings, one of which corrects an earlier draft of this script:

   (1) The memory ratio collapses from 1.9 (space-only, 4C) to ~1e-4, but it does NOT
       keep falling: it PLATEAUS at an M-independent floor.  The floor is
            ||B_2||_1 / ||B_1||_1  ->  exp(-2 pi^2 r)  = {np.exp(-2*np.pi**2*cp.R):.3e}   at r = {cp.R}.
       Mechanism: at the COARSE NYQUIST mode the two nearest aliases sit at theta = +-pi/M,
       exactly degenerate, so e_2/e_1 ~ |g(pi/M)|^{{M^2}} -> exp(-2 r pi^2), with the M's
       cancelling.  Verified over r = 0.15..0.45 and M = 8..24 (see table below): agreement
       to a few percent across three decades.
       Interpretation: the residual memory is precisely the part of the highest coarse mode
       that one coarse step has NOT yet mixed away.

   (2) The MARKOV term B_1 is NONNEGATIVE for every M, under both coarsenings.  Under
       diffusive scaling its mass is 1 to within the memory floor (for M >= 5), so a
       positive, memoryless coarse scheme exists with relative error ~1e-4.  Under
       space-only coarsening B_1 is also positive but its mass is M(1-2r), which exceeds 1
       for M > 1/(1-2r) = {1/(1-2*cp.R):.0f}; the memory then has to carry negative mass.

   (3) NOT the claim an earlier draft made: the full memory scheme is still NOT positive
       under diffusive coarsening (min entry ~ -3e-5).  The tension does not vanish; it
       becomes numerically negligible, bounded by the same exp(-2 pi^2 r) floor.""")
print(f"\n   memory floor vs exp(-2 pi^2 r):")
print(f"   {'r':>7} {'exp(-2 pi^2 r)':>16} " + "".join(f"{'M=%d' % M:>12}" for M in [8, 12, 15, 20, 24]))
for r in [0.45, 0.35, 0.25, 0.15]:
    ra = r * cp.C * cp.DX / cp.ALPHA; wv = np.array([r + ra / 2, 1 - 2 * r, r - ra / 2])
    row = []
    for M in [8, 12, 15, 20, 24]:
        B, _ = coarse_memory(M, w=wv, K=M * M)
        row.append(f"{np.abs(B[1]).sum()/np.abs(B[0]).sum():>12.3e}")
    print(f"   {r:>7.2f} {np.exp(-2*np.pi**2*r):>16.3e} " + "".join(row))
print("""
   => Memory is the price of coarse-graining space faster than the dynamics mixes.
      Coarse-grain space and time together at the diffusive ratio and the price drops to
      exp(-2 pi^2 r) -- small, M-independent, and set entirely by the per-step mixing r.""")

# ---------------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
for i, M in enumerate([2, 3, 4, 6, 8]):
    ps = np.arange(1, 8)
    ax.semilogy(ps, [max(fitres[(M, p)], 1e-17) for p in ps], 'os^vD'[i] + '-', color=f"C{i}", label=f"M={M}")
    ax.axvline(M, color=f"C{i}", ls=':', lw=1, alpha=0.55)
ax.set_xlabel("p  (coarse time levels used)"); ax.set_ylabel("relative one-step residual")
ax.set_title("MZ memory terminates exactly at p = M\n(dotted: p = M)", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

ax = axes[1]
for i, M in enumerate([4, 6, 8, 12]):
    B, _ = coarse_memory(M)
    ax.semilogy(np.arange(1, M + 1), [np.abs(B[p]).sum() for p in range(M)], 'os^v'[i] + '-',
                color=f"C{i}", label=f"M={M}, space only")
    Bd, _ = coarse_memory(M, K=M * M)
    ax.semilogy(np.arange(1, M + 1), [max(np.abs(Bd[p]).sum(), 1e-18) for p in range(M)],
                'os^v'[i] + '--', color=f"C{i}", alpha=0.55, label=f"M={M}, diffusive $M^2$")
ax.axhline(np.exp(-2 * np.pi**2 * cp.R), color='k', ls='-.', lw=1.4,
           label=r"floor $e^{-2\pi^2 r}$")
ax.set_xlabel("memory level p"); ax.set_ylabel(r"$\|B_p\|_1$")
ax.set_title("space-only: memory dominates\ndiffusive: crushed to an $M$-independent floor", fontsize=10)
ax.legend(fontsize=6.2, ncol=2); ax.grid(alpha=0.25, which='both'); ax.set_ylim(1e-9, 30)

ax = axes[2]
rr = np.linspace(0.08, 0.49, 120)
ax.semilogy(rr, np.exp(-2 * np.pi**2 * rr), 'k-', lw=2, label=r"$e^{-2\pi^2 r}$ (predicted floor)")
for i, M in enumerate([8, 12, 20, 24]):
    ys = []
    for r in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        ra = r * cp.C * cp.DX / cp.ALPHA
        B, _ = coarse_memory(M, w=np.array([r + ra / 2, 1 - 2 * r, r - ra / 2]), K=M * M)
        ys.append(np.abs(B[1]).sum() / np.abs(B[0]).sum())
    ax.semilogy([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10], ys, 'os^v'[i], ms=6,
                alpha=0.8, label=f"measured, M={M}")
ax.axvline(1 - 1 / np.sqrt(2), color='C3', ls='--', lw=1.3)
ax.annotate("$M=2$ positivity\nthreshold\n$r=1-1/\\sqrt{2}$",
            xy=(1 - 1 / np.sqrt(2), 2e-3), xytext=(0.145, 3e-4),
            fontsize=7, color="C3",
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.9))
ax.set_xlabel(r"$r=\alpha\Delta t/\Delta x^2$  (per-step mixing)")
ax.set_ylabel(r"$\|B_2\|_1/\|B_1\|_1$ (residual memory)")
ax.set_title("the memory floor is set by $r$ alone,\nnot by the coarsening factor", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.25, which='both')

fig.tight_layout()
fig.savefig(f"{FIG}/task4_memory.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task4_memory.png")
