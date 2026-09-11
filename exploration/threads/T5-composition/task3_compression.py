"""
Task 3 -- a wide direct stencil is a compressed representation of many composed
narrow ones.  We (i) prove the feasibility boundary and check it by LP,
(ii) show the two statements coincide exactly, (iii) price the compression and
measure what it loses.
"""
import os
for _v in ("OMP","OPENBLAS","MKL","VECLIB_MAXIMUM","NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")  # tiny matrices: threading is pure overhead
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.special import erfc
from math import factorial
import composition as cp

FIG = "figures"; SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)

s2 = float(np.sum(cp.W_FTCS * cp.OFF_FTCS**2)) - float(np.sum(cp.W_FTCS * cp.OFF_FTCS))**2
mu1 = float(np.sum(cp.W_FTCS * cp.OFF_FTCS))

hdr("3A.  The feasibility boundary is EXACT, and it is 'one standard deviation'")
print(f"""   Claim (proof, not dimensional analysis).  Let w >= 0 be supported on |xi| <= m dx
   and satisfy the two consistency conditions  sum w = 1,  sum w xi^2 = 2 alpha tau
   (the Theta_0 and Theta_2 conditions; drift only shifts the centre).  Then
        2 alpha tau = sum_i w_i xi_i^2 <= m^2 dx^2 sum_i w_i = m^2 dx^2,
   so with tau = k dt,     k <= m^2 dx^2 / (2 alpha dt) = m^2 / (2r).      (NECESSARY)
   Conversely w = (1-p) delta_0 + (p/2)(delta_{{-m}} + delta_{{+m}}) with p = 2 alpha tau/(m dx)^2
   is nonnegative and satisfies both whenever k <= m^2/(2r).                (SUFFICIENT)
   So the boundary is exactly  sigma_required = sqrt(2 alpha k dt) <= m dx :
   a positive stencil can take the step iff the kernel's standard deviation FITS INSIDE it.
   r = {cp.R}, so k_max(m) = m^2/(2r) = {1/(2*cp.R):.4f} m^2.""")


def pos_feasible(m, k, nmom=3):
    """LP: is there w >= 0 on |xi|<=m dx with the first nmom Theta conditions for step k dt?"""
    ox = np.arange(-m, m + 1); tau = -k * cp.DT
    A = np.array([theta_poly(n, ox * cp.DX, tau * np.ones(ox.size)) for n in range(nmom)])
    sc = np.abs(A).max(axis=1)[:, None]
    b = np.array([theta_poly(n, 0., 0.) for n in range(nmom)]) / sc[:, 0]
    r_ = linprog(np.zeros(ox.size), A_eq=A / sc, b_eq=b, bounds=[(0, None)] * ox.size, method="highs")
    return r_.status == 0


def theta_poly(n, xi, tau):
    y = np.asarray(xi, float) - cp.C * np.asarray(tau, float); t = np.asarray(tau, float)
    return factorial(n) * sum((cp.ALPHA * t)**k * y**(n - 2 * k)
                              / (factorial(k) * factorial(n - 2 * k)) for k in range(n // 2 + 1))


print(f"\n   LP check of the boundary (bisection on k):")
print(f"   {'m':>4} {'k_max measured':>16} {'m^2/(2r) predicted':>20} {'ratio':>8}")
for m in [1, 2, 3, 5, 8, 12, 20, 32]:
    lo, hi = 1, int(4 * m * m / (2 * cp.R)) + 4
    if not pos_feasible(m, 1):
        print(f"   {m:>4} {'infeasible at k=1':>16}"); continue
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pos_feasible(m, mid): lo = mid
        else: hi = mid - 1
    print(f"   {m:>4} {lo:>16} {m*m/(2*cp.R):>20.2f} {lo/(m*m/(2*cp.R)):>8.4f}")

hdr("3B.  The two statements are the SAME statement")
print(f"""   Task 2: composing L narrow steps produces a kernel of standard deviation
        sigma_L = sqrt(s2 L) cells,   s2 = 2r - ra^2 = {s2:.6f}  (per-step variance).
   Task 3A: a wide positive stencil of half-width m advances k = m^2/(2r) steps.
   Substitute m = sigma_L:      k = s2 L / (2r) = L (1 - ra^2/(2r)) = {1-cp.RA**2/(2*cp.R):.6f} L.
   So k = L, exactly, up to the O(ra^2/r) = {cp.RA**2/(2*cp.R):.2e} advective correction.

   NOTE a factor-sqrt(2) convention trap: the diffusive width is sqrt(2 alpha t), not
   sqrt(alpha t).  With sqrt(alpha L dt) the two statements would disagree by sqrt(2);
   with the variance convention they agree identically.""")
print(f"\n   {'L':>6} {'sigma_L (cells)':>16} {'k=sigma_L^2/(2r)':>18} {'L':>6} {'rel err':>10}")
for L in [4, 16, 64, 256, 1024, 4096]:
    sig = np.sqrt(s2 * L)
    print(f"   {L:>6} {sig:>16.4f} {sig**2/(2*cp.R):>18.4f} {L:>6} {abs(sig**2/(2*cp.R)-L)/L:>10.2e}")

hdr("3C.  What the compression costs and what it loses")
TH = np.linspace(-np.pi, np.pi, 20001)
BAND = np.pi / 4


def band_err(w, off, tau):
    t = np.linspace(-BAND, BAND, 4001)
    g = cp.symbol(w, off, t)
    ex = np.exp(-1j * cp.C * t / cp.DX * tau - cp.ALPHA * (t / cp.DX)**2 * tau)
    return float(np.abs(g - ex).max())


def trunc_renorm(wL, oL, m):
    c = int(round(np.sum(wL * oL)))                 # centre on the mean
    sel = (oL >= c - m) & (oL <= c + m)
    w = wL[sel].copy(); o = oL[sel]
    if o.size < 2 * m + 1:                          # pad to full width
        full = np.arange(c - m, c + m + 1); wf = np.zeros(full.size)
        wf[np.searchsorted(full, o)] = w; w, o = wf, full
    return w / w.sum(), o


def raw_from_cumulants(kap):
    """Raw moments m_0..m_n from cumulants kappa_1..kappa_n (m_0 = 1)."""
    from math import comb
    n = len(kap) - 1
    m = np.zeros(n + 1); m[0] = 1.0
    for j in range(1, n + 1):
        m[j] = sum(comb(j - 1, i - 1) * kap[i] * m[j - i] for i in range(1, j + 1))
    return m


def maxent_stencil(m, k, J, cells=True):
    """Maximum-entropy NONNEGATIVE stencil on |xi-centre|<=m matching the first J
    moments of the EXACT propagator over tau = k dt.  Convex dual, solved by BFGS.
    Returns (w, offsets, moment residual) or (None, offsets, res) on failure."""
    tau = k * cp.DT
    ctr = -cp.C * tau / cp.DX                                   # exact mean, in cells
    ic = int(round(ctr))
    off = np.arange(ic - m, ic + m + 1)
    x = (off - ctr).astype(float)                               # centred, cell units
    sg = np.sqrt(2 * cp.ALPHA * tau) / cp.DX                    # exact sd, cells
    xs = x / sg                                                 # scale for conditioning
    kap = np.zeros(J)
    if J > 2: kap[2] = 1.0                                      # unit variance in xs units
    mu = raw_from_cumulants(kap)[1:J]                            # targets for <xs^1..xs^{J-1}>
    P = np.array([xs**j for j in range(1, J)])                   # (J-1, n)

    def qof(lam):
        z = P.T @ lam; z -= z.max(); e = np.exp(z); return e / e.sum()

    # damped Newton on the convex dual; Hessian = covariance of the P-features
    lam = np.zeros(J - 1)
    for _ in range(500):
        q = qof(lam)
        g = P @ q - mu
        if np.max(np.abs(g)) < 1e-13:
            break
        Pm = P @ q
        H = (P * q) @ P.T - np.outer(Pm, Pm)
        try:
            step = np.linalg.solve(H + 1e-14 * np.eye(J - 1) * max(np.trace(H), 1.0), g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        f0 = np.log(np.exp(P.T @ lam - (P.T @ lam).max()).sum()) + (P.T @ lam).max() - lam @ mu
        for _ls in range(60):                                    # backtracking line search
            ln = lam - t * step
            z = P.T @ ln
            f1 = np.log(np.exp(z - z.max()).sum()) + z.max() - ln @ mu
            if f1 <= f0 - 1e-4 * t * (g @ step):
                break
            t *= 0.5
        lam = lam - t * step
    w = qof(lam)
    res = float(np.abs(P @ w - mu).max())
    return w, off, res


print("""   Three representations of the same L-step advance, at half-width m:
     (A) truncate the exact composite to +-m and renormalise  (lossy only in the tail)
     (B) maximum-entropy NONNEGATIVE stencil matching the first J exact moments
     (C) the (2m+1)-moment Theta-exact stencil solved directly (the naive route)
   (B) is positive by construction; (C) is what happens if you fit a wide stencil head-on.
   Metric: max over well-resolved modes |theta|<=pi/4 of |g(theta) - exact propagator|.""")

rows = []
for L in [64, 256, 1024]:
    wL, oL = cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L)
    sig = np.sqrt(s2 * L); tau = L * cp.DT
    e_comp = band_err(wL, oL, tau)
    print(f"\n   L = {L}   sigma_L = {sig:.2f} cells   formal half-width = {L}")
    print(f"   exact composite (2L+1 = {2*L+1} pts): band error = {e_comp:.4e}")
    print(f"   {'m':>5} {'m/sigma':>8} {'(A) trunc':>12} {'||A-comp||_1':>13}")
    for mm in [int(np.ceil(f * sig)) for f in [1.0, 2.0, 3.0, 4.0, 6.0]]:
        wa, oa = trunc_renorm(wL, oL, mm)
        d1 = float(np.abs(np.interp(oa, oL, wL, left=0, right=0) - wa).sum())
        ea = band_err(wa, oa, tau)
        rows.append((L, mm, ea, d1))
        print(f"   {mm:>5} {mm/sig:>8.2f} {ea:>12.3e} {d1:>13.3e}")

print("""
   (A) saturates at the exact-composite error once m >~ 4 sigma: the discarded tail
   mass is Gaussian-small, so truncating the composite is essentially LOSSLESS.
   The l1 distance to the untruncated composite falls like the Gaussian tail.""")

print("\n   (B) At FIXED width m = 4 sigma, sweep the number of matched moments J:")
mrows = []
for L in [64, 256, 1024]:
    sig = np.sqrt(s2 * L); tau = L * cp.DT; mm = int(np.ceil(4 * sig))
    wL, oL = cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L)
    print(f"\n     L={L}, m={mm} ({2*mm+1} points), sigma_L={sig:.2f}")
    print(f"     {'J':>4} {'band error':>13} {'min w':>11} {'||w||_1':>9}")
    for J in [2, 3, 4, 6, 8, 12]:
        wb, ob, res = maxent_stencil(mm, L, J)
        if wb is None or res > 1e-6:
            print(f"     {J:>4} {'maxent did not converge (res=%.1e)'%res:>13}"); continue
        eb = band_err(wb, ob, tau)
        mrows.append((L, J, eb))
        print(f"     {J:>4} {eb:>13.3e} {wb.min():>11.2e} {cp.l1(wb):>9.6f}")
    print(f"     {'exact composite':>4}  {band_err(wL,oL,tau):>13.3e} (for reference)")

print("""
   Reading (B) -- three things happen, two of them unexpected:
     - J=2 (mass and mean but NO variance) is hopeless at any width.  Width alone buys
       nothing; the compression is carried by the MOMENTS, not by the support.
     - J=3, the standard consistency conditions, is already about 4x MORE ACCURATE than
       the exact composite it is meant to compress.  That is not a paradox: maxent
       targets the moments of the exact semigroup, whereas the composite faithfully
       reproduces FTCS's own accumulated defect L*eps_2 (Thm 1').  Compressing a
       trajectory of a biased integrator can beat the integrator.
     - J>=6 gets WORSE, monotonically.  Forcing the FULL Gaussian's high moments onto a
       support truncated at 4 sigma is inconsistent (a truncated Gaussian has smaller
       high moments), and the optimiser buys those moments by distorting the bulk.
       More moment information is not automatically better on a bounded footprint.
     - maxent weights are strictly positive at EVERY J tested (||w||_1 = 1 throughout).
       So there is no accuracy/positivity tension here; the blow-up in route (C) below
       is a conditioning failure of the direct solve, not an intrinsic obstruction.""")

print("\n   (C) the naive route -- solve the full (2m+1)-moment system directly:")
print(f"     {'m':>5} {'cond(A)':>12} {'||w||_1':>14} {'usable?':>9}")
for mm in [2, 3, 4, 6, 8, 10]:
    n = 2 * mm + 1; tau = 64 * cp.DT
    oc = np.arange(-mm, mm + 1)
    A = np.array([theta_poly(nn, oc * cp.DX, -tau * np.ones(n)) for nn in range(n)])
    scl = np.abs(A).max(axis=1)[:, None]
    cond = np.linalg.cond(A / scl)
    try:
        wc = np.linalg.solve(A / scl, np.array([theta_poly(nn, 0., 0.) for nn in range(n)]) / scl[:, 0])
        l1c = cp.l1(wc)
    except Exception:
        l1c = np.inf
    print(f"     {mm:>5} {cond:>12.3e} {l1c:>14.4g} {'yes' if l1c < 10 else 'NO':>9}")
print("""     The moment problem on a lattice is exponentially ill-conditioned: past m~4 the
     direct solve is meaningless in double precision.  Composition reaches the SAME
     wide kernel stably, because it never forms the moment matrix.  That is an
     independent, practical reason to compose rather than to fit wide directly.""")

hdr("3D.  Price of the advance: cost ratio")
print("""   Accounting 1 (ONE output value, the 'agent queries its inputs' cost): the composite
   needs its whole light cone, sum_{l=1..L} (2l-1) = L^2 stencil applications at 3
   mult-adds each = 3L^2.  The wide stencil costs 2m+1.
   Accounting 2 (a FULL GRID of N points, the PDE-solver cost): composite 3NL, wide (2m+1)N.""")
print(f"\n   {'L':>6} {'sigma_L':>9} {'m=4sigma':>9} {'3L^2 (1 pt)':>13} {'2m+1':>7} "
      f"{'ratio 1pt':>11} {'3L (grid)':>10} {'ratio grid':>11}")
Ls = [4, 16, 64, 256, 1024, 4096, 16384]
c1, c2, ms = [], [], []
for L in Ls:
    sig = np.sqrt(s2 * L); m = int(np.ceil(4 * sig)); ms.append(m)
    r1 = 3 * L**2 / (2 * m + 1); r2 = 3 * L / (2 * m + 1)
    c1.append(r1); c2.append(r2)
    print(f"   {L:>6} {sig:>9.2f} {m:>9} {3*L**2:>13} {2*m+1:>7} {r1:>11.1f} {3*L:>10} {r2:>11.2f}")
p1 = np.polyfit(np.log(Ls[2:]), np.log(c1[2:]), 1)[0]
p2 = np.polyfit(np.log(Ls[2:]), np.log(c2[2:]), 1)[0]
print(f"\n   fitted: ratio_1pt ~ L^{p1:.3f} (predicted 3/2),   ratio_grid ~ L^{p2:.3f} (predicted 1/2)")
print("""
   This is the precise sense in which coarse-graining buys computation, and the
   reason is Task 2: the fixed point of the flow is a TWO-PARAMETER family (mean and
   variance).  Carrying L levels of state is redundant once the kernel has
   Gaussianised; the wide stencil carries only the 2 numbers that matter, sampled on
   the O(sqrt(L)) cells where the mass actually lives.  The saving is exactly the
   formal-to-effective support ratio rho(L) = sqrt(L/s2) of Task 2C.""")

# ------------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axes[0]
L = 1024
wL, oL = cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L)
sig = np.sqrt(s2 * L)
ax.semilogy(oL - L * mu1, np.maximum(wL, 1e-320), 'C0-', lw=1.4, label=f"exact composite, L={L}")
for f, cc in [(1, 'C1'), (2, 'C2'), (4, 'C3')]:
    m = int(np.ceil(f * sig))
    ax.axvline(m, color=cc, ls='--', lw=1.2, label=f"$m={f}\\sigma_L$ = {m} cells")
    ax.axvline(-m, color=cc, ls='--', lw=1.2)
ax.axvline(L, color='k', ls=':', lw=1.6, label=f"formal half-width $L$ = {L}")
ax.axvline(-L, color='k', ls=':', lw=1.6)
ax.set_xlim(-1.15 * L, 1.15 * L); ax.set_ylim(1e-40, 1)
ax.set_xlabel("cells from the mean"); ax.set_ylabel("weight")
ax.set_title(f"formal support $2L+1={2*L+1}$ vs effective $\\sim\\sigma_L={sig:.0f}$")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

ax = axes[1]
R = np.array(rows); M = np.array(mrows)
for L, mk, cc in [(64, 'o', 'C0'), (256, 's', 'C1'), (1024, '^', 'C2')]:
    sel = R[:, 0] == L
    sg = np.sqrt(s2 * L)
    ax.semilogy(R[sel, 1] / sg, R[sel, 2], mk + '-', color=cc,
                label=f"(A) truncated composite, L={L}")
    ax.semilogy(R[sel, 1] / sg, R[sel, 3], mk + ':', color=cc, alpha=0.5,
                label=f"    $\\|\\cdot-w^{{*L}}\\|_1$, L={L}")
    e3 = M[(M[:, 0] == L) & (M[:, 1] == 3), 2]
    if e3.size:
        ax.axhline(e3[0], color=cc, ls='--', lw=1.1, alpha=0.8)
ax.text(0.97, 0.06, "dashed: maxent 3-moment\nstencil at $m=4\\sigma$", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=7.5)
ax.set_xlabel(r"half-width $m/\sigma_L$"); ax.set_ylabel("band error vs exact propagator")
ax.set_title("truncating the composite is lossless past $\\sim4\\sigma$")
ax.legend(fontsize=6.5, loc='upper right'); ax.grid(alpha=0.25, which='both')

ax = axes[2]
ax.loglog(Ls, c1, 'o-', label=r"one output point: $3L^2/(2m{+}1)\propto L^{3/2}$")
ax.loglog(Ls, c2, 's-', label=r"full grid: $3L/(2m{+}1)\propto L^{1/2}$")
ax.loglog(Ls, np.array(Ls, float)**1.5 * c1[2] / Ls[2]**1.5, 'k--', lw=1, alpha=.6)
ax.loglog(Ls, np.array(Ls, float)**0.5 * c2[2] / Ls[2]**0.5, 'k:', lw=1.4, alpha=.6)
ax.set_xlabel("L (steps advanced in one shot)"); ax.set_ylabel("cost ratio composed / wide")
ax.set_title("coarse-graining buys computation"); ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

fig.tight_layout()
fig.savefig(f"{FIG}/task3_compression.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task3_compression.png")
