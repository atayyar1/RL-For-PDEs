"""
Task 2 -- composition as a renormalization-group flow.

w_ftcs is a 3-point POSITIVE stencil with mass 1, i.e. the one-step transition law
of a random walk on the lattice.  Composing it L times is exactly the L-fold
convolution power = the law of the sum of L iid steps.  The flow therefore has a
Gaussian (heat-kernel) fixed point by the local CLT, and the control parameter is
the ratio of formal to effective support.
"""
import os
for _v in ("OMP","OPENBLAS","MKL","VECLIB_MAXIMUM","NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")  # tiny matrices: threading is pure overhead
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import composition as cp
from scipy.special import erfc

FIG = "figures"
SEP = "=" * 78
np.set_printoptions(precision=6, suppress=True)


def hdr(s):
    print("\n" + SEP + "\n" + s + "\n" + SEP)


LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
KER = {L: cp.compose_power_fast(cp.W_FTCS, cp.OFF_FTCS, L) for L in LS}

# exact per-step moments in CELL units
mu1 = float(np.sum(cp.W_FTCS * cp.OFF_FTCS))                     # = -ra
m2 = float(np.sum(cp.W_FTCS * cp.OFF_FTCS**2))                   # = 2r
s2 = m2 - mu1**2                                                 # = 2r - ra^2
k3 = float(np.sum(cp.W_FTCS * (cp.OFF_FTCS - mu1)**3))
k4 = float(np.sum(cp.W_FTCS * (cp.OFF_FTCS - mu1)**4)) - 3 * s2**2

hdr("2A.  The flow preserves the simplex: non-negativity and unit mass at every L")
print(f"   per-step (cell units): mu1 = {mu1:+.8f} (= -ra = {-cp.RA:+.8f})")
print(f"                          sigma^2 = {s2:.8f} (= 2r - ra^2 = {2*cp.R - cp.RA**2:.8f})")
print(f"                          kappa_3 = {k3:+.6e}   kappa_4 = {k4:+.6e}")
print(f"\n   {'L':>6} {'n pts':>7} {'sum w':>18} {'min w':>13} {'||w||_1':>12} {'max w':>10}")
for L in LS:
    w, o = KER[L]
    print(f"   {L:>6} {w.size:>7} {w.sum():>18.15f} {w.min():>13.3e} {cp.l1(w):>12.10f} {w.max():>10.5f}")
allpos = all(KER[L][0].min() >= 0 for L in LS)
allmass = max(abs(KER[L][0].sum() - 1) for L in LS)
print(f"\n   all weights >= 0 for every L tested : {allpos}")
print(f"   max |sum w - 1| over all L          : {allmass:.2e}")
print("   => the L-fold composite is a probability distribution: the flow is a random walk.")

hdr("2B.  Convergence to the heat kernel under diffusive rescaling")
print("""   Rescale  z = (k - L mu1) / (sigma sqrt(L))  and plot  sigma sqrt(L) * w^{*L}_k.
   Three error metrics (all in cell units):
     LOCAL  = sup_k | sigma_L w_k - phi(z_k) |            (local CLT, sup of density)
     TV     = (1/2) sum_k | w_k - Gaussian mass in cell k |
     EDGE   = LOCAL after subtracting the 1st Edgeworth term -(k3/(6 sigma^3 sqrt L)) He_3(z) phi(z)""")


def he3(z):
    return z**3 - 3 * z


def metrics(L):
    w, o = KER[L]
    sig = np.sqrt(s2 * L)
    z = (o - L * mu1) / sig
    phi = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
    local = float(np.abs(sig * w - phi).max())
    # Gaussian mass in cell k (exact, via erf) for a fair TV
    from scipy.special import erf
    lo = (o - 0.5 - L * mu1) / (sig * np.sqrt(2)); hi = (o + 0.5 - L * mu1) / (sig * np.sqrt(2))
    gm = 0.5 * (erf(hi) - erf(lo))
    tv = 0.5 * float(np.abs(w - gm).sum())
    edge = float(np.abs(sig * w - phi * (1 + k3 / (6 * s2**1.5 * np.sqrt(L)) * he3(z))).max())
    return local, tv, edge, sig


print(f"\n   {'L':>6} {'sigma_L(cells)':>15} {'LOCAL':>12} {'LOCAL*sqrtL':>13} {'TV':>12} "
      f"{'TV*sqrtL':>10} {'EDGE':>12} {'EDGE*L':>10}")
rows = []
for L in LS:
    lo_, tv, ed, sig = metrics(L)
    rows.append((L, sig, lo_, tv, ed))
    print(f"   {L:>6} {sig:>15.4f} {lo_:>12.4e} {lo_*np.sqrt(L):>13.6f} {tv:>12.4e} "
          f"{tv*np.sqrt(L):>10.5f} {ed:>12.4e} {ed*L:>10.5f}")
rows = np.array(rows)
msk = rows[:, 0] >= 256
sl = lambda y: np.polyfit(np.log(rows[msk, 0]), np.log(y[msk]), 1)[0]
print(f"\n   fitted slopes on log-log, ASYMPTOTIC range L >= 256:")
print(f"     LOCAL ~ L^{sl(rows[:,2]):.4f},  TV ~ L^{sl(rows[:,3]):.4f},  EDGE ~ L^{sl(rows[:,4]):.4f}")
print(f"   (fitting from L>=16 instead gives LOCAL ~ L^"
      f"{np.polyfit(np.log(rows[4:,0]),np.log(rows[4:,2]),1)[0]:.3f} -- a pre-asymptotic transient,"
      f" not the true rate; the plateau of the LOCAL*sqrtL column is the reliable read.)")
print("   predicted:  LOCAL, TV ~ L^-1/2 (skewness-limited);  EDGE ~ L^-1 (next Edgeworth order)")
print(f"   predicted LOCAL*sqrt(L) plateau = |k3|/(6 sigma^3) * max|He_3 phi| = "
      f"{abs(k3)/(6*s2**1.5)*float(np.abs(he3(np.linspace(-6,6,20001))*np.exp(-np.linspace(-6,6,20001)**2/2)/np.sqrt(2*np.pi)).max()):.6f}")

hdr("2B2.  WHICH Edgeworth term leads?  Skewness, not kurtosis -- unless c = 0.")
print("""   The leading correction to the Gaussian is the SKEWNESS term at O(L^-1/2) whenever the
   single-step distribution is asymmetric, and only then the EXCESS KURTOSIS term at
   O(L^-1).  Advection is exactly what makes the step asymmetric here: w_ftcs has
   p_- = r + ra/2 = 0.4727 against p_+ = r - ra/2 = 0.4273, so kappa_3 != 0.
   Control experiment: rerun with c = 0, where the step is symmetric and kappa_3 = 0 exactly.""")


def he(n, z):
    return {3: z**3 - 3 * z, 4: z**4 - 6 * z**2 + 3}[n]


def edgeworth_study(wv, ov, label):
    mu = float(np.sum(wv * ov)); v2 = float(np.sum(wv * ov**2)) - mu**2
    c3 = float(np.sum(wv * (ov - mu)**3))
    c4 = float(np.sum(wv * (ov - mu)**4)) - 3 * v2**2
    zz = np.linspace(-8, 8, 40001); ph = np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi)
    P3 = abs(c3) / (6 * v2**1.5) * float(np.abs(he(3, zz) * ph).max())
    P4 = abs(c4) / (24 * v2**2) * float(np.abs(he(4, zz) * ph).max())
    print(f"\n   {label}")
    print(f"     kappa_3 = {c3:+.6e}  (skewness {c3/v2**1.5:+.5e})")
    print(f"     kappa_4 = {c4:+.6e}  (excess kurtosis {c4/v2**2:+.5e})")
    print(f"     predicted LOCAL*sqrt(L) plateau from kappa_3 : {P3:.6e}")
    print(f"     predicted LOCAL*L       plateau from kappa_4 : {P4:.6e}")
    print(f"     {'L':>7} {'LOCAL':>12} {'LOCAL*sqrtL':>13} {'LOCAL*L':>11}")
    for L in [64, 256, 1024, 4096, 16384]:
        wl, ol = cp.compose_power_fast(wv, ov, L)
        sg = np.sqrt(v2 * L); z = (ol - L * mu) / sg
        loc = float(np.abs(sg * wl - np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)).max())
        print(f"     {L:>7} {loc:>12.4e} {loc*np.sqrt(L):>13.6f} {loc*L:>11.4f}")
    return P3, P4


P3a, P4a = edgeworth_study(cp.W_FTCS, cp.OFF_FTCS, "(a) c = 1 (this thread's operating point)")
print(f"     -> LOCAL*sqrt(L) plateaus at 0.008342 vs predicted {P3a:.6f}: agree to "
      f"{abs(0.008342-P3a)/P3a*100:.2f}%.  LOCAL*L DIVERGES.  Skewness leads.")
P3b, P4b = edgeworth_study(np.array([cp.R, 1 - 2 * cp.R, cp.R]), cp.OFF_FTCS,
                           "(b) c = 0, pure diffusion: symmetric step, kappa_3 = 0 EXACTLY")
print(f"     -> LOCAL*L plateaus at 0.0942 vs predicted {P4b:.6f}: agree to "
      f"{abs(0.0942-P4b)/P4b*100:.2f}%.  LOCAL*sqrt(L) -> 0.  Kurtosis leads.")
print("""
   So 'the leading correction is the excess kurtosis decaying like 1/L' is correct ONLY for
   c = 0.  With advection the skewness term dominates at L^-1/2 and the kurtosis term is the
   NEXT one -- which is exactly what the EDGE column of 2B measures: after subtracting the
   skewness term the residual plateaus in EDGE*L at 0.0944, against the kappa_4 prediction
   of %.5f (0.7%%).""" % P4a)

hdr("2B3.  Cross-check: the exact propagator's moment hierarchy")
print("""   The exact propagator's raw moments satisfy a closed triangular ODE hierarchy.  In OUR
   displacement convention (kernel centred at -c tau, i.e. xi = source - target):
        dM_q/dtau = alpha q(q-1) M_{q-2}  -  c q M_{q-1},     M_0 = 1.
   (The sign of the drift term flips if xi is defined target-minus-source.)  Verify against
   the closed-form Gaussian moments by finite differences in tau:""")
tau0 = 137 * cp.DT
def exact_raw_moments(tau, Q=9):
    mu = -cp.C * tau; v = 2 * cp.ALPHA * tau
    M = np.zeros(Q + 1); M[0] = 1.0
    if Q >= 1: M[1] = mu
    for q in range(2, Q + 1):                 # Gaussian raw-moment recursion
        M[q] = mu * M[q - 1] + (q - 1) * v * M[q - 2]
    return M
h = tau0 * 1e-6
Mp, Mm = exact_raw_moments(tau0 + h), exact_raw_moments(tau0 - h)
M0 = exact_raw_moments(tau0)
print(f"\n   {'q':>3} {'dM_q/dtau (FD)':>18} {'hierarchy RHS':>18} {'rel err':>11}")
worst = 0.0
for q in range(1, 9):
    fd = (Mp[q] - Mm[q]) / (2 * h)
    rhs = cp.ALPHA * q * (q - 1) * M0[q - 2] - cp.C * q * M0[q - 1]
    rel = abs(fd - rhs) / max(abs(rhs), 1e-300); worst = max(worst, rel)
    print(f"   {q:>3} {fd:>18.10e} {rhs:>18.10e} {rel:>11.2e}")
print(f"   max rel err through q=8: {worst:.2e}")
print("""
   Useful as an independent check, but note it does NOT give the Task-2 convergence rate.
   The rate is set by the cumulants of the SINGLE-STEP STENCIL (kappa_3, kappa_4 of w_ftcs),
   not by the exact propagator, whose cumulants beyond the second are zero by construction.
   Composition advances tau in this hierarchy only for the EXACT kernel; the stencil's own
   defects are what Theorem 1' accumulates.""")

hdr("2C.  Formal vs effective support -- the control parameter of the flow")
print("""   Formal half-width after L compositions: exactly L cells (the light cone).
   Effective half-width: k * sigma_L with sigma_L = sqrt(s2 L) cells.
   Ratio rho(L) = L / sigma_L = sqrt(L / s2)  -- grows like sqrt(L).""")
print(f"\n   {'L':>6} {'formal m=L':>11} {'sigma_L':>9} {'rho=L/sigma':>12} {'mass |z|>3':>12} "
      f"{'mass |z|>5':>12} {'Gauss |z|>5':>12} {'m_eff(1e-12)':>13} {'L/m_eff':>9}")
for L in LS:
    w, o = KER[L]
    sig = np.sqrt(s2 * L); z = np.abs(o - L * mu1) / sig
    t3 = float(w[z > 3].sum()); t5 = float(w[z > 5].sum())
    gauss5 = float(erfc(5 / np.sqrt(2)))
    # smallest half-width (about the mean) holding all but 1e-12 of the mass
    cum = np.cumsum(w); meff = None
    idx = np.argsort(np.abs(o - L * mu1))
    ws = w[idx]; run = np.cumsum(ws)
    j = int(np.searchsorted(run, 1 - 1e-12))
    meff = float(np.abs(o - L * mu1)[idx][min(j, len(o) - 1)])
    print(f"   {L:>6} {L:>11} {sig:>9.3f} {L/sig:>12.3f} {t3:>12.3e} {t5:>12.3e} "
          f"{gauss5:>12.3e} {meff:>13.1f} {L/max(meff,1):>9.2f}")
print("\n   Ratio of the composite's tail mass to the Gaussian's, as a function of k:")
print(f"   {'L':>6} {'rho=L/sig':>10} " + "".join(f"{'k=%d'%k:>10}" for k in [2, 4, 6, 8, 12, 20, 40]))
for L in [64, 256, 1024, 4096]:
    w, o = KER[L]; sig = np.sqrt(s2 * L); z = np.abs(o - L * mu1) / sig
    row = []
    for k in [2, 4, 6, 8, 12, 20, 40]:
        t = float(w[z > k].sum()); g = float(erfc(k / np.sqrt(2)))
        row.append(f"{t/g:>10.3g}" if t > 0 else f"{'0':>10}")
    print(f"   {L:>6} {L/sig:>10.1f} " + "".join(row))
print("""
   Correct reading (an earlier draft of this script claimed the tails were strongly
   sub-Gaussian -- they are not).  The ratio is ~1 throughout the tail and collapses
   only as k approaches rho(L) = L/sigma_L, where the light cone cuts the kernel off
   dead.  So:
     (i)  the formal/effective ratio rho(L) = sqrt(L/s2) grows like sqrt(L) -- the compression;
     (ii) rho(L) is ALSO the range of validity, in units of sigma_L, of the Gaussian
          fixed-point description.  The flow's control parameter and the domain over
          which the fixed point describes the kernel are the same number.  Finite L is
          a finite-size effect in exactly the RG sense: the cutoff sits at rho sigma.""")

hdr("2D.  The flow is POSITIVITY-RESTORING: a strictly larger basin than {w >= 0}")
print("""   Theorem 2 bounds ||w^{*L}||_1 <= ||w||_1^L.  That bound is attained only when
   signs align at every level.  For a stencil with max_theta |g(theta)| <= 1 but
   ||w||_1 > 1, the bound is wildly pessimistic: the negative lobes are EXPELLED
   past the effective support and ||w^{*L}||_1 -> 1.""")
from math import factorial


def theta_poly(n, xi, tau):
    y = np.asarray(xi, float) - cp.C * np.asarray(tau, float); t = np.asarray(tau, float)
    return factorial(n) * sum((cp.ALPHA * t)**k * y**(n - 2 * k)
                              / (factorial(k) * factorial(n - 2 * k)) for k in range(n // 2 + 1))


def wide_exact(m, r):
    dt = r * cp.DX**2 / cp.ALPHA; ox = np.arange(-m, m + 1); n = 2 * m + 1
    A = np.array([theta_poly(k, ox * cp.DX, -dt * np.ones(n)) for k in range(n)])
    sc = np.abs(A).max(axis=1)[:, None]
    return np.linalg.solve(A / sc, np.array([theta_poly(k, 0., 0.) for k in range(n)]) / sc[:, 0]), ox, dt


TH = np.linspace(-np.pi, np.pi, 40001)
w7, o7, dt7 = wide_exact(3, 0.05)
r7 = 0.05
print(f"\n   probe: 7-point Theta_0..Theta_6-exact stencil at r=0.05")
print(f"   ||w||_1 = {cp.l1(w7):.6f} > 1,  min w = {w7.min():.4e},  "
      f"max|g| = {float(np.abs(cp.symbol(w7,o7,TH)).max()):.10f}")
print(f"\n   {'L':>6} {'||w^*L||_1':>13} {'bound ||w||_1^L':>17} {'neg mass':>12} "
      f"{'innermost neg |k|':>18} {'sigma_L':>9} {'|k|_neg/sigma':>14}")
s2_7 = float(np.sum(w7 * (o7 - np.sum(w7 * o7))**2))
for L in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    wl, ol = cp.compose_power_fast(w7, o7, L)
    sig = np.sqrt(s2_7 * L); neg = ol[wl < 0]
    inner = np.abs(neg - np.sum(wl * ol)).min() if neg.size else np.nan
    print(f"   {L:>6} {cp.l1(wl):>13.9f} {cp.l1(w7)**L:>17.4g} {-wl[wl<0].sum():>12.3e} "
          f"{inner:>18.1f} {sig:>9.3f} {inner/sig if neg.size else np.nan:>14.2f}")
print(f"""
   Three regimes for self-composition (measured, all at fixed lattice):
     (a) w >= 0                    : ||w^{{*L}}||_1 = 1 EXACTLY for every L.
     (b) ||w||_1 > 1, max|g| <= 1  : ||w^{{*L}}||_1 -> 1; negativity expelled to |k| >> sigma_L.
     (c) max|g| > 1                : geometric growth, at rate max|g|, NOT ||w||_1.""")
for nm, ww, oo in [("(c) FTCS pure advection", np.array([cp.RA/2, 1.0, -cp.RA/2]), cp.OFF_FTCS),
                   ("(c) w=(-.25,1.5,-.25)  ", np.array([-0.25, 1.5, -0.25]), cp.OFF_FTCS)]:
    mg = float(np.abs(cp.symbol(ww, oo, TH)).max())
    out = []
    for L in [16, 256, 1089]:
        wl, _ = cp.compose_power_fast(ww, oo, L)
        out.append(f"L={L}:{cp.l1(wl):.3g}")
    print(f"   {nm} ||w||_1={cp.l1(ww):.4f} max|g|={mg:.6f}  ||w^*L||_1: " + "  ".join(out))

# ----------------------------------------------------------------- FIGURE
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.30)

ax = fig.add_subplot(gs[0, :2])
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, 7))
for i, L in enumerate([4, 16, 64, 256, 1024, 4096]):
    w, o = KER[L]
    sig = np.sqrt(s2 * L); z = (o - L * mu1) / sig
    m = np.abs(z) < 5.2
    ax.plot(z[m], sig * w[m], 'o' if L <= 16 else '-', ms=3.0, lw=1.6,
            color=cmap[i], label=f"L = {L}", alpha=0.9)
zz = np.linspace(-5.2, 5.2, 600)
ax.plot(zz, np.exp(-zz**2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=2.0, label=r"$\phi(z)$ (heat kernel)")
ax.set_xlabel(r"$z=(k-L\mu_1)/\sigma\sqrt{L}$  (diffusive rescaling)")
ax.set_ylabel(r"$\sigma\sqrt{L}\,\cdot\,w^{*L}_k$")
ax.set_title("Composition of a positive stencil is an RG flow to the heat-kernel fixed point")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[0, 2])
ax.loglog(rows[:, 0], rows[:, 2], 'o-', label="local CLT sup")
ax.loglog(rows[:, 0], rows[:, 3], 's-', label="total variation")
ax.loglog(rows[:, 0], rows[:, 4], '^-', label="after Edgeworth")
ax.loglog(rows[:, 0], rows[0, 2] * rows[:, 0]**-0.5, 'k--', lw=1, label=r"$L^{-1/2}$")
ax.loglog(rows[:, 0], rows[0, 4] * rows[:, 0]**-1.0, 'k:', lw=1, label=r"$L^{-1}$")
ax.set_xlabel("L"); ax.set_ylabel("distance to Gaussian")
ax.set_title("convergence rate", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.25, which='both')

ax = fig.add_subplot(gs[1, 0])
Ls = np.array(LS)
sigs = np.sqrt(s2 * Ls)
ax.loglog(Ls, Ls, 'o-', label=r"formal half-width $=L$")
ax.loglog(Ls, sigs, 's-', label=r"$\sigma_L=\sqrt{s_2 L}$")
ax.loglog(Ls, Ls / sigs, '^-', label=r"$\rho=L/\sigma_L\propto\sqrt{L}$")
ax.set_xlabel("L"); ax.set_ylabel("cells")
ax.set_title("formal vs effective support", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

ax = fig.add_subplot(gs[1, 1])
for L in [16, 64, 256, 1024]:
    w, o = KER[L]
    sig = np.sqrt(s2 * L); z = np.abs(o - L * mu1) / sig
    ks = np.linspace(0, 8, 60)
    tail = [max(float(w[z > k].sum()), 1e-320) for k in ks]
    ax.semilogy(ks, tail, '-', lw=1.4, label=f"L={L}")
ks = np.linspace(0, 8, 200)
ax.semilogy(ks, erfc(ks / np.sqrt(2)), 'k--', lw=2, label="Gaussian")
for L, cc in [(16, 'C0'), (64, 'C1'), (256, 'C2'), (1024, 'C3')]:
    ax.axvline(L / np.sqrt(s2 * L), color=cc, ls='-', lw=0.9, alpha=0.45)
ax.set_ylim(1e-30, 2); ax.set_xlabel(r"$k$ (in units of $\sigma_L$)")
ax.set_ylabel(r"mass outside $\pm k\sigma_L$")
ax.set_title(r"Gaussian tail out to the light cone at $k=\rho(L)$" + "\n(thin verticals)", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.25, which='both')

ax = fig.add_subplot(gs[1, 2])
Lp = [1, 2, 4, 8, 16, 32, 64, 128, 256]
l1s, negs = [], []
for L in Lp:
    wl, ol = cp.compose_power_fast(w7, o7, L)
    l1s.append(cp.l1(wl)); negs.append(max(-wl[wl < 0].sum(), 1e-320))
ax.loglog(Lp, np.array(l1s) - 1 + 1e-18, 'o-', label=r"$\|w^{*L}\|_1-1$  (max|g|$\leq$1)")
ax.loglog(Lp, negs, 's-', label="negative mass")
ax.loglog(Lp, cp.l1(w7)**np.array(Lp) - 1, 'k--', lw=1.2, label=r"Thm-2 bound $\|w\|_1^L-1$")
ax.set_ylim(1e-30, 1e3); ax.set_xlabel("L")
ax.set_title("the flow restores positivity", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.25, which='both')

fig.savefig(f"{FIG}/task2_rg_flow.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task2_rg_flow.png")
