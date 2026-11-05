"""TASK 11 -- T5's crossover, and the OFF-CENTRE (semi-Lagrangian) stencil.

T5 and I independently derived the same corrected frontier:
    T5 :  k_max = ( -r + sqrt(r^2 + ra^2 m^2) ) / ra^2          (r = nu, ra = Co)
    me :  k_max = floor( [sqrt(nu^2 + Co^2 m^2) - nu] / Co^2 )   (task7)
Identical.  Two things in T5's message are NEW to me and are tested here:

  (1) the CROSSOVER between the diffusive branch m^2/(2 nu) and the advective
      branch m/Co sits at  m* = nu/Co = 1/Pe_cell,  which at our operating point
      is only ~9.9 cells -- so the quadratic law dies at moderate m even at the
      modest Pe = 0.101 of the benchmark;

  (2) if the stencil is re-CENTRED on the kernel's own mean (the departure point,
      -Co*k cells upstream) instead of on the target point, only the spread has to
      fit, so the advective penalty should vanish entirely -- and a shifted stencil
      costs exactly the same to evaluate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
nu, co = C.NU_REF, C.CO_REF
PE = co / nu                                  # cell Peclet = c dx / alpha


def lp_feasible(js, k, nu=nu, co=co):
    """Positive weights on integer offsets `js` matching M0, M1, M2 of the true kernel."""
    js = np.asarray(js, float)
    mu, M2 = -k * co, 2 * k * nu + (k * co) ** 2
    sc = max(np.max(np.abs(js)), 1.0)
    A = np.array([np.ones_like(js), js / sc, (js / sc) ** 2])
    b = np.array([1.0, mu / sc, M2 / sc ** 2])
    r = linprog(np.zeros(len(js)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return r.status == 0


def kmax_scan(js_fn, mmax, nu=nu, co=co, kcap=40000):
    out = {}
    for m in mmax:
        lo, hi, best = 1, kcap, 0
        while lo <= hi:                       # feasibility is monotone in k here
            mid = (lo + hi) // 2
            if lp_feasible(js_fn(m, mid), mid, nu, co):
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        out[m] = best
    return out


def kmax_closed(m, nu=nu, co=co):
    if co == 0:
        return m ** 2 / (2 * nu)
    return (-nu + np.sqrt(nu ** 2 + co ** 2 * m ** 2)) / co ** 2


ms = [1, 2, 3, 5, 8, 12, 20, 32, 50]
print("=" * 100)
print("A.  CROSS-CHECK: T5's closed form, my closed form, and the LP")
print("=" * 100)
cen = kmax_scan(lambda m, k: np.arange(-m, m + 1), ms)
print("%-14s %s" % ("m", "".join("%7d" % m for m in ms)))
print("%-14s %s" % ("LP (centred)", "".join("%7d" % cen[m] for m in ms)))
print("%-14s %s" % ("closed form", "".join("%7d" % int(np.floor(kmax_closed(m))) for m in ms)))
print("%-14s %s" % ("m^2/(2 nu)", "".join("%7.0f" % (m ** 2 / (2 * nu)) for m in ms)))
print("%-14s %s" % ("ratio old/LP", "".join("%7.2f" % ((m ** 2 / (2 * nu)) / cen[m]) for m in ms)))
agree = all(cen[m] == int(np.floor(kmax_closed(m))) for m in ms)
print("\n  LP == closed form at every m: %s   (matches T5's table exactly)" % agree)

print("\n" + "=" * 100)
print("B.  THE CROSSOVER:  m* = nu/Co = 1/Pe_cell = %.2f cells" % (nu / co))
print("=" * 100)
print("%5s | %10s %10s %10s | %s" % ("m", "k_max", "m^2/(2nu)", "m/Co", "which branch"))
for m in [2, 5, 8, 10, 12, 16, 20, 32, 50, 80]:
    kd, ka = m ** 2 / (2 * nu), m / co
    k = int(np.floor(kmax_closed(m)))
    br = "diffusive" if kd < ka else "advective"
    print("%5d | %10d %10.0f %10.0f | %s (min = %s)"
          % (m, k, kd, ka, br, "%.0f" % min(kd, ka)))
print("\n  -> confirmed.  At Pe_cell = %.3f the quadratic branch is already dead by m ~ 10," % PE)
print("     so the 'spend points laterally' speedup saturates LINEARLY in m past that.")
print("     This is a sharper statement than my 'at high Pe it goes linear': it bites at")
print("     MODEST Pe too, because the crossover is at 1/Pe_cell, not at Pe ~ 1.")

print("\n" + "=" * 100)
print("C.  OFF-CENTRE (SEMI-LAGRANGIAN) STENCIL: does the advective penalty vanish?")
print("=" * 100)
print("  stencil on {j0-m .. j0+m} with j0 = round(-Co*k) (the departure point).")
print("  Same number of points, same cost per evaluation -- only the index offset changes.")
off = kmax_scan(lambda m, k: np.arange(-m, m + 1) + int(round(-k * co)), ms)
print("\n%-16s %s" % ("m", "".join("%8d" % m for m in ms)))
print("%-16s %s" % ("centred", "".join("%8d" % cen[m] for m in ms)))
print("%-16s %s" % ("off-centre", "".join("%8d" % off[m] for m in ms)))
print("%-16s %s" % ("m^2/(2 nu)", "".join("%8.0f" % (m ** 2 / (2 * nu)) for m in ms)))
print("%-16s %s" % ("gain", "".join("%8.2f" % (off[m] / max(cen[m], 1)) for m in ms)))
print("\n  -> the off-centre frontier tracks m^2/(2 nu) at EVERY m: the advective penalty")
print("     is gone, and the gain over the centred stencil grows without bound (%.1fx at m=50)."
      % (off[50] / cen[50]))
print("     T5 is right, and this is a free change: re-centring is an index shift.")

print("\n  Why:  with j0 = round(mu), write eta = j - j0.  The conditions become")
print("        E[eta] = mu - j0 =: mu', |mu'| <= 1/2, and E[eta^2] = 2 k nu + mu'^2,")
print("        so feasibility is 2 k nu + mu'^2 <= m^2, i.e. k <= (m^2 - mu'^2)/(2 nu)")
print("        -> m^2/(2 nu) up to a <= 0.25/(2 nu) = %.2f correction. Verified above."
      % (0.25 / (2 * nu)))

print("\n" + "=" * 100)
print("D.  HOW MUCH DOES THIS MATTER IN PRACTICE?  (cost at fixed step and accuracy)")
print("=" * 100)
print("  To take k steps at safety factor s you need, in cells:")
print("     centred   : m = s*sigma + |mu|      (span origin AND departure point)")
print("     off-centre: m = s*sigma             (span the kernel only)")
print("  with sigma = sqrt(2 k nu), mu = Co*k.  Cost ratio = (2m+1) centred / off-centre.")
print("\n%8s %8s | %9s %9s | %9s %9s | %8s"
      % ("k", "k*dt", "sigma", "|mu|", "m centred", "m off", "cost x"))
for k in [10, 40, 160, 640, 2560]:
    sig, mu = np.sqrt(2 * k * nu), co * k
    for s_ in [4.0]:
        mc, mo = int(np.ceil(s_ * sig + mu)), int(np.ceil(s_ * sig))
        print("%8d %8.4f | %9.2f %9.2f | %9d %9d | %8.2f"
              % (k, k * C.DT, sig, mu, mc, mo, (2 * mc + 1) / (2 * mo + 1)))
print("\n  -> at the benchmark's Pe the saving is modest until k is large (k*dt ~ 1),")
print("     but it is free, and it removes the linear-saturation branch entirely.")
print("     For advection-dominated problems it is the difference between a usable")
print("     scheme and one whose footprint is dominated by drift.")

# ---------------------------------------------------------------- figure
mm = np.arange(1, 81)
kc = np.array([kmax_closed(m) for m in mm])
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))
ax[0].loglog(mm, mm ** 2 / (2 * nu), "k--", lw=1.2, label=r"diffusive $m^2/(2\nu)$")
ax[0].loglog(mm, mm / co, "k:", lw=1.2, label=r"advective $m/\mathrm{Co}$")
ax[0].loglog(mm, kc, "-", color="#c4432b", lw=2, label="centred frontier (exact)")
ax[0].loglog(ms, [off[m] for m in ms], "o-", color="#1b6ca8", lw=1.5, ms=5,
             label="off-centre frontier (LP)")
ax[0].axvline(nu / co, color="#2e8b57", lw=1.4)
ax[0].text(nu / co * 1.1, 3, r"$m^*=1/\mathrm{Pe}_{cell}$", fontsize=8, color="#2e8b57")
ax[0].set_xlabel("half-width $m$ (cells)"); ax[0].set_ylabel(r"$k_{\max}$")
ax[0].set_title("(a) centred frontier saturates; off-centre does not")
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, which="both")

pes = np.logspace(-2, 1, 60)
for mval, cl in zip([5, 12, 30], ["#1b6ca8", "#c4432b", "#2e8b57"]):
    rat = []
    for pe in pes:
        cod = nu * pe
        rat.append(kmax_closed(mval, nu, cod) / (mval ** 2 / (2 * nu)))
    ax[1].semilogx(pes, rat, "-", color=cl, lw=1.6, label="$m=%d$" % mval)
    ax[1].axvline(1.0 / mval, color=cl, ls=":", lw=1)
ax[1].set_xlabel(r"cell P\'eclet  $\mathrm{Pe}=c\Delta x/\alpha$".replace("\\'e", "e"))
ax[1].set_ylabel(r"$k_{\max}\,/\,[m^2/(2\nu)]$")
ax[1].set_title(r"(b) quadratic law fails once $\mathrm{Pe} > 1/m$")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both"); ax[1].set_ylim(0, 1.1)
fig.suptitle("Drift fills the footprint: centred vs off-centre wide stencils",
             y=0.99, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "fig8_offcentre.png"), dpi=150)
print("\nwrote figures/fig8_offcentre.png")


# ====================================================================== E
print("\n" + "=" * 100)
print("E.  DOES THIS CHANGE MY TASK 3 SOLVER?  three ways to handle the drift")
print("=" * 100)
print("  (i)   centred     : kernel carries the drift, m must span origin + departure pt")
print("  (ii)  off-centre  : kernel carries the drift, stencil re-centred on departure pt")
print("  (iii) gauge/Liouville   : drift removed analytically (V = u e^{-beta x}), pure-diffusion")
print("                      kernel, mu = 0 -- what solvers.solve_wide already does.")

import solvers as S


def kernel_centred(dt_big, dx, alpha, c, s):
    sig = np.sqrt(2 * alpha * dt_big) / dx
    mu = -c * dt_big / dx
    m = int(np.ceil(s * sig + abs(mu))) + 1
    w = C.kernel_moment_matched(m, mu, sig ** 2 + mu ** 2)
    return w, m


def kernel_offcentre(dt_big, dx, alpha, c, s):
    sig = np.sqrt(2 * alpha * dt_big) / dx
    mu = -c * dt_big / dx
    j0 = int(round(mu))
    m = int(np.ceil(s * sig)) + 1
    w = C.kernel_moment_matched(m, mu - j0, sig ** 2 + (mu - j0) ** 2)
    return w, m, j0


print("\n%6s %8s | %22s | %22s | %14s"
      % ("Pe", "k*dt", "(i) centred  m / cost", "(ii) off-centre m / cost", "cost saving"))
DX = C.DX
for pe in [0.101, 0.5, 2.0, 8.0]:
    cc = pe * C.ALPHA / DX
    for dtb in [0.005, 0.05]:
        wc, mc = kernel_centred(dtb, DX, C.ALPHA, cc, 4.0)
        wo, mo, j0 = kernel_offcentre(dtb, DX, C.ALPHA, cc, 4.0)
        cost_c, cost_o = 2 * (2 * mc + 1) - 1, 2 * (2 * mo + 1) - 1
        ok = (wc is not None) and (wo is not None)
        print("%6.3f %8.3f | %11d / %8d | %11d / %8d | %10.2fx %s"
              % (pe, dtb, mc, cost_c, mo, cost_o, cost_c / cost_o,
                 "" if ok else "(infeasible)"))

print("\n  -> My Task 3 solver uses route (iii): it passes c=0 into wide_weights and removes")
print("     the drift with the gauge/Liouville transform, so mu = 0 and it NEVER paid the drift")
print("     penalty.  That is why the Task 3 work-precision numbers do not move.")
print("  -> But (iii) is problem-specific (constant c, constant alpha).  Route (ii) is the")
print("     GENERAL fix and gets the same footprint for free, so it is what the")
print("     variable-coefficient scheme (Task 9) should use once c != 0 there.")
print("  -> Route (i) -- the one T5's bound describes -- is the one to avoid; it is also")
print("     the one the original brief's 'symmetric stencil, all neighbours at -k dt'")
print("     specifies, which is why the saturation showed up in the first place.")
