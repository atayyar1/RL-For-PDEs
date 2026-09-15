"""
task7_bigstep.py -- the integrator built from MOMENT rows rather than Taylor rows.

Rows:  sum_i w_i dx_i^q = Mfrak_q(tau),  q = 0..p,   with w >= 0.

The Taylor rows of RESULTS.md are the q<=2 truncation with c^2 tau^2 dropped, so
they are only valid for tau ~ h^2/alpha.  The moment rows are exact at every tau,
which is what makes a single giant step possible.

Tested on a PERIODIC domain first (where the translation-invariant propagator is
exact), then on the Dirichlet problem, where it is not -- that distinction turns
out to matter a great deal and is reported rather than hidden.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J, moments as MO

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
C, AL = 1.0, 0.1
# sigma at k CFL steps is dx*sqrt(0.9k) cells, so 6 sigma needs ~114 cells at
# k=400 REGARDLESS of dx: the domain must hold >228 cells or the stencil is
# truncated and the whole comparison is an artefact of that truncation.
NX = 512; DX = 1.0 / NX; X = np.arange(NX) * DX
DT = min(0.9 * DX / C, 0.45 * DX ** 2 / AL)

rng = np.random.default_rng(0)
KM = np.arange(1, 9); AMP = KM ** -1.5; PH = rng.uniform(0, 2 * np.pi, len(KM))
def u_per(t):
    """Exact periodic solution of u_t + c u_x = alpha u_xx."""
    return (np.cos(2 * np.pi * np.outer(X - C * t, KM) + PH)
            * (AMP * np.exp(-AL * (2 * np.pi * KM) ** 2 * t))).sum(-1)

print("=" * 88); print("TASK 7 -- BIG-STEP INTEGRATOR FROM MOMENT ROWS"); print("=" * 88)
PBEST = 8
print(f"  nx={NX} dx={DX:.5g}  CFL dt={DT:.5g}  (c={C}, alpha={AL})")

def ftcs_step(u, dt):
    r = AL * dt / DX ** 2; ra = C * dt / DX
    return (r + ra / 2) * np.roll(u, 1) + (1 - 2 * r) * u + (r - ra / 2) * np.roll(u, -1)

def big_stencil(tau, p, nsig=6.0, nonneg=True):
    """Moment-matched stencil for a lag tau, matching moments q=0..p."""
    sig = np.sqrt(2 * AL * tau)
    R = int(np.ceil((C * tau + nsig * sig) / DX))
    if R > NX // 2 - 1:
        return None, None, R          # kernel does not fit: refuse, do not truncate
    
    offs = np.arange(-R, R + 1) * DX
    Mf = MO.moments_from_symbol({1: -C, 2: AL}, tau, p)
    w = (MO.w_moment_positive_smooth(offs, Mf, p) if nonneg
         else MO.w_moment_minnorm(offs, Mf, p))
    return w, np.arange(-R, R + 1), R

def apply_stencil(u, w, ioff):
    out = np.zeros_like(u)
    for wi, o in zip(w, ioff):
        out += wi * np.roll(u, -o)
    return out

# ------------------------------------------- 1. periodic: accuracy and feasibility
print("\n--- 1. PERIODIC domain: one big step vs k FTCS steps ---")
print(f"  {'k':>6} {'tau':>9} {'R':>5} {'pts':>5} {'p':>3} {'w>=0?':>6} "
      f"{'big-step err':>14} {'FTCS err':>12} {'ratio':>8}")
rows = []
for k in [10, 50, 100, 200, 400]:
    tau = k * DT
    u0 = u_per(0.0); ue = u_per(tau)
    uf = u0.copy()
    for _ in range(k):
        uf = ftcs_step(uf, DT)
    e_ftcs = np.max(np.abs(uf - ue))
    for p in [2, 4, 6, 8]:
        w, ioff, R = big_stencil(tau, p)
        if w is None:
            why = "TOOBIG" if R > NX // 2 - 1 else "INFEAS"
            print(f"  {k:>6} {tau:>9.5f} {R:>5} {'-':>5} {p:>3} {why:>6}")
            continue
        ub = apply_stencil(u0, w, ioff)
        e_big = np.max(np.abs(ub - ue))
        if p == PBEST:
            rows.append((k, tau, 2 * R + 1, e_big, e_ftcs))
        print(f"  {k:>6} {tau:>9.5f} {R:>5} {2*R+1:>5} {p:>3} {'yes':>6} "
              f"{e_big:>14.4e} {e_ftcs:>12.4e} {e_ftcs/e_big:>8.1f}x")

# --------------------------------------------------------------- 2. honest cost
print("\n--- 2. COST, counted honestly ---")
print(f"  {'k':>6} {'stencil pts':>12} {'big: mults/pt':>15} {'FTCS: mults/pt':>16} "
      f"{'full-grid speedup':>18} {'1-point cone':>14}")
for (k, tau, npts, eb, ef) in rows:
    big = npts
    ftcs = 3 * k
    cone = sum(min(NX, 2 * (k - j) + 1) for j in range(k))
    print(f"  {k:>6} {npts:>12} {big:>15} {ftcs:>16} {ftcs/big:>17.1f}x "
          f"{cone/big:>13.0f}x")
print("""  Two different speedups, and they must not be conflated:
    * FULL-GRID advance: both methods touch every point, so the ratio is just
      3k / npts -- a single-digit to ~10x win, and FFT convolution would do
      better still.
    * ONE point at the horizon: FD must fill the whole numerical cone, the
      stencil does not.  That is where the hundreds-fold figure comes from.
      It is real but it answers a different question.""")

# ------------------------------------- 3. does it survive DISCOVERED coefficients?
print("\n--- 3. rows built from DISCOVERED coefficients (task 6 accuracy levels) ---")
print(f"  {'eta':>8} {'c_hat':>10} {'alpha_hat':>11} {'k=400 big-step err':>20} {'vs true rows':>14}")
w, ioff, R = big_stencil(400 * DT, 8)
e_ref = np.max(np.abs(apply_stencil(u_per(0.0), w, ioff) - u_per(400 * DT)))
for eta, ch, ah in [(0.0, 1.0000000, 0.10000000), (1e-6, 1.0000000, 0.09999900),
                    (1e-4, 1.0000700, 0.10000400), (1e-3, 0.9993200, 0.09947300),
                    (1e-2, 0.9979700, 0.09905500)]:
    tau = 400 * DT
    sig = np.sqrt(2 * ah * tau)
    Rr = int(np.ceil((ch * tau + 6 * sig) / DX))
    offs = np.arange(-Rr, Rr + 1) * DX
    Mf = MO.moments_from_symbol({1: -ch, 2: ah}, tau, 8)
    wr = MO.w_moment_positive_smooth(offs, Mf, 8)
    if wr is None:
        print(f"  {eta:>8.0e} {ch:>10.6f} {ah:>11.6f}  no nonnegative stencil")
        continue
    e = np.max(np.abs(apply_stencil(u_per(0.0), wr, np.arange(-Rr, Rr + 1)) - u_per(tau)))
    print(f"  {eta:>8.0e} {ch:>10.6f} {ah:>11.6f} {e:>20.4e} {e/e_ref:>13.1f}x")

# ------------------------------------------------- 4. the Dirichlet reality check
print("\n--- 4. DIRICHLET domain: where the big step actually breaks ---")
sold = J.AdvDiff(alpha=AL, c=C)
xd = np.linspace(0, 1, NX + 1)
tau = 400 * DT
sig = np.sqrt(2 * AL * tau)
reach = C * tau + 4 * sig
print(f"  tau={tau:.4f}: drift c*tau={C*tau:.4f}, sigma={sig:.4f}, "
      f"kernel reach ~{reach:.3f} of a unit domain")
w, ioff, R = big_stencil(tau, 8)
u0 = sold.u(xd, 0.10)
ue = sold.u(xd, 0.10 + tau)
ub = np.zeros_like(u0)
for wi, o in zip(w, ioff):
    idx = np.clip(np.arange(len(xd)) + o, 0, len(xd) - 1)
    ub += wi * np.where((np.arange(len(xd)) + o < 0) |
                        (np.arange(len(xd)) + o >= len(xd)), 0.0, u0[idx])
err = np.abs(ub - ue)
print(f"  {'distance from boundary':>26} {'max error':>12}")
for d in [0.0, 0.1, 0.2, 0.3, 0.4]:
    m = (xd >= d) & (xd <= 1 - d)
    print(f"  {('> ' + str(d)):>26} {err[m].max():>12.3e}")
print(f"""
  The translation-invariant propagator is NOT the Dirichlet Green's function:
  it needs method-of-images corrections. With a kernel reaching {reach:.2f} on a
  unit domain, NO interior point is more than a few sigma from a boundary, so a
  single 400-step jump is simply not available for this BVP. The clean big-step
  result of section 1 is a periodic/free-space result and must be labelled as
  such. Boundaries, not stability, are the binding constraint on step size.""")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ks = [r[0] for r in rows]
ax[0].loglog(ks, [r[3] for r in rows], "o-", label="one moment-row big step ($p$=8)")
ax[0].loglog(ks, [r[4] for r in rows], "s--", label="$k$ FTCS steps")
ax[0].set_xlabel("$k$ (CFL steps spanned)"); ax[0].set_ylabel("max error")
ax[0].set_title("Periodic domain: one big step vs FTCS", fontsize=10)
ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
ax[1].semilogy(xd, err + 1e-18, lw=1.2)
ax[1].set_xlabel("$x$"); ax[1].set_ylabel("error of one big step")
ax[1].set_title("Dirichlet domain: boundaries dominate", fontsize=10)
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task7_bigstep.png"), dpi=150)
print("\n  figure -> figures/fig_task7_bigstep.png")
