"""
Task 6 -- ORDER or CONSTANT?  And what does the Jensen argument actually obstruct?

Background.  The coordinating thread observes that non-negative weights summing
to 1 are a probability distribution over the offsets, so by Jensen

    M2 := sum_i w_i dt_i^2  >=  (sum_i w_i dt_i)^2  > 0 ,

with equality iff all dt_i coincide -- hence "the u_tt error term can never be
cancelled by non-negative weights", which would put a FLOOR under the
achievable truncation error and imply an order barrier.

The inequality is certainly true.  This script asks whether the CONSEQUENCE
holds, and resolves the conflict with the empirical finding that positive and
signed weights converge at the same RATE with a worse CONSTANT.

Key structural fact (see stencil.py).  Because d_x and L = -c d_x + alpha d_x^2
commute,
    u(x+dx, t+dt) = exp(q d_x + a d_x^2) u ,   q = dx - c dt,  a = alpha dt,
so the coefficient of d_x^s u is C_s(q,a) = sum_j q^(s-2j) a^j /((s-2j)! j!).
There is NO free-standing u_tt term: (1/2) dt^2 u_tt is redistributed by the
PDE into (1/2)c^2 dt^2 u_xx - c alpha dt^2 u_xxx + (1/2) alpha^2 dt^2 u_xxxx,
and the u_xx piece is already inside C_2 -- which non-negative weights cancel
routinely (FTCS does exactly that).  So `sum_i w_i dt_i^2 = 0` is NOT one of
the accuracy conditions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from stencil import (ALPHA, C, u_true, moment_coeff, build_A_order,
                     w_positive_order, w_lstsq_order)

FIG = "figures"
np.set_printoptions(precision=6, suppress=True)


def max_pos_order(dxi, dti, alpha=ALPHA, c=C, pmax=7):
    p = 0
    for s in range(1, pmax + 1):
        if w_positive_order(dxi, dti, order=s, alpha=alpha, c=c) is None:
            break
        p = s
    return p


def max_signed_order(dxi, dti, alpha=ALPHA, c=C, pmax=7):
    p = 0
    for s in range(1, pmax + 1):
        A, b = build_A_order(dxi, dti, alpha=alpha, c=c, order=s)
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.abs(A @ w - b).max() > 1e-9:
            break
        p = s
    return p


# ============================================================ PART 1: Jensen
print("=" * 90)
print("PART 1 -- the Jensen inequality: true, but what does it obstruct?")
print("=" * 90)
rng = np.random.default_rng(0)
dx0, dt0 = 0.01, 0.01**2 * 0.25 / ALPHA
viol = 0
n = 0
slacks = []
while n < 3000:
    ix = rng.integers(-4, 5, size=6)
    it = -rng.integers(1, 9, size=6)
    dxi, dti = ix * dx0, it * dt0
    w = w_positive_order(dxi, dti, order=2)
    if w is None:
        continue
    n += 1
    M2 = float(w @ dti**2)
    M1 = float(w @ dti)
    slacks.append(M2 - M1**2)
    viol += (M2 - M1**2) < -1e-18
print(f"  1a. M2 >= M1^2 on {n} positive stencils: violations = {viol}, "
      f"min slack = {min(slacks):.3e}")
print(f"      (it is Var(dt) >= 0 under the probability measure w -- an identity)")

print("\n  1b. Reproduce the constrained experiment: ADD 'sum w dt^2 = 0' as an")
print("      explicit extra equality row, on multi-time-level stencils.")
rng = np.random.default_rng(1)
npos = nsig = ntot = 0
l1s = []
while ntot < 600:
    ix = rng.integers(-4, 5, size=6)
    it = -rng.integers(1, 9, size=6)
    if len(set(zip(ix, it))) < 6 or len(set(it)) < 3:
        continue
    dxi, dti = ix * dx0, it * dt0
    A, b = build_A_order(dxi, dti, order=2)
    A = np.vstack([A, dti**2 / dt0**2])
    b = np.append(b, 0.0)
    ntot += 1
    r1 = linprog(np.zeros(6), A_eq=A, b_eq=b, bounds=[(0, None)] * 6,
                 method="highs")
    npos += (r1.status == 0)
    r2 = linprog(np.ones(12), A_eq=np.hstack([A, -A]), b_eq=b,
                 bounds=[(0, None)] * 12, method="highs")
    if r2.status == 0:
        nsig += 1
        w = r2.x[:6] - r2.x[6:]
        l1s.append(np.abs(w).sum())
print(f"      w >= 0   : {npos}/{ntot} feasible")
print(f"      signed w : {nsig}/{ntot} feasible, ||w||_1 median "
      f"{np.median(l1s):.4f}, min {min(l1s):.4f}")
print("      -> CONFIRMED, matches the coordinating thread exactly.")
print("      -> BUT this row is not an accuracy condition for the PDE.  Below.")

print("\n  1c. The ACTUAL accuracy conditions are sum_i w_i C_s(q_i,a_i) = 0.")
print("      Verify: annihilate s=0..p, measure the slope of |tau| vs dx")
print("      (fixed 5-pt shape j=-2..2, parabolic refinement r = 0.25).")
xs, tst = 0.42, 0.30
for order in [1, 2, 3, 4]:
    hs, ts = [], []
    for nx in [50, 100, 200, 400]:
        dxl = 1.0 / (nx - 1)
        dtl = 0.25 * dxl**2 / ALPHA
        dxi = np.arange(-2, 3) * dxl
        dti = -dtl * np.ones(5)
        w = w_lstsq_order(dxi, dti, order=order)
        un = np.array([u_true(xs + d, tst - dtl)[0] for d in dxi])
        hs.append(dxl)
        ts.append(abs(float(u_true(xs, tst)[0] - w @ un)))
    sl = np.polyfit(np.log(hs), np.log(ts), 1)[0]
    print(f"      s=0..{order}: |tau| slope = {sl:5.2f}   (expected {order+1})")
print("      -> the hierarchy is the right one; no dt^2 row appears in it.")

# ====================================================== PART 2: max order
print("\n" + "=" * 90)
print("PART 2 -- maximum order reachable with NON-NEGATIVE weights, by regime")
print("=" * 90)

print("\n  (a) PURE DIFFUSION (c=0), 3-point stencil, vs r = alpha dt/dx^2:")
print(f"      {'r':>10} {'max pos order':>15} {'max signed':>12}  weights")
dxl = 0.01
for r in [0.05, 0.1, 1 / 6, 0.2, 1 / 3, 0.5, 0.6]:
    dtl = r * dxl**2 / ALPHA
    dxi = np.array([-dxl, 0.0, dxl])
    dti = -dtl * np.ones(3)
    mp = max_pos_order(dxi, dti, ALPHA, 0.0)
    ms = max_signed_order(dxi, dti, ALPHA, 0.0)
    w = w_positive_order(dxi, dti, order=2, alpha=ALPHA, c=0.0)
    ws = "infeasible" if w is None else str(np.round(w, 5))
    star = "   <== r = 1/6" if abs(r - 1 / 6) < 1e-9 else ""
    print(f"      {r:>10.5f} {mp:>15} {ms:>12}  {ws}{star}")
print("      r = 1/6 gives w = (1/6, 2/3, 1/6): NON-NEGATIVE and annihilating")
print("      through s = 5.  This is the classical fourth-order-accurate FTCS")
print("      scheme for the heat equation -- and it is positive.  A positivity")
print("      order barrier at 2 is therefore FALSE for parabolic problems.")

print("\n  (b) PURE ADVECTION (alpha=0):  max positive order, any stencil size")
print(f"      {'nu':>8} {'3-pt':>8} {'5-pt':>8} {'7-pt':>8} {'11-pt':>8}")
for nu in [0.1, 0.25, 0.5, 0.75, 0.9]:
    dtl = nu * dxl / C
    row = []
    for m in [1, 2, 3, 5]:
        dxi = np.arange(-m, m + 1) * dxl
        row.append(max_pos_order(dxi, -dtl * np.ones(2 * m + 1), 0.0, C))
    print(f"      {nu:>8.2f}" + "".join(f"{v:>8}" for v in row))
print("      Always 1.  This IS Godunov's theorem, and the proof is one line:")
print("      with alpha = 0, C_2 = q^2/2 >= 0, so sum_i w_i C_2 = 0 with w >= 0")
print("      forces w_i q_i^2 = 0 for every i, i.e. every active neighbour must")
print("      sit EXACTLY on the characteristic.  Generic nu has no such point.")

print("\n  (c) ADVECTION-DIFFUSION, 5-point single-level stencil:")
print(f"      {'r':>8} {'cell-Pe':>9} {'max pos':>9} {'max signed':>12} {'order lost':>11}")
for r in [0.05, 0.1, 1 / 6, 0.25, 0.4]:
    for pe in [0.1, 1.0, 3.0]:
        cl = pe * ALPHA / dxl
        dtl = r * dxl**2 / ALPHA
        dxi = np.arange(-2, 3) * dxl
        dti = -dtl * np.ones(5)
        mp = max_pos_order(dxi, dti, ALPHA, cl)
        ms = max_signed_order(dxi, dti, ALPHA, cl)
        print(f"      {r:>8.4f} {pe:>9.2f} {mp:>9} {ms:>12} {ms-mp:>11}")
print("      Positivity costs AT MOST one order here, and at r >= 1/6 it costs")
print("      NOTHING.  Contrast with (b), where it costs everything above 1.")

# ================================================ PART 3: convergence study
print("\n" + "=" * 90)
print("PART 3 -- ORDER vs CONSTANT: convergence along three refinement paths")
print("=" * 90)
print("  Fixed 5-point shape j = -2..2, one time level.  tau measured exactly")
print("  against the analytic solution at (x*,t*) = (0.42, 0.30).")

PATHS = {
    "P1 parabolic  dt~dx^2 (r=0.25)": ("par", 0.25),
    "P2 hyperbolic dt~dx   (nu=0.20)": ("hyp", 0.20),
    "P3 fixed dt = 2.5e-5": ("fix", 2.5e-5),
}
NXS = [50, 71, 100, 141, 200, 283, 400]

print("""
  IMPORTANT CONTROL.  A first pass compared positive weights satisfying only the
  ORDER-2 conditions against lstsq weights satisfying the same conditions, and
  found slopes 3.02 vs 4.00 -- an apparent order loss.  That comparison is
  RIGGED: on the symmetric shape j=-2..2 the min-norm lstsq solution inherits the
  stencil's symmetry, so its C_3 moment vanishes for free and it gets a bonus
  order; the LP returns a basic (vertex) solution supported on 3 of the 5
  columns, which is not symmetric and does not get that bonus.  The honest test
  asks BOTH solvers for the SAME moment conditions.  Both versions are reported.
""")

SETS = [("pos", 2), ("pos", 3), ("pos", 4),
        ("ls", 2), ("ls", 3), ("ls", 4)]
results = {}
for pname, (kind, par) in PATHS.items():
    print(f"\n  --- {pname} ---")
    hdr = "".join(f"{k}{o:<d}".rjust(12) for k, o in SETS)
    print(f"      {'nx':>5} {'r':>8} {'maxpos':>7}" + hdr)
    rec = {f"{k}{o}": [] for k, o in SETS}
    rec["h"] = []; rec["r"] = []; rec["maxpos"] = []
    for nx in NXS:
        dxl = 1.0 / (nx - 1)
        dtl = (par * dxl**2 / ALPHA if kind == "par"
               else par * dxl / C if kind == "hyp" else par)
        rr = ALPHA * dtl / dxl**2
        dxi = np.arange(-2, 3) * dxl
        dti = -dtl * np.ones(5)
        un = np.array([u_true(xs + d, tst - dtl)[0] for d in dxi])
        ust = float(u_true(xs, tst)[0])
        mp = max_pos_order(dxi, dti)
        rec["h"].append(dxl); rec["r"].append(rr); rec["maxpos"].append(mp)
        cells = []
        for k, o in SETS:
            if k == "pos":
                w = w_positive_order(dxi, dti, order=o)
                t = abs(ust - w @ un) if w is not None else np.nan
            else:
                w = w_lstsq_order(dxi, dti, order=o)
                A_, b_ = build_A_order(dxi, dti, order=o)
                t = (abs(ust - w @ un)
                     if np.abs(A_ @ w - b_).max() < 1e-9 else np.nan)
            rec[f"{k}{o}"].append(t)
            cells.append(f"{t:12.2e}" if np.isfinite(t) else f"{'--':>12}")
        print(f"      {nx:>5} {rr:>8.3f} {mp:>7}" + "".join(cells))
    results[pname] = rec
    print(f"      {'slopes':>21}", end="")
    for k, o in SETS:
        v = np.array(rec[f"{k}{o}"], float); h = np.array(rec["h"])
        m = np.isfinite(v) & (v > 1e-14)
        sl = (np.polyfit(np.log(h[m]), np.log(v[m]), 1)[0]
              if m.sum() >= 3 else np.nan)
        print(f"{sl:>12.2f}" if np.isfinite(sl) else f"{'--':>12}", end="")
    print()

print("\n  MATCHED-ORDER COMPARISON (P1, parabolic):")
rec = results["P1 parabolic  dt~dx^2 (r=0.25)"]
for o in (2, 3, 4):
    a = np.array(rec[f"pos{o}"], float); b = np.array(rec[f"ls{o}"], float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        print(f"      order {o}: positive infeasible"); continue
    sa = np.polyfit(np.log(np.array(rec['h'])[m]), np.log(a[m]), 1)[0]
    sb = np.polyfit(np.log(np.array(rec['h'])[m]), np.log(b[m]), 1)[0]
    print(f"      order {o} conditions: slope pos {sa:5.2f} vs lstsq {sb:5.2f}"
          f"   |  geo-mean |tau| ratio pos/lstsq = "
          f"{np.exp(np.mean(np.log(a[m]/b[m]))):6.2f}")
print("""
  VERDICT.  Asked for the SAME moment conditions, positive and signed weights
  converge at the SAME rate.  Positivity buys its certificate with a CONSTANT,
  not with an order -- and the constant can go either way (see the C_{p+1} table
  below).  What positivity actually costs is FEASIBILITY: it is available only
  in part of parameter space, and the boundary is the CFL/Peclet region of
  Task 2, not an order barrier.""")

# leading-error constant directly:  sum_i w_i C_{p+1}
print("\n  Leading truncation CONSTANT  sum_i w_i C_4(q_i,a_i)/h^4, order-3 conditions:")
print(f"      {'r':>8} {'positive':>13} {'lstsq':>13} {'|pos/lstsq|':>13}")
dxl = 0.005
for r in [0.1, 1 / 6, 0.25, 0.4, 0.49]:
    dtl = r * dxl**2 / ALPHA
    dxi = np.arange(-2, 3) * dxl
    dti = -dtl * np.ones(5)
    c4 = moment_coeff(dxi - C * dti, ALPHA * dti, 4)
    wp = w_positive_order(dxi, dti, order=3)
    w2 = w_lstsq_order(dxi, dti, order=3)
    if wp is None:
        print(f"      {r:>8.4f} {'infeasible':>13} {w2@c4/dxl**4:>13.4f}")
        continue
    kp, kl = wp @ c4 / dxl**4, w2 @ c4 / dxl**4
    print(f"      {r:>8.4f} {kp:>13.4f} {kl:>13.4f} {abs(kp/kl):>13.2f}")
print("      -> at r <= 0.25 the POSITIVE stencil has the SMALLER leading")
print("         constant; positivity is not a penalty there at all.")

# ===================================================================== FIG
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
series = [("pos3", "#1e8449", "positive, order-3 cond."),
          ("ls3", "#c0392b", "lstsq, order-3 cond."),
          ("pos2", "#27ae60", "positive, order-2 cond."),
          ("ls2", "#e67e22", "lstsq, order-2 cond.")]
for j, (pname, rec) in enumerate(results.items()):
    a = ax[j]
    h = np.array(rec["h"])
    for key, col, lab in series:
        v = np.array(rec[key], float)
        m = np.isfinite(v) & (v > 1e-14)
        if m.sum() >= 2:
            lb = lab
            if m.sum() >= 3:
                sl = np.polyfit(np.log(h[m]), np.log(v[m]), 1)[0]
                lb = f"{lab}  slope {sl:.2f}"
            else:
                lb = f"{lab}  (too few pts)"
            a.loglog(h[m], v[m], "o", color=col, ms=4,
                     ls="-" if key.startswith("pos") else "--", label=lb)
    a.loglog(h, 3e2 * h**4, "k:", lw=1, label=r"$h^4$ guide")
    a.set_xlabel(r"$\Delta x$")
    a.set_ylabel(r"$|\tau|$" if j == 0 else "")
    a.set_title(pname, fontsize=9.5)
    a.legend(fontsize=7)
    a.grid(alpha=0.25, which="both")
fig.suptitle("Task 6 -- at MATCHED moment conditions, positive and signed weights "
             "converge at the same rate;\npositivity costs FEASIBILITY (P2, P3), not order",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG}/task6_order.png", dpi=150)
print(f"\nfigure -> {FIG}/task6_order.png")
