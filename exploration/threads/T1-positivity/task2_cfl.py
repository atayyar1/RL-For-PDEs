"""
Task 2 -- Does the positivity LP recover the textbook CFL conditions?

Method.  For a fixed classical stencil geometry we sweep the relevant
dimensionless parameter and ask the LP whether some w >= 0 solves A w = b.  The
numerically detected boundary is refined by bisection and compared with theory.

Schemes
  (a) FTCS diffusion            c=0, 3-pt, order 2   expect r <= 1/2
  (b) first-order upwind        alpha=0, 2-pt, order 1  expect 0 <= nu <= 1
      + 3-pt centred, order 1                           expect nu <= 1 too
  (c) FTCS advection-diffusion  3-pt, order 2        expect r <= 1/2 and a
                                                     cell-Peclet condition
  (d) Lax-Wendroff / centred    alpha=0, 3-pt, order 2  expect FAILURE

NOTE on the u_xx moment row.  Two variants (see stencil.py):
  'given': p_i = 1/2 dx_i^2 + alpha dt_i           (the existing code)
  'exact': p_i = 1/2 (dx_i - c dt_i)^2 + alpha dt_i (true 2nd-order moment)
The 'exact' row includes -c dx dt (from dx dt u_xt) and 1/2 c^2 dt^2 (from
1/2 dt^2 u_tt).  Which one you use changes the scheme you recover and the
positivity region, so both are reported.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stencil import DX, DT, ALPHA, C, build_A, w_positive

FIG = "figures"


def feasible(dxi, dti, alpha, c, variant="given", order=2):
    return w_positive(dxi, dti, alpha=alpha, c=c, variant=variant,
                      order=order) is not None


def unique_w(dxi, dti, alpha, c, variant="given", order=2):
    """Square-system solve (only when n == number of rows)."""
    A, b = build_A(dxi, dti, alpha=alpha, c=c, variant=variant, order=order)
    return np.linalg.solve(A, b)


def bisect_boundary(fn, lo, hi, iters=60):
    """fn(x) True for x=lo, False for x=hi.  Return the crossing."""
    if not fn(lo) or fn(hi):
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("=" * 78)
print("(a) FTCS DIFFUSION   c = 0, 3-point stencil at dt = -Dt, order 2")
print("=" * 78)
alpha, dx = 0.1, 0.01
dxi = np.array([-dx, 0.0, dx])


def feas_a(r):
    dt = r * dx**2 / alpha
    return feasible(dxi, -dt * np.ones(3), alpha, 0.0)


rstar = bisect_boundary(feas_a, 0.01, 2.0)
print(f"  positivity boundary (LP, bisected) : r* = {rstar:.12f}")
print(f"  textbook FTCS diffusion condition  : r* = 0.5")
print(f"  |error| = {abs(rstar-0.5):.3e}     -> EXACT MATCH")
for r in [0.3, 0.5, 0.5 + 1e-9, 0.7]:
    dt = r * dx**2 / alpha
    w = unique_w(dxi, -dt * np.ones(3), alpha, 0.0)
    print(f"    r = {r:<12.9g} w = [{w[0]:+.4f} {w[1]:+.4f} {w[2]:+.4f}]  "
          f"||w||_1 = {np.abs(w).sum():.6f}   feasible = {feas_a(r)}")
print("  closed form: w = (r, 1-2r, r); the only sign change is w_0 = 1-2r.")

print("\n" + "=" * 78)
print("(b) PURE ADVECTION, FIRST ORDER  alpha = 0, order 1")
print("=" * 78)
c, dx = 1.0, 0.01
up_x = np.array([-dx, 0.0])                      # upwind (c > 0)
ce_x = np.array([-dx, 0.0, dx])                  # centred
dn_x = np.array([0.0, dx])                       # downwind


def mk(xs, nu):
    dt = nu * dx / c
    return xs, -dt * np.ones(len(xs))


for name, xs in (("upwind  2-pt", up_x), ("centred 3-pt", ce_x),
                 ("downwind 2-pt", dn_x)):
    f = lambda nu, xs=xs: feasible(*mk(xs, nu), 0.0, c, order=1)
    nustar = bisect_boundary(f, 1e-4, 5.0)
    print(f"  {name}: positivity boundary nu* = {nustar:.12f}   "
          f"(feasible at nu=0.5: {f(0.5)})")
print("  textbook Courant condition: nu = c Dt/Dx <= 1")
print("  downwind stencil is infeasible for every nu > 0 -- the characteristic")
print("  foot x* - c Dt lies OUTSIDE its x-span.  This is the CFL condition in")
print("  its literal form: the stencil must contain the domain of dependence.")
for nu in [0.5, 1.0, 1.5]:
    xs, ts = mk(up_x, nu)
    w = unique_w(xs, ts, 0.0, c, order=1)
    print(f"    upwind nu={nu:<4} w = [{w[0]:+.4f} {w[1]:+.4f}]   "
          f"||w||_1 = {np.abs(w).sum():.4f}   (closed form (nu, 1-nu))")

print("\n" + "=" * 78)
print("(c) FTCS ADVECTION-DIFFUSION  3-point, order 2")
print("=" * 78)
alpha, dx = 0.1, 0.01
dxi = np.array([-dx, 0.0, dx])
print("  'given' row -> unique w = (r + nu/2,  1 - 2r,  r - nu/2)")
print("      positivity  <=>  r <= 1/2   AND   r >= nu/2  <=>  cell-Pe <= 2")
print("  'exact' row -> unique w = ((2r+nu^2+nu)/2, 1-2r-nu^2, (2r+nu^2-nu)/2)")
print("      positivity  <=>  2r + nu^2 <= 1  AND  2r + nu^2 >= nu (i.e. Pe <= 2/(1-nu))")
print("      with nu = r*Pe the second condition is r Pe^2 - Pe + 2 >= 0, which")
print("      for r < 1/8 is violated on an INTERVAL of Pe -- the feasible set is")
print("      DISCONNECTED in Pe.  The 'given' row has no such structure.")
def feas_runs(fn, lo, hi, n=4000):
    """Return ALL maximal feasible runs (the set need not be an interval), each
    edge refined by bisection.  Returns a list of (left, right) pairs."""
    xs = np.linspace(lo, hi, n)
    fl = np.array([fn(x) for x in xs])
    if not fl.any():
        return []

    def edge(a_, b_, want_true_at_a):
        for _ in range(60):
            m = 0.5 * (a_ + b_)
            if fn(m) == want_true_at_a:
                a_ = m
            else:
                b_ = m
        return 0.5 * (a_ + b_)

    runs = []
    i = 0
    while i < n:
        if not fl[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fl[j + 1]:
            j += 1
        left = lo if i == 0 else edge(xs[i], xs[i - 1], True)
        right = hi if j == n - 1 else edge(xs[j], xs[j + 1], True)
        runs.append((left, right))
        i = j + 1
    return runs


def fmt_runs(runs):
    if not runs:
        return "empty"
    return " U ".join(f"[{a:.6f}, {b:.6f}]" for a, b in runs)


def pred_r_runs(nu, variant):
    """Analytic feasible r-set at fixed nu for the 3-pt adv-diff stencil."""
    if variant == "given":
        # w = (r+nu/2, 1-2r, r-nu/2):  r >= nu/2  and  r <= 1/2
        return [(nu / 2, 0.5)] if nu / 2 <= 0.5 else []
    # w = ((2r+nu^2+nu)/2, 1-2r-nu^2, (2r+nu^2-nu)/2)
    lo, hi = max(0.0, (nu - nu**2) / 2), (1 - nu**2) / 2
    return [(lo, hi)] if lo <= hi else []


def pred_pe_runs(r, variant):
    """Analytic feasible cell-Peclet interval at fixed r.  Note nu = r * Pe."""
    if variant == "given":
        # r >= nu/2 = r*Pe/2  ->  Pe <= 2 ;  r <= 1/2 independent of Pe
        return [(0.0, 2.0)] if r <= 0.5 else []
    # nu = r*Pe.
    # w0 >= 0: 2r + (r Pe)^2 <= 1     ->  Pe <= sqrt(1-2r)/r
    # w+ >= 0: r Pe^2 - Pe + 2 >= 0   ->  OUTSIDE the roots (1 -+ sqrt(1-8r))/(2r)
    if r > 0.5:
        return []
    hi = np.sqrt(1 - 2 * r) / r
    if r >= 0.125:
        return [(0.0, hi)]                   # w+ condition never binds
    p_lo = (1 - np.sqrt(1 - 8 * r)) / (2 * r)
    p_hi = (1 + np.sqrt(1 - 8 * r)) / (2 * r)
    runs = [(0.0, min(p_lo, hi))]
    if hi > p_hi:                            # a SECOND, disconnected branch
        runs.append((p_hi, hi))
    return runs


for variant in ("given", "exact"):
    print(f"\n  --- variant = {variant} ---")
    cc = 1.0
    print(f"    {'nu':>6} {'r set (LP)':>30} {'r set (theory)':>30}")
    for nu_t in [0.02, 0.2, 0.5, 0.9]:
        def fr(rv, nu_t=nu_t):
            dtl = nu_t * dx / cc
            al = rv * dx**2 / dtl
            return feasible(dxi, -dtl * np.ones(3), al, cc, variant=variant)
        runs = feas_runs(fr, 0.002, 0.62)
        pr = pred_r_runs(nu_t, variant)
        print(f"    {nu_t:>6.2f} {fmt_runs(runs):>30} {fmt_runs(pr):>30}")
    print(f"    {'r':>6} {'cell-Pe set (LP)':>44} {'cell-Pe set (theory)':>44}")
    for r_t in [0.1, 0.25, 0.45]:
        def fpe(pe, r_t=r_t):
            cl = pe * alpha / dx
            dtl = r_t * dx**2 / alpha
            return feasible(dxi, -dtl * np.ones(3), alpha, cl, variant=variant)
        runs = feas_runs(fpe, 0.01, 25.0, n=8000)
        pr = pred_pe_runs(r_t, variant)
        print(f"    {r_t:>6.2f} {fmt_runs(runs):>44} {fmt_runs(pr):>44}")

print("\n" + "=" * 78)
print("(d) PURE ADVECTION AT SECOND ORDER -- POSITIVITY FAILS (Godunov)")
print("=" * 78)
c, dx = 1.0, 0.01
dxi3 = np.array([-dx, 0.0, dx])
dxi5 = np.array([-2 * dx, -dx, 0.0, dx, 2 * dx])
print(f"{'nu':>7} {'3pt feas':>10} {'5pt feas':>10} {'LW w':>34} {'||w||_1':>10}")
for nu in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1]:
    dt = nu * dx / c
    f3 = feasible(dxi3, -dt * np.ones(3), 0.0, c, variant="exact")
    f5 = feasible(dxi5, -dt * np.ones(5), 0.0, c, variant="exact")
    w = unique_w(dxi3, -dt * np.ones(3), 0.0, c, variant="exact")
    print(f"{nu:>7.2f} {str(f3):>10} {str(f5):>10} "
          f"[{w[0]:+.4f} {w[1]:+.4f} {w[2]:+.4f}]".rjust(34)
          + f"{np.abs(w).sum():>10.4f}")
print("  The 'exact' row reproduces Lax-Wendroff exactly: w = (nu(1+nu)/2,")
print("  1-nu^2, -nu(1-nu)/2), with ||w||_1 = 1 + nu(1-nu) <= 1.25.")
nu = 0.5
dt = nu * dx / c
w = unique_w(dxi3, -dt * np.ones(3), 0.0, c, variant="exact")
lw = np.array([nu * (1 + nu) / 2, 1 - nu**2, -nu * (1 - nu) / 2])
print(f"    check at nu=0.5: moment solve {w}  vs  LW closed form {lw}  "
      f"max|diff| = {np.abs(w-lw).max():.2e}")
w_given = unique_w(dxi3, -dt * np.ones(3), 0.0, c, variant="given")
print(f"    the 'given' row instead gives {w_given} = FTCS-central,")
print(f"    ||w||_1 = {np.abs(w_given).sum():.4f} = 1 + nu: unconditionally unstable.")
print("\n  WHY positivity can never hold here: with alpha = 0 the u_xx moment is")
print("  p_i = 1/2 (dx_i - c dt_i)^2 >= 0, and p_i = 0 only for a neighbour")
print("  sitting exactly on the characteristic.  A convex combination of")
print("  non-negative numbers vanishes only if all the active ones vanish, so no")
print("  nonnegative w exists unless a neighbour lies exactly at the characteristic")
print("  foot (nu an integer).  This is Godunov's theorem: no linear scheme of")
print("  order >= 2 for pure advection is monotone.  Diffusion is what buys")
print("  positivity back -- alpha dt_i < 0 pulls near neighbours' p_i below zero.")
print(f"\n  Diffusion needed for the centre point to have p < 0:")
print(f"    1/2 (c Dt)^2 < alpha Dt  <=>  Dt < 2 alpha / c^2 = "
      f"{2*ALPHA/C**2:.4g}   (benchmark Dt = {DT:.4g}: satisfied by "
      f"{2*ALPHA/C**2/DT:.0f}x)")

# =============================================================== FIGURE
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))

# (a)
a = ax[0, 0]
rs = np.linspace(0.05, 1.0, 400)
alpha, dx = 0.1, 0.01
dxi = np.array([-dx, 0.0, dx])
ws = np.array([unique_w(dxi, -(r * dx**2 / alpha) * np.ones(3), alpha, 0.0)
               for r in rs])
fe = np.array([feas_a(r) for r in rs])
a.plot(rs, np.abs(ws).sum(1), color="#2874a6", lw=2, label=r"$\|w\|_1$")
a.plot(rs, ws.min(1), color="#c0392b", lw=1.5, ls="--", label=r"$\min_i w_i$")
a.fill_between(rs, 0, 1, where=fe, color="#27ae60", alpha=0.15,
               transform=a.get_xaxis_transform(), label="positivity feasible")
a.axvline(0.5, color="k", ls=":", lw=1.5)
a.text(0.51, 1.6, r"textbook $r=1/2$", fontsize=9)
a.axhline(1.0, color="gray", lw=0.8)
a.set_xlabel(r"$r=\alpha\Delta t/\Delta x^2$")
a.set_title(f"(a) FTCS diffusion: LP boundary $r^*={rstar:.9f}$")
a.legend(fontsize=8, loc="upper left")
a.set_ylim(-1, 2.2)

# (b)
a = ax[0, 1]
nus = np.linspace(0.02, 2.0, 400)
c, dx = 1.0, 0.01
wu = np.array([unique_w(*mk(up_x, nu), 0.0, c, order=1) for nu in nus])
fu = np.array([feasible(*mk(up_x, nu), 0.0, c, order=1) for nu in nus])
fc = np.array([feasible(*mk(ce_x, nu), 0.0, c, order=1) for nu in nus])
a.plot(nus, np.abs(wu).sum(1), color="#2874a6", lw=2, label=r"upwind $\|w\|_1$")
a.fill_between(nus, 0, 1, where=fu, color="#27ae60", alpha=0.15,
               transform=a.get_xaxis_transform(), label="upwind feasible")
a.fill_between(nus, 0, 1, where=fc, color="#8e44ad", alpha=0.10,
               transform=a.get_xaxis_transform(), label="centred 3-pt feasible")
a.axvline(1.0, color="k", ls=":", lw=1.5)
a.text(1.03, 2.2, r"Courant $\nu=1$", fontsize=9)
a.axhline(1.0, color="gray", lw=0.8)
a.set_xlabel(r"$\nu=c\Delta t/\Delta x$")
a.set_title("(b) pure advection, 1st order: LP boundary $\\nu^*=1$")
a.legend(fontsize=8, loc="upper left")
a.set_ylim(0, 3)

# (c) 2-D map
a = ax[1, 0]
alpha, dx = 0.1, 0.01
dxi = np.array([-dx, 0.0, dx])
NR, NP = 160, 160
rgrid = np.linspace(0.02, 0.85, NR)
pegrid = np.linspace(0.05, 12.0, NP)
Mg = np.zeros((NP, NR))
Me = np.zeros((NP, NR))
for i, pe in enumerate(pegrid):
    cl = pe * alpha / dx
    for j, rv in enumerate(rgrid):
        dtl = rv * dx**2 / alpha
        Mg[i, j] = feasible(dxi, -dtl * np.ones(3), alpha, cl, variant="given")
        Me[i, j] = feasible(dxi, -dtl * np.ones(3), alpha, cl, variant="exact")
a.contourf(rgrid, pegrid, Mg, levels=[0.5, 1.5], colors=["#27ae60"], alpha=0.30)
a.contour(rgrid, pegrid, Mg, levels=[0.5], colors=["#1e8449"], linewidths=2)
a.contour(rgrid, pegrid, Me, levels=[0.5], colors=["#8e44ad"], linewidths=2,
          linestyles="--")
a.axvline(0.5, color="k", ls=":", lw=1.5)
a.axhline(2.0, color="k", ls=":", lw=1.5)
a.text(0.03, 2.2, "cell-Pe = 2", fontsize=9)
a.text(0.505, 11.0, "r = 1/2", fontsize=9, rotation=90, va="top")
a.set_xlabel(r"$r=\alpha\Delta t/\Delta x^2$")
a.set_ylabel(r"cell Peclet $=c\Delta x/\alpha$")
a.set_title("(c) FTCS adv-diff: green = feasible ('given');\n"
            "purple dashed = 'exact' row boundary")

# (d)
a = ax[1, 1]
nus = np.linspace(0.02, 1.2, 400)
c, dx = 1.0, 0.01
wl = np.array([unique_w(dxi3, -(nu * dx / c) * np.ones(3), 0.0, c,
                        variant="exact") for nu in nus])
fl = np.array([feasible(dxi3, -(nu * dx / c) * np.ones(3), 0.0, c,
                        variant="exact") for nu in nus])
a.plot(nus, np.abs(wl).sum(1), color="#2874a6", lw=2,
       label=r"Lax-Wendroff $\|w\|_1=1+\nu(1-\nu)$")
a.plot(nus, wl.min(1), color="#c0392b", lw=1.5, ls="--", label=r"$\min_i w_i$")
a.plot(nus, 1 + nus, color="#e67e22", lw=1.2, ls="-.",
       label=r"FTCS-central ('given' row): $1+\nu$")
a.axhline(1.0, color="gray", lw=0.8)
a.axvline(1.0, color="k", ls=":", lw=1.5)
a.text(0.05, 0.35, "positivity feasible NOWHERE\n(except $\\nu\\in\\{0,1\\}$)\n"
       "= Godunov's theorem", fontsize=9,
       bbox=dict(fc="#fdebd0", ec="#e67e22"))
a.set_xlabel(r"$\nu=c\Delta t/\Delta x$")
a.set_title("(d) 2nd-order pure advection: L$^2$-stable but NOT monotone")
a.legend(fontsize=8, loc="upper left")
a.set_ylim(-0.4, 2.4)

fig.tight_layout()
fig.savefig(f"{FIG}/task2_cfl.png", dpi=150)
print(f"\nfigure -> {FIG}/task2_cfl.png")
