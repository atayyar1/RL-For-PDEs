"""Step 0 -- the apparatus floor, before any arm is compared (CHECKS.md #1).

  F0.1  reference floor: exact series vs itself (refined), and CN vs exact at alpha=0.1,
        CN Richardson floor at the alphas where the exact series is unusable.
  F0.2  uniform reductions: on the uniform grid with K=3, every propagator arm must
        equal Lax-Wendroff-diffusion and molfe must equal FTCS, to machine precision.
  F0.3  boundary/fallback floor: how many interior nodes are positivity-infeasible at
        each disorder level and K (static, since the points do not move).
  F0.4  floor of the marching loop itself: march the exact 3-point uniform LW operator
        and report its error vs the exact solution (that is the discretisation error
        any arm at the same resolution has to beat or match).
Prints tables only.
"""
import numpy as np
import g1lib as g

np.set_printoptions(precision=3, linewidth=140)

print("=== F0.1 reference floor ===")
print(f"{'alpha':>7} {'nx':>5} {'exact self-consist':>19} {'CN vs exact':>12} {'CN Richardson':>14}")
x = np.linspace(0, 1, 101)
times = np.linspace(0, 0.5, 11)[1:]
for alpha in [0.1, 0.03, 0.01, 0.003]:
    p = g.Problem(alpha=alpha, c=1.0)
    ex = None
    if alpha >= 0.03:
        U1 = np.array([p.u_true(x, t) for t in times])
        U2 = np.array([p.u_true(x, t, n_series=800, nq=80001) for t in times])
        ex = float(np.abs(U1 - U2).max())
    cn = g.Reference(p, times, x, force_cn=True, nx_ref=4001, dt_max=2e-4)
    vs = float(np.abs(cn.U - U1).max()) if alpha == 0.1 else np.nan
    print(f"{alpha:7.3f} {101:5d} {ex if ex is not None else np.nan:19.2e} {vs:12.2e} {cn.floor:14.2e}")

print("\n=== F0.2 uniform reductions (nx=101, K=3, disorder 0) ===")
for alpha in [0.1, 0.01]:
    p = g.Problem(alpha=alpha, c=1.0)
    x = g.make_points(101, 0.0)
    dt = g.time_step(x, p, 0.5)
    dx = x[1] - x[0]
    lw, ft = g.lw_weights(p, dx, dt), g.ftcs_weights(p, dx, dt)
    print(f"alpha={alpha} dx={dx:.4f} dt={dt:.3e} r={p.alpha*dt/dx**2:.3f} nu={p.c*dt/dx:.3f} Pe={p.c*dx/p.alpha:.2f}")
    print(f"  {'arm':10} {'max|w_row - LW|':>16} {'max|w_row - FTCS|':>18} {'min w':>9} {'max l1':>8}")
    for arm in g.ARMS:
        op = g.Operator(x, p, dt, arm, 3)
        i = 50
        row = op.M1[i, [i - 1, i, i + 1]]
        print(f"  {arm:10} {np.abs(row - lw).max():16.2e} {np.abs(row - ft).max():18.2e} {op.min_w.min():9.4f} {op.l1.max():8.4f}")

print("\n=== F0.3 positivity-infeasible interior nodes (static), nx=101, alpha=0.1, safety 0.5 ===")
p = g.Problem(alpha=0.1, c=1.0)
print(f"{'kind':7} {'disord':>6} {'K':>3} {'infeasible/99':>14} {'maxentw K used (max)':>21}")
for kind, dis in [("jitter", 0.0), ("jitter", 0.1), ("jitter", 0.25), ("jitter", 0.45), ("random", 0.0)]:
    x = g.make_points(101, dis, seed=1, kind=kind)
    dt = g.time_step(x, p, 0.5)
    for K in [3, 5, 7]:
        a = g.Operator(x, p, dt, "maxent", K)
        b = g.Operator(x, p, dt, "maxentw", K)
        print(f"{kind:7} {dis:6.2f} {K:3d} {a.fallback.sum():14d} {b.K_used.max():21d}  (widened fallback {b.fallback.sum()})")

print("\n=== F0.4 marching-loop floor: uniform 3-pt LW (=maxent, K=3, disorder 0) vs exact, alpha=0.1 ===")
for nx in [51, 101, 201]:
    x = g.make_points(nx, 0.0)
    dt = g.time_step(x, p, 0.5)
    T = 0.5
    nt = int(np.ceil(T / dt))
    dt = T / nt
    op = g.Operator(x, p, dt, "maxent", 3)
    times = dt * np.arange(1, nt + 1)
    ref = g.Reference(p, times, x)
    res = op.march(p.u_true(x, 0.0), None, nt, ref_fn=lambda k: ref.at(k - 1), sample_every=1)
    print(f"nx={nx:4d} nt={nt:5d} dt={dt:.2e}  L_inf(T)={res['err'][-1]:.2e}  max_t L_inf={res['err'].max():.2e}  "
          f"ref floor={ref.floor:.1e}  rho(M)={op.spectral_radius():.6f}")
