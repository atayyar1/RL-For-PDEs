"""CHECKS.md #3: the LP arm has a zero objective, so every feasible point is optimal and the
solver returns an arbitrary vertex. Re-solve with 20 random objectives on a fixed geometry
and report what survives: growth, error, number of non-zeros, and the spread. Compare to
maxent (unique interior point) and to the min-norm QP (strictly convex, no degeneracy).
Prints tables only.
"""
import numpy as np
from scipy.optimize import linprog
import g1lib as g
from core import stencil as st

p = g.Problem(alpha=0.1, c=1.0)
for periodic, nx in [(False, 101), (True, 100)]:
    for kind, dis in [("jitter", 0.25), ("random", 0.0)]:
        x = g.make_points(nx, dis, seed=0, kind=kind, periodic=periodic)
        dt = g.time_step(x, p, 0.5, periodic)
        T = 0.5; nt = int(np.ceil(T / dt)); m = max(1, nt // 50)
        samp = np.arange(m, nt + 1, m); samp = samp if samp[-1] == nt else np.append(samp, nt)
        times = dt * samp
        if periodic:
            U = np.array([g.u_periodic(p, x, t) for t in times]); u0 = g.u_periodic(p, x, 0.0)
        else:
            U = g.Reference(p, times, x).U; u0 = p.u_true(x, 0.0)
        k_of = {int(s): k for k, s in enumerate(samp)}
        for K in [5, 7]:
            print(f"\n=== periodic={periodic} {kind}:{dis} K={K} alpha=0.1 nt={nt} ===")
            print(f"{'objective':>14}{'growth':>10}{'errT':>11}{'nnz/node (mean)':>17}{'max l1':>8}{'fb':>4}")
            errs, grows = [], []
            for trial in range(-2, 20):
                if trial == -2:
                    op = g.Operator(x, p, dt, "maxent", K, periodic=periodic); name = "maxent"
                elif trial == -1:
                    op = g.Operator(x, p, dt, "lp", K, periodic=periodic); name = "lp (zero obj)"
                else:
                    op = g.Operator(x, p, dt, "lprand", K, periodic=periodic, seed=1000 + trial); name = f"lp rand #{trial}"
                res = op.march(u0, None, nt, ref_fn=lambda k: U[k_of[k]], sample_every=m)
                nnz = np.mean([(np.abs(op.M1[i]) > 1e-12).sum() for i in (range(nx) if periodic else range(1, nx - 1))])
                gr = op.growth(nt); e = res["err"][-1]
                if trial >= 0:
                    errs.append(e); grows.append(gr)
                if trial < 3 or trial == 19:
                    print(f"{name:>14}{gr:10.3f}{e:11.2e}{nnz:17.2f}{op.l1.max():8.4f}{op.fallback.sum():4d}")
            errs, grows = np.array(errs), np.array(grows)
            print(f"{'20 random: ':>14} growth in [{grows.min():.3f}, {grows.max():.3f}]  errT in [{errs.min():.2e}, {errs.max():.2e}]  spread x{errs.max()/errs.min():.1f}")
