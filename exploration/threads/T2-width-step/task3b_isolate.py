"""TASK 3b -- isolate WHERE the win comes from.

(1) head-to-head: frontier stencil vs the CHEAPEST FTCS reaching the same error
(2) pure diffusion c=0: removes the Cole-Hopf/semi-Lagrangian advantage that the
    wide solver enjoys over central-difference FTCS/CN
(3) wall-clock cross-check of the flop model
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import core as C, solvers as S

ALPHA = 0.1
HERE = os.path.dirname(os.path.abspath(__file__))


def errinf(x, u, ex, T):
    return float(np.max(np.abs(u - ex(x, T))))


def sweep(T_end, c):
    ex = C.Exact(C.ic_sine, ALPHA, c)
    Ns = [13, 17, 25, 35, 50, 71, 100, 141, 200, 283, 400, 566, 800]
    ftcs, cn, wide, front, spec = [], [], [], [], []
    for N in Ns:
        x, u, i = S.solve_ftcs(N, T_end, ALPHA, c, C.ic_sine)
        ftcs.append((i["flops"], errinf(x, u, ex, T_end), N))
        x, u, i = S.solve_spectral(N, T_end, ALPHA, c, C.ic_sine)
        spec.append((i["flops"], errinf(x, u, ex, T_end), N))
        for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            x, u, i = S.solve_cn(N, T_end, ALPHA, c, C.ic_sine, nb)
            cn.append((i["flops"], errinf(x, u, ex, T_end), N, nb))
            for s in [1.0, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
                x, u, i = S.solve_wide(N, T_end, ALPHA, c, C.ic_sine, nb, s)
                if u is None or not np.all(np.isfinite(u)):
                    continue
                r = (i["flops"], errinf(x, u, ex, T_end), N, nb, s, i["m"], i["P"])
                wide.append(r)
                if s <= 1.05:
                    front.append(r)
    return ex, ftcs, cn, wide, front, spec


def cheapest(lst, tgt):
    ok = [r for r in lst if r[1] <= tgt]
    return min(ok, key=lambda r: r[0]) if ok else None


print("#" * 104)
print("#  (1)+(2)  WORK-PRECISION WITH AND WITHOUT ADVECTION")
print("#" * 104)
summary = {}
for c in [0.0, 1.0]:
    for T_end in [0.05, 0.2]:
        ex, ftcs, cn, wide, front, spec = sweep(T_end, c)
        key = "c%.0f_T%.2f" % (c, T_end)
        summary[key] = {}
        print("\n--- c = %.1f , T_end = %.2f %s" % (c, T_end,
              "(pure diffusion: all methods see the same operator)" if c == 0 else ""))
        print("%-9s %13s %13s %13s %13s %13s | %10s %10s"
              % ("target", "FTCS", "CN", "WIDE best", "WIDE frontier", "DST exact",
                 "W/FTCS", "W/CN"))
        for tgt in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            a, b, w, f, sp = (cheapest(ftcs, tgt), cheapest(cn, tgt),
                              cheapest(wide, tgt), cheapest(front, tgt),
                              cheapest(spec, tgt))
            fm = lambda r: ("%13.3e" % r[0]) if r else "%13s" % "--"
            rr = lambda p, q: ("%9.1fx" % (p[0] / q[0])) if (p and q) else "%10s" % "--"
            print("%-9.0e %s %s %s %s %s | %s %s"
                  % (tgt, fm(a), fm(b), fm(w), fm(f), fm(sp), rr(a, w), rr(b, w)))
            summary[key]["%.0e" % tgt] = dict(
                ftcs=a[0] if a else None, cn=b[0] if b else None,
                wide=w[0] if w else None, frontier=f[0] if f else None,
                spec=sp[0] if sp else None)

        # head-to-head frontier vs FTCS at matched accuracy
        print("\n  HEAD-TO-HEAD: each frontier config vs the cheapest FTCS of equal/better error")
        print("  %6s %5s %4s %4s | %11s %11s | %11s %11s | %s"
              % ("N", "nb", "m", "P", "front err", "front cost", "FTCS err", "FTCS cost",
                 "frontier/FTCS cost"))
        fr = sorted(front, key=lambda r: r[1])
        seen = set()
        for r in fr:
            keyn = round(np.log10(max(r[1], 1e-300)), 1)
            if keyn in seen:
                continue
            seen.add(keyn)
            a = cheapest(ftcs, r[1])
            if a is None:
                continue
            print("  %6d %5d %4d %4d | %11.3e %11.3e | %11.3e %11.3e | %8.2fx"
                  % (r[2], r[3], r[5], r[6], r[1], r[0], a[1], a[0], r[0] / a[0]))
            if len(seen) > 7:
                break

# ------------------------------------------------------------------ wall clock
print("\n" + "#" * 104)
print("#  (3)  WALL-CLOCK CROSS-CHECK OF THE FLOP MODEL   (N=400, T=0.05, c=1)")
print("#" * 104)
N, T_end, c = 400, 0.05, 1.0
ex = C.Exact(C.ic_sine, ALPHA, c)
cases = [("FTCS @CFL", lambda: S.solve_ftcs(N, T_end, ALPHA, c, C.ic_sine)),
         ("CN nb=64", lambda: S.solve_cn(N, T_end, ALPHA, c, C.ic_sine, 64)),
         ("wide nb=4 s=6", lambda: S.solve_wide(N, T_end, ALPHA, c, C.ic_sine, 4, 6.0)),
         ("wide nb=4 s=1", lambda: S.solve_wide(N, T_end, ALPHA, c, C.ic_sine, 4, 1.0)),
         ("DST exact", lambda: S.solve_spectral(N, T_end, ALPHA, c, C.ic_sine))]
print("%-16s %12s %12s %12s %14s" % ("method", "einf", "flops", "wall [ms]", "flops/wall"))
for nm, fn in cases:
    x, u, i = fn()
    t0 = time.perf_counter(); nrep = 3
    for _ in range(nrep):
        fn()
    wt = (time.perf_counter() - t0) / nrep * 1e3
    print("%-16s %12.3e %12.3e %12.3f %14.3e"
          % (nm, errinf(x, u, ex, T_end), i["flops"], wt, i["flops"] / (wt * 1e-3)))
print("\n  NOTE: wall time is numpy-overhead dominated for small N; flop counts are")
print("  the primary metric.  Methods with many steps (FTCS, CN) pay python loop")
print("  overhead, so wall time FLATTERS the wide stencil relative to the flop model.")

json.dump(summary, open(os.path.join(HERE, "task3b_summary.json"), "w"), indent=1)
