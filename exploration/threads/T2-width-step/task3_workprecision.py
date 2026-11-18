"""TASK 3 -- work-precision.  Does the wide positive explicit stencil beat
CFL-limited FTCS and Crank-Nicolson?  Includes the decisive CONTROL:
the frontier stencil of half-width m vs plain FTCS on a grid of spacing m*dx.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C, solvers as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
ALPHA, CVEL = 0.1, 1.0
NS = [25, 50, 100, 200, 400, 800]


def errs(x, u, ex, T):
    ut = ex(x, T)
    d = u - ut
    return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def pareto(pts):
    """pts = list of (cost, err, label-dict); keep non-dominated."""
    pts = sorted(pts, key=lambda p: p[0])
    out, best = [], np.inf
    for p in pts:
        if p[1] < best * (1 - 1e-12):
            out.append(p); best = p[1]
    return out


def run(T_end, c=CVEL, ic=C.ic_sine, tag="sine"):
    ex = C.Exact(ic, ALPHA, c, nmodes=1200 if tag != "sine" else 400)
    res = dict(ftcs=[], cn=[], wide=[], spec=[], frontier=[],
               rkc1=[], rkc2=[])

    for N in NS:
        x, u, i = S.solve_ftcs(N, T_end, ALPHA, c, ic)
        ei, e2 = errs(x, u, ex, T_end)
        res["ftcs"].append(dict(N=N, cost=i["flops"], einf=ei, e2=e2, steps=i["steps"]))

        x, u, i = S.solve_spectral(N, T_end, ALPHA, c, ic)
        ei, e2 = errs(x, u, ex, T_end)
        res["spec"].append(dict(N=N, cost=i["flops"], einf=ei, e2=e2))

        for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            x, u, i = S.solve_cn(N, T_end, ALPHA, c, ic, nb)
            ei, e2 = errs(x, u, ex, T_end)
            res["cn"].append(dict(N=N, nb=nb, cost=i["flops"], einf=ei, e2=e2))

        # --- RKC1 / RKC2: the real competitor (footprint s, dt ~ s^2 dx^2/(2 alpha))
        dx = 1.0 / (N - 1)
        for sst in [2, 4, 8, 16, 32]:
            for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                dt = T_end / nb
                lim1 = sst ** 2 * dx ** 2 / (2 * ALPHA)
                lim2 = 0.653 * sst ** 2 * dx ** 2 / (4 * ALPHA)
                for order, lim, key in [(1, lim1, "rkc1"), (2, lim2, "rkc2")]:
                    if dt > lim:
                        continue                      # outside the stability limit
                    try:
                        x, u, i = S.solve_rkc(N, T_end, ALPHA, c, ic, nb, sst, order=order)
                    except Exception:
                        continue
                    if u is None or not np.all(np.isfinite(u)):
                        continue
                    ei, e2 = errs(x, u, ex, T_end)
                    if ei > 1e3:
                        continue
                    res[key].append(dict(N=N, nb=nb, s=sst, cost=i["flops"],
                                         einf=ei, e2=e2))

        for s in [1.0, 1.05, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
            for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                x, u, i = S.solve_wide(N, T_end, ALPHA, c, ic, nb, s)
                if u is None or not np.all(np.isfinite(u)):
                    continue
                ei, e2 = errs(x, u, ex, T_end)
                rec = dict(N=N, nb=nb, s=s, m=i["m"], P=i["P"], cost=i["flops"],
                           einf=ei, e2=e2, sigma=i["sigma_cells"], wmin=i["wmin"])
                res["wide"].append(rec)
                if s <= 1.05:
                    res["frontier"].append(rec)
    return res, ex


def main():
    allres = {}
    for T_end in [0.05, 0.2]:
        print("\n" + "#" * 110)
        print("#  T_end = %.2f   alpha=%.2f  c=%.2f   IC = sin(pi x)+0.5 sin(2 pi x)"
              % (T_end, ALPHA, CVEL))
        print("#" * 110)
        t0 = time.time()
        res, ex = run(T_end)
        allres["T%.2f" % T_end] = res

        # ---------------- headline table: best cost to reach a target accuracy
        print("\nCOST (flops) TO REACH A TARGET L-infinity ERROR  (lower is better)")
        print("%-10s %12s %12s %12s %12s %12s %12s %12s"
              % ("target", "FTCS", "CN", "RKC1", "RKC2", "WIDE(any s)",
                 "WIDE(s<=1.05)", "DST-exact"))
        for tgt in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
            row = ["%.0e" % tgt]
            for key in ["ftcs", "cn", "rkc1", "rkc2", "wide", "frontier", "spec"]:
                ok = [r["cost"] for r in res[key] if r["einf"] <= tgt]
                row.append("%12.3e" % min(ok) if ok else "%12s" % "--")
            print("%-10s %s" % (row[0], " ".join(row[1:])))

        # ---------------- ratios
        print("\nSPEEDUP of the wide stencil (flops FTCS / flops WIDE) at equal accuracy:")
        for tgt in [1e-3, 1e-4, 1e-5, 1e-6]:
            a = [r["cost"] for r in res["ftcs"] if r["einf"] <= tgt]
            b = [r["cost"] for r in res["wide"] if r["einf"] <= tgt]
            cc = [r["cost"] for r in res["cn"] if r["einf"] <= tgt]
            f = [r["cost"] for r in res["frontier"] if r["einf"] <= tgt]
            print("  %.0e : vs FTCS %s   vs CN %s   | frontier-only vs FTCS %s"
                  % (tgt,
                     ("%8.1fx" % (min(a) / min(b))) if a and b else "   n/a  ",
                     ("%8.1fx" % (min(cc) / min(b))) if cc and b else "   n/a  ",
                     ("%8.2fx" % (min(a) / min(f))) if a and f else "   n/a  "))

        # ---------------- best configurations
        print("\nBest wide configs on the Pareto front (T=%.2f):" % T_end)
        pf = pareto([(r["cost"], r["einf"], r) for r in res["wide"]])
        print("%8s %6s %5s %5s %5s %5s %12s %12s"
              % ("cost", "N", "nb", "s", "m", "P", "einf", "sigma_cells"))
        for cst, e, r in pf[::max(len(pf)//12, 1)]:
            print("%8.2e %6d %5d %5.2f %5d %5d %12.3e %12.2f"
                  % (cst, r["N"], r["nb"], r["s"], r["m"], r["P"], e, r["sigma"]))
        print("  (%.1f s)" % (time.time() - t0))

    # ------------------------------------------------------------------ CONTROL
    print("\n" + "#" * 110)
    print("#  CONTROL EXPERIMENT: is the FRONTIER stencil just coarse-grid FTCS?")
    print("#  half-width m on a grid of N points  vs  plain FTCS on ~N/m points")
    print("#" * 110)
    T_end = 0.05
    ex = C.Exact(C.ic_sine, ALPHA, 0.0)
    print("%4s %4s %6s | %12s %12s | %12s %12s | %s"
          % ("N", "m", "N/m", "wide einf", "wide flops", "FTCS einf", "FTCS flops",
             "err ratio"))
    ctrl = []
    for N in [400, 800]:
        for m in [2, 4, 8, 16]:
            # frontier: sigma = m cells  ->  dt_big = (m dx)^2/(2 alpha)
            dx = 1.0 / (N - 1)
            dtb = (m * dx) ** 2 / (2 * ALPHA)
            nb = max(int(round(T_end / dtb)), 1)
            x, u, i = S.solve_wide(N, T_end, ALPHA, 0.0, C.ic_sine, nb, 1.0)
            if u is None:
                continue
            ew, _ = errs(x, u, ex, T_end)
            Nc = max(int(round((N - 1) / m)) + 1, 4)
            xc, uc, ic_ = S.solve_ftcs(Nc, T_end, ALPHA, 0.0, C.ic_sine)
            ec, _ = errs(xc, uc, ex, T_end)
            print("%4d %4d %6.1f | %12.3e %12.3e | %12.3e %12.3e | %8.2f"
                  % (N, i["m"], (N - 1) / i["m"], ew, i["flops"], ec, ic_["flops"],
                     ew / ec))
            ctrl.append(dict(N=N, m=i["m"], ew=ew, cw=i["flops"], ec=ec, cc=ic_["flops"]))
    print("\n  -> if 'err ratio' ~ 1 while wide flops >> FTCS flops, the frontier")
    print("     stencil is an m-fold redundant re-implementation of coarse FTCS.")

    # ------------------------------------------------------------------ BOUNDARIES
    print("\n" + "#" * 110)
    print("#  BOUNDARY HANDLING COST AND ERROR")
    print("#" * 110)
    T_end = 0.05
    ex = C.Exact(C.ic_sine, ALPHA, CVEL)
    print("%4s %4s %4s | %12s %12s | %s"
          % ("N", "nb", "m", "images", "zero-pad", "comment"))
    for N in [100, 400]:
        for nb, s in [(1, 4.0), (4, 4.0), (16, 4.0)]:
            r = {}
            for bm in ["images", "zeropad"]:
                x, u, i = S.solve_wide(N, T_end, ALPHA, CVEL, C.ic_sine, nb, s, bmode=bm)
                r[bm] = errs(x, u, ex, T_end)[0] if u is not None else np.nan
                mm = i["m"]
            print("%4d %4d %4d | %12.3e %12.3e | zero-pad penalty %.0fx"
                  % (N, nb, mm, r["images"], r["zeropad"], r["zeropad"] / r["images"]))
    # substepping cost
    print("\n  Alternative boundary fix: FTCS sub-stepping in a wall layer.")
    print("  To advance k dt with FTCS you need a halo of k cells, and k = m^2/(2 nu),")
    print("  so the 'layer' is m + k cells wide, NOT m:")
    print("  %6s %6s %8s %10s %14s" % ("m", "k", "layer", "N=400?", "layer/N"))
    for m in [4, 8, 16, 24]:
        k = int(m ** 2 / (2 * 0.45))
        print("  %6d %6d %8d %10s %14.2f"
              % (m, k, m + k, "yes" if m + k < 400 else "OVERRUNS", (m + k) / 400))

    json.dump(allres, open(os.path.join(HERE, "task3_results.json"), "w"),
              indent=1, default=float)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, T_end in zip(axes, [0.05, 0.2]):
        res = allres["T%.2f" % T_end]
        for key, col, lab, mk in [("ftcs", "#1b6ca8", "FTCS @ CFL", "o"),
                                  ("cn", "#2e8b57", "Crank-Nicolson", "^"),
                                  ("rkc1", "#8c564b", "RKC1 ($s$ stages)", "v"),
                                  ("rkc2", "#17a2b2", "RKC2 ($s$ stages)", "P"),
                                  ("wide", "#c4432b", "wide positive (best $s$)", "s"),
                                  ("frontier", "#e08a1e", r"wide @ frontier ($s\!\approx\!1$)", "d"),
                                  ("spec", "#6a3d9a", "DST exact (reference)", "*")]:
            pf = pareto([(r["cost"], r["einf"], r) for r in res[key]])
            if not pf:
                continue
            ax.loglog([p[0] for p in pf], [p[1] for p in pf], mk + "-", color=col,
                      ms=5, lw=1.5, label=lab)
        ax.set_xlabel("cost  [flops]"); ax.set_ylabel(r"$\|u-u_{\rm exact}\|_\infty$ at $T$")
        ax.set_title("$T_{\\rm end}=%.2f$" % T_end)
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
        ax.set_ylim(1e-13, 1)
    fig.suptitle("Work-precision, $u_t+u_x=0.1\\,u_{xx}$ on $[0,1]$, Dirichlet", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_workprecision.png"), dpi=150)
    print("\nwrote figures/fig3_workprecision.png")


if __name__ == "__main__":
    main()
