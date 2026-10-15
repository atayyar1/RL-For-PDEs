"""TASK 4 -- adaptive stencil half-width.

IMPORTANT STRUCTURAL POINT (tested below).  With a single GLOBAL big step k dt,
positivity forces  m_j >= sigma = sqrt(2 alpha k dt)/dx  at EVERY point: you may
not shrink the stencil near a steep gradient, because the measure you must
represent has that variance everywhere.  What can legitimately adapt is the
SAFETY FACTOR s_j = m_j/sigma, and the truncation error is governed by
    LTE ~ (M_{2p+2} mismatch) * u^{(2p+2)}
so you need LARGE s where the solution is ROUGH and s ~ 1 suffices where it is
SMOOTH -- the OPPOSITE of "large m where smooth, small m near steep gradients".
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C, solvers as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
ALPHA = 0.1


# ---------------------------------------------------------------- exact solutions
class ExactSine:
    def __init__(s, alpha=ALPHA): s.a = alpha
    def __call__(s, x, t):
        x = np.atleast_1d(np.asarray(x, float))
        return (np.sin(np.pi * x) * np.exp(-s.a * np.pi ** 2 * t)
                + 0.5 * np.sin(2 * np.pi * x) * np.exp(-s.a * 4 * np.pi ** 2 * t))


class ExactSeries:
    """u_t = alpha u_xx, u(0)=u(1)=0, from given sine coefficients."""
    def __init__(s, bn, alpha=ALPHA):
        s.bn = np.asarray(bn); s.n = np.arange(1, len(bn) + 1); s.a = alpha
    def __call__(s, x, t):
        x = np.atleast_1d(np.asarray(x, float))
        return (np.sin(np.pi * np.outer(x, s.n))
                * (s.bn * np.exp(-s.a * (s.n * np.pi) ** 2 * t))).sum(-1)


def square_exact(a=0.3, b=0.5, nm=4000):
    n = np.arange(1, nm + 1)
    return ExactSeries(2 * (np.cos(n * np.pi * a) - np.cos(n * np.pi * b)) / (n * np.pi))


def gauss_exact(x0=0.35, sg=0.03, nm=2000, nq=200001):
    xq = np.linspace(0, 1, nq); f = np.exp(-((xq - x0) ** 2) / (2 * sg ** 2))
    n = np.arange(1, nm + 1)
    bn = np.array([2 * np.trapz(f * np.sin(k * np.pi * xq), xq) for k in n])
    return ExactSeries(bn)


# ---------------------------------------------------------------- adaptive solver
def smoothness(u, dx):
    """Cheap indicator: |D^4 u| * dx^4 normalised -- large where the solution is rough."""
    d4 = np.zeros_like(u)
    d4[2:-2] = u[:-4] - 4 * u[1:-3] + 6 * u[2:-2] - 4 * u[3:-1] + u[4:]
    d4[:2] = d4[2]; d4[-2:] = d4[-3]
    scale = np.maximum(np.max(np.abs(u)), 1e-30)
    return np.abs(d4) / scale


def solve_adaptive(N, T_end, alpha, ic, nsteps, s_lo=1.2, s_hi=6.0,
                   policy="rough_wide", fixed_s=None, ind_ref=None):
    """policy: 'rough_wide'  -> s large where rough      (theory)
               'smooth_wide' -> s large where smooth     (prompt hypothesis)
               'fixed'       -> s = fixed_s everywhere
    Weights are cached per half-width; each row is a positive measure so the
    scheme is monotone regardless of how m varies in space."""
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    dtb = T_end / nsteps
    sig = np.sqrt(2 * alpha * dtb) / dx
    u = np.asarray(ic(x), float); u[0] = u[-1] = 0.0
    L = 2 * (N - 1)
    flops = 0
    mhist = []
    for _ in range(nsteps):
        if policy == "fixed":
            svec = np.full(N, fixed_s)
        else:
            ind = smoothness(u, dx)
            r = ind / (ind_ref if ind_ref else max(ind.max(), 1e-30))
            r = np.clip(r, 0, 1) ** 0.25
            svec = s_lo + (s_hi - s_lo) * (r if policy == "rough_wide" else (1 - r))
        mvec = np.maximum(np.ceil(svec * sig).astype(int) + 1, 2)
        mhist.append(mvec.copy())
        # extended (odd-reflected) array
        U = np.zeros(L); U[:N] = u; U[N:] = -u[-2:0:-1]
        cache = {}
        out = np.zeros(N)
        for mval in np.unique(mvec):
            if mval not in cache:
                w = C.kernel_moment_matched(int(mval), 0.0, sig ** 2)
                if w is None:
                    p = sig ** 2 / mval ** 2
                    w = np.zeros(2 * mval + 1); w[0] = w[-1] = p / 2; w[mval] = 1 - p
                cache[mval] = w
            w = cache[mval]
            sel = np.where(mvec == mval)[0]
            nz = np.nonzero(w)[0]
            acc = np.zeros(len(sel))
            for q in nz:
                acc += w[q] * U[(sel + (q - mval)) % L]
            out[sel] = acc
            flops += len(sel) * (2 * len(nz) - 1)
        out[0] = out[-1] = 0.0
        u = out
    return x, u, dict(flops=flops, mhist=mhist, sigma=sig, steps=nsteps)


def einf(x, u, ex, T):
    return float(np.max(np.abs(u - ex(x, T))))


def main():
    cases = [("smooth sine", C.ic_sine, ExactSine(), 0.05),
             ("narrow Gaussian (sd=0.03)", C.ic_gauss, gauss_exact(), 0.02),
             ("square wave [0.3,0.5]", C.ic_square, square_exact(), 0.02)]
    N = 200
    allres = {}
    for name, ic, ex, T_end in cases:
        print("\n" + "=" * 104)
        print("CASE: %s    N=%d  T_end=%.3f  (pure diffusion c=0)" % (name, N, T_end))
        print("=" * 104)
        recs = []
        for nb in [2, 4, 8, 16, 32, 64, 128]:
            row = dict(nb=nb)
            for lab, kw in [("s=1.2", dict(policy="fixed", fixed_s=1.2)),
                            ("s=3", dict(policy="fixed", fixed_s=3.0)),
                            ("s=6", dict(policy="fixed", fixed_s=6.0)),
                            ("adapt rough-wide", dict(policy="rough_wide")),
                            ("adapt smooth-wide", dict(policy="smooth_wide"))]:
                x, u, i = solve_adaptive(N, T_end, ALPHA, ic, nb, **kw)
                row[lab] = (einf(x, u, ex, T_end), i["flops"],
                            float(np.mean(i["mhist"][0])))
            x, u, i = S.solve_ftcs(N, T_end, ALPHA, 0.0, ic)
            row["FTCS"] = (einf(x, u, ex, T_end), i["flops"], 1.0)
            recs.append(row)
        hdr = ["s=1.2", "s=3", "s=6", "adapt rough-wide", "adapt smooth-wide"]
        print("%4s | " % "nb" + " | ".join("%22s" % h for h in hdr))
        print("%4s | " % "" + " | ".join("%10s %11s" % ("einf", "flops")
                                         for _ in hdr))
        for r in recs:
            print("%4d | " % r["nb"] + " | ".join(
                "%10.3e %11.3e" % (r[h][0], r[h][1]) for h in hdr))
        print("FTCS @CFL: einf = %.3e  flops = %.3e" % (recs[0]["FTCS"][0],
                                                        recs[0]["FTCS"][1]))
        # efficiency: best error at comparable cost
        print("\n  cost to reach the best fixed-s error, adaptive vs fixed:")
        for r in recs[::2]:
            best_fixed = min([r[h] for h in ["s=1.2", "s=3", "s=6"]],
                             key=lambda z: z[0])
            ar = r["adapt rough-wide"]; asw = r["adapt smooth-wide"]
            print("   nb=%3d  best fixed (e=%.2e, c=%.2e) | rough-wide (%.2e, %.2e) "
                  "err x%.2f cost x%.2f | smooth-wide (%.2e, %.2e) err x%.2f"
                  % (r["nb"], best_fixed[0], best_fixed[1], ar[0], ar[1],
                     ar[0] / best_fixed[0], ar[1] / best_fixed[1],
                     asw[0], asw[1], asw[0] / best_fixed[0]))
        allres[name] = [{k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in r.items()} for r in recs]

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2))
    for col, (name, ic, ex, T_end) in enumerate(cases):
        ax = axes[0, col]
        for lab, cl in [("s=1.2", "#e08a1e"), ("s=3", "#1b6ca8"), ("s=6", "#2e8b57"),
                        ("adapt rough-wide", "#c4432b"),
                        ("adapt smooth-wide", "#777777")]:
            pts = [(r[lab][1], r[lab][0]) for r in allres[name]]
            pts.sort()
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", ms=4,
                      color=cl, label=lab)
        f = allres[name][0]["FTCS"]
        ax.loglog([f[1]], [f[0]], "k*", ms=12, label="FTCS @ CFL")
        ax.set_title(name, fontsize=10); ax.grid(alpha=.3, which="both")
        ax.set_xlabel("flops"); ax.set_ylabel(r"$\|e\|_\infty$")
        if col == 0:
            ax.legend(fontsize=7)
        # bottom row: profile + chosen m
        ax2 = axes[1, col]
        x, u, i = solve_adaptive(N, T_end, ALPHA, ic, 16, policy="rough_wide")
        ax2.plot(x, ic(x), color="#bbbbbb", lw=1, label="initial")
        ax2.plot(x, u, color="#c4432b", lw=1.6, label="adaptive @ $T$")
        ax2.plot(x, ex(x, T_end), "k--", lw=1, label="exact")
        ax2.set_xlabel("$x$"); ax2.set_ylabel("$u$")
        a3 = ax2.twinx()
        a3.plot(x, i["mhist"][0], color="#1b6ca8", lw=1, alpha=.7)
        a3.set_ylabel("chosen $m$ (first step)", color="#1b6ca8", fontsize=8)
        a3.tick_params(axis="y", labelcolor="#1b6ca8", labelsize=7)
        if col == 0:
            ax2.legend(fontsize=7)
    fig.suptitle("Adaptive half-width: wide where ROUGH beats wide where SMOOTH", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_adaptive.png"), dpi=150)
    print("\nwrote figures/fig4_adaptive.png")
    json.dump(allres, open(os.path.join(HERE, "task4_results.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
