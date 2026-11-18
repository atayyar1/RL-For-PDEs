"""TASK 2 -- truncation error along the positivity frontier (rigorous version).

Frontier kernel (c=0), EXACTLY moment-consistent for every m:
    w_{+-m} = p/2,  w_0 = 1-p,   p = 2 k nu / m^2 <= 1
    -> M2 = 2 k nu (exact),  M4 = p m^4.
Symbol error   E(theta) = W - exp(-k nu theta^2) = (M4/24 - M2^2/8) theta^4 + O(theta^6)
             = (p m^4/24)(1 - 3p) theta^4 ;  at the frontier p -> 1: -(m^4/12) theta^4.
Per unit simulated time (k dt = m^2 dx^2/(2 alpha) when p=1):
    LTE / (k dt) = alpha (m dx)^2 |u_xxxx| / 6        ->  C = alpha/6
For c != 0 with the SPEC rows the dropped u_tt term contributes
    LTE / (k dt) = (pi^2 c^2 / (4 alpha)) (m dx)^2 |u|   (mode 1)
so BOTH terms are O((m dx)^2) per unit time and their ratio is
    (3/2) (c / (alpha pi))^2 ,  = 15.2 for alpha=0.1, c=1.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
HERE = os.path.dirname(os.path.abspath(__file__))
nu, co = C.NU_REF, C.CO_REF


def frontier_kernel(m, k, nu=nu):
    """Extremal symmetric positive kernel with M2 = 2 k nu exactly (c = 0)."""
    p = 2.0 * k * nu / m ** 2
    assert p <= 1 + 1e-12, (m, k, p)
    w = np.zeros(2 * m + 1)
    w[0] = w[-1] = p / 2
    w[m] = 1.0 - p
    return w


def lte(w, m, k, exact, T0=0.10, c=C.CVEL, npts=401):
    lo, hi = m * C.DX, 1.0 - m * C.DX
    xs = np.linspace(lo, hi, npts)
    j = np.arange(-m, m + 1)
    ap = np.zeros_like(xs)
    for wi, ji in zip(w, j):
        if wi != 0.0:
            ap += wi * exact(xs + ji * C.DX, T0)
    return np.max(np.abs(ap - exact(xs, T0 + k * C.DT)))


def main():
    T0 = 0.10
    # single-mode reference so that u_xxxx = pi^4 u exactly
    ex0_1 = C.Exact(lambda x: np.sin(np.pi * x), C.ALPHA, 0.0)
    ex0_2 = C.Exact(C.ic_sine, C.ALPHA, 0.0)
    ex1_1 = C.Exact(lambda x: np.sin(np.pi * x), C.ALPHA, C.CVEL)
    U1 = np.max(np.abs(ex0_1(np.linspace(0, 1, 2001), T0)))
    U4 = np.pi ** 4 * U1                                    # max |u_xxxx|, mode 1

    ms = list(range(2, 26))
    rows = []
    print("=" * 118)
    print("A.  c = 0, FRONTIER k = k_max(m) = floor(m^2/(2 nu)).  IC = sin(pi x), T0 = %.2f"
          % T0)
    print("    prediction:  LTE/(k dt) = C (m dx)^2 |u_xxxx| with C = alpha/6 = %.5f"
          % (C.ALPHA / 6))
    print("=" * 118)
    print("%3s %5s %8s %6s | %11s %11s | %11s %11s"
          % ("m", "k", "k*dt", "p", "LTE_inf", "LTE/(k dt)", "C measured", "C pred"))
    for m in ms:
        k = int(np.floor(m ** 2 / (2 * nu)))
        w = frontier_kernel(m, k)
        p = 2 * k * nu / m ** 2
        e = lte(w, m, k, ex0_1, T0, c=0.0)
        Cm = e / (k * C.DT) / ((m * C.DX) ** 2 * U4)
        rows.append(dict(m=m, k=k, p=p, e=e, rate=e / (k * C.DT), Cm=Cm))
        if m % 2 == 0 or m < 6:
            print("%3d %5d %8.4f %6.3f | %11.4e %11.4e | %11.5f %11.5f"
                  % (m, k, k * C.DT, p, e, e / (k * C.DT), Cm, C.ALPHA / 6))
    mdx = np.array([r["m"] for r in rows]) * C.DX
    rate = np.array([r["rate"] for r in rows])
    sl = np.polyfit(np.log(mdx), np.log(rate), 1)[0]
    print("\n  fitted slope d log[LTE/(k dt)] / d log(m dx) = %.4f   (prediction 2)" % sl)
    print("  mean measured C over m>=6 = %.5f   (prediction alpha/6 = %.5f)"
          % (np.mean([r["Cm"] for r in rows if r["m"] >= 6]), C.ALPHA / 6))

    # ---- symbol-level check of the theta^4 coefficient
    print("\n" + "=" * 118)
    print("B.  SYMBOL CHECK:  E(theta) = (M4/24 - M2^2/8) theta^4 + O(theta^6)")
    print("=" * 118)
    print("%3s %5s | %14s %14s | %8s" % ("m", "k", "measured coef", "predicted coef", "ratio"))
    th = 1e-3
    for m in [4, 8, 12, 16, 20, 24]:
        k = int(np.floor(m ** 2 / (2 * nu)))
        w = frontier_kernel(m, k)
        j = np.arange(-m, m + 1)
        M2 = (w * j ** 2).sum(); M4 = (w * j ** 4).sum()
        meas = np.real(C.symbol_error(w, k, nu, 0.0, np.array([th]))[0]) / th ** 4
        pred = M4 / 24 - M2 ** 2 / 8
        print("%3d %5d | %14.6e %14.6e | %8.5f" % (m, k, meas, pred, meas / pred))

    # ---- c != 0 : which term dominates
    print("\n" + "=" * 118)
    print("C.  c = 1, SPEC ROWS:  the dropped u_tt term.  Which term limits accuracy?")
    print("=" * 118)
    C_tt = np.pi ** 2 * C.CVEL ** 2 / (4 * C.ALPHA)
    C_xx = C.ALPHA * np.pi ** 4 / 6
    print("  predicted rates (per unit time, per (m dx)^2, mode 1, units of |u|):")
    print("     u_tt  term : pi^2 c^2/(4 alpha)   = %8.3f" % C_tt)
    print("     u_xxxx term: alpha pi^4/6         = %8.3f" % C_xx)
    print("     ratio                             = %8.3f   = (3/2)(c/(alpha pi))^2 = %.3f"
          % (C_tt / C_xx, 1.5 * (C.CVEL / (C.ALPHA * np.pi)) ** 2))
    print("\n%3s %5s | %11s %11s | %11s %11s"
          % ("m", "k", "LTE/(k dt)", "/(mdx)^2/|u|", "pred tot", "pred/meas"))
    rows1 = []
    U1a = np.max(np.abs(ex1_1(np.linspace(0, 1, 2001), T0)))
    for m in ms:
        k = C.kmax_analytic(m, nu, co)
        ok, w = C.w_positive(np.arange(-m, m + 1) * C.DX,
                             -k * C.DT * np.ones(2 * m + 1), C.ALPHA, C.CVEL)
        e = lte(w, m, k, ex1_1, T0)
        r = e / (k * C.DT) / ((m * C.DX) ** 2 * U1a)
        rows1.append(dict(m=m, k=k, e=e, rate=e / (k * C.DT), norm=r))
        if m % 3 == 0:
            print("%3d %5d | %11.4e %11.4f | %11.4f %11.4f"
                  % (m, k, e / (k * C.DT), r, C_tt + C_xx, (C_tt + C_xx) / r))
    print("\n  -> VERDICT: with the spec's 3 rows and c != 0, the u_tt error term")
    print("     dominates the (m dx)^2 u_xxxx term by ~%.0fx.  Both scale as (m dx)^2"
          % (C_tt / C_xx))
    print("     per unit time, because k ~ m^2, so the SCALING claim survives but the")
    print("     CONSTANT is set by advection, not by the stencil width.")

    # ---- corrected rows + safety factor
    print("\n" + "=" * 118)
    print("D.  CORRECTED ROWS (M2 = 2 k nu + (k Co)^2) and safety factor s = m/sigma")
    print("=" * 118)
    corr = []
    for m in ms:
        for s in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            k = int(np.floor(m ** 2 / (2 * nu * s ** 2)))
            if k < 1:
                continue
            w = C.kernel_exact_semigroup(m, k, nu, co, order="exact")
            if w is None:
                continue
            e = lte(w, m, k, ex1_1, T0)
            corr.append(dict(m=m, k=k, s=s, e=e, rate=e / (k * C.DT),
                             nnz=int((w > 1e-16).sum())))
    print("%3s | " % "m" + " ".join("%11s" % ("s=%.1f" % s)
                                    for s in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]))
    for m in [4, 8, 12, 16, 20, 24]:
        line = "%3d | " % m
        for s in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            d = [x for x in corr if x["m"] == m and x["s"] == s]
            line += "%11.3e " % (d[0]["rate"] if d else np.nan)
        print(line)
    print("  (LTE per unit simulated time; s=1 is the frontier)")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
    ax[0].loglog(mdx, rate, "o-", color="#1b6ca8", label=r"$c=0$ frontier, measured")
    ax[0].loglog(mdx, C.ALPHA / 6 * mdx ** 2 * U4, "k--", lw=1.3,
                 label=r"$\frac{\alpha}{6}(m\Delta x)^2|u_{xxxx}|$")
    r1 = np.array([r["rate"] for r in rows1])
    ax[0].loglog(mdx, r1, "s-", color="#c4432b", label=r"$c=1$ spec rows, measured")
    ax[0].loglog(mdx, (C_tt + C_xx) * mdx ** 2 * U1a, "k:", lw=1.3,
                 label=r"$[\frac{\pi^2c^2}{4\alpha}+\frac{\alpha\pi^4}{6}](m\Delta x)^2|u|$")
    ax[0].set_xlabel(r"$m\Delta x$"); ax[0].set_ylabel(r"LTE$_\infty$ / (k$\Delta t$)")
    ax[0].set_title("(a) truncation rate along the frontier")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, which="both")

    for s, cl in zip([1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                     plt.cm.viridis(np.linspace(0, .88, 6))):
        d = sorted([x for x in corr if x["s"] == s], key=lambda z: z["m"])
        if d:
            ax[1].loglog([x["m"] * C.DX for x in d], [x["rate"] for x in d],
                         "o-", ms=3.5, color=cl, label="$s=%.1f$" % s)
    ax[1].set_xlabel(r"$m\Delta x$"); ax[1].set_ylabel(r"LTE$_\infty$ / (k$\Delta t$)")
    ax[1].set_title("(b) corrected rows: safety factor $s=m/\\sigma$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")

    th = np.linspace(1e-3, np.pi, 500)
    for m, cl in zip([6, 12, 20], ["#1b6ca8", "#c4432b", "#2e8b57"]):
        k = int(np.floor(m ** 2 / (2 * nu)))
        ax[2].semilogy(th, np.abs(C.symbol_error(frontier_kernel(m, k), k, nu, 0., th)),
                       color=cl, lw=1.4, label="$m=%d$, frontier ($s$=1)" % m)
        ks = max(int(np.floor(m ** 2 / (2 * nu * 9))), 1)
        ws = C.kernel_exact_semigroup(m, ks, nu, 0.0)
        if ws is not None:
            ax[2].semilogy(th, np.abs(C.symbol_error(ws, ks, nu, 0., th)), "--",
                           color=cl, lw=1.2, label="$m=%d$, $s=3$" % m)
    ax[2].axvline(np.pi * C.DX, color="k", lw=.8, ls="-.")
    ax[2].text(np.pi * C.DX * 1.15, 1e-13, "mode 1", fontsize=7, rotation=90)
    ax[2].set_xlabel(r"$\theta=\xi\Delta x$"); ax[2].set_ylabel(r"$|W(\theta)-\hat G(\theta)|$")
    ax[2].set_xscale("log"); ax[2].set_ylim(1e-17, 10)
    ax[2].set_title("(c) one-big-step symbol error, $c=0$")
    ax[2].legend(fontsize=7, loc="lower right"); ax[2].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_truncation.png"), dpi=150)
    print("\nwrote figures/fig2_truncation.png")
    json.dump(dict(c0=rows, c1=rows1, corrected=corr),
              open(os.path.join(HERE, "task2_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
