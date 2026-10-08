"""TASK 10 -- the price of positivity: achievable error vs ||w||_1.

For a fixed (m,k) solve the LP

    minimise  t
    over w, t   s.t.   |W(theta_i) - G(theta_i)| <= t   for a grid of theta,
                       sum_i w_i = 1,
                       ||w||_1 <= B

W(theta) = sum_j w_j e^{i j theta},  G(theta) = exp(-i k Co theta - k nu theta^2).

Because the consistency row sum w = 1 is imposed, ||w||_1 >= 1 with equality
IF AND ONLY IF w >= 0.  So B = 1 is exactly the positive/monotone case and
B > 1 buys accuracy by giving up monotonicity.  The curve error(B) is the
Pareto frontier of the whole idea.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
nu, co = C.NU_REF, C.CO_REF


def best_error(m, k, B, nu=nu, co=co, th_max=np.pi / 2, nth=64):
    """min over w of max_theta |W-G| subject to sum w = 1 and ||w||_1 <= B."""
    j = np.arange(-m, m + 1)
    n = len(j)
    th = np.linspace(1e-6, th_max, nth)
    E = np.exp(1j * np.outer(th, j))                   # nth x n
    G = np.exp(-1j * k * co * th - k * nu * th ** 2)
    # variables: p (n), q (n), t   with w = p - q, p,q >= 0
    # |Re(E(p-q)) - ReG| <= t  ->  +-(Re E)(p-q) - t <= +-ReG
    ReE, ImE = E.real, E.imag
    A_ub = np.vstack([
        np.hstack([ReE, -ReE, -np.ones((nth, 1))]),
        np.hstack([-ReE, ReE, -np.ones((nth, 1))]),
        np.hstack([ImE, -ImE, -np.ones((nth, 1))]),
        np.hstack([-ImE, ImE, -np.ones((nth, 1))]),
        np.hstack([np.ones((1, n)), np.ones((1, n)), np.zeros((1, 1))]),  # ||w||_1 <= B
    ])
    b_ub = np.concatenate([G.real, -G.real, G.imag, -G.imag, [B]])
    A_eq = np.hstack([np.ones((1, n)), -np.ones((1, n)), np.zeros((1, 1))])
    r = linprog(np.r_[np.zeros(2 * n), 1.0], A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=np.array([1.0]),
                bounds=[(0, None)] * (2 * n) + [(0, None)], method="highs")
    if r.status != 0:
        return np.nan, None
    w = r.x[:n] - r.x[n:2 * n]
    return r.x[-1], w


def main():
    """NOTE ON THE METRIC.  The error is measured over the RESOLVED band
    theta in (0, pi/2].  Measuring out to theta = pi is useless: W(pi) = sum_j
    w_j (-1)^j is REAL for any real w, while G(pi) has imaginary part
    -sin(k Co pi) e^{-k nu pi^2}, so the full-band error is floored by an
    ALIASING term that no choice of weights -- positive or signed -- can touch.
    That floor masks the whole effect; my first pass hit it and wrongly concluded
    positivity was free everywhere."""
    Bs = np.r_[1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0, 4.0, 6.0, 10.0]

    print("=" * 104)
    print("A.  DIFFUSION-DOMINATED (at the frontier, s = m/sigma = 1): is positivity free?")
    print("=" * 104)
    res_a = {}
    print("  %-22s | %s" % ("(m, k)", "err at ||w||_1 = 1, 1.5, 3, 10   ->  total gain"))
    for m, k in [(6, 40), (12, 160), (20, 273)]:
        e = np.array([best_error(m, k, B)[0] for B in Bs])
        res_a[(m, k)] = e
        print("  m=%2d k=%3d (s=%.2f)     | %.3e  %.3e  %.3e  %.3e   ->  %.2fx"
              % (m, k, m / np.sqrt(2 * k * nu), e[0], e[4], e[7], e[-1], e[0] / e[-1]))
    print("\n  -> essentially FLAT: a 10x budget in ||w||_1 buys ~1.1x accuracy.")
    print("     The binding constraint is the SUPPORT, not the sign.  A kernel with")
    print("     sigma = m cells keeps ~32%% of its mass beyond +-m, and no weight vector")
    print("     on {-m..m} can represent mass it cannot reach.  POSITIVITY IS FREE HERE.")

    print("\n" + "=" * 104)
    print("B.  ADVECTION-DOMINATED: positivity costs once the kernel is SUB-CELL")
    print("    m=8, k=1, Co=0.5 (worst-case half-cell shift); sweep nu, i.e. sigma=sqrt(2nu)")
    print("=" * 104)
    print("  %8s %8s | %11s %11s %11s | %12s"
          % ("nu", "sigma", "B=1 (pos)", "B=1.5", "B=10", "gain"))
    res_b = {}
    nus = [0.45, 0.2, 0.1, 0.05, 0.02, 0.01, 0.003]
    for nu_ in nus:
        e = np.array([best_error(8, 1, B, nu=nu_, co=0.5)[0] for B in Bs])
        res_b[nu_] = e
        print("  %8.3f %8.3f | %11.4e %11.4e %11.4e | %10.1fx"
              % (nu_, np.sqrt(2 * nu_), e[0], e[4], e[-1], e[0] / max(e[-1], 1e-16)))
    print("\n  -> up to ~2e6x.  This is GODUNOV'S BARRIER in the LP: a monotone linear")
    print("     scheme is at most first-order, so a narrow kernel displaced by a")
    print("     non-integer number of cells needs negative weights (Lax-Wendroff and up).")
    print("     Note the cost of positivity appears exactly where the METHOD is weak")
    print("     anyway -- sub-cell kernels mean the step is small, i.e. no super-stepping.")

    print("\n" + "=" * 104)
    print("C.  WHICH KNOB ACTUALLY PAYS AT THE FRONTIER?  (m = 12 fixed)")
    print("=" * 104)
    m, kf = 12, 160
    e_pos = best_error(m, kf, 1.0)[0]
    print("  %6s | %12s %12s | %s" % ("k", "err (B=1)", "cost/time", "vs frontier"))
    for k in [160, 80, 40, 20, 10, 5]:
        e = best_error(m, k, 1.0)[0]
        print("  %6d | %12.4e %12.3e | %10s better, %2dx dearer"
              % (k, e, (2 * (2 * m + 1) - 1) / (k * C.DT),
                 ("%.1fx" % (e_pos / e)) if e > 1e-12 else ">1e10x", kf // k))
    print("\n  -> backing off k buys 1e4x for 16x cost; relaxing ||w||_1 to 10 buys 1.1x.")
    print("     CONCLUSION: monotonicity is NOT what limits this scheme in its own")
    print("     (diffusion-dominated) regime.  Giving it up would buy almost nothing.")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.9))
    cols = ["#1b6ca8", "#c4432b", "#2e8b57"]
    for mk, cl in zip(res_a, cols):
        m, k = mk
        ax[0].semilogy(Bs, res_a[mk], "o-", color=cl, ms=4,
                       label="$m=%d$, $k=k_{\\max}=%d$" % (m, k))
    for k, ls in zip([40, 10], ["--", ":"]):
        ax[0].semilogy(Bs, [max(best_error(12, k, B)[0], 1e-13) for B in Bs], ls,
                       color="#777777", lw=1.4, label="$m=12$, $k=%d$ (backed off)" % k)
    ax[0].axvline(1.0, color="k", lw=1.2, ls="--")
    ax[0].text(1.08, 3e-5, "positive / monotone", rotation=90, fontsize=8, va="center")
    ax[0].set_xlabel(r"$\|w\|_1$   ($1 \Leftrightarrow$ non-negative weights)")
    ax[0].set_ylabel(r"best achievable $\max_{\theta\leq\pi/2}|W-\hat G|$")
    ax[0].set_title("(a) diffusion-dominated: positivity is free")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, which="both")

    for nu_, cl in zip(nus, plt.cm.plasma(np.linspace(0, .85, len(nus)))):
        ax[1].semilogy(Bs, np.maximum(res_b[nu_], 1e-13), "o-", color=cl, ms=3.5,
                       label=r"$\sigma=%.2f$ cells" % np.sqrt(2 * nu_))
    ax[1].axvline(1.0, color="k", lw=1.2, ls="--")
    ax[1].set_xlabel(r"$\|w\|_1$")
    ax[1].set_ylabel(r"best achievable $\max_{\theta\leq\pi/2}|W-\hat G|$")
    ax[1].set_title("(b) advection-dominated ($\\mathrm{Co}=0.5$): positivity costs")
    ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3, which="both")
    fig.suptitle("Achievable one-step error vs $\\|w\\|_1$ -- the price of monotonicity",
                 y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig6_l1_pareto.png"), dpi=150)
    print("\nwrote figures/fig6_l1_pareto.png")


if __name__ == "__main__":
    main()
