"""Does the projected / learned operator reduce to FTCS or Lax-Wendroff when the points are uniform?

For each arm, on a uniform periodic grid, K = 3, 5, 7, 9: the interior weights, their distance
from the 3-point LW-diffusion row (propagator arms) or the FTCS row (MOL arms), the sign
structure, and the one-step symbol |g(theta)|. Prints tables; figures/uniform_reduction.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import g1lib as g

np.set_printoptions(precision=4, suppress=True, linewidth=150)
N = 100
x = g.make_points(N, 0.0, periodic=True)
h = 1.0 / N
th = np.linspace(-np.pi, np.pi, 1025)


def row_and_symbol(op, i=50, K=9):
    offs = np.arange(-K, K + 1)
    w = np.array([op.M1[i, (i + o) % N] for o in offs])
    gsym = np.array([np.sum(w * np.exp(-1j * t * offs)) for t in th])
    return offs, w, gsym


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cases = [(0.1, "Pe = 0.10, r = 0.25, nu = 0.025"), (0.01, "Pe = 1.0, r = 0.25, nu = 0.25"), (0.003, "Pe = 3.3, r = 0.15, nu = 0.5")]
for ci, (alpha, label) in enumerate(cases):
    p = g.Problem(alpha=alpha, c=1.0)
    dt = g.time_step(x, p, 0.5, periodic=True)
    lw, ft = g.lw_weights(p, h, dt), g.ftcs_weights(p, h, dt)
    print(f"\n=== alpha={alpha}  {label}  dt={dt:.3e}  LW row={lw}  FTCS row={ft}  LW max|g|={g.st.symbol_max(lw,[-1,0,1]):.4f}  FTCS max|g|={g.st.symbol_max(ft,[-1,0,1]):.4f}")
    print(f"{'arm':10}{'K':>3}{'max|w-LW| (3 central)':>24}{'max|w-FTCS|':>13}{'|w| outside 3pt':>17}{'min w':>9}{'l1':>8}{'max|g|':>9}{'|g(pi)|':>9}")
    ax_w, ax_g = axes[0, ci], axes[1, ci]
    for arm in ["minnorm", "projgauss", "spectral", "spectralg", "molfe", "maxent", "lp"]:
        for K in [3, 5, 7, 9]:
            op = g.Operator(x, p, dt, arm, K, periodic=True)
            offs, w, gsym = row_and_symbol(op)
            c3 = w[(offs >= -1) & (offs <= 1)]
            outside = np.abs(w[(offs < -1) | (offs > 1)]).max()
            print(f"{arm:10}{K:3d}{np.abs(c3-lw).max():24.2e}{np.abs(c3-ft).max():13.2e}{outside:17.2e}{w.min():9.4f}{np.abs(w).sum():8.4f}{np.abs(gsym).max():9.4f}{abs(gsym[0]):9.4f}")
            if K == 5:
                ax_w.plot(offs[3:-3], w[3:-3], "o-", ms=4, label=arm)
                ax_g.plot(th, np.abs(gsym), label=arm)
    ax_w.plot([-1, 0, 1], lw, "k*", ms=10, label="LW (3-pt)")
    ax_w.set_title(f"K=5 weights, {label}"); ax_w.set_xlabel("offset / h"); ax_w.axhline(0, color="gray", lw=0.5)
    ax_g.axhline(1, color="gray", lw=0.5); ax_g.set_ylim(0, 1.5); ax_g.set_xlabel("theta"); ax_g.set_title("|g(theta)| one-step symbol, K=5")
    if ci == 0:
        ax_w.legend(fontsize=7); ax_g.legend(fontsize=7)
plt.tight_layout()
plt.savefig("figures/uniform_reduction.png", dpi=150)
print("\nsaved figures/uniform_reduction.png")

# the mechanism, in closed form: min-norm p=2 Laplacian on 5 uniform points
print("\n=== closed form: min-norm 5-pt second-derivative weights * h^2 (offsets -2..2) ===")
D1, D2 = g.spatial_ops(x, 5, periodic=True)
print("measured :", np.array([D2[50, (50 + o) % N] for o in [-2, -1, 0, 1, 2]]) * h**2)
print("predicted:", np.array([2, -1, -2, -1, 2]) / 7.0, "  (a=2/7, b=-1/7, c=-2/7: minimise 2a^2+2b^2+(2a+2b)^2 s.t. b+4a=1)")
print("D2 symbol at theta=pi  * h^2 :", (4 * np.cos(2 * np.pi) - 2 * np.cos(np.pi) - 2) / 7.0, " (> 0: anti-diffusive at Nyquist)")
