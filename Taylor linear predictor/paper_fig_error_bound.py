"""
paper_fig_error_bound.py
========================

Paper figure: numerical verification of the local predictor error bounds, for
the generic predictor and for the PDE-substituted one, on the SAME stencils.

PDE:      u_t + c u_x = alpha u_xx
Test field (an exact solution of it):
          u(x,t) = sin(pi (x - c t)) * exp(-alpha pi^2 t)

Generic predictor -- moment rows [1, dx, dt]:

        |e| <= ||w||_1 * C * h^2 ,   C = max(|u_xx|,|u_xt|,|u_tt|),
                                     h = max_i ||dz_i||_2 .

PDE-substituted predictor -- moment rows
[1, (dx - c dt)/h, (dx^2/2 + alpha dt)/h^2] -- annihilates the u_x AND the
u_xx group, leaving only the mixed / temporal curvature and the third-order
remainder:

        |e| <= ||w||_1 * max_i [ |dx dt| C_xt + dt^2 C_tt / 2
                                 + ( |dx|^3 C_xxx + 3 dx^2 |dt| C_xxt
                                   + 3 |dx| dt^2 C_xtt + |dt|^3 C_ttt ) / 6 ] ,

which is O(||w||_1 h^3) under the parabolic scaling the solver actually uses,
dt ~ dx^2 / alpha.

NOTE on the test field.  The earlier version of this figure used
u = sin(pi x) cos(pi t), which does NOT satisfy the advection-diffusion
equation, so the PDE-substituted predictor is not defined for it. Both
predictors are therefore evaluated here on the exact solution above. The
generic bound does not use the PDE, so this is a fair common ground.

Panels:
    (a) one stencil: n = 5 neighbours against m = 3 moment conditions;
    (b) z* fixed, parabolic refinement dx ~ s, dt ~ 0.45 s^2 / alpha:
        error and bound for both predictors, log-log, showing rates 2 and 3;
    (c) fixed stencil scale, z* varying over 200 random targets: every point
        of both families falls below its own bound.

Outputs into FIG_DIR:
    fig_error_bound.pdf   <- vector, use this in LaTeX
    fig_error_bound.png   <- 600 dpi raster preview

Usage:
    python paper_fig_error_bound.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "Figures")

C_ADV   = 0.8      # advection speed
ALPHA   = 0.05     # diffusivity
DT_FAC  = 0.45     # dt = DT_FAC * dx^2 / alpha, the solver's CFL ratio


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Exact solution and its derivatives
# ═══════════════════════════════════════════════════════════════════════════
# u = sin(pi xi) E,  xi = x - c t,  E = exp(-alpha pi^2 t).
# u is an eigenfunction, so d/dt = -(c d/dx + alpha pi^2) acting on u and on
# every x-derivative of it. That gives every mixed partial in closed form
# without a CAS.
_SIN_CYCLE = (np.sin, np.cos, lambda z: -np.sin(z), lambda z: -np.cos(z))


def d_x(m, x, t):
    """m-th x-derivative of u."""
    xi = np.pi * (np.asarray(x) - C_ADV * np.asarray(t))
    return (np.pi**m) * _SIN_CYCLE[m % 4](xi) * np.exp(-ALPHA * np.pi**2 * np.asarray(t))


def deriv(m, n, x, t):
    """d^m/dx^m d^n/dt^n of u."""
    lam, out = ALPHA * np.pi**2, 0.0
    for k in range(n + 1):
        coef = ((-1) ** n) * _binom(n, k) * (C_ADV ** k) * (lam ** (n - k))
        out = out + coef * d_x(m + k, x, t)
    return out


def _binom(n, k):
    from math import comb
    return comb(n, k)


def u(x, t):
    return d_x(0, x, t)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  The two predictors and the two bounds
# ═══════════════════════════════════════════════════════════════════════════
_B = np.array([1.0, 0.0, 0.0])
KEYS = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


def w_generic(dx, dt):
    """Moment rows [1, dx, dt]: kills u, u_x, u_t."""
    A = np.array([np.ones(len(dx)), dx, dt])
    return np.linalg.lstsq(A, _B, rcond=None)[0], A


def w_pde(dx, dt):
    """PDE-substituted rows: kills u, the u_x group and the u_xx group."""
    h = max(np.max(np.abs(dx)), np.sqrt(ALPHA * np.max(np.abs(dt))), 1e-300)
    A = np.array([np.ones(len(dx)),
                  (dx - C_ADV * dt) / h,
                  (0.5 * dx**2 + ALPHA * dt) / h**2])
    return np.linalg.lstsq(A, _B, rcond=None)[0], A, h


def hull_constants(x0, t0, dx, dt):
    """Suprema of the needed derivatives over the box spanned by the stencil."""
    gx = np.linspace(min(0.0, dx.min()), max(0.0, dx.max()), 11)
    gt = np.linspace(min(0.0, dt.min()), max(0.0, dt.max()), 11)
    X, T = np.meshgrid(x0 + gx, t0 + gt)
    return {k: np.abs(deriv(k[0], k[1], X, T)).max() for k in KEYS}


def bound_generic(w, dx, dt, C):
    h = np.max(np.sqrt(dx**2 + dt**2))
    Cmax = max(C[(2, 0)], C[(1, 1)], C[(0, 2)])
    return np.sum(np.abs(w)) * Cmax * h**2


def bound_pde(w, dx, dt, C):
    second = np.abs(dx * dt) * C[(1, 1)] + 0.5 * dt**2 * C[(0, 2)]
    third  = (np.abs(dx)**3 * C[(3, 0)]
              + 3 * dx**2 * np.abs(dt) * C[(2, 1)]
              + 3 * np.abs(dx) * dt**2 * C[(1, 2)]
              + np.abs(dt)**3 * C[(0, 3)]) / 6.0
    return np.sum(np.abs(w)) * np.max(second + third)


def h_iso(dx, dt):
    return np.max(np.sqrt(dx**2 + dt**2))


def unit_disc(rng, n=5):
    """n directions drawn uniformly in the unit disc."""
    ang = rng.uniform(0.0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0.0, 1.0, n))
    return rad * np.cos(ang), rad * np.sin(ang)


def parabolic_stencil(s, p, q):
    """Scale a unit-disc pattern to the aspect ratio the solver marches on:
    dx ~ s, dt ~ 0.45 s^2 / alpha. The neighbourhood is the ellipse with
    semi-axes (s, 0.45 s^2 / alpha), not a disc -- that anisotropy is exactly
    what turns the O(h^2) bound into O(h^3)."""
    return s * p, (DT_FAC * s**2 / ALPHA) * q


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Experiment 1 -- z* fixed, parabolic refinement
# ═══════════════════════════════════════════════════════════════════════════
x_star, t_star = 0.5, 0.3

_rng_dirs = np.random.default_rng(3)
p_fixed, q_fixed = unit_disc(_rng_dirs)

s_values = np.logspace(np.log10(0.12), np.log10(0.0035), 24)
H, E_gen, B_gen, E_pde, B_pde = [], [], [], [], []

for s in s_values:
    dx, dt = parabolic_stencil(s, p_fixed, q_fixed)
    ui, u0 = u(x_star + dx, t_star + dt), u(x_star, t_star)
    C = hull_constants(x_star, t_star, dx, dt)

    wg, _    = w_generic(dx, dt)
    wp, _, _ = w_pde(dx, dt)

    H.append(h_iso(dx, dt))
    E_gen.append(abs(wg @ ui - u0)); B_gen.append(bound_generic(wg, dx, dt, C))
    E_pde.append(abs(wp @ ui - u0)); B_pde.append(bound_pde(wp, dx, dt, C))

H = np.array(H)
E_gen, B_gen = np.array(E_gen), np.array(B_gen)
E_pde, B_pde = np.array(E_pde), np.array(B_pde)


def _rate(y):
    return float(np.mean(np.diff(np.log(y[-5:])) / np.diff(np.log(H[-5:]))))


rate_gen, rate_pde = _rate(E_gen), _rate(E_pde)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Experiment 2 -- stencil scale fixed, z* varying
# ═══════════════════════════════════════════════════════════════════════════
s_fixed = 0.03
n_pts   = 200
_rng2   = np.random.default_rng(11)

eg, bg, ep, bp = [], [], [], []
while len(eg) < n_pts:
    xt, tt = _rng2.uniform(0.25, 0.75), _rng2.uniform(0.25, 0.75)
    pp, qq = unit_disc(_rng2)
    dx, dt = parabolic_stencil(s_fixed, pp, qq)

    wg, Ag    = w_generic(dx, dt)
    wp, Ap, _ = w_pde(dx, dt)
    if np.linalg.cond(Ag) > 1e6 or np.linalg.cond(Ap) > 1e4:
        continue                                   # inadmissible for either

    ui, u0 = u(xt + dx, tt + dt), u(xt, tt)
    C = hull_constants(xt, tt, dx, dt)
    eg.append(abs(wg @ ui - u0)); bg.append(bound_generic(wg, dx, dt, C))
    ep.append(abs(wp @ ui - u0)); bp.append(bound_pde(wp, dx, dt, C))

eg, bg, ep, bp = map(np.array, (eg, bg, ep, bp))
ratio_gen, ratio_pde = eg / bg, ep / bp


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Demo stencil for panel (a)
# ═══════════════════════════════════════════════════════════════════════════
s_demo = 0.10
_rng_d = np.random.default_rng(15)
p_demo, q_demo = unit_disc(_rng_d)
dx_demo, dt_demo = parabolic_stencil(s_demo, p_demo, q_demo)
Rx_demo = s_demo                          # ellipse semi-axes of the
Rt_demo = DT_FAC * s_demo**2 / ALPHA      # neighbourhood being sampled
x_demo, t_demo = 0.5, 0.30


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Plot style -- matches fig_placement_comparison
# ═══════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.labelsize":     9.5,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    6.6,
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "figure.dpi":         160,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

C_GEN  = "#2c6fac"   # generic predictor
C_PDE  = "#d1495b"   # PDE-substituted predictor
C_REF  = "#7a848e"   # guides
C_OK   = "#2e7d5b"   # admissible region


def style_ax(ax):
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#8d959d')
    ax.tick_params(colors='#3a3f45')
    ax.grid(True, alpha=0.30, linestyle=(0, (1.5, 2.5)), linewidth=0.6,
            color='#b8bfc7')
    ax.set_axisbelow(True)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.95),
                                        constrained_layout=True)

    # ── (a) one stencil ────────────────────────────────────────────────────
    style_ax(ax1)
    Rx, Rt = Rx_demo, Rt_demo
    for kw in (dict(fill=True, facecolor=C_GEN, alpha=0.055, lw=0),
               dict(fill=False, edgecolor=C_GEN, lw=0.9,
                    linestyle=(0, (4, 2.5)))):
        ax1.add_patch(patches.Ellipse((x_demo, t_demo), 2 * Rx, 2 * Rt,
                                      zorder=1, **kw))

    for xn, tn in zip(x_demo + dx_demo, t_demo + dt_demo):
        ax1.plot([x_demo, xn], [t_demo, tn], color=C_GEN, alpha=0.35,
                 linewidth=0.7, zorder=2)
    ax1.scatter(x_demo + dx_demo, t_demo + dt_demo, s=34, color=C_GEN, zorder=4,
                edgecolors='white', linewidths=0.6, label=r'$z_i,\; i=1,\dots,5$')
    ax1.scatter([x_demo], [t_demo], s=150, color=C_PDE, marker='*', zorder=5,
                edgecolors='white', linewidths=0.5,
                label=r'$z^\ast=(x^\ast,t^\ast)$')

    ax1.annotate('', xy=(x_demo + Rx, t_demo), xytext=(x_demo, t_demo),
                 arrowprops=dict(arrowstyle='->', color=C_REF, lw=0.9,
                                 shrinkA=0, shrinkB=0))
    ax1.text(x_demo + 0.52 * Rx, t_demo + 0.055 * Rt, r'$r$',
             ha='center', va='bottom', fontsize=7.4, color=C_REF)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t$')
    ax1.set_title(r'(a)  $n=5$,  $m=3$', pad=6)
    ax1.legend(fontsize=6.4, framealpha=0.95, edgecolor='#c8ced4',
               loc='upper left', handletextpad=0.35, borderpad=0.3,
               labelspacing=0.35, handlelength=1.4, borderaxespad=0.3)
    pad_x, pad_t = 1.45 * Rx, 1.45 * Rt
    ax1.set_xlim(x_demo - 1.55 * Rx, x_demo + 1.55 * Rx)
    ax1.set_ylim(t_demo - 1.55 * Rt, t_demo + 1.55 * Rt)
    ax1.set_box_aspect(1)

    # ── (b) z* fixed, parabolic refinement ─────────────────────────────────
    style_ax(ax2)
    ax2.set_xscale('log'); ax2.set_yscale('log')

    ax2.plot(H, B_gen, ls=(0, (4, 2.5)), color=C_GEN, lw=1.1, zorder=3)
    ax2.plot(H, E_gen, '-', color=C_GEN, lw=1.5, zorder=4, label='generic')
    ax2.plot(H, B_pde, ls=(0, (4, 2.5)), color=C_PDE, lw=1.1, zorder=3)
    ax2.plot(H, E_pde, '-', color=C_PDE, lw=1.5, zorder=4,
             label='PDE')

    ax2.set_xlabel(r'$h=\max_i\|\Delta z_i\|$')
    ax2.set_ylabel(r'error,  bound')
    ax2.set_title(r'(b)  $z^\ast$ fixed,  $\Delta t\sim\Delta x^{2}/\alpha$', pad=6)
    ax2.legend(fontsize=6.4, framealpha=0.95, edgecolor='#c8ced4',
               loc='lower right', handletextpad=0.5, borderpad=0.3,
               labelspacing=0.35, handlelength=1.5, borderaxespad=0.4,
               title=r'solid $|e|$,  dashed bound', title_fontsize=6.0)
    ax2.set_box_aspect(1)

    # ── (c) z* varying ─────────────────────────────────────────────────────
    style_ax(ax3)
    allv = np.concatenate([bg, eg, bp, ep])
    lims = [allv.min() * 0.35, allv.max() * 3.0]

    ax3.fill_between(lims, [lims[0]] * 2, lims, alpha=0.07, color=C_OK,
                     lw=0, zorder=1)
    ax3.plot(lims, lims, ls=(0, (4, 2.5)), color=C_REF, lw=1.0, zorder=2)
    ax3.scatter(bg, eg, color=C_GEN, alpha=0.7, s=13, edgecolors='white',
                linewidths=0.25, zorder=3, label='generic')
    ax3.scatter(bp, ep, color=C_PDE, alpha=0.7, s=13, edgecolors='white',
                linewidths=0.25, zorder=4, label='PDE')

    ax3.set_xlim(lims); ax3.set_ylim(lims)
    ax3.set_xscale('log'); ax3.set_yscale('log')

    _lo, _hi = np.log10(lims)
    _mid = 10 ** (_lo + 0.55 * (_hi - _lo))
    ax3.text(_mid, _mid * 1.3, r'$|e|=\mathrm{bound}$', fontsize=7.2,
             color=C_REF, rotation=45, rotation_mode='anchor',
             ha='center', va='bottom')

    ax3.set_xlabel(r'bound')
    ax3.set_ylabel(r'$|e|$')
    ax3.set_title(r'(c)  $h$ fixed,  $z^\ast$ varying', pad=6)
    ax3.legend(fontsize=6.4, framealpha=0.95, edgecolor='#c8ced4',
               loc='upper left', handletextpad=0.35, borderpad=0.3,
               labelspacing=0.35, handlelength=1.0, borderaxespad=0.3)
    ax3.set_aspect('equal')

    fig.suptitle(
        r'$u_t+c\,u_x=\alpha u_{xx}$,'
        r'$\;\;u=\sin(\pi(x-ct))\,e^{-\alpha\pi^{2}t}$,'
        rf'$\;\;c={C_ADV}$, $\alpha={ALPHA}$',
        fontsize=9)

    out_pdf = os.path.join(FIG_DIR, "fig_error_bound.pdf")
    out_png = os.path.join(FIG_DIR, "fig_error_bound.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)

    print(f"observed rate  generic {rate_gen:.2f}   PDE {rate_pde:.2f}")
    print(f"generic : all below bound {bool(np.all(ratio_gen <= 1))}   "
          f"max {ratio_gen.max():.3f}   median {np.median(ratio_gen):.3f}")
    print(f"PDE     : all below bound {bool(np.all(ratio_pde <= 1))}   "
          f"max {ratio_pde.max():.3f}   median {np.median(ratio_pde):.3f}")
    print(f"accuracy gain at smallest h : {E_gen[-1] / E_pde[-1]:.0f}x")
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
