"""
paper_fig_cones.py
==================

Do the finite-difference domain of dependence and the learned policy's
dependency cone have the same shape?

For FTCS on a 3-point stencil, the set of grid nodes that influence
u(x*, t*) is the backward cone |i - i*| <= j* - j: half-width falls by exactly
one cell per time level.

The analogue for the learned policy is NOT simply where the walkers went; it is
the transitive closure of the GFDM stencils, starting from z*'s own stencil and
following each evaluated point back through the neighbours it was built from,
down to t = 0. `env.stencil_log` records exactly that graph.

This script overlays the two cones and compares their half-width profiles.

Run paper_fig_comparison.py first if you want its figure too; this script
imports the environment and the rollout helper from it.

Outputs into FIG_DIR:
    fig_cones.pdf   <- vector, use this in LaTeX
    fig_cones.png   <- 600 dpi raster preview

Usage:
    python paper_fig_cones.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

import paper_fig_comparison as P

FIG_DIR = P.FIG_DIR
dx, dt  = P.dx, P.dt


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Rollout + the two cones
# ═══════════════════════════════════════════════════════════════════════════
def cell(x, t):
    return (round(x / dx), round(t / dt))


def build():
    model = MaskablePPO.load(P.MODEL, device="cpu")
    res = P.rollout(lambda o, m: int(
        model.predict(o, deterministic=True, action_masks=m)[0]))
    env = res['env']

    i_star, j_star = round(env.x_star / dx), round(env.t_star / dt)

    placed = {cell(x, t) for (x, t, _) in env.visited._pts
              if t > 1e-12 and 0.0 < x < 1.0}

    # dependency closure of z*
    stencil = {cell(*k): [cell(a, b) for a, b in v]
               for k, v in env.stencil_log.items()}
    closure = {cell(a, b) for a, b in res['stencil']}
    frontier = list(closure)
    while frontier:
        c = frontier.pop()
        for nb in stencil.get(c, []):
            if nb not in closure:
                closure.add(nb)
                frontier.append(nb)

    return res, env, i_star, j_star, placed, closure


def envelope(cells, i_star):
    """level -> max |i - i*| present at that level."""
    lv = {}
    for (i, j) in cells:
        lv[j] = max(lv.get(j, -1), abs(i - i_star))
    return lv


def fit_slope(lv, j_lo, j_hi):
    js = np.array([j for j in range(j_lo, j_hi + 1) if j in lv])
    ws = np.array([lv[j] for j in js], dtype=float)
    return np.polyfit(js, ws, 1)[0]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Style
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
    "legend.fontsize":    6.8,
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

C_FD   = "#5c6b7a"   # finite-difference cone
C_RL   = "#2c6fac"   # learned cone
C_STAR = "#d1495b"
C_GRID = "#b8bfc7"


def style_ax(ax):
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#8d959d')
    ax.tick_params(colors='#3a3f45')
    ax.grid(True, alpha=0.30, linestyle=(0, (1.5, 2.5)), linewidth=0.6,
            color='#b8bfc7')
    ax.set_axisbelow(True)


def cone_edges(ax, lv, i_star, color, lw=1.3, ls='-', label=None,
               fill_alpha=0.0):
    """Draw the MEASURED envelope {level: half-width}, so the boundary is
    guaranteed to contain every point of that cone rather than an idealised
    straight line that might cut one off."""
    js = np.array(sorted(lv))
    ws = np.array([lv[j] for j in js], dtype=float)
    xl, xr, ts = (i_star - ws) * dx, (i_star + ws) * dx, js * dt
    if fill_alpha:
        ax.fill_betweenx(ts, xl, xr, color=color, alpha=fill_alpha,
                         lw=0, zorder=1)
    ax.plot(xl, ts, ls=ls, color=color, lw=lw, zorder=5, label=label)
    ax.plot(xr, ts, ls=ls, color=color, lw=lw, zorder=5)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    res, env, i_star, j_star, placed, closure = build()

    env_fd = {j: j_star - j for j in range(0, j_star + 1)}
    env_cl = envelope(closure, i_star)
    j_top  = max(env_cl)
    apex_rl = j_top + int(round(env_cl[j_top]))       # where the RL cone closes

    slope_fd = fit_slope(env_fd, 0, j_star)
    slope_cl = fit_slope(env_cl, 1, j_top)

    print(f"reached={res['reached']}  k={res['k_term']}  evals={res['n_eval']}  "
          f"err={res['error']:.3e}")
    print(f"FD      cone: apex level {j_star:2d}, half-width at t=0 "
          f"{env_fd[0]:2d} cells, slope {slope_fd:+.3f} cells/step")
    print(f"learned cone: apex level {apex_rl:2d}, half-width at t=0 "
          f"{env_cl.get(0, 0):2d} cells, slope {slope_cl:+.3f} cells/step")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9),
                                   constrained_layout=True)

    # ── (a) overlay in the x-t plane ───────────────────────────────────────
    style_ax(ax1)
    t_hi = (apex_rl + 1.2) * dt

    cone_edges(ax1, env_fd, i_star, C_FD, lw=1.2, ls=(0, (4, 2.5)),
               label='FD domain of dependence', fill_alpha=0.10)
    cone_edges(ax1, env_cl, i_star, C_RL, lw=1.2, ls='-',
               label='learned dependency cone', fill_alpha=0.07)

    # points
    cl_only = np.array([[i * dx, j * dt] for (i, j) in closure
                        if (i, j) not in placed and j * dt <= t_hi])
    pl = np.array([[i * dx, j * dt] for (i, j) in placed])
    if len(cl_only):
        ax1.scatter(cl_only[:, 0], cl_only[:, 1], s=11, facecolors='none',
                    edgecolors=C_RL, linewidths=0.7, zorder=6)
    ax1.scatter(pl[:, 0], pl[:, 1], s=12, color=C_RL, linewidths=0, zorder=7)
    ax1.scatter([env.x_star], [env.t_star], s=110, marker='*', color=C_STAR,
                edgecolors='white', linewidths=0.5, zorder=8)

    ax1.set_xlim(env.x_star - 16 * dx, env.x_star + 16 * dx)
    ax1.set_ylim(-0.6 * dt, t_hi)
    ax1.set_yticks(np.arange(0, apex_rl + 1, 2) * dt)
    ax1.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _p: f"{v / dt:.0f}"))
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t/\Delta t$')
    ax1.set_title(r'(a)  cones about $z^\ast$', pad=6)
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='#c8ced4',
               handletextpad=0.5, borderpad=0.3, labelspacing=0.35,
               handlelength=1.6, borderaxespad=0.3)

    # ── (b) half-width profiles ────────────────────────────────────────────
    style_ax(ax2)
    jf = np.array(sorted(env_fd))
    jc = np.array(sorted(k for k in env_cl if k >= 1))
    ax2.plot(jf, [env_fd[j] for j in jf], 'o-', color=C_FD, lw=1.3, ms=3.2,
             markeredgewidth=0, label=rf'FD,  slope ${slope_fd:+.2f}$')
    ax2.plot(jc, [env_cl[j] for j in jc], 's-', color=C_RL, lw=1.3, ms=3.0,
             markeredgewidth=0, label=rf'learned,  slope ${slope_cl:+.2f}$')

    ax2.axvline(j_star, color=C_STAR, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax2.text(j_star - 0.35, ax2.get_ylim()[1] * 0.97, r'$t^\ast$',
             color=C_STAR, fontsize=7.6, ha='right', va='top')

    ax2.set_xlabel(r'$t/\Delta t$')
    ax2.set_ylabel(r'half-width  [cells]')
    ax2.set_title(r'(b)  half-width per level', pad=6)
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor='#c8ced4',
               handletextpad=0.5, borderpad=0.3, labelspacing=0.35,
               handlelength=1.6, borderaxespad=0.3)
    ax2.set_xlim(-0.4, apex_rl + 0.6)
    ax2.set_ylim(0, max(env_cl.values()) + 1.6)
    ax2.set_box_aspect(0.86)

    out_pdf = os.path.join(FIG_DIR, "fig_cones.pdf")
    out_png = os.path.join(FIG_DIR, "fig_cones.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
