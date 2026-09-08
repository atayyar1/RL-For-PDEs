"""
paper_fig_characteristics.py
============================

Does the placement rule follow the characteristic?

The question is worth asking because the two rows have a genuine transport
direction: the advection run solves u_t + c u_x = alpha u_xx with c = 0.5, and
the Burgers run solves u_t + u u_x = alpha u_xx, whose transport speed is the
solution itself.  If the policy had learned the equation rather than the
shortest route to the query, the cloud it builds should lean along that
direction.

This figure puts the two side by side.  Left column: the converged rollout in
the x-t plane with the characteristic field drawn through it -- straight lines
of slope c for advection, and for Burgers the curves obtained by integrating
dx/dt = u(x,t) on the Hopf-Cole reference solution.  Right column: the half
width of the visited cloud against lookback tau = (t* - t)/dt, on log axes,
against the three scales that could set it,

    characteristic   c tau dt / dx           (Burgers: the integrated curve)
    diffusive        sqrt(alpha tau dt) / dx
    action cone      tau                     (one cell per time level)

The action cone is the set of points a walker could still reach z* from, not a
bound on where a walker may be: a walker not heading for z* can and does sit
outside it, which is why the tau = 1 marker is above the dotted line.

WHAT IT SHOWS.  The characteristic is not a scale the lattice can resolve.
Both runs take dt from the diffusive CFL limit, so the advective Courant number
is 0.23 (advection) and 0.046 (Burgers): over the whole lookback to t* the
characteristic through z* drifts 1.14 and 0.92 cells respectively.  A walker
moves in whole cells, so following it is not an available behaviour, and the
figure should be read as ruling the explanation out rather than testing it.
What the cloud does sit on is the action cone: the converged Burgers envelope
is w = tau + 1 cells for every 2 <= tau <= 19, and the advection envelope is
w = tau exactly for 2 <= tau <= 4 -- one cell per level in both, which is the
negative result about geometry that Section 5 asks for.
The diffusive scale, sqrt(alpha tau dt), is the only physical scale in range,
and it is about 3 cells at t* for Burgers and 1.5 for advection: the stencil at
z* lives on that scale even though the cloud that feeds it does not.

Data: the converged panels of fig_training_grid (advection 3.0M, Burgers
17.6M), imported from paper_fig_training_grid so the two figures cannot drift
apart.  PDE constants come from the first code cell of the notebook that
produced each run, executed rather than copied for the same reason.

Outputs into FIG_DIR:
    fig_characteristics.pdf   <- vector, use this in LaTeX
    fig_characteristics.png   <- 600 dpi raster preview

Usage:
    python paper_fig_characteristics.py
"""

import io
import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import paper_fig_training_grid as G

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "Figures")

NB_ADV = "5.2 Toy RL joint advection.ipynb"
NB_BUR = "6.2 RL Burger.ipynb"


def pde_setup(nb, cell=1):
    """Execute the notebook's PDE cell: grid, constants and u_true."""
    doc = json.load(io.open(os.path.join(ROOT, nb), encoding="utf-8"))
    src = [''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code']
    ns = {'np': np}
    exec(compile(src[cell], nb, 'exec'), ns)
    return ns


# ═══════════════════════════════════════════════════════════════════════════
# Characteristics
# ═══════════════════════════════════════════════════════════════════════════
def characteristic(x0, t0, t_end, speed, n=400):
    """Integrate dx/dt = speed(x, t) from (x0, t0) to t_end (either direction),
    midpoint rule.  For advection speed is a constant function."""
    h = (t_end - t0) / n
    x, t = float(x0), float(t0)
    xs, ts = [x], [t]
    for _ in range(n):
        k1 = speed(x, t)
        k2 = speed(x + 0.5 * h * k1, t + 0.5 * h)
        x += h * k2; t += h
        xs.append(x); ts.append(t)
    return np.array(xs), np.array(ts)


def half_width(data, ix_star, jt_star):
    """Widest evaluated point at each lookback, in cells.  tau = jt* excluded:
    that level is the initial condition the walkers are seeded on, not a
    placement the policy made."""
    pts = [p for c in data['chains'] for p in G.decode(c)]
    w = {}
    for ix, jt in pts:
        if 0 < jt <= jt_star:
            w.setdefault(jt_star - jt, []).append(abs(ix - ix_star))
    return {tau: max(v) for tau, v in sorted(w.items()) if 1 <= tau < jt_star}


# ═══════════════════════════════════════════════════════════════════════════
# Style -- matches the other paper figures
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

C_STENCIL = "#d1495b"
C_CHAR    = "#2a3d45"
C_DIFF    = "#3d8bcd"
C_ACTION  = "#8d99ae"
C_GRID    = "#d6dade"


def draw_xt(ax, data, nx, ix_star, jt_star, dx, dt, alpha, speed, jmax):
    """Rollout in the x-t plane with the characteristic field through it."""
    ax.set_axisbelow(True)
    ax.grid(True, ls=(0, (1, 2.6)), lw=0.5, color=C_GRID)

    # characteristic field, seeded across the domain at t = 0
    for x0 in np.linspace(0.04, 0.96, 13):
        xs, ts = characteristic(x0, 0.0, jmax * dt, speed)
        ax.plot(xs, ts / dt, '-', color=C_CHAR, lw=0.5, alpha=0.28, zorder=2)
    # the one through z*, traced back to t = 0 and on to the top of the panel
    xb, tb = characteristic(ix_star * dx, jt_star * dt, 0.0, speed)
    xf, tf = characteristic(ix_star * dx, jt_star * dt, jmax * dt, speed)
    ax.plot(np.r_[xb[::-1], xf], np.r_[tb[::-1], tf] / dt, '-',
            color=C_CHAR, lw=1.3, zorder=6)

    # the two other scales that could set the width, drawn about z*
    tau = np.linspace(0, jt_star, 200)
    for s in (+1, -1):
        ax.plot(ix_star * dx + s * np.sqrt(alpha * tau * dt), jt_star - tau,
                '--', color=C_DIFF, lw=0.9, zorder=6)
        ax.plot(ix_star * dx + s * tau * dx, jt_star - tau, ':',
                color=C_ACTION, lw=1.0, zorder=2)

    fine = nx > 50
    lw, ms = (0.45, 1.5) if fine else (0.8, 2.6)
    for chain in data['chains']:
        pts = G.decode(chain)
        col = plt.cm.viridis(0.04 + 0.92 * pts[0][0] / (nx - 1))
        xs = [ix * dx for ix, _ in pts]
        ts = [jt for _, jt in pts]
        if len(pts) > 1:
            ax.plot(xs, ts, '-', color=col, lw=lw, alpha=0.6, zorder=3)
        tail = [(x, t) for x, t in zip(xs, ts) if t > 0]
        head = [(x, t) for x, t in zip(xs, ts) if t == 0]
        if tail:
            ax.plot(*zip(*tail), ls='none', marker='o', ms=ms, mfc=col,
                    mec='none', alpha=0.85, zorder=4)
        if head:
            ax.plot(*zip(*head), ls='none', marker='^', ms=ms + 0.7, mfc=col,
                    mec='none', alpha=0.85, zorder=5)

    ax.plot([ix * dx for ix, _ in data['stencil']],
            [jt for _, jt in data['stencil']], ls='none', marker='o',
            ms=ms + 1.4, mfc=C_STENCIL, mec='none', zorder=7)
    ax.plot([ix_star * dx], [jt_star], ls='none', marker='*', ms=8.5,
            mfc=C_STENCIL, mec='white', mew=0.5, zorder=8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.9, jmax + 1.0)
    ax.set_yticks(np.arange(0, jmax + 1, max(1, int(round(jmax / 4)))))
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(length=3, width=0.8)
    for s in ax.spines.values():
        s.set_color("#3a3f45")


def draw_scales(ax, data, ix_star, jt_star, dx, dt, alpha, char_offset):
    """Cloud half-width against lookback, with the three candidate scales."""
    w = half_width(data, ix_star, jt_star)
    tau = np.array(sorted(w))
    y = np.array([w[t] for t in tau], float)
    g = np.linspace(1, jt_star, 200)

    ax.set_axisbelow(True)
    ax.grid(True, which='both', ls=(0, (1, 2.6)), lw=0.5, color=C_GRID)
    ax.plot(g, g, ':', color=C_ACTION, lw=1.4, zorder=2,
            label=r'action bound  $\tau$')
    ax.plot(g, np.sqrt(alpha * g * dt) / dx, '--', color=C_DIFF, lw=1.2,
            zorder=3, label=r'diffusive  $\sqrt{\alpha\tau\Delta t}/\Delta x$')
    ax.plot(g, char_offset(g), '-', color=C_CHAR, lw=1.3, zorder=4,
            label='characteristic')
    ax.plot(tau, y, ls='none', marker='o', ms=3.4, mfc=C_STENCIL, mec='none',
            zorder=5, label='visited cloud')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.88, jt_star * 1.15)
    ax.set_ylim(0.02, jt_star * 2.2)
    xt = [1, 2, 3, 4, 5] if jt_star <= 6 else [1, 2, 5, 10, 20]
    ax.set_xticks(xt); ax.set_xticklabels([str(v) for v in xt])
    ax.set_xticks([], minor=True)
    ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    ax.tick_params(length=3, width=0.8, which='both')
    for s in ax.spines.values():
        s.set_color("#3a3f45")
    return tau, y


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    A = pde_setup(NB_ADV)
    B = pde_setup(NB_BUR)

    rows = [
        ("advection", r"$u_t + c\,u_x = \alpha u_{xx}$,  $c=0.5$",
         G.ADV_LATE, 20, 10, 5, A['dx'], A['dt'], A['alpha'],
         (lambda x, t, c=A['c']: c),
         (lambda g, c=A['c'], dt=A['dt'], dx=A['dx']: c * g * dt / dx)),
        ("Burgers", r"$u_t + u\,u_x = \alpha u_{xx}$",
         G.BUR_LATE, 100, 50, 20, B['dx'], B['dt'], B['alpha'],
         (lambda x, t, u=B['u_true']: u(x, t)), None),
    ]

    # the Burgers characteristic offset is the integrated curve, not a line
    xb, tb = characteristic(50 * B['dx'], 20 * B['dt'], 0.0,
                            lambda x, t: B['u_true'](x, t), n=2000)
    off = np.abs(xb - 50 * B['dx']) / B['dx']
    lev = (20 * B['dt'] - tb) / B['dt']
    rows[1] = rows[1][:-1] + (lambda g: np.interp(g, lev, off),)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.7), squeeze=False,
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    report = {}
    for ri, (name, eq, data, nx, ixs, jts, dx, dt, alpha, speed, coff) in enumerate(rows):
        jmax = max(jt for c in data['chains'] for _, jt in G.decode(c))
        draw_xt(axes[ri][0], data, nx, ixs, jts, dx, dt, alpha, speed, jmax)
        tau, y = draw_scales(axes[ri][1], data, ixs, jts, dx, dt, alpha, coff)
        report[name] = (tau, y, coff(np.array([float(jts)]))[0],
                        np.sqrt(alpha * jts * dt) / dx)

        axes[ri][0].set_ylabel(r"$t/\Delta t$")
        axes[ri][1].set_ylabel("half-width  [cells]")
        axes[ri][0].annotate(name, xy=(-0.26, 0.5), xycoords='axes fraction',
                             rotation=90, ha='center', va='center', fontsize=9.5)
        if ri == 0:
            axes[ri][0].set_title("converged rollout, characteristics through it",
                                  fontsize=9, pad=7)
            axes[ri][1].set_title("what sets the width of the cloud",
                                  fontsize=9, pad=7)
        axes[ri][0].text(0.03, 0.955, eq, transform=axes[ri][0].transAxes,
                         ha='left', va='top', fontsize=7.6, color="#3a3f45")
        if ri == 1:
            axes[ri][0].set_xlabel(r"$x$")
            axes[ri][1].set_xlabel(r"lookback  $\tau=(t^\ast-t)/\Delta t$")

    handles = [
        plt.Line2D([], [], ls='-', lw=1.3, color=C_CHAR, label='characteristic'),
        plt.Line2D([], [], ls='--', lw=1.2, color=C_DIFF,
                   label=r'diffusive $\sqrt{\alpha\tau\Delta t}/\Delta x$'),
        plt.Line2D([], [], ls=':', lw=1.4, color=C_ACTION,
                   label=r'action cone about $z^\ast$, one cell per level'),
        plt.Line2D([], [], ls='-', lw=1.0, marker='o', ms=3.0, mec='none',
                   color=plt.cm.viridis(0.45), label='walker path'),
        plt.Line2D([], [], ls='none', marker='o', ms=3.4, color=C_STENCIL,
                   mec='none', label='visited cloud half-width'),
        plt.Line2D([], [], ls='none', marker='*', ms=8, color=C_STENCIL,
                   mec='none', label=r'$z^\ast$'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=3,
               frameon=False, handletextpad=0.5, columnspacing=1.9,
               borderaxespad=0.0)

    out_pdf = os.path.join(FIG_DIR, "fig_characteristics.pdf")
    out_png = os.path.join(FIG_DIR, "fig_characteristics.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)

    for name, (tau, y, cstar, dstar) in report.items():
        print(f"{name:10s} envelope w(tau) - tau = "
              f"{(y - tau).astype(int).tolist()}"
              f"   (0 or 1 means the cloud rides the action cone)")
        print(f"{'':10s} at t*:  characteristic {cstar:.2f} cells | "
              f"diffusive {dstar:.2f} cells | action cone {int(tau.max())+1} cells")
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
