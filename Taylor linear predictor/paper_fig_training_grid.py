"""
paper_fig_training_grid.py
==========================

Training-progress grid: one row per PDE, one column per stage of training.
Each panel is a single deterministic rollout -- the points the policy evaluated
in the x-t plane, the walker paths that placed them, and, when the episode
terminated on an admissible stencil, the five points that stencil used at the
query z*.  Read across a row, the three panels show the same thing happening
three times over: an early policy scatters evaluations across the whole domain
and spends its budget, a converged one commits a few walkers to the cone that
actually carries information to z*.

Rows
    diffusion   t* = 10 dt, n_x =  30, runs1/t10_lam0.001_beta100_K250_N25/seed2
    advection   t* =  5 dt, n_x =  20, runs_adv8/t5_lam0.001_beta1000_K400_N25/seed8
    Burgers     t* = 20 dt, n_x = 100, runs8-21/t20_errtol0.001_werr1_wtime0.1_N70_c0.3_R3/seed6

Panel data is stored as lattice indices: x = ix*dx with dx = 1/(n_x - 1) and
t = jt*dt, so the vertical axis is t/dt and a row needs only its own n_x and
(ix*, jt*).  A chain is stored as its first site plus one character per step,
'-' '0' '+' for the move in x, because the environment advances a walker by one
time level per step and by at most one cell in x.

PROVENANCE.  The other figures in this paper re-run the rollout from the
checkpoint.  That is not available for these nine panels: checkpoints survive
only every 1.5M steps (diffusion, advection) and every 1.2M (Burgers), and none
of the nine steps shown -- 0.1M to 3.1M, 13.9M, 17.6M -- lands on one.  The
archived figure is the only surviving record of those rollouts, so each panel is
read back out of its PNG by paper_fig_training_grid_digitise.py, which recovers
lattice indices rather than estimated coordinates and checks the result three
ways (the star lands on its known lattice site; no walker holds two places in
one time level; no chain makes an illegal move).  Run that script to regenerate
the literals below.  The one panel that does have a checkpoint, advection at
3.0M, was re-run and reproduces its recovered stencil exactly.

Outputs into FIG_DIR:
    fig_training_grid.pdf   <- vector, use this in LaTeX
    fig_training_grid.png   <- 600 dpi raster preview

Usage:
    python paper_fig_training_grid.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "Figures")


# ═══════════════════════════════════════════════════════════════════════════
# Panel data.  Regenerate with:  python paper_fig_training_grid_digitise.py
# chains: (ix0, jt0, moves) -- one character per time level after the first
# ═══════════════════════════════════════════════════════════════════════════
DIF_EARLY = dict(          # traj_t10dt_00200k_timeout.png
    stencil=[],
    chains=[
        (2,0,""), (3,0,""), (4,0,"+++++++++++"), (5,0,""),
        (6,0,"++++++++++"), (7,0,""), (8,0,""), (10,0,"+++++++"),
        (11,0,""), (12,0,""), (13,0,"++++"), (14,0,""), (15,0,""),
        (16,0,""), (17,0,""), (18,0,"---"), (19,0,""), (20,0,"----"),
        (22,0,""), (23,0,"----------"), (24,0,"+++"), (25,0,""),
        (26,0,""), (27,0,"")])

DIF_MID = dict(          # traj_t10dt_01000k_reached.png
    stencil=[(14,8), (14,9), (15,8), (15,9), (16,8)],
    chains=[
        (2,0,""), (3,0,""), (4,0,""), (5,0,""), (6,0,"++++++++0"),
        (7,0,""), (8,0,""), (10,0,"+++++++-"), (11,0,""), (12,0,""),
        (13,0,"+"), (14,0,""), (15,0,"0000"), (16,0,""), (17,0,""),
        (18,0,""), (19,0,"--"), (20,0,""), (22,0,""), (23,0,"--------0"),
        (24,0,""), (25,0,""), (26,0,""), (27,0,""), (27,0,"+")])

DIF_LATE = dict(          # traj_t10dt_03100k_reached.png
    stencil=[(12,6), (15,8), (16,7), (16,9), (17,6)],
    chains=[
        (2,0,""), (3,0,""), (4,0,""), (5,0,""), (6,0,"++++++"), (7,0,""),
        (8,0,""), (10,0,"++++"), (11,0,""), (12,0,""), (13,0,"+"),
        (14,0,""), (15,0,""), (16,0,""), (17,0,""), (18,0,""),
        (19,0,"--"), (20,0,""), (22,0,""), (23,0,"--------+"), (24,0,""),
        (25,0,""), (26,0,""), (27,0,"")])

ADV_EARLY = dict(          # traj_t5dt_00200k_timeout.png
    stencil=[],
    chains=[
        (2,0,"-000"), (3,0,"-000-00"), (3,0,"00"), (4,0,"+++++++"),
        (5,0,"-0-+000"), (6,0,"+0-0000"), (7,0,"-----+-"),
        (7,0,"++++0++"), (8,0,"+++++++++"), (9,0,"+++"), (10,0,""),
        (11,0,""), (11,0,"0+"), (12,0,""), (13,0,"-"), (13,0,"000+++0"),
        (14,0,"00++0--"), (15,0,"0+++-+-"), (16,0,""), (17,0,"-------"),
        (17,0,"00+-+-+"), (17,0,"+0")])

ADV_MID = dict(          # traj_t5dt_02100k_reached.png
    stencil=[(8,3), (9,4), (11,3), (12,3), (12,4)],
    chains=[
        (2,0,""), (3,0,""), (4,0,""), (5,0,"++++"), (6,0,""), (7,0,""),
        (8,0,"+++++++"), (9,0,"+++++"), (10,0,""), (11,0,"0"), (12,0,""),
        (13,0,""), (14,0,"--"), (15,0,""), (15,0,"0"), (16,0,""),
        (17,0,""), (17,0,"---0")])

ADV_LATE = dict(          # traj_t5dt_03000k_reached.png
    stencil=[(8,3), (9,4), (11,3), (12,3), (12,4)],
    chains=[
        (2,0,""), (3,0,""), (4,0,""), (5,0,"++++"), (6,0,""), (7,0,""),
        (8,0,"+++++"), (9,0,"++++++"), (10,0,""), (11,0,""), (12,0,""),
        (13,0,""), (14,0,""), (15,0,""), (16,0,""), (17,0,"")])

BUR_EARLY = dict(          # traj_t20dt_x50_00100k_reached.png
    stencil=[(48,18), (50,18), (50,19), (50,21), (50,22)],
    chains=[
        (3,0,"+++++"), (5,0,""), (6,0,""), (7,0,"-"), (9,0,""), (10,0,""),
        (12,0,"-----------"), (13,0,"00000000000"), (14,0,""), (16,0,""),
        (17,0,""), (18,0,""), (20,0,"------"), (21,0,"-----"),
        (23,0,"-------"), (24,0,""), (25,0,""), (27,0,""), (28,0,"+++++"),
        (29,0,"++++++00000"), (31,0,"0"), (32,0,"+++0"), (34,0,""),
        (35,0,""), (36,0,"-"), (38,0,""), (39,0,""), (40,0,"-------"),
        (42,0,""), (43,0,""), (45,0,""), (46,0,""),
        (47,0,"+++++++0000000000000000"), (48,18,""),
        (49,0,"0-00000000000000"), (50,0,""), (50,21,"0"), (51,0,"++"),
        (53,0,""), (54,0,"----000000000000000"), (55,0,""),
        (57,0,"----+"), (58,0,""), (60,0,"+++++++-+-+-+-+-+-+-+-+"),
        (61,0,""), (62,0,""), (64,0,""), (65,0,""),
        (66,0,"00000-000000000"), (68,0,"-"), (69,0,"+---"), (71,0,""),
        (72,0,""), (73,0,"-------"), (75,0,"---------"),
        (76,0,"+0++++++++++++++++++++0"), (77,0,""), (79,0,"++++++++++"),
        (80,0,"------------0---0000000"), (82,0,"00"),
        (83,0,"+++++++0000++++++++0000"), (84,0,""), (86,0,""),
        (87,0,"000++++0000++"), (88,0,"+++++"), (90,0,""), (91,0,""),
        (93,0,""), (94,0,"00000000000000"), (95,0,""), (97,0,""),
        (97,0,"--")])

BUR_MID = dict(          # traj_t20dt_x50_13901k_reached.png
    stencil=[(48,17), (49,18), (49,19), (53,17), (53,18)],
    chains=[
        (3,0,""), (5,0,"+++++++"), (6,0,""), (7,0,""), (9,0,""),
        (10,0,""), (12,0,""), (13,0,""), (14,0,""), (16,0,""), (17,0,""),
        (18,0,""), (20,0,""), (21,0,""), (23,0,""), (24,0,""), (25,0,""),
        (27,0,"++++++++++++"), (28,0,""), (29,0,""), (31,0,""), (32,0,""),
        (34,0,""), (35,0,"+++++++"), (36,0,""), (38,0,""), (39,0,"++++"),
        (40,0,""), (42,0,""), (43,0,""), (45,0,""), (46,0,""),
        (47,0,"++++-0++-0++++--"), (49,0,"0-000000000000000+0"),
        (50,0,""), (51,0,"++"), (53,0,""), (53,17,"0"), (54,0,"----"),
        (54,18,"+-+-+"), (55,0,""), (57,0,"--------"), (58,0,""),
        (60,0,""), (61,0,""), (62,0,"-----------"), (64,0,""), (65,0,""),
        (66,0,"--------------"), (68,0,""), (69,0,""),
        (71,0,"-----------------"), (72,0,""), (73,0,""), (75,0,""),
        (76,0,""), (77,0,""), (79,0,""), (80,0,""), (82,0,""), (83,0,""),
        (84,0,""), (86,0,""), (87,0,""), (88,0,""), (90,0,""),
        (91,0,"-------"), (93,0,""), (94,0,""), (95,0,""), (97,0,"")])

BUR_LATE = dict(          # traj_t20dt_x50_17601k_reached.png
    stencil=[(47,19), (48,18), (49,17), (52,19), (53,18)],
    chains=[
        (3,0,""), (5,0,""), (6,0,""), (7,0,""), (9,0,""), (10,0,""),
        (12,0,""), (13,0,""), (14,0,""), (16,0,""), (17,0,""), (18,0,""),
        (20,0,""), (21,0,""), (23,0,""), (24,0,""), (25,0,""), (27,0,""),
        (28,0,""), (29,0,""), (31,0,""), (32,0,""), (34,0,""), (35,0,""),
        (36,0,""), (38,0,""), (39,0,"+++++"), (40,0,""), (42,0,""),
        (43,0,""), (45,0,""), (46,0,""), (47,0,""), (49,0,""), (50,0,""),
        (51,0,""), (53,0,""), (54,0,""), (55,0,""), (57,0,"-----------"),
        (58,0,""), (60,0,""), (61,0,""), (62,0,"----------------"),
        (64,0,""), (65,0,""), (66,0,"--------------------"), (68,0,""),
        (69,0,""), (71,0,"-------------------"), (72,0,""), (73,0,""),
        (75,0,""), (76,0,""), (77,0,""), (79,0,""), (80,0,""), (82,0,""),
        (83,0,""), (84,0,""), (86,0,""), (87,0,""), (88,0,""), (90,0,""),
        (91,0,""), (93,0,""), (94,0,""), (95,0,""), (97,0,"")])

# ═══════════════════════════════════════════════════════════════════════════
# Rows.  (n_x, ix*, jt*) and the three stages, left to right.
# ═══════════════════════════════════════════════════════════════════════════
ROWS = [
    ("diffusion", 30, 15, 10,
     [(r"$0.2$M",  False, DIF_EARLY),
      (r"$1.0$M",  True,  DIF_MID),
      (r"$3.1$M",  True,  DIF_LATE)]),
    ("advection", 20, 10, 5,
     [(r"$0.2$M",  False, ADV_EARLY),
      (r"$2.1$M",  True,  ADV_MID),
      (r"$3.0$M",  True,  ADV_LATE)]),
    ("Burgers", 100, 50, 20,
     [(r"$0.1$M",  True,  BUR_EARLY),
      (r"$13.9$M", True,  BUR_MID),
      (r"$17.6$M", True,  BUR_LATE)]),
]

COLUMNS = ["early", "mid-training", "converged"]

MOVE = {'-': -1, '0': 0, '+': 1}


def decode(chain):
    """(ix0, jt0, moves) -> [(ix, jt), ...]"""
    ix, jt, moves = chain
    out = [(ix, jt)]
    for m in moves:
        ix += MOVE[m]; jt += 1
        out.append((ix, jt))
    return out


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
C_GRID    = "#d6dade"


def draw(ax, data, nx, ix_star, jt_star, reached, jmax):
    dx   = 1.0 / (nx - 1)
    fine = nx > 50                       # the Burgers row carries ~70 walkers
    lw   = 0.45 if fine else 0.8
    ms   = 1.5 if fine else 2.6

    ax.set_axisbelow(True)
    ax.grid(True, ls=(0, (1, 2.6)), lw=0.5, color=C_GRID)

    for chain in data['chains']:
        pts = decode(chain)
        col = plt.cm.viridis(0.04 + 0.92 * pts[0][0] / (nx - 1))
        xs  = [ix * dx for ix, _ in pts]
        ts  = [jt for _, jt in pts]
        if len(pts) > 1:
            ax.plot(xs, ts, '-', color=col, lw=lw, alpha=0.85, zorder=3,
                    solid_capstyle='round')
        head = [(x, t) for x, t in zip(xs, ts) if t == 0]
        tail = [(x, t) for x, t in zip(xs, ts) if t > 0]
        if tail:
            ax.plot(*zip(*tail), ls='none', marker='o', ms=ms, mfc=col,
                    mec='none', zorder=4)
        if head:
            ax.plot(*zip(*head), ls='none', marker='^', ms=ms + 0.7, mfc=col,
                    mec='none', zorder=5)

    if reached:
        for ix, jt in data['stencil']:
            ax.plot([ix_star * dx, ix * dx], [jt_star, jt],
                    ls=(0, (2.5, 1.8)), color=C_STENCIL, lw=0.8, alpha=0.9,
                    zorder=6)
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


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.55), squeeze=False,
                             sharex=True, constrained_layout=True)

    for ri, (name, nx, ix_star, jt_star, stages) in enumerate(ROWS):
        jmax = max(jt for _, _, d in stages for c in d['chains']
                   for _, jt in decode(c))
        for ci, (steplab, reached, data) in enumerate(stages):
            ax = axes[ri][ci]
            draw(ax, data, nx, ix_star, jt_star, reached, jmax)
            if ci:
                ax.tick_params(labelleft=False)
            if ri == 0:
                ax.set_title(COLUMNS[ci], fontsize=9.5, pad=14)
            # points at t = 0 are the given initial condition; everything
            # above it is an evaluation the rollout paid for
            nev = sum(1 for c in data['chains'] for _, jt in decode(c) if jt)
            ax.text(0.5, 1.015,
                    steplab + r" steps  $\cdot$  "
                    + ("reached" if reached else "timeout")
                    + rf"  $\cdot$  {nev} evals",
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=7.2, color="#3a3f45")
            if ri == 2:
                ax.set_xlabel(r"$x$")
        axes[ri][0].set_ylabel(r"$t/\Delta t$")
        axes[ri][0].annotate(name, xy=(-0.30, 0.5), xycoords='axes fraction',
                             rotation=90, ha='center', va='center',
                             fontsize=9.5)

    handles = [
        plt.Line2D([], [], ls='-', lw=1.0, marker='o', ms=3.0, mec='none',
                   color=plt.cm.viridis(0.45),
                   label='walker path, evaluated points'),
        plt.Line2D([], [], ls='none', marker='^', ms=4.0, mec='none',
                   color=plt.cm.viridis(0.45), label='walker start'),
        plt.Line2D([], [], ls=(0, (2.5, 1.8)), lw=0.8, color=C_STENCIL,
                   marker='o', ms=3.4, mec='none',
                   label=r'stencil at $z^\ast$'),
        plt.Line2D([], [], ls='none', marker='*', ms=8, color=C_STENCIL,
                   mec='none', label=r'$z^\ast=(x^\ast,t^\ast)$'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=4,
               frameon=False, handletextpad=0.5, columnspacing=1.9,
               borderaxespad=0.0)

    out_pdf = os.path.join(FIG_DIR, "fig_training_grid.pdf")
    out_png = os.path.join(FIG_DIR, "fig_training_grid.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)

    for name, _, _, _, stages in ROWS:
        for steplab, reached, d in stages:
            pts = [p for c in d['chains'] for p in decode(c)]
            print(f"{name:10s} {steplab:8s} chains={len(d['chains']):>3} "
                  f"points={len(pts):>4} stencil={len(d['stencil'])} "
                  f"{'reached' if reached else 'timeout'}")
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
