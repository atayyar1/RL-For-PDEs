"""
paper_fig_when.py
=================

Paper figure: "where, whom, and WHEN".

A placement policy decides which walker to move (whom) and in which direction
(where).  This figure isolates a third decision that is easy to miss: the ORDER
in which the points are computed (when).  Every evaluated value is itself a
prediction built from the neighbours that already exist -- and in this
environment moves are solved non-causally (`causal_moves = False`), so a point
may be predicted from neighbours both below and above it in time.  Two rollouts
that end with the same point cloud and the same stencil at z* can therefore
carry very different error.

EVIDENCE.  In the archived advection run

    t10_errtol0.0003_werr1_wtime0.1_N30_c0.3_R4/seed3

the checkpoints at 6.0M and 7.2M training steps produce the IDENTICAL set of 11
points, the identical three walker chains, and the identical five-point stencil
at z*.  They differ only by two adjacent transpositions in the construction
order, and their errors are 8.89e-5 and 7.69e-5.  The archived figure at 7.0M
steps shows a third value, 3.50e-5, on the same cloud.

Because the point set fixes the stencil, it also fixes the weights w: the error
at z* is |sum_i w_i u_i - u_exact| with w FIXED.  The entire spread comes from
the computed values u_i -- that is, from the order alone.

Enumerating every interleaving of the three walker chains that preserves each
chain's internal order gives 11550 admissible construction orders, all of them
feasible.  Over that set the error at z* spans

        min 9.52e-08     median 1.17e-04     max 1.20e-03

a factor of ~1.3e4, on a fixed point cloud.

PANELS.  Three orders of that cloud, matching the three archived figures:
    (a) 8.89e-5  -- the order the policy takes at the 6.0M checkpoint
    (b) 3.50e-5  -- an order reproducing the value in the 7.0M archived figure.
                    Checkpoints are kept every 1.2M steps so that one was not
                    retained, and 17 of the 11550 orders reproduce this error to
                    within 0.2%; this order is therefore representative, not
                    attributed.
    (c) 7.69e-5  -- the order the policy takes at the 7.2M checkpoint
Each point carries its rank in the construction order.

The (a) and (c) replays are verified bit-for-bit against the live rollouts.

Outputs into FIG_DIR:
    fig_when.pdf   <- vector, use this in LaTeX
    fig_when.png   <- 600 dpi raster preview

Usage:
    python paper_fig_when.py            # figure only
    python paper_fig_when.py --enumerate    # also sweep all 11550 orders (~30 s)
"""

import os
import sys
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sb3_contrib import MaskablePPO

import _adv_env

ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "Figures")
RUN     = os.path.join(ROOT, "t10_errtol0.0003_werr1_wtime0.1_N30_c0.3_R4",
                       "t10_errtol0.0003_werr1_wtime0.1_N30_c0.3_R4", "seed3")
CK      = os.path.join(RUN, "checkpoints", "sched")

ns     = _adv_env.load({})
Env    = ns['PDEEnvironment']
dx, dt = ns['dx'], ns['dt']
u_true = ns['u_true']

ENV_KW = dict(R_bubble=4, t_star_steps=10, N_particles=30, err_tol=3e-4,
              w_err=1, w_time=0.1, K_max=2000, c_shape=0.3, x_star_schedule=[50])

# the middle panel, given by grid indices (mapped to exact floats below)
ORDER_B = [(49, 2), (48, 4), (52, 2), (51, 4), (47, 2), (47, 4),
           (51, 6), (47, 6), (51, 8), (48, 6), (46, 8)]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Live rollouts -- exact coordinates and construction order
# ═══════════════════════════════════════════════════════════════════════════
def live_rollout(step):
    model = MaskablePPO.load(os.path.join(CK, f"sched_{step}_steps.zip"),
                             device='cpu')
    env = Env(**ENV_KW)
    obs, _ = env.reset(seed=0)
    seq, done, info = [], False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True,
                             action_masks=env.action_masks())
        obs, _, done, _, info = env.step(a)
        if info['accepted']:
            i = info['walker']
            seq.append((i, float(env.walker_x[i]), float(env.walker_t[i])))
    return seq, info['error'], info['k_term']


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Replay an arbitrary construction order on a freshly seeded field
# ═══════════════════════════════════════════════════════════════════════════
_env = Env(**ENV_KW)
_env.reset(seed=0)
_V = _env.visited
_SNAP = (_V._n, _V._store[:_V._n].copy(), _V._normstore[:_V._n].copy(),
         set(_V._idx_set))
U_REF = u_true(_env.x_star, _env.t_star)


def replay(order):
    """order: list of exact (x, t) floats. Returns (error, stencil)."""
    n, st, nst, idx = _SNAP
    _V._n = n
    _V._store[:n] = st
    _V._normstore[:n] = nst
    _V._idx_set = set(idx)

    for (x, t) in order:
        # moves are solved with the env's own causality setting -- here
        # causal_moves = False, which is precisely why order matters
        uh, ok, _, _ = _env._gfdm_solve(x, t, causal=_env.causal_moves,
                                        box_R=None)
        if not ok:
            return None, None
        _V.add(x, t, uh)

    up, ok, nbc, _ = _env._gfdm_solve(_env.x_star, _env.t_star,
                                      causal=False, box_R=_env.R_bubble)
    return (None, None) if not ok else (abs(up - U_REF), nbc)


def enumerate_orders(chains):
    """Every interleaving preserving each chain's internal order."""
    lens = [len(c) for c in chains]
    n_tot = sum(lens)
    slots = range(n_tot)
    out = []
    for p0 in itertools.combinations(slots, lens[0]):
        r0 = [s for s in slots if s not in p0]
        for p1 in itertools.combinations(r0, lens[1]):
            p2 = [s for s in r0 if s not in p1]
            order = [None] * n_tot
            for k, s in enumerate(p0): order[s] = chains[0][k]
            for k, s in enumerate(p1): order[s] = chains[1][k]
            for k, s in enumerate(p2): order[s] = chains[2][k]
            e, _ = replay(order)
            if e is not None:
                out.append(e)
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Plot style -- matches the other paper figures
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
C_WALK    = {0: "#3b528b", 1: "#1f968b", 2: "#7ad151"}


def fmt_err(e):
    m, ex = f"{e:.1e}".split("e")
    return rf"${m}\times 10^{{{int(ex)}}}$"


def draw(ax, order, chains, walker_of, stencil, xlim, ylim):
    x_star, t_star = _env.x_star, _env.t_star
    rank = {(round(x / dx), round(t / dt)): k + 1
            for k, (x, t) in enumerate(order)}

    Rx, Rt = _env.R_bubble * dx, _env.R_bubble * dt
    ax.add_patch(patches.Rectangle((x_star - Rx, t_star - Rt), 2 * Rx, 2 * Rt,
                                   fill=False, edgecolor=C_STENCIL,
                                   ls=(0, (4, 2.5)), lw=0.9, zorder=2))
    for (xn, tn) in stencil:
        ax.plot([x_star, xn], [t_star, tn], ls=(0, (2.5, 1.8)),
                color=C_STENCIL, lw=0.8, alpha=0.85, zorder=3)

    for w, ch in chains.items():
        ax.plot([p[0] for p in ch], [p[1] for p in ch], '-', color=C_WALK[w],
                lw=1.0, alpha=0.5, zorder=4, solid_capstyle='round')

    for (x, t) in order:
        key = (round(x / dx), round(t / dt))
        w = walker_of[key]
        ax.plot([x], [t], 'o', ms=10.0, mfc='white', mec=C_WALK[w], mew=1.2,
                zorder=6)
        ax.text(x, t, str(rank[key]), ha='center', va='center', fontsize=5.6,
                color=C_WALK[w], zorder=7)

    ax.scatter([x_star], [t_star], s=140, marker='*', color=C_STENCIL,
               edgecolors='white', linewidths=0.5, zorder=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$x$')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    seq_a, err_a_live, k_a = live_rollout(6000000)
    seq_c, err_c_live, k_c = live_rollout(7200000)

    chains = {}
    for w, x, t in seq_c:
        chains.setdefault(w, []).append((x, t))
    coord = {(round(x / dx), round(t / dt)): (x, t)
             for ch in chains.values() for (x, t) in ch}
    walker_of = {(round(x / dx), round(t / dt)): w
                 for w, ch in chains.items() for (x, t) in ch}

    order_a = [(x, t) for _, x, t in seq_a]
    order_c = [(x, t) for _, x, t in seq_c]
    order_b = [coord[p] for p in ORDER_B]

    err_a, st_a = replay(order_a)
    err_b, st_b = replay(order_b)
    err_c, st_c = replay(order_c)

    assert err_a == err_a_live and err_c == err_c_live, \
        "replay does not reproduce the live rollout bit-for-bit"
    assert len({frozenset(map(tuple, np.round(o / np.array([dx, dt])).astype(int)))
                for o in (np.array(order_a), np.array(order_b), np.array(order_c))}) == 1, \
        "the three orders do not share one point set"
    assert len({frozenset(map(tuple, np.round(s / np.array([dx, dt])).astype(int)))
                for s in (st_a, st_b, st_c)}) == 1, \
        "the three orders do not share one stencil"

    print(f"(a) 6.0M checkpoint      k={k_a}  |e| = {err_a:.4e}   [live match]")
    print(f"(b) representative order       |e| = {err_b:.4e}")
    print(f"(c) 7.2M checkpoint      k={k_c}  |e| = {err_c:.4e}   [live match]")
    print("shared 11-point set and shared 5-point stencil at z*: confirmed")

    if "--enumerate" in sys.argv:
        errs = enumerate_orders([chains[w] for w in sorted(chains)])
        print(f"\nall {len(errs)} construction orders of this point set:")
        print(f"  min {errs.min():.3e}   median {np.median(errs):.3e}   "
              f"max {errs.max():.3e}   spread {errs.max()/errs.min():.0f}x")

    xlim = (45.2 * dx, 55.0 * dx)
    ylim = (0.7 * dt, 15.2 * dt)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), sharey=True,
                             constrained_layout=True)
    panels = [("a", order_a, st_a, err_a),
              ("b", order_b, st_b, err_b),
              ("c", order_c, st_c, err_c)]

    for ax, (tag, order, st, err) in zip(axes, panels):
        draw(ax, order, chains, walker_of, st, xlim, ylim)
        ax.set_title(rf"({tag})  $|e|=$ " + fmt_err(err), fontsize=9, pad=5)

    axes[0].set_ylabel(r'$t/\Delta t$')
    for ax in axes:
        ax.set_yticks(np.arange(2, 15, 2) * dt)
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _p: f"{v / dt:.0f}"))
        ax.set_xticks(np.arange(46, 55, 2) * dx)
        ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _p: f"{v:.2f}"))
        ax.tick_params(length=3, width=0.8)
        for s in ax.spines.values():
            s.set_color("#3a3f45")

    handles = [
        plt.Line2D([], [], ls='-', lw=1.0, marker='o', ms=5.5, mfc='white',
                   mec=C_WALK[w], mew=1.2, color=C_WALK[w], label=f'walker {w}')
        for w in sorted(chains)
    ] + [
        plt.Line2D([], [], ls=(0, (2.5, 1.8)), lw=0.8, color=C_STENCIL,
                   label=r'stencil at $z^\ast$'),
        plt.Line2D([], [], ls=(0, (4, 2.5)), lw=0.9, color=C_STENCIL,
                   label=r'bubble, $R=4$'),
        plt.Line2D([], [], ls='none', marker='*', ms=8, color=C_STENCIL,
                   label=r'$z^\ast=(x^\ast,t^\ast)$'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=6,
               frameon=False, handletextpad=0.5, columnspacing=1.2,
               borderaxespad=0.0)
    fig.suptitle(r'same 11 points, same stencil at $z^\ast$, '
                 r'three construction orders', fontsize=9)

    out_pdf = os.path.join(FIG_DIR, "fig_when.pdf")
    out_png = os.path.join(FIG_DIR, "fig_when.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
