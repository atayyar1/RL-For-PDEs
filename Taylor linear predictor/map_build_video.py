"""
map_build_video.py

Video of one map being built backward from z*, in the order the points were expanded
(latest t first). Needs only the finished map m: m["stencil"] keeps the expansion order.

Each frame: expanded points coloured by their influence |beta| on z*, pending points (needed
but not yet given a stencil) as open circles, and the stencils of the newest points in red.

    from map_build_video import make_build_video
    make_build_video(env.m, "build.mp4", is_known=is_known, title="PPO, seed 1")
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass


def make_build_video(m, path, is_known, title="", n_frames=120, fps=12, dpi=100):
    """m: finished map; is_known(q): True for IC / wall points (they are never computed)."""
    order = list(m["stencil"])                                   # expansion order
    z = m["z_star"]
    beta = {p: abs(m["beta"][p]) for p in order}
    cuts = np.unique(np.linspace(1, len(order), n_frames).astype(int))
    P = np.array(order)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(P[:, 0].min() - 3, P[:, 0].max() + 3)
    ax.set_ylim(-0.5, z[1] + 1)
    ax.set_xlabel("ix"); ax.set_ylabel("it")
    ax.axhline(0, color="k", lw=1)                               # the initial condition
    done_sc = ax.scatter([], [], s=12, c=[], cmap="viridis", norm=LogNorm(1e-8, 1))
    pend_sc = ax.scatter([], [], s=14, facecolors="none", edgecolors="gray", linewidths=0.8)
    new_lines, = ax.plot([], [], "-", color="crimson", lw=0.8)
    ax.plot(*z, "r*", ms=14)
    fig.colorbar(done_sc, ax=ax, label="|beta|  (influence on z*)")
    head = ax.set_title("")

    def update(f):
        k = cuts[f]
        done = order[:k]
        done_set = set(done)
        pending = {q for p in done for q in m["stencil"][p][0] if not is_known(q) and q not in done_set}
        D = np.array(done)
        done_sc.set_offsets(D); done_sc.set_array(np.array([beta[p] for p in done]))
        pend_sc.set_offsets(np.array(list(pending)) if pending else np.empty((0, 2)))
        xs, ys = [], []
        for p in order[max(0, k - 5):k]:                         # stencils of the 5 newest points
            for q in m["stencil"][p][0]:
                xs += [p[0], q[0], np.nan]; ys += [p[1], q[1], np.nan]
        new_lines.set_data(xs, ys)
        head.set_text(f"{title}   {k} / {len(order)} points expanded, {len(pending)} pending")
        return done_sc, pend_sc, new_lines, head

    anim = animation.FuncAnimation(fig, update, frames=len(cuts), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"{len(cuts)} frames, {len(order)} points | saved: {path}")
    return path
