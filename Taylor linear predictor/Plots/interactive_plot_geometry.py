import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def interactive_plot(
    point_data,
    visited,
    u_true,
    fd_interp,
    T,
    dt_sc,
    ic_x,
    ic_t,
    bc_x=None,
    bc_t=None,
    alpha=0.05,
    n_exact=None,
):
    """
    Interactive plotter for the geometry-only rollout notebook.

    Unlike the richer PDE-informed plotter, this version does not require
    per-point metadata such as box_x/box_t or stored neighbour arrays.
    When those fields are missing, it infers a reasonable neighbour stencil
    from the visited points so clicking remains informative instead of failing.
    """

    point_data = list(point_data or [])
    visited_arr = _coerce_visited(visited)

    px = np.array([p.get("x", np.nan) for p in point_data], dtype=float)
    pt = np.array([p.get("t", np.nan) for p in point_data], dtype=float)
    pu = np.array([p.get("u", np.nan) for p in point_data], dtype=float)
    perr = np.array([p.get("err", 0.0) for p in point_data], dtype=float)

    dt_scale = max(float(dt_sc), 1e-12) if dt_sc is not None else 1e-12
    dx_scale = _estimate_dx(visited_arr[:, 0]) if visited_arr.size else 1.0
    time_tol = max(dt_scale * 1e-9, 1e-12)

    blue = "#2C6FAC"
    orange = "#E87722"
    red = "#C0392B"
    green = "#27AE60"
    grey = "#7F8C8D"

    def eval_fd(x_vals, t_val):
        x_arr = np.atleast_1d(np.asarray(x_vals, dtype=float))
        t_arr = np.full(x_arr.shape, float(t_val), dtype=float)
        pts = np.column_stack([x_arr, t_arr])

        try:
            vals = np.asarray(fd_interp(pts), dtype=float).reshape(-1)
            if vals.size == x_arr.size:
                return vals
        except Exception:
            pass

        try:
            vals = np.array([float(fd_interp((xi, float(t_val)))) for xi in x_arr], dtype=float)
            if vals.size == x_arr.size:
                return vals
        except Exception:
            pass

        return np.array([float(fd_interp(float(xi), float(t_val))) for xi in x_arr], dtype=float)

    def infer_neighbours(p, max_neighbours=5):
        nb_x = np.asarray(p.get("nb_x", []), dtype=float)
        nb_t = np.asarray(p.get("nb_t", []), dtype=float)
        nb_u = np.asarray(p.get("nb_u", []), dtype=float)

        if nb_x.size and nb_t.size:
            box_x = float(p.get("box_x", max(np.max(np.abs(nb_x - p["x"])), dx_scale)))
            box_t = float(p.get("box_t", max(np.max(np.abs(nb_t - p["t"])), dt_scale)))
            return nb_x, nb_t, nb_u, box_x, box_t, True

        if not visited_arr.size:
            return np.array([]), np.array([]), np.array([]), None, None, False

        past_mask = visited_arr[:, 1] < (float(p["t"]) - time_tol)
        past = visited_arr[past_mask]
        if not past.size:
            return np.array([]), np.array([]), np.array([]), None, None, False

        dist2 = ((past[:, 0] - p["x"]) / dx_scale) ** 2 + ((past[:, 1] - p["t"]) / dt_scale) ** 2
        order = np.argsort(dist2)[:max_neighbours]
        nb = past[order]

        box_x = float(max(np.max(np.abs(nb[:, 0] - p["x"])), dx_scale))
        box_t = float(max(np.max(np.abs(nb[:, 1] - p["t"])), dt_scale))
        return nb[:, 0], nb[:, 1], nb[:, 2], box_x, box_t, False

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[0.1, 0.9],
        hspace=0.25,
        wspace=0.3,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax_text = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCCCC")
        ax.tick_params(colors="#444444", labelsize=9)
        ax.grid(True, alpha=0.2, linestyle="--", color="#AAAAAA")

    style_ax(ax_left)
    style_ax(ax_right)
    ax_text.axis("off")

    text_display = ax_text.text(
        0.5,
        0.5,
        "Click a computed point to inspect its local stencil and solution slice.",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            facecolor="#D6EAF8",
            alpha=0.85,
            edgecolor=blue,
            linewidth=1.5,
        ),
    )

    ax_left.scatter(ic_x, ic_t, s=14, color=green, zorder=3, label="IC", alpha=0.75)

    if bc_x is not None and bc_t is not None:
        ax_left.scatter(
            bc_x,
            bc_t,
            s=24,
            color=grey,
            marker="D",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
            alpha=0.8,
            label="BC",
        )

    sc = None
    if len(point_data):
        sc = ax_left.scatter(
            px,
            pt,
            c=perr,
            cmap="Reds",
            s=32,
            zorder=4,
            edgecolors="white",
            linewidths=0.3,
            norm=Normalize(vmin=0.0, vmax=max(float(np.nanmax(perr)), 1e-10)),
            label="Computed",
        )
        plt.colorbar(sc, ax=ax_left, label="Error |u - u_FD|", shrink=0.85)
    else:
        text_display.set_text("No computed points are available yet. Run the rollout cell first.")

    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_ylim(-0.003, T * 1.06)
    ax_left.set_xlabel("x", fontsize=11)
    ax_left.set_ylabel("t", fontsize=11)
    ax_left.set_title("Geometry-only rollout points", fontsize=10, fontweight="bold")

    handles, labels = ax_left.get_legend_handles_labels()
    if handles:
        ax_left.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

    x_plot = np.linspace(0.0, 1.0, 300)
    ax_right.set_xlabel("x", fontsize=11)
    ax_right.set_ylabel("u", fontsize=11)
    ax_right.set_title("Solution slice at clicked t", fontsize=10, fontweight="bold")
    ax_right.text(
        0.5,
        0.5,
        "Select a point on the left.",
        transform=ax_right.transAxes,
        ha="center",
        va="center",
        color=grey,
        fontsize=11,
    )

    interactive_artists = []

    def clear_interactive():
        for artist in interactive_artists:
            try:
                artist.remove()
            except Exception:
                pass
        interactive_artists.clear()

    def on_click(event):
        if not len(point_data):
            return
        if event.inaxes != ax_left or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        xy_disp = ax_left.transData.transform(np.column_stack([px, pt]))
        click_disp = ax_left.transData.transform([[event.xdata, event.ydata]])[0]
        dists = np.sqrt(np.sum((xy_disp - click_disp) ** 2, axis=1))

        idx = int(np.argmin(dists))
        if dists[idx] > 20:
            return

        p = point_data[idx]
        xq = float(p["x"])
        tq = float(p["t"])
        uq = float(p["u"])
        eq = float(p.get("err", 0.0))
        u_fd_q = float(eval_fd([xq], tq)[0])

        nb_x, nb_t, nb_u, box_x, box_t, from_payload = infer_neighbours(p)

        clear_interactive()

        if box_x is not None and box_t is not None:
            rect = patches.Rectangle(
                (xq - box_x, tq - box_t),
                2.0 * box_x,
                box_t,
                linewidth=1.6,
                edgecolor=red,
                facecolor=red,
                alpha=0.07,
                zorder=2,
            )
            ax_left.add_patch(rect)
            interactive_artists.append(rect)

        if nb_x.size and nb_t.size:
            for xi, ti in zip(nb_x, nb_t):
                line, = ax_left.plot([xq, xi], [tq, ti], "-", color=blue, linewidth=1.2, alpha=0.5, zorder=3)
                interactive_artists.append(line)

            nb_sc = ax_left.scatter(
                nb_x,
                nb_t,
                s=80,
                color=blue,
                marker="s",
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
                label="Stencil",
            )
            interactive_artists.append(nb_sc)

        node = ax_left.scatter(
            [xq],
            [tq],
            s=180,
            color=red,
            marker="o",
            zorder=6,
            edgecolors="black",
            linewidths=1.2,
        )
        interactive_artists.append(node)

        info_parts = [
            f"Point ({xq:.3f}, {tq:.4f})",
            f"u_hat={uq:.5f}",
            f"u_FD={u_fd_q:.5f}",
            f"err={eq:.2e}",
        ]
        if "cond" in p:
            info_parts.append(f"cond={float(p['cond']):.1e}")
        if "w1" in p:
            info_parts.append(f"||w||_1={float(p['w1']):.3f}")
        if nb_x.size:
            source = "stored" if from_payload else "inferred"
            info_parts.append(f"neighbours={len(nb_x)} ({source})")
        text_display.set_text(" | ".join(info_parts))

        ax_right.cla()
        style_ax(ax_right)

        u_fd_slice = eval_fd(x_plot, tq)
        ax_right.plot(x_plot, u_fd_slice, "-", color=green, linewidth=2.2, label="FD reference")

        same_t = np.isclose(pt, tq, rtol=0.0, atol=time_tol)
        if same_t.any():
            xs = px[same_t]
            us = pu[same_t]
            order = np.argsort(xs)
            ax_right.plot(
                xs[order],
                us[order],
                "-o",
                color=orange,
                linewidth=2.0,
                markersize=4.5,
                label="Geometry rollout",
            )

        ax_right.scatter(
            [xq],
            [uq],
            s=150,
            color=red,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="Clicked point",
            zorder=5,
        )

        if nb_x.size and nb_u.size:
            ax_right.scatter(
                nb_x,
                nb_u,
                s=80,
                color=blue,
                marker="s",
                edgecolors="white",
                linewidths=0.8,
                alpha=0.75,
                label="Stencil values",
                zorder=4,
            )

        ax_right.set_xlabel("x", fontsize=11)
        ax_right.set_ylabel("u", fontsize=11)
        ax_right.set_title(f"Solution slice at t = {tq:.4f}", fontsize=10, fontweight="bold")

        handles, labels = ax_right.get_legend_handles_labels()
        if handles:
            ax_right.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    n_text = "auto" if n_exact is None else str(n_exact)
    fig.suptitle(
        f"Interactive geometry-only rollout | alpha={alpha} | T={T} | n={n_text}",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    plt.show()
    return fig


def _coerce_visited(visited):
    if visited is None:
        return np.empty((0, 3), dtype=float)
    try:
        arr = np.asarray(list(visited), dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3]


def _estimate_dx(x_vals):
    x_unique = np.unique(np.asarray(x_vals, dtype=float))
    if x_unique.size < 2:
        return 1.0
    diffs = np.diff(x_unique)
    diffs = diffs[diffs > 1e-12]
    if diffs.size == 0:
        return 1.0
    return float(np.min(diffs))
