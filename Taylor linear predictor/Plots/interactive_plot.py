import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    del visited, alpha, n_exact

    point_data = list(point_data or [])
    px = np.array([p["x"] for p in point_data], dtype=float)
    pt = np.array([p["t"] for p in point_data], dtype=float)
    pu = np.array([p["u"] for p in point_data], dtype=float)

    time_tol = max(float(dt_sc) * 1e-9, 1e-12) if dt_sc is not None else 1e-12

    def get_series(field):
        if not point_data:
            return None
        if any(field not in p or p[field] is None for p in point_data):
            return None
        try:
            return np.array([float(p[field]) for p in point_data], dtype=float)
        except Exception:
            return None

    def eval_true_pairs(x_vals, t_vals):
        if u_true is None:
            return None

        x_arr = np.asarray(x_vals, dtype=float).reshape(-1)
        t_arr = np.asarray(t_vals, dtype=float).reshape(-1)
        if x_arr.size != t_arr.size:
            raise ValueError("x_vals and t_vals must have the same size")

        try:
            vals = np.asarray(u_true(x_arr, t_arr), dtype=float).reshape(-1)
            if vals.size == x_arr.size:
                return vals
            if vals.size == 1:
                return np.full(x_arr.shape, float(vals[0]), dtype=float)
        except Exception:
            pass

        return np.array([float(u_true(float(xi), float(ti))) for xi, ti in zip(x_arr, t_arr)], dtype=float)

    def eval_true_slice(x_vals, t_val):
        x_arr = np.asarray(x_vals, dtype=float).reshape(-1)
        t_arr = np.full(x_arr.shape, float(t_val), dtype=float)
        return eval_true_pairs(x_arr, t_arr)

    def eval_fd_pairs(x_vals, t_vals):
        if fd_interp is None:
            return None

        x_arr = np.asarray(x_vals, dtype=float).reshape(-1)
        t_arr = np.asarray(t_vals, dtype=float).reshape(-1)
        if x_arr.size != t_arr.size:
            raise ValueError("x_vals and t_vals must have the same size")

        pts = np.column_stack([x_arr, t_arr])

        try:
            vals = np.asarray(fd_interp(pts), dtype=float).reshape(-1)
            if vals.size == x_arr.size:
                return vals
        except Exception:
            pass

        try:
            vals = np.array(
                [float(fd_interp((float(xi), float(ti)))) for xi, ti in zip(x_arr, t_arr)],
                dtype=float,
            )
            if vals.size == x_arr.size:
                return vals
        except Exception:
            pass

        return np.array([float(fd_interp(float(xi), float(ti))) for xi, ti in zip(x_arr, t_arr)], dtype=float)

    def eval_fd_slice(x_vals, t_val):
        x_arr = np.asarray(x_vals, dtype=float).reshape(-1)
        t_arr = np.full(x_arr.shape, float(t_val), dtype=float)
        return eval_fd_pairs(x_arr, t_arr)

    def finite_max(values, default=1e-10):
        if values is None:
            return default
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return default
        return max(float(vals.max()), default)

    def fmt_value(value, number_fmt=".5f", missing="n/a"):
        if value is None:
            return missing
        try:
            if np.isfinite(value):
                return format(float(value), number_fmt)
        except Exception:
            pass
        return missing

    u_true_pts = get_series("u_true")
    if u_true_pts is None and len(point_data):
        u_true_pts = eval_true_pairs(px, pt)

    u_fd_pts = get_series("u_fd")
    if u_fd_pts is None and len(point_data):
        u_fd_pts = eval_fd_pairs(px, pt)

    err_true = get_series("err_true")
    if err_true is None and u_true_pts is not None:
        err_true = np.abs(pu - u_true_pts)

    err_fd = get_series("err_fd")
    if err_fd is None:
        legacy_err = get_series("err")
        if legacy_err is not None:
            err_fd = legacy_err
        elif u_fd_pts is not None:
            err_fd = np.abs(pu - u_fd_pts)

    if err_true is not None:
        perr = err_true
        err_label = r"Error $|u - u_{true}|$"
        err_title = "coloured by error vs true solution"
    elif err_fd is not None:
        perr = err_fd
        err_label = r"Error $|u - u_{FD}|$"
        err_title = "coloured by error vs FD reference"
    else:
        perr = np.zeros_like(pu)
        err_label = "Error"
        err_title = "coloured by stored error"

    perr_plot = np.where(np.isfinite(perr), perr, 0.0)

    blue = "#2C6FAC"
    orange = "#E87722"
    red = "#C0392B"
    green = "#27AE60"
    grey = "#7F8C8D"
    purple = "#8E44AD"

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

    def style_ax(ax):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCCCC")
        ax.tick_params(colors="#444444", labelsize=9)
        ax.grid(True, alpha=0.2, linestyle="--", color="#AAAAAA")

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")

    text_display = ax_text.text(
        0.5,
        0.5,
        "Click a visited point to compare your rollout against the true and FD references.",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            facecolor="#D6EAF8",
            alpha=0.8,
            edgecolor=blue,
            linewidth=1.5,
        ),
    )

    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    style_ax(ax_left)
    style_ax(ax_right)

    ax_left.scatter(ic_x, ic_t, s=12, color=green, zorder=3, label="IC", alpha=0.7)

    if bc_x is not None and bc_t is not None:
        ax_left.scatter(
            bc_x,
            bc_t,
            s=25,
            color=purple,
            marker="D",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            alpha=0.8,
            label="BC",
        )

    sc = ax_left.scatter(
        px,
        pt,
        c=perr_plot,
        cmap="Reds",
        s=30,
        zorder=4,
        edgecolors="white",
        linewidths=0.3,
        norm=Normalize(vmin=0.0, vmax=finite_max(perr_plot)),
        label="Visited",
    )

    plt.colorbar(sc, ax=ax_left, label=err_label, shrink=0.85)

    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_ylim(-0.003, T * 1.06)
    ax_left.set_xlabel("$x$", fontsize=11)
    ax_left.set_ylabel("$t$", fontsize=11)
    ax_left.set_title(
        f"Visited points\n({err_title})",
        fontsize=10,
        fontweight="bold",
    )
    ax_left.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

    x_plot = np.linspace(0.0, 1.0, 300)
    ax_right.set_xlabel("$x$", fontsize=11)
    ax_right.set_ylabel("$u$", fontsize=11)
    ax_right.set_title(
        "Solution slice at clicked $t$\n(click a point on the left)",
        fontsize=10,
        fontweight="bold",
    )
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
        if not point_data:
            return
        if event.inaxes != ax_left or event.button != 1:
            return

        xc, tc = event.xdata, event.ydata
        if xc is None or tc is None:
            return

        xy_disp = ax_left.transData.transform(np.column_stack([px, pt]))
        click_disp = ax_left.transData.transform([[xc, tc]])[0]
        dists = np.sqrt(np.sum((xy_disp - click_disp) ** 2, axis=1))

        idx = int(np.argmin(dists))
        if dists[idx] > 20:
            return

        p = point_data[idx]
        xq = float(p["x"])
        tq = float(p["t"])
        uq = float(p["u"])

        u_true_q = float(u_true_pts[idx]) if u_true_pts is not None else None
        if u_true_q is None or not np.isfinite(u_true_q):
            u_true_eval = eval_true_pairs([xq], [tq])
            u_true_q = float(u_true_eval[0]) if u_true_eval is not None else None

        u_fd_q = float(u_fd_pts[idx]) if u_fd_pts is not None else None
        if u_fd_q is None or not np.isfinite(u_fd_q):
            u_fd_eval = eval_fd_pairs([xq], [tq])
            u_fd_q = float(u_fd_eval[0]) if u_fd_eval is not None else None

        err_true_q = float(err_true[idx]) if err_true is not None else None
        if (err_true_q is None or not np.isfinite(err_true_q)) and u_true_q is not None and np.isfinite(u_true_q):
            err_true_q = abs(uq - u_true_q)

        err_fd_q = float(err_fd[idx]) if err_fd is not None else None
        if (err_fd_q is None or not np.isfinite(err_fd_q)) and u_fd_q is not None and np.isfinite(u_fd_q):
            err_fd_q = abs(uq - u_fd_q)

        clear_interactive()

        bx = float(p["box_x"])
        bt = float(p["box_t"])

        rect = patches.Rectangle(
            (xq - bx, tq - bt),
            2.0 * bx,
            bt,
            linewidth=1.8,
            edgecolor=red,
            facecolor=red,
            alpha=0.07,
            zorder=2,
        )

        ax_left.add_patch(rect)
        interactive_artists.append(rect)

        for xi, ti in zip(p["nb_x"], p["nb_t"]):
            line, = ax_left.plot(
                [xq, xi],
                [tq, ti],
                "-",
                color=blue,
                linewidth=1.2,
                alpha=0.5,
                zorder=3,
            )
            interactive_artists.append(line)

        nb_sc = ax_left.scatter(
            p["nb_x"],
            p["nb_t"],
            s=80,
            color=blue,
            marker="s",
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
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

        text_display.set_text(
            f"Point: ({xq:.3f}, {tq:.4f})  |  "
            f"Predicted: {uq:.5f}  |  "
            f"True: {fmt_value(u_true_q)}  |  "
            f"FD: {fmt_value(u_fd_q)}  |  "
            f"|u-u_true|: {fmt_value(err_true_q, '.2e')}  |  "
            f"|u-u_FD|: {fmt_value(err_fd_q, '.2e')}  |  "
            f"||w||_1: {p['w1']:.3f}  |  "
            f"cond(A): {p['cond']:.1e}  |  "
            f"neighbours: {len(p['nb_x'])}"
        )

        ax_right.cla()
        style_ax(ax_right)

        u_true_slice = eval_true_slice(x_plot, tq)
        if u_true_slice is not None:
            ax_right.plot(
                x_plot,
                u_true_slice,
                "-",
                color=green,
                linewidth=2.4,
                label="True reference",
            )

        u_fd_slice = eval_fd_slice(x_plot, tq)
        if u_fd_slice is not None:
            ax_right.plot(
                x_plot,
                u_fd_slice,
                "--",
                color=grey,
                linewidth=2.0,
                label="FD reference",
            )

        mask = np.isclose(pt, tq, rtol=0.0, atol=time_tol)
        if mask.any():
            xs = px[mask]
            us = pu[mask]

            order = np.argsort(xs)
            xs = xs[order]
            us = us[order]

            ax_right.plot(
                xs,
                us,
                "-o",
                color=orange,
                linewidth=2.0,
                markersize=5,
                label="Your method",
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
        )

        ax_right.scatter(
            p["nb_x"],
            p["nb_u"],
            s=80,
            color=blue,
            marker="s",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.7,
            label="Neighbours used",
        )

        ax_right.set_xlabel("$x$", fontsize=11)
        ax_right.set_ylabel("$u$", fontsize=11)
        ax_right.set_title(
            f"Solution slice at $t = {tq:.4f}$",
            fontsize=10,
            fontweight="bold",
        )

        handles, labels = ax_right.get_legend_handles_labels()
        if handles:
            ax_right.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
