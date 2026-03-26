import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def interactive_plot_rbf(*args, alpha=1.0, n_exact=None, labels=("GFDM", "RBF")):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    def _looks_like_point_data(obj):
        return isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], dict) and "x" in obj[0]

    def _looks_like_visited(obj):
        return isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], tuple) and len(obj[0]) >= 3

    def _pack(point_data):
        return {
            "point_data": point_data,
            "px": np.array([p["x"] for p in point_data]),
            "pt": np.array([p["t"] for p in point_data]),
            "pu": np.array([p["u"] for p in point_data]),
            "perr": np.array([p["err"] for p in point_data]),
        }

    def _build_side(ax, data, title, ic_x, ic_t, bc_x, bc_t, norm, cmap_label):
        _style_ax(ax)
        ax.scatter(ic_x, ic_t, s=12, color=GREEN, zorder=2, label="IC", alpha=0.7)

        if bc_x is not None and bc_t is not None:
            ax.scatter(
                bc_x,
                bc_t,
                s=25,
                color=PURPLE,
                marker="D",
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                alpha=0.8,
                label="BC",
            )

        sc = ax.scatter(
            data["px"],
            data["pt"],
            c=data["perr"],
            cmap="Reds",
            s=30,
            zorder=4,
            edgecolors="white",
            linewidths=0.3,
            norm=norm,
            label="Visited",
        )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")
        plt.colorbar(sc, ax=ax, label=cmap_label, shrink=0.85)

    def _clear_artists(side):
        for artist in side["artists"]:
            try:
                artist.remove()
            except Exception:
                pass
        side["artists"].clear()

    def _nearest_clicked_index(ax, data, event):
        if event.inaxes != ax or event.button != 1:
            return None

        xc, tc = event.xdata, event.ydata
        if xc is None or tc is None:
            return None

        xy_disp = ax.transData.transform(np.column_stack([data["px"], data["pt"]]))
        click_disp = ax.transData.transform([[xc, tc]])[0]
        dists = np.sqrt(np.sum((xy_disp - click_disp) ** 2, axis=1))
        idx = int(np.argmin(dists))

        if dists[idx] > 20:
            return None
        return idx

    def _match_index(source_point, target_data):
        dx = target_data["px"] - source_point["x"]
        dt = target_data["pt"] - source_point["t"]

        exact_mask = (np.abs(dx) < 1e-12) & (np.abs(dt) < 1e-12)
        if np.any(exact_mask):
            return int(np.flatnonzero(exact_mask)[0]), "exact", 0.0

        x_scale = max(float(np.ptp(target_data["px"])), 1e-12)
        t_scale = max(float(np.ptp(target_data["pt"])), 1e-12)
        dist = np.sqrt((dx / x_scale) ** 2 + (dt / t_scale) ** 2)
        idx = int(np.argmin(dist))
        return idx, "nearest", float(dist[idx])

    def _draw_stencil(side, point, color, emphasize=False):
        ax = side["ax"]

        rect = patches.Rectangle(
            (point["x"] - point["box_x"], point["t"] - point["box_t"]),
            2 * point["box_x"],
            point["box_t"],
            linewidth=1.7,
            edgecolor=color,
            facecolor=color,
            alpha=0.07,
            zorder=2,
        )
        ax.add_patch(rect)
        side["artists"].append(rect)

        for xi, ti in zip(point["nb_x"], point["nb_t"]):
            line, = ax.plot(
                [point["x"], xi],
                [point["t"], ti],
                "-",
                color=BLUE,
                linewidth=1.1,
                alpha=0.45,
                zorder=3,
            )
            side["artists"].append(line)

        nb_sc = ax.scatter(
            point["nb_x"],
            point["nb_t"],
            s=72,
            color=BLUE,
            marker="s",
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
        )
        side["artists"].append(nb_sc)

        node = ax.scatter(
            [point["x"]],
            [point["t"]],
            s=210 if emphasize else 170,
            color=color,
            marker="o",
            zorder=6,
            edgecolors="black",
            linewidths=1.2,
        )
        side["artists"].append(node)

    def _style_ax(ax):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCCCC")
        ax.tick_params(colors="#444444", labelsize=9)
        ax.grid(True, alpha=0.2, linestyle="--", color="#AAAAAA")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.003, T * 1.06)
        ax.set_xlabel("$x$", fontsize=11)
        ax.set_ylabel("$t$", fontsize=11)

    BLUE = "#2C6FAC"
    ORANGE = "#E87722"
    RED = "#C0392B"
    GREEN = "#27AE60"
    PURPLE = "#8E44AD"

    if len(args) >= 10 and _looks_like_point_data(args[0]) and _looks_like_point_data(args[1]):
        point_data_left, point_data_right = args[0], args[1]
        visited_left, visited_right = args[2], args[3]
        u_true, fd_interp = args[4], args[5]
        T, dt_sc = args[6], args[7]
        ic_x, ic_t = args[8], args[9]
        bc_x = args[10] if len(args) > 10 else None
        bc_t = args[11] if len(args) > 11 else None
        comparison_mode = True
        del visited_left, visited_right, fd_interp, dt_sc
    elif len(args) >= 8 and _looks_like_point_data(args[0]) and _looks_like_visited(args[1]):
        point_data = args[0]
        visited = args[1]
        u_true, fd_interp = args[2], args[3]
        T, dt_sc = args[4], args[5]
        ic_x, ic_t = args[6], args[7]
        bc_x = args[8] if len(args) > 8 else None
        bc_t = args[9] if len(args) > 9 else None
        comparison_mode = False
        del visited, fd_interp, dt_sc
    else:
        raise TypeError(
            "interactive_plot_rbf expects either "
            "(point_data, visited, u_true, fd_interp, T, dt_sc, ic_x, ic_t, bc_x=None, bc_t=None) "
            "or "
            "(point_data_gfdm, point_data_rbf, visited_gfdm, visited_rbf, u_true, fd_interp, "
            "T, dt_sc, ic_x, ic_t, bc_x=None, bc_t=None)."
        )

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#F8F9FA")

    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[0.12, 0.88],
        hspace=0.22,
        wspace=0.22,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")

    if comparison_mode:
        left = _pack(point_data_left)
        right = _pack(point_data_right)

        left_err_max = float(np.max(left["perr"])) if len(left["perr"]) else 0.0
        right_err_max = float(np.max(right["perr"])) if len(right["perr"]) else 0.0
        norm = Normalize(vmin=0.0, vmax=max(left_err_max, right_err_max, 1e-10))

        text_display = ax_text.text(
            0.5,
            0.5,
            f"Click a point in {labels[0]} or {labels[1]} to compare both stencils and both errors",
            ha="center",
            va="center",
            fontsize=11.5,
            bbox=dict(boxstyle="round", facecolor="#D6EAF8", alpha=0.85, edgecolor=BLUE, linewidth=1.5),
        )

        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        _build_side(
            ax_left,
            left,
            f"{labels[0]} visited points\n(coloured by error)",
            ic_x,
            ic_t,
            bc_x,
            bc_t,
            norm,
            f'{labels[0]} error $|u - u_{{FD}}|$',
        )
        _build_side(
            ax_right,
            right,
            f"{labels[1]} visited points\n(coloured by error)",
            ic_x,
            ic_t,
            bc_x,
            bc_t,
            norm,
            f'{labels[1]} error $|u - u_{{FD}}|$',
        )

        left_side = {"ax": ax_left, "data": left, "artists": []}
        right_side = {"ax": ax_right, "data": right, "artists": []}

        def _update_text(source_label, left_point, right_point, match_mode, match_distance):
            text_display.set_text(
                f"Selected in {source_label}: ({left_point['x'] if source_label == labels[0] else right_point['x']:.3f}, "
                f"{left_point['t'] if source_label == labels[0] else right_point['t']:.4f})"
                f"  |  {labels[0]} err: {left_point['err']:.2e}"
                f"  |  {labels[1]} err: {right_point['err']:.2e}"
                f"  |  {labels[0]} u: {left_point['u']:.5f}"
                f"  |  {labels[1]} u: {right_point['u']:.5f}"
                f"  |  True: {u_true(left_point['x'], left_point['t']):.5f}"
                f"  |  match: {match_mode}{'' if match_mode == 'exact' else f' ({match_distance:.3e})'}"
            )

        def _set_comparison(source_key, source_idx):
            if source_key == "left":
                source_point = left["point_data"][source_idx]
                other_idx, match_mode, match_distance = _match_index(source_point, right)
                other_point = right["point_data"][other_idx]
                _clear_artists(left_side)
                _clear_artists(right_side)
                _draw_stencil(left_side, source_point, RED, emphasize=True)
                _draw_stencil(right_side, other_point, ORANGE)
                _update_text(labels[0], source_point, other_point, match_mode, match_distance)
            else:
                source_point = right["point_data"][source_idx]
                other_idx, match_mode, match_distance = _match_index(source_point, left)
                other_point = left["point_data"][other_idx]
                _clear_artists(left_side)
                _clear_artists(right_side)
                _draw_stencil(left_side, other_point, RED)
                _draw_stencil(right_side, source_point, ORANGE, emphasize=True)
                _update_text(labels[1], other_point, source_point, match_mode, match_distance)
            fig.canvas.draw_idle()

        def on_click(event):
            left_idx = _nearest_clicked_index(ax_left, left, event)
            if left_idx is not None:
                _set_comparison("left", left_idx)
                return

            right_idx = _nearest_clicked_index(ax_right, right, event)
            if right_idx is not None:
                _set_comparison("right", right_idx)

        fig.canvas.mpl_connect("button_press_event", on_click)

        fig.suptitle(
            rf"Interactive {labels[0]} vs {labels[1]} comparison"
            rf"  $u_t = \alpha u_{{xx}}$, $\alpha = {alpha}$, $T = {T}$"
            + ("" if n_exact is None else rf", $n = {n_exact}$ neighbours"),
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )

    else:
        data = _pack(point_data)
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(data["perr"])) if len(data["perr"]) else 0.0, 1e-10))

        text_display = ax_text.text(
            0.5,
            0.5,
            "Click any RBF point to inspect its stencil and error",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="#D6EAF8", alpha=0.85, edgecolor=BLUE, linewidth=1.5),
        )

        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        _build_side(
            ax_left,
            data,
            "RBF visited points\n(coloured by error)",
            ic_x,
            ic_t,
            bc_x,
            bc_t,
            norm,
            "RBF error $|u - u_{FD}|$",
        )

        _style_ax(ax_right)
        ax_right.set_title("Local RBF stencil\n(click a point on the left)", fontsize=10, fontweight="bold")
        ax_right.text(
            0.5,
            0.5,
            "Selected point and neighbours appear here",
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
            transform=ax_right.transAxes,
        )

        left_side = {"ax": ax_left, "data": data, "artists": []}
        right_side = {"ax": ax_right, "data": data, "artists": []}

        def _set_single(idx):
            point = data["point_data"][idx]
            _clear_artists(left_side)
            _clear_artists(right_side)

            _draw_stencil(left_side, point, RED, emphasize=True)
            _draw_stencil(right_side, point, ORANGE, emphasize=True)

            text_display.set_text(
                f"Point: ({point['x']:.3f}, {point['t']:.4f})"
                f"  |  Predicted: {point['u']:.5f}"
                f"  |  True: {u_true(point['x'], point['t']):.5f}"
                f"  |  Error: {point['err']:.2e}"
                f"  |  ||w||_1: {point['w1']:.3f}"
                f"  |  cond: {point['cond']:.1e}"
                f"  |  neighbours: {len(point['nb_x'])}"
            )
            fig.canvas.draw_idle()

        def on_click(event):
            idx = _nearest_clicked_index(ax_left, data, event)
            if idx is not None:
                _set_single(idx)

        fig.canvas.mpl_connect("button_press_event", on_click)

        fig.suptitle(
            rf"Interactive RBF stencil viewer  $u_t = \alpha u_{{xx}}$, $\alpha = {alpha}$, $T = {T}$"
            + ("" if n_exact is None else rf", $n = {n_exact}$ neighbours"),
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )

    plt.show()
