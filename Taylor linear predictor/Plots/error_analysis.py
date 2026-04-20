import numpy as np
import matplotlib.pyplot as plt


def error_analysis(point_data, u_true=None, fd_interp=None):
    point_data = list(point_data or [])

    if not point_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No point data available for error analysis.", ha="center", va="center")
        plt.tight_layout()
        plt.show()
        return

    px = np.array([p["x"] for p in point_data], dtype=float)
    pt = np.array([p["t"] for p in point_data], dtype=float)
    pu = np.array([p["u"] for p in point_data], dtype=float)
    cond_vals = np.array([p["cond"] for p in point_data], dtype=float)
    w1_vals = np.array([p["w1"] for p in point_data], dtype=float)

    def get_series(field):
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

        try:
            vals = np.asarray(u_true(x_arr, t_arr), dtype=float).reshape(-1)
            if vals.size == x_arr.size:
                return vals
            if vals.size == 1:
                return np.full(x_arr.shape, float(vals[0]), dtype=float)
        except Exception:
            pass

        return np.array([float(u_true(float(xi), float(ti))) for xi, ti in zip(x_arr, t_arr)], dtype=float)

    def eval_fd_pairs(x_vals, t_vals):
        if fd_interp is None:
            return None

        x_arr = np.asarray(x_vals, dtype=float).reshape(-1)
        t_arr = np.asarray(t_vals, dtype=float).reshape(-1)
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

        return None

    def get_fd_regular_grid_error_series(max_points=None):
        if fd_interp is None or u_true is None:
            return None

        grid = getattr(fd_interp, "grid", None)
        values = getattr(fd_interp, "values", None)
        if grid is None or values is None or len(grid) != 2:
            return None

        try:
            x_grid = np.asarray(grid[0], dtype=float).reshape(-1)
            t_grid = np.asarray(grid[1], dtype=float).reshape(-1)
            u_grid = np.asarray(values, dtype=float)
        except Exception:
            return None

        if u_grid.ndim != 2 or u_grid.shape != (x_grid.size, t_grid.size):
            return None

        # Match the FD solver's computed interior points: skip IC (t=0) and boundaries.
        x_inner = x_grid[1:-1]
        t_inner = t_grid[1:]
        if x_inner.size == 0 or t_inner.size == 0:
            return None

        xx, tt = np.meshgrid(x_inner, t_inner, indexing="ij")
        u_true_grid = eval_true_pairs(xx.ravel(order="F"), tt.ravel(order="F"))
        if u_true_grid is None:
            return None

        fd_errs = np.abs(u_grid[1:-1, 1:].ravel(order="F") - u_true_grid)
        if max_points is not None:
            fd_errs = fd_errs[:max_points]

        return fd_errs

    u_true_vals = get_series("u_true")
    if u_true_vals is None:
        u_true_vals = eval_true_pairs(px, pt)

    u_fd_vals = get_series("u_fd")
    if u_fd_vals is None:
        u_fd_vals = eval_fd_pairs(px, pt)

    # error of your method against truth
    err_mine = get_series("err_true")
    if err_mine is None and u_true_vals is not None:
        err_mine = np.abs(pu - u_true_vals)

    # error of FD against truth
    err_fd_true_points = None
    if u_fd_vals is not None and u_true_vals is not None:
        err_fd_true_points = np.abs(u_fd_vals - u_true_vals)

    err_fd_true_curve = get_fd_regular_grid_error_series(max_points=len(point_data))
    if err_fd_true_curve is None:
        err_fd_true_curve = err_fd_true_points

    series = []
    if err_mine is not None:
        series.append(
            {
                "label": "GFDM vs Analytical",
                "curve_errs": np.asarray(err_mine, dtype=float),
                "scatter_errs": np.asarray(err_mine, dtype=float),
                "color": "#2C6FAC",
                "linestyle": "-",
            }
        )
    if err_fd_true_curve is not None:
        series.append(
            {
                "label": "FD vs Analytical",
                "curve_errs": np.asarray(err_fd_true_curve, dtype=float),
                "scatter_errs": None
                if err_fd_true_points is None
                else np.asarray(err_fd_true_points, dtype=float),
                "color": "#E87722",
                "linestyle": "--",
            }
        )

    if not series:
        series.append(
            {
                "label": "Stored error",
                "curve_errs": np.zeros(len(point_data), dtype=float),
                "scatter_errs": np.zeros(len(point_data), dtype=float),
                "color": "#2C6FAC",
                "linestyle": "-",
            }
        )

    tiny = np.finfo(float).tiny

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))

    for item in series:
        label = item["label"]
        color = item["color"]
        linestyle = item["linestyle"]
        curve_errs = item["curve_errs"]
        scatter_errs = item["scatter_errs"]

        cumulative_error = np.cumsum(curve_errs)
        point_counts = np.arange(1, curve_errs.size + 1)
        avg_error = cumulative_error / point_counts

        ax[0].plot(point_counts, cumulative_error, linewidth=2, color=color, linestyle=linestyle, label=label)
        ax[1].plot(point_counts, np.maximum(avg_error, tiny), linewidth=2, color=color, linestyle=linestyle, label=label)

        if scatter_errs is None:
            continue

        finite_mask = np.isfinite(scatter_errs) & np.isfinite(cond_vals) & (cond_vals > 0)
        if np.any(finite_mask):
            ax[2].scatter(
                cond_vals[finite_mask],
                np.maximum(scatter_errs[finite_mask], tiny),
                alpha=0.55,
                s=26,
                color=color,
                label=label,
            )

        finite_mask = np.isfinite(scatter_errs) & np.isfinite(w1_vals) & (w1_vals > 0)
        if np.any(finite_mask):
            ax[3].scatter(
                w1_vals[finite_mask],
                np.maximum(scatter_errs[finite_mask], tiny),
                alpha=0.55,
                s=26,
                color=color,
                label=label,
            )

    ax[0].set_xlabel("Points evaluated")
    ax[0].set_ylabel("Accumulated error")
    ax[0].set_title("Total error accumulation")
    ax[0].grid(True)

    ax[1].set_yscale("log")
    ax[1].set_xlabel("Points evaluated")
    ax[1].set_ylabel("Average error")
    ax[1].set_title("Running average error")
    ax[1].grid(True)

    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("cond(A)")
    ax[2].set_ylabel("Error")
    ax[2].set_title("Error vs conditioning")
    ax[2].grid(True)

    ax[3].set_yscale("log")
    ax[3].set_xlabel(r"$||w||_1$")
    ax[3].set_ylabel("Error")
    ax[3].set_title("Error vs weight magnitude")
    ax[3].grid(True)

    for axis in ax:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(framealpha=0.9)

    plt.tight_layout()
    plt.show()
