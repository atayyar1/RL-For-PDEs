import numpy as np
import matplotlib.pyplot as plt


BLUE = "#2C6FAC"
ORANGE = "#E87722"
GREEN = "#27AE60"
RED = "#C0392B"


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(True, alpha=0.25, linestyle="--", color="#AAAAAA")


def _coerce_point_data(run):
    if isinstance(run, dict):
        if "point_data" in run:
            run = run["point_data"]
        else:
            raise ValueError("Run dictionary must contain a 'point_data' key.")
    elif isinstance(run, (list, tuple)) and len(run) == 2:
        maybe_visited, maybe_point_data = run
        if isinstance(maybe_point_data, (list, tuple)) and len(maybe_point_data) > 0:
            first_point = maybe_point_data[0]
            if isinstance(first_point, dict) and "x" in first_point and "t" in first_point and "u" in first_point:
                run = maybe_point_data

    if not isinstance(run, (list, tuple)) or len(run) == 0:
        raise ValueError("Each run must be a non-empty list of point dictionaries.")

    first = run[0]
    if not isinstance(first, dict) or "x" not in first or "t" not in first or "u" not in first:
        raise ValueError("Each run must contain dictionaries with at least 'x', 't', and 'u'.")

    x = np.array([p["x"] for p in run], dtype=float)
    t = np.array([p["t"] for p in run], dtype=float)
    u = np.array([p["u"] for p in run], dtype=float)

    return {"x": x, "t": t, "u": u, "raw": run}


def collect_runs(run_once, n_runs, seeds=None, base_seed=0):
    """
    Repeats a user-supplied solver function and stores the point clouds.

    Parameters
    ----------
    run_once : callable
        Callable with signature run_once() or run_once(seed). It must return
        either point_data directly, a dictionary containing 'point_data', or
        a tuple of the form (visited, point_data).
    n_runs : int
        Number of Monte Carlo repetitions.
    seeds : sequence[int] | None
        Explicit seeds to use. If None, seeds base_seed, ..., base_seed+n_runs-1
        are used.
    base_seed : int
        Starting seed used when seeds is None.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    if seeds is None:
        seeds = list(range(base_seed, base_seed + n_runs))
    elif len(seeds) != n_runs:
        raise ValueError("len(seeds) must match n_runs.")

    runs = []
    for seed in seeds:
        np.random.seed(seed)
        try:
            result = run_once(seed)
        except TypeError:
            result = run_once()
        if isinstance(result, dict) and "point_data" in result:
            runs.append(result["point_data"])
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            runs.append(result)
        else:
            runs.append(result)

    return runs


def _bin_single_run(run_arrays, x_edges, t_edges):
    x_idx = np.digitize(run_arrays["x"], x_edges) - 1
    t_idx = np.digitize(run_arrays["t"], t_edges) - 1

    nx = len(x_edges) - 1
    nt = len(t_edges) - 1

    valid = (x_idx >= 0) & (x_idx < nx) & (t_idx >= 0) & (t_idx < nt)

    grid_sum = np.zeros((nt, nx), dtype=float)
    grid_count = np.zeros((nt, nx), dtype=int)

    for xi, ti, ui in zip(x_idx[valid], t_idx[valid], run_arrays["u"][valid]):
        grid_sum[ti, xi] += ui
        grid_count[ti, xi] += 1

    grid_mean = np.full((nt, nx), np.nan, dtype=float)
    mask = grid_count > 0
    grid_mean[mask] = grid_sum[mask] / grid_count[mask]

    return grid_mean


def prepare_uq_grid(runs, nx=40, nt=40, xlim=(0.0, 1.0), tlim=None, u_true=None):
    """
    Maps each scattered run onto the same coarse (x, t) grid, then computes
    run-to-run statistics in each cell.
    """
    if len(runs) < 2:
        raise ValueError("At least two runs are needed for uncertainty plots.")

    runs_arr = [_coerce_point_data(run) for run in runs]

    if tlim is None:
        t_max = max(float(np.max(run["t"])) for run in runs_arr)
        tlim = (0.0, t_max)

    x_edges = np.linspace(xlim[0], xlim[1], nx + 1)
    t_edges = np.linspace(tlim[0], tlim[1], nt + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    stacked = np.stack([_bin_single_run(run, x_edges, t_edges) for run in runs_arr], axis=0)
    run_count = np.sum(~np.isnan(stacked), axis=0)

    with np.errstate(invalid="ignore"):
        u_mean = np.nanmean(stacked, axis=0)
        u_std = np.nanstd(stacked, axis=0, ddof=0)
        u_q05 = np.nanquantile(stacked, 0.05, axis=0)
        u_q95 = np.nanquantile(stacked, 0.95, axis=0)

    true_grid = None
    mean_abs_error = None
    if u_true is not None:
        X, T = np.meshgrid(x_centers, t_centers)
        true_grid = u_true(X, T)
        with np.errstate(invalid="ignore"):
            mean_abs_error = np.nanmean(np.abs(stacked - true_grid[None, :, :]), axis=0)

    return {
        "stacked": stacked,
        "run_count": run_count,
        "x_edges": x_edges,
        "t_edges": t_edges,
        "x_centers": x_centers,
        "t_centers": t_centers,
        "u_mean": u_mean,
        "u_std": u_std,
        "u_q05": u_q05,
        "u_q95": u_q95,
        "true_grid": true_grid,
        "mean_abs_error": mean_abs_error,
    }


def plot_uq_slice(runs, t_value, nx=50, nt=50, band_sigma=2.0, ax=None, u_true=None):
    stats = prepare_uq_grid(runs, nx=nx, nt=nt, u_true=u_true)
    time_idx = int(np.argmin(np.abs(stats["t_centers"] - t_value)))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    _style_ax(ax)

    x = stats["x_centers"]
    mean_slice = stats["u_mean"][time_idx]
    std_slice = stats["u_std"][time_idx]
    q05_slice = stats["u_q05"][time_idx]
    q95_slice = stats["u_q95"][time_idx]
    count_slice = stats["run_count"][time_idx]

    valid = count_slice > 0
    ax.plot(x[valid], mean_slice[valid], color=BLUE, linewidth=2.2, label="Mean u")
    ax.fill_between(
        x[valid],
        mean_slice[valid] - band_sigma * std_slice[valid],
        mean_slice[valid] + band_sigma * std_slice[valid],
        color=BLUE,
        alpha=0.18,
        label=rf"Mean $\pm$ {band_sigma:.1f}$\sigma$",
    )
    ax.fill_between(
        x[valid],
        q05_slice[valid],
        q95_slice[valid],
        color=ORANGE,
        alpha=0.14,
        label="5%-95% band",
    )

    if u_true is not None:
        t_slice = float(stats["t_centers"][time_idx])
        ax.plot(
            x[valid],
            u_true(x[valid], t_slice),
            color=RED,
            linewidth=2.0,
            linestyle="--",
            label="Theoretical",
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_title(f"Run-to-run spread at t = {stats['t_centers'][time_idx]:.4f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    return ax, stats


def plot_uq_mean_error_heatmap(runs, nx=50, nt=50, ax=None, u_true=None):
    stats = prepare_uq_grid(runs, nx=nx, nt=nt, u_true=u_true)

    if stats["mean_abs_error"] is None:
        raise ValueError("u_true must be provided to plot the mean error heatmap.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.imshow(
        stats["mean_abs_error"],
        origin="lower",
        aspect="auto",
        extent=[
            stats["x_edges"][0],
            stats["x_edges"][-1],
            stats["t_edges"][0],
            stats["t_edges"][-1],
        ],
        cmap="Reds",
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label="Mean |u - u_true|")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_title("Mean error across runs", fontsize=10, fontweight="bold")

    return ax, stats


def sample_probes(runs, probes):
    """
    Samples u from each run at user-selected probe locations by taking the
    nearest scattered point in normalized (x, t) coordinates.
    """
    runs_arr = [_coerce_point_data(run) for run in runs]
    all_x = np.concatenate([run["x"] for run in runs_arr])
    all_t = np.concatenate([run["t"] for run in runs_arr])

    x_scale = max(float(np.ptp(all_x)), 1e-12)
    t_scale = max(float(np.ptp(all_t)), 1e-12)

    values = []
    for x_probe, t_probe in probes:
        probe_vals = []
        for run in runs_arr:
            dist = np.sqrt(((run["x"] - x_probe) / x_scale) ** 2 + ((run["t"] - t_probe) / t_scale) ** 2)
            idx = int(np.argmin(dist))
            probe_vals.append(run["u"][idx])
        values.append(np.array(probe_vals, dtype=float))

    return values


def plot_probe_histograms(runs, probes, bins=15, axes=None, u_true=None):
    probe_values = sample_probes(runs, probes)

    if axes is None:
        fig, axes = plt.subplots(1, len(probes), figsize=(5 * len(probes), 4))
        if len(probes) == 1:
            axes = [axes]
    else:
        fig = axes[0].figure

    for ax, (x_probe, t_probe), vals in zip(axes, probes, probe_values):
        _style_ax(ax)
        ax.hist(vals, bins=bins, color=GREEN, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(vals), color=RED, linewidth=2, label="Mean")
        ax.axvline(np.mean(vals) + np.std(vals), color=ORANGE, linestyle="--", linewidth=1.6, label="+1 std")
        ax.axvline(np.mean(vals) - np.std(vals), color=ORANGE, linestyle="--", linewidth=1.6, label="-1 std")
        if u_true is not None:
            u_exact = float(u_true(x_probe, t_probe))
            ax.axvline(
                u_exact,
                color=BLUE,
                linestyle="-.",
                linewidth=2.0,
                label="Theoretical",
            )
        ax.set_xlabel("$u$")
        ax.set_ylabel("Count")
        ax.set_title(f"Probe at (x={x_probe:.2f}, t={t_probe:.3f})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return axes, probe_values


def plot_uq_dashboard(runs, t_value=None, probes=None, nx=50, nt=50, bins=15, u_true=None):
    """
    One-call dashboard with three practical first-pass UQ plots:
    1. Mean and spread of u(x, t_value) with theoretical comparison
    2. Mean absolute error heatmap over the full domain
    3. Histograms at selected probe locations
    """
    runs_arr = [_coerce_point_data(run) for run in runs]
    t_max = max(float(np.max(run["t"])) for run in runs_arr)

    if t_value is None:
        t_value = 0.8 * t_max

    if probes is None:
        probes = [(0.25, 0.5 * t_max), (0.5, 0.8 * t_max), (0.75, 0.8 * t_max)]

    n_probe = len(probes)
    fig, axes = plt.subplots(2, max(2, n_probe), figsize=(5.2 * max(2, n_probe), 8.0))
    fig.patch.set_facecolor("#F8F9FA")

    top_axes = np.atleast_1d(axes[0])
    bottom_axes = np.atleast_1d(axes[1])

    plot_uq_slice(runs, t_value=t_value, nx=nx, nt=nt, ax=top_axes[0], u_true=u_true)
    plot_uq_mean_error_heatmap(runs, nx=nx, nt=nt, ax=top_axes[1], u_true=u_true)

    for extra_ax in top_axes[2:]:
        extra_ax.axis("off")

    plot_probe_histograms(runs, probes=probes, bins=bins, axes=bottom_axes[:n_probe], u_true=u_true)
    for extra_ax in bottom_axes[n_probe:]:
        extra_ax.axis("off")

    fig.suptitle("GFDM uncertainty quantification across repeated random runs", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
