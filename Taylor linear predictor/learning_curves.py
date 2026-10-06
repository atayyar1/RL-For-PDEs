"""
learning_curves.py

Plot a training run from its snapshots.json (written by SnapshotCallback): return, points and
exact bound of the policy's map against training steps, with baselines as horizontal lines.

    from learning_curves import plot_learning_curves
    plot_learning_curves(os.path.join(seed_dir, "snapshots.json"),
                         baselines={"LW": {"return": -2.0, "points": 400, "bound": 3.0e-5}})
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(snap_path, baselines=None, path=None):
    """baselines: {name: {"return": .., "points": .., "bound": ..}} (any key may be left out)."""
    with open(snap_path) as f:
        hist = json.load(f)
    step = np.array([h["step"] for h in hist])
    keys = [("return", "return (higher is better)", "linear"),
            ("points", "points in the map", "linear"),
            ("bound", "exact bound  sum |beta delta|", "log")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, label, scale) in zip(axes, keys):
        y = np.array([h[k] if h[k] is not None else np.nan for h in hist], dtype=float)
        ax.plot(step, y, "o-", ms=4, label="policy")
        for i, (name, b) in enumerate((baselines or {}).items()):
            if k in b:
                ax.axhline(b[k], ls="--", color=f"C{i + 1}", label=name)
        ax.set_yscale(scale); ax.set_xlabel("training steps"); ax.set_title(label); ax.legend(fontsize=8)
        if "episodes" in hist[0]:                               # second x-axis: maps built so far
            eps = np.array([h["episodes"] for h in hist], dtype=float)
            if eps[-1] > eps[0]:
                sec = ax.secondary_xaxis("top", functions=(lambda s: np.interp(s, step, eps),
                                                           lambda e: np.interp(e, eps, step)))
                sec.set_xlabel("maps built (episodes)")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()
