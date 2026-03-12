import numpy as np
import matplotlib.pyplot as plt


def error_analysis(point_data):

    px   = np.array([p['x'] for p in point_data])
    pt   = np.array([p['t'] for p in point_data])
    perr = np.array([p['err'] for p in point_data])
    cond_vals = np.array([p['cond'] for p in point_data])
    w1_vals   = np.array([p['w1'] for p in point_data])

    # sort by time
    order = np.argsort(pt)
    pt = pt[order]
    perr = perr[order]

    # accumulated error
    cumulative_error = np.cumsum(perr)

    # running average error
    avg_error = cumulative_error / np.arange(1, len(perr)+1)

    # ─────────────────────────────────────
    # Figure layout
    # ─────────────────────────────────────
    fig, ax = plt.subplots(1, 4, figsize=(18,4))

    # accumulated error
    ax[0].plot(pt, cumulative_error, linewidth=2)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Accumulated error")
    ax[0].set_title("Total error accumulation")
    ax[0].grid(True)

    # average error
    ax[1].plot(pt, avg_error, linewidth=2)

    ax[1].set_yscale("log")

    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Average error")
    ax[1].set_title("Running average error")
    ax[1].grid(True)

    # conditioning vs error
    ax[2].scatter(cond_vals, perr, alpha=0.6)

    ax[2].set_xscale("log")
    ax[2].set_yscale("log")

    ax[2].set_xlabel("cond(A)")
    ax[2].set_ylabel("Error")

    ax[2].set_title("Error vs conditioning")
    ax[2].grid(True)

    # weight magnitude vs error
    ax[3].scatter(w1_vals, perr, alpha=0.6)

    ax[3].set_yscale("log")

    ax[3].set_xlabel("||w||₁")
    ax[3].set_ylabel("Error")

    ax[3].set_title("Error vs weight magnitude")
    ax[3].grid(True)

    plt.tight_layout()
    plt.show()