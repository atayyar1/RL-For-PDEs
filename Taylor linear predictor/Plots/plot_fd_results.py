import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --------------------------------------------------
# Colors
# --------------------------------------------------
BLUE = '#2C6FAC'
GREEN = '#27AE60'
ORANGE = '#E87722'
GREY = '#7F8C8D'


# --------------------------------------------------
# Axis styling
# --------------------------------------------------
def style_ax(ax):

    ax.set_facecolor('white')

    ax.spines[['top', 'right']].set_visible(False)

    ax.spines[['left', 'bottom']].set_color('#CCCCCC')

    ax.tick_params(colors='#444444', labelsize=9)

    ax.grid(True, alpha=0.25, linestyle='--', color='#AAAAAA')


# --------------------------------------------------
# Main plotting function
# --------------------------------------------------
def plot_fd_results(
    x,
    t,
    T,
    U_method,
    U_fd,
    u_true,
    w,
    w_fd,
    title="Method comparison"
):

    # --------------------------------------------------
    # Compute errors
    # --------------------------------------------------
    X, Tgrid = np.meshgrid(x, t)

    U_true = u_true(X, Tgrid)

    err_method_vs_true = np.abs(U_method - U_true)
    err_fd_vs_true = np.abs(U_fd - U_true)
    err_method_vs_fd = np.abs(U_method - U_fd)

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#F8F9FA')

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.38,
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.08
    )

    # --------------------------------------------------
    # Panel 1: Weight comparison
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)

    n = len(w)
    xpos = np.arange(n)
    width = 0.35

    # automatic stencil labels
    if n == 3:
        labels = ['Centre', 'Left', 'Right']
    elif n == 5:
        labels = ['Centre', 'Left', 'Right', 'Left2', 'Right2']
    else:
        labels = [f'w{i}' for i in range(n)]

    ax1.bar(xpos - width/2, w, width, color=BLUE, label='Our weights')
    ax1.bar(xpos + width/2, w_fd, width, color=ORANGE, label='FD weights')

    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels)

    ax1.set_ylabel('Weight value')

    ax1.set_title(f'Weight comparison ({n}-point stencil)', fontsize=10, fontweight='bold')

    ax1.legend(fontsize=9)

    # --------------------------------------------------
    # Panel 2: Solution at final time
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)

    ax2.plot(x, u_true(x, T), '-', color=GREY, linewidth=2.5, label='True solution')

    ax2.plot(x, U_fd[-1], '--', color=GREEN, linewidth=2, label='FD')

    ax2.plot(
        x,
        U_method[-1],
        'o-',
        color=BLUE,
        linewidth=1.5,
        markersize=4,
        label='Our method',
        markeredgecolor='white'
    )

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$u$')

    ax2.set_title(f'Solution at $t = {T:.3f}$', fontsize=10, fontweight='bold')

    ax2.legend(fontsize=9)

    # --------------------------------------------------
    # Panel 3: Error vs time
    # --------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)

    err_method_t = err_method_vs_true.mean(axis=1)
    err_fd_t = err_fd_vs_true.mean(axis=1)

    ax3.semilogy(t, err_method_t, '-', color=BLUE, linewidth=2, label='Our vs True')

    ax3.semilogy(t, err_fd_t, '--', color=GREEN, linewidth=2, label='FD vs True')

    ax3.set_xlabel('$t$')
    ax3.set_ylabel('Mean error')

    ax3.set_title('Mean error over time', fontsize=10, fontweight='bold')

    ax3.legend(fontsize=9)

    # --------------------------------------------------
    # Panel 4: Our method heatmap
    # --------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])

    im4 = ax4.imshow(
        U_method,
        origin='lower',
        aspect='auto',
        extent=[0, 1, 0, T],
        cmap='viridis'
    )

    plt.colorbar(im4, ax=ax4, label='$u$', shrink=0.85)

    ax4.set_xlabel('$x$')
    ax4.set_ylabel('$t$')

    ax4.set_title('Our method solution', fontsize=10, fontweight='bold')

    # --------------------------------------------------
    # Panel 5: FD heatmap
    # --------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])

    im5 = ax5.imshow(
        U_fd,
        origin='lower',
        aspect='auto',
        extent=[0, 1, 0, T],
        cmap='viridis'
    )

    plt.colorbar(im5, ax=ax5, label='$u$', shrink=0.85)

    ax5.set_xlabel('$x$')
    ax5.set_ylabel('$t$')

    ax5.set_title('FD reference solution', fontsize=10, fontweight='bold')

    # --------------------------------------------------
    # Panel 6: Error heatmap
    # --------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])

    im6 = ax6.imshow(
        err_method_vs_fd,
        origin='lower',
        aspect='auto',
        extent=[0, 1, 0, T],
        cmap='Reds'
    )

    plt.colorbar(im6, ax=ax6, label='$|u_{method} - u_{FD}|$', shrink=0.85)

    ax6.set_xlabel('$x$')
    ax6.set_ylabel('$t$')

    ax6.set_title('Error: Our method vs FD', fontsize=10, fontweight='bold')

    # --------------------------------------------------
    # Figure title
    # --------------------------------------------------
    fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.show()