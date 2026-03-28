import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
def interactive_plot(point_data, visited, u_true, fd_interp,
                     T, dt_sc, ic_x, ic_t,
                     bc_x=None, bc_t=None,
                     alpha=0.05, n_exact=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    px = np.array([p['x'] for p in point_data])
    pt = np.array([p['t'] for p in point_data])
    pu = np.array([p['u'] for p in point_data])
    perr = np.array([p['err'] for p in point_data])

    time_tol = max(float(dt_sc) * 1e-9, 1e-12) if dt_sc is not None else 1e-12

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

    # ─────────────────────────────────────────────
    # Interactive figure
    # ─────────────────────────────────────────────
    BLUE   = '#2C6FAC'
    ORANGE = '#E87722'
    RED    = '#C0392B'
    GREEN  = '#27AE60'
    GREY   = '#7F8C8D'

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#F8F9FA')

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[0.1, 0.9],
        hspace=0.25,
        wspace=0.3,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.08
    )

    def style_ax(ax):
        ax.set_facecolor('white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#CCCCCC')
        ax.tick_params(colors='#444444', labelsize=9)
        ax.grid(True, alpha=0.2, linestyle='--', color='#AAAAAA')

    # ── Info bar ──────────────────────────────────
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')

    text_display = ax_text.text(
        0.5, 0.5,
        'Click on any visited point to see its neighbours, search box, and solution slice',
        ha='center', va='center', fontsize=12,
        bbox=dict(
            boxstyle='round',
            facecolor='#D6EAF8',
            alpha=0.8,
            edgecolor=BLUE,
            linewidth=1.5
        )
    )

    ax_left  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    style_ax(ax_left)
    style_ax(ax_right)

    # IC points
    # IC points
    ax_left.scatter(ic_x, ic_t, s=12, color=GREEN, zorder=3, label='IC', alpha=0.7)

    # BC points (always visible like IC)
    if bc_x is not None:
        ax_left.scatter(
            bc_x, bc_t,
            s=25,
            color='purple',
            marker='D',
            edgecolors='white',
            linewidths=0.5,
            zorder=4,
            alpha=0.8,
            label='BC'
        )
    sc = ax_left.scatter(
        px, pt,
        c=perr,
        cmap='Reds',
        s=30,
        zorder=4,
        edgecolors='white',
        linewidths=0.3,
        norm=Normalize(vmin=0, vmax=max(perr.max(), 1e-10)),
        label='Visited'
    )

    plt.colorbar(sc, ax=ax_left, label='Error $|u - u_{FD}|$', shrink=0.85)

    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_ylim(-0.003, T * 1.06)

    ax_left.set_xlabel('$x$', fontsize=11)
    ax_left.set_ylabel('$t$', fontsize=11)

    ax_left.set_title(
        'Visited points — click to inspect\n(coloured by error vs FD reference)',
        fontsize=10,
        fontweight='bold'
    )

    ax_left.legend(fontsize=8, framealpha=0.9, edgecolor='#CCCCCC')

    # ── Right panel default ───────────────────────

    x_plot = np.linspace(0, 1, 300)

    ax_right.set_xlabel('$x$', fontsize=11)
    ax_right.set_ylabel('$u$', fontsize=11)

    ax_right.set_title(
        'Solution slice at clicked $t$\n(click a point on the left)',
        fontsize=10,
        fontweight='bold'
    )

    ax_right.legend(fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')

    # ─────────────────────────────────────────────
    # Click handler
    # ─────────────────────────────────────────────

    _interactive_artists = []

    def clear_interactive():
        for artist in _interactive_artists:
            try:
                artist.remove()
            except Exception:
                pass
        _interactive_artists.clear()

    def on_click(event):

        if event.inaxes != ax_left or event.button != 1:
            return

        xc, tc = event.xdata, event.ydata
        if xc is None or tc is None:
            return

        xy_disp = ax_left.transData.transform(np.column_stack([px, pt]))
        click_disp = ax_left.transData.transform([[xc, tc]])[0]
        dists = np.sqrt(np.sum((xy_disp - click_disp)**2, axis=1))

        idx = int(np.argmin(dists))
        if dists[idx] > 20:
            return

        p = point_data[idx]
        xq, tq, uq, eq = p['x'], p['t'], p['u'], p['err']
        u_fd_q = float(eval_fd([xq], tq)[0])

        clear_interactive()

        bx = p['box_x']
        bt = p['box_t']

        rect = patches.Rectangle(
            (xq - bx, tq - bt),
            2 * bx, bt,
            linewidth=1.8,
            edgecolor=RED,
            facecolor=RED,
            alpha=0.07,
            zorder=2
        )

        ax_left.add_patch(rect)
        _interactive_artists.append(rect)

        for xi, ti in zip(p['nb_x'], p['nb_t']):
            line, = ax_left.plot(
                [xq, xi], [tq, ti],
                '-', color=BLUE, linewidth=1.2, alpha=0.5, zorder=3
            )
            _interactive_artists.append(line)

        nb_sc = ax_left.scatter(
            p['nb_x'], p['nb_t'],
            s=80, color=BLUE, marker='s', zorder=5,
            edgecolors='white', linewidths=0.8
        )
        _interactive_artists.append(nb_sc)

        node = ax_left.scatter(
            [xq], [tq],
            s=180, color=RED, marker='o', zorder=6,
            edgecolors='black', linewidths=1.2
        )
        _interactive_artists.append(node)

        text_display.set_text(
            f'Point: ({xq:.3f}, {tq:.4f})  |  '
            f'Predicted: {uq:.5f}  |  '
            f'FD: {u_fd_q:.5f}  |  '
            f'Error: {eq:.2e}  |  '
            f'||w||_1: {p["w1"]:.3f}  |  '
            f'cond(A): {p["cond"]:.1e}  |  '
            f'neighbours: {len(p["nb_x"])}'
        )

        ax_right.cla()
        style_ax(ax_right)

        u_fd_slice = eval_fd(x_plot, tq)

        ax_right.plot(
            x_plot, u_fd_slice,
            '-', color=GREEN, linewidth=2.2,
            label='FD reference'
        )

        mask = np.isclose(pt, tq, rtol=0.0, atol=time_tol)

        if mask.any():
            xs = px[mask]
            us = pu[mask]

            order = np.argsort(xs)
            xs = xs[order]
            us = us[order]

            ax_right.plot(
                xs, us,
                '-o',
                color=ORANGE,
                linewidth=2,
                markersize=5,
                label='Your method'
            )

        ax_right.scatter(
            [xq], [uq],
            s=150, color=RED,
            marker='*',
            edgecolors='black',
            linewidths=0.8,
            label='Clicked point'
        )

        ax_right.scatter(
            p['nb_x'], p['nb_u'],
            s=80, color=BLUE, marker='s',
            edgecolors='white',
            linewidths=0.8,
            alpha=0.7,
            label='Neighbours used'
        )
 

        ax_right.set_xlabel('$x$', fontsize=11)
        ax_right.set_ylabel('$u$', fontsize=11)

        ax_right.set_title(
            f'Solution slice at $t = {tq:.4f}$',
            fontsize=10,
            fontweight='bold'
        )

        ax_right.legend(fontsize=8.5, framealpha=0.9, edgecolor='#CCCCCC')

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    fig.suptitle(
        rf'Interactive scattered integrator — $u_t = \alpha u_{{xx}}$, '
        rf'$\alpha = {alpha}$, $T = {T}$, random neighbour selection, '
        rf'$n = {n_exact}$ neighbours',
        fontsize=12,
        fontweight='bold',
        y=0.99
    )

    plt.show()
