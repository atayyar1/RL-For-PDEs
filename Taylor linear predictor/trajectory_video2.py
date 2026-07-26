"""
trajectory_video.py

Generates an animated MP4 of the RL point-placement process for a single
deterministic rollout of a trained model, in the same visual style as the
final static plot (traj_t{t_star_steps}dt_final_reached.png).

Design notes (why it's built this way):

[... original notes unchanged ...]

--- NEW: state/action schematic panel ---

Right of the trajectory plot, two plain numeric tables replace a literal
node-and-edge NN diagram:

- "State (input)" panel: N x 2 table, columns = [dx_rel, dt_rel], one row
  per walker, exact values printed as text. This matches the u-free
  observation space (2*N) -- u was dropped from _build_obs to test
  generalization across t* (a walker's u depended on t* even when its
  (dx_rel, dt_rel) didn't, which broke the intended invariance).
- "Action probs (output)" panel: N x n_actions table, the actual masked
  softmax distribution the policy computed over all N*n_actions joint
  actions at that step (not just the argmax), exact probability printed
  per cell.

No heatmap coloring -- values are text only, per request. The row of the
walker that acted is still tinted (color-matched to its trajectory-panel
color) so it's identifiable among 26 rows without reading every line; in
the output table the specific chosen (walker, action) cell gets an extra
highlight. This is a structural marker, not a magnitude-based color scale,
so it doesn't reintroduce the "heatmap" being removed.

Only DECISIONS THAT LED TO AN ACCEPTED MOVE are logged and shown, one per
animation "point" (same unit the trajectory panel already advances on).
Steps where the GFDM solve rejected the proposed point are not shown in
this panel -- this matches the existing hold_frames/point_idx schedule
without restructuring it. If you also want rejected attempts visible,
that's a bigger change to the frame schedule and should be a separate
pass.

Architecture note: this assumes the trained model's policy is a plain
MlpPolicy with no policy_kwargs override (confirmed against the training
cell -- default SB3 net_arch, which for MaskablePPO is
Linear(3N, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, N*n_actions)
for both pi and vf, verified empirically against a fresh model instance).
This panel doesn't draw the architecture live per frame (it doesn't
change), only the two tables update.
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


def make_trajectory_video1(model, PDEEnvironmentClass, t_star_steps, N_particles,
                           img_dir=".", label="final",
                           hold_frames=6, fps=10 / 3, dpi=120,
                           verbose=True, x_star_steps=None, R_bubble=4):   # NEW
    """
    Parameters
    ----------
    model : trained MaskablePPO model
    PDEEnvironmentClass : the PDEEnvironment class object itself (pass the
        class, not an instance -- e.g. `PDEEnvironment`)
    t_star_steps, N_particles : same meaning as in plot_trajectories
    img_dir : directory to save the mp4 (same folder you pass to
        plot_trajectories, so the video lands next to the pngs)
    label : filename label, matches plot_trajectories's `label` argument
    hold_frames : number of frames each point's stencil lines stay visible
        before being dropped (the point itself always persists)
    fps : playback framerate. Default is 10/3 (~3.33) -- 3x slower than
        the original default of 10, same frame count, just held on screen
        3x longer.
    dpi : output resolution
    verbose : print progress

    Returns
    -------
    str : path to the saved mp4
    """
    # ── 1. Fresh deterministic rollout, recording global acceptance order ──
    env = PDEEnvironmentClass(
            t_star_steps=t_star_steps, N_particles=N_particles,
            R_bubble=R_bubble,
            x_star_schedule=None if x_star_steps is None else [x_star_steps])
    obs, _ = env.reset()

    N = env.N
    n_actions = len(env.ACTIONS)
    action_labels = [f"dx={dx_i:+d}\ndt={dt_i:+d}" for dx_i, dt_i in env.ACTIONS.values()]

    trajectories = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}
    acceptance_order = []  # [(x, t, walker_id), ...] in the order actually accepted
    decision_log = []      # parallel to acceptance_order: (state_vec, walker_i, action_j, probs)
    done = False
    info = {}

    while not done:
        action_masks = env.action_masks()
        obs_before = np.array(obs, dtype=np.float32)

        # masked action-probability distribution the policy actually used
        obs_tensor, _ = model.policy.obs_to_tensor(obs_before)
        dist = model.policy.get_distribution(obs_tensor, action_masks=action_masks[None])
        probs = dist.distribution.probs.detach().cpu().numpy()[0]

        action, _ = model.predict(obs_before, deterministic=True, action_masks=action_masks)
        obs, reward, done, _, info = env.step(action)

        if info['accepted']:
            i = info['walker']
            x_new, t_new = env.walker_x[i], env.walker_t[i]
            trajectories[i].append((x_new, t_new))
            acceptance_order.append((x_new, t_new, i))

            j = int(action) % n_actions
            decision_log.append((obs_before, i, j, probs))

    reached = info.get('reached', False)
    outcome = 'reached' if reached else 'timeout'
    n_points = len(acceptance_order)

    if n_points == 0:
        raise RuntimeError("Rollout accepted zero points -- nothing to animate.")

    if verbose:
        print(f"Rollout done: {outcome}, {n_points} accepted points, "
              f"{env.step_count} total steps.")

    # z*'s own stencil — the exact solve step() scored: box points on a
    # bubble-fill, global nearest neighbours on a timeout (both non-causal).
    z_star_nb = None
    _box_R = env.R_bubble if reached else None
    _, success, nb_coords, _ = env._gfdm_solve(
        env.x_star, env.t_star, causal=False, box_R=_box_R)
    if success:
        z_star_nb = nb_coords

    # ── 2. Static background (IC/BC) ────────────────────────────────────────
    visited_log = [(x, t) for (x, t, _) in env.visited._pts]
    vx = [p[0] for p in visited_log]
    vt = [p[1] for p in visited_log]

    colors = plt.cm.viridis(np.linspace(0, 1, env.N))

    # ── 3. Figure layout: trajectory | state table | prob table ────────────
    fig = plt.figure(figsize=(15, 6.5), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.15, 1.35])
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_state = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[0, 2])

    def draw_static_background():
        ax_traj.scatter(vx, vt, s=4, color='lightgray', zorder=1)
        for i in range(env.N):
            x0, t0 = trajectories[i][0]
            ax_traj.scatter(x0, t0, s=80, color=colors[i], marker='^', zorder=4)
        # bubble box around z*
        Rx = env.R_bubble * env.dx
        Rt = env.R_bubble * env.dt
        ax_traj.add_patch(patches.Rectangle(
            (env.x_star - Rx, env.t_star - Rt), 2 * Rx, 2 * Rt,
            fill=False, edgecolor='red', linestyle='--', linewidth=1.5, zorder=6))
        ax_traj.scatter(env.x_star, env.t_star, s=150, color='red', marker='*', zorder=7)
        ax_traj.set_xlabel('x')
        ax_traj.set_ylabel('t')
        ax_traj.set_xlim(0, 1)
        ax_traj.set_ylim(0, env.t_star * 2)

    def draw_table(ax, mat, col_labels, fmt, acted_row, acted_col=None):
        """Plain numeric table -- no heatmap fill. acted_row gets a tint
        matching that walker's trajectory color; acted_col (if given) gets
        an extra highlight on top of that."""
        ax.clear()
        ax.axis('off')

        cell_text = [[fmt(v) for v in row] for row in mat]
        row_labels = [str(i) for i in range(mat.shape[0])]

        tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                        loc='center', cellLoc='center', rowLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.scale(1, 0.78)

        n_cols = mat.shape[1]
        acted_color = colors[acted_row]
        # tint the whole acted row (data cells + its row-label cell)
        for c in range(-1, n_cols):
            cell = tbl[(acted_row + 1, c)]  # +1: row 0 is the header
            cell.set_facecolor(acted_color)
            cell.set_alpha(0.35)
        if acted_col is not None:
            chosen_cell = tbl[(acted_row + 1, acted_col)]
            chosen_cell.set_facecolor('lime')
            chosen_cell.set_alpha(0.6)
            chosen_cell.set_edgecolor('black')
            chosen_cell.set_linewidth(1.5)
        return tbl

    # ── 4. Frame schedule ────────────────────────────────────────────────────
    # Each point gets (1 + hold_frames) frames: the stencil is shown for the
    # first `hold_frames` of them, then dropped. Point + path segment persist
    # in every subsequent frame regardless.
    total_frames = n_points * (1 + hold_frames)

    def frame_state(frame_idx):
        point_idx = min(frame_idx // (1 + hold_frames), n_points - 1)
        offset = frame_idx % (1 + hold_frames)
        show_stencil = offset < hold_frames
        return point_idx, show_stencil

    def update(frame_idx):
        point_idx, show_stencil = frame_state(frame_idx)

        ax_traj.clear()
        draw_static_background()

        # Rebuild each walker's path from scratch up to point_idx (inclusive).
        paths = {i: [trajectories[i][0]] for i in range(env.N)}
        for k in range(point_idx + 1):
            x_k, t_k, wi = acceptance_order[k]
            paths[wi].append((x_k, t_k))

        for i in range(env.N):
            xs = [p[0] for p in paths[i]]
            ts = [p[1] for p in paths[i]]
            if len(xs) > 1:
                ax_traj.plot(xs, ts, '-', linewidth=1.5, color=colors[i], zorder=2)
            ax_traj.scatter(xs, ts, s=30, color=colors[i], zorder=3)

        x_cur, t_cur, _ = acceptance_order[point_idx]
        if show_stencil:
            key = (round(x_cur, 8), round(t_cur, 8))
            nb_coords = env.stencil_log.get(key)
            if nb_coords is not None:
                for (nx_, nt_) in nb_coords:
                    ax_traj.plot([x_cur, nx_], [t_cur, nt_], '--', color='black',
                                 linewidth=0.6, alpha=0.6, zorder=5)

        if point_idx == n_points - 1 and z_star_nb is not None:
            for (nx_, nt_) in z_star_nb:
                ax_traj.plot([env.x_star, nx_], [env.t_star, nt_], '--', color='red',
                             linewidth=0.8, alpha=0.7, zorder=6)
            ax_traj.scatter(z_star_nb[:, 0], z_star_nb[:, 1], s=40, color='red',
                             zorder=6, alpha=0.7)

        xtag = f'x_idx={x_star_steps}' if x_star_steps is not None else f'x_star={env.x_star:.2f}'
        title = (f't_star={t_star_steps}\u00b7dt  |  {xtag}  |  '
                 f'{outcome}  |  point {point_idx + 1}/{n_points}')
        ax_traj.set_title(title)

        # ── state / action panel for this decision ──
        state_vec, acted_i, acted_j, probs = decision_log[point_idx]
        state_mat = np.column_stack([state_vec[:N], state_vec[N:2 * N]])
        t_star_norm = state_vec[2 * N]       # NEW
        x_star_norm = state_vec[2 * N + 1]   # NEW  

        prob_mat = probs.reshape(N, n_actions)

        draw_table(ax_state, state_mat, ['dx_rel', 'dt_rel'],
                   fmt=lambda v: f"{v:.2f}", acted_row=acted_i)
        ax_state.set_title(
            f'State (input)  |  walker {acted_i} acted\n'
            f't*_norm={t_star_norm:.1f}  x*_norm={x_star_norm:.1f}',   # NEW
            fontsize=9)

        draw_table(ax_prob, prob_mat, action_labels,
                   fmt=lambda v: f"{v:.2f}", acted_row=acted_i, acted_col=acted_j)
        ax_prob.set_title('Action probs (output, masked)', fontsize=10)

        return []

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    fname = f'traj_t{t_star_steps}dt_{label}_{outcome}.mp4'
    out_path = os.path.join(img_dir, fname)
    os.makedirs(img_dir, exist_ok=True)

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"Saved video: {out_path}")

    return out_path