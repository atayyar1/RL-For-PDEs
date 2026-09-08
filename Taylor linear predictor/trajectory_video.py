"""
trajectory_video.py

Generates an animated MP4 of the RL point-placement process for a single
deterministic rollout of a trained model, in the same visual style as the
final static plot (traj_t{t_star_steps}dt_final_reached.png).

Design notes (why it's built this way):

- PDEEnvironment is defined inline in the training notebook, not as an
  importable module. Rather than refactor that out (risky one step before
  a deadline), this file takes the class itself as an argument and
  instantiates it locally. Zero changes required to existing notebook code.

- The rollout here is a FRESH rollout, not a reuse of the one that produced
  the static png. This is safe because the rollout is fully deterministic:
  walker init positions are computed from a fixed formula (no RNG), the
  GFDM solve is pure linear algebra, and model.predict(deterministic=True)
  does not sample. Re-running the same model through a fresh env reproduces
  the identical sequence of accepted points and stencils.

- env.stencil_log already stores GFDM neighbour coordinates for every
  accepted point (not just z*), populated inside PDEEnvironment.step().
  No changes to the environment are needed to get "what neighbours it
  took" for each point in the sequence.

- Global acceptance order (interleaved across all walkers) is recorded
  here during the rollout, since plot_trajectories only keeps per-walker
  lists, not one combined chronological sequence.

- Each frame is a full redraw (ax.clear() + rebuild) rather than
  incremental artist updates. Slower per frame, but avoids any state
  shared across frame calls -- correctness over speed for what is a
  one-off figure generation, not a real-time loop.
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


def make_trajectory_video(model, PDEEnvironmentClass, t_star_steps, N_particles,
                           img_dir=".", label="final",
                           hold_frames=6, fps=10, dpi=120,
                           verbose=True, R_bubble=3, x_star_steps=None):
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
    fps : playback framerate
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

    trajectories = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}
    acceptance_order = []  # [(x, t, walker_id), ...] in the order actually accepted
    done = False
    info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, reward, done, _, info = env.step(action)
        if info['accepted']:
            i = info['walker']
            x_new, t_new = env.walker_x[i], env.walker_t[i]
            trajectories[i].append((x_new, t_new))
            acceptance_order.append((x_new, t_new, i))

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

    fig, ax = plt.subplots(figsize=(8, 6))

    def draw_static_background():
        ax.scatter(vx, vt, s=4, color='lightgray', zorder=1)
        for i in range(env.N):
            x0, t0 = trajectories[i][0]
            ax.scatter(x0, t0, s=80, color=colors[i], marker='^', zorder=4)
        # bubble box around z*
        Rx = env.R_bubble * env.dx
        Rt = env.R_bubble * env.dt
        ax.add_patch(patches.Rectangle(
            (env.x_star - Rx, env.t_star - Rt), 2 * Rx, 2 * Rt,
            fill=False, edgecolor='red', linestyle='--', linewidth=1.5, zorder=6))
        ax.scatter(env.x_star, env.t_star, s=150, color='red', marker='*', zorder=7)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, env.t_star * 2)

    # ── 3. Frame schedule ────────────────────────────────────────────────────
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

        ax.clear()
        draw_static_background()

        # Rebuild each walker's path from scratch up to point_idx (inclusive).
        # Recomputed every call on purpose -- no state shared across frames.
        paths = {i: [trajectories[i][0]] for i in range(env.N)}
        for k in range(point_idx + 1):
            x_k, t_k, wi = acceptance_order[k]
            paths[wi].append((x_k, t_k))

        for i in range(env.N):
            xs = [p[0] for p in paths[i]]
            ts = [p[1] for p in paths[i]]
            if len(xs) > 1:
                ax.plot(xs, ts, '-', linewidth=1.5, color=colors[i], zorder=2)
            ax.scatter(xs, ts, s=30, color=colors[i], zorder=3)

        # Current point's stencil connections -- transient
        x_cur, t_cur, _ = acceptance_order[point_idx]
        if show_stencil:
            key = (round(x_cur, 8), round(t_cur, 8))
            nb_coords = env.stencil_log.get(key)
            if nb_coords is not None:
                for (nx_, nt_) in nb_coords:
                    ax.plot([x_cur, nx_], [t_cur, nt_], '--', color='black',
                            linewidth=0.6, alpha=0.6, zorder=5)

        # z*'s stencil stays permanently once we're on the final point
        # (drawn for both bubble-fill and timeout — a solve happens either way)
        if point_idx == n_points - 1 and z_star_nb is not None:
            for (nx_, nt_) in z_star_nb:
                ax.plot([env.x_star, nx_], [env.t_star, nt_], '--', color='red',
                        linewidth=0.8, alpha=0.7, zorder=6)
            ax.scatter(z_star_nb[:, 0], z_star_nb[:, 1], s=40, color='red',
                       zorder=6, alpha=0.7)

        title = (f't_star={t_star_steps}\u00b7dt  |  {outcome}  |  '
                 f'point {point_idx + 1}/{n_points}')
        ax.set_title(title)
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


def make_training_progress_video(img_dir, t_star_steps, fps=3, dpi=120,
                                  outcome_filter=None, out_name=None,
                                  verbose=True):
    """
    Stitches the periodic snapshot pngs saved by PlotCallback during
    training (plus the final plot_trajectories call) into a single
    time-lapse mp4 showing how the learned trajectory structure evolved
    over training.

    Pure post-processing: reads pngs that already exist in img_dir. No
    model, no environment, no rollout. Can be called from the training
    notebook right after training finishes, or completely independently,
    any time later, on any img_dir that already has the pngs -- including
    old runs you're not retraining.

    Sorting is done by parsing the checkpoint step number out of each
    filename (not by alphabetical filename order, which only happens to
    work here by accident of zero-padding). The `final` checkpoint is
    always placed last regardless of its numeric non-existence.

    Parameters
    ----------
    img_dir : directory containing traj_t{t_star_steps}dt_*.png files
    t_star_steps : filters to only this target's images, so curriculum
        runs with multiple t_star stages in the same folder don't get
        mixed into one video
    fps : frames per second. Low fps (e.g. 2-3) = slower, more readable;
        with dozens of checkpoints this can still run quick, tune to taste
    dpi : output resolution
    outcome_filter : None (include both 'reached' and 'timeout' frames),
        or 'reached' / 'timeout' to only include one
    out_name : output filename, defaults to
        traj_t{t_star_steps}dt_training_progress.mp4
    verbose : print progress

    Returns
    -------
    str : path to the saved mp4
    """
    pattern = re.compile(
        rf'^traj_t{t_star_steps}dt_(?:(\d+)k|(final))_(reached|timeout)\.png$'
    )

    candidates = []
    for path in glob.glob(os.path.join(img_dir, f'traj_t{t_star_steps}dt_*.png')):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            continue
        step_str, is_final, outcome = m.groups()
        if outcome_filter is not None and outcome != outcome_filter:
            continue
        sort_key = float('inf') if is_final else int(step_str)
        candidates.append((sort_key, path))

    if not candidates:
        raise RuntimeError(
            f"No matching pngs found in {img_dir} for t_star_steps={t_star_steps}. "
            f"Check the folder path and that filenames follow the "
            f"traj_t{{t}}dt_{{step}}k_{{outcome}}.png / "
            f"traj_t{{t}}dt_final_{{outcome}}.png convention."
        )

    candidates.sort(key=lambda c: c[0])
    paths_in_order = [c[1] for c in candidates]

    if verbose:
        print(f"Found {len(paths_in_order)} frames, from "
              f"{os.path.basename(paths_in_order[0])} to "
              f"{os.path.basename(paths_in_order[-1])}")

    first_img = plt.imread(paths_in_order[0])
    h, w = first_img.shape[0], first_img.shape[1]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    im_artist = ax.imshow(first_img)

    def update(frame_idx):
        img = plt.imread(paths_in_order[frame_idx])
        im_artist.set_data(img)
        return [im_artist]

    ani = animation.FuncAnimation(fig, update, frames=len(paths_in_order), blit=True)

    if out_name is None:
        out_name = f'traj_t{t_star_steps}dt_training_progress.mp4'
    out_path = os.path.join(img_dir, out_name)

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"Saved video: {out_path}")

    return out_path
