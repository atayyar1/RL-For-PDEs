"""
training_progress_video.py

Create an MP4 time-lapse from the trajectory PNG figures generated during
training by plot_trajectories / PlotCallback.

Expected input filenames inside img_dir:
    traj_t{t_star_steps}dt_{step}k_reached.png
    traj_t{t_star_steps}dt_{step}k_timeout.png
    traj_t{t_star_steps}dt_final_reached.png
    traj_t{t_star_steps}dt_final_timeout.png
    traj_t{t_star_steps}dt_x{x_star_steps}_{step}k_reached.png
    traj_t{t_star_steps}dt_x{x_star_steps}_final_reached.png

Example:
    from training_progress_video import make_training_progress_video

    make_training_progress_video(
        img_dir=img_dir,
        t_star_steps=t_star_steps,
        fps=3
    )
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_training_progress_video(
    img_dir,
    t_star_steps,
    *plot_args,
    fps=3,
    dpi=120,
    outcome_filter=None,
    out_name=None,
    verbose=True,
    label="final",
    x_star_steps=None,
):
    """
    Stitches the saved trajectory PNGs from training into one MP4 video.

    This is pure post-processing: it does NOT use the model, environment,
    or rollout. It only reads the PNG files already saved in img_dir.

    Parameters
    ----------
    img_dir : str
        Folder containing the trajectory PNGs generated during training.

    t_star_steps : int
        The same t_star_steps used during training. This filters the files
        so different target-time runs do not get mixed together.

    *plot_args
        Optional legacy plot metadata arguments. Some notebooks pass
        N_particles, lam, beta, K_max, c_shape here to mirror
        plot_trajectories(...). They are accepted and ignored.

    fps : int, default=3
        Frames per second for the output video. Lower = slower and easier
        to show to an advisor. Good values: 2, 3, 5.

    dpi : int, default=120
        Output video resolution scaling.

    outcome_filter : None, 'reached', or 'timeout', default=None
        If None, include both reached and timeout figures.
        If 'reached', include only reached figures.
        If 'timeout', include only timeout figures.

    out_name : str or None, default=None
        Output filename. If None, defaults to:
            traj_t{t_star_steps}dt_training_progress.mp4

    verbose : bool, default=True
        Print progress messages.

    label : str, default='final'
        Label used by the final plot_trajectories call. With the default,
        this includes files like traj_t{t_star_steps}dt_final_reached.png.

    x_star_steps : int or None, default=None
        Optional x-star schedule tag. If None, the helper prefers untagged
        files, or auto-selects the only x-tagged set in img_dir.

    Returns
    -------
    str
        Path to the saved MP4 file.
    """

    if plot_args:
        legacy_outcomes = (None, "reached", "timeout")
        looks_like_old_options = (
            len(plot_args) <= 2
            or (len(plot_args) == 3 and plot_args[2] in legacy_outcomes)
            or (len(plot_args) == 4 and plot_args[2] in legacy_outcomes)
            or (
                len(plot_args) == 5
                and plot_args[2] in legacy_outcomes
                and isinstance(plot_args[4], bool)
            )
        )

        if looks_like_old_options:
            if len(plot_args) >= 1:
                fps = plot_args[0]
            if len(plot_args) >= 2:
                dpi = plot_args[1]
            if len(plot_args) >= 3:
                outcome_filter = plot_args[2]
            if len(plot_args) >= 4:
                out_name = plot_args[3]
            if len(plot_args) >= 5:
                verbose = plot_args[4]
        # Otherwise these are plot_trajectories metadata arguments
        # (N_particles, lam, beta, K_max, c_shape). They are intentionally
        # ignored because this helper only needs saved PNGs.

    if outcome_filter not in (None, "reached", "timeout"):
        raise ValueError("outcome_filter must be None, 'reached', or 'timeout'.")

    if label is None:
        label = "final"
    label = str(label)

    pattern = re.compile(
        rf"^traj_t{re.escape(str(t_star_steps))}dt"
        rf"(?P<x_tag>_x[^_]+)?_"
        rf"(?:(?P<step>\d+)k|(?P<label>{re.escape(label)}))_"
        rf"(?P<outcome>reached|timeout)\.png$"
    )

    all_candidates = []

    search_pattern = os.path.join(img_dir, f"traj_t{t_star_steps}dt*.png")
    for path in glob.glob(search_pattern):
        fname = os.path.basename(path)
        match = pattern.match(fname)

        if not match:
            continue

        x_tag = match.group("x_tag") or ""
        step_str = match.group("step")
        is_labelled_final = match.group("label") is not None
        outcome = match.group("outcome")

        if outcome_filter is not None and outcome != outcome_filter:
            continue

        # Numeric checkpoints come first in order; final always goes last.
        sort_key = float("inf") if is_labelled_final else int(step_str)
        all_candidates.append((x_tag, sort_key, path))

    requested_x_tag = None
    if x_star_steps is not None:
        requested_x_tag = f"_x{x_star_steps}"
        candidates = [
            (sort_key, path)
            for x_tag, sort_key, path in all_candidates
            if x_tag == requested_x_tag
        ]
        selected_x_tag = requested_x_tag
    else:
        tags = sorted({x_tag for x_tag, _, _ in all_candidates})
        if "" in tags:
            selected_x_tag = ""
        elif len(tags) == 1:
            selected_x_tag = tags[0]
        elif len(tags) > 1:
            available = ", ".join(tag[1:] for tag in tags)
            raise RuntimeError(
                "Found multiple x-star tagged trajectory sets "
                f"({available}) in {img_dir}. Pass x_star_steps=... "
                "to choose one."
            )
        else:
            selected_x_tag = ""

        candidates = [
            (sort_key, path)
            for x_tag, sort_key, path in all_candidates
            if x_tag == selected_x_tag
        ]

    if not candidates:
        x_tag_hint = requested_x_tag or selected_x_tag
        x_tag_hint = x_tag_hint if x_tag_hint else ""
        raise RuntimeError(
            f"No matching trajectory PNGs found in: {img_dir}\n"
            f"Expected files like:\n"
            f"  traj_t{t_star_steps}dt{x_tag_hint}_10k_reached.png\n"
            f"  traj_t{t_star_steps}dt{x_tag_hint}_20k_timeout.png\n"
            f"  traj_t{t_star_steps}dt{x_tag_hint}_{label}_reached.png\n"
            f"Check img_dir, t_star_steps, and your saved filenames."
        )

    candidates.sort(key=lambda item: item[0])
    paths_in_order = [path for _, path in candidates]

    if verbose:
        print(f"Found {len(paths_in_order)} images.")
        print(f"First frame: {os.path.basename(paths_in_order[0])}")
        print(f"Last frame:  {os.path.basename(paths_in_order[-1])}")

    first_img = plt.imread(paths_in_order[0])
    height, width = first_img.shape[0], first_img.shape[1]

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    image_artist = ax.imshow(first_img)

    def update(frame_idx):
        img = plt.imread(paths_in_order[frame_idx])
        image_artist.set_data(img)
        return [image_artist]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(paths_in_order),
        blit=True,
    )

    if out_name is None:
        label_part = "" if label == "final" else f"_{label}"
        out_name = (
            f"traj_t{t_star_steps}dt{selected_x_tag}"
            f"{label_part}_training_progress.mp4"
        )

    out_path = os.path.join(img_dir, out_name)

    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"Saved training progress video: {out_path}")

    return out_path
