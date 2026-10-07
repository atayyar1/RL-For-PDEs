"""
map_progress_video.py

Stitch the snapshot figures saved during training (images/map_*.png, then map_final.png)
into one MP4. Pure post-processing: no model, no environment.

    from map_progress_video import make_progress_video
    make_progress_video(img_dir, fps=2)
"""
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def _ffmpeg_writer(fps):
    """FFmpeg writer using the ffmpeg shipped with imageio-ffmpeg (pip install imageio-ffmpeg)."""
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    return animation.FFMpegWriter(fps=fps)


def make_progress_video(img_dir, fps=2, out_name="training_progress.mp4", dpi=100):
    """Snapshots in training order, the final map last. Returns the path of the video."""
    frames = sorted(glob.glob(os.path.join(img_dir, "map_[0-9]*.png")))
    final = os.path.join(img_dir, "map_final.png")
    if os.path.exists(final):
        frames.append(final)
    if not frames:
        raise FileNotFoundError(f"no map_*.png in {img_dir}")

    first = plt.imread(frames[0])
    h, w = first.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    im = ax.imshow(first)

    def update(k):
        img = plt.imread(frames[k])
        im.set_data(img[:h, :w])                # figures can differ by a pixel; crop to the first one
        return [im]

    path = os.path.join(img_dir, out_name)
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(path, writer=_ffmpeg_writer(fps), dpi=dpi)
    plt.close(fig)
    print(f"{len(frames)} frames: {os.path.basename(frames[0])} ... {os.path.basename(frames[-1])}")
    print("saved:", path)
    return path
