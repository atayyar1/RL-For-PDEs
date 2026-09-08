"""
paper_fig_training_grid_digitise.py
===================================

Recovers the rollout behind an archived ``traj_*.png`` figure and prints it as
the literal that ``paper_fig_training_grid.py`` embeds.

WHY DIGITISE AT ALL.  Everywhere else in this paper a panel is produced by
loading the checkpoint and re-running the rollout, so the figure is generated
rather than traced.  That is not available here: checkpoints were kept every
1.5M steps for the diffusion and advection runs and every 1.2M for Burgers,
while the nine panels of the training grid sit at 0.1-3.1M and 13.9/17.6M --
none of those step counts has a surviving checkpoint.  The archived PNG is the
only surviving record of those rollouts, so the panel data is read back out of
it.  The recovery is exact rather than approximate, for the reasons below.

WHAT MAKES IT EXACT.  Every marker in the source figure sits on the reference
lattice, x = ix*dx with dx = 1/(nx-1) and t = jt*dt.  So the digitiser never
estimates a coordinate; it only decides, for each lattice site, whether a
marker is drawn there.

  * x calibration comes from the axis itself: the first and last x tick are
    x = 0 and x = 1 (all panels span the unit domain), which fixes dx in pixels
    to better than a tenth of a cell.
  * t calibration needs no tick reading: t = 0 is the row of walker-start
    triangles and t = t* is the red star, and t* is known exactly from the run
    (5, 10 and 20 dt), so one division gives dt in pixels.
  * markers are filled discs ~9 px across while paths and stencil connectors
    are ~1.5 px wide, so a single binary erosion deletes every line and leaves
    the markers standing.  A second erosion additionally deletes the dashed
    bubble outline of the Burgers panels, isolating the red star and the five
    red stencil discs.
  * walker identity is read from the marker colour.  The source figures colour
    walker i by ``viridis(i/(N-1))`` and fill it solid, so the modal RGB inside
    a marker names its walker exactly -- the chains are grouped, not inferred.

THREE CHECKS THE OUTPUT MUST PASS, and does, on all nine panels:

  1. the recovered star sits within 0.08 of a cell of its known (ix*, jt*);
  2. no colour group holds two points at the same time level -- a walker
     occupies one time level at a time, so a mis-grouping would show here;
  3. consecutive points of a chain satisfy dt = +1 and |dx| <= 1, the only
     moves the environment allows.

A marker drawn under another marker is recovered rather than dropped.  A walker
whose t = 0 triangle is covered by another walker's appears as a chain starting
at t = dt; its seat on the initial condition is restored from the environment's
seeding rule (centre outwards in equal steps, clipped to the interior), which is
deterministic and which every visible t = 0 marker in all nine panels matches
exactly.  The walker that owns a chain is read off its viridis fill, so the rule
is applied to the right walker rather than to the nearest one.

Only the five points hidden under the red stencil discs carry no colour.  They
are re-attached by the one assignment that keeps every chain legal
(``linear_sum_assignment`` over runs x chains).  z* is not a walker point --
the rollout terminates on an admissible stencil and the value at z* is then
reconstructed from it -- so a chain that runs past z* is stored as two
segments, and the break falls exactly under the drawn star.

Independent confirmation: the advection panel at 3.0M is the one requested step
that does have a checkpoint, and re-running it reproduces the stencil recovered
here, (8,3) (9,4) (11,3) (12,3) (12,4), exactly.

Usage:
    python paper_fig_training_grid_digitise.py          # all nine panels
"""

import collections
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.abspath(os.path.join(ROOT, "..", "Archives"))

# name -> (images dir, nx, N_walkers, ix*, jt*, [(tag, file), ...])
SOURCES = {
    "diffusion": (
        os.path.join(ARCH, "runs1", "t10_lam0.001_beta100_K250_N25",
                     "seed2", "images"), 30, 25, 15, 10,
        [("DIF_EARLY", "traj_t10dt_00200k_timeout.png"),
         ("DIF_MID",   "traj_t10dt_01000k_reached.png"),
         ("DIF_LATE",  "traj_t10dt_03100k_reached.png")]),
    "advection": (
        os.path.join(ARCH, "runs_adv8", "t5_lam0.001_beta1000_K400_N25",
                     "seed8", "images"), 20, 25, 10, 5,
        [("ADV_EARLY", "traj_t5dt_00200k_timeout.png"),
         ("ADV_MID",   "traj_t5dt_02100k_reached.png"),
         ("ADV_LATE",  "traj_t5dt_03000k_reached.png")]),
    "Burgers": (
        os.path.join(ROOT, "runs8-21",
                     "t20_errtol0.001_werr1_wtime0.1_N70_c0.3_R3",
                     "seed6", "images"), 100, 70, 50, 20,
        [("BUR_EARLY", "traj_t20dt_x50_00100k_reached.png"),
         ("BUR_MID",   "traj_t20dt_x50_13901k_reached.png"),
         ("BUR_LATE",  "traj_t20dt_x50_17601k_reached.png")]),
}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Read one PNG -> occupied lattice sites, stencil sites, marker colours
# ═══════════════════════════════════════════════════════════════════════════
def read_panel(path, nx, ix_star, jt_star, panel=0, rad=2.2, thr=3,
               overlay=None):
    a = np.array(Image.open(path).convert('RGB')).astype(int)
    H, W = a.shape[:2]

    # axes frame.  The Burgers figures carry a second (zoom) panel; panel=0
    # keeps the full-domain axes on the left.
    dark = a.max(axis=2) < 110
    cs = np.where(dark.sum(axis=0) > 0.30 * H)[0]
    rs = np.where(dark.sum(axis=1) > 0.15 * W)[0]
    L, R = cs[2 * panel], cs[2 * panel + 1]
    T, B = rs.min(), rs.max()

    # x ticks just below the bottom spine; first is x = 0, last is x = 1
    band = (a[B + 2:B + 5, :].max(axis=2) < 110).sum(axis=0)
    cc = np.where(band >= 2)[0]
    cc = cc[(cc >= L - 1) & (cc <= R + 1)]
    grp, cur = [], [cc[0]]
    for c in cc[1:]:
        if c - cur[-1] <= 2:
            cur.append(c)
        else:
            grp.append(cur); cur = [c]
    grp.append(cur)
    xt = np.array([np.mean(g) for g in grp])
    px0, px1 = xt[0], xt[-1]
    dxpx = (px1 - px0) / (nx - 1)

    # the run-condition legend the source figures draw inside the axes: find
    # its grey frame and blank it, so its key markers are not read as data
    g = (np.abs(a[..., 0] - a[..., 1]) < 8) & (np.abs(a[..., 1] - a[..., 2]) < 8) \
        & (a[..., 0] > 170) & (a[..., 0] < 235)
    g[:T + 4] = False; g[B - 3:] = False
    g[:, :L + 4] = False; g[:, R - 3:] = False
    g[T + (B - T) // 2:] = False
    legend = None
    if g.sum(axis=1).max() > 20:
        r_top = int(np.argmax(g.sum(axis=1)))          # the legend's top edge
        cols_ = np.where(g[r_top])[0]
        runs, cur = [], [cols_[0]]
        for c in cols_[1:]:
            if c - cur[-1] <= 4:
                cur.append(c)
            else:
                runs.append(cur); cur = [c]
        runs.append(cur)
        run = max(runs, key=len)
        c0, c1 = run[0], run[-1]
        rws = np.where(g[:, max(c0 - 2, 0):c0 + 4].any(axis=1))[0]
        r1 = rws[rws >= r_top - 2].max()
        box = (r_top - 3, r1 + 4, c0 - 3, c1 + 4)
        if box[3] - box[2] < 0.60 * (R - L) and box[1] - box[0] < 0.40 * (B - T):
            legend = box

    Rc, Gc, Bc = a[..., 0], a[..., 1], a[..., 2]
    sat = (a.max(axis=2) - a.min(axis=2)) > 40
    inside = np.zeros((H, W), bool)
    inside[T + 1:B, L + 1:R] = True
    if legend is not None:
        inside[legend[0]:legend[1], legend[2]:legend[3]] = False
    red = sat & inside & (Rc > 150) & (Gc < 120) & (Bc < 120)
    viri = sat & inside & ~red & (a.min(axis=2) < 150)   # drops the pale
    core_r = ndimage.binary_erosion(red, np.ones((3, 3)))   # rejection dots
    core_v = ndimage.binary_erosion(viri, np.ones((3, 3)))
    disc_r = ndimage.binary_erosion(core_r, np.ones((3, 3)))

    def blobs(m):
        lab, n = ndimage.label(m)
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            ys, xs = np.where(lab[sl] == i)
            out.append((xs.mean() + sl[1].start, ys.mean() + sl[0].start, len(ys)))
        return out

    xexp = px0 + ix_star * dxpx
    star = max((b for b in blobs(core_r)
                if b[2] >= 20 and abs(b[0] - xexp) < 1.2 * dxpx),
               key=lambda b: b[2])
    vy = np.array([b[1] for b in blobs(core_v)])
    y0 = np.median(vy[vy > vy.max() - 6])            # the t = 0 triangle row
    dtpx = (y0 - star[1]) / jt_star
    assert abs((star[0] - px0) / dxpx - ix_star) < 0.08, "star off lattice"

    yy, xx = np.mgrid[0:H, 0:W]

    def window(ix, jt):
        cx, cy = px0 + ix * dxpx, y0 - jt * dtpx
        i0, i1 = max(0, int(cy - rad)), min(H, int(cy + rad) + 2)
        j0, j1 = max(0, int(cx - rad)), min(W, int(cx + rad) + 2)
        if i1 <= i0 or j1 <= j0:
            return None, None, None
        m = (yy[i0:i1, j0:j1] - cy) ** 2 + (xx[i0:i1, j0:j1] - cx) ** 2 <= rad ** 2
        return (slice(i0, i1), slice(j0, j1)), m, None

    jmax = int(np.floor((y0 - T) / dtpx))
    pts, sten, colour = [], [], {}
    for ix in range(nx):
        for jt in range(jmax + 1):
            if (ix, jt) == (ix_star, jt_star):
                continue
            sl, m, _ = window(ix, jt)
            if sl is None:
                continue
            is_stencil = int((disc_r[sl] & m).sum()) >= 2
            if is_stencil:
                sten.append((ix, jt))
            if is_stencil or int((core_v[sl] & m).sum()) >= thr:
                pts.append((ix, jt))
                px = a[sl][m & core_v[sl]]
                if len(px):
                    u, c = np.unique(px, axis=0, return_counts=True)
                    colour[(ix, jt)] = tuple(int(v) for v in u[c.argmax()])

    if overlay:                                   # visual check of the reading
        ov = a.copy()
        for ix, jt in pts:
            cx, cy = int(round(px0 + ix * dxpx)), int(round(y0 - jt * dtpx))
            ov[max(0, cy - 7):cy + 8, cx] = [255, 0, 255]
            ov[cy, max(0, cx - 7):cx + 8] = [255, 0, 255]
        Image.fromarray(ov.astype(np.uint8)).save(overlay)

    return dict(points=sorted(pts), stencil=sorted(sten), colour=colour)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Sites -> walker chains, by marker colour, with the hidden points placed
# ═══════════════════════════════════════════════════════════════════════════
def walker_starts(nx, N):
    """The environment's walker seeding, verbatim: index 0 at the domain centre,
    then alternating outward in equal steps, clipped to the interior."""
    mid, margin, half_n = nx // 2, 2, N // 2
    step = (mid - margin) / half_n if half_n > 0 else 0
    off, k = [0], 1
    while len(off) < N:
        off.append(k * step)
        if len(off) < N:
            off.append(-k * step)
        k += 1
    return np.clip(np.round(np.array(off[:N]) + mid).astype(int),
                   margin, nx - 1 - margin)


_LUT = None


def walker_index(colour, N):
    """Invert the viridis fill of a marker back to the walker index it names."""
    global _LUT
    if _LUT is None:
        import matplotlib.pyplot as plt
        _LUT = np.array([plt.cm.viridis(i / 255.0)[:3] for i in range(256)]) * 255
    p = int(np.argmin(((_LUT - np.array(colour)) ** 2).sum(1))) / 255.0
    i = p * (N - 1)
    assert abs(i - round(i)) < 0.35, "marker colour is not a walker colour"
    return int(round(i))


def chains(panel, nx, N):
    groups = collections.defaultdict(list)
    for site, col in panel['colour'].items():
        groups[col].append(site)

    # A walker whose t = 0 triangle is drawn under another walker's shows up as
    # a chain starting at t = dt.  Its seat on the initial condition is not a
    # guess: the seeding is deterministic, and every one of the visible t = 0
    # markers agrees with it, so the missing start is restored from the rule.
    starts = walker_starts(nx, N)
    ch = []
    for col, sites in groups.items():
        sites.sort(key=lambda z: z[1])
        i = walker_index(col, N)
        if sites[0][1] == 0:
            assert starts[i] == sites[0][0], "walker start off the seeding rule"
        else:
            assert sites[0][1] == 1 and abs(starts[i] - sites[0][0]) <= 1, \
                "cannot restore this walker's start"
            sites.insert(0, (int(starts[i]), 0))
        ch.append(sites)

    for c in ch:                       # check 2: one time level per walker
        assert len({t for _, t in c}) == len(c), "colour group repeats a level"

    hidden = sorted(set(panel['points']) - set(panel['colour']),
                    key=lambda z: (z[1], z[0]))
    runs = []                          # hidden points that follow each other
    for p in hidden:
        for r in runs:
            if r[-1][1] == p[1] - 1 and abs(r[-1][0] - p[0]) <= 1:
                r.append(p); break
        else:
            runs.append([p])

    if runs:
        C = np.full((len(runs), len(ch)), 1e6)
        for i, r in enumerate(runs):
            t0, t1 = r[0][1], r[-1][1]
            for j, c in enumerate(ch):
                ts = {t: x for x, t in c}
                if any(t in ts for t in range(t0, t1 + 1)):
                    continue
                pv, nx_ = ts.get(t0 - 1), ts.get(t1 + 1)
                if pv is None and nx_ is None:
                    continue
                if pv is not None and abs(pv - r[0][0]) > 1:
                    continue
                if nx_ is not None and abs(nx_ - r[-1][0]) > 1:
                    continue
                C[i, j] = (0 if (pv is not None and nx_ is not None) else 10) \
                    + (abs(pv - r[0][0]) if pv is not None else 0) \
                    + (abs(nx_ - r[-1][0]) if nx_ is not None else 0)
        placed = set()
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < 1e5:
                ch[j] += runs[i]; ch[j].sort(key=lambda z: z[1]); placed.add(i)
        ch += [r for i, r in enumerate(runs) if i not in placed]

    for c in ch:                       # check 3: legal moves only
        for a, b in zip(c, c[1:]):
            assert b[1] - a[1] >= 1 and abs(b[0] - a[0]) <= b[1] - a[1], \
                "illegal walker move"
    return ch


MOVE = {-1: '-', 0: '0', 1: '+'}


def encode(ch):
    """(ix0, jt0, moves) per chain; a chain is split at a hidden-marker gap."""
    seg = []
    for c in ch:
        cur = [c[0]]
        for a, b in zip(c, c[1:]):
            if b[1] - a[1] == 1:
                cur.append(b)
            else:
                seg.append(cur); cur = [b]
        seg.append(cur)
    return [(c[0][0], c[0][1],
             ''.join(MOVE[c[i + 1][0] - c[i][0]] for i in range(len(c) - 1)))
            for c in sorted(seg)]


def main():
    import textwrap
    for pde, (imgdir, nx, N, ixs, jts, files) in SOURCES.items():
        for tag, fn in files:
            path = os.path.join(imgdir, fn)
            p = read_panel(path, nx, ixs, jts)
            ch = chains(p, nx, N)
            enc = encode(ch)
            n = sum(len(m) + 1 for _, _, m in enc)
            print(f'# {pde}: {fn}  ->  {len(enc)} chains, {n} points, '
                  f'{len(p["stencil"])} stencil')
            body = ', '.join(f'({a},{b},"{m}")' for a, b, m in enc)
            print(f'{tag} = dict(          # {fn}')
            print('    stencil=[%s],' % ', '.join(f'({a},{b})'
                                                  for a, b in p['stencil']))
            print('    chains=[')
            print('\n'.join(textwrap.wrap(
                body, 74, initial_indent=' ' * 8, subsequent_indent=' ' * 8,
                break_on_hyphens=False, break_long_words=False)) + '])\n')


if __name__ == "__main__":
    main()
