"""
paper_fig_comparison.py
=======================

Paper figure: three-panel comparison of point-placement strategies for
predicting u(x*, t*) of the 1-D diffusion problem

        u_t = alpha u_xx ,   u(x,0) = sin(pi x) ,   u(0,t)=u(1,t)=0 ,

    (a) uniform finite differences (FTCS),
    (b) masked-random walkers,
    (c) the learned MaskablePPO policy.

All three panels share axes and are annotated with the number of solution
evaluations spent and the resulting error at z* = (x*, t*), so the panels are
directly comparable.

FD cost.  FD_MODE = 'cone' (default) charges the FTCS domain of dependence of
z* only -- the pyramid |i - i*| <= t*_steps - j -- and draws the rest of the
marched grid faintly.  That is the strongest honest FD baseline: it is the
minimum work any explicit 3-point scheme must do for that one value.
FD_MODE = 'full' charges the whole marched grid instead; both counts are
printed either way.

The environment below is a stand-alone copy of the diffusion "bubble"
environment as it stood for the run

    runs7-24/t10_errtol0.001_werr1_wtime0.1_N50_c0_R2/seed4

i.e. |ACTIONS| = 3, observation = 2N+2, N = 50, R_bubble = 2, c_shape = 0.
Those two shapes are asserted against the checkpoint at load time, so if you
point RUN_DIR at a different run the script will tell you rather than silently
producing a wrong rollout.

Outputs (into FIG_DIR):
    fig_placement_comparison.pdf   <- vector, use this in LaTeX
    fig_placement_comparison.png   <- 600 dpi raster preview

Usage:
    python paper_fig_comparison.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import RegularGridInterpolator
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO


# ═══════════════════════════════════════════════════════════════════════════
# 0.  User settings
# ═══════════════════════════════════════════════════════════════════════════
ROOT     = os.path.dirname(os.path.abspath(__file__))
RUN_DIR  = os.path.join(ROOT, "runs7-24",
                        "t10_errtol0.001_werr1_wtime0.1_N50_c0_R2", "seed4")
MODEL    = os.path.join(RUN_DIR, "checkpoints", "sched_final.zip")
FIG_DIR  = os.path.join(ROOT, "Figures")

# environment constants for this run (encoded in the run-directory name)
T_STAR_STEPS = 10        # t* = 10 * dt
X_STAR_IDX   = 50        # x* = x_fd[50]
N_PARTICLES  = 50
R_BUBBLE     = 2
ERR_TOL      = 1e-3
W_ERR        = 1.0
W_TIME       = 0.1
C_SHAPE      = 0.0
K_MAX        = 2000

# None -> a fresh draw of the masked-random baseline on every run.
# Set to an int to freeze panel (b) and reproduce a figure exactly.
RANDOM_SEED  = None      # seed for the masked-random baseline rollout
ENV_SEED     = None      # seed passed to env.reset()

# 'cone' -> highlight only the FTCS domain of dependence of z* (the pyramid),
#           with the rest of the marched grid drawn faintly;
# 'full' -> charge and highlight the whole marched grid.
FD_MODE      = 'cone'

# which panels get the bubble zoom inset: subset of {'a', 'b', 'c'}
INSET_PANELS = {'c'}
SHOW_STARTS  = True      # walker initial positions (triangles at t = 0)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PDE, reference grid, analytic solution
# ═══════════════════════════════════════════════════════════════════════════
alpha = 0.05

nx    = 100
T     = 1
x_fd  = np.linspace(0.0, 1.0, nx)
dx    = x_fd[1] - x_fd[0]

dt_cfl = 0.45 * dx**2 / alpha
nt     = max(int(np.ceil(T / dt_cfl)) + 1, 10)
t_fd   = np.linspace(0.0, T, nt)
dt     = t_fd[1] - t_fd[0]


def u_true(x, t):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def build_fd_reference():
    """FTCS march on the full uniform grid. Returns (u_grid, interpolator)."""
    u = np.zeros((nx, nt))
    u[:, 0] = u_true(x_fd, 0.0)
    r_diff = alpha * dt / dx**2
    for it in range(nt - 1):
        un = u[:, it]
        u[1:-1, it + 1] = un[1:-1] + r_diff * (un[2:] - 2 * un[1:-1] + un[:-2])
        u[0,  it + 1] = 0.0
        u[-1, it + 1] = 0.0
    interp = RegularGridInterpolator((x_fd, t_fd), u, method='linear',
                                     bounds_error=False, fill_value=None)
    return u, interp


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Visited set + GFDM weights  (verbatim from the run)
# ═══════════════════════════════════════════════════════════════════════════
class VisitedSet:
    def __init__(self, cap0=8192):
        self._cap = cap0
        self._n   = 0
        self._store     = np.empty((cap0, 3), dtype=float)   # (x, t, u)
        self._normstore = np.empty((cap0, 2), dtype=float)   # (x/dx, t/dt)
        self._idx_set   = set()
        self.rejected_history = []

    def _grow(self):
        self._cap *= 2
        s  = np.empty((self._cap, 3)); s[:self._n]  = self._store[:self._n]
        ns = np.empty((self._cap, 2)); ns[:self._n] = self._normstore[:self._n]
        self._store, self._normstore = s, ns

    def add(self, x, t, u):
        if self._n == self._cap:
            self._grow()
        self._store[self._n]     = (x, t, u)
        self._normstore[self._n] = (x / dx, t / dt)
        self._idx_set.add((round(x / dx), round(t / dt)))
        self._n += 1

    @property
    def _pts(self):
        return self._store[:self._n]

    def contains(self, x, t):
        return (round(x / dx), round(t / dt)) in self._idx_set

    def find_neighbours(self, x_star, t_star, n, causal=True, box_R=None):
        if self._n < n:
            return None
        pts_arr  = self._store[:self._n]
        norm_arr = self._normstore[:self._n]

        mask = (pts_arr[:, 1] <= t_star) if causal else np.ones(self._n, bool)

        same_cell = (np.round(pts_arr[:, 0] / dx) == round(x_star / dx)) & \
                    (np.round(pts_arr[:, 1] / dt) == round(t_star / dt))
        mask &= ~same_cell

        if box_R is not None:
            in_box = (np.abs(pts_arr[:, 0] - x_star) <= box_R * dx + 1e-9) & \
                     (np.abs(pts_arr[:, 1] - t_star) <= box_R * dt + 1e-9)
            mask &= in_box

        if mask.sum() < n:
            return None
        norm_c = norm_arr[mask]
        pts_c  = pts_arr[mask]
        target = np.array([x_star / dx, t_star / dt])
        d2  = np.sum((norm_c - target) ** 2, axis=1)
        idx = np.argpartition(d2, n - 1)[:n]
        idx = idx[np.argsort(d2[idx])]
        return [tuple(row) for row in pts_c[idx]]

    def count_in_box(self, x_star, t_star, R):
        ci, cj = round(x_star / dx), round(t_star / dt)
        cnt = 0
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                if di == 0 and dj == 0:
                    continue
                if (ci + di, cj + dj) in self._idx_set:
                    cnt += 1
        return cnt

    def __len__(self):
        return self._n


def solve_weights(dx_i, dt_i):
    """PDE-substituted Taylor moment conditions, minimum-norm solution."""
    h = max(np.max(np.abs(dx_i)), np.sqrt(alpha * np.max(np.abs(dt_i))), 1e-12)
    A = np.array([
        np.ones(len(dx_i)),
        dx_i / h,
        (0.5 * dx_i**2 + alpha * dt_i) / h**2,
    ])
    b = np.array([1.0, 0.0, 0.0])
    if np.linalg.cond(A) > 1e4:
        return None, 'cond'
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w, 'ok'


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Environment  (3-action version used by the runs7-24 checkpoints)
# ═══════════════════════════════════════════════════════════════════════════
class PDEEnvironment(gym.Env):
    ACTIONS = {0: (-1, 1),
               1: ( 0, 1),
               2: ( 1, 1)}

    def __init__(self, t_star_schedule=None, t_star_steps=2, N_particles=27,
                 err_tol=1e-3, w_err=0.8, w_time=0.2, K_max=5000,
                 c_shape=0.05, x_star_schedule=None, R_bubble=4):
        super().__init__()
        self.alpha    = alpha
        self.c_shape  = c_shape
        self.ERR_CEIL = 2.2
        self.n_neighbours = 5
        self.R_bubble = R_bubble
        self.nx, self.nt = nx, nt
        self.x_fd, self.t_fd = x_fd, t_fd
        self.dx, self.dt, self.T = dx, dt, T

        self.t_star_schedule = dict(t_star_schedule) if t_star_schedule else None

        if x_star_schedule is not None:
            self.x_star_choices = np.array(
                [self.x_fd[int(ix)] for ix in x_star_schedule], dtype=float)
        else:
            self.x_star_choices = None

        self.x_star = self.x_fd[self.nx // 2]
        self.t_star_steps = t_star_steps
        self.t_star = t_star_steps * dt

        self.K_max   = K_max
        self.err_tol = err_tol
        self.w_err   = w_err
        self.w_time  = w_time
        self.phi_scale = 1.0
        self.eps_x = 0.5 * dx
        self.eps_t = 0.5 * dt
        self.retry_after = 50

        # walker initial positions: frozen to domain centre, independent of x*
        self.N = N_particles
        mid, margin = self.nx // 2, 2
        half_n = self.N // 2
        step = (mid - margin) / half_n if half_n > 0 else 0
        offsets, k = [0], 1
        while len(offsets) < self.N:
            offsets.append(k * step)
            if len(offsets) < self.N:
                offsets.append(-k * step)
            k += 1
        init_idx = np.clip(np.round(np.array(offsets[:self.N]) + mid).astype(int),
                           margin, self.nx - 1 - margin)
        self.walker_init_x = self.x_fd[init_idx]

        _, self.fd_interp = build_fd_reference()

        self.visited = None
        self.walker_x = self.walker_t = self.walker_u = None
        self.step_count = 0
        self.stencil_log = {}

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * self.N + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.N * len(self.ACTIONS))

    # ── episode life-cycle ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.t_star_schedule is not None:
            self.t_star_steps = int(self.np_random.choice(
                list(self.t_star_schedule.keys())))
            self.t_star = self.t_star_steps * self.dt
            self.K_max  = self.t_star_schedule[self.t_star_steps]

        if self.x_star_choices is not None:
            self.x_star = float(self.np_random.choice(self.x_star_choices))

        self.visited = VisitedSet()
        self.stencil_log = {}

        for ix in range(self.nx):
            self.visited.add(self.x_fd[ix], 0.0, u_true(self.x_fd[ix], 0.0))
        for it in range(1, self.nt):
            self.visited.add(0.0, self.t_fd[it], 0.0)
            self.visited.add(1.0, self.t_fd[it], 0.0)

        self.walker_x = np.array(self.walker_init_x, dtype=float)
        self.walker_t = np.zeros(self.N)
        self.walker_u = np.array([u_true(x, 0.0) for x in self.walker_x])
        self.step_count = 0
        self.rejected_history = []
        self.gfdm_rejected_cells = {}
        self.phi_scale = max(self._mean_target_distance(), 1e-12)
        self._phi_prev = self._potential()
        return self._build_obs(), {}

    def _build_obs(self):
        return np.array([
            *((self.walker_x - self.x_star) / self.dx),
            *((self.walker_t - self.t_star) / self.dt),
            self.t_star / self.dt,
            self.x_star / self.dx,
        ], dtype=np.float32)

    def _mean_target_distance(self):
        dx_rel = (self.walker_x - self.x_star) / self.dx
        dt_rel = (self.walker_t - self.t_star) / self.dt
        return float(np.min(np.sqrt(dx_rel**2 + dt_rel**2)))

    def _potential(self):
        if self.c_shape == 0.0:
            return 0.0
        return -self.c_shape * self._mean_target_distance() / max(self.phi_scale, 1e-12)

    def _gfdm_solve(self, x_star, t_star, causal=True, box_R=None):
        nb = self.visited.find_neighbours(x_star, t_star, self.n_neighbours,
                                          causal=causal, box_R=box_R)
        if nb is None:
            return None, False, None, 'no_nb'
        nb = np.array(nb)
        w, reason = solve_weights(nb[:, 0] - x_star, nb[:, 1] - t_star)
        if w is None:
            return None, False, None, reason
        u_hat = float(w @ nb[:, 2])
        if abs(u_hat) > 1.1:
            return None, False, None, 'bounds_u'
        return u_hat, True, nb[:, :2], 'ok'

    def step(self, action):
        self.step_count += 1
        done, reward = False, 0.0

        n_act = len(self.ACTIONS)
        i, j = int(action) // n_act, int(action) % n_act
        dx_idx, dt_idx = self.ACTIONS[j]

        info = {'walker': i, 'accepted': False, 'rejected': None, 'reached': False}

        x_new = self.walker_x[i] + dx_idx * self.dx
        t_new = self.walker_t[i] + dt_idx * self.dt
        t_ceil = self.t_star + self.R_bubble * self.dt + self.eps_t

        if not (0.0 <= x_new <= 1.0 and 0.0 < t_new <= t_ceil):
            info['rejected'] = 'bounds'
        elif self.visited.contains(x_new, t_new):
            info['rejected'] = 'collision'
        else:
            u_hat, ok, nb_coords, reason = self._gfdm_solve(x_new, t_new)
            if not ok:
                info['rejected'] = reason
                self.gfdm_rejected_cells[
                    (round(x_new / self.dx), round(t_new / self.dt))] = len(self.visited)
            else:
                self.visited.add(x_new, t_new, u_hat)
                self.walker_x[i], self.walker_t[i], self.walker_u[i] = x_new, t_new, u_hat
                self.stencil_log[(round(x_new, 8), round(t_new, 8))] = nb_coords
                info['accepted'] = True

        if not info['accepted']:
            self.rejected_history.append((x_new, t_new, info['rejected']))

        bubble_full = (self.visited.count_in_box(self.x_star, self.t_star,
                                                 self.R_bubble) >= self.n_neighbours)
        timeout = self.step_count >= self.K_max

        if bubble_full or timeout:
            done = True
            info['reached'] = bool(bubble_full)
            info['timeout'] = bool(timeout and not bubble_full)
            box_R = self.R_bubble if bubble_full else None
            u_pred, _, _, _ = self._gfdm_solve(self.x_star, self.t_star,
                                               causal=False, box_R=box_R)
            u_ref = u_true(self.x_star, self.t_star)
            if u_pred is None:
                err, err_norm = None, 1.0
            else:
                err = abs(u_pred - u_ref)
                err_norm = float(np.clip(
                    np.log10(max(err, 1e-12) / self.err_tol) / 3.0, 0.0, 1.0))
            time_norm = self.step_count / max(self.K_max, 1)
            reward = -self.w_err * err_norm - self.w_time * time_norm
            info.update(u_pred=u_pred, u_ref=u_ref, error=err,
                        k_term=self.step_count, err_norm=err_norm,
                        time_norm=time_norm)

        phi_new = 0.0 if done else self._potential()
        reward += phi_new - self._phi_prev
        self._phi_prev = phi_new
        return self._build_obs(), reward, done, False, info

    def action_masks(self):
        n_act = len(self.ACTIONS)
        adx = np.fromiter((self.ACTIONS[j][0] for j in range(n_act)), int, n_act)
        adt = np.fromiter((self.ACTIONS[j][1] for j in range(n_act)), int, n_act)

        x_new = self.walker_x[:, None] + adx[None, :] * self.dx
        t_new = self.walker_t[:, None] + adt[None, :] * self.dt
        t_ceil = self.t_star + self.R_bubble * self.dt + self.eps_t

        ok = ((x_new >= 0.0) & (x_new <= 1.0) &
              (t_new > 0.0) & (t_new <= t_ceil)).ravel()

        cx = np.round(x_new / self.dx).astype(np.int64).ravel()
        ct = np.round(t_new / self.dt).astype(np.int64).ravel()

        n_pts   = len(self.visited)
        idx_set = self.visited._idx_set
        rej     = self.gfdm_rejected_cells

        for k in range(self.N * n_act):
            if not ok[k]:
                continue
            key = (int(cx[k]), int(ct[k]))
            if key in idx_set:
                ok[k] = False
                continue
            n_fail = rej.get(key)
            if n_fail is not None and n_pts < n_fail + self.retry_after:
                ok[k] = False

        if not ok.any():
            ok[:] = True
        return ok


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Rollouts
# ═══════════════════════════════════════════════════════════════════════════
def make_env():
    return PDEEnvironment(t_star_steps=T_STAR_STEPS, N_particles=N_PARTICLES,
                          err_tol=ERR_TOL, w_err=W_ERR, w_time=W_TIME,
                          K_max=K_MAX, c_shape=C_SHAPE,
                          x_star_schedule=[X_STAR_IDX], R_bubble=R_BUBBLE)


def rollout(policy, seed=ENV_SEED):
    """policy(obs, mask) -> action. Returns a dict of everything the figure needs."""
    env = make_env()
    obs, _ = env.reset(seed=seed)
    trajectories = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}
    done, info = False, {}
    while not done:
        mask = env.action_masks()
        obs, reward, done, _, info = env.step(policy(obs, mask))
        if info['accepted']:
            i = info['walker']
            trajectories[i].append((env.walker_x[i], env.walker_t[i]))

    reached = info.get('reached', False)
    _, s_ok, nb_coords, _ = env._gfdm_solve(
        env.x_star, env.t_star, causal=False,
        box_R=env.R_bubble if reached else None)

    # evaluations = points added on top of the seeded IC row + boundary columns
    n_seed = env.nx + 2 * (env.nt - 1)
    return dict(env=env, trajectories=trajectories, info=info, reached=reached,
                stencil=nb_coords if s_ok else None,
                n_eval=len(env.visited) - n_seed,
                error=info.get('error'), k_term=info.get('k_term', env.step_count))


def fd_cone_mask():
    """(nx, n_rows) boolean: nodes in the FTCS domain of dependence of z*.

    A node (i, j) with 1 <= j <= t*_steps feeds u(x*, t*) through the 3-point
    stencil iff |i - i*| <= t*_steps - j. Row j = 0 is the initial condition
    and the two boundary columns are given data, so neither is charged.
    """
    n_rows = min(nt, T_STAR_STEPS + R_BUBBLE + 4)
    cone   = np.zeros((nx, n_rows), dtype=bool)
    ii     = np.arange(nx)
    for j in range(0, T_STAR_STEPS + 1):
        cone[np.abs(ii - X_STAR_IDX) <= T_STAR_STEPS - j, j] = True
    given = np.zeros_like(cone)
    given[:, 0]  = True          # initial condition
    given[0, :]  = True          # boundaries
    given[-1, :] = True
    return cone, given, n_rows


def fd_result():
    """Uniform FTCS to t*: cost, error and the 3-point stencil at z*."""
    u_grid, _ = build_fd_reference()
    x_star, t_star = x_fd[X_STAR_IDX], T_STAR_STEPS * dt
    u_pred = u_grid[X_STAR_IDX, T_STAR_STEPS]
    err = abs(u_pred - u_true(x_star, t_star))

    cone, given, _ = fd_cone_mask()
    n_cone = int((cone & ~given).sum())          # minimal FD work for z* alone
    n_full = (nx - 2) * T_STAR_STEPS             # what a solver actually marches

    stencil = np.array([[x_fd[X_STAR_IDX + d], t_fd[T_STAR_STEPS - 1]]
                        for d in (-1, 0, 1)])
    return dict(x_star=x_star, t_star=t_star, u_pred=u_pred, error=err,
                n_eval=n_cone if FD_MODE == 'cone' else n_full,
                n_cone=n_cone, n_full=n_full, stencil=stencil)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Plot style
# ═══════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.labelsize":     9.5,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "lines.antialiased":  True,
    "figure.dpi":         160,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,      # embed TrueType, keeps text selectable
    "ps.fonttype":        42,
})

C_IDLE    = "#dfe3e8"   # grid nodes a solver marches but z* does not need
C_GRID    = "#b8bfc7"   # initial / boundary data (given, not charged)
C_EVAL    = "#3b4a5a"   # evaluated points of the active method
C_BUBBLE  = "#d1495b"   # bubble box / z*
C_STENCIL = "#d1495b"   # stencil legs
C_TRAJ    = plt.cm.viridis


def draw_panel(ax, kind, res, xlim, ylim, traj_lw=1.0, pt_s=9.0, show_starts=True):
    """kind in {'fd', 'rl'}."""
    if kind == 'fd':
        x_star, t_star = res['x_star'], res['t_star']
        cone, given, n_rows = fd_cone_mask()
        XX, TT = np.meshgrid(x_fd, t_fd[:n_rows], indexing='ij')

        if FD_MODE == 'cone':
            charged = cone & ~given                 # what z* actually costs
            idle    = (~cone) & (~given) & (TT <= t_star + 1e-12)
        else:
            charged = (TT <= t_star + 1e-12) & ~given
            idle    = np.zeros_like(charged)

        ax.scatter(XX[idle], TT[idle], s=pt_s * 0.55, color=C_IDLE,
                   linewidths=0, zorder=1, rasterized=True)
        ax.scatter(XX[given], TT[given], s=pt_s * 0.55, color=C_GRID,
                   linewidths=0, zorder=2, rasterized=True)
        ax.scatter(XX[charged], TT[charged], s=pt_s * 0.55, color=C_EVAL,
                   linewidths=0, zorder=3, rasterized=True)
        stencil = res['stencil']
    else:
        env  = res['env']
        traj = res['trajectories']
        x_star, t_star = env.x_star, env.t_star

        pts = env.visited._pts
        keep = (pts[:, 1] <= ylim[1]) & (pts[:, 1] >= ylim[0])
        ax.scatter(pts[keep, 0], pts[keep, 1], s=pt_s * 0.55, color=C_GRID,
                   linewidths=0, zorder=1, rasterized=True)

        colors = C_TRAJ(np.linspace(0.05, 0.95, env.N))
        for i, tr in traj.items():
            xs = [p[0] for p in tr]; ts = [p[1] for p in tr]
            if len(xs) > 1:
                ax.plot(xs, ts, '-', lw=traj_lw, color=colors[i], alpha=0.85,
                        zorder=3, solid_capstyle='round')
                ax.scatter(xs[1:], ts[1:], s=pt_s, color=colors[i],
                           linewidths=0, zorder=4)
        if show_starts:
            ax.scatter(env.walker_init_x, np.zeros(env.N), s=11, marker='^',
                       color=colors, linewidths=0, zorder=5)
        stencil = res['stencil']

    # bubble box
    Rx, Rt = R_BUBBLE * dx, R_BUBBLE * dt
    ax.add_patch(patches.Rectangle((x_star - Rx, t_star - Rt), 2 * Rx, 2 * Rt,
                                   fill=False, edgecolor=C_BUBBLE, ls=(0, (4, 2.5)),
                                   lw=1.0, zorder=6))
    # stencil legs + nodes
    if stencil is not None:
        for (xn, tn) in stencil:
            ax.plot([x_star, xn], [t_star, tn], ls=(0, (2.5, 1.8)),
                    color=C_STENCIL, lw=0.9, alpha=0.9, zorder=7)
        ax.scatter(stencil[:, 0], stencil[:, 1], s=18, color=C_STENCIL,
                   linewidths=0, zorder=8)
    # z*
    ax.scatter([x_star], [t_star], s=115, marker='*', color=C_BUBBLE,
               edgecolors='white', linewidths=0.5, zorder=9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x$")


def add_inset(ax, kind, res, pad=R_BUBBLE + 2):
    x_star = res['x_star'] if kind == 'fd' else res['env'].x_star
    t_star = res['t_star'] if kind == 'fd' else res['env'].t_star
    zx = (x_star - pad * dx, x_star + pad * dx)
    zt = (t_star - pad * dt, t_star + pad * dt)

    axin = ax.inset_axes([0.035, 0.60, 0.42, 0.375], xlim=zx, ylim=zt,
                         xticks=[], yticks=[], zorder=10)
    draw_panel(axin, kind, res, xlim=zx, ylim=zt,
               traj_lw=1.0, pt_s=11.0, show_starts=False)
    axin.set_xlabel("")
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_facecolor("white")
    axin.patch.set_alpha(1.0)
    for s in axin.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#5a6570")

    ax.indicate_inset_zoom(axin, edgecolor="#5a6570", lw=0.55,
                           linestyle=(0, (2, 2)), alpha=0.75)
    return axin


def fmt_err(e):
    if e is None:
        return "failed"
    m, ex = f"{e:.1e}".split("e")
    return rf"${m}\times 10^{{{int(ex)}}}$"


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Build the figure
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── load the trained policy, and check it matches this environment ─────
    probe = make_env()
    model = MaskablePPO.load(MODEL, device="cpu")
    assert model.observation_space.shape == probe.observation_space.shape, (
        f"checkpoint obs {model.observation_space.shape} != env "
        f"{probe.observation_space.shape}; check N_PARTICLES / obs layout")
    assert model.action_space.n == probe.action_space.n, (
        f"checkpoint action space {model.action_space.n} != env "
        f"{probe.action_space.n}; check N_PARTICLES / len(ACTIONS)")

    rng = np.random.default_rng(RANDOM_SEED)

    def pi_random(obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))

    def pi_learned(obs, mask):
        a, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return int(a)

    res_fd  = fd_result()
    res_rnd = rollout(pi_random)
    res_rl  = rollout(pi_learned)

    for tag, r in (("random", res_rnd), ("learned", res_rl)):
        print(f"{tag:8s}  reached={r['reached']}  k={r['k_term']}  "
              f"evals={r['n_eval']}  err={r['error']}")
    print(f"{'FD':8s}  cone={res_fd['n_cone']}  full grid={res_fd['n_full']}  "
          f"err={res_fd['error']:.3e}   (charging: {FD_MODE})")

    # ── shared axes ────────────────────────────────────────────────────────
    t_star = T_STAR_STEPS * dt
    xlim = (-0.012, 1.012)
    ylim = (-0.9 * dt, t_star + (R_BUBBLE + 3.2) * dt)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.05), sharey=True,
                             constrained_layout=True)

    panels = [
        (axes[0], 'a', 'fd', res_fd,  "(a) uniform finite differences"),
        (axes[1], 'b', 'rl', res_rnd, "(b) masked-random walkers"),
        (axes[2], 'c', 'rl', res_rl,  "(c) learned policy"),
    ]

    for ax, key, kind, res, title in panels:
        draw_panel(ax, kind, res, xlim, ylim, show_starts=SHOW_STARTS)
        ax.set_title(title, fontsize=9, pad=13.5)
        # a rollout that hit K_max never filled the bubble: its stencil at z*
        # comes from global nearest neighbours, not from inside the box, so the
        # panel must say so rather than pass as a successful run
        tag = "" if res.get('reached', True) else r" $\cdot$ timed out"
        ax.text(0.5, 1.012,
                rf"${res['n_eval']:,}$ evaluations $\cdot$ error "
                + fmt_err(res['error']) + tag,
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7.6,
                color="#3a3f45")
        if key in INSET_PANELS:
            add_inset(ax, kind, res)

    # y in units of dt: the horizon (t* = 10 dt) and the bubble (R = 2) stay legible
    axes[0].set_ylabel(r"$t/\Delta t$")
    for ax in axes:
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks(np.arange(0, T_STAR_STEPS + R_BUBBLE + 1, 2) * dt)
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _p: f"{v / dt:.0f}"))
        ax.tick_params(length=3, width=0.8)
        for s in ax.spines.values():
            s.set_color("#3a3f45")

    # shared legend
    mid = C_TRAJ(0.45)
    handles = [
        plt.Line2D([], [], ls='none', marker='o', ms=3.0, color=C_GRID,
                   label='initial / boundary data'),
        plt.Line2D([], [], ls='none', marker='o', ms=3.0, color=C_IDLE,
                   label='grid node not needed for ' + r'$z^\ast$'),
        plt.Line2D([], [], ls='none', marker='o', ms=3.0, color=C_EVAL,
                   label='evaluated point'),
        plt.Line2D([], [], ls='-', lw=1.1, marker='o', ms=3.0, color=mid,
                   label='walker path'),
        plt.Line2D([], [], ls='none', marker='^', ms=4.0, color=mid,
                   label='walker start'),
        plt.Line2D([], [], ls=(0, (4, 2.5)), lw=1.0, color=C_BUBBLE,
                   label=rf'bubble, $R={R_BUBBLE}$'),
        plt.Line2D([], [], ls=(0, (2.5, 1.8)), lw=0.9, color=C_STENCIL,
                   marker='o', ms=3.0, label=r'stencil at $z^\ast$'),
        plt.Line2D([], [], ls='none', marker='*', ms=8, color=C_BUBBLE,
                   label=r'$z^\ast=(x^\ast,t^\ast)$'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=4,
               frameon=False, handletextpad=0.5, columnspacing=1.6,
               labelspacing=0.5, borderaxespad=0.0)

    out_pdf = os.path.join(FIG_DIR, "fig_placement_comparison.pdf")
    out_png = os.path.join(FIG_DIR, "fig_placement_comparison.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
