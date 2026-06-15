# %% [markdown]
# Libraries

# %%
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
import gymnasium as gym
from gymnasium import spaces

# %% [markdown]
# PDE and FD reference Parameters
# 

# %%
# ── PDE ────────────────────────────────────────────────────────
alpha = 0.05

# ── FD reference grid ──────────────────────────────────────────
nx    = 10
T     = 0.5
x_fd  = np.linspace(0.0, 1.0, nx)
dx    = x_fd[1] - x_fd[0]

dt_cfl = 0.45 * dx**2 / alpha
nt     = max(int(np.ceil(T / dt_cfl)) + 1, 10)
t_fd   = np.linspace(0.0, T, nt)
dt     = t_fd[1] - t_fd[0]

# %% [markdown]
# Analytical diffusion solution
# 

# %%
def u_true(x, t):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)

# %% [markdown]
# FD For diffusion

# %%
def build_fd_reference():
    u = np.zeros((nx, nt))
    u[:, 0] = u_true(x_fd, 0.0)

    r_diff = alpha * dt / dx**2

    for it in range(nt - 1):
        un = u[:, it]
        u[1:-1, it+1] = (un[1:-1]
                         + r_diff * (un[2:] - 2*un[1:-1] + un[:-2]))
        u[0,   it+1] = 0.0
        u[-1,  it+1] = 0.0

    interp = RegularGridInterpolator(
        (x_fd, t_fd), u, method='linear',
        bounds_error=False, fill_value=None
    )
    return u, interp

# %% [markdown]
# __init__

# %%
class VisitedSet:

    def __init__(self, rebuild_every=200):
        self._pts           = []
        self._norm          = []
        self._tree          = None
        self._dirty         = 0
        self._rebuild_every = rebuild_every

    def add(self, x, t, u):
        self._pts.append((x, t, u))
        self._norm.append((x / dx, t / dt))
        self._dirty += 1
        if self._dirty >= self._rebuild_every:
            self._rebuild()

    def _rebuild(self):
        self._tree  = KDTree(np.array(self._norm))
        self._dirty = 0

    def _ensure(self):
        if self._tree is None:
            self._rebuild()

    def find_neighbours(self, x_star, t_star, n):
        candidates = [(x, t, u) for (x, t, u) in self._pts
                    if 0 < (t_star - t)]
        
        if len(candidates) < n:
            return None
        
        pts     = np.array([(x, t) for (x, t, _) in candidates])
        tree    = KDTree(pts)
        _, idxs = tree.query([x_star, t_star], k=n)
        
        return [candidates[i] for i in idxs]
    def __len__(self):
        return len(self._pts)

# %% [markdown]
# GFDM Solver
# 

# %%
def solve_weights(dx_i, dt_i):
    """
    Pure diffusion constraint matrix.
    Returns (w, success).
    """
    h   = max(np.max(np.abs(dx_i)), 1e-12)

    A = np.array([
        np.ones(len(dx_i)),
        dx_i / h,
        (0.5 * dx_i**2 + alpha * dt_i) / h**2,
    ])
    b = np.array([1.0, 0.0, 0.0])

    cond = np.linalg.cond(A)
    if cond > 1e4:
        return None, False

    w = np.linalg.lstsq(A, b, rcond=None)[0]

    if np.max(np.abs(w)) > 3.0 / len(dx_i):
        return None, False
    if np.min(w) < -1.0:
        return None, False

    return w, True

# %% [markdown]
# PDEEnvironment

# %%
class PDEEnvironment(gym.Env):

    def __init__(self):
        super().__init__()

        # ── PDE constants ───────────────────────────────────────────────────────
        self.alpha = alpha

        # ── Grid variables ──────────────────────────────────────────────────────
        self.nx   = nx
        self.nt   = nt
        self.x_fd = x_fd
        self.t_fd = t_fd
        self.dx   = dx
        self.dt   = dt
        self.T    = T

        # ── Target point (z*) ───────────────────────────────────────────────────
        self.x_star = 0.5
        self.t_star = 1 * dt

        # ── RL parameters ───────────────────────────────────────────────────────
        self.N      = 5
        self.K_max  = 20
        self.lam    = 0.001
        self.Kx     = 1
        self.Kt     = 1

        # ── GFDM neighbours ─────────────────────────────────────────────────────
        self.n_neighbours = 5

        # ── Termination tolerance ───────────────────────────────────────────────
        self.eps_x = 0.5 * dx
        self.eps_t = 0.5 * dt

        # ── Walker initial positions ─────────────────────────────────────────────
        self.walker_init_x = [0.3, 0.4, 0.5, 0.6, 0.7]

        # ── FD reference built once ──────────────────────────────────────────────
        _, self.fd_interp = build_fd_reference()

        # ── Episode state (reset() fills these) ──────────────────────────────────
        self.visited        = None
        self.walker_x       = None
        self.walker_t       = None
        self.walker_u       = None
        self.step_count     = 0
        self.current_walker = 0          # ADDED: round-robin tracker
        self.stencil_log    = {}         # ADDED: log of stencils used
        # ── Gym spaces ───────────────────────────────────────────────────────────
        # CHANGED: single walker local obs (x_i, t_i, u_i, x*, t*) — 5 dims
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        # CHANGED: one action per step, not one per walker
        self.action_space = spaces.Discrete(6)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ── Fresh visited list ───────────────────────────────────────────────────
        self.visited = VisitedSet()
        self.stencil_log = {}  # Clear stencil log at reset

        # ── Seed with IC ─────────────────────────────────────────────────────────
        for ix in range(self.nx):
            x = self.x_fd[ix]
            self.visited.add(x, 0.0, u_true(x, 0.0))

        # ── Seed with BCs ────────────────────────────────────────────────────────
        for it in range(1, self.nt):
            self.visited.add(0.0, self.t_fd[it], 0.0)
            self.visited.add(1.0, self.t_fd[it], 0.0)

        # ── Place walkers at t=0 ─────────────────────────────────────────────────        
        shuffled      = np.random.permutation(self.walker_init_x)
        self.walker_t = np.zeros(self.N)
        self.walker_x = np.array(shuffled, dtype=float)
        self.walker_u = np.array([u_true(x, 0.0) for x in shuffled])

        # ── Reset counters ───────────────────────────────────────────────────────
        self.step_count     = 0
        self.current_walker = 0          # ADDED: start round-robin at walker 0

        return self._build_local_obs(0), {}


    def _build_local_obs(self, i):       # CHANGED: was _build_state(), now per-walker
        return np.array([
            self.walker_x[i],
            self.walker_t[i],
            self.walker_u[i],
            self.x_star,
            self.t_star
        ], dtype=np.float32)


    def _decode_action(self, action_idx):
        # dx ∈ {-1, 0, +1} · Δx  → 3 options
        # dt ∈ {0, 1}      · Δt  → 2 options
        # layout: action_idx = dx_idx * 2 + dt_idx
        # NOTE: action 2 (dx=0, dt=0) always rejected — no-op

        dx_idx = action_idx // 2
        dt_idx = action_idx % 2

        dx = (dx_idx - self.Kx) * self.dx
        dt = dt_idx * self.dt

        return dx, dt


    def _check_collision(self, x, t):
        for (xv, tv, _) in self.visited._pts:
            if abs(xv - x) < self.eps_x and abs(tv - t) < self.eps_t:
                return True
        return False




    def _gfdm_solve(self, x_star, t_star):
        nb = self.visited.find_neighbours(x_star, t_star, self.n_neighbours)
        if nb is None:
            return None, False, None

        nb   = np.array(nb)
        dx_i = nb[:, 0] - x_star
        dt_i = nb[:, 1] - t_star

        if not (np.min(dx_i) < 0 < np.max(dx_i)):
            return None, False, None

        w, success = solve_weights(dx_i, dt_i)
        if not success:
            return None, False, None

        u_hat = float(w @ nb[:, 2])
        return u_hat, True, nb[:, :2]
    def step(self, action):

        self.step_count += 1
        done   = False
        reward = 0.0
        i      = self.current_walker
        info   = {'walker': i, 'accepted': False, 'rejected': None, 'reached': False}

        dx, dt = self._decode_action(action)
        x_new  = self.walker_x[i] + dx
        t_new  = self.walker_t[i] + dt

        if not (0.0 <= x_new <= 1.0 and 0.0 < t_new <= self.T):
            info['rejected'] = 'bounds'

        elif self._check_collision(x_new, t_new):
            info['rejected'] = 'collision'

        else:
            u_hat, success, nb_coords = self._gfdm_solve(x_new, t_new)  # CHANGED: unpack 3
            if not success:
                info['rejected'] = 'gfdm'
            else:
                self.visited.add(x_new, t_new, u_hat)
                self.walker_x[i] = x_new
                self.walker_t[i] = t_new
                self.walker_u[i] = u_hat
                self.stencil_log[(round(x_new, 8), round(t_new, 8))] = nb_coords  # ADDED
                info['accepted'] = True

                if (abs(x_new - self.x_star) < self.eps_x and
                    abs(t_new - self.t_star) < self.eps_t):
                    done = True
                    info['reached'] = True

        if done:
            u_pred, _, _   = self._gfdm_solve(self.x_star, self.t_star)
            u_ref          = u_true(self.x_star, self.t_star)

            if u_pred is None:
                reward        = -1.0
                info['error'] = None
            else:
                reward         = -abs(u_pred - u_ref) - self.lam * self.step_count
                info['u_pred'] = u_pred
                info['u_ref']  = u_ref
                info['error']  = abs(u_pred - u_ref)

        elif self.step_count >= self.K_max:
            done   = True
            reward = -1.0
            info['timeout'] = True

        self.current_walker = (self.current_walker + 1) % self.N
        return self._build_local_obs(self.current_walker), reward, done, False, info



# %%
env      = PDEEnvironment()
state, _ = env.reset()

trajectories = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}

done           = False
total_accepted = 0
total_rejected = {}

while not done:
    action = np.random.randint(6)                              # single action
    state, reward, done, _, info = env.step(action)

    total_accepted += int(info['accepted'])                    # bool not list
    if info['rejected']:                                       # string not list of tuples
        reason = info['rejected']
        total_rejected[reason] = total_rejected.get(reason, 0) + 1

    if info['accepted']:                                       # bool not list
        i = info['walker']
        trajectories[i].append((env.walker_x[i], env.walker_t[i]))

visited_log = [(x, t) for (x, t, _) in env.visited._pts]

print(f"Steps:          {env.step_count}")
print(f"Reward:         {reward:.6f}")
print(f"Reached:        {info.get('reached', False)}")
print(f"Timeout:        {info.get('timeout', False)}")
print(f"Total accepted: {total_accepted}")
print(f"Rejections:     {total_rejected}")
print

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(8, 6))

vx = [p[0] for p in visited_log]
vt = [p[1] for p in visited_log]
ax.scatter(vx, vt, s=4, color='lightgray', zorder=1, label='Visited')

colors = cm.tab10.colors
for i, traj in trajectories.items():
    xs = [p[0] for p in traj]
    ts = [p[1] for p in traj]
    ax.plot(xs, ts, '-', color=colors[i], linewidth=1.5, zorder=2)
    ax.scatter(xs, ts, s=30, color=colors[i], zorder=3)
    ax.scatter(xs[0], ts[0], s=80, color=colors[i],
               marker='^', zorder=4, label=f'Walker {i}')

ax.scatter(env.x_star, env.t_star, s=150, color='red',
           marker='*', zorder=5, label='z*')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Walker trajectories in x-t domain')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, env.t_star * 2)
plt.tight_layout()
plt.show()

# %%
env.reset()
for step in range(15):
    obs, reward, done, _, info = env.step(3)
    if done:
        print(f"step {step+1} | reached={info.get('reached')} | reward={reward:.6f}")
        break

# %%
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=150_000)

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
obs, _ = env.reset()
trajectories = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if info['accepted']:
        i = info['walker']
        trajectories[i].append((env.walker_x[i], env.walker_t[i]))
        
visited_log = [(x, t) for (x, t, _) in env.visited._pts]
fig, ax = plt.subplots(figsize=(8, 6))

vx = [p[0] for p in visited_log]
vt = [p[1] for p in visited_log]
ax.scatter(vx, vt, s=4, color='lightgray', zorder=1, label='Visited')

colors = cm.tab10.colors
for i, traj in trajectories.items():
    xs = [p[0] for p in traj]
    ts = [p[1] for p in traj]
    ax.plot(xs, ts, '-', color=colors[i], linewidth=1.5, zorder=2)
    ax.scatter(xs, ts, s=30, color=colors[i], zorder=3)
    ax.scatter(xs[0], ts[0], s=80, color=colors[i],
               marker='^', zorder=4, label=f'Walker {i}')

ax.scatter(env.x_star, env.t_star, s=150, color='red',
           marker='*', zorder=5, label='z*')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Walker trajectories in x-t domain')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, env.t_star * 2)
plt.tight_layout()
plt.show()

# %%
%matplotlib widget

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Use logged stencils instead of post-hoc reconstruction ───────────────────
stencils = {key: coords for key, coords in env.stencil_log.items()}

# ── Build plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

vx = [p[0] for p in visited_log]
vt = [p[1] for p in visited_log]
ax.scatter(vx, vt, s=4, color='lightgray', zorder=1, label='Visited')

colors     = cm.tab10.colors
all_points = []

for i, traj in trajectories.items():
    xs = [p[0] for p in traj]
    ts = [p[1] for p in traj]
    ax.plot(xs, ts, '-', color=colors[i], linewidth=1.5, zorder=2)
    ax.scatter(xs[0], ts[0], s=80, color=colors[i], marker='^',
               zorder=4, label=f'Walker {i}')
    for (x, t) in traj[1:]:
        ax.scatter(x, t, s=60, color=colors[i], zorder=3)
        all_points.append((x, t, i))

ax.scatter(env.x_star, env.t_star, s=200, color='red',
           marker='*', zorder=5, label='z*')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Click any walker point to see its GFDM stencil')
ax.legend()
plt.tight_layout()

# ── Click handler ─────────────────────────────────────────────────────────────
active_lines = []

def on_click(event):
    global active_lines

    if event.inaxes != ax or not all_points:
        return

    cx, ct  = event.xdata, event.ydata
    best_pt = min(all_points,
                  key=lambda p: ((p[0]-cx)/env.dx)**2 + ((p[1]-ct)/env.dt)**2)
    x, t, i = best_pt
    key     = (round(x, 8), round(t, 8))

    for line in active_lines:
        line.remove()
    active_lines = []

    if key in stencils:
        for (nx, nt) in stencils[key]:
            line, = ax.plot([x, nx], [t, nt], '-',
                            color=colors[i], alpha=0.75,
                            linewidth=2, zorder=2)
            active_lines.append(line)

        dot, = ax.plot(x, t, 'o', color=colors[i],
                       markersize=10, markeredgecolor='black',
                       markeredgewidth=1.5, zorder=6)
        active_lines.append(dot)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()


