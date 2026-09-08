"""
paper_fig_beta_sweep.py
=======================

Paper figure: the accuracy/cost weight in the reward controls the stencil
geometry the policy discovers.

Three archived runs at t* = 10 dt, sweeping the accuracy weight beta of the
ORIGINAL reward

        R = -( beta * |e| + lambda * k )      on reach,
        R = -1 - lambda * k                   on timeout,

with lambda = 0.01 fixed:

    beta = 1e3   Archives/runs9/t10_lam0.01_beta1000_K500_N45/seed6
    beta = 1e4   Archives/runs9/t10_lam0.01_beta10000_K500_N45/seed6
    beta = 1e5   Archives/runs8/t10_lam0.01_beta100000_K500_N45/seed6

Reading beta in the current parametrisation.  One extra evaluation costs
lambda; it pays for itself only while it reduces |e| by more than lambda/beta.
The policy therefore drives the error down to

        eps_eff = lambda / beta

and stops -- exactly the role err_tol plays in the current reward, where
err_norm = clip(log10(|e|/err_tol)/3, 0, 1) saturates at 0 below err_tol.
So the sweep is a sweep over an effective error tolerance,
eps_eff = 1e-5, 1e-6, 1e-7.  This is a marginal-rate argument, not an
identity: the old reward is linear in |e|, the new one log-scaled and clipped.

ENVIRONMENT.  The nx = 50 / N = 45 / |A| = 6 variant of the environment these
runs used was never committed, so it is reconstructed here and VERIFIED: each
rollout reproduces the step count printed on the archived figure
(k = 13, 15, 11) and the stencil read off those images.  Reconstructed
settings: observation [walker_x, walker_t, walker_u, x*, t*] (3N+2, raw
coordinates), ACTIONS = {(-1,1),(-1,2),(0,1),(0,2),(1,1),(1,2)}, termination on
reaching z* within half a cell, no two-sided geometry test and no |w| caps in
the solve.

ERRORS.  The numbers printed on the archived PNGs were produced at different
times with different versions of solve_weights and are not mutually
consistent (two of the three disagree with the figures in `To Bakarji/`).
They are recomputed here with the CURRENT solver -- h = max(max|dx|,
sqrt(alpha max|dt|)), cond < 1e4, minimum-norm w -- so all three panels, and
the rest of the paper, use one definition.

Outputs into FIG_DIR:
    fig_beta_sweep.pdf   <- vector, use this in LaTeX
    fig_beta_sweep.png   <- 600 dpi raster preview

Usage:
    python paper_fig_beta_sweep.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

ROOT     = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(ROOT, "Figures")
ARCHIVES = os.path.abspath(os.path.join(ROOT, "..", "Archives"))

LAMBDA       = 0.01
T_STAR_STEPS = 10
N_PARTICLES  = 45
K_MAX        = 500

RUNS = [
    (1e3, os.path.join(ARCHIVES, "runs9", "t10_lam0.01_beta1000_K500_N45",
                       "seed6", "checkpoints", "tstar10_final.zip"), 13),
    (1e4, os.path.join(ARCHIVES, "runs9", "t10_lam0.01_beta10000_K500_N45",
                       "seed6", "checkpoints", "tstar10_final.zip"), 15),
    (1e5, os.path.join(ARCHIVES, "runs8", "t10_lam0.01_beta100000_K500_N45",
                       "seed6", "checkpoints", "tstar10_final_continued.zip"), 11),
]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Grid and exact solution  (the era's constants)
# ═══════════════════════════════════════════════════════════════════════════
alpha = 0.05
nx, T = 50, 1
x_fd  = np.linspace(0.0, 1.0, nx)
dx    = x_fd[1] - x_fd[0]
dt_cfl = 0.45 * dx**2 / alpha
nt    = max(int(np.ceil(T / dt_cfl)) + 1, 10)
t_fd  = np.linspace(0.0, T, nt)
dt    = t_fd[1] - t_fd[0]


def u_true(x, t):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Visited set + weights (current solver, applied uniformly)
# ═══════════════════════════════════════════════════════════════════════════
class VisitedSet:
    def __init__(self):
        self._pts, self._norm, self._idx = [], [], set()

    def add(self, x, t, u):
        self._idx.add((round(x / dx), round(t / dt)))
        self._pts.append((x, t, u))
        self._norm.append((x / dx, t / dt))

    def contains(self, x, t):
        return (round(x / dx), round(t / dt)) in self._idx

    def find_neighbours(self, xs, ts, n):
        if len(self._pts) < n:
            return None
        P = np.asarray(self._pts)
        Nn = np.asarray(self._norm)
        m = P[:, 1] < ts                       # strictly causal, as in the era
        if m.sum() < n:
            return None
        d2 = np.sum((Nn[m] - np.array([xs / dx, ts / dt]))**2, axis=1)
        idx = np.argpartition(d2, n - 1)[:n]
        idx = idx[np.argsort(d2[idx])]
        return [tuple(r) for r in P[m][idx]]

    def __len__(self):
        return len(self._pts)


def solve_weights(dxi, dti):
    h = max(np.max(np.abs(dxi)), np.sqrt(alpha * np.max(np.abs(dti))), 1e-12)
    A = np.array([np.ones(len(dxi)),
                  dxi / h,
                  (0.5 * dxi**2 + alpha * dti) / h**2])
    if np.linalg.cond(A) > 1e4:
        return None, 'cond'
    return np.linalg.lstsq(A, np.array([1.0, 0.0, 0.0]), rcond=None)[0], 'ok'


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Reconstructed environment (verified against k = 13, 15, 11)
# ═══════════════════════════════════════════════════════════════════════════
ACTIONS = {0: (-1, 1), 1: (-1, 2),
           2: ( 0, 1), 3: ( 0, 2),
           4: ( 1, 1), 5: ( 1, 2)}


class PDEEnvironment(gym.Env):
    def __init__(self, N=N_PARTICLES, t_star_steps=T_STAR_STEPS,
                 K_max=K_MAX, margin=2):
        super().__init__()
        self.A, self.nA = ACTIONS, len(ACTIONS)
        self.N, self.K_max = N, K_max
        self.dx, self.dt, self.T = dx, dt, T
        self.nx, self.nt = nx, nt
        self.x_fd, self.t_fd = x_fd, t_fd
        self.x_star = x_fd[nx // 2]
        self.t_star = t_star_steps * dt
        self.n_neighbours = 5
        self.eps_x, self.eps_t = 0.5 * dx, 0.5 * dt

        mid, half = nx // 2, N // 2
        step = (mid - margin) / half if half else 0
        off, k = [0], 1
        while len(off) < N:
            off.append(k * step)
            if len(off) < N:
                off.append(-k * step)
            k += 1
        self.walker_init_x = x_fd[np.clip(
            np.round(np.array(off[:N]) + mid).astype(int), margin, nx - 1 - margin)]

        self.observation_space = spaces.Box(-np.inf, np.inf, (3 * N + 2,), np.float32)
        self.action_space = spaces.Discrete(N * self.nA)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visited = VisitedSet()
        self.stencil_log = {}
        for ix in range(nx):
            self.visited.add(x_fd[ix], 0.0, u_true(x_fd[ix], 0.0))
        for it in range(1, nt):
            self.visited.add(0.0, t_fd[it], 0.0)
            self.visited.add(1.0, t_fd[it], 0.0)
        self.walker_x = np.array(self.walker_init_x, float)
        self.walker_t = np.zeros(self.N)
        self.walker_u = np.array([u_true(x, 0.0) for x in self.walker_x])
        self.step_count = 0
        self.gfdm_rejected_set = set()
        return self._obs(), {}

    def _obs(self):
        return np.array([*self.walker_x, *self.walker_t, *self.walker_u,
                         self.x_star, self.t_star], dtype=np.float32)

    def gfdm(self, xs, ts):
        nb = self.visited.find_neighbours(xs, ts, self.n_neighbours)
        if nb is None:
            return None, False, None, None, 'no_nb'
        nb = np.array(nb)
        w, r = solve_weights(nb[:, 0] - xs, nb[:, 1] - ts)
        if w is None:
            return None, False, None, None, r
        uh = float(w @ nb[:, 2])
        if abs(uh) > 1.1:
            return None, False, None, None, 'bounds_u'
        return uh, True, nb[:, :2], w, 'ok'

    def step(self, action):
        self.step_count += 1
        done = False
        i, j = int(action) // self.nA, int(action) % self.nA
        ddx, ddt = self.A[j]
        info = {'walker': i, 'accepted': False, 'rejected': None, 'reached': False}

        xn = self.walker_x[i] + ddx * dx
        tn = self.walker_t[i] + ddt * dt
        if not (0.0 <= xn <= 1.0 and 0.0 < tn <= self.T):
            info['rejected'] = 'bounds'
        elif self.visited.contains(xn, tn):
            info['rejected'] = 'collision'
        else:
            uh, ok, nbc, _, r = self.gfdm(xn, tn)
            if not ok:
                info['rejected'] = r
                self.gfdm_rejected_set.add(int(action))
            else:
                self.gfdm_rejected_set.clear()
                self.visited.add(xn, tn, uh)
                self.walker_x[i], self.walker_t[i], self.walker_u[i] = xn, tn, uh
                self.stencil_log[(round(xn, 8), round(tn, 8))] = nbc
                info['accepted'] = True
                if (abs(xn - self.x_star) < self.eps_x and
                        abs(tn - self.t_star) < self.eps_t):
                    done = True
                    info['reached'] = True

        if done or self.step_count >= self.K_max:
            done = True
            info['k_term'] = self.step_count
        return self._obs(), 0.0, done, False, info

    def action_masks(self):
        m = np.ones(self.N * self.nA, bool)
        for i in range(self.N):
            for j in range(self.nA):
                ddx, ddt = self.A[j]
                xn = self.walker_x[i] + ddx * dx
                tn = self.walker_t[i] + ddt * dt
                if not (0.0 <= xn <= 1.0 and 0.0 < tn <= self.T
                        and not self.visited.contains(xn, tn)):
                    m[i * self.nA + j] = False
        for a in self.gfdm_rejected_set:
            m[a] = False
        if not m.any():
            m[:] = True
        return m


def rollout(path):
    model = MaskablePPO.load(path, device='cpu')
    env = PDEEnvironment()
    obs, _ = env.reset(seed=0)
    traj = {i: [(env.walker_x[i], env.walker_t[i])] for i in range(env.N)}
    done, info = False, {}
    while not done:
        a, _ = model.predict(obs, deterministic=True,
                             action_masks=env.action_masks())
        obs, _, done, _, info = env.step(a)
        if info['accepted']:
            i = info['walker']
            traj[i].append((env.walker_x[i], env.walker_t[i]))

    u_pred, ok, nbc, w, _ = env.gfdm(env.x_star, env.t_star)
    err = None if u_pred is None else abs(u_pred - u_true(env.x_star, env.t_star))
    return dict(env=env, traj=traj, info=info, stencil=nbc if ok else None,
                w1=None if w is None else float(np.sum(np.abs(w))),
                err=err, k=info.get('k_term'),
                n_eval=len(env.visited) - (nx + 2 * (nt - 1)))


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Plot style -- matches the other paper figures
# ═══════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.labelsize":     9.5,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    6.6,
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "figure.dpi":         160,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

C_GRID    = "#b8bfc7"
C_STENCIL = "#d1495b"
C_TRAJ    = plt.cm.viridis


def fmt_err(e):
    if e is None:
        return "failed"
    m, ex = f"{e:.1e}".split("e")
    return rf"${m}\times 10^{{{int(ex)}}}$"


def draw(ax, res, xlim, ylim):
    env, traj = res['env'], res['traj']

    pts = np.asarray(env.visited._pts)
    keep = (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1])
    ax.scatter(pts[keep, 0], pts[keep, 1], s=5, color=C_GRID, linewidths=0,
               zorder=1, rasterized=True)

    colors = C_TRAJ(np.linspace(0.05, 0.95, env.N))
    for i, tr in traj.items():
        xs = [p[0] for p in tr]; ts = [p[1] for p in tr]
        if len(xs) > 1:
            ax.plot(xs, ts, '-', lw=1.0, color=colors[i], alpha=0.9, zorder=3,
                    solid_capstyle='round')
            ax.scatter(xs[1:], ts[1:], s=11, color=colors[i], linewidths=0, zorder=4)
    ax.scatter(env.walker_init_x, np.zeros(env.N), s=10, marker='^',
               color=colors, linewidths=0, zorder=5)

    if res['stencil'] is not None:
        for (xn, tn) in res['stencil']:
            ax.plot([env.x_star, xn], [env.t_star, tn], ls=(0, (2.5, 1.8)),
                    color=C_STENCIL, lw=0.9, alpha=0.9, zorder=6)
        ax.scatter(res['stencil'][:, 0], res['stencil'][:, 1], s=20,
                   color=C_STENCIL, linewidths=0, zorder=7)
    ax.scatter([env.x_star], [env.t_star], s=120, marker='*', color=C_STENCIL,
               edgecolors='white', linewidths=0.5, zorder=8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$x$')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    results = []
    for beta, path, k_ref in RUNS:
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint: {path}")
        r = rollout(path)
        r['beta'], r['k_ref'] = beta, k_ref
        assert r['k'] == k_ref, (
            f"beta={beta:g}: reproduced k={r['k']} but the archived figure says "
            f"{k_ref}; the environment reconstruction is wrong")
        results.append(r)
        print(f"beta={beta:.0e}  eps_eff={LAMBDA/beta:.0e}  k={r['k']} (archived "
              f"{k_ref})  |e|={r['err']:.3e}  ||w||_1={r['w1']:.4f}  "
              f"evals={r['n_eval']}")

    t_star = T_STAR_STEPS * dt
    xlim = (-0.012, 1.012)
    ylim = (-0.85 * dt, t_star + 2.6 * dt)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95), sharey=True,
                             constrained_layout=True)

    for ax, res, tag in zip(axes, results, "abc"):
        draw(ax, res, xlim, ylim)
        eps = int(round(np.log10(LAMBDA / res['beta'])))
        ax.set_title(rf"({tag})  $\varepsilon_{{\mathrm{{eff}}}}=10^{{{eps}}}$,"
                     rf"$\;\;|e|=$ " + fmt_err(res['err']),
                     fontsize=9, pad=13.5)
        ax.text(0.5, 1.012, rf"$k={res['k']}$ evaluations",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7.4,
                color="#3a3f45")

    axes[0].set_ylabel(r'$t/\Delta t$')
    for ax in axes:
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks(np.arange(0, T_STAR_STEPS + 1, 2) * dt)
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _p: f"{v / dt:.0f}"))
        ax.tick_params(length=3, width=0.8)
        for s in ax.spines.values():
            s.set_color("#3a3f45")

    mid = C_TRAJ(0.45)
    handles = [
        plt.Line2D([], [], ls='none', marker='o', ms=3.0, color=C_GRID,
                   label='initial / boundary data'),
        plt.Line2D([], [], ls='-', lw=1.1, marker='o', ms=3.0, color=mid,
                   label='walker path'),
        plt.Line2D([], [], ls='none', marker='^', ms=4.0, color=mid,
                   label='walker start'),
        plt.Line2D([], [], ls=(0, (2.5, 1.8)), lw=0.9, color=C_STENCIL,
                   marker='o', ms=3.0, label=r'stencil at $z^\ast$'),
        plt.Line2D([], [], ls='none', marker='*', ms=8, color=C_STENCIL,
                   label=r'$z^\ast=(x^\ast,t^\ast)$'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=5,
               frameon=False, handletextpad=0.5, columnspacing=1.5,
               borderaxespad=0.0)

    out_pdf = os.path.join(FIG_DIR, "fig_beta_sweep.pdf")
    out_png = os.path.join(FIG_DIR, "fig_beta_sweep.png")
    fig.savefig(out_pdf); fig.savefig(out_png)
    plt.close(fig)
    print("\nsaved:", out_pdf, "\n      ", out_png)


if __name__ == "__main__":
    main()
