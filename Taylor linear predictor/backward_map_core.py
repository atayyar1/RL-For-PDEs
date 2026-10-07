"""
backward_map_core.py

Everything from notebook 11 (cells 1-9 and the later fixes) that the comparison notebook needs,
as plain definitions with no checks, plots or training runs:

    PDE, grid, exact solution     alpha, c, nx, nt, dx, dt, x_grid, u0, make_u_true, u_true
    predictor and score           solve_weights, score, L_SOL, COND_MAX, W_MAX
    backward map                  is_known, weights, point_score, start_map, next_point, expand, build_map, evaluate
    window and rules              WINDOW, place, admissible, pick_mask, lowest_score, lw, greedy_at, fixed_rule
    numbers                       report, reference, map_return
    environment                   MapEnv, MapEnvGen, run_policy
    figure                        plot_map
    classical schemes             classical(scheme, k, z)

The initial condition only enters when a map is evaluated: every map-building rule (and the agent) sees
geometry only. To test another IC, pass its exact solution to report(m, u_fn) / classical(..., u_fn).

    from backward_map_core import *
"""
import heapq, itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gymnasium as gym
from gymnasium import spaces
from scipy.linalg import solve_banded

# ── PDE:  u_t + a(u) u_x = alpha u_xx   on [0, 1],  u = 0 at both walls ──
alpha = 0.1
c     = 1.0
a     = lambda u: c * np.ones_like(u)          # advection speed (Burgers: a(u) = u)
A_MAX = 1.0                                    # max |a(u)|, for the CFL limit
u0    = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)

# ── Grid ──
nx, T = 100, 0.5
dx = 1.0 / (nx - 1)
nt = int(np.ceil(T / min(0.9 * dx / A_MAX, 0.45 * dx**2 / alpha))) + 1
dt = T / (nt - 1)
x_grid = np.arange(nx) * dx

# ── Exact solution for any initial condition: u = exp(b x - alpha b^2 t) v, v solves the heat equation ──
_xq = np.linspace(0.0, 1.0, 80001)
_b  = c / (2 * alpha)
_n  = np.arange(1, 801)

def make_u_true(ic):
    """Exact solution u(x, t) for the initial condition ic(x) (zero at both walls)."""
    bn = np.array([2 * np.trapezoid(ic(_xq) * np.exp(-_b * _xq) * np.sin(k * np.pi * _xq), _xq) for k in _n])
    def u(x, t):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        v = (np.sin(np.pi * np.outer(x, _n)) * (bn * np.exp(-alpha * (_n * np.pi)**2 * t))).sum(axis=1)
        return np.exp(_b * x - alpha * _b**2 * t) * v
    return u

u_true = make_u_true(u0)

# ── Taylor predictor: 3 rows (value, u_x, u_xx) with u_t = alpha u_xx - a u_x substituted ──
COND_MAX, W_MAX = 1e4, 2.0
def solve_weights(dxs, dts, a_loc):
    """Min-norm weights of the neighbours at offsets (dxs, dts). None if not admissible."""
    xi = dxs - a_loc * dts
    h  = max(np.abs(dxs).max(), np.sqrt(alpha * np.abs(dts).max()))
    A  = np.array([np.ones_like(dxs), xi / h, (0.5 * xi**2 + alpha * dts) / h**2])
    if np.linalg.matrix_rank(A) < 3 or np.linalg.cond(A) > COND_MAX:
        return None
    w = np.linalg.pinv(A) @ np.array([1.0, 0.0, 0.0])
    return w if np.abs(w).sum() <= W_MAX else None

# ── Geometry score: leftover on the heat polynomials T3, T4 (no solution values) ──
L_SOL = 1 / (2 * np.pi)
def score(DX, DT, w, a_loc):
    xi = DX - a_loc * DT
    m3 = w @ (xi**3 + 6 * alpha * DT * xi)
    m4 = w @ (xi**4 + 12 * alpha * DT * xi**2 + 12 * (alpha * DT)**2)
    return abs(m3) / (6 * L_SOL**3) + abs(m4) / (24 * L_SOL**4)

# ── Backward map ──
is_known = lambda q: q[1] == 0 or q[0] in (0, nx - 1)
a_loc = c

_w_cache = {}
def weights(p, nbs):
    offs = tuple((q[0] - p[0], q[1] - p[1]) for q in nbs)
    if offs not in _w_cache:
        _w_cache[offs] = solve_weights(np.array([o[0] for o in offs]) * dx, np.array([o[1] for o in offs]) * dt, a_loc)
    return _w_cache[offs]

def point_score(p, nbs, w):
    return score(np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt, w, a_loc)

def start_map(z_star):
    return {"z_star": z_star, "pending": [(-z_star[1], z_star[0])], "beta": {z_star: 1.0}, "stencil": {}}

def next_point(m):
    it, ix = m["pending"][0]
    return (ix, -it)

def expand(m, nbs):
    """Give the next point the stencil nbs and pass its influence down: beta_q += beta_p * w_q."""
    it, ix = heapq.heappop(m["pending"])
    p = (ix, -it)
    w = weights(p, nbs)
    m["stencil"][p] = (nbs, w)
    for q, wq in zip(nbs, w):
        if is_known(q):
            continue
        if q not in m["beta"]:
            m["beta"][q] = 0.0
            heapq.heappush(m["pending"], (-q[1], q[0]))
        m["beta"][q] += m["beta"][p] * wq
    return p, w

def build_map(z_star, choose):
    """Whole map; choose(m, p) returns the neighbours of p."""
    m = start_map(z_star)
    while m["pending"]:
        expand(m, choose(m, next_point(m)))
    return m

def evaluate(m, u_fn=None):
    """Values at every computed point, from the IC (of u_fn, default u_true) and the zero walls."""
    U0 = (u_fn or u_true)(x_grid, 0.0)
    u_hat = {}
    value = lambda q: (U0[q[0]] if q[1] == 0 else 0.0) if is_known(q) else u_hat[q]
    for p in sorted(m["stencil"], key=lambda p: p[1]):
        nbs, w = m["stencil"][p]
        u_hat[p] = sum(wq * value(q) for q, wq in zip(nbs, w))
    return u_hat

# ── Window and admissible stencils ──
R_X, K_T = 3, 3
WINDOW  = [(dix, dit) for dit in range(-1, -K_T - 1, -1) for dix in range(-R_X, R_X + 1)]
N_CELLS = len(WINDOW)
N_NB    = 5

def place(p, k):
    return (min(max(p[0] + WINDOW[k][0], 0), nx - 1), max(p[1] + WINDOW[k][1], 0))

def situation(p):
    return tuple((place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS))

_sets_cache = {}
def admissible(p):
    """(stencils as tuples of cells, cell-usage matrix, scores)."""
    key = situation(p)
    if key not in _sets_cache:
        cells, seen = [], set()
        for k in range(N_CELLS):
            if place(p, k) not in seen:
                seen.add(place(p, k)); cells.append(k)
        sets, scores = [], []
        for s in itertools.combinations(cells, min(N_NB, len(cells))):
            nbs = [place(p, k) for k in s]
            w = weights(p, nbs)
            if w is not None:
                sets.append(s); scores.append(point_score(p, nbs, w))
        uses = np.zeros((len(sets), N_CELLS), dtype=bool)
        for i, s in enumerate(sets):
            uses[i, list(s)] = True
        _sets_cache[key] = (sets, uses, np.array(scores))
    return _sets_cache[key]

def pick_mask(p, picks):
    _, uses, _ = admissible(p)
    ok = uses[:, picks].all(axis=1) if picks else np.ones(len(uses), dtype=bool)
    mask = uses[ok].any(axis=0)
    mask[picks] = False
    return mask

# ── Map-building rules (no learning) ──
lw = lambda m, p: [(p[0] - 1, p[1] - 1), (p[0], p[1] - 1), (p[0] + 1, p[1] - 1)]     # Lax-Wendroff

def lowest_score(m, p):
    sets, _, sc = admissible(p)
    return [place(p, k) for k in sets[int(np.argmin(sc))]]

_ref_cache = {}
def reference(z):
    """(score sum, points) of LW's map: the ruler of the reward."""
    if z not in _ref_cache:
        m = build_map(z, lw)
        _ref_cache[z] = (sum(abs(m["beta"][p]) * point_score(p, *m["stencil"][p]) for p in m["stencil"]), len(m["stencil"]))
    return _ref_cache[z]

def greedy_at(lam):
    """Best immediate reward at every point: beta-weighted score + lam x new points."""
    def choose(m, p):
        B_ref, N_ref = reference(m["z_star"])
        sets, uses, sc = admissible(p)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)], dtype=float)
        return [place(p, k) for k in sets[int(np.argmin(abs(m["beta"][p]) * sc / B_ref + lam * (uses @ new) / N_ref))]]
    return choose

def fixed_rule(s):
    """Stencil s (tuple of window cells) wherever admissible, the lowest-score stencil elsewhere (walls, IC)."""
    s, allowed = tuple(sorted(s)), {}
    def choose(m, p):
        key = situation(p)
        if key not in allowed:
            allowed[key] = set(admissible(p)[0])
        return [place(p, k) for k in s] if s in allowed[key] else lowest_score(m, p)
    return choose

BEST_FIXED_50_20 = tuple(sorted(WINDOW.index(o) for o in [(-1, -1), (-2, -2), (1, -2), (0, -3), (3, -3)]))

# ── Numbers for a finished map ──
def report(m, u_fn=None):
    """points, work (points x neighbours), true error at z*, exact bound sum|beta delta|, for the IC of u_fn."""
    u_fn = u_fn or u_true
    U = lambda q: u_fn(q[0] * dx, q[1] * dt)[0]
    z = m["z_star"]
    delta = {p: U(p) - sum(wq * U(q) for q, wq in zip(nbs, w)) for p, (nbs, w) in m["stencil"].items()}
    return {"points": len(m["stencil"]),
            "work":   sum(len(nbs) for nbs, _ in m["stencil"].values()),
            "error":  abs(U(z) - evaluate(m, u_fn)[z]),
            "bound":  sum(abs(m["beta"][p] * delta[p]) for p in delta)}

def map_return(m, z, lam):
    B_ref, N_ref = reference(z)
    B = sum(abs(m["beta"][p]) * point_score(p, *m["stencil"][p]) for p in m["stencil"])
    return -B / B_ref - lam * (len(m["stencil"]) - 1) / N_ref

# ── Environment (as trained in notebook 11) ──
N_OBS = 4 * N_CELLS + 7

class MapEnv(gym.Env):
    def __init__(self, queries, lam=1.0, cap=3.0):
        super().__init__()
        self.queries, self.lam, self.cap = queries, lam, cap
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_CELLS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z = self.queries[self.np_random.integers(len(self.queries))]
        self.B_ref, self.N_ref = reference(self.z)
        self.m, self.picks, self.B = start_map(self.z), [], 0.0
        self._play_forced()
        return self._obs(), {}

    def _complete(self, cells):
        p = next_point(self.m)
        cells = sorted(cells)
        nbs = [place(p, k) for k in cells]
        if weights(p, nbs) is None:
            raise RuntimeError(f"inadmissible stencil {[WINDOW[k] for k in cells]} at p = {p}, z* = {self.z}")
        n_new = sum(1 for q in set(nbs) if not is_known(q) and q not in self.m["beta"])
        _, w = expand(self.m, nbs)
        term = abs(self.m["beta"][p]) * point_score(p, nbs, w)
        self.B += term
        return -term / self.B_ref - self.lam * n_new / self.N_ref

    def _play_forced(self):
        r = 0.0
        while self.m["pending"] and len(admissible(next_point(self.m))[0]) == 1:
            r += self._complete(admissible(next_point(self.m))[0][0])
        return r

    def _obs(self):
        if not self.m["pending"]:
            return np.zeros(N_OBS, np.float32)
        p = next_point(self.m)
        picked, exists, known = np.zeros(N_CELLS), np.zeros(N_CELLS), np.zeros(N_CELLS)
        picked[self.picks] = 1
        for k in range(N_CELLS):
            q = place(p, k)
            known[k], exists[k] = is_known(q), q in self.m["beta"]
        _, uses, sc = admissible(p)
        rel = np.log10(sc / sc.min())
        ok = uses[:, self.picks].all(axis=1) if self.picks else np.ones(len(rel), dtype=bool)
        reach = np.where(uses[ok], rel[ok, None], 3.0).min(axis=0)
        reach[self.picks] = 3.0
        pend = [(ix, -it) for it, ix in self.m["pending"]]
        glob = [np.log10(abs(self.m["beta"][p]) + 1e-12), p[1], p[0] - self.z[0], len(self.picks),
                len(self.m["stencil"]) / self.N_ref, len(pend) / self.N_ref, sum(abs(self.m["beta"][q]) for q in pend)]
        return np.concatenate([picked, exists, known, np.minimum(reach, 3.0), glob]).astype(np.float32)

    def step(self, k):
        self.picks.append(int(k))
        p = next_point(self.m)
        if len(self.picks) < len(admissible(p)[0][0]):
            return self._obs(), 0.0, False, False, {}
        r = self._complete(self.picks)
        self.picks = []
        r += self._play_forced()
        n = len(self.m["stencil"]) + len(self.m["pending"])
        if self.m["pending"] and n <= self.cap * self.N_ref:
            return self._obs(), r, False, False, {}
        for it, ix in self.m["pending"]:
            q = (ix, -it)
            r -= abs(self.m["beta"][q]) * admissible(q)[2].max() / self.B_ref
        info = {"finished": not self.m["pending"], "points": len(self.m["stencil"]),
                "score_ratio": self.B / self.B_ref, "points_ratio": len(self.m["stencil"]) / self.N_ref}
        return np.zeros(N_OBS, np.float32), r, True, False, info

    def action_masks(self):
        return pick_mask(next_point(self.m), self.picks)

class MapEnvGen(MapEnv):
    """MapEnv with 'levels above the IC' clipped at 10 and 'distance to the nearest wall' (clipped at 10)."""
    def _obs(self):
        o = super()._obs()
        if self.m["pending"]:
            p = next_point(self.m)
            o[4 * N_CELLS + 1] = min(p[1], 10)
            o[4 * N_CELLS + 2] = min(p[0], nx - 1 - p[0], 10)
        return o

def run_policy(model, z, lam=1.0, env_cls=MapEnv):
    """One episode on z* with the policy's most likely picks. Returns the finished env (map in env.m)."""
    env = env_cls(queries=[z], lam=lam)
    obs, _ = env.reset(seed=0)
    done, ret = False, 0.0
    while not done:
        a_, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, r, done, _, info = env.step(a_)
        ret += r
    env.ret, env.info = ret, info
    return env

# ── Classical schemes on the grid refined k times (r = alpha dt/dx^2 kept fixed) ──
def classical(scheme, k, z, u_fn=None):
    """(work, true error at z*) for FD-3, LW, Taylor-5, FD-5 + RK4 or CN; work = points in the cone x per-point cost."""
    u_fn = u_fn or u_true
    n, h = (nx - 1) * k + 1, dx / k
    tau = dt / k**2 if scheme != "CN" else 4 * dt / k**2
    ix, steps = z[0] * k, round(z[1] * dt / tau)
    u = u_fn(np.arange(n) * h, 0.0)
    r, nu = alpha * tau / h**2, c * tau / h
    d2 = lambda v: np.r_[0, v[2:] - 2 * v[1:-1] + v[:-2], 0]
    d1 = lambda v: np.r_[0, v[2:] - v[:-2], 0] / 2
    def d4(v):
        a2, a1 = d2(v), d1(v)
        a2[2:-2] = (-v[4:] + 16 * v[3:-1] - 30 * v[2:-2] + 16 * v[1:-3] - v[:-4]) / 12
        a1[2:-2] = (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / 12
        return a2, a1
    if scheme == "Taylor-5":
        xi, ta = np.arange(-2, 3) * h + c * tau, -tau * np.ones(5)
        A = np.array([np.ones(5), xi, xi**2 + 2 * alpha * ta, xi**3 + 6 * alpha * ta * xi,
                      xi**4 + 12 * alpha * ta * xi**2 + 12 * (alpha * ta)**2])
        W5 = np.linalg.solve(A, [1, 0, 0, 0, 0])
    if scheme == "CN":
        ab = np.zeros((3, n - 2)); ab[0, 1:], ab[1], ab[2, :-1] = -r / 2 + nu / 4, 1 + r, -r / 2 - nu / 4
    for _ in range(steps):
        if scheme == "FD-3":
            u = u + r * d2(u) - nu * d1(u)
        elif scheme == "LW":
            u = u + (r + nu**2 / 2) * d2(u) - nu * d1(u)
        elif scheme == "FD-5 + RK4":
            f = lambda v: (lambda a2, a1: r * a2 - nu * a1)(*d4(v))
            k1 = f(u); k2 = f(u + k1 / 2); k3 = f(u + k2 / 2); k4 = f(u + k3)
            u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        elif scheme == "Taylor-5":
            v = u + (r + nu**2 / 2) * d2(u) - nu * d1(u)
            v[2:-2] = W5 @ np.array([u[:-4], u[1:-3], u[2:-2], u[3:-1], u[4:]])
            u = v
        elif scheme == "CN":
            u = np.r_[0, solve_banded((1, 1), ab, (u + r / 2 * d2(u) - nu / 2 * d1(u))[1:-1]), 0]
        u[0] = u[-1] = 0.0
    err = abs(u[ix] - u_fn(z[0] * dx, z[1] * dt)[0])
    spread = {"FD-3": 1, "LW": 1, "Taylor-5": 2, "FD-5 + RK4": 8}.get(scheme)
    pts = (n - 2) * steps if spread is None else sum(min(n - 2, ix + spread * j) - max(1, ix - spread * j) + 1 for j in range(steps))
    per_point = {"FD-3": 3, "LW": 3, "Taylor-5": 5, "FD-5 + RK4": 20, "CN": 6}[scheme]
    return pts * per_point, err

# ── One figure for any finished map ──
def plot_map(m, title="", path=None, u_fn=None):
    """2x2: map coloured by influence | zoom on z* with stencils | accumulated error | error share per point."""
    u_fn = u_fn or u_true
    z, rep = m["z_star"], report(m, u_fn)
    U = lambda q: u_fn(q[0] * dx, q[1] * dt)[0]
    pts = list(m["stencil"])
    P = np.array(pts)
    beta = np.array([abs(m["beta"][p]) for p in pts])
    share = np.array([abs(m["beta"][p] * (U(p) - sum(wq * U(q) for q, wq in zip(*m["stencil"][p])))) for p in pts])
    u_hat = evaluate(m, u_fn)
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13, 9))
    sc = a1.scatter(P[:, 0], P[:, 1], c=beta, s=8, cmap="viridis", norm=LogNorm(1e-8, 1))
    a1.plot(*z, 'r*', ms=14); fig.colorbar(sc, ax=a1, label="|beta|")
    a1.set_xlabel("ix"); a1.set_ylabel("it"); a1.set_title("Map, coloured by influence on z*")
    top = [p for p in pts if p[1] >= z[1] - 2 * K_T]
    for p in top:
        for q in m["stencil"][p][0]:
            a2.plot([p[0], q[0]], [p[1], q[1]], '-', color='gray', lw=0.6, alpha=0.6)
    Tp = np.array(top)
    a2.scatter(Tp[:, 0], Tp[:, 1], c=[abs(m["beta"][p]) for p in top], s=30, cmap="viridis", norm=LogNorm(1e-8, 1), zorder=3)
    a2.plot(*z, 'r*', ms=16, zorder=4); a2.set_ylim(z[1] - 2 * K_T - 0.5, z[1] + 0.5)
    a2.set_xlabel("ix"); a2.set_ylabel("it"); a2.set_title(f"Zoom on z*: stencils of the top {2 * K_T} levels")
    acc = np.array([abs(u_hat[p] - U(p)) for p in pts])
    sc = a3.scatter(P[:, 1], np.maximum(acc, 1e-16), c=np.abs(P[:, 0] - z[0]), s=8, cmap="plasma")
    a3.plot(z[1], rep["error"], 'r*', ms=16, label=f"error at z* = {rep['error']:.1e}")
    fig.colorbar(sc, ax=a3, label="|ix - ix*|")
    a3.set_yscale("log"); a3.set_xlabel("it"); a3.set_ylabel("|u_hat - u|"); a3.legend(fontsize=8)
    a3.set_title("Accumulated error at every computed point")
    sc = a4.scatter(P[:, 1], np.maximum(share, 1e-16), c=np.abs(P[:, 0] - z[0]), s=8, cmap="plasma")
    fig.colorbar(sc, ax=a4, label="|ix - ix*|")
    a4.set_yscale("log"); a4.set_ylim(max(share.max() * 1e-8, 1e-16), share.max() * 3)
    a4.set_xlabel("it"); a4.set_ylabel("|beta_p delta_p|")
    a4.set_title(f"Error share per point (sum = exact bound {rep['bound']:.1e})")
    fig.suptitle(f"{title}   z* = {z}  |  {rep['points']} points, work {rep['work']}  |  "
                 f"exact bound {rep['bound']:.1e}, error {rep['error']:.1e}")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    else:
        plt.show()
