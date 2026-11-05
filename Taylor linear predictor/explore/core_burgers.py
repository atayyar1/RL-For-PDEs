"""Core of notebook "12.3 Robust Backward Burger Shock.ipynb" (PROBLEM = "inflow_front": a front forms and leaves),
extracted as a plain module, code kept as written. Same cells as explore/core.py; checks and pictures left out."""
import os, heapq, itertools, time, warnings
import numpy as np


# ── Problem settings: change PROBLEM, everything below follows ──
#   "walls"         : u0 = sin(pi x) + 0.5 sin(2 pi x), u = 0 at both walls, alpha = 0.1   (notebook 12, for comparison)
#   "inflow_smooth" : u0 = 1 + 0.8 sin(2 pi x / 1.5), time-dependent Dirichlet data from the exact solution, alpha = 0.1
#   "inflow_front"  : same, alpha = 0.01: a front forms and leaves (t ~ 0.25-0.35); a second one enters through the left wall (t ~ 0.7)
PROBLEM = os.environ.get("BURGERS_PROBLEM", "inflow_front")   # "walls", "inflow_smooth", or "inflow_front"
alpha, T = {"walls": (0.1, 0.5), "inflow_smooth": (0.1, 0.5), "inflow_front": (0.01, 1.1)}[PROBLEM]

# ── PDE:  u_t + a(u) u_x = alpha u_xx  on [0, 1] ──
a = lambda u: u                                # advection speed: Burgers
if PROBLEM == "walls":
    u0 = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)
    F0 = lambda x: (1 - np.cos(np.pi * x)) / np.pi + (1 - np.cos(2 * np.pi * x)) / (4 * np.pi)   # integral of u0 from 0
    A_MAX = 1.5                                # bound on |u| (the IC peaks at about 1.26)
else:
    C_BG, A_W, L_W = 1.0, 0.8, 1.5             # background speed, amplitude, wavelength: u stays in [0.2, 1.8] > 0
    X_PK = 0.5                                 # the peak starts at the middle, so u changes a lot there
    u0 = lambda x: C_BG + A_W * np.cos(2 * np.pi * (x - X_PK) / L_W)
    F0 = lambda x: C_BG * x + A_W * L_W / (2 * np.pi) * (np.sin(2 * np.pi * (x - X_PK) / L_W) + np.sin(2 * np.pi * X_PK / L_W))
    A_MAX = C_BG + A_W

# ── Grid ──
# explicit advection + diffusion (LW, FD-3) is stable for 2r + nu^2 <= 1, r = alpha dt/dx^2, nu = A_MAX dt/dx; 0.9 for margin
nx = 50
dx = 1.0 / (nx - 1)
_c2, _c1 = (A_MAX / dx)**2, 2 * alpha / dx**2
nt = int(np.ceil(T / ((-_c1 + np.sqrt(_c1**2 + 4 * _c2 * 0.9)) / (2 * _c2)))) + 1
dt = T / (nt - 1)
x_grid = np.arange(nx) * dx

# ── Exact solution (Hopf-Cole), as a weighted average so it stays accurate for small alpha ──
#    u(x, t) = sum (x - y)/t W / sum W,   W = exp(-F0(y)/(2 alpha) - (x - y)^2 / (4 alpha t)),   y on a uniform grid.
#    "walls": F0 mirrored about every wall (even, period 2) -> phi_x = 0 -> u = 0 at x = 0 and 1.
#    inflow : the whole line, no mirrors; the weight peaks near the foot of the characteristic, y ~ x - u t,
#             so the y range reaches A_MAX * T to the left. Both cover 8 kernel widths. Exponents shifted by their max.
_s = 8 * np.sqrt(2 * alpha * T)
if PROBLEM == "walls":
    _lo, _hi = -np.ceil(_s), 1 + np.ceil(_s)
    _Y  = np.linspace(_lo, _hi, int(round((_hi - _lo) * 10000)) + 1)
    _yf = np.mod(_Y, 2.0)
    _FY = F0(np.where(_yf > 1, 2 - _yf, _yf)) / (2 * alpha)
else:
    _lo, _hi = -A_MAX * T - _s, 1 + _s
    _Y  = np.linspace(_lo, _hi, int(round((_hi - _lo) * 10000)) + 1)
    _FY = F0(_Y) / (2 * alpha)

def u_true(x, t):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if t <= 0:
        return u0(x)
    out = np.empty(len(x))
    for i in range(0, len(x), 50):                             # chunks of x, to keep memory small
        xs = x[i:i + 50, None]
        E  = -_FY - (xs - _Y)**2 / (4 * alpha * t)
        W  = np.exp(E - E.max(axis=1, keepdims=True))
        out[i:i + 50] = ((xs - _Y) / t * W).sum(1) / W.sum(1)
    return out

# exact values on the grid, one time level at a time when first needed (u_true ONLY for checks, reports, and wall data)
_U_rows = {}
def U_row(it):
    if it not in _U_rows:
        _U_rows[it] = u_true(x_grid, it * dt)
    return _U_rows[it]
Ug = lambda q: U_row(q[1])[q[0]]

# ── Boundary data: u at x = 0 and x = 1 as functions of t (given data, like the IC) ──
# on the grid's time levels it is exact (U_row); between levels (refined classical schemes) a cubic spline through a fine table
from scipy.interpolate import CubicSpline
_tb = np.linspace(0.0, T, 2001)
_gb = np.array([u_true([0.0, 1.0], t) for t in _tb])            # (time, [left, right])
_gs = CubicSpline(_tb, _gb)                                     # cubic in t: accurate enough for high-order schemes too
g_left  = lambda t: float(_gs(t)[0])
g_right = lambda t: float(_gs(t)[1])



# ── Taylor predictor: 3 rows (value, u_x, u_xx) with u_t = alpha u_xx - a u_x substituted ──
COND_MAX, W_MAX = 1e4, 2.0
def solve_weights_many(DX, DT, a_loc):
    """Min-norm weights for M stencils at once: DX, DT of shape (M, n) = neighbour offsets, a_loc = local speed.
    Returns W (M, n) and ok (M,): admissible = rank 3, cond <= COND_MAX and ||w||_1 <= W_MAX."""
    xi = DX - a_loc * DT                                              # offset in the moving frame
    h  = np.maximum(np.abs(DX).max(1), np.sqrt(alpha * np.abs(DT).max(1)))[:, None]
    A  = np.stack([np.ones_like(DX), xi / h, (0.5 * xi**2 + alpha * DT) / h**2], axis=1)   # (M, 3, n)
    s  = np.linalg.svd(A, compute_uv=False)                           # singular values, largest first
    ok = (s[:, -1] > s[:, 0] * DX.shape[1] * np.finfo(float).eps) & (s[:, 0] <= COND_MAX * s[:, -1])
    W  = np.linalg.pinv(A) @ np.array([1.0, 0.0, 0.0])
    return W, ok & (np.abs(W).sum(1) <= W_MAX)

def solve_weights(dxs, dts, a_loc):
    """One stencil: weights, or None if not admissible."""
    W, ok = solve_weights_many(dxs[None], dts[None], a_loc)
    return W[0] if ok[0] else None

# ── Local error estimate (score): the stencil's error on the local Taylor polynomial of u around p ──
# PDE-specific part: Taylor coefficients with every time derivative replaced using Burgers, u_t = alpha u_xx - u u_x
def taylor_coeffs(d):
    """d = (u, u_x, u_xx, u_xxx, u_xxxx) at p. Returns {(i, j): (d/dx)^i (d/dt)^j u / (i! j!)} for weight i + 2j <= 4."""
    u, u1, u2, u3, u4 = d
    ut   = alpha * u2 - u * u1
    uxt  = alpha * u3 - u * u2 - u1**2                                     # Burgers term: - u_x^2
    uxxt = alpha * u4 - u * u3 - 3 * u1 * u2                               # Burgers term: - 3 u_x u_xx
    utt  = alpha**2 * u4 - 2 * alpha * u * u3 + u**2 * u2 + 2 * u1 * (u * u1 - 2 * alpha * u2)   # Burgers term: 2 u_x (...)
    return {(1, 0): u1, (2, 0): u2 / 2, (3, 0): u3 / 6, (4, 0): u4 / 24,
            (0, 1): ut, (1, 1): uxt, (2, 1): uxxt / 2, (0, 2): utt / 2}

def score(DX, DT, w, d):
    """|sum_q w_q P(dx_q, dt_q)|, P = Taylor polynomial of u around p (without u_p). One stencil (n,) or M stencils (M, n)."""
    P = sum(c * DX**i * DT**j for (i, j), c in taylor_coeffs(d).items())
    return np.abs((w * P).sum(-1))

def x_derivs(f, x, h=2e-3):
    """(u, u_x, u_xx, u_xxx, u_xxxx) of a function f at x, by 7-point central differences."""
    v = f(x + h * np.arange(-3, 4))
    C = [[0, 0, 0, 1, 0, 0, 0], np.array([-1, 9, -45, 0, 45, -9, 1]) / 60, np.array([2, -27, 270, -490, 270, -27, 2]) / 180,
         np.array([1, -8, 13, 0, -13, 8, -1]) / 8, np.array([-1, 12, -39, 56, -39, 12, -1]) / 6]
    return tuple(np.dot(c, v) / h**k for k, c in enumerate(C))



# ── Rough guess of u and its x-derivatives, used ONLY to build the map (speed, weights, score, influence) ──
# Lax-Wendroff on a grid K_COARSE times coarser, from the initial condition to T, with the same boundary data; computed once.
K_COARSE = {"walls": 5, "inflow_smooth": 3, "inflow_front": 2}[PROBLEM]
from scipy.interpolate import make_interp_spline
xc  = np.linspace(0.0, 1.0, (nx - 1) // K_COARSE + 1)
hc  = xc[1] - xc[0]
_c2, _c1 = (A_MAX / hc)**2, 2 * alpha / hc**2                                 # same stability rule as the main grid
tau = T / int(np.ceil(T / ((-_c1 + np.sqrt(_c1**2 + 4 * _c2 * 0.9)) / (2 * _c2))))
tc  = np.arange(0.0, T + tau / 2, tau)
Uc  = [u0(xc)]
for j in range(1, len(tc)):
    u, ui = Uc[-1], Uc[-1][1:-1]
    r, nu = alpha * tau / hc**2, a(ui) * tau / hc
    n = u.copy()
    n[1:-1] = ui + (r + nu**2 / 2) * (u[2:] - 2 * ui + u[:-2]) - nu / 2 * (u[2:] - u[:-2])
    n[0], n[-1] = g_left(tc[j]), g_right(tc[j])                               # boundary data
    Uc.append(n)
G_COARSE = len(xc) * len(tc)                                                  # cost of the guess (grid points)

# ── Smooth fit in x at each coarse time: a quintic spline through the coarse values, differentiated exactly ──
Dc = np.zeros((5, len(tc), nx))                                               # (derivative order, coarse time, our x)
for j, u in enumerate(Uc):
    sp = make_interp_spline(xc, u, k=5)
    for k in range(5):
        Dc[k, j] = sp.derivative(k)(x_grid) if k else sp(x_grid)

# ── Linear in time onto our grid: GUESS[k, it, ix] = k-th x-derivative of the guess at grid point (ix, it) ──
tg = np.arange(nt) * dt
jj = np.minimum((tg // tau).astype(int), len(tc) - 2)
f  = ((tg - tc[jj]) / tau)[None, :, None]
GUESS = (1 - f) * Dc[:, jj] + f * Dc[:, jj + 1]
guess_derivs = lambda q: GUESS[:, q[1], q[0]]                                 # (u, u_x, u_xx, u_xxx, u_xxxx) at q



# ── Known data: the initial condition and the walls are given, not computed ──
# their values are the exact solution there: u0 at t = 0, the boundary data g_left / g_right on the walls
is_known    = lambda q: q[1] == 0 or q[0] in (0, nx - 1)
known_value = Ug

# ── Speed at p for its weights: from the guess, rounded to A_STEP so weights can be cached ──
A_STEP  = 0.02
a_build = lambda p: A_STEP * round(a(guess_derivs(p)[0]) / A_STEP)

_w_cache = {}
def weights(p, nbs):
    """Taylor weights of p's neighbours nbs (grid points), with p's guessed speed; None if not admissible.
    Cached by (shape, speed). Solved with the neighbours sorted (latest level first, then left to right), so the
    result never depends on the order the neighbours were picked in."""
    order = sorted(range(len(nbs)), key=lambda i: (-nbs[i][1], nbs[i][0]))
    offs  = tuple((nbs[i][0] - p[0], nbs[i][1] - p[1]) for i in order)
    key   = (offs, a_build(p))
    if key not in _w_cache:
        _w_cache[key] = solve_weights(np.array([o[0] for o in offs]) * dx, np.array([o[1] for o in offs]) * dt, key[1])
    if _w_cache[key] is None:
        return None
    w = np.empty(len(nbs)); w[order] = _w_cache[key]                  # back to the order of nbs
    return w

def point_score(p, nbs, w):
    """Estimated local error of p's stencil, with the guessed derivatives at p."""
    return score(np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt, w, guess_derivs(p))
# ── Same-level neighbours (upwind only) ──
#   SAME_LEVEL = k > 0 adds the cells (-1, 0) .. (-k, 0): neighbours on p's OWN time level, to its left (upwind for u > 0).
#   To stay explicit (no linear solve): within a level, values are computed LEFT to right (left neighbours are ready),
#   and points are expanded RIGHT to left (every user of a point has passed its influence down before the point is
#   expanded, so beta stays exact). SAME_LEVEL = 0 restores the original window and order (needed for the old agents).
SAME_LEVEL = 0
ORDER = -1 if SAME_LEVEL else 1                      # sign of ix in the pending heap: -1 = right to left within a level

# ── The map, built backward from z* ──
# Points wait in `pending` and are expanded LATEST t first (and, within a level, in the order above): every point that can
# use p is expanded before p, so p's influence beta_p is complete when p is expanded.
def start_map(z_star):
    return {"z_star": z_star, "pending": [(-z_star[1], ORDER * z_star[0])], "beta": {z_star: 1.0}, "stencil": {}}

def pending_points(m):
    """The pending points as (ix, it)."""
    return [(ORDER * k, -it) for it, k in m["pending"]]
# ── Same-level neighbours on BOTH sides ──
#   SAME_LEVEL = k > 0 adds the cells (-k..-1, 0) and (1..k, 0): neighbours on p's own time level, left AND right.
#   No linear solve: within a level the same-level links must form no cycle (a mask forbids any link that would close one),
#   so the level can be computed in dependency order. Because a point's same-level users can come after it, its influence
#   beta is only final when its whole level is done: beta is passed down ONE LEVEL AT A TIME, when the level is complete.
#   SAME_LEVEL = 0 gives back the original window (needed for the old agents).

# ── The map, built backward from z* ──
# Points wait in `pending` and are expanded latest t first, left to right within a level. Stencils are chosen point by
# point; when the last point of a level is expanded, the level is "closed": its beta is passed down (users before the
# points they use), which completes the beta of the lower points before they are expanded.
def start_map(z_star):
    return {"z_star": z_star, "pending": [(-z_star[1], z_star[0])], "beta": {z_star: 1.0}, "stencil": {},
            "same": {}, "level": []}               # same: p -> same-level points p uses | level: expanded points of the open level

def next_point(m):
    it, ix = m["pending"][0]
    return (ix, -it)

def reaches(m, a, b):
    """True if same-level point a uses b, directly or through other same-level points."""
    stack, seen = [a], set()
    while stack:
        q = stack.pop()
        if q == b:
            return True
        if q not in seen:
            seen.add(q); stack.extend(m["same"].get(q, []))
    return False

def forbidden(m, p):
    """Window cells p may not use: same-level points that already use p (directly or not), which would close a cycle."""
    return [k for k in range(N_CELLS) if WINDOW[k][1] == 0 and not is_known(place(p, k)) and reaches(m, place(p, k), p)]

def users_first(m, pts):
    """The points of one level ordered so that every point comes before the same-level points it uses."""
    pts, order, done = set(pts), [], set()
    def visit(q):                                  # depth-first: a point is placed after all its same-level users
        if q in done:
            return
        done.add(q)
        for r in pts:
            if q in m["same"].get(r, []):
                visit(r)
        order.append(q)
    for q in sorted(pts):
        visit(q)
    return order

def close_level(m):
    """Pass the open level's beta down (users first) and return the level's points (their beta is now final)."""
    pts = m["level"]
    for p in users_first(m, pts):
        nbs, w = m["stencil"][p]
        for q, wq in zip(nbs, w):
            if not is_known(q):
                m["beta"][q] += m["beta"][p] * wq
    m["level"] = []
    return pts

def expand(m, nbs):
    """Give the next point the stencil nbs. Returns (p, w, closed): closed = the points whose beta became final now."""
    it, ix = heapq.heappop(m["pending"])
    p = (ix, -it)
    w = weights(p, nbs)
    m["stencil"][p] = (nbs, w)
    m["same"][p] = [q for q in nbs if q[1] == p[1] and not is_known(q)]
    for q in nbs:
        if not is_known(q) and q not in m["beta"]:
            m["beta"][q] = 0.0
            heapq.heappush(m["pending"], (-q[1], q[0]))
    m["level"].append(p)
    level_done = not m["pending"] or m["pending"][0][0] != it
    return p, w, (close_level(m) if level_done else [])

def build_map(z_star, choose):
    """Whole map; choose(m, p) returns the neighbours of p (it must respect forbidden(m, p))."""
    m = start_map(z_star)
    while m["pending"]:
        expand(m, choose(m, next_point(m)))
    return m

# ── Forward pass: values bottom-up; within a level, every point after the same-level points it uses ──
# The map fixes the neighbours (chosen with the guessed speed). Each point then re-solves its weights with the speed
# of its own computed value (picard steps; 0 = keep the map's weights), so the guessed speed never enters the values.
# The weights actually used are kept in m["w_used"] (report needs them).
def forward_order(m):
    order = []
    for t in sorted({p[1] for p in m["stencil"]}):
        order += users_first(m, [p for p in m["stencil"] if p[1] == t])[::-1]
    return order

def evaluate(m, picard=1):
    u_hat, m["w_used"] = {}, {}
    value = lambda q: known_value(q) if is_known(q) else u_hat[q]
    for p in forward_order(m):
        nbs, w = m["stencil"][p]
        v = np.array([value(q) for q in nbs])
        DX, DT = np.array([q[0] - p[0] for q in nbs]) * dx, np.array([q[1] - p[1] for q in nbs]) * dt
        for _ in range(picard):
            w_new = solve_weights(DX, DT, a(w @ v))             # speed from the value just computed
            if w_new is None:                                   # not admissible: keep the last weights
                break
            w = w_new
        m["w_used"][p], u_hat[p] = w, w @ v
    return u_hat
lw = lambda m, p: [(p[0] - 1, p[1] - 1), (p[0], p[1] - 1), (p[0] + 1, p[1] - 1)]


# ── The window: candidate neighbour cells below p ──
R_X, K_T = 3, 3                       # |dix| <= 3 cells, 1 to 3 levels down
WINDOW  = ([(k, 0) for k in list(range(-SAME_LEVEL, 0)) + list(range(1, SAME_LEVEL + 1))]
           + [(dix, dit) for dit in range(-1, -K_T - 1, -1) for dix in range(-R_X, R_X + 1)])
N_CELLS = len(WINDOW)
N_NB    = 5                           # neighbours per stencil
TERMS   = list(taylor_coeffs((0.0,) * 5))                             # the (i, j) Taylor terms of the score

def place(p, k):
    """Grid point of window cell k seen from p. Past a wall or below t = 0 -> moved onto it."""
    return (min(max(p[0] + WINDOW[k][0], 0), nx - 1), max(p[1] + WINDOW[k][1], 0))

# ── Every admissible stencil at p ──
# Cached by p's local situation and guessed speed: the stencils, which cells they use, and their moments
# sum_q w_q dx^i dt^j for each Taylor term. The score at p is then |moments @ Taylor coefficients at p|.
_sets_cache = {}
def admissible(p):
    """(stencils as tuples of cells, cell-usage matrix, scores at p). Cells that land on the same point count once."""
    key = (tuple((place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS)), a_build(p))
    if key not in _sets_cache:
        cells, seen = [], set()
        for k in range(N_CELLS):
            if place(p, k) not in seen:
                seen.add(place(p, k)); cells.append(k)
        sets = list(itertools.combinations(cells, min(N_NB, len(cells))))
        off  = np.array([[(place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in s] for s in sets])   # (M, n, 2)
        DX, DT = off[..., 0] * dx, off[..., 1] * dt
        W, ok  = solve_weights_many(DX, DT, a_build(p))                # all candidate stencils in one solve
        sets   = [s for s, o in zip(sets, ok) if o]
        mom    = np.stack([(W[ok] * DX[ok]**i * DT[ok]**j).sum(1) for i, j in TERMS], axis=1)
        uses   = np.zeros((len(sets), N_CELLS), dtype=bool)
        for i, s in enumerate(sets):
            uses[i, list(s)] = True
        _sets_cache[key] = (sets, uses, mom)
    sets, uses, mom = _sets_cache[key]
    c = taylor_coeffs(guess_derivs(p))
    return sets, uses, np.abs(mom @ np.array([c[t] for t in TERMS]))

def allowed_sets(m, p):
    """Admissible stencils at p that use no forbidden (cycle-closing) cell: (sets, uses, scores) restricted to them.
    Cached on the map for the current point (it only changes when a stencil is placed), so the 5 picks of a point share it."""
    key = (p, len(m["stencil"]))
    if m.get("_allowed_key") != key:
        sets, uses, sc = admissible(p)
        bad = forbidden(m, p)
        ok = ~uses[:, bad].any(axis=1) if bad else np.ones(len(sets), dtype=bool)
        m["_allowed"] = ([s for s, o in zip(sets, ok) if o], uses[ok], sc[ok]) if bad else (sets, uses, sc)
        m["_allowed_key"] = key
    return m["_allowed"]

def pick_mask(m, p, picks):
    """Cell k may be picked next if some allowed stencil contains the picks so far plus k."""
    _, uses, _ = allowed_sets(m, p)
    ok = uses[:, picks].all(axis=1) if picks else np.ones(len(uses), dtype=bool)
    mask = uses[ok].any(axis=0)
    mask[picks] = False
    return mask

def lowest_score(m, p):
    sets, _, sc = allowed_sets(m, p)
    return [place(p, k) for k in sets[int(np.argmin(sc))]]

# ── Report of a finished map: points, true error at z*, exact bound sum |beta delta| (u_true ONLY for reporting) ──
# ── Report of a finished map: points, true error at z*, exact bound sum |beta delta| (u_true ONLY for reporting) ──
# with the weights the forward pass actually used (live speed); beta is recomputed with them, so e(z*) = sum beta delta
def report(m):
    """Points, true error at z*, exact bound sum |beta delta| with the weights the forward pass used (u_true ONLY here)."""
    z = m["z_star"]
    u_hat, W = evaluate(m), m["w_used"]
    beta = {z: 1.0}
    for p in forward_order(m)[::-1]:                              # every user before the points it uses
        for q, wq in zip(m["stencil"][p][0], W[p]):
            if not is_known(q):
                beta[q] = beta.get(q, 0.0) + beta[p] * wq
    delta = {p: Ug(p) - W[p] @ np.array([Ug(q) for q in m["stencil"][p][0]]) for p in m["stencil"]}
    return {"points": len(m["stencil"]), "error": abs(Ug(z) - u_hat[z]),
            "bound": sum(abs(beta[p] * delta[p]) for p in delta)}


# ── Units of the reward ──
#   accuracy: the estimated error share |beta_p| s_p, in units of EPS (a fixed target error for the problem, same for every query)
#   cost:     new points, in units of N_LW (the point count of the Lax-Wendroff map of the same z*: "a classical solve")
EPS = {"walls": 1e-5, "inflow_smooth": 1e-3, "inflow_front": 1e-4}[PROBLEM]
_nlw_cache = {}
def n_lw(z):
    """Points of the LW map of z* (the cost unit)."""
    if z not in _nlw_cache:
        _nlw_cache[z] = len(build_map(z, lw)["stencil"])
    return _nlw_cache[z]

# ── State switch: per-cell physics features ──
#   STATE_X = True adds, for every window cell q, two numbers to the state:
#     xi_q / dx : the cell's offset from the foot of the characteristic (with p's guessed speed): which side is upwind, how far
#     R_q       : the cell's Taylor remainder, the part of the local Taylor polynomial the three conditions do NOT cancel,
#                 so the score of ANY stencil at p is |sum_q w_q R_q|  (signed log of R_q / dx^3, which spans ~1 .. 1000)
STATE_X = True
def cell_features(p):
    d, ap = guess_derivs(p), a_build(p)
    off = np.array([(place(p, k)[0] - p[0], place(p, k)[1] - p[1]) for k in range(N_CELLS)], dtype=float)
    DX, DT = off[:, 0] * dx, off[:, 1] * dt
    xi = DX - ap * DT
    P = sum(c * DX**i * DT**j for (i, j), c in taylor_coeffs(d).items())
    R = P - d[1] * xi - d[2] * (0.5 * xi**2 + alpha * DT)
    return np.concatenate([xi / dx, np.sign(R) * np.log10(1 + np.abs(R) / dx**3)])




# ── Self-contained: the rules this cell needs (overrides any earlier, broken copies) ──
def score_sum(m):
    """B = sum |beta_p| s_p: the map's estimated error bound."""
    return sum(abs(m["beta"][p]) * point_score(p, *m["stencil"][p]) for p in m["stencil"])

def greedy_rule(lam):
    """At every point the allowed stencil with the best one-step reward (beta_p as known when p is expanded)."""
    def choose(m, p):
        sets, uses, sc = allowed_sets(m, p)
        new = np.array([not is_known(place(p, k)) and place(p, k) not in m["beta"] for k in range(N_CELLS)])
        cost = abs(m["beta"][p]) * sc / EPS + lam * (uses.astype(float) @ new) / n_lw(m["z_star"])
        return [place(p, k) for k in sets[int(np.argmin(cost))]]
    return choose

