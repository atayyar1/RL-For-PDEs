# ── Bigger action space: 21-cell window (|dix| <= 3, 3 levels down), reward measured against FD ──
R_X, K_T = 3, 3
WINDOW  = [(dix, dit) for dit in range(-1, -K_T - 1, -1) for dix in range(-R_X, R_X + 1)]
N_CELLS = len(WINDOW)
N_OBS   = 3 * N_CELLS + 5
for cache in (_w_cache, _sets_cache, _ref_cache):
    cache.clear()

_big = {}
def _situation(ix, it):
    return tuple((q[0] - ix, q[1] - it) for q in (place(ix, it, a) for a in range(N_CELLS)))

def big_sets(ix, it):
    """All admissible stencils here, checked in one vectorised pass: (list of sets, cell-usage matrix, scores)."""
    key = _situation(ix, it)
    if key not in _big:
        cells = distinct_cells(ix, it)
        combos = np.array(list(itertools.combinations(cells, min(N_NB, len(cells)))))
        offs = np.array(key, dtype=float)[combos]
        DX, DT = offs[..., 0] * dx, offs[..., 1] * dt
        h  = np.maximum(np.abs(DX).max(1), np.sqrt(alpha * np.abs(DT).max(1)))[:, None]
        xi = DX - c * DT
        A  = np.stack([np.ones_like(DX), xi / h, (0.5 * xi**2 + alpha * DT) / h**2], axis=1)
        sv = np.linalg.svd(A, compute_uv=False)
        ok = (sv[:, -1] > sv[:, 0] * 1e-12 * max(A.shape[1:])) & (sv[:, 0] <= COND_MAX * sv[:, -1])
        w  = np.linalg.pinv(A)[:, :, 0]
        ok &= np.abs(w).sum(1) <= W_MAX
        m3 = np.einsum('mk,mk->m', w, xi**3 + 6 * alpha * DT * xi)
        m4 = np.einsum('mk,mk->m', w, xi**4 + 12 * alpha * DT * xi**2 + 12 * (alpha * DT)**2)
        sc = np.abs(m3) / (6 * L_SOL**3) + np.abs(m4) / (24 * L_SOL**4)
        uses = np.zeros((ok.sum(), N_CELLS), dtype=bool)
        np.put_along_axis(uses, combos[ok], True, axis=1)
        _big[key] = ([tuple(cm) for cm in combos[ok]], uses, sc[ok])
    return _big[key]

admissible_sets = lambda ix, it: big_sets(ix, it)[0]

def choose_ref(m, p):
    sets, _, sc = big_sets(*p)
    return sets[int(np.argmin(sc))]

def pick_mask(ix, it, picks):
    """Vectorised: cell a is allowed if some admissible stencil contains the picks so far plus a."""
    _, uses, _ = big_sets(ix, it)
    ok = uses[:, picks].all(axis=1) if picks else np.ones(len(uses), dtype=bool)
    mask = uses[ok].any(axis=0)
    mask[picks] = False
    return mask

def reachable_score(ix, it, picks):
    _, uses, sc = big_sets(ix, it)
    rel = np.log10(sc) - np.log10(sc).min()
    ok = uses[:, picks].all(axis=1) if picks else np.ones(len(rel), dtype=bool)
    out = np.where(uses[ok], rel[ok, None], 3.0).min(axis=0) if ok.any() else np.full(N_CELLS, 3.0)
    out[picks] = 3.0
    return np.minimum(out, 3.0)

def fd_map(z):
    """FD's cone as a map in our format: 3-point stencils one level down, influence passed down from z*."""
    zx, zt = z
    m = {"z_star": z, "beta": {z: 1.0}, "stencil": {}, "pending": []}
    for it in range(zt, 0, -1):
        for ix in range(max(1, zx - (zt - it)), min(nx - 2, zx + (zt - it)) + 1):
            p = (ix, it)
            nbs = [(ix - 1, it - 1), (ix, it - 1), (ix + 1, it - 1)]
            m["stencil"][p] = (nbs, W_FD)
            for q, wq in zip(nbs, W_FD):
                if not is_known(*q):
                    m["beta"][q] = m["beta"].get(q, 0.0) + m["beta"][p] * wq
    return m

def reference(z):
    """Now FD: its score sum and its point count. Every return reads 'relative to finite differences'."""
    if z not in _ref_cache:
        m = fd_map(z)
        _ref_cache[z] = (sum(abs(m["beta"][p]) * score(*p, nbs, w) for p, (nbs, w) in m["stencil"].items()), len(m["stencil"]))
    return _ref_cache[z]

# checks
print(f"{N_CELLS} cells | admissible stencils in the interior: {len(admissible_sets(50, 20))}")
e32 = PendingStateEnv(queries=[(50, 20)], lam=1.0); o, _ = e32.reset(seed=0)
rng, total, steps, d, t0 = np.random.default_rng(0), 0.0, 0, False, time.time()
while not d:
    o, r, d, _, info = e32.step(rng.choice(np.flatnonzero(e32.action_masks()))); total += r; steps += 1
print(f"obs size {o.shape[0]} | random episode: {steps} picks, {info['n_points']} points, score {info['score_ratio']:.1f}x FD, "
      f"{steps / (time.time() - t0):.0f} picks/s")
print(f"sum of rewards {total:.3f} = -score_ratio - lam*(points-1)/N_FD = {-info['score_ratio'] - (info['n_points'] - 1) / e32.N_ref:.3f}")
