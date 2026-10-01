# ── Stencil from picked cells, and its geometry score ──
L_SOL = 1 / (2 * np.pi)            # solution length scale: shortest mode of the IC, sin(2 pi x)

_w_cache = {}
def stencil(ix, it, cells):
    """(neighbour points, weights) for the picked cells, weights computed at the PLACED positions.
    None if the stencil is inadmissible."""
    nbs  = [place(ix, it, a) for a in sorted(cells)]
    offs = tuple((q[0] - ix, q[1] - it) for q in nbs)
    if offs not in _w_cache:
        _w_cache[offs] = solve_weights(np.array([o[0] for o in offs]) * dx, np.array([o[1] for o in offs]) * dt)
    w = _w_cache[offs]
    return None if w is None else (nbs, w)

def score(ix, it, nbs, w):
    """Estimated local error of the stencil, from geometry only:
    what is left of the Taylor series after the 3 rows cancel T0, T1, T2."""
    DX = np.array([q[0] - ix for q in nbs]) * dx
    DT = np.array([q[1] - it for q in nbs]) * dt
    xi = DX - c * DT
    m3 = w @ (xi**3 + 6 * alpha * DT * xi)                                  # leftover on T3
    m4 = w @ (xi**4 + 12 * alpha * DT * xi**2 + 12 * (alpha * DT)**2)       # leftover on T4
    return abs(m3) / (6 * L_SOL**3) + abs(m4) / (24 * L_SOL**4)

# check at p = (50, 20): score vs the true local error (u_true used ONLY for this check)
p = (50, 20)
for name, cells in [("row 1 level down ", [0, 1, 2, 3, 4]), ("row 2 levels down", [5, 6, 7, 8, 9])]:
    nbs, w = stencil(*p, cells)
    delta = u_true(p[0] * dx, p[1] * dt)[0] - sum(wq * u_true(q[0] * dx, q[1] * dt)[0] for q, wq in zip(nbs, w))
    print(f"{name}: weights {np.round(w, 3)}  score {score(*p, nbs, w):.1e}  true local error {abs(delta):.1e}")

print("4 IC points at (1, 1):", stencil(1, 1, distinct_cells(1, 1)))
