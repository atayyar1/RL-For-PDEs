# ── Admissible stencils at a point, and the mask for the next pick ──
_sets_cache = {}
def admissible_sets(ix, it):
    """Every choice of n_picks(ix, it) distinct cells whose stencil is admissible."""
    key = tuple((q[0] - ix, q[1] - it) for q in (place(ix, it, a) for a in range(N_CELLS)))  # the local situation
    if key not in _sets_cache:
        cells = distinct_cells(ix, it)
        _sets_cache[key] = [s for s in itertools.combinations(cells, n_picks(ix, it))
                            if stencil(ix, it, s) is not None]
    return _sets_cache[key]

def pick_mask(ix, it, picks):
    """Cell a may be picked next if some admissible stencil contains the picks so far plus a."""
    mask = np.zeros(N_CELLS, dtype=bool)
    for s in admissible_sets(ix, it):
        if set(picks) <= set(s):
            mask[[a for a in s if a not in picks]] = True
    return mask

for p in [(50, 10), (50, 1), (1, 1), (1, 5)]:
    print(f"at {p}: {len(admissible_sets(*p)):3d} admissible stencils of size {n_picks(*p)} | "
          f"first pick allowed on {pick_mask(*p, []).sum()} cells")
print("at (50, 10) after picking cells 0 and 9:", np.flatnonzero(pick_mask(50, 10, [0, 9])))
