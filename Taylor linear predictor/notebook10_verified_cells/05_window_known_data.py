# ── Downward window: neighbours only from earlier time levels ──
R_X, K_T = 2, 2                     # |dix| <= 2 cells,  dit in {-1, -2}
WINDOW  = [(dix, dit) for dit in range(-1, -K_T - 1, -1) for dix in range(-R_X, R_X + 1)]
N_CELLS = len(WINDOW)
N_NB    = 5                         # neighbours per stencil

U0 = u_true(np.arange(nx) * dx, 0.0)      # initial condition on the grid

def is_known(ix, it):
    """Initial condition or wall: the value is given, not computed."""
    return it == 0 or ix == 0 or ix == nx - 1

def known_value(ix, it):
    return U0[ix] if it == 0 else 0.0

def place(ix, it, a):
    """Grid point of window cell a seen from (ix, it). Past a wall or below t = 0 -> moved onto it."""
    dix, dit = WINDOW[a]
    return (min(max(ix + dix, 0), nx - 1), max(it + dit, 0))

def distinct_cells(ix, it):
    """One window cell per distinct grid point (cells moved onto the same wall/IC point count once)."""
    out, used = [], set()
    for a in range(N_CELLS):
        q = place(ix, it, a)
        if q not in used:
            used.add(q)
            out.append(a)
    return out

def n_picks(ix, it):
    """Stencil size at (ix, it): 5, or fewer if the window has fewer distinct points."""
    return min(N_NB, len(distinct_cells(ix, it)))

print(f"{N_CELLS} window cells:", WINDOW)
print("from (50, 10):", [place(50, 10, a) for a in range(N_CELLS)])
print("from (1, 1):  ", [place(1, 1, a) for a in range(N_CELLS)])
print("distinct points / stencil size at (50,10):", len(distinct_cells(50, 10)), n_picks(50, 10),
      "| at (1,1):", len(distinct_cells(1, 1)), n_picks(1, 1))
