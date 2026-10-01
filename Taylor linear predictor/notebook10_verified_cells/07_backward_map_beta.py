# ── The map, built backward from z* ──
def start_map(z_star):
    """Empty map for the query z*: only z* is pending, with influence 1."""
    return {"z_star":  z_star,
            "pending": [(-z_star[1], z_star[0])],   # heap: always pop the LATEST t first
            "beta":    {z_star: 1.0},               # influence of each computed point on z*
            "stencil": {}}                          # point -> (neighbour points, weights)

def next_point(m):
    """The point that gets its stencil next (latest t), without removing it."""
    neg_it, ix = m["pending"][0]
    return ix, -neg_it

def expand(m, cells):
    """Give the next point the stencil of the picked cells and pass its influence down.
    Returns (point, new points created, its term |beta_p| * s_p)."""
    neg_it, ix = heapq.heappop(m["pending"])
    p = (ix, -neg_it)
    nbs, w = stencil(*p, cells)
    m["stencil"][p] = (nbs, w)
    n_new = 0
    for q, wq in zip(nbs, w):
        if is_known(*q):
            continue                                 # IC or wall: exact, a leaf
        if q not in m["beta"]:
            m["beta"][q] = 0.0
            heapq.heappush(m["pending"], (-q[1], q[0]))
            n_new += 1
        m["beta"][q] += m["beta"][p] * wq
    return p, n_new, abs(m["beta"][p]) * score(*p, nbs, w)

# check: build a whole map where every point uses "5 in a row, one level down"
m = start_map((50, 6))
B = 0.0
while m["pending"]:
    p, n_new, term = expand(m, [0, 1, 2, 3, 4])
    B += term
print("points computed:", len(m["stencil"]), "| sum of |beta| * score:", f"{B:.2e}")
for it in range(6, 0, -1):
    print(f"  level {it}: {sum(1 for q in m['beta'] if q[1] == it):2d} points, sum of beta = {sum(b for q, b in m['beta'].items() if q[1] == it):.3f}")
