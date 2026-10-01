# ── Reference map and greedy baseline ──
def build_map(z_star, choose):
    """Build a whole map; choose(m, p) returns the cells for point p. Returns (map, sum |beta| s, points)."""
    m, B = start_map(z_star), 0.0
    while m["pending"]:
        p = next_point(m)
        _, _, term = expand(m, choose(m, p))
        B += term
    return m, B, len(m["stencil"])

def n_new_points(m, p, cells):
    """How many new computed points this stencil would create."""
    return sum(1 for q in set(place(*p, a) for a in cells) if not is_known(*q) and q not in m["beta"])

def s_of(p, cells):
    return score(*p, *stencil(*p, cells))

# Reference: every point takes its lowest-score admissible stencil (ignores influence and cost)
choose_ref = lambda m, p: min(admissible_sets(*p), key=lambda s: s_of(p, s))

# Greedy: every point takes the stencil with the lowest immediate reward cost
def make_greedy(B_ref, N_ref, lam):
    return lambda m, p: min(admissible_sets(*p),
                            key=lambda s: abs(m["beta"][p]) * s_of(p, s) / B_ref + lam * n_new_points(m, p, s) / N_ref)

def true_error(m):
    z = m["z_star"]
    return abs(evaluate(m)[z] - u_true(z[0] * dx, z[1] * dt)[0])    # u_true ONLY for reporting

for z in [(50, 6), (50, 10), (50, 20)]:
    m_ref, B_ref, N_ref = build_map(z, choose_ref)
    print(f"z* = {z}")
    print(f"   FD        : error {fd_reference(*z)[1]:.2e}, {fd_reference(*z)[2]:4d} points")
    print(f"   reference : error {true_error(m_ref):.2e}, {N_ref:4d} points, score sum {B_ref:.2e}")
    for lam in [0.0, 1.0]:
        m_g, B_g, N_g = build_map(z, make_greedy(B_ref, N_ref, lam))
        print(f"   greedy lam={lam:.0f}: error {true_error(m_g):.2e}, {N_g:4d} points, score sum {B_g:.2e}")
