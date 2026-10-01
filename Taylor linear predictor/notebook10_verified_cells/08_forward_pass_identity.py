# ── Forward pass: compute the values bottom-up, from the IC and walls to z* ──
def evaluate(m):
    """u_hat at every computed point of a finished map (no pending points left)."""
    u_hat = {}
    value = lambda q: known_value(*q) if is_known(*q) else u_hat[q]
    for p in sorted(m["stencil"], key=lambda p: p[1]):          # increasing t: neighbours are ready
        nbs, w = m["stencil"][p]
        u_hat[p] = sum(wq * value(q) for q, wq in zip(nbs, w))
    return u_hat

# check on the map from Cell 7 (u_true used ONLY for checking)
z = m["z_star"]
u_hat = evaluate(m)
err = u_true(z[0] * dx, z[1] * dt)[0] - u_hat[z]
print(f"error at z* = {z}: {abs(err):.2e}   (FD: {fd_reference(*z)[1]:.2e} with {fd_reference(*z)[2]} points)")

# the error splits exactly into one term per point: e(z*) = sum_p beta_p * delta_p
U = lambda q: u_true(q[0] * dx, q[1] * dt)[0]
delta = {p: U(p) - sum(wq * U(q) for q, wq in zip(nbs, w)) for p, (nbs, w) in m["stencil"].items()}
print(f"true error {err:+.6e}   sum of beta*delta {sum(m['beta'][p] * delta[p] for p in delta):+.6e}")
