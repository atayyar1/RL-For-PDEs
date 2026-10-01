# ── FD reference: the tight central stencil at every point ──
r, nu = alpha * dt / dx**2, c * dt / dx
W_FD = np.array([r + nu / 2, 1 - 2 * r, r - nu / 2])      # weights for (x-dx, x, x+dx)
_fd_cache = {}
def fd_reference(ix, it):
    """FD at z* = (ix, it): (u_fd, E_FD, C_FD) = value, error, number of computed points."""
    if (ix, it) in _fd_cache:
        return _fd_cache[(ix, it)]
    u = u_true(np.arange(nx) * dx, 0.0)                    # level 0: initial condition
    for _ in range(it):
        u[1:-1] = W_FD[0] * u[:-2] + W_FD[1] * u[1:-1] + W_FD[2] * u[2:]
        u[0] = u[-1] = 0.0                                   # boundaries
    u_fd = u[ix]
    E_FD = abs(u_fd - u_true(ix * dx, it * dt)[0])
    C_FD = sum(min(nx - 2, ix + j) - max(1, ix - j) + 1 for j in range(it))
    _fd_cache[(ix, it)] = (u_fd, E_FD, C_FD)
    return _fd_cache[(ix, it)]

for z in [(50, 6), (50, 10), (50, 20)]:
    u_fd, E_FD, C_FD = fd_reference(*z)
    print(f"FD at z* = {z}: error {E_FD:.2e} with {C_FD} computed points")

COND_MAX = 1e4
W_MAX    = 2     # cap on ||w||_1, the per-level error-amplification factor (thesis Thm 2.7)

def solve_weights(dxs, dts, w_max=W_MAX):
    """Min-norm weights for the 3 PDE-substituted Taylor rows (constant, u_x, u_xx).
    None if rank-deficient, ill-conditioned (cond > COND_MAX), or ||w||_1 > w_max."""
    h = max(np.abs(dxs).max(), np.sqrt(alpha * np.abs(dts).max()))
    A = np.array([np.ones_like(dxs),
                  (dxs - c * dts) / h,
                  (0.5 * (dxs - c * dts)**2 + alpha * dts) / h**2])
    if np.linalg.matrix_rank(A) < 3 or np.linalg.cond(A) > COND_MAX:
        return None
    w = np.linalg.pinv(A) @ np.array([1.0, 0.0, 0.0])
    return w if np.abs(w).sum() <= w_max else None

w3 = solve_weights(np.array([-1, 0, 1]) * dx, np.array([-1, -1, -1]) * dt)
print("3 points one level down:", np.round(w3, 5))
print("Lax-Wendroff weights:   ", np.round([r + nu/2 + nu**2/2, 1 - 2*r - nu**2, r - nu/2 + nu**2/2], 5))
