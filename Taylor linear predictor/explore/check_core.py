"""Reproduce the notebook checks with explore/core.py: LW map identity at (10, 40) and score vs true local error."""
from core import *

z = (10, 40)
m = build_map(z, lw)
err = Ug(z) - evaluate(m)[z]
delta = {p: Ug(p) - w @ np.array([Ug(q) for q in nbs]) for p, (nbs, w) in m["stencil"].items()}
print(f"LW map at {z}: {len(m['stencil'])} points, error {abs(err):.2e}, "
      f"identity {err:.10e} vs {sum(m['beta'][p] * delta[p] for p in delta):.10e}")

# score vs true local error (cell 6 check, exact derivatives)
U = lambda q: u_true(q[0] * dx, q[1] * dt)[0]
rng = np.random.default_rng(0)
cells = [(i, j) for j in (-1, -2, -3) for i in range(-3, 4)]
S, D = [], []
while len(S) < 300:
    p = (int(rng.integers(3, nx - 3)), int(rng.integers(3, nt)))
    nb = [cells[k] for k in rng.choice(len(cells), 5, replace=False)]
    DX, DT = np.array([q[0] for q in nb]) * dx, np.array([q[1] for q in nb]) * dt
    ap = float(speed(p[0] * dx))
    w = solve_weights(DX, DT, ap)
    if w is None:
        continue
    d = x_derivs(lambda x: u_true(x, p[1] * dt), p[0] * dx)
    S.append(score(DX, DT, w, d, ap)); D.append(abs(U(p) - w @ np.array([U((p[0] + q[0], p[1] + q[1])) for q in nb])))
print(f"corr(log score, log true error) = {np.corrcoef(np.log(S), np.log(D))[0, 1]:.2f}")
