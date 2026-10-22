"""Fair comparison: classical solves at the unseen Burgers queries (error at z*, cost = grid points evaluated).
    PDE=burgers python explore/classical_burgers.py
Map recipes from combo_*.json / sweep_*.json for reference (their cost = map points; the guess is extra, shared)."""
import os, json
os.environ["PDE"] = "burgers"
from stop import *
from residual import coarse_solution
from evaluate import Q_UNSEEN, gm

rows = {}
for name, K, scheme in [("LW full grid", 1, "lw"), ("FD-5 + RK4 full grid", 1, "fd5"), ("FD-5 + RK4 2x coarser", 2, "fd5")]:
    xc, tc, U = coarse_solution(K, scheme)
    cost = len(xc) * (len(tc) - 1) * (4 if scheme == "fd5" else 1)
    errs = []
    for z in Q_UNSEEN:
        t = z[1] * dt
        j = int(np.argmin(np.abs(tc - t)))                       # solves are on their own time steps: nearest level
        errs.append(abs(np.interp(z[0] * dx, xc, U[j]) - Ug(z)) if abs(tc[j] - t) < 1e-9 else abs(np.interp(z[0] * dx, xc, U[j]) - u_true([z[0] * dx], tc[j])[0]))
    rows[name] = (cost, errs)
B = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_inflow_front.json")))["R"]
lw_err = [B[str(z)]["LW map"]["error"] for z in Q_UNSEEN]
lw_pts = [B[str(z)]["LW map"]["points"] for z in Q_UNSEEN]
print(f"notebook LW map: median {np.median(lw_pts):.0f} points, median error {np.median(lw_err):.1e}")
for n, (c, e) in rows.items():
    print(f"{n:24s}: cost {c:6d} grid points (all queries at once) | median error {np.median(e):.1e} | "
          f"geo-mean error / notebook LW map x{gm(np.array(e) / np.array(lw_err)):.2f}")
