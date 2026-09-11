"""Program-level figures. Run from exploration/:  python figures/make_figures.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from core.pde import Problem
from core import stencil as st

OUT = os.path.dirname(os.path.abspath(__file__))
p = Problem()
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------------------------ FIG 1
# The Pareto frontier: how much accuracy does ||w||_1 buy?
# min |sum w dt^2|  s.t.  A w = b,  ||w||_1 <= L.  (LP via w = pp - nn, epigraph on t.)
def achievable(dxi, dti, L):
    A, b = st.rows_taylor(dxi, dti, p)
    n = len(dxi)
    g = dti**2 / np.max(dti**2)
    # vars: pp(n), nn(n), t(1)
    c = np.zeros(2 * n + 1); c[-1] = 1.0
    Aeq = np.hstack([A, -A, np.zeros((A.shape[0], 1))])
    Aub = np.vstack([
        np.concatenate([g, -g, [-1.0]]),            #  g.w - t <= 0
        np.concatenate([-g, g, [-1.0]]),            # -g.w - t <= 0
        np.concatenate([np.ones(2 * n), [0.0]]),    #  ||w||_1 <= L
    ])
    bub = np.array([0.0, 0.0, L])
    r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=b,
                bounds=[(0, None)] * (2 * n) + [(0, None)], method="highs")
    return r.x[-1] if r.status == 0 else np.nan

rng = np.random.default_rng(3)
m, npts = 6, 12
dxi = rng.integers(-m, m + 1, npts) * p.dx
dti = -rng.integers(1, 9, npts) * p.dt
Ls = np.linspace(1.0, 2.2, 60)
vals = np.array([achievable(dxi, dti, L) for L in Ls])
jensen = (np.linalg.lstsq(*st.rows_taylor(dxi, dti, p), rcond=None)[0] @ dti) ** 2

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].semilogy(Ls, np.maximum(vals, 1e-20) / np.max(dti**2), lw=2, color="#2a6fb5")
ax[0].axvline(1.0, color="#c0392b", ls="--", lw=1.2)
ax[0].text(1.02, 0.5, "positivity\n‖w‖₁ = 1", color="#c0392b", fontsize=8,
           transform=ax[0].get_xaxis_transform(), va="center")
ax[0].set_xlabel("‖w‖₁ budget"); ax[0].set_ylabel("achievable |Σ wᵢ Δtᵢ²|  (normalised)")
ax[0].set_title("Accuracy has a price in stability", fontsize=10)

# ------------------------------------------------------------------ FIG 1b
# Jensen: the dt^2 moment vs its lower bound, over positive stencils
xs, ys = [], []
for _ in range(1500):
    k = rng.integers(5, 12)
    dx2 = rng.integers(-6, 7, k) * p.dx
    dt2 = -rng.integers(1, 9, k) * p.dt
    A, b = st.rows_taylor(dx2, dt2, p)
    w = st.solve_positive(A, b)
    if w is None:
        continue
    xs.append((w @ dt2) ** 2); ys.append(w @ dt2**2)
xs, ys = np.array(xs), np.array(ys)
ax[1].loglog(xs, ys, ".", ms=2.5, alpha=0.4, color="#2a6fb5")
lim = [min(xs.min(), ys.min()) * 0.7, max(xs.max(), ys.max()) * 1.4]
ax[1].plot(lim, lim, "--", color="#c0392b", lw=1.3, label="Jensen bound  Σw Δt² = (Σw Δt)²")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("(Σ wᵢ Δtᵢ)²"); ax[1].set_ylabel("Σ wᵢ Δtᵢ²")
ax[1].set_title(f"No positive stencil crosses it  (0/{len(xs)})", fontsize=10)
ax[1].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pareto.png"); plt.close(fig)
print("wrote fig_pareto.png")

# ------------------------------------------------------------------ FIG 2
# Query-driven cost: moment stencil vs explicit FD
def gauss_w(m_half, tau, order):
    idx = np.arange(-m_half, m_half + 1)
    A, b = st.rows_moment(idx * p.dx, np.full(len(idx), -tau), order, p)
    return st.solve_positive(A, b)

Js = [25, 50, 100, 200, 400, 800]
fd_cost, fd_err, mo_cost, mo_err = [], [], [], []
x_star = 0.5; jstar = int(round(x_star / p.dx))
for J in Js:
    tau = J * p.dt; u_ref = p.u_true(x_star, tau)
    fd_cost.append(sum(min(p.nx - 2, 2 * (J - q) + 1) for q in range(J)))
    # march FTCS
    u = p.u_true(p.x, 0.0).copy()
    w3 = st.solve_positive(*st.rows_taylor(np.array([-p.dx, 0, p.dx]), np.array([-p.dt] * 3), p))
    for _ in range(J):
        u = np.concatenate([[0.0], np.convolve(u, w3[::-1], "same")[1:-1], [0.0]])
    fd_err.append(abs(u[jstar] - u_ref))
    sig = np.sqrt(2 * p.alpha * tau)
    best = None
    for g in [2, 3, 4]:
        mh = int(np.ceil(g * sig / p.dx))
        if jstar - mh < 0 or jstar + mh > p.nx - 1:
            continue
        for order in [4, 6, 8]:
            w = gauss_w(mh, tau, order)
            if w is None:
                continue
            e = abs(w @ p.u_true(p.x[jstar - mh: jstar + mh + 1], 0.0) - u_ref)
            if best is None or e < best[1]:
                best = (2 * mh + 1, e)
    mo_cost.append(best[0]); mo_err.append(best[1])

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].loglog(Js, fd_cost, "o-", color="#c0392b", label="explicit FD (fill the cone)")
ax[0].loglog(Js, mo_cost, "s-", color="#2a6fb5", label="moment stencil (one step)")
ax[0].set_xlabel("t* (CFL steps)"); ax[0].set_ylabel("evaluations for ONE query")
ax[0].set_title("Cost of a single query", fontsize=10); ax[0].legend(fontsize=7)
ax[1].loglog(fd_cost, fd_err, "o-", color="#c0392b", label="explicit FD")
ax[1].loglog(mo_cost, mo_err, "s-", color="#2a6fb5", label="moment stencil")
ax[1].set_xlabel("evaluations"); ax[1].set_ylabel("error at the query")
ax[1].set_title("Work–precision, query-driven", fontsize=10); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_query.png"); plt.close(fig)
print("wrote fig_query.png")
for J, fc, fe, mc, me in zip(Js, fd_cost, fd_err, mo_cost, mo_err):
    print(f"  t*={J:>4}: FD {fc:>7} / {fe:.2e}   moment {mc:>4} / {me:.2e}"
          f"   -> {fc/mc:>6.0f}x cheaper, {fe/me:>7.1f}x more accurate")
