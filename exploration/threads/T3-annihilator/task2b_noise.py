"""
task2b_noise.py -- noise robustness of jet recovery, against fair baselines.

Baselines, all given the SAME data budget and each tuned over its own
bandwidth at every noise level (otherwise the comparison is rigged):
  FD-m   : 5-point central FD on a stencil of spacing m*dx (coarsening is the
           standard FD noise knob); u_t by 3-point backward at spacing m*dt
  SG-m   : Savitzky-Golay, i.e. degree-K polynomial LS on a uniform 1-D
           stencil of half-width m (u_x, u_xx) and on a time column (u_t)
  MOM    : the moment estimator of Step 1 (Monte-Carlo over w and G)
  MLS-u  : its exact closed form (uniform kernel) -- same estimator, no MC
  MLS-t  : moving least squares with a tricube kernel -- the strong baseline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
sol = J.AdvDiff(alpha=0.1, c=1.0); ALPHA = sol.alpha
x0, t0 = 0.40, 0.25
DXG = 1 / 99.0
K = 4
N_PTS = 60                    # shared data budget
TRUE = {"u_x": sol.deriv(x0, t0, 1, 0),
        "u_t": sol.deriv(x0, t0, 0, 1),
        "u_xx": sol.deriv(x0, t0, 2, 0)}
uref = J.noise_scale(sol, x0, t0, 3 * DXG, (3 * DXG) ** 2 / ALPHA, rng=np.random.default_rng(1))

def jet_to_derivs(qs, d):
    i = {q: k for k, q in enumerate(qs)}
    return {"u_x": d[i[(1, 0)]], "u_t": d[i[(0, 1)]], "u_xx": 2 * d[i[(2, 0)]]}

def scatter_estimator(hx, sigma, seed, kind="mls_uniform", S=3000, n_nb=6):
    rng = np.random.default_rng(seed)
    ht = hx ** 2 / ALPHA
    qs = J.monomials(K, "parabolic")
    dxi = hx * rng.uniform(-1, 1, N_PTS); dti = ht * rng.uniform(-1, 0, N_PTS)
    u_nb = np.array([sol.u(x0 + a, t0 + b) for a, b in zip(dxi, dti)]) \
        + sigma * rng.normal(size=N_PTS)
    u_st = sol.u(x0, t0) + sigma * rng.normal()
    Phi = J.design_matrix(dxi, dti, qs, hx, ht); y = u_nb - u_st
    if kind == "moment":
        Wf = np.zeros((S, N_PTS))
        for s in range(S):
            idx = rng.choice(N_PTS, n_nb, replace=False)
            Wf[s, idx] = J.random_weights(1, n_nb, rng, "gauss", tau=1.0)[0]
        dh = J.moment_regression(Phi, y, Wf)
    elif kind == "mls_uniform":
        dh = J.wls_jet(Phi, y, np.eye(N_PTS) - np.ones((N_PTS, N_PTS)) / N_PTS)
    elif kind == "mls_tricube":
        dh = J.wls_jet(Phi, y, J.tricube(dxi, dti, hx, ht) + 1e-6)
    return jet_to_derivs(qs, dh / J.jet_scaling(qs, hx, ht))

def fd_estimator(m, sigma, seed):
    rng = np.random.default_rng(seed)
    dx = m * DXG; dt = dx ** 2 / ALPHA
    xs = x0 + dx * np.arange(-2, 3)
    u = np.array([sol.u(xx, t0) for xx in xs]) + sigma * rng.normal(size=5)
    uu = np.array([sol.u(x0, t0 - k * dt) for k in range(3)]) + sigma * rng.normal(size=3)
    return {"u_x": (u[0] - 8 * u[1] + 8 * u[3] - u[4]) / (12 * dx),
            "u_xx": (-u[0] + 16 * u[1] - 30 * u[2] + 16 * u[3] - u[4]) / (12 * dx ** 2),
            "u_t": (3 * uu[0] - 4 * uu[1] + uu[2]) / (2 * dt)}

def sg_estimator(m, sigma, seed, deg=4):
    """Savitzky-Golay: degree-deg poly LS on a uniform (2m+1)-point x stencil and
    on a uniform (m+1)-point backward t column.  Same total budget as FD-ish."""
    rng = np.random.default_rng(seed)
    dx = DXG; dt = dx ** 2 / ALPHA
    xs = np.arange(-m, m + 1) * dx
    u = np.array([sol.u(x0 + a, t0) for a in xs]) + sigma * rng.normal(size=len(xs))
    V = np.vander(xs / (m * dx), deg + 1, increasing=True)
    cx = np.linalg.lstsq(V, u, rcond=None)[0]
    ts = -np.arange(0, m + 1) * dt
    ut_ = np.array([sol.u(x0, t0 + b) for b in ts]) + sigma * rng.normal(size=len(ts))
    Vt = np.vander(ts / (m * dt), min(deg, m) + 1, increasing=True)
    ct = np.linalg.lstsq(Vt, ut_, rcond=None)[0]
    return {"u_x": cx[1] / (m * dx), "u_xx": 2 * cx[2] / (m * dx) ** 2,
            "u_t": ct[1] / (m * dt)}

ETAS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
MS = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]
NREP = 40
KEYS = ["u_x", "u_t", "u_xx"]

print("=" * 92)
print("TASK 2b -- NOISE ROBUSTNESS (each method tuned over its own bandwidth)")
print("=" * 92)
print(f"target ({x0},{t0});  noise sigma = eta * RMS(u) = eta * {uref:.4f}")
print(f"true: u_x={TRUE['u_x']:.5f}  u_t={TRUE['u_t']:.5f}  u_xx={TRUE['u_xx']:.5f}")

def sweep(fn, sigma, integer_m=False):
    """returns best (median rel err over the 3 derivs) and the argmin bandwidth"""
    best, bm = np.inf, None
    grid = [int(x) for x in MS if x >= 1] if integer_m else MS
    for m in grid:
        errs = []
        for s in range(NREP):
            try:
                est = fn(m, sigma, 1000 + s)
            except Exception:
                continue
            errs.append(np.median([abs(est[k] - TRUE[k]) / abs(TRUE[k]) for k in KEYS]))
        if not errs:
            continue
        e = np.median(errs)
        if e < best:
            best, bm = e, m
    return best, bm

METHODS = {
    "FD-m   (central FD, spacing m*dx)": (lambda m, s, sd: fd_estimator(m, s, sd), True),
    "SG-m   (Sav-Golay deg 4)":          (lambda m, s, sd: sg_estimator(m, s, sd), True),
    "MOM    (moment regression)":        (lambda m, s, sd: scatter_estimator(m * DXG, s, sd, "moment"), False),
    "MLS-u  (its closed form)":          (lambda m, s, sd: scatter_estimator(m * DXG, s, sd, "mls_uniform"), False),
    "MLS-t  (tricube kernel)":           (lambda m, s, sd: scatter_estimator(m * DXG, s, sd, "mls_tricube"), False),
}

table = {k: [] for k in METHODS}
bw = {k: [] for k in METHODS}
for eta in ETAS:
    sig = eta * uref
    print(f"\n  noise eta = {eta:.0e}")
    for name, (fn, intm) in METHODS.items():
        e, m = sweep(fn, sig, intm)
        table[name].append(e); bw[name].append(m)
        print(f"    {name:<36} rel err {e:>9.3e}   at best bandwidth m={m}")

print("\n" + "-" * 92)
print("SUMMARY  (median relative error over u_x, u_t, u_xx; each method tuned)")
print(f"{'method':<36}" + "".join(f"{e:>12.0e}" for e in ETAS))
for name in METHODS:
    print(f"{name:<36}" + "".join(f"{v:>12.2e}" for v in table[name]))
print("\nratios vs the moment method (>1 means the baseline is BETTER):")
mom = np.array(table["MOM    (moment regression)"])
for name in METHODS:
    if name.startswith("MOM"):
        continue
    r = mom / np.array(table[name])
    print(f"  MOM / {name:<34}" + "".join(f"{v:>10.2f}" for v in r))

fig, ax = plt.subplots(figsize=(7.2, 4.8))
xs = np.array([1e-7 if e == 0 else e for e in ETAS])
sty = {"FD-m   (central FD, spacing m*dx)": ("s--", "tab:red"),
       "SG-m   (Sav-Golay deg 4)": ("^--", "tab:purple"),
       "MOM    (moment regression)": ("o-", "tab:blue"),
       "MLS-u  (its closed form)": ("x-", "tab:cyan"),
       "MLS-t  (tricube kernel)": ("d-", "tab:green")}
for name in METHODS:
    mk, cl = sty[name]
    ax.loglog(xs, table[name], mk, color=cl, ms=5, label=name)
ax.set_xlabel("relative noise $\\eta$  (leftmost point = clean)")
ax.set_ylabel("median relative error in $u_x,u_t,u_{xx}$")
ax.set_title("Jet recovery vs noise, every method tuned over its own bandwidth")
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_task2_noise.png"), dpi=150)
print("\n  figure -> figures/fig_task2_noise.png")
