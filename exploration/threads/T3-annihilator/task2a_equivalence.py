"""
task2a_equivalence.py -- THE NOVELTY QUESTION, answered first.

Claim under test
----------------
Step 1 ("draw random geometries G and random weights w, regress the observed
prediction error e(w,G) on the moment vector M(w,G)") is presented as a NEW way
to recover the local jet without finite differencing.

Algebra
-------
With sum_i w_i = 1 and y_i = u_i - u*,
        M(w) = Phi^T w        (Phi = scaled-monomial design matrix, n x Q)
        e(w) = y^T w
The least-squares normal equations over S weight draws are
        (sum_s M M^T) d = sum_s M e
    <=> Phi^T ( sum_s w w^T ) Phi d = Phi^T ( sum_s w w^T ) y
    <=> Phi^T W Phi d = Phi^T W y,      W := sum_s w^(s) w^(s)T

which is EXACTLY weighted least squares on the pointwise Taylor residuals,
i.e. a local polynomial / moving-least-squares fit constrained to interpolate
u* exactly.  The identity is algebraic and holds for ANY finite S -- it is not
a large-sample statement.  Random w only supply Monte-Carlo noise in W.

Multiple random geometries change nothing: pad each w to the union of all
sampled neighbours and the same identity holds on the pooled cloud.

This script verifies the identity to machine precision and measures the price
paid for the randomisation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jetlib as J

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
rng = np.random.default_rng(0)
np.set_printoptions(precision=6, suppress=False)

sol = J.AdvDiff(alpha=0.1, c=1.0)
x0, t0 = 0.40, 0.25
K, GRADING = 4, "parabolic"
qs = J.monomials(K, GRADING)
hx = 3 * (1 / 99.0)              # ~0.0303
ht = hx ** 2 / sol.alpha         # parabolic time radius ~9.2e-3

print("=" * 78)
print("TASK 2a -- IS THE MOMENT METHOD NEW?   (algebraic identity test)")
print("=" * 78)
print(f"target z*=({x0},{t0})  hx={hx:.4g}  ht={ht:.4g}  monomials(K={K},{GRADING})={qs}")

# ---------------------------------------------------------------- single geometry
n = 12
dxi = hx * rng.uniform(-1, 1, n)
dti = ht * rng.uniform(-1, 0, n)
u_nb = np.array([sol.u(x0 + a, t0 + b) for a, b in zip(dxi, dti)])
u_st = sol.u(x0, t0)
y = u_nb - u_st
Phi = J.design_matrix(dxi, dti, qs, hx, ht)

print("\n--- A. one fixed geometry, S random weight vectors ---")
print(f"{'S':>8} {'||d_moment - d_WLS(W=sum ww^T)||_inf':>38} {'rel jet err':>14}")
d_true = J.true_jet(sol, x0, t0, qs) * J.jet_scaling(qs, hx, ht)
for S in [10, 50, 200, 2000, 20000]:
    W = J.random_weights(S, n, rng, "gauss", tau=1.0)
    d_mom = J.moment_regression(Phi, y, W)
    d_wls = J.wls_jet(Phi, y, W.T @ W)                 # exact same normal eqns
    err_id = np.max(np.abs(d_mom - d_wls))
    rel = np.linalg.norm(d_mom - d_true) / np.linalg.norm(d_true)
    print(f"{S:>8} {err_id:>38.3e} {rel:>14.3e}")

print("\n  -> identity holds to machine precision for EVERY S.")
print("     (it is exact algebra, not a limit theorem)")

# ------------------------------------------------------- many random geometries
print("\n--- B. many random geometries, pooled onto the union cloud ---")
S, n_nb, n_pool = 4000, 6, 40
dxp = hx * rng.uniform(-1, 1, n_pool)
dtp = ht * rng.uniform(-1, 0, n_pool)
up = np.array([sol.u(x0 + a, t0 + b) for a, b in zip(dxp, dtp)])
yp = up - u_st
Php = J.design_matrix(dxp, dtp, qs, hx, ht)

Wfull = np.zeros((S, n_pool))
for s in range(S):
    idx = rng.choice(n_pool, n_nb, replace=False)
    Wfull[s, idx] = J.random_weights(1, n_nb, rng, "gauss", tau=1.0)[0]
d_mom = J.moment_regression(Php, yp, Wfull)
d_wls = J.wls_jet(Php, yp, Wfull.T @ Wfull)
print(f"  S={S} draws of {n_nb}-point geometries from a {n_pool}-point pool")
print(f"  ||d_moment - d_WLS||_inf / ||d|| = "
      f"{np.max(np.abs(d_mom - d_wls)) / np.linalg.norm(d_wls):.3e}")
print("  -> random GEOMETRIES also collapse to one weighted fit on the pooled cloud.")

# --------------------------------------------- what does the implied W look like?
Wimp = Wfull.T @ Wfull
diag = np.diag(Wimp)
off = Wimp - np.diag(diag)
print(f"\n  implied W: diag mean={diag.mean():.4g} (spread {diag.std()/diag.mean():.1%}), "
      f"max|offdiag|/mean(diag)={np.max(np.abs(off))/diag.mean():.3f}")
print("  -> W is ~ (uniform diagonal) + small coupling: the moment method silently")
print("     chooses a FLAT kernel over the sampling box.  Not an especially good one.")

# ------------------------------------------------------------------------------
# D.  What IS the limiting estimator?  -> plain uniform-weight MLS.
# ------------------------------------------------------------------------------
print("\n--- D. identify the S->infinity estimator ---")
print("  With w = 1/n + z, z ~ N(0, tau^2 (I - 11^T/n)) on a FIXED geometry,")
print("      E[w w^T] = tau^2 (I - 11^T/n) + 11^T/n^2 .")
print("  As tau grows the centering matrix dominates, and WLS with W = I - 11^T/n")
print("  is *exactly* the profiled-out form of UNCONSTRAINED uniform-weight MLS.")
d_unc, c0 = J.mls_jet_unconstrained(dxi, dti, u_nb, qs, hx, ht)
C = np.eye(n) - np.ones((n, n)) / n
d_cen = J.wls_jet(Phi, y, C)
print(f"\n  ||WLS(W = I - 11^T/n)  -  unconstrained uniform MLS||_inf = "
      f"{np.max(np.abs(d_cen - d_unc)):.3e}   (exact identity)")
print(f"{'tau':>8} {'||d_moment(S=1e5) - uniform MLS||_inf':>40}")
for tau in [1.0, 3.0, 10.0, 100.0]:
    Wt = J.random_weights(100000, n, rng, "gauss", tau=tau)
    dm = J.moment_regression(Phi, y, Wt)
    print(f"{tau:>8.0f} {np.max(np.abs(dm - d_unc)):>40.3e}")
print("\n  VERDICT: Step 1 is not a new estimator.  It is plain local polynomial")
print("  least squares, reached by an expensive Monte-Carlo detour.")

# ------------------------------------------------------------------------------
# C.  price of the randomisation, clean vs noisy
# ------------------------------------------------------------------------------
print("\n--- E. statistical price of the randomisation (clean vs noisy) ---")
S_, n_nb, n_pool = None, 6, 40
dxp = hx * rng.uniform(-1, 1, n_pool)
dtp = ht * rng.uniform(-1, 0, n_pool)
up = np.array([sol.u(x0 + a, t0 + b) for a, b in zip(dxp, dtp)])
Php = J.design_matrix(dxp, dtp, qs, hx, ht)
uref = J.noise_scale(sol, x0, t0, hx, ht, rng=np.random.default_rng(1))
d_true = J.true_jet(sol, x0, t0, qs) * J.jet_scaling(qs, hx, ht)

# exact S->inf limit of sum_s w w^T for random m-subsets of the pool
m, npl, tau = n_nb, n_pool, 1.0
a_ii = (m / npl) * (tau ** 2 * (1 - 1 / m) + 1 / m ** 2)
b_ij = (m * (m - 1) / (npl * (npl - 1))) * (-tau ** 2 / m + 1 / m ** 2)
Winf = (a_ii - b_ij) * np.eye(npl) + b_ij * np.ones((npl, npl))

Ss = np.array([20, 50, 100, 300, 1000, 3000, 10000, 30000])
curves = {}
for eta in [0.0, 1e-4, 1e-2]:
    noise = eta * uref * np.random.default_rng(99).normal(size=n_pool)
    ypn = (up + noise) - (sol.u(x0, t0) + (eta * uref * np.random.default_rng(98).normal()))
    floor = np.linalg.norm(J.wls_jet(Php, ypn, Winf) - d_true) / np.linalg.norm(d_true)
    row = []
    for S2 in Ss:
        errs = []
        for rep in range(10):
            Wf = np.zeros((S2, n_pool))
            for s in range(S2):
                idx = rng.choice(n_pool, n_nb, replace=False)
                Wf[s, idx] = J.random_weights(1, n_nb, rng, "gauss", tau=1.0)[0]
            try:
                dm = J.moment_regression(Php, ypn, Wf)
            except np.linalg.LinAlgError:
                continue
            errs.append(np.linalg.norm(dm - d_true) / np.linalg.norm(d_true))
        row.append(np.median(errs))
    curves[eta] = (np.array(row), floor)
    print(f"  noise {eta:>7.0e}: S=20 {row[0]:.3e}  S=1e3 {row[3]:.3e}  "
          f"S=3e4 {row[-1]:.3e}   WLS floor {floor:.3e}")

fig, ax = plt.subplots(figsize=(6.4, 4.3))
cols = {0.0: "tab:blue", 1e-4: "tab:orange", 1e-2: "tab:green"}
for eta, (row, floor) in curves.items():
    ax.loglog(Ss, row, "o-", color=cols[eta], ms=4,
              label=f"moment regression, noise {eta:.0e}")
    ax.axhline(floor, color=cols[eta], ls="--", lw=1)
ax.set_xlabel("number of $(w,G)$ samples $S$")
ax.set_ylabel("relative jet error")
ax.set_title("Moment regression relaxes onto its own WLS limit (dashed).\n"
             "Clean data is bias-dominated: $S$ buys nothing.", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_equivalence.png"), dpi=150)
print("  figure -> figures/fig_equivalence.png")
