"""The query-driven win: cost of ONE point at (x*,t*), FD vs one wide positive step.

This is the setting Ali's MDP actually targets, and it is where the asymptotics
are lopsided:

  explicit FD  must fill its NUMERICAL cone, which under CFL grows 1 cell/step
               -> cost ~ min(nx, 2J) * J   evaluations for a query J steps up

  wide positive stencil  jumps in ONE step, needing only the PHYSICAL domain of
               dependence, half-width m = sqrt(2 alpha t*)/dx
               -> cost ~ (2m+1)  evaluations

The ratio grows without bound in J. This, not the marching speedup, is the claim.
"""
import numpy as np
import scheme_mdp as S

print(f"grid: NX={S.NX}, dx={S.DX:.4g}, dt_cfl={S.DT:.4g}, alpha={S.ALPHA}, c={S.C}")
print()
print(f"{'t* (steps)':>10} {'t* phys':>9} {'m needed':>9} {'k_max(m)':>9} | "
      f"{'wide cost':>10} {'FD cost':>10} {'ratio':>9} | {'wide err':>10} {'FD err':>10}")

x_star = 0.5
for J in [25, 50, 100, 200, 400, 800, 1600, 3200]:
    t_star = J * S.DT
    if t_star > 0.45:
        break
    # physical domain of dependence: one sigma of the heat kernel, plus advective shift
    sigma = np.sqrt(2 * S.ALPHA * t_star)
    m = int(np.ceil(sigma / S.DX))
    if 2 * m + 1 > S.NX:
        m = (S.NX - 1) // 2
    km = S.k_max(m)
    w = S.weights(m, min(J, km))
    # one giant step from the exact initial condition
    k_use = min(J, km)
    if w is not None and k_use == J:
        lo, off = S._window(int(round(x_star / S.DX)), m)
        wj = S.weights(m, J, offsets=off)
        u0 = S.u_true(S.X[lo: lo + 2 * m + 1], 0.0)
        pred = float(wj @ u0) if wj is not None else np.nan
    else:
        pred = np.nan
    err_wide = abs(pred - float(S.u_true(x_star, t_star)[0]))
    cost_wide = 2 * m + 1

    # explicit FD: fill the numerical cone (1 cell/step, clipped to the domain)
    cost_fd = sum(min(S.NX - 2, 2 * (J - q) + 1) for q in range(J))
    uf = S.u_true(S.X, 0.0).copy()
    for _ in range(J):
        nxt = S.advance(uf, 1, 1)
        uf = nxt
    err_fd = abs(uf[int(round(x_star / S.DX))] - float(S.u_true(x_star, t_star)[0]))

    print(f"{J:>10} {t_star:>9.4g} {m:>9} {km:>9} | {cost_wide:>10} {cost_fd:>10} "
          f"{cost_fd/cost_wide:>8.0f}x | {err_wide:>10.2e} {err_fd:>10.2e}")
