"""Query cost vs accuracy, sweeping the kernel capture factor gamma.

m = ceil(gamma * sigma / dx),  sigma = sqrt(2 alpha t*).
gamma=1 sits exactly on the positivity frontier and truncates the kernel hard.
Going wider costs linearly in gamma but the truncated tail mass falls like
erfc(gamma/sqrt2) -- so accuracy is EXPONENTIAL in cost. That is the real claim.
"""
import numpy as np
from scipy.special import erfc
import scheme_mdp as S

x_star = 0.5
jstar = int(round(x_star / S.DX))

for J in [100, 400, 1600]:
    t_star = J * S.DT
    sigma = np.sqrt(2 * S.ALPHA * t_star)
    u_ref = float(S.u_true(x_star, t_star)[0])
    # FD reference cost + error
    cost_fd = sum(min(S.NX - 2, 2 * (J - q) + 1) for q in range(J))
    uf = S.u_true(S.X, 0.0).copy()
    for _ in range(J):
        uf = S.advance(uf, 1, 1)
    err_fd = abs(uf[jstar] - u_ref)

    print(f"\n=== t* = {J} CFL steps = {t_star:.4g};  sigma = {sigma:.4g} = {sigma/S.DX:.1f} cells;"
          f"  u_ref = {u_ref:.6g} ===")
    print(f"  explicit FD:  cost = {cost_fd:>8}   err = {err_fd:.3e}")
    print(f"  {'gamma':>6} {'m':>5} {'cost':>7} {'k_max(m)':>9} {'err':>11} {'tail erfc':>11} {'vs FD cost':>11}")
    for g in [1, 2, 3, 4, 5, 6]:
        m = int(np.ceil(g * sigma / S.DX))
        if jstar - m < 0 or jstar + m > S.NX - 1:
            print(f"  {g:>6} {m:>5}   -- stencil exceeds the domain --")
            continue
        km = S.k_max(m)
        if km < J:
            print(f"  {g:>6} {m:>5} {2*m+1:>7} {km:>9}   (k_max < J: needs >1 step)")
            continue
        w = S.weights(m, J)
        if w is None:
            print(f"  {g:>6} {m:>5}   infeasible")
            continue
        u0 = S.u_true(S.X[jstar - m: jstar + m + 1], 0.0)
        err = abs(float(w @ u0) - u_ref)
        print(f"  {g:>6} {m:>5} {2*m+1:>7} {km:>9} {err:>11.3e} {erfc(g/np.sqrt(2)):>11.2e}"
              f" {cost_fd/(2*m+1):>10.0f}x")
