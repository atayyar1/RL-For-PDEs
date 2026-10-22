"""Stops by one step of policy improvement (rollout), no PPO.

The myopic stop compares the stop cost |beta_p| g_err(p) with ONE step of expansion. The real price of expanding p is
its whole subtree: every future score and every future point. Build the base policy's map once, and for each of its
points q compute bottom-up (estimates only: score, guess, beta):
    A_q = s_q + sum_r |w_r| A_r          estimated error at q's subtree per unit influence (an upper bound with sharing)
    N_q = 1 + sum_r N_r / users(r)       points of q's subtree, a shared point split between its users
Then rebuild with: stop p  iff  |beta_p| g_err(p) / EPS  <  |beta_p| A_p / EPS + lam N_p / N_LW   (stop is cheaper than
the expected cost-to-go of expanding). Points the base map does not have are expanded. Repeating with the improved map
as the base is policy iteration.
"""
from robust import *


def cost_to_go(m):
    """A (error per unit beta) and N (points) for every expanded point of a finished map."""
    users = {}
    for p, (nbs, w) in m["stencil"].items():
        for q in set(nbs):
            users[q] = users.get(q, 0) + 1
    A, N = {}, {}
    for p in forward_order(m):                                   # bottom-up: the points a stencil uses come first
        nbs, w = m["stencil"][p]
        A[p], N[p] = point_score(p, nbs, w), 1.0
        for q, wq in zip(nbs, w):
            if q in A:                                           # an expanded point (not data, not stopped)
                A[p] += abs(wq) * A[q]
                N[p] += N[q] / users[q]
            elif q in m["stopped"]:                              # a stopped point costs its guess error
                A[p] += abs(wq) * guess_err(q)
    return A, N


def rollout_rule(base, lam, A, N, stopped=()):
    """base rule + a stop wherever stopping beats the base map's cost-to-go (points stopped before stay stopped)."""
    def choose(m, p):
        if p in stopped:
            return None
        if p in A and p != m["z_star"]:
            b = abs(m["beta"][p])
            if b * guess_err(p) / EPS < b * A[p] / EPS + lam * N[p] / n_lw(m["z_star"]):
                return None
        return base(m, p)
    return choose


def policy_iteration(z, base, lam=1.0, iters=2):
    """Base map, then `iters` rounds of rollout stops. Returns the list of maps (round 0 = base)."""
    maps = [build_map_s(z, base)]
    for _ in range(iters):
        A, N = cost_to_go(maps[-1])
        maps.append(build_map_s(z, rollout_rule(base, lam, A, N, maps[-1]["stopped"])))
    return maps


if __name__ == "__main__":
    for z in [(12, 40), (25, 40), (37, 55)]:
        for name, base in [("LW", lw), ("greedy", greedy_rule(1.0))]:
            for i, mm in enumerate(policy_iteration(z, base)):
                r = report_s(mm)
                print(f"{z} {name:6s} round {i}: {r['points']:5d} points {r['stops']:4d} stops | error {r['error']:.2e} | bound {r['bound']:.2e}")
