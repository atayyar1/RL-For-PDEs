"""Is there an ORDER BARRIER for positive stencils?  (Godunov / Bolley-Crouzeix)

Claim: for any positive consistent stencil, Jensen's inequality forces
    M_tt := sum_i w_i (dt_i)^2  >=  ( sum_i w_i dt_i )^2  =  (effective step)^2 > 0
so the u_tt error term can NEVER be cancelled by a non-negative weight vector.
Non-negative weights => w is a probability distribution => variance >= 0.
Only a SIGNED weight vector can cancel it -- and signed means ||w||_1 > 1.

That is a genuine stability/accuracy Pareto frontier, and it is the discrete
analogue of Godunov's theorem.
"""
import numpy as np
from scipy.optimize import linprog
import scheme_mdp as S

print("=== A. Jensen obstruction: can any positive w cancel the Dt^2 moment? ===")
rng = np.random.default_rng(0)
viol = 0; n = 0; slack = []
for _ in range(3000):
    m = rng.integers(2, 9); k = int(rng.integers(1, max(2, S.k_max(m))))
    # allow neighbours on SEVERAL past time levels (the scattered space-time cloud)
    npts = rng.integers(5, 14)
    off_x = rng.integers(-m, m + 1, size=npts) * S.DX
    off_t = -rng.integers(1, k + 1, size=npts) * S.DT
    A, b = S.build_A(off_x, off_t)
    r = linprog(np.zeros(npts), A_eq=A, b_eq=b, bounds=[(0, None)] * npts, method="highs")
    if r.status != 0:
        continue
    w = r.x; n += 1
    M1 = w @ off_t          # effective time advance
    M2 = w @ off_t**2       # the uncancellable moment
    slack.append(M2 - M1**2)
    if M2 < M1**2 - 1e-18:
        viol += 1
print(f"  positive stencils sampled: {n}")
print(f"  violations of  M2 >= M1^2 : {viol}   (expected 0 -- this is Jensen)")
print(f"  min slack M2-M1^2 = {min(slack):.3e}   (>=0 always; =0 iff all dt_i equal)")

print("\n=== B. Can a SIGNED stencil cancel it?  (order barrier is escapable only by ||w||_1>1) ===")
def try_cancel(off_x, off_t, positive):
    """Add the row  sum w dt_i^2 = 0  and ask for feasibility."""
    A, b = S.build_A(off_x, off_t)
    A2 = np.vstack([A, off_t**2 / (np.max(np.abs(off_t))**2)])
    b2 = np.append(b, 0.0)
    nn = len(off_x)
    if positive:
        r = linprog(np.zeros(nn), A_eq=A2, b_eq=b2, bounds=[(0, None)] * nn, method="highs")
        return (r.x if r.status == 0 else None)
    r = linprog(np.ones(2 * nn), A_eq=np.hstack([A2, -A2]), b_eq=b2,
                bounds=[(0, None)] * 2 * nn, method="highs")
    return (r.x[:nn] - r.x[nn:]) if r.status == 0 else None

ok_pos = ok_sgn = trials = 0; l1s = []
for _ in range(600):
    m = int(rng.integers(3, 9)); k = max(2, int(rng.integers(2, 12)))
    npts = int(rng.integers(8, 16))
    off_x = rng.integers(-m, m + 1, size=npts) * S.DX
    off_t = -rng.integers(1, k + 1, size=npts) * S.DT
    if len(set(off_t)) < 2:      # need >=2 time levels for any hope
        continue
    trials += 1
    if try_cancel(off_x, off_t, True)  is not None: ok_pos += 1
    ws = try_cancel(off_x, off_t, False)
    if ws is not None:
        ok_sgn += 1; l1s.append(np.abs(ws).sum())
print(f"  multi-time-level stencils tried: {trials}")
print(f"  can cancel Dt^2 with w >= 0 : {ok_pos:>4}  ({ok_pos/max(trials,1):.1%})")
print(f"  can cancel Dt^2 with signed w: {ok_sgn:>4}  ({ok_sgn/max(trials,1):.1%})")
if l1s:
    print(f"  when signed succeeds, ||w||_1: median {np.median(l1s):.3f}, min {min(l1s):.3f}, max {max(l1s):.3f}")

print("\n=== C. Observed convergence order, positive vs signed, parabolic refinement ===")
print("   fixed stencil shape, refine h with dt ~ h^2 (parabolic scaling)")
def trunc(scale, positive):
    """Truncation error of a 5-point 2-level stencil at spatial scale `scale`*DX."""
    off_x = np.array([-1, 0, 1, -1, 1], float) * scale * S.DX
    off_t = np.array([-1, -1, -1, -2, -2], float) * (scale**2) * S.DT
    A, b = S.build_A(off_x, off_t)
    nn = 5
    if positive:
        r = linprog(off_x**4, A_eq=A, b_eq=b, bounds=[(0, None)] * nn, method="highs")
        w = r.x if r.status == 0 else None
    else:
        A2 = np.vstack([A, off_t**2 / np.max(off_t**2), off_x * off_t / np.max(np.abs(off_x * off_t))])
        b2 = np.append(b, [0.0, 0.0])
        r = linprog(np.ones(2 * nn), A_eq=np.hstack([A2, -A2]), b_eq=b2,
                    bounds=[(0, None)] * 2 * nn, method="highs")
        w = (r.x[:nn] - r.x[nn:]) if r.status == 0 else None
    if w is None: return None, None
    xs = 0.5; ts = 0.02
    e = max(abs(w @ u_nb(xs + off_x, ts + off_t) - float(S.u_true(xs, ts)[0])) for _ in [0])
    return e, np.abs(w).sum()

def u_nb(xs, ts):
    return np.array([float(S.u_true(a, b)[0]) for a, b in zip(xs, ts)])

print(f"  {'scale':>6} {'h':>10} | {'positive err':>13} {'||w||1':>7} | {'signed err':>13} {'||w||1':>7}")
prev_p = prev_s = None
for sc in [16, 8, 4, 2, 1]:
    ep, lp = trunc(sc, True); es, ls = trunc(sc, False)
    rp = f"{np.log2(prev_p/ep):.2f}" if (prev_p and ep) else "  - "
    rs = f"{np.log2(prev_s/es):.2f}" if (prev_s and es) else "  - "
    print(f"  {sc:>6} {sc*S.DX:>10.2e} | {ep:>13.3e} {lp:>7.3f} | {es:>13.3e} {ls:>7.3f}   order p={rp} s={rs}")
    prev_p, prev_s = ep, es
