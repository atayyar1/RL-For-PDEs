"""Initial-condition wavenumber sweep for the UNFILTERED consistency-only arms (periodic, exact
reference). The operator is linear and fixed, so its growth envelope does not depend on the
data; what depends on the data is how much of the unstable mode is present at t = 0 and hence
when the blow-up becomes visible. Reports, per arm and IC mode m (u0 = sin(2 pi m x)):
L_inf error at T, the first time max|u| exceeds 10 (blow-up time), and the growth envelope.
Prints tables only.
"""
import numpy as np
import g1lib as g

p = g.Problem(alpha=0.1, c=1.0)
T = 0.5
modes = [1, 2, 4, 8, 16, 25]
for kind, dis in [("jitter", 0.0), ("jitter", 0.25), ("random", 0.0)]:
    x = g.make_points(100, dis, seed=0, kind=kind, periodic=True)
    dt = g.time_step(x, p, 0.5, True); nt = int(np.ceil(T / dt)); m_s = max(1, nt // 200)
    print(f"\n=== periodic {kind}:{dis}, alpha=0.1, K=5, nt={nt}, dt={dt:.2e} ===")
    print(f"{'arm':10}{'growth':>10}" + "".join(f"{'m=%d' % m:>22}" for m in modes))
    print(f"{'':10}{'':>10}" + "".join(f"{'errT / t_blowup':>22}" for m in modes))
    for arm in ["molfe", "molrk3", "minnorm", "spectral", "maxent"]:
        op = g.Operator(x, p, dt, arm, 5, periodic=True)
        line = f"{arm:10}{min(op.growth(nt), 1e99):10.2e}"
        for m in modes:
            k = 2 * np.pi * m
            u0 = np.sin(k * x)
            ref = lambda s: np.exp(-p.alpha * k**2 * s * dt) * np.sin(k * (x - p.c * s * dt))
            res = op.march(u0, None, nt, ref_fn=ref, sample_every=m_s)
            over = np.where(res["maxu"] > 10)[0]
            tb = res["steps"][over[0]] * dt if len(over) else np.inf
            e = res["err"][-1] if len(res["err"]) and np.isfinite(res["err"][-1]) and res["steps"][-1] == nt else np.inf
            line += f"{e:11.2e} /{tb:8.3f} "
        print(line)
    # reference scale: exact solution amplitude at T for each mode
    print("exact amplitude at T:", "  ".join(f"m={m}: {np.exp(-p.alpha*(2*np.pi*m)**2*T):.1e}" for m in modes))
