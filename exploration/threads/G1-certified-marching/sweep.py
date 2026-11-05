"""Sweep: march every arm on every geometry / K / alpha, record stability and error.

One CSV row per (arm, geometry, seed, K, alpha). No interpretation is printed
(CHECKS.md #2); summarize.py builds the tables and RESULTS.md reads them.

Usage: python3 sweep.py --tag main [--levels 1] [--safety 0.5] [--alphas 0.1 0.03 0.01 0.003]
                        [--nx 101] [--seeds 0 1 2] [--Ks 3 5 7] [--arms ...] [--T 0.5]
"""
import argparse, csv, os, sys, time
import numpy as np
import g1lib as g

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--levels", type=int, default=1)
ap.add_argument("--safety", type=float, default=0.5)
ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.03, 0.01, 0.003])
ap.add_argument("--nx", type=int, default=101)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--Ks", type=int, nargs="+", default=[3, 5, 7])
ap.add_argument("--geoms", nargs="+", default=["jitter:0", "jitter:0.1", "jitter:0.25", "jitter:0.45", "random:0"])
ap.add_argument("--arms", nargs="+", default=g.ARMS)
ap.add_argument("--T", type=float, default=0.5)
ap.add_argument("--nsamp", type=int, default=50)
ap.add_argument("--npow", type=int, default=10000)
ap.add_argument("--periodic", action="store_true")
args = ap.parse_args()

if args.levels == 2:
    args.arms = [a for a in args.arms if a not in ("molfe", "molrk3")]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"sweep_{args.tag}.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
FIELDS = ["periodic", "arm", "kind", "disorder", "seed", "K", "alpha", "safety", "levels", "nx", "dt", "nt",
          "Pe", "r_mean", "nu_mean", "hmin_over_h", "n_fallback", "K_used_max", "min_w", "max_l1",
          "rho", "pow_nt", "growth", "errT", "errmax", "maxu", "blowup", "ref_floor", "secs", "eps"]
done = set()
if os.path.exists(out):
    with open(out) as f:
        for row in csv.DictReader(f):
            done.add((row["arm"], row["kind"], row["disorder"], row["seed"], row["K"], row["alpha"]))
fh = open(out, "a", newline="")
wr = csv.DictWriter(fh, fieldnames=FIELDS)
if not done:
    wr.writeheader()

for alpha in args.alphas:
    prob = g.Problem(alpha=alpha, c=1.0)
    for geom in args.geoms:
        kind, dis = geom.split(":")
        dis = float(dis)
        for seed in (args.seeds if not (kind == "jitter" and dis == 0) else [0]):
            x = g.make_points(args.nx, dis, seed=seed, kind=kind, periodic=args.periodic)
            h = 1.0 / args.nx if args.periodic else 1.0 / (args.nx - 1)
            dt = g.time_step(x, prob, args.safety, args.periodic)   # exact CFL-based step, never rounded
            nt = int(np.ceil(args.T / dt))
            m = max(1, nt // args.nsamp)                    # sample every m steps
            samp = np.arange(m, nt + 1, m)
            if samp[-1] != nt:
                samp = np.append(samp, nt)
            times = dt * samp
            if args.periodic:
                class _R:  # exact periodic reference
                    floor = 0.0
                    U = np.array([g.u_periodic(prob, x, tt) for tt in times])
                    def at(self, k): return self.U[k]
                ref = _R()
            else:
                ref = g.Reference(prob, times, x)
            step_to_k = {int(s): k for k, s in enumerate(samp)}
            u0 = g.u_periodic(prob, x, 0.0) if args.periodic else prob.u_true(x, 0.0)
            for K in args.Ks:
                for arm in args.arms:
                    key = (arm, kind, str(dis), str(seed), str(K), str(alpha))
                    if key in done:
                        continue
                    t0 = time.time()
                    op = g.Operator(x, prob, dt, arm, K, levels=args.levels, seed=seed, periodic=args.periodic, nt=nt)
                    u1 = None
                    if args.levels == 2:
                        op1 = g.Operator(x, prob, dt, arm, K, levels=1, seed=seed, periodic=args.periodic)
                        u1 = op1.M1 @ u0
                    rho = op.spectral_radius()
                    pow_nt = op.power_norm(nt)
                    growth = op.growth(nt)
                    res = op.march(u0, u1, nt, ref_fn=lambda k: ref.at(step_to_k[k]), sample_every=m)
                    err = res["err"]
                    finite = np.isfinite(err)
                    complete = len(res["steps"]) == len(samp) and res["steps"][-1] == nt
                    row = dict(periodic=int(args.periodic), arm=arm, kind=kind, disorder=dis, seed=seed, K=K, alpha=alpha, safety=args.safety,
                               levels=args.levels, nx=args.nx, dt=dt, nt=nt, Pe=prob.c * h / alpha,
                               r_mean=alpha * dt / h**2, nu_mean=prob.c * dt / h, hmin_over_h=np.diff(x).min() / h,
                               n_fallback=int(op.fallback.sum()), K_used_max=int(op.K_used.max()),
                               min_w=float(op.min_w.min()), max_l1=float(op.l1.max()), rho=rho,
                               pow_nt=pow_nt, growth=growth,
                               errT=float(err[-1]) if complete and np.isfinite(err[-1]) else np.inf,
                               errmax=float(err[finite].max()) if finite.any() else np.inf,
                               maxu=float(np.nanmax(res["maxu"])),
                               blowup=int(not (complete and np.isfinite(err[-1]) and res["maxu"].max() < 10)),
                               ref_floor=ref.floor, secs=time.time() - t0, eps=op.eps)
                    wr.writerow(row); fh.flush()
                    print(f"{arm:10} {kind:6} {dis:4.2f} s{seed} K{K} a{alpha:<6} growth={growth:9.2e} |M^nt|={pow_nt:9.2e} "
                          f"errT={row['errT']:9.2e} fb={row['n_fallback']:2d} minw={row['min_w']:8.4f} l1={row['max_l1']:6.3f} {row['secs']:5.1f}s")
fh.close()
print("maxent paths:", g.MAXENT_PATH)
