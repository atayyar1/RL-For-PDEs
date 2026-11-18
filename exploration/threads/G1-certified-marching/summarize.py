"""Pivot the sweep CSVs into tables. Prints numbers only.

Usage: python3 summarize.py results/sweep_main.csv [more.csv ...]
"""
import csv, sys
import numpy as np

rows = []
for fn in sys.argv[1:]:
    with open(fn) as f:
        for r in csv.DictReader(f):
            for k in ("disorder", "alpha", "safety", "dt", "Pe", "r_mean", "nu_mean", "hmin_over_h", "min_w",
                      "max_l1", "rho", "pow_nt", "growth", "errT", "errmax", "maxu", "ref_floor", "secs"):
                r[k] = float(r[k])
            for k in ("seed", "K", "levels", "nx", "nt", "n_fallback", "K_used_max", "blowup", "periodic"):
                r[k] = int(r[k])
            r["geom"] = f"{r['kind']}:{r['disorder']:.2f}"
            rows.append(r)

arms = [a for a in ["minnorm", "projgauss", "spectral", "spectralg", "molfe", "molrk3", "molfilt", "maxent", "lp", "lprand", "maxentw"]
        if any(r["arm"] == a for r in rows)]


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fmt_e(v):
    return f"{v:9.2e}" if np.isfinite(v) else "      inf"


def table(title, groups, gkey, gvals):
    print(f"\n### {title}")
    print(f"{'arm':10}" + "".join(f"{str(v):>22}" for v in gvals))
    for a in arms:
        line = f"{a:10}"
        for v in gvals:
            rs = [r for r in groups if r["arm"] == a and r[gkey] == v]
            if not rs:
                line += f"{'-':>22}"; continue
            n = len(rs)
            unst = sum(r["growth"] > 100 for r in rs)
            blow = sum(r["blowup"] for r in rs)
            errs = np.array([r["errT"] for r in rs])
            med = np.median(errs)
            line += f"  {unst:2d}/{n:<2d} {blow:2d}bl {fmt_e(med)}"
        print(line)
    print("  cell = (#growth>100)/(#runs)  (#blow-ups)  median L_inf(T)")


print(f"{len(rows)} rows, arms: {arms}")
print(f"periodic={sorted(set(r['periodic'] for r in rows))} levels={sorted(set(r['levels'] for r in rows))} safety={sorted(set(r['safety'] for r in rows))} nx={sorted(set(r['nx'] for r in rows))}")

geoms = sorted(set(r["geom"] for r in rows), key=lambda s: (s.split(':')[0] == 'random', float(s.split(':')[1])))
alphas = sorted(set(r["alpha"] for r in rows), reverse=True)
Ks = sorted(set(r["K"] for r in rows))

table("by geometry (all alpha, all K)", rows, "geom", geoms)
table("by alpha (all geometries, all K)", rows, "alpha", alphas)
table("by K (all geometries, all alpha)", rows, "K", Ks)
for a in alphas:
    table(f"alpha={a} (Pe={rows[0]['nx'] and 1/(rows[0]['nx']-1)/a:.2f}): by geometry", sel(alpha=a), "geom", geoms)

print("\n### worst growth per arm, and where")
for a in arms:
    rs = [r for r in rows if r["arm"] == a]
    w = max(rs, key=lambda r: r["growth"])
    print(f"{a:10} growth_max={w['growth']:.2e} at {w['geom']} s{w['seed']} K{w['K']} a{w['alpha']}  |M^nt|={w['pow_nt']:.2e} "
          f"errT={fmt_e(w['errT'])}  min_w={w['min_w']:.3f} max_l1={w['max_l1']:.3f}  fb={w['n_fallback']}")
print("\n### growth distribution per arm: count growth<=1+1e-9 / <=10 / <=100 / >100 ; and O(1)-wrong-but-bounded (growth<=100, errT>0.1)")
for a in arms:
    gs = np.array([r["growth"] for r in rows if r["arm"] == a]); es = np.array([r["errT"] for r in rows if r["arm"] == a])
    print(f"{a:10} n={len(gs):4d}  <=1: {np.sum(gs<=1+1e-9):4d}  <=10: {np.sum(gs<=10):4d}  <=100: {np.sum(gs<=100):4d}  >100: {np.sum(gs>100):4d}   bounded-but-wrong: {np.sum((gs<=100)&(es>0.1)):4d}")

print("\n### fallback (positivity-infeasible after workaround) per arm: mean nodes / 99, max")
for a in ["maxent", "lp", "lprand", "maxentw"]:
    rs = [r for r in rows if r["arm"] == a]
    if rs:
        fb = np.array([r["n_fallback"] for r in rs])
        print(f"{a:10} mean={fb.mean():5.2f} max={fb.max():3d}  runs with fallback>0: {np.sum(fb>0)}/{len(rs)}")

print("\n### certificate vs outcome (all runs): rows = max_l1 <= 1+1e-9 ?, cols = growth <= 1+1e-9 / <=100 / >100")
for cert in (True, False):
    rs = [r for r in rows if (r["max_l1"] <= 1 + 1e-9) == cert]
    g1 = sum(r["growth"] <= 1 + 1e-9 for r in rs); g100 = sum(r["growth"] <= 100 for r in rs)
    print(f"  certified={cert!s:5}: n={len(rs):4d}  growth<=1: {g1:4d}  growth<=100: {g100:4d}  growth>100: {len(rs)-g100:4d}")

print("\n### accuracy where everything is stable: median errT ratio arm / maxent, per (alpha, K), stable runs only")
print(f"{'alpha':>6} {'K':>2}" + "".join(f"{a:>11}" for a in arms))
for al in alphas:
    for K in Ks:
        base = {(r["geom"], r["seed"]): r["errT"] for r in sel(alpha=al, K=K, arm="maxent") if r["growth"] <= 100}
        line = f"{al:6} {K:2d}"
        for a in arms:
            rat = [r["errT"] / base[(r["geom"], r["seed"])] for r in sel(alpha=al, K=K, arm=a)
                   if (r["geom"], r["seed"]) in base and r["growth"] <= 100 and np.isfinite(r["errT"])]
            line += f"{np.median(rat):11.3f}" if rat else f"{'-':>11}"
        print(line)

print("\n### pairing: for each config, consistency-only arm's outcome vs positivity feasibility of the SAME stencil (maxent n_fallback at same geom/seed/K/alpha)")
fb_of = {(r["geom"], r["seed"], r["K"], r["alpha"]): r["n_fallback"] for r in rows if r["arm"] == "maxent"}
print(f"{'arm':10}{'feasible everywhere & stable':>30}{'feasible & UNSTABLE':>22}{'infeasible somewhere & stable':>31}{'infeasible & unstable':>23}{'feasible & bounded-wrong(err>0.1)':>35}")
for a in ["minnorm", "projgauss", "spectral", "spectralg", "molfe", "molrk3", "molfilt", "maxent", "lp", "maxentw"]:
    c = np.zeros(5, int)
    for r in rows:
        if r["arm"] != a: continue
        key = (r["geom"], r["seed"], r["K"], r["alpha"])
        if key not in fb_of: continue
        feas = fb_of[key] == 0; unst = r["growth"] > 100
        c[0] += feas and not unst; c[1] += feas and unst; c[2] += (not feas) and not unst; c[3] += (not feas) and unst
        c[4] += feas and not unst and r["errT"] > 0.1
    print(f"{a:10}{c[0]:30d}{c[1]:22d}{c[2]:31d}{c[3]:23d}{c[4]:35d}")

print("\n### the 'feasible & UNSTABLE' cases in detail (consistency-only arms)")
for a in ["minnorm", "projgauss", "spectral", "spectralg"]:
    for r in rows:
        key = (r["geom"], r["seed"], r["K"], r["alpha"])
        if r["arm"] == a and key in fb_of and fb_of[key] == 0 and r["growth"] > 100:
            print(f"  {a:10} {r['geom']} s{r['seed']} K{r['K']} a{r['alpha']} growth={r['growth']:.2e} errT={fmt_e(r['errT'])} min_w={r['min_w']:.3f} l1={r['max_l1']:.3f} nu={r['nu_mean']:.3f} r={r['r_mean']:.3f}")

print("\n### maxentw: residual fallback after widening (nodes), by geometry x alpha")
for geom in geoms:
    line = f"{geom:14}"
    for al in alphas:
        rs = sel(arm="maxentw", geom=geom, alpha=al)
        line += f"  a{al}: " + (f"{np.mean([r['n_fallback'] for r in rs]):5.1f}/{np.max([r['n_fallback'] for r in rs]):2d}" if rs else "   -   ")
    print(line)

print("\n### molfilt: tuned filter strength eps (1/16 = Nyquist zeroed on uniform grid; inf = not stabilisable) by geometry x alpha, K=5: median / max")
for geom in geoms:
    line = f"{geom:14}"
    for al in alphas:
        rs = [r for r in rows if r["arm"] == "molfilt" and r["geom"] == geom and r["alpha"] == al and r["K"] == 5]
        if rs:
            e = np.array([float(r.get("eps", "nan")) for r in rs])
            line += f"  a{al}: {np.median(e):.4f}/{e.max():.4f}"
        else:
            line += "   -   "
    print(line)
