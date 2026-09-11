"""
Task 2 -- p* across system classes, with three competing hypotheses on trial.

Everything is a circulant lattice operator specified by its symbol g(theta), so one
machinery covers diffusion, advection-diffusion, the exact heat semigroup, a
scale-separated two-timescale operator, and dispersive propagators.

THE THREE HYPOTHESES

  H_gap    (original brief)  p* is small when scale separation is good.
  H_spread (team lead)       p* is a monotone, system-independent function of the
                             ALIAS SPREAD -- how much the fine operator distinguishes
                             modes the coarse grid cannot represent.
  H_sign   (POSITIONING.md)  p* is controlled by whether the unresolved aliases avoid
                             the positive real axis.  Magnitude and spread are secondary.

The discriminating cases are built in on purpose:
  * the EXACT HEAT SEMIGROUP has all g > 0, tiny unresolved spread, and a superb gap.
    H_gap and H_spread predict it is the easiest system; H_sign predicts infeasible at
    every M and every depth.
  * the TWO-TIMESCALE operator is perfect scale separation with all g > 0.  Same split.
  * DISPERSIVE UNITARY has |g| = 1 -- no decay at all, the worst possible gap.

Dependent variable: p* = min depth P (half-width <= 3) admitting an EXACT NON-NEGATIVE
compact coarse law, detected by delta(P) = min ||b||_1 - 1 = 0.  delta is scale-free and
cannot decay by renormalisation, unlike a residual on the exactness equations.

Run: python3 task2_systems.py      (~3 min)
"""
import os
for _v in ("OMP", "OPENBLAS", "MKL", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

N = 120
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)


def wrap(th):
    """Map a lattice wavenumber to its physical branch in (-pi, pi]."""
    return (th + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------- systems
def sys_ftcs(r, ra=0.0):
    return lambda th: (1 - 2 * r) + 2 * r * np.cos(th) - 1j * ra * np.sin(th)

def sys_exact_heat(r):
    """Exact continuum heat propagator over one step: g = exp(-r theta^2), ALL POSITIVE."""
    return lambda th: np.exp(-r * wrap(th) ** 2) + 0j

def sys_two_timescale(g_slow=0.95, g_fast=0.05, cut=np.pi / 2, sharp=12.0):
    """Perfect scale separation: slow band near 1, fast band near 0, both POSITIVE."""
    def g(th):
        t = np.abs(wrap(th))
        w = 0.5 * (1 + np.tanh(sharp * (t - cut) / np.pi))
        return (g_slow * (1 - w) + g_fast * w) + 0j
    return g

def sys_disp_unitary(b3=0.0345):
    """Exact u_t = u_xxx propagator, |g| = 1 exactly: no decay, worst possible gap."""
    return lambda th: np.exp(1j * b3 * (wrap(th) * N) ** 3 * 1.0 / N ** 3 * 1.0) \
        if False else (lambda t: np.exp(1j * b3 * wrap(t) ** 3 * 1.0))(th)

def sys_disp_damped(b3=0.0345, r=0.45):
    """Dispersive phase times FTCS damping, so the modulus decays and can flip sign."""
    gu, gd = sys_disp_unitary(b3), sys_ftcs(r)
    return lambda th: gu(th) * gd(th)


SYSTEMS = [
    ("FTCS diff r=0.15",      sys_ftcs(0.15)),
    ("FTCS diff r=0.25",      sys_ftcs(0.25)),
    ("FTCS diff r=0.35",      sys_ftcs(0.35)),
    ("FTCS diff r=0.45",      sys_ftcs(0.45)),
    ("FTCS diff r=0.50",      sys_ftcs(0.50)),
    ("adv-diff Pe=0.5",       sys_ftcs(0.45, 0.45 * 0.5)),
    ("adv-diff Pe=2",         sys_ftcs(0.45, 0.45 * 2.0)),
    ("adv-diff Pe=8",         sys_ftcs(0.30, 0.30 * 8.0)),
    ("exact heat r=0.45",     sys_exact_heat(0.45)),
    ("exact heat r=1.50",     sys_exact_heat(1.50)),
    ("two-timescale .95/.05", sys_two_timescale()),
    ("dispersive unitary",    sys_disp_unitary()),
    ("dispersive damped",     sys_disp_damped()),
]


# ------------------------------------------------------------------ diagnostics
def alias_g(g, M, q, n=N):
    return g(2 * np.pi * (q + (n // M) * np.arange(M)) / n)

def alias_spread(g, M, n=N):
    """max over coarse modes of the diameter of the alias class in the complex plane."""
    out = 0.0
    for q in range(n // M):
        a = alias_g(g, M, q, n)
        out = max(out, float(np.abs(a[:, None] - a[None, :]).max()))
    return out

def sign_diagnostic(g, M, n=N):
    """Largest real-positive unresolved alias at the constant coarse mode.

    POSITIONING.md: a non-negative coarse law of ANY depth/width requires every
    unresolved alias here to avoid (0, 1).  Returns (max offending value, n_offending).
    """
    a = alias_g(g, M, 0, n)[1:]
    off = [z.real for z in a if abs(z.imag) < 1e-9 and 1e-9 < z.real < 1 - 1e-9]
    return (max(off) if off else 0.0), len(off)

def max_modulus_unresolved(g, M, n=N):
    return max(float(np.abs(alias_g(g, M, q, n)[1:]).max()) for q in range(n // M))


# ------------------------------------------------------------------ feasibility
def defect(g, M, P, s, n=N):
    """min ||b||_1 - 1 over EXACT compact coarse laws of depth P, half-width s."""
    Nc = n // M
    ks = np.arange(-s, s + 1)
    nb = P * len(ks)
    rows, rhs = [], []
    for q in range(Nc):
        e = np.exp(1j * 2 * np.pi * q * ks / Nc)
        for gl in alias_g(g, M, q, n):
            rows.append(np.outer(gl ** (P - np.arange(1, P + 1)), e).ravel())
            rhs.append(gl ** P)
    A, y = np.array(rows), np.array(rhs)
    Ar = np.vstack([np.hstack([A.real, -A.real]), np.hstack([A.imag, -A.imag])])
    br = np.concatenate([y.real, y.imag])
    res = linprog(np.ones(2 * nb), A_eq=Ar, b_eq=br, bounds=[(0, None)] * 2 * nb, method="highs")
    return None if res.status != 0 else float(np.abs(res.x[:nb] - res.x[nb:]).sum()) - 1.0

def p_star(g, M, P_max=8, s_max=3, tol=1e-9):
    for P in range(1, P_max + 1):
        for s in range(0, s_max + 1):
            d = defect(g, M, P, s)
            if d is not None and abs(d) < tol:
                return P, s
    return None, None


# =========================================================================== 1
hdr("1.  p* and the diagnostics, across system classes")
print("   p* = minimal depth (half-width <= 3) for an exact NON-NEGATIVE coarse law;")
print("   '--' = none at depth <= 8.  'bad g' = largest real-positive unresolved alias at q=0")
print("   (the POSITIONING.md obstruction: non-zero here => provably infeasible at ANY depth).\n")
rows = []
for M in (2, 3, 4):
    print(f"   --- M = {M} " + "-" * 60)
    print(f"   {'system':>24} {'spread':>9} {'max|g_un|':>10} {'bad g':>8} {'#bad':>5} {'p*':>10}")
    for name, g in SYSTEMS:
        sp = alias_spread(g, M)
        mg = max_modulus_unresolved(g, M)
        bad, nbad = sign_diagnostic(g, M)
        P, s = p_star(g, M)
        rows.append(dict(M=M, name=name, spread=sp, maxg=mg, bad=bad, nbad=nbad, P=P, s=s))
        print(f"   {name:>24} {sp:>9.4f} {mg:>10.4f} {bad:>8.4f} {nbad:>5} "
              f"{('P=%d,s=%d' % (P, s)) if P else '--':>10}")

# =========================================================================== 2
hdr("2.  VERDICT ON THE THREE HYPOTHESES")
feas = [d for d in rows if d["P"]]
infeas = [d for d in rows if not d["P"]]
print(f"   {len(feas)} feasible cases, {len(infeas)} infeasible, out of {len(rows)}.\n")

print("   H_sign -- does 'bad g > 0' coincide exactly with infeasibility?")
viol = [d for d in rows if (d["nbad"] > 0) != (d["P"] is None)]
print(f"      cases where the sign diagnostic and feasibility DISAGREE: {len(viol)} / {len(rows)}")
for d in viol:
    print(f"        M={d['M']:<2} {d['name']:>24}  bad={d['bad']:.4f} (#{d['nbad']})  "
          f"p*={('P=%d' % d['P']) if d['P'] else '--'}")

print("\n   H_spread -- is p* a monotone function of the alias spread?")
print("      feasible cases, sorted by spread:")
for d in sorted(feas, key=lambda x: x["spread"]):
    print(f"        spread {d['spread']:>7.4f}  M={d['M']:<2} {d['name']:>24}  P={d['P']}")
print("      infeasible cases, sorted by spread:")
for d in sorted(infeas, key=lambda x: x["spread"])[:8]:
    print(f"        spread {d['spread']:>7.4f}  M={d['M']:<2} {d['name']:>24}  --")
sp_f = [d["spread"] for d in feas]
sp_i = [d["spread"] for d in infeas]
if sp_f and sp_i:
    print(f"      feasible spread range   [{min(sp_f):.4f}, {max(sp_f):.4f}]")
    print(f"      infeasible spread range [{min(sp_i):.4f}, {max(sp_i):.4f}]")
    print(f"      OVERLAP: {'YES -- spread does not separate the two classes' if (min(sp_i) < max(sp_f)) else 'no'}")

print("\n   H_gap -- do the best-separated systems have the smallest p*?")
for nm in ("exact heat r=1.50", "two-timescale .95/.05", "dispersive unitary"):
    for d in rows:
        if d["name"] == nm and d["M"] == 2:
            print(f"      {nm:>24} at M=2: max|g_unresolved| = {d['maxg']:.4f}, "
                  f"p* = {('P=%d' % d['P']) if d['P'] else '--'}")

# ---------------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for d in rows:
    ok = d["P"] is not None
    ax.scatter(d["spread"], d["P"] if ok else 9, marker="o" if ok else "x",
               s=60, c="C0" if ok else "C3", alpha=0.8)
ax.axhline(8.5, color="k", lw=0.8, ls=":")
ax.text(0.02, 9.15, "infeasible (depth > 8)", fontsize=8)
ax.set_xlabel("alias spread"); ax.set_ylabel("p*  (minimal positive depth)")
ax.set_title("H_spread: p* vs alias spread\nfeasible (o) and infeasible (x) overlap in spread", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[1]
for d in rows:
    ok = d["P"] is not None
    ax.scatter(d["bad"], d["P"] if ok else 9, marker="o" if ok else "x",
               s=60, c="C0" if ok else "C3", alpha=0.8)
ax.axhline(8.5, color="k", lw=0.8, ls=":")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("largest real-positive unresolved alias at q=0")
ax.set_ylabel("p*  (minimal positive depth)")
ax.set_title("H_sign: everything feasible sits at exactly 0\n(the obstruction is a sign condition)", fontsize=10)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIG}/task2_hypotheses.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task2_hypotheses.png")
