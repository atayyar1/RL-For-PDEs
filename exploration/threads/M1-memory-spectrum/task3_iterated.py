"""
Task 3 -- iterated coarse-graining, and the thing that actually varies.

The brief asks: coarse-grain by 2 twice versus by 4 once; does memory accumulate,
compose, or saturate?

Part 1 shows that question is a TAUTOLOGY for decimation: the two routes produce the
same observable, hence the same alias set, hence bit-identical coarse laws and an
identical positivity-feasible set.  Memory is a function of the total ratio alone.

Part 2 prices it.  The exact coarse law needs M time levels of an N/M-point field, so
the exact coarse representation costs M * (N/M) = N numbers at EVERY M -- exactly the
fine state.  Demanding exactness, decimation compresses nothing.

Part 3 is the measurement that is not a tautology.  The free choice nobody made is the
RESTRICTION OPERATOR.  Only aliases that survive the restriction need to be reproduced,
and a box average annihilates every unresolved alias at the constant coarse mode --
which is exactly where the positivity obstruction of POSITIONING.md lives.  So the
prediction is sharp: block averaging should admit non-negative coarse laws at M >= 4,
where decimation provably cannot at any depth or width.

Run: python3 task3_iterated.py        (~2 min)
"""
import os
for _v in ("OMP", "OPENBLAS", "MKL", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ.setdefault(_v + "_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

N = 120                       # fine periodic grid, divisible by 2,3,4,5,6,8,10,12
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
SEP = "=" * 78
def hdr(s): print("\n" + SEP + "\n" + s + "\n" + SEP)


def g_sym(theta, r, ra=0.0):
    """FTCS symbol.  ra = 0 is pure diffusion."""
    return (1 - 2 * r) + 2 * r * np.cos(theta) - 1j * ra * np.sin(theta)


def alias_thetas(M, q, n=N):
    """The M fine wavenumbers that alias onto coarse mode q at coarsening M."""
    Nc = n // M
    return 2 * np.pi * (q + Nc * np.arange(M)) / n


# ---------------------------------------------------------------- restrictions
def restrict_weight(kind, M, q, n=N):
    """|W| of the restriction operator at each of the M aliases of coarse mode q.

    Decimation keeps every M-th point: W = 1 on every alias, so every alias must be
    reproduced by the coarse law.
    Box average over M consecutive fine points: W(th_l) = (1/M)(1-e^{i M th_l})/(1-e^{i th_l}).
    Since M*th_l = th_c + 2 pi l, the numerator is the same for every l; at q = 0 it
    VANISHES, so every unresolved alias is annihilated at the constant coarse mode.
    """
    th = alias_thetas(M, q, n)
    if kind == "decimate":
        return np.ones(M)
    num = 1 - np.exp(1j * M * th)
    den = 1 - np.exp(1j * th)
    W = np.where(np.abs(den) < 1e-13, 1.0 + 0j, num / np.where(np.abs(den) < 1e-13, 1.0, den) / M)
    return np.abs(W)


# ---------------------------------------------------------------- feasibility
def exact_depth(M, r, ra=0.0, kind="decimate", n=N, tol=1e-12):
    """Minimal P such that a coarse law of depth P can be exact (no positivity).

    Per coarse mode the surviving aliases must be roots of z^P - sum_j Bhat_j z^{P-j};
    a monic polynomial with a given root set exists at degree = number of DISTINCT
    surviving roots, so the answer is the max over coarse modes of that count.
    """
    Nc = n // M
    best = 0
    for q in range(Nc):
        g = g_sym(alias_thetas(M, q, n), r, ra)
        keep = g[restrict_weight(kind, M, q, n) > 1e-10]
        # count distinct roots
        d = 0
        for i, gi in enumerate(keep):
            if not any(abs(gi - keep[j]) < tol for j in range(i)):
                d += 1
        best = max(best, d)
    return best


def positive_feasible(M, P, s, r, ra=0.0, kind="decimate", n=N, tol=1e-9, drop_Q=-1):
    """min ||b||_1 over EXACT compact laws of depth P and half-width s, b signed.

    Exactness at (q=0, l=0) forces sum b = 1, so the value is >= 1 and equals 1 exactly
    when a NON-NEGATIVE exact law exists.  Returns (defect, b) or None if the LP fails.
    Only aliases surviving the restriction are constrained.
    """
    Nc = n // M
    ks = np.arange(-s, s + 1)
    nb = P * len(ks)
    rows, rhs = [], []
    for q in range(Nc):
        e = np.exp(1j * 2 * np.pi * q * ks / Nc)
        g = g_sym(alias_thetas(M, q, n), r, ra)
        W = restrict_weight(kind, M, q, n)
        # drop_Q >= 0: also delete the UNRESOLVED constraints for coarse modes within
        # drop_Q of the constant mode, to locate how wide the obstructed neighbourhood is.
        near = min(q, Nc - q) <= drop_Q
        for li, (gl, wl) in enumerate(zip(g, W)):
            if wl <= 1e-10:
                continue                      # annihilated by the restriction: no constraint
            if near and li != 0:
                continue
            rows.append(np.outer(gl ** (P - np.arange(1, P + 1)), e).ravel())
            rhs.append(gl ** P)
    A, y = np.array(rows), np.array(rhs)
    Ar = np.vstack([np.hstack([A.real, -A.real]), np.hstack([A.imag, -A.imag])])
    br = np.concatenate([y.real, y.imag])
    res = linprog(np.ones(2 * nb), A_eq=Ar, b_eq=br, bounds=[(0, None)] * 2 * nb, method="highs")
    if res.status != 0:
        return None
    b = res.x[:nb] - res.x[nb:]
    return float(np.abs(b).sum()) - 1.0, b.reshape(P, len(ks))


def p_star(M, r, ra=0.0, kind="decimate", P_max=10, s_max=3, tol=1e-9, drop_Q=-1):
    """Smallest (P, s) admitting an exact NON-NEGATIVE compact coarse law."""
    for P in range(1, P_max + 1):
        for s in range(0, s_max + 1):
            out = positive_feasible(M, P, s, r, ra, kind, tol=tol, drop_Q=drop_Q)
            if out is not None and out[0] < tol:
                return P, s
    return None, None


# =========================================================================== 1
hdr("1.  ROUTE INDEPENDENCE -- the (2 then 2) vs (4 at once) comparison is a tautology")
print("""   Decimating by 2 and then by 2 again retains fine indices 0,4,8,... -- the SAME
   index set as decimating by 4.  The coarse law is a property of the observable, so
   the two routes cannot differ.  Concretely, the level-2 alias class of coarse mode q
   is the union over the two level-1 modes {q, q+N/4} of their level-1 alias pairs,
   which is exactly the 4-decimation alias class of q.  Checked numerically:""")
r = 0.45
print(f"\n   {'total M':>8} {'route':>22} {'alias set (sorted, coarse mode q=1)':>40} {'max diff vs direct':>20}")
for Mtot, steps in [(4, [2, 2]), (8, [2, 2, 2]), (8, [2, 4]), (6, [2, 3])]:
    Nc = N // Mtot
    q = 1
    direct = np.sort_complex(g_sym(alias_thetas(Mtot, q), r))
    # iterated: track the fine modes reaching coarse mode q after the given steps.
    # After cumulative ratio C the level has N/C modes, and aliasing by `st` maps
    # mode m -> {m + (N/(C*st))*j}.  Wavenumber indices stay absolute throughout.
    modes, C = np.array([q]), 1
    for st in steps:
        C *= st
        modes = np.concatenate([modes + (N // C) * j for j in range(st)])
    iterated = np.sort_complex(g_sym(2 * np.pi * modes / N, r))
    d = np.abs(direct - iterated).max()
    print(f"   {Mtot:>8} {'x'.join(map(str, steps)):>22} {str(np.round(direct.real, 4)):>40} {d:>20.2e}")
print("""
   Every route returns the identical alias set, therefore the identical characteristic
   polynomial, therefore bit-identical B_p and an identical positivity-feasible set.
   Memory neither accumulates, composes, nor saturates: it is a function of the TOTAL
   coarsening ratio and nothing else.  There is no measurement to make here.""")

# =========================================================================== 2
hdr("2.  THE PRICE OF EXACTNESS IS THE WHOLE COMPRESSION")
print("""   The exact coarse law needs `exact depth` time levels of an N/M-point field.
   State cost = depth * N/M.  Compare with the fine state, N.""")
print(f"\n   {'M':>4} {'N/M cells':>11} {'exact depth':>13} {'coarse state':>14} {'vs fine N=120':>15}")
for M in [2, 3, 4, 5, 6, 8, 10, 12]:
    d = exact_depth(M, r, ra=0.0)
    print(f"   {M:>4} {N // M:>11} {d:>13} {d * (N // M):>14} {d * (N // M) / N:>15.3f}")
print("""
   Exactly 1.000 at every M.  Decimation plus exact memory is a change of basis, not a
   compression: you get back precisely the state you saved.  So "does compression across
   scales have a bounded price?" has an exact answer in this setting -- the price is
   100%, at every ratio, by every route.  Compression only exists if you accept a
   tolerance, and then the depth is set by how fast the unresolved aliases decay, which
   is the classical Mori-Zwanzig / spectral-gap story.""")

# =========================================================================== 3
hdr("3.  THE FREE CHOICE THAT ACTUALLY MATTERS: the restriction operator")
print("""   Only aliases that SURVIVE the restriction need to be reproduced.  Decimation keeps
   all M (W = 1 everywhere).  A box average over M fine points has
        W(th_l) = (1/M)(1 - e^{i M th_l}) / (1 - e^{i th_l}),   M th_l = th_c + 2 pi l,
   so at the constant coarse mode (th_c = 0) the numerator vanishes and W = 0 on every
   unresolved alias.  That is precisely where the positivity obstruction lives:
   POSITIONING.md shows a non-negative coarse law requires every unresolved alias at
   q = 0 to have g <= 0, which fails for M >= 4 at every admissible r.
   PREDICTION: block averaging voids the obstruction and admits positive laws at M >= 4.""")
print(f"\n   surviving unresolved aliases at q=0, M=4, r=0.45:")
for kind in ("decimate", "box"):
    W = restrict_weight(kind, 4, 0)
    g = g_sym(alias_thetas(4, 0), r).real
    print(f"     {kind:>9}: |W| = {np.round(W, 6)}   g = {np.round(g, 4)}   "
          f"unresolved kept: {int((W[1:] > 1e-10).sum())}")

rs = [0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
Ms = [2, 3, 4, 5, 6, 8]
res = {}
for kind in ("decimate", "box"):
    print(f"\n   --- p* (depth P, half-width s) for an exact NON-NEGATIVE coarse law, {kind} ---")
    print(f"   {'M':>3} " + "".join(f"{'r=%.2f' % rr:>12}" for rr in rs))
    for M in Ms:
        row = []
        for rr in rs:
            P, s = p_star(M, rr, 0.0, kind)
            res[(kind, M, rr)] = (P, s)
            row.append(f"{'P=%d,s=%d' % (P, s) if P else '--':>12}")
        print(f"   {M:>3} " + "".join(row))

print("""
   '--' means no exact non-negative law at depth <= 10, half-width <= 3.

   *** THE PREDICTION FAILED. ***  Box averaging deletes exactly the q=0 constraints that
   carry the obstruction, and M >= 4 is STILL infeasible at every r.  The restriction
   operator is very nearly irrelevant (it moves one cell: M=2, r=0.25 goes P=9 -> P=8).""")

# =========================================================================== 4
hdr("4.  WHY the prediction failed: the obstruction is a NEIGHBOURHOOD, not a point")
print("""   b is a fixed finite set of numbers and the exactness constraints are continuous in
   the coarse wavenumber th_c, so satisfying them at th_c = 2 pi / Nc (adjacent to the
   constant mode) very nearly forces the th_c = 0 condition.  Deleting one point of a
   near-continuum cannot help.  Measured: delete the unresolved constraints for every
   coarse mode within drop_Q of the constant mode, and find where feasibility appears.""")
print(f"\n   M=4, r=0.45, pure diffusion, depth <= 10, half-width <= 3.  Nc = {N // 4} coarse modes.")
print(f"   {'drop_Q':>8} {'modes deleted':>15} {'fraction of band':>18} {'p* (P,s)':>14}")
for Q in [-1, 0, 1, 2, 3, 5, 8, 11, 14]:
    P, s = p_star(4, 0.45, 0.0, "decimate", drop_Q=Q)
    ndel = 0 if Q < 0 else min(2 * Q + 1, N // 4)
    tag = f"P={P},s={s}" if P else "--"
    print(f"   {Q:>8} {ndel:>15} {ndel / (N // 4):>18.2f} {tag:>14}")
print("""
   Feasibility only appears once a large fraction of the spectral band is exempted, which
   is not a coarse-graining any more.  The obstruction is robust: it is a property of the
   low-wavenumber END of the spectrum, not of the single constant mode, and no choice of
   restriction operator that keeps the coarse field a local average of the fine one can
   evade it.  This CLOSES the decimation-vs-block-averaging question raised in
   POSITIONING.md section 4 -- I flagged it as load-bearing and it is not.""")

# ---------------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
for M in [2, 3, 4, 5, 6, 8, 10, 12]:
    d = exact_depth(M, r, ra=0.0)
    ax.plot(M, d * (N // M) / N, 'o', color='C0', ms=7)
ax.axhline(1.0, color='k', lw=1, ls='--')
ax.set_xlabel("coarsening factor M"); ax.set_ylabel("coarse state / fine state")
ax.set_ylim(0, 1.25)
ax.set_title("exact memory costs exactly what\ndecimation saves, at every M", fontsize=10)
ax.grid(alpha=0.25)

for ai, kind in enumerate(("decimate", "box")):
    ax = axes[1 + ai]
    Z = np.full((len(Ms), len(rs)), np.nan)
    for i, M in enumerate(Ms):
        for j, rr in enumerate(rs):
            P, _ = res[(kind, M, rr)]
            Z[i, j] = P if P else np.nan
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis", vmin=1, vmax=10,
                   extent=[-0.5, len(rs) - 0.5, -0.5, len(Ms) - 0.5])
    for i in range(len(Ms)):
        for j in range(len(rs)):
            ax.text(j, i, "-" if np.isnan(Z[i, j]) else f"{int(Z[i, j])}",
                    ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(len(rs))); ax.set_xticklabels([f"{x:.2f}" for x in rs], fontsize=8)
    ax.set_yticks(range(len(Ms))); ax.set_yticklabels(Ms, fontsize=8)
    ax.set_xlabel("diffusion number r"); ax.set_ylabel("coarsening M")
    ax.set_title(f"p* (depth) -- {kind}\n'-' = infeasible at any depth <= 10", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout()
fig.savefig(f"{FIG}/task3_restriction.png", dpi=150, bbox_inches="tight")
print(f"\n   wrote {FIG}/task3_restriction.png")
