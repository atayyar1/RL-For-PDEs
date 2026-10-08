"""
Task 4b -- decompose the recursion  e* = sum_i w_i e_i + tau.

Task 4 measures the composite outcome.  The two terms trade off:
  * positivity caps the AMPLIFICATION term at ||w||_1 = 1, but
  * insisting on positivity restricts which geometries you may use, which can
    make the LOCAL TRUNCATION term tau larger.

tau is measured exactly, with no rollout: apply the stencil to the analytic
solution,   tau = u_true(z*) - sum_i w_i u_true(z_i),   so nothing propagates.

Also measured: how loose the worst-case bound |e*| <= ||w||_1 max|e_i| really
is when the incoming errors are NOT sign-aligned with w.  This is the reason
the Task 4 effect sizes are far below ||w||_1^L.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stencil import (ALPHA, C, DX, DT, NX, X_FD, u_true, w_lstsq, w_minL1,
                     w_positive_enum, positive_feasible_hull, moments_solvable)

FIG = "figures"
rng = np.random.default_rng(11)

WINDOWS = {"benign": (2, 3), "wide": (4, 5), "sprint": (3, 5)}


def sample(sel, rg):
    """One target point and a candidate neighbour set, as in task4."""
    i = int(rg.integers(6, NX - 6))
    lev = int(rg.integers(8, 30))
    M, K = WINDOWS[sel]
    if sel == "sprint":
        s = 1 if rg.random() < 0.5 else -1
        cells = [(i + s * d, lev - k) for k in range(1, K + 1)
                 for d in range(0, M + 1)]
    else:
        cells = [(i + d, lev - k) for k in range(1, K + 1)
                 for d in range(-M, M + 1)]
    cells = [cc for cc in cells if 0 <= cc[0] < NX and cc[1] >= 0]
    if len(cells) < 5:
        return None
    pick = [cells[j] for j in rg.choice(len(cells), 5, replace=False)]
    dxi = np.array([X_FD[q[0]] - X_FD[i] for q in pick])
    dti = np.array([(q[1] - lev) * DT for q in pick])
    if not moments_solvable(dxi, dti):
        return None
    return i, lev, pick, dxi, dti


print("=" * 92)
print("LOCAL TRUNCATION ERROR tau = u_true(z*) - sum_i w_i u_true(z_i)")
print("=" * 92)
print(f"  {'selector':>8} {'policy':>10} {'median |tau|':>13} {'p90 |tau|':>12} "
      f"{'mean ||w||_1':>13} {'feasible':>10} {'median h/dx':>12}")

store = {}
for sel in WINDOWS:
    rows = {k: [] for k in ("lstsq", "minL1", "mask")}
    l1s = {k: [] for k in rows}
    hs = {k: [] for k in rows}
    nfeas = 0
    ntot = 0
    rg = np.random.default_rng(5)
    while ntot < 6000:
        s = sample(sel, rg)
        if s is None:
            continue
        i, lev, pick, dxi, dti = s
        ntot += 1
        u_nb = np.array([u_true(X_FD[q[0]], q[1] * DT)[0] for q in pick])
        u_st = float(u_true(X_FD[i], lev * DT)[0])
        h = max(np.max(np.abs(dxi)), np.sqrt(ALPHA * np.max(np.abs(dti))))

        wl = w_lstsq(dxi, dti)
        rows["lstsq"].append(u_st - wl @ u_nb)
        l1s["lstsq"].append(np.abs(wl).sum())
        hs["lstsq"].append(h)

        wp = w_positive_enum(dxi, dti)
        if wp is not None:
            nfeas += 1
            wm = wp
        else:
            wm, _ = w_minL1(dxi, dti)
            if wm is None:
                wm = wl
        rows["minL1"].append(u_st - wm @ u_nb)
        l1s["minL1"].append(np.abs(wm).sum())
        hs["minL1"].append(h)

        # mask: resample up to 8 times until positivity is feasible
        chosen = None
        for _ in range(8):
            s2 = sample(sel, rg)
            if s2 is None:
                continue
            if positive_feasible_hull(s2[3], s2[4]):
                chosen = s2
                break
        if chosen is None:
            chosen = s
        i2, lev2, pick2, dxi2, dti2 = chosen
        w2 = w_positive_enum(dxi2, dti2)
        if w2 is None:
            w2, _ = w_minL1(dxi2, dti2)
        if w2 is None:
            w2 = w_lstsq(dxi2, dti2)
        u_nb2 = np.array([u_true(X_FD[q[0]], q[1] * DT)[0] for q in pick2])
        u_st2 = float(u_true(X_FD[i2], lev2 * DT)[0])
        h2 = max(np.max(np.abs(dxi2)), np.sqrt(ALPHA * np.max(np.abs(dti2))))
        rows["mask"].append(u_st2 - w2 @ u_nb2)
        l1s["mask"].append(np.abs(w2).sum())
        hs["mask"].append(h2)

    store[sel] = (rows, l1s, hs, nfeas / ntot)
    for pol in ("lstsq", "minL1", "mask"):
        t = np.abs(np.array(rows[pol]))
        print(f"  {sel:>8} {pol:>10} {np.median(t):>13.4e} "
              f"{np.percentile(t,90):>12.4e} {np.mean(l1s[pol]):>13.4f} "
              f"{(nfeas/ntot if pol=='minL1' else float('nan')):>10.3f} "
              f"{np.median(hs[pol])/DX:>12.3f}")
    print()

print("=" * 92)
print("HOW LOOSE IS THE WORST-CASE BOUND |e*| <= ||w||_1 max|e_i| ?")
print("=" * 92)
print("  Draw random incoming errors e_i and compare the realised |sum w_i e_i|")
print("  with the bound ||w||_1 max|e_i|, for three error models.")
rg = np.random.default_rng(7)
sel = "sprint"
sampled = []
while len(sampled) < 3000:
    s = sample(sel, rg)
    if s is None:
        continue
    _, _, _, dxi, dti = s
    w = w_lstsq(dxi, dti)
    if np.abs(w).sum() > 1.05:
        sampled.append(w)
print(f"  {len(sampled)} lstsq stencils with ||w||_1 > 1.05 "
      f"(mean {np.mean([np.abs(w).sum() for w in sampled]):.3f})")
print(f"  {'error model':>34} {'median realised/bound':>23}")
for name, gen in [
    ("adversarial: e_i = sign(w_i)", lambda w, rg: np.sign(w)),
    ("iid Gaussian", lambda w, rg: rg.standard_normal(len(w))),
    ("smooth (all e_i equal)", lambda w, rg: np.ones(len(w))),
]:
    ratios = []
    for w in sampled:
        e = gen(w, rg)
        bound = np.abs(w).sum() * np.abs(e).max()
        ratios.append(abs(w @ e) / bound)
    print(f"  {name:>34} {np.median(ratios):>23.4f}")
print("\n  The bound is TIGHT only for the sign-aligned model.  For iid errors the")
print("  realised amplification is a small fraction of ||w||_1, and for perfectly")
print("  smooth errors it collapses to 1/||w||_1 (because sum_i w_i = 1 exactly).")
print("  THIS is why the Task 4 effect sizes are far below ||w||_1^L: a marching")
print("  rollout on a smooth solution generates errors that are neither adversarial")
print("  nor iid, but strongly spatially correlated -- the regime the bound handles")
print("  worst.  Positivity is insurance against the adversarial case, and the")
print("  premium is small; it is not a large expected-case win.")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
cols = {"lstsq": "#c0392b", "minL1": "#2874a6", "mask": "#1e8449"}
for j, sel in enumerate(WINDOWS):
    rows, l1s, hs, ff = store[sel]
    a = ax[j]
    for pol in ("lstsq", "minL1", "mask"):
        t = np.abs(np.array(rows[pol]))
        t = t[t > 0]
        a.hist(np.log10(t), bins=60, alpha=0.5, color=cols[pol],
               label=f"{pol}: med {np.median(t):.1e}, "
                     r"$\overline{\|w\|_1}$=" + f"{np.mean(l1s[pol]):.2f}")
    a.set_xlabel(r"$\log_{10}|\tau|$  (local truncation error)")
    a.set_ylabel("count" if j == 0 else "")
    a.set_title(f"selector = {sel}  (feasible {ff:.0%})")
    a.legend(fontsize=7.5)
fig.suptitle("Task 4b -- truncation error is the price of the positivity constraint",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG}/task4b_truncation.png", dpi=150)
print(f"\nfigure -> {FIG}/task4b_truncation.png")
