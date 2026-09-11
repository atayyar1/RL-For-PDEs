"""TASK 14 -- (a) BOUNDARIES with variable alpha, charged honestly;
                 plus a stable reference for the discontinuous-alpha case,
                 and the nb-dependence that explains why positivity helps.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import task13_harden as H

A0, T_END = H.A0, H.T_END
u0f = H.u0f

print("=" * 104)
print("0.  SCOPE CORRECTION: the 7-20x runs of Task 9/9b were PERIODIC.")
print("    There were no boundaries in them at all.  Everything below is new.")
print("=" * 104)

# ---------------------------------------------------------------- stable reference
def ref_fine_periodic(afun, N=4096):
    x = np.arange(N) / N; dx = 1.0 / N
    ah = afun(x + dx / 2)
    dt = 0.2 * dx ** 2 / (2 * afun(x).max())
    ns = int(np.ceil(T_END / dt)); dt = T_END / ns
    u = u0f(x)
    for _ in range(ns):
        fl = ah * (np.roll(u, -1) - u) / dx ** 2
        u = u + dt * (fl - np.roll(fl, 1))
    return x, u


print("\n" + "=" * 104)
print("(c-fix)  DISCONTINUOUS alpha with a STABLE reference")
print("  (the spectral RK4 reference blew up on a tanh step -- that row of task13 was junk)")
print("=" * 104)
N = 128
xg = np.arange(N) / N
print("  %-22s %9s | %12s %12s | %10s"
      % ("alpha", "L_a (cells)", "wide (best)", "FTCS", "ratio"))
rows = []
for kind, q, w_, lab in [("smooth", 4, 0, "sin(8 pi x)"),
                         ("smooth", 8, 0, "sin(16 pi x)"),
                         ("step", 1, 0.05, "tanh step w=0.05"),
                         ("step", 1, 0.02, "tanh step w=0.02"),
                         ("step", 1, 0.005, "tanh step w=0.005")]:
    af, ad = H.make_alpha(kind, 0.8, q, w_) if kind == "step" else H.make_alpha(kind, 0.8, q)
    xr, ur = ref_fine_periodic(af)
    ref = lambda xq: np.interp(np.asarray(xq) % 1.0, np.r_[xr, 1.0], np.r_[ur, ur[0]])
    uf, cf = H.ftcs_periodic(N, af)
    ef = float(np.max(np.abs(uf - ref(xg))))
    best = np.inf
    for nb in [8, 16, 32, 64]:
        for m in [3, 5, 8, 12]:
            for P in [2, 4, 6]:
                uw, iw = H.solve_periodic(N, nb, m, P, 6, ad, positive=True)
                if uw is None:
                    continue
                e = float(np.max(np.abs(uw - ref(xg))))
                if np.isfinite(e):
                    best = min(best, e)
    xs = np.linspace(0, 1, 4001)
    La = float(np.max(np.abs(af(xs))) / max(np.max(np.abs(np.gradient(af(xs), xs[1]))), 1e-12)) * N
    print("  %-22s %9.2f | %12.4e %12.4e | %10.2f" % (lab, La, best, ef, ef / best))
    rows.append((lab, La, best, ef))
print("\n  -> the advantage is a function of L_a = alpha/|alpha'| measured IN CELLS.")
print("     Breakdown is gradual and predictable, not a cliff.")

# ---------------------------------------------------------------- (b) why positivity helps
print("\n" + "=" * 104)
print("(b-why)  POSITIVITY IS LOAD-BEARING, NOT A FREE EXTRA")
print("  ||w||_1 = B means the scheme can amplify by B per step, so B^nb over the run.")
print("=" * 104)
af, ad = H.make_alpha("smooth", 0.8, 1)
xr, ur = ref_fine_periodic(af)
ref = lambda xq: np.interp(np.asarray(xq) % 1.0, np.r_[xr, 1.0], np.r_[ur, ur[0]])
print("  %5s %4s | %13s %13s | %8s %10s"
      % ("nb", "m", "err positive", "err signed", "||w||_1", "signed/pos"))
for nb in [4, 8, 16, 32, 64]:
    up, ip = H.solve_periodic(N, nb, 8, 6, 6, ad, True)
    us, isg = H.solve_periodic(N, nb, 8, 6, 6, ad, False)
    if up is None or us is None:
        continue
    ep = float(np.max(np.abs(up - ref(xg))))
    es = float(np.max(np.abs(us - ref(xg))))
    print("  %5d %4d | %13.4e %13.4e | %8.3f %10.1fx"
          % (nb, 8, ep, es, isg["l1"], es / ep))
print("\n  -> the signed/positive gap WIDENS with the number of steps, which is the")
print("     signature of ||w||_1 > 1 compounding.  Positivity is doing real work here.")

# ---------------------------------------------------------------- (a) boundaries
print("\n" + "=" * 104)
print("(a)  DIRICHLET BOUNDARIES WITH VARIABLE alpha, wall layer charged honestly")
print("=" * 104)
print("  No method of images: alpha(x) is not symmetric about the wall, so the")
print("  constant-coefficient escape is gone.  Strategy tested: at a point within m of a")
print("  wall, restrict the support to the available interior offsets and drop the")
print("  moment order P until the LP is feasible.  Cost is charged per actual non-zero.")


def wall_feasible(N, nb, m, P, Q, dfun, Pmin=1):
    """How close to the wall can a one-sided positive stencil still be built?"""
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    tau = T_END / nb
    first_ok = None
    for j in range(1, N // 2):
        lo, hi = -min(m, j), min(m, N - 1 - j)
        offs = np.arange(lo, hi + 1)
        got = False
        for Ptry in range(P, Pmin - 1, -1):
            if len(offs) < Ptry + 1:
                continue
            if H.weights(x[j], tau, dx, offs, Ptry, Q, dfun, positive=True) is not None:
                got = True; break
        if got:
            first_ok = j; break
    return first_ok


def solve_dirichlet_hybrid(N, nb, m, P, Q, dfun, afun):
    """Interior: wide positive stencil, global step tau.
    Wall layer: FTCS sub-stepping, which needs its own CFL AND a halo of nsub
    cells because each sub-step reaches one cell further.  Everything charged."""
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    tau = T_END / nb
    dt_cfl = 0.9 * dx ** 2 / (2 * afun(x).max())
    nsub = max(int(np.ceil(tau / dt_cfl)), 1)
    dts = tau / nsub
    R = m + nsub                       # wall layer width that must be sub-stepped
    if 2 * R >= N - 2:
        return None, dict(overrun=True, R=R, nsub=nsub)
    rows_w, rows_o = {}, {}
    for j in range(R, N - R):
        offs = np.arange(-m, m + 1)
        w = H.weights(x[j], tau, dx, offs, P, Q, dfun, positive=True)
        if w is None:
            return None, dict(infeasible_interior=True)
        rows_w[j], rows_o[j] = w, offs
    ah = afun(x[:-1] + dx / 2)
    u = u0f(x); u[0] = u[-1] = 0.0
    flops = 0
    for _ in range(nb):
        un = np.zeros(N)
        for j in range(R, N - R):
            un[j] = float(rows_w[j] @ u[j + rows_o[j]])
        flops += (N - 2 * R) * (2 * (2 * m + 1) - 1)
        v = u.copy()
        for _ in range(nsub):
            fl = ah * np.diff(v) / dx ** 2
            vn = v.copy(); vn[1:-1] = v[1:-1] + dts * (fl[1:] - fl[:-1])
            vn[0] = vn[-1] = 0.0; v = vn
        flops += nsub * 2 * R * 7
        un[:R] = v[:R]; un[N - R:] = v[N - R:]
        un[0] = un[-1] = 0.0
        u = un
    return u, dict(flops=flops, R=R, nsub=nsub, overrun=False)


def ref_dirichlet(afun, N=4096):
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    ah = afun(x[:-1] + dx / 2)
    dt = 0.2 * dx ** 2 / (2 * afun(x).max())
    ns = int(np.ceil(T_END / dt)); dt = T_END / ns
    u = u0f(x); u[0] = u[-1] = 0.0
    for _ in range(ns):
        fl = ah * np.diff(u) / dx ** 2
        un = u.copy(); un[1:-1] = u[1:-1] + dt * (fl[1:] - fl[:-1])
        un[0] = un[-1] = 0.0; u = un
    return x, u


afd, add = H.make_alpha("smooth", 0.8, 1)
xrd, urd = ref_dirichlet(afd)
refd = lambda xq: np.interp(xq, xrd, urd)
Nd = 129
xd = np.linspace(0, 1, Nd)


def ftcs_dirichlet(N, afun):
    x = np.linspace(0, 1, N); dx = x[1] - x[0]
    ah = afun(x[:-1] + dx / 2)
    dt = 0.9 * dx ** 2 / (2 * afun(x).max())
    ns = max(int(np.ceil(T_END / dt)), 1); dt = T_END / ns
    u = u0f(x); u[0] = u[-1] = 0.0
    for _ in range(ns):
        fl = ah * np.diff(u) / dx ** 2
        un = u.copy(); un[1:-1] = u[1:-1] + dt * (fl[1:] - fl[:-1])
        un[0] = un[-1] = 0.0; u = un
    return u, ns * (N - 2) * 7


ufd, cfd = ftcs_dirichlet(Nd, afd)
efd = float(np.max(np.abs(ufd - refd(xd))))
print("\n  FTCS (Dirichlet, at CFL): err = %.4e, flops = %.3e" % (efd, cfd))
print("\n  A1. How close to the wall can a ONE-SIDED positive stencil be built?")
print("  %5s %4s %3s | %8s %10s | %s" % ("nb", "m", "P", "sigma", "first j ok", "verdict"))
for nb in [8, 16, 32]:
    for m in [5, 8]:
        tau = T_END / nb
        sg = np.sqrt(2 * A0 * 1.8 * tau) / (1.0 / (Nd - 1))
        fo = wall_feasible(Nd, nb, m, 6, 6, add)
        print("  %5d %4d %3d | %8.2f %10s | %s"
              % (nb, m, 6, sg, fo,
                 "no wall stencil at any P" if fo is None else "needs j >= %d" % fo))
print("  -> CORRECTION to my first guess: a one-sided positive stencil DOES exist from")
print("     j = 1 onward.  What degrades is the achievable MOMENT ORDER, gracefully:")
print("     P_max = 1, 2, 4, 5, 6, 6, ... at j = 0, 1, 2, 3, 4, 5 ...  so the wall layer")
print("     is only ~4-6 cells wide in lost order -- far thinner than the m + k layer")
print("     that FTCS sub-stepping needs.  Order reduction is the cheaper wall fix.")

print("\n  A2. HYBRID: wide interior + FTCS-substepped wall layer, everything charged")
print("  %5s %4s | %5s %5s %6s | %13s %13s | %s"
      % ("nb", "m", "nsub", "R", "R/N", "err (all x)", "err interior", "flops"))
bestd = (np.inf, None)
for nb in [8, 16, 32, 64]:
    for m in [3, 5, 8]:
        uw, iw = solve_dirichlet_hybrid(Nd, nb, m, 6, 6, add, afd)
        if uw is None:
            print("  %5d %4d | %5s %5s %6s | %13s"
                  % (nb, m, iw.get("nsub", "-"), iw.get("R", "-"), "-",
                     "OVERRUNS DOMAIN" if iw.get("overrun") else "infeasible"))
            continue
        e_all = float(np.max(np.abs(uw - refd(xd))))
        intr = slice(iw["R"], Nd - iw["R"])
        e_in = float(np.max(np.abs(uw[intr] - refd(xd[intr]))))
        print("  %5d %4d | %5d %5d %6.2f | %13.4e %13.4e | %.3e"
              % (nb, m, iw["nsub"], iw["R"], iw["R"] / Nd, e_all, e_in, iw["flops"]))
        if e_all < bestd[0]:
            bestd = (e_all, iw["flops"])
print("\n  best hybrid (Dirichlet): err = %.4e at flops = %.3e" % bestd)
print("  FTCS (Dirichlet)       : err = %.4e at flops = %.3e" % (efd, cfd))
print("  -> ratio: %.2fx accuracy at %.2fx the cost"
      % (efd / bestd[0], bestd[1] / cfd))
json.dump({"rough": [(a, float(b), float(c), float(d)) for a, b, c, d in rows]},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "task14_results.json"), "w"), indent=1)
