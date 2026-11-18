"""TASK 5 -- the (m,k) decision problem, framed as an RL agent would see it.

State  : local smoothness (effective wavenumber theta = xi dx) and alpha (via nu).
Action : (m, k)  -- stencil half-width and number of base steps to leap.
Reward : accuracy per unit cost.  We use the scalarisation

    R_lambda(m,k) = -log10( LTE per unit simulated time )
                    - lambda * log10( cost per unit simulated time )

everything measured EXACTLY from the symbol:
    err_step(theta) = |W(theta) - exp(-i k Co theta - k nu theta^2)|
    err_rate        = err_step / (k dt)
    cost_rate       = (2P-1) / (k dt)      [flops per grid point per unit time]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
DX, DT = C.DX, C.DT


def eval_mk(m, k, theta, nu, co, design="best"):
    """Return (err_rate, cost_rate) for action (m,k) at local wavenumber theta."""
    mu, var = -k * co, 2 * k * nu
    M2 = var + mu ** 2
    if M2 > m ** 2:
        return np.inf, np.inf                      # positivity infeasible
    w = C.kernel_moment_matched(m, mu, M2)
    if w is None:
        p = M2 / m ** 2
        w = np.zeros(2 * m + 1); w[0] = w[-1] = p / 2; w[m] = 1 - p
        j = np.arange(-m, m + 1)
        need = mu - (w * j).sum()
        if abs(need) > w[m]:
            return np.inf, np.inf
        w[m] -= abs(need); w[m + int(np.sign(need))] += abs(need)
    e = abs(C.symbol_error(w, k, nu, co, np.array([theta]))[0])
    e = max(e, 1e-14)          # double-precision / aliasing floor: below this the
                               # 'optimum' would be chasing roundoff noise
    P = int((w > 1e-16).sum())
    return e / (k * DT), (2 * P - 1) / (k * DT)


# Action-space caps are PHYSICAL, not arbitrary:
#   m <= MMAX  : a stencil cannot be centred within m cells of a wall (N=100 grid)
#   k <= KMAX  : k*dt must not overshoot the simulation horizon (k*dt <= ~T_end=0.2)
MMAX, KMAX = 30, 400


def landscape(theta, nu, co, mmax=MMAX, kmax=KMAX):
    ms = np.arange(1, mmax + 1)
    ks = np.unique(np.round(np.logspace(0, np.log10(kmax), 40)).astype(int))
    E = np.full((len(ms), len(ks)), np.inf)
    Cst = np.full((len(ms), len(ks)), np.inf)
    for i, m in enumerate(ms):
        for j, k in enumerate(ks):
            E[i, j], Cst[i, j] = eval_mk(m, k, theta, nu, co)
    return ms, ks, E, Cst


def main():
    nu, co = C.NU_REF, 0.0            # pure diffusion: cleanest decision problem
    print("#" * 104)
    print("#  THE (m,k) DECISION PROBLEM.  nu = %.3f, dx = %.5f, dt = %.3e" % (nu, DX, DT))
    print("#" * 104)

    # local smoothness proxy: theta = xi*dx.  theta=pi*dx <-> mode 1; larger = rougher
    thetas = [np.pi * DX, 4 * np.pi * DX, 16 * np.pi * DX, 0.5, 1.0, 2.0]
    labels = ["very smooth (mode 1)", "smooth (mode 4)", "moderate (mode 16)",
              "rough (th=0.5)", "rough (th=1.0)", "near-grid (th=2.0)"]

    out = {}
    for lam in [0.5, 1.0, 2.0]:
        print("\n" + "=" * 104)
        print("lambda = %.1f   (reward = -log10 err_rate - %.1f log10 cost_rate)"
              % (lam, lam))
        print("=" * 104)
        print("%-22s %6s | %4s %5s %5s %8s | %11s %11s | %8s %8s"
              % ("state (smoothness)", "theta", "m*", "k*", "s*", "R*",
                 "err_rate*", "cost_rate*", "R(1,1)", "R*-R(1,1)"))
        for th, lab in zip(thetas, labels):
            ms, ks, E, Cst = landscape(th, nu, co)
            R = -np.log10(np.maximum(E, 1e-300)) - lam * np.log10(np.maximum(Cst, 1e-300))
            R[~np.isfinite(E)] = -np.inf
            i, j = np.unravel_index(np.argmax(R), R.shape)
            e11, c11 = eval_mk(1, 1, th, nu, co)
            R11 = -np.log10(max(e11, 1e-300)) - lam * np.log10(c11)
            sstar = ms[i] / np.sqrt(2 * ks[j] * nu)
            print("%-22s %6.3f | %4d %5d %5.2f %8.3f | %11.3e %11.3e | %8.3f %8.3f"
                  % (lab, th, ms[i], ks[j], sstar, R[i, j], E[i, j], Cst[i, j],
                     R11, R[i, j] - R11))
            out.setdefault("lam%.1f" % lam, []).append(
                dict(theta=th, label=lab, m=int(ms[i]), k=int(ks[j]),
                     s=float(sstar), R=float(R[i, j]), R11=float(R11)))

    # ------------------------------------------------------- tolerance-constrained
    print("\n" + "=" * 104)
    print("TOLERANCE-CONSTRAINED OPTIMUM: cheapest (m,k) with err_rate <= eps")
    print("(this is the decision an agent actually faces; the scalarised reward above")
    print(" has its optimum ON THE BOUNDARY of the action space, which is degenerate)")
    print("=" * 104)
    tol_rows = []
    for eps in [1e-3, 1e-5, 1e-7, 1e-9]:
        print("\n  eps = %.0e  (LTE per unit simulated time)" % eps)
        print("  %-22s | %4s %6s %6s | %12s | %12s %10s"
              % ("state", "m*", "k*", "s*", "cost_rate*", "cost(1,1)", "speedup"))
        for th, lab in zip(thetas, labels):
            ms, ks, E, Cst = landscape(th, nu, co)
            ok = E <= eps
            if not ok.any():
                print("  %-22s | %s" % (lab, "  unreachable"))
                continue
            Cm = np.where(ok, Cst, np.inf)
            i, j = np.unravel_index(np.argmin(Cm), Cm.shape)
            e11, c11 = eval_mk(1, 1, th, nu, co)
            s = ms[i] / np.sqrt(2 * ks[j] * nu)
            feas11 = e11 <= eps
            print("  %-22s | %4d %6d %6.2f | %12.4e | %12.4e %9.2fx%s"
                  % (lab, ms[i], ks[j], s, Cst[i, j], c11, c11 / Cst[i, j],
                     "" if feas11 else "  (m=k=1 MISSES eps)"))
            tol_rows.append(dict(eps=eps, label=lab, m=int(ms[i]), k=int(ks[j]),
                                 s=float(s), cost=float(Cst[i, j]),
                                 cost11=float(c11), feas11=bool(feas11)))
    out["tolerance"] = tol_rows

    # ---------------------------------------------------------------- flatness
    print("\n" + "=" * 104)
    print("FLATNESS OF THE OPTIMUM  (how much reward is lost by a wrong action?)")
    print("=" * 104)
    lam = 1.0
    for th, lab in zip(thetas[:4], labels[:4]):
        ms, ks, E, Cst = landscape(th, nu, co)
        R = -np.log10(np.maximum(E, 1e-300)) - lam * np.log10(np.maximum(Cst, 1e-300))
        R[~np.isfinite(E)] = -np.inf
        Rm = R.max()
        finite = R[np.isfinite(R)]
        frac = [float(np.mean(finite >= Rm - d)) for d in [0.1, 0.3, 1.0, 3.0]]
        i, j = np.unravel_index(np.argmax(R), R.shape)
        # sensitivity to +-1 in m and factor 2 in k
        def safe(a, b):
            return R[a, b] if 0 <= a < R.shape[0] and 0 <= b < R.shape[1] else -np.inf
        dm = min(Rm - safe(i - 1, j), Rm - safe(i + 1, j))
        dk = min(Rm - safe(i, j - 1), Rm - safe(i, j + 1))
        print("%-22s R*=%7.3f | frac of feasible actions within 0.1/0.3/1/3 of R*: "
              "%5.1f%% %5.1f%% %5.1f%% %5.1f%% | dR for m+-1: %6.3f  k+-1step: %6.3f"
              % (lab, Rm, 100 * frac[0], 100 * frac[1], 100 * frac[2], 100 * frac[3],
                 dm, dk))
    print("\n  A LARGE 'frac within 0.1' means the reward plateau is wide and an RL")
    print("  agent cannot distinguish good actions from the optimum.")

    # ---------------------------------------------------------------- alpha sweep
    print("\n" + "=" * 104)
    print("OPTIMAL ACTION vs alpha (through nu = alpha dt/dx^2), lambda = 1")
    print("=" * 104)
    print("%8s %8s | " % ("alpha", "nu") + " | ".join("%16s" % l for l in labels[:4]))
    print("%8s %8s | " % ("", "") + " | ".join("%16s" % "(m*, k*, s*)" for _ in labels[:4]))
    alpha_rows = []
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0]:
        nu_a = alpha * DT / DX ** 2
        cells = []
        for th in thetas[:4]:
            ms, ks, E, Cst = landscape(th, nu_a, 0.0)
            R = -np.log10(np.maximum(E, 1e-300)) - np.log10(np.maximum(Cst, 1e-300))
            R[~np.isfinite(E)] = -np.inf
            i, j = np.unravel_index(np.argmax(R), R.shape)
            s = ms[i] / np.sqrt(2 * ks[j] * nu_a)
            cells.append("(%2d,%4d,%4.1f)" % (ms[i], ks[j], s))
            alpha_rows.append(dict(alpha=alpha, nu=nu_a, theta=float(th),
                                   m=int(ms[i]), k=int(ks[j]), s=float(s)))
        print("%8.2f %8.2f | " % (alpha, nu_a) + " | ".join("%16s" % c for c in cells))
    out["alpha"] = alpha_rows

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, th, lab in zip(axes, [np.pi * DX, 0.5, 1.5],
                           ["very smooth ($\\theta=\\pi\\Delta x$)",
                            "rough ($\\theta=0.5$)", "very rough ($\\theta=1.5$)"]):
        ms, ks, E, Cst = landscape(th, nu, 0.0)
        R = -np.log10(np.maximum(E, 1e-300)) - np.log10(np.maximum(Cst, 1e-300))
        R[~np.isfinite(E)] = np.nan
        im = ax.pcolormesh(ks, ms, R, shading="nearest", cmap="viridis")
        ax.set_xscale("log"); ax.set_xlabel("$k$ (base steps leapt)")
        ax.set_ylabel("$m$ (half-width)")
        kk = np.array(ks, float)
        ax.plot(kk, np.sqrt(2 * kk * nu), "w--", lw=1.6, label=r"positivity frontier $m=\sigma$")
        ax.plot(kk, 3 * np.sqrt(2 * kk * nu), "w:", lw=1.4, label=r"$m=3\sigma$")
        i, j = np.unravel_index(np.nanargmax(R), R.shape)
        ax.plot(ks[j], ms[i], "r*", ms=15, label="optimum")
        ax.set_ylim(1, 30); ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(im, ax=ax, label="reward $R_{\\lambda=1}$")
    fig.suptitle("Reward landscape over actions $(m,k)$; white dashed = positivity frontier",
                 y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(OUT, "fig5_decision.png"), dpi=150)
    # ------------------------------------------------- boundary-optimum diagnostic
    print("\n" + "=" * 104)
    print("IS THE OPTIMUM INTERIOR?  (fraction of states whose optimal m sits ON the cap)")
    print("=" * 104)
    oncap = 0; tot = 0
    for lam in [0.5, 1.0, 2.0]:
        for th in thetas:
            ms, ks, E, Cst = landscape(th, nu, co)
            R = -np.log10(np.maximum(E, 1e-300)) - lam * np.log10(np.maximum(Cst, 1e-300))
            R[~np.isfinite(E)] = -np.inf
            i, j = np.unravel_index(np.argmax(R), R.shape)
            tot += 1; oncap += int(ms[i] >= MMAX - 1)
    print("  %d / %d optimal actions have m* >= m_cap-1  (cap = %d)" % (oncap, tot, MMAX))
    print("  -> the (m,k) objective has NO interior optimum in m: reward increases")
    print("     monotonically with width until a PHYSICAL cap (wall distance, or the")
    print("     simulation horizon k*dt <= T_end) stops it.  The only genuinely")
    print("     interior quantity is the safety factor s = m/sigma, and even that is")
    print("     set by the tolerance:  s* ~ sqrt(2 ln(1/eps)).")

    print("\nwrote figures/fig5_decision.png")
    json.dump(out, open(os.path.join(HERE, "task5_results.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
