"""TASK 9 -- THE DECISIVE NOVELTY TEST: variable-coefficient diffusion.

u_t = d_x( alpha(x) d_x u ) = alpha u_xx + alpha' u_x   on periodic [0,1].

Here NO closed-form propagator exists, so "sample the heat kernel" is unavailable.
The construction under test uses ONLY local data: the Taylor coefficients of
alpha at x_j.  If it survives here it is genuinely more than an exponential
integrator; if it does not, the idea is confined to constant coefficients.

Local rows at x_j.  With s = x - x_j and alpha(x_j+s) = sum_q a_q s^q,
    L(s^p) = sum_q a_q p(p-1) s^(p-2+q) + sum_q (q+1) a_(q+1) p s^(p-1+q)
Build L as a matrix on {s^0..s^P}, exponentiate, and read off the targets
    M_p = [exp(tau L)]_(0,p)   =  value at x_j of the exact evolution of s^p.
Then solve for w >= 0 on {-m..m} with sum_i w_i (i dx)^p = M_p, p = 0..P.
This never uses a fundamental solution.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

A0, AMP = 0.1, 0.8                      # alpha(x) = A0 (1 + AMP sin 2 pi x)
alpha_f = lambda x: A0 * (1 + AMP * np.sin(2 * np.pi * x))


def alpha_taylor(x0, Q):
    """a_q = alpha^(q)(x0)/q!  for the analytic alpha above."""
    k = 2 * np.pi
    a = []
    for q in range(Q + 1):
        d = A0 * AMP * k ** q * np.sin(k * x0 + q * np.pi / 2)     # q-th derivative
        if q == 0:
            d += A0
        a.append(d / __import__('math').factorial(q))
    return np.array(a)


def L_matrix(x0, P, Q):
    """Matrix of L = alpha d_xx + alpha' d_x on {s^0..s^P}, truncated to degree P."""
    a = alpha_taylor(x0, Q + 1)
    M = np.zeros((P + 1, P + 1))
    for p in range(P + 1):
        for q in range(Q + 1):
            if p >= 2 and p - 2 + q <= P:
                M[p - 2 + q, p] += a[q] * p * (p - 1)
            if p >= 1 and p - 1 + q <= P and q + 1 < len(a):
                M[p - 1 + q, p] += (q + 1) * a[q + 1] * p
    return M


def targets(x0, tau, P, Q):
    return expm(tau * L_matrix(x0, P, Q))[0, :]


def wide_weights_var(x0, tau, dx, m, P, Q):
    """Non-negative weights on {-m..m} matching the local propagator moments."""
    d = np.arange(-m, m + 1) * dx
    tg = targets(x0, tau, P, Q)
    sc = np.array([max(abs(m * dx) ** p, 1e-300) for p in range(P + 1)])
    A = np.array([d ** p / sc[p] for p in range(P + 1)])
    b = tg / sc
    # least-infeasibility LP: minimise slack, then report
    r = linprog(np.zeros(len(d)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if r.status != 0:
        return None
    w = r.x
    if abs(w.sum() - 1) > 1e-6:
        return None
    return w


# ---------------------------------------------------------------- reference
def reference(u0f, T, Nref=1024, nsub=60000):
    x = np.arange(Nref) / Nref
    u = u0f(x)
    al = alpha_f(x)
    kk = 2 * np.pi * np.fft.fftfreq(Nref, d=1.0 / Nref)

    def rhs(v):
        vh = np.fft.fft(v)
        vx = np.real(np.fft.ifft(1j * kk * vh))
        vxx = np.real(np.fft.ifft(-(kk ** 2) * vh))
        dal = np.real(np.fft.ifft(1j * kk * np.fft.fft(al)))
        return al * vxx + dal * vx

    dt = T / nsub
    for _ in range(nsub):
        k1 = rhs(u); k2 = rhs(u + dt / 2 * k1); k3 = rhs(u + dt / 2 * k2); k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x, u


# ---------------------------------------------------------------- solvers
def ftcs_var(N, T, u0f):
    x = np.arange(N) / N; dx = 1.0 / N
    ah = alpha_f(x + dx / 2)                       # alpha at j+1/2
    dt = 0.9 * dx ** 2 / (2 * alpha_f(x).max())
    ns = max(int(np.ceil(T / dt)), 1); dt = T / ns
    u = u0f(x)
    for _ in range(ns):
        fl = ah * (np.roll(u, -1) - u) / dx ** 2
        u = u + dt * (fl - np.roll(fl, 1))
    return x, u, dict(flops=ns * N * 7, steps=ns)


def rkc1_var(N, T, u0f, s, nb):
    x = np.arange(N) / N; dx = 1.0 / N
    ah = alpha_f(x + dx / 2)
    dt = T / nb

    def L(v):
        fl = ah * (np.roll(v, -1) - v) / dx ** 2
        return fl - np.roll(fl, 1)

    lam = 4 * alpha_f(x).max() / dx ** 2
    if dt > 2 * s ** 2 / lam:
        return x, None, dict(flops=np.inf, unstable=True)
    u = u0f(x)
    for _ in range(nb):
        Y0 = u.copy(); Y1 = u + (dt / s ** 2) * L(u)
        Tm2, Tm1 = 1.0, 1.0
        for j in range(2, s + 1):
            Tj = 2 * Tm1 - Tm2
            mu, nuj = 2 * Tm1 / Tj, -Tm2 / Tj
            Y2 = mu * Y1 + nuj * Y0 + (mu / s ** 2) * dt * L(Y1)
            Y0, Y1 = Y1, Y2
            Tm2, Tm1 = Tm1, Tj
        u = Y1
    return x, u, dict(flops=nb * s * N * 11, steps=nb, unstable=False)


def wide_var(N, T, u0f, nb, m, P, Q):
    x = np.arange(N) / N; dx = 1.0 / N
    tau = T / nb
    W = np.zeros((N, 2 * m + 1))
    for j in range(N):
        w = wide_weights_var(x[j], tau, dx, m, P, Q)
        if w is None:
            return x, None, dict(flops=np.inf, infeasible=True, j=j)
        W[j] = w
    u = u0f(x)
    idx = (np.arange(N)[:, None] + np.arange(-m, m + 1)[None, :]) % N
    for _ in range(nb):
        u = (W * u[idx]).sum(1)
    P_nz = int((W > 1e-16).sum(1).mean())
    return x, u, dict(flops=nb * N * (2 * (2 * m + 1) - 1), steps=nb, m=m,
                      P_nz=P_nz, wmin=float(W.min()), infeasible=False)


def main():
    T = 0.02
    u0f = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))
    print("Building spectral reference ...")
    xr, ur = reference(u0f, T)
    # EXACT (spectral) evaluation of the reference at arbitrary x: linear interp
    # would floor every error near 2e-5 and silently corrupt the comparison.
    _uh = np.fft.fft(ur) / len(ur)
    _kk = np.fft.fftfreq(len(ur), d=1.0 / len(ur))

    def ref(xq):
        xq = np.atleast_1d(np.asarray(xq, float))
        return np.real(np.exp(2j * np.pi * np.outer(xq, _kk)) @ _uh)

    # convergence check of the reference itself
    _, ur2 = reference(u0f, T, Nref=1536, nsub=90000)
    _uh2 = np.fft.fft(ur2) / len(ur2); _kk2 = np.fft.fftfreq(len(ur2), d=1.0 / len(ur2))
    r2 = np.real(np.exp(2j * np.pi * np.outer(xr, _kk2)) @ _uh2)
    print("  reference self-consistency (Nref 1024 vs 1536): %.3e"
          % float(np.max(np.abs(ur - r2))))

    N = 128
    x = np.arange(N) / N
    err = lambda u: float(np.max(np.abs(u - ref(x))))

    print("\n" + "=" * 104)
    print("VARIABLE-COEFFICIENT DIFFUSION  alpha(x) = %.2f (1 + %.1f sin 2 pi x),"
          " ratio alpha_max/alpha_min = %.1f" % (A0, AMP, (1 + AMP) / (1 - AMP)))
    print("N = %d, T = %.3f.  Reference: Fourier pseudospectral + RK4 (Nref=1024)." % (N, T))
    print("=" * 104)

    xf, uf, inf_ = ftcs_var(N, T, u0f)
    print("\n  FTCS (conservative, at CFL): err = %.4e, flops = %.3e, steps = %d"
          % (err(uf), inf_["flops"], inf_["steps"]))

    print("\n  A. DOES A NON-NEGATIVE LOCAL STENCIL EXIST AT ALL?  (feasibility vs P, m)")
    print("     tau = T/nb; sigma_cells = sqrt(2 alpha_max tau)/dx")
    dx = 1.0 / N
    print("  %5s %8s %6s | %s" % ("nb", "sig_cells", "P", "smallest feasible m (LP over all x)"))
    for nb in [64, 16, 4]:
        tau = T / nb
        sig = np.sqrt(2 * alpha_f(x).max() * tau) / dx
        for P in [2, 4, 6]:
            got = None
            for m in range(1, 41):
                if all(wide_weights_var(xx, tau, dx, m, P, 6) is not None
                       for xx in x[::8]):
                    got = m; break
            print("  %5d %8.2f %6d | %s" % (nb, sig, P, got if got else "none up to 40"))

    print("\n  B. WORK-PRECISION ON THE VARIABLE-COEFFICIENT PROBLEM")
    print("  %-34s %12s %12s %10s" % ("method", "err", "flops", "min w"))
    rows = []
    print("  %-34s %12.4e %12.3e %10s" % ("FTCS @ CFL", err(uf), inf_["flops"], "-"))
    rows.append(("FTCS", err(uf), inf_["flops"]))
    for s_, nb in [(4, 16), (8, 8), (16, 4)]:
        xx, ur2, i2 = rkc1_var(N, T, u0f, s_, nb)
        if ur2 is not None:
            print("  %-34s %12.4e %12.3e %10s"
                  % ("RKC1 s=%d, nb=%d" % (s_, nb), err(ur2), i2["flops"], "-"))
            rows.append(("RKC1 s=%d" % s_, err(ur2), i2["flops"]))
    best = {}
    for nb in [64, 32, 16, 8, 4, 2]:
        for P in [2, 4, 6]:
            for mm in [3, 5, 8, 12, 18, 25]:
                xx, uw, iw = wide_var(N, T, u0f, nb, mm, P, 6)
                if uw is None or not np.all(np.isfinite(uw)):
                    continue
                e = err(uw)
                if e > 1e3:
                    continue
                best[(nb, P, mm)] = (e, iw["flops"], iw["wmin"])
    top = sorted(best.items(), key=lambda kv: kv[1][0])[:6]
    for (nb, P, mm), (e, fl, wmn) in top:
        print("  %-34s %12.4e %12.3e %10.2e"
              % ("wide nb=%d, m=%d, P=%d rows" % (nb, mm, P), e, fl, wmn))
        rows.append(("wide nb=%d m=%d P=%d" % (nb, mm, P), e, fl))

    print("\n  C. HOW FAR CAN THE STEP BE PUSHED? (cheapest config reaching each tolerance)")
    print("  %-10s | %14s %14s %14s" % ("tolerance", "FTCS", "RKC1", "wide positive"))
    rkc_all = []
    for s_ in [2, 4, 8, 16, 32]:
        for nb in [1, 2, 4, 8, 16, 32, 64, 128]:
            xx, u2, i2 = rkc1_var(N, T, u0f, s_, nb)
            if u2 is not None and np.all(np.isfinite(u2)):
                rkc_all.append((err(u2), i2["flops"]))
    for tol in [1e-3, 1e-4, 1e-5, 1e-6]:
        a = inf_["flops"] if err(uf) <= tol else None
        b = min([f for e, f in rkc_all if e <= tol], default=None)
        c = min([v[1] for v in best.values() if v[0] <= tol], default=None)
        f = lambda z: ("%14.3e" % z) if z else "%14s" % "--"
        print("  %-10.0e | %s %s %s" % (tol, f(a), f(b), f(c)))

    print("\n  D. FAIR COMPARISON: let FTCS and RKC1 REFINE THE GRID too.")
    print("     (their O(dx^2) spatial error, not the CFL, is what binds at N=128;")
    print("      the wide scheme uses P=6 local rows so it is higher order in space.)")
    print("  %6s | %12s %12s | %12s %12s" % ("N", "FTCS err", "FTCS flops",
                                             "RKC1 best err", "at flops"))
    fair = {"ftcs": [], "rkc": [], "wide": []}
    for Nn in [64, 128, 181, 256, 362, 512, 724]:
        xf2, uf2, if2 = ftcs_var(Nn, T, u0f)
        xg = np.arange(Nn) / Nn
        e = float(np.max(np.abs(uf2 - ref(xg))))
        fair["ftcs"].append((e, if2["flops"], Nn))
        br = (np.inf, np.inf)
        for s_ in [2, 4, 8, 16, 32, 64]:
            for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                xx, u2, i2 = rkc1_var(Nn, T, u0f, s_, nb)
                if u2 is None or not np.all(np.isfinite(u2)):
                    continue
                e2 = float(np.max(np.abs(u2 - ref(xg))))
                if e2 < br[0]:
                    br = (e2, i2["flops"])
                fair["rkc"].append((e2, i2["flops"], Nn))
        print("  %6d | %12.4e %12.3e | %12.4e %12.3e" % (Nn, e, if2["flops"], br[0], br[1]))

    for Nn in [48, 64, 96, 128]:
        xg = np.arange(Nn) / Nn
        for nb in [4, 8, 16, 32, 64]:
            for P in [2, 4, 6]:
                for mm in [3, 5, 8, 12, 18]:
                    xx, uw2, iw2 = wide_var(Nn, T, u0f, nb, mm, P, 6)
                    if uw2 is None or not np.all(np.isfinite(uw2)):
                        continue
                    e2 = float(np.max(np.abs(uw2 - ref(xg))))
                    if e2 < 1e3:
                        fair["wide"].append((e2, iw2["flops"], Nn))

    def cheapest(lst, tol):
        v = [f for e, f, _ in lst if e <= tol]
        return min(v) if v else None

    print("\n  FAIR work-precision (grid refined for every method):")
    print("  %-10s | %13s %13s %13s | %10s %10s"
          % ("tolerance", "FTCS", "RKC1", "wide positive", "vs FTCS", "vs RKC1"))
    for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        a, b, c = (cheapest(fair["ftcs"], tol), cheapest(fair["rkc"], tol),
                   cheapest(fair["wide"], tol))
        f = lambda z: ("%13.3e" % z) if z else "%13s" % "--"
        r = lambda p, q: ("%9.1fx" % (p / q)) if (p and q) else "%10s" % "--"
        print("  %-10.0e | %s %s %s | %s %s"
              % (tol, f(a), f(b), f(c), r(a, c), r(b, c)))
    json.dump({k: [list(map(float, t)) for t in v] for k, v in fair.items()},
              open(os.path.join(HERE, "task9_results.json"), "w"), indent=1)

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    ax[0].plot(x, u0f(x), color="#bbbbbb", lw=1.2, label="initial")
    ax[0].plot(xr, ur, "k--", lw=1.4, label="spectral reference at $T$")
    bk = top[0][0]
    xx, uw, _ = wide_var(N, T, u0f, bk[0], bk[2], bk[1], 6)
    ax[0].plot(x, uw, "o", color="#c4432b", ms=3,
               label="wide positive ($nb$=%d, $m$=%d, $P$=%d)" % (bk[0], bk[2], bk[1]))
    a2 = ax[0].twinx(); a2.plot(x, alpha_f(x), color="#1b6ca8", lw=1, alpha=.5)
    a2.set_ylabel(r"$\alpha(x)$", color="#1b6ca8", fontsize=9)
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u$")
    ax[0].set_title("(a) variable-coefficient solution"); ax[0].legend(fontsize=7.5)

    def pf(lst):
        pts = sorted([(f, e) for e, f, _ in lst]); out = []; bb = np.inf
        for f_, e_ in pts:
            if e_ < bb * (1 - 1e-12):
                out.append((f_, e_)); bb = e_
        return out
    for key, cl, mk, lab in [("ftcs", "#1b6ca8", "o", "FTCS @ CFL"),
                             ("rkc", "#8c564b", "v", "RKC1 ($s$ stages)"),
                             ("wide", "#c4432b", "s", "wide positive (local rows)")]:
        q = pf(fair[key])
        ax[1].loglog([p[0] for p in q], [p[1] for p in q], mk + "-", color=cl,
                     ms=5, lw=1.5, label=lab)
    ax[1].set_xlabel("flops"); ax[1].set_ylabel(r"$\|e\|_\infty$ at $T$")
    ax[1].set_title("(b) work-precision, variable coefficients")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
    fig.suptitle(r"Variable-coefficient test: $u_t=\partial_x(\alpha(x)\partial_x u)$,"
                 r" $\alpha_{max}/\alpha_{min}=9$", y=0.99, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig7_varcoef.png"), dpi=150)
    print("\nwrote figures/fig7_varcoef.png")


if __name__ == "__main__":
    main()
