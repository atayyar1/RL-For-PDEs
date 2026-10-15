"""TASK 1 -- map the positivity frontier k_max(m) exactly, and derive its constant."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import core as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def frontier_lp(ms, nu, co, dx, dt, alpha, c, kcap=1200):
    """Exhaustive LP scan (NOT bisection: feasibility in k need not be monotone)."""
    out = {}
    for m in ms:
        ks, feas = [], []
        kdiff = int(np.floor(m ** 2 / (2 * nu))) + 4
        for k in range(1, min(kdiff, kcap) + 1):
            j = np.arange(-m, m + 1)
            ok, _ = C.w_positive(j * dx, -k * dt * np.ones(2 * m + 1), alpha, c)
            ks.append(k); feas.append(ok)
        feas = np.array(feas); ks = np.array(ks)
        kmax = ks[feas].max() if feas.any() else 0
        monotone = bool(np.all(feas[:feas.argmin()] if (~feas).any() else feas))
        # is the feasible set an initial run 1..kmax?
        contiguous = bool(np.all(feas[:kmax])) if kmax > 0 else True
        out[m] = dict(kmax=int(kmax), contiguous=contiguous,
                      nfeas=int(feas.sum()))
    return out


def main():
    t0 = time.time()
    res = {}

    # ---------------------------------------------------------------- A. canonical
    ms = list(range(1, 26))
    print("=== A. canonical setup: alpha=0.1, c=1.0, nu=%.6f, Co=%.6f, Pe_cell=%.4f"
          % (C.NU_REF, C.CO_REF, C.CVEL * C.DX / C.ALPHA))
    lp = frontier_lp(ms, C.NU_REF, C.CO_REF, C.DX, C.DT, C.ALPHA, C.CVEL)
    print(" m | k_max(LP) | analytic | floor(m^2/2nu) | 2nu/Co^2 | contiguous")
    rows = []
    for m in ms:
        ka = C.kmax_analytic(m, C.NU_REF, C.CO_REF)
        kd = int(np.floor(m ** 2 / (2 * C.NU_REF)))
        kadv = 2 * C.NU_REF / C.CO_REF ** 2
        print("%3d | %9d | %8d | %14d | %8.1f | %s"
              % (m, lp[m]["kmax"], ka, kd, kadv, lp[m]["contiguous"]))
        rows.append(dict(m=m, k_lp=lp[m]["kmax"], k_analytic=ka, k_diff=kd,
                         k_adv=kadv, contiguous=lp[m]["contiguous"]))
    res["canonical"] = rows
    agree = all(r["k_lp"] == r["k_analytic"] for r in rows)
    print("LP == analytic for all m:", agree)

    # pure diffusion
    print("\n=== B. pure diffusion c=0 (same nu)")
    lp0 = frontier_lp(ms, C.NU_REF, 0.0, C.DX, C.DT, C.ALPHA, 0.0)
    rows0 = []
    for m in ms:
        kd = int(np.floor(m ** 2 / (2 * C.NU_REF)))
        rows0.append(dict(m=m, k_lp=lp0[m]["kmax"], k_diff=kd))
    print(" m | k_max(LP) | floor(m^2/2nu)")
    for r in rows0:
        print("%3d | %9d | %14d" % (r["m"], r["k_lp"], r["k_diff"]))
    res["pure_diffusion"] = rows0
    agree0 = all(r["k_lp"] == r["k_diff"] for r in rows0)
    print("c=0: k_max == floor(m^2/(2 nu)) exactly for all m:", agree0)

    # ---------------------------------------------------------------- C. Peclet sweep
    print("\n=== C. cell-Peclet sweep (dx, dt frozen at canonical values)")
    pes = [0.0, 0.02, 0.05, 0.101, 0.2, 0.5, 1.0, 2.0, 5.0]
    pe_rows = []
    for pe in pes:
        # Pe = c dx/alpha ;  keep alpha, dx, dt -> c = Pe*alpha/dx
        cc = pe * C.ALPHA / C.DX
        co = cc * C.DT / C.DX
        lpp = frontier_lp(ms, C.NU_REF, co, C.DX, C.DT, C.ALPHA, cc, kcap=900)
        kadv = np.inf if co == 0 else 2 * C.NU_REF / co ** 2
        pe_rows.append(dict(pe=pe, c=cc, co=co, kadv=kadv,
                            kmax=[lpp[m]["kmax"] for m in ms]))
        print("Pe=%5.3f c=%7.3f Co=%8.5f  k_adv=%9.1f  k_max(m=1,5,12,25)=%s"
              % (pe, cc, co, kadv,
                 [lpp[m]["kmax"] for m in (1, 5, 12, 25)]))
    res["peclet"] = [{k: (v if not isinstance(v, float) or np.isfinite(v) else 1e99)
                      for k, v in r.items()} for r in pe_rows]

    # ---------------------------------------------------------------- D. nu sweep
    print("\n=== D. nu sweep at c=0 (does the constant 1/(2 nu) hold?)")
    nu_rows = []
    for nu in [0.05, 0.1, 0.2, 0.45, 0.8, 1.5]:
        dtl = nu * C.DX ** 2 / C.ALPHA
        lpn = frontier_lp(list(range(1, 13)), nu, 0.0, C.DX, dtl, C.ALPHA, 0.0)
        ok = all(lpn[m]["kmax"] == int(np.floor(m ** 2 / (2 * nu)))
                 for m in range(1, 13))
        print("nu=%5.2f  k_max(m=1..6)=%s   matches floor(m^2/2nu): %s"
              % (nu, [lpn[m]["kmax"] for m in range(1, 7)], ok))
        nu_rows.append(dict(nu=nu, kmax=[lpn[m]["kmax"] for m in range(1, 13)],
                            matches=bool(ok)))
    res["nu_sweep"] = nu_rows

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    mm = np.array(ms)
    kl = np.array([r["k_lp"] for r in rows])
    kl0 = np.array([r["k_lp"] for r in rows0])
    ax[0].plot(mm, kl0, "o-", color="#1b6ca8", label="LP, $c=0$")
    ax[0].plot(mm, kl, "s-", color="#c4432b", label="LP, $c=1$ (Pe=0.101)")
    ax[0].plot(mm, mm ** 2 / (2 * C.NU_REF), "k--", lw=1.2,
               label=r"$m^2/(2\nu)=(m\Delta x)^2/(2\alpha\Delta t)$")
    ax[0].axhline(2 * C.NU_REF / C.CO_REF ** 2, color="#c4432b", ls=":", lw=1.2,
                  label=r"$2\nu/\mathrm{Co}^2=2\alpha/(c^2\Delta t)$")
    ax[0].set_xlabel("half-width $m$"); ax[0].set_ylabel(r"$k_{\max}$")
    ax[0].set_yscale("log"); ax[0].set_title("(a) positivity frontier")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)

    ax[1].plot(mm, kl0 / (mm ** 2 / (2 * C.NU_REF)), "o-", color="#1b6ca8",
               label="$c=0$")
    ax[1].plot(mm, kl / (mm ** 2 / (2 * C.NU_REF)), "s-", color="#c4432b",
               label="$c=1$")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("half-width $m$")
    ax[1].set_ylabel(r"$k_{\max}\,/\,[m^2/(2\nu)]$")
    ax[1].set_title("(b) ratio to closed form"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3); ax[1].set_ylim(0, 1.15)

    for r in pe_rows:
        if r["pe"] == 0:
            lab = "Pe = 0"
        else:
            lab = "Pe = %.3g" % r["pe"]
        ax[2].plot(mm, r["kmax"], "-", lw=1.4, label=lab)
    ax[2].plot(mm, mm ** 2 / (2 * C.NU_REF), "k--", lw=1.2, label="$m^2/(2\\nu)$")
    ax[2].set_yscale("log"); ax[2].set_xlabel("half-width $m$")
    ax[2].set_ylabel(r"$k_{\max}$"); ax[2].set_title("(c) effect of cell Peclet")
    ax[2].legend(fontsize=7, ncol=2); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_frontier.png"), dpi=150)
    print("\nwrote figures/fig1_frontier.png   (%.1f s)" % (time.time() - t0))

    with open(os.path.join(os.path.dirname(OUT), "task1_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
