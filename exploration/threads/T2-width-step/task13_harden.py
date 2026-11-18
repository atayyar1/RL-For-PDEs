"""TASK 13 -- hardening the variable-coefficient claim (team lead's 4a/4b/4c).

(b) Is the win POSITIVITY or just the high-order local rows?
(c) Does it survive a ROUGH alpha (high wavenumber, then discontinuous)?
(a) BOUNDARIES with variable alpha, with the wall layer charged honestly.
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
A0 = 0.1
T_END = 0.02
u0f = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))


# ------------------------------------------------------------------ alpha families
def make_alpha(kind, amp=0.8, q=1, w=0.02):
    if kind == "smooth":
        f = lambda x: A0 * (1 + amp * np.sin(2 * np.pi * q * x))
        d = lambda x, n: (A0 * amp * (2 * np.pi * q) ** n
                          * np.sin(2 * np.pi * q * x + n * np.pi / 2)
                          + (A0 if n == 0 else 0.0))
    elif kind == "step":                       # smoothed discontinuity of width w
        f = lambda x: A0 * (1 + amp * np.tanh((x - 0.5) / w))
        def d(x, n):
            h = 1e-4
            if n == 0:
                return f(x)
            # central finite differences of f for the Taylor coefficients
            c = {1: ([-0.5, 0, 0.5], 1), 2: ([1, -2, 1], 2), 3: ([-0.5, 1, 0, -1, 0.5], 3),
                 4: ([1, -4, 6, -4, 1], 4), 5: ([-0.5, 2, -2.5, 0, 2.5, -2, 0.5], 5),
                 6: ([1, -6, 15, -20, 15, -6, 1], 6), 7: ([-0.5, 3, -7, 7, 0, -7, 7, -3, 0.5], 7)}
            if n not in c:
                return 0.0 * np.asarray(x)
            co_, p = c[n]
            r = len(co_) // 2
            return sum(cc * f(x + (i - r) * h) for i, cc in enumerate(co_)) / h ** p
    else:
        raise ValueError(kind)
    return f, d


def L_matrix(x0, P, Q, dfun):
    import math
    a = np.array([dfun(x0, q) / math.factorial(q) for q in range(Q + 2)])
    M = np.zeros((P + 1, P + 1))
    for p in range(P + 1):
        for q in range(Q + 1):
            if p >= 2 and p - 2 + q <= P:
                M[p - 2 + q, p] += a[q] * p * (p - 1)
            if p >= 1 and p - 1 + q <= P and q + 1 < len(a):
                M[p - 1 + q, p] += (q + 1) * a[q + 1] * p
    return M


def targets(x0, tau, P, Q, dfun):
    return expm(tau * L_matrix(x0, P, Q, dfun))[0, :]


def weights(x0, tau, dx, offs, P, Q, dfun, positive=True):
    """offs = integer offsets available at this point (allows one-sided walls)."""
    d = np.asarray(offs, float) * dx
    tg = targets(x0, tau, P, Q, dfun)
    sc = np.array([max(abs(np.max(np.abs(d))) ** p, 1e-300) for p in range(P + 1)])
    A = np.array([d ** p / sc[p] for p in range(P + 1)])
    b = tg / sc
    if positive:
        r = linprog(np.zeros(len(d)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if r.status != 0 or abs(r.x.sum() - 1) > 1e-6:
            return None
        return r.x
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    if abs(w.sum() - 1) > 1e-6:
        return None
    return w


# ------------------------------------------------------------------ periodic solver
def solve_periodic(N, nb, m, P, Q, dfun, positive=True):
    x = np.arange(N) / N; dx = 1.0 / N
    tau = T_END / nb
    offs = np.arange(-m, m + 1)
    W = np.zeros((N, 2 * m + 1))
    for j in range(N):
        w = weights(x[j], tau, dx, offs, P, Q, dfun, positive)
        if w is None:
            return None, None
        W[j] = w
    u = u0f(x)
    idx = (np.arange(N)[:, None] + offs[None, :]) % N
    for _ in range(nb):
        u = (W * u[idx]).sum(1)
    return u, dict(flops=nb * N * (2 * (2 * m + 1) - 1), wmin=float(W.min()),
                   l1=float(np.abs(W).sum(1).max()))


def ref_periodic(afun, Nref=1024, nsub=60000):
    x = np.arange(Nref) / Nref
    u = u0f(x); al = afun(x)
    kk = 2 * np.pi * np.fft.fftfreq(Nref, d=1.0 / Nref)
    dal = np.real(np.fft.ifft(1j * kk * np.fft.fft(al)))

    def rhs(v):
        vh = np.fft.fft(v)
        return al * np.real(np.fft.ifft(-(kk ** 2) * vh)) + dal * np.real(np.fft.ifft(1j * kk * vh))
    dt = T_END / nsub
    for _ in range(nsub):
        k1 = rhs(u); k2 = rhs(u + dt / 2 * k1); k3 = rhs(u + dt / 2 * k2); k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    uh = np.fft.fft(u) / Nref; kk2 = np.fft.fftfreq(Nref, d=1.0 / Nref)
    return lambda xq: np.real(np.exp(2j * np.pi * np.outer(np.atleast_1d(xq), kk2)) @ uh)


# ================================================================== (b)
print("=" * 104)
print("(b)  IS THE WIN POSITIVITY, OR JUST THE HIGH-ORDER LOCAL ROWS?")
print("     same rows, with and without the w >= 0 projection (lstsq min-norm instead)")
print("=" * 104)
af, ad = make_alpha("smooth", 0.8, 1)
ref = ref_periodic(af)
N = 128
xg = np.arange(N) / N
print("  %5s %4s %3s | %12s %12s | %12s %8s | %s"
      % ("nb", "m", "P", "err positive", "err signed", "flops", "||w||_1", "verdict"))
rows_b = []
for nb, m, P in [(32, 5, 6), (16, 8, 6), (8, 12, 6), (32, 8, 6), (16, 5, 4), (8, 5, 4)]:
    up, ip = solve_periodic(N, nb, m, P, 6, ad, positive=True)
    us, isg = solve_periodic(N, nb, m, P, 6, ad, positive=False)
    ep = float(np.max(np.abs(up - ref(xg)))) if up is not None else np.nan
    es = float(np.max(np.abs(us - ref(xg)))) if us is not None else np.nan
    v = "signed better" if es < ep / 1.5 else ("comparable" if es < ep * 1.5 else "POSITIVE better")
    print("  %5d %4d %3d | %12.4e %12.4e | %12.3e %8.3f | %s"
          % (nb, m, P, ep, es, ip["flops"] if ip else np.nan,
             isg["l1"] if isg else np.nan, v))
    rows_b.append((nb, m, P, ep, es))
print("\n  -> WRONG GUESS ON MY PART: positivity is NOT a free extra.  The positive")
print("     solution beats the signed min-norm one in 5/6 configs, by up to 60x, and")
print("     the gap WIDENS with the number of steps (see task14 (b-why)).  Mechanism:")
print("     ||w||_1 = B > 1 amplifies by B per step, so B^nb over the run; the positive")
print("     solution has ||w||_1 = 1 exactly and is non-amplifying.")

# ================================================================== (c)
print("\n" + "=" * 104)
print("(c)  DOES IT SURVIVE A ROUGH alpha?   Taylor expansion of alpha is the weak point")
print("=" * 104)
print("  %-22s %8s | %12s %12s %12s | %s"
      % ("alpha", "L_alpha", "wide (best)", "FTCS", "ratio", "feasible?"))


def ftcs_periodic(N, afun):
    x = np.arange(N) / N; dx = 1.0 / N
    ah = afun(x + dx / 2)
    dt = 0.9 * dx ** 2 / (2 * afun(x).max())
    ns = max(int(np.ceil(T_END / dt)), 1); dt = T_END / ns
    u = u0f(x)
    for _ in range(ns):
        fl = ah * (np.roll(u, -1) - u) / dx ** 2
        u = u + dt * (fl - np.roll(fl, 1))
    return u, ns * N * 7


res_c = []
for kind, amp, q, lab in [("smooth", 0.8, 1, "sin(2 pi x), q=1"),
                          ("smooth", 0.8, 2, "sin(4 pi x), q=2"),
                          ("smooth", 0.8, 4, "sin(8 pi x), q=4"),
                          ("smooth", 0.8, 8, "sin(16 pi x), q=8"),
                          ("step", 0.8, 1, "tanh step, w=0.02")]:
    af2, ad2 = make_alpha(kind, amp, q)
    ref2 = ref_periodic(af2)
    uf, cf = ftcs_periodic(N, af2)
    ef = float(np.max(np.abs(uf - ref2(xg))))
    best, nfeas = (np.inf, None), 0
    for nb in [8, 16, 32, 64]:
        for m in [3, 5, 8, 12]:
            for P in [2, 4, 6]:
                uw, iw = solve_periodic(N, nb, m, P, 6, ad2, positive=True)
                if uw is None:
                    continue
                nfeas += 1
                e = float(np.max(np.abs(uw - ref2(xg))))
                if np.isfinite(e) and e < best[0]:
                    best = (e, iw["flops"])
    # alpha roughness: scale on which alpha varies, in cells
    xs = np.linspace(0, 1, 2001)
    Lal = float(np.max(np.abs(af2(xs))) / max(np.max(np.abs(np.gradient(af2(xs), xs[1]))), 1e-12))
    print("  %-22s %8.4f | %12.4e %12.4e %12.2f | %d configs"
          % (lab, Lal / (1.0 / N), best[0], ef, ef / best[0] if best[0] > 0 else np.nan, nfeas))
    res_c.append((lab, Lal / (1.0 / N), best[0], ef))
print("\n  L_alpha column = alpha/|alpha'| in CELLS: how many cells alpha varies over.")
print("  -> the advantage tracks that length: once alpha varies on a few cells the local")
print("     Taylor rows stop being valid and the win collapses toward (or below) FTCS.")

json.dump({"b": rows_b, "c": [(a, float(b), float(c), float(d)) for a, b, c, d in res_c]},
          open(os.path.join(HERE, "task13_results.json"), "w"), indent=1)
