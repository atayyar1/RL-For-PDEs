"""TASK 9b -- remove the ORDER confound from the variable-coefficient test.

The wide positive stencil uses P=6 local rows, so it is 6th-order in space,
while FTCS / RKC1 use a 3-point (2nd-order) Laplacian.  A fair competitor is a
HIGH-ORDER spatial discretisation driven by RKC1 -- same order, same footprint
budget, but signed weights.  If the wide positive scheme still wins, the win is
about the scheme; if it does not, the win was just spatial order.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import task9_varcoef as T9

A0, AMP = T9.A0, T9.AMP
alpha_f = T9.alpha_f
T = 0.02
u0f = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))

print("Building spectral reference ...")
xr, ur = T9.reference(u0f, T)
_uh = np.fft.fft(ur) / len(ur)
_kk = np.fft.fftfreq(len(ur), d=1.0 / len(ur))
ref = lambda xq: np.real(np.exp(2j * np.pi * np.outer(np.atleast_1d(xq), _kk)) @ _uh)

# 4th- and 6th-order central coefficients
C2 = {1: np.array([-0.5, 0, 0.5]), 2: np.array([1.0, -2.0, 1.0])}
C4 = {1: np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
      2: np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])}
C6 = {1: np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
      2: np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])}
ORD = {2: C2, 4: C4, 6: C6}


def make_L(N, order):
    x = np.arange(N) / N; dx = 1.0 / N
    al = alpha_f(x)
    dal = A0 * AMP * 2 * np.pi * np.cos(2 * np.pi * x)
    c1, c2 = ORD[order][1], ORD[order][2]
    r = len(c1) // 2
    offs = np.arange(-r, r + 1)

    def L(v):
        ux = sum(c * np.roll(v, -o) for c, o in zip(c1, offs)) / dx
        uxx = sum(c * np.roll(v, -o) for c, o in zip(c2, offs)) / dx ** 2
        return al * uxx + dal * ux
    lam = al.max() * (2 * np.pi * N / 2) ** 2 * 1.05   # conservative spectral radius
    flops_per_pt = 2 * (2 * r + 1) * 2 + 3
    return L, lam, flops_per_pt, x


def rkc1(N, order, s, nb):
    L, lam, fpp, x = make_L(N, order)
    dt = T / nb
    if dt > 2 * s ** 2 / lam:
        return None, np.inf
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
        if not np.all(np.isfinite(u)):
            return None, np.inf
    return (u, x), nb * s * N * (fpp + 4)


print("\n" + "=" * 100)
print("HIGH-ORDER-IN-SPACE RKC1 vs THE WIDE POSITIVE STENCIL (variable coefficients)")
print("=" * 100)
pts = {2: [], 4: [], 6: []}
for order in [2, 4, 6]:
    for N in [32, 48, 64, 96, 128, 192]:
        for s in [2, 4, 8, 16, 32, 64]:
            for nb in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                if s * nb > 2048:            # cap total stage count: cost is hopeless beyond
                    continue
                out, fl = rkc1(N, order, s, nb)
                if out is None:
                    continue
                u, x = out
                e = float(np.max(np.abs(u - ref(x))))
                if np.isfinite(e) and e < 1e3:
                    pts[order].append((e, fl, N, s, nb))

# wide positive, same sweep as task9
wide = []
for N in [48, 64, 96, 128]:
    x = np.arange(N) / N
    for nb in [4, 8, 16, 32, 64]:
        for P in [2, 4, 6]:
            for mm in [3, 5, 8, 12, 18]:
                xx, uw, iw = T9.wide_var(N, T, u0f, nb, mm, P, 6)
                if uw is None or not np.all(np.isfinite(uw)):
                    continue
                e = float(np.max(np.abs(uw - ref(x))))
                if e < 1e3:
                    wide.append((e, iw["flops"], N, mm, P))

ch = lambda L_, t: min([p[1] for p in L_ if p[0] <= t], default=None)
print("\n%-10s | %12s %12s %12s | %14s | %9s %9s"
      % ("tolerance", "RKC1 (2nd)", "RKC1 (4th)", "RKC1 (6th)", "wide positive",
         "vs 4th", "vs 6th"))
for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    a2, a4, a6 = ch(pts[2], tol), ch(pts[4], tol), ch(pts[6], tol)
    c = ch(wide, tol)
    f = lambda z: ("%12.3e" % z) if z else "%12s" % "--"
    r = lambda p, q: ("%8.1fx" % (p / q)) if (p and q) else "%9s" % "--"
    print("%-10.0e | %s %s %s | %14s | %s %s"
          % (tol, f(a2), f(a4), f(a6), ("%.3e" % c) if c else "--", r(a4, c), r(a6, c)))

print("\nBest config reaching 1e-6:")
for lab, L_ in [("RKC1 4th", pts[4]), ("RKC1 6th", pts[6]), ("wide positive", wide)]:
    ok = [p for p in L_ if p[0] <= 1e-6]
    if ok:
        b = min(ok, key=lambda p: p[1])
        print("  %-16s err=%.3e flops=%.3e  params=%s" % (lab, b[0], b[1], b[2:]))
    else:
        print("  %-16s did not reach 1e-6 in the sweep" % lab)


# ------------------------------------------------------------------ RK4 competitor
def rk4(N, order, nb):
    """4th order in TIME, high order in space, CFL-limited (no super-stepping).
    This removes RKC1's first-order-in-time handicap; if the wide positive stencil
    still wins, the advantage really is the large stable step."""
    L, lam, fpp, x = make_L(N, order)
    dt = T / nb
    if dt * lam > 2.78:                       # RK4 real-axis stability limit
        return None, np.inf
    u = u0f(x)
    for _ in range(nb):
        k1 = L(u); k2 = L(u + dt / 2 * k1); k3 = L(u + dt / 2 * k2); k4 = L(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(u)):
            return None, np.inf
    return (u, x), nb * 4 * N * (fpp + 4)


print("\n" + "=" * 100)
print("RK4 (4th order in TIME) + high-order space -- removes RKC1's temporal handicap")
print("=" * 100)
r4 = {4: [], 6: []}
for order in [4, 6]:
    for N in [32, 48, 64, 96, 128, 192]:
        for nb in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            out, fl = rk4(N, order, nb)
            if out is None:
                continue
            u, x = out
            e = float(np.max(np.abs(u - ref(x))))
            if np.isfinite(e) and e < 1e3:
                r4[order].append((e, fl, N, nb))

print("%-10s | %13s %13s | %14s | %9s %9s"
      % ("tolerance", "RK4+FD4", "RK4+FD6", "wide positive", "vs FD4", "vs FD6"))
for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    a4, a6, c = ch(r4[4], tol), ch(r4[6], tol), ch(wide, tol)
    f = lambda z: ("%13.3e" % z) if z else "%13s" % "--"
    r = lambda p, q: ("%8.1fx" % (p / q)) if (p and q) else "%9s" % "--"
    print("%-10.0e | %s %s | %14s | %s %s"
          % (tol, f(a4), f(a6), ("%.3e" % c) if c else "--", r(a4, c), r(a6, c)))
print("\nBest RK4+FD6 reaching 1e-6:", end=" ")
ok = [p for p in r4[6] if p[0] <= 1e-6]
print(min(ok, key=lambda p: p[1]) if ok else "did not reach it in the sweep")
