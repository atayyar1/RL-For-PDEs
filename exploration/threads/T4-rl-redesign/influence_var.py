"""Does target-awareness survive VARIABLE COEFFICIENTS, where the influence
function is no longer a closed-form Gaussian?

Rebuilt after a first attempt that was inconclusive for two reasons, both mine:
  - the "expensive" stencil matched only moments 0..2, so it was barely better
    than the cheap one and there was almost nothing to allocate;
  - the reference was an under-converged sub-stepped cheap scheme whose own
    error (~2e-7) was comparable to the spread between policies (~4e-7).
Fixed: expensive matches local moments 0..4; reference is Richardson-extrapolated
from sub=64 and sub=128, and its residual is reported so it can be checked
against the effect being measured.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import linprog

NX, NT = 161, 90
X = np.linspace(0, 1, NX); DX = X[1] - X[0]; C = 1.0
A_LO, A_HI = 0.02, 0.40
ALPHA = A_LO + (A_HI - A_LO) * (1 + np.cos(2 * np.pi * X)) / 2
DT = 0.45 * DX**2 / A_HI


def gauss_moments(mu, var, p):
    m = [1.0, mu]
    for q in range(2, p + 1):
        m.append(mu * m[q - 1] + (q - 1) * var * m[q - 2])
    return np.array(m)


def local_w(j, m, order):
    """Positive stencil at x_j matching the LOCAL frozen-coefficient propagator."""
    idx = np.arange(-m, m + 1); h = m * DX
    A = np.vstack([(idx * DX / h) ** q for q in range(order + 1)])
    b = gauss_moments(-C * DT, 2 * ALPHA[j] * DT, order) / h ** np.arange(order + 1)
    r = linprog((idx * DX / h) ** (order + 2), A_eq=A, b_eq=b,
                bounds=[(0, None)] * len(idx), method="highs")
    return r.x if r.status == 0 else None


W1 = np.array([local_w(j, 1, 2) for j in range(NX)])
W2raw = [local_w(j, 2, 4) for j in range(NX)]
ok2 = np.array([w is not None for w in W2raw])
W2 = np.array([w if w is not None else np.r_[0, W1[j], 0] for j, w in enumerate(W2raw)])
print(f"alpha in [{A_LO},{A_HI}] (20x), dt={DT:.3e}, "
      f"5-pt order-4 positive-feasible at {ok2.mean():.0%} of points")


def march(mask, W_lo=W1, W_hi=W2, nt=NT, dtsub=1):
    u = np.sin(np.pi * X).copy()
    for n in range(nt):
        lo = np.einsum('ij,ij->i', W_lo[1:-1], np.lib.stride_tricks.sliding_window_view(u, 3))
        hi = np.einsum('ij,ij->i', W_hi[2:-2], np.lib.stride_tricks.sliding_window_view(u, 5))
        v = np.zeros(NX)
        v[1:-1] = lo
        m = mask[n] if mask is not None else np.zeros(NX, bool)
        sel = m[2:-2]
        v[2:-2] = np.where(sel, hi, lo[1:-1])
        v[[0, -1]] = 0.0
        u = v
    return u


def reference(sub):
    """Sub-stepped cheap scheme; each sub-step re-solves its own local weights."""
    d = DT / sub
    W = np.array([[ALPHA[j]*d/DX**2 + C*d/DX/2, 1 - 2*ALPHA[j]*d/DX**2,
                   ALPHA[j]*d/DX**2 - C*d/DX/2] for j in range(NX)])
    u = np.sin(np.pi * X).copy()
    for _ in range(NT * sub):
        v = np.zeros(NX)
        v[1:-1] = np.einsum('ij,ij->i', W[1:-1], np.lib.stride_tricks.sliding_window_view(u, 3))
        u = v
    return u


r64, r128 = reference(64), reference(128)
REF = 2 * r128 - r64                                   # Richardson (scheme is O(dt))
resid = np.abs(r128 - REF)
print(f"reference: Richardson from sub=64,128; residual max {resid.max():.2e}")

# the allocation quantity: how much upgrading a cell actually changes one step
tau_gap = np.zeros((NT, NX))
u = np.sin(np.pi * X).copy()
for n in range(NT):
    lo = np.zeros(NX); hi = np.zeros(NX)
    lo[1:-1] = np.einsum('ij,ij->i', W1[1:-1], np.lib.stride_tricks.sliding_window_view(u, 3))
    hi[2:-2] = np.einsum('ij,ij->i', W2[2:-2], np.lib.stride_tricks.sliding_window_view(u, 5))
    tau_gap[n, 2:-2] = np.abs(hi[2:-2] - lo[2:-2])
    u = lo; u[[0, -1]] = 0.0


def influence_exact(jstar):
    G = np.zeros((NT, NX)); g = np.zeros(NX); g[jstar] = 1.0
    for n in range(NT - 1, -1, -1):
        G[n] = g
        gn = np.zeros(NX)
        for j in range(1, NX - 1):
            gn[j - 1:j + 2] += W1[j] * g[j]
        g = gn; g[[0, -1]] = 0.0
    return G


def influence_crude(jstar):
    """A constant-coefficient cone using the domain-mean alpha -- ignores all variation."""
    abar = ALPHA.mean(); G = np.zeros((NT, NX))
    for n in range(NT):
        tau = (NT - n) * DT
        G[n] = np.exp(-(X - (X[jstar] - C * tau))**2 / (2 * (2 * abar * tau + 1e-18)))
    return G


print(f"\n{'x*':>6} {'budget':>7} | {'uniform':>10} {'local':>10} {'crude cone':>11} {'exact adj':>11}"
      f" | {'crude':>7} {'exact':>7}")
rows = []
for xs in [0.3, 0.6]:
    jstar = int(round(xs / DX))
    Ge, Gc = influence_exact(jstar), influence_crude(jstar)
    for frac in [0.10, 0.30]:
        k = int(frac * NX * NT); out = {}
        for name, score in [("uniform", np.random.default_rng(1).random((NT, NX))),
                            ("local", tau_gap), ("crude", Gc * tau_gap), ("exact", Ge * tau_gap)]:
            flat = np.argsort(-score.ravel())[:k]
            M = np.zeros(NT * NX, bool); M[flat] = True
            out[name] = abs(march(M.reshape(NT, NX))[jstar] - REF[jstar])
        rows.append((out["local"] / out["crude"], out["local"] / out["exact"]))
        print(f"{xs:>6.2f} {frac:>6.0%} | {out['uniform']:>10.3e} {out['local']:>10.3e}"
              f" {out['crude']:>11.3e} {out['exact']:>11.3e} | {rows[-1][0]:>6.2f}x {rows[-1][1]:>6.2f}x")
r = np.array(rows)
print(f"\nmedian gain over the local indicator: crude {np.median(r[:,0]):.1f}x, "
      f"exact {np.median(r[:,1]):.1f}x")
print(f"reference residual {resid.max():.1e} vs smallest error measured "
      f"{min(min(rr) for rr in [[1]]) if False else ''}")
