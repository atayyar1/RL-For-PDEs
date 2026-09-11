"""Variable coefficients with an EXACT reference, via a manufactured solution.

Two earlier attempts were floored by reference error. Fix: choose u(x,t), derive
the source S = u_t + c u_x - alpha(x) u_xx analytically, and march with S. The
reference is then exact to machine precision.

High spatial wavenumber so the SPATIAL stencil choice actually matters (both
options share the same dt, so shared temporal error would otherwise floor them).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.optimize import linprog

NX, NT, K = 161, 90, 6
X = np.linspace(0, 1, NX); DX = X[1] - X[0]; C = 1.0
A_LO, A_HI = 0.02, 0.40
ALPHA = A_LO + (A_HI - A_LO) * (1 + np.cos(2 * np.pi * X)) / 2
DT = 0.45 * DX**2 / A_HI
LAM = 0.5

u_ex  = lambda x, t: np.sin(K * np.pi * x) * np.exp(-LAM * t)
ut_ex = lambda x, t: -LAM * np.sin(K * np.pi * x) * np.exp(-LAM * t)
ux_ex = lambda x, t: K * np.pi * np.cos(K * np.pi * x) * np.exp(-LAM * t)
uxx_ex= lambda x, t: -(K * np.pi)**2 * np.sin(K * np.pi * x) * np.exp(-LAM * t)
S     = lambda x, t: ut_ex(x, t) + C * ux_ex(x, t) - ALPHA * uxx_ex(x, t)

def gm(mu, var, p):
    m = [1.0, mu]
    for q in range(2, p + 1):
        m.append(mu * m[q-1] + (q-1) * var * m[q-2])
    return np.array(m)

def local_w(j, m, order):
    idx = np.arange(-m, m+1); h = m * DX
    A = np.vstack([(idx * DX / h) ** q for q in range(order+1)])
    b = gm(-C*DT, 2*ALPHA[j]*DT, order) / h ** np.arange(order+1)
    r = linprog((idx*DX/h)**(order+2), A_eq=A, b_eq=b,
                bounds=[(0,None)]*len(idx), method="highs")
    return r.x if r.status == 0 else None

W1 = np.array([local_w(j,1,2) for j in range(NX)])
W2raw = [local_w(j,2,4) for j in range(NX)]
ok2 = np.array([w is not None for w in W2raw])
W2 = np.array([w if w is not None else np.r_[0, W1[j], 0] for j,w in enumerate(W2raw)])
print(f"alpha 20x variation, wavenumber k={K}, dt={DT:.3e}; "
      f"order-4 positive-feasible at {ok2.mean():.0%} of points")

sw = np.lib.stride_tricks.sliding_window_view
def march(M):
    u = u_ex(X, 0.0).copy()
    for n in range(NT):
        lo = np.einsum('ij,ij->i', W1[1:-1], sw(u, 3))
        hi = np.einsum('ij,ij->i', W2[2:-2], sw(u, 5))
        v = np.zeros(NX); v[1:-1] = lo
        if M is not None:
            v[2:-2] = np.where(M[n][2:-2], hi, lo[1:-1])
        v += DT * S(X, n * DT)
        v[0], v[-1] = u_ex(0.0, (n+1)*DT), u_ex(1.0, (n+1)*DT)
        u = v
    return u

REF = u_ex(X, NT * DT)
e_cheap = np.abs(march(None) - REF)
e_all   = np.abs(march(np.ones((NT, NX), bool)) - REF)
print(f"dynamic range: all-cheap err {np.median(e_cheap):.3e} -> "
      f"all-expensive {np.median(e_all):.3e}  ({np.median(e_cheap)/np.median(e_all):.1f}x)")

tau_gap = np.zeros((NT, NX)); u = u_ex(X, 0.0).copy()
for n in range(NT):
    lo = np.zeros(NX); hi = np.zeros(NX)
    lo[1:-1] = np.einsum('ij,ij->i', W1[1:-1], sw(u,3))
    hi[2:-2] = np.einsum('ij,ij->i', W2[2:-2], sw(u,5))
    tau_gap[n,2:-2] = np.abs(hi[2:-2] - lo[2:-2])
    u = lo + DT*S(X, n*DT); u[0], u[-1] = u_ex(0.0,(n+1)*DT), u_ex(1.0,(n+1)*DT)

def infl_exact(js):
    G = np.zeros((NT,NX)); g = np.zeros(NX); g[js] = 1.0
    for n in range(NT-1,-1,-1):
        G[n] = g; gn = np.zeros(NX)
        for j in range(1,NX-1): gn[j-1:j+2] += W1[j]*g[j]
        g = gn; g[[0,-1]] = 0.0
    return G

def infl_crude(js):
    ab = ALPHA.mean(); G = np.zeros((NT,NX))
    for n in range(NT):
        t = (NT-n)*DT
        G[n] = np.exp(-(X-(X[js]-C*t))**2/(2*(2*ab*t+1e-18)))
    return G

print(f"\n{'x*':>6} {'budget':>7} | {'uniform':>10} {'local':>10} {'crude':>10} {'exact':>10}"
      f" | {'crude':>7} {'exact':>7}")
rows=[]
for xs in [0.3,0.6]:
    js = int(round(xs/DX)); Ge, Gc = infl_exact(js), infl_crude(js)
    for frac in [0.10,0.30]:
        k = int(frac*NX*NT); out={}
        for name,sc in [("uniform",np.random.default_rng(1).random((NT,NX))),
                        ("local",tau_gap),("crude",Gc*tau_gap),("exact",Ge*tau_gap)]:
            f = np.argsort(-sc.ravel())[:k]
            M = np.zeros(NT*NX,bool); M[f]=True
            out[name] = abs(march(M.reshape(NT,NX))[js] - REF[js])
        rows.append((out["local"]/out["crude"], out["local"]/out["exact"]))
        print(f"{xs:>6.2f} {frac:>6.0%} | {out['uniform']:>10.3e} {out['local']:>10.3e}"
              f" {out['crude']:>10.3e} {out['exact']:>10.3e} | {rows[-1][0]:>6.2f}x {rows[-1][1]:>6.2f}x")
r = np.array(rows)
print(f"\nmedian gain over the local indicator: crude cone {np.median(r[:,0]):.1f}x, "
      f"exact adjoint {np.median(r[:,1]):.1f}x")
print(f"crude retains {100*np.median(r[:,0])/np.median(r[:,1]):.0f}% of the exact-adjoint gain")
