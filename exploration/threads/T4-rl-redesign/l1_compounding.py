"""Does ||w||_1 > 1 COMPOUND in practice? (high-order regime, where it can show)

First attempt used 3 Taylor rows at m=2,3: truncation error swamped amplification
and positive vs min-norm were identical to 4 digits. T2's result is at P=6 rows,
m=8 -- high order, so truncation is tiny and amplification can dominate. Redone
there.

Reconciliation being tested: von Neumann needs translation invariance. A fixed
uniform stencil has a symbol and the l1 bound is loose (T5: Lax-Wendroff). With
variable coefficients there is no single symbol -- is compounding real there?
"""
import sys, os
sys.path.insert(0, "/Users/josephbakarji/Documents/00-projects/01-tests/RL-For-PDEs/exploration")
import numpy as np
from scipy.optimize import linprog

NX = 201
X = np.linspace(0, 1, NX); DX = X[1]-X[0]; C = 1.0
ALPHA = 0.02 + 0.38*(1 + np.cos(2*np.pi*X))/2
DT = 0.45*DX**2/ALPHA.max()

def gm(mu, var, p):
    m = [1.0, mu]
    for q in range(2, p+1): m.append(mu*m[q-1] + (q-1)*var*m[q-2])
    return np.array(m)

def rows(j, m, P):
    idx = np.arange(-m, m+1); h = m*DX
    A = np.vstack([(idx*DX/h)**q for q in range(P+1)])
    b = gm(-C*DT, 2*ALPHA[j]*DT, P) / h**np.arange(P+1)
    return A, b

def w_pos(j, m, P):
    A, b = rows(j, m, P)
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=b, bounds=[(0,None)]*A.shape[1], method="highs")
    return r.x if r.status == 0 else None

def w_min(j, m, P):
    A, b = rows(j, m, P)
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w if abs(w.sum()-1) < 1e-6 else None

m, P = 8, 6
WP = [w_pos(j, m, P) for j in range(NX)]
WM = [w_min(j, m, P) for j in range(NX)]
good = [j for j in range(m, NX-m) if WP[j] is not None and WM[j] is not None]
print(f"m={m}, P={P}:  {len(good)}/{NX-2*m} interior points solvable both ways")
print(f"  mean ||w||_1   positive {np.mean([np.abs(WP[j]).sum() for j in good]):.6f}"
      f"   min-norm {np.mean([np.abs(WM[j]).sum() for j in good]):.6f}"
      f"   max min-norm {max(np.abs(WM[j]).sum() for j in good):.4f}")
print(f"  min-norm negative weights at {sum(1 for j in good if WM[j].min()<-1e-12)}/{len(good)} points")

W3 = [np.array([ALPHA[j]*DT/DX**2 + C*DT/DX/2, 1-2*ALPHA[j]*DT/DX**2,
                ALPHA[j]*DT/DX**2 - C*DT/DX/2]) for j in range(NX)]

def march(W, n):
    u = np.sin(np.pi*X).copy()
    for _ in range(n):
        v = np.zeros(NX)
        for j in range(NX):
            if 1 <= j < m or NX-m <= j < NX-1: v[j] = W3[j] @ u[j-1:j+2]
            elif m <= j < NX-m and W[j] is not None: v[j] = W[j] @ u[j-m:j+m+1]
            elif 0 < j < NX-1: v[j] = W3[j] @ u[j-1:j+2]
        v[[0,-1]] = 0.0; u = v
    return u

def ref(n, sub=32):
    u = np.sin(np.pi*X).copy(); d = DT/sub
    Ws = [np.array([ALPHA[j]*d/DX**2 + C*d/DX/2, 1-2*ALPHA[j]*d/DX**2,
                    ALPHA[j]*d/DX**2 - C*d/DX/2]) for j in range(NX)]
    for _ in range(n*sub):
        v = np.zeros(NX)
        for j in range(1, NX-1): v[j] = Ws[j] @ u[j-1:j+2]
        u = v
    return u

print(f"\n  {'steps':>6} {'positive':>13} {'min-norm':>13} {'ratio':>9}")
for n in [4, 8, 16, 32]:
    R = ref(n)
    ep = np.max(np.abs(march(WP, n) - R)); em = np.max(np.abs(march(WM, n) - R))
    print(f"  {n:>6} {ep:>13.3e} {em:>13.3e} {em/ep:>8.2f}x")
