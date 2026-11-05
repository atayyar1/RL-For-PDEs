"""Verify T3's reopening: arbitrary-order rows on MIXED TIME LEVELS, scattered.

Generating function: sum_i w_i exp[s*A_i + s^2*B_i] = 1 with A = dx - c*dt,
B = alpha*dt.  Coefficient of s^q in exp(sA + s^2 B) is

    sum_j A^(q-2j) B^j / ((q-2j)! j!)

so row q is that, with RHS delta_{q0}.  At q=2 it must reduce to the F8-corrected
row 1/2(dx - c dt)^2 + alpha*dt.  This is the one construction that handles
scattered points at arbitrary order -- rows_moment needs a single time level,
rows_taylor is capped at p=2.
"""
import sys, os
sys.path.insert(0, "/Users/josephbakarji/Documents/00-projects/01-tests/RL-For-PDEs/exploration")
import numpy as np
from math import factorial
from core.pde import Problem
from core import stencil as st

p = Problem(nx=401)

def rows_taylor_p(dxi, dti, order, prob=p):
    A = dxi - prob.c*dti
    B = prob.alpha*dti
    h = max(np.max(np.abs(dxi)), np.sqrt(prob.alpha*np.max(np.abs(dti))), 1e-12)
    R = []
    for q in range(order+1):
        row = np.zeros(len(dxi))
        for j in range(q//2 + 1):
            row += A**(q-2*j) * B**j / (factorial(q-2*j)*factorial(j))
        R.append(row / h**q)
    b = np.zeros(order+1); b[0] = 1.0
    return np.array(R), b

print("=== 1. Does it reduce to the corrected row at p=2? ===")
rng = np.random.default_rng(0)
dxi = rng.uniform(-5,5,7)*p.dx; dti = -rng.uniform(0.5,4,7)*p.dt
A2,_ = rows_taylor_p(dxi,dti,2); A1,_ = st.rows_taylor(dxi,dti,p)
# compare row 2 up to the shared h-normalisation
print(f"   max |row2_new - row2_corrected| = {np.abs(A2[2]-A1[2]).max():.2e}")

print("\n=== 2. Scattered, mixed time levels: feasibility and floor vs order ===")
def exact(x,t,ks=(1,2,3),amps=(1,.5,.25)):
    return sum(a*np.sin(k*np.pi*(x-p.c*t))*np.exp(-p.alpha*(k*np.pi)**2*t)
               for k,a in zip(ks,amps))

print(f"{'order p':>8} {'n pts':>6} | {'feasible %':>11} {'median |err|':>14} {'maxent vs LP':>13}")
for order in [2, 4, 6]:
    npts = max(3*(order+1), 12)
    feas = 0; errs = []; ratios = []
    for trial in range(250):
        xs, ts = 0.5, 30*p.dt
        dxi = rng.uniform(-6,6,npts)*p.dx
        dti = -rng.uniform(0.3,3.0,npts)*p.dt      # MIXED time levels
        Ar,br = rows_taylor_p(dxi,dti,order)
        if np.linalg.matrix_rank(Ar) < order+1: continue
        w = st.solve_maxent(Ar,br)
        if w is None: continue
        feas += 1
        u_nb = exact(xs+dxi, ts+dti); uref = exact(xs, ts)
        errs.append(abs(w@u_nb - uref))
        wl = st.solve_positive(Ar,br)
        if wl is not None:
            e_lp = abs(wl@u_nb-uref)
            if errs[-1] > 0: ratios.append(e_lp/errs[-1])
    print(f"{order:>8} {npts:>6} | {100*feas/250:>10.1f}% {np.median(errs):>14.3e}"
          f" {np.median(ratios) if ratios else float('nan'):>12.1f}x")
