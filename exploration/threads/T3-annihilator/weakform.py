"""
weakform.py -- weak-form (integral) SINDy, the closest competitor.

Library terms are d_x^a ( u^p ) (the Messenger-Bortz form).  Integrating
against a compactly supported test function phi on a spacetime patch moves
every derivative onto phi:

    int phi d_x^a(u^p)  =  (-1)^a int (d_x^a phi) u^p
    int phi u_t         =  -int (d_t phi) u

so no derivative of the DATA is ever taken.  Note the structural restriction:
only terms that are exact x-derivatives of functions of u can appear.  A term
like u*u_xx has no divergence form and is simply not expressible.
"""
import numpy as np


def _poly_pow(p):
    """coefficients of (1 - z^2)^p in increasing powers of z"""
    c = np.array([1.0])
    base = np.array([1.0, 0.0, -1.0])
    for _ in range(p):
        c = np.convolve(c, base)
    return c


def _dpoly(c, k):
    for _ in range(k):
        c = np.polynomial.polynomial.polyder(c)
    return c


class WeakSINDy:
    def __init__(self, mx=12, mt=12, px=6, pt=6):
        self.mx, self.mt, self.px, self.pt = mx, mt, px, pt
        self.cx = _poly_pow(px)
        self.ct = _poly_pow(pt)

    def _phi(self, a, b, Hx, Ht):
        """phi values and (d_x^a d_t^b phi) on the patch node grid"""
        xi = np.linspace(-1, 1, 2 * self.mx + 1)
        ta = np.linspace(-1, 1, 2 * self.mt + 1)
        fx = np.polynomial.polynomial.polyval(xi, _dpoly(self.cx, a)) / Hx ** a
        ft = np.polynomial.polynomial.polyval(ta, _dpoly(self.ct, b)) / Ht ** b
        return np.outer(fx, ft)

    def assemble(self, grid, Un, centers, terms, Hx_cells=None, Ht_cells=None):
        """terms: list of (p, a) meaning d_x^a (u^p).  Returns Theta, b, and the
        weak LHS int phi u_t."""
        Hx_cells = Hx_cells or self.mx
        Ht_cells = Ht_cells or self.mt
        dx, dt = grid.dx, grid.dt
        Hx, Ht = Hx_cells * dx, Ht_cells * dt
        # node spacing inside the patch (patch resampled onto its own node grid)
        sx = max(1, Hx_cells // self.mx)
        st = max(1, Ht_cells // self.mt)
        wx = np.ones(2 * self.mx + 1); wx[0] = wx[-1] = 0.5
        wt = np.ones(2 * self.mt + 1); wt[0] = wt[-1] = 0.5
        quad = np.outer(wx, wt) * (sx * dx) * (st * dt)

        P = {(a, b): self._phi(a, b, Hx, Ht) for (a, b) in
             [(0, 0), (0, 1)] + [(t[1], 0) for t in terms]}
        rows, rhs = [], []
        for (i, j) in centers:
            U = Un[i - self.mx * sx: i + self.mx * sx + 1: sx,
                   j - self.mt * st: j + self.mt * st + 1: st]
            if U.shape != (2 * self.mx + 1, 2 * self.mt + 1):
                continue
            rhs.append(-np.sum(P[(0, 1)] * U * quad))          # int phi u_t
            rows.append([((-1) ** a) * np.sum(P[(a, 0)] * U ** p * quad)
                         for (p, a) in terms])
        return np.array(rows), np.array(rhs)
