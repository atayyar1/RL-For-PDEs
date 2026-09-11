"""Linear advection-diffusion: PDE setup, grids, and exact solutions.

    u_t + c u_x = alpha u_xx      on [0,1],  u(0,t) = u(1,t) = 0

Everything downstream imports its grid and its ground truth from here, so that
no two threads can silently disagree about dx, dt or u_true.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Problem:
    alpha: float = 0.1
    c: float = 1.0
    nx: int = 201
    T: float = 0.5

    # ---- grid -------------------------------------------------------------
    @property
    def x(self):
        return np.linspace(0.0, 1.0, self.nx)

    @property
    def dx(self):
        return 1.0 / (self.nx - 1)

    @property
    def dt(self):
        """Explicit CFL step: the tighter of the advective and diffusive limits."""
        adv = 0.9 * self.dx / abs(self.c) if self.c else np.inf
        dif = 0.45 * self.dx**2 / self.alpha if self.alpha else np.inf
        return min(adv, dif)

    @property
    def r(self):
        """Diffusion number alpha dt / dx^2. Positivity of FTCS needs r <= 1/2."""
        return self.alpha * self.dt / self.dx**2

    @property
    def nu(self):
        """Courant number c dt / dx."""
        return self.c * self.dt / self.dx

    @property
    def cell_peclet(self):
        """c dx / alpha. Positivity of centred advection needs <= 2."""
        return self.c * self.dx / self.alpha if self.alpha else np.inf

    # ---- exact solution ---------------------------------------------------
    def u_true(self, x, t, ic="sin2", n_series=400, nq=40001):
        """Analytic solution for u0(x) = sin(pi x) + 0.5 sin(2 pi x), u=0 at the walls.

        Substituting u = exp(beta x - alpha beta^2 t) v with beta = c/(2 alpha)
        removes advection and leaves the heat equation for v, solved by its sine
        series. Cached on (ic, n_series, nq).
        """
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, float))
        if t == 0.0:
            # BUG FIX: at t=0 the transformed series has NOT been damped, and
            # v0 = u0*exp(-beta x) has non-zero second derivative at the walls, so
            # its sine series converges only as O(n^-3). The reconstruction is then
            # multiplied by exp(beta x) -- at beta=5 that is 12x at midspan -- giving
            # ~5e-7 error in the INITIAL CONDITION while the solution at t>0 is exact
            # to 2e-16 because the modes are damped. Return the analytic form instead.
            u0 = {"sin2": np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x),
                  "sin":  np.sin(np.pi * x)}[ic]
            return float(u0[0]) if scalar else u0
        bn, narr, beta = self._series(ic, n_series, nq)
        v = (np.sin(np.pi * np.outer(x, narr)) *
             (bn * np.exp(-self.alpha * (narr * np.pi) ** 2 * t))).sum(-1)
        out = np.exp(beta * x - self.alpha * beta**2 * t) * v
        return float(out[0]) if scalar else out

    _CACHE = {}

    def _series(self, ic, n_series, nq):
        key = (self.alpha, self.c, ic, n_series, nq)
        if key not in Problem._CACHE:
            beta = self.c / (2 * self.alpha) if self.alpha else 0.0
            xq = np.linspace(0.0, 1.0, nq)
            u0 = {"sin2": np.sin(np.pi * xq) + 0.5 * np.sin(2 * np.pi * xq),
                  "sin":  np.sin(np.pi * xq)}[ic]
            v0 = u0 * np.exp(-beta * xq)
            narr = np.arange(1, n_series + 1)
            bn = np.array([2.0 * np.trapz(v0 * np.sin(n * np.pi * xq), xq) for n in narr])
            Problem._CACHE[key] = (bn, narr, beta)
        return Problem._CACHE[key]


DEFAULT = Problem()
