"""
griddata.py -- a shared, cached, noisy spacetime grid.

Every method in tasks 3-5 sees EXACTLY the same measured data: one uniform
(nx x nt) grid of u values plus one realisation of additive Gaussian noise.
That is what makes the comparison against weak-form SINDy fair -- the jet
route and WSINDy consume the same bytes.
"""
import os
import numpy as np

CACHE = "/private/tmp/claude-502/-Users-josephbakarji-Documents-05-nexus/" \
        "662f8a62-5a19-43c9-b637-f4798806d47e/scratchpad"


class Grid:
    def __init__(self, sol, nx=200, t0=0.10, t1=0.40, dt=None, alpha_scale=None, tag=""):
        self.sol = sol
        self.x = np.linspace(0, 1, nx)
        self.dx = self.x[1] - self.x[0]
        a = alpha_scale if alpha_scale is not None else getattr(sol, "alpha", 0.1)
        self.dt = dt if dt is not None else self.dx ** 2 / a
        self.t = np.arange(t0, t1 + 0.5 * self.dt, self.dt)
        self.nx, self.nt = len(self.x), len(self.t)
        os.makedirs(CACHE, exist_ok=True)
        f = os.path.join(CACHE, f"grid_{tag}_{nx}_{self.nt}_{t0}_{t1}.npy")
        if os.path.exists(f):
            self.U = np.load(f)
        else:
            self.U = np.empty((self.nx, self.nt))
            for j, tt in enumerate(self.t):
                self.U[:, j] = sol.u(self.x, tt)
            np.save(f, self.U)
        self.urms = float(np.sqrt(np.mean(self.U ** 2)))

    def noisy(self, eta, seed=0):
        if eta == 0:
            return self.U.copy()
        rng = np.random.default_rng(seed)
        return self.U + eta * self.urms * rng.normal(size=self.U.shape)

    # ---- local scattered cloud around a grid target, past-only in t ------------
    def cloud(self, i, j, m, ht_cells, Un, subsample=None, rng=None):
        """Neighbours within |dx| <= m*dx and -ht_cells*dt <= dt <= 0."""
        ii = np.arange(max(i - m, 0), min(i + m + 1, self.nx))
        jj = np.arange(max(j - ht_cells, 0), j + 1)
        I, Jd = np.meshgrid(ii, jj, indexing="ij")
        I, Jd = I.ravel(), Jd.ravel()
        keep = ~((I == i) & (Jd == j))
        I, Jd = I[keep], Jd[keep]
        if subsample is not None and len(I) > subsample:
            sel = (rng or np.random.default_rng(0)).choice(len(I), subsample, replace=False)
            I, Jd = I[sel], Jd[sel]
        return (self.x[I] - self.x[i], (Jd - j) * self.dt, Un[I, Jd])
