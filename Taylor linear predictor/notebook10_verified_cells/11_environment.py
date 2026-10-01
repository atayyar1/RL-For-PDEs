# ── The environment: one step = one pick; a reward every time a stencil is completed ──
_ref_cache = {}
def reference(z_star):
    """(B_ref, N_ref) for this query: score sum and point count of the reference map."""
    if z_star not in _ref_cache:
        _, B_ref, N_ref = build_map(z_star, choose_ref)
        _ref_cache[z_star] = (B_ref, N_ref)
    return _ref_cache[z_star]

N_OBS = 3 * N_CELLS + 5

class LocalRewardEnv(gym.Env):
    def __init__(self, queries, lam=1.0, cap=3.0):
        super().__init__()
        self.queries, self.lam, self.cap = queries, lam, cap
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_CELLS)          # one action = one window cell

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.z_star = self.queries[self.np_random.integers(len(self.queries))]
        self.B_ref, self.N_ref = reference(self.z_star)
        self.m, self.picks, self.B = start_map(self.z_star), [], 0.0
        self._play_forced()                                   # only matters if z* is one level above the IC
        return self._obs(), {}

    def _complete(self, cells):
        """Expand the next point with these cells; return its reward."""
        p, n_new, term = expand(self.m, cells)
        self.B += term
        return -term / self.B_ref - self.lam * n_new / self.N_ref

    def _play_forced(self):
        """Points with exactly one admissible stencil (e.g. one level above the IC) need no decision."""
        r = 0.0
        while self.m["pending"] and len(admissible_sets(*next_point(self.m))) == 1:
            r += self._complete(admissible_sets(*next_point(self.m))[0])
        return r

    def _obs(self):
        if not self.m["pending"]:
            return np.zeros(N_OBS, np.float32)
        p = next_point(self.m)
        picked, exists, known = np.zeros(N_CELLS), np.zeros(N_CELLS), np.zeros(N_CELLS)
        picked[self.picks] = 1
        for a in range(N_CELLS):
            q = place(*p, a)
            known[a]  = is_known(*q)
            exists[a] = q in self.m["beta"]
        glob = [np.log10(abs(self.m["beta"][p]) + 1e-12),     # influence of this point on z*
                p[1],                                        # levels above the IC
                p[0] - self.z_star[0],                       # offset to z* in x
                len(self.picks),                             # pick number
                len(self.m["stencil"]) / self.N_ref]         # points used so far
        return np.concatenate([picked, exists, known, glob]).astype(np.float32)

    def step(self, a):
        self.picks.append(int(a))
        p = next_point(self.m)
        if len(self.picks) < n_picks(*p):                     # still choosing this point's stencil
            return self._obs(), 0.0, False, False, {}
        r = self._complete(self.picks)
        self.picks = []
        r += self._play_forced()
        N = len(self.m["stencil"]) + len(self.m["pending"])
        if self.m["pending"] and N <= self.cap * self.N_ref:  # map not finished
            return self._obs(), r, False, False, {}
        # finished, or over the point cap: each unexpanded point costs its worst admissible stencil
        for neg_it, ix in self.m["pending"]:
            q = (ix, -neg_it)
            r -= abs(self.m["beta"][q]) * max(s_of(q, s) for s in admissible_sets(*q)) / self.B_ref
        info = {"finished": not self.m["pending"],
                "n_points": len(self.m["stencil"]),
                "score_ratio": self.B / self.B_ref,
                "points_ratio": len(self.m["stencil"]) / self.N_ref,
                "error": true_error(self.m) if not self.m["pending"] else np.nan}   # u_true ONLY for logging
        return np.zeros(N_OBS, np.float32), r, True, False, info

    def action_masks(self):
        return pick_mask(*next_point(self.m), self.picks)

# check: one episode with random (masked) picks
env = LocalRewardEnv(queries=[(50, 10)])
obs, _ = env.reset(seed=0)
rng = np.random.default_rng(0)
total, steps, done = 0.0, 0, False
while not done:
    a = rng.choice(np.flatnonzero(env.action_masks()))
    obs, r, done, _, info = env.step(a)
    total += r; steps += 1
print(f"obs size {N_OBS} | {steps} picks | {info['n_points']} points | error {info['error']:.2e} "
      f"| score {info['score_ratio']:.1f}x reference | points {info['points_ratio']:.2f}x reference")
print(f"sum of rewards {total:.4f}   check: -score_ratio - lam*(points - 1)/N_ref = "
      f"{-info['score_ratio'] - env.lam * (info['n_points'] - 1) / env.N_ref:.4f}")
