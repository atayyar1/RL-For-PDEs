# ── Score in the state: for each cell, the best stencil score still reachable if it is picked next ──
_scored_cache = {}
def scored_sets(ix, it):
    """For every admissible stencil at (ix, it): which cells it uses (bool matrix) and log10 of its
    score relative to the best one here. Cached per local situation."""
    key = tuple((q[0] - ix, q[1] - it) for q in (place(ix, it, a) for a in range(N_CELLS)))
    if key not in _scored_cache:
        sets = admissible_sets(ix, it)
        uses = np.zeros((len(sets), N_CELLS), dtype=bool)
        for k, s in enumerate(sets):
            uses[k, list(s)] = True
        logs = np.log10([s_of((ix, it), s) for s in sets])
        _scored_cache[key] = (uses, logs - logs.min())
    return _scored_cache[key]

def reachable_score(ix, it, picks):
    """Per cell: log10(best score reachable with picks + that cell / best score at this point).
    0 = the best stencil is still reachable through this cell. Cells that can't be picked get the cap 3."""
    uses, rel = scored_sets(ix, it)
    ok = uses[:, picks].all(axis=1) if picks else np.ones(len(rel), dtype=bool)   # stencils still reachable
    out = np.where(uses[ok], rel[ok, None], 3.0).min(axis=0) if ok.any() else np.full(N_CELLS, 3.0)
    out[picks] = 3.0                                                               # already picked
    return np.minimum(out, 3.0)

class ScoreStateEnv(LocalRewardEnv):
    """LocalRewardEnv plus one number per window cell: how good a stencil can still be built through it."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS + N_CELLS,), dtype=np.float32)

    def _obs(self):
        if not self.m["pending"]:
            return np.zeros(N_OBS + N_CELLS, np.float32)
        extra = reachable_score(*next_point(self.m), self.picks)
        return np.concatenate([super()._obs(), extra]).astype(np.float32)

# check at (50, 10): before any pick, and after picking cell 0
print("before any pick:  ", np.round(reachable_score(50, 10, []), 2))
print("after picking 0:  ", np.round(reachable_score(50, 10, [0]), 2))
e16 = ScoreStateEnv(queries=[Z_TRAIN]); o, _ = e16.reset(seed=0)
print("obs size:", o.shape[0])
