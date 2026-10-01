# ── State fix: let the agent see the work still pending ──
class PendingStateEnv(ScoreStateEnv):
    """ScoreStateEnv plus the pending workload: how many points still need a stencil, and their total influence."""
    N_EXTRA = 2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS + N_CELLS + self.N_EXTRA,), dtype=np.float32)

    def _obs(self):
        if not self.m["pending"]:
            return np.zeros(N_OBS + N_CELLS + self.N_EXTRA, np.float32)
        pend = [(ix, -neg_it) for neg_it, ix in self.m["pending"]]
        extra = [len(pend) / self.N_ref,                                   # points still waiting
                 sum(abs(self.m["beta"][q]) for q in pend)]               # their total influence on z*
        return np.concatenate([super()._obs(), extra]).astype(np.float32)

def make_env_cls(cls, queries, lam, seed):
    def _init():
        env = Monitor(cls(queries=queries, lam=lam))
        env.reset(seed=seed)
        return env
    return _init

def evaluate_cls(cls, model, z, lam):
    e = cls(queries=[z], lam=lam); o, _ = e.reset(seed=0); d = False; tot = 0.0
    while not d:
        a, _ = model.predict(o, action_masks=e.action_masks(), deterministic=True)
        o, r, d, _, info = e.step(a); tot += r
    return tot, info, e

e26 = PendingStateEnv(queries=[Z_HARD], lam=LAM_HARD); o, _ = e26.reset(seed=0)
for k in range(30):
    o, r, d, _, _ = e26.step(np.flatnonzero(e26.action_masks())[0])
print("obs size:", o.shape[0], "| pending points / N_ref:", round(float(o[-2]), 3), "| pending influence:", round(float(o[-1]), 3))
