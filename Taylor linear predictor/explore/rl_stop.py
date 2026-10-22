"""MaskablePPO with a stop action (one extra action: "take the guess value at p, expand nothing").

    python explore/rl_stop.py train  <run_name> <steps> <queries: "25,45" or "multi">  [norm]
    python explore/rl_stop.py eval   <run_name>
Reward per point (estimates only):  expanded  -|beta_p| s_p / EPS - lam new / N_LW   (cell 15)
                                    stopped   -|beta_p| g_err(p) / EPS
norm = "lw": the accuracy part is divided by B_LW / EPS (the LW map's score sum of the same query), so every query's LW
map scores about -1 - lam and multi-query returns are comparable across t*.
State: cell 15's + log10 of the stop cost |beta_p| g_err / EPS (what stopping would cost now).
"""
import sys, json
from env import *                     # cell 15 environment, on top of core (advection)
from stop import build_map_s, report_s, stop_point, guess_err, in_band_fraction
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

STOP = N_CELLS
N_OBS_S = N_OBS + 1
_blw = {}
def b_lw(z):
    """Score sum of the LW map of z* (in EPS): the per-query accuracy unit when norm = 'lw'."""
    if z not in _blw:
        _blw[z] = score_sum(build_map(z, lw)) / EPS
    return _blw[z]


class StopEnv(MapEnv):
    def __init__(self, queries, lam_range=(1.0, 1.0), cap=3.0, norm=None):
        super().__init__(queries, lam_range, cap)
        self.norm = norm
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(N_OBS_S,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_CELLS + 1)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.m["stopped"] = set()
        self.unit = b_lw(self.z) if self.norm == "lw" else 1.0
        return self._obs(), info

    def _accuracy(self, pts):
        return super()._accuracy(pts) / self.unit

    def _obs(self):
        o = super()._obs()
        if not self.m["pending"]:
            return np.zeros(N_OBS_S, np.float32)
        p = next_point(self.m)
        return np.r_[o, np.log10(abs(self.m["beta"][p]) * guess_err(p) / EPS + 1e-12)].astype(np.float32)

    def step(self, k):
        if k != STOP:
            obs, r, done, trunc, info = super().step(k)
            return (np.zeros(N_OBS_S, np.float32) if done else obs), r, done, trunc, self._info(info, done)
        p = next_point(self.m)
        cost = abs(self.m["beta"][p]) * guess_err(p)
        self.B += cost
        r = -cost / EPS / self.unit
        stop_point(self.m)
        if self.m["level"] and (not self.m["pending"] or -self.m["pending"][0][0] != p[1]):
            r -= self._accuracy(close_level(self.m))
        r += self._play_forced()
        if self.m["pending"]:
            return self._obs(), r, False, False, {}
        r -= self._accuracy(close_level(self.m)) if self.m["level"] else 0.0
        info = {"finished": True, "points": len(self.m["stencil"]), "lam": self.lam,
                "score_ratio": self.B / EPS, "points_ratio": len(self.m["stencil"]) / self.N_ref}
        return np.zeros(N_OBS_S, np.float32), r, True, False, self._info(info, True)

    def _info(self, info, done):
        if done:
            info["stops"] = len(self.m["stopped"])
        return info

    def action_masks(self):
        p = next_point(self.m)
        return np.r_[pick_mask(self.m, p, self.picks), not self.picks and p != self.z]


def make_env(queries, seed, norm):
    def _init():
        e = Monitor(ActionMasker(StopEnv(queries, norm=norm), lambda e: e.action_masks()))
        e.reset(seed=seed)
        return e
    return _init


def run_policy(model, z, norm=None):
    env = StopEnv([z], norm=norm)
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, r, done, _, info = env.step(int(a))
    return env.m


Q_TRAIN_MULTI = [(ix, it) for it in (30, 45, 50) for ix in (8, 20, 30, 42)]     # disjoint from the test queries
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    run_dir = os.path.join(RUNS, name)
    if mode == "train":
        steps, qarg = int(float(sys.argv[3])), sys.argv[4]
        norm = sys.argv[5] if len(sys.argv) > 5 else None
        queries = Q_TRAIN_MULTI if qarg == "multi" else [tuple(int(v) for v in qarg.split(","))]
        os.makedirs(run_dir, exist_ok=True)
        json.dump(dict(steps=steps, queries=queries, norm=norm), open(os.path.join(run_dir, "config.json"), "w"))
        for z in queries:
            n_lw(z); b_lw(z)
        vec = SubprocVecEnv([make_env(queries, 100 + i, norm) for i in range(4)])
        model = MaskablePPO("MlpPolicy", vec, seed=1, verbose=0, device="cpu", n_steps=1024, batch_size=1024,
                            gamma=1.0, gae_lambda=0.97, learning_rate=3e-4, ent_coef=0.0)
        t0, hist = time.time(), []
        for chunk in range(10):                                   # 10 chunks: a learning curve on the first query
            model.learn(steps // 10, reset_num_timesteps=False)
            r = report_s(run_policy(model, queries[0], norm))
            hist.append(dict(step=model.num_timesteps, **r))
            print(f"{model.num_timesteps:>9,} steps ({(time.time() - t0) / 60:.0f} min): {r['points']} points, "
                  f"{r['stops']} stops, error {r['error']:.2e}, bound {r['bound']:.2e}", flush=True)
            json.dump(hist, open(os.path.join(run_dir, "curve.json"), "w"), default=float)
        model.save(os.path.join(run_dir, "model"))
        vec.close()
    else:
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
        model = MaskablePPO.load(os.path.join(run_dir, "model"), device="cpu")
        from evaluate import Q_UNSEEN, Q_NEW
        out = {}
        for z in [tuple(q) for q in cfg["queries"][:1]] + Q_UNSEEN + Q_NEW:
            m = run_policy(model, z, cfg["norm"])
            out[str(z)] = dict(report_s(m), band=in_band_fraction(m))
            print(z, {k: f"{v:.3g}" for k, v in out[str(z)].items()}, flush=True)
        json.dump(out, open(os.path.join(run_dir, "eval.json"), "w"), default=float)
