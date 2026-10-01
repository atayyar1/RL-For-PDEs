# ── One model for many queries: train on 15, test on 6 it has never seen ──
TRAIN_Q = [(x, t) for t in (10, 15, 20, 25, 30) for x in (30, 50, 70)]
TEST_Q  = [(40, 12), (60, 18), (40, 22), (60, 28), (50, 35), (50, 40)]     # last two: longer than any training query
LAM_MQ, SEED, TOTAL = 8.0, 1, 10_000_000

t0 = time.time()
for z in TRAIN_Q + TEST_Q:
    reference(z)                     # computed once here, then copied to every worker process
print(f"reference maps for {len(TRAIN_Q) + len(TEST_Q)} queries: {time.time() - t0:.0f} s")

class MultiProgressLog(EpisodeLog):
    """Every `every` steps: points and score relative to each query's own reference map (last 50 episodes)."""
    def __init__(self, every=500_000):
        super().__init__()
        self.every, self.next_print, self.t0 = every, every, time.time()
    def _on_step(self):
        super()._on_step()
        if self.num_timesteps >= self.next_print:
            recent = self.episodes[-50:]
            if recent:
                print(f"   {self.num_timesteps / 1e6:4.1f}M steps | {len(self.episodes):5d} episodes | last 50: "
                      f"points {np.mean([x['points_ratio'] for x in recent]):.2f}x ref, score {np.mean([x['score_ratio'] for x in recent]):.2f}x ref "
                      f"| {time.time() - self.t0:5.0f} s")
            self.next_print += self.every
        return True

vec   = SubprocVecEnv([make_env_cls(PendingStateEnv, TRAIN_Q, LAM_MQ, 100 * SEED + i) for i in range(N_ENVS)])
model = MaskablePPO("MlpPolicy", vec, n_steps=256, batch_size=256, gamma=1.0, gae_lambda=1.0, seed=SEED, verbose=0)
model.learn(total_timesteps=TOTAL, callback=MultiProgressLog())
vec.close()
model.save(f"local_reward_multi_lam{LAM_MQ:.0f}_seed{SEED}_{date.today()}")

# evaluation: PPO against the best heuristic on every query
def compare(z):
    B_ref, N_ref = reference(z)
    tot, info, e_z = evaluate_cls(PendingStateEnv, model, z, LAM_MQ)
    best = None
    for name, ch in [("reference", choose_ref), ("greedy", make_greedy(B_ref, N_ref, LAM_MQ)), ("fewest", choose_min)]:
        _, B_b, N_b = build_map(z, ch)
        r = -B_b / B_ref - LAM_MQ * (N_b - 1) / N_ref
        if best is None or r > best[0]:
            best = (r, N_b, name)
    return tot, info["n_points"], best, exact_bound(e_z.m)

wins = {"train": 0, "test": 0}
print(f"\nlam = {LAM_MQ}               PPO (ret / pts)   best rule (ret / pts / which)      FD pts   PPO bound   FD error")
for label, queries in [("train", TRAIN_Q), ("test", TEST_Q)]:
    for z in queries:
        tot, n_pts, best, bound = compare(z)
        win = tot > best[0]
        wins[label] += win
        print(f"{label:5s} z* = {str(z):9s}  {tot:7.2f} / {n_pts:4d}    {best[0]:7.2f} / {best[1]:4d} / {best[2]:9s}   "
              f"{fd_reference(*z)[2]:5d}   {bound:.1e}   {fd_reference(*z)[1]:.1e}{'  <- PPO wins' if win else ''}")
print(f"\nPPO beats every heuristic on {wins['train']}/{len(TRAIN_Q)} training queries and {wins['test']}/{len(TEST_Q)} unseen queries")
