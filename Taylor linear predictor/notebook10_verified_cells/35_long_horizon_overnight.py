# ── Overnight: (A) is the best fixed stencil the same at every t*?  (B) long horizons, where the walls matter ──
A_QUERIES  = [(50, 10), (50, 20), (50, 30)]
LONG_TRAIN = [(x, t) for t in (60, 80, 100) for x in (30, 50, 70)]
LONG_TEST  = [(40, 70), (60, 90), (50, 120)]
FAR_TEST   = [(50, 300), (50, 1000)]
LONG_TOTAL = 20_000_000
N_JOBS     = 12
from joblib import Parallel, delayed

class LongEnv(PendingStateEnv):
    """PendingStateEnv with 'levels above the IC' clipped at 10: the agent only needs to know when the IC is close,
    and the policy can then be applied at horizons far beyond training (t* = 300, 1000)."""
    def _obs(self):
        o = super()._obs()
        o[3 * N_CELLS + 1] = min(o[3 * N_CELLS + 1], 10.0)
        return o

work = lambda m: sum(len(set(nbs)) for nbs, w in m["stencil"].values())

def brute_force(z, lam, stencils):
    """Every stencil in `stencils` as a fixed rule at z*. Returns rows (stencil, points, score sum), best first at this lam."""
    chunks = np.array_split(np.arange(len(stencils)), N_JOBS)
    res = Parallel(n_jobs=N_JOBS)(delayed(fixed_chunk)([stencils[i] for i in idx], z) for idx in chunks)
    B_fd, N_fd = reference(z)
    rows = [r for part in res for r in part]
    return sorted(rows, key=lambda r: r[2] / B_fd + lam * (r[1] - 1) / N_fd)

def fixed_map(z, s):
    return build_map(z, lambda m, p: s if s in set(admissible_sets(*p)) else choose_ref(m, p))[0]

# ---- (A) best fixed stencil at t* = 10, 20, 30 ----
print("(A) best fixed stencil per horizon, lam = 1")
interior = admissible_sets(50, 20)
top = {}
for z in A_QUERIES:
    t0 = time.time()
    rows = brute_force(z, 1.0, interior)
    top[z] = [r[0] for r in rows[:50]]
    s, N, B = rows[0]
    print(f"   z* = {z}: best {[WINDOW[a] for a in s]}  {N} points, bound {exact_bound(fixed_map(z, s)):.1e}  ({time.time() - t0:.0f} s)")
print("   same best stencil at every t*:", len({tuple(top[z][0]) for z in A_QUERIES}) == 1)

# ---- (B) long horizons: train, then compare with FD-3, FD-5, lowest score and the best fixed stencils ----
print("\n(B) long horizons, lam = 1")
for z in LONG_TRAIN + LONG_TEST + FAR_TEST:
    reference(z)
vec   = SubprocVecEnv([make_env_cls(LongEnv, LONG_TRAIN, 1.0, 700 + i) for i in range(N_ENVS)])
model = MaskablePPO("MlpPolicy", vec, n_steps=256, batch_size=256, gamma=1.0, gae_lambda=1.0, seed=7, verbose=0)
model.learn(total_timesteps=LONG_TOTAL, callback=MultiProgressLog(every=LONG_TOTAL // 20))
vec.close()
model.save(f"local_reward_big21_long_clip_lam1_seed7_{date.today()}")

row5 = tuple(sorted(WINDOW.index((d, -1)) for d in (-2, -1, 0, 1, 2)))
candidates = list(dict.fromkeys(s for z in A_QUERIES for s in top[z]))           # union of the top-50 from (A)
print(f"\n{'':5s} {'query':10s} | {'FD-3':>22s} | {'FD-5':>22s} | {'lowest score':>22s} | {'best fixed (of top-50s)':>24s} | {'PPO':>22s}")
print(f"{'':5s} {'':10s} | {'pts   work   bound':>22s} | {'pts   work   bound':>22s} | {'pts   work   bound':>22s} | {'pts   work   bound':>24s} | {'pts   work   bound':>22s}")
for label, queries in [("train", LONG_TRAIN), ("test", LONG_TEST), ("far", FAR_TEST)]:
    for z in queries:
        t0 = time.time()
        best_s = brute_force(z, 1.0, candidates)[0][0]
        _, _, e_z = evaluate_cls(LongEnv, model, z, 1.0)
        maps = [fd_map(z), fixed_map(z, row5), build_map(z, choose_ref)[0], fixed_map(z, best_s), e_z.m]
        cells = [f"{len(m['stencil']):5d} {work(m):6d} {exact_bound(m):8.1e}" for m in maps]
        print(f"{label:5s} {str(z):10s} | " + " | ".join(f"{c:>22s}" if i != 3 else f"{c:>24s}" for i, c in enumerate(cells)) + f"   ({time.time() - t0:.0f} s)")
