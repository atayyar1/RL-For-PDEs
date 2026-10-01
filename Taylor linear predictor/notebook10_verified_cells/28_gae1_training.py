# ── Same as Cell 27 (pending workload in the state), plus gae_lambda = 1.0  (the setup that made training reliable: 3/3 seeds) ──
SEEDS, TOTAL = [1, 2, 3], 3_000_000
results_gae = {}

for seed in SEEDS:
    print(f"seed {seed}")
    vec   = SubprocVecEnv([make_env_cls(PendingStateEnv, [Z_HARD], LAM_HARD, 100 * seed + i) for i in range(N_ENVS)])
    model = MaskablePPO("MlpPolicy", vec, n_steps=256, batch_size=256, gamma=1.0, gae_lambda=1.0, seed=seed, verbose=0)
    model.learn(total_timesteps=TOTAL, callback=ProgressLog())
    vec.close()
    model.save(f"local_reward_pending_gae1_{Z_HARD[0]}_{Z_HARD[1]}_lam{LAM_HARD:.0f}_seed{seed}_{date.today()}")

    tot, info, e_s = evaluate_cls(PendingStateEnv, model, Z_HARD, LAM_HARD)
    T = np.array(list(terms(e_s.m).values()))
    results_gae[seed] = {"return": tot, "points": info["n_points"], "error": info["error"], "bound": exact_bound(e_s.m),
                         "spread": np.std(np.log10(T)), "wasted": np.mean(T < 0.01 * T.mean())}
    print(f"seed {seed} final: return {tot:.2f}, {info['n_points']} points
")
