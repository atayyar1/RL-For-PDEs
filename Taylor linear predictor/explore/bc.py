"""Warm start: clone greedy + myopic stop into the PPO policy (and its value head), then fine-tune with PPO.
    python explore/bc.py <run_name> <ppo_steps> <queries: "25,45" or "multi"> [norm]
The teacher's stencil is fed as picks in ascending cell order (the mask allows any order). Value target = the
return-to-go of the teacher's episode (gamma = 1), so PPO does not start with a random critic.
"""
import torch as th
from rl_stop import *
from evaluate import greedy_stop


def teacher_episode(z, norm):
    env = StopEnv([z], norm=norm)
    obs, _ = env.reset(seed=0)
    rule = greedy_stop(1.0)
    O, M, A, Rw = [], [], [], []
    done = False
    while not done:
        p = next_point(env.m)
        nbs = rule(env.m, p)
        if nbs is None:
            acts = [STOP]
        else:
            sets, _, _ = allowed_sets(env.m, p)
            acts = sorted(next(s for s in sets if [place(p, k) for k in s] == nbs))     # the teacher's cells
        for a in acts:
            O.append(obs); M.append(env.action_masks()); A.append(a)
            obs, r, done, _, info = env.step(a)
            Rw.append(r)
            if done:
                break
    G = np.cumsum(Rw[::-1])[::-1]                       # return-to-go
    return np.array(O), np.array(M), np.array(A), G


if __name__ == "__main__":
    name, steps, qarg = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
    norm = sys.argv[4] if len(sys.argv) > 4 else None
    queries = Q_TRAIN_MULTI if qarg == "multi" else [tuple(int(v) for v in qarg.split(","))]
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(dict(steps=steps, queries=queries, norm=norm, bc=True), open(os.path.join(run_dir, "config.json"), "w"))
    data = [teacher_episode(z, norm) for z in queries]
    O, M, A, G = (np.concatenate([d[i] for d in data]) for i in range(4))
    print(f"teacher: {len(A)} picks from {len(queries)} queries, {np.mean(A == STOP) * len(A):.0f} stops", flush=True)

    vec = SubprocVecEnv([make_env(queries, 100 + i, norm) for i in range(4)])
    model = MaskablePPO("MlpPolicy", vec, seed=1, verbose=0, device="cpu", n_steps=1024, batch_size=1024,
                        gamma=1.0, gae_lambda=0.97, learning_rate=1e-4, ent_coef=0.0)
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=1e-3)
    O_t, M_t, A_t, G_t = th.tensor(O), th.tensor(M), th.tensor(A), th.tensor(G, dtype=th.float32)
    for ep in range(300):                               # behaviour cloning + value regression
        idx = th.randperm(len(A_t))
        for b in range(0, len(idx), 1024):
            j = idx[b:b + 1024]
            dist = pol.get_distribution(O_t[j], action_masks=M_t[j].numpy())
            v = pol.predict_values(O_t[j]).squeeze(-1)
            loss = -dist.log_prob(A_t[j]).mean() + 0.5 * ((v - G_t[j]) ** 2).mean() / (G_t.abs().mean() ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 299:
            with th.no_grad():
                acc = (pol.get_distribution(O_t, action_masks=M_t.numpy()).distribution.probs.argmax(1) == A_t).float().mean()
            print(f"BC epoch {ep}: loss {loss.item():.3f}, pick accuracy {acc:.3f}", flush=True)
    r = report_s(run_policy(model, queries[0], norm))
    print(f"after BC: {r['points']} points, {r['stops']} stops, error {r['error']:.2e}, bound {r['bound']:.2e}", flush=True)
    hist = [dict(step=0, **r)]
    t0 = time.time()
    for chunk in range(10 if steps else 0):
        model.learn(steps // 10, reset_num_timesteps=False)
        r = report_s(run_policy(model, queries[0], norm))
        hist.append(dict(step=model.num_timesteps, **r))
        print(f"{model.num_timesteps:>9,} steps ({(time.time() - t0) / 60:.0f} min): {r['points']} points, "
              f"{r['stops']} stops, error {r['error']:.2e}, bound {r['bound']:.2e}", flush=True)
        json.dump(hist, open(os.path.join(run_dir, "curve.json"), "w"), default=float)
    model.save(os.path.join(run_dir, "model"))
    vec.close()
