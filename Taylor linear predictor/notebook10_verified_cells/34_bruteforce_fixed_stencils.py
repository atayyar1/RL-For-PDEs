# ── Brute force: every fixed stencil in the 21-cell window, used everywhere, at z* = (50, 20) ──
from joblib import Parallel, delayed

N_JOBS = 12

Z_BF = (50, 20)
B_fd, N_fd = reference(Z_BF)
interior = admissible_sets(*Z_BF)

def fixed_chunk(chunk, z):
    """For each stencil in the chunk: build the map that uses it wherever it is admissible
    (lowest-score stencil elsewhere, i.e. at edges and near the IC). Returns (stencil, points, score sum)."""
    set_cache = {}
    def allowed(p):
        key = _situation(*p)
        if key not in set_cache:
            set_cache[key] = set(admissible_sets(*p))
        return set_cache[key]
    out = []
    for s in chunk:
        _, B, N = build_map(z, lambda m, p: s if s in allowed(p) else choose_ref(m, p))
        out.append((s, N, B))
    return out

t0 = time.time()
chunks = np.array_split(np.arange(len(interior)), N_JOBS)
res = Parallel(n_jobs=N_JOBS)(delayed(fixed_chunk)([interior[i] for i in idx], Z_BF) for idx in chunks)
rows = [r for part in res for r in part]
print(f"{len(rows)} fixed stencils in {time.time() - t0:.0f} s")

ppo = {1.0: (-1.34, 305, 1.0e-05), 4.0: (-3.93, 323, 1.7e-05)}          # Cell 33 results at (50, 20)
for lam in (1.0, 4.0):
    ret = lambda N, B: -B / B_fd - lam * (N - 1) / N_fd
    best = sorted(rows, key=lambda r: -ret(r[1], r[2]))[:3]
    print(f"\nlambda = {lam}:  PPO return {ppo[lam][0]:.2f} ({ppo[lam][1]} points, bound {ppo[lam][2]:.1e})   FD {-1 - lam * (N_fd - 1) / N_fd:.2f}")
    for s, N, B in best:
        m_s = build_map(Z_BF, lambda m, p, s=s: s if s in set(admissible_sets(*p)) else choose_ref(m, p))[0]
        print(f"   fixed {[WINDOW[a] for a in s]}: return {ret(N, B):.2f}, {N} points, bound {exact_bound(m_s):.1e}")
    beats = sum(ret(N, B) > ppo[lam][0] for _, N, B in rows)
    print(f"   fixed stencils that beat PPO: {beats} of {len(rows)}")
