# ── Restore: definitions from Cells 12, 14, 17, 20, 21, 25 without re-running any training ──
import time
from matplotlib.colors import LogNorm

class EpisodeLog(BaseCallback):
    """Keeps the info of every finished episode."""
    def __init__(self):
        super().__init__()
        self.episodes = []
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.episodes.append(info)
        return True

class ProgressLog(EpisodeLog):
    """EpisodeLog that also prints a status line every `every` steps."""
    def __init__(self, every=100_000):
        super().__init__()
        self.every, self.next_print, self.t0 = every, every, time.time()
    def _on_step(self):
        super()._on_step()
        if self.num_timesteps >= self.next_print:
            recent = self.episodes[-20:]
            if recent:
                print(f"   {self.num_timesteps / 1e6:4.1f}M steps | {len(self.episodes):4d} episodes | "
                      f"last 20: {np.mean([x['n_points'] for x in recent]):5.0f} pts, score {np.mean([x['score_ratio'] for x in recent]):5.2f}x "
                      f"| {time.time() - self.t0:4.0f} s")
            self.next_print += self.every
        return True

Z_TRAIN, LAM        = (50, 10), 1.0
Z_HARD,  LAM_HARD   = (50, 20), 8.0
N_ENVS              = 12

def terms(m):
    """Each computed point's share of the score sum, relative to the reference."""
    B_ref, _ = reference(m["z_star"])
    return {p: abs(m["beta"][p]) * score(*p, nbs, w) / B_ref for p, (nbs, w) in m["stencil"].items()}

def exact_bound(m):
    """sum over points of |beta_p * delta_p|: the true error with no cancellation allowed (u_true, checking only)."""
    U = lambda q: u_true(q[0] * dx, q[1] * dt)[0]
    return sum(abs(m["beta"][p] * (U(p) - sum(wq * U(q) for q, wq in zip(nbs, w)))) for p, (nbs, w) in m["stencil"].items())

choose_min = lambda m, p: min(admissible_sets(*p), key=lambda s: (n_new_points(m, p, s), s_of(p, s)))

# returns recorded so far at z* = (50, 20), lam = 8 (Cells 20, 23, 25, 27)
results         = {s: {"return": r} for s, r in {0: -8.29, 1: -11.59, 2: -15.04, 3: -14.39, 4: -13.49, 5: -14.36}.items()}
results_pending = {s: {"return": r} for s, r in {1: -13.52, 2: -8.23, 3: -14.05}.items()}
print("restored")
