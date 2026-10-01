# Where we are: notebook 10, step by step

*2026-09-30. Plain-language explanation of everything in notebook 10, what we ran, and what is going on now.*

---

## 1. The goal in one sentence

We want the solution of the PDE at **one point** $z^* = (x^*, t^*)$, as accurately as possible, while computing **as few other points as possible**. The RL agent decides *which* points to compute.

---

## 2. The pieces

**Grid.** Every point we can use sits on a fixed space-time grid: position index `ix` (0 to 99) and time index `it`. One step in `it` is one time level.

**Query $z^*$.** The point we want. In our tests, $z^* = (50, 10)$ or $(50, 20)$: middle of the domain, 10 or 20 time levels up.

**Known data.** The initial condition (level `it = 0`) and the two walls (`ix = 0` and `ix = 99`). Their values are given for free.

**Stencil.** To compute a point $p$, we take **5 neighbours** from the two levels just below it (a window of 10 cells: 5 cells one level down, 5 cells two levels down). The agent chooses which 5.

**Weights.** Given the 5 neighbours, the value at $p$ is a weighted sum:

$$
\hat u(p) = w_1 \hat u(q_1) + \dots + w_5 \hat u(q_5).
$$

The weights come from the thesis's Taylor/PDE rows (chapter 2), the same method as always. The agent never chooses weights, only neighbours.

**Admissible.** Some choices of 5 neighbours give bad weights (rank-deficient, ill-conditioned, or $\|w\|_1 > 2$). Those are forbidden. The **mask** only allows picks that can still end in an admissible stencil.

**Map.** The set of all computed points and their stencils. It is built **backward**: start at $z^*$, choose its stencil, then choose a stencil for each of its neighbours, and so on, until every branch reaches the initial condition or a wall.

**Forward pass.** Once the map is finished, compute the values from the bottom up: known data → level 1 → level 2 → … → $z^*$. This gives $\hat u(z^*)$.

---

## 3. How good is a map? Three quantities per point

**Influence $\beta_p$.** How much the value at $p$ affects the answer at $z^*$.
- $z^*$ has influence 1.
- When $p$ gets its stencil, each neighbour $q$ receives $\beta_p \times w_q$.
- A point used by many points with large weights has a large influence.

**Local error $\delta_p$.** How wrong $p$'s stencil would be *even if its 5 neighbours were exact*. Computing it needs the exact solution, so we only use it for checking.

**The key fact (exact, checked to 16 digits):**

$$
\text{error at } z^* = \sum_{\text{all points } p} \beta_p \, \delta_p .
$$

The total error is a sum of one piece per point. So each stencil choice can be graded on its own piece.

**Score $s_p$.** An *estimate* of $\delta_p$ that does **not** need the exact solution. It measures how much of the next Taylor terms the 5 weighted neighbours fail to cancel, using only their positions and weights. It tracks the true local error reasonably well (correlation about 0.7), but not perfectly.

---

## 4. The RL game

**Step.** The agent picks one cell from the window. Five steps make one stencil.

**State** (what the agent sees before each pick):
- for each of the 10 cells: already picked? already an existing point? known data?
- for each cell: how good a stencil can still be finished through it (added in Cell 16)
- the current point's influence $\beta_p$, its height above the initial condition, its offset from $z^*$
- how many points are used so far
- *(Cell 26 adds: how many points are still waiting, and their total influence)*

It **never** contains solution values.

**Reward.** Zero for picks 1–4. When the 5th pick completes a stencil:

$$
r_p = -\frac{\beta_p\, s_p}{B_\text{ref}} \;-\; \lambda\,\frac{\text{new points created}}{N_\text{ref}} .
$$

- **First term:** this point's share of the estimated error.
- **Second term:** the price of the new points this stencil forces us to compute.
- $B_\text{ref}$ and $N_\text{ref}$ are the score and point count of the reference map (below). They put every query on the same scale.

**λ (lambda).** How expensive a point is compared with accuracy. Small λ: accuracy matters most. Large λ: fewer points matters most.

**Episode.** One whole map, from $z^*$ down to the known data. The **return** is the sum of all rewards in the episode:

$$
\text{return} = -\,(\text{score of the map, relative to reference}) \;-\; \lambda \times (\text{points, relative to reference}).
$$

Higher (closer to 0) is better.

**Policy.** A small neural network: state in, probabilities over the 10 cells out.

**PPO.** The training algorithm. It plays many episodes, sees which picks led to better-than-expected return, and makes those picks more likely. It also trains a second network, the **value network**, which predicts "how much return is still to come from this state". PPO uses that prediction to judge each pick.

**Seed.** The random number that fixes the network's starting weights and the random choices during training. Different seeds = independent training runs of the same method.

---

## 5. What we compare against (no learning)

| Name | Rule at every point | Looks ahead? |
|---|---|---|
| **FD** | the classical finite-difference scheme | — |
| **Reference** | the stencil with the lowest score | no |
| **Greedy** | the stencil with the best immediate reward (score + λ·new points) | one step |
| **Fewest points** | the stencil creating the fewest new points | one step |

None of them plans ahead. RL's job is to beat them by planning: making choices whose payoff comes later.

---

## 6. What we ran and what we learned

| # | Setting | Result | What it taught us |
|---|---|---|---|
| 1 | $z^*=(50,10)$, λ=1, PPO from scratch | return −3.45 vs reference −1.99. PPO built **exactly FD's footprint** (100 points) with a 2-level stencil, 18× more accurate than FD. | PPO finds a clean structure, but not the best one. |
| 2 | same + entropy bonus; same + score in state | identical −3.45 | Not an exploration or information problem. |
| 3 | mixing policies (no training) | the return gets *worse* before it gets better | There is a **dip** between the two solutions. PPO can't cross it. |
| 4 | warm start (copy the reference, then PPO) | holds about −2.0, doesn't improve | Starting in the right place works, but nothing better was found at λ=1. |
| 5 | λ sweep (no training) | at λ=4–8 no single rule wins; greedy gets worse | This is the regime where planning should matter. |
| 6 | $z^*=(50,20)$, λ=8, PPO, **seed 0** | **−8.29, beats every rule**; 363 points (fewer than FD's 400); error bound 8× below FD's error; error spread evenly across points | PPO *can* find a better map than all the rules. |
| 7 | same, seeds 1–5 | all **worse** than every rule (585–766 points) | Seed 0 was not reproducible: 1 of 6. |
| 8 | seed 1 again with "pending points" in the state | still bad (680 points) | That fix alone doesn't help. |

---

## 7. The problem right now, in plain words

Look at how the failed runs behave during training (seed 3):

| training steps | points in the map | score | return |
|---|---|---|---|
| 0.3M – 1.0M | about 405 | about 2.9× | about **−9.2** |
| 1.6M – 3.0M | about 768 | about 2.6× | about **−14.4** |

PPO first finds a decent map, then **slowly makes it worse**: it nearly doubles the number of points to get a slightly better score. By its own reward that is a bad trade, yet it keeps making it.

**Why this can happen.** When a stencil reaches wide, it creates new points. Their cost comes in two parts:
- **now:** λ per new point. PPO sees this.
- **much later:** every new point needs its own stencil and may create more points. That cost arrives hundreds of steps later.

PPO judges each pick by looking only about 20 steps ahead, and trusts the value network's prediction for everything after. If the value network underestimates the later cost, every wide stencil looks slightly better than it really is, and the policy drifts toward bigger and bigger maps. That matches the curves: slow, one-directional, over a million steps.

**What we tried:** showing the agent how much work is still pending (Cell 26), so the value network can predict the later cost. Seed 1 still drifted.

**What to try next:** stop relying on the value network's guess. With `gae_lambda=1.0`, PPO judges each pick by the *actual* return of the rest of the episode. That is noisier but honest about late costs. It's a one-argument change.

---

## 8. What is solid, and what is not

**Solid:**
- The engine: map, weights, influence, the exact error identity, the score, the mask.
- The heuristic maps are 10–14× more accurate than FD at similar cost, with no learning.
- A better map than every rule **exists and is reachable** (seed 0: 363 points, error spread evenly across points).

**Not solid yet:**
- PPO reaching it **reliably**. Currently 1 of 6 runs.
- Anything at other queries, other λ, or longer horizons.

**The decision in front of us:** fix the drift (try `gae_lambda=1.0`) before scaling to longer horizons such as $t^* = 30$–$40$, where planning has the most room.
