# Backward map with a reward for every point

*Draft, 2026-09-29. Proposal for notebook 10. Every number here was computed in this session; see the "What is verified" section at the end for what is and isn't established.*

---

## 1. The story

**Walkers (notebook 6.3).** Walkers start on the initial condition (IC) and move forward in time until they reach the query point $z^* = (x^*, t^*)$. Two problems. They have no sense of direction, and the reward comes only at the end: roughly a thousand moves, then one number. Changing one early move can change the whole outcome, and the agent cannot tell which move mattered.

**Backward map (notebook 8.3).** Start *at* $z^*$ and go backward. Pass 1: choose each point's neighbours and weights, down to the IC and the walls. Pass 2: compute forward from the known data up to $z^*$. This fixed "you have to find $z^*$", because we start there. It did **not** fix the terminal reward: hundreds of choices, one error at the end.

**This formulation.** Keep the backward map exactly as it is, and give **every point its own reward at the moment its stencil is chosen**. Two facts make this possible:

1. The error at $z^*$ splits *exactly* into one piece per point (section 3).
2. Each piece can be estimated from the stencil's **geometry alone**, with no solution values (section 4).

So each decision is graded immediately, and no exact solution is needed to grade it.

---

## 2. Setup (unchanged from 8.3 / 9)

PDE on $x \in [0,1]$, with $u(0,t) = u(1,t) = 0$:

$$
u_t + c\,u_x = \alpha\,u_{xx}, \qquad c = 1,\ \alpha = 0.1 .
$$

Points live on the lattice $(i\,\Delta x,\ n\,\Delta t)$. The IC ($n=0$) and the walls ($i = 0,\ n_x-1$) are **known data**. Every other point $p$ in the map is computed from $k=5$ neighbours:

$$
\hat u(p) = \sum_{q \in \mathcal N(p)} w_{pq}\,\hat u(q).
$$

The weights are the min-norm solution of the three PDE-substituted Taylor rows (notebook 9, Cell 4, with the $u_{xx}$ fix). The stencil is admissible if the rows have rank 3, $\operatorname{cond} \le 10^4$ and $\|w\|_1 \le 2$.

**One change from 8.3: neighbours only from earlier times** ($\Delta t < 0$). Reasons:

- It makes the map a tree with no loops. In notebook 9, loops made $(I-W)$ singular: in 13% of states, the error at $z^*$ was above $10^{-2}$.
- It is required for section 3: a point's influence $\beta_p$ must be final when $p$ is expanded.

---

## 3. The error splits exactly across points

Write $u$ for the exact solution and $\hat u$ for the computed one. The **local error** of $p$'s stencil is what the stencil gets wrong *even if all its neighbours were exact*:

$$
\delta_p = u(p) - \sum_q w_{pq}\, u(q).
$$

Let $e_p = u(p) - \hat u(p)$. Subtracting the two formulas gives

$$
e_p = \delta_p + \sum_q w_{pq}\, e_q , \qquad e_q = 0 \text{ on the IC and walls.}
$$

Unroll this from $z^*$ down to the known data:

$$
\boxed{\ e(z^*) = \sum_{p} \beta_p\, \delta_p\ }
\qquad
\beta_{z^*} = 1, \qquad
\beta_q = \sum_{p \text{ uses } q} \beta_p\, w_{pq}.
$$

$\beta_p$ is the **influence** of $p$ on $z^*$: the sum over all paths from $z^*$ down to $p$ of the product of the weights along the path.

**Why $\beta_p$ is known the moment $p$ is expanded.** Always expand the pending point with the **latest** $t$ (a heap, as in 8.3). Every point that could use $p$ lies at a later time, so it has already been expanded, and $\beta_p$ is complete. This settles the "which point next" question from the handoff: the order is required, not a free design choice.

**Checked numerically** on FD-cone clouds: $\sum_p \beta_p\delta_p$ equals the true error at $z^*$ to $10^{-16}$ for $z^* = (50,5),\ (30,10),\ (50,20)$.

---

## 4. Scoring a stencil from its geometry

### 4.1 Taylor expansion with the PDE substituted

Take a neighbour $q$ at offset $(\Delta x, \Delta t)$ from $p$, and let $\xi = \Delta x - c\,\Delta t$ (the offset seen in the frame moving with the flow). Use the PDE to turn every time derivative into space derivatives (thesis §2.5). The Taylor series of any solution becomes

$$
u(q) = \sum_{n \ge 0} \frac{\partial_x^n u(p)}{n!}\; T_n(q),
$$

where $T_n$ are the **heat polynomials**:

$$
T_0 = 1, \quad
T_1 = \xi, \quad
T_2 = \xi^2 + 2\alpha\Delta t, \quad
T_3 = \xi^3 + 6\alpha\Delta t\,\xi, \quad
T_4 = \xi^4 + 12\alpha\Delta t\,\xi^2 + 12(\alpha\Delta t)^2 .
$$

$T_0, T_1, T_2$ are exactly the three Taylor rows already in `solve_weights`.

### 4.2 What the weights cancel and what is left

Weight the neighbours and add:

$$
\sum_q w_q u(q)
= \underbrace{\Big(\sum w_q\Big)}_{=1} u
+ \underbrace{\Big(\sum w_q T_1\Big)}_{=0} u_x
+ \underbrace{\Big(\sum w_q T_2\Big)}_{=0} \frac{u_{xx}}{2}
+ \Big(\sum w_q T_3\Big) \frac{u_{xxx}}{6}
+ \Big(\sum w_q T_4\Big) \frac{u_{xxxx}}{24}
+ \dots
$$

The three rows force the first three brackets to $1, 0, 0$. So the local error is what survives:

$$
\delta_p \approx -\,m_3(p)\,\frac{u_{xxx}(p)}{6} - m_4(p)\,\frac{u_{xxxx}(p)}{24},
\qquad
m_n(p) = \sum_q w_{pq}\,T_n(q).
$$

- $m_3, m_4$ are **plain numbers from the offsets and weights**. No solution values.
- $u_{xxx}(p)$ is unknown, but it is **the same for every choice of neighbours at $p$**, so when comparing choices at $p$ it drops out.

### 4.3 The score

Replace the derivatives by their typical size $|\partial_x^n u| \sim U / L^n$, where $L$ is the solution's length scale:

$$
\boxed{\ s_p = \frac{|m_3(p)|}{6L^3} + \frac{|m_4(p)|}{24L^4}\ }
\qquad
L = \frac{1}{2\pi} \text{ (shortest mode of the IC).}
$$

$L$ is the one hyperparameter. For a given IC it is read off the data; it is not tuned.

**Example at $p = (50, 20)$:**

| Choice of 5 neighbours | Score $s_p$ (no solution) | True $\lvert\delta_p\rvert$ (only to check) |
|---|---|---|
| 5 in a row, 1 level down | $1.7\times10^{-6}$ | $6.8\times10^{-7}$ |
| 5 in a row, 2 levels down | $1.0\times10^{-5}$ (6× worse) | $3.6\times10^{-6}$ (5× worse) |

The score gets the size roughly right (within a factor of about 3) and the ratio between the two choices almost exactly.

---

## 5. The MDP

### Start
An empty map for a query $z^*$, sampled from a training set of queries:

$$
\beta_{z^*} = 1, \qquad \text{pending} = \{z^*\}, \qquad \text{map} = \varnothing .
$$

### Active point
The pending point with the latest $t$ (section 3). It stays active for 5 steps, one per neighbour pick.

### Action
Pick one cell of a **downward window** around the active point $p$:

$$
\mathcal W = \{(\Delta i, \Delta n) : |\Delta i| \le 2,\ \Delta n \in \{-1, -2\}\} \qquad (10 \text{ cells}).
$$

Five picks make one stencil, as in 8.3. The **mask** is 8.3's `pick_mask`, restricted to $\mathcal W$: no repeats, and after the 5th pick the stencil must be admissible. Cells past a wall or below $t = 0$ are placed on the wall or the IC, and the weights are computed for the *placed* positions.

### State (what the agent sees, no solution values)
For each of the 10 window cells:

$$
\big[\ \text{picked already},\ \ \text{point already exists (free)},\ \ \text{point is IC/wall}\ \big]
$$

plus globals:

$$
\big[\ \log_{10}|\beta_p|,\ \ n_p \ (\text{levels above the IC}),\ \ i_p - i^* \ (\text{offset to } z^*),\ \ \text{distance to each wall (cells, clipped)},\ \ \text{pick number},\ \ \tfrac{\text{points used}}{N_\text{ref}}\ \big].
$$

Everything is in grid units relative to $p$ (thesis §5.3.2, version 2), so one policy serves every $z^*$.

### Transition
After the 5th pick:

1. Store $p$'s stencil $(\mathcal N(p), w)$.
2. For each neighbour $q$ that is not known data: $\beta_q \mathrel{+}= \beta_p\,w_{pq}$, and push $q$ onto pending if it is new.
3. The next active point is the latest-$t$ pending point.

### Reward
Zero for picks 1–4. On the 5th pick:

$$
r_p = -\,\frac{|\beta_p|\, s_p}{B_\text{ref}} \;-\; \lambda\,\frac{n_\text{new}(p)}{N_\text{ref}},
$$

- $n_\text{new}(p)$: points this stencil created.
- $B_\text{ref}, N_\text{ref}$: score total and point count of the reference map for this $z^*$ (every point uses "5 in a row, one level down"). Both are computable without the solution, and they put all $t^*$ on the same scale.

Summed over the episode:

$$
\sum r = -\,\frac{\sum_p |\beta_p|\,s_p}{B_\text{ref}} \;-\; \lambda\,\frac{N}{N_\text{ref}}
$$

In words: an estimated error bound relative to the reference map, plus cost relative to the reference map. The sum uses absolute values, so the agent cannot score well through lucky cancellations.

### Termination
Pending is empty: every branch has landed on the IC or a wall. A safety cap on points (e.g. $3N_\text{ref}$) ends the episode early. Each point left unexpanded then costs $-|\beta_q|\,s_\text{max}/B_\text{ref}$.

### Discount, policy
$\gamma = 1$. MaskablePPO, MLP policy, as in 8.3.

### Where the exact solution appears
**Only in logging:** after each episode, the forward pass gives $\hat u(z^*)$, and we record $|\hat u(z^*) - u(z^*)|$ to check that the proxy is steering correctly. It is never in the state and never in the reward.

---

## 6. Pseudocode

```text
episode(z*):
    beta = {z*: 1};  pending = heap[z*];  stencil = {};  N = 0
    while pending not empty and N < cap:
        p = pending.pop_latest_t()
        picks = []
        repeat 5 times:
            obs = observe(p, picks, beta[p])
            a   = policy(obs, mask = pick_mask(p, picks))
            picks.append(a)                          # reward 0
        nbs, w = placed_points(p, picks), weights(placed offsets)
        stencil[p] = (nbs, w)
        n_new = 0
        for q, w_q in zip(nbs, w):
            if q is IC or wall: continue
            if q not in beta: beta[q] = 0; pending.push(q); n_new += 1
            beta[q] += beta[p] * w_q
        N += n_new
        reward = -|beta[p]| * score(nbs, w) / B_ref  -  lam * n_new / N_ref
    # logging only:
    u_hat = forward pass over stencil, in increasing t
    log |u_hat(z*) - u_true(z*)|
```

---

## 7. What is verified, and what isn't

**Verified in this session:**

| Claim | Evidence |
|---|---|
| $e(z^*) = \sum \beta_p \delta_p$ exactly | matches to $10^{-16}$ at 3 queries |
| Earlier-time-only neighbours remove the blow-ups | notebook 9 loop: worst single-step jump fell from 3–4 decades to 0.3–1.8 |
| Local score tracks local error | 400 random stencils: corr. of $\log$ = 0.75; best-25% vs worst-25% geometries, 16× apart (ran with an earlier $m_4$ weighting, not yet rerun with $L$) |
| **Map total tracks the true error at $z^*$** | 120 random maps per query: corr. of $\log(\text{true err})$ with $\log\sum\lvert\beta\rvert s$ = **0.72** ($t^*=6$), **0.63** ($t^*=10$). Same as the exact $\sum\lvert\beta\delta\rvert$ (0.71, 0.63) |
| Optimising the total helps | best-10%-score maps: true error $4.5\times10^{-6}$ / $1.6\times10^{-5}$; worst-10%: $1.6\times10^{-4}$ / $2.6\times10^{-4}$ |
| There is room above the reference map | reference map: $4.2\times10^{-6}$ (66 pts) at $t^*=6$; best random map: $8.2\times10^{-7}$. FD: $1.5\times10^{-5}$ |

**Not established yet:**

- That PPO learns this. Nothing has been trained.
- That it beats **greedy on the score**: at each point, take the admissible stencil with the smallest $|\beta_p| s_p + \lambda n_\text{new}$. This is the baseline RL has to beat. RL's possible advantage is planning: reusing points across siblings, and how a choice now shapes later $\beta$.
- Behaviour at larger $t^*$ (only $t^* = 6, 10$ checked at map level).

---

## 8. Open choices (decide when building)

1. **Reward scale.** A linear ratio (above) lets very bad maps dominate a batch. Clip or use a log if training is unstable.
2. **Window size.** Start with the 10-cell window. Widen to $|\Delta i| \le 3$ or $\Delta n \ge -3$ only once training works.
3. **$\lambda$.** Price of a point relative to the reference map.
4. **Baselines in the eval loop from the start:** FD, the reference map, greedy-on-score, 8.3's trained policy (on matching $t^*$), F6.

## 9. Later, not now

- **Longer jumps and higher order** (F6 direction): widen the action to jump length and moment count. The same $\beta$, $\delta$ and score machinery applies, using higher $T_n$.
- **Burgers / variable $\alpha(x)$:** the score's $L$ becomes local, estimated from computed values. Only then do solution values enter the state.
