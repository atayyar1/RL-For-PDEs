# Learning where to add points: goal-oriented refinement toward a query

## The problem

We want the solution of a time-dependent PDE at **one point** in space and time, the *query* `z* = (x*, t*)`, as accurately as possible while computing as few other points as possible.

Running example: advection–diffusion on `[0, 1]` with zero boundary values,

```
u_t + c u_x = α u_xx,     c = 1,   α = 0.1
```

Points live on a fixed space–time lattice (`Δx`, `Δt` as in the FD reference). The initial condition (`t = 0`) and the walls (`x = 0, 1`) are **known data**.

## The building block (unchanged from the thesis)

Every computed point `p` is a weighted sum of `k` other points of the cloud:

```
û(p) = Σ wᵢ û(zᵢ)
```

- **Neighbours:** the `k` nearest points of the current cloud (KD-tree, normalised distance `x/Δx`, `t/Δt`), in **any direction**, including later times.
- **Weights:** the three PDE-substituted Taylor rows (constant, `u_x`, `u_xx`), solved with the **pseudo-inverse**. With `k = 5` the system is underdetermined on purpose and the weights are min-norm (notebook 1).
- **Admissibility:** a stencil is usable only if the rows have full rank, `cond(A) ≤ 10⁴`, and `‖w‖₁ ≤ w_max`. The last check is the thesis's own §5.10 recommendation: `‖w‖₁` is the per-level amplification factor.

Because neighbours may lie above a point, values are not computed bottom-up. All of them come from **one sparse linear solve**:

```
(I − W) û = b
```

where `W` holds the weights between computed points and `b` collects `weight × value` for known neighbours. On a cloud without loops this gives exactly the old bottom-up values.

## Why a new formulation

| Earlier formulation | What failed |
|---|---|
| Walkers from the IC (thesis ch. 5) | one terminal reward for hundreds of moves; the cloud's shape hidden from the agent; ~80 % of episodes scored a constant; geometric error growth |
| Backward map from the query | terminal-only reward again; no values while deciding (blocks Burgers); episodes grow with `t*`; with all-direction neighbours, maps can close into data-free "islands" |

The related work we checked (Foucart et al. 2023; ASMR; Learning to Discretize; multi-agent Dec-MDP; BPTTS) solves these with **local decisions, one shared policy, and local or dense rewards**. The formulation below follows that pattern.

## The idea

Always have an answer at `z*`, and let the agent **add one point at a time where it improves that answer most**.

## Two fields computed at every step (no exact solution needed)

- **β, influence of each point on the answer.** One extra sparse solve with the same matrix:

  ```
  (I − W)ᵀ β = e_z*
  ```

  `β_p` is how much of `û(z*)` comes from point `p`. It is the discrete adjoint, and for positive weights the chance that a backward random walk from `z*` visits `p`.

- **η, local error indicator.** At each computed point, the disagreement between two admissible stencils (e.g. `k = 5` vs `k = 7`) evaluated on the computed values:

  ```
  η_p = | û_p(k=7) − û_p(k=5) |
  ```

`Σ β_p η_p` estimates the error at `z*`. This is the classical dual-weighted-residual (DWR) estimate.

## The learning problem

| | |
|---|---|
| **Episode** | One query `z*`. Start: known data + `z*` (seed, see open questions). One solve gives a first estimate `û(z*)`. |
| **Step** | Add one lattice point. Recompute the KNN stencils that change (local), re-solve for `û` and `β`, update `η`. |
| **Action** | Which empty lattice cell to add, from the **candidates** (empty cells within a window of an existing point, below a ceiling `t* + H`), plus **stop**. Candidates that would leave any stencil inadmissible are masked. |
| **Observation** | Around each candidate, a local patch of fields: occupancy, `û`, `β`, `η`, `‖w‖₁`; plus the budget used. One shared network scores every candidate (local policy, as in Foucart/ASMR). |
| **Reward** | `r_k = (1/3)[log₁₀ e_k − log₁₀ e_{k+1}] − λ`, where `e_k` is the error at `z*` after `k` points. |
| **Discount** | `γ = 1`. |
| **Termination** | Point budget `B` reached, or the **stop** action. |

**The reward telescopes.** Summed over an episode it is

```
(1/3) log₁₀(e_0 / e_final) − λ · (points added)
```

so the return is exactly "decades of error removed, minus cost", while every step gets credit for its own point. `λ` sets the price of a point in decades of error. Errors are floored (e.g. `10⁻³ e_tol`) so that lucky cancellation cannot earn unbounded reward.

**Where the exact solution is used.** Only at `z*`, only in the training reward. Never in the observation. The solution-free version replaces `e_k` by the DWR estimate `|Σ β η|`.

## What this fixes

| Problem | Here |
|---|---|
| One terminal reward | per-step, telescoping reward |
| No values while deciding | `û`, `β`, `η` available at every step (Burgers: Picard linearisation as in notebook 6.2) |
| Episodes grow with `t*` | episode length = points added, set by the budget |
| Walker selection, hidden cloud | no walkers; the cloud is in the state |
| Islands, failed solves | the cloud always contains known data; masked admissibility keeps the solve well posed |
| Generalisation | local shared policy: train small, deploy on larger `t*`, other `x*`, other ICs |

## What counts as winning

Reported as **work–precision curves** (error at `z*` vs points), over several queries:

- **FD**: the full cone below `z*`.
- **Uniform refinement**: add points in a fixed coarse-to-fine order.
- **Random candidate** (masked).
- **Greedy DWR**: add the candidate next to the largest `β·η`. This is the classical goal-oriented method and **the main baseline**.

The learned policy has to beat greedy DWR in at least one of:

- **Planning:** better error for the same budget (greedy is one step ahead).
- **Cost:** needs no adjoint solve at deployment (ablation without `β` in the observation).
- **Transfer:** trained on toy problems, works on larger `t*`, other ICs, or Burgers.

## Ablations

- **Without `β`** in the observation. `β` *is* the domain of dependence. If the agent still recovers the diffusive `√(ατ)` fan and the upstream lean (the envelope measurement of §5.10), the physics is discovered, not given.
- **Without `η`.**
- **`‖w‖₁` limit on / off**: tests the amplification diagnosis of §5.9.

## Traps

- **Luck.** A single `|e(z*)|` can cancel by chance, and the log reward amplifies that. Floor the error, report `Σ β η` alongside, and evaluate over several queries.
- **Peeking.** `u_true` may appear in the reward during training, never in the observation.
- **Greedy is strong.** Without the greedy-DWR baseline, a good result does not show that RL was needed.

## Where this sits

- **Goal-oriented adaptivity / DWR:** `β` is the adjoint and `Σ β η` the error estimate. The RL policy replaces the greedy marking rule.
- **RL for adaptive mesh refinement** (Foucart et al. 2023; ASMR; Yang et al.): local sequential decisions with a shared policy. Here it is mesh-free (GFDM), in space–time, and aimed at one query instead of a global error.
- **GFDM / meshless stencil selection:** KNN stencils with admissibility checks (thesis ch. 3).
- **Walk-on-spheres / Monte Carlo:** with positive weights, `β` is a visit density of backward random walks. The same cloud could be evaluated by sampling walks instead of solving (Ulam–von Neumann). That is a possible second evaluator.

In one sentence: *start from a one-jump answer at `z*`, and let RL add points where they cut the error at `z*` most, with credit given to each point as it is added.*

## Open questions

- **Seed.** At the start, `z*`'s nearest known points can all lie on one wall, a rank-deficient stencil. Options: (a) take `z*`'s stencil as the nearest *admissible* set of known points; (b) start from a coarse seed lattice below `z*`. Option (a) keeps the start free of hand-made structure.
- **Candidate set and policy shape.** v1: a fixed box around `z*` as the action space, masked to candidates, with an image observation and a CNN. v2: a fully convolutional or per-candidate scorer, so the policy is translation-equivariant and size-independent.
- **Several points per step.** Faster episodes, at the cost of joint-action coordination.
- **Solution-free reward.** Train with `|Σ β η|` instead of the true error, and measure how much is lost.
- **Continuous positions.** Off-lattice points, to make the method truly mesh-free.
- **Burgers.** Stencils need `û`, which is available here; check that the Picard linearisation keeps the solve well posed.
