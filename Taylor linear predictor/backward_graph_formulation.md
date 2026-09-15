# Learning where to compute: the backward-graph formulation

## The problem

We want the solution of a time-dependent PDE at **one point** in space and time, the *query* `z* = (x*, t*)`, as accurately as possible while computing as few other points as possible.

The running example is advection–diffusion on `[0, 1]` with zero boundary values:

```
u_t + c u_x = α u_xx,     c = 1,   α = 0.1
```

Every point we are allowed to compute lives on a fixed space–time grid. The initial condition (`t = 0`) and the boundaries (`x = 0, 1`) are known for free.

## The building block

A point's value is predicted as a weighted sum of a few earlier points:

```
u(point) ≈ w₁·u₁ + w₂·u₂ + w₃·u₃
```

The weights come from a Taylor expansion with the PDE substituted in (the local linear predictor / GFDM idea). Once the neighbours are chosen the weights are fixed. They depend only on *where* the neighbours are, not on any values. So **the only real choice in the whole method is which neighbours each point uses.** That choice is the geometry of the scheme, and it is what we want to learn.

## Where it started, and why it stalled

The first formulation sent *walkers* forward from the initial condition. They moved one step at a time, left behind a cloud of computed points, and the agent was rewarded once, at the end, for accuracy at the query and total cost.

It never learned accuracy, for three reasons that turned out to be structural:

- **Credit assignment.** Hundreds of moves shared one reward. The agent could not tell which move had hurt, and one bad move could spoil ninety-nine good ones.
- **Partial observability.** What decided success was the shape of the cloud, which the agent could not see. It only saw walker positions.
- **Waste.** Walkers wandered, and much of what they computed never influenced the query at all.

It learned to reach the query quickly, but not to reach it accurately, and at long horizons it learned nothing.

## The turn: start from the answer and work backward

Instead of walking forward and hoping to arrive, start **at the query** and ask: *which earlier points do I need?*

1. The query picks a stencil: three earlier points.
2. Each of those points needs a stencil of its own, so each picks one.
3. Keep going down. A branch stops when it lands on the initial condition or a boundary, where the value is known.
4. When every branch has stopped, the **map** is complete.
5. Now compute **forward**: known values at the bottom, then level by level upward, each point a weighted sum of its neighbours, until you reach the query.

Two things make this cheap. Points shared by several stencils are computed once, so **reuse is free**. And every point in the map exists because something above it needs it, so **nothing is wasted**.

Building the map needs only geometry. Values are plugged in afterwards, in one pass.

## Influence and blame

Two numbers per point carry the whole idea.

- **β, influence.** How much of the query's answer comes from this point. If `z* = 0.5·A + 0.5·B` and `A = 0.3·C + 0.7·D`, then `D` contributes `0.35` of the answer. β starts at 1 at the query and is passed down the map, each point handing its share to its neighbours in proportion to the weights.
- **τ, own error.** The mistake this point's stencil would make even if its neighbours were perfect.

The error at the query splits exactly:

```
error(z*) = Σ over all points of  β · τ
```

Every point's contribution to the final error, its **blame**, is `β · τ`, and it is known at the moment that point's stencil is chosen. β is already complete, because the map is built from the top down, and τ follows from the choice.

That single identity removes the credit-assignment problem. A bad choice at an important point is penalised, and penalised immediately.

## The learning problem

| | |
|---|---|
| **Episode** | Build the map for one query. It always finishes; there is no "reaching". |
| **Step** | Choose the stencil for one point, taking points latest time first. |
| **Action** | A stencil shape from a menu: how far back, how wide, how far to lean sideways. Shapes that leave the grid or give negative weights are masked out. |
| **Observation** | Geometry only: the point's position relative to the query and the domain, its influence β, and for each action how many *new* points it would create (which choices reuse). **Never the exact solution.** |
| **Reward** | `−|β·τ|` in units of FD's error, minus the new points created in units of FD's cost. Charged right after each choice. |
| **Discount** | `γ = 1`, so the return is exactly the objective. |

No shaping is needed. The reward terms are the objective itself, split into per-step pieces. With the absolute value, the return is an upper bound on the error that gives no credit for errors cancelling by luck.

Why negative weights are masked: with positive weights summing to one, a point is an average of its neighbours and cannot amplify their errors. A negative weight lets errors grow level after level, which is the amplification that ruined the walker runs.

## What counts as winning

- **FD** lives inside this action space: always pick the tightest stencil one level down. It is exact and expensive.
- **A single fixed stencil everywhere**, for each shape in the menu, traces what "no adaptivity" can reach.
- **Random allowed stencils** are the floor.
- **Greedy** picks, at each point, the stencil with the smallest blame plus a price for the future work it creates. It computes τ from the exact solution, so it is a reference, not a solver. It shows how much per-point choice can buy.

The learned policy should beat FD and every fixed stencil, and approach greedy **without ever seeing the exact solution**. That last part is why learning is needed at all: a policy maps local geometry to a stencil, so it can run where no exact solution exists.

## Two traps to remember

- **Luck.** At a single query, point errors can cancel almost perfectly and make a mediocre stencil look spectacular. Judge rules on `Σ|β·τ|`, not only on `|error|`, and across several queries.
- **Peeking.** Anything that uses the exact solution to *choose*, as greedy does, is an oracle. The exact solution may appear in the reward during training, never in the observation.

## Where this sits

- **Monte Carlo PDE solvers (walk-on-spheres).** With positive weights summing to one, the weights are probabilities. β is the chance a backward random walk from the query passes through a point, and the answer is the walk's average. Walk-on-spheres estimates that average by sampling; the map computes it exactly. Walk-on-spheres has sampling noise and no discretisation error; the map has the reverse. Both evaluate a single point without solving everywhere. Related: the Markov chain approximation method (positive FD weights as transition probabilities), and learned importance sampling for these walks.
- **Goal-oriented error estimation.** Weighting local errors by their influence on one quantity of interest is the dual-weighted residual idea; β is a discrete adjoint.
- **RL for discretisation.** *Learning to Discretize* casts a PDE solver as an MDP but learns stencil *weights* over the whole field. Here the weights are fixed by consistency and the *placement* is learned for one query.
- **RL for mesh refinement.** Swarm RL for adaptive meshes uses local, per-element rewards over a global mesh, close in spirit to per-point blame.

In one sentence: *a lattice analogue of walk-on-spheres that computes the walk's average exactly instead of sampling it, with the stencil choice learned for accuracy and cost.*

## Open questions

- **The action menu is restrictive.** Templates force the three neighbours into a row at one level. The natural next step is to let the agent place each neighbour anywhere in a window below the point: mixed depths, asymmetric shapes, and 4–5-point stencils.
- **The mask is not physics-free.** Which shapes give positive weights depends on `α` and `c`, so part of the physics (width growing with depth, an upstream lean) is visible in the mask. An ablation with the mask off separates what is discovered from what is given.
- **The reward needs the exact solution.** Training uses problems with known solutions. A learned error estimate would remove that dependency.
- **Nonlinear equations.** For Burgers, the advection speed depends on `u`, which is unknown while building the map backward. Stencil choice would need a predicted value.
- **Generalisation.** A single query can be solved by search. Learning earns its place by transferring across query locations and horizons, and by planning ahead where greedy only looks one point ahead, which is what the critic is for.
