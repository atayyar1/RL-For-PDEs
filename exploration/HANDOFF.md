# Hand-off — what changed in the repo, and what it means for the thesis

For Ali. This branch (`jo-exploration`) touches 15 of your files. Nothing is merged; review and
take what you want. Everything below is measured and the scripts are in `exploration/threads/`.

---

## 1. One bug, and it is worth fixing before any more runs

The third row of the constraint matrix in `solve_weights` was

```python
(0.5 * dx_i**2 + alpha * dt_i) / h**2          # was
(0.5 * (dx_i - c_local*dt_i)**2 + alpha*dt_i) / h**2   # is
```

Substituting `u_t = αu_xx − cu_x` **recursively** also reduces `u_xt` and `u_tt` to x-derivatives,
which contributes `−c·ΔxΔt` and `½c²Δt²`. The clean form is **½ξ² + αΔt with ξ = Δx − cΔt** — row 2
is already ξ, so row 3 is just the diffusive moment condition in the characteristic frame. Verified
symbolically, and independently by three other routes.

**What it costs you and what it buys you.** At your benchmark (ν = 0.045) results move by <0.2%, so
the diffusion numbers in the thesis stand. But with the old row the 3-point centred stencil is
FTCS-central, which is *unconditionally unstable*; with the correction it is **exactly
Lax–Wendroff**. The old row could not express a second-order advection scheme at all. So:

- **Any Péclet sweep needs rerunning.** So does the Burgers work, where `c_local = mean(u_nb)` makes
  the dropped term state-dependent.
- Your framework is *more* expressive than the thesis claims — it recovers Lax–Wendroff, which is a
  result worth stating.

Also added: a partition-of-unity guard (`abs(w.sum()-1) > 1e-6 → reject`). `lstsq` was silently
returning `Σw = 0.478` on rank-deficient stencils. The corrected row happens to remove that
degeneracy too, but keep the guard — it is free.

## 2. Three things that bear directly on the RL design

**(a) Do not use `(m, k)` as the action space.** There is no interior optimum: reward rises
monotonically with width until a physical cap, and 15 of 18 optimal actions sit *on* the cap. An
agent will learn "pick the biggest m" and nothing else. The one genuine interior optimum is the
safety factor, **s\* ≈ √(2 ln(1/ε))** — that is the meaningful thing to ask a policy for.

**(b) One-sided stencils are never positive-feasible.** 0 out of 15,501, exhaustively, both sides.
Downwind fails always; upwind fails unless the window exceeds 2/Pe ≈ 20 cells. This matters for your
reward: the time penalty pushes the policy toward sprinting one-sided at the target, and such a
policy is not merely inaccurate — it is **categorically outside the feasible set**. Worth checking
whether your late-training trajectories are doing this.

**(c) The positivity certificate is cheaper than what you already run.** Feasibility has an exact
closed form — the origin must lie in the convex hull of the planar points (row₂ᵢ, row₃ᵢ), an
`atan2`-and-sort test. 0 disagreements with the LP in 80,000 stencils, ~50× faster than an LP and
**cheaper than the `lstsq` it replaces** (16.9 µs vs 17.8 µs). It is a drop-in action mask. And
`lstsq` throws away an available positivity certificate **47.9% of the time**.

Caveat that must travel with it: **the certificate is necessary, not sufficient.** At its own
feasibility boundary the only admissible stencil is the extremal two-point measure — consistent,
positive, stable, and up to 10¹¹ times less accurate than a stencil backed off that boundary. Pair
it with s ≥ s\*, never use it alone.

## 3. Two things in the environment that will be asked about

**`causal_moves` defaults to `False`** and `make_env` never passes it, so every intermediate solve
searches all visited points rather than past ones. The paper draft says the opposite ("every
intermediate solve during the rollout uses causal neighbours only"). In practice the nearest points
are usually below, so it may rarely bite — but a reviewer who reads the code will find it. Set it
`True`, rerun, and see whether anything moves.

**The terminal solve is interpolation, not prediction.** Walkers may rise to `t* + R·Δt` and the
star solve is `causal=False`, so the agent is scored on *surrounding* z\*, not on forecasting it.
That is the most likely explanation for reach → 1.00 while error degrades: filling a box around a
point makes interpolation trivially easy. Make it a switch and report both; the causal version is
the scientifically interesting one and much harder.

Related: `err_norm` clips after three decades, so once saturated the only live gradient is the time
penalty — which selects exactly the sprint geometry that (b) says is infeasible.

## 4. A stronger spine for the workshop paper

The current framing is "RL discovers non-trivial placement strategies at lower cost". That invites
the reviewer question *why learn this at all*, and the honest answer is better than the claim:

> **Godunov's theorem says no linear monotone scheme can be high-order.** The classical escape is
> to abandon linearity — flux limiters, ENO, WENO: schemes that choose their stencil from the local
> solution. A learned, state-dependent selector is exactly such a scheme, general rather than
> hand-designed. **The order barrier is the reason to learn, not a decoration on it.**

And for the parabolic half, which Godunov does not cover, the argument is different but also
available: positivity constrains **geometry**, one-sided geometries are excluded outright, and the
feasible boundary's location depends on the local cell Péclet number. **A fixed rule cannot track a
boundary that moves with the solution.** That is a defensible reason for learning that does not
depend on beating anyone's wall-clock.

Two scopes that should be on the claim rather than in a limitations section, if you use any of the
constant-coefficient speedup numbers from this branch: they are **periodic** (no boundaries), and
the variable-coefficient method needs **L_α = α/|α′| ≳ 6 cells**.

## 5. What we looked into that did not pan out

Recorded so you do not spend time on them. The wide-stencil frontier is **exactly first-order
Runge–Kutta–Chebyshev** (agreement 2e-16) — no novelty there. Jet-recovery-by-moment-regression is
**exactly moving least squares** (5e-16). Adjoint-weighted effort allocation is **dual-weighted
residual**. Phase 1 hit ten classical results in one session; `REEVALUATION.md` has the list and the
filter that would have predicted it. Read `CHECKS.md` before trusting any number in this tree,
including ours — three of our own confident results were wrong and were caught only by cross-checking.

## Where things are

`core/` is a small shared library (`Problem`, `rows_taylor`, `rows_moment`, `solve_maxent`,
`positive_feasible`, `k_max`, `symbol_max`) with 13 tests, each locking a numbered finding; run
`python tests/test_core.py`. `FINDINGS.md` is F1–F42 with every retraction in place.
`threads/T*/RESULTS.md` and `threads/M*/` carry the detail, each with its scope limits stated first.
