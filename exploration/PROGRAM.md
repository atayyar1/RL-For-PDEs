# Do We Still Need Differential Equations?

**A research program, shaped as a book.**
Joseph Bakarji · opened 2026-09-11 · branch `jo-exploration`

---

## The question

Differential equations buy three things: **locality** (a rule that refers only to a
neighbourhood), **generalization** (the same rule works everywhere and for all time), and
**superposition** (for linear ones, solutions compose). They cost three things: they demand
smoothness, they must be *discovered* through estimated derivatives, and they must be
*solved* — and solving means discretizing, which means throwing the continuum away again.

The seed observation (voice memo, 2025-08-19): most equations we seek to discover will be
solved on a machine anyway. The continuum is a detour. So ask directly for the thing the
machine actually runs — **an integrator** — and ask what it must satisfy.

The reframing that organizes the whole program:

> A numerical scheme is a pair **(geometry G, weights w)**. Consistency determines **w**
> given **G**. So the entire free content of a scheme is its **geometry** — and geometry is
> exactly what an agent can choose.

## Status legend

`[P]` proved · `[M]` measured, reproducible · `[O]` open · `[X]` tried and failed

---

## Part I — The scheme is the object

The discrete is primary; the continuum is the convenient limit. A predictor
û(z\*) = Σᵢ wᵢ u(zᵢ) over scattered space-time neighbours is the general form. Classical
finite differences are the special case where someone chose G in advance.

- `[M]` The framework reproduces FTCS and 4th-order FD exactly (Ali's Exp. 3).
- `[P]` With the PDE substituted *recursively*, the consistency rows are simply
  **1, ξ, ½ξ² + αΔt** where ξ = Δx − cΔt is the characteristic offset. In the frame moving
  at speed c the equation is pure diffusion, so the rows are the diffusive moment conditions
  in ξ. (**F8** — the original code dropped two terms; corrected across the repo.)
- `[M]` With the corrected row the 3-point centred stencil is exactly Lax–Wendroff; with
  the old row it was FTCS-central, unconditionally unstable. So the old row could not
  express a second-order advection scheme at all.

## Part II — Stability is positivity

- `[P]` Σwᵢ = 1 always, so **‖w‖₁ = 1 ⟺ all wᵢ ≥ 0 ⟺ convex combination ⟺ discrete
  maximum principle**.
- `[P]` Under recursion e\* = Σwᵢeᵢ + τ, so ‖w‖₁ = 1 gives **linear** error accumulation and
  ‖w‖₁ = 1+δ gives **geometric**. ‖w‖₁ is the amplification factor.
- `[M]` The classical CFL conditions are recovered *as positivity boundaries* to 6+ digits:
  r ≤ ½ (diffusion), Courant ≤ 1 (upwind), cell-Pe ≤ 2 (centred advection–diffusion).
  **The CFL condition was always a convex-combination condition.**
- `[M]` Positivity feasibility has an exact closed form — the origin must lie in the convex
  hull of the planar points (row₂ᵢ, row₃ᵢ). An `atan2`-and-sort test, 0 disagreements in
  80,000 stencils, **50–60× faster than the LP and cheaper than the `lstsq` it replaces**.
- `[M]` `lstsq` discards an available positivity certificate **47.9% of the time** on
  stencils that *are* positive-feasible.

## Part III — What positivity actually costs

**Retracted**: this part previously argued a trilemma — locality, positivity, expressiveness,
pick two — from a Jensen obstruction. The inequality is true; the consequence is false. See F10.
Under a PDE constraint u_tt is not independent, so the obstructed moment is redistributed onto
x-derivatives the other moments cancel. **FTCS at r = 1/6 has weights (1/6, 2/3, 1/6) ≥ 0 and is
fourth-order.** Verified independently by measured convergence.

What survives, and it is sharper:

- `[P]` **Hyperbolic: positivity costs exactly one order.** Godunov's theorem, and the mechanism
  is visible in this language: with α = 0 the second moment is q²/2 ≥ 0, so cancelling it with
  w ≥ 0 forces every active neighbour exactly onto the characteristic. Measured: upwind 1.00,
  Lax–Wendroff 2.00.
- `[M]` **Parabolic: positivity costs no order at all.** At matched moment conditions the rates
  are identical, and at order 3 the positive stencil is 2.6× *more* accurate. What it costs is
  feasibility: under Δt ∼ Δx the max positive order collapses 3 → 2 → 1 as r grows.
- `[M]` **The real constraint is on geometry.** One-sided stencils are never positive-feasible
  (0/15 501). The feasible depth saturates at 2α/(c²Δt) regardless of width. The frontier has
  three branches, `min(m²/2r, m/ν, 2α/c²Δt)`, and the binding one changes with cell Péclet.
- `[P]` **Bolley–Crouzeix (1978)**, checked against the literature: positivity preservation
  requires *either* time order ≤ 1 *or* a stability condition tying Δt to the spatial
  discretization. Unconditionally positive ⇒ first order; conditionally positive ⇒ no barrier.
  Our r = 1/6 example is on the second branch, so it is consistent rather than contradictory.

> **The thing positivity constrains is which stencils exist — and that is exactly what an agent
> chooses.** A policy that sprints one-sided toward its target is not merely inaccurate; it is
> outside the feasible set. That is a better foundation for the programme than the barrier it
> replaces, because it bears directly on the decision being learned.

## Part IV — Why learning is forced

- For **hyperbolic** problems no linear scheme is both monotone and high-order. The classical
  escape is to abandon linearity: flux limiters, ENO/WENO — schemes that **choose their stencil
  from the local solution**. A learned state-dependent selector is exactly such a scheme.
- For **parabolic** problems the argument is different, since positivity costs no order there.
  What it costs is feasibility, and the feasible set is *shaped*: one-sided geometries excluded,
  depth saturating, the boundary's location set by local cell Péclet. **A fixed rule cannot track
  a boundary that moves with the solution.** That, not an order barrier, is the case for learning.
- `[P]` Positivity also makes *planning* tractable: with ‖w‖₁ = 1 errors are additive, so the
  optimal scheme schedule is an exact shortest-path problem. With ‖w‖₁ > 1 it is
  history-dependent and search is genuinely required. **That is the honest boundary between
  when you can plan and when you must learn.**
- `[X]` RL over global scheme selection (m, k) is *not* needed: 15/18 optimal actions sit on
  a boundary, so the optimal policy is trivial. Learning must act where the choice is
  genuinely local and state-dependent.

## Part V — Discovery without derivatives

- `[P]` For u_t = Lu the propagator exp(τL) has moments obeying a **closed triangular ODE
  hierarchy driven by the PDE coefficients alone**:
  `dM_q/dτ = α q(q−1) M_{q−2} + c q M_{q−1}`. Verified to 1.6e-9 through q = 8. No
  fundamental solution — so it generalizes to any linear PDE and locally to variable and
  state-dependent coefficients.
- `[M]` Matching moments q = 0…p with w ≥ 0 turns a *query* at (x\*, t\*) into one
  convolution. At t\* = 400 CFL steps: cost 153 vs FD's 69,898, error 1.3e-7 vs 2.6e-5.
- `[P]` The Taylor rows are the p = 2 case. **The moment formulation subsumes and corrects
  the Taylor formulation.**
- `[O]` Estimate the hierarchy's coefficients from data. The moments are the transition
  statistics of the predictor — drift, spread, and higher cumulants — so discovery becomes
  Kramers–Moyal estimation *constrained to yield a stable integrator*. That constraint is
  what no existing discovery method imposes.
- `[O]` **The failure signature matters more than the success.** By Pawula, data with no
  local PDE should make the hierarchy refuse to truncate, or force negative weights. A
  method that loudly says "no local rule exists here" is worth more than one that always
  returns an equation.

## Part VI — Agents at all scales

- `[P]` Stencils compose into stencils; ‖w‖₁ is submultiplicative. So **‖w‖₁ = 1 is exactly
  the condition for an agent to compose with itself indefinitely without amplification.**
  A scale-free agent is a positive one. *Stability is the tax on composition.*
- `[M/O]` Composing a positive stencil L times is a random walk; the RG fixed point is the
  heat kernel. Wide stencils and deep compositions are two representations of one operator.
- `[O]` Under partial observation the coarse law acquires memory (Mori–Zwanzig, cf. the
  2025-01-09 memo). Prediction: **more memory buys back positivity at fixed locality.** If
  that trade curve is real it is the quantitative form of the trilemma.

## What would falsify the program

- ~~If positivity costs *order* rather than a constant~~ — **settled, and it went the other way**:
  positivity costs no order in the parabolic case and exactly one in the hyperbolic case. The
  trilemma framing is retracted (F10).
- The experiment that could end Part V: take weak-form SINDy's better coefficients, build the
  same rows, impose positivity. If that works equally well, the moment route is unnecessary even
  for the one claim that survived it.
- If the moment hierarchy cannot be estimated from noisy data better than weak-form SINDy,
  Part V is a reformulation, not a method.
- If coarse-graining preserves positivity without memory, Part VI loses its tension.

## Deliberately not claimed

- A faster solver for the heat equation. On linear constant-coefficient problems with
  uniform grids, spectral methods beat everything here, and the wide-stencil frontier is
  **exactly first-order Runge–Kutta–Chebyshev** (machine precision, T2 Task 6). The value is
  in regimes where those do not apply: unknown equations, scattered points, variable and
  state-dependent coefficients.
