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

## Part III — The trilemma

The central claim of the book, reached by three independent routes.

- `[P]` **Jensen obstruction.** Non-negative weights summing to 1 are a probability
  distribution, so Σwᵢ Δtᵢ² ≥ (Σwᵢ Δtᵢ)² > 0. The u_tt error term **can never** be cancelled
  by any positive w. Measured: 0/600 multi-level stencils cancel it with w ≥ 0; 600/600 do
  with signed w, at minimum ‖w‖₁ = 1.018. **Accuracy beyond the barrier has a price in
  ‖w‖₁, and the price is measurable.**
- `[M]` **Godunov.** Positivity fails for second-order advection exactly where the theorem
  says it must: no linear monotone scheme is second-order.
- `[P]` **Pawula.** The propagator moments are a Kramers–Moyal expansion. A KM expansion
  truncating above order 2 must truncate at 2, or the propagator stops being a non-negative
  density. So **a local positive propagator is necessarily second-order.**

> ### Locality, positivity, expressiveness — pick two.
>
> Second-order PDEs are the fixed point where all three nearly coexist. That is why they are
> everywhere, and it is exactly what you surrender to go beyond them: either locality
> (integro-differential operators, memory) or positivity (signed weights, geometric error
> growth under composition).

This is the book's answer to its title question, and it is not a hedge.

## Part IV — Why learning is forced

- Part III says no **linear** scheme is both monotone and high-order. The classical escape
  is to abandon linearity: flux limiters, ENO/WENO — schemes that **choose their stencil
  from the local solution**.
- A learned, state-dependent stencil selector is exactly such a nonlinear scheme, general
  rather than hand-designed. **The order barrier is the reason to learn**, not a decoration
  on it.
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

- If positivity costs *order* rather than a constant on every refinement path, the
  "stability is cheap" half of the story collapses. (T1, Task 6 — currently open.)
- If the moment hierarchy cannot be estimated from noisy data better than weak-form SINDy,
  Part V is a reformulation, not a method.
- If coarse-graining preserves positivity without memory, Part VI loses its tension.

## Deliberately not claimed

- A faster solver for the heat equation. On linear constant-coefficient problems with
  uniform grids, spectral methods beat everything here, and the wide-stencil frontier is
  **exactly first-order Runge–Kutta–Chebyshev** (machine precision, T2 Task 6). The value is
  in regimes where those do not apply: unknown equations, scattered points, variable and
  state-dependent coefficients.
