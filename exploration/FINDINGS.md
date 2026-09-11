# Findings log

## F1 — The positivity frontier is exact (2026-09-11)
For a symmetric uniform stencil of half-width m taking one step of k·Δt, the largest k
admitting a non-negative consistent weight vector is **exactly**

    k_max(m) = floor[ (mΔx)² / (2αΔt) ]

Verified by bisection at NX=201, α=0.1, c=1: m=1→1 (pred 1.1), 2→4 (4.4), 3→10 (10.0),
4→17 (17.8), 6→40 (40.0), 8→71 (71.1), 12→160 (160.0), 16→284 (284.4). Exact to the floor
at every m. The constant is 1, i.e. the capture factor γ=1 — the frontier sits where the
stencil half-width equals one standard deviation of the heat kernel it must represent.

## F2 — Cost per unit time decreases with width, but accuracy binds first
Cost/time ∝ (2m+1)/k_max(m) falls monotonically: 3.00 (m=1) → 0.116 (m=16), suggesting
~26× headroom. **Measured speedup at matched accuracy is only ~2.2×** (best fixed scheme
m=5,k=8: cost 109k vs FTCS 239k, but error 3.7e-4 vs 9.2e-5). The best scheme uses k=8
while k_max(5)=27 — so it stops *far* short of the positivity limit. Accuracy, not
stability, is the binding constraint. The naive cost argument is wrong by an order of
magnitude. → This significantly weakens the standalone case for M2.

## F3 — ⭐ The Jensen obstruction: accuracy beyond 2nd order requires ‖w‖₁ > 1
Non-negative weights summing to 1 are a probability distribution over the offsets.
Therefore, by Jensen,

    M₂ := Σᵢ wᵢ Δtᵢ²  ≥  (Σᵢ wᵢ Δtᵢ)²  =  (effective time advance)²  >  0

with equality iff all Δtᵢ are equal. **The u_tt error term can never be cancelled by any
non-negative weight vector**, no matter how many time levels the stencil spans.

Measured: 0 violations in 2550 sampled positive stencils; min slack −2e-22 (machine zero).
Then, on multi-time-level stencils where cancelling Δt² is attempted explicitly:

| | can cancel Σw Δt² = 0 |
|---|---|
| w ≥ 0 | **0 / 600 (0.0%)** |
| signed w | **600 / 600 (100%)**, ‖w‖₁ median 1.286, min 1.018 |

So the trade is exact and priced: **the minimum ‖w‖₁ needed to buy the next order of
accuracy is measurable, and it is > 1.** Stability and accuracy are on a genuine Pareto
frontier, and ‖w‖₁ is the currency.

This is the discrete mechanism behind the classical order barriers — Godunov's theorem
(linear monotone schemes for hyperbolic problems are at most 1st order) and the
Bolley–Crouzeix result for positive parabolic schemes. *Open*: whether the Jensen
obstruction costs **order** or only a **constant** depends on the refinement path. A
fixed-shape parabolic refinement showed both positive and signed weights converging at
the same rate with a 20–50× constant penalty for positivity, not a different slope.
Nailing this down is assigned to T1.

## F4 — ⭐⭐ Why RL, and not just a better hand-designed scheme
F3 says: no **linear** scheme can be both monotone and high-order. The classical escape
is to give up linearity — flux limiters, ENO/WENO — schemes that **choose their stencil
based on the local solution**.

An RL agent selecting stencil geometry from the local state *is exactly such a nonlinear
scheme*, learned rather than hand-designed.

This inverts the project's justification. RL is not a fashionable wrapper on a numerical
method; **Godunov's theorem is the reason a learned, data-dependent stencil choice is
necessary at all.** Hand-designed limiters are the human solution to the same obstruction.
That is the spine M1 should be built on.

## F5 — Positivity makes planning tractable (and its absence is the argument for search)
With ‖w‖₁ = 1 errors accumulate linearly, so total error is **additive** over steps and
the optimal scheme schedule is an exact shortest-path problem, solvable by DP.
With ‖w‖₁ > 1 accumulation is multiplicative and history-dependent; the objective no
longer factorises and search is genuinely required.
→ This gives T4 a *computable optimal policy* to validate RL against, and simultaneously
explains where RL becomes necessary rather than merely convenient.

## F6 — ⭐⭐⭐ Moment rows replace Taylor rows, and the giant step works
The 3-row Taylor/PDE system is only consistent for a *small* step. Widening the stencil
adds freedom but no constraints, so accuracy plateaus (measured: error stuck at 2.5e-2 at
t*=400 steps regardless of stencil width γ=1…5). The fix is to match more moments.

For u_t = Lu, the exact propagator over τ is exp(τL). Its raw moments against monomials
are computable from the **PDE coefficients alone** via a closed triangular ODE hierarchy
obtained by integration by parts:

    dM_q/dτ = α·q(q−1)·M_{q−2} + c·q·M_{q−1},        M_q(0) = δ_{q0}

Verified against the closed-form Gaussian moments to 1.6e-9 (ODE tolerance) through q=8.
**No fundamental solution is used.** This generalises to any linear PDE and, locally, to
variable and state-dependent coefficients — so it is not "sample the known Green's function".

Matching moments q = 0…p with w ≥ 0, one giant step from the initial condition:

| t* (CFL steps) | explicit FD cost / err | moment stencil cost / err | |
|---|---|---|---|
| 100 | 10,198 / 8.38e-6 | 59 / **3.2e-7** (γ=3, p=6) | **173× cheaper, 26× more accurate** |
| 400 | 69,898 / 2.57e-5 | 153 / **1.33e-7** (γ=4, p=8) | **457× cheaper, 193× more accurate** |

(FD baseline re-run with the F8-corrected row, which improves it ~2.4×. The moment-stencil
numbers are unchanged — they are built from monomial moment rows and never touch `build_A`,
so F6 is independent of both the F8 row bug and T1's amended k_max frontier.)

The trade is clean and structural: **stencil width buys moment order, subject to
positivity.** Measured feasibility: γ=2 supports p≤4; γ=3 supports p≤6; γ=4 supports p≤8.

This is the classical **truncated moment problem**: a non-negative w matching a target
moment sequence on a given support exists iff the sequence is realisable (Hankel PSD).
Positivity feasibility is not an ad-hoc filter — it is a named, well-studied condition.

## F7 — ⭐⭐⭐ Pawula's theorem, and an answer to the motivating question
The propagator moments are the transition statistics of the "agent": M₁ is drift
(advection), M₂ is spread (diffusion), higher moments are non-Gaussian corrections.
This is exactly a Kramers–Moyal expansion.

**Pawula's theorem**: a Kramers–Moyal expansion that truncates at any order > 2 must
truncate at 2, or the propagator ceases to be a non-negative density.

Read in this framework that says: **a local, positive propagator is necessarily
second-order.** Anything richer must give up either locality (→ integro-differential
operators, memory, Mori–Zwanzig — cf. the 2025-01-09 voice memo) or positivity (→ signed
weights, ‖w‖₁ > 1, geometric error growth under composition).

So the motivating question — *do we still need differential equations?* — gets a sharp
answer in this language:

> **Locality, positivity, expressiveness: pick two.**
>
> Second-order PDEs are exactly the fixed point where all three nearly coexist. That is
> why they are ubiquitous, and it is also precisely what you must give up to go beyond them.

This is the same trilemma T5 is measuring as "positivity, locality, accuracy — pick two",
and the same one F3 prices via ‖w‖₁. Three independent routes to one statement.

## F8 — ⭐ Bug in the third constraint row (confirmed symbolically)
`build_A` in the RL-For-PDEs code uses row 3 = `½Δx² + αΔt`. Expanding the Taylor series and
substituting u_t = αu_xx − cu_x *recursively* (so that u_xt and u_tt also reduce to
x-derivatives), the true coefficient of u_xx is

    ½Δx² − c·ΔxΔt + ½c²Δt² + αΔt  =  **½(Δx − cΔt)² + αΔt**

Verified with sympy: exact match. The code drops `−c·ΔxΔt` and `½c²Δt²`.

The correct form is the natural one: **½(characteristic offset)² + αΔt**. In the moving
frame ξ = Δx − cΔt the equation is pure diffusion, so the second-order row is just
½ξ² + αΔt. Row 2 is already ξ itself — the two rows are ξ and ½ξ² + αΔt, which is obvious
in hindsight and makes the error hard to spot.

Magnitude at the benchmark (c=1, α=0.1, Δx=5e-3, Δt=1.125e-4): the dropped term is
2cΔt/Δx = 4.5% of row 3. Harmless for the reported diffusion-dominated results; **not
harmless in any advection-dominated regime**, and T1 reports that with the corrected row the
3-point centred stencil is exactly Lax–Wendroff, whereas with the current row it is
FTCS-central, which is unconditionally unstable. So the shipped row cannot produce a
second-order advection scheme at all. Credit: found by T1.

**The moment hierarchy gets this right automatically.** Imposing Σw·[½(Δx−cΔt)² + αΔt] = 0
with Δtᵢ = −τ is algebraically identical to M₂ = (cτ)² + 2ατ with M₁ = −cτ, which is exactly
the F6 hierarchy. So the moment formulation *subsumes and corrects* the Taylor formulation —
an independent argument for adopting it.

## F9 — Vertical stencils: a real but guarded silent failure
If all neighbours share the same Δx, rows 2 and 3 are both affine in Δt, so A has rank 2,
b lies outside its range, and `lstsq` returns a vector with residual 0.72 and **Σw = 0.478**
— silently, with no error. The environment would then predict a garbage value.

**Severity correction (I disagree with T1's framing here).** Measured over 15,948 random
5-point stencils: only 2 (0.01%) give Σw ≠ 1, and **0 of them pass the `cond(A) < 1e4` gate**
(the exactly-vertical case has cond = 1.75e16). So the existing conditioning check does catch
this in practice. It is a latent trap, not an active corruption — the fix is a cheap explicit
`assert |Σw − 1| < 1e-6` rather than relying on conditioning to imply consistency.

This also settles the earlier disagreement about conditioning: `cond < 1e4` is doing its job
as a **rank guard**. It was never an amplification guard, and ‖w‖₁ still needs its own
control. Both T1's independent count (2 of 15,948) and mine agree exactly.

## F1 — AMENDED
`k_max(m) = floor[(mΔx)²/(2αΔt)]` is exact only on the **diffusive branch** (small m).
T1 reports two further bounds binding at larger m, with reachable depth saturating near
435 steps regardless of width; the formula overstates by 2.5× at m=25 and 4× at m=40.
My verification only ran to m=16, entirely inside the diffusive branch. The F6
query-driven results use γ≤4 (m≤76 at t*=400) and so may sit outside it — **flagged for
recheck against T1's corrected frontier.**
