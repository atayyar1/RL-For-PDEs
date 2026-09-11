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

## F3 — ⛔ RETRACTED (see F10). The Jensen obstruction: accuracy beyond 2nd order requires ‖w‖₁ > 1
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

## F7 — ⛔ PARTLY RETRACTED (see F10). Pawula's theorem
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


## F10 — ⛔ RETRACTION of F3's consequence and F7's trilemma
**The Jensen inequality is true; the order barrier drawn from it is false.**

Under a PDE constraint u_tt is *not* an independent error term. ∂ₓ commutes with the
generator, so u(x+Δx, t+Δt) = exp(q∂ₓ + a∂ₓ²)u with q = Δx − cΔt, a = αΔt: every time
derivative collapses into two numbers. The ½Δt²u_tt contribution is redistributed across
u_xx, u_xxx and u_xxxx, where the remaining moments cancel it routinely. The Jensen
constraint binds for the *generic* hierarchy (exactness for arbitrary smooth f(x,t)) and is
**vacuous for the PDE-constrained one**. Credit: T1.

Counterexample, verified independently here by measured convergence:

| r = αΔt/Δx² | weights | w ≥ 0 | observed order | error at Δx=1/320 |
|---|---|---|---|---|
| 0.10 | (0.1000, 0.8000, 0.1000) | yes | 2.00 | 5.2e-7 |
| 0.25 | (0.2500, 0.5000, 0.2500) | yes | 2.00 | 6.5e-7 |
| **1/6** | **(0.1667, 0.6667, 0.1667)** | **yes** | **4.00** | **3.4e-12** |
| 0.50 | (0.5000, 0.0000, 0.5000) | yes | 2.00 | 2.6e-6 |

At *matched* moment conditions positivity costs nothing: identical rates at order 3 and 4,
and at order 3 the positive stencil is **2.6× more accurate**. My earlier "20–50× constant
penalty" was an artefact of comparing at mismatched conditions, where a symmetric min-norm
solution collects a free extra order. The penalty was not even constant — it grew like 1/h.

**What is actually true:**
- **Hyperbolic**: positivity costs exactly one order — Godunov. Verified: upwind (positive)
  1.00, Lax–Wendroff (signed) 2.00. T1 confirms max positive order = 1 at 3, 5, 7, 11 points
  for every Courant number.
- **Parabolic**: positivity costs *no* order. It costs **feasibility** — under Δt ∼ Δx the
  max positive order collapses 3 → 2 → 1 as r runs 0.98 → 7.98.
- **Pawula** survives only in its proper form: it concerns which *generators* admit
  non-negative propagators, not the approximation order of a positive scheme.

**RESOLVED by literature check** (was flagged as unverified recall). Bolley & Crouzeix (1978):
to preserve positivity for the heat equation a discrete method must **either** use time
discretization of order at most one, **or** impose stability conditions relating the time step
to the spatial discretization. So any *unconditionally* positive method is at most first-order
in time — and there is no barrier for *conditionally* positive ones.

T1's caution was exactly right and the reconciliation is exactly conditional vs unconditional.
Our r = 1/6 counterexample lives on the second branch: it is second-order in time and fourth-order
in space, but only under the imposed relation r = αΔt/Δx² = 1/6 — a stability condition tying Δt
to Δx, which is precisely the escape clause the theorem names. No contradiction.

## F11 — The frontier has three branches, and one is m-independent
`k_max = min( m²/(2r), m/ν, 2α/(c²Δt) )` — LP-verified for every m ≤ 25 at c = 0 and c = 1.
The third branch saturates at **435 steps** here and does not improve with width. Crossover
of the first two at m = 2/Pe = 19.8 cells. So F1's quadratic law is exact only up to m ≈ 20;
it overstates by 4.1× at m = 40 and 14.8× at m = 76. **The F6 query experiment at γ=4 (m=76,
k=400) sits at 92% of the true frontier, not the 16% headroom the quadratic law implied** —
and the binding constraint there does not depend on m at all. T2 notes this advective cap is
an artefact of the 3-row spec and vanishes once u_tt is kept.

## F12 — One-sided stencils are never positive-feasible
0 / 15 501, exhaustively, both sides. Downwind fails because q = Δx + c|Δt| > 0 always;
upwind fails unless the window exceeds 2α/(cΔx) = 2/Pe = 19.8 cells, with the measured
crossover exactly there (0.000 at D=19, 0.038 at D=20).

This is the most useful result for the RL thread. A time-pressured policy that sprints
one-sided toward its target is not merely inaccurate — it is **categorically outside the
feasible set**. The constraint positivity imposes is on *geometry*, which is exactly what
the agent chooses.

## F13 — ⛔ AMENDED: ‖w‖₁ = 1 is sufficient, not necessary
Credit: T5. The bound |e| ≤ ‖w‖₁^L is a worst case over adversarial error patterns and is
generically nowhere near attained. Verified here: Lax–Wendroff at ν = 0.4 has
w = (0.28, 0.84, −0.12), **‖w‖₁ = 1.24**, yet **max|symbol| = 1.000000** exactly — von Neumann
stable. Under 512-fold composition ‖w‖₁ reaches only **1.59 against a bound of 6.8 × 10⁴⁷**.
T5 reports the mechanism: composition *expels* the negative lobes (innermost negative weight
migrates 6σ → 25σ, negative mass 10⁻² → 10⁻⁹⁰) and ‖w^{*L}‖₁ → 1.

So the correct statement of the program's central object:

| | ‖w‖₁ = 1 (positivity) | von Neumann max\|ĝ\| ≤ 1 |
|---|---|---|
| sufficient for stability | yes | yes |
| necessary | **no** | yes (asymptotically) |
| exact at every depth L | **yes, no transient** | asymptotic only |
| needs translation invariance | **no** | **yes** |
| computable on scattered geometry | **yes** | **no** |

**Positivity is the certificate you can always compute; the symbol is the sharp criterion you
can only sometimes compute.** In the meshfree setting this program targets there is no symbol,
so positivity is the only certificate available — at the real cost of rejecting perfectly stable
schemes such as Lax–Wendroff. That cost must be stated, not hidden.

## F11 — AMENDED: the frontier conflict resolved, and a sharp closed form
T1 and T5 reported different frontiers. Both were right **for different rows**, and I verified
this directly against the LP:

| m | LP, uncorrected row | T1's min(3 branches) | LP, corrected row | T5's closed form |
|---|---|---|---|---|
| 8 | 71 | 71.1 | 62 | 62.2 |
| 16 | 284 | 284.4 | 196 | 196.1 |
| 20 | 435 | 435.6 | 273 | 273.2 |
| 32 | 435 | 435.6 | 519 | 519.1 |
| 76 | 435 | 435.6 | 1468 | 1468.3 |

T1's 435-step saturation is an **artefact of the uncorrected row** (T2 predicted exactly this:
"it vanishes once u_tt is kept"). With the corrected row — now the code's default — there is no
saturation and T5's closed form is sharp at every m:

    k_max = ( −r + √(r² + ν²m²) ) / ν²,     r = αΔt/Δx²,  ν = cΔt/Δx

because the consistency conditions fix the **raw** second moment 2αkΔt + (ckΔt)² — variance
*plus mean squared* — which a probability measure on {−m..m} can realise only while it stays
below m²Δx². Limits: ν → 0 gives m²/(2r); m → ∞ gives m/ν. Now in `core.stencil.k_max`, with the
test bisecting against the LP rather than assuming any formula.

**Consequence for F6**: my γ=4 query experiment (m=76, k=400) sits at **27%** of the true
frontier, not the 92% I reported from T1's numbers. The earlier "9% headroom" warning is void.

## F14 — ⭐ Coarse-graining destroys positivity, and memory is its price
Credit: T5, and this is the tension I asked for.
- Spatial coarse-graining by factor M admits an **exact, finite** Mori–Zwanzig memory of exactly
  **M − 1 lags** (Cayley–Hamilton on the M-dimensional alias subspace). MZ kernels are normally
  infinite; here the truncation is exact.
- **Positivity fails for M ≥ 3 at every r.** M = 2 survives only if r ≥ 1 − 1/√2 = 0.29289
  (analytic; bisection gives 0.293048). Since FTCS needs r ≤ ½, coarsening by 2 is safe only in
  the narrow window 0.293 ≤ r ≤ 0.5 — i.e. only when the fine scheme runs near its stability limit.
- Under **RG-consistent** coarsening (diffusive, Δt → M²Δt) memory collapses to an M-independent
  floor e^{−2π²r}, verified to a few percent over three decades in r.

> **Memory is the price of coarse-graining space faster than the dynamics mixes.**

So a trilemma does survive — but it is about *coarse-graining*, not about accuracy order:
**you cannot coarse-grain by M ≥ 3, stay local and memoryless, and stay positive.** That is a
measured result, and it is the one I should have had instead of the one I published.

## F15 — Composition: orders take the min, and cumulant defects are exactly additive
Credit: T5. The propagator is a Lévy kernel whose cumulants are all linear in τ, and cumulants
add under convolution, so cumulant defects are **exactly** additive: ε_n(w^{*L}) = L·ε_n(w), with
no remainder. This explains why modified-equation coefficients are step-count independent —
α_eff = 0.09977043 at every L from 1 to 512, all digits identical.
Composition order takes the **min** of the parts and can never fall below it, so the
counterexample I asked T5 to hunt for provably does not exist (600 random heterogeneous
compositions searched; none). T5 retracts its own sub-Gaussian tail claim: the tails track the
Gaussian to a few percent and cut off only at k → ρ(L) = √(L/s₂), which is simultaneously the
compression ratio and the fixed point's range of validity.

## F16 — ⭐ Target-awareness pays, and a crude cone captures it (constant coefficients)
The decisive test for the query-driven framing: is the optimal allocation of computational
effort a *local* formula, or does it need to know the query?

Linearised, the error at a query is `E = Σ G(j,n)·τ(j,n)`, with τ the local truncation and
**G the influence function** — the adjoint solution, one backward sweep from a delta at z*.
Three greedy policies on the same budget, marching the real mixed scheme and measuring the
true error against the exact solution (161 × 120 levels, t* = 0.021):

| budget | uniform | local indicator | **target-aware** | gain vs local |
|---|---|---|---|---|
| 5% | 1.8e-5 | 1.9e-5 | **8.9e-6** | 2.1× |
| 15% | 1.6e-5 | 1.7e-5 | **1.1e-6** | 15× |
| 35% | 1.3e-5 | 1.3e-5 | **7.7e-8** | 167× |

Median gain **18×** across three query locations. And note the second column: **the standard
local truncation indicator is no better than uniform, sometimes worse.** Effort spent where
truncation is large is wasted if that error never reaches the query.

This is the first result in the programme that supports the RL framing rather than undercutting
it, and it validates Ali's specific design choice to put z* in the state. A first pass also
suggested a *crude* influence estimate — a constant-coefficient Gaussian cone using the
domain-mean α, ignoring all spatial structure — retains essentially all of the exact adjoint's
gain, which is what makes it learnable. That sub-result is from the inconclusive run below and
needs redoing.

## F17 — ⛔ INCONCLUSIVE: the variable-coefficient extension, and why (three failed attempts)
Whether F16 survives variable coefficients is **not settled**. Three attempts, each defeated by
a different flaw, all mine:

1. Reference was a sub-stepped cheap scheme whose own error (~2e-7) was comparable to the spread
   between policies (~4e-7); and the "expensive" stencil matched only moments 0..2, so it was
   barely better than cheap.
2. Rebuilt with order-4 moments and a Richardson-extrapolated reference. Reference residual
   1.6e-7 against a 2.6e-7 signal — still floored.
3. Rebuilt with a **manufactured solution**, so the reference is exact. This finally removed the
   reference problem and exposed the real one: **the wide "expensive" stencil is 0.8× — it is
   *worse* than the narrow one.** There is nothing to allocate, so the target-aware policy, which
   upgrades more effectively inside the influence cone, does *more* damage.

**The real finding hiding in attempt 3**: under variable coefficients a wider stencil commits a
larger **coefficient-freezing error** — moment rows built from α(x_j) are applied over a window
where α differs. Widening trades discretization error for coefficient-variation error, so the
optimal width is *narrower* than in the constant-coefficient case, and can be 1.

This bears directly on the width-buys-timestep story (T2) and cuts the same way: the wide-stencil
advantage **erodes precisely in the regime the programme cares about** — variable or unknown
coefficients — which is also the regime where spectral methods and RKC do not apply. The
wide-stencil idea has now lost its edge at both ends.

**To settle F16 under variable coefficients** the upgrade must actually be an upgrade. Either
correct the moment rows for local coefficient variation (include α'(x), α''(x) terms), or make
the expensive option a *sub-stepping* upgrade rather than a *widening* one. Not yet done.
