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

## F18 — ⛔ F2 RETRACTED (partly): the "(kΔt)² accuracy floor" was one bad configuration
F2 claimed accuracy binds long before stability, giving only a ~2.2× speedup, and that a
"(kΔt)² floor" was irreducible. T2 refutes this: I measured it at a single capture factor
s = mΔx/σ = 1.86, using the 3-row Taylor system. Sweeping s with moment rows, **pure diffusion**:

| s = mΔx/σ | 2.00 | 4.33 | 6.33 | 8.33 |
|---|---|---|---|---|
| error | 8.7e-4 | 8.8e-7 | 2.4e-11 | **1.5e-14** |

Nine orders at **fixed** kΔt, with w ≥ 0 throughout. So there is no floor; the constant is
~exp(−s²/2) and s is a free design choice I had pinned at a bad value. F2's speedup number is
void. Credit: T2.

**But it does not reproduce with advection**, and this is an open discrepancy. At c = 1
(cell-Pe 0.05, drift 0.9 cells) the error floors at **3.83e-7** from s ≈ 6 onward, regardless of
stencil width or moment order up to p = 10. Ruled out as causes:
- solver tolerance — the moment residual is 1.1e-16;
- boundaries — free-space convolution and the Dirichlet series agree to 2.2e-16;
- window centring — centring on the drifted kernel centre changes nothing past s ≈ 6.
- the exact **sampled heat kernel hits the identical floor**, so it is not the LP's doing.

Cause unresolved. It matters because F6's headline numbers were measured in the advective case.
Flagged for T2's exact configuration rather than guessed at.

## F19 — ⛔ Two more of my claims fall
- **The 3-row LP does not recover the heat kernel.** I suggested to T2 that positivity plus the
  three consistency rows would land near the sampled Gaussian. It does not: the LP returns a
  basic feasible solution with **≤3 non-zeros** (obvious in hindsight — three equality
  constraints), TV distance 0.88 from the kernel at m = 12. Verified here.
- **F17's conclusion is wrong.** I reported that the wide-stencil advantage "erodes under
  variable coefficients" because widening incurs coefficient-freezing error. The *diagnosis* was
  right for my implementation and the *conclusion* was wrong: freezing is avoidable. T2 builds
  rows from the local Taylor coefficients of α(x) — exponentiating on a polynomial basis, then
  projecting to w ≥ 0, with no fundamental solution anywhere — and reports **7–20× over RK4+FD6**,
  an order-matched, temporally-accurate competitor, at α_max/α_min = 9, while staying monotone.
  That reverses F17 and is the strongest engineering result the programme has produced.

## F20 — T2's other corrections (self-reported, worth recording)
- Withdrew its own advection cap `k ≤ 2α/c²` and the cell-Péclet barrier `ν·Pe² ≤ 2`; both were
  artefacts of the dropped u_tt terms. A positive stencil exists at **every** cell Péclet up to
  20, and at high Pe the frontier is **linear** in m, not quadratic.
- Its corrected frontier `floor([√(r² + ν²m²) − r]/ν²)` now agrees with T5's and with my own LP
  check — three independent routes to the same closed form.
- Found a metric error of its own: measuring the symbol to θ = π hits an aliasing floor where
  W(π) is real for any real w, which made positivity look free everywhere. On the resolved band
  θ ≤ π/2: **diffusion-dominated, positivity is nearly free** (a 10× ‖w‖₁ budget buys 1.1×);
  **advection-dominated sub-cell, it costs >10⁵×** (Godunov again).
- Consequence worth keeping: monotonicity is *not* what limits this scheme in the diffusive
  regime, which weakens "we preserve monotonicity" as a selling point.

## F21 — ⛔ RESOLVED: the "advective accuracy floor" was a bug in my own reference
F18 flagged an unexplained 3.83e-7 floor in the advective case, surviving every width, moment
order, centring and solver tolerance, and afflicting the exact sampled kernel identically.

**Cause: `Problem.u_true(x, 0)` was wrong.** The Dirichlet solution is built as
u = exp(βx − αβ²t)·v with β = c/(2α) = 5, where v solves the heat equation via a sine series.
But v₀ = u₀·exp(−βx) has non-zero second derivative at the walls, so its sine series converges
only as O(n⁻³) — and the reconstruction is then multiplied by exp(βx), which is 12× at midspan.
For **t > 0** the modes are damped by exp(−αn²π²t) and the series is exact to 2e-16; at **t = 0**
nothing damps it, so the *initial condition* fed to every stencil carried ~5e-7 of error while the
*reference* it was compared against was exact. Two completely different weight vectors (the LP
solution and the sampled kernel) gave identical errors precisely because they shared the bad input.

Fixed: `u_true` now returns the analytic initial condition at t = 0. Locked by
`test_F21_initial_condition_is_exact`. Diagnosis chain, all wrong before the right one: solver
tolerance (no, residual 1e-16), boundaries (no, 2e-16), window centring (no), conditioning (no,
cond = 95), moment truncation (no, predicted 8e-12), `u_true` at t = τ (no, exact).

**Consequences, both in T2's favour:**

1. **T2's nine-orders claim now reproduces in the advective case too.** At τ = 40 CFL steps,
   sweeping moment order at s = 4.3: 1.8e-6 (p=4) → 9.2e-10 (p=6) → 2.1e-11 (p=8) → **7.6e-15**
   (p=10), w ≥ 0 throughout. F18's "open discrepancy" is closed and was mine.

2. **F6's headline numbers were floored by the same bug and are much better than reported:**

   | t* (CFL steps) | FD cost / err | moment cost / err | |
   |---|---|---|---|
   | 100 | 10,198 / 8.8e-6 | 97 / **1.2e-14** | 105× cheaper, ~7×10⁸ more accurate |
   | 400 | 69,898 / 2.6e-5 | 191 / **6.2e-9** | 366× cheaper, 4219× more accurate |

   (Previously reported as 1.3e-7 error at t* = 400 — that was the bug, not the method. The cost
   ratios fall slightly because the optimum now prefers a wider, costlier, far more accurate
   stencil. At t* = 1600 the required stencil exceeds the domain, which is a real limit.)

## F22 — Off-centre stencils: the frontier saturation was an artefact of my own brief
T2 and T5 independently converged on this and I verified it by LP. Placing the stencil on
{j₀−m … j₀+m} with j₀ = round(−c·kΔt/Δx) — a free index shift — removes the drift penalty
entirely, because in the shifted index the residual drift is |μ′| ≤ ½ and the condition becomes
k ≤ (m² − μ′²)/(2r) → m²/(2r).

| m | 5 | 8 | 12 | 20 | 32 | 50 |
|---|---|---|---|---|---|---|
| centred (LP) | 26 | 62 | 124 | 273 | 519 | 903 |
| **off-centre (LP)** | **27** | **71** | **159** | **444** | **1137** | **2777** |
| m²/(2r) | 28 | 71 | 160 | 444 | 1138 | 2778 |

Off-centre tracks the pure-diffusive branch at every m; the gain grows without bound (3.08× at
m = 50). As T2 observes, the centred stencil is exactly what **my original brief specified**
("symmetric stencil, all neighbours at −kΔt"), so the saturation was an artefact of my problem
statement, which then propagated into T1's three-branch frontier and T5's work. Note it does
*not* affect the accuracy floor of F21 — the two are independent.

## F23 — ⭐⭐ The memory/positivity trade curve is real, and for small M the price is zero
T5's decisive experiment, formulated exactly rather than fitted: a compact coarse law of
half-width s and memory depth p is exact iff `Σⱼ B̂ⱼ(q)·g_l^{p+1−j} = g_l^{p+1}` over all
(mode, alias) pairs; solve unconstrained, then again under w ≥ 0 by NNLS. Initial-condition
independent.

At fixed minimal locality (3 coarse weights per lag), **memory buys back positivity
geometrically**:

| coarsening M | positivity defect decay per lag | extra lags per decade |
|---|---|---|
| 3 | 0.0966^p | 0.99 |
| 4 | 0.4414^p | 2.82 |
| 6 | 0.7370^p | 7.54 |

**Stronger than I predicted: for small M the price is exactly zero.** Exact non-negative compact
coarse laws exist — M=2 at (s=1, p=1), M=3 at (s=2, p=5) — with mass exactly 1, validated outside
the system they were fitted in (one-step error 1e-16, 400-step unforced rollout stable at 1e-15).
The M=2 law came out identical to all digits via the independent Cayley–Hamilton route of F14.

**The structure is not what "adding memory" suggests.** The M=3 law has B₁ = B₂ = 0 *exactly* —
it is a pure-delay scheme starting at lag 2. Buying back positivity means **moving the whole law
backwards in time, not correcting a Markov law**. At M=4 this stops (geometric convergence, no
snap to zero — a failure to find, not a proof).

So F14's "coarse-graining destroys positivity" is refined: it destroys *Markov* positivity. Memory
restores it, and at small M restores it exactly.

## F24 — ⛔ F7/Pawula fully dead, and now for the right reason
F10 retracted "local + positive ⟹ 2nd order" on the strength of the r=1/6 counterexample. T5
supplies the actual reason, which is sharper: **Pawula forbids the stencil's own cumulants from
terminating; accuracy only requires finitely many moments to be matched.** Those are different
conditions, and I conflated them.

Verified here — positive stencils matching propagator moments 0…p, all weights non-negative:

| p | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| points | 15 | 23 | 31 | 37 | 43 | 47 |
| min w | 2.5e-1 | 8.6e-2 | 2.0e-2 | 4.8e-3 | 5.8e-4 | 2.2e-4 |
| error | 1.0e-4 | 8.0e-7 | 2.4e-9 | 7.7e-12 | 6.4e-15 | 6.9e-15 |

Positive **and** twelfth-order. T5 adds that the real cap is set by r: order 10 at r ≥ 0.40,
order 2 at r ≤ 0.15. The classical result that genuinely caps order is Godunov's, which is
hyperbolic and does not bite in this parabolic regime.

## F25 — Four more corrections from T5, all to me
- **Wide ≠ composed.** I told T5 these were "two representations of the same operator". False.
  At L=40 on an identical 81-point footprint: composite 3.7e-6, single wide stencil at p=8
  **2.3e-11** — 1.6×10⁵ better. (At p=2 the wide stencil is *worse* than the composite, 1.5e-3.)
  Same footprint, different operators; which wins depends entirely on moments matched.
- **Edgeworth mechanism.** I predicted the leading correction was excess kurtosis at L⁻¹. It is
  **skewness at L^(−1/2)** — advection makes the step asymmetric (p₋ = 0.4727 vs p₊ = 0.4273).
  My mechanism holds only at c = 0, which T5 ran as a control. Both verified against analytic
  constants: c=1 skewness 0.008342 vs 0.008313 (0.34%); c=0 kurtosis 0.0942 vs 0.094195 (0.01%).
- **RG fixed point.** It is the Gaussian with α_eff = α − c²Δt/2, not α — the composite
  accumulates FTCS's own defect.
- **Compression cost.** One output value costs L² evaluations, not L: the light cone. My Task 3
  framing understated the composed route's cost.

Also: my corrected row 3 is exactly ½Θ₂ in T5's formulation, which already used and generalises it
— a fourth independent route to the F8 correction.

## F26 — ⭐⭐⭐ The positivity certificate is necessary, not sufficient — and at its boundary it certifies a useless scheme
T5's finding, verified here, and it is the most important qualification the programme has produced
because "impose positivity" was its one actionable recommendation.

At the frontier k = k_max(m) the only feasible measure is the **extremal two-point one**,
≈ ½(δ₋ₘ + δ₊ₘ): mass sits at the ends with essentially nothing between.

| m | k_max | non-zero weights | w(−m) | w(+m) | max interior \|w\| | ‖w‖₁ |
|---|---|---|---|---|---|---|
| 3 | 9 | 3 | 0.505 | 0.323 | 1.7e-1 | 1.000000 |
| 6 | 39 | 3 | 0.572 | 0.416 | 1.2e-2 | 1.000000 |
| 12 | 147 | 3 | 0.636 | 0.354 | 9.5e-3 | 1.000000 |

(T5 reports interior weights *identically* zero; with centred stencils and c ≠ 0 I get a small
residue, which the off-centre construction of F22 removes. Same structure either way.)

That stencil is **consistent** — all moment conditions exact — **positive**, and **stable**,
‖w‖₁ = 1.000000. And it is worthless:

| m | k | s = mΔx/σ | frontier error | same m, backed off | ratio |
|---|---|---|---|---|---|
| 6 | 39 | 1.01 | 1.5e-4 | 8.9e-16 | 1.7×10¹¹ |
| 12 | 147 | 1.04 | 2.0e-3 | 2.9e-15 | 7.0×10¹¹ |
| 20 | 368 | 1.10 | 1.2e-2 | 1.1e-12 | 1.0×10¹⁰ |

> **Consistency + positivity + stability do not imply accuracy. The feasibility frontier is a
> stability boundary, not an accuracy one.**

This explains, retrospectively, nearly everything the programme measured. At the frontier
s = mΔx/√(2ατ) = 1 **by construction** — and s = 1 is exactly where every accuracy sweep found the
worst results. It is the same pathology T2 found from the other side (at the frontier the m
residue classes never exchange information, so the scheme is m decoupled coarse solves), and the
same reason every Pareto-optimal configuration T2 found sits at s = 3–6.

**The design rule that follows.** Positivity must be paired with an accuracy criterion; alone it
is a filter that admits the worst scheme in the feasible set. Concretely: do *not* take the
largest positivity-feasible k. Take the largest k with s = mΔx/√(2ατ) ≥ s*, where T2's measured
interior optimum **s\* ≈ √(2 ln(1/ε))** is exactly the missing accuracy criterion. The two
results compose into one usable rule.

Two corollaries from T5, recorded as theirs:
- **RKC is not positive away from the frontier** (min w = −0.111 at k/k_max = 0.9). The classical
  scheme occupies a single point of the positive cone and leaves it the moment you back off for
  accuracy. Maximum entropy is the alternative — same feasible set, interior point, positive at
  every moment budget tested.
- T5 marks its own L^(1/2) grid speedup as a rediscovery of the RKC stability scaling, and
  corrects a label used across threads: V = u·e^(−βx) on *linear* advection–diffusion is the
  **gauge/Liouville transform**, not Cole–Hopf (which is the nonlinear Burgers→heat map).

---
# Phase 2 — the price of compression

## F27 — ⭐ The exact memory depth is algebraic; the *positive* memory depth is the live quantity
De-risk check on Phase 2's central question, run before committing to it
(`threads/M1-memory-spectrum/derisk.py`).

**Exact depth is spectrum-independent.** Coarse-graining the periodic FTCS stencil by M, a coarse
law of depth M−1 is exact for *every* r, because the monic polynomial whose roots are the M
aliased symbols always exists. That is Cayley–Hamilton on the alias subspace. Verified to 1e-16
residual for M = 2…6. **So p\* is not the spectral gap** — the classical part of it carries no
spectral information at all.

**The positivity-constrained depth is a different, much larger quantity** (independent NNLS
implementation, periodic FTCS, N = 120, reproducing T5's structure by a different method):

| M | exact depth | minimum **positive** depth |
|---|---|---|
| 2 | 1 | 2 (s=1) |
| 3 | 2 | **6 (s=2)** |
| ≥ 4 | 3+ | > 10 |

Exactness needs 2 lags at M=3; positivity needs 6.

**And it depends on r, with a sharp threshold structure:**

| r | M=2 | M=3 | M=4 |
|---|---|---|---|
| 0.10–0.20 | >8 | >8 | >8 |
| **0.2929** | **2 (s=1)** | >8 | >8 |
| 0.45 | 2 | 6 (s=2) | >8 |
| 0.50 | 2 | 3 (s=1) | >8 |

The M=2 threshold lands at **r = 0.2929**, matching T5's analytic 1 − 1/√2 = 0.29289 — independent
confirmation by a different method.

**Reading — ⛔ STRUCK, see F36.** I wrote: "you can coarse-grain cheaply only if the fine dynamics
already mixes across a coarse cell in one step." **That is the wrong mechanism.** The threshold is
the condition that the unresolved alias **flips sign**, not that it becomes small. At M=2, r=0.24
gives a mode damped 25× per step and is infeasible; r=0.45 gives one damped 1.25× per step and is
feasible. *The strongly mixing case is the impossible one.*

**The consequence that matters for the framework**: for **M ≥ 4, no r inside FTCS's own stability
range (r ≤ ½) admits a positive coarse law at depth ≤ 8** — large single-jump coarse-grainings
look unboundedly expensive. But M = 2 is cheap over a wide range of r.

Which makes one experiment decisive, and it is now M1's priority:

> **Coarse-grain by 2 twice, versus by 4 once.**
>
> If iterating cheap steps stays cheap while the single big jump does not, deep hierarchies are
> viable *only if built in small steps* — a measured statement about why multiscale structure
> looks the way it does. If memory compounds across iterated steps instead, hierarchies are
> expensive however they are built.

## F28 — T2's final report: the claim survives, properly bounded
Four scope questions answered; two of T2's own guesses reversed.

**Scope correction on the headline.** The 7–20× over RK4+FD6 was measured **periodic — no
boundaries at all**. That must be on the abstract.

**Boundaries, priced honestly.** T2 reverses its own earlier guess: a one-sided *positive* stencil
does exist from j = 1, and what degrades is achievable moment order, gracefully —
P_max = 1, 2, 4, 6, 6 at j = 0, 1, 2, 3, 4. So the order-reduction layer is **4–6 cells**, far
thinner than the m+k layer sub-stepping needs. With the wall charged flop-by-flop: **1.17e-6 at
9.57e4 flops vs FTCS 1.74e-4 at 1.17e5 — 148× accuracy at 0.82× cost.** But this is **against
FTCS only**; RK4+FD6 was not re-run with walls, so 148× is *not* comparable to the 7–20×.

**Rough α: breaks where predicted, with a clean scaling.** The advantage tracks
**L_α = α/|α′| in cells** — exactly the validity range of the local Taylor rows:

| L_α (cells) | 45.8 | 22.9 | 11.5 | 5.7 | 5.8 (step) | 1.4 (step) |
|---|---|---|---|---|---|---|
| ratio vs FTCS | 2386× | 2430× | 27× | 1.3× | **0.79×** | **0.01×** |

Gone by ~6 cells, actively harmful below ~2. Gradual and predictable, not a cliff. (T2 discarded
its own first tanh run — the spectral RK4 reference blew up — and redid it.)

**Positivity is load-bearing, not free.** T2 guessed wrong and says so: same rows, w ≥ 0 vs signed
min-norm, positive wins 5/6 by up to 60×, and the gap **widens with step count** — 4.0× at 16
steps, **78.1× at 32**. Attributed to ‖w‖₁ = B > 1 compounding as B^n. So the claim is
"local-α rows **and** positivity", not "positivity free on top".

## F29 — ⛔ I could not reproduce the ‖w‖₁ compounding, twice, and both failures were mine
This matters because it is the reconciliation of a real tension: T5 showed the ℓ¹ bound is
astronomically loose for a *fixed uniform* stencil (Lax–Wendroff, ‖w‖₁ = 1.24, symbol modulus
exactly 1, 512-fold composition reaching 1.59 against a bound of 6.8e47). T2 reports genuine
compounding with *variable coefficients*. The proposed reconciliation — von Neumann needs
translation invariance, which variable coefficients destroy — is plausible and would matter.

Two attempts, neither able to see the effect:
1. **3 Taylor rows at m = 2, 3.** Positive and min-norm errors identical to 4 digits at every step
   count, despite mean ‖w‖₁ of 1.12 and 1.27. Truncation from the low-order rows swamped any
   amplification.
2. **Moment rows at P = 6, m = 8** (T2's regime). min-norm ‖w‖₁ = 1.47, negative weights at
   **every** point — and still ratio 1.00× at 4, 8, 16 and 32 steps. Diagnosis: only 106 of 185
   interior points were solvable both ways, so 43% of the domain fell back to the *same* 3-point
   scheme in both arms and dominated the error, which sat flat at 1.05e-5 in n.

**The pattern is mine, not the method's.** This is the same failure as the advective-floor saga
(three attempts) and the variable-coefficient influence test (three attempts): I build a
measurement in which the quantity of interest sits below the noise floor of my own setup, then
read the floor as a result. Standing correction for future work: **before running a comparison,
measure the floor of the apparatus and check the expected effect exceeds it.**

T2's configuration has been requested rather than its number accepted or rejected.

## F30 — ⭐⭐ The constructive completion of F26: use maximum entropy, not the LP
F19 recorded that the 3-row LP does not recover the heat kernel (≤3 non-zeros, TV 0.88) and I
treated that as a curiosity. F26 then showed the frontier stencil is the extremal two-point
measure and is useless. **These are the same fact, and it has a fix.**

**Linear programming returns vertices.** With 3 equality rows, any LP hands back a weight vector
with at most 3 non-zeros — the extremal measure. So `solve_positive` was always returning the
worst member of the feasible set, at every k, not only at the frontier.

Maximising −Σ w log w over the same set returns the interior point instead. Since
w ∝ exp(A[1:]ᵀλ) and the rows are 1, ξ, ξ², the answer is **a discrete Gaussian in the
characteristic offset — the propagator itself, recovered without being told it**. (T5 proposed
max-entropy; T2 showed it coincides with their independently-derived two-parameter family to
1e-16, so these are one construction with two derivations.)

Measured, nx = 401, τ = 40 and 160 CFL steps:

| s = mΔx/σ | LP vertex error | max-entropy error | gain |
|---|---|---|---|
| 2 | 5.00e-6 | 4.09e-6 | 1.2× |
| 4 | 8.63e-6 | 5.32e-8 | 162× |
| 6 | 9.92e-6 | **7.41e-12** | **1.3 × 10⁶×** |
| 2 (τ=160) | 6.37e-4 | 6.79e-5 | 9.4× |
| 4 | 1.45e-3 | 9.70e-7 | 1494× |
| 6 | 2.11e-3 | **1.54e-10** | **1.4 × 10⁷×** |

**The LP error gets *worse* with width** — 5.0e-6 → 9.9e-6 — because the vertex simply moves its
three masses further apart. Max-entropy converts width into accuracy exponentially.

So F26's design rule was incomplete. The full rule has three parts, none sufficient alone:

> 1. **Positivity feasibility** — the certificate: does a stable scheme exist here at all?
> 2. **Maximum entropy** — which member of the feasible set to take. Not any feasible point.
> 3. **s ≥ s\* ≈ √(2 ln 1/ε)** — how much width to spend.

`solve_maxent` added to `core/stencil.py` (Newton on the convex dual, backtracking), with
`test_F30_maxent_beats_the_lp_vertex` asserting both that max-entropy wins and that the LP vertex
fails to improve with width. 13/13 passing.

## F31 — T2 / T5 cross-validation closes Phase 1
- **Max-entropy = T2's construction**, agreeing to 2.8e-17 at five different (m,k). One
  construction, two derivations; T5's convex dual is the better presentation since it extends to
  more moments without a bespoke parametrisation.
- **RKC positivity reproduced to three digits**: min w = −0.0000 at ρ=1 (interior non-zeros: 0),
  −0.1115 at ρ=0.9, −0.2651 at ρ=0.3. **Classical RKC meets the positive cone at exactly one
  point — its extreme vertex — and leaves it the moment you back off for accuracy.**
- **Frontier degeneracy now has three independent derivations** (forced ±m measure; residue
  classes mod m never exchanging information; T_m(cos θ) = cos(mθ)).
- **Safety factors made explicit**: none of T2's headline numbers were frontier numbers. The
  7–20× variable-coefficient result is at s = 2.60; the constant-coefficient Pareto points are at
  s = 1.5–6, never 1. So F26 does not undercut them.

## F32 — ⭐⭐⭐ Memory-free coarse-graining and uselessness are the same phenomenon
T5's result, verified here, and it reframes Phase 2's central question.

The Phase-1 frontier stencil ½(δ₋ₘ + δ₊ₘ) has symbol cos(mθ). Coarse-grained by M = m, every
aliased mode gets the **same** eigenvalue, g_l = cos(2πmq/N) independent of the alias index. The
alias subspace is therefore an eigenspace, the minimal polynomial has degree 1, and the coarse law
is **exactly Markov with zero memory** — measured alias spread ~4e-15 and exact depth 0 at
M = m = 3, 4, 6, against the M−1 lags FTCS requires.

And that is the same degeneracy that makes the frontier stencil useless (F26). Hence:

> **The only operator you can coarse-grain for free is one that has already discarded what the
> coarse grid discards.** Memory-free coarse-graining and carrying no information are one
> phenomenon, not two.

The general form:

> exact memory depth = (number of **distinct** aliased eigenvalues) − 1.
>
> Zero memory ⟺ the symbol is constant on alias classes ⟺ the fine operator cannot distinguish
> modes the coarse grid cannot represent. An accurate operator evolves them differently, so it
> must pay memory. **Memory is the price of being able to distinguish what the coarse grid
> cannot.**

**This changes Phase 2's question.** It was "does compression across scales have a bounded price?"
It should be: *the price is zero only for operators that have already thrown away what is being
compressed.* For anything informative, the price is strictly positive — and the live question is
its rate, not its existence.

**And it changes what to measure.** The exact depth is a **rank statistic** and therefore jumps.
Interpolating u → (1−a)·FTCS + a·cos(3θ):

| a | alias spread | exact depth |
|---|---|---|
| 0.00 | 1.5588 | 2 |
| 0.50 | 0.7794 | 2 |
| 0.99 | **0.0156** | **2** |
| 1.00 | 0.0000 | **0** |

At a = 0.99 the operator is within 1.6% of degenerate and the exact depth is still 2. So the exact
depth says almost nothing about how close a system is to memory-free. The right pair is:

- **independent variable: alias spread** — how much the fine symbol varies across an alias class,
  i.e. how much the operator distinguishes what the coarse grid cannot;
- **dependent variable: positive depth** p\*(ε) — minimum memory depth admitting a *non-negative*
  compact coarse law. Continuous in the spread, because it is approximation not annihilation.

This independently re-derives F27 (exact depth = M−1 always, spectrum-independent, because it is
pure rank; the positive depth is the live quantity) from a second direction.

**Framework reading**: an agent that can compress without memory has nothing left to reconstruct.
Compression–reconstruction requires memory precisely because reconstruction requires having kept
something.

## F33 — ⛔ Phase 2 is largely dead, and three things I published are wrong
Both positioning threads came back before running experiments, which is what the discipline was
for. Between them they answered two of Phase 2's tasks, named its central object three times over,
and corrected three claims of mine — one of which is on the published page.

### The prior art
- **M2's kill shot**: *"Minimizing memory as an objective for coarse-graining"* — Guttenberg,
  Dama, Saunders, Voth, Weare & Dinner, J. Chem. Phys. **138**, 094111 (2013). M2's thread title,
  verbatim, thirteen years old. One search found it.
- **p\* has three names already**: the **McMillan degree** unconstrained; the **minimal positive
  realization order** with non-negativity (known to vastly exceed it — Benvenuti–Farina 2004,
  Benvenuti 2022); and the minimal *k* for **strong k-lumpability** (Gurvits–Ledoux;
  Geiger–Temmel, on Burke–Rosenblatt 1958), with p\*=0 being Kemeny–Snell lumpability exactly.
- **"Memory is the price of positivity" is fifty years old in other units.** Depth-p AR on K
  variables = Markov on pK variables, so it *is* "the minimal positive realization can be strictly
  larger than the minimal realization" — the central classical fact of positive realization theory.
  M1's and M2's shared thesis, translated.
- **Memory vs coarsening ratio is published**: Parish & Duraisamy measure τ ∝ Δ^1.5 across Burgers,
  homogeneous turbulence and channel flow — Phase 2's Task 2 headline, in the nonlinear setting
  the brief called out of reach.
- Also: PCCA+ (Deuflhard–Weber 2005) is structurally the shape of M2's hoped-for Task 2 answer;
  information-optimal RG does not increase interaction range (Lenggenhager et al., PRX 2020);
  van Enter–Fernández–Sokal (1993) is the honest ancestor of "coarse-graining destroys positivity";
  and p\* is **statistical complexity** in different units — for linear systems the ε-machine *is*
  the minimal realisation. Algebraic multigrid has the same problem in our own field, with learned
  prolongation since 2020, and my brief failed to name it.

### Phase 2's central experiment, answered for free and negatively
I called "coarse-grain by 2 twice versus by 4 once" the single most important measurement.
**It needs no experiment**: decimating by 2 twice retains the *same index set* as decimating by 4,
and the exact MZ law is a property of the observable. So memory depends only on the **total**
coarsening ratio — path-independent, never compounding. You cannot buy cheap hierarchies by taking
small steps. The framework question splits: the exactness price is cheap and linear; the positivity
price is infinite past M=4 and equally path-independent.

### My prediction was inverted, and M1's obstruction explains my own data
A non-negative coarse law of **any** depth or width requires every unresolved alias at the constant
coarse mode to satisfy g ≤ 0 — a **sign** condition, not a gap condition. (In substance,
Perron–Frobenius on a non-negative companion matrix; M1 claims novelty only for the reading.)
For FTCS, g_l = 1 − 4r sin²(πl/M), giving r ≥ 1/(4sin²(π/M)): 0.25 (M=2), 0.333 (M=3),
**0.50 (M=4)** — and FTCS needs r ≤ ½, so **p\* = ∞ for every M ≥ 4**.

Verified here against the threshold table I measured in F27 *before* the memo existed — **7 for 7**,
including all three infeasible cases. I had reported that table as an empirical threshold and read
it as "fast mixing makes coarse-graining cheap." That reading is **backwards**: at r=0.10, M=2 the
unresolved mode has g = +0.6 — decaying cleanly, perfectly well-behaved — and is fatal at every
depth; at r=0.45 it has g = −0.8 — barely decaying, but oscillating — and is free.

> **Coarse-graining preserves a maximum principle when the modes you discard oscillate, and
> destroys it when they merely decay. Scale separation is not what makes hierarchies cheap.**

### Three corrections to what I published
1. **F23's "memory buys back positivity geometrically" holds only to M = 3.** Past that the defect
   decays to an infimum **never attained** — M1 has a proof where T5 had "a failure to find". The
   trade curve terminates at M=3. My phrase "the price is exactly zero" was written against my
   expectation and not checked against the M≥4 rows. **This is on the published page.**
2. **F23's "pure-delay, B₁ = B₂ = 0" is a reach artefact, not a positivity phenomenon.** Verified:
   P·A^j·P^T is *exactly diagonal for j < M* at M = 2, 3, 4, because the fine stencil reaches one
   cell and coarse cells sit M apart — the coarse law **cannot** depend on neighbours before lag M,
   for purely geometric reasons. I published it as the striking structural finding of Phase 2.
3. **F14's "exactly M−1 lags — unusual, since MZ kernels are normally infinite" is wrong twice.**
   p\* ≤ ⌈N/K⌉−1 holds trivially for every rank-K projection and decimation attains the ceiling, so
   M−1 is the *worst case*, not special structure. And finite exact recurrences are generic in
   discrete-time linear systems, so the "normally infinite" framing should be retired.

### What survives
- **M1's obstruction and its inversion** — the sign condition, and the falsification experiment it
  implies (approved: two-timescale, advection-dominated, dispersive, r-sweep, plus decimation vs
  block-averaging, since every number above rests on a restriction operator nobody chose).
- **M2's degeneracy result**, and it is the more valuable: every 20-dim invariant subspace of the
  FTCS ring hits p\* = 0 exactly, including the **fastest** (VAMP-2 = 1.39 vs 15.66 for the
  slowest). Memory depth is a rank condition, blind to eigenvalue magnitude. So minimise-memory is
  strictly weaker than maximise-VAMP and its optimal set is the whole Grassmannian of invariant
  subspaces. Which qualifies the framework's own sentence: **"attend to whatever lets you forget"
  is under-determined** — forgetting is cheap in many directions, and what selects the useful ones
  is retained predictive content, not Markovianity.
- One uncleared corner (minimise memory subject to non-negative **and** banded), declined on M2's
  own 1-in-3 estimate.

### And a process failure of mine
`task5_trilemma.py` does exist, in T5's working directory. I copied their tree into the repo early
and never re-synced, so M1 read a stale snapshot and re-derived F23 from scratch believing the
script was missing. All four Phase-1 threads have now been rsynced. The silver lining is that F23's
existence claim is now confirmed by three independent methods.

## F34 — T3 closes: the detector survives, the big-step claim is bandwidth-dependent
**The mathematics has a cleaner statement than my ODE.** T3 verified the hierarchy to **1e-17**
(superseding my 1.6e-9, which was solver tolerance) and replaced it with an identity:

    Σ_q M_q s^q / q!  =  exp(τ · P(s)),    P(s) = αs² − cs

The stencil's moment generating function is **the exponential of the generator's symbol**, so
cumulants are κ_q = τ·q!·[s^q]P(s) and κ₃…κ₈ vanish exactly — *that is Pawula*. Generalises to any
linear PDE in one line, and should be the presentation.

**Best result in the thread: the detector measures the exponent, not just the failure.** On
non-local data the signature is bandwidth, α_eff(R) ∼ R^(2−2s), and it recovers
**s = 0.999, 0.905, 0.760, 0.520 against truths 1.0, 0.9, 0.75, 0.5**. A method that reports *how*
non-local the data is, rather than a spurious local PDE, is worth more than one that fits. (Clean
data only; unreliable by η = 1e-2.) T3's own first prediction here was wrong — it guessed
κ₂ ∼ τ^(1/s) — and it reports that.

**The big-step claim is bandwidth-dependent, and my check disagrees with T3's in one direction.**
T3 reports that simply sampling the Gaussian kernel beats the moment construction by 4–7 orders
"at every bandwidth". Measured here on exact multi-mode data after the F21 fix:

| data | s | sampled kernel | 3-row LP | 3-row maxent | moment p=8 |
|---|---|---|---|---|---|
| 4 modes | 4 | 4.9e-8 | 3.3e-8 | 1.1e-10 | **2.4e-17** |
| 4 modes | 6 | 2.2e-12 | 7.2e-6 | 1.2e-14 | 6.0e-16 |
| 40 modes | 4 | 5.1e-7 | 2.5e-4 | 5.0e-7 | **6.2e-8** |
| 40 modes | 6 | **2.5e-11** | 1.5e-3 | 2.0e-11 | 3.9e-7 |

**Narrowband: high-order moment matching beats the kernel by 3–9 orders** — backwards from T3's
claim. **Broadband at large s: the kernel wins**, as they say, because matching many moments on a
wide stencil over-fits low-frequency content the data does not have. And **3-row maxent tracks the
sampled kernel everywhere** — the same object, per F30. If T3's comparison used NNLS or an LP for
the moment construction, it was comparing the kernel against the *vertex*, which would produce
exactly their margins.

**Their claim-(c) refutation is very likely the same artefact.** They found NNLS 33,000× worse than
plain LS, "returning a sparse solution, a bad fit to a smooth Gaussian". NNLS returns vertices for
the same reason LP does. Asked to re-run with `solve_maxent` before writing the certificate off.

**Accepted without argument**: the 457× (now 366×) is the **single-point cone ratio**; full-grid is
**5.1×**. The step is unusable within ~2–3σ of a Dirichlet boundary. And on a uniform grid, kernel
sampling makes the moment machinery redundant — *that recipe is an exponential integrator*.

Which is the day's recurring lesson once more: **the machinery only earns its place where the
classical method does not reach.** T3's scattered-geometry experiment is therefore the whole
thread, and is now the only remaining Phase-1 item.

## F35 — M2 closed cleanly, with a caveat against its own memo
`POSITIONING.md` now opens with a STOP box: the Guttenberg citation with DOI and PMID, a four-row
table of the positivity literatures with entry points, three one-line redirections, and a plain
statement that no experiments were run. It also records, unprompted, that **its own Q4 search was
wrong on the first pass** — it shared my prior that positivity was the thin spot and found the four
literatures only after deliberately searching against itself. So its 1-in-3 estimate on the one
uncleared corner came from a process that had already failed in that exact area, and should be
discounted rather than treated as calibrated. If anyone revisits it, the first move is a
positive-realization search with a bandwidth constraint, not experiments.

That is the second time today a thread has argued against its own conclusion in writing. It is the
behaviour most worth keeping from this session's method.

## F36 — ⛔⛔ Phase 2 is closed: the price of exact compression is 100% of the state
M1's result, verified here, and it ends the phase by argument rather than exhaustion.

**Exact depth × (N/M) = M × (N/M) = N, at every M from 2 to 12** — every row exactly 1.000.
Storing M coarse levels of N/M values each is storing N numbers. **Decimation plus exact memory
costs the entire state.**

So Phase 2's framing question — *does compression across scales have a bounded price?* — has an
exact answer, and it is the least interesting one: **the price is the whole thing.** Compression
only exists once you accept a *tolerance*, and then the depth is set by how fast the unresolved
aliases decay, which is classical Mori–Zwanzig. The question collapses onto a dichotomy with a
classical answer on each branch.

**A refinement in M1's favour.** I checked the observability stack's rank and it is *not* full:
23/24 at M=2, 21/24 at M=4, 13/24 at M=12 — the periodic FTCS operator has degenerate eigenvalues
(modes k and N−k share a symbol). So "you get back precisely the state you saved" understates it:
you pay the full N and **still do not recover the state**. Lossy, at full price.

*How I found that*: I wrote a script whose print statement asserted "the stack is full rank N"
while the data underneath said otherwise — the same error M1 caught in T5's Observation 4 ("the
prose was written against the expectation rather than the table"), committed by me two hours after
recording theirs.

**Task 3 is a tautology, measured not argued.** Decimating by 2 twice retains fine indices
0,4,8,… — the same index set as decimating by 4. Alias sets are **bit-identical** (max difference
0.00e+00) for M=4 via 2×2, M=8 via 2×2×2, M=8 via 2×4, M=6 via 2×3. Memory neither accumulates,
composes nor saturates. My framework reading — *deep hierarchies are viable only if built in small
steps* — comes out **negative**: the route is free, so small steps buy exactly nothing, and the
M ≥ 4 wall cannot be walked around two at a time.

**M1's replacement experiment failed, and reports it.** It predicted block averaging would admit
positive laws at M ≥ 4, since a box average annihilates every unresolved alias at q = 0 — clean
mechanism, wrong: **one cell differs out of 42**, and M ≥ 4 is infeasible at every r under both
restrictions. Its follow-up explanation was also wrong — exempting 97% of the band still leaves
M=4 infeasible. So **the obstruction lives at essentially every coarse mode**; the q=0 proof is
where it is *provable*, not where it lives. The general version stays conjectured.

This also closes the decimation-vs-block-averaging question flagged as load-bearing for every
number in F14/F23. **It was not load-bearing.**

**And the M ≥ 4 wall has a proof**, not just my empirical "no r admits a positive law at depth ≤ 8":
it holds at *any* depth and *any* width.

### Status
Phase 2: closed. One time-boxed item outstanding — the dispersive test (u_t = u_xxx), approved
solely because the sign condition is *proved at q = 0 and conjectured elsewhere*, is the phase's
only surviving result, and is on the published page. Either outcome is useful: confirmation makes
it quotable on a second system class; refutation means pulling it back to q = 0 before anyone
builds on it.

## F37 — T2 withdraws its own result, and my reconciliation with it
**T2 audited its own experiment and retracted the 78.1×.** Three faults, all found by T2:
1. **Inadequate reference** — fine-FTCS at N=4096 is only good to 1.4e-7 against a spectral
   reference, *at or above* the positive arm's error. Exactly the failure mode I hit three times
   today, independently arrived at.
2. **Mechanism refuted by direct test** — actual amplification of the composite operator is
   ‖A^nb‖∞ = **1.0003** at nb=16 and **1.0026** at nb=32, with ‖w‖₁ = 1.53 per step. **No
   compounding.** And the constant-α control shows the same effect, so **my translation-invariance
   reconciliation is not needed and is probably wrong** — the ℓ¹ bound is loose in both settings,
   and T5's looseness result generalises.
3. **Unfair comparison** — 7 equations in 17 unknowns leaves a 10-dimensional feasible set, and it
   compared an arbitrary positive *vertex* against an arbitrary min-2-norm point. The spread across
   choices of point at **fixed rows** is 12–60×, as large as the claimed effect. That is F30 turned
   into a critique of its own experiment.

**Under a matched rule** (min ‖w‖₂ subject to moments, with and without w ≥ 0) positivity still
helps — 6.1× at nb=16, **6746×** at nb=64 (constant α), 11.8× (variable α) — but with a different
mechanism: as nb grows, σ shrinks at fixed m, the moment rows under-determine the weights ever more
badly, the unconstrained solution develops an oscillatory component, and **w ≥ 0 cannot oscillate**.
T2's reading: *positivity is a regulariser on an under-determined moment system, not a stability
certificate* — and it reports min-‖w‖₁ signed within 3× at nb=32, so the benefit is not unique to
positivity.

**My check disagrees with the 3×, and the resolution is more interesting than either version.**
At identical rows, four choices of point in the same solution set:

| k | s | pts | LP vertex | **maxent** | min-L2 signed | min-L1 signed | maxent / best signed |
|---|---|---|---|---|---|---|---|
| 40 | 4 | 49 | 8.6e-6 | **5.3e-8** | 3.9e-5 | 1.1e-5 | 209× |
| 40 | 6 | 73 | 9.9e-6 | **7.4e-12** | 1.2e-4 | 4.7e-5 | 6.4×10⁶ |
| 160 | 6 | 145 | 2.1e-3 | **1.5e-10** | 1.8e-3 | 1.1e-3 | 6.8×10⁶ |

But the honest reading is **not** "positivity wins". Note the LP vertex is *positive and terrible*,
min-L2 is *signed and terrible*, maxent is *positive and excellent*. **Positivity alone predicts
nothing; the choice of point predicts everything.** And maxent's form w ∝ exp(λ·rows) is
automatically non-negative — positivity is a *consequence* of maxent, not a separable constraint —
so this test cannot cleanly attribute the gain to positivity as such.

**What is robust across both of us**: at fixed rows the choice of point in the solution set spans
six orders of magnitude, and maximum entropy is far the best. That is stronger and more useful than
either "positivity is a certificate" or "positivity is a regulariser", and it supersedes both.

**Coverage question resolved**: T2 discards any configuration where *any* point fails, so all its
numbers are at **100% coverage**. My 57% run was not the same experiment, which alone explains why
I could not reproduce it.

**T2's own summary, worth recording verbatim in substance**: it has now been wrong twice on this
sub-point in *opposite* directions — first "positivity is free", then "positivity is load-bearing
via ‖w‖₁" — and both times it inferred a mechanism from an uncontrolled comparison. That is the
same error I made about mixing-vs-sign, and the same one M1 caught in T5's prose. Three threads,
one failure mode: **stating the mechanism before controlling the comparison.**

## F38 — ⭐⭐⭐ The falsification run: accuracy and coarse-grainability are in tension
M1's last experiment, 13 circulant systems × M ∈ {2,3,4} = 39 cases, three hypotheses on trial.
**Only 7 of 39 admit an exact non-negative coarse law, and M = 4 admits none for any system.**

| hypothesis | whose | verdict |
|---|---|---|
| **H_gap** — scale separation makes coarse-graining cheap | my original brief | **falsified and inverted** |
| **H_spread** — p\* is monotone in alias spread | my reframing | **falsified** |
| **H_sign** — every unresolved alias needs g ≤ 0 | M1's obstruction | **survives**, 0/39 counterexamples |

**H_spread**: feasible spread range [1.40, 2.00], infeasible [0.52, 4.80] — complete overlap, and
non-monotone *inside* the feasible set (spread 1.5588 → P=6, 1.7321 → P=3, 1.8000 → P=2). The
spread is a fine measure of how much the operator distinguishes what the coarse grid cannot; it
simply is not what prices positivity. My last redirection to M1 was wrong.

**H_sign** is **necessary but not sufficient** — all 16 cases with an offending alias are
infeasible as predicted, but 15 further cases have no offending alias and are still infeasible.

### The headline, verified here independently

| system, M=2 | worst unresolved g | feasible? |
|---|---|---|
| two-timescale 0.95/0.05 (best separation) | +0.050 | **no** |
| **exact heat semigroup** | +0.635 (any τ) | **no** |
| FTCS r=0.10 | +0.600 | **no** |
| **FTCS r=0.45 (crudest)** | **−0.800** | **yes** |

**The exact heat semigroup — the most accurate diffusion operator there is, every eigenvalue real
and positive — admits no non-negative coarse law at any M, any depth, any τ**, because
exp(−αk²τ) > 0 always. The crude FTCS stencil, whose high modes overshoot to g = −0.8, does.

> **Accuracy and coarse-grainability are in tension.** T5 found the only operator you can
> coarse-grain *for free* is one that has already discarded what the coarse grid discards; M1 finds
> the only operator you can coarse-grain *positively* is one that is crude in a specific way — its
> discarded modes must oscillate. Two independent routes, same conclusion.

### A theorem, and it is the cleanest thing the programme produced
For |g| = 1 every term of Σ_jk b_jk e^{ikθ_c} g^{−j} = 1 has unit modulus, so a convex combination
equals 1 only if **every atom** equals 1. Hence a unitary propagator admits a non-negative coarse
law only if it is an exact lattice translation:

> **Non-dissipative dynamics cannot be positively coarse-grained at all.**

### M1's own scorecard: 2 of 4
Confirmed: scale separation makes positivity *harder*; p\* decreases as r increases.
**Falsified: advection helps** (Pe=0.5 matches pure diffusion; Pe=2 infeasible at every M).
**Falsified: dispersive is the easy case** — infeasible at M=2,3,4, unitary and damped alike.
**My original brief was right and M1 was wrong**; it had applied a necessary condition as if it
were sufficient, and says so.

### Another instance of the vertex problem
B₁ = 0 exactly in every non-negative law M1 found — that half of F23 is confirmed. But **B₂ =
0.6917·I, not zero.** The LP objective is degenerate (every feasible non-negative law has ‖b‖₁ = 1
exactly), so the solver returns an arbitrary **vertex of a feasible set**. So "pure delay" is a
property of *a particular vertex*, not of the closure — F30's lesson for the third time. My reach
argument explains why B₁…B_{M−1} must be **diagonal**, which is consistent with B₂ = 0.6917·I and
is a strictly weaker claim than vanishing.

### Do not quote
- M=2 at r=0.25 is **undetermined, not feasible** — defect 3.6e-10, inside M1's stated noise floor,
  and exactly the degenerate boundary case (g = 0) the obstruction does not cover.
- The adv-diff Pe=8 row is von Neumann unstable (max|g| = 2.44) and is a labelled control only.

### Status: the programme's experiments are closed
Durable output of Phase 2: **one sentence** — coarse-graining preserves a maximum principle when
the discarded modes oscillate and destroys it when they merely decay — and **one theorem** —
non-dissipative dynamics cannot be positively coarse-grained. Everything else came back negative,
tautological, or classical. The remaining M ≥ 4 infeasibility is a **proof gap** (the all-q
obstruction is conjectured), which is a mathematics task, not a compute task.

## F39 — ⛔ The published sentence was a false biconditional; corrected
M1 flagged this as urgent and was right. The page carried:

> ~~"Coarse-graining preserves a maximum principle when the discarded modes oscillate, and destroys
> it when they merely decay."~~

The first clause asserts sufficiency and **it is false**. Dispersive propagators are as oscillatory
as it gets — unitary, |g| = 1, entirely non-decaying — and are infeasible at M = 2, 3 and 4, damped
and undamped alike. **Oscillation is necessary, not sufficient.** The tested form is one-directional:

> **A discarded mode that decays monotonically — a real eigenvalue in (0,1) — destroys the coarse
> maximum principle, at every memory depth and every stencil width.**

Proved at the constant coarse mode, verified 16/16 where it applies, never falsified across 39
cases and 13 system classes. The all-q version stays **conjectured**, and the block-averaging
control (R4) showed the proof's location is not the phenomenon's location. Page corrected to v9
with the retraction visible; M1 struck it from its own files so it cannot be re-quoted.

Worth noting the sequence: M1 ran the dispersive case, found it refuted its own prediction, and
then told me the sentence *I* had published was half wrong — applying the criterion I had stated
when clearing the experiment ("refutation means pulling it back to q=0"). That is the thread
holding me to my own stated standard, which is the behaviour the method was for.

## F40 — T3 closes: the answer is no, and the fallback does not hold either
**Does the jet/moment route contribute anything once weak-form SINDy supplies the coefficients? No.**
Rollout error is truncation-dominated below |Δc|/c ≈ 3e-3; above it the sensitivity is linear
(d log err / d log|Δc| = 0.898). WSINDy sits at **3.4e-4 even at 1% noise** — below threshold
everywhere.

> Where the jet route is good enough to matter, it does not matter; where it would matter, it is
> the one that fails.

At η ≤ 1e-4 the *true* coefficients sometimes give a worse rollout (6.68e-4) than estimated ones
(3.88e-4).

**Positivity is worth 0.16%.** Certified vs uncertified with true coefficients: 6.6807e-4 vs
6.6913e-4. Certified-with-wrong never beat uncertified-with-truth at any drift.

**The ‖w‖₁ correction is confirmed and stronger than I stated: 444 levels at ‖w‖₁ = 9.2, no
divergence.** T3 annotated rather than silently edited its earlier 37.3% figure, which was von
Neumann on uniform one-step stencils — sharp there, and it does not transfer.

**The certificate never misleads**: max|u| pinned at 1.2942 ≤ 1.2989 at every drift. A convex
combination cannot amplify, so the stability claim stays true *while certifying the wrong
operator*. At η = 1e-2 it **withdrew rather than lied** (α̂ 39% low breaks feasibility; c error
alone never did, even at 30%).

**T3 retracts its own Part II §5 on my suggestion, and then declines the credit for it.**
`solve_maxent` overturns "moment rows are the wrong tool" — max|w_ME − w_K| = **7.2e-8**, i.e.
max-entropy reconstructs the Gaussian propagator from moments alone, beating its min-norm QP by
22,000×. **But the gain is not positivity**: the LP vertex is equally non-negative and is the worst
of three at large spread; everything sits within a factor of 3; at small spread non-positive
min-norm beats maxent; and maxent fails on 2–6% of feasible geometries, where the LP fallback is
the worse point. That independently confirms F37's reading — *the choice of point predicts
everything, positivity alone predicts nothing.*

`manuscript_outline.md` withdrawn. The one salvageable result is unrelated to integrators: the
non-locality detector α_eff(R) ∼ R^(2−2s), recovering the fractional order to 0.02 absolute. A short
methods note needing real data and a comparison against MSD-scaling estimators — and it should not
carry this thread's framing.

## F41 — Two standing checks, from three threads' worth of the same error
Both are failure modes where **the output looks like a result**.

1. **Never write the conclusion into the output text.** Read it off the table instead. M1 caught
   this in T5's Observation 4 only because it ran the script before writing the criticism and the
   table contradicted the paragraph above it. I then committed the identical bug two hours later
   (asserting "the stack is full rank N" while the data said 23/24).
2. **A degenerate objective makes every feasible point optimal**, so the solver returns an
   arbitrary vertex and *any structure you see in it is an artefact of the pivot rule.* The check
   is cheap: **re-solve with a perturbed or randomised objective and see which features survive.**
   One root cause behind three separate threads — M1's B₂ = 0.6917·I vs T5's B₂ = 0 (F38),
   `solve_positive` returning a useless 3-non-zero stencil (F30), and an uncontrolled positivity
   comparison (F37). Credit: M1 for the general statement.

A third, from my own record: **before running a comparison, measure the noise floor of the
apparatus and check the expected effect exceeds it** (F29) — which T2 arrived at independently
when it withdrew its own 78.1× (F37).

## F42 — ⭐⭐ T3 reopens: high order on scattered mixed time levels works, and F30 needs scoping
T3 found its Part III verdict was measured in the one configuration where it could not hold —
**3 rows with 6 neighbours**, where the estimator provably cannot matter — and rebuilt it.

**The construction.** From the generating function Σᵢwᵢ exp[s·Aᵢ + s²·Bᵢ] = 1 with A = Δx − cΔt,
B = αΔt, the order-q row is Σⱼ A^(q−2j)Bʲ / ((q−2j)! j!). This handles **arbitrary order on mixed
time levels**, which neither existing routine can: `rows_moment` needs a single time level and
`rows_taylor` is capped at p = 2. Verified here: **it reduces to the F8-corrected row at p = 2
exactly (0.00e+00)**.

**Verified independently, scattered points at mixed time levels:**

| order p | points | positive-feasible | median error | maxent vs LP |
|---|---|---|---|---|
| 2 | 12 | 78.8% | 4.35e-7 | 1.6× |
| 4 | 15 | 60.0% | 2.95e-11 | 1.0× |
| 6 | 21 | 71.6% | **1.11e-15** | **0.5×** |

Eight orders of accuracy from p=2 to p=6, and — the part both T3 and I expected to fail —
**positivity survives on scattered geometry at 6th order.** T3 measures 99.2% feasibility on its
geometry distribution; I get 71.6% on a deliberately wider, more adversarial one. Either way it
does not die.

### ⛔ Scope limit on F30, from my own measurement
The last column reverses. At p = 6 the **LP vertex beats maxent (0.5×)** on scattered mixed-level
geometry. F30's headline — maxent over the LP vertex by up to 1.3×10⁶ — was measured on **uniform,
single-time-level, wide stencils at p = 2**, where the feasible set is enormous (3 rows, 145
points) and the vertex is pathological. At high order the feasible set is small, the vertex is no
longer extremal in a damaging way, and entropy-maximisation is not what minimises truncation.
T3 reports the same from the other side: the LP vertex degrades as the stencil widens while maxent
stays flat, but at p = 2 everything is floored, capping the gap at ~10× in *their* regime.

**So the rule is narrower than I wrote it**: prefer maximum entropy when the moment system is badly
under-determined (few rows, many points). At high order with few excess degrees of freedom, check
rather than assume.

### The finding worth keeping
At p = 6 the propagator route hits the truncation floor exactly and beats weak-form SINDy **50× on
clean data**, 3.3× at η = 1e-6 — but WSINDy wins from η ≥ 1e-4, and **no estimator is accurate
enough to exploit p = 6 beyond that** (WSINDy sits 30× above the floor). Hence:

> **Data-driven integrators are coefficient-limited, not order-limited.**
> Match the consistency order to the coefficient accuracy you can actually get.

That is a clean design rule, it is new, and it is the right note for this programme to end on —
it says exactly where the effort should go, which is not into higher-order rows.

**And one correction to me.** I told T3 its NNLS result was the vertex phenomenon. It checked:
NNLS returned **163–167 non-zeros out of 171** — not sparse, so my mechanism was wrong. The vertex
effect is real but belongs to the *equality-constrained* problem (3 rows ⇒ ≤3 non-zeros), not to
their overdetermined regression, where NNLS fails because a bound constraint picks a worse point
from a huge null space at cond ≈ 1e17. My *conclusion* was right — re-run with an interior
parametrisation (softmax) and positivity helps under noise, 2.19e-4 vs 4.44e-4 at η=1e-4 — but for
a different reason than I gave. Fourth instance today of right conclusion, wrong mechanism.
