# manuscript_outline.md

## Read this first (REVISED after round 2)

**My earlier recommendation was "don't write the paper". That has changed.** The
constant-coefficient story is still a rediscovery of Runge-Kutta-Chebyshev, and I stand by
that. But the variable-coefficient test the team lead asked for came back positive, and it
is the one regime where neither RKC nor exponential integrators nor an FFT solve applies.
That is a paper, with a much narrower and more honest claim than the original one.

**Scope, which must sit on the headline number.** The advantage is a function of
`L_alpha = alpha/|alpha'|` measured IN CELLS -- the validity range of the local Taylor rows:
~2400x over FTCS at `L_alpha >= 20` cells, 27x at 11, gone by 6, and actively HARMFUL below
~2 (a tanh step of width 0.005 loses to FTCS by 100x). The periodic runs also had no
boundaries; with Dirichlet walls charged honestly the measured result is 148x accuracy at
0.82x cost versus FTCS -- but that is not order-matched, so it is not comparable to the
7-20x figure. Any abstract must carry the `L_alpha` scope.

**The result that carries it.** On `u_t = d_x(alpha(x) d_x u)` with `alpha_max/alpha_min = 9`,
a stencil built from purely local data -- the Taylor coefficients of `alpha` at the point,
exponentiated on a polynomial basis, then projected onto non-negative weights -- beats every
competitor I could construct, including `RK4 + 6th-order finite differences` (4th order in
time, 6th in space) by **6.7-20x at every tolerance from 1e-3 to 1e-7**, while additionally
being monotone. It reaches 1e-6 and 1e-7 where standard FTCS and RKC1 do not arrive at all.

The structural reason is worth stating plainly because it is the actual contribution:
**the construction obtains high order in space and in time simultaneously from a single
local solve**, whereas stabilised Runge-Kutta methods separate the two and must pay
stability to buy temporal order. RKC1 in fact gets *worse* when driven by a higher-order
spatial operator, because its spectral radius grows while its temporal order stays at one.

---

## (A) The defensible paper

**Title.** *Monotone high-order explicit stepping for variable-coefficient diffusion via
local propagator moments*

**Venue.** *JCP* or *SINUM* as a full paper — the variable-coefficient result carries it.
(My round-1 recommendation of a short note assumed the constant-coefficient case was all
there was.)

**Primary claim.** For `u_t = d_x(alpha(x) d_x u)`, matching the moments of the *locally
exponentiated* operator with non-negative weights yields a scheme that is simultaneously
(i) high order in space, (ii) high order in time, (iii) explicit with a step far above the
parabolic CFL, and (iv) monotone — and it beats order-matched RK4 + FD6 by 7-20x on
work-precision. No closed-form propagator is used anywhere.

**Supporting results (all proved or measured in this thread).**

1. For the linear constant-coefficient advection–diffusion operator, the set of `(m,k)` for
   which a **monotone** (convex-combination) explicit stencil of half-width `m` advances
   `kΔt` in one shot is *exactly*
   `ψ(k·Co) ≤ 2kν ≤ m²`, where `ψ` is the lower convex envelope of `j²` on `ℤ`.
   The proof is two lines (extremal moments of a lattice probability measure) and the
   constant is exactly 1, not asymptotic. LP-verified for `m ≤ 25`.
2. The upper bound `2kν ≤ m²` **coincides with the RKC1 stability limit**, and at equality the
   RKC1 operator *is* the convex combination `½(δ_{−m}+δ_{+m})`. This gives a real-space proof
   that RKC1 is L∞-monotone at its stability limit — stronger than the usual `|R_s| ≤ 1`
   L²-statement, and, as far as I can establish, not stated in this form. Strictly below the
   limit the Chebyshev stencil is *not* in general a non-negative measure while the LP
   construction always is, so the two families separate there.
3. The lower bound `2kν ≥ ψ(k·Co)` is new content and is the interesting half: it says
   **advection caps the monotone step independently of `m`**, at `kΔt ≤ 2α/c²` — "the
   diffusive spread over a step must exceed the advective displacement over it" — and that no
   monotone stencil of *any* width exists once `ν·Pe² > 2`. Widening does not evade the
   cell-Péclet barrier. The `ψ`-vs-`μ²` gap `f(1−f) ≤ ¼` is a genuine lattice effect, verified
   by LP.
4. Accuracy along the frontier: `LTE/(kΔt) = (α/6)(mΔx)²|u_xxxx|` for pure diffusion
   (measured slope 1.978, `C = 0.0156` vs `α/6 = 0.01667`); with advection the dropped `u_tt`
   term dominates by `(3/2)(c/απ)² = 15.2`, and is removable by correcting the second-moment
   row to `2kν + (kCo)²`. The order-`2p` frontiers are `m²/3` (4 moments) and
   `m²/(3/(3−√6))` (6 moments), both confirmed numerically.

**Explicit non-claims, stated in the abstract.** Quadratic step growth with footprint is
classical (RKC/STS) and is *not* claimed as new. The scheme is not competitive with an FFT
solve on problems where an FFT applies. The method needs the BVP Green's function for
boundaries and therefore does not extend to variable coefficients or general geometry.

**Figures (all already produced).**
1. `fig1_frontier.png` — LP frontier vs closed form; ratio panel; Péclet sweep showing the
   `m`-independent advection cap and the `ν·Pe² = 2` cutoff.
2. `fig2_truncation.png` — `(mΔx)²` law with the `α/6` constant; symbol error vs `θ`.
3. **New panel needed** — the `ψ` envelope vs `μ²`, with the LP-measured `k_max` where they
   differ (data exist in `task1b_theory.py`, not yet plotted).
4. `fig3_workprecision.png` — but **re-framed**: add RKC1 and RKC2 curves, and label the
   frontier curve "= RKC1". Without RKC on this plot the figure is misleading.
5. `fig4_adaptive.png` — optional; the useful content is the negative (adapt wide-where-rough,
   and even then uniform width wins).

**Sections.** 1. The monotone `(m,k)` region, exactly. 2. Relation to RKC1 (identity proof +
Table C of `task6_rkc.py`). 3. The advective lower bound and the Péclet cutoff. 4. Truncation
along the frontier. 5. Numerical confirmation. 6. Limitations (boundaries; linear
constant-coefficient only).

**Honest assessment of strength.** The variable-coefficient result (new section, see below)
is the paper. The constant-coefficient material becomes the *analysis* section: it is where
the frontier can be derived exactly and where the RKC identity must be disclosed up front.

**Required new section — variable coefficients.** Local rows from the Taylor coefficients of
`alpha`; feasibility of non-negative weights as a function of `(P, sigma, m)` (matching more
moments needs a wider stencil, and below `sigma ~ 1.4` cells no positive stencil matches 5+
moments at any width -- Godunov); work-precision against FTCS, RKC1+FD2/4/6, and RK4+FD4/6.
Figure: `fig7_varcoef.png`.

**Still missing before submission** (do not skip):
* **RKC2 and higher-order stabilised methods** as competitors. I tested RKC1 thoroughly and
  RK4 as a temporally-accurate control, but not RKC2/ROCK2/ROCK4, which are the methods a
  referee will name first. My RKC2 implementation here is indicative only.
* **A defence of the `L_alpha` scope.** The natural referee question is whether the method
  is just "high-order FD where the coefficient is smooth". Task 13b answers half of it --
  positivity is load-bearing, not decorative (signed min-norm rows are up to 60x worse, and
  the gap widens with step count as `||w||_1^nb`) -- but the framing needs care.
* **Nonlinearity** (e.g. porous-medium or a reaction term). Variable coefficients are the
  first step; nonlinear is where the claim would really bite, and where monotonicity earns
  its keep.
* **2-D**, where wide stencils cost `O(m^2)` per point rather than `O(m)` and the whole
  cost argument has to be redone. This could reverse the verdict and must be checked.
* **Boundaries for variable coefficients** -- PARTLY DONE (Task 14a). A one-sided positive
  stencil does exist from j = 1; what degrades is the achievable moment order, and only
  over a 4-6 cell layer (P_max = 1,2,4,6,6,... vs distance). Measured hybrid: 148x accuracy
  at 0.82x cost vs FTCS with walls charged. STILL MISSING: the order-matched RK4+FD6
  competitor with walls, and the cheaper order-reduction wall treatment (I priced the
  FTCS-substep one, which is the more expensive of the two).
* **Off-centre stencils in the variable-coefficient scheme** (T5's result, verified in
  Task 11). A stencil centred on the target point must span both the origin and the
  departure point, which caps `k_max` linearly in `m` once `m > 1/Pe_cell`; re-centring on
  the departure point restores `m^2/(2 nu)` at every `m` for free. My constant-coefficient
  solver dodged this with a gauge/Liouville transform, which does not exist for variable
  coefficients -- so the variable-coefficient scheme must use the off-centre form as soon
  as it carries advection. This is a required change, not an optimisation.

---

## (B) The over-claiming version, and why it fails

> *"Beating the parabolic CFL limit with wide positive explicit stencils: 5000× speedups
> over FTCS."*

Three referee objections, all fatal, all confirmed in this thread:

1. **"This is RKC1."** Same Δt, same footprint, same cost, agreement to 2×10⁻¹⁶
   (`task6_rkc.py` §B). RKC2 gets 0.65·m² with *second*-order temporal accuracy, which is
   strictly better for the same footprint on any problem where accuracy matters.
2. **"Your speedup is the exponential integrator, not the stencil."** Every Pareto-optimal
   configuration has `s = 3–6`, never `s ≈ 1`. At `s ≈ 1` the scheme lies *on* the
   Crank–Nicolson curve, and the control experiment shows it reproduces coarse-grid FTCS at
   `7–15×` the cost (it is `m` decoupled sub-lattice solves). The win comes from replacing the
   difference operator with the exact semigroup kernel, and a DST solve does that better.
3. **"Your boundary treatment is the exact Green's function."** Zero-padding costs
   `10³–10⁴×` in error; sub-stepping needs a wall layer of `m + k` cells, which overruns the
   domain by `m = 24`. The working method uses odd reflection plus a gauge/Liouville change of
   variable — available only because the problem is constant-coefficient on an interval, i.e.
   exactly the case where you would have used an FFT.

---

## (C) Related work that must be engaged (do not skip any of these)

* **Runge–Kutta–Chebyshev / super-time-stepping.** van der Houwen & Sommeijer; Verwer,
  Hundsdorfer & Sommeijer (RKC); Alexiades, Amiez & Gremaud (STS); Meyer, Balsara & Aslam
  (RKL). *This is the primary comparison and belongs in the abstract, not the related work.*
* **SSP / monotonicity theory.** Gottlieb, Ketcheson & Shu; Ketcheson's `C_SSP ≤ s` bound.
  Needed to explain why quadratic monotone growth here does not contradict the linear SSP
  barrier: SSP constrains convex combinations of forward-Euler steps of a *fixed* operator and
  must hold for all problems satisfying the forward-Euler condition, whereas here the spatial
  footprint widens and the argument is specific to the linear constant-coefficient operator.
* **Wide-stencil monotone schemes for degenerate elliptic operators.** Motzkin–Wasow;
  Oberman; Froese & Oberman. Same positivity device (a positive lattice measure matching
  prescribed moments), different purpose (Barles–Souganidis convergence for Monge–Ampère/HJB).
* **Exponential integrators and kernel/semigroup methods.** Hochbruck & Ostermann; Greengard &
  Strain's fast Gauss transform; real-space heat-kernel convolution. This is what the accurate
  (`s ≈ 4–6`) regime is.
* **Semi-Lagrangian schemes.** The first-moment condition `M1 = −k·Co` is exactly "centre the
  stencil at the departure point"; the positivity of interpolation weights is a classical
  concern there (Staniforth & Côté).
* **Lattice random walks / probabilistic schemes.** A positive stencil summing to one *is* a
  random-walk transition kernel; the frontier condition is "the walk's variance per step
  cannot exceed `m²`".

---

## Recommended next step for the wider project

Pursue the paper, but on the variable-coefficient claim only, and run the missing
comparisons above first (RKC2/ROCK, nonlinearity, 2-D cost) before writing. The
constant-coefficient transferable results are:

* the exact monotone region `ψ(kCo) ≤ 2kν ≤ m²` and its two physical readings (width ≥ kernel
  σ; diffusive spread ≥ advective displacement) — a **hard constraint an RL agent can be given
  for free** rather than made to discover;
* the design rule `s* = sqrt(2 ln(1/ε))`, which is the one genuinely interior optimum found in
  this thread and the only sensible thing to ask an agent to learn;
* the negative result that `(m,k)` has no interior optimum, so any RL formulation over that
  action space will be driven to a boundary and will learn nothing interesting;
* **and the one place an RL agent could add real value**: in the variable-coefficient case
  the feasible `(P, m, k)` region depends on `alpha` and its derivatives locally, the
  feasibility boundary is not available in closed form, and the optimal local choice varies
  in space. That is a genuine decision problem, unlike the constant-coefficient one.
