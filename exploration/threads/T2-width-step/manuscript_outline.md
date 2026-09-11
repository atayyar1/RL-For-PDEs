# manuscript_outline.md

## Read this first

The work-precision answer in Task 3 is **yes** — the wide positive explicit stencil beats
CFL-limited FTCS by up to 5×10³ and Crank–Nicolson by up to 2×10² in flops at equal accuracy.
The brief says to draft a manuscript outline if the answer is yes, so here it is.

**But my recommendation is not to write the obvious paper.** Task 6 established, to
round-off, that the scheme at its positivity frontier is *first-order Runge–Kutta–Chebyshev
with `s = m` stages* — the same polynomial of the same operator, the same `Δt = m²Δx²/(2α)`,
the same footprint, the same cost. And the regime that actually produces the headline
speedups (`s = m/σ ≈ 4–6`) is a discretised heat kernel, i.e. a real-space exponential
integrator, which a DST/FFT exact solve still beats by 2–10×. A paper claiming "large
explicit time steps from wide positive stencils" would be a rediscovery, and a referee who
knows RKC would say so in the first paragraph.

Below: (A) the paper I think is *defensible*, narrow but honest; (B) what the over-claiming
version would look like and why it fails; (C) the related work that must be engaged.

---

## (A) The defensible paper

**Title.** *Positivity-preserving super-time-stepping: a real-space certificate for
Runge–Kutta–Chebyshev, and its exact frontier*

**Venue.** A short note — *BIT*, *JCP* Short Note, or *Applied Mathematics Letters*. Not a
full-length methods paper; there is not enough new material for one.

**Claim (deliberately narrow, and all of it is proved or measured here).**

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

**Honest assessment of strength.** Item 3 is the only unambiguously new mathematics, items 1–2
are a clean re-derivation with a genuine strengthening (L∞ over L²), item 4 is textbook-adjacent.
That is a note, not a paper. If T1/T3/T5 in the wider project turn up a nonlinear or
variable-coefficient extension of the positivity certificate, items 1–3 become the technical
core of something larger and should be held back until then.

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
   domain by `m = 24`. The working method uses odd reflection plus a Cole–Hopf change of
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

Do not pursue the scheme paper. The transferable results are:

* the exact monotone region `ψ(kCo) ≤ 2kν ≤ m²` and its two physical readings (width ≥ kernel
  σ; diffusive spread ≥ advective displacement) — a **hard constraint an RL agent can be given
  for free** rather than made to discover;
* the design rule `s* = sqrt(2 ln(1/ε))`, which is the one genuinely interior optimum found in
  this thread and the only sensible thing to ask an agent to learn;
* the negative result that `(m,k)` has no interior optimum, so any RL formulation over that
  action space will be driven to a boundary and will learn nothing interesting.
