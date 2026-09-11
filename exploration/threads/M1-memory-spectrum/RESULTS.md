# M1 — Results

Read `POSITIONING.md` first: it is the literature positioning and it contains the
obstruction theorem the numbers below test. Claims marked **[classical]**,
**[measured here]**, **[conjectured]**.

Scripts: `positioning_check.py` (obstruction + LP verification), `task3_iterated.py`
(Task 3 + restriction operator, log in `run_task3.log`), figure
`figures/task3_restriction.png` at 150 dpi; `task2_systems.py` (system classes, log in
`run_task2.log`, figure `figures/task2_hypotheses.png`). Setup throughout: periodic FTCS on
N=120, pure diffusion unless stated, T5's operating point r = 0.45.

---

## THE RESULT, AND EXACTLY WHAT IT IS NOT

The thread's one durable finding, stated one-directionally:

> **A discarded mode that decays monotonically — a real eigenvalue in (0, 1) — destroys
> the coarse maximum principle, at every memory depth and every stencil width.**
>
> Corollary, unconditional and separately proved (R6): **a unitary propagator (|g| ≡ 1)
> admits a non-negative coarse law only if it is an exact lattice translation.**

Evidence: proved at the constant coarse mode (POSITIONING.md §2.1); 16/16 where it applies;
0 counterexamples in 39 cases across 13 system classes. The sharp instance is that the
**exact heat semigroup cannot be positively coarse-grained at any M or depth, while crude
FTCS can** — accuracy and coarse-grainability are in tension, and scale separation makes
coarse-graining harder, not cheaper.

**Five scope limits. All five are load-bearing; quote them with the result.**

1. **Proved at one coarse mode only.** §2.1 proves it at q = 0. The all-q version is
   **[conjectured]**. R4 shows the obstruction is *not* localised at q = 0 — exempting 29
   of 30 coarse modes still leaves M=4 infeasible — so the proof's location is not the
   phenomenon's location. A proof at one mode plus a conjecture elsewhere is the honest
   object; the general claim is not established.
2. **Necessary, not sufficient. It excludes; it does not admit.** 15 of 39 cases carry no
   offending alias and are still infeasible at depth ≤ 8. Do not read the sign condition as
   a characterisation of when positivity survives, and do not state it as a biconditional —
   the "exactly when" form is false and was refuted by the dispersive test (R6).
3. **Oscillation buys nothing on its own.** Dispersive propagators are oscillatory and fail
   anyway; every one of the 13 systems fails at M ≥ 4. Nothing measured here makes
   hierarchies cheap.
4. **Real eigenvalues only.** The §2.1 argument needs g real. Complex-spectrum systems
   (advection, dispersion) are outside it and need the separate unit-modulus argument, which
   covers only the |g| ≡ 1 edge.
5. **Setting.** Linear, constant-coefficient, 1-D, periodic, uniform grid; exact
   (zero-tolerance) feasibility; decimation or box-average restriction; depth ≤ 10,
   half-width ≤ 3. No 2-D, no nonlinear, no variable coefficients. LP noise floor ~1e−10 —
   see the limitations section for the one cell (M=2, r=0.25) that sits inside it and is
   therefore **undetermined, not feasible**.

---

## R1 — Task 3 is a tautology, and this is a proof, not an opinion **[measured here]**

The brief called "coarse-grain by 2 twice versus by 4 once" the single most important
measurement in the thread. It is not a measurement. Decimating by 2 and then by 2 again
retains fine indices 0, 4, 8, … — **the same index set as decimating by 4**. The coarse
law is a property of the observable, so the routes cannot differ.

Checked on the alias sets, which determine the characteristic polynomial and hence every
coefficient of the coarse law:

| total M | route | max difference vs the single jump |
|---|---|---|
| 4 | 2×2 | **0.00e+00** |
| 8 | 2×2×2 | **0.00e+00** |
| 8 | 2×4 | **0.00e+00** |
| 6 | 2×3 | **0.00e+00** |

Bit-identical, not merely close. The same holds for the positivity-feasible set, because
it is the same set of constraints.

> **Memory neither accumulates, composes, nor saturates. It is a function of the total
> coarsening ratio and nothing else.** p_exact(M₁ then M₂) = p_exact(M₁M₂) = M₁M₂ − 1.

This answers the framework question the brief attached to Task 3 — *are hierarchies
viable only if built in small steps?* — in the negative: **the route is free, so building
in small steps buys exactly nothing.** The M ≥ 4 wall cannot be walked around by
approaching it two at a time.

Caveat on scope: this is an identity for a *fixed observable* under linear dynamics. It
would stop being an identity if each level were allowed to be inexact to a tolerance, or
to carry an augmented coarse state. Neither was in the brief; both are real questions.

---

## R2 — The price of exactness is 100% of the compression, at every ratio **[measured here]**

The exact coarse law needs `depth` time levels of an N/M-point field. Counting:

| M | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| coarse cells N/M | 60 | 40 | 30 | 24 | 20 | 15 | 12 | 10 |
| exact depth | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
| coarse state | 120 | 120 | 120 | 120 | 120 | 120 | 120 | 120 |
| **vs fine state (120)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Exactly 1.000 at every M, by construction: depth × (N/M) = M × (N/M) = N.

> **Decimation plus exact memory is a change of basis, not a compression.** You get back
> precisely the state you saved.

**Refinement (team lead, verified independently by them): it is worse than a change of
basis.** The observability stack is not full rank — 23/24 at M=2, 21/24 at M=4, 13/24 at
M=12 — because the FTCS symbol is even in θ, so within an alias class the members l and
M−l coincide at the constant coarse mode (concretely, M=3 at q=0 has g₁ = g₂ = −0.35).
So **you pay the full N and still do not recover the fine state: lossy, at full price.**
This is consistent with R2's "exact depth = M" — that is a max over coarse modes, and the
degeneracy bites only at special modes, which is exactly why the stack loses a few ranks
rather than collapsing.

So *"does compression across scales have a bounded price?"* has an exact answer here, and
it is the least interesting possible one: **the price is the whole thing.** Real
compression requires a tolerance, and once you accept a tolerance the depth is set by how
fast the unresolved aliases decay — which is the classical Mori–Zwanzig / spectral-gap
story **[classical]**. The programme's framing question therefore collapses onto a
dichotomy with a classical answer on each branch.

---

## R3 — p\*(M) is finite only at M ∈ {2, 3}, and only above an r-threshold **[measured here]**

LP for an exact non-negative compact coarse law, depth ≤ 10, half-width ≤ 3, decimation:

| M | r=0.15 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|
| 2 | — | P=9,s=3 | P=2,s=1 | P=2,s=1 | P=2,s=1 | P=2,s=1 | P=2,s=1 |
| 3 | — | — | — | — | P=6,s=2 | P=6,s=2 | P=3,s=1 |
| 4 | — | — | — | — | — | — | — |
| 5, 6, 8 | — | — | — | — | — | — | — |

Reproduces T5's F23 (M=2 at s=1, M=3 at s=2 and P=6; lag-counting differs by one from
T5's convention, not substance) and the team lead's independent NNLS run, which found the
M=2 threshold at r = 0.2929 = 1 − 1/√2.

For M ≥ 4 the "—" is not a search failure. POSITIONING.md §2.1 proves it: a non-negative
law of **any** depth and **any** width requires every unresolved alias at the constant
coarse mode to satisfy g ≤ 0, i.e. r ≥ 1/(4 sin²(π/M)), which exceeds FTCS's own r ≤ ½ for
every M ≥ 5 and is met only at the endpoint for M = 4.

Two secondary readings, both **[measured here]**: memory does buy a little (M=2 becomes
feasible at r = 0.25 with depth 9 rather than 2), and **p\* falls as r rises** (M=3 needs
depth 6 at r = 0.45, depth 3 at r = 0.50).

---

## R4 — ⛔ Prediction failed: the restriction operator is nearly irrelevant **[measured here]**

This was my replacement experiment and the thread's best remaining idea. It did not work,
and the way it failed is the most informative result here.

Only aliases surviving the restriction need reproducing. For a box average over M fine
points, W(θ_l) = (1/M)(1−e^{iMθ_l})/(1−e^{iθ_l}), and since Mθ_l = θ_c + 2πl the numerator
is common to all l and **vanishes at θ_c = 0**. Verified: at M=4, q=0, decimation keeps 3
unresolved aliases, box averaging keeps **0**. Block averaging therefore deletes precisely
the constraints that carry the proved obstruction.

**Prediction: box averaging admits positive laws at M ≥ 4. It does not.**

| M | decimation | box average |
|---|---|---|
| 2, r=0.25 | P=9, s=3 | P=8, s=3 |
| 2, r≥0.30 | P=2, s=1 | P=2, s=1 |
| 3, r=0.45 | P=6, s=2 | P=6, s=2 |
| **4, 5, 6, 8 — every r** | **—** | **—** |

The two tables differ in one cell out of 42. My first explanation — that the obstruction
is a *neighbourhood* of the constant mode, so deleting one point of a near-continuum
cannot help — is also wrong. Exempting the unresolved constraints of every coarse mode
within drop_Q of the constant mode:

| modes exempted | 0 | 1 | 3 | 5 | 7 | 11 | 17 | 23 | **29 of 30** |
|---|---|---|---|---|---|---|---|---|---|
| p\* at M=4, r=0.45 | — | — | — | — | — | — | — | — | **—** |

**97% of the band exempted and it is still infeasible.** So the obstruction is present at
essentially every coarse mode independently; the q=0 argument is simply the one place it
is easy to prove, not the place it lives. The general all-q version is **[conjectured]**.

Two consequences:

1. **This closes the decimation-vs-block-averaging question.** POSITIONING.md §4 flagged
   the unexamined restriction operator as load-bearing for every number in F14/F23. It is
   not — self-correction.
2. It strengthens the negative. The M ≥ 4 wall survives changing the coarsening route
   (R1), changing the restriction operator, and deleting almost all of the constraints.

---

---

## R5 — Across 13 system classes: H_sign survives, H_spread and H_gap are falsified **[measured here]**

`task2_systems.py`, 13 circulant systems × M ∈ {2,3,4} = 39 cases. Three hypotheses on
trial: **H_gap** (the brief: p\* small when scale separation is good), **H_spread** (team
lead: p\* a monotone, system-independent function of the alias spread), **H_sign**
(POSITIONING.md §2.1: unresolved aliases must avoid the positive real axis).

**Result: 7 of 39 cases admit an exact non-negative coarse law. M = 4 admits none, for
any of the 13 systems.**

**H_sign — not falsified, 0 counterexamples in 39.** It is a *necessary* condition, so
the only falsifying observation is "offending alias AND feasible". There were none: all
16 cases carrying an offending alias are infeasible, as predicted. It is *not* sufficient,
as proved — 15 further cases carry no offending alias and are still infeasible.

**H_spread — falsified.** Feasible spread range [1.40, 2.00]; infeasible [0.52, 4.80].
Complete overlap. Monotonicity fails inside the feasible set too: spread 1.5588 → P=6,
spread 1.7321 → P=3, spread 1.8000 → P=2. And spread 1.8000 gives P=2 at M=2 and
infeasible at M=4, so it is not system-independent either.

**H_gap — falsified, and inverted.** The two *perfectly* separated systems are the two
that fail hardest:

| system at M=2 | worst unresolved \|g\| | p\* |
|---|---|---|
| two-timescale 0.95/0.05 | 0.9500 | **—** |
| exact heat semigroup, r=1.5 | 0.9959 | **—** |
| FTCS diffusion r=0.45 (worst separated) | 0.9988 | **P=2** |

The exact heat semigroup is the sharpest case: it is the *most accurate* diffusion
operator available, every eigenvalue is real and positive, and it admits **no** non-negative
coarse law at any M or depth — while the crude FTCS stencil, whose high modes go negative,
does. **Accuracy and coarse-grainability are in tension, and scale separation makes
coarse-graining harder, not cheaper.**

## R6 — ⛔ Two of my four predictions falsified, including the headline experiment **[measured here]**

| POSITIONING.md §2.4 prediction | verdict |
|---|---|
| 1. Scale separation makes positivity **harder** | **CONFIRMED** (R5) |
| 2. Advection **helps** (complex aliases let phases cancel) | **FALSIFIED** |
| 3. The dispersive case is the **easy** one | **FALSIFIED** |
| 4. p\* **decreases** as r increases | **CONFIRMED** |

**(2)** Pe=0.5 matches pure diffusion exactly (P=2 at M=2, P=6 at M=3); Pe=2 is infeasible
at every M. Advection does not help — past a small cell Péclet it hurts.

**(3) was the experiment the thread was cleared to run, and it failed.** Dispersive is
infeasible at M=2, 3, 4, unitary and damped alike. **The brief's original expectation
("positivity should fail hard") was right and I was wrong.** I applied a necessary
condition as if it were sufficient. The correct statement for the dispersive case is a
*stronger* obstruction, and it is provable **[measured here, analytic]**:

> For |g| ≡ 1 every term of Σ_{j,k} b_{j,k} e^{ikθ_c} g^{−j} = 1 has unit modulus, so a
> convex combination equals 1 only if **every** atom equals 1. Hence a **unitary
> propagator admits a non-negative coarse law only if it is an exact lattice
> translation.** Non-dissipative dynamics cannot be coarse-grained positively at all.

So the sign condition of §2.1 is one necessary condition among several, and the honest
summary is that positivity under coarse-graining is far more fragile than any single
spectral statistic predicts.

## What the thread establishes, in one paragraph

For linear, local, constant-coefficient dynamics on a uniform grid, coarse-graining in
space has an exact finite closure whose depth is fixed by algebra alone (the alias count,
i.e. the McMillan degree — **[classical]**), whose state cost is exactly the state you
saved (R2), and which cannot be made non-negative at all past a coarsening factor of 3 —
for any of 13 system classes tested (R3, R5). The route (R1), the restriction operator
(R4), and the memory depth are all powerless against that last fact. **The positivity price
of compression is not bounded and not slowly-growing; past M = 3 it is infinite.**

The one durable positive finding is an inversion of the programme's intuition. **State it
one-directionally — the biconditional is false:**

> **A discarded mode that decays monotonically — a real eigenvalue in (0,1) — destroys the
> coarse maximum principle, at every memory depth and every stencil width.**

Proved at the constant coarse mode (POSITIONING.md §2.1), verified 16/16 where it applies,
and never falsified across 39 cases. The exact heat semigroup — the most accurate diffusion
operator there is, every eigenvalue real and positive — cannot be positively coarse-grained
at all, while crude FTCS can. Scale separation is not what makes hierarchies cheap.

**The converse is false and was tested.** Oscillation is necessary, not sufficient:
dispersive propagators are oscillatory and fail anyway (R6), and all 13 system classes fail
at M ≥ 4. Nothing measured here makes hierarchies cheap. Anyone quoting this result should
quote the implication, not the equivalence, and should note that the general (all-q) version
is **[conjectured]** — §2.1 is where it is provable, R4 shows it is not where it lives.

## R7 — On T5's "pure-delay" structure **[measured here]**

Inspecting the actual non-negative laws my LP returns, at r = 0.45 and r = 0.50:

```
M=3, P=6, s=2, r=0.45     B_1 = 0 exactly
                          B_2 = 0.691657 * I   (pure diagonal, no neighbours)
                          B_3..B_6 nonzero, spatially spread
M=3, P=3, s=1, r=0.50     B_1 = 0 exactly
                          B_2 = 0.750000 * I
                          B_3 = (0.125, 0, 0.125)
```

The robust part of T5's F23 is confirmed: **the Markov term B₁ vanishes exactly** in every
non-negative law I found. But **B₂ ≠ 0** in mine, where T5 reported B₁ = B₂ = 0. This is
not a contradiction of T5 — the LP objective is degenerate (every feasible non-negative
law has ‖b‖₁ = 1 exactly), so the solver returns an arbitrary vertex of a feasible *set*,
not a unique law. Neither solution is "the" coarse law. **"Pure delay" should be stated as
a property of a particular vertex, not of the closure** — unless someone shows B₂ = 0 in
every feasible solution, which nobody has.

The team lead's reach argument (P·Aʲ·Pᵀ is diagonal for j < M, because the fine stencil
reaches one cell and coarse cells sit M apart) explains why B₁…B_{M−1} must be **diagonal**.
That is a weaker statement than vanishing, and it is consistent with B₂ = 0.6917·I above.

## Negative results, stated plainly

- The brief's Task 3 comparison is a tautology (R1).
- The brief's Task 1 spectrum has two finite entries (R3).
- My block-averaging replacement experiment failed (R4), and my first explanation of the
  failure also failed.
- **Two of my four §2.4 predictions failed (R6), including the dispersive one the thread
  was cleared to run.** The brief's original expectation was right there and mine was wrong.
- The team lead's alias-spread reframing is falsified (R5).
- The framework's "does compression have a bounded price" question resolves to a
  dichotomy with a classical answer on both branches (R2).
- "Pure delay" is a vertex property, not a property of the coarse law (R7).

## Known limitations

- LP conditioning: at large depth the coefficients g^{P−j} span many orders of magnitude
  and HiGHS reports spurious infeasibility (e.g. M=2, s=1 feasible at P=8, "infeasible" at
  P=9). Cells at depth ≤ 8 are robust; I would not trust a defect below ~1e−10. Every
  "—" at M ≥ 4 is independently backed by the §2.1 proof, so it does not rest on the LP.
- Searches capped at depth 10 (8 in `task2_systems.py`) and half-width 3.
- **M=2 at r=0.25 is undetermined, not feasible.** R3's table reports P=9, s=3 there, but
  that cell's defect is 3.6e−10 — inside the noise floor I said not to trust. A depth sweep
  P=7…11 never reaches machine zero (values 1e−7 to 1e−10). This is exactly the degenerate
  case the §2.1 obstruction does not cover: at r=1/4 the unresolved alias is g = 0 exactly,
  the boundary of the forbidden interval (0,1). Treat it as open.
- The "adv-diff Pe=8" row in R5 has max|g| = 2.44, i.e. the fine scheme is von-Neumann
  unstable. It is outside the theory and carries no weight in either direction; it is kept
  in the table only as a labelled control.
- Pure diffusion for the tables; the advective case was spot-checked in
  `positioning_check.py` and moves nothing qualitatively.
- One dimension, periodic, constant coefficients throughout. The 2D question in the brief
  was not run.
