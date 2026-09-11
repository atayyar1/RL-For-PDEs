# M1 — Results

Read `POSITIONING.md` first: it is the literature positioning and it contains the
obstruction theorem the numbers below test. Claims marked **[classical]**,
**[measured here]**, **[conjectured]**.

Scripts: `positioning_check.py` (obstruction + LP verification), `task3_iterated.py`
(Task 3 + restriction operator, log in `run_task3.log`), figure
`figures/task3_restriction.png` at 150 dpi. Setup throughout: periodic FTCS on N=120,
pure diffusion unless stated, T5's operating point r = 0.45.

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

## What the thread establishes, in one paragraph

For linear, local, constant-coefficient dynamics on a uniform grid, coarse-graining in
space has an exact finite closure whose depth is fixed by algebra alone (the alias count,
i.e. the McMillan degree — **[classical]**), whose state cost is exactly the state you
saved, and which cannot be made non-negative at all past a coarsening factor of 3. The
route, the restriction operator, and the memory depth are all powerless against that last
fact. **The positivity price of compression is not bounded and not slowly-growing; past
M = 3 it is infinite.** Whether that survives contact with a system whose unresolved
spectrum is oscillatory rather than monotone is the open question, and it is the one
prediction in POSITIONING.md §2.4 still worth testing.

## Negative results, stated plainly

- The brief's Task 3 comparison is a tautology (R1).
- The brief's Task 1 spectrum has two finite entries (R3).
- My own headline replacement experiment failed (R4), and my first explanation of the
  failure also failed.
- The framework's "does compression have a bounded price" question resolves to a
  dichotomy with a classical answer on both branches (R2).

## Known limitations

- LP conditioning: at large depth the coefficients g^{P−j} span many orders of magnitude
  and HiGHS reports spurious infeasibility (e.g. M=2, s=1 feasible at P=8, "infeasible" at
  P=9). Cells at depth ≤ 8 are robust; I would not trust a defect below ~1e−10. Every
  "—" at M ≥ 4 is independently backed by the §2.1 proof, so it does not rest on the LP.
- Searches capped at depth 10 and half-width 3.
- Pure diffusion for the tables; the advective case was spot-checked in
  `positioning_check.py` and moves nothing qualitatively.
- One dimension, periodic, constant coefficients throughout. The 2D question in the brief
  was not run.
