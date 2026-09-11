# T1 — Positivity as a stability certificate for PDE-constrained meshfree stencils

Code: `stencil.py` (solvers) + `task{1..5}_*.py`. Figures in `figures/`, 150 dpi.
Theory and proof sketches in `THEORY.md`.

Benchmark parameters throughout: `alpha=0.1, c=1.0, nx=100`, so
`dx = 0.0101010101`, `dt = 4.591368e-04`, `nt = 1090`,
`r = alpha dt/dx^2 = 0.450000`, `nu = c dt/dx = 0.045455`, `cell-Pe = c dx/alpha = 0.101010`.

---

## Headline

1. **Positivity ⟺ ‖w‖₁ = 1 is exact**, and it is the *right* certificate: it reproduces
   every classical CFL condition tested, to 6+ digits, as an LP feasibility boundary.
2. **The LP is unnecessary.** Positivity feasibility has a closed form: the origin must
   lie in the convex hull of the planar points `(q_i, p_i)`. An `atan2`-and-sort test is
   **exactly** equivalent (0 disagreements in 80 000 stencils) and is **26× faster than
   the LP scalar, ~2000× faster batched** — cheaper than the `lstsq` it replaces. This
   removes the cost objection to running it inside an RL step.
3. **Two claims in the brief did not survive.** The conditioning/‖w‖₁ story is weaker
   than stated, and `k_max ≈ (m dx)²/(2 alpha dt)` is only the small-`m` branch of a
   three-way minimum (see 3A).
4. **A bug-level finding:** the third row of the existing `build_A` is not the exact
   second-order PDE-constrained moment (see "Row 3" below). It matters for which
   classical scheme you recover and for the shape of the stability region.
5. **Effect size on error accumulation is real but modest in the benign regime and only
   becomes qualitative under adversarial geometry.** Numbers in Task 4 — stated plainly,
   including where the improvement is a factor of ~2 and not more.

---

## A finding about `build_A` itself (affects Tasks 2 and 4)

Substituting `u_t = -c u_x + alpha u_xx` into the Taylor expansion of
`u(x*+dx, t*+dt)` and collecting the `u_xx` coefficient gives

```
1/2 dx^2  +  alpha dt  -  c dx dt  +  1/2 c^2 dt^2   =   1/2 (dx - c dt)^2 + alpha dt
```

The existing row is `1/2 dx^2 + alpha dt`; it drops `-c dx dt` (from `dx dt u_xt`) and
`1/2 c^2 dt^2` (from `1/2 dt^2 u_tt`). Both are the same order as the retained terms
unless `|c dt| << |dx|`. **Confirmed symbolically by the coordinating thread (exact sympy
match).**

**The form that makes it obvious.** Let `xi = dx - c dt` be the characteristic offset. In
the frame moving with the characteristic the equation is *pure diffusion*, so the two rows
are simply

```
row 2 = xi           row 3 = 1/2 xi^2 + alpha dt
```

The bug is easy to miss precisely because row 2 was *already* written in `xi` — only row 3
was left in `dx`.

**Independent cross-check via propagator moments.** For a single time level `dt_i = -tau`,
imposing rows 2–3 is algebraically identical to matching the first two spatial moments of
the exact drift-diffusion propagator:

```
M1 = sum_i w_i dx_i   = -c tau
M2 = sum_i w_i dx_i^2 = (c tau)^2 + 2 alpha tau
```

Verified here to machine precision on random stencils (`M1`, `M2` matched to 1e-16). The
`'given'` row instead gives `M2 = 2 alpha tau`, **missing the `(c tau)^2` advective-drift
variance**. Three independent routes — Taylor expansion, sympy, and propagator moments —
now agree.

Consequences, measured in `task2_cfl.py`:

| | 3-pt centred, pure advection | FTCS adv–diff positivity region |
|---|---|---|
| `'given'` row | FTCS-central, `w = (nu/2, 1, -nu/2)`, `‖w‖₁ = 1+nu` — **unconditionally unstable** | `r ≤ 1/2` and `Pe ≤ 2` |
| `'exact'` row | **exactly Lax–Wendroff**, `w = (nu(1+nu)/2, 1-nu², -nu(1-nu)/2)` | `2r+nu² ≤ 1` and `Pe ≤ 2/(1-nu)` |

At the benchmark parameters `nu = 0.045`, so the two agree to ~0.2% and nothing in the
benchmark is affected. But the `'given'` row *cannot* produce a second-order-accurate
advection scheme, and its stability region is the wrong shape at larger Courant number.
Both variants are implemented (`variant='given' | 'exact'`); `'given'` is the default so
existing results reproduce.

---

## Task 1 — Positivity as the stability certificate

`task1_certificate.py` → `figures/task1_certificate.png`

### Sanity checks (all reproduced exactly)

| quantity | brief | measured |
|---|---|---|
| FTCS weights | `[0.4727, 0.1000, 0.4273]` | `[0.472727, 0.100000, 0.427273]`, `‖w‖₁ = 1.0000000000` |
| random 5-pt `‖w‖₁`, median / p90 / p99 / max | 1.09 / 2.04 / 7.11 / 37.4 | **1.09 / 2.04 / 7.11 / 37.4** |
| one-sided vs two-sided median | 2.99 vs 1.06 | **2.99 vs 1.06** |

### Claims verified (4000 random 5-point stencils)

| claim | result |
|---|---|
| C1 `sum(w) = 1` for every solver | max deviation `8.1e-15` (lstsq), `1.1e-13` (min-L1) |
| C2 positive-feasible ⇒ `‖w‖₁ = 1` | max `‖‖w‖₁ - 1‖ = 4.4e-16` |
| C3 infeasible ⇒ `min‖w‖₁ > 1` **strictly** | smallest infeasible value `1.002373`; median `1.373`; max `18.656` |
| C4 min-L1 = 1 on all feasible | max deviation `1.3e-15` |
| C5 `‖w‖₁(lstsq) ≥ min‖w‖₁` always | true; medians `1.0836` vs `1.0000` |
| C6 the `h` row-scaling changes feasibility | **no** — 0/2000 disagreements (it is a diagonal left-multiply with zero RHS on rows 2–3) |
| C7 LP ≡ `C(5,3)` enumeration | 0/20000 disagreements |
| C8 LP ≡ convex-hull test | 0/20000 disagreements |

Positive-feasible fraction of random 5-point stencils: **67.1 %**.

**The sharpest number:** on 4000 stencils that *are* positive-feasible, `lstsq` lands on
a nonnegative solution only **52.0 %** of the time; the other **47.9 %** come back with
`‖w‖₁ > 1.001` (p75 = 1.089, p90 = 1.178, p99 = 1.328, max = 1.483). The geometry was
fine; the solver threw the certificate away. This is the strongest single argument for
the change, and it is free — see Task 5.

### A third failure mode — a latent trap, not an active hazard (severity corrected)

For a **vertical** stencil (all neighbours at one `x`, stacked over time) `dx_i` is
constant, so `q_i = dx − c dt_i` and `p_i = ½dx² + alpha dt_i` are both *affine in the
single varying quantity* `dt_i`. Rows 2 and 3 then lie in `span{1, dt}`, `A` drops to
rank 2, and `b = (1,0,0)` is **outside its range**: the moment system has no solution at
all, for any solver, positive or not. Physically obvious in hindsight — you cannot
recover two spatial derivatives from data at one spatial location.

`np.linalg.lstsq` does **not** signal this. It returns the least-squares point, whose
residual `|Aw−b|` is 0.44–0.55, satisfying none of the moment conditions.

**Correction to my earlier framing.** I first wrote that this "is what crashed the first
1000-level rollout", implying an active hazard. That was too strong, and the coordinating
thread's pushback is correct. Checked explicitly:

- My `stencil.py:w_lstsq` has **no** `cond` gate (deliberately — it returns the raw
  min-norm solution), and `task4_rollout.py` has none either: its only two `cond` mentions
  are comments. So the guard **was not active** in the rollout that crashed.
- The pre-existing `threads/rollout.py:24` **does** carry `if np.linalg.cond(A)>1e4:
  return None`, and would not have crashed.

The accurate claim is therefore: **this bites only when the conditioning guard is
disabled.** It is a latent trap, not a corruption of any reported result.

Frequency, independently measured twice and agreeing exactly: of 15 948 draws with
distinct `(i,j)` pairs, **2 (0.013 %)** are rank-deficient, and **0 of them pass
`cond < 1e4`** (their cond values are `1.7e16` and `7.3e15`).

The gate also separates *cleanly* — there is no near-miss band:

| | max `|Aw−b|` | max `|Σw − 1|` |
|---|---|---|
| kept by `cond < 1e4` (47 806) | **1.48e−14** | 6.22e−15 |
| kept with `cond ∈ (1e2, 1e4]` (33) | 5.33e−15 | — |
| rejected (2) | **0.468** | — |

**Recommended fix** (agreed): an explicit `assert abs(w.sum() - 1) < 1e-6`, or the
residual check now in `stencil.py:moments_solvable`, rather than relying on conditioning
to imply consistency. Conditioning happens to catch it; it does not mean it.

### What was weaker than the brief stated

> "the conditioning test does NOT control `‖w‖₁`"

Half right. `corr(cond(A), ‖w‖₁) = 0.822` — a **strong** correlation, not none. What is
true is that *thresholding* at `cond < 1e4` is useless as a positivity or amplification
control. In the brief's own sampling it rejects **2 of 15 948** stencils — and all 2 are
the rank-deficient vertical ones above, not high-`‖w‖₁` ones. Within the accepted set
`‖w‖₁` still reaches 37.4. In Task 3C's window sampling it keeps 99.9 % of stencils at
precision 0.473, exactly the base rate: **zero information about positivity.**

The honest statement is: `cond(A)` is strongly correlated with `‖w‖₁` but the threshold
that is actually used is a *rank* guard, not an amplification guard. It removes the
geometries where no weights exist and leaves the geometries where bad weights exist.

---

## Task 2 — CFL ⟺ positivity

`task2_cfl.py` → `figures/task2_cfl.png`. Boundaries located by bisecting LP feasibility
to 60 iterations.

### (a) FTCS diffusion — **exact match**
Unique 3×3 solution `w = (r, 1-2r, r)`; only `w_0 = 1-2r` can change sign.
LP boundary `r* = 0.500000001`, textbook `1/2`, error `1e-9` (the bisection tolerance).

### (b) First-order advection — **exact match, and it is literally CFL**

| stencil | LP boundary | textbook |
|---|---|---|
| 2-pt upwind | `nu* = 1.0000001` | Courant ≤ 1 |
| 3-pt centred | `nu* = 1.0000000` | Courant ≤ 1 |
| 2-pt **downwind** | infeasible for every `nu > 0` | unstable |

The downwind result is the point: infeasibility is not about weights, it is that the
characteristic foot `x* - c dt` lies outside the stencil's x-span. Positivity *is* the
statement that the numerical domain of dependence contains the physical one.

### (c) FTCS advection–diffusion — **exact match, plus a surprise**

`'given'` row, unique solution `w = (r + nu/2, 1-2r, r - nu/2)`:
`w_0 ≥ 0 ⟺ r ≤ 1/2`, `w_+ ≥ 0 ⟺ r ≥ nu/2 ⟺ cell-Pe ≤ 2`. **Both textbook conditions,
recovered exactly.** LP vs theory, agreeing to 6 decimals:

| `nu` | feasible `r` set (LP) | theory |
|---|---|---|
| 0.02 | `[0.010000, 0.500000]` | `[0.010000, 0.500000]` |
| 0.50 | `[0.250000, 0.500000]` | `[0.250000, 0.500000]` |
| 0.90 | `[0.450000, 0.500000]` | `[0.450000, 0.500000]` |

cell-Pe boundary: `2.000001`, `2.000000`, `2.000000` at `r = 0.1, 0.25, 0.45`.

**Surprise:** with the `'exact'` row the feasible set is **disconnected**. Since
`nu = r·Pe`, `w_+ ≥ 0` becomes `r Pe² - Pe + 2 ≥ 0`, which for `r < 1/8` fails on an
interval between the roots. At `r = 0.1` the LP returns

```
[0.010000, 2.763933] U [7.236068, 8.944272]      theory: [0, 2.763932] U [7.236068, 8.944272]
```

i.e. a second stable branch at high cell-Péclet. Verified by hand: at `Pe = 8`
(`nu = 0.8`), `w = (0.82, 0.16, 0.02)` — all non-negative, sums to 1. A monotone
stability region for this scheme is **not convex**, which is worth knowing before anyone
tries to learn or parameterise it.

### (d) Second-order advection — **positivity fails, as expected, and it is Godunov's theorem**

With `alpha = 0` the `u_xx` moment is `p_i = 1/2 (dx_i - c dt_i)² ≥ 0`, zero only for a
neighbour exactly on the characteristic. A convex combination of non-negative numbers
vanishes only if all active ones vanish ⇒ **infeasible for every non-integer Courant
number, at any stencil size** (verified for 3-pt and 5-pt, `nu ∈ [0.1, 1.1]`; feasible
only at `nu = 1`).

The unique 3-point solution is exactly Lax–Wendroff (`max|diff| = 0.00e+00` vs the closed
form at `nu = 0.5`), with `‖w‖₁ = 1 + nu(1-nu) ≤ 1.25`. LW is `L²`-stable for `nu ≤ 1`
and never monotone. **So `‖w‖₁ > 1` does not imply instability** — this is the honest
limitation of positivity as a criterion, and it is not a numerical artefact but a theorem.
Diffusion is what buys positivity back: the centre point gets `p < 0` iff
`dt < 2 alpha/c² = 0.2`, satisfied at the benchmark by a factor of 436.

---

## Task 3 — Feasibility geography

`task3_geography.py` → `figures/task3_geography.png`

### 3A — symmetric stencil, half-width `m`, single time level `-k dt`

**The predicted `k_max ≈ (m dx)²/(2 alpha dt)` is correct only for small `m`.**

| `m` | `k_max` (LP) | `(m dx)²/(2α dt)` | `m dx/(c dt)` | `2α/(c² dt)` | min of three |
|---|---|---|---|---|---|
| 3 | 10 | 10.0 | 66 | 435.6 | 10.0 |
| 8 | 71 | 71.1 | 176 | 435.6 | 71.1 |
| 12 | 160 | 160.0 | 264 | 435.6 | 160.0 |
| 19 | 401 | 401.1 | 418 | 435.6 | 401.1 |
| 20 | 435 | 444.4 | 440 | 435.6 | 435.6 |
| 25 | **435** | 694.4 | 550 | 435.6 | 435.6 |
| 40 | **435** | 1777.8 | 880 | 435.6 | 435.6 |

Ratio to the claimed formula: `0.976` for `m ≤ 12`, **`0.394`** for `m ≥ 25`.
Ratio to `min(three bounds)`: `0.976` and **`0.999`**.

So there are three bounds, not one — diffusive `m²dx²/(2α dt)`, advective `m dx/(c dt)`,
and a "vertex" bound `k ≤ 2α/(c² dt)` from the origin having to sit above the moment
parabola. **Reachable depth saturates at 435 steps however wide the stencil**, and the
crossover is at `m = 2 alpha/(c dx) = 2/Pe = 19.8` cells. Any speedup claim built on the
diffusive formula alone will be too optimistic by 2.5× at `m = 25` and 4× at `m = 40`.

### 3B — spreading neighbours over several time levels

**First attempt was vacuous and is recorded as such.** Asking for the largest `K` such
that the block `{|j| ≤ m} × {1 ≤ k ≤ K}` is feasible returns "unbounded" for every
`m ≥ 1` — but only because that block *contains the 3-point FTCS stencil*, which is
feasible here. The question measures nothing.

Reformulated: constrain every neighbour to depth `k ≥ k_min` and allow a band of `D`
extra levels. Largest feasible `k_min`:

| `m` | D=0 | D=1 | D=3 | D=10 | D=50 | `m²/(2r)` |
|---|---|---|---|---|---|---|
| 3 | 10 | 10 | 10 | 10 | 10 | 10.0 |
| 8 | 71 | 71 | 71 | 71 | 71 | 71.1 |
| 20 | 435 | 435 | 435 | 435 | 435 | 444.4 |
| 30 | 435 | 435 | 435 | 435 | 435 | 1000.0 |

**Every column is identical: extra time levels buy nothing.** Feasibility needs one
neighbour with `p_i ≥ 0`, and the best candidate is always the widest neighbour at the
*shallowest* level, so the bound depends on `k_min` alone:
`m ≥ sqrt(2 alpha k_min dt)/dx` — the half-width must be at least the **diffusion length
of the jump**. Only x-width extends reach.

### 3C — random 5-point stencils, feasible fraction vs window shape (M, K)

| | M=1 | M=2 | M=3 | M=4 | M=6 | M=8 | M=12 |
|---|---|---|---|---|---|---|---|
| K=1 | — | 1.000 | 0.698 | 0.546 | 0.370 | 0.270 | 0.187 |
| K=3 | 0.273 | 0.789 | 0.726 | 0.600 | 0.429 | 0.316 | 0.209 |
| K=8 | 0.035 | 0.360 | 0.664 | 0.702 | 0.590 | 0.464 | 0.297 |
| K=20 | 0.008 | 0.079 | 0.278 | 0.536 | 0.670 | 0.624 | 0.470 |
| K=40 | 0.002 | 0.018 | 0.091 | 0.219 | 0.556 | 0.661 | 0.596 |

There is a **ridge**, not a monotone trend: for each time depth `K` an optimal width
`M* ≈ sqrt(2 r K)` maximises the feasible fraction. Too narrow and no neighbour reaches
past the diffusion length; too wide and random draws stop bracketing the characteristic.
`M=2, K=1` is 100 % feasible (it is FTCS).

### Cheap geometric predictors (candidate RL action masks)

39 795 stencils drawn from random windows, base rate feasible **0.473**:

| predictor | precision | recall | keeps | cost |
|---|---|---|---|---|
| accept everything | 0.473 | 1.000 | 1.000 | — |
| `cond(A) < 1e4` | **0.473** | 1.000 | 0.999 | 1 SVD |
| `cond(A) < 1e2` | 0.474 | 1.000 | 0.998 | 1 SVD |
| x-spread ≥ 2 | 0.487 | 1.000 | 0.971 | O(n) |
| two-sided in x | 0.527 | 1.000 | 0.897 | O(n) |
| **P1** brackets characteristic: `min q ≤ 0 ≤ max q` | 0.507 | 1.000 | 0.933 | O(n) |
| **P2** straddles diffusion length: `min p ≤ 0 ≤ max p` | 0.636 | 1.000 | 0.743 | O(n) |
| **P1 ∧ P2** (necessary condition) | **0.678** | **1.000** | 0.697 | O(n) |
| P1 ∧ P2 ∧ two-sided | 0.702 | 1.000 | 0.674 | O(n) |
| P1 ∧ P2 ∧ depth ≤ 8 | 0.720 | **0.414** | 0.272 | — (not necessary: loses 59 % of valid actions) |

Recall is 1.000 for the necessary conditions *by construction* — they cannot reject a
feasible stencil. The measured quantity is precision.

### 3D — one-sided stencils are *never* positive-feasible (a sharp result)

Exhaustive enumeration of all `C(20,5) = 15 504` 5-subsets of a one-sided window
(`d ∈ 0..3` on one side of `x*`, `k ∈ 1..5`):

| window | feasible |
|---|---|
| downwind (`dx ≥ 0`) | **0 / 15 501** |
| upwind (`dx ≤ 0`) | **0 / 15 501** |

Both have a proof.

*Downwind.* `q_i = dx_i − c dt_i = dx_i + c|dt_i| > 0` for every neighbour, so `min q > 0`:
the stencil never brackets the characteristic foot. P1 fails identically.

*Upwind.* The `d = 0` column has `q = ck Δt > 0`, `p = −αk Δt < 0`, so all of its points lie
on the single ray with direction `(c, −α)`. The `d ≥ 1` points all have `q < 0`. Take the
direction `u = (−ε, −1)`. Then `u·v > 0` for the `d = 0` points whenever `ε < α/c`, and for
the `d ≥ 1` points whenever `ε > ½ D Δx`, where `D` is the window half-width. Such an `ε`
exists — i.e. a strictly separating direction exists, so `0 ∉ conv` — iff

> `½ D Δx < α/c`, i.e. **`D < 2α/(c Δx) = 2/Pe`**.

**Prediction, and it lands on the cell.** A one-sided upwind window becomes feasible only
once `D ≥ 2/Pe = 19.80`:

| `D` (cells) | 4 | 8 | 12 | 16 | **19** | **20** | 22 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| feasible frac | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** | **0.038** | 0.084 | 0.114 | 0.130 |

The same `2/Pe` constant governs the 3A crossover and the 3B reach bound. It is the
single number that organises the whole feasibility geography.

This also explains the brief's observation that one-sided stencils have median
`‖w‖₁ = 2.99`: they are not merely bad, they are **categorically outside** the positive
cone until the window exceeds `2/Pe` cells.

**Recommendation:** don't use any of these as the mask. Use the **exact** hull test — it
costs 0.2 µs batched (Task 5), less than the O(n) screens cost in Python, and gives
precision 1.000. `P1 ∧ P2` is worth having only as a branch-free pre-filter inside a
fused kernel. **Conditioning-based masks are worthless here** — `cond < 1e4` has
precision equal to the base rate, i.e. it carries zero information about positivity.

---

## Task 5 — Cost

`task5_cost.py`. 4000 random 5-point stencils, macOS / numpy 1.26.4 / scipy 1.15.3, single core.

| method | µs / solve | vs LP |
|---|---|---|
| `linprog` min-L1 LP (3×10 split) | 510.94 | 1.1× slower |
| `linprog` feasibility LP (HiGHS) | 452.77 | 1.0× |
| enum `C(5,3)` = ten 3×3 solves | 52.30 | **8.7× faster** |
| `lstsq` (min 2-norm) — *the incumbent* | 17.82 | 25.4× faster |
| **convex-hull test (exact)** | **16.89** | **26.8× faster** |
| `P1 ∧ P2` necessary screen | 5.40 | 83.9× faster |

Batched (the form an RL action mask would actually call):

| batch | µs / stencil | vs LP |
|---|---|---|
| hull, B=64 | 0.421 | 1128× |
| hull, B=512 | 0.182 | **2607×** |
| hull, B=4096 | 0.216 | 2203× |
| `P1 ∧ P2`, B=4096 | 0.060 | 7925× (necessary only) |

### Is there a closed form? Yes — and it is *not* the `C(n,3)` enumeration.

Row 1 puts `w` on the simplex, so rows 2–3 say the convex combination of the **planar**
points `(q_i, p_i)` is the origin. Hence

> **positivity feasible ⟺ `0 ∈ conv{(q_i, p_i)}`**

a 2-D point-in-hull query. By Gordan's theorem that holds iff the polar angles
`atan2(p_i, q_i)` have no gap wider than `π`: `n` `atan2` calls and a sort, `O(n log n)`,
no linear algebra, exact.

### Equivalence — verified, not assumed

60 000 stencils (`|dx| ≤ 6` cells, depth ≤ 39), base rate feasible 0.563:

| comparison | disagreements |
|---|---|
| `linprog` vs `C(5,3)` enumeration | **0** |
| `linprog` vs hull test | **0** |
| `linprog` vs batched hull test | **0** |

(Plus 20 000 more in Task 1, also 0 — 80 000 total.) Enumeration weights on 1033 feasible
stencils: `max|Aw - b| = 2.2e-16`, `min_i w_i = 0`. The `C(5,3)` enumeration *is* exactly
equivalent, as the hint predicted, and *is* 8.7× faster than `linprog` — but the hull test
is another 3× faster than that scalar, and 250× faster batched.

**Answer to the question that motivated Task 5:** the certificate is not a cost. It is
*cheaper than the `lstsq` it replaces* (16.9 µs vs 17.8 µs scalar; 0.2 µs batched). There
is no performance reason to keep the min-2-norm solver.

### Caveat

The hull test answers *feasibility*. If you also want the weights you still need a solve —
use the `C(5,3)` enumeration (52 µs), not `linprog` (453 µs). A natural split for an RL
env: batched hull test to build the action mask, then one enumeration for the chosen action.

