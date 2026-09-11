# T2 — Width vs. time-step frontier for wide positive explicit stencils

**Problem.** `u_t + c u_x = α u_xx` on `[0,1]`, `u(0)=u(1)=0`.
**Canonical setup** (as specified): `α=0.1, c=1, nx=100, Δx=0.01010101, Δt=4.59137e-04`,
so `ν = αΔt/Δx² = 0.45`, `Co = cΔt/Δx = 0.045455`, cell Péclet `Pe = cΔx/α = 0.101`.

---

## ERRATUM AND UPDATE (round 2)

Three corrections, two of them to my own earlier conclusions:

1. **`build_A` row 3 was wrong** (team lead's catch, confirmed symbolically and
   numerically). It must be `½(Δx − cΔt)² + αΔt`, not `½Δx² + αΔt`. Since shift and
   propagator commute for the constant-coefficient operator,
   `u(x+δ,t+τ) = exp[ξ∂_x + ατ∂_x²]u` **exactly**, with `ξ = δ − cτ`; the `u_xx` row is
   `ατ + ξ²/2`. Two of my Task 1 findings are **WITHDRAWN**: (a) the "advection cap
   `k ≤ 2α/c²`" and (b) the "cell-Péclet barrier `ν·Pe² ≤ 2`, beyond which no positive
   stencil exists at any width". Both were artifacts of the dropped terms. Corrected
   results in `task7_corrected_rows.py`. Everything in Tasks 3, 4 and 5 already used the
   corrected moment condition (`M₂ = 2kν + (kCo)²`), so those stand unchanged.
2. **The claimed irreducible `O((kΔt)²)` error floor does not exist** (`task8_jensen.py`).
   The Jensen argument imposes `Σw Δt² = 0`, which is not a consistency condition.
3. **My own `‖w‖₁` metric was initially saturated by aliasing**, which made positivity look
   free everywhere. Fixed by measuring on the resolved band `θ ≤ π/2` (`task10`).
4. **The quadratic law dies at `m ≈ 1/Pe_cell ≈ 10`, not at high Péclet** (T5, verified in
   Task 11) — but only for a stencil centred on the target point. Re-centring on the
   departure point restores `m²/(2ν)` at every `m`, for free.

**And the decisive new result: the construction survives variable coefficients**, where no
closed-form propagator exists — see the new Task 9. That, not the constant-coefficient
work-precision curves, is what makes this worth continuing.

---

## THE SHORT ANSWER

| question | answer |
|---|---|
| Is `k_max ≈ (mΔx)²/(2αΔt)` right? | **Exactly right**, constant = 1, provable in one line. Verified by LP for every `m ≤ 25`. |
| Does truncation error go like `C·(mΔx)²`? | **Yes**, `C = α/6` times `\|u_xxxx\|`, measured slope 1.978, measured `C = 0.0156` vs `α/6 = 0.01667`. |
| Does the wide positive stencil beat CFL-limited FTCS on work-precision? | **Yes — but not at the frontier.** Best configs beat FTCS by **393×** (`T=0.05`, tol `1e-4`) and **5454×** (tol `1e-5`). The *frontier* configs (`s≈1`) beat FTCS by only ~12–24×, and are **beaten by a fairly coarsened FTCS** at equal accuracy. |
| Does it beat Crank–Nicolson? | **Yes, 28–105×** in the main sweep (up to 215× on the finer N-sweep in `task3b`), again only in the over-resolved regime. At the frontier it sits *exactly on top of* the CN work-precision curve. |
| Is it new? | **At constant coefficients, no; at variable coefficients, yes — 7–20× over a fair competitor, but only while `α` varies over ≳20 cells (see Task 13c).** The frontier scheme is **first-order Runge–Kutta–Chebyshev with `s = m` stages, to round-off (2e-16)**, and the accurate regime is heat-kernel convolution (a DST/FFT exact solve is still **2–10× cheaper**). **At variable coefficients, yes** — see Task 9: the purely local construction beats grid-refined FTCS by 83× and RKC1 by 179× at tol 1e-4, and still beats **RK4+FD6** (4th order in time, 6th in space — the strongest competitor I could build) by **6.7–20×** at every tolerance, while remaining monotone. |
| Does relaxing positivity buy anything? | **In the diffusion-dominated regime, essentially nothing** (1.1× for a 10× `‖w‖₁` budget). The binding constraint is stencil *support*, not sign. Positivity only costs once the kernel is sub-cell and advection-dominated (then up to >10⁵× — Godunov's barrier). |
| Is there a "genuine optimum in `(m,k)`"? | **No interior optimum.** Reward increases monotonically in width until a *physical* cap (wall distance, or `kΔt ≤ T_end`). 15 / 18 optima sit on the cap. |

**Bottom line for the larger project.** At constant coefficients this is a clean negative on
novelty: the width↔step trade is real, quantitatively exact, derivable in closed form — and
it is precisely the classical Chebyshev super-time-stepping trade, rediscovered in real
space. **At variable coefficients it is a positive**, and that is where the work should go:
the local-propagator-moment construction beats the strongest order-matched explicit
competitor I could build by 7–20× while staying monotone, in a regime RKC, exponential
integrators and FFT solvers do not cover. Before writing anything, run the comparisons
listed as missing in `manuscript_outline.md` — especially RKC2/ROCK, nonlinearity, and the
2-D cost model, which could reverse the verdict.

---

## TASK 1 — The positivity frontier (`figures/fig1_frontier.png`)

### Closed form, derived and verified

The three consistency rows in `build_A`, in moment language (`M_p = Σ_j w_j j^p`), are

```
M0 = 1 ,   M1 = -k·Co ,   M2 = 2·k·ν
```

`w` is a **probability measure on the integer lattice `{-m..m}`**. Therefore:

* `M2 ≤ m²` always, with equality **only** for mass at `±m` (upper concave envelope of `j²`
  on `[-m,m]` is the constant chord `m²`);
* `M2 ≥ ψ(M1)`, where `ψ` is the **lower convex envelope of `j²` on `ℤ`**,
  `ψ(μ) = ⌊μ⌋² + f(2⌊μ⌋+1)`, `f = frac(|μ|)`. Note `ψ(μ) − μ² = f(1−f) ≤ 1/4`.

Hence feasibility is **exactly** `ψ(k·Co) ≤ 2kν ≤ m²`, giving

```
k_max(m) = floor( min[ m²/(2ν) ,  2ν/Co² ] )      (up to the ≤¼ lattice correction)
         = floor( min[ (mΔx)²/(2αΔt) ,  2α/(c²Δt) ] )
```

**LP verification, m = 1…25:** `k_max(LP) == k_max(analytic)` for **every** m, both at `c=0`
and `c=1`. The predicted constant is exactly 1, not approximate.

| m | 1 | 2 | 3 | 5 | 8 | 12 | 19 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|
| `k_max`, c=0 | 1 | 4 | 10 | 27 | 71 | 160 | 401 | 444 | 694 |
| `k_max`, c=1 | 1 | 4 | 10 | 27 | 71 | 160 | 401 | **435** | **435** |
| `floor(m²/2ν)` | 1 | 4 | 10 | 27 | 71 | 160 | 401 | 444 | 694 |

(The preliminary measurement is reproduced: m=3 → 10 so 8 ✓/16 ✗; m=5 → 27 so 16 ✓/32 ✗.)

### Does advection tighten or loosen the frontier?

**Tightens, and in a way that is independent of `m`.** The variance-nonnegativity constraint
`M2 ≥ M1²` gives the second, `m`-free ceiling

```
k·Δt ≤ 2α/c²        ⟺        sqrt(2α·kΔt) ≥ c·kΔt
```

i.e. **the diffusive spreading over the step must exceed the advective displacement over the
step**. For the canonical numbers this is `k ≤ 435.6`, which starts binding at `m = 20`
(exactly where the LP frontier flattens). In cell-Péclet terms, advection caps the frontier
whenever `Pe > 2/m`.

**High-Péclet cutoff.** Setting `k=1` gives `Co² ≤ 2ν`, i.e. `ν·Pe² ≤ 2`. For `ν = 0.45` the
critical cell Péclet is `sqrt(2/ν) = 2.108`: above it **no positive stencil exists for any
`(m,k)`** — verified by LP at `m = 40`. This is the classical cell-Péclet-2 barrier, and
**widening the stencil does not help at all**. The reason is that the 3 rows demand *exactly*
the physical diffusion and no numerical diffusion; upwinding evades the barrier by adding
numerical diffusion, i.e. by violating row 3.

### The advection cap is an artefact of the spec's rows

The true solution operator over `kΔt` is convolution with a Gaussian of mean `-k·Co` and
variance `2kν`, whose **second moment is `2kν + (k·Co)²`**. Row 3 asks for `2kν`; the missing
`(k·Co)²` *is* the dropped `u_tt` term. With the corrected row the frontier becomes

```
2kν + (k·Co)² ≤ m²            (stencil must cover displacement² + variance)
```

and the spurious `2ν/Co²` cap vanishes (m=30: 435 → 477; m=40: 435 → 688; LP-confirmed).

### Higher-order frontiers

Matching more Gaussian moments costs frontier:

| moments matched | required | predicted `k_max` ratio | measured (m=24) |
|---|---|---|---|
| `M0,M1,M2` | `M2 ≤ m²` | 1 | 1 |
| + `M4 = 3M2²` | `M2 ≤ m²/3` | **3** | **3.00** |
| + `M6 = 15M2³` | `M2 ≤ m²/5.449` | **3/(3−√6) = 5.449** | **5.47** |

(The 6-moment constant comes from the Hausdorff condition on `y = j²/m²`:
`3(1−q)(1−5q) ≥ (1−3q)² ⟹ q ≤ 1 − √24/6`.)

---

## TASK 2 — Truncation error along the frontier (`figures/fig2_truncation.png`)

Working with the **symbol error** `E(θ) = W(θ) − exp(−ik·Co·θ − kν·θ²)`, `θ = ξΔx`,
`W(θ) = Σ_j w_j e^{ijθ}`:

### c = 0 — the prediction is confirmed, with the constant

Symmetric `w`: `E(θ) = (M4/24 − M2²/8)·θ⁴ + O(θ⁶)`.
**Verified to 5 significant figures** (m=4…24, ratio measured/predicted = 0.99987–1.00000).

At the frontier the measure is forced to `±m`, so `M4 = m⁴`, `M2 = m²`, `E = −(m⁴/12)θ⁴`, and
since `kΔt = (mΔx)²/(2α)`:

```
LTE per unit simulated time  =  (α/6)·(mΔx)²·|u_xxxx|
```

Measured: **slope `d log(LTE/kΔt)/d log(mΔx) = 1.978`** (predicted 2); **`C = 0.0156`**
averaged over `m ≥ 6` (predicted `α/6 = 0.01667`; the ~6% deficit is the opposite-signed `θ⁶`
term). So the `C·(mΔx)²` prediction is **confirmed, with `C = α/6 · |u_xxxx|`**.

### c ≠ 0 — which term actually limits accuracy (the crux)

**The `u_tt` term dominates, by a factor `(3/2)·(c/(απ))² = 15.2`** for mode 1.

```
u_tt  contribution :  (π²c²/4α)·(mΔx)²·|u|      = 24.674 (mΔx)²|u|
u_xxxx contribution:  (απ⁴/6)·(mΔx)²·|u|        =  1.623 (mΔx)²|u|
```

Measured normalised rate: 55.6 at m=3 falling to 16.9 at m=24 (the θ²-truncation of the
`u_tt` series is only asymptotic in `kCo`, which reaches 7.3 at m=12) — bracketing the
predicted 26.3 within a factor 2.

**The important structural point: both terms scale as `(mΔx)²` per unit time**, because
`k ∝ m²`. So the `(mΔx)²` *scaling* claim survives advection intact; only the *constant*
changes, and at `α=0.1, c=1` it is set by advection, not by stencil width. The `u_tt` term is
free to remove (correct row 3 to `M2 = 2kν + (kCo)²`), after which `C = α/6` governs again.

### The safety factor is what really controls accuracy

With corrected rows and `s = m/σ`, `σ = sqrt(2kν)` (LTE per unit simulated time):

| m \ s | 1 (frontier) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 8 | 6.4e-2 | 1.8e-2 | 7.3e-4 | 1.3e-5 | 4.8e-10 | — |
| 16 | 2.1e-1 | 5.3e-2 | 4.1e-3 | 1.1e-4 | 1.1e-6 | — |
| 24 | 3.6e-1 | 1.1e-1 | 1.0e-2 | 4.0e-4 | 3.8e-6 | — |

Going from `s=1` to `s=5` buys **5 orders of magnitude** of accuracy for `5×` the points per
step and `25×` more steps. **The frontier is the worst place on this table to operate.**

---

## TASK 3 — Work-precision (`figures/fig3_workprecision.png`) — THE HEADLINE

Cost model: a `P`-term dot product = `2P−1` flops. FTCS = 5 flops/pt/step; CN = 10
flops/pt/step (5 for the RHS + Thomas with a precomputed factorisation: 2 forward, 3 back);
wide = `2P−1` with `P` the actual number of non-zero weights. Wall-clock cross-check in
`task3b_isolate.py` (flop model is the primary metric; wall time actually *flatters* the wide
stencil because FTCS/CN pay Python per-step overhead).

### Cost (flops) to reach a target L∞ error at `T_end`

**`T_end = 0.05`**

| target | FTCS | CN | RKC1 | RKC2 | **wide (best s)** | wide @ frontier (s≈1) | DST exact |
|---|---|---|---|---|---|---|---|
| 1e-3 | 5.34e4 | 3.84e3 | 2.77e4 | 5.53e4 | **8.34e2** | 1.50e3 | 1.31e3 |
| 1e-4 | 4.37e5 | 3.17e4 | 4.52e5 | 4.56e5 | **1.11e3** | 3.73e4 | 1.31e3 |
| 1e-5 | 2.83e7 | 5.11e5 | — | 1.47e7 | **5.19e3** | 1.22e6 | 3.12e3 |
| 1e-6 | — | — | — | — | **2.43e4** | — | 7.24e3 |
| 1e-8 | — | — | — | — | **5.48e5** | — | 3.70e4 |

**`T_end = 0.20`**

| target | FTCS | CN | RKC1 | RKC2 | **wide (best s)** | wide @ frontier | DST exact |
|---|---|---|---|---|---|---|---|
| 1e-3 | 2.14e5 | 1.57e4 | 2.21e5 | 2.26e5 | **2.03e3** | 9.02e3 | 1.31e3 |
| 1e-4 | 1.41e7 | 1.27e5 | — | 1.83e6 | **2.49e3** | 1.49e5 | 1.31e3 |
| 1e-5 | 1.13e8 | 1.02e6 | — | 2.94e7 | **9.80e3** | — | 3.12e3 |

**Note on RKC.** RKC1/RKC2 are *not* faster than FTCS at these tolerances, and that is
expected: RKC buys **stability**, not accuracy, and on this problem accuracy (the `O(Δx²)`
spatial error and the temporal order) is the binding constraint, not the CFL. RKC wins on
long horizons where the step count, not the grid, dominates. It is on the plot because it is
the correct point of comparison for the *stability* claim — and the frontier stencil
coincides with RKC1 exactly (Task 6).

**Speedups at equal accuracy:** vs FTCS **393× / 5454×** (1e-4 / 1e-5, T=0.05) and
**5655× / 11555×** (T=0.2). Vs CN **28× / 98×** (T=0.05) and **51× / 104×** (T=0.2).

### But the win is not the claimed mechanism. Three controls say so.

**(1) Pure diffusion (`c=0`) removes the gauge-transform / semi-Lagrangian advantage.** The win
survives (1224× vs FTCS, 11× vs CN at 1e-4), so it is not an advection-treatment artefact.

**(2) The frontier stencil is coarse-grid FTCS, run `m`-fold redundantly.**
Half-width `m` on `N` points vs plain FTCS on `N/m` points, `T=0.05`, `c=0`:

| N | m | wide L∞ | wide flops | FTCS(N/m) L∞ | FTCS flops | err ratio | **cost ratio** |
|---|---|---|---|---|---|---|---|
| 400 | 9 | 2.04e-4 | 4.98e4 | 1.98e-4 | 6.86e3 | 1.03 | **7.3×** |
| 400 | 18 | 8.85e-4 | 1.19e4 | 7.96e-4 | 8.40e2 | 1.11 | **14.2×** |
| 800 | 9 | 5.10e-5 | 3.99e5 | 4.94e-5 | 5.54e4 | 1.03 | **7.2×** |
| 800 | 17 | 2.21e-4 | 9.98e4 | 1.98e-4 | 6.86e3 | 1.11 | **14.5×** |

Same error, `7–15×` the cost. At the frontier the `m` sub-lattices `{j mod m}` **never
exchange information** — the scheme is `m` independent coarse FTCS solves.

**(3) The Pareto-optimal configurations are never at the frontier.** Every point on the wide
Pareto front has `s = 3–6`; `s ≈ 1` configs lie *on* the Crank–Nicolson curve (see the orange
and green curves in the figure, which overlap).

**Where the win actually comes from:** using the **exact continuum semigroup kernel** instead
of a difference operator removes the `O(Δx²)` spatial error entirely. That is an exponential-
integrator effect, not a wide-stencil effect. Confirmation: a DST-based exact solve is
**2–10× cheaper still** than the best wide stencil at every tolerance.

### Boundaries — the real practical cost

A half-width-`m` stencil cannot be centred within `m` cells of a wall. Options measured:

* **Odd reflection (method of images)** — exact for `u=0` Dirichlet (and, via
  `V = u e^{−βx}`, `β = c/2α`, exact for advection–diffusion too). Costs nothing extra.
  **This is what makes the scheme work at all.**
* **Zero-padding (the naive thing)** — catastrophic: `865× – 20 200×` worse error.
  | N | nb | m | images | zero-pad | penalty |
  |---|---|---|---|---|---|
  | 100 | 1 | 41 | 8.18e-5 | 1.11e-1 | 1351× |
  | 100 | 16 | 11 | 1.35e-6 | 2.73e-2 | 20223× |
  | 400 | 4 | 81 | 3.48e-5 | 7.06e-2 | 2030× |
* **FTCS sub-stepping in a wall layer** — **not viable**. To advance `kΔt` by FTCS you need a
  halo of `k` cells, and `k = m²/(2ν)`, so the layer is `m + k` wide, not `m`:
  `m=8 → 79 cells; m=16 → 300 cells; m=24 → 664 cells`, which overruns a 400-point domain.

**Honest caveat:** images requires the Green's function of the boundary-value problem in
closed form. That is available here because the problem is constant-coefficient on an
interval. It is *not* available for variable coefficients, nonlinearity, or general geometry —
which is exactly where FTCS/CN/RKC are used. This is the method's binding limitation.

---

## TASK 4 — Adaptive `m` (`figures/fig4_adaptive.png`)

### The premise needs correcting first

With a **single global** big step `kΔt`, positivity forces `m_j ≥ σ = sqrt(2αkΔt)/Δx` at
*every* point — you may **not** shrink the stencil near a steep gradient, because the measure
you must represent has that variance everywhere. What can legitimately adapt is the safety
factor `s_j = m_j/σ`. And since the leading error is `(moment mismatch) × u^{(2p+2)}`, you
need **large `s` where the solution is ROUGH and `s ≈ 1` where it is SMOOTH** — the *opposite*
of "large m where smooth, small m near steep gradients".

### Measured (N=200, pure diffusion; L∞ error / flops)

| case | policy | error @ comparable cost |
|---|---|---|
| smooth sine, nb=32 | **rough-wide** | **4.66e-9** |
| | smooth-wide | 1.85e-5 (**3970× worse**) |
| narrow Gaussian, nb=32 | **rough-wide** | **1.99e-6** |
| | smooth-wide | 8.95e-4 (**450× worse**) |
| narrow Gaussian, nb=128 | **rough-wide** | **2.21e-8** |
| | smooth-wide | 5.17e-5 (**2340× worse**) |

**The prompt's hypothesis is refuted decisively: adapt wide-where-rough.**

### But adaptivity still does not pay

**Uniform `s = 6` dominates both adaptive policies on the work-precision Pareto front.**
Smooth sine: uniform `s=6` at nb=16 gives `6.3e-11` for `4.0e5` flops; adaptive rough-wide at
nb=32 gives `4.7e-9` for `5.2e5` flops — more expensive *and* 75× worse. Reason: cost is
**linear** in `s` while error falls **super-exponentially** in `s`, so there is never a reason
to economise where the solution is smooth. Adaptivity buys 10–40% cost; uniform width buys
orders of magnitude of accuracy for the same money.

### The hard case kills the advantage entirely

**Square wave IC:** every method — FTCS, all fixed `s`, both adaptive policies — saturates at
`6.34e-3`, the `O(Δx)` error of representing a jump whose location is not on a grid point.
The wide stencil's advantage is confined to smooth data and **vanishes completely at a
discontinuity.** The narrow Gaussian (sd=0.03, ~6 cells) is fine — resolved data, full
advantage.

---

## TASK 5 — The decision problem for an RL agent (`figures/fig5_decision.png`)

State = local smoothness `θ = ξΔx` and `α` (through `ν`). Action = `(m,k)`.
Reward `R_λ = −log10(err_rate) − λ·log10(cost_rate)`, both per unit simulated time, computed
exactly from the symbol. Error floored at `1e-14` (below that the optimum chases round-off).
Action caps are **physical**: `m ≤ 30` (wall distance on a 100-point grid), `k ≤ 400`
(`kΔt ≤ T_end`).

### Optimal action vs. state (λ = 1)

| state | θ | m* | k* | s* | R* | R(1,1) | **gap** |
|---|---|---|---|---|---|---|---|
| very smooth (mode 1) | 0.032 | 30 | 19 | 7.3 | 7.80 | −0.19 | **8.0** |
| smooth (mode 4) | 0.127 | 30 | 16 | 7.9 | 7.65 | −2.59 | **10.2** |
| moderate (mode 16) | 0.508 | 29 | 14 | 8.2 | 7.55 | −4.98 | **12.5** |
| rough (θ=1.0) | 1.0 | 29 | 16 | 7.6 | 7.66 | −6.09 | **13.8** |
| near-grid (θ=2.0) | 2.0 | 28 | 16 | 7.4 | 7.68 | −7.02 | **14.7** |

vs `α` (λ=1, `k*` scales as `1/ν ∝ 1/α`, `s*` essentially constant at 7–8):

| α | ν | (m*, k*, s*) smooth | (m*, k*, s*) rough |
|---|---|---|---|
| 0.01 | 0.045 | (30, 186, 7.3) | (30, 159, 7.9) |
| 0.10 | 0.45 | (30, 19, 7.3) | (29, 14, 8.2) |
| 1.00 | 4.5 | (30, 2, 7.1) | (24, 1, 8.0) |

### Tolerance-constrained form (the decision actually faced)

Cheapest `(m,k)` with `err_rate ≤ ε`: **`s* ≈ sqrt(2 ln(1/ε))`** — a clean closed-form
design rule, confirmed:

| ε | 1e-3 | 1e-5 | 1e-7 | 1e-9 |
|---|---|---|---|---|
| measured `s*` | 3.2–4.5 | 5.0–5.5 | 5.4–6.3 | 6.3–7.3 |
| `sqrt(2 ln(1/ε))` | 3.72 | 4.80 | 5.68 | 6.44 |

But note the cost speedup vs the naive `(m=1,k=1)` **at fixed tolerance is only 1–3×** (and
`<1×` at `ε ≤ 1e-9`). The huge reward gaps above come from *accuracy at fixed cost*, not
*cost at fixed accuracy*.

### Verdict on learnability

* **No interior optimum in `m`. 15 / 18 optimal actions sit on the cap.** Reward increases
  monotonically with width until a physical constraint stops it. The optimal policy is
  therefore trivial — "take `m` as large as the wall and the horizon allow" — which an RL
  agent will find instantly, and which tells you the `(m,k)` parametrisation is degenerate.
* The landscape is **not** flat: only **0.1–3%** of feasible actions lie within 0.3 of `R*`,
  and losing 1 in `m` costs up to **4.6** reward units near the frontier cliff. So the
  *constraint boundary* is sharp and learnable; the *interior* is a monotone ramp.
* **Recommendation for the RL project: do not pose the action as `(m,k)`.** Pose it as the
  safety factor `s` at a given tolerance, where `s* = sqrt(2 ln(1/ε))` is a real, interior,
  state-dependent optimum — and then check whether the agent recovers that closed form. That
  is a meaningful test; "learn to pick the biggest `m`" is not.

---

## TASK 7 — The corrected consistency rows (`task7_corrected_rows.py`)

Confirmed the team lead's correction symbolically and by LP. With the corrected row 3:

* LP feasibility `==` the analytic rule `ψ(kCo) ≤ 2kν + (kCo)² ≤ m²` for every probed `(m,k)`.
* **Closed form:** `k_max = floor( [sqrt(ν² + Co²m²) − ν] / Co² )`, → `m²/(2ν)` as `Co → 0`.
  Physically `σ² + μ² ≤ (mΔx)²`: the stencil's second moment must cover the diffusive
  variance **plus** the squared advective displacement.
* **The `k ≈ 435` saturation is an artifact.** The buggy rows overstate `k_max` at small `m`
  (m=3: 10 vs 9) and then cap it at `2ν/Co²`; the corrected frontier does not saturate
  (m=30: 477, m=40: 688).
* **The cell-Péclet barrier is withdrawn.** Corrected lower bound reduces to `2kν ≥ f(1−f)`
  with `f = frac(kCo)`, and `f(1−f) ≤ 1/4`, so for `ν ≥ 1/8` it never binds. A positive
  stencil exists at every cell Péclet tested up to **Pe = 20**. At high Pe the frontier
  becomes `k_max ~ m/Co` — **linear** in `m`, not quadratic: the stencil must reach the
  departure point.
* Confirmed: at `m=k=1, α=0` the corrected rows give **exactly Lax–Wendroff** (and its
  negative weight for `0 < Co < 1` is Godunov's barrier).

---

## TASK 15 — T5 round 2: maxent, RKC positivity, and safety-factor scoping

### The two "alternative" constructions are the same object

T5's `maxent_stencil` (convex dual) and my `kernel_moment_matched` (2-parameter root-find)
agree to **1e-16**. They must: maximising `−Σ w log w` subject to `Σw = 1`, `Σw j = μ`,
`Σw j² = M₂` gives `w_j ∝ exp(λ₁ j + λ₂ j²)`, a discrete Gaussian — the same
two-parameter family my routine solves for. Band errors identical to all digits shown.
**Write it up as one construction with two derivations, not as two alternatives.**

### RKC touches the positive cone at exactly one point

Recovering the RKC1 stencil by exact DFT of `T_m(1 − 2ρ sin²(θ/2))`, `ρ = k/k_max`:

| m | ρ = 1.0 | 0.9 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|---|
| min w (m=8) | **−0.0000** | −0.1115 | −0.1610 | −0.2188 | −0.2651 |
| ‖w‖₁ (m=8) | **1.0000** | 2.3777 | 2.5964 | 2.5000 | 2.3421 |
| interior non-zeros | **0** | 15 | 15 | 15 | 15 |

Reproduces T5's −0.111 exactly. At `ρ = 1` the stencil is precisely
`½(δ₋ₘ + δ₊ₘ)` — zero interior weights, `‖w‖₁ = 1`. So **classical RKC meets the positive
cone at one point, its extreme vertex, and leaves it the moment you back off for accuracy.**
That is the sharpest available statement of what the LP/moment construction adds, and it
supersedes the qualitative version I gave in Task 6.

### Every speedup number, with its safety factor

T5 is right that a speedup quoted at `k = k_max` would be quoting a scheme with no accuracy
(the frontier stencil is the bimodal extremal measure; band error ≈ 1). It does not affect
my numbers, because none of them are frontier numbers — but the scoping belongs on the page:

| result | configuration | **s = m/σ** |
|---|---|---|
| 393× / 5454× vs FTCS (const-coef, T=0.05) | Pareto points | **1.5–6**, never 1 |
| 7–20× vs RK4+FD6 (variable-coef) | N=128, nb=32, m=5, P=6 | **2.60** |
| " (next two Pareto points) | nb=16 m=8 / nb=8 m=12 | 2.95 / 3.12 |

The `s ≈ 1` frontier configurations appear on the Pareto front only at the cheap, useless
end (`‖e‖∞ ≈ 3e-2`). Consistent with Task 3's separate "wide @ frontier" curve, which sits
on top of Crank–Nicolson.

### Terminology correction (T5)

`V = u e^{−βx}`, `β = c/2α`, on **linear** advection–diffusion is the **gauge (Liouville)
transform**, not Cole–Hopf (which is the nonlinear map taking Burgers to the heat
equation). Corrected throughout `RESULTS.md`, `manuscript_outline.md`, `solvers.py` and the
project memory. The substance is unchanged: it is problem-specific, and off-centring
(Task 11) is the general route.

---

## TASK 12 — The advective floor: not in the scheme (`task12_advective_floor.py`)

**Scope answer first: my nine-orders sweep WAS advective** — `c = 1`, `Co = 0.04545`, drift
0.45 cells, `Exact(ic_sine, 0.1, 1.0)`. It does not need rescoping.

**Reproducing the team lead's exact configuration** (`N = 201`, `Δx = 0.005`, `ν = 0.45`,
`Co = 0.0225`, `Pe = 0.05`, `k = 40`, `σ = 6` cells, drift 0.9 cells) gives **no floor**:

| s | 2.00 | 4.33 | 5.00 | 6.33 | 7.00 | **8.33** |
|---|---|---|---|---|---|---|
| LTE, c=1 | 8.5e-5 | 3.5e-7 | 2.3e-8 | 2.3e-11 | 3.6e-13 | **2.2e-15** |
| LTE, c=0 | 1.6e-5 | 6.4e-8 | 4.3e-9 | 4.1e-12 | 6.1e-14 | **1.1e-15** |

Advection costs a factor ~2, not a floor. Single exact Fourier modes (no reference error at
all) reach 1.8e-14 / 2.5e-14 / 5.6e-15 / 6.6e-17 for n = 1,2,4,8.

**Mechanisms ruled out quantitatively.** Kernel aliasing is the only drift-sensitive
candidate, and it is `exp(−4π²kν) = 2.4e-309` at `kν = 18` — the drift contributes a
*phase*, not a magnitude. Symbol error of the moment-matched kernel at modes 1 and 4:
1.2e-16 and 2.2e-16. Varying the reference's mode count (100→800) and quadrature
(4001→160001) moves the measured LTE only between 2.2e-15 and 3.2e-15.

**What I think it is.** Their own observation is the tell: the LP weights *and* the exact
sampled heat kernel hit the **identical** 3.83e-7. Two very different weight constructions
cannot share a floor that originates in the weights — a shared floor is a property of what
they are compared **against**. Ranked:

1. **A reference evolved numerically over `τ`.** My emulation (series at `T₀`, finely-stepped
   FTCS to `T₀+τ`) gives a 5.1e-6 discrepancy — right order, and flat in both `s` and
   moment order `p`, exactly the reported signature. **Most likely.**
2. The two time levels evaluated with different accuracy (different mode count, quadrature,
   or one analytic and one numerical). Same signature.
3. Absolute L∞ where `e^{βx} = e⁵ = 148` amplifies the right half — shifts by ~10², not 10⁸.

**Why my measurement is immune:** my `Exact` is a truncated sine series in `V = u e^{−βx}`
in which *every mode is an exact solution of the PDE*. Applying a consistent stencil to it
and comparing at `t+τ` measures the stencil's truncation error with no reference error
leaking in. **Cheap discriminator for them:** apply the stencil to one exact Fourier mode
(§E of the script). Machine precision there but 3.83e-7 against the full reference ⇒ cause
1 or 2. **So the F6 headline numbers are not capped by any advective floor.**

---

## TASKS 13–14 — Hardening the variable-coefficient claim

### (b) Is the win positivity, or just the high-order local rows?  **Positivity is load-bearing**

I guessed it would be a free extra. Wrong — same rows, `w ≥ 0` vs signed min-norm:

| nb | m | P | err positive | err signed | ‖w‖₁ signed |
|---|---|---|---|---|---|
| 32 | 5 | 6 | **7.29e-8** | 2.02e-7 | 1.35 |
| 16 | 8 | 6 | **1.62e-6** | 6.04e-6 | 1.44 |
| 8 | 12 | 6 | **3.96e-6** | 6.50e-5 | 1.46 |
| 32 | 8 | 6 | **2.63e-7** | 1.58e-5 | 1.53 |

Positive wins 5/6, by up to 60×. And the gap **widens with step count** — at `m=8, P=6`:
4.0× at `nb=16`, **78.1×** at `nb=32`. That is the signature of `‖w‖₁ = B > 1` compounding
as `B^nb`; the positive solution has `‖w‖₁ = 1` exactly and cannot amplify. **So the claim
is "local-α rows *and* positivity", not "local-α rows, positivity free".**

### (c) Does it survive a rough α?  **It degrades predictably with α/|α'| in cells**

(`task13`'s tanh row was junk — the spectral RK4 reference blew up; redone in `task14` with
a stable fine-FTCS reference.)

| α | `L_α = α/\|α'\|` (cells) | wide (best) | FTCS | **ratio** |
|---|---|---|---|---|
| sin 2πx | 45.8 | 7.29e-8 | 1.74e-4 | **2386×** |
| sin 4πx | 22.9 | 1.00e-7 | 2.44e-4 | **2430×** |
| sin 8πx | 11.5 | 2.06e-5 | 5.59e-4 | **27×** |
| sin 16πx | 5.7 | 1.43e-3 | 1.88e-3 | **1.3×** |
| tanh step, w=0.05 | 14.4 | 1.73e-4 | 2.57e-4 | 1.5× |
| tanh step, w=0.02 | 5.8 | 4.81e-4 | 3.82e-4 | **0.79× (loses)** |
| tanh step, w=0.005 | 1.4 | 6.42e-2 | 8.74e-4 | **0.01× (loses badly)** |

**The advantage is a function of how many cells `α` varies over**, which is exactly the
validity range of the local Taylor rows. It is ~2400× at `L_α ≳ 20` cells, ~27× at 11, gone
by 6, and actively harmful below ~2. Breakdown is gradual and predictable, not a cliff.
**This is the scope that must sit on the headline number.**

### (a) Boundaries with variable α

**Scope correction: the 7–20× runs of Tasks 9/9b were periodic — no boundaries at all.**
With variable `α` the method of images is gone (`α` is not symmetric about the wall).

*What actually happens at a wall* (correcting my own first guess that no one-sided positive
stencil exists): one **does** exist from `j = 1` onward; what degrades is the achievable
**moment order**, and gracefully — highest feasible `P` vs distance from the wall:

| j | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| `P_max` (nb=16, m=8) | 1 | 2 | 3 | 4 | 5 | 6 | 6 |
| `P_max` (nb=32, m=8) | 1 | 2 | 4 | 6 | 6 | 6 | 6 |

So the order-loss layer is only **4–6 cells**, far thinner than the `m + k` layer that FTCS
sub-stepping needs in the constant-coefficient case.

*Measured, with the wall layer charged honestly* (hybrid: wide interior + FTCS-substepped
wall layer of width `R = m + n_sub`, every flop counted):

| nb | m | n_sub | R | R/N | err (all x) | err interior | flops |
|---|---|---|---|---|---|---|---|
| 16 | 8 | 9 | 17 | 0.13 | 6.72e-6 | 4.29e-6 | 8.44e4 |
| 32 | 5 | 5 | 10 | 0.08 | **1.17e-6** | 7.87e-7 | 9.57e4 |
| 32 | 8 | 5 | 13 | 0.10 | 2.40e-6 | 1.61e-6 | 1.38e5 |

**Best hybrid 1.17e-6 at 9.57e4 flops vs FTCS 1.74e-4 at 1.17e5 — 148× the accuracy at
0.82× the cost, walls included.** Caveats: many `(nb,m)` combinations are infeasible
(interior LP fails whenever `σ > m`), and this Dirichlet comparison is against **FTCS only**
— I did not re-run the order-matched `RK4+FD6` competitor with walls, so **the 148× here is
not comparable to the 7–20× of the periodic case**.

---

## TASK 11 — The drift branch and the off-centre stencil (`figures/fig8_offcentre.png`)

From T5. Their corrected frontier `k_max = (−r + √(r² + ra²m²))/ra²` is algebraically
identical to mine from Task 7, and **our LP tables agree at every m** (1, 4, 9, 26, 62, 124,
273, 519, 903 for m = 1,2,3,5,8,12,20,32,50) — an independent cross-check, from a different
direction, of the corrected rows. Two things in their message were new to me:

### (1) The crossover is at moderate m, even at modest Péclet

| m | 5 | 8 | 10 | 12 | 16 | 20 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|
| `k_max` | 26 | 62 | 91 | 124 | 196 | 273 | 519 | 903 |
| `m²/(2ν)` | 28 | 71 | 111 | 160 | 284 | 444 | 1138 | 2778 |
| ratio | 0.94 | 0.87 | **0.82** | 0.78 | 0.69 | 0.61 | 0.46 | 0.33 |

Two distinct thresholds, worth keeping separate: the quadratic law becomes **materially
wrong** (≈25%) at `m ≈ ν/Co = 1/Pe_cell = 9.9`, which is T5's `m*`; the two branch formulas
**cross** at `m = 2ν/Co = 19.8`. Either way the point stands and is sharper than my earlier
"at high Pe it goes linear": **at Pe = 0.101 the quadratic branch is already dead by m ≈ 10**,
so lateral spending saturates linearly well before the stencil gets wide.

### (2) Re-centring on the departure point removes the penalty entirely — for free

Put the stencil on `{j₀−m .. j₀+m}` with `j₀ = round(−Co·k)`. With `η = j − j₀` the
conditions become `E[η] = μ' := μ−j₀` (so `|μ'| ≤ ½`) and `E[η²] = 2kν + μ'²`, hence
`k ≤ (m² − μ'²)/(2ν) → m²/(2ν)`. Verified by LP:

| m | 5 | 8 | 12 | 20 | 32 | 50 |
|---|---|---|---|---|---|---|
| centred | 26 | 62 | 124 | 273 | 519 | 903 |
| **off-centre** | 27 | 71 | 159 | 444 | 1137 | **2777** |
| `m²/(2ν)` | 28 | 71 | 160 | 444 | 1138 | 2778 |
| gain | 1.04× | 1.15× | 1.28× | 1.63× | 2.19× | **3.08×** |

The off-centre frontier tracks `m²/(2ν)` at every m and the gain grows without bound. This
costs nothing — re-centring is an index shift.

### Does it change my Task 3 numbers?  No, and here is why

`solvers.solve_wide` passes `c = 0` into `wide_weights` and removes the drift analytically
with the gauge/Liouville transform `V = u e^{−βx}`, so `μ = 0` and it **never paid the drift
penalty**. Footprint cost of the three routes (`s = 4`):

| Pe | kΔt | (i) centred | (ii) off-centre | saving |
|---|---|---|---|---|
| 0.101 | 0.05 | m=46 | m=41 | 1.12× |
| 0.5 | 0.05 | m=66 | m=41 | 1.61× |
| 2.0 | 0.05 | m=139 | m=41 | 3.38× |
| 8.0 | 0.05 | m=433 | m=41 | **10.5×** |

So: route (iii) gauge/Liouville is what I used and it is fine for constant coefficients, but it is
**problem-specific**. Route (ii) is the general fix and gets the same footprint for free —
**it is what the variable-coefficient scheme of Task 9 must use once `c ≠ 0`**, since no
gauge/Liouville transform exists there. Route (i), the centred stencil T5's bound describes, is
the one to avoid — and it is exactly what the original brief specified ("symmetric stencil,
all neighbours at −kΔt"), which is why the saturation appeared at all.

---

## TASK 8 — The "irreducible `O((kΔt)²)` floor" is not real (`task8_jensen.py`)

**Structural refutation.** After recursive substitution of `u_t = Lu` there is no
free-standing `u_tt` term to cancel. The conditions are moments of `ξ = Δx − cΔt` alone:

```
E[ξ] = 0,  E[ξ²] = 2α|τ|,  E[ξ^(2p)] = (2p−1)!!·(2α|τ|)^p,  E[ξ^odd] = 0
```

These are exactly the moments of a **Gaussian**, which is a **positive** measure. Every
order is simultaneously satisfiable with `w ≥ 0`. `ΣwΔt² = 0` is simply not one of the
conditions, so "0/600 stencils can cancel `ΣwΔt²`" is a correct computation of the wrong
constraint.

**Numerical refutation.** Hold `kΔt` **fixed** and vary `s = m/σ`. A floor would not move:

| kΔt = 0.00459 | s=2.0 | s=3.3 | s=4.3 | s=5.3 | s=6.3 | s=7.3 |
|---|---|---|---|---|---|---|
| measured LTE | 8.06e-5 | 6.21e-6 | 2.55e-7 | 3.21e-9 | 1.31e-11 | **1.82e-14** |
| LTE / (kΔt)² | 3.82 | 0.29 | 1.2e-2 | 1.5e-4 | 6.2e-7 | **8.6e-10** |

**Nine orders of magnitude at fixed `kΔt`, with `w ≥ 0` throughout.**

What *is* true: at **fixed `s`** the error scales as `(kΔt)²` (my Task 2 result, since
`(mΔx)² = 2αs²kΔt`). So the coordinating thread measured a real `(kΔt)²` scaling — but
its constant is `~exp(−s²/2)`, a design choice, not a floor. Their `m=5, k=8` config has
`s = 1.86`; sweeping `s` is what was missing.

**Does the 3-row LP recover the heat kernel?** **No.** It returns a vertex with ≤ 3
non-zeros, total-variation distance 0.67–0.92 from the kernel, and a symbol error 4 orders
of magnitude worse. Even the most-spread-out feasible point is not the kernel. Three moment
conditions plus positivity leave a polytope whose vertices are 3-point measures; recovering
the Green's function requires imposing the higher moments, i.e. knowing it already.

---

## TASK 9 — VARIABLE COEFFICIENTS: the decisive test (`figures/fig7_varcoef.png`)

`u_t = ∂_x(α(x)∂_x u)` on periodic `[0,1]`, `α(x) = 0.1(1 + 0.8 sin 2πx)`, so
`α_max/α_min = 9`. **No closed-form propagator exists**, so "sample the heat kernel" is
unavailable. Reference: Fourier pseudospectral + RK4 at `N_ref = 1024`, evaluated
**spectrally** at arbitrary `x` (linear interpolation floors every error near 2e-5 and
silently corrupts the comparison — I hit that and fixed it).

**The construction.** Purely local: with `s = x − x_j` and `α(x_j+s) = Σ_q a_q s^q`,

```
L(s^p) = Σ_q a_q p(p−1) s^(p−2+q)  +  Σ_q (q+1)a_(q+1) p s^(p−1+q)
```

Build `L` as a matrix on `{s⁰..s^P}`, exponentiate, and read the targets
`M_p = [exp(τL)]_(0,p)`. Then solve for `w ≥ 0` on `{-m..m}` with `Σ_i w_i (iΔx)^p = M_p`.
**Only `α` and its derivatives at `x_j` are used — no fundamental solution anywhere.**

### Do non-negative solutions exist?  Yes, with a feasibility rule

| `nb` | `σ` (cells) | smallest feasible `m`, P=2 | P=4 | P=6 |
|---|---|---|---|---|
| 64 | 1.36 | 2 | **none ≤ 40** | **none ≤ 40** |
| 16 | 2.72 | 3 | 5 | 7 |
| 4 | 5.43 | 6 | 10 | 13 |

Matching more moments needs a wider stencil, and at `σ ≲ 1.4` cells no positive stencil can
match 5 or 7 moments at any width — Godunov again. So `P`, `σ` and `m` are coupled, and the
scheme wants **large** steps (it is the small-step regime that is infeasible). Weights were
non-negative at every grid point in every feasible configuration.

### Work-precision, with the grid refined for every method

| tolerance | FTCS | RKC1 | **wide positive** | vs FTCS | vs RKC1 |
|---|---|---|---|---|---|
| 1e-3 | 1.48e4 | 4.51e4 | **4.03e3** | 3.7× | 11.2× |
| 1e-4 | 3.33e5 | 7.21e5 | **4.03e3** | **82.6×** | **178.8×** |
| 1e-5 | 2.13e7 | — | **1.08e4** | **1977×** | — |
| 1e-6 | — | — | **8.60e4** | — | — |
| 1e-7 | — | — | **8.60e4** | — | — |

**This is the first result in the thread that RKC does not already own**, and it is exactly
where the team lead predicted the decision lies. The scheme delivers high spatial order, a
large stable step, and monotonicity from a single local construction.

### Removing the order confound (`task9b_fairorder.py`)

The wide scheme uses `P=6` local rows, so it is 6th-order in space while FTCS/RKC1 use a
3-point Laplacian. Part of the win above is spatial order, not stepping. Two stronger
competitors, both with signed weights:

| tolerance | RKC1+FD4 | RKC1+FD6 | RK4+FD4 | **RK4+FD6** | wide positive | **vs RK4+FD6** |
|---|---|---|---|---|---|---|
| 1e-3 | 5.53e4 | 1.43e5 | 5.53e4 | 7.17e4 | **4.03e3** | **17.8×** |
| 1e-4 | 1.33e6 | — | 1.66e5 | 7.17e4 | **4.03e3** | **17.8×** |
| 1e-5 | — | — | 1.33e6 | 2.15e5 | **1.08e4** | **20.0×** |
| 1e-6 | — | — | 1.06e7 | 5.73e5 | **8.60e4** | **6.7×** |
| 1e-7 | — | — | — | 1.72e6 | **8.60e4** | **20.0×** |

`RK4+FD6` is the strongest competitor I could construct: 4th order in time, 6th in space,
CFL-limited but with no meaningful temporal error. **The wide positive stencil still wins
6.7–20× at every tolerance**, and additionally retains monotonicity, which `RK4+FD6` does
not. Note RKC1 gets *worse* with higher-order spatial operators: it is only **first order in
time**, and a higher-order operator has a larger spectral radius, hence more stages.

**So the honest variable-coefficient headline is 7–20× over a fair order-matched competitor**
(not the 83×/179× against standard FTCS/RKC1). The structural reason is that the wide
positive stencil gets **high order in space and in time simultaneously from one local
construction**, whereas RKC separates them and pays stability for temporal order.

---

## TASK 10 — The price of monotonicity (`figures/fig6_l1_pareto.png`)

For fixed `(m,k)`, minimise `max_θ |W−ġ|` over `w` subject to `Σw = 1` and `‖w‖₁ ≤ B`.
Since `Σw = 1`, `‖w‖₁ ≥ 1` with equality **iff** `w ≥ 0`, so `B = 1` is exactly the
monotone case and `B > 1` is the price of giving it up.

**Metric caveat (my error, now fixed):** measured out to `θ = π` the error is floored by an
aliasing term — `W(π) = Σw_j(−1)^j` is **real** for any real `w`, while `ġ(π)` has
imaginary part `−sin(kCoπ)e^{−kνπ²}`. Nothing, signed or not, can touch that. Measured on
the resolved band `θ ≤ π/2`:

**(a) Diffusion-dominated, at the frontier — positivity is FREE.**

| (m, k) | B=1 | B=1.5 | B=3 | B=10 | total gain |
|---|---|---|---|---|---|
| (6, 40) | — | — | — | — | **1.09×** |
| (12, 160) | 3.007e-1 | 2.928e-1 | 2.825e-1 | 2.729e-1 | **1.10×** |
| (20, 273) | — | — | — | — | **1.10×** |

A 10× budget in `‖w‖₁` buys 1.1× accuracy. **The binding constraint is stencil support,
not sign**: a kernel with `σ = m` cells keeps ~32% of its mass beyond `±m`, and no weight
vector on `{-m..m}` can represent mass it cannot reach.

**(b) Advection-dominated with a sub-cell kernel — positivity is expensive.**
`m=8, k=1, Co=0.5`:

| σ (cells) | 0.95 | 0.45 | 0.32 | 0.20 | 0.14 | 0.077 |
|---|---|---|---|---|---|---|
| B=1 (positive) | 1.4e-7 | 5.2e-2 | 1.3e-1 | 1.7e-1 | 1.9e-1 | 2.0e-1 |
| B=1.5 | ~0 | 2.4e-8 | 1.6e-7 | 9.5e-6 | 3.0e-5 | 7.2e-5 |
| gain | — | **>10⁵×** | **>10⁵×** | **>10⁵×** | **>10⁵×** | **>10⁵×** |

This is **Godunov's barrier** in the LP. Note it bites exactly where the method has no
super-stepping to offer anyway (a sub-cell kernel means a small step).

**(c) Which knob pays?** At `m=12`: backing `k` off from 160 to 10 buys **1.4e5×** accuracy
for 16× cost; relaxing `‖w‖₁` to 10 buys 1.1×. **Monotonicity is not what limits this
scheme in its own regime** — which also means "we keep monotonicity" is a weak selling
point there, since nobody would pay much to give it up.

---

## TASK 6 — Positioning against RKC / super-time-stepping (THE NOVELTY QUESTION)

**The frontier scheme is first-order Runge–Kutta–Chebyshev, exactly.**

RKC1 uses `R_s(z) = T_s(1 + z/s²)`, bounded by 1 on `z ∈ [−2s², 0]`. With the 3-point
Laplacian (`λ_max = 4α/Δx²`) this gives `Δt ≤ s²Δx²/(2α)` — **identical** to the positivity
frontier `kΔt = m²Δx²/(2α)`.

The algebra: with `z = Δt_base·λ(θ) = −4ν sin²(θ/2)`,

```
T_m(1 + z/(2ν)) = T_m(cos θ) = cos(m θ)
```

and `cos(mθ)` is precisely the symbol of the frontier kernel `½(δ_{−m} + δ_{+m})`.

**Numerical confirmation** (`task6_rkc.py`, N=401): running an actual RKC1 recurrence with
`s = m` stages against the frontier stencil, at `Δt = m²Δx²/(2α)`:

| s = m | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| ‖RKC1 − frontier stencil‖∞ | 2.2e-16 | 4.4e-16 | 6.7e-16 | 8.9e-16 |

Same polynomial of the same operator. **Same Δt, same footprint, same cost.**

### Comparison at equal stencil footprint `m`

| method | `Δt_max` | footprint | order | monotone (L∞) |
|---|---|---|---|---|
| forward Euler / FTCS | `1·Δx²/(2α)` | 1 | 1(t), 2(x) | yes |
| **RKC1, s = m stages** | **`m²·Δx²/(2α)`** | m | 1(t), 2(x) | yes, at the limit |
| **wide positive, half-width m** | **`m²·Δx²/(2α)`** | m | 1(t), 2(x) | yes |
| RKC2, s = m stages | `0.65·m²·Δx²/(2α)` | m | **2(t)**, 2(x) | no |
| RKL1 (Legendre STS) | `~0.5·m²·Δx²/(2α)` | m | 1(t), 2(x) | no |
| explicit SSP-RK, s stages | `m·Δx²/(2α)` (SSP bound `C ≤ s`) | m | ≤4 | yes |
| Crank–Nicolson | unlimited | global | 2(t), 2(x) | no |

### What, honestly, is left

1. **The quadratic width↔step scaling is not new.** It is the Chebyshev/super-time-stepping
   result, with the *same constant*, obtained by a different route.
2. **The SSP barrier is genuinely beaten** — explicit SSP methods obey `C_SSP ≤ s`
   (Ketcheson), i.e. *linear* growth of the monotone step in stage count, while this reaches
   `m²`. There is no contradiction: the SSP bound constrains convex combinations of
   forward-Euler steps of a *fixed* operator and must hold for *all* problems satisfying the
   forward-Euler condition, whereas here the spatial footprint itself widens and the argument
   is specific to the constant-coefficient linear heat operator.
3. **The one real increment**: the LP/moment view gives a *real-space certificate of
   monotonicity*, and shows RKC1 is a convex combination *exactly at* its stability limit.
   Below the limit the Chebyshev stencil `T_s(1 − (4ν'/s²)sin²(θ/2))` is **not** in general a
   non-negative measure, while the LP construction always is. That is a narrow but real
   distinction (positivity-preserving super-time-stepping).
4. **The accurate regime is a different known thing**: a moment-matched positive kernel with
   `s = m/σ ≈ 4–6` is a discretised heat kernel, i.e. a real-space exponential integrator /
   Gaussian convolution. Related: fast Gauss transform, exponential integrators, and
   semi-Lagrangian schemes (the mean condition `M1 = −kCo` *is* "centre the stencil at the
   departure point"). Adjacent literature on wide-stencil monotone schemes for degenerate
   elliptic operators (Motzkin–Wasow; Oberman; Froese–Oberman) uses the same positivity idea
   for a different purpose.

---

## Files

* `core.py` — setup, LP feasibility, `ψ` envelope, analytic frontier, kernel designs, symbol error
* `solvers.py` — FTCS, Crank–Nicolson, wide positive stencil (images / zero-pad), DST reference; flop accounting
* `task1_frontier.py` / `task1b_theory.py` — frontier map, lattice correction, Péclet cutoff, corrected rows, higher-order frontiers
* `task2_truncation.py` — LTE and symbol-error scaling
* `task3_workprecision.py` / `task3b_isolate.py` — work-precision, coarse-FTCS control, boundary costs, wall-clock
* `task4_adaptive.py` — adaptive half-width, three ICs
* `task5_decision.py` — reward landscape, tolerance-constrained optimum, flatness
* `task6_rkc.py` — the RKC identity
* `task7_corrected_rows.py` — corrected `build_A`, corrected frontier, Péclet sweep redone
* `task8_jensen.py` — refutation of the irreducible-`(kΔt)²` claim; LP vs heat kernel
* `task9_varcoef.py`, `task9b_fairorder.py` — **variable coefficients** (the decisive test)
* `task10_l1_pareto.py` — achievable error vs `‖w‖₁`
* `task11_offcentre.py` — T5 cross-check, drift branch, off-centre stencil
* `task12_advective_floor.py` — the 3.83e-7 floor is in the reference, not the scheme
* `task13_harden.py`, `task14_bnd_and_step.py` — positivity vs order, rough α, boundaries
* `task15_maxent_and_scope.py` — maxent ≡ moment-matched Gaussian; RKC positivity; safety factors
* `figures/fig1…fig7*.png` (150 dpi), `task*_results.json`, `task3_log.txt`, `task5_log.txt`
