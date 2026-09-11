# T2 — Width vs. time-step frontier for wide positive explicit stencils

**Problem.** `u_t + c u_x = α u_xx` on `[0,1]`, `u(0)=u(1)=0`.
**Canonical setup** (as specified): `α=0.1, c=1, nx=100, Δx=0.01010101, Δt=4.59137e-04`,
so `ν = αΔt/Δx² = 0.45`, `Co = cΔt/Δx = 0.045455`, cell Péclet `Pe = cΔx/α = 0.101`.

---

## THE SHORT ANSWER

| question | answer |
|---|---|
| Is `k_max ≈ (mΔx)²/(2αΔt)` right? | **Exactly right**, constant = 1, provable in one line. Verified by LP for every `m ≤ 25`. |
| Does truncation error go like `C·(mΔx)²`? | **Yes**, `C = α/6` times `\|u_xxxx\|`, measured slope 1.978, measured `C = 0.0156` vs `α/6 = 0.01667`. |
| Does the wide positive stencil beat CFL-limited FTCS on work-precision? | **Yes — but not at the frontier.** Best configs beat FTCS by **393×** (`T=0.05`, tol `1e-4`) and **5454×** (tol `1e-5`). The *frontier* configs (`s≈1`) beat FTCS by only ~12–24×, and are **beaten by a fairly coarsened FTCS** at equal accuracy. |
| Does it beat Crank–Nicolson? | **Yes, 28–105×** in the main sweep (up to 215× on the finer N-sweep in `task3b`), again only in the over-resolved regime. At the frontier it sits *exactly on top of* the CN work-precision curve. |
| Is it new? | **No.** The frontier scheme is **first-order Runge–Kutta–Chebyshev with `s = m` stages, to round-off (2e-16)** — same Δt limit, same footprint, same cost. The accurate regime is real-space heat-kernel convolution, i.e. an exponential integrator. A DST/FFT exact solve is still **2–10× cheaper** than the best wide stencil. |
| Is there a "genuine optimum in `(m,k)`"? | **No interior optimum.** Reward increases monotonically in width until a *physical* cap (wall distance, or `kΔt ≤ T_end`). 15 / 18 optima sit on the cap. |

**Bottom line for the larger project: this is a clean negative on novelty and a positive
on mechanism.** The width↔step trade is real, quantitatively exact, and derivable in closed
form — and it is precisely the classical Chebyshev super-time-stepping trade, rediscovered
in real space. Its one genuinely new ingredient (unconditional *monotonicity*, not just
L²-stability) is worth keeping.

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

**(1) Pure diffusion (`c=0`) removes the Cole–Hopf/semi-Lagrangian advantage.** The win
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
* `figures/fig1…fig5*.png` (150 dpi), `task*_results.json`, `task3_log.txt`, `task5_log.txt`
