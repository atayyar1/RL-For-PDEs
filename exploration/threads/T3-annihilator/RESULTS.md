# T3 — The annihilator / moment route to PDE discovery

Working directory: `~/Documents/00-projects/00-active/neural-integrators/threads/T3-annihilator/`

---

## VERDICT ON THE NOVELTY QUESTION (Task 2) — read this first

**Step 1 is not a new estimator. It is weighted local polynomial least squares
(moving least squares), reached by an expensive Monte-Carlo detour.**

This is not a "roughly similar in practice" statement. It is an exact algebraic
identity. With `Σᵢwᵢ = 1`, write `yᵢ = uᵢ − u*` and let `Φ` be the design matrix of
scaled monomials over the neighbours. Then `M(w) = Φᵀw` and `e(w) = yᵀw`, so the
least-squares normal equations over `S` weight draws are

```
 (Σₛ M Mᵀ) d = Σₛ M e
 ⇔  Φᵀ (Σₛ w wᵀ) Φ d  =  Φᵀ (Σₛ w wᵀ) y
 ⇔  Φᵀ W Φ d = Φᵀ W y ,        W := Σₛ w⁽ˢ⁾w⁽ˢ⁾ᵀ
```

which is exactly weighted least squares on the *pointwise* Taylor residuals.
Verified in `task2a_equivalence.py`: `‖d_moment − d_WLS‖∞ ≈ 5e-16` for **every**
`S` from 10 to 20 000. Random *geometries* change nothing — pad each `w` to the
union of sampled neighbours and the identity still holds (`2e-14`).

Two sharper consequences:

- With the natural weight law `w = 1/n + z`, `z ~ N(0, τ²(I − 11ᵀ/n))`, the
  `S→∞` estimator is `WLS(W = I − 11ᵀ/n)`, and **that is identically the
  profiled-out form of plain unconstrained uniform-weight MLS**. Measured
  agreement: `1.1e-5` at `τ=1`, falling to `9.1e-8` at `τ=100`.
- The constraint `Σw = 1` is therefore not a new ingredient. It is local
  polynomial regression that interpolates the target exactly — a known MLS
  variant, and under noise a *disadvantage*, since it forces the estimator to
  trust the noisy target value.

The randomisation is strictly a cost: it adds Monte-Carlo variance (2x worse at
`S=20` under noise), it silently picks a flat kernel over the sampling box
(a tricube kernel beats it 1.8x on clean data), and its MC residue is amplified
in exactly the ill-conditioned directions that matter.

### And it buys no noise robustness either

`task2b_noise.py`, every method tuned over its own bandwidth at every noise
level (median relative error over `u_x, u_t, u_xx`):

| method | clean | 1e-6 | 1e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|---|---|
| FD, spacing `m·dx` | 4.2e-7 | 5.2e-5 | 2.2e-3 | 1.1e-2 | 6.6e-2 | 6.1e-1 |
| Savitzky-Golay deg 4 | 4.2e-7 | 1.0e-4 | 2.1e-3 | 9.6e-3 | 3.6e-2 | 3.6e-1 |
| **moment regression** | 6.8e-7 | 1.3e-4 | 2.6e-3 | 1.2e-2 | 6.6e-2 | 3.1e-1 |
| its closed form (MLS-uniform) | 9.2e-7 | 1.4e-4 | 2.5e-3 | 1.2e-2 | 6.9e-2 | 3.3e-1 |
| MLS, tricube kernel | 3.0e-7 | 1.2e-4 | 2.9e-3 | 1.3e-2 | 6.9e-2 | 6.5e-1 |

All five curves lie on top of each other (`figures/fig_task2_noise.png`).

**Where the "far more noise-robust than finite differences" impression comes
from**: comparing against *untuned* FD at grid scale. Coarsening the FD stencil
is the standard FD noise knob, and using it erases the gap entirely.

| η | FD at grid scale (m=1) | FD tuned | moment tuned | FD(m=1)/MOM | FD(tuned)/MOM |
|---|---|---|---|---|---|
| 1e-6 | 3.8e-4 | 5.0e-5 | 1.1e-4 | 3.4x | 0.44x |
| 1e-4 | 3.8e-2 | 2.5e-3 | 2.3e-3 | 16.7x | 1.09x |
| 1e-3 | 3.8e-1 | 1.1e-2 | 1.1e-2 | 33.6x | 0.99x |
| 1e-2 | 3.8e+0 | 6.9e-2 | 6.8e-2 | 56.0x | 1.02x |
| 1e-1 | 3.8e+1 | 6.1e-1 | 3.3e-1 | 115.8x | 1.86x |

So the premise "the derivative estimation is the weak link, and averaging over
many stencils fixes it" is **false as stated**. Bandwidth selection fixes it,
and every method has that knob.

---

## What actually survives

1. The **parabolic grading** `deg(a,b) = a + 2b` is the right truncation for a
   parabolic PDE and is not what one writes down by default (Task 1).
2. **Step 2 works**, but weak-form SINDy — the closest competitor — beats it at
   every noise level from 1e-4 up, and beats it decisively on *term selection*
   (Task 3).
3. The **state-dependent-coefficient diagnostic** is genuinely good: on Burgers
   the recovered advection coefficient tracks the local `u` with slope `-0.992`
   (Task 4b).
4. Nonlocal data **does** trigger a loud alarm, but only if you report the
   residual and sweep the bandwidth — the selected term list alone looks
   innocent (Task 4c).
5. The **positivity certificate** on `w` is the one thing this formulation gives
   that a jet estimate cannot (Task 5). It is a property of the *weights*.

---

## Task 1 — Jet recovery (`task1_jet_recovery.py`)

Target `z* = (0.40, 0.25)`, `α=0.1`, `c=1`, 60 scattered neighbours, one-sided in
time. Exact solution and exact analytic mixed derivatives (PDE residual verified
to `4e-16`).

**Grading.** Parabolic beats total degree at equal cost. At `h=3dx`, parabolic
`K=4` uses `Q=8` monomials with `cond(Φ)=18.8`; total degree `K=4` needs `Q=14`
with `cond(Φ)=532`, for comparable accuracy.

**Truncation bias** scales as `h^(K+1−p)` with `p = a+2b` the parabolic degree,
confirmed by fitted log-log slopes (`figures/fig_task1_bias.png`):

| component | p | K=3 predicted / measured | K=4 predicted / measured |
|---|---|---|---|
| `d₁₀` | 1 | 3 / 2.43 | 4 / 4.01 |
| `d₀₁` | 2 | 2 / 2.00 | 3 / 1.13\* |
| `d₂₀` | 2 | 2 / 2.00 | 3 / 2.79 |
| `d₃₀` | 3 | 1 / 0.85 | 2 / 1.99 |
| `d₁₁` | 3 | 1 / 2.81\* | 2 / 2.01 |

\* these two are at the floating-point floor over the fitted range, so the slope
is meaningless there, not evidence against the scaling.

**Bias-variance.** Optimal `h` grows with noise exactly as `h* ~ (σ/√n)^(1/(K+1))`
predicts (`figures/fig_task1_biasvar.png`): best `h/dx` for `u_xx` is
`0.5 → 2 → 6 → 12` as `η` goes `0 → 1e-6 → 1e-4 → 1e-2`.

**Recoverability** (`K=4` parabolic, `h=3dx`, 60 points, median of 24 seeds):

| q | p | clean | η=1e-4 | η=1e-2 | verdict |
|---|---|---|---|---|---|
| (1,0) | 1 | 2.3e-4 | 5.5e-4 | 4.7e-2 | solid |
| (2,0) | 2 | 2.3e-3 | 4.1e-2 | 3.9e+0 | usable to 1e-4 |
| (0,1) | 2 | 3.4e-4 | 3.9e-3 | 4.1e-1 | usable to 1e-4 |
| (3,0) | 3 | 1.1e-2 | 1.3e-1 | 1.4e+1 | clean data only |
| (1,1) | 3 | 2.5e-1 | 2.9e-1 | 1.2e+1 | **hopeless** |
| (4,0) | 4 | 1.2e+0 | 2.2e+1 | 2.0e+3 | **hopeless** |
| (2,1) | 4 | 3.9e-2 | 2.8e-1 | 2.7e+1 | clean data only |
| (0,2) | 4 | 2.2e-2 | 2.1e-1 | 2.3e+1 | clean data only |

Only `u_x`, `u_t`, `u_xx` are usable at realistic noise. Anything needing a
third derivative or a mixed `u_xt` is out of reach — which bounds what Step 2
can ever discover.

**Sample count.** Minimum `S` for identifiability is `Q`. Beyond that `S` only
reduces Monte-Carlo noise about the closed form, and on clean data (bias
dominated) it buys nothing at all.

---

## Task 3 — Relation finding (`task3_relations.py`)

Shared noisy grid `nx=200`, `dx=5.0e-3`, `nt=1189`, `dt=2.5e-4`, `t∈[0.1,0.4]`.
Library of 9 candidates: `1, u, u², u_x, u·u_x, u_xx, u·u_xx, u_x², u_xxx`.
Both routes consume the same bytes.

| η | jet: `c` | jet: `α` | jet terms | weak: `c` | weak: `α` | weak terms |
|---|---|---|---|---|---|---|
| 0 | 1.000008 | 0.100000 | **correct** | 1.000037 | 0.100012 | **correct** |
| 1e-6 | 1.000027 | 0.100001 | **correct** | 1.000037 | 0.100012 | **correct** |
| 1e-4 | 1.000280 | 0.100003 | **correct** | 1.000034 | 0.100009 | **correct** |
| 1e-3 | 0.996959 | 0.099693 | 7 terms ✗ | 0.999897 | 0.099965 | **correct** |
| 1e-2 | 0.987729 | 0.098360 | 9 terms ✗ | 0.999291 | 0.099894 | **correct** |
| 1e-1 | 0.727327 | 0.035184 | 9 terms ✗ | 0.089025 | 0.004257 | ✗ |

**Head to head on max relative coefficient error** — weak-form SINDy wins at
1e-4, 1e-3 and 1e-2 by 3x, 9x and 15x, and is the only one that keeps the
correct sparse support past `η=1e-4`. The jet route wins only on clean data
(where it is 14x better) and at `η=1e-1` where both have failed.

Term selection is the jet route's real weakness. At `η=1e-3` no STLSQ threshold
recovers `{u_x, u_xx}`: at `0.001–0.05` it keeps all nine terms, and by `0.2` it
still keeps `1, u, u²`. The low-order library columns absorb the jets'
systematic bias.

Accuracy vs targets `N` (η=1e-3, m=8): `7.1e-1` at `N=10`, `8.4e-3` at `N=100`,
`3.8e-3` at `N=800` — saturating at the bias floor, as expected.

---

## Task 4 — Generalisation (`task4_generalise.py`)

**(a) Pure diffusion, `c=0`.** No hallucinated advection: `ĉ` is thresholded to
exactly `0` at η = 0, 1e-5, 1e-4 and 1e-2, and `α̂ = 0.100000 … 0.095405`.

**(b1) Burgers from a travelling wave — an identifiability trap worth
recording.** The Taylor shock `u = s − a·tanh(a(x−st)/2ν)` satisfies
`u_t = −s·u_x` *identically*, so the library columns obey the exact dependence
`(s−u)u_x + ν u_xx = 0` and Burgers is **not identifiable** from it, by any
method. The pipeline returns `u_t = −u_x`, which is true for this data and is the
parsimonious answer. Forcing the true support recovers `[−0.99987, 0.04991]`, so
the *jets* were fine — the data was the problem. My first pass through this task
mistook the flat `a_eff ≈ −1` for a method failure; it is not.

**(b2) Burgers from multi-mode Cole-Hopf data — the real test.** Using
`φ = 1 + 0.8cos(πx)e^(−νπ²t) + 0.15cos(2πx)e^(−4νπ²t)`, `u = −2ν(log φ)_x`, with
exact mixed derivatives by truncated 2-variable Taylor arithmetic (Burgers
residual verified to `1e-17`):

| η | coef(`u·u_x`) | coef(`u_xx`) | residual | selected |
|---|---|---|---|---|
| 0 | **−1.00000** | **0.05000** | 8.9e-6 | `{u·u_x, u_xx}` ✓ |
| 1e-6 | −1.00001 | 0.05000 | 4.1e-4 | ✓ |
| 1e-5 | −1.00038 | 0.05001 | 4.2e-3 | ✓ |
| 1e-4 | −1.00112 | 0.05002 | 1.1e-2 | ✓ |
| 1e-3 | −0.95238 | 0.04846 | 3.9e-2 | 8 terms ✗ |

And the coefficient genuinely tracks the state. Binning 6000 targets by local `u`
and fitting `u_t = a_eff·u_x + b_eff·u_xx` inside each bin gives
`a_eff = −0.9922·⟨u⟩ − 0.0006` (truth: slope −1, intercept 0) while
`b_eff = 0.04997 ± 0.00098` stays pinned at `ν = 0.05`
(`figures/fig_task4_burgers.png`). **This is the strongest positive result in
the thread.**

**(c) Spectral fractional diffusion `u_t = −α(−Δ)^s u` — the negative control.**
A caveat that matters: with only a handful of Fourier modes *any* operator is
matched exactly by a finite-order local one, so this test is vacuous unless the
field is genuinely multi-scale. With 20 modes (`bₙ = n^-1.5`), on **clean** data:

| s | residual floor | `α_eff` drift over m ∈ [3,12] |
|---|---|---|
| 1.0 (local) | 4.2e-4 | 1.009 |
| 0.9 | 1.1e-1 | 1.098 |
| 0.75 | 3.3e-1 | 1.418 |
| 0.5 | 6.9e-1 | 1.807 |

Two alarms fire, both only for `s<1`: an **irreducible 11–69% residual on clean
data**, and **coefficient drift with bandwidth**. A genuine local PDE leaves
`4e-4` and shrinks. But STLSQ's selected term list looks innocent in every case,
so **the residual and the bandwidth sweep are mandatory reporting, not optional**
(`figures/fig_task4_nonlocal.png`).

---

## Task 5 — Closing the loop (`task5_rollout.py`)

Rows built from the *recovered* coefficients, `nx=100`, `dx=1.01e-2`,
`dt=4.59e-4`, 1089 levels to `T=0.5`, 5-point stencil.

| η | `ĉ` | `α̂` | max err @T (recovered) | (true coeffs) | ratio |
|---|---|---|---|---|---|
| 0 | 1.000008 | 0.100000 | 1.39e-3 | 1.39e-3 | 1.00 |
| 1e-6 | 1.000004 | 0.099999 | 1.39e-3 | 1.39e-3 | 1.00 |
| 1e-4 | 1.000259 | 0.100084 | 1.02e-3 | 1.39e-3 | 0.74 |
| 1e-3 | 0.998125 | 0.099784 | 2.44e-3 | 1.39e-3 | 1.75 |
| 1e-2 | 0.973276 | 0.092792 | 3.53e-2 | 1.39e-3 | 25.4 |

The loop closes: discovery yields a working integrator, and up to `η=1e-3` it is
within 2x of the true-coefficient integrator. Discovery accuracy sets integrator
accuracy directly — at `η=1e-2` the penalty is 25x.

A controlled sweep perturbing `c` alone gives a fitted exponent of **1.037** over
`|Δc|/c ∈ [1e-2, 1e-1]`, i.e. discovery error propagates **linearly**; nothing
amplifies. (The whole-range fit of 1.50 is contaminated — the max-norm error is
signed and cancels against the scheme's own truncation error near the floor, so
"error minus floor" is not a clean quantity there.) `|Δc|·T·max|u_x|` is a valid
upper bound at every point.

### Positivity — tested properly

My first pass asserted that min-norm weights "go negative and the rollout
diverges". The data did not support that, and `‖w‖₁ > 1` is the wrong test. The
exact criterion for a one-step stencil is the von Neumann amplification factor
`g = max_θ |Σⱼ wⱼ e^{ijθ}|`. Since `‖w‖₁ ≥ g` always, `‖w‖₁ > 1` does *not* imply
instability — but `w ≥ 0` with `Σw = 1` forces `g = 1` exactly. Positivity is
**sufficient, not necessary**.

| stencil (k=1) | mode | ‖w‖₁ | min w | g | max err @T |
|---|---|---|---|---|---|
| symmetric [-2..2] | min-norm | 1.00000 | +0.034 | 1.000000 | 9.9e-4 |
| symmetric [-2..2] | positive | 1.00000 | 0.000 | 1.000000 | 1.0e-3 |
| offset [-3..1] | min-norm | 1.03615 | −0.018 | 1.000000 | 2.4e-3 |
| offset [-3..1] | positive | 1.00000 | 0.000 | 1.000000 | 1.6e-2 |
| symmetric [-3..3] | min-norm | 1.16649 | −0.046 | 1.000000 | 2.7e-3 |
| symmetric [-3..3] | positive | 1.00000 | 0.000 | 1.000000 | 2.4e-3 |
| upwind-only [-4..0] | min-norm | 1.75615 | −0.190 | **1.449** | **5.0e+53** |
| upwind-only [-4..0] | positive | — | — | — | infeasible (correctly refused) |

Over 4000 random 4–7 point one-step stencils in [-5,5] with the recovered
coefficients: min-norm is genuinely unstable (`g>1`) in **37.3%** of cases
(median `g = 1.288`, which over 1089 levels amplifies by `3.3e+119`); a
nonnegative stencil exists in **46.5%**; and positivity rescues an otherwise
unstable geometry in **7.1%**.

Honest reading: on well-conditioned symmetric stencils the min-norm solution is
already nonnegative and positivity changes nothing, and on lopsided stencils
imposing it can *cost* accuracy (offset [-3..1]: 1.6e-2 vs 2.4e-3). It buys a
stability guarantee, not a free lunch.

**But the real argument is not this 1-D table.** On a uniform one-step stencil
you can just compute `g` and reject. A meshfree solver has scattered,
level-varying geometry where no Fourier symbol exists and `g` is not computable.
There, `w ≥ 0` with `Σw = 1` still certifies max-norm non-amplification, and it
is a single LP. That certificate is a property of the **weights**. A jet
estimate — however obtained, by MLS or by moments or by finite differences —
does not provide one. This is the only place where "go through the weights"
beats "estimate the derivatives, then build a scheme".

---

## Files

| file | contents |
|---|---|
| `jetlib.py` | monomials/gradings, moment & WLS/MLS estimators, exact solutions (`AdvDiff`, `BurgersTanh`, `BurgersColeHopf`, `FracDiff`), truncated 2-var Taylor arithmetic, solver rows, STLSQ |
| `griddata.py` | shared cached noisy spacetime grid + local cloud extraction |
| `weakform.py` | weak-form (integral) SINDy |
| `task2a_equivalence.py` | **the novelty test** — run this first |
| `task1_jet_recovery.py` | grading, truncation bias, bias-variance, recoverability |
| `task2b_noise.py` | noise robustness vs FD / Savitzky-Golay / MLS |
| `task3_relations.py` | Step 2 + head-to-head vs weak-form SINDy |
| `task4_generalise.py` | diffusion, Burgers (both solutions), fractional |
| `task5_rollout.py` | discovered rows → integrator, von Neumann stability |
| `figures/*.png` | 8 figures, 150 dpi |

## Recommendation

Do not write the paper that was proposed. The headline claim — a new,
derivative-free, better-conditioned route to the local jet — is an exact
reparameterisation of moving least squares, and it is not more noise-robust than
a properly tuned finite difference, let alone than weak-form SINDy.

What is publishable is much narrower and is about the *integrator*, not the
*estimator*: see `manuscript_outline.md`.
