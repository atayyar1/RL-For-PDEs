# Log

Headlines only, newest first. Detail lives in `FINDINGS.md` (numbered F1–F9) and in
`threads/T*/RESULTS.md`. Every claim marked `[M]` has a test in `tests/test_core.py`.

---

## 2026-09-11 — Session 1

**Set up**
- Branch `jo-exploration` off `main`. New `exploration/` tree: `core/` (shared library),
  `tests/`, `threads/T1…T5`, `figures/`.
- `core/pde.py` + `core/stencil.py` consolidate everything verified so far, so no two
  threads can disagree about dx, dt, u_true, or the consistency rows.
- `tests/test_core.py`: 9 tests, all passing, each one locking down a numbered finding.

**Fixed in Ali's code**
- **Corrected the u_xx consistency row in 15 files** (all active notebooks + 2 scripts;
  Archives untouched). Was `½Δx² + αΔt`, should be `½(Δx − cΔt)² + αΔt`. Substituting
  u_t = αu_xx − cu_x *recursively* also reduces u_xt and u_tt, contributing `−c·ΔxΔt` and
  `½c²Δt²`. Verified symbolically; confirmed the corrected row recovers Lax–Wendroff exactly
  for pure advection, where the old row gave unconditionally-unstable FTCS-central.
  Each patch carries an inline explanatory comment. Benchmark results move <0.2%; advection
  results change qualitatively.
- **Added a partition-of-unity guard** to `solve_weights`. `lstsq` silently returned
  Σw = 0.478 on rank-deficient systems.
- **Bonus**: the corrected row *also removes* that rank degeneracy. The old row was affine in
  Δt so a single-x stencil gave rank 2; ξ² is quadratic in Δt, restoring rank 3.

**Found**
- `[P]` **The trilemma**: locality, positivity, expressiveness — pick two. Reached
  independently via Jensen, Godunov, and Pawula. This is the program's answer to its own
  question. (F3, F7)
- `[P]` **Moment rows subsume Taylor rows** and extend to arbitrary order. Propagator moments
  come from a closed ODE hierarchy in the PDE coefficients alone — no Green's function. (F6)
- `[M]` **Query-driven win**: one point at t\* = 400 CFL steps costs 153 vs FD's 69,898, at
  error 1.3e-7 vs 2.6e-5. (F6)
- `[M]` **Positivity certificate is free**: exact `atan2` convex-hull test, 50–60× faster than
  the LP and cheaper than `lstsq`. `lstsq` throws away an available certificate 47.9% of the
  time. (T1)
- `[M]` **CFL conditions are positivity boundaries**, recovered to 6+ digits across schemes.

**Ruled out** (negative results, kept deliberately)
- `[X]` The wide-stencil frontier **is** first-order Runge–Kutta–Chebyshev, to machine
  precision (2.2e-16 at s=m=2…16). Same Δt, footprint, and cost. No novelty there.
- `[X]` At the frontier the m sub-lattices never exchange information — the scheme is m
  redundant coarse FTCS solves at 7–15× the cost. The apparent win was an artefact.
- `[X]` RL over global scheme selection (m, k) is unnecessary: 15/18 optimal actions sit on
  a boundary, so the optimal policy is trivial.
- `[X]` My own claim that `cond < 1e4` is uninformative about ‖w‖₁ was wrong (corr = 0.822).
  The accurate statement: it is a **rank** guard, not an amplification guard.
- `[X]` `k_max(m) = (mΔx)²/(2αΔt)` is exact only for the *uncorrected* row on the diffusive
  branch. With the corrected row it is an upper bound, tight to ~15% at m = 16.

**RETRACTED, same session**
- `[X]` **The trilemma.** I claimed locality/positivity/expressiveness — pick two, from a Jensen
  obstruction, and published it. The inequality holds; the consequence does not. Under a PDE
  constraint u_tt is not independent. **FTCS at r = 1/6: weights (1/6, 2/3, 1/6) ≥ 0, order 4.00,
  error 3.4e-12 against 5.2e-7 at r = 0.1.** Found by T1, verified independently here, locked into
  `tests/test_core.py` as `test_F3_RETRACTED_positivity_does_not_cap_order`. The lab notebook page
  carries the retraction prominently rather than a silent patch.
- `[X]` The "20–50× constant penalty" for positivity was an artefact of mismatched moment
  conditions, and was not constant — it grew like 1/h.
- What replaces it: positivity costs **one order in the hyperbolic case** (Godunov, verified) and
  **no order in the parabolic case**. Its real cost is **feasibility of geometry** — one-sided
  stencils are never feasible (0/15 501), and the frontier saturates. Better for the programme,
  since geometry is what the agent chooses.

**Second round of corrections (T5)**
- `[X]` **‖w‖₁ = 1 is sufficient, not necessary.** Mine again. Lax–Wendroff at ν=0.4 has
  ‖w‖₁ = 1.24 and max|symbol| = 1.000000 — stable. 512-fold composition reaches ‖w‖₁ = 1.59
  against a bound of 6.8e47. The honest framing: positivity is the certificate you can **always
  compute**; von Neumann is the sharp criterion you can compute only under translation
  invariance, which scattered geometry lacks. Positivity's real cost is rejecting good schemes.
- `[M]` **Frontier conflict between T1 and T5 resolved**: different rows. T1's 435-step
  saturation is an artefact of the uncorrected row. With the corrected row T5's closed form
  `k_max = (−r + √(r²+ν²m²))/ν²` is sharp at every m (LP-verified to ±1 at m = 2…50). Now in
  `core.stencil.k_max`; the test bisects against the LP rather than trusting any formula.
  My γ=4 query experiment sits at 27% of the true frontier, not the 92% I reported.
- `[M]` **A trilemma does survive — about coarse-graining, not accuracy.** Positivity fails for
  M ≥ 3 at every r; M = 2 survives only for r ≥ 1 − 1/√2. Coarse-graining admits an exact finite
  MZ memory of M−1 lags. *Memory is the price of coarse-graining space faster than the dynamics
  mixes.*

**Third round — and this one was my bug, not a claim**
- `[X]` **The "advective accuracy floor" (F18) was `u_true(x,0)` returning a bad initial
  condition.** The transformed sine series is undamped at t=0, so the IC carried ~5e-7 error
  while the reference at t=τ was exact to 2e-16. Six wrong diagnoses before the right one. Fixed
  and locked by a test.
- `[M]` Consequences both favour T2: its nine-orders result now reproduces **with advection**
  (7.6e-15 at p=10), and **F6's numbers improve by orders of magnitude** — at t*=400,
  6.2e-9 not 1.3e-7, i.e. 366× cheaper and 4219× more accurate than FD.
- `[X]` **The frontier saturation was an artefact of my own brief.** I specified "symmetric
  stencil"; off-centring by round(−c·kΔt/Δx) — a free index shift — restores the full diffusive
  branch (2777 vs 903 at m=50). That error propagated into T1's and T5's work.

**T5 lands the decisive experiment (and corrects me four more times)**
- `[M]` **Memory buys back positivity, geometrically — and at small M the price is exactly zero.**
  Exact non-negative compact coarse laws at M=2 (s=1,p=1) and M=3 (s=2,p=5), validated out of
  sample (400-step unforced rollout stable at 1e-15). The M=3 law has B1 = B2 = 0 exactly: a
  pure-delay scheme. Buying back positivity means moving the law backwards in time, not
  correcting a Markov law.
- `[X]` **F7/Pawula dead for the right reason now.** Pawula forbids the stencil's *cumulants from
  terminating*; accuracy needs finitely many *moments matched*. Different conditions, and I
  conflated them. Verified: positive stencils reach order 12 (min weight 2.2e-4, error 6.9e-15).
- `[X]` **Wide != composed.** I claimed an identity. On an identical 81-point footprint at L=40:
  composite 3.7e-6 vs wide p=8 **2.3e-11**. At p=2 the wide stencil is *worse*. Same footprint,
  different operators.
- `[X]` Edgeworth: leading correction is **skewness at L^-1/2**, not kurtosis at L^-1 — my
  mechanism holds only at c=0. RG fixed point has alpha_eff = alpha - c^2 dt/2, not alpha. One
  composed output costs L^2 evaluations, not L (light cone).

**Open / next**
- T1 Task 6: does positivity cost *order*, or only a constant? Refinement-path dependent.
- T3: estimate the moment hierarchy from noisy data; establish what it does that weak-form
  SINDy does not; design the "no local PDE exists" failure signature.
- T5: does coarse-graining trade memory against positivity at fixed locality?
- Ali's workshop paper: reframe the spine onto "the order barrier is the reason to learn".

**Communication channel**
- Living lab notebook published as an Artifact:
  https://claude.ai/code/artifact/9cde25ae-280f-426f-93d4-2a43b425b8ea
  Source is `exploration/notebook/page.html` (figures embedded as data URIs, regenerate with
  `python figures/make_figures.py` then re-embed). Republishing the same file path keeps the URL,
  so this page is updated in place as threads report — it is the running record, not a snapshot.
