# Manuscript outline — what is actually publishable from T3

**Read `RESULTS.md` first.** The method as originally framed does not hold up:
Step 1 is exactly moving least squares (proved by an algebraic identity, verified
to machine precision), and it gives no noise advantage over a tuned finite
difference. That paper cannot be written honestly.

This outline is for the narrower claim that **does** survive. It is about the
integrator, not the estimator.

---

## Working title

*Discovering a certified-stable meshfree integrator: from local jets to
nonnegative PDE-constrained stencils*

## The one-sentence claim

Sparse regression on local jets recovers not just a PDE but, through the
annihilator of the jet subspace, the constraint rows of a meshfree stencil —
and on scattered geometry those rows admit a nonnegativity certificate that
guarantees max-norm stability, which no derivative estimate can provide.

## What is NOT claimed (state this in the introduction, not the discussion)

1. Not a new derivative estimator. The moment regression of
   `e(w,G) = M(w,G)·d` is identically weighted local polynomial least squares
   (`Φᵀ W Φ d = Φᵀ W y`, `W = Σₛ w wᵀ`), and with the natural weight law its
   limit is plain uniform-weight MLS. Include the identity and the numerical
   verification as a short subsection; it is a useful negative result and it
   pre-empts the obvious referee objection.
2. Not more noise-robust than finite differences. Once the FD stencil spacing is
   tuned — the standard FD noise knob — the two are within a factor of 2 at
   every noise level from 1e-6 to 1e-1.
3. Not competitive with weak-form SINDy on coefficient recovery or on term
   selection. WSINDy wins by 3–15x from `η=1e-4` up and is the only method that
   holds the correct sparse support past `η=1e-4`.

## Positioning against prior work

- **PDE-FIND / SINDy** (Rudy et al. 2017; Brunton et al. 2016). Same Step 2.
  Our jets come from MLS instead of FD; that is a bandwidth choice, not a new
  idea, and we say so.
- **Weak-form / integral SINDy** (Messenger & Bortz 2021; Reinbold et al. 2020).
  **The closest competitor and the one to beat.** It avoids pointwise
  derivatives entirely by integrating against compactly supported test
  functions, and it is more noise-robust than anything here. Two things it does
  not do, and these are the honest wedge:
  (i) its library is structurally restricted to terms of the form `∂ₓᵃ(f(u))`,
      so a non-divergence term like `u·u_xx` is simply not expressible;
  (ii) it returns *coefficients*, not *stencil weights*, so it hands you no
      discretisation and no stability certificate. Getting from a WSINDy
      coefficient vector to a stable meshfree scheme is a separate problem —
      which is precisely the one this paper solves.
  A local jet, by contrast, is pointwise, which is what makes the
  state-dependent-coefficient diagnostic below possible.
- **Moving least squares / GFDM / RBF-FD** (Lancaster & Salkauskas 1981;
  Benito et al. 2001; Flyer et al. 2016). Our Step 1 *is* MLS. The contribution
  is not the fit; it is constraining the weights by the discovered annihilator
  and then imposing `w ≥ 0`.
- **Data-driven discretisation** (Bar-Sinai et al. 2019). Learns stencil
  coefficients directly from data with a neural network. Closest in *goal*.
  Difference: we obtain the rows from an interpretable discovered PDE and get
  an LP-checkable stability certificate rather than a learned black box.
- **Monotone / positive finite-volume schemes** (Harten 1983; LeVeque).
  Positivity as a stability mechanism is classical. New here: positivity as a
  *feasibility test on scattered, discovered stencils*, decided by one LP.

## Contributions, ranked by how well they held up

1. **The annihilator correspondence, stated cleanly.** For `u_t + c u_x = α u_xx`
   the jet subspace is `ρ^⊥` with `ρ = (c, 1, −2α)` on `(d₁₀, d₀₁, d₂₀)`;
   zero-error weights require `M ∈ span{ρ}`; a basis of `ρ^⊥` gives exactly the
   two solver rows `Σw(Δx − cΔt) = 0` and `Σw(½Δx² + αΔt) = 0`. Discovery of the
   relation *is* discovery of the rows. Short, exact, and the organising idea.
2. **Positivity as a certificate on discovered rows.** With recovered
   coefficients, `w ≥ 0` with `Σw = 1` forces the amplification factor to `1`.
   Over 4000 random scattered stencils, min-norm weights are genuinely unstable
   (`g > 1`) in 37.3% of cases, median `g = 1.288` — which over 1089 steps
   amplifies by `1e+119`, and we exhibit a rollout that reaches `5e+53`.
   Nonnegative weights exist in 46.5% of geometries and rescue 7.1%. Report the
   cost too: on lopsided stencils positivity can lose an order of magnitude of
   accuracy. Emphasise that on scattered geometry `g` is not even computable,
   so the LP is the only available certificate.
3. **State-dependent coefficient detection.** On multi-mode Cole-Hopf Burgers,
   binning targets by local `u` recovers `a_eff = −0.9922⟨u⟩ − 0.0006` against a
   truth of `−u`, while `b_eff = 0.04997 ± 0.00098` holds at `ν = 0.05`. Weak-form
   SINDy, being patch-integrated, cannot produce this pointwise map as directly.
4. **Two nonlocality alarms, and the warning that neither is automatic.**
   Irreducible clean-data residual (11–69% for `s < 1` vs `4e-4` for `s = 1`) and
   coefficient drift with bandwidth (1.8x at `s = 0.5` vs 1.009 at `s = 1`).
   Critically: STLSQ's selected term list looks innocent in every case, so these
   must be *reported*, not left to the reader.
5. **The parabolic grading `a + 2b`.** `Q = 8` and `cond(Φ) = 18.8` versus
   `Q = 14` and `cond(Φ) = 532` for total degree at the same accuracy. Small but
   practical, and it explains the observed bias law `h^(K+1−p)`.
6. **Two negative results worth the space** (§ "What is NOT claimed").

## Suggested structure

1. Introduction — derivative estimation in PDE discovery; why the *scheme*, not
   the coefficients, is the useful output.
2. The annihilator correspondence (contribution 1).
3. Jet recovery is moving least squares — the identity, the verification, and
   the parabolic grading. Honest and short.
4. Step 2: relations, and the head-to-head with weak-form SINDy including the
   losses.
5. Certified-stable discovered stencils (contribution 2). **The core.**
6. Diagnostics: state dependence, nonlocality alarms, and the travelling-wave
   identifiability trap.
7. Limits and failure modes.

## What must be done before this is submittable

- **Scattered-geometry rollout.** Every stability number here is on a uniform
  one-step stencil, where von Neumann applies and the LP is not needed. The
  central claim lives on genuinely scattered, level-varying geometry; that
  experiment does not exist yet and the paper has no core without it.
- **2-D.** Everything here is 1-D. The annihilator bookkeeping generalises
  cleanly; the positivity feasibility rate almost certainly does not, and the
  feasibility fraction in 2-D is the number that decides whether this is
  practical.
- **A real problem.** Advection-diffusion with an analytic solution is a unit
  test, not evidence.
- **WSINDy → rows.** The fair comparison for contribution 2 is: take WSINDy's
  (better) coefficients, build the same rows, impose positivity. If that works
  as well, the jet route is unnecessary even here and the paper reduces to
  "positivity certificates for discovered stencils" — which is still a paper,
  but a different and smaller one. **Run this experiment before writing
  anything.** It is the one that can kill the remaining claim.

## Honest assessment of venue

If the scattered-geometry and 2-D experiments work, this is a solid
*Journal of Computational Physics* or *CMAME* paper about certified-stable
data-driven discretisation. It is not a methods-breakthrough paper about PDE
discovery, and it should not be dressed as one.
