# Positioning — is "discover the discrete integrator from data" an open problem?

2026-09-13. Twelve web searches and three full-text reads, done *before* any experiment. The
question was whether the space is crowded. Answer: **much more crowded than the programme assumed,
including one paper eleven days old that is close to Ali's exact construction — but two specific
gaps survive full-text inspection.**

## The map

| axis of the programme | occupied by | when |
|---|---|---|
| learn the kernel on a fixed grid, moment-constrained, discover the PDE | PDE-Net; PDE-Net 2.0 (symbolic) | 2018–19 |
| known part as FD operators + unknown part as NN ("partially known physics") | PDE-Net++ | 2023 |
| data-dependent stencil coefficients on a coarse grid (a learned limiter) | Bar-Sinai et al., PNAS | 2019 |
| data-only learned stencils *and* time-stepping, fixed grid, "stable by design" | STENCIL-NET | 2021 |
| discrete evolution operator from data; parametrised, partially observed, **with memory** | flow-map learning, Xiu group | 2019–23 |
| geometry → consistent weights on **scattered** points, GNN, moment constraints | NeMDO, arXiv 2603.24641 | Mar 2026 |
| same, consistency exact by projection, data-free spectral objective | **SpeND, arXiv 2609.02833** | **Sep 2, 2026** |
| RL chooses the time step | Dellnitz et al. | 2021 |
| RL chooses mesh refinement | Yang; Foucart | 2021–23 |
| learned schemes with hard guarantees (finite volume, on a mesh) | several | 2026 |
| positivity-preserving learned ODE integrators via NSFD | arXiv 2607.10858 | Jul 2026 |
| active learning / OED for PDE surrogates and discovery | several | 2022–25 |

Every *component* of "discover the discrete integrator, constrain or match it to known physics,
place points actively, generalise the classical schemes" is published. The two closest — NeMDO and
SpeND — are precisely "learn the map from local scattered geometry to consistent stencil weights",
which is Ali's construction with a network in place of the pseudoinverse.

## What survives full-text inspection

Read in full, both 2026 papers share the same three absences:

1. **⛔ Corrected by G1 (full-PDF read).** SpeND does not time-march — confirmed verbatim: "work in
   progress addresses … eigenvalue stability and sustained resolving power over long integration
   times." But **NeMDO does**: §V.F runs the 2-D weakly-compressible Taylor–Green vortex on a
   disordered Eulerian node set to t\* = 10. Its stability comes from **added dissipation, not the
   operator** — "to improve numerical stability we dealiase the solution with a high-order filter
   at each time-step … we train a hyperviscous operator GNN^hyp." No long-time study, no unfiltered
   run, an empirical eigenvalue section (§V.C) but no theorem. My original "neither marches" came
   from a fetch summary that missed §V.F; recorded as an instance of trusting the summariser.
2. **Neither discusses the sign of the learned weights** — no positivity, monotonicity, maximum
   principle, or von Neumann analysis. Consistency only.
3. **Neither chooses where to evaluate.** Point sets are given (k-nearest in NeMDO; prescribed
   disordered distributions in SpeND). Nothing is adaptive about *where*.

Those three absences are exactly this programme's surviving results: the positivity certificate is
the only computable stability certificate on scattered geometry (no symbol, so no von Neumann);
maximum entropy is the constructive choice of point within the feasible set; and the
generating-function rows reduce to FTCS / Lax–Wendroff exactly on a uniform grid.

## The two gaps, and what would close each

**G1 — certified scattered learned operators under time-marching.** Sharpened after the NeMDO
correction: **can a positivity certificate replace the ad-hoc filter?** NeMDO marches only with a
high-order dealiasing filter plus a trained hyperviscous operator; SpeND does not march. Neither has
a sign condition on the weights (grep over both full PDFs: zero relevant hits). Nearest misses —
Nasser–Adcroft 2606.17497 (TVD-penalised learned FV), Gueyffier 2607.20171 (entropy-stable learned
FV), FINO 2509.26186 — are grid/mesh-based or soft. Prior-art risk **low**. **Kill condition:
if consistency-only rollouts are already stable on every test, positivity is redundant here.**

**G2 — active placement when the operator is unknown.** Every RL-for-numerics paper uses a *known*
solver; every learned-operator paper uses *given* points. The coupling — the integrator chooses
where to look *and* learns the kernel from what it sees — is not in the map. It has content only if
acting changes what you know, i.e. the operator is unknown and locally varying (T2's band,
L_α ∈ [6, 20] cells). Prior-art risk **medium** — this is a combination, and combinations are where
positioning has failed before. **Kill condition: if "estimate the operator everywhere, then place by
dual-weighted residual" matches the joint policy, the coupling is fictitious.**

## Honest framing for the book

The *generalisation* Joseph describes exists as a class — learned discretisations, 2018–2026. What
does not exist is the combination with certificates and active placement on scattered geometry.
Relative to SpeND that is an **increment**, not a foundation. It is a good increment: it answers
the question SpeND's authors say they are working on, with a certificate they do not have. But it
should be pitched as that.

Not verified: arXiv 2607.20171 ("Hard Guarantees at a Measured Price") — the search snippet
described entropy-stable learned FV with zero negativity events; the fetched page resolved to a
different paper. Treated as "FV on a mesh, not scattered" pending a proper read.
