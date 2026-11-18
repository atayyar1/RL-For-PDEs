# G1 positioning — full-text check of SpeND and NeMDO, and the search for a closing paper

2026-09-13. Both papers read from the PDF (`pdftotext`, grep over the full text), not from
abstracts. Eight web searches for a 2024–2026 paper that time-marches a learned scattered-point
operator with a positivity / monotonicity guarantee. **Verdict: the gap is open, but the brief's
first premise needs a correction — NeMDO does time-march.**

## The three claimed absences, checked against the full text

| absence | SpeND (2609.02833, Sep 2026) | NeMDO (2603.24641, Mar 2026) |
|---|---|---|
| 1. no time-marching with the learned operator | **Confirmed.** Only modal analysis (Fig. 1) and a static convergence test (Fig. 2). Conclusion: *"Work in progress addresses the remaining validation: out-of-distribution validation, and time-dependent solves demonstrating eigenvalue stability and sustained resolving power over long integration times."* | **Refuted.** Sec. V.F marches the 2-D weakly compressible Navier–Stokes equations (Eqs. 26a–b) on an *Eulerian* disordered node set, Taylor–Green vortex, Re = 100, Ma = 0.1, to t* = 10, with NeMDO p=2 vs Wendland C2 vs LABFM p=4. **But** stability is obtained by adding dissipation, not from the operator: *"to improve numerical stability we dealiase the solution with a high-order filter at each time-step [70] … for our framework we train a hyperviscous operator GNN^hyp_{p=4}."* The time integrator is not named (ref. [70] is Jameson–Schmidt–Turkel RK). No long-time study, no growth measurement, no unfiltered run. |
| 2. no discussion of the sign of the weights | **Confirmed.** Zero hits for positiv / monoton / maximum principle / M-matrix / diagonally dominant / negative weight. The only stability-flavoured content is the loss (Eq. 4): a per-stencil penalty on ℑ{k_eff} > 0, *"positive values implying unconditional instability"*, weighted by an asymmetric γ. It is a soft, per-stencil, Fourier-proxy regulariser on scattered offsets — not a certificate. | **Confirmed.** Same grep, same result (the only "positive" hit is Wendland's *positive definite* RBFs). The nearest remark: *"the higher-order consistency requirement leads to increased weight magnitudes compared to the second-order counterpart."* |
| 3. no stability theorem / guarantee | **Confirmed.** None. | **Confirmed as a theorem; not as an analysis.** Sec. V.C builds the global derivative matrix G^D on 2500 nodes at disorder ε = 1.0 and plots its spectrum: x-derivative *"tightly clustered near the imaginary axis"*, Laplacian *"no eigenmodes exhibit growth in time (ℜ(μ) ≤ 0 ∀μ) for any of the operators considered."* Empirical, semi-discrete (no time step), one geometry. Limitations (Sec. VII) are about fixed stencil size and disorder-dependent accuracy; nothing on stability. |

Two further structural facts that matter for the experiment design:

- **Both learn *spatial* operators (∂x, Laplacian) and leave time integration to a separate
  classical integrator** (method of lines). This programme's object is the *one-step space–time
  propagator* w with rows from `rows_taylor` / `rows_moment`. The two coincide only up to the
  (cΔt)² Lax–Wendroff term (F8). So the faithful "consistency-only learned operator, marched" arm
  is **MOL: projected spatial weights + forward Euler / RK**, and the propagator-row min-norm arm is
  a second, closer-to-home control. Both are run.
- **NeMDO's consistency is soft** (*"NeMDO is not a formally consistent method (polynomial
  consistency is learnt, not enforced)"*); **SpeND's is exact by projection**, Eq. 5:
  ŵ = w̃ − C⁺(C w̃ − b), with w̃ = f_θ(normalised stencil geometry), p = 4, |N_a| = 30, 2-D.
  The minimum-norm solution is the special case w̃ = 0 of that projection, so `solve_minnorm`
  *is* the SpeND projection with a trivial network. A second surrogate (a Gaussian kernel guess
  for w̃) and a SpeND-style spectral objective in the null-space directions are also run, so that
  "consistency-only" is not one arbitrary point of the affine subspace (check 3).

## Search for a closing paper (2024–2026): none found

Candidates that came up and why each does not close G1:

| paper | what it is | why it does not close the gap |
|---|---|---|
| Nasser & Adcroft, arXiv 2606.17497 (Jun 2026), *Design principles for stable and generalizable data-driven discretizations … linear hyperbolic conservation laws* | learned FV reconstruction / flux limiter for 1-D linear advection; TVD-style penalty inspired by the local discrete maximum principle | **uniform grid**; monotonicity is a **soft training penalty**, not a guarantee; stability is empirical |
| Gueyffier, arXiv 2607.20171 (Jul 2026), *Hard Guarantees at a Measured Price* | learned FV for 2-D Euler, admissible by construction, entropy-stable interior flux | **unstructured mesh with a FV structure**, not scattered-point stencils; the guarantee is entropy stability / admissibility, not weight positivity; its headline is that the *unlearned* skeleton wins |
| Cheng et al., arXiv 2509.26186 (Sep 2025), *PDE Solvers Should Be Local* (FINO) | learnable convolutional stencils + learned time-stepping; composition error bound under a Lipschitz condition | **convolutional ⇒ uniform grid**; the bound is a Lipschitz composition estimate, not positivity |
| Shaffer et al., arXiv 2605.08436 (May 2026), meshfree exterior calculus (MEEC-Net) | structure-preserving (exact discrete conservation) learning on point clouds | conservation, not monotonicity / maximum principle; flux law learned, not stencil weights; no positivity statement |
| Shaffer et al., arXiv 2602.02788 (Feb 2026), Geo-NeW | learned FEM operator + FEEC spaces | mesh-based, steady-state |
| STENCIL-NET (Sci. Rep. 2023), Bar-Sinai et al. (PNAS 2019), PDE-Net | learned stencils + time-stepping, "stable by design" via RK/TVD inductive bias | **Cartesian grids**; no positivity certificate |
| arXiv 2607.10858 (Jul 2026) positivity-preserving learned ODE integrators via NSFD | positivity for **ODE** integrators | no spatial operator, no scattered geometry |
| Trask et al. GMLS-Nets | CNN-like layers on unstructured stencils via GMLS | regression architecture; no marching guarantee |

Search strings used (all Sep 13, 2026): learned meshfree stencil positivity-preserving time integration;
"discrete maximum principle" learned discretization stencil scattered; RBF-FD neural learned weights
M-matrix monotonicity; STENCIL-NET stable by design; graph neural network learned differential operator
point cloud positivity/monotone/maximum principle; structure-preserving learned discretization point
cloud maximum principle; learned GFDM/MLS neural stencil stability time integration; learned/neural
meshfree advection monotone/upwind/positivity guarantee. Every hit that mentioned positivity or a
maximum principle was on a grid or a mesh; every scattered-point hit was consistency-only.

## What the experiment therefore has to show

The open question is exactly as the brief states it, with one refinement: NeMDO's TGV run is
evidence that a consistency-only scattered operator *can* be marched — **with a filter and a
trained hyperviscosity**. So the kill condition should be read as: if the unfiltered
consistency-only operator is stable on every test here, positivity is redundant *and* NeMDO's
filter was unnecessary too. If it is unstable and positivity fixes it, the result is a certificate
where the two papers currently have a regulariser (SpeND's ℑ{k_eff} penalty) and a filter (NeMDO).

Prior-art risk after this read: **low**. Proceeding.
