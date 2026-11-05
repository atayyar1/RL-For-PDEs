# G2 — Literature positioning: active placement when the operator is unknown

**Thread question**: unknown, spatially varying α(x) with L_α = α/|α′| in [6, 20] cells; predict
u(x\*, t\*) from a budget of N point evaluations of the true system; each evaluation both feeds
the reconstruction and the local estimate of α. Does a policy that treats the two jointly beat
(A) estimate-then-place-by-DWR and (B) myopic information gain?

**Status: positioning done 2026-09-13, 17 searches, 6 full-text reads, 12 abstract-level checks, no experiments run.
Waiting for a reply before running anything.**

**Verdict in one line: the coupling is published as a principle (Feldbaum 1960), as a theorem
with the exact "acting only pays when the operator is unknown" statement (Krause & Guestrin
2007), and as the static A-vs-C comparison (Zimmerman 2006). What is not published is this
instance — a PDE-constrained local estimator, a single space-time target, and the L_α band —
and the theorem already predicts the shape of what the instance would show.**

---

## ⛔ STOP — read this box first

**If you are about to think "the agent must trade off learning the operator against using it,
and that trade-off is what RL is for", that is Feldbaum's dual control (1960–61): the control has
a *dual effect*, probing to learn the parameters and caution to use them. Sixty-five years of
literature, with a survey (Wittenmark 1995) and a 2026 arXiv line still active.**

**If you are about to think "when the operator is known the optimal placement is computable
before acting, so sequential design has value only when it is unknown", that sentence is in the
introduction of:**

> **A. Krause & C. Guestrin, "Nonmyopic active learning of Gaussian processes: an
> exploration–exploitation approach", *ICML* 2007.**
>
> "if the GP model parameters are completely known, the predictive variances do not depend on
> observed values, and hence nothing is lost by committing to sampling locations in advance. …
> In the case of unknown parameters however, this independence is no longer true."
>
> Their Theorem 1 **bounds the advantage of any sequential policy over the best a priori design
> by the entropy H(Θ) of the prior over the unknown kernel parameters** (plus a submodularity
> slack). Section 6 extends it to **nonstationary GPs with local structure** — spatially varying
> kernel parameters, i.e. the spatial analogue of α(x). They compare a priori design, random,
> three exploration strategies (IE, ITE, IGE), and the joint exploration–exploitation policy, on
> river-pH and Intel-lab temperature data, and report a ~50% RMS gap between nonstationary
> sequential and isotropic/a priori, with "none of the exploration strategies dominates".

**If you are about to think "policy A — spend a fixed fraction on estimating α, then place by
the estimated operator — is the obvious straw man", it is a named design criterion in
geostatistics, and its comparison against the joint objective has been done:**

> **D. L. Zimmerman, "Optimal network design for spatial prediction, covariance parameter
> estimation, and empirical prediction", *Environmetrics* 17:635–652 (2006).** Three criteria —
> prediction with known covariance, covariance-parameter estimation, and *empirical* (plug-in)
> prediction that accounts for the parameters having to be estimated from the same data. Finding:
> the first two objectives "are largely antithetical and lead to quite different optimal
> designs", and the empirical-prediction optimum is a compromise — mostly space-filling with a
> few clustered pairs to pin the covariance. Companion: Zhu & Stein, *JABES* 11:24–49 (2006),
> "Spatial sampling design for prediction with estimated parameters"; Diggle & Lophaven,
> *Scand. J. Stat.* 33:53–64 (2006), "Bayesian geostatistical design".

That is A (design for prediction with plug-in parameters), B (design for parameter estimation),
and C (design for the end-to-end empirical prediction error) — in the *a priori* setting.
Krause–Guestrin is the *sequential* setting. Between them the coupling in the brief is covered.

---

## The map — every item the brief asked me to check

| work | operator known to the method? | chooses *where* to evaluate? | evaluations serve both purposes? | objective |
|---|---|---|---|---|
| **Bayesian OED for PDE coefficient inversion** (Alexanderian et al. 2014–; sparsified A-opt; robust A-opt 2023; multi-type sensors 2026) | forward model known, coefficient field unknown | yes, sensor placement | no — evaluations serve inference of the field only | posterior variance of the *parameter* |
| **Goal-oriented OED** (Attia, Alexanderian, Saibaba 2018; Wu–Chen–Ghattas 2023; GO-OED nonlinear/MCMC 2024; quadratic-approx GO-OED 2024) | same | yes, **batch / a priori** | evaluations serve inference; the QoI is *computed through* the inferred parameter | posterior variance / EIG of a **prediction QoI** — this is the field-target coupling, without sequentiality |
| **Sequential OED via policy-gradient RL** (Shen & Huan, CMAME 2023; **2601.05868, Jan 2026, infinite-dimensional**, "sequential multi-sensor placement for contaminant source tracking") | forward model known; unknown is a *source / random field* | yes, sequential, an RL policy, and they explicitly say the policy "generalizes batch and greedy" | inference only | D-optimality / EIG on the parameter; the abstract says other utilities "can also be used" — a goal-oriented utility is one line away |
| **Uciński 2004** (book), moving sensors for DPS identification | PDE structure known; **spatially varying coefficients** unknown | yes, sensor trajectories, sequential variants | inference only | D-optimal Fisher information on the coefficients |
| **Targeted observations + parameter estimation in DA** (Bellsky, Kostelich, Mahalov, *Chaos* 2014; Frontiers Climate 2022 EAKF/TVR) | model known up to *global* parameters | yes, targeted observations | 2022: **different** observation sets for state and for parameters, by design ("excessive parameter corrections") | state RMSE / parameter spread; parameters not spatially varying |
| **Krause & Guestrin 2007** | kernel parameters unknown, **nonstationary extension** | yes, sequential | **yes** — the same observations reduce H(Θ) and predictive variance | mutual information / RMS over the **whole field** |
| **Zimmerman 2006; Zhu–Stein 2006; Diggle–Lophaven 2006** | covariance parameters unknown | a priori design | **yes** — empirical-kriging criterion | prediction MSE with plug-in parameters, field-averaged |
| **Li / Musekamp et al., "Active learning for neural PDE solvers", 2408.01536** and its 2026 follow-ups (2605.21348 physics-based AL for neural operators) | classical solver **known** and queried as oracle | chooses which *instances* (ICs, PDE parameters) to simulate — **not spatial locations** | the queried solutions train the surrogate only | surrogate test error |
| **RLMesh, 2603.02066** (Mar 2026), RL-guided mesh for PDE surrogates | classical solver generates data; surrogate is the learned operator | **yes** — the RL agent allocates mesh points, and the mesh feeds surrogate training | closest of the RL group: mesh choice both resolves the solution and trains the operator | surrogate accuracy via a proxy-model reward; no local unknown coefficient, no single target; read at abstract level only |
| **Dellnitz et al. 2021/SISC 2023**, RL time-stepping | **known** right-hand side; RL controls step size of a classical scheme | when (step), not where | no | error/cost of a known integrator |
| **Yang et al. 2021/2023; Foucart et al. 2023**, RL for AMR | **known** PDE; the RL policy replaces the error estimator | yes, refinement | no — the operator is never learned | error/cost of a known solver |
| **STENCIL-NET 2101.06182** | **unknown** PDE, learned discrete operator | **no** — "regular Cartesian grids"; "solution-adaptive" means the stencil *coefficients* depend on the local solution, not the point set | n/a | rollout accuracy |
| **Bar-Sinai et al., PNAS 2019** | **known** PDE, learned coarse-grid derivative estimates | **no** — fixed coarse grid; data-dependent coefficients | n/a | coarse rollout accuracy |
| **PAGP 2204.02583** (physics-assisted GP with active learning, forward + inverse) | inverse mode: **constant** λ as an extra hyperparameter | active learning by posterior variance, **run only on the forward discrete-time model**; not on the inverse problem | no | field L2 error |
| **PINN residual-based adaptive sampling** (RAR-G / RAR-D, Wu et al. 2023) | unknown coefficient possible in inverse PINNs | chooses *collocation* points where the residual is enforced, not measurements of the true system | no | training loss |

**Answering the brief's specific questions plainly**

- Dellnitz 2021, Yang 2021, Foucart 2023: **known operator, all three.** The RL agent replaces a step-size controller or an error estimator for a solver it does not learn. None of them has anything to learn about the dynamics by acting; the value of RL there is amortising the error estimator, which is why T4's "RL over scheme parameters has its optimum on a boundary" finding was predictable.
- STENCIL-NET and Bar-Sinai 2019: **"solution-adaptive" never means choosing *where*.** Fixed regular grids in both; adaptivity is in the coefficients as a function of the local solution values.
- Li et al. 2408.01536 and its citations: active learning over *which simulations to run* (ICs and PDE parameters), with a known classical solver as oracle. Not spatial placement, and the queried evaluations do not feed a reconstruction. The one RL paper in that cluster that does choose spatial resolution and does feed the learned operator is RLMesh (2603.02066), but it has no locally varying unknown coefficient and no single target.
- Bayesian OED for coefficient identification: chooses where, and the coefficient field is unknown, but the evaluations serve inference only; the goal-oriented variants route a prediction QoI through the inferred parameter, which is the coupling, done a priori; the sequential RL variants (Huan group) make it sequential, for parameter EIG. **I did not find a paper that is simultaneously sequential, goal-oriented on a single prediction target, and inverting a spatially varying coefficient.** All three components are there and the Huan-group framework accommodates the combination by changing the reward; treat "not found" as "not found in 17 searches", not as absent.

---

## What each arm in the brief already is

| brief | literature name | where |
|---|---|---|
| **A, decoupled** (fixed fraction to estimate α everywhere, then DWR with the estimate) | plug-in / empirical kriging design; "explore then exploit" with a fixed switch | Zimmerman 2006; Zhu–Stein 2006; KG07's two-phase algorithm with the stopping rule of their Corollary 2 replaced by a fixed budget |
| **B, myopic information gain** (α-uncertainty × influence on the target) | goal-oriented greedy sequential design; KG07's IGE weighted by the QoI sensitivity | Attia et al. 2018 (batch, goal-oriented); KG07 §5 (sequential, field-oriented); the combination is one line |
| **C, joint** (parametric placement policy optimised end-to-end on final error at the target) | sequential goal-oriented OED with an amortised policy | Shen & Huan 2023; 2601.05868; KG07's nonmyopic policy (with MI, not a target) |

And the expected result is not open either: **KG07 Theorem 1 says the C-over-A gap is bounded by
the prior entropy of the unknown operator restricted to what matters for the target.** In the
L_α band that entropy is largest in the middle — at L_α ≫ 20 a global fit removes it cheaply
(A ≈ C), at L_α < 6 the local estimator cannot reduce it at any budget (A ≈ B ≈ C ≈ bad). So a
sweep will show a hump in the band with zero at both edges, *whatever the hump's height*. The
experiment can only measure how much of the bound is attained; it cannot discover the shape.

---

## What is not published — honestly sized

1. **The estimator.** Every work above learns the operator through a GP kernel, a Fisher
   information matrix, or a neural surrogate. This programme's estimator is different in kind:
   α(x) recovered from the **PDE residual on scattered mixed-time-level points** via the
   generating-function rows (T3/F42), and the reconstruction done by **max-entropy consistent
   stencils** on the same points. With a GP kernel, learning the parameters needs *close pairs*
   (Zimmerman's clustered points); with a residual estimator it needs *local clusters spanning
   two time levels* of ≥ 4–5 points. So the geometry the joint policy learns would be a different
   object from the geostatistical compromise design. That is an *observation about what C would
   look like*, not a claim of content.

2. **The target.** KG07 and the geostatistical designs are field-oriented (MI / average MSE).
   Goal-oriented OED is target-oriented but a priori, or sequential with parameter EIG. The
   *single space-time point target with an influence cone* is this programme's framing (F16) and
   is DWR on the reconstruction side. Coupling the cone to an unknown-operator entropy is an
   instance, not a new principle.

3. **The band.** T2/F28 measured where the local-α rows live (L_α ≳ 6 cells) and where a global
   fit suffices (≳ 20). Nobody has measured the attained sequential advantage as a function of
   L_α with a PDE-constrained estimator. This is the one number the thread could produce that is
   not in the literature, and it is a number, not a result.

**Prior-art risk after this search: high on the framing, low on the instance.** Two of this
programme's threads were killed by one search each; this one survives as an instance only, and
its most likely outcome is confirming a 2007 theorem's bound in a new setting.

---

## Recommendation

**Do not run this as a novelty thread.** The coupling has content, and the content is
Feldbaum's, Krause–Guestrin's, and Zimmerman's. Writing it up as "active placement with an
unknown operator" invites the reviewer to cite ICML 2007 and stop reading.

**If it is run at all, run it as a one-day measurement with a different question:** *how much of
the Krause–Guestrin entropy bound does a PDE-constrained local estimator attain, across L_α?* Same
three arms, same kill condition, same discipline — but the deliverable is the attained fraction of
a known bound, and the interesting failure is if it attains *more* than the bound allows (which
would mean the bound's assumptions — discretised parameter prior, submodular MI — do not transfer
to residual estimators and the theorem needs a PDE version). That is a real question; it is also
a small one.

**The additional kill condition** the literature imposes, on top of the brief's: if C's learned
geometry is "a few clusters to pin α plus a DWR cone to the target", it is Zimmerman's compromise
design with a cone, and the thread should close with that sentence.

**Where the programme's actual gap sits, given this map:** every work in the table with an
unknown operator learns it *globally* (GP kernel, surrogate, FIM over the whole field) and every
work with a target uses a *known* operator for the adjoint. The estimator that is local in the
sense of T2/T3 — the operator known only within L_α of where you have looked — has no adjoint
and no global posterior. REEVALUATION.md's direction (2), "operator learning where no adjoint
exists — learn the influence function", is the version of this thread that the table does not
cover. G2 as briefed uses the estimated operator to *build* the adjoint (arm A does exactly that),
which reintroduces the known-operator assumption one step later. That is the filter from
REEVALUATION.md failing in the method, and it is the reason to stop here rather than after the
sweep.

---

## Sources read in full or at abstract+section level

- Krause & Guestrin, ICML 2007 — full text (PDF, 8 pp).
- Zimmerman, Environmetrics 2006 — abstract and design-criteria section.
- PAGP, arXiv 2204.02583 — full text; §2.6 (active learning) and §3 (examples).
- Shen & Huan, CMAME 2023 (arXiv 2110.15335); 2601.05868 — abstracts.
- Attia, Alexanderian, Saibaba 2018 (arXiv 1802.06517) — abstract + intro.
- Dellnitz et al. (arXiv 2104.03562); Yang et al. (2103.01342); Foucart et al. 2023; STENCIL-NET
  (2101.06182); Bar-Sinai et al. PNAS 2019; Musekamp et al. (2408.01536); RLMesh (2603.02066) —
  abstracts, with the specific known/unknown and where/when questions checked.
- Bellsky, Kostelich, Mahalov, Chaos 2014 — abstract only (full text paywalled).
- Frontiers in Climate 2022 (EAKF/TVR adaptive observations for parameters) — full text.
- Uciński 2004 — publisher description and table of contents; not re-read.

Checked last: arXiv 2507.02500 ("goal-oriented optimal sensor placement … dynamic sensor
steering, crisis management") — unknown is a **source term**, operator known, C-optimal design on
a prediction QoI, both static placement and dynamic steering. Sequential + goal-oriented on a PDE
inverse problem, but not a coefficient field. It confirms the pattern in the table rather than
closing the "not found" cell.

---

## Addendum (2026-09-13, on request) — learning the influence function where no adjoint exists

**Closed, from three sides.** The influence function G(x,t; x\*,t\*) *is* the Green's function of
the operator, restricted to one target column, so "learn the influence function from evaluations
alone with an unknown operator" is the title of an existing literature: **Boullé, Kim, Shi &
Townsend, "Learning Green's functions associated with time-dependent PDEs", *JMLR* 23 (2022)**
(parabolic, from input–output pairs, with sample-complexity bounds via randomised SVD of
Hilbert–Schmidt operators and a hierarchical space-time partition); Boullé & Townsend, "Elliptic
PDE learning is provably data-efficient" (2023); and, verbatim, **"Operator learning without the
adjoint", arXiv 2401.17739** — none of which chooses where to evaluate or targets a point. The
*target-specific, adjoint-free* estimate from samples is **ensemble sensitivity — Ancell & Hakim,
*Mon. Wea. Rev.* 135:4117 (2007)**: regress an ensemble of states onto a single-point forecast
metric (their example is sea-level pressure at one point in Washington State), which they prove
equals the error covariance projected onto the adjoint sensitivity, and which they then use for
*observation targeting* — i.e. F16's influence-weighted placement with the adjoint replaced by a
regression on evaluations, and no operator opened. And **KG07 §6 does already contain the
corner**: in a GP the influence of an observation at s on the prediction at y is the kernel through
Σ_yA Σ_AA⁻¹, so learning the local kernel parameters Θ⁽ⁱ⁾ by exploration *is* learning the influence
structure, and Ancell–Hakim's regression is the same covariance object estimated from an ensemble
rather than a parametric prior; what KG07 lacks is only the single-point target (its criterion is
field-wide MI), which Ancell–Hakim supplies. The two learned-adjoint papers the lead named do not
drop the operator: **arXiv 2102.12450** solves a *known* adjoint PDE in strong form with a
feedforward network (an accelerator), and **E2N, arXiv 2207.11233** replaces the error-estimation
step with a network trained on local mesh/physics features, with the primal solved classically.
**Nearest miss, not found**: choosing *where* to evaluate in order to learn one target column of
the Green's function (active learning of a Green's function, or active ensemble sensitivity). The
closest are "Learning Where to Simulate" (arXiv 2606.09949) and physics-based active learning for
neural operators (2605.21348), both field-wide surrogate training — but by KG07 Theorem 1 that
missing piece is again "reduce the entropy of the covariance parameters that matter for the
target", an instance of their exploration phase rather than a new problem. Thread closed.
