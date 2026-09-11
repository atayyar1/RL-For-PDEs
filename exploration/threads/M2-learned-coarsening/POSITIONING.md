# M2 — Literature positioning

**Thread question**: choose the coarse variables to minimise the memory the coarse law requires,
subject to the coarse law being non-negative / stable.

**Written before any experiments, as instructed.** Every claim is labelled
**[classical]** (someone else's result, cited), **[measured here]** (run today in
`positioning_check.py`), or **[conjectured]**.

---

## Verdict, up front

I tried to talk myself out of this thread and I mostly succeeded.

1. **The thread's title is a 2013 JCP paper.** Guttenberg, Dama, Saunders, Voth, Weare & Dinner,
   *"Minimizing memory as an objective for coarse-graining"*, J. Chem. Phys. **138**, 094111 (2013).
   They define a memory measure over choices of retained coordinates, minimise it, and report
   that it "suggests coordinate selections that are physically intuitive and reflect molecular
   structure." **That is Task 3, stated and answered, thirteen years ago.** I did not know this
   paper this morning; it took one search.

2. **Task 1 has a classical answer three times over**, and I verified the linear-algebra version
   numerically today rather than trusting my memory of it: minimal exact memory depth is the
   Krylov/observability index of the pair (P, A) minus one; zero memory ⟺ the coarse subspace is
   invariant. This is exact lumpability (Kemeny–Snell), minimal realisation theory (Kalman), and
   the statement that the Mori–Zwanzig kernel vanishes on invariant subspaces, all at once.

3. **The unconstrained problem is worse than classical — it is degenerate.** [measured here]
   *Every* invariant subspace of the right dimension achieves p\* = 0 exactly, including the
   **fastest** one. Memory depth is a rank condition; it is blind to eigenvalue magnitude. So
   "minimise memory" does **not** recover the slow variables on its own and is **not** the same
   objective as VAMP. VAMP's optimum is one point in memory-minimisation's enormous optimal set.
   Answering Q1: *not identical, strictly weaker, and degenerate without a second criterion.*

4. **Q4 — the thing the brief believed was thin — is not thin.** Requiring the reduced law to be
   non-negative/positive has at least three established homes: the **positive realization problem**
   (Anderson–Deistler–Farina–Benvenuti 1996; Benvenuti–Farina tutorial 2004; Benvenuti survey,
   *Automatica* **143**, 2022), **positivity-preserving model order reduction** (Li–Lam et al.; Grussler–Damm;
   Cortese–Grigoletto–Ticozzi–Ferrante 2025), and **PCCA+** (Deuflhard & Weber, *LAA* **398**, 161–184, 2005), which finds a
   rotation of the dominant invariant subspace making the memberships non-negative and the coarse
   propagator a stochastic matrix. Worse for us: the framework claim that *positivity is bought
   with extra state dimension* is the **known central fact** of positive realization theory —
   the minimal positive realization can be strictly larger than the minimal realization, and
   characterising minimality there is itself still open.

5. **Q5 — p\* is a reparameterisation of statistical complexity** in this setting. [classical]
   Causal states are the minimal sufficient statistic for prediction (Crutchfield–Young 1989;
   Shalizi–Crutchfield 2001); for a linear system the causal-state construction *is* the minimal
   realisation, and depth-p AR on K variables = Markov on pK variables, so "lags" and "state
   dimension" are the same currency in different units. Not a new quantity.

**Recommendation: do not run Tasks 1 and 3 as stated.** Task 1's answer is classical and I have
already verified it; Task 3 is Guttenberg et al. 2013. Task 2 as stated is largely covered by
positive realization theory. There is **one uncleared corner**, described in §7, and it is much
smaller than the thread's brief. I would rather hand M1 the two facts below and close M2 than
run a reduced version of it — but §7 is a genuine question and the call is yours.

Two facts M1 can use immediately, both measured today:

- Decimation by M is the **worst possible** projection for memory: p\* ≤ ⌈N/K⌉ − 1 holds trivially
  for every P (the Krylov stack saturates at rank N), and decimation attains that ceiling exactly.
  So F14's "exactly M − 1 lags" is not special structure — it is the trivial upper bound.
- At M = 3, the one-step coarse map P A Pᵀ for the 3-point FTCS stencil is **exactly diagonal**
  (measured: VAMP-2 = 0.2000 = 20 × (1−2r)²): the coarse cells cannot see each other in one step,
  because the stencil reaches 1 fine cell and the coarse spacing is 3. That is a mechanical reason
  for F23's pure-delay structure, and it says the pure delay is a statement about *reach*, not
  about positivity.

---

## 1. Markov state models, VAMP, tICA, TCCA, diffusion maps, committors

**Is "minimise memory" identical to "maximise the VAMP score"? No — and I can show the gap.**

### The analytic relationship

Fine dynamics `u_{n+1} = A u_n` on R^N; coarse variables `y_n = P u_n`, P of rank K.

> **[classical]** An exact *Markov* coarse law `y_{n+1} = Â y_n` exists iff `PA = ÂP` for some Â,
> i.e. iff `rowspace(P)·A ⊆ rowspace(P)` — the coarse subspace is A-invariant. Equivalently
> `range(Pᵀ)` is Aᵀ-invariant, i.e. spanned by left eigenvectors of A.

This single condition is simultaneously:

| field | name | reference |
|---|---|---|
| Markov chains | **exact lumpability** | Kemeny & Snell, *Finite Markov Chains* (1960); Buchholz, *J. Appl. Prob.* **31** (1994) |
| control | unobservable subspace is A-invariant; **exact reduction** | Kalman (1963); Luenberger |
| statistical mechanics | the **Mori–Zwanzig memory kernel vanishes** | Mori (1965), Zwanzig (1961) |
| transfer operators | the subspace is **Koopman-invariant** | Wu & Noé, *J. Nonlinear Sci.* **30** (2020) |

> **[classical]** The minimal memory depth is the **Krylov / observability index minus one**:
> p\* = min{ q : `P A^q ∈ rowspan{P, PA, …, PA^{q−1}}` } − 1.

**[measured here]** `positioning_check.py`, FTCS on a 60-cell ring at r = 0.45, ν = 0.0225.
Rank computation and an independent held-out AR regression on 400+ random initial conditions
agree exactly; the regression at depth p\* has relative error ~2e-15 and at depth p\*−1 has
relative error O(1), so p\* is minimal and the identification is not an artefact of an
underdetermined fit.

| projection | K | p\* | AR resid at p\* | at p\*−1 |
|---|---|---|---|---|
| decimation M=2 | 30 | 1 | 2.1e-15 | 1.2e+00 |
| decimation M=3 | 20 | 2 | 2.2e-15 | 4.4e-01 |
| decimation M=4 | 15 | 3 | 3.7e-15 | 2.5e-01 |
| decimation M=6 | 10 | 5 | 2.5e-15 | 7.4e-02 |
| block average M=3 | 20 | 2 | 2.1e-15 | — |
| random dense | 20 | 2 | 9.9e-15 | — |

This reproduces T5/F14's "exactly M − 1 lags" by a completely different route (rank of a Krylov
stack, not Cayley–Hamilton on the alias subspace), which is a useful cross-check — **but it also
shows the result is the trivial ceiling**, since p\* ≤ ⌈N/K⌉ − 1 = M − 1 for *any* rank-K P.

### Where VAMP and memory come apart

**[measured here]** Same ring, K = 20, three different A-invariant subspaces:

| dim-20 invariant subspace | p\* | VAMP-2 | slowest \|λ\| retained |
|---|---|---|---|
| 20 **slowest** eigendirections | **0** | 15.66 | 1.000000 |
| 20 **middle** eigendirections | **0** | 4.19 | 0.628255 |
| 20 **fastest** eigendirections | **0** | 1.39 | 0.429393 |
| decimation by M = 3 (not invariant) | 2 | 0.20 | — |

Memory depth cannot tell these apart. VAMP-2 separates them by an order of magnitude. So:

> **[classical, restated]** VAMP-optimality ⟹ zero memory. Zero memory ⇏ VAMP-optimality.
> Minimising memory is the *strictly weaker* condition, and its optimal set is the whole
> Grassmannian of K-dimensional invariant subspaces.

Concretely: VAMP-2 maximises Σ σ_i² of the Koopman operator restricted to the subspace, which is
a *magnitude* criterion; Markovianity is an *invariance* criterion. They coincide only because
the top-K singular subspace happens to be invariant. Any objective of the form "minimise memory"
must be regularised by an accuracy/retained-variance term to have a unique answer, and once you
add that term you are doing VAMP — **with one exception**, which is the positivity constraint (§7).

### The rest of the family

- **tICA** (Pérez-Hernández, Paul, Giorgino, De Fabritiis, Noé 2013) = VAMP-2 for reversible
  dynamics; **TCCA** = time-lagged canonical correlation analysis, which is VAMP for the
  linear-Gaussian case.
- **Diffusion maps** (Coifman–Lafon) give the same slow eigenfunctions from a kernel.
- **Committor as the optimal reaction coordinate**: Berezhkovskii & Szabo; E & Vanden-Eijnden's
  transition path theory. **[classical]**
- **SGOOP** (Tiwary & Berne, PNAS 2016) *literally* optimises collective variables to maximise
  the spectral gap of the coarse dynamics — the closest thing in this family to "attend to
  whatever lets you forget", and a decade old.
- **MSM practice**: the Chapman–Kolmogorov test and implied-timescale convergence (Prinz et al.,
  *JCP* **134**, 174105, 2011) are exactly "tune the clustering and lag until the coarse dynamics
  is as Markovian as possible." That has been the community's daily method since ~2007.
- **"Slicing and Dicing: Optimal Coarse-Grained Representation to Preserve Molecular Kinetics"**
  (Yang, Templeton, Rosenberger, Bittracher, Nüske, Noé, Clementi, *ACS Cent. Sci.* 2023) selects
  the coarse mapping to preserve the slow spectrum, and explicitly reports that information-
  theoretic and structure-based mappings fail to do so.
- **Legoll & Lelièvre**, *Nonlinearity* **23** (2010), "Effective dynamics using conditional
  expectations", plus the quantitative follow-ups with Olla, Sharma, and Zhang: error bounds for
  the Markovian effective dynamics in terms of how slow the reaction coordinate is. This is the
  rigorous version of "good coarse variables are the ones that let you forget."

> **Framework note.** "Attend to whatever lets you forget" is a good sentence and it is not a new
> idea. It is the founding premise of the collective-variable literature. The honest version of
> the framework claim is narrower: *given* that you must forget, the interesting structure is in
> what the forgetting costs, not in the fact that good variables minimise it.

---

## 2. Optimal prediction and lumpability

**[classical, both.]**

**Optimal prediction** (Chorin, Hald & Kupferman, *PNAS* **97** (2000); *Physica D* **166** (2002);
Chorin & Stinis, *CRM* 2006) takes the projection as **given** — usually conditional expectation
onto the resolved variables — and derives the memory term. It answers "what memory does this
projection require," which is our p\*, and does not ask "which projection minimises it." That gap
is real but it is the gap that Guttenberg et al. 2013 closed.

Relevant caveat from that literature, which bears on the thread's framing: Mori's *linear*
projection's quality "depends almost entirely on the choice of resolved variables," while
Zwanzig's *nonlinear* projection satisfies an optimality condition but needs conditional
expectations. So "choose the projection" is a recognised weak point, not an unnoticed one.

**Lumpability** (Kemeny & Snell 1960): a partition is (strongly) lumpable iff within-block row
sums to each block are constant — the indicator subspace is invariant. **Weak lumpability**
(Rubino & Sericola) is initial-distribution dependent. **Nearly completely decomposable** chains
(Simon & Ando 1961; Courtois 1977) are the approximate theory. Optimal-lumping *algorithms* exist
(signature-based symbolic algorithms, TACAS 2007; spectral lumping for non-reversible chains,
Deuflhard–Weber lineage). Note the important structural point: **lumping is restricted to
partitions, which buys non-negativity of the coarse law for free** and pays for it in
Markovianity. That is one half of the trade this thread wanted to discover.

---

## 3. Renormalisation group / real-space decimation

**"Choose the blocking so the coarse Hamiltonian stays simple" is the central technical problem of
real-space RG, and it has an information-theoretic optimum with a proof.** [classical]

- Kadanoff block spins; decimation generically **proliferates couplings** — long-range and
  multi-spin terms appear. That proliferation is the spatial analogue of memory growth, and it has
  been the known obstruction since the 1970s.
- **Koch-Janusz & Ringel**, *Nature Physics* **14** (2018): choose the coarse-graining rule by
  maximising **real-space mutual information** (RSMI) between the block variable and its
  environment.
- **Lenggenhager, Gökmen, Ringel, Huber & Koch-Janusz**, *PRX* **10**, 011037 (2020), "Optimal
  Renormalization Group Transformation from Information Theory": **proves** that an RSMI-optimal
  coarse-graining *does not increase the range of interactions in the renormalised Hamiltonian*,
  and that it is optimal in suppressing complexity measures at fixed retained information.
- Follow-up: "Minimizing couplings in renormalization by preserving short-range mutual
  information" (arXiv:2107.00990) — the objective in the title is this thread's objective in the
  spatial direction.
- The deep negative result on the constrained side: **van Enter, Fernández & Sokal**,
  *J. Stat. Phys.* **72** (1993) — RG transformations can map a Gibbs measure to a **non-Gibbsian**
  one, i.e. no well-defined coarse Hamiltonian exists at all. That is the sharpest known version
  of "coarse-graining can destroy the structure you wanted to preserve," and it is the honest
  ancestor of F14's "coarse-graining destroys positivity."

There is also a numerical-analysis version we should not ignore, because it is the *same problem
in our own setting*: **algebraic multigrid** chooses coarse variables and the interpolation
operator, and the Galerkin coarse operator PᵀAP suffers **stencil growth** (the coarse law becomes
less local as you go down the hierarchy) — addressed by **non-Galerkin coarse grids**
(Falgout & Schroder, Bienz et al.). Additionally, AMG practitioners know the Galerkin coarse
operator can lose the **M-matrix** property, which is the positivity/discrete-maximum-principle
property we care about. And AMG coarsening and interpolation are now routinely *learned*
(Luz, Galun, Maron, Basri & Irani, ICML 2020; Greenfeld et al. 2019; RL-based coarsening).

**This is the closest field to the thread, in our own problem class, and the brief did not name it.**

---

## 4. Has anyone required the learned coarse law to be POSITIVE / monotone / stable?

**Yes. Repeatedly, in at least four literatures.** This was the question the brief expected to be
thin, and it is the best-populated of the five.

**(a) The positive realization problem.** Given a transfer function, when does a state-space
realization with non-negative matrices exist, and of what minimal dimension?
Anderson, Deistler, Farina & Benvenuti (*IEEE TCAS* 1996) and Farina (1996) settled existence;
Benvenuti & Farina, "A tutorial on the positive realization problem" (*IEEE TAC* **49**, 2004) is
the standard reference; Benvenuti, "Minimal positive realizations: a survey"
(*Automatica* **143**, 110422, 2022) is current. Existence is characterised by an invariant **polyhedral
cone** (Ohta, Maeda & Kodama, *SIAM J. Control Optim.* **22**, 1984).

The key known fact, which is exactly this thread's thesis:

> **[classical]** The minimal *positive* realization can have dimension **strictly larger** than
> the order of the transfer function, and characterising that minimal dimension is still open.

Since a depth-p AR law on K coarse variables is a Markov law on pK variables, "memory is the price
of positivity" **is** "the minimal positive realization is bigger than the minimal realization,"
in different units. M1's and M2's central framework claim is a fifty-year-old result in control
theory, stated in a currency we had not translated into.

**(b) Positivity-preserving model order reduction.** Li, Lam, Wang and co-workers on
positivity-preserving H∞ model reduction for positive systems (bilinear-matrix-inequality
formulations); Grussler & Damm on balanced truncation preserving cone-invariance; Reis & Virnik on
positivity-preserving balanced truncation for descriptor systems; and most directly
**Cortese, Grigoletto, Ticozzi & Ferrante, "Robust, positive and exact model reduction via
monotone matrices" (arXiv:2406.11696, 2025)** — they characterise when a projection onto the
reachable subspace admits a *non-negative* full-rank factorisation, and when minimal positive
reduction fails they **enlarge the state space** as the explicit price. Their Example 2 pays one
extra dimension. That is our trade, worked as an example, last year.

**(c) PCCA+.** Deuflhard & Weber, "Robust Perron cluster analysis in conformation dynamics" (*LAA* **398**, 161–184, 2005); Röblitz & Weber
(*ADAC* 2013); complex-eigenvalue extension (Fackeldey/Weber lineage, *JCAM* 2024). PCCA+ takes the
dominant invariant subspace and finds the **linear transformation that makes the memberships
non-negative and partition-of-unity**, so the coarse propagator is a genuine stochastic matrix.
This is, structurally, "the slow subspace, rotated until the coarse law is positive" — the exact
shape of the Task 2 answer the brief hoped would be novel. Related: Kube & Weber (*JCP* 2007) fit
the coarse **generator** subject to it being a proper rate matrix.

**(d) Structure-preserving learning.** Stability-constrained operator inference (Sawant, Kramer,
Peherstorfer, *CMAME* 2023; Goyal & Benner on guaranteed-stable quadratic models); stability-
constrained DMD; MSM estimators that impose row-stochasticity and detailed balance by construction
(Prinz et al. 2011).

**What is genuinely not there**, as far as I can find: nobody in (a)–(d) imposes a **locality /
bandwidth** constraint on the positive reduced law *while also* choosing the projection. Positive
realization theory allows any non-negative matrices; PCCA+ allows any stochastic coarse matrix;
positivity-preserving MOR allows dense reduced operators. **Locality is the assumption none of
them drop.** See §7.

**And a warning about the difficulty if this thread proceeds.** Asking "which K eigenvalues can we
retain such that some non-negative coarse matrix has that spectrum" is the **nonnegative inverse
eigenvalue problem**. It is open for n ≥ 5 (posed by Suleĭmanova, 1949), and its real version is
**NP-hard** (arXiv:1608.00931). A positivity-constrained coarse-variable search contains NIEP as a
subproblem. That is not a reason not to run experiments, but it is a reason to expect that any
clean characterisation we find will be special to the FTCS structure rather than general.

---

## 5. Information-theoretic framings — is p\* statistical complexity?

**Essentially yes, in this setting.** [classical]

- **Causal states / ε-machines** (Crutchfield & Young, *PRL* **63** (1989); Shalizi & Crutchfield,
  *J. Stat. Phys.* **104** (2001)): the causal states are the coarsest partition of pasts that is
  sufficient for predicting futures, and are proved **optimal, minimal, and unique**. Statistical
  complexity C_μ = H[causal states] is *literally* the minimal memory needed for optimal
  prediction.
- The ε-machine achieves p = 0 by **enlarging the state**, exactly the trade in §4(a). Our p\*
  fixes the state dimension at K and asks for lags. Depth-p on K variables ≡ Markov on pK.
  So p\* and C_μ measure the same thing in different units, and neither is new.
- For a **linear** system, the causal-state construction collapses to the minimal realisation:
  two pasts are equivalent iff they produce the same future, which for `y_n = P A^n u_0` means
  equal projections onto the observable subspace. So computational mechanics and Kalman
  realisation theory coincide here — which is why our §1 rank computation *is* the ε-machine.
- **Past–future information bottleneck** (Creutzig, Globerson & Tishby 2009; Still 2014) and the
  **Gaussian IB** (Chechik, Globerson, Tishby & Weiss, *JMLR* 2005): for linear-Gaussian processes
  the IB optimum is given by canonical correlation analysis between past and future — i.e.
  **TCCA, i.e. VAMP**. So the information-theoretic and the spectral routes are known to be the
  same route, which closes the circle with §1.
- **Predictive information** (Bialek, Nemenman & Tishby 2001): the subextensive part of the entropy,
  the same quantity from the entropy side.

The one currency that is *not* obviously covered: our p\* is measured in **lags at fixed locality
and fixed positivity**, and neither C_μ nor IB carries a positivity or bandwidth constraint. That
is the same residue as §4.

---

## 6. Summary table

| brief's question | verdict | the citation that settles it |
|---|---|---|
| Is minimise-memory = maximise-VAMP? | **No — strictly weaker, and degenerate.** Measured today: all invariant subspaces tie at p\*=0, including the fastest. VAMP's optimum is one element of that set. | Wu & Noé 2020; degeneracy measured here |
| Is choosing variables to minimise memory new? | **No.** | Guttenberg, Dama, Saunders, Voth, Weare & Dinner, *JCP* **138**, 094111 (2013) |
| Optimal prediction | takes P as given; the gap is real but was closed in 2013 | Chorin–Hald–Kupferman 2000/2002 |
| Lumpability | exact condition = invariance; partitions buy positivity, pay Markovianity | Kemeny–Snell 1960; Buchholz 1994 |
| RG: optimal blocking | **solved with a theorem**: information-optimal blocking does not increase coupling range | Lenggenhager et al., *PRX* **10**, 011037 (2020) |
| **Positivity constraint — "the thin one"** | **not thin: four literatures**, and "positivity costs state dimension" is the central classical fact | Benvenuti–Farina, *IEEE TAC* 2004; Benvenuti, *Automatica* **143** (2022); Cortese et al. 2025; Deuflhard–Weber, *LAA* **398** (2005) |
| Is p\* statistical complexity? | **yes, in different units**; for linear systems ε-machine = minimal realisation | Shalizi–Crutchfield 2001 |
| Same problem in our own field | AMG coarsening: stencil growth, lost M-matrix property, learned prolongation | Falgout–Schroder non-Galerkin; Luz et al. ICML 2020 |

Applying Phase 2's own filter: M2 drops **"the projection is given"** — and then, in every
formulation I can write down, reintroduces it as *the leading invariant subspace*, which is the
classical answer to the classical problem. That is the same failure mode as moment-regression → MLS
and target-aware allocation → DWR.

---

## 7. What, if anything, is left

One corner is uncleared, and only one. All three constraints must be live simultaneously:

> **Minimise memory depth over projections P, subject to the coarse law being (i) non-negative
> AND (ii) compactly supported / banded of half-width s.**

Why the combination might not be covered:

- Positive realization theory (§4a) and positivity-preserving MOR (§4b) allow **dense** reduced
  operators. Adding a bandwidth constraint changes the feasible set from "some non-negative matrix
  with this spectrum" to "some non-negative **banded** matrix", which is a strictly harder and, as
  far as I can find, unstudied realisability question.
- PCCA+ (§4c) fixes the subspace first (dominant invariant) and never considers locality or memory.
- RG (§3) cares about locality and about the coarse law's form, but works with Hamiltonians and
  mutual information, not with non-negativity of a propagator or with autoregressive depth. The
  RSMI theorem bounds *coupling range*, not positivity.
- AMG (§3) cares about locality (stencil growth) and about the M-matrix property, but does **not**
  consider memory — coarse-grid operators there are strictly one-step.

So the intersection is empty in the literature, and the natural first question is sharp:

> **Q.** At fixed coarse dimension K and fixed bandwidth s, does the positivity-constrained
> memory-minimising projection differ from the slow subspace, and if so, in what way?

**[conjectured]** It does, and the reason is that the slow subspace of a diffusion operator is
spanned by *oscillatory* eigenvectors (sines), which are sign-indefinite, so the coarse law they
induce cannot be a convex combination of anything. Non-negativity should push the optimum towards
**localised, positive, partition-like** coarse variables — i.e. towards block averages and away
from Fourier modes — and memory is then what pays for the accuracy the block average loses.
If that is what happens, the finding is "positivity turns VAMP into PCCA+ with a locality
constraint, and the memory depth is the residue", which is a modest but real and measurable
statement.

**Honest assessment of its value**: this is a *much* smaller claim than the brief's. It is a
constrained-optimisation characterisation in one specific linear problem class, it will not
generalise without work, and its nearest neighbours (PCCA+, positive MOR, non-Galerkin AMG) are
close enough that a referee would ask about all three. I would rate the chance that a serious
version of it is already published at maybe one in three — I could not find it, but §4 shows my
search has been wrong about this area once already today.

---

## 8. Recommendation

**Kill Tasks 1 and 3.** Task 1 is classical and I have already run the verification that settles it
(§1, and the numbers are in the table — hand them to M1 rather than repeating them). Task 3 is
Guttenberg et al. 2013, including its conclusion.

**Task 2 is the only one worth touching, and only in the narrowed form of §7** — with the bandwidth
constraint live, because without it the answer is positive realization theory.

Two things I would do instead, both cheap, if you want value out of this thread:

1. **Hand M1 the reframing.** M1's p\*(M) is the **minimal positive realization dimension gap** in
   different units, and that reframing comes with a fifty-year literature, a survey
   (Benvenuti, *Automatica* **143**, 2022), known bounds, and known hard cases. M1 should not measure p\*(M) as a novel
   scaling law without checking it against the positive-realization bounds first. The trivial
   ceiling p\* ≤ ⌈N/K⌉ − 1 and the fact that decimation *attains* it (measured, §1) also matter to
   M1: F14's "exactly M − 1 lags" is the worst case, so any p\*(M) curve should be reported
   relative to that ceiling, not in absolute lags.

2. **Report the degeneracy as the finding.** "Minimising memory does not select slow variables —
   every invariant subspace is equally Markov, including the fastest one" is a clean, measured,
   slightly counterintuitive statement that directly qualifies the framework claim. *Attend to
   whatever lets you forget* is under-determined: forgetting is cheap in many directions, and what
   picks out the useful ones is not Markovianity but **retained predictive content** — which is the
   VAMP score, or, with the positivity constraint, something more like a partition. That is a
   sharper and more honest version of the framework sentence than the one we started with, and it
   cost one afternoon rather than a thread.

**Awaiting your call before running anything further.**

---

### Sources

- [Minimizing memory as an objective for coarse-graining (JCP 2013)](https://pubs.aip.org/aip/jcp/article-abstract/138/9/094111/72810/Minimizing-memory-as-an-objective-for-coarse)
- [Variational Approach for Learning Markov Processes from Time Series Data (VAMP)](https://link.springer.com/article/10.1007/s00332-019-09567-y)
- [Optimal Renormalization Group Transformation from Information Theory (PRX 2020)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.011037)
- [Mutual Information, Neural Networks and the Renormalization Group](https://arxiv.org/abs/1704.06279)
- [Minimizing couplings in renormalization by preserving short-range mutual information](https://arxiv.org/pdf/2107.00990)
- [Minimal positive realizations: A survey (Automatica 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0005109822002758)
- [A Tutorial on the Positive Realization Problem](https://www.researchgate.net/publication/3031788_A_Tutorial_on_the_Positive_Realization_Problem)
- [Robust, positive and exact model reduction via monotone matrices](https://arxiv.org/html/2406.11696)
- [Fuzzy spectral clustering by PCCA+](https://link.springer.com/article/10.1007/s11634-013-0134-6)
- [Slicing and Dicing: Optimal Coarse-Grained Representation to Preserve Molecular Kinetics](https://pubs.acs.org/doi/10.1021/acscentsci.2c01200)
- [Effective dynamics using conditional expectations (Legoll & Lelièvre)](https://arxiv.org/abs/0906.4865)
- [Computational Mechanics: Pattern and Prediction, Structure and Simplicity](https://arxiv.org/abs/cond-mat/9907176)
- [Non-Galerkin Coarse Grids for Algebraic Multigrid](https://www.osti.gov/biblio/1237540)
- [The real nonnegative inverse eigenvalue problem is NP-hard](https://arxiv.org/pdf/1608.00931)
- [Nonequilibrium Statistical Mechanics and Optimal Prediction of Partially-Observed Complex Systems](https://arxiv.org/pdf/2203.16048)
