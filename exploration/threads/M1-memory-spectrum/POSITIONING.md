# M1 — Literature positioning, and a recommendation against the thread as scoped

**Date**: 2026-09-11. **Deliverable**: Task 0 only. No Task 1–4 experiments were run.
Supporting script: `positioning_check.py` (analytic checks + LP verification, ~2 min).

Every claim is marked **[classical]**, **[measured here]**, or **[conjectured]**.

---

## Verdict, up front

**Do not run M1 as scoped.** Three of the four tasks are either already answered in the
literature or answerable analytically, and I answered two of them while writing this memo.
The central object p\* already has two names in two literatures. And the method the brief
proposes — LP feasibility over non-negative multistep coefficients — is the standard method
of the SSP-multistep literature, i.e. exactly the Phase-1 failure mode: drop an assumption,
then reintroduce it by using the classical method for the classical problem.

**But one thing in this thread is worth half a day**, and it is not any of Tasks 1–4. It is
that the brief's own central prediction is, I believe, *exactly backwards*:

> The brief predicts: *p\* is small when there is scale separation and large when there is not
> — p\* measures the failure of scale separation.*
>
> The obstruction says: the constraint is a **sign** condition on the unresolved spectrum, not
> a **gap** condition. A perfectly separated unresolved mode (amplification g → 0⁺, dies in one
> step) makes a non-negative coarse law **impossible at every memory depth**. A barely separated
> one that *oscillates* (g < 0) is free. **p\* measures the failure of oscillation, not the
> failure of scale separation.**

That inverts the prediction, it is cheap to test, and it matters to the programme, because
"hierarchies are cheap when scales separate" is the intuition Phase 2 is built on.

---

## 0. What I did

Read `T5-composition/{THEORY.md, task4_mz.py, composition.py}`, `FINDINGS.md` F23/F14,
`PHASE2.md`, `REEVALUATION.md`, `core/`. Then, because Q2 asks for an analytic prediction, I
derived the positivity obstruction by hand, verified it against T5's numbers by an independent
route (closed form + LP), and ran four literature searches.

`task5_trilemma.py`, named in the brief, **does not exist** in `T5-composition/` — only
`task1..task4` are there. F23's numbers therefore had no script behind them that I could read,
so I re-derived and re-measured them from scratch. **They reproduce** (§6).

---

## 1. Mori–Zwanzig / optimal prediction: memory length vs coarsening ratio

**[classical]** The framework itself (Zwanzig 1960/61, Mori 1965) and the numerical-analysis
line (Chorin–Hald–Kupferman, *Optimal prediction with memory*, Physica D 2002; the *t*-model;
Stinis's finite-memory hierarchies; Givon–Kupferman–Stuart, Nonlinearity 2004) all say the
memory kernel decays on the timescale of the unresolved variables, and that with scale
separation ε the closure is Markovian to O(ε). This is the standard picture and it is the one
the brief's prediction is drawn from.

**The specific question — memory length as a function of the coarsening ratio — has been asked
and measured.** Parish & Duraisamy, *A priori estimation of memory effects in coarse-grained
nonlinear systems using the Mori–Zwanzig formalism* (arXiv:1611.06277) measure memory length
against filter width across **Burgers, homogeneous turbulence, and channel flow**, and report a
power law τ ∝ Δ^1.5 in the coarsening ratio. That is Task 2's headline measurement, in the
nonlinear setting M1 lists as out of reach, published. **[classical]**

**Is there an existing notion of "minimal memory depth"?** Yes, two, and T5's Theorem 3 is the
first of them:

- **[classical]** For a *linear* system, the minimal order of an exact recurrence for a linear
  observable is the **McMillan degree** of the coarse transfer function (equivalently the
  Hankel rank; minimal realization theory, Kalman). T5's "exact, finite MZ memory of exactly
  M−1 lags, via Cayley–Hamilton on the M-dimensional alias subspace" is the statement that the
  coarse transfer function is rational of degree M. The finiteness T5 calls "unusual" is
  unusual only relative to the *continuous-time, infinite-dimensional* MZ setting; in
  discrete-time linear systems theory a finite exact recurrence is the generic case, not the
  exception. This does not make T5 wrong — the derivation is correct and the alias-subspace
  route is clean — but the headline "MZ kernels are normally infinite, this one terminates"
  should be retired.
- **[classical]** With the non-negativity constraint, the minimal order is the **minimal
  positive realization order**. See Benvenuti & Farina's tutorial (IEEE TAC 2004) and the 2022
  Automatica survey *Minimal positive realizations: a survey*. The known headline is precisely
  the phenomenon T5 found: **the minimal positive realization order can be strictly, and
  sometimes vastly, larger than the McMillan degree**, and characterising minimality for
  positive systems is an acknowledged open problem. "p\*(3) = 5 > M−1 = 2" is an instance.

---

## 2. Is p\*(M) the spectral gap in disguise? — **No, and the reason is the thread's one real finding**

This is the question I was asked to answer loudly. The answer is no, but not because p\* is
subtler than the gap — because it is governed by a *different feature of the same spectrum*.

### 2.1 The obstruction (analytic)

Write a compact coarse law of depth P and half-width s as

    U^{n+1}(x) = Σ_{j≥1} Σ_k b_{j,k} U^{n+1−j}(x + k Δx_c),      b ≥ 0.

Exactness on a fine mode with amplification g and coarse wavenumber θ_c reads
`Σ_{j,k} b_{j,k} e^{ikθ_c} g^{−j} = 1`. Evaluate at the **constant coarse mode** q = 0, where
θ_c = 0. With β_j := Σ_k b_{j,k} ≥ 0 this is

    φ(y) := Σ_{j≥1} β_j y^j = 1        at    y = 1/g.

The alias l = 0 at q = 0 has g = 1 and gives Σ_j β_j = 1 — unit mass, automatic. So φ has
non-negative coefficients, φ(0) = 0, φ(1) = 1, and is strictly increasing on [0,∞). **y = 1 is
its only positive solution.** Hence:

> **Obstruction.** A non-negative coarse law of **any** depth and **any** width exists only if
> every unresolved alias at the constant coarse mode satisfies **g ≤ 0**. No unresolved g may
> lie in (0,1). More memory cannot help.

**[classical]** in substance: β ≥ 0 with Σβ = 1 makes the companion matrix of the recurrence
non-negative and stochastic, so this is Perron–Frobenius — the Perron root 1 is the only
eigenvalue on the positive real axis. I am claiming novelty only for the *reading*, not the
mathematics.

### 2.2 What it predicts, and it is not a gap

For FTCS pure diffusion, g_l(0) = 1 − 4r sin²(πl/M), so the condition is
**r ≥ 1/(4 sin²(π/M))**, binding at l = 1. **[measured here]**, LP-verified:

| M | r needed (necessary) | FTCS allows r ≤ ½ | verdict |
|---|---|---|---|
| 2 | 0.2500 | yes | feasible |
| 3 | 0.3333 | yes | feasible |
| 4 | 0.5000 | only at the endpoint | **infeasible for r < ½** |
| 5 | 0.7236 | no | **infeasible, any r** |
| 6 | 1.0000 | no | **infeasible, any r** |
| ≥7 | > 1 | no | **infeasible, any r** |

So **p\*(M) = ∞ for every M ≥ 4** in this system. **That is Task 1, answered.** The
"p\*(M) spectrum for M = 2…12" the brief wants to map has two finite entries.

LP scan for the exact non-negative law, depth ≤ 8, half-width ≤ 3, pure diffusion
(`positioning_check.py` §E) **[measured here]**:

| M | r=0.20 | 0.25 | 0.28 | 0.293 | 0.30 | 0.34 | 0.36 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | — | — | p=3,s=2 | p=1,s=1 | p=1,s=1 | p=1,s=1 | p=1,s=1 | p=1,s=1 | p=1,s=1 | p=1,s=1 |
| 3 | — | — | — | — | — | — | — | p=5,s=2 | p=5,s=2 | p=2,s=1 |
| 4 | — | — | — | — | — | — | — | — | — | — |

Two things to read off. **Memory does buy something**: at M=2, r=0.28 lies below T5's Markov+1
threshold 1−1/√2 = 0.2929 and a law exists at depth 3 instead of 1. And **p\* falls as r rises**:
M=3 needs p=5 at r=0.45 but only p=2 at r=0.50.

### 2.3 The inversion

The obstruction cares about the **sign** of the unresolved eigenvalue, not its magnitude:

- g → 0⁺ — the *best possible* scale separation, an unresolved mode annihilated in one step —
  is **fatal** at every depth.
- g = −0.8 — the *worst* separation in the admissible range, a slowly-decaying oscillatory mode
  — is **free** (this is exactly why M=2 works).

Direct test against the two standard gap proxies, pure diffusion at r = 0.45, half-width 3 so
locality does not bind. ρ_M is the geometric decay rate of the positivity defect per lag
**[measured here]**:

| M | ρ_M measured | max_{l≠0} \|g_l(0)\| | max_q 2nd-largest \|g\| | max **positive** g_l(0) |
|---|---|---|---|---|
| 3 | 0.030 | 0.350 | 0.756 | 0.000 |
| 4 | 0.391 | 0.800 | 0.800 | 0.100 |
| 5 | 0.569 | 0.628 | 0.828 | 0.378 |
| 6 | 0.640 | 0.800 | 0.879 | 0.550 |
| 8 | 0.722 | 0.800 | 0.922 | 0.736 |

Both gap proxies are flat in M (0.63–0.80, 0.76–0.92). ρ_M rises monotonically 0.03 → 0.72 and
tracks the **largest positive unresolved alias** — the obstruction quantity. **p\* is not a
reparameterisation of the spectral gap.** **[measured here]**

### 2.4 Four falsifiable predictions that invert the brief

**[conjectured]**, all cheap to test, all following from §2.1:

1. **Scale separation makes positivity *harder*.** A two-timescale linear system whose fast
   mode has real positive amplification g_fast ∈ (0,1) has p\* = ∞ at every M. The brief
   predicts p\* small here.
2. **Advection *helps*.** Advection makes the unresolved g complex, so phases can cancel and
   the real-positive obstruction weakens. The brief has no prediction here; the naive one is
   that advection hurts.
3. **The dispersive case is the *easy* one, not the hard one.** For u_t = u_xxx the symbol lies
   on the unit circle and is real-positive only at isolated points, so the obstruction is
   generically empty. The brief predicts "positivity should fail hard".
4. **Increasing r (more damping per step, i.e. *coarser in time already*) lowers p\*.** Partly
   measured already in §2.2 (M=3: p=5 at r=0.45, p=2 at r=0.50).

Prediction 3 is the sharpest test: it is a sign flip on a case the brief lists as an expected
negative control.

---

## 3. Has anyone asked for a positivity/monotonicity constraint on an MZ coarse law?

This is where the brief thinks the thread is thin, and it is the closest to right — but the
gap is narrower than it looks, because the constraint has been studied in every neighbouring
formulation.

- **[classical]** *Positivity of multistep schemes.* Bolley–Crouzeix (1978) — already hit in
  Phase 1 — is exactly "which multistep schemes are unconditionally positive". The modern line
  is SSP linear multistep methods (Shu 1988; Lenferink 1989; Hundsdorfer–Ruuth–Spiteri;
  Ketcheson–Gottlieb–Macdonald, arXiv:1504.03930). **Their method is LP feasibility over
  non-negative multistep coefficients, with a duality argument for infeasibility** — which is
  the method I used above and the method Task 1 proposes. The prose in that literature —
  *"any method with non-negative coefficients preserves discrete monotonicity"*, and existence
  proofs run as LP feasibility plus dual infeasibility — is the same machinery on the same
  object. **This is the strongest prior-art hit against Task 1 as a method.**
- **[classical]** *T5's "pure-delay" surprise is the known shape of optimal monotone multistep
  methods.* Optimal SSP LMMs are sparse and put their weight at late lags; "buy back
  monotonicity by moving the law backwards in time" is the known trade in that literature
  (more steps buy a larger monotonicity threshold). F23 reports B₁ = B₂ = 0 as a structural
  surprise; I would expect it to be the SSP structure, and I would want that checked before it
  is written up as novel. I flag this as **[conjectured]** — I have not matched a specific
  optimal SSP method to T5's M=3 law.
- **[classical]** *Positivity destroyed by decimation is the RG pathology.* Griffiths–Pearce
  (1979) and van Enter–Fernández–Sokal (*Regularity properties and pathologies of
  position-space renormalization-group transformations*, J. Stat. Phys. 72, 879, 1993): a
  decimated Gibbs measure is provably not Gibbsian for any reasonable interaction. The
  conceptual content — **decimation breaks the positivity/locality structure, and not as a
  small perturbation** — is famous, and F14/F23's "coarse-graining destroys positivity" is its
  scheme-theoretic shadow.
- **[measured here, weak]** A search for the four-way combination (positivity-preserving +
  coarse-grained + Mori–Zwanzig memory + maximum principle) returns the components separately
  and nothing joining them. That is the one honest point in the thread's favour, and it is a
  negative search result, not evidence of a gap.

---

## 4. Markov state models / lumpability — **p\* already has a name**

**[classical]** and this is the single most important citation in the memo.

- Strong lumpability (Kemeny–Snell): a partition gives an exactly Markov lumped chain iff the
  row sums into each block are constant across each source block. **p\* = 0 ⟺ strong
  lumpability.**
- The generalisation is standard: *a lumping is **strongly k-lumpable** iff the lumped process
  is a k-th order Markov chain for every starting distribution of the original chain.*
  Higher-order lumpability is analysed by Gurvits & Ledoux (Linear Algebra Appl. 2005) and by
  Geiger & Temmel, *Lumpings of Markov chains, entropy rate preservation, and higher-order
  lumpability* (arXiv:1212.4375, J. Appl. Prob.). The background fact is Burke & Rosenblatt
  (1958), *A Markovian function of a Markov chain*.

**p\*(M, tolerance 0) is the minimal k for which the coarse-graining is strongly k-lumpable.**
The definition the brief proposes as the programme's new measurable quantity is a named object
in the Markov-chain literature, with a small literature attached.

Two honest caveats that keep this from being a total match:

1. T5 coarse-grains by **decimation** (keep every M-th point), not by **lumping** (sum or
   average over blocks). Lumpability theory is about the latter. Decimation of a field is not a
   probabilistic lumping, so the correspondence is at the level of the operator, not literal.
   **T5 never examined this choice.** I expected it to be load-bearing and said so here.
   **It is not — see RESULTS.md R4, which retracts this paragraph.** Block averaging
   annihilates exactly the unresolved aliases at q = 0, i.e. exactly the constraints that
   carry the §2.1 obstruction, and it changes one cell in a 42-cell feasibility table.
   Exempting 97% of all coarse modes still leaves M = 4 infeasible. So the obstruction
   lives at essentially every mode independently, and §2.1 is where it is provable rather
   than where it lives.
2. FTCS with r ≤ ½ and small c *is* a doubly stochastic circulant, so the fine dynamics really
   is a random walk on Z_N and the probabilistic language is not a metaphor here.

---

## 5. What survives — Task 3, answered analytically, for free

**[measured here / analytic]** Task 3 ("is p\* bounded under iterated coarsening — coarsen by 2
twice vs by 4 at once — the single most important measurement in the thread") does not need an
experiment. Decimating by 2 and then by 2 again yields *the same observable* as decimating by 4:
the retained index set is identical. The exact MZ law is a property of the observable, so:

> **Memory depends only on the total coarsening ratio. It does not accumulate and does not
> compound.** p_exact(M₁ then M₂) = p_exact(M₁M₂) = M₁M₂ − 1.

So the framework question splits, and the split is the informative part:

- **the exactness price is cheap** — linear in the total ratio, path-independent, no compounding;
- **the positivity price is infinite past M = 4**, and also path-independent, so you cannot
  dodge it by taking small steps.

That is a real answer to "does compression across scales have a bounded price", for this system:
*bounded if you only want closure, unbounded the moment you also want the coarse law to be a
probability kernel.* It took a paragraph, not a thread.

---

## 6. Corrections to T5

I reproduced F23 independently (closed form for M=2, LP for the rest) before criticising it.

**Confirmed [measured here].** T5's headline numbers all reproduce:
- M=2 exact law: B₁ = (2−4r)·I, B₂ = (r², −(1−4r+2r²), r²), non-negative iff r ≥ 1−1/√2 =
  0.292893. Matches T5's Cayley–Hamilton output to 1e−16 at every r tested.
- Exact non-negative compact laws at **M=2 (s=1, p=1)** and **M=3 (s=2, p=5)** — both found
  independently by LP, defect 0 to 2e−16.
- Defect decay at fixed minimal locality s=1: **0.0966 per lag at M=3** (I get 0.0966),
  ~0.44 at M=4, ~0.74 at M=6. All three reproduce.

**Correction 1 — "memory buys back positivity" is the wrong verb for M ≥ 4.** The defect decays
geometrically, as T5 says, but for M ≥ 4 it decays to an infimum that is **never attained**: by
§2.1 no finite depth is feasible at any width. T5 wrote "at M=4 this stops (geometric
convergence, no snap to zero — a failure to find, not a proof)". It is now a proof of the
negative. The trade curve is real; it terminates at M=3.

**Correction 2 — T5's Observation 4 contradicts T5's own output.** THEORY.md §6 Observation 4,
FINDINGS row 18 and PHASE2's framing all state that under diffusive M² time coarsening "the
coarse law becomes Markov to round-off, stays positive, and keeps Σ‖B_p‖₁ = 1", concluding
"there is no price at all". I ran `task4_mz.py` unmodified. Its own table prints:

| M | K=M² | Σ‖B_p‖₁ | positive? | resid p=1 |
|---|---|---|---|---|
| 2 | 4 | **1.819200** | **no** | 5.6e−2 |
| 3 | 9 | 1.012120 | **no** | 2.8e−3 |
| 4 | 16 | 1.056295 | **no** | 6.5e−3 |
| 5 | 25 | 1.000336 | **no** | 8.2e−5 |
| 6 | 36 | 1.000649 | **no** | 8.0e−5 |

Every row says `no`; Σ‖B_p‖₁ is 1.82 at M=2, not 1; and the p=1 residual at M=2 is 5.6e−2,
which is not round-off. My independent computation gives min B_p = −1.1e−1 at M=2. The
obstruction explains why: under K = M² the M=2 unresolved alias becomes (−0.8)⁴ = **+0.4096**,
real and positive — forbidden. **Time-coarsening turns a harmless oscillatory alias into a
fatal monotone one.** The surviving true claim is the narrower one: memory is crushed for
M ≥ 3 (‖B₂‖/‖B₁‖ falls to 3e−4). "The tension dissolves" and "there is no price at all" do not
survive, and PHASE2.md inherits this.

**Numerical caveat on my own LP [measured here].** At large P the coefficients g^{P−j} span many
orders of magnitude and HiGHS reports spurious infeasibility (e.g. M=2, s=1 is feasible at P=8
and "infeasible" at P=9). Those cells are conditioning, not mathematics. The feasible cells and
the §2.2 thresholds are robust; I would not trust a defect below ~1e−10.

---

## 7. Recommendation

**Kill Tasks 1, 3 and 4.** Task 1 is answered (§2.2: finite only at M ∈ {2,3}, and only above
an r-threshold). Task 3 is answered (§5: total ratio only, no compounding). Task 4 (nonlinear)
is out of reach by the thread's own method, and the nonlinear version of its question was
published by Parish & Duraisamy.

**Do not run Task 2 as written.** Its method — LP over non-negative multistep coefficients — is
the SSP-multistep method, and its object — minimal order of a non-negative realization / minimal
k for strong k-lumpability — is named in two literatures. Running it reproduces the Phase-1
pattern exactly: drop *locality in time*, then reintroduce it by using the classical method on
the classical object.

**Run instead, for half a day, one falsification experiment.** Test §2.4's four predictions at
matched coarsening: two-timescale (predicted p\* = ∞ *because* it separates), advection-dominated
(predicted easier), dispersive u_t = u_xxx (predicted **easy**, against the brief's "fails
hard"), and the r-sweep (predicted p\* decreasing in r). The deliverable is one sentence, and
it is a sentence about the programme rather than about numerical schemes:

> ~~Coarse-graining preserves a maximum principle when the modes you throw away oscillate, and
> destroys it when they merely decay.~~

**⚠ That sentence is half wrong and must not be quoted. It was written before the
dispersive test; RESULTS.md R6 refutes its first clause.** The biconditional is false:
oscillation is *necessary*, not sufficient. The correct, tested statement is
one-directional:

> **A discarded mode that decays monotonically (a real eigenvalue in (0,1)) destroys the
> coarse maximum principle, at every memory depth and every stencil width.** Proved at the
> constant coarse mode (§2.1), verified 16/16 where it applies, never falsified in 39 cases.
> **Oscillation does not buy positivity back** — dispersive propagators are oscillatory and
> still fail, and every one of 13 system classes fails at M ≥ 4.
>
> Scale separation is not what makes hierarchies cheap. But nothing measured here makes
> them cheap.

If that survives, it is worth a paragraph in the book and a figure. If it fails, M1 closes and
the programme has lost a day instead of a week. ~~Either way I would also spend an hour on the **decimation-vs-block-averaging** question
(§4, caveat 1), because every number in F14/F23 depends on a restriction operator nobody
chose deliberately.~~ **Ran it. It fails — see RESULTS.md R4.** The restriction operator
changes one cell in 42 and the obstruction survives exempting 97% of the spectral band.

**What I would not do is look for a third framing.** PHASE2.md says: *"If both, the right move
is to say so and stop, not to find a third framing."* p\* is not the spectral gap — but it is
higher-order lumpability, and it is minimal positive realization order. That is the same
outcome by a different route, and the honest response is the one PHASE2 already wrote down.

---

## Sources

- [Minimal positive realizations: a survey (Automatica 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0005109822002758)
- [Minimal state-space realization in linear system theory: an overview](https://www.sciencedirect.com/science/article/pii/S0377042700003411)
- [Parish & Duraisamy, A priori estimation of memory effects in coarse-grained nonlinear systems using the Mori–Zwanzig formalism](https://arxiv.org/html/1611.06277)
- [Incorporation of memory effects in coarse-grained modeling via the Mori–Zwanzig formalism (J. Chem. Phys. 143, 243128)](https://pubs.aip.org/aip/jcp/article/143/24/243128/963573/Incorporation-of-memory-effects-in-coarse-grained)
- [Ketcheson, Gottlieb & Macdonald, Existence and optimality of strong stability preserving linear multistep methods: a duality-based approach](https://arxiv.org/abs/1504.03930)
- [Geiger & Temmel, Lumpings of Markov chains, entropy rate preservation, and higher-order lumpability](https://arxiv.org/pdf/1212.4375)
- [van Enter, Fernández & Sokal, Regularity properties and pathologies of position-space renormalization-group transformations](https://arxiv.org/pdf/hep-lat/9210032)
- [Maximum-principle-satisfying and positivity-preserving high-order schemes for conservation laws (survey)](https://royalsocietypublishing.org/doi/10.1098/rspa.2011.0153)
