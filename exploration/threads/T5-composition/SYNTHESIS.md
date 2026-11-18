# T5 — Synthesis: positivity, locality, accuracy, and what composition costs

One object, measured from three sides. The spine below is the coordinating thread's; the
classical anchors are named as such, and what is mine is marked.

| | classical | measured here |
|---|---|---|
| positivity is the condition for safe composition | Godunov; maximum-principle / monotone schemes | $\ell_1$ submultiplicativity, the three growth regimes, defect amplification by $\|w\|_1$ |
| positivity + locality caps expressiveness | **Pawula** (1967) | the cap is set by $r$, not by "2"; measured order 10 at $r=0.45$, order 2 at $r\le0.15$ |
| the escape is memory, i.e. giving up locality | **Mori–Zwanzig** | memory is exactly $M-1$ lags and terminates; the trade curve $R_{\rm pos}\sim\rho(M)^p$ |
| width buys step length quadratically | **Runge–Kutta–Chebyshev** | the frontier stencil is the extremal two-point measure — consistent, positive, stable, and useless |

The contribution is the unification in stencil language plus **a continuous price where the
classical results give yes/no answers** — and, in two places, a correction to the expected
form of the answer.

---

## 1. Agents compose into agents — proved

Composition is the pushforward of the product measure under displacement addition;
convolution when the sub-agents are identical. Closure is one line, no hypotheses: *if $w$
and every sub-agent reproduce a translation-invariant class $\mathcal S$ at their own
centres, so does the composite.* Two consequences:

- **Orders take the min, not the sum**, and can never drop below it — so the "composition
  degrades the order" counterexample does not exist. I searched (600 random heterogeneous
  compositions): none, to $10^{-14}$.
- **The class cannot be space-time polynomials.** A single-time-level stencil can never
  reproduce $\Pi_r$ for $r\ge1$ ($M_{0,1}=\tau\sum w_i=\tau\ne0$). The right class is the
  PDE-adapted heat/drift polynomials $\Theta_n$. This is also, incidentally, the corrected
  third row the coordinating thread flagged: $\tfrac12\Theta_2=\tfrac12(\xi-c\tau)^2+\alpha\tau$,
  and the $\Theta_n$ formulation generalises it to every order.

**The technical heart.** The exact propagator is a Lévy kernel whose cumulants are all
*linear in $\tau$*, and cumulants add under convolution. So **cumulant defects are exactly
additive**, $\varepsilon_n(w^{*L})=L\varepsilon_n(w)$, with no remainder — which is why
modified-equation coefficients are step-count independent: $\alpha_{\rm eff}=\alpha-c^2\Delta t/2
=0.09977043$ at every $L$ from 1 to 512, all digits identical. For heterogeneous sub-agents
the defects are averaged **with signed weights**, $D_n({\rm comp})=D_n(w)+\sum_iw_id_n^{(i)}$,
so $\|w\|_1>1$ amplifies their spread and the bound is attained when signs align.

---

## 2. Positivity and safe composition — true, but not in the proposed form

Submultiplicativity holds (0/4000 violations); positivity is preserved; $w\ge0$ gives
$\|w^{*L}\|_1=1$ exactly at every $L$. But the *iterated* bound $\|w\|_1^L$ is generically
nowhere near attained, so "$\|w\|_1=1$ is exactly the condition" is **sufficient, not
necessary**:

| | |
|---|---|
| $w\ge0$ | $\|w^{*L}\|_1=1$ exactly, every $L$, no transient |
| $\|w\|_1>1$, $\max_\theta\lvert g\rvert\le1$ | $\|w^{*L}\|_1\to1$; the bound is wildly pessimistic |
| $\max_\theta\lvert g\rvert>1$ | geometric growth at rate $\max\lvert g\rvert$, **not** $\|w\|_1$ |

The middle row is the interesting one: **the flow restores positivity**. For a stencil with
$\|w\|_1=1.0205$, $\min w=-5.6\times10^{-3}$, the negative lobes are *expelled* — innermost
negative weight moving $6.3\sigma_L\to25\sigma_L$, negative mass $10^{-2}\to10^{-90}$, against
a Theorem-2 bound of 180. So positivity is the **scale-free** condition (nothing to wait for),
von Neumann the asymptotic one.

What genuinely fails under composition is the **sign of $\kappa_2$**: consistency pins the
defects but not their sign, and $\kappa_2<0$ gives a perfectly good one-step scheme whose
composite is not a kernel at all. Lax equivalence in cumulant language: *consistency fixes
$\varepsilon_n$; stability fixes $\mathrm{sign}(\kappa_2)$.*

---

## 3. The RG picture, and which Edgeworth term leads

For $w\ge0$ composition is literally a random walk; the fixed point is the Gaussian heat
kernel. **But not the one you would first write down**: the composite converges to a Gaussian
with the *renormalised* diffusivity $\alpha_{\rm eff}=\alpha-c^2\Delta t/2$, not $\alpha$ —
because the composite faithfully accumulates FTCS's own defect $L\varepsilon_2$ (§1). The
variance is $2\alpha L\Delta t-Lc^2\Delta t^2$, not $2\alpha L\Delta t$.

**The leading correction is skewness, not kurtosis — except at $c=0$, where it is kurtosis.**
Advection is exactly what makes the step asymmetric ($p_-=0.4727$ vs $p_+=0.4273$,
$\kappa_3=+7.708\times10^{-2}$). Both regimes verified against their analytic constants:

| | leading term | predicted plateau | measured | agreement |
|---|---|---|---|---|
| $c=1$ (this thread) | skewness, $L^{-1/2}$ | $0.008313$ | $0.008342$ | **0.34 %** |
| $c=0$ (control) | excess kurtosis, $L^{-1}$ | $0.094195$ | $0.0942$ | **0.01 %** |

At $c=1$ the kurtosis term is the *next* one: after subtracting the skewness term the residual
plateaus at $L^{-1}$ with constant $0.0944$ against the $\kappa_4$ prediction $0.09376$ (0.7 %).
The moment hierarchy $dM_q/d\tau=\alpha q(q-1)M_{q-2}-cqM_{q-1}$ (drift sign set by our
displacement convention) verifies to $8.6\times10^{-11}$ through $q=8$, but it describes the
**exact** propagator, whose cumulants above the second vanish by construction — the convergence
rate is set by the *single-step stencil's* cumulants, so the hierarchy is a cross-check, not a
route to the rate.

**Control parameter.** $\rho(L)=L/\sigma_L=\sqrt{L/s_2}$, formal over effective support
(4096 vs 61 cells at $L=4096$). The tails are **not** sub-Gaussian: they track the Gaussian to
a few percent all the way out and collapse only as $k\to\rho(L)$. So $\rho$ is simultaneously
the compression ratio and the fixed point's range of validity.

---

## 4. Wide stencils are compressed composites — an identity, with a correction

The boundary is sharp in both directions: a positive stencil of half-width $m$ **about its
target** can take a step $k\Delta t$ iff the exact propagator's *raw* second moment fits inside
it, $2\alpha k\Delta t+(ck\Delta t)^2\le m^2\Delta x^2$ — variance **plus mean squared**. Hence
$k_{\max}(m)=(-r+\sqrt{r^2+r_a^2m^2})/r_a^2$, which LP bisection reproduces **exactly** at
$m=1\dots50$. This corrects the companion thread's $m^2/(2\alpha\Delta t)$, which is only the
diffusion-limited branch: past $m\sim r/r_a=9.9$ cells the bound is advection-limited
($k\to m/r_a$) and $m^2/(2r)$ overestimates by $2.2\times$ at $m=32$, $3.1\times$ at $m=50$.

With the drift kept the equivalence is exact — feeding the composite's half-width
$\sqrt{Ls_2+L^2r_a^2}$ into $k_{\max}$ returns $L$ to within $r_a^2/(2r+2r_a^2L)$.

Two refinements from T2-width-step, both adopted. The advective saturation is an **artefact of
centring the stencil on its target** (what the brief specified, not a necessity): putting the
support on $\{j_0\pm m\}$ with $j_0=\mathrm{round}(-r_ak)$ gives $k\le(m^2-\mu'^2)/2r\to m^2/2r$,
recovering the diffusive branch at every $m$ for free. And there are *two* thresholds — the
quadratic law goes materially wrong ($\approx25\%$) at $m^\*=1/\mathrm{Pe}_{\rm cell}=9.9$, while the
branch formulas cross at $2m^\*=19.8$.

**The classical anchor, and a caveat that matters.** The $m^2/(2r)$ law *is* the
**Runge–Kutta–Chebyshev** stability scaling: first-order RKC with $m$ stages has
$P_m(z)=T_m(1+z/m^2)$, stable for $|z|\le2m^2$, and $z\in[-4rk,0]$ gives $k\le m^2/(2r)$ exactly.
So this law and the $L^{1/2}$ grid speedup are classical, rediscovered in stencil language; the
composition picture explains why, but the scaling is not new.

What *is* worth knowing is what sits at the frontier. At $k=m^2/(2r)$ the argument becomes the
averaging operator, $T_m(\cos\theta)=\cos(m\theta)$, and the RKC stencil is **exactly**
$\tfrac12(\delta_{-m}+\delta_{+m})$ — the extremal measure of my own sufficiency proof, and the LP
vertex an earlier draft dismissed as "pathological". It is consistent (all three moments exact),
positive, and stable ($\|w\|_1=1$) — with band error $\approx1.0$. A bimodal kernel with the right
mass, the right variance, and the wrong shape.

> **Consistency + positivity + stability do not imply accuracy.** Accuracy is an independent
> fourth axis, and the feasibility frontier is a *stability* boundary. Any speedup quoted at the
> frontier carries that caveat.

RKC away from the frontier is not positive ($\min w=-0.111$ at $k/k_{\max}=0.9$, $m=8$; T2
reproduces to three digits): the classical scheme occupies one point of the positive cone, its
extreme vertex, and leaves it as soon as you back off for accuracy. Maximum entropy is the
opposite choice — same feasible set, interior point, accuracy instead of extremality. (At $J=3$
that stencil *is* the moment-matched discrete Gaussian, verified against T2's independently
root-found kernel to $10^{-16}$: one construction, two derivations, not a choice.)

**Why the frontier is empty, and what it says about memory.** $\tfrac12(\delta_{-m}+\delta_{+m})$
couples $j$ only to $j\pm m$, so the lattice splits into $m$ residue classes that never exchange
information; on one class the stencil is $(\tfrac12,0,\tfrac12)$, i.e. FTCS for pure diffusion at
$r_c=\alpha\tau/(m\Delta x)^2=0.5$ **exactly** — its own stability limit. Coarsen that operator by
$M=m$ and all $M$ aliased eigenvalues coincide, so the minimal polynomial has degree 1 and the
coarse law is **exactly Markov, zero memory** — against the $M-1$ lags FTCS needs. So the frontier
is precisely where coarse-graining is *free*, and that is the same degeneracy that makes it carry
no information. **Memory-free coarse-graining and uselessness are one phenomenon, not two.**

But "the same operator" overstates it. Both *converge* to the same Gaussian; as finite objects
they differ, and the difference is measurable and goes the surprising way: **the 3-moment
maximum-entropy positive stencil is about $4\times$ more accurate than the composite it
compresses**, because it targets the exact semigroup while the composite reproduces
$L\varepsilon_2$. Matching *more* moments then makes it monotonically **worse** on a truncated
footprint. Cost ratios are $L^{3/2}$ for one output value (the light cone is $L^2$
point-evaluations, not $L$) and $L^{1/2}$ on a full grid.

---

## 5. The decisive experiment: positivity, locality, accuracy

**Pawula's theorem** (Phys. Rev. 162, 186, 1967): for a Markov process with Kramers–Moyal
expansion $\partial_tP=\sum_n(-\partial_x)^n[D^{(n)}P]$, non-negativity of $P$ forces
$[D^{(m+n)}]^2\le D^{(2m)}D^{(2n)}$; hence if any even $D^{(2j)}$, $j\ge2$, vanishes then all
$D^{(n)}$, $n\ge3$, vanish. The expansion stops at 2 or has infinitely many non-zero terms;
no truncation at finite order $\ge3$ is consistent with a density.

**Translating this as "a local positive propagator is necessarily second-order" is wrong, and
the distinction is the substance.** Pawula constrains the stencil's own cumulant sequence — it
may not *terminate*. Accuracy requires only *matching* finitely many moments of the target.
Concretely, at $r=0.45$ the 5-point stencil exact on $\Theta_0..\Theta_4$ is **positive**
($\min w=0.0577$, $\|w\|_1=1$): fourth order, local, positive. Its own $\kappa_1..\kappa_4$
match the kernel while $\kappa_5\dots\kappa_{10}$ are all non-zero — exactly as Pawula
requires — and Cauchy–Schwarz holds with 0 violations.

There *is* an order cap on positive local stencils, but it is set by $r$:

| $r$ | 0.45 | 0.40 | 0.35 | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| max spatial order, $w\ge0$ | **10** | **10** | 8 | 6 | 6 | 6 | 2 | 2 | 2 |

The cap tracks the kernel width $\sigma=\sqrt{2r}$ cells: when $\sigma\ll1$ the exact kernel is
nearly a delta and its high moments cannot be reproduced by a non-negative lattice measure.
The classical result that *does* cap order for positive schemes is **Godunov's theorem**, which
is hyperbolic (monotone linear schemes for advection are at most first order); here $r$ is held
fixed with real diffusion present, the problem is parabolic, and Godunov's bound does not bite.

### The trade curve — memory buys back positivity, geometrically

Exact formulation, no trajectory fitting: a compact coarse law with half-width $s$ and memory
depth $p$ reproduces the fine dynamics iff
$\sum_j\widehat{B_j}(q)\,g_l^{\,p+1-j}=g_l^{\,p+1}$ for every coarse mode $q$ and alias $l$ —
solved once unconstrained ($R_{\rm free}$) and once with all weights $\ge0$ ($R_{\rm pos}$).
Initial-condition independent. At fixed minimal locality $s=1$:

| $M$ | decay per lag | extra lags per decade |
|---|---|---|
| 2 | exact at $p=1$ | — (positivity is free) |
| 3 | $0.0966^p$ | **0.99** |
| 4 | $0.4414^p$ | **2.82** |
| 6 | $0.7370^p$ | **7.54** |

**The prediction is confirmed, and quantified.** Memory buys back positivity geometrically,
and the exchange rate degrades sharply with the coarsening factor. Every solution has weights
$\ge0$ with total mass $S\le1$, hence satisfies a discrete maximum principle
$\max|U^{n+1}|\le S\max_j\max|U^{n+1-j}|$ — unconditionally $\ell_\infty$-stable, no transient.

**Stronger than the prediction:** for small $M$ the price is not merely small, it is **zero**.
An *exact* non-negative compact coarse law exists — $M=2$ at $s{=}1,p{=}1$ (6 weights) and $M=3$
at $s{=}2,p{=}5$ (30 weights), both with mass exactly 1 and $\min w\ge0$. Validated outside the
system they were fitted in: one-step error $\sim10^{-16}$ and a 400-step *unforced* self-rollout
staying at $10^{-15}$. The $M=2$ law reproduces, to all digits, the one derived independently from
Cayley–Hamilton on the alias block — two unrelated routes agreeing exactly.

Locality and memory are interchangeable: at $M=3$, widening from 3 to 5 weights per lag cuts the
memory needed for exactness from $p\approx15$ to $p=5$. And the structure of the solution is not
what "adding memory" suggests: the $M=3$ law has $B_1=B_2=0$ **exactly** — a **pure-delay** scheme
starting at lag 2. Buying back positivity does not mean correcting a Markov law; it means moving
the whole law backwards in time.

At $M=4$ this stops: $s{=}2\to4$ moves the $p{=}17$ residual only from $1.0\times10^{-8}$ to
$1.5\times10^{-9}$, with no snap to zero. Geometric convergence, not exact representability —
though that is a failure to find within $s\le4,p\le17$, not a proof.

So the honest form of the trilemma: **positivity, locality, accuracy — you can often have all
three, and where you cannot, the price is continuous and measurable rather than a prohibition.**

---

## 6. Coarse-graining and memory

The coarse law is **exactly** an order-$M$ recurrence: the alias class of each coarse mode is an
$M$-dimensional invariant subspace, and Cayley–Hamilton on that block gives a memory of exactly
$M-1$ lags that **terminates** — unusual, since MZ kernels are normally infinite. Two answers
against the expected form:

- **The kernel does not decay.** $\|B_2\|_1>\|B_1\|_1$ for every $M$ from 2 to 20; at $M=20$ the
  largest is $\|B_5\|_1=7.96$ against $\|B_1\|_1=2.00$. Nothing to truncate.
- **The naive coarse law loses positivity for $M\ge3$**, and $\sum_p\|B_p\|_1$ grows like
  $e^{0.19M}$. $M=2$ survives iff $r\ge1-1/\sqrt2=0.292893$ (analytic; bisection $0.293048$).
  Since FTCS needs $r\le\tfrac12$, the safe window is $r\in[0.293,0.5]$.

Note this is the *same* condition as the order cap in §5: **positivity needs enough mixing per
step**, in both settings. And it is why the trade curve exists — the exact $B_p$ are negative,
but extra memory provides the freedom to trade back into the non-negative cone.

Confirmed by the RG-consistent coarsening: pairing $\Delta x\to M\Delta x$ with
$\Delta t\to M^2\Delta t$ collapses the memory to an **$M$-independent floor** $e^{-2\pi^2 r}$
(verified to a few percent for $r=0.15$–$0.45$, $M=8$–$24$), set by the two exactly degenerate
aliases at the coarse Nyquist. The residual memory is precisely the part of the highest coarse
mode that one coarse step has not yet mixed away.

> **Memory is the price of coarse-graining space faster than the dynamics mixes.**

---

## 7. Status

**Proved.** Closure (Thm 1) and the $\Pi_r$ obstruction; the binomial moment law; exact
additivity of cumulant defects (Thm 1′) and $L$-independence of modified-equation coefficients;
$\ell_1$ submultiplicativity and preservation of positivity (Thm 2); the sharp wide-stencil
boundary $2\alpha k\Delta t+(ck\Delta t)^2\le m^2\Delta x^2$ both ways; exactness and finiteness
of the order-$M$ coarse recurrence (Thm 3); the $M=2$ threshold $r\ge1-\tfrac12\sqrt{2-r_a^2}$.

**Classical, not mine** (named as such throughout): Pawula; Mori–Zwanzig; Godunov; the RKC
stability scaling behind $k\le m^2/2r$ and the $L^{1/2}$ speedup; maximum-norm stability of
dissipative schemes behind the positivity-restoration regime.

**Numerically established** (one operating point, one equation, uniform lattice). The three
$\|w^{*L}\|_1$ regimes; both Edgeworth regimes against their analytic constants (0.34 %, 0.01 %);
$\rho(L)$ as compression ratio and validity range; $L^{3/2}$/$L^{1/2}$ cost ratios; maxent
beating the composite and degrading with more moments; the order cap vs $r$ (order 10 at
$r=0.45$); non-decay of the space-only memory kernel; loss of positivity for $M\ge3$ and its
$e^{0.19M}$ growth; the $e^{-2\pi^2r}$ floor; **the trade curve and the exact non-negative
compact coarse laws at $M=2,3$**; the frontier stencil being the extremal two-point measure with
band error $\approx1$.

**Scope bound on anything claimed jointly with T2.** Their variable-coefficient advantage tracks
$L_\alpha=\alpha/|\alpha'|$ measured *in cells*: $\sim\!2400\times$ over FTCS at $L_\alpha\ge20$, $27\times$ at
11, gone by 6, and actively harmful below $\approx2$. That is the validity range of the local Taylor
rows, and therefore of the moment machinery this whole thread is built on. Their quoted speedups
are all off the frontier (safety factor $s=m/\sigma\in[1.5,6]$ constant-coefficient, $s=2.60$ at the
variable-coefficient winner), so the frontier caveat above does not touch them.

**Still a story.** That the flow's positivity restoration is a theorem — local CLT covers the
bulk, the far tails are not controlled, and some large-$L$ "$\min w=0$" readings are underflow.
Whether an exact non-negative compact coarse law exists for $M\ge4$ at all: none found within
39 weights and $p<20$, which is a failure to find, **not** a proof of non-existence. Whether
any of this survives variable coefficients, non-uniform lattices, nonlinearity, or $>1$
dimension — every result leans on constant coefficients (cumulant additivity, circulant
aliasing) and none has been tested outside that. Whether "$\rho(L)$ is the RG control parameter"
is more than a relabelling of the CLT.

**Worth pulling.** Is the exponent in the $e^{-2\pi^2 r}$ floor universal across schemes at
fixed $r$? Is the trade-curve rate $\rho(M)$ predictable from the alias spectrum rather than
measured? And composition reaches wide kernels that direct moment-matching cannot touch in
double precision ($\mathrm{cond}\sim10^{17}$ at $m=10$) — that is an algorithm, not just an
observation, and it has not been written down as one.
