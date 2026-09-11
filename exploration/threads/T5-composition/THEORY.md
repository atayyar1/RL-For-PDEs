# T5 — Composition of local linear predictors: precise statements

Setting throughout: the constant-coefficient advection–diffusion equation

$$u_t + c\,u_x = \alpha\,u_{xx},\qquad \alpha=0.1,\ c=1,$$

on a uniform lattice $\Delta x = 1/99 = 1.0101\times10^{-2}$, $\Delta t = 4.59137\times10^{-4}$, so
$r = \alpha\Delta t/\Delta x^2 = 0.45$ and $r_a = c\Delta t/\Delta x = 0.0454\overline{54}$.
The reference agent is the FTCS stencil

$$w_{\rm FTCS} = \big(r + \tfrac{r_a}{2},\ 1-2r,\ r - \tfrac{r_a}{2}\big) = (0.472727,\ 0.1,\ 0.427273)$$

on offsets $(-1,0,+1)$ at time offset $-1$. It is non-negative with $\|w\|_1=1$.

All claims below are labelled **[P]** proved, **[N]** numerically verified, or **[C]** conjectural.
Verification scripts: `task1_theory.py` (§1–3), `task2_rg.py` (§4), `task3_compression.py` (§5), `task4_mz.py` (§6).

---

## 0. Definitions

An **agent** is a finitely supported signed measure on the displacement lattice: a set of
displacements $\delta_i=(\xi_i,\tau_i)\in\mathbb{R}^2$ (in practice $(\,k\Delta x,\ -j\Delta t)$) with
weights $w_i$, acting as

$$\hat u(z^\*) \;=\; \sum_i w_i\,u(z^\*+\delta_i).$$

**Raw moments** $M_{p,q}(w)=\sum_i w_i\,\xi_i^p\tau_i^q$; **symbol** (moment generating function)
$\widehat W(s)=\sum_i w_i e^{\langle s,\delta_i\rangle}$, so $M_{p,q}=\partial_{s_x}^p\partial_{s_t}^q\widehat W(0)$;
**cumulants** $\kappa_n$ from $K_w=\log\widehat W$. The **von Neumann symbol** is
$g(\theta)=\sum_i w_i e^{\mathrm{i}k_i\theta}$ for single-time-level stencils with offsets $k_i$ in cells.

**Composition.** Agent $w$ at $z^\*$ consumes the values $u(z^\*+\delta_i)$, each of which is itself
produced by an agent $v_i$ centred at $z^\*+\delta_i$ with displacements $\epsilon_{ij}$:

$$\hat u(z^\*)=\sum_i w_i \sum_j v_{ij}\, u\big(z^\*+\delta_i+\epsilon_{ij}\big).$$

---

## 1. Closure

> **Proposition 1 (support/weight law).** **[P,N]**
> The composite is the agent with displacements $\delta_i+\epsilon_{ij}$ and weights $w_i v_{ij}$
> (coincident displacements summed). Equivalently it is the pushforward of the product measure
> under addition. If all sub-agents are the same measure $v$ relative to their own centres, the
> composite is the **convolution** $w*v$.

Immediate, and the technical content is only that *displacements add*. Verified to exact equality
in `task1_theory.py §1A`.

> **Theorem 1 (closure of exactness).** **[P]**
> Let $\mathcal S$ be any set of functions closed under the translations in use — for a
> constant-coefficient PDE, any linear space of exact solutions. If $w$ reproduces every $u\in\mathcal S$
> at its own centre and **every** sub-agent $v_i$ reproduces every $u\in\mathcal S$ at its own centre,
> then the composite reproduces every $u\in\mathcal S$ at $z^\*$.
>
> *Proof.* $\sum_i w_i\sum_j v_{ij}u(z^\*+\delta_i+\epsilon_{ij}) = \sum_i w_i u(z^\*+\delta_i)=u(z^\*)$. $\square$

No hypothesis on supports, weights, homogeneity, or positivity. Two consequences worth stating:

* **Order is inherited, and it takes the minimum, not the sum.** Degree-2 $\circ$ degree-2 = degree-2.
  The intuition "orders multiply" is wrong; the composite cannot fall *below* $\min(r_w,r_v)$ either,
  so **there is no counterexample of the "composition degrades the order" type.** Searched anyway
  (200 random heterogeneous compositions at $r=1,2,3$): max relative defect $10^{-14}$, i.e. none.

* **Which class?** Not the space-time polynomials.

> **Proposition 2 (obstruction).** **[P,N]**
> A single-time-level stencil (all $\tau_i=\tau$) can never reproduce space-time polynomials of total
> degree $\ge 1$, because $M_{0,1}=\tau\sum_i w_i=\tau\neq 0$.
> (`task1_theory.py §1D`: best achievable residual $0.707$ for FTCS's footprint, $8\times10^{-16}$ once
> two time levels are available.)

The right class is the **PDE-adapted (heat/drift) polynomials**, the Appell sequence

$$\Theta_n(x,t)=H_n(x-ct,\;t),\qquad H_n(y,t)=n!\sum_{k}\frac{(\alpha t)^k\,y^{\,n-2k}}{k!\,(n-2k)!},$$

i.e. $\Theta_0=1$, $\Theta_1=y$, $\Theta_2=y^2+2\alpha t$, $\Theta_3=y^3+6\alpha t\,y$, …, each an exact
solution (verified symbolically, residual $0$ for $n\le4$). These *are* reachable at one time level,
and Theorem 1 applies verbatim. Verified on $\mathcal S=\Theta_{\le r}$, $r=1,2,3$, 200 random
heterogeneous compositions each: composite defect $\le 1.1\times10^{-14}$, defect at degree $r{+}1$
order $10^{-1}$ (so the test has power).

### 1.1 The moment composition law

> **Proposition 3 (moments convolve binomially).** **[P,N]**
> If all sub-agents share the same moments, $\widehat W_{\rm comp}=\widehat W\cdot\widehat V$ and
> $$M_{p,q}(w*v)=\sum_{a\le p}\sum_{b\le q}\binom{p}{a}\binom{q}{b}M_{a,b}(w)\,M_{p-a,q-b}(v).$$
> In general (sub-agents differing),
> $$M_{p,q}(\mathrm{comp})=\sum_i w_i\sum_{a\le p,\,b\le q}\binom{p}{a}\binom{q}{b}\xi_i^a\tau_i^b\,M_{p-a,q-b}(v_i).$$

This is the "convolution/shift structure": moments are **not** products. With $M_{0,0}=1$,
$M_{1,0}(w*v)=M_{1,0}(w)+M_{1,0}(v)$ — *additive*. Verified to $6\times10^{-15}$ relative
(homogeneous) and $5\times10^{-16}$ (heterogeneous) over all $p\le4,q\le3$.

### 1.2 The technical heart: cumulant defects are exactly additive

The exact propagator of $u_t+cu_x=\alpha u_{xx}$ over time $\tau>0$, as a kernel in the displacement
$\xi$, is $\mathcal N(-c\tau,\ 2\alpha\tau)$. Its cumulants are

$$\kappa_1=-c\tau,\qquad \kappa_2=2\alpha\tau,\qquad \kappa_{n\ge3}=0,$$

**all linear in $\tau$** — the kernel is infinitely divisible (a Lévy semigroup). Define the
**cumulant defect** of an agent advancing time $\tau$:

$$\varepsilon_n(w) \;:=\; \kappa_n(w)-\kappa_n^{\rm exact}(\tau).$$

> **Theorem 1′ (exact additivity of defects).** **[P,N]**
> Cumulants add under convolution ($K_{w*v}=K_w+K_v$), and the exact target is linear in $\tau$.
> Hence for two agents each advancing $\tau$,
> $$\varepsilon_n(w*v)=\varepsilon_n(w)+\varepsilon_n(v)\quad\textbf{exactly},\qquad
>   \varepsilon_n(w^{*L})=L\,\varepsilon_n(w).$$
> No $O(\cdot)$ remainder. Verified to $3\times10^{-13}$ relative up to $L=512$.

> **Corollary (modified equation).** **[P,N]** The effective diffusivity of the $L$-fold composite,
> $\kappa_2/(2L\Delta t)=\alpha+\varepsilon_2/(2\Delta t)$, is **independent of $L$**.
> For FTCS, $\varepsilon_2=-c^2\Delta t^2$ exactly, so $\alpha_{\rm eff}=\alpha-\tfrac{c^2\Delta t}{2}=0.09977043$
> at every $L\in\{1,\dots,512\}$ (measured; all digits identical). This *is* FTCS's classical negative
> numerical diffusion, recovered as an additive defect — and it explains why modified-equation
> coefficients are step-count independent.

> **Proposition 4 ($\Theta_n$-exactness $\Leftrightarrow$ $\varepsilon_n=0$).** **[N]**
> The $\Theta_n$ defect equals $\varepsilon_n$ exactly when $\varepsilon_1=\dots=\varepsilon_{n-1}=0$.
> For FTCS ($\varepsilon_2,\varepsilon_3\neq0$) they agree exactly at $n\le3$ and to $8\times10^{-6}$
> relative at $n=4$.

### 1.3 Heterogeneous sub-agents: defects are averaged with *signed* weights

$\Theta_n$ is a binomial-type (Appell) sequence, $\Theta_n(z+z')=\sum_k\binom{n}{k}\Theta_k(z)\Theta_{n-k}(z')$.
Pushing that through the composition:

> **Proposition 5.** **[P,N]** With $D_n$ the $\Theta_n$ defect,
> $$D_n(\mathrm{comp}) = D_n(w) + \sum_i w_i\sum_k\binom{n}{k}\Theta_k(\delta_i)\,d^{(i)}_{n-k},$$
> which collapses, when every sub-agent is $\Theta_{n-1}$-exact, to
> $$D_n(\mathrm{comp}) = D_n(w) + \sum_i w_i\,d^{(i)}_n,
> \qquad\big|D_n(\mathrm{comp})-D_n(w)\big|\le \|w\|_1\max_i\big|d^{(i)}_n\big|,$$
> with the bound **attained** when $\mathrm{sign}(d^{(i)}_n)=\mathrm{sign}(w_i)$.

Verified at $\|w\|_1\in\{1.34,1.5,3,8\}$: predicted and measured composite defects agree to
round-off. **This is where $\|w\|_1>1$ hurts:** it amplifies the *spread* of the sub-agents' defects.
It is the composition-theoretic content of the empirical finding in `rollout.py` that min-norm
(sign-indefinite) stencils lose to positive ones under heterogeneous rollout.

---

## 2. Submultiplicativity

> **Theorem 2.** **[P,N]** $\ \|w_{\rm comp}\|_1\le\|w\|_1\cdot\max_i\|v_i\|_1$, with equality iff, for
> every attained displacement, the contributing terms $w_iv_{ij}$ all share one sign.
> Consequences: positivity is preserved under composition (products and sums of non-negatives), so
> $w\ge0,\ \sum w=1$ gives $\|w^{*L}\|_1=1$ for **every** $L$; and $\|w\|_1>1$ admits geometric growth.

Verified: $0$ violations in $4000$ random heterogeneous compositions (ratio lhs/rhs median $0.617$,
max $0.99976$); $2000$ positive$\times$positive compositions all positive with mass $1$;
FTCS gives $\|w^{*L}\|_1=1.000000000000$ for $L$ up to $512$; $w=(-0.25,1.5,-0.25)$ gives
$\|w^{*L}\|_1=2^L$ exactly up to $L=16$.

### 2.1 The iterated bound is *not* generically attained — and this matters

$\|w\|_1$ is the $\ell_\infty\to\ell_\infty$ operator norm (the maximum-principle constant);
$\max_\theta|g(\theta)|$ is the $\ell_2$ one (von Neumann). Always
$\max_\theta|g|\le\|w\|_1$. Measured, three regimes:

| regime | $\|w^{*L}\|_1$ behaviour |
|---|---|
| (a) $w\ge0$, $\sum w=1$ | $=1$ **exactly**, every $L$ |
| (b) $\|w\|_1>1$ but $\max_\theta\lvert g\rvert\le1$ | $\to 1$; Theorem-2 bound wildly pessimistic |
| (c) $\max_\theta\lvert g\rvert>1$ | geometric growth at rate $\max\lvert g\rvert$, **not** $\|w\|_1$ |

Example of (b): the 7-point $\Theta_0..\Theta_6$-exact stencil at $r=0.05$ has $\|w\|_1=1.020503$,
$\min w=-5.64\times10^{-3}$, $\max|g|=1.0000000000$. Then

| $L$ | 1 | 8 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| $\|w^{*L}\|_1$ | 1.020503 | 1.000745 | 1.0000000 | 1.0000000 | 1.0000000 |
| Thm-2 bound $\|w\|_1^L$ | 1.021 | 1.176 | 1.915 | 3.665 | 180.5 |
| negative mass | $1.0\times10^{-2}$ | $3.7\times10^{-4}$ | $9.7\times10^{-13}$ | $4.7\times10^{-25}$ | $1.6\times10^{-90}$ |
| innermost negative $\lvert k\rvert/\sigma_L$ | 6.3 | 4.4 | 8.9 | 12.0 | 25.2 |

Example of (c): FTCS pure advection ($\alpha=0$) has $\|w\|_1=1.0455$ but $\max|g|=1.001033$;
$\|w^{*L}\|_1$ reaches $18.4$ at $L=1089$ and $418$ at $L=4096$, versus a Theorem-2 bound of $10^{21}$
and $10^{79}$.

> **Observation 1 (the flow restores positivity).** **[N, mechanism C]**
> For $\max_\theta|g|\le1$ with equality only at $\theta=0$ and $\kappa_2>0$, the negative lobes are not
> cancelled — they are **expelled**: the innermost negative weight moves out to $\sim25\sigma_L$ while
> the negative mass decays faster than exponentially. The composite is non-negative throughout its
> effective support. A local-CLT argument gives this in the bulk; controlling the far tails to a
> proof is **[C]** (and "$\min w=0$ at large $L$" in our tables is partly floating-point underflow,
> so the reliable statements are the negative-mass and innermost-$|k|/\sigma$ columns).
> Related classical material: $\ell_\infty$ (maximum-norm) stability of dissipative difference schemes.

**Amendment to the thread's starting hypothesis.** "$\|w\|_1=1$ is exactly the condition under which
an agent composes with itself indefinitely without amplification" is **sufficient but not necessary**.
Precisely:

* $w\ge0$ is *necessary and sufficient* for **exact** non-amplification at every $L$ including $L=1$
  (a maximum principle at every scale, with no transient);
* $\max_\theta|g|\le1$ is *necessary and sufficient* for non-amplification **asymptotically**, and the
  flow repairs a non-positive start.

So positivity is the *scale-free* condition (nothing to wait for), while von Neumann stability is the
*asymptotic* one. That distinction is the honest version of "stability is the tax on composition".

---

## 3. What actually degrades under composition

Ordered by how sharp they are.

1. **Not the order.** Theorem 1 forbids it. **[P]**
2. **The inhomogeneous one-step conditions are not inherited as identities.** Two agents each exact
   for step $\Delta t$ do not compose to one exact for $2\Delta t$: $M_2(w*w)=1.84076\times10^{-4}$
   versus the required $1.84498\times10^{-4}$, short by exactly $2\varepsilon_2$. Theorem 1′ says the
   shortfall is exactly additive, never worse. **[P,N]**
3. **Heterogeneous defect amplification by $\|w\|_1$** (Proposition 5). **[P,N]**
4. **The sharp failure: individually consistent, composite ill-posed.** Consistency constrains
   $\varepsilon_n$ but not its **sign**. A stencil with $\kappa_2<0$ is a perfectly good first-order
   consistent one-step scheme whose composite is not a kernel at all. Measured: **[N]**

   | scheme | $\kappa_2$ | $\|w\|_1$ | $\max\lvert g\rvert$ | $\max\lvert g\rvert^{1089}$ |
   |---|---|---|---|---|
   | FTCS advection–diffusion, $r=0.45$ | $+9.16\times10^{-5}$ | 1.00000 | 1.000000 | 1 |
   | FTCS pure advection ($\alpha=0$) | $-2.11\times10^{-7}$ | 1.04545 | 1.001033 | 3.08 |
   | central advection $+$ tiny diffusion $r=5\times10^{-4}$ | $-1.09\times10^{-7}$ | 1.04445 | 1.000275 | 1.35 |

   The last two are first-order **consistent** ($M_0=1$, $M_1=-c\Delta t$) and still lose.
   In this language Lax equivalence reads: *consistency fixes the defects $\varepsilon_n$; stability
   fixes $\mathrm{sign}(\kappa_2)$.*
5. **A trade-off we looked for and did not find.** We built the unique 5-point one-level stencil
   exact on $\Theta_0..\Theta_4$ expecting high order to lose to positivity at large $L$. At $r=0.45$
   that stencil is itself **positive** ($\min w=0.0577$, $\|w\|_1=1$) and beats FTCS by $3$–$5000\times$
   at every $L$ up to $1089$. There is no crossover. The trade-off only appears below $r\approx0.17$,
   where the 5-point exact stencil goes indefinite ($\|w\|_1$ up to $1.014$) — and even there
   $\max|g|=1$, so by Observation 1 it is regime (b) and the flow repairs it. **[N]**
   Honest conclusion: at this operating point **high order and positivity are compatible**, and the
   claimed tension between them is not supported.

---

## 4. Composition as renormalization (summary of §Task 2)

For $w\ge0$ the $L$-fold composite is the law of a sum of $L$ i.i.d. lattice steps, so the flow is a
random walk and the fixed point is the heat kernel. Per-step, in cell units:
$\mu_1=-r_a=-0.0454545$, $s_2=2r-r_a^2=0.897934$, $\kappa_3=+1.22\times10^{-1}$, $\kappa_4=-8.0\times10^{-1}$.

* **[N]** Non-negativity and unit mass hold at every $L\le4096$ ($|\sum w-1|\le 4\times10^{-16}$).
* **[P,N]** Diffusive collapse. With $z=(k-L\mu_1)/\sigma_L$, $\sigma_L=\sqrt{s_2L}$:
  the local-CLT sup-distance and the total variation both decay like $L^{-1/2}$
  (fitted $L^{-0.563}$, $L^{-0.532}$ on $L\ge256$), and $L^{-1/2}\times$ them plateau at
  $0.00842$ / $0.0115$. The predicted plateau from the first Edgeworth term,
  $|\kappa_3|/(6s_2^{3/2})\cdot\max|He_3\phi| = 0.008313$, matches to $1.3\%$.
  Subtracting that term gives $L^{-0.9995}$ (predicted $L^{-1}$).
  *Caveat:* fitting from $L\ge16$ instead returns $L^{-0.78}$, a pre-asymptotic transient.
* **[P,N]** Formal vs effective support: formal half-width $=L$ exactly; effective $\sigma_L=\sqrt{s_2L}$;
  ratio $\rho(L)=L/\sigma_L=\sqrt{L/s_2}\propto\sqrt L$. At $L=4096$: $4096$ vs $60.6$ cells, $\rho=67.5$.
* **[N] Correction to a natural guess:** the tails are **not** strongly sub-Gaussian. The ratio of the
  composite's tail mass to the Gaussian's is $\approx1$ throughout ($1.00$ at $k=4$, $0.97$ at $k=6$,
  $0.96$ at $k=8$ for $L=4096$) and collapses only as $k\to\rho(L)$, where the light cone cuts it off
  dead. So **$\rho(L)$ is simultaneously the compression ratio and the range of validity (in units of
  $\sigma_L$) of the fixed-point description** — a finite-size effect in exactly the RG sense.

---

## 5. Wide stencils are compressed composites (summary of §Task 3)

> **Proposition 6 (the feasibility boundary, with proof).** **[P,N]**
> Let $w\ge0$ be supported on $|\xi|\le m\Delta x$ with $\sum_i w_i=1$ and $\sum_i w_i\xi_i^2=2\alpha\tau$
> ($\Theta_0$ and $\Theta_2$). Then $2\alpha\tau=\sum_i w_i\xi_i^2\le m^2\Delta x^2$, so with $\tau=k\Delta t$
> $$k\ \le\ \frac{m^2\Delta x^2}{2\alpha\Delta t}=\frac{m^2}{2r}. \qquad\textbf{(necessary)}$$
> Conversely $w=(1-p)\delta_0+\tfrac p2(\delta_{-m}+\delta_{+m})$ with $p=2\alpha\tau/(m\Delta x)^2$ is
> non-negative and satisfies both whenever $k\le m^2/(2r)$. **(sufficient)**
> So the boundary is exactly $\sigma_{\rm required}=\sqrt{2\alpha k\Delta t}\le m\Delta x$:
> **a positive stencil can take the step iff the kernel's standard deviation fits inside it.**
> LP bisection reproduces $k_{\max}(m)/(m^2/2r)\in[0.98,1.11]$ for $m\in\{1,2,3,5,8,12,20,32\}$.

> **Corollary (the two thread results are one statement).** **[P,N]**
> Substituting $m=\sigma_L=\sqrt{s_2L}$ gives $k=s_2L/(2r)=L\,(1-r_a^2/2r)=0.99770\,L$.
> So $k=L$ up to the $O(r_a^2/r)=2.3\times10^{-3}$ advective correction.
> **Convention trap:** the diffusive width is $\sqrt{2\alpha t}$, not $\sqrt{\alpha t}$; with
> $\sqrt{\alpha L\Delta t}$ the two statements disagree by $\sqrt2$, with the variance convention
> they agree identically.

**Cost.** One output value: the composite needs its whole light cone,
$\sum_{\ell=1}^L(2\ell-1)=L^2$ applications $\times3$ mult-adds $=3L^2$; the wide stencil costs $2m+1$
with $m=\lceil4\sigma_L\rceil$. Ratio $\propto L^{3/2}$ (fitted $L^{1.506}$). Full grid of $N$ points:
$3NL$ vs $(2m+1)N$, ratio $\propto L^{1/2}$ (fitted $L^{0.506}$). At $L=4096$: $10^5\times$ and $25\times$.
Note the grid ratio is $<1$ below $L\approx8$ — compression only pays past a crossover.

**What the compression loses** (measured by the max over $|\theta|\le\pi/4$ of $|g-\text{exact}|$):

* Truncating the exact composite to $\pm m$ and renormalising saturates at the composite's own error
  once $m\gtrsim4\sigma_L$; the $\ell_1$ distance to the untruncated composite falls like the Gaussian
  tail ($5\times10^{-5}$ at $4\sigma$, $10^{-9}$ at $6\sigma$). **Essentially lossless.**
* **[N] Two results that were not expected.** (i) The maximum-entropy non-negative stencil matching
  only the **3** standard consistency moments is about $4\times$ *more accurate* than the composite it
  compresses ($3.8\times10^{-4}$ vs $8.8\times10^{-4}$ at $L=1024$) — because it targets the exact
  semigroup while the composite faithfully reproduces FTCS's accumulated $L\varepsilon_2$.
  (ii) Matching **more** moments makes it **worse** monotonically ($J=6,8,12$ give $1.1,2.2,10.6\times10^{-3}$):
  forcing the full Gaussian's high moments onto a support truncated at $4\sigma$ is inconsistent, and
  the optimiser pays for them by distorting the bulk.
* Width alone buys nothing: $J=2$ (mass and mean, no variance) has error $0.80$ at any width.
* Maxent weights are strictly positive at every $J$ tested, so there is no accuracy/positivity tension here.
* **[N]** The naive route — solving the full $(2m+1)$-moment system directly — is useless: the lattice
  moment matrix has $\mathrm{cond}=4\times10^{4},\,2\times10^{8},\,6\times10^{11},\,5\times10^{17}$ at
  $m=2,4,6,10$, giving $\|w\|_1$ up to $10^{14}$. Composition reaches the same wide kernel **stably**,
  because it never forms the moment matrix. An independent practical reason to compose.

---

## 6. Coarse-graining and memory (summary of §Task 4)

Periodic fine grid $N=120$, retain every $M$-th point, $N_c=N/M$.

> **Theorem 3 (exact, finite Mori–Zwanzig memory).** **[P,N]**
> The fine operator is circulant with symbol $g(\theta_j)$. Restriction to every $M$-th point aliases
> the $M$ fine modes $j,\,j+N_c,\dots,j+(M-1)N_c$ onto one coarse mode. That alias class is an
> $M$-dimensional **invariant** subspace of the fine operator, and the coarse observable is one linear
> functional on it; Cayley–Hamilton on the $M\times M$ block gives an exact order-$M$ recurrence
> $$U^{n+1}=B_1U^n+B_2U^{n-1}+\dots+B_MU^{n-M+1},\qquad
> \widehat{B_p}(q)=(-1)^{p+1}e_p\big(g_0(q),\dots,g_{M-1}(q)\big),$$
> with $e_p$ the elementary symmetric functions. **The memory is exactly $M-1$ lags and it terminates.**
> This is unusual — MZ kernels are normally infinite — and the finiteness comes from the fine operator
> being finite-dimensional *and* block-diagonal over alias classes.
> Measured residual at $M-1$ lags: $6\times10^{-16}$ to $1.5\times10^{-15}$ for $M=2..12$;
> at $M-2$ lags: $8.4\times10^{-1}$ down to $3.1\times10^{-4}$; Markov only: $0.16$–$0.87$.
> **Memory required grows linearly in the coarsening factor $M$.**

Note the coarse law needs $M$ initial time levels to start — the standard MZ "memory of the
initialisation", here made concrete.

> **Observation 2 (the kernel does not decay).** **[N]** $\|B_2\|_1>\|B_1\|_1$ for every $M\ge2$:
> the lag-1 operator is *larger* than the Markov operator. Space-only coarse-graining at the fine time
> step does not add a small memory correction to a Markov law — memory dominates it. There is no
> exponential decay to truncate, hence no cheap approximate closure in this variable.

> **Observation 3 (positivity has a threshold at $M=3$).** **[N]** The signs alternate with the lag
> ($\widehat{B_p}=(-1)^{p+1}e_p$). Whether that lands on a non-negative stencil depends on the signs of
> the aliased $g_l$. Measured at $r\in\{0.45,0.35,0.25,0.15,0.05\}$:
> **$M=2$ is positive, exactly, at every $r$** — the max principle survives coarsening by 2, with
> $\sum_p\|B_p\|_1=1$; explicitly $U^{n+1}_p=(2-4r)U^n_p+B_2U^{n-1}$ with $B_2\ge0$ of mass $4r-1$…
> (at $r=0.45$: $B_1=0.2\,I$, $B_2\approx(0.2235,0.394,0.1825)$, total mass 1).
> **$M\ge3$ is not positive at any $r$ tested**, and $\sum_p\|B_p\|_1$ grows geometrically in $M$.

> **Observation 4 (the tension dissolves under RG-consistent coarsening).** **[N]**
> Space-only coarsening is not the RG-natural move. Diffusive scaling pairs $\Delta x\to M\Delta x$ with
> $\Delta t\to M^2\Delta t$, holding $r$ fixed — one coarse step $=K=M^2$ fine steps. Then the non-zero
> aliases satisfy $|g_l|<1$ strictly and are raised to the power $M^2$, so $e_p$ for $p\ge2$ is crushed:
> the coarse law becomes Markov to round-off, stays positive, and keeps $\sum_p\|B_p\|_1=1$.
> **Memory is the price of coarse-graining space faster than the dynamics mixes. Coarse-grain space
> and time at the diffusive ratio and there is no price.**

---

## 7. Status table

| # | Statement | Status |
|---|---|---|
| 1 | Composition = pushforward under displacement addition; convolution if homogeneous | **[P,N]** |
| 2 | Exactness on a translation-invariant class is inherited verbatim; order takes the min | **[P,N]** |
| 3 | Single-time-level stencils cannot reproduce space-time polynomials of degree $\ge1$ | **[P,N]** |
| 4 | Moments convolve binomially; general heterogeneous law | **[P,N]** |
| 5 | Cumulant defects are **exactly** additive; modified-equation coefficients are $L$-independent | **[P,N]** |
| 6 | Heterogeneous defects averaged with signed weights; bound $\|w\|_1\max_i\lvert d_i\rvert$ attained | **[P,N]** |
| 7 | $\ell_1$ submultiplicativity; positivity preserved; $\|w^{*L}\|_1=1$ iff $w\ge0$ | **[P,N]** |
| 8 | The **iterated** bound $\|w\|_1^L$ is not generically attained; three regimes | **[N]** |
| 9 | The flow expels negativity beyond the effective support (positivity restored) | **[N]**, proof **[C]** |
| 10 | Diffusive collapse at $L^{-1/2}$, Edgeworth constant matched to 1.3%; $L^{-1}$ after correction | **[P,N]** |
| 11 | $\rho(L)=\sqrt{L/s_2}$ is both the compression ratio and the fixed point's range of validity | **[N]** |
| 12 | Positive wide-stencil feasibility boundary $k\le m^2/2r$, with proof; $=$ the composed width | **[P,N]** |
| 13 | Compression cost ratios $L^{3/2}$ (one point) and $L^{1/2}$ (grid) | **[P,N]** |
| 14 | Truncating the composite past $4\sigma$ is lossless; more moments can be worse | **[N]** |
| 15 | Coarse MZ memory is exact, finite, of length $M-1$ | **[P,N]** |
| 16 | Memory kernel does not decay; lag-1 term exceeds the Markov term | **[N]** |
| 17 | Coarse positivity: survives $M=2$ exactly, fails for $M\ge3$ | **[N]** |
| 18 | Under diffusive $M^2$ time coarsening, memory is crushed and positivity survives | **[N]** |
