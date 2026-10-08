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
$\mu_1=-r_a=-0.0454545$, $s_2=2r-r_a^2=0.897934$, $\kappa_3=+7.708490\times10^{-2}$, $\kappa_4=-1.515976$ (skewness $\gamma_1=+0.0906$, excess kurtosis $\gamma_2=-1.880$).

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

> **Proposition 6 (the feasibility boundary, sharp, drift included).** **[P,N]**
> Let $w\ge0$ be supported on $|\xi|\le m\Delta x$ (measured **about the target point**) and satisfy
> $\Theta_0,\Theta_1,\Theta_2$ for a step $\tau=k\Delta t$. Those conditions *fix* the raw moments:
> $$M_0=1,\qquad M_1=-ck\Delta t,\qquad M_2=2\alpha k\Delta t+(ck\Delta t)^2,$$
> i.e. $M_2$ is the exact propagator's raw second moment — **variance plus mean squared**.
>
> *Necessary:* $M_2=\sum_iw_i\xi_i^2\le m^2\Delta x^2\sum_iw_i=m^2\Delta x^2$, so in cell units
> $$r_a^2k^2+2rk-m^2\le0\iff k\le k_{\max}(m)=\frac{-r+\sqrt{r^2+r_a^2m^2}}{r_a^2}.$$
> *Sufficient:* the conditions ask for a probability measure on $[-m,m]$ with mean $\mu=-r_ak$ and
> variance $v=2rk$. The largest variance available at mean $\mu$ on $[-m,m]$ is $m^2-\mu^2$, so such a
> measure exists iff $v\le m^2-\mu^2$, i.e. iff $M_2=v+\mu^2\le m^2$ — **the same inequality**.
> The bound is therefore sharp up to lattice integrality, with the extremal measure at the endpoints.
>
> Two branches, with **two thresholds worth keeping apart** (refinement from T2-width-step):
> $m^\*=r/r_a=1/\mathrm{Pe}_{\rm cell}=9.90$ cells is where the quadratic law first goes *materially*
> wrong ($\approx25\%$); the two branch formulas actually **cross** at $2m^\*=19.80$. The threshold
> is $1/\mathrm{Pe}_{\rm cell}$, so saturation bites at *modest* Péclet, not at $\mathrm{Pe}\sim1$.
>
> $$r_am\ll r:\ k\to \frac{m^2}{2r}\ \ \text{(diffusion-limited)},\qquad r_am\gg r:\ k\to\frac{m}{r_a}\ \ \text{(advection-limited, a CFL bound)}.$$
> LP bisection reproduces $k_{\max}$ **exactly** at $m=1,2,3,5,8,12,20,32,50$
> ($k=1,4,9,26,62,124,273,519,903$).

> **Correction, and a message for the companion thread.** **[N]**
> The familiar $k\approx(m\Delta x)^2/(2\alpha\Delta t)=m^2/2r$ is only the *diffusion-limited branch*.
> At this operating point it already overestimates the true boundary by $1.29\times$ at $m=12$,
> $1.63\times$ at $m=20$, $2.19\times$ at $m=32$ and $3.08\times$ at $m=50$, and the gap keeps growing,
> because beyond $m\approx10$ cells the drift — not the diffusive spread — is what fills the footprint.

> **Corollary (the two thread results are one statement).** **[P,N]**
> The $L$-fold composite has raw second moment $M_2=Ls_2+(Lr_a)^2$ in cell units, i.e. an effective
> half-width **about the target** of $m=\sqrt{Ls_2+L^2r_a^2}$ — diffusive spreading *and* advective
> displacement, exactly what a wide stencil must span. Feeding that $m$ into $k_{\max}$ returns $k=L$:
>
> | $L$ | 4 | 16 | 64 | 256 | 1024 | 4096 |
> |---|---|---|---|---|---|---|
> | $m$ | 1.904 | 3.860 | 8.120 | 19.112 | 55.552 | 195.810 |
> | $k_{\max}(m)$ | 3.991 | 15.966 | 63.886 | 255.730 | 1023.588 | 4095.525 |
> | rel. err | $2.3\times10^{-3}$ | $2.1\times10^{-3}$ | $1.8\times10^{-3}$ | $1.1\times10^{-3}$ | $4.0\times10^{-4}$ | $1.2\times10^{-4}$ |
>
> The residual is exactly $r_a^2/(2r+2r_a^2L)$.
> *Convention trap:* the diffusive width is $\sqrt{2\alpha t}$, not $\sqrt{\alpha t}$; with the latter the
> two statements disagree by $\sqrt2$. With the variance convention **and** the drift term kept, they
> agree identically.

> **The advective branch is an artefact of centring, and it is avoidable.** **[P, confirmed by T2]**
> Requiring the stencil to be centred on its target is what the brief specified, not a necessity.
> Put the support on $\{j_0-m,\dots,j_0+m\}$ with $j_0=\mathrm{round}(-r_ak)$. In $\eta=j-j_0$ the
> conditions become $E[\eta]=\mu'$ with $|\mu'|\le\tfrac12$ (the rounding remainder) and
> $E[\eta^2]=2rk+\mu'^2$, so the endpoint bound gives
> $$k\le\frac{m^2-\mu'^2}{2r}\ \longrightarrow\ \frac{m^2}{2r}:$$
> the **diffusive branch is recovered at every $m$**, with no advective penalty, and a shifted
> stencil costs no more to evaluate. T2's LP confirms (27, 71, 159, 444, 1137, 2777 against
> $m^2/2r=$ 28, 71, 160, 444, 1138, 2778); the gain over centred grows without bound, $3.1\times$
> at $m=50$. So the two half-widths below are not just bookkeeping — (ii) is strictly better and
> is what a general variable-coefficient scheme should use.

### 5.1 Classical anchor: this is Runge–Kutta–Chebyshev  **[classical]**

The semi-discrete diffusion operator has eigenvalues $\lambda(\theta)=-(4\alpha/\Delta x^2)\sin^2(\theta/2)$,
so over a superstep $\tau=k\Delta t$, $z=\lambda\tau\in[-4rk,0]$. First-order RKC with $m$ stages has
stability polynomial $P_m(z)=T_m(1+z/m^2)$ and real-axis stability $|z|\le2m^2$, hence
$4rk\le2m^2\iff k\le m^2/(2r)$ — **exactly the diffusive branch**. So the quadratic width–step law
and the $L^{1/2}$ grid speedup of §5.2 are the **RKC stability scaling**, rediscovered in stencil
language. Verified independently here; flagged by T2-width-step, who also showed the frontier
scheme *is* first-order RKC with $s=m$ stages to $2\times10^{-16}$. The composition picture explains
*why* (a degree-$m$ polynomial in the shift operator spans $m$ cells and absorbs a step $\sim m^2$),
but the scaling is not new.

> **Observation 8 (the frontier is a stability boundary, not an accuracy one).** **[P,N]**
> At $k=m^2/(2r)$ exactly, $1+z/m^2=(S_{+1}+S_{-1})/2$, the averaging operator with symbol
> $\cos\theta$; and $T_m(\cos\theta)=\cos(m\theta)$, the symbol of $(S_{+m}+S_{-m})/2$. So the RKC
> frontier stencil is **exactly the two-point endpoint measure** $\tfrac12(\delta_{-m}+\delta_{+m})$
> — verified to machine zero at $m=3,6,12$. That is precisely the extremal measure in the
> Proposition 6 sufficiency proof, and precisely the LP vertex an earlier draft of `task3` found
> and dismissed as "pathological". It is not pathological; it is the frontier scheme.
>
> | $m$ | support | weights | $\sum w$ | $M_2/(2\alpha\tau)$ | band error |
> |---|---|---|---|---|---|
> | 3 | $\{-3,3\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 0.766 |
> | 6 | $\{-6,6\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 1.004 |
> | 12 | $\{-12,12\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 1.000 |
>
> The scheme is **consistent** (all three moment conditions exact), **positive**, and **stable**
> ($\|w\|_1=1$) — and its band error is $\approx1$: a bimodal kernel with the right mass and the
> right variance and the wrong shape. **Consistency + positivity + stability do not imply
> accuracy.** Accuracy is an independent fourth axis, and any speedup quoted *at* the frontier
> carries that caveat.
>
> RKC away from the frontier is **not positive**: at $m=8$, $k/k_{\max}=0.9$ gives $\min w=-0.111$,
> $\|w\|_1=2.378$ (T2 reproduces to three digits by exact DFT of $T_m(1-2\rho\sin^2(\theta/2))$, and
> counts 15 non-zero interior weights there against 0 at the frontier). So the classical scheme
> occupies exactly one point of the positive cone — its extreme vertex — and leaves it the moment
> you back off for accuracy. The maximum-entropy stencil of §5.2 is the opposite choice: same
> feasible set, interior point, accuracy instead of extremality.

> **Observation 9 (why the frontier carries no information, and the link to §6).** **[P,N]**
> $\tfrac12(\delta_{-m}+\delta_{+m})$ couples $j$ only to $j\pm m$, so the lattice splits into $m$
> residue classes mod $m$ that never exchange information. On one class the stencil is
> $(\tfrac12,0,\tfrac12)$ — FTCS for pure diffusion at coarse ratio
> $r_c=\alpha\tau/(m\Delta x)^2=0.500000000000$ **exactly**, FTCS's own stability limit. The frontier
> scheme is coarse-grid FTCS at its limit, run $m$-fold redundantly. (Independently found by T2's
> Task-3 control experiment; $T_m(\cos\theta)=\cos(m\theta)$ is the quickest route.)
>
> Now coarsen *that* operator by $M=m$ with the §6 machinery. Its symbol is $\cos(m\theta)$, and the
> aliases of coarse mode $q$ sit at $\theta_l=2\pi(q+lN_c)/N$, so $m\theta_l=2\pi mq/N+2\pi l$ and
> $$g_l=\cos(2\pi mq/N)\quad\text{independent of }l.$$
> All $M$ aliased eigenvalues coincide, the minimal polynomial has degree 1, and the coarse law is
> **exactly Markov — zero memory** (measured spread $\max_l|g_l-g_0|\le5.6\times10^{-15}$ for
> $M=m=3,4,6$). Contrast FTCS coarsened by the same $M$, which needs exactly $M-1$ lags.
>
> So the frontier scheme is precisely the point where **coarse-graining is free** — and that is the
> *same* degeneracy that makes it carry no information. Memory-free coarse-graining and uselessness
> are one phenomenon, not two.

**Two half-widths, not interchangeable.** (i) about the *target point* (Prop. 6 above): the stencil
must span both the origin and the drifted centre, so drift eats footprint. (ii) about the *kernel's
own mean* (cost and accuracy below): the stencil is re-centred at $-r_aL$ cells, so only the spread
matters and $m=\lceil4\sigma_L\rceil$. Cost uses (ii), since a shifted stencil is no more expensive.

**Cost.** One output value: the composite needs its whole light cone,
$\sum_{\ell=1}^L(2\ell-1)=L^2$ applications $\times3$ mult-adds $=3L^2$; the wide stencil costs $2m+1$
with $m=\lceil4\sigma_L\rceil$. Ratio $\propto L^{3/2}$ (fitted $L^{1.506}$). Full grid of $N$ points:
$3NL$ vs $(2m+1)N$, ratio $\propto L^{1/2}$ (fitted $L^{0.506}$). At $L=4096$: $10^5\times$ and $25\times$.
Note the grid ratio is $<1$ below $L\approx8$ — compression only pays past a crossover.

**What the compression loses** (measured by the max over $|\theta|\le\pi/4$ of $|g-\text{exact}|$):

* Truncating the exact composite to $\pm m$ and renormalising saturates at the composite's own error
  once $m\gtrsim4\sigma_L$; the $\ell_1$ distance to the untruncated composite falls like the Gaussian
  tail ($5\times10^{-5}$ at $4\sigma$, $10^{-9}$ at $6\sigma$). **Essentially lossless.**
* **[N] At $J=3$ the maximum-entropy stencil *is* the moment-matched discrete Gaussian**, not an
  alternative to it: maximising $-\sum w\log w$ under $\sum w=1$, $\sum wj=\mu$, $\sum wj^2=M_2$ gives
  $w_j\propto e^{\lambda_1j+\lambda_2j^2}$ by construction ($\log w$ quadratic in $j$ to
  $2.5\times10^{-14}$ here). T2 verified agreement with their independently root-found kernel to
  $2.8\times10^{-17}$–$1.0\times10^{-16}$ at five $(m,k)$ pairs, band errors identical to every digit.
  One construction, two derivations. The convex dual is the presentation of choice because it
  extends past $J=3$ without a bespoke parametrisation — and since going past $J=3$ on a truncated
  footprint makes things worse, $J=3$ (the Gaussian) is also the right stopping point.
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

> **Observation 2 (the kernel does not decay).** **[N]** $\|B_2\|_1>\|B_1\|_1$ for every
> $M$ from 2 to 20 (it dips below only at $M=24$), and at large $M$ the largest operator is
> not the Markov one at all — at $M=20$, $\|B_5\|_1=7.96$ against $\|B_1\|_1=2.00$.
> Space-only coarse-graining at the fine time step does not add a small memory correction
> to a Markov law: memory dominates it, with no exponential decay to truncate and hence no
> cheap approximate closure in this variable. $\sum_p\|B_p\|_1$ grows geometrically,
> $\sim e^{0.189M}$ (1.00, 1.12, 1.46, 1.93, 2.52, 4.04, 6.07, 8.71, 13.99, 30.60, 62.15 for
> $M=2,\dots,24$). Since $\|\cdot\|_1$ is the $\ell_\infty$ amplification bound per coarse
> step, the coarse scheme loses the maximum principle by a factor compounding with $M$.

> **Observation 3 (positivity: a threshold in *both* $M$ and $r$).** **[P,N]**
> The signs alternate with the lag ($\widehat{B_p}=(-1)^{p+1}e_p$), so positivity is not automatic.
> Measured over $r\in\{0.45,0.35,0.30,0.25,0.15,0.05\}$ and $M\in\{2,\dots,24\}$:
>
> * **$M\ge3$ is never positive**, at any $r$ tested.
> * **$M=2$ is positive iff $r$ is large enough.** The centre weight of $B_2$ is (sympy)
>   $$(B_2)_0 = -(1-2r)^2+2r^2-\tfrac{r_a^2}{2} = -\tfrac{1}{2}\big(4r^2-8r+2+r_a^2\big),$$
>   non-negative iff $r\ge 1-\tfrac12\sqrt{2-r_a^2}\to 1-\tfrac{1}{\sqrt2}=0.292893$ as $r_a\to0$.
>   Bisection gives $r^\*=0.293048$ (the gap is the $r_a$ term).
>   At $r=0.45$ the coarse law is the positive mass-1 three-term scheme
>   $U^{n+1}_j=0.2\,U^n_j+B_2*U^{n-1}$, $B_2\approx(0.2235,0.3940,0.1825)$.
>
> Since FTCS itself needs $r\le\tfrac12$, the safe window is $r\in[0.293,\,0.5]$: coarsening by 2
> preserves the maximum principle **only when the fine scheme is run near its stability limit**,
> so that one fine step already mixes a full cell. **An earlier draft of this file claimed
> $M=2$ was positive at every $r$; that is false.**

> **Observation 4 (RG-consistent coarsening: memory falls to an $M$-independent floor).** **[N]**
> Diffusive scaling pairs $\Delta x\to M\Delta x$ with $\Delta t\to M^2\Delta t$, holding $r$ fixed;
> one coarse step is $K=M^2$ fine steps. Then:
>
> 1. The memory ratio collapses from $\approx1.9$ to $\approx10^{-4}$ but **plateaus**, at a floor
>    that does not depend on $M$:
>    $$\frac{\|B_2\|_1}{\|B_1\|_1}\;\longrightarrow\;e^{-2\pi^2 r}.$$
>    *Mechanism:* at the **coarse Nyquist** mode the two nearest aliases sit at $\theta=\pm\pi/M$,
>    exactly degenerate, so $e_2/e_1\sim|g(\pi/M)|^{M^2}\to e^{-2r\pi^2}$ — the $M$'s cancel.
>    Verified for $r=0.15\dots0.45$, $M=8\dots24$, agreeing to a few percent over three decades:
>
>    | $r$ | $e^{-2\pi^2r}$ | $M{=}8$ | $M{=}12$ | $M{=}20$ | $M{=}24$ |
>    |---|---|---|---|---|---|
>    | 0.45 | $1.388\times10^{-4}$ | $1.160\times10^{-4}$ | $1.299\times10^{-4}$ | $1.373\times10^{-4}$ | $1.216\times10^{-4}$ |
>    | 0.35 | $9.990\times10^{-4}$ | $9.158\times10^{-4}$ | $9.683\times10^{-4}$ | $9.958\times10^{-4}$ | $9.301\times10^{-4}$ |
>    | 0.25 | $7.192\times10^{-3}$ | $7.010\times10^{-3}$ | $7.136\times10^{-3}$ | $7.202\times10^{-3}$ | $7.053\times10^{-3}$ |
>    | 0.15 | $5.177\times10^{-2}$ | $5.189\times10^{-2}$ | $5.201\times10^{-2}$ | $5.201\times10^{-2}$ | $5.198\times10^{-2}$ |
>
>    *Interpretation:* the residual memory is exactly the part of the highest coarse mode that
>    one coarse step has not yet mixed away.
> 2. **The Markov term $B_1$ is non-negative for every $M$, under both coarsenings.** Under
>    diffusive scaling its mass is 1 to within the floor (for $M\ge5$), so a positive memoryless
>    coarse scheme exists with relative error $\sim10^{-4}$. Under space-only coarsening $B_1$ is
>    also positive but has mass $M(1-2r)$, exceeding 1 for $M>1/(1-2r)=10$, so the memory must
>    then carry negative mass.
> 3. **The full memory scheme is still not positive** under diffusive coarsening (min entry
>    $\approx-3\times10^{-5}$). The tension does not vanish; it becomes numerically negligible,
>    bounded by the same $e^{-2\pi^2r}$ floor. **An earlier draft claimed positivity was restored
>    here; it is not.**

**Summary of §6.** Memory is the price of coarse-graining space *faster than the dynamics mixes*.
Coarse-grain space and time together at the diffusive ratio and the price falls to $e^{-2\pi^2 r}$ —
small, $M$-independent, and set entirely by the per-step mixing number $r$.

---

## 6.5 The trilemma: positivity, locality, accuracy  (§Task 5)

**Pawula's theorem** (R. F. Pawula, *Phys. Rev.* **162**, 186, 1967). For a Markov process with
Kramers–Moyal expansion $\partial_tP=\sum_{n\ge1}(-\partial_x)^n\!\left[D^{(n)}P\right]$,
non-negativity of $P$ forces the generalised Cauchy–Schwarz inequality
$\left[D^{(m+n)}\right]^2\le D^{(2m)}D^{(2n)}$. Hence if any **even** coefficient $D^{(2j)}$ with
$j\ge2$ vanishes, then $D^{(n)}=0$ for all $n\ge3$. The expansion therefore either stops at
$n=2$ (Fokker–Planck) or has infinitely many non-zero terms: a truncation at any finite order
$N$ with $3\le N<\infty$ is inconsistent with a non-negative density. **[classical]**

> **Proposition 7 (what Pawula does *not* say).** **[N]**
> "A local positive propagator is necessarily second-order" is **false**. Pawula constrains the
> stencil's own cumulant sequence — it may not *terminate* beyond order 2. Accuracy requires only
> *matching* finitely many moments of the target kernel, which is a different condition.
> Demonstration: at $r=0.45$ the 5-point stencil exact on $\Theta_0..\Theta_4$ has
> $\min w=0.0577>0$ and $\|w\|_1=1$ — local, positive, fourth order in space. Its own cumulants
> $\kappa_1..\kappa_4$ match the kernel while $\kappa_5,\dots,\kappa_{10}$ are all non-zero,
> exactly as Pawula requires, and Cauchy–Schwarz holds with 0 violations over $1\le m,n\le4$.

> **Observation 5 (the real order cap is set by $r$).** **[N]**
> Largest $m$ for which the $(2m+1)$-point $\Theta_0..\Theta_{2m}$-exact stencil stays non-negative:
>
> | $r$ | 0.45 | 0.40 | 0.35 | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
> |---|---|---|---|---|---|---|---|---|---|
> | $\sigma=\sqrt{2r}$ (cells) | 0.949 | 0.894 | 0.837 | 0.775 | 0.707 | 0.632 | 0.548 | 0.447 | 0.316 |
> | max spatial order, $w\ge0$ | **10** | **10** | 8 | 6 | 6 | 6 | 2 | 2 | 2 |
>
> The cap tracks the kernel width: when $\sigma\ll1$ cell the exact kernel is nearly a delta and
> its high moments cannot be reproduced by a non-negative lattice measure. Note this is the *same*
> condition as Observation 3 ($M=2$ coarsening is positive iff $r\ge1-1/\sqrt2$): **positivity
> needs enough mixing per step**, in both settings.
> The classical result that *does* cap order is **Godunov's theorem** — a linear,
> constant-coefficient, monotonicity-preserving scheme for the *advection* equation is at most
> first-order accurate. That is hyperbolic; here $r$ is fixed with real diffusion present, the
> problem is parabolic, and Godunov's bound does not bite.

### The trade curve

Exact formulation, no trajectory fitting. A compact coarse law
$U^{n+1}=\sum_{j=1}^{p+1}B_jU^{n+1-j}$ with $B_j$ of half-width $s$ reproduces the fine dynamics
iff, for every coarse mode $q$ and every alias $l$,
$$\sum_{j=1}^{p+1}\widehat{B_j}(q)\,g_l^{\,p+1-j}=g_l^{\,p+1},$$
$M N_c$ complex equations in $(p+1)(2s+1)$ real unknowns shared across all $q$. The $(q,l)=(0,0)$
row is $\sum_{j,k}(B_j)_k=1$, so mass conservation is already imposed. Solved twice: free
($R_{\rm free}$, `lstsq`) and non-negative ($R_{\rm pos}$, exact active-set NNLS).
Initial-condition independent.

> **Observation 6 (memory buys back positivity, geometrically).** **[N]**
> At fixed minimal locality $s=1$ (3 coarse weights per lag):
>
> | $M$ | decay per extra lag | extra lags per decade of accuracy |
> |---|---|---|
> | 2 | exact at $p=1$ | — (positivity is free) |
> | 3 | $0.0966^p$ | 0.99 |
> | 4 | $0.4414^p$ | 2.82 |
> | 6 | $0.7370^p$ | 7.54 |
>
> Every solution has weights $\ge0$ with total mass $S\le1$ (mass is met only to within the
> residual: $S=1$ to $10^{-8}$ for $M=2,3,4$ at $p=20$; $S=0.99995$ for $M=6$ where
> $R_{\rm pos}=1.1\times10^{-3}$). Hence each satisfies a discrete maximum principle
> $\max|U^{n+1}|\le S\max_j\max|U^{n+1-j}|$: unconditionally $\ell_\infty$-stable, no transient.
> The exchange rate degrades sharply with the coarsening factor.

> **Observation 7 (for small $M$ the price is zero).** **[N]**
> An **exact** non-negative compact coarse law exists:
>
> | $M$ | $s$ | $p$ | weights | $R_{\rm pos}$ | $\min w$ | mass |
> |---|---|---|---|---|---|---|
> | 2 | 1 | 1 | 6 | $3.7\times10^{-16}$ | $2.6\times10^{-17}$ | $1.000000000000$ |
> | 3 | 2 | 5 | 30 | $5.9\times10^{-16}$ | $0$ | $1.000000000000$ |
>
> Locality and memory are interchangeable: at $M=3$, widening from 3 to 5 weights per lag cuts
> the memory needed for exactness from $p\approx15$ to $p=5$.
> Both verified outside the system they were fitted in: one-step relative error
> $4.7\times10^{-16}$ / $6.7\times10^{-16}$ over 5 random trajectories, and a 400-step *unforced*
> self-rollout staying at $3.2\times10^{-15}$ / $4.4\times10^{-16}$.
> The $M=2$ law is $B_1=0.2\,I$, $B_2=(0.2235,0.3940,0.1826)$ — identical to all digits with the
> law derived independently in Observation 3 from Cayley–Hamilton on the alias block.
> The $M=3$ law has $B_1=B_2=0$ **exactly**: a **pure-delay** scheme starting at lag 2. Buying
> back positivity with memory is not adding a correction to a Markov law; it is moving the whole
> law backwards in time.
>
> For $M\ge4$ no exact non-negative compact law was found: at $M=4$, $s=2\to4$ moves the $p=17$
> residual only from $1.0\times10^{-8}$ to $1.5\times10^{-9}$, with no snap to zero — qualitatively
> unlike $M=3$. Searched $s\le4$, $p\le17$: a failure to find, **not** a proof of non-existence.

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
| 17 | Coarse positivity: $M\ge3$ never; $M=2$ iff $r\ge1-1/\sqrt2=0.2929$ (analytic + bisection) | **[P,N]** |
| 24 | The $m^2/2r$ law is the RKC stability scaling (classical); verified independently | **[P,N]** |
| 25 | RKC frontier stencil $=\tfrac12(\delta_{-m}+\delta_{+m})$: consistent, positive, stable, band error $\approx1$ | **[P,N]** |
| 26 | Off-centring removes the advective branch entirely: $k\le(m^2-\mu'^2)/2r$ | **[P]**, T2-confirmed |
| 27 | Maxent at $J{=}3$ *is* the moment-matched discrete Gaussian (one construction) | **[P,N]**, T2-confirmed |
| 28 | Frontier scheme $=$ coarse FTCS at $r_c{=}1/2$ on $m$ decoupled sublattices; coarse law exactly Markov | **[P,N]** |
| 19 | Pawula constrains the stencil's own cumulants (may not terminate), not its accuracy order | **[N]** |
| 20 | Order cap on positive local stencils is set by $r$: order 10 at $r=0.45$, order 2 at $r\le0.15$ | **[N]** |
| 21 | Trade curve: memory buys back positivity geometrically, rate degrading with $M$ | **[N]** |
| 22 | Exact non-negative compact coarse laws at $M=2$ ($s{=}1,p{=}1$) and $M=3$ ($s{=}2,p{=}5$) | **[N]** |
| 23 | Leading Edgeworth term is skewness ($L^{-1/2}$) unless $c=0$, then kurtosis ($L^{-1}$) | **[P,N]** |
| 18 | Under diffusive $M^2$ coarsening memory falls to the $M$-independent floor $e^{-2\pi^2r}$; $B_1\ge0$ always, full scheme still not positive | **[N]** |
