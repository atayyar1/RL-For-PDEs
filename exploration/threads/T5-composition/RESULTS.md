# T5 — Results: numbers and figures

Reproduce with `python3 task{1,2,3,4}_*.py`; full console output is saved in
`out_task1.txt … out_task4.txt`. Figures are 150 dpi in `figures/`.
All scripts pin BLAS to one thread — on a loaded machine, thread oversubscription made a
120×120 matvec loop take 83.8 s instead of 0.003 s.

**Operating point.** $\alpha=0.1$, $c=1$, $\Delta x=1.0101\times10^{-2}$, $\Delta t=4.59137\times10^{-4}$,
$r=0.45$, $r_a=0.0454545$, cell Péclet $c\Delta x/\alpha=0.1010$.
$w_{\rm FTCS}=(0.472727,\,0.1,\,0.427273)$, $\|w\|_1=1$, $\min w=0.1$.
Per step in cell units: $\mu_1=-0.0454545$, $s_2=2r-r_a^2=0.897934$, $\kappa_3=+7.708490\times10^{-2}$, $\kappa_4=-1.515976$ (skewness $\gamma_1=+0.0906$, excess kurtosis $\gamma_2=-1.880$).

---

## Task 1 — composition law and the elementary theorems (`out_task1.txt`)

| check | result |
|---|---|
| composite = $\sum_{ij}w_iv_{ij}\delta_{\delta_i+\epsilon_{ij}}$ vs brute force | err $0.00$ |
| uniform sub-agents $\equiv$ `np.convolve` | err $0.00$ |
| binomial moment law, homogeneous, all $p\le4,q\le3$ | max rel. err $6.0\times10^{-15}$ |
| binomial moment law, heterogeneous | max rel. err $4.6\times10^{-16}$ |
| sympy: $\widehat W_{\rm comp}-\widehat W\widehat V$ | $0$ |
| sympy: $\kappa_2(w)+\kappa_2(v)-\kappa_2(w*v)$ | $0$ |
| PDE residual of $\Theta_n$, $n=0..4$ | $0$ (symbolic) |

**Theorem 1 (closure).** 200 random heterogeneous compositions per $r$, class $\Theta_{\le r}$:

| $r$ | composite defect at degree $\le r$ | at degree $r{+}1$ |
|---|---|---|
| 1 | $4.6\times10^{-16}$ | $6.9\times10^{-1}$ |
| 2 | $3.4\times10^{-15}$ | $5.2\times10^{-1}$ |
| 3 | $1.1\times10^{-14}$ | $5.1\times10^{-1}$ |

Order is inherited and takes the **min**. The obstruction that forces the class to be $\Theta_r$ and
not $\Pi_r$: best achievable $\Pi_1$-exactness residual is $0.707$ at one time level, $8.0\times10^{-16}$
at two.

**$\Theta_n$-defect $=\varepsilon_n$** for FTCS: $n=2$: both $-2.108066\times10^{-7}$;
$n=3$: both $+7.944448\times10^{-8}$; $n=4$: $-1.578149$ vs $-1.578162\times10^{-8}$ (agree to $8\times10^{-6}$
relative, as expected once lower defects are non-zero).

**Theorem 1′ (defects add exactly).**

| $L$ | 1 | 4 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| $\varepsilon_2(w^{*L})$ | $-2.108\times10^{-7}$ | $-8.432\times10^{-7}$ | $-6.746\times10^{-6}$ | $-2.698\times10^{-5}$ | $-1.079\times10^{-4}$ |
| rel. err vs $L\varepsilon_2$ | 0 | $1.3\times10^{-13}$ | $6.4\times10^{-14}$ | $1.3\times10^{-13}$ | $3.2\times10^{-13}$ |
| $\alpha_{\rm eff}$ | 0.09977043 | 0.09977043 | 0.09977043 | 0.09977043 | 0.09977043 |

$\alpha-\tfrac{c^2\Delta t}{2}=0.09977043$; $\varepsilon_2=-c^2\Delta t^2$ exactly.

**Theorem 2 (submultiplicativity).** $0/4000$ violations; ratio lhs/rhs median $0.617$, max $0.99976$.
$2000/2000$ positive$\times$positive compositions positive with mass 1.
FTCS: $\|w^{*L}\|_1=1.000000000000$ for $L=1..512$ (min weight $8.3\times10^{-190}$ at $L=512$).
$w=(-0.25,1.5,-0.25)$: $\|w^{*L}\|_1=2^L$ exactly, $L=1..16$.

**Heterogeneous defect amplification** (Prop. 5), predicted vs measured agree to round-off:

| $\|w\|_1$ | 1.343 | 1.5 | 3.0 | 8.0 |
|---|---|---|---|---|
| $D_2({\rm comp})$ measured | $6.3218\times10^{-8}$ | $9.5285\times10^{-8}$ | $4.0138\times10^{-7}$ | $1.4217\times10^{-6}$ |
| $D_2(w)+\sum_iw_id_i$ | $6.3218\times10^{-8}$ | $9.5285\times10^{-8}$ | $4.0138\times10^{-7}$ | $1.4217\times10^{-6}$ |
| bound $\|w\|_1\max_i\lvert d_i\rvert$ | $5.571\times10^{-7}$ | $6.223\times10^{-7}$ | $1.245\times10^{-6}$ | $3.319\times10^{-6}$ |

**Consistent but ill-posed.** FTCS pure advection: $\kappa_2=-2.11\times10^{-7}$, $\|w\|_1=1.0455$,
$\max|g|=1.001033$, $\max|g|^{1089}=3.08$. Central advection + $r=5\times10^{-4}$:
$\kappa_2=-1.09\times10^{-7}$, $\max|g|^{1089}=1.35$. Both first-order consistent.

**The trade-off we did not find.** The unique 5-point $\Theta_0..\Theta_4$-exact stencil at $r=0.45$ is
$w_5=(0.070576,0.204350,0.482708,0.184684,0.057682)$ — **positive**, $\|w\|_1=1$, $\min w=0.0577$.
It beats FTCS at every $L$: band error $7.08\times10^{-4}$ vs $2.14\times10^{-2}$ at $L=1$, and
$1.72\times10^{-7}$ vs $8.80\times10^{-4}$ at $L=1089$ (ratio $0.0002$). **No crossover.**
$w_5$ only goes indefinite below $r\approx0.17$ ($\min w=-0.0014$ at $r=0.15$, $\|w\|_1$ peaking at
$1.0139$ near $r=0.08$), and even there $\max|g|=1$ exactly.

---

## Task 2 — composition as RG flow (`out_task2.txt`, `figures/task2_rg_flow.png`)

**Simplex preserved.** For $L=1\dots4096$: $\min w\ge0$ always, $|\sum w-1|\le4\times10^{-16}$.

**Diffusive collapse.**

| $L$ | $\sigma_L$ (cells) | LOCAL | LOCAL$\cdot\sqrt L$ | TV | TV$\cdot\sqrt L$ | EDGE | EDGE$\cdot L$ |
|---|---|---|---|---|---|---|---|
| 16 | 3.79 | $1.537\times10^{-2}$ | 0.0615 | $1.463\times10^{-2}$ | 0.0585 | $1.569\times10^{-2}$ | 0.251 |
| 256 | 15.16 | $6.302\times10^{-4}$ | 0.01008 | $7.860\times10^{-4}$ | 0.01258 | $3.683\times10^{-4}$ | 0.0943 |
| 1024 | 30.32 | $2.740\times10^{-4}$ | 0.00877 | $3.655\times10^{-4}$ | 0.01170 | $9.218\times10^{-5}$ | 0.0944 |
| 4096 | 60.65 | $1.316\times10^{-4}$ | 0.00842 | $1.793\times10^{-4}$ | 0.01147 | $2.306\times10^{-5}$ | 0.0944 |

Fitted on $L\ge256$: LOCAL $\sim L^{-0.563}$, TV $\sim L^{-0.532}$, EDGE $\sim L^{-0.9995}$
(predicted $-\tfrac12,-\tfrac12,-1$). **Edgeworth constant check:** predicted plateau
$|\kappa_3|/(6s_2^{3/2})\max|He_3\phi|=0.008313$, measured $0.00842$ — **1.3 %**.
*Caveat:* fitting from $L\ge16$ gives $L^{-0.779}$, a pre-asymptotic transient.

**Formal vs effective support.** Formal half-width $=L$; $\sigma_L=\sqrt{s_2L}$;
$\rho(L)=L/\sigma_L=\sqrt{L/s_2}$. At $L=4096$: $4096$ vs $60.6$ cells, $\rho=67.5$.

**Tails (correcting a natural guess).** Ratio of composite tail mass to Gaussian tail mass:

| $L$ | $\rho$ | $k{=}2$ | $k{=}4$ | $k{=}6$ | $k{=}8$ | $k{=}12$ | $k{=}20$ |
|---|---|---|---|---|---|---|---|
| 64 | 8.4 | 0.889 | 0.668 | 0.167 | $1.0\times10^{-3}$ | 0 | 0 |
| 1024 | 33.8 | 0.972 | 1.02 | 0.900 | 0.668 | 0.293 | $3.9\times10^{-5}$ |
| 4096 | 67.5 | 0.992 | 1.00 | 0.967 | 0.959 | 0.884 | 0.164 |

The tails are **Gaussian, not sub-Gaussian**; they collapse only as $k\to\rho(L)$.
So $\rho(L)$ is simultaneously the compression ratio and the range of validity of the fixed point.

**Positivity restoration.** 7-point $\Theta_0..\Theta_6$-exact stencil at $r=0.05$:
$\|w\|_1=1.020503$, $\min w=-5.64\times10^{-3}$, $\max|g|=1.0000000000$.

| $L$ | 1 | 8 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| $\|w^{*L}\|_1$ | 1.020503 | 1.000745 | 1.000000000 | 1.000000000 | 1.000000000 |
| Thm-2 bound | 1.021 | 1.176 | 1.915 | 3.665 | 180.5 |
| negative mass | $1.03\times10^{-2}$ | $3.72\times10^{-4}$ | $9.72\times10^{-13}$ | $4.75\times10^{-25}$ | $1.58\times10^{-90}$ |
| innermost neg. $\lvert k\rvert/\sigma_L$ | 6.31 | 4.43 | 8.85 | 11.99 | 25.24 |

Regime (c): FTCS pure advection $\|w^{*L}\|_1 = 1.70,\,5.50,\,18.4,\,418$ at $L=16,256,1089,4096$
against a Theorem-2 bound of $2.0,\,8.8\times10^4,\,1.1\times10^{21},\,1.2\times10^{79}$.

---

## Task 3 — wide stencils as compressed composites (`out_task3.txt`, `figures/task3_compression.png`)

**Feasibility boundary — corrected.** The three consistency conditions fix the raw second moment to
$M_2=2\alpha k\Delta t+(ck\Delta t)^2$ (variance **plus mean squared**); $M_2\le m^2\Delta x^2$ is both
necessary and sufficient, giving $k_{\max}(m)=(-r+\sqrt{r^2+r_a^2m^2})/r_a^2$.

| $m$ | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 32 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| LP $k_{\max}$ | 1 | 4 | 9 | 26 | 62 | 124 | 273 | 519 | 903 |
| corrected formula | 1 | 4 | 9 | 26 | 62 | 124 | 273 | 519 | 903 |
| old $m^2/(2r)$ | 1.11 | 4.44 | 10.00 | 27.78 | 71.11 | 160.00 | 444.44 | 1137.78 | 2777.78 |
| old / LP | 1.111 | 1.111 | 1.111 | 1.068 | 1.147 | **1.290** | **1.628** | **2.192** | **3.076** |
| branch | diff | diff | diff | diff | diff | adv | adv | adv | adv |

The corrected formula matches the LP **exactly at every $m$**. The familiar $m^2/(2r)$ is only the
diffusion-limited branch; the crossover is at $m\sim r/r_a=\alpha/(c\Delta x)=9.90$ cells, beyond which
the bound is advection-limited ($k\to m/r_a$) and $m^2/(2r)$ overestimates by an unboundedly growing
factor.

**Equivalence.** The $L$-fold composite has raw second moment $Ls_2+(Lr_a)^2$, i.e. half-width about
the target $m=\sqrt{Ls_2+L^2r_a^2}$. Feeding that into $k_{\max}$ returns $L$:

| $L$ | 4 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| $m$ | 1.9039 | 3.8595 | 8.1198 | 19.1122 | 55.5515 | 195.8101 |
| $k_{\max}(m)$ | 3.9910 | 15.9658 | 63.8864 | 255.7298 | 1023.5876 | 4095.5252 |
| rel. err | $2.25\times10^{-3}$ | $2.14\times10^{-3}$ | $1.77\times10^{-3}$ | $1.06\times10^{-3}$ | $4.03\times10^{-4}$ | $1.16\times10^{-4}$ |

Residual $=r_a^2/(2r+2r_a^2L)$ exactly.

**What compression loses** (band error vs exact propagator, $|\theta|\le\pi/4$):

$L=1024$, $\sigma_L=30.32$, exact composite (2049 pts) $=8.830\times10^{-4}$:

| $m/\sigma_L$ | 1.02 | 2.01 | 3.00 | 4.02 | 6.00 |
|---|---|---|---|---|---|
| truncate+renormalise | $3.77\times10^{-1}$ | $6.19\times10^{-2}$ | $4.74\times10^{-3}$ | $8.57\times10^{-4}$ | $8.83\times10^{-4}$ |
| $\ell_1$ dist. to composite | $2.99\times10^{-1}$ | $4.25\times10^{-2}$ | $2.54\times10^{-3}$ | $5.27\times10^{-5}$ | $1.63\times10^{-9}$ |

Maximum-entropy positive stencil at $m=4\sigma_L$, sweeping matched moments $J$:

| $J$ | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| $L=1024$ band error | $8.03\times10^{-1}$ | $3.81\times10^{-4}$ | $3.82\times10^{-4}$ | $1.08\times10^{-3}$ | $2.21\times10^{-3}$ | $1.06\times10^{-2}$ |

$\min w>0$ and $\|w\|_1=1.000000$ at every $J$. Two surprises: $J=3$ is $\approx4\times$ **better** than
the composite it compresses (it targets the exact semigroup, not FTCS's accumulated $L\varepsilon_2$),
and $J\ge6$ is monotonically **worse** (forcing the untruncated Gaussian's high moments onto a
$4\sigma$ support is inconsistent).

**The naive direct solve is unusable:** $\mathrm{cond}=4.0\times10^{4},\,2.3\times10^{8},\,5.2\times10^{11},\,5.3\times10^{17}$
at $m=2,4,6,10$, giving $\|w\|_1$ up to $1.4\times10^{14}$.

**Cost ratio (composed / wide), $m=\lceil4\sigma_L\rceil$:**

| $L$ | 4 | 64 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| one output point | 2.8 | 195.0 | 12840 | $1.03\times10^5$ | $8.28\times10^5$ |
| full grid | 0.71 | 3.05 | 12.54 | 25.23 | 50.52 |

Fitted $L^{1.506}$ and $L^{0.506}$ (predicted $3/2$, $1/2$). The grid ratio is $<1$ below $L\approx8$.

---

## Task 4 — coarse-graining and memory (`out_task4.txt`, `figures/task4_memory.png`)

Periodic $N=120$; $M$ restricted to divisors of $N$ (aliasing is exact only then).

**Exact finite memory.** Residual of the order-$M$ recurrence with the analytic $B_p$:

| $M$ | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| $M{-}1$ lags | $5.9\times10^{-16}$ | $5.5\times10^{-16}$ | $6.9\times10^{-16}$ | $1.0\times10^{-15}$ | $1.5\times10^{-15}$ |
| $M{-}2$ lags | $8.4\times10^{-1}$ | $2.4\times10^{-1}$ | $2.8\times10^{-2}$ | $5.4\times10^{-3}$ | $3.1\times10^{-4}$ |
| Markov only | $8.4\times10^{-1}$ | $8.7\times10^{-1}$ | $4.9\times10^{-1}$ | $3.0\times10^{-1}$ | $3.3\times10^{-1}$ |

Blind least-squares fit reproduces this: residual hits round-off at $p=M$ and not one lag before
(e.g. $M=6$: $3.2\times10^{-1},\,7.6\times10^{-2},\,2.7\times10^{-2},\,6.9\times10^{-3},\,1.3\times10^{-3},\,5.5\times10^{-10}$
for $p=1..6$).

**No decay (space-only, $K=1$).** $\|B_p\|_1$ at $M=8$: $0.800,\,1.336,\,0.914,\,0.581,\,0.295,\,0.085$.
$\|B_2\|_1>\|B_1\|_1$ for $M=2..20$. At $M=20$ the largest is $\|B_5\|_1=7.96$ vs $\|B_1\|_1=2.00$.
$\sum_p\|B_p\|_1$: 1.00, 1.12, 1.46, 1.93, 2.52, 4.04, 6.07, 8.71, 13.99, 30.60, 62.15
for $M=2,3,4,5,6,8,10,12,15,20,24$ — fitted $\sim e^{0.189M}$.

**Positivity (space-only).**

| $M$ \ $r$ | 0.45 | 0.35 | 0.30 | 0.25 | 0.15 | 0.05 |
|---|---|---|---|---|---|---|
| 2 | **POS** (1.00) | **POS** (1.00) | **POS** (1.00) | neg (1.25) | neg (1.89) | neg (2.61) |
| 3 | neg (1.12) | neg (1.17) | neg (1.51) | neg (2.13) | neg (3.81) | neg (5.85) |
| 4 | neg (1.46) | neg (1.47) | neg (2.26) | neg (3.52) | neg (7.09) | neg (12.00) |
| 8 | neg (4.04) | neg (4.71) | neg (9.16) | neg (19.33) | neg (64.51) | neg (167.90) |

$M=2$ threshold: bisection $r^\*=0.293048$; analytic $1-\tfrac12\sqrt{2-r_a^2}\to1-1/\sqrt2=0.292893$.
Sympy: $(B_2)_0=-(4r^2-8r+2+r_a^2)/2$, roots $1\pm\tfrac{1}{\sqrt2}$.
At $r=0.45$: $U^{n+1}_j=0.2U^n_j+B_2*U^{n-1}$ with $B_2\approx(0.2235,0.3940,0.1825)$, mass 1.

**Diffusive coarsening ($K=M^2$).**

| $M$ | $\|B_1\|_1$ | $\min B_1$ | $\sum B_1$ | $\|B_2\|/\|B_1\|$ | $\sum_p\|B_p\|_1-1$ | $\min$ all $B_p$ |
|---|---|---|---|---|---|---|
| 2 | 1.409600 | $-5.7\times10^{-17}$ | 1.409600 | $2.91\times10^{-1}$ | $8.19\times10^{-1}$ | $-1.10\times10^{-1}$ |
| 5 | 0.999991 | $-1.5\times10^{-16}$ | 0.999991 | $3.45\times10^{-4}$ | $3.36\times10^{-4}$ | $-1.13\times10^{-4}$ |
| 12 | 1.000000 | $+3.9\times10^{-6}$ | 1.000000 | $1.30\times10^{-4}$ | $1.30\times10^{-4}$ | $-3.12\times10^{-5}$ |
| 24 | 1.000000 | $+4.1\times10^{-2}$ | 1.000000 | $1.22\times10^{-4}$ | $1.22\times10^{-4}$ | $-4.03\times10^{-5}$ |

**Memory floor $=e^{-2\pi^2 r}$, independent of $M$** (see THEORY.md §6 table; agreement to a few
percent for $r=0.15..0.45$, $M=8..24$). $B_1\ge0$ for every $M$ under both coarsenings; the full
memory scheme remains slightly non-positive under diffusive coarsening ($\approx-3\times10^{-5}$).

---

## Task 3 addendum — RKC anchor and the frontier (`out_task3.txt` §3E)

$k\le m^2/(2r)$ **is** the first-order Runge–Kutta–Chebyshev stability scaling
($P_m(z)=T_m(1+z/m^2)$, $|z|\le2m^2$, $z\in[-4rk,0]$). Verified independently here; the companion
thread showed the frontier scheme *is* first-order RKC with $s=m$ stages to $2\times10^{-16}$.
**Classical, not a new result.**

At the frontier $1+z/m^2=(S_{+1}+S_{-1})/2$ and $T_m(\cos\theta)=\cos(m\theta)$, so the stencil is
exactly $\tfrac12(\delta_{-m}+\delta_{+m})$ (machine zero at $m=3,6,12$):

| $m$ | support | weights | $\sum w$ | $M_2/(2\alpha\tau)$ | band error |
|---|---|---|---|---|---|
| 3 | $\{-3,3\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 0.766 |
| 6 | $\{-6,6\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 1.004 |
| 12 | $\{-12,12\}$ | $(0.5,0.5)$ | 1.0000000000 | 1.0000000000 | 1.000 |

Consistent, positive, stable — band error $\approx1$. **Consistency + positivity + stability do not
imply accuracy.** Away from the frontier RKC is not positive:

| $m$ | $k/k_{\max}$ | 1.00 | 0.90 | 0.70 | 0.50 |
|---|---|---|---|---|---|
| 8 | $\min w$ | $0$ | $-1.115\times10^{-1}$ | $-1.610\times10^{-1}$ | $-2.188\times10^{-1}$ |
| 8 | $\|w\|_1$ | 1.000000 | 2.377675 | 2.596448 | 2.500000 |
| 16 | $\min w$ | $0$ | $-1.830\times10^{-1}$ | $-1.297\times10^{-1}$ | $-1.736\times10^{-1}$ |

**Why the frontier carries no information.** $\tfrac12(\delta_{-m}+\delta_{+m})$ couples $j$ only to
$j\pm m$, so the lattice splits into $m$ residue classes that never exchange information. On one
class the stencil is $(\tfrac12,0,\tfrac12)$ = FTCS for pure diffusion at
$r_c=\alpha\tau/(m\Delta x)^2=0.500000000000$ exactly — its own stability limit. Coarsening that
operator by $M=m$ gives $g_l=\cos(2\pi mq/N)$ independent of $l$ (measured spread
$\le5.6\times10^{-15}$ for $M=m=3,4,6$), so the minimal polynomial has degree 1 and the coarse law is
**exactly Markov, zero memory** — against the $M-1$ lags FTCS needs. The frontier is where
coarse-graining is free, and that is the same degeneracy that makes it useless.

**Maxent $\equiv$ moment-matched Gaussian.** At $J=3$, $\log w$ is quadratic in $j$ to
$2.5\times10^{-14}$ (it is $e^{\lambda_1j+\lambda_2j^2}$ by construction). T2 verified agreement with
their independently root-found kernel to $2.8\times10^{-17}$–$1.0\times10^{-16}$ at
$(m,k)=(8,4),(12,10),(20,25),(30,50),(40,100)$, band errors identical to every digit. One
construction, two derivations.

**Off-centring removes the advective branch.****Off-centring removes the advective branch.** Support on $\{j_0\pm m\}$, $j_0=\mathrm{round}(-r_ak)$:
$k\le(m^2-\mu'^2)/(2r)\to m^2/(2r)$, $|\mu'|\le\tfrac12$. T2's LP: 27, 71, 159, 444, 1137, 2777 vs
$m^2/(2r)=$ 28, 71, 160, 444, 1138, 2778. Gain over centred $3.1\times$ at $m=50$. Two thresholds:
materially wrong at $m^\*=1/\mathrm{Pe}_{\rm cell}=9.90$, branch crossing at $2m^\*=19.80$.

---

## Task 5 — positivity / locality / accuracy (`out_task5.txt`, `figures/task5_trilemma.png`)

Exact Fourier formulation (no trajectory fitting), so initial-condition independent:
a compact coarse law of half-width $s$ and memory depth $p$ is exact iff
$\sum_j\widehat{B_j}(q)g_l^{p+1-j}=g_l^{p+1}$ for all $MN_c$ (mode, alias) pairs.
$R_{\rm free}$ = best unconstrained (`lstsq`), $R_{\rm pos}$ = best with all weights $\ge0$
(exact active-set NNLS). The $(q,l)=(0,0)$ row *is* mass conservation, so it is already imposed.

**Sanity.** $R_{\rm free}$ hits round-off exactly at $p=M-1$ at full support (Thm 3), and
$R_{\rm pos}$ does not for $M\ge3$ — the exact coarse law is not representable non-negatively.

**Locality vs memory at $M=3$** ($R_{\rm pos}$; must be non-increasing in $s$):

| $p$ | $s{=}1$ | $s{=}2$ | $s{=}3$ | $s{=}4$ | $s{=}6$ | full |
|---|---|---|---|---|---|---|
| 2 | $6.09\times10^{-2}$ | $6.09\times10^{-2}$ | $6.09\times10^{-2}$ | $6.09\times10^{-2}$ | $6.09\times10^{-2}$ | $6.09\times10^{-2}$ |
| 4 | $7.30\times10^{-4}$ | $7.30\times10^{-4}$ | $7.30\times10^{-4}$ | $7.30\times10^{-4}$ | $7.30\times10^{-4}$ | $7.30\times10^{-4}$ |
| 5 | $5.83\times10^{-5}$ | $5.88\times10^{-16}$ | $5.77\times10^{-16}$ | $5.75\times10^{-16}$ | $5.70\times10^{-16}$ | $6.07\times10^{-16}$ |
| 7 | $5.81\times10^{-7}$ | $5.14\times10^{-16}$ | $4.92\times10^{-16}$ | $4.91\times10^{-16}$ | $4.63\times10^{-16}$ | $4.32\times10^{-16}$ |

Locality is irrelevant up to $p=4$ and then decisive: at $p=5$, going from 3 to 5 weights per
lag takes the scheme from $6\times10^{-5}$ to **machine zero**.

**The trade curve at minimal locality $s=1$** (3 coarse weights per lag), $R_{\rm pos}$:

| $M$ | $p{=}0$ | $p{=}2$ | $p{=}5$ | $p{=}10$ | $p{=}15$ | $p{=}20$ | decay/lag | lags/decade |
|---|---|---|---|---|---|---|---|---|
| 2 | $9.88\times10^{-1}$ | $3.4\times10^{-16}$ | $5.0\times10^{-16}$ | $5.0\times10^{-16}$ | $5.4\times10^{-16}$ | $4.5\times10^{-16}$ | exact at $p{=}1$ | — |
| 3 | $9.88\times10^{-1}$ | $6.09\times10^{-2}$ | $5.83\times10^{-5}$ | $5.04\times10^{-10}$ | $4.50\times10^{-15}$ | $1.17\times10^{-15}$ | $0.0966$ | **0.99** |
| 4 | $9.88\times10^{-1}$ | $2.25\times10^{-1}$ | $2.49\times10^{-2}$ | $3.74\times10^{-4}$ | $6.63\times10^{-6}$ | $1.15\times10^{-7}$ | $0.4414$ | **2.82** |
| 6 | $9.88\times10^{-1}$ | $3.01\times10^{-1}$ | $9.87\times10^{-2}$ | $2.50\times10^{-2}$ | $5.48\times10^{-3}$ | $1.14\times10^{-3}$ | $0.7370$ | **7.54** |

All weights $\ge0$ with total mass $S\le1$ ($S=1$ to $10^{-8}$ for $M=2,3,4$ at $p=20$;
$S=0.99995$ at $M=6$, where $R_{\rm pos}$ is still $1.1\times10^{-3}$), so each obeys
$\max|U^{n+1}|\le S\max_j\max|U^{n+1-j}|$ — a discrete maximum principle.

**Exact non-negative compact coarse laws** (the price is zero, not merely small):

| $M$ | $s$ | $p$ | weights | $R_{\rm pos}$ | $\min w$ | mass |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 6 | $3.67\times10^{-16}$ | $2.6\times10^{-17}$ | 1.000000000000 |
| 3 | 2 | 5 | 30 | $5.88\times10^{-16}$ | 0 | 1.000000000000 |

For $M=4$, widening the support does **not** produce the snap-to-zero that $M=3$ showed:

| $R_{\rm pos}$ | $p{=}5$ | $p{=}8$ | $p{=}11$ | $p{=}14$ | $p{=}17$ |
|---|---|---|---|---|---|
| $s{=}2$ | $2.49\times10^{-2}$ | $1.26\times10^{-3}$ | $2.49\times10^{-5}$ | $5.22\times10^{-7}$ | $1.01\times10^{-8}$ |
| $s{=}3$ | $2.49\times10^{-2}$ | $1.26\times10^{-3}$ | $2.26\times10^{-5}$ | $2.27\times10^{-7}$ | $2.26\times10^{-9}$ |
| $s{=}4$ | $2.49\times10^{-2}$ | $1.26\times10^{-3}$ | $2.26\times10^{-5}$ | $2.22\times10^{-7}$ | $1.51\times10^{-9}$ |

Going from 3 to 5 weights per lag took $M=3$ from $6\times10^{-5}$ to machine zero at $p=5$;
$s{=}2\to4$ at $M=4$ moves $p{=}17$ only from $1.0\times10^{-8}$ to $1.5\times10^{-9}$. $M=4$ looks
qualitatively different — geometric convergence rather than exact representability. **A failure
to find within $s\le4$, $p\le17$, not a proof of non-existence.**

**Independent validation** of both exact laws (evolve the fine scheme, restrict, test outside the
Fourier system they were fitted in):

| | one-step rel. err (5 trajectories × 75 steps) | 400-step self-rollout, no re-injection |
|---|---|---|
| $M=2$, $s{=}1$, $p{=}1$ | $4.74\times10^{-16}$ | max abs err $3.23\times10^{-15}$ |
| $M=3$, $s{=}2$, $p{=}5$ | $6.71\times10^{-16}$ | max abs err $4.44\times10^{-16}$ |

The $M=2$ law comes out as $B_1=0.2\,I$, $B_2=(0.2235,0.3940,0.1826)$ — reproducing, to all digits,
the law derived independently in Task 4D from Cayley–Hamilton on the alias block. Two unrelated
routes agree exactly.

The $M=3$ law has $B_1=B_2=0$ **exactly**: the non-negative representation uses no dependence on
the two most recent coarse levels at all. It is a **pure-delay** scheme starting at lag 2 — so
"buying back positivity with memory" is not adding a correction to a Markov law, it is moving the
whole law backwards in time.

**Pawula.** At $r=0.45$ the 5-point $\Theta_0..\Theta_4$-exact stencil is positive
($\min w=0.05768$, $\|w\|_1=1$): local, positive, 4th order. Its cumulants $\kappa_1..\kappa_4$
match the kernel ($\kappa_3=2.9\times10^{-22}$, $\kappa_4=3.3\times10^{-24}$) while
$\kappa_5=1.27\times10^{-11},\dots,\kappa_{10}=-2.14\times10^{-18}$ are all non-zero — exactly
as Pawula requires. Cauchy–Schwarz: 0 violations over $1\le m,n\le4$.

**Order cap on positive local stencils** (largest $m$ with the $(2m+1)$-point
$\Theta_0..\Theta_{2m}$-exact stencil still $\ge0$):

| $r$ | 0.45 | 0.40 | 0.35 | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| $\sigma=\sqrt{2r}$ cells | 0.949 | 0.894 | 0.837 | 0.775 | 0.707 | 0.632 | 0.548 | 0.447 | 0.316 |
| max spatial order | **10** | **10** | 8 | 6 | 6 | 6 | 2 | 2 | 2 |

---

## Task 2 addendum — which Edgeworth term leads (`out_task2.txt` §2B2–2B3)

| | $\kappa_3$ | $\kappa_4$ | leading term | predicted | measured | agree |
|---|---|---|---|---|---|---|
| $c=1$ (operating point) | $+7.708490\times10^{-2}$ | $-1.515976$ | skewness, $L^{-1/2}$ | 0.008313 | 0.008342 | **0.34 %** |
| $c=0$ (control) | $0$ exactly | $-1.530000$ | kurtosis, $L^{-1}$ | 0.094195 | 0.0942 | **0.01 %** |

At $c=1$, LOCAL$\cdot L$ diverges (0.114 → 1.068 over $L=64..16384$); at $c=0$,
LOCAL$\cdot\sqrt L\to0$. Moment hierarchy $dM_q/d\tau=\alpha q(q-1)M_{q-2}-cqM_{q-1}$ verified
against closed-form Gaussian moments to $8.6\times10^{-11}$ through $q=8$.

---

## Corrections made during this work

Recorded because the value of the thread depends on them being visible.

1. Theorem 1 was first tested against space-time polynomials $\Pi_r$ and "failed". The test was
   wrong twice over: the moment matrix in physical units has $\mathrm{cond}\sim10^{12}$ so `lstsq`
   silently returned a non-solution, and more fundamentally a single-time-level stencil **cannot**
   be $\Pi_r$-exact for $r\ge1$. The correct class is $\Theta_r$.
2. "The tails are sub-Gaussian" — **false**. They are Gaussian out to $k\sim\rho(L)$.
3. "The 4th-order 5-point stencil loses to positive FTCS at large $L$" — **false** at this operating
   point; that stencil is itself positive and wins everywhere.
4. "$M=2$ coarse-graining is positive at every $r$" — **false**; only for $r\ge1-1/\sqrt2$.
5. "Diffusive coarsening restores positivity" — **false**; it drives the violation to $\sim3\times10^{-5}$
   but not to zero.
6. The feasibility boundary was first stated as $k\le m^2/(2r)$, dropping the drift from the raw
   second moment. The correct sharp boundary is $k\le(-r+\sqrt{r^2+r_a^2m^2})/r_a^2$; the old form
   overestimates by $2.2\times$ at $m=32$ and $3.1\times$ at $m=50$. The numbers first written into
   this file for that table were also not the ones the script produced — they have been replaced
   with the actual output.  Same failure mode as item 8.
7. The LP "positive wide stencil" first used a $\xi^4$ objective, which selects a bimodal vertex
   $\tfrac12(\delta_{-\sigma}+\delta_{+\sigma})$ with band error $\approx1$. Replaced by maximum entropy.
8. $\kappa_3$ and $\kappa_4$ of the single step were written into this file and THEORY.md as
   $+0.122$ and $-0.800$; the script prints $+7.708490\times10^{-2}$ and $-1.515976$. Corrected.
9. The Task-3C LP vertex $\tfrac12(\delta_{-\sigma}+\delta_{+\sigma})$ was dismissed as "pathological".
   It is the RKC frontier scheme and the extremal measure of my own Prop. 6 sufficiency proof.
   Replacing it with maximum entropy was the right accuracy choice, but the characterisation was
   wrong.
10. $M=7,9$ were included in an early coarsening scan; they do not divide $N=120$, so the aliasing
   argument does not apply. An assertion now guards this.
