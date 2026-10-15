# THEORY — what survived

Notation. A target point $z^*=(x^*,t^*)$ is predicted from $n$ scattered neighbours at
offsets $(\delta x_i,\delta t_i)=(x_i-x^*,\,t_i-t^*)$, $\delta t_i<0$, as
$\hat u^*=\sum_i w_i u_i$. For $u_t+cu_x=\alpha u_{xx}$ the PDE-constrained Taylor
moment conditions are $A w=b$ with

$$A=\begin{pmatrix}1&\cdots&1\\ q_1/h&\cdots&q_n/h\\ p_1/h^2&\cdots&p_n/h^2\end{pmatrix},\qquad b=(1,0,0)^\top,$$

$$q_i=\delta x_i-c\,\delta t_i,\qquad p_i=\tfrac12\delta x_i^2+\alpha\,\delta t_i \ \ (\text{'given'}),\qquad p_i=\tfrac12 q_i^2+\alpha\,\delta t_i\ \ (\text{'exact'}).$$

---

## Claim 1 (the identity). $\ \|w\|_1=1 \iff w\ge 0$.

*Proof.* Row 1 forces $\sum_i w_i=1$ for **every** solution of $Aw=b$, whatever
solver produced it. Hence $\|w\|_1=\sum_i|w_i|\ \ge\ \big|\sum_i w_i\big|=1$, with
equality in the triangle inequality iff all $w_i$ have the same sign; the sign must be
$+$ because the sum is $+1$. $\square$

Consequences. $\min\|w\|_1 = 1$ exactly when a nonnegative stencil exists, and
$\min\|w\|_1>1$ *strictly* otherwise (verified numerically: the smallest
infeasible value seen over 4000 random stencils was $1.002373$). So the min-$L^1$ LP
is the correct graceful degradation: it returns the certificate when one exists and the
least-bad amplification when one does not.

## Claim 1b (the rows, in the form that makes the bug obvious).

Write $\xi=\delta x-c\,\delta t$ for the **characteristic offset**. In the frame moving
with the characteristic the equation is *pure diffusion*, so the two annihilation rows are
simply

$$\text{row 2}=\xi,\qquad \text{row 3}=\tfrac12\xi^2+\alpha\,\delta t .$$

That is the whole content of the corrected $C_2$. The existing code's
$\tfrac12\delta x^2+\alpha\delta t$ replaces $\tfrac12\xi^2$ by $\tfrac12\delta x^2$,
dropping $-c\,\delta x\,\delta t$ and $\tfrac12c^2\delta t^2$. The error is easy to miss
because row 2 is *already* written in $\xi$ — only row 3 was left in $\delta x$.

**Independent cross-check via propagator moments.** For a single time level
$\delta t_i=-\tau$, imposing $\sum_iw_i\big[\tfrac12\xi_i^2+\alpha\delta t_i\big]=0$
together with $\sum_iw_i\xi_i=0$ is *algebraically identical* to matching the first two
spatial moments of the exact propagator,

$$M_1=\sum_iw_i\,\delta x_i=-c\tau,\qquad M_2=\sum_iw_i\,\delta x_i^2=(c\tau)^2+2\alpha\tau,$$

i.e. the mean and second moment of the Gaussian drift–diffusion kernel. Verified
numerically to machine precision on random stencils. The `'given'` row instead yields
$M_2=2\alpha\tau$, **missing the $(c\tau)^2$ advective-drift variance**. So the moment
hierarchy of Claim 7 both subsumes and corrects the Taylor rows, and two independent
derivations (symbolic Taylor, propagator moments) agree.

## Claim 2 (error propagation). Under recursion, $\|w\|_1$ is the per-level amplification.

If each neighbour value carries error $e_i$ and the local truncation error is $\tau$,
then $e^*=\sum_i w_ie_i+\tau$, so $|e^*|\le\|w\|_1\max_i|e_i|+|\tau|$. Over $L$ levels
with uniform bound $\|w\|_1\le g$,

$$|e^{(L)}|\ \le\ \tau\,\frac{g^L-1}{g-1}\quad (g>1),\qquad |e^{(L)}|\ \le\ L\,\tau\quad (g=1).$$

Linear vs geometric. With $g=1$ the only growth is the unavoidable accumulation of
fresh truncation error; the scheme is $\ell^\infty$-stable uniformly in $L$.

## Claim 3 (discrete maximum principle). $w\ge0$ with $\sum w_i=1$ $\iff$ local DMP.

$\hat u^*=\sum_i w_iu_i$ is then a convex combination, so
$\min_i u_i\le \hat u^*\le\max_i u_i$: no new extremum is created. Propagating this
over a rollout gives a global bound $\|u^{(L)}\|_\infty\le\|u^{(0)}\|_\infty$, i.e.
monotonicity in the Godunov sense. The amplification matrix $W$ of a level has
$\|W\|_\infty=\max_i\sum_j|W_{ij}|=\max_i\|w^{(i)}\|_1$, so Claim 1 says positivity is
exactly the statement $\|W\|_\infty=1$.

## Claim 4 (the geometric characterisation — the main new result).

**Positivity is feasible $\iff$ the origin lies in the convex hull of the planar points
$(q_i,p_i)$.**

*Proof.* Row 1 restricts $w$ to the simplex $\Delta^{n-1}$; rows 2–3 then say
$\sum_i w_i(q_i,p_i)=(0,0)$. A point is a convex combination of $\{(q_i,p_i)\}$ iff it
lies in their convex hull. $\square$

Row scaling by $h$ is an invertible diagonal map on the plane and cannot change hull
membership — so the $h$ normalisation is irrelevant to positivity (it only changes
which solution `lstsq` picks).

**Corollary 4a (test).** By Gordan's theorem, $0\in\mathrm{conv}\{v_i\}$ iff no open
half-plane through the origin contains every $v_i$, i.e. iff the polar angles
$\theta_i=\operatorname{atan2}(p_i,q_i)$ have **no gap wider than $\pi$**. This is an
exact $O(n\log n)$ test: $n$ `atan2` calls and a sort. Verified against HiGHS on
$8\times10^4$ stencils with **zero** disagreements, at $1/27$ the cost scalar and
$1/2000$ the cost batched.

**Corollary 4b (two necessary conditions, $O(n)$, no transcendentals).**
Projecting the hull condition on the two axes:

* $\min_i q_i\le 0\le\max_i q_i$ — *the stencil must bracket the foot of the
  characteristic* $x^*-c|\delta t|$. This is the CFL condition in its literal,
  original form: the numerical domain of dependence must contain the physical one.
* $\min_i p_i\le 0\le\max_i p_i$ — with the 'exact' row, $p_i\ge0\iff
  |q_i|\ge\sqrt{2\alpha|\delta t_i|}$, so *the stencil must contain a neighbour
  farther than the diffusion length of its own time offset, and one nearer.*

Both are necessary, neither is sufficient, and their conjunction is not sufficient
either (measured precision $0.678$ at base rate $0.473$, recall $1$ by construction).

**Corollary 4c (reach).** For any stencil all of whose neighbours sit at depth
$\ge k\Delta t$, feasibility requires some $p_i\ge0$, hence
$\tfrac12(m\Delta x)^2\ge\alpha k\Delta t$:

$$\boxed{\,m\ \ge\ \sqrt{2\alpha k\Delta t}\,/\,\Delta x\,}$$

the half-width must be at least the **diffusion length of the jump**. Spreading
neighbours over extra time levels does not relax this — the binding neighbour is the
widest one at the *shallowest* level. Two further bounds bind at large $m$:
$k\le m\Delta x/(c\Delta t)$ (advective reach) and $k\le 2\alpha/(c^2\Delta t)$
(the origin must lie above the moment parabola).

## Claim 5 (classical schemes). Positivity reproduces the textbook CFL conditions exactly.

| scheme | geometry | positivity condition | textbook |
|---|---|---|---|
| FTCS diffusion | 3-pt, $c=0$ | $r\le\frac12$ | $r\le\frac12$ |
| upwind advection | 2-pt, $\alpha=0$, order 1 | $0\le\nu\le1$ | Courant $\le1$ |
| downwind | 2-pt | never | unstable |
| FTCS adv–diff ('given') | 3-pt | $r\le\frac12$ **and** $\mathrm{Pe}\le2$ | both |
| FTCS adv–diff ('exact') | 3-pt | $2r+\nu^2\le1$ **and** $\mathrm{Pe}\le\frac{2}{1-\nu}$ | — |

with $r=\alpha\Delta t/\Delta x^2$, $\nu=c\Delta t/\Delta x$, $\mathrm{Pe}=c\Delta x/\alpha$.
All boundaries were located by bisecting the LP and agree with the closed forms to
$\le10^{-6}$.

## Claim 6 (the honest limit). Positivity $\ne$ von Neumann stability.

Positivity is **strictly stronger** than $L^2$ stability, and the gap is not an
artefact — it is Godunov's theorem.

* With $\alpha=0$ and second-order accuracy, $p_i=\tfrac12 q_i^2\ge0$ with equality only
  for a neighbour exactly on the characteristic. A convex combination of non-negative
  numbers vanishes only if every active one vanishes, so **positivity is infeasible for
  all non-integer Courant numbers**, at any $n$. The unique 3-point solution is exactly
  Lax–Wendroff, $w=\big(\tfrac{\nu(1+\nu)}2,\,1-\nu^2,\,-\tfrac{\nu(1-\nu)}2\big)$, with
  $\|w\|_1=1+\nu(1-\nu)\le1.25$ — $L^2$-stable for $\nu\le1$, never monotone.
  This is Godunov's theorem recovered as an LP infeasibility.
* Therefore $\|w\|_1>1$ **does not imply instability**. $\|W\|_\infty>1$ while
  $\|W\|_2\le1$: the $\ell^\infty$ bound ignores the cancellation between neighbouring
  stencils that Fourier analysis exploits on a *structured, uniform* grid with *smooth*
  error modes.
* The bound $\|w\|_1^L$ is a **worst case over error sign patterns**. It is attained
  only when the incoming errors align adversarially with $\mathrm{sign}(w)$. Random,
  mesh-incoherent errors realise far less (Task 4 measures how much less).
* Diffusion is what buys positivity back: $\alpha\delta t_i<0$ pushes near neighbours'
  $p_i$ below zero. The centre point has $p<0$ iff $\Delta t<2\alpha/c^2$.

**Scope.** Positivity is the right certificate when (i) stencil geometry varies
point-to-point so no Fourier symbol exists, (ii) the recursion depth $L$ is large and
uncontrolled, and (iii) errors are not smooth grid modes — i.e. precisely the meshfree /
learned-integrator setting. On a uniform grid marching a smooth solution it is
conservative, and insisting on it forbids second-order advection entirely.

---

# Added: the order barrier (Task 6)

## Claim 7 (the moment hierarchy). Every time derivative collapses into two numbers.

Because $\partial_x$ and $L=-c\partial_x+\alpha\partial_x^2$ commute,

$$u(x+\delta x,\;t+\delta t)=e^{\delta x\partial_x}e^{\delta t L}u=\exp\!\big(q\,\partial_x+a\,\partial_x^2\big)u,\qquad q=\delta x-c\,\delta t,\quad a=\alpha\,\delta t .$$

So the coefficient of $\partial_x^s u$ for neighbour $i$ is the $s$-th Taylor coefficient of
$\exp(q\xi+a\xi^2)$:

$$C_s(q,a)=\sum_{j=0}^{\lfloor s/2\rfloor}\frac{q^{\,s-2j}a^{\,j}}{(s-2j)!\,j!},\qquad
C_0=1,\;\;C_1=q,\;\;C_2=\tfrac{q^2}2+a,\;\;C_3=\tfrac{q^3}6+qa,\;\;C_4=\tfrac{q^4}{24}+\tfrac{q^2a}2+\tfrac{a^2}2 .$$

The accuracy conditions are $\sum_i w_iC_s(q_i,a_i)=\delta_{s0}$ for $s=0..p$, giving
$|\tau|=O(h^{p+1})$. **Verified**: measured slopes of $|\tau|$ vs $\Delta x$ under parabolic
refinement were $2.00,\,4.00,\,4.02,\,\approx5$ for $p=1,2,3,4$ (the $p=2$ case gains a
free order from the symmetry of the stencil).

$C_2$ is exactly the `'exact'` row of `build_A`, confirming Claim 6's correction.

## Claim 8 (what the Jensen argument does and does not show).

**The inequality is true.** For $w\ge0$, $\sum_iw_i=1$, the $w_i$ are a probability
measure on the offsets, so

$$M_2=\sum_i w_i\,\delta t_i^2\;\ge\;\Big(\sum_i w_i\,\delta t_i\Big)^2=M_1^2>0,$$

strictly, since every $\delta t_i<0$ forces $M_1<0$. This is just
$\mathrm{Var}_w(\delta t)\ge0$, with equality iff all $\delta t_i$ coincide. Reproduced
here: 0 violations in 3000 positive stencils.

**But the consequence depends on which hierarchy you impose.**

* *Generic (non-PDE) hierarchy.* If the stencil must reproduce every bivariate monomial
  $\delta x^m\delta t^n$ through total degree $p$ — exactness for arbitrary smooth
  functions of $(x,t)$ — then $\sum_iw_i\delta t_i^2=0$ **is** a required condition and
  Jensen forbids it for $w\ge0$. **The obstruction is real here.** Reproduced: adding
  that row explicitly gives 0/600 feasible for $w\ge0$ and 600/600 for signed $w$ with
  $\min\|w\|_1=1.018$.
* *PDE-constrained hierarchy (the one that governs a solver's accuracy).* There is **no
  free-standing $u_{tt}$ term**. The PDE redistributes
  $\tfrac12\delta t^2u_{tt}=\tfrac12c^2\delta t^2u_{xx}-c\alpha\delta t^2u_{xxx}+\tfrac12\alpha^2\delta t^2u_{xxxx}$,
  and the $u_{xx}$ piece is already inside $C_2$ — which non-negative weights cancel
  routinely (FTCS does exactly that). **$\sum_iw_i\delta t_i^2=0$ is not one of the
  conditions**, so Jensen does not bind.

**Decisive counterexample.** Pure diffusion, 3-point stencil, $r=\alpha\Delta t/\Delta x^2=1/6$:

$$w=\big(\tfrac16,\ \tfrac23,\ \tfrac16\big)\ \ge 0,\qquad\text{annihilates } s=0\ldots5 .$$

This is the classical fourth-order-accurate FTCS scheme for the heat equation, and it is
**positive**. A positivity order barrier at 2 is therefore false for parabolic problems.
For 5-point advection–diffusion stencils, positivity costs **at most one** order, and at
$r\ge1/6$ it costs **nothing**.

## Claim 9 (Godunov's theorem, recovered in two lines).

**Statement (Godunov, 1959).** A linear, constant-coefficient scheme
$u_j^{n+1}=\sum_ka_ku_{j+k}^n$ for $u_t+cu_x=0$ that is monotonicity-preserving
(equivalently, for linear schemes, $a_k\ge0$ for all $k$) is at most **first-order
accurate**.

*Proof in this framework.* With $\alpha=0$ we have $a=0$, so $C_s=q^s/s!$. For even $s$,
$C_s\ge0$, vanishing only at $q=0$. Hence $\sum_iw_iC_2=\tfrac12\sum_iw_iq_i^2=0$ with
$w\ge0$ forces $w_iq_i^2=0$ for every $i$: every active neighbour must sit **exactly on
the characteristic** through $z^*$. For non-integer Courant number no grid point does. $\square$

**Verified**: max positive order $=1$ for every $\nu\in\{0.1,\dots,0.9\}$ and for 3-, 5-,
7- and 11-point stencils.

## Claim 10 (honest positioning — is any of this new?).

* **In the hyperbolic case, the Jensen/convexity argument *is* Godunov's theorem.** Same
  mechanism (a convex combination of non-negative numbers vanishes only if every active
  term vanishes), same conclusion. It is a clean re-derivation, **not a new result.**
  Saying so plainly is the useful outcome.
* Variance/Jensen arguments of this kind are standard in the positivity-preserving and
  SSP literature.
* The *specific* claim of a $\delta t^2$ floor is an artefact of the generic hierarchy and
  does not survive the PDE substitution — so it is not the mechanism behind the classical
  parabolic barriers either.
* **What is genuinely worth keeping** is not the barrier but its *price*. Classical order
  barriers are yes/no theorems. Here, when positivity is infeasible, $\min\|w\|_1>1$ is a
  **continuous, measurable** cost of buying the next order — a Pareto frontier rather than
  a prohibition. That reframing is the contribution; the barrier itself is classical.

**On Bolley–Crouzeix — stated with my actual confidence.** I am confident there is a
classical order barrier for **unconditionally** positivity-preserving one-step methods for
parabolic problems (Bolley & Crouzeix, *Conservation de la positivité lors de la
discrétisation des problèmes d'évolution paraboliques*, RAIRO Anal. Numér. **12** (1978)),
and that the barrier usually quoted in the SSP literature for unconditional positivity is
**order 1**. I am **not** confident about an order-**2** barrier as stated, and the
numerics here contradict any order-2 barrier for *conditionally* positive schemes — the
$r=1/6$ FTCS scheme is positive and fourth-order. The reconciliation is the
**conditional / unconditional** distinction: barriers of that family constrain schemes
that remain positive for *all* $\Delta t$, whereas we accept a CFL-type step restriction,
and under a step restriction high-order positive parabolic schemes exist. **This exact
statement should be checked against the paper before being relied on.**

## Claim 11 (scope — and why the barrier is a reason to learn, not an obstacle).

The barrier in Claim 9 applies to schemes that are **linear in the data with a
state-independent stencil**. The classical escape is to abandon linearity: flux limiters,
ENO/WENO select the stencil *from the local solution*, and are therefore nonlinear maps
even for a linear PDE. A learned, state-dependent stencil selector is exactly such a
nonlinear scheme, so Godunov's theorem does not apply to it. **The order barrier is the
reason to learn the selector, not an obstacle to it** — and the positivity LP supplies
the per-decision certificate such a selector needs, at 0.2 µs (Task 5).

One caveat for honesty: the Task 3 feasibility geography and the Task 4 rollout both
describe the *linear-per-step* scheme. A state-dependent selector escapes the barrier but
also forfeits the clean $\|W\|_\infty\le1$ argument, because the amplification matrix then
depends on the solution; the guarantee that survives is the local one (no new extremum is
created at any single prediction), which is still exactly what a discrete maximum
principle needs.
