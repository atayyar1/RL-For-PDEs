# G1 — Certified scattered-point operators under time-marching

**Kill condition not met, in two different ways.** The consistency-only operator *as both papers
build it* — projected spatial ∂x, ∂xx plus explicit time-stepping — is unstable in **167 of 195**
periodic configurations (86 %), including on the **uniform grid at K ≥ 5, at any time step**, for a
closed-form reason (§2). The same consistency rows written as a one-step propagator are unstable
in 45 of 195 — and **every one of those 45 is a configuration where positivity is infeasible**
(0 feasible-and-unstable out of 89) **[measured, periodic + Dirichlet, 2 safety factors]**. So
positivity is not redundant, but its role differs by formulation: for the papers' object it is a
certificate that *replaces NeMDO's filter* at 5–34× (periodic) to 200–2400× (Dirichlet) lower
error, and works on the disordered sets where no filter strength stabilises at all; for the
one-step object it is a *detector* — infeasibility predicts every instability and the widening
policy fixes 19 of 45 — plus a 5–50× accuracy gain where feasible. The unfiltered arm has no
smooth-data regime: blow-up time is independent of the initial wavenumber (§6).

Code: `g1lib.py` (library), `floor.py` (apparatus floor), `sweep.py` (main runs, CSV per row),
`summarize.py` (tables from CSV), `uniform_reduction.py`, `degeneracy.py`, `figures.py`.
Results in `results/sweep_*.csv`, figures in `figures/` (150 dpi). Positioning in `POSITIONING.md`.
Every claim below is tagged **[classical]**, **[measured]** or **[conjectured]**.

## Scope limits, stated before the results

1. **1-D, linear, constant coefficients** (`u_t + c u_x = α u_xx`, c = 1), Eulerian points that do not
   move. Both papers' operators are 2-D and NeMDO's application is nonlinear; nothing here says what
   a 2-D GNN does. What transfers is the *mechanism*: a consistency constraint fixes M rows of an
   N-weight stencil and leaves N − M free directions, and marching tests those directions.
2. **No network was trained.** The "learned" arms are the *projection structure* of SpeND (Eq. 5,
   ŵ = w̃ − C⁺(Cw̃ − b)) with three explicit raw outputs w̃ — zero (= minimum norm), a Gaussian
   kernel, and a SpeND-style spectral least-squares objective with and without SpeND's one-sided
   |g| > 1 penalty — plus the method-of-lines form (projected spatial ∂x, ∂xx + forward Euler / RK3)
   that is literally what both papers produce. A trained network could land elsewhere in the
   affine subspace; these arms bracket what consistency alone allows, not what training finds.
3. **Stencils are K-nearest neighbours** (K = 3, 5, 7) at one time level, plus a 2K mixed-level
   variant. Both papers use larger stencils (30 in SpeND, 10–100 in NeMDO) in 2-D; a 1-D K = 7 has
   4 free directions, which is the same order of under-determination as their p = 2 / p = 4 cases
   per dimension, but not the same count.
4. **Stability is measured on the assembled linear map** M (u^{n+1} = M u^n), by the growth
   envelope max_n ‖Mⁿ‖∞ over n ∈ {1, 10, 100, 1000, 10⁴, n_T}, and by marching. The spectral
   radius is *not* used: for these nonnormal matrices `eigvals` returned ρ = 0.699 and 0.900 for
   two matrices equal to 5e-16 **[measured]**. "Unstable" below means growth > 100; every
   certified arm has growth ≤ 1 + 1e-9 by construction (‖w‖₁ = 1 row-wise ⇒ ‖M‖∞ = 1).
5. **The reference.** The shared exact series `Problem.u_true` is exact only for α ≥ 0.05:
   self-consistency 3e-15 at α = 0.1, 3e-11 at α = 0.03, and **unusable at α ≤ 0.01** (it returns
   max|u| = 1.4e5 at α = 0.01 and 1e55 at α = 0.003 for a solution bounded by 1.3 — the
   exp(βx) transformation with β = c/2α overflows) **[measured, floor.py F0.1]**. For α < 0.05 the
   Dirichlet reference is Crank–Nicolson on 16001 points, dt = 5e-5, which agrees with the exact
   series to 1.3e-8 at α = 0.1 and has a Richardson floor of 3e-6 at α = 0.003. The periodic runs
   use the exact two-mode solution (roundoff-exact at every α). Reported errors are all ≥ 100× above
   the relevant floor except where marked.
6. **Cell Péclet is varied by α at fixed nx** (Pe = c h/α = 0.1, 0.33, 1, 3.3 on nx = 101; 10 on the
   periodic nx = 100 set). On the Dirichlet problem, Pe > 2 means the outflow layer (width α/c) is
   under-resolved by every arm at this nx, so Dirichlet errors at Pe = 3.3 are O(1e-1) for all arms
   and only stability is compared there; the periodic set has no layer and is the clean Pe test.
7. The time step is `safety · min(h_min/c, h_min²/2α)` on the **actual** minimum spacing, safety
   ∈ {0.5, 0.9}. On disordered sets this makes the local r = αΔt/h_loc² small at wide gaps, which
   is where positivity feasibility is lost (see §4). Both papers would face the same choice.

## 1. Apparatus floor (CHECKS.md #1) — `floor.py`

| check | result |
|---|---|
| exact series vs itself (n_series, nq doubled), α = 0.1 / 0.03 | 3.1e-15 / 3.1e-11 |
| exact series at α = 0.01, 0.003 | **wrong** (max\|u\| = 1.4e5, 1e55) — replaced by CN |
| CN 16001 vs exact, α = 0.1 | 1.3e-8; CN Richardson floor at α = 0.003: 3.1e-6 |
| uniform K = 3: every propagator arm vs Lax–Wendroff-diffusion row | ≤ 1.3e-15 |
| uniform K = 3: MOL-FE vs FTCS row | 8.9e-16 |
| marching loop, uniform 3-pt LW, nx = 51 / 101 / 201, T = 0.5 | L∞(T) = 5.4e-4 / 1.35e-4 / 3.4e-5 (2nd order) |

So the discretisation error of a *good* scheme at nx = 101 is ~1e-4 (Dirichlet, Pe = 0.1); the
effects claimed below are factors of 10–10⁶ above that, or are blow-ups.

Two apparatus bugs found and worked around, both worth reporting to the programme:

- **`core.solve_maxent` discards converged answers.** On 10–25 % of stencils that
  `positive_feasible` certifies — including K = 3, where the feasible set is a single strictly
  positive point — it returns `None`. Traced: Newton reaches |g| ≈ 1e-13…1e-10, the Armijo test
  then fails at the roundoff floor of the dual, the step shrinks to 1e-7 and 200 iterations run
  out. The iterate is correct (identical to the LP point at K = 3). `g1lib.w_maxent` retries at
  tol = 1e-9 and only then falls back; the fallback-to-LP path was taken 20 times in 25,721 solves (0.08 %); the tol = 1e-9 retry rescued 13 %.
  Any earlier thread that counted `solve_maxent is None` as "infeasible" over-counted.
- **Spectral radius is not a stability measure here** (item 4 above).

## 2. Does the projected operator reduce to FTCS / Lax–Wendroff on a uniform grid? — `uniform_reduction.py`, `figures/uniform_reduction.png`

**[measured]** At K = 3 the system is fully determined: every propagator arm equals the
LW-diffusion row to 1e-15 and MOL-FE equals FTCS to 1e-16, at Pe = 0.1, 1 and 3.3.

**Limit of the construction, stated plainly: the projected operator reduces to the classical
scheme only on the minimal (3-point) stencil.** At K ≥ 5 nothing reduces to LW — not the
minimum-norm point, not the Gaussian projection, not the spectral objective, not maximum entropy,
not the LP vertex. LW is the unique 3-point member of the affine subspace and every arm spends the
free directions elsewhere, so a "generalises the classical schemes" claim for this construction is
true for three-point stencils and false beyond them; what generalises is the *consistency*, not
the scheme. Measured deviations:
max|w − LW| on the three central weights is 0.05–0.26 (minnorm), 0.04–0.09 (projgauss),
0.05 (spectral), 0.04–0.09 (maxent), 0.03–0.44 (LP vertex), and every arm puts 1e-2…1e-1 of weight
outside ±h. The generating-function rows reduce; the *projection* reduces only when there is
nothing to project.

**[measured, then classical]** On the uniform periodic grid every one-step arm — including
`minnorm` with min w = −0.08 and ‖w‖₁ = 1.30 — has max|g(θ)| = 1.0000. The only arm with
max|g| > 1 on the uniform grid is the **method-of-lines** form: 1.143 (K = 5), 1.054 (K = 7),
1.030 (K = 9) at r = 0.25, and 1.086 at Pe = 3.3. The mechanism is closed-form: the minimum-norm
p = 2 second-derivative stencil on 5 uniform points is **h²·D₂ = (2, −1, −2, −1, 2)/7** (minimise
2a² + 2b² + (2a + 2b)² subject to b + 4a = 1; measured to 1e-16), so its symbol at θ = π is
**+4/(7h²) — anti-diffusive at the Nyquist mode** — and forward Euler gives |g(π)| = 1 + 4r/7 for
every r > 0. The SpeND projection with a trivial raw output is therefore unconditionally unstable
under explicit marching on a uniform grid, at any time step, purely because p = 2 consistency
constrains nothing at high k. (SpeND's own loss targets ∂x only; NeMDO's Laplacian is learnt with a
GNN whose bias is unknown; neither reports the sign of D₂'s symbol.)

## 3. Degeneracy of the LP arm (CHECKS.md #3) — `degeneracy.py`

**[measured]** Fixed geometry, α = 0.1, K ∈ {5, 7}, zero objective vs 20 random objectives:

| geometry | growth (all 21) | LP errT range | maxent errT | nnz/node |
|---|---|---|---|---|
| jitter 0.25, K=5, Dirichlet | 1.000 | 6.1e-4 … 6.3e-3 (×3.7… ×10) | 2.2e-4 | 3.00 |
| jitter 0.25, K=7, Dirichlet | 1.000 | 8.1e-4 … 1.1e-2 | 2.2e-4 | 3.00 |
| random, K=7, Dirichlet | 1.000 | 3.0e-3 … 2.0e-2 | 1.2e-3 | 3.00 |
| jitter 0.25, K=5, periodic | 1.000 | 4.9e-4 … 4.9e-3 | 2.6e-4 | 3.00 |
| random, K=5 (2 infeasible nodes) | 648 … 985 | 1.98 … 2.02 | 2.00 | 3.04 |

What survives the pivot rule: the **stability verdict** (growth exactly 1 for every vertex whenever
feasible; 650–985 for every vertex when two nodes are infeasible). What does not: the **accuracy** —
a ×4–10 spread across vertices, every vertex 3–50× worse than the unique max-entropy point. This is
F30/F38 on scattered marching geometry, with the vertex effect confirmed as an *equality-constrained*
phenomenon (exactly 3 non-zeros per node at K = 5 and K = 7).

## 4. The sweep — `sweep.py`, `summarize.py`, `results/summary_*.txt`, `figures/fig1_growth.png`

Grid: 5 geometries (uniform; jitter 0.10, 0.25, 0.45 h; random gaps 0.3 + Exp(0.7)) × up to 3 seeds
× K ∈ {3, 5, 7} × α ∈ {0.1, 0.03, 0.01, 0.003 (+0.001 periodic)} × 11 arms, T = 0.5, n_T = 500–200 000
steps, on the Dirichlet set (nx = 101) and the periodic set (nx = 100), safety 0.5 and 0.9, plus a
mixed-time-level set. 6,730 runs. "Unstable" = growth > 100; "bounded-but-wrong" = growth ≤ 100 and
L∞(T) > 0.1 (exact max|u(T)| = 0.60).

### 4.1 Stability: who blows up, and where (periodic, safety 0.5; 195 runs per arm)

| arm | what it is | unstable | growth ≤ 10 | unstable among the 89 configs where positivity is feasible everywhere | unstable among the 106 with an infeasible node |
|---|---|---|---|---|---|
| molfe | projected ∂x, ∂xx + forward Euler (SpeND/NeMDO object) | **167** | 28 | **67** | 100 |
| molrk3 | same + RK3 | 160 | 35 | 67 | 93 |
| molfilt | molfe + tuned hyperviscous filter (NeMDO-style) | 105 | 86 | 23 | 82 |
| minnorm | one-step rows, SpeND projection with w̃ = 0 | 45 | 150 | **0** | 45 |
| projgauss | one-step rows, Gaussian w̃ projected | 45 | 150 | 0 | 45 |
| spectral / spectralg | one-step rows, SpeND-style LS objective (± hinge on \|g\|>1) | 50 | 134 | **5** | 45 |
| maxent, lp, lprand | certified where feasible, min-norm fallback where not | 45 | 150 | 0 | 45 |
| maxentw | maxent, stencil widened to ≤ K+6 until feasible | **26** | 169 | 0 | 26 |

**[measured]** Same structure on the Dirichlet set (minnorm 0/89 feasible-unstable, 38/67 infeasible-unstable;
molfe 67/89 and 61/67) and at safety 0.9 (minnorm 0/68 and 33/67; molfe 54/68 and 64/67).

Three readings, in decreasing strength:

1. **The certificate is exact and sufficient, as advertised**: over all 6,730 runs, every operator
   with ‖w‖₁ = 1 on every row had growth ≤ 1 + 1e-9 (536/536 periodic, 526/526 Dirichlet, 397/397
   and 362/362 at safety 0.9, 149/149 mixed-level). And **not necessary**: 826 of 1,609 uncertified
   periodic operators were also stable — F13, on scattered marching geometry.
2. **The method-of-lines form is the unstable one, and its instability is not a disorder effect.**
   It fails on the uniform grid (11/15 periodic, 8/12 Dirichlet) at every α, K ≥ 5, because the
   min-norm Laplacian is anti-diffusive at Nyquist (§2). RK3 does not help (160 vs 167). Disorder
   makes it worse (45/45 at jitter 0.45 and random). This is the object SpeND defers and NeMDO
   filters.
3. **For the one-step form, instability and positivity-infeasibility coincide.** Every unstable
   minnorm/projgauss run (45 periodic, 38 Dirichlet, 33 at safety 0.9, 18 mixed-level) has at least
   one node where no positive stencil exists; on the 89 (68, 30) configurations where one exists at
   every node, the consistency-only operator was stable in every case, though with growth up to
   10 and 5–50× the error (§4.3). The infeasible nodes are the geometric ones: K-nearest stencils
   that are one-sided (a gap > 2h on one side, F12) or whose characteristic foot lands between two
   nodes with |ξ_L|ξ_R > 2αΔt (the local Péclet/Courant band, §4.4). Where the consistency-only
   operator was *infeasible but stable* (61 periodic), it was bounded-but-wrong in 13 (minnorm) —
   e.g. random K = 5, α = 0.1: growth 600, L∞(T) = 2.0 with max|u| never exceeding 2.4, an exact
   eigenvalue 1 on a downwind-decoupled block (`fig3_march.png`, right).

**[conjectured]** Reading 3 suggests a theorem-shaped statement — *for p = 2 one-step rows on
K-nearest 1-D stencils, the min-norm consistent operator is stable whenever the positive cone is
non-empty at every node* — but the spectral arm's 5 feasible-and-unstable cases (all jitter 0.45,
K = 5, ν ≈ 0.001, growth 1e15–∞ over 10⁵ steps) show it is false for other points of the affine
subspace. Feasibility bounds the *min-norm* point, not the subspace.

### 4.2 Can the certificate replace the filter? — `molfilt` vs `maxent`

The filter arm reproduces NeMDO's stabilisation: after each forward-Euler step apply
F = I − ε h⁴D₄ with D₄ the unique p = 4 stencil on the 5 nearest nodes ((1,−4,6,−4,1) on a uniform
grid). **Tuning procedure, stated exactly because the accuracy numbers depend on it**: for each
configuration ε is scanned over {0} ∪ 28 log-spaced values in [1e-4, 1/8] and set to the
*smallest* value whose growth envelope over the full horizon is ≤ 10 (growth ≤ 1 is not required;
a bounded transient is accepted). This is the most favourable setting the filter can have: the
stable set in ε is a band (§4.2, next paragraph), every stable ε above the chosen one adds
dissipation and therefore error, and every ε below it is unstable. A "better tuned" filter in
the sense of more accurate does not exist within this family; a different filter family (sharper
than a 5-point D₄) is not constructible from p = 4 consistency on scattered nodes without a wider
stencil, and the wider min-norm D₄ is sign-indefinite (measured, the K = 7 version was unstable
on its own). So the comparison is accuracy at matched stability with the filter at its optimum. Walls and the two
nodes beside each wall are left unfiltered (a one-sided D₄ is not dissipative; with it filtered,
every Dirichlet run failed — `results/sweep_filt_dir_s05_wallbug.csv`).

**[measured]** Tuned ε at K = 5, α = 0.1: uniform 0.0089, jitter 0.10 0.0053, jitter 0.25 0.0024
(same on both domains); at Pe = 3.3 slightly less. **At jitter 0.45 and on random points no ε in
[0, 1/8] stabilises** (105/195 unstable; the stable set in ε is a band — too weak leaves the PDE
step unstable, too strong makes F itself unstable: ‖F^{n_T}‖ = ∞ at ε = 1/16 with no PDE step at
all on jitter ≥ 0.25). At Pe = 10 (α = 0.001) no ε works on any geometry.

Accuracy at matched stability, median L∞(T) ratio to `maxent` over the configurations where both
are stable **[measured]**:

| | periodic K=5 | periodic K=7 | Dirichlet K=5 | Dirichlet K=7 |
|---|---|---|---|---|
| α = 0.1 (Pe 0.1) | 15 | 33 | **2360** | **2600** |
| α = 0.03 | 18 | 13 | 594 | 800 |
| α = 0.01 | 19 | 34 | 200 | 221 |
| α = 0.003 (Pe 3.3) | 11 | 5.6 | 18 | 8.9 |

The Dirichlet numbers are the filter paying its dissipation where the solution has structure: the
error sits at x = 0.92–0.99 (0.05–0.14, the outflow layer) while the interior error is 1.7e-3
against maxent's 3.9e-5. Against the classical 3-point FTCS on the same points (molfe K = 3, where it is stable — 28/65 runs, all on feasible geometry),
the filtered K = 5 operator is 7× worse (periodic, α = 0.1) — the filter costs more than it buys.

So on this problem: **the certificate replaces the filter at lower error everywhere the filter
works, and works where the filter does not** (jitter 0.45: maxentw 0/9 unstable at α ≤ 0.01 vs
molfilt 9/9; random: maxentw 0/9 at α = 0.1 vs 9/9). The one place both fail is Pe > 2 with a
small Courant number, §4.4.

### 4.3 Accuracy where everything is stable (periodic, safety 0.5; median ratio to maxent)

| α | K | minnorm | projgauss | spectral | molfilt | LP vertex | random-objective LP | FTCS (molfe K=3) |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 5 | 13.6 | 10.1 | 2.8 | 15 | 23.6 | 12.2 | 2.2 |
| 0.1 | 7 | 52 | 25 | 5.9 | 33 | 33 | 10.8 | |
| 0.01 | 5 | 6.8 | 5.3 | 1.7 | 19 | 5.6 | 4.2 | 12.7 |
| 0.003 | 5 | 1.8 | 1.4 | **0.53** | 11 | 0.98 | 1.0 | 87 |
| 0.001 | 5 | 1.0 | 0.74 | **0.45** | 2.5 | 1.0 | 1.0 | — |

**[measured]** At Pe ≤ 1 maximum entropy is the most accurate point of the feasible set by 3–50×
over the other consistency-only points, and the LP vertex is the worst positive point (F30
reproduced on marching geometry; degeneracy §3). At Pe ≥ 3.3 the ranking inverts: the spectral
objective beats maxent by 2×, and maxent ≈ minnorm because at safety 0.5 most nodes are
infeasible and fall back (§4.4) — a scope limit on F30 in the same direction as F42: prefer
maximum entropy when the feasible set is large; check when it is small.

Uniform-grid absolute numbers, α = 0.1, K = 5, periodic: maxent 6.0e-6 (below the 3-point LW floor
of 1.5e-4 — the wider positive stencil is a higher-accuracy propagator, F30), minnorm 5.6e-4,
spectral 1.1e-4, molfilt 8.0e-4, LP vertex 6.0e-3.

### 4.4 Where positivity is infeasible: the Péclet–Courant band — `figures/fig4_feasibility.png`

**[classical, derived here]** For the p = 2 one-step rows the third row is ½ξ² − αΔt with ξ the
offset from the characteristic foot, so a positive stencil exists iff two nodes straddle the foot
with |ξ_L|·ξ_R ≤ 2αΔt. On a uniform grid this is ν(1 − ν) ≤ 2r, i.e. **ν ≥ 1 − 2/Pe** for Pe > 2:
at high Péclet, positivity needs a *large* Courant number (the foot near a node), and widening the
stencil cannot help because the nearest straddling pair is always the best chord.

**[measured]** Fraction of nodes infeasible at K = 5: 0 for Pe ≤ 1 on every jittered set (4 % on
random, the one-sided gaps); at Pe = 3.3 with safety 0.5 (ν = 0.5·h_min/h < 0.4 on jittered sets):
45–81 %; with safety 0.9: 0 % on uniform and jitter 0.10, 43 % on jitter 0.25, 79 % on jitter 0.45.
Widening (maxentw) removes almost none of these (47 → 44.7 nodes on jitter 0.10), as derived. At Pe = 10 everything is infeasible at
safety 0.5 (ν ≥ 0.8 needed) and the certified arms reduce to minnorm.

This is the honest limit of the certificate on this problem: it is a *local* CFL-type condition, and
when it fails the fix is the time step, not the stencil. In that regime the consistency-only
one-step arms were stable anyway (the foot is inside the hull; only the sign is lost) and the
spectral objective is the most accurate choice.

### 4.5 Mixed time levels (Dirichlet, safety 0.5, K ∈ {3, 5}, 2K-point stencils from t−Δt and t−2Δt)

**[measured]** Same pattern: minnorm/projgauss 0 feasible-and-unstable of 30, 18 of 24 infeasible
configurations unstable; maxent 0/30; maxentw 10/24. The spectral arm is unstable in **15 of 30
feasible** configurations here — its per-point mixed-level target is a weak proxy — so the
SpeND-style objective is the least robust choice on multi-level stencils. Accuracy at α = 0.1:
maxent 2.0e-4, LP 1.2e-3, minnorm 1.9e-3.

## 5. Dirichlet vs periodic: what the walls add

**[measured]** Boundary stencils change no verdict (the pairing tables agree to within a few
configurations) but they raise the bounded-but-wrong count: minnorm 22 feasible-and-bounded-wrong
on Dirichlet vs 7 periodic, the **LP vertex 30** (its 3-non-zero extremal stencils near the outflow
layer are the F26 useless scheme, stable and wrong), maxent 2. The reference floor at the walls
is 1.3e-8 (exact) / 3e-6 (CN), far below these.

## 6. Initial-condition wavenumber — `ic_sweep.py`

**[measured]** Unfiltered MOL-FE, periodic, K = 5, α = 0.1, u₀ = sin(2πm x), m ∈ {1, 2, 4, 8, 16, 25}:
blow-up time (max|u| > 10) is 0.072–0.075 on the uniform grid, 0.010–0.027 on jitter 0.25,
0.010–0.032 on random, monotone in m but never absent — the operator is linear and fixed, its
unstable mode is seeded by roundoff, and smooth data buys ≈ 300 steps at most. There is no
"stable on smooth data" regime for this arm. The one-step arms on the same sets: minnorm error
5.6e-4 → 5e-15 as m rises (the exact solution decays as e^{−α k² T}), maxent 1.1e-6 → 1e-16.

## 7. What this does and does not say about SpeND / NeMDO

- **Says** [measured, 1-D]: the *structure* they use — p = 2 consistency on K-nearest scattered
  stencils with the remaining directions chosen by an objective that is not a sign constraint —
  produces spatial operators whose explicit marching is unstable on most geometries, on the
  uniform grid at any Δt for the trivial objective, and on strongly disordered sets for every
  objective tried, including SpeND's |g| ≤ 1 hinge (5 feasible-and-unstable, identical to the
  un-hinged arm) and NeMDO's hyperviscous filter (no stabilising ε at jitter 0.45 / random).
  NeMDO's unreported TGV stability is consistent with mild disorder plus a filter tuned in the
  band found here.
- **Says** [measured]: a positivity certificate on the one-step operator is computable on this
  geometry (no symbol needed), is exact (536/536), and where it holds the maximum-entropy point
  is 5–50× more accurate than any other consistent point at Pe ≤ 1 and 15–2600× more accurate
  than the tuned filter.
- **Does not say**: anything about trained GNN outputs in 2-D, about p = 4 stencils of 30 points,
  or about nonlinear problems. The transfer is the mechanism (§2), which is dimension-free.
- **Kill condition, restated**: the unfiltered consistency-only arm is not stable on every test;
  positivity is not redundant. But the honest form of the result is *detector + accuracy*, not
  *fix*: on the one-step operator, wherever positivity could fix an instability the min-norm point
  was already stable, and wherever the min-norm point was unstable positivity was infeasible.

