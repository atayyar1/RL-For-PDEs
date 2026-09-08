# Project Context — Handoff Summary

*Written to seed a fresh conversation. Everything below is either verified in
code/data during the previous session, or explicitly marked as a hypothesis.*

---

## 1. What the project is

**RL for adaptive point placement in space--time meshfree PDE solving.**
Ali Tayyar, American University of Beirut, supervised by Dr. Joseph Bakarji.

Two coupled components:

1. **Reconstruction environment** — a GFDM-like local linear predictor that
   estimates the solution at a target space--time point `z* = (x*,t*)` from `n`
   previously evaluated neighbours. **Validated and essentially finished.**
2. **RL agent** — learns *where* to place the next evaluation points so that
   the prediction at `z*` is accurate and cheap. **Works at short and medium
   horizons; the central claim is not yet secure** (see §6).

**Immediate goal: a NeurIPS workshop paper.** Note this changes the framing
from the thesis version — an ML audience will care more about the MDP design,
the ablations and the failure analysis than a numerical-analysis audience
would. The design-history record (§5) is unusually strong content for a
workshop.

---

## 2. The reconstruction environment (settled)

Predictor: `û* = Σ wᵢ uᵢ`, weights determined by Taylor moment conditions with
the PDE **substituted into** the expansion (not appended as an extra row).

For `u_t + c u_x = α u_xx` the constraint matrix has **three rows**:

```
[ 1 ,  (Δxᵢ − c Δtᵢ)/h ,  (½Δxᵢ² + α Δtᵢ)/h² ] w = (1, 0, 0)ᵀ
h = max( max|Δxᵢ| , sqrt(α·max|Δtᵢ|) )
```

Solved as the minimum-norm solution `w = A⁺b` with `n = 5 > 3` (underdetermined).

**Error bound** (proved, and verified numerically — max ratio 0.549 over 80
random targets):

```
|u* − û*| ≤ ‖w‖₁ · C · h²      C = max(|u_xx|, |u_xt|, |u_tt|)
```

### Results established during the previous session (all numerically verified)

- **Substitution beats augmentation decisively.** At n=5, r=0.05 on an exact
  heat solution: no PDE → ‖w‖₁ 1.443, err 6.35e−3; PDE **appended** → ‖w‖₁
  9.482, err 3.39e−4; PDE **substituted** → ‖w‖₁ **1.000**, err **5.23e−6**.
  Appending constraints to an underdetermined system can only raise min‖w‖.
- **Constraint count.** With substitution, order `p` needs **p+1** rows, not
  (p+1)(p+2)/2. So O(h⁴) needs 4 rows and 5 neighbours, not 10 and 11.
- **Positivity ⟹ stability.** If `wᵢ ≥ 0` then `Σwᵢ = 1` makes `û*` a convex
  combination ⟹ discrete maximum principle, `‖w‖₁ = 1`. Error accumulation over
  k prediction levels is **linear** if ‖w‖₁ = 1 and **geometric** (`A^k`) if
  ‖w‖₁ > 1. On a regular 3-point stencil this reduces exactly to the CFL
  condition `αΔt/Δx² ≤ ½`.
- **cond(A) cannot detect instability.** Front stencil: cond 4.64, ‖w‖₁ 1.000,
  all weights positive. Single-walker zigzag: cond 5.66, ‖w‖₁ **2.133**,
  min wᵢ = −0.30. Both pass `cond < 1e4` by three orders of magnitude.
- **ξ-form improvement (found, NOT yet in the code).** The consistent third row
  for advection--diffusion uses the characteristic coordinate ξ = x − ct:
  `½Δξᵢ² + αΔtᵢ` with `Δξᵢ = Δxᵢ − cΔtᵢ`, not `½Δxᵢ²`. Verified 3–7× smaller
  error at every radius with ‖w‖₁ no worse. Effect scales with the aspect ratio
  `cΔt/Δx`, which is ~0.045 on the current grid, so the per-solve gain there is
  small.

---

## 3. The RL formulation (current)

| Element | Current form |
|---|---|
| γ | 1.0 (γ=0.99 tested, failed) |
| State | `ℝ^{2N+2}`: relative offsets in grid cells, plus `t*/Δt`, `x*/Δx`. **No uᵢ** |
| Action | `Discrete(N·|𝒜|)`, flattened; `i = a // |𝒜|`, `j = a mod |𝒜|` |
| `𝒜` | `{(−1,1), (0,1), (1,1)}` — δt always +1 |
| Termination | **bubble**: ≥ 5 evaluated points inside `x* ± RΔx`, `t* ± RΔt` |
| Reward | `−w_err·e_norm − w_time·(K/K_max)` + potential shaping |
| `e_norm` | `clip(log10(e/e_tol)/3, 0, 1)` |
| Algorithm | MaskablePPO (sb3-contrib), MlpPolicy, 12 SubprocVecEnv, CPU |

**Policy architecture history:** Gen 1 fully shared (combinatorial, failed) →
Gen 2 one walker at a time (worked, but no walker selection) → Gen 3a
MultiDiscrete (rank-one factorisation ⟹ deterministic/stochastic argmax gap) →
**Gen 3b flattened joint softmax (current)** → Gen 4 shared encoder +
joint softmax (proposed, not implemented).

**Action masking** was added because rejected moves leave the observation
unchanged, so a deterministic policy re-selects the same action forever
(livelock). This was the single most consequential implementation change.

**Admissibility constraints:** two-sided coverage and `cond(A) < 1e4` are
active; the two weight constraints (`max|wᵢ| ≤ 3/n`, `min wᵢ ≥ −1`) are
**commented out** — which, per §2, removes the only control over ‖w‖₁ and hence
over the error-accumulation rate.

---

## 4. Code state (`6.2 RL Burger.ipynb`)

Done:
- potential now uses the **mean** over walkers (was `np.min`, despite the name
  `_mean_target_distance`)
- `c_shape` raised 0.001 → **0.1**

**Not done:**
- **termination still `count_in_box` only** — does not check that an admissible
  stencil exists. *This is the blocker.*
- `err_tol` still fixed at 1e−3 (saturates to 0 at short horizons, 1 at long)
- ξ-form third row not adopted
- ‖w‖₁ neither penalised nor logged
- `KDTree` imported in cell 1 but never used; `find_neighbours` is a linear scan
- no `VecNormalize`

Current config: `N=35`, `R_bubble=4`, `K_max=2000`, `t_star_schedule={50:5000}`,
`x_star_schedule=[50]`, `seeds=[6]`, 100M timesteps ≈ 350 min on 12 cores.

---

## 5. The experimental record (mined, laptop-accessible)

**~100 distinct configurations, 328 tensorboard event files, 668 checkpoints,
~9,000 figures**, split between the working folder and `../Archives/`.

**Directory names encode the constants**, in three eras:
`t{t*}_lam{λ}_beta{β}_K{Kmax}_N{N}` → `t{t*}_errtol_werr_wtime_N_c` →
same with `_R{R}` appended.

**Structural choices are NOT in the names but ARE in the checkpoints:**
`observation_space._shape ÷ N` gives the state version; `action_space.n ÷ N`
gives |𝒜|; `policy_class.__module__` says whether masking was on. Saved image
filenames gained an `_x{index}` field exactly when `x*` entered the state, which
is an independent dating fingerprint.

**State space evolved through five stages** (recovered from checkpoints):
`3N+2` (runs1–8) → `3N`, target removed (runs9,10,13) → `2N` (runs14) →
`2N+1`, t* restored (runs14–17) → `2N+2`, x* added (runs21 onward).
`runs9` and `runs10` are a **fully controlled pair** (same t*, N, λ, β) differing
only in the state.

### Iteration outcomes recovered from saved notebook outputs

| Iteration | first → last | Conclusion |
|---|---|---|
| Gen 1 shared | reach 0.70 → **0.55**, rew −0.30 → −0.45 | degraded, stopped at 163k steps |
| Gen 2 one walker | reach 0.04 → **0.98**, err 1.1e−3 → 5.6e−4 | worked; no walker selection |
| Gen 3b flattened | reach 0.71 → 0.80, err 1.5e−5 → 4.3e−5 | retained |
| λ=0.001, no β | reach → **1.00**, rew → **−0.009**, err **flat**, ep_len 225 → **8.6** | accuracy term inactive; agent just rushed |
| λ=0.005, β=100 | reach 0.70 → **0.00**, ep_len → budget | timing out beat reaching |
| curriculum λ=0.05, β=100 | reach 0.79 → **0.00**, entropy → 4e−4 | same, plus policy collapse |
| bubble termination | reach 1.00, err 2.1e−3 → **6.9e−4** | largest single improvement |

β was swept over **10² to 10⁷**; no value satisfied both failure modes, which
is what motivated normalisation. Both **joint** schedules (`t5-7-10`, etc.) and
**sequential curricula** (`curriculum_t2/t4/t5/t6/t10/t15`) exist in the
archive — the chapter's claim that only the joint form was tested is wrong.

### Current-era results

| t\* | e_tol | N | c_shape | R | steps | reach | err on reach |
|---|---|---|---|---|---|---|---|
| 10 | 1e−3 | 30 | 1e−3 | 3 | 4M | 1.000 | 2.4e−4 |
| 50 | 1e−3 | 35 | 1e−3 | 3 | 100M | 0.724 | 6.2e−3 |
| 50 | 1e−3 | 50 | **0** | **4** | 10M | **0.999** | **3.6e−3** |
| 100 | 1e−3 | 90 | 1e−4 | **4** | 21M | **1.000** | **2.2e−3** |

**Every good run has R = 4; every R = 3 run is worse.**

---

## 6. The central open question

The chapter currently claims long horizons don't converge. **That is wrong** —
t\*=50 and t\*=100 both converge at R=4. What separates success from failure is
**R and N**, not the horizon.

But there is a problem with reading those as successes:

- **Entropy in the successful runs is 86–88% of uniform** (5.03 nats against
  ln(300)=5.70; 5.38 against ln(540)=6.29). A policy that reaches ~100% of the
  time has barely committed to anything.
- Every action has δt=+1, so walkers march upward regardless of policy.
- `walker_init_x` is computed from `nx//2` and is **explicitly independent of
  x\***. With N=50 the spacing is ~1.92 cells, so with R=4 five walkers start
  inside the bubble's x-range, straddling x\*. With N=35 the spacing is ~2.82
  cells and fewer do.

**Hypothesis (not yet tested):** reaching is trivially satisfiable by walkers
marching straight up, the successful configurations are those where the
*initialisation* happens to place walkers on both sides of x\* inside the
bubble, and the resulting single-column stencils are near-degenerate — which
would explain the ~80% failed-final-solve rate, the flat entropy, and the R/N
dependence simultaneously. It would also explain why x\* generalisation fails,
since the initialisation does not move with x\*.

**The one experiment that settles it: run the masked-random policy (cell 14 —
no training required) at N=50, R=4, t\*=50 and compare reach rate and error to
the trained policy.** If they match, the current results are an artifact of
initialisation. Nothing else should be built on top until this is known.

Two other laptop-scale measurements, both from the 668 saved checkpoints:
**‖w‖₁ over rollouts** (tests the degeneracy mechanism directly) and the
**envelope exponent** `w ∝ τ^p` (p≈½ diffusive, p≈1 at wave speed
characteristic, p≈1 at 22 cells/step = shortest-path artifact).

---

## 7. Known problems, ranked

1. **Termination ≠ admissible stencil.** `count_in_box ≥ 5` counts points
   without checking two-sided coverage or conditioning. ~80% of episodes in the
   failing configs end with `u_pred = None` → `e_norm = 1` regardless of
   behaviour. The reward is constant for those episodes.
2. **Error grows geometrically with horizon.** Implied amplification
   `A = 355^(1/40) ≈ 1.158` per level, versus `A = 1` for positive weights.
   Root cause: the ‖w‖₁ constraints were relaxed.
3. **Walker initialisation is independent of x\*** — explains the x\*
   generalisation failure without invoking learning at all.
4. **Reward band is narrow.** `e_norm` is informative only for
   e ∈ [e_tol, 10³·e_tol]; short horizons sit below it, long horizons above.
5. **Reward requires `u_true`**, so the method can only be trained where the
   answer is known — a hard limit on the "on-demand integrator" framing. The
   bound `‖w‖₁·C·h²` is computable without ground truth and tracks the true
   error closely (max ratio 0.549).
6. **Single fixed initial condition**, so `u_true` is a deterministic function
   of (x,t) and conditioning on (x\*,t\*) is sufficient to identify the
   solution — discovery and memorisation are not separated.

---

## 8. Strategic options for the paper

**Paper A (safe, writable now):** contribution is the reconstruction
environment — PDE-substituted space--time GFDM, proved+verified error bound,
exact recovery of FTCS and 4th-order stencils, validated 30,000-point rollout —
with the RL presented as a formulation plus an honest failure analysis.

**Paper B (stronger):** requires the RL to work at a meaningful horizon *and*
beat a baseline. Critical path: fix termination → confirm convergence →
work--precision diagram (error vs points evaluated, versus uniform grid and
masked-random) → envelope figure.

**For a NeurIPS workshop specifically**, the design-history and ablation
material is a genuine asset: the reward-ordering failure with data, the
rank-one/argmax-gap analysis, the livelock-from-state-invariance observation,
and the R/N ablation are all the kind of thing workshop reviewers value.

**Missing baseline.** Random and uniform are both trivial. The real competitor
is a **greedy placer** that puts the next point where it most reduces
`‖w‖₁·C·h²` — classical optimal design, no learning, computable without ground
truth. Nobody has built it. It sharpens the question to "does RL beat greedy by
planning ahead where greedy is myopic?"

**Structural critiques worth weighing:** (a) with one IC and a known solution
the optimal cloud is a fixed geometric pattern that could be written down, so
the current setup cannot demonstrate what RL is for; (b) spending thousands of
evaluations to predict one value is not an integrator — the natural task is
"given a budget of K evaluations, minimise error over a region," which makes
the reward dense, removes the termination pathology, and makes the uniform-grid
comparison natural.

---

## 9. Documents

| File | State |
|---|---|
| `MAIN.tex` | 29-page thesis progress document, chapters 1–6 |
| `RL Formulation.pdf` | Chapter 5 as currently compiled (16 pp, pp. 26–41) |
| `chapter2_revised.tex` | Rewritten Chapter 2 — drop-in replacement, not yet merged |
| `chapter5_revised.tex` | Rewritten Chapter 5 — largely merged already |
| `chapter5_experimental_record.tex` | §5.11 experimental record — merged, figures added |
| `RL_for_PDEs__Ali_Tayyar (1).pdf` | 5-page extended abstract (older; overclaims — attributes FD-stencil recovery to the learned policy when it is a property of the reconstruction environment, and claims a Pareto-front result the body says is pending) |
| `compute_options.pdf` | Compute/cloud costing report |

**Known LaTeX issues in the compiled chapter:** `[?]` for `ng1999policy` (bibtex
entry never added); `??` twice in §5.9 (missing labels); §5.7 table has
unfilled `N`/`A` placeholders and stale `c_shape`/`K_max`/`R`; §5.3.3 stops at
action-space Version 4 while the code uses Version 5; §5.8.2 contradicts §5.11.

---

## 10. Suggested first moves in the new conversation

1. Decide **Paper A or Paper B**, and the workshop's page limit and deadline.
2. Run the **masked-random baseline** — it is cheap and it determines whether
   the current results survive.
3. Draft the **GFDM section** first; it is finished, validated, and
   `chapter2_revised.tex` already contains the maths.
