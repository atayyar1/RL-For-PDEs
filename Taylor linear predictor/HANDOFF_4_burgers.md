# Handoff 4: clean pipeline done, comparisons done, next is Burgers (2026-10-07)

Read this first. It replaces `HANDOFF_3_clean_notebook.md` for current status (that file, `local_reward_formulation.md` and `where_we_are.md` are the history and the design math).

---

## 0. How to work with Ali (the user)

- Ali (MSc thesis, AUB; advisor **Joseph Bakarji**) builds notebooks by pasting **one numbered cell at a time**. For each cell: run it yourself first, then send a markdown cell + the code + the **expected output**, and explain it in plain language. Add a figure when it helps.
- Keep cells lean: prints and figures yes, but no unneeded extras (no name labels, redundant checks, long comments). Keep the method **general** (nonlinear PDEs come next): PDE-specific parts in one place.
- Short replies, one topic at a time. Explain derivations simply (the long notation-heavy version was rejected; a short 5-step version worked).
- No hidden hard-coded fallbacks. If a fallback is needed, make it explicit and report how often it is used.
- Long training runs on the **remote machine** (`10.30.116.107`, Remote Desktop, 12 cores, user `amt35`, Python 3.13). Test small locally, then send the cell. On the remote machine, `tensorboard` and `imageio-ffmpeg` had to be installed.
- Ali writes notes inside handoffs as `(Ali: ...)`. Treat them as open questions.
- Commits: Ali asks for them; end messages with the Co-Authored-By line from the system prompt.

---

## 1. The project in one page

**Goal (thesis "RL for PDEs", see `RL_for_PDEs__Ali_Tayyar.pdf` for notation and prior chapters):** predict the solution at one query point z* = (x*, t*) of a time-dependent PDE with as few computed points as possible and a controlled error, by letting an RL agent decide **which points to compute and which neighbours each point uses**.

**Current formulation (backward map, notebook 10 → clean notebook 11):**
1. **Predictor** (thesis §2.5): u_p ≈ Σ w_q u_q from 5 neighbours. Expand each u_q in Taylor series about p, replace time derivatives using the PDE (u_t = α u_xx − a u_x). With ξ = Δx − aΔt, the weights satisfy 3 rows: Σw = 1, Σwξ = 0, Σw(½ξ² + αΔt) = 0 (the **corrected** third row; the thesis PDF eqs. 2.35–2.36 and §3.3.2 still show the old wrong row). 5 points, 3 rows → underdetermined; min-norm weights. **Admissible** if rank 3, cond ≤ 1e4, ‖w‖₁ ≤ 2. Three points one level down give exactly Lax–Wendroff (check in cell 3).
2. **Backward map**: start at z*, give it a stencil, every non-known neighbour (not IC, not wall) becomes a new point, repeat down to the IC/walls. Points expanded **latest t first** (heap), so the influence β_p is complete when p is expanded. **β_q += β_p w_q**, β_z* = 1.
3. **Exact identity**: error(z*) = Σ_p β_p δ_p, with δ_p = u_p − Σ w_q u_q the local error (verified to 10 digits). Exact bound = Σ|β_p δ_p|. This is the discrete adjoint / DWR representation (Becker & Rannacher 2001), the exact version of thesis Corollary 2.7.
4. **Geometry score** (no solution values): s_p = |Σ w T₃|/(6L³) + |Σ w T₄|/(24L⁴), heat polynomials T₃ = ξ³ + 6αΔt ξ, T₄ = ξ⁴ + 12αΔt ξ² + 12(αΔt)², L = 1/(2π). It ranks stencils correctly (corr 0.98 at an interior point) and over-estimates by about 2× there (real |u_xxx| = 119 vs assumed 1/L³ = 248).
5. **RL**: the agent picks the 5 window cells of each point one at a time (masked to admissible completions). Reward per completed stencil r = −|β_p| s_p / B_LW − λ·(new points)/N_LW. Total return = −B/B_LW − λ(N−1)/N_LW, where B_LW, N_LW are the score sum and point count of the **Lax–Wendroff map** of the same z* (LW scores about −1 − λ). λ = price of one new point.

**Simplified "simple words" story:** the error at z* splits exactly into one piece per computed point (influence × local error); the agent spends accurate stencils where the influence is high and skips work where it is tiny.

---

## 2. Test problem (linear)

u_t + a(u) u_x = α u_xx on [0, 1], u = 0 at both walls, u(x,0) = sin πx + ½ sin 2πx, α = 0.1, a(u) = c = 1. nx = 100, T = 0.5, dt from CFL (nt = 1090, dx = 0.0101, dt = 4.59e-4, r = αdt/dx² = 0.45). Exact solution: u = exp(b x − αb²t)·v, b = c/(2α), v a sine series (800 modes). Main query z* = (50, 20) (grid indices).

Ali's professor said "advection–diffusion with Dirichlet BCs doesn't work": it is well-posed for α > 0, but the outflow wall creates a boundary layer of width about α/c (resolved here: cell Péclet 0.05); with α → 0 the outflow Dirichlet condition becomes ill-posed. Decision: keep it as is.

---

## 3. The clean notebook: `11. Backward Map RL.ipynb` (project folder)

Cells (in the file; the numbering used in conversation is in brackets):

| # | Content | Key names |
|---|---|---|
| [1] | imports | numpy, gym, sb3, MaskablePPO, ActionMasker |
| [2] | PDE, grid, exact solution, plot | `alpha, c, a(u), A_MAX, u0, nx, nt, dx, dt, x_grid, u_true(x,t)` |
| [3] | FD baselines + predictor | `fd_step, FD_SCHEMES {"FD-3","LW","FD-5"(dt/2)}, fd_reference(ix,it,scheme)` → (value, error, points, work); `COND_MAX=1e4, W_MAX=2, solve_weights(dxs, dts, a_loc)` |
| [4] | geometry score + 300-stencil scatter | `L_SOL, score(DX, DT, w, a_loc)` |
| [5] | backward map, β, forward pass, identity check | `U0, is_known(q), known_value(q), a_loc=c, weights(p, nbs)` (cached by shape), `point_score, start_map, next_point, expand(m, nbs), build_map(z, choose), evaluate(m)` |
| [6] | 21-cell window, admissible stencils, mask, lowest-score rule | `R_X=K_T=3, WINDOW, N_CELLS=21, N_NB=5, place(p,k), admissible(p)` → (sets, uses, scores), `pick_mask(p, picks), lowest_score(m,p)` |
| [7] | environment | `lw` rule (defined at the top of this cell), `report(m)` → points/work/error/bound, `reference(z)` (LW map), `MapEnv(queries, lam, cap=3)`, obs size 91 |
| [8] | 2×2 map figure | `plot_map(m, title, path)` (influence map, zoom with stencils, accumulated error, error share) |
| [9] | training helpers | `make_env, run_policy, MapLogCallback, SnapshotCallback` (snapshots count episodes; **updated versions with `env_cls` live in notebook cell 30**, copy them into cell 9) |
| [10] | training | `QUERIES, Z_PLOT, LAM, SEEDS, TOTAL, N_ENVS=12, PPO_KW = dict(n_steps=256, batch_size=256, gamma=1, gae_lambda=1, lr=3e-4, ent_coef=0)`; run folder `runs{m}-{d}/{run_name}/seed{s}/{images, checkpoints, logs}` + `config.json`, `snapshots.json` |
| [11] | videos and curves | uses project files `map_progress_video.py`, `map_build_video.py`, `learning_curves.py`; `baseline(rule)` |
| [12] | Comparison 1: best fixed stencil (brute force) + histogram/scatter (12b) | `fixed_rule(s), map_return(m,z,lam), fixed_chunk`, joblib, `rows` |
| [13] | Comparison 2: accuracy vs evaluated points | `classical(scheme, k)` for FD-3, LW, Taylor-5, FD-5+RK4, CN on grids refined k = 1..4; `greedy` rule; `curves, pts_map` |
| [14] | Comparison 3: how many Taylor rows | **code cell is NOT in the saved notebook file** (only its markdown). The code is in the appendix below |
| [15] | Multi-query training | `MapEnvGen` (levels above IC clipped at 10, distance to nearest wall clipped at 10 instead of offset to z*), TRAIN_Q, run folder `multi15_...` |

State (91 numbers) of `MapEnv`, for the current point p: per window cell (21 × 4): picked, already in map (reuse is free), known data (IC/wall), best score still reachable through it (log10 ratio, capped at 3); global (7): log|β_p|, levels above IC, x-offset to z*, pick number, points used/N_LW, pending points/N_LW, pending total |β|. Points with a single admissible stencil are auto-played. Cap: 3·N_LW points.

Project helper files (tested): `map_progress_video.py` (`make_progress_video(img_dir, fps)`), `map_build_video.py` (`make_build_video(m, path, is_known, title)`), `learning_curves.py` (`plot_learning_curves(snapshots.json, baselines, path)`). Both video files set the ffmpeg path from `imageio_ffmpeg` at save time (`_ffmpeg_writer`).

`.gitignore` (repo root `RL For PDEs/`): `local_reward_*.zip` added; `runs*/` already ignored. Two old zips in `test_run/` are still tracked (Ali was asked, no answer).

---

## 4. Results so far (linear PDE, z* = (50, 20), λ = 1, LW ruler)

**Single-query PPO run** (`runs10-6/single_z50-20_lam1_R3K3`, 5M steps, 28 min, about 3,050 maps): learned to LW level by 0.5M, passed lowest score at 0.7M, plateau from 1.6M at return ≈ −1.05. Final: **323 points, work 1615, bound 4.2e-6, error 8.2e-8 (cancellation), return −1.08**. Reproduces (slightly better than) notebook 10's 305 points / 1.0e-5. Pipeline confirmed.

**Comparison table:**

| Method | Points | Work | Error | Exact bound | Return |
|---|---|---|---|---|---|
| Best fixed stencil (brute force, 3-row min-norm) | 168 | 840 | 8.4e-7 | 1.1e-6 | −0.50 |
| PPO agent | 323 | 1615 | 8.2e-8 | 4.2e-6 | −1.08 |
| Greedy (one-step reward) | 313 | 1565 | 1.3e-7 | 4.6e-6 | not computed yet |
| Lowest score | 670 | 3350 | 1.5e-7 | 1.6e-7 | −1.69 |
| LW | 400 | 1200 | 3.0e-5 | 3.0e-5 | −2.00 |
| Random | 659 | 3295 | 7.8e-5 | 6.5e-4 | about −17.8 |
| Taylor-5 (our predictor, 5 rows, regular 5-pt row, classical sweep) | 780 | | 3.3e-8 | | about −1.95 (est.) |
| FD-5 + RK4 (k=1) | 6468 (×4 stages) | | 3.8e-8 | | |
| CN (k=1, 4× dt) | 490 | | 1.8e-5 | | |
| FD-3 (k=1) | 400 | 1200 | 5.2e-5 | | |

- Only **23 of 18,694** fixed stencils beat the agent (old agent: 75), but the best fixed stencil beats it on points, work and bound. All winners jump **3 levels down** (the agent mostly uses 1–2).
- Classical 2nd-order schemes need about 87,000 points (LW refined 4×) to reach 1.9e-6; every map method beats them by orders of magnitude.
- 4th-order curves (Taylor-5, FD-5+RK4) flatten at about 2e-8 under refinement (2nd-order closure next to the walls once the cone reaches them); only their k = 1 points are clean.
- Agent and greedy have error ≪ bound (cancellation luck); lowest score and best fixed have error ≈ bound.

**Comparison 3 (how many Taylor rows), all with the longer score T₃..T₆:**

| Predictor | Lowest score: points / bound | Best fixed: points / bound |
|---|---|---|
| 3 rows, min-norm (current) | 939 / 1.9e-6 (cell 6 with the short score: 670 / 1.6e-7) | 168 / 1.1e-6 |
| **3 rows, min-score** | **400 / 4.9e-8** | **160 / 1.7e-7** |
| 5 rows exact | 587 / 4.3e-8 | 162 / 1.7e-7 |

**Key finding:** keep the 3 rows (underdetermined, robust on any geometry) but spend the 2 free degrees on **cancelling the T₃/T₄ leftover instead of minimising the norm** → about 1 order of magnitude better, same as 5 rows, and the map methods now sit on/below the 4th-order frontier (160 points at 1.7e-7; 400 points at 4.9e-8 vs Taylor-5's 780 at 3.3e-8). Error ≈ bound for all min-score maps. Ali's original choice (underdetermined) was right for robustness (5-row square systems are ill-conditioned on irregular stencils, which is what broke earlier work); min-score gets the order back. **Plan: make min-score the predictor going forward** (agent retrain, Burgers). Caveat: min-score weights depend on L (balance of T₃ vs T₄).

**Interpretation agreed with Ali:**
- On this linear, uniform PDE the method works (big gains over classical 2nd-order), but RL is not needed: one offline search (about 50–100 s) finds a uniform stencil that beats the agent, and is cheaper at use time.
- Ali's counter-argument: for **many queries** a search may need to be redone each time while the agent decides in one pass (amortisation). Test: the multi-query run.
- The return depends on λ; a λ-sweep (one agent per λ) would give a curve to compare with the classical accuracy-vs-points curves (to do later).
- GFDM literature does not search stencils by error: stars are chosen by geometric rules (nearest neighbours, four-quadrant / eight-octant criteria; Benito, Ureña, Gavete), accuracy is tuned through weighting functions. Related: Seibold's minimal positive stencils, RBF-FD stencil selection (Davydov), DRP optimised coefficients (Tam & Webb). Space-time shape selection by an adjoint-weighted score looks new (not verified by a search). A GFDM quadrant/octant baseline should be added.

**Side idea (Ali will discuss with Joseph):** a separate paper without RL: GFDM/Taylor predictor + exact adjoint identity + geometry score + goal-oriented stencil-shape selection (+ min-score weights). Reviewers would want irregular nodes or variable coefficients, and a GFDM baseline.

---

## 5. Running / pending

- **Multi-query run (cell [15])** on the remote machine. Settings chosen: TRAIN_Q = t* ∈ {10, 20, 30, 40, 50} × x* ∈ {30, 50, 70} (Ali asked for bigger t*), TOTAL = 20M (about 2.5–3 h), λ = 1, seed 1, MapEnvGen. Each episode draws one query uniformly (by episode; big t* dominates the steps). Ali restarted the notebook before running it: needed cells 1–9, the cell-10 settings only (`json, time, CheckpointCallback, N_ENVS, PPO_KW`), then cell 15 with the updated helpers at its top. Status unknown at handoff; ask Ali for the snapshot lines.
- **Cell [16] to write:** generalisation table on training + unseen queries ((40,12), (60,18), (40,22), (60,28), (50,35), (50,40) and longer t* = 60, 80, 100): agent (no retraining) vs best fixed stencil searched per query vs the (50,20) winner reused vs LW vs lowest score; points, work, bound, return; total search time vs one training.
- **Greedy return** (2-line check) not computed yet.
- Copy the updated `make_env / run_policy / SnapshotCallback` (with `env_cls`) into cell 9.
- Save the cell [14] code into the notebook (appendix below).
- Split cell [5] into machinery and the check part (`lw` now lives in cell 7).

---

## 6. Next: Burgers (new notebook, same structure)

u_t + u u_x = α u_xx (6.2 used α = 0.1, IC sin πx + ½ sin 2πx, Dirichlet walls, Hopf–Cole exact solution, code in `6.2 RL Burger.ipynb` cell 3). **Open question:** keep α = 0.1 (mild steepening, comparable to 6.2) or use α = 0.01–0.03 (real front, where position-dependent stencils should matter).

The nonlinear term enters in **three places**:
1. **Weights** need a = u at p, but the backward map is built before any values exist.
2. **Taylor expansion:** with u_t = αu_xx − u u_x, the second-order terms gain −Δx Δt u_x² (and more): freezing a = u_p is exact to first order only; the leftover is large at steep fronts. This is the Cauchy–Kowalewski blow-up of terms known from Lax–Wendroff-type methods for nonlinear problems (modern fix: approximate the extra terms numerically: compact approximate Taylor methods, automatic differentiation).
3. **β:** weights depend on values → linearised adjoint (β_q += β_p (w_q + Σ_r ∂w_r/∂u_q u_r)). For Burgers, DWR literature uses the **secant coefficient a = (u + u_h)/2**, which makes the error representation **exact** (u² − u_h² = (u + u_h)(u − u_h)). We can test exactness with the exact solution.

Literature (quick skim) on the nonlinear term in GFDM/meshless Burgers: explicit "linear" schemes take the coefficient from already-computed values (previous level); implicit ones use Newton–Raphson (space-time GFDM), Picard (random-feature methods) or quasilinearisation; upwind/flux limiters for steep fronts. A 2026 paper "Optimized Stencil Strategy for the GFDM: application to steady nonlinear problems" (arXiv 2607.07165) must be read for overlap.

**Proposed design (not yet agreed; Ali asked for the literature check, then wanted a new session):**
- **Backward build:** the agent and the score use a cheap guess of u (characteristics from the IC, u₀(x − a t), or a coarse FD solve) for a and for a **local length scale L** (from guessed u_x, u_xx). Add the estimated u_x² leftover to the score. State gains local solution features (local a, |u_x|, L).
- **Forward evaluation:** weights with a from the neighbours' computed values (standard explicit GFDM choice, as in 6.2), optionally 1–2 Picard updates recomputing weights with computed u.
- **β:** linearised; test the secant version for exactness.
- Use the **min-score** predictor.

Questions still open for Ali: (1) A+C (guess + Picard) vs B (weights computed forward from neighbours, as 6.2)? (2) Which guess: characteristics or coarse FD? (3) α?

First Burgers cells that do not depend on these decisions: [1] imports, [2] PDE + Hopf–Cole + plot of the steepening front, [3] FD baselines (FD-3, upwind, LW-type) and their errors at a few z*.

---

## 7. Other open threads

- **Walkers (6.2 formulation):** β and the identity work as an analysis tool (backward sweep in placement order after an episode: which steps mattered, wasted points). As a training reward it needs reward redistribution (RUDDER-like), and Burgers makes it linearised. Not started.
- **Thesis PDF updates** (Joseph cares): corrected third row, backward construction, "chooses stencils and dependencies", admissibility changes (two-sided coverage no longer enforced), lattice caveat, geometry-score derivation, min-score weights.
- **Literature:** read the 2026 ASMR++ local-rewards paper (Freymuth et al., per-element reward = error reduction − cost; closest RL work) and VDGN (posthumous credit assignment) before novelty claims. Skim found no RL with adjoint-weighted per-decision rewards for a pointwise QoI.
- **Comparison extras:** classical DWR refinement baseline; GFDM quadrant/octant star baseline; λ sweep; offline vs online cost table.
- Upwind was considered as a baseline; Ali chose not to add it (ablation later).

---

## Appendix A: cell [14] code (Comparison 3, missing from the saved notebook)

```python
from math import factorial

def heat_poly(n, xi, adt):
    """T_n(xi, alpha dt) = n! sum_m xi^(n-2m) (alpha dt)^m / ((n-2m)! m!)."""
    return sum(factorial(n) / (factorial(n - 2 * j) * factorial(j)) * xi**(n - 2 * j) * adt**j for j in range(n // 2 + 1))

def rows(dxs, dts, a_loc, n_rows):
    xi, adt = dxs - a_loc * dts, alpha * dts
    h = max(np.abs(dxs).max(), np.sqrt(alpha * np.abs(dts).max()))
    return np.array([heat_poly(n, xi, adt) / (factorial(n) * h**n) for n in range(n_rows)]), xi, adt

def weights_5row(dxs, dts, a_loc):
    A, _, _ = rows(dxs, dts, a_loc, 5)
    if np.linalg.matrix_rank(A) < 5 or np.linalg.cond(A) > COND_MAX:
        return None
    w = np.linalg.solve(A, np.eye(5)[0])
    return w if np.abs(w).sum() <= W_MAX else None

def weights_minscore(dxs, dts, a_loc):
    """3 rows; the 2 free degrees minimise the T3/T4 leftover (the score) instead of the norm."""
    A, xi, adt = rows(dxs, dts, a_loc, 3)
    if np.linalg.matrix_rank(A) < 3 or np.linalg.cond(A) > COND_MAX:
        return None
    C = np.array([heat_poly(3, xi, adt) / (6 * L_SOL**3), heat_poly(4, xi, adt) / (24 * L_SOL**4)])
    Q = C.T @ C + 1e-6 * np.trace(C.T @ C) * np.eye(len(xi))     # leftover on T3, T4 (+ a tiny norm term)
    K = np.block([[2 * Q, A.T], [A, np.zeros((3, 3))]])          # minimise w'Qw  subject to  A w = (1, 0, 0)
    w = np.linalg.solve(K, np.r_[np.zeros(len(xi)), 1.0, 0.0, 0.0])[:len(xi)]
    return w if np.abs(w).sum() <= W_MAX else None

def score_long(DX, DT, w, a_loc):
    xi, adt = DX - a_loc * DT, alpha * DT
    return sum(abs(w @ heat_poly(n, xi, adt)) / (factorial(n) * L_SOL**n) for n in range(3, 7))

# The cell then swaps solve_weights/score to each variant (a stencil failing admissibility with the new weights
# keeps its 3-row min-norm weights; the share is reported), clears _w_cache/_sets_cache, runs lowest_score and the
# brute force (fixed_rule + joblib), reports points/work/error/bound, restores the originals in a finally block,
# and plots the results on top of cell 13's curves. Rerun cells 3-5 if it was interrupted (originals not restored).
```

## Appendix B: pitfalls (from all sessions)

- Weights must be computed at the **placed** (wall/IC-clamped) positions.
- Same-level or upward neighbours create loops → singular (I − W) (notebook 9's failure).
- Compare returns only within one ruler (FD-3 vs LW normalisation give different λ meanings).
- True error can be 10–50× below the bound by cancellation: always report the exact bound.
- `sed` with `\u` mangles LaTeX/Python; use the Edit tool.
- PPO updates in blocks of n_steps × n_envs = 3072 steps, so the final model is one update after the last snapshot.
- Explicit FD-5 (forward Euler) is unstable at r = 0.45 (needs r ≤ 3/8); RK4 is stable there.
- Joblib + notebook functions: pass index chunks, not `np.array(list_of_tuples, dtype=object)` (rows become unhashable arrays).
- Rerunning cell 9's tiny test overwrites the variable `model`.
