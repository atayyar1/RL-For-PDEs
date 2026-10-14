# Handoff 5: notebook rebuild, new reward/state, and what training showed (Oct 13–14, 2026)

Ali rebuilt notebook 12 cell by cell in a new notebook. Every cell below was tested in a scratch copy on all
relevant problems before being sent. Keep cells simple (Ali's rule: no code that isn't needed).

## 1. The pipeline (cells, in order)

| cell | content | key decisions |
|---|---|---|
| 1 | problem, grid, exact solution, boundary data | `PROBLEM` switch: `"walls"` (old), `"inflow_smooth"` (u0 = 1 + 0.8 sin(2πx/1.5), α = 0.1, T = 0.5, **working problem**), `"inflow_front"` (α = 0.01, T = 1.1). Time-dependent **Dirichlet** data from the exact solution (inflow left, outflow right). dt from **2r + ν² ≤ 0.9** (LW was unstable with the old rule). Exact solution: Hopf–Cole as a weighted average (cosine series fails at α = 0.01), checked by FD convergence (÷4) and against the cosine series (≤ 5e-14 at α = 0.1). `U_row/Ug` = cached exact rows. `g_left/g_right` = cubic spline of wall data. |
| 2 | Taylor predictor (`solve_weights_many`, `solve_weights`) | unchanged; `fd_reference` removed (baselines live in one comparison cell). |
| 3 | score (`taylor_coeffs`, `score`, `x_derivs`) | unchanged; check on random points anywhere, colour = |u_x|: corr 1.00 smooth, 0.95 front. |
| 4 | coarse guess | coarse LW with the boundary data, **quintic spline** in x (sine series rang at fronts), same stability rule; **K_COARSE = 4** (smooth), 2 (front). |
| 5 | map machinery | `known_value = Ug` (non-zero walls); `weights` solves in sorted order (pick-order crash fix); identity e = Σβδ exact; forcing u = 0 at walls makes the error 70–500× worse (walls really used). |
| 5b | map-level estimate check | Σ|β|s vs true Σ|βδ|: 0.97–1.06 for LW/random maps in smooth problems; 0.12–0.8 on the front (speed error, see §3). |
| 6 | window, `admissible`, `pick_mask`, `lowest_score`, `report` (uses `Ug`) | unchanged otherwise. |
| 7 | environment `MapEnv` | reward **r_p = −|β_p| s_p / EPS − λ·(new points)/N_LW** (linear, so it sums to the bound; no log). λ drawn log-uniform per episode from `lam_range` and put in the state (ASMR++ idea); `n_lw(z)` = LW map size. State (75): picked/exists/known per cell + log|β|, height/z*, offset/nx, 2 wall distances, guessed u, u_x, u_xx (L_SCALE = 0.1), points used, pending count, pending influence, log λ. **No score/reach/regret** (no search). |
| 8 | `plot_map` | left: map coloured by |β|; right: exact u(x,t*), IC, computed points of top levels, exact vs predicted at z*. |
| 9 | `make_env(queries, lam_range, seed)`, `run_policy(model, z, lam)`, callbacks | λ fixed at evaluation via `lam_range=(lam, lam)`. |
| 10 | scan (`score_sum`, `greedy_rule`, EPS) | at z* = (50, 30): EPS = 1e-6 from the lowest-score map; greedy balance at λ ≈ 1. **Note:** the scratch/imitation runs used the provisional EPS = 1e-5 (scan cell was not run). |
| 11 | training (single query, from scratch) | `skew_slope(m)` = lean along the characteristic. |
| 12 | imitation (DAgger from greedy) + PPO fine-tune | `MapEnvX` adds u_xxx, u_xxxx. Rename trap: never name a helper `evaluate` (overwrites the forward pass). |

## 2. Training results (inflow_smooth, z* = (50, 30), λ = 1, EPS = 1e-5)

| run | return | points | bound | note |
|---|---|---|---|---|
| untrained | −31.3 | 1547 | 3.0e-4 | random admissible picks; **skew already 0.94** (admissibility favours upwind) |
| scratch, γ = gae = 1, 8M | −10.8 | ~1600 | 9.0e-5 | plateau from ~1M steps, 982 maps |
| scratch, gae 0.95, 3.5M | −11.2 | ~1700 | 9.2e-5 | same plateau → not credit assignment |
| imitation (4 DAgger rounds, 42k states) | −15.4 | 499 | 1.5e-4 | P(greedy's pick) stuck at **0.50** with 5× more data |
| + PPO 3M (lr 1e-4, clip 0.1, gae 0.95) | −10.4 | 878 | 9.4e-5 | drifted to LW-level; skew 8.8 (odd) |
| LW map | ≈ −10.7 | 900 | 9.7e-5 | |
| greedy (internal gauge only) | −0.71 | 528 | 9.0e-6 | Ali: greedy is NOT a comparison (too expensive) |

**Reading:** the agent learns to avoid bad stencils and to control the point count, but not which 5-cell combinations
are accurate (accuracy needs exact cancellation of Taylor terms: all 5 cells right). Earlier good results
(notebooks 10, 12) all had score-derived features in the state; removing them is the main cause of the drop.
Training longer is not the fix (flat curves; imitation saturated).

## 3. Findings to keep in mind

- **Score exploitation:** optimizing the score finds stencils whose *guessed* Taylor terms cancel (fake-perfect):
  estimate/true 0.002–0.05 for lowest-score maps. Finer guess (smaller K) fixes the *ranking* (lowest-score maps
  7–14× better than LW at t* = 100–150 with K = 2) but not calibration. Two-guess max made maps unstable (rejected).
  **Never** use the exact error in the reward (Ali: defeats the purpose).
- **Front problem:** also needs a speed fix (weights use guessed speed): double pass (reweight with computed speed)
  or a speed-sensitivity term in the score. Deferred.
- **Physical time:** in inflow_smooth, it* = 30 is t ≈ 0.014 (nothing moves). Proposed: **nx = 50** (dt 4× larger,
  maps half as wide); not yet done, needs re-running the checks (K probably 2 then).
- **True error vs bound:** error at z* is often ≪ bound (sign cancellation, e.g. LW error 9.5e-9 vs bound 9.7e-5);
  judge maps by the bound, report both.

## 4. Next steps (agreed direction, Oct 14)

1. **Is the agent learning anything beyond "safe stencils"?** shape check (are its stencils one fixed scheme?),
   ≥ 3 seeds, held-out queries, and a state-blind control (state zeroed: does it reach the same level?).
2. **λ sweep** (one run with λ sampled in (0.1, 10) + evaluation at 5 λ): prediction: points follow λ, bound stays ~1e-4.
   Compare with classical refinement curves (LW, FD-3, CN, FD-5+RK4, WENO5+RK3) on accuracy vs points.
3. Comparison cell for the new problems (classical schemes with the boundary data; WENO ghosts need the data).
4. Ideas from the literature if accuracy must improve without search:
   **A. per-candidate scoring** (network scores each candidate stencil from its own moments/features, one decision
   per point; Gasse et al. 2019, Kool et al. 2019); **B. learned continuous weights** on fixed/candidate stencils
   (RL-WENO: Wang, Shen, Long, Dong 2020; Bar-Sinai et al. 2019): the 2 free weights of a 5-point stencil
   (min-score weights gave ~10× before).
5. Grid change nx = 50 for more physical time per level.
