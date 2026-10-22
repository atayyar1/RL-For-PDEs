# Handoff 6: what was done (Oct 14–21, 2026)

A factual record of the work in this chat: what was changed, what was run, and the numbers it gave. No conclusions or plans.
Working style during the chat: Ali runs every cell himself; cells (or only the changed parts) are sent in chat; scratch copies
were used only to check cells before sending; no scratch runs while his training runs.

---

## 1. Starting point (notebook 12.3, before this chat)
- Burgers, backward map from z*, per-point reward from e(z*) = Σ β_p δ_p, MaskablePPO, window R_X = K_T = 3 (21 cells, 5-point
  stencils, 3 Taylor conditions), IC `inflow_smooth` u0 = 1 + 0.8 sin(2πx/1.5), nx = 100, EPS = 1e-5, runs in `runs10-14`
  (single query (50, 30)). See HANDOFF_5.

## 2. Test 1 on the nx = 100 runs (scratch)
At (50, 30): LW 900 pts, bound 9.7e-5, true error 9.5e-9; lowest score 1497 pts, 1.0e-5; greedy 528 pts, 9.0e-6;
the four trained agents 878–1951 pts, bound 8.1e-5 – 9.4e-5. Timing: greedy 0.36 ms/point (cached stencil table, scoring
≈ 0.03 ms/point), one network call ≈ 0.22 ms. Skew slopes 3–9 (Courant ≈ 0.04 cells/level).

## 3. Notebook 12 "reach" ablation (scratch)
Notebook 12 agent (`runs10-7/burgers_single_z50-20_lam1_R3K3`) at (50, 20): full state −1.02, 242 pts, bound 6.0e-5;
reach shuffled or removed −1.55, ~240–258 pts, bound 1.3e-4; LW −2.00, 400 pts, 1.4e-4.

## 4. Grid nx = 50, larger t*, live speed
- nx = 50 (dt = 1.818e-3, nt = 276), K_COARSE = 3 (guess 629 pts).
- With the guessed speed, lowest score / greedy had est/bound ≈ 0.00 and no gain over LW at t* ≥ 100. Diagnosis at (25, 100)
  (exact values only for the diagnosis): exact speed alone → greedy bound 3.4e-4 (LW 1.6e-3); exact derivatives alone → no change.
- Added the live-speed forward pass: `evaluate(m, picard=1)` re-solves each point's weights with a(û_p); weights kept in
  `m["w_used"]`; `report` recomputes β with them (identity checked exact). Rules at (25, 50): greedy error 2.2e-5 vs LW 3.9e-4.
- EPS changed 1e-5 → 1e-3. Change in u at x* for it* = 30/50/100/150: max over x 28/43/74/101 % of amplitude.

## 5. Old IC (sine), nx = 50: single-query runs
- `runs10-15/inflow_smooth_scratch_gae0.97_z25-50_lam1-1` (seed 1, 20M, λ = 1, lr 3e-4, gae 0.97, ent 0): final −1.47,
  1882 pts, bound 1.4e-4 (true error 1.1e-6) at (25, 50).
- Test 2 (11 unseen queries it* 30–100 × ix* 12/25/38): agent pts ×1.08, bound ×0.30, error ×0.16 vs LW (10/11 wins);
  agent's top stencil (5-point row one level down) ×0.27 / ×0.13; best fixed (picked by exact bound) ×0.27 / ×0.11;
  fixed stencil picked by the estimate ×0.89 / ×0.79; lowest score bound ×1.26, error ×0.15.
- Stencil probe: the agent's choice between its 2 main stencils was explained at 93 % by `exists[(1,-1)]`; 12 distinct
  per-cell state patterns over 1294 points; agreement with slope sign 72 %; "slope rule" / centred / shifted row bounds similar.
- Classical comparison (FD-3, LW, CN, FD-5+RK4, WENO5+RK3, grids 1–4×, same boundary data, WENO ghosts from u_true) at
  (25, 50), (38, 75), (25, 100): agent 1e-6 – 8e-5; FD-3/LW/CN 7e-4 – 2e-3; FD-5/WENO5 4e-8 – 1.6e-6 with 4–7× the points.
- Green's-function check: β on the IC row vs the adjoint. First reference used φ_τ = u φ_x + α φ_xx (mass mismatch 0.73 vs 0.88);
  corrected to the conservative form φ_τ = (u φ)_x + α φ_xx (adjoint of the frozen-speed operator). Plot agent vs exact sent;
  LW-overlay version sent.
- Seeds: seed 2 (`runs10-20/..._seed2`) −1.61, 2110 pts, bound 1.4e-4, error 9.7e-5; seed 3 stopped early (2 ckpts);
  seed 4 (`runs10-20/..._seed4`) −1.47, 1863 pts, bound 1.3e-4, error 1.0e-4.
- Other runs on this IC: λ = 5 (`..._ent0_z25-50_lam5-5`, 8M): ~2080 pts, −6.2 (LW at λ=5 ≈ −5.7).
  DAgger clone of greedy (`runs10-15/inflow_smooth_imitation_z25-50_lam1`): clone 715 pts, bound 1.3e-3, depth 1/2/3 = 6/25/68 %;
  + PPO 5M: −1.40, 1590 pts, bound 2.9e-4, depth 82/6/12 %. Anchor (PPO + fading BC) cell written, not run.
- Multi-query from scratch, 15 queries it* 40–80 × ix* 15/25/35, lr 3e-4→0 (`runs10-19/inflow_smooth_multi15_gae0.97_lam1-1`,
  stopped ~16M): single vs multi table — multi 2M wins 2/14 (training queries), 2/11 (unseen); multi 16M 0/14, 0/11.
- A front-problem multi run exists (`runs10-17/inflow_front_multi15_...`, α = 0.01, trained by mistake on PROBLEM = inflow_front):
  agent ≈ LW (−5.36 vs −5.26 at (25, 75)); lowest score / greedy estimate B ≪ exact bound there.
- Depth test (12.3.1, old IC, rules only): best uniform stencils with all cells ≥ 2 levels down (and ≥ 3) vs LW and agent
  (filter bug: "depth 1" allowed all cells and picked the depth-2 stencil). At (25, 50): depth 2 1451 pts / 9.8e-5,
  depth 3 818 pts / 1.7e-4, LW 1824 / 3.9e-4; at it* = 100–150 depth 3 saves 10–25 % points.

## 6. New IC "D" (peak at the middle)
- u0 = 1 + 0.8 cos(2π(x − 0.5)/1.5), F0 = x + (0.8·1.5/2π)(sin(2π(x−0.5)/1.5) + sin(2π·0.5/1.5)); u ∈ [0.6, 1.8].
  Checks passed (FD ÷4, score corr 0.98, guess errors). Change at x* = 0.5: −0.23/−0.48/−0.66 at it* 50/100/150.
- Candidate ICs compared (sign-changing A, zero-at-middle B, peak-middle with sign change C, D); D chosen.
- In 12.3 the `else:` branch of the problem cell now uses IC D (also affects `inflow_front`).
- Overnight 3-seed run on IC D at (25, 75) started (`runs10-20/inflow_smooth_peak0.5_scratch_z25-75_lam1-1_seed1`) and stopped at 3M.

## 7. Same-level neighbours (12.3)
- `SAME_LEVEL = k` adds cells (−k..−1, 0) and (1..k, 0). No linear solve: cycle mask (`forbidden`, `reaches`), β passed down
  per level (`close_level`, `users_first`), values in dependency order (`forward_order`), accuracy reward paid when a level
  closes, `allowed_sets` cached per point, `pick_mask(m, p, picks)`, rules use `allowed_sets`. Identity checked exact; 73k
  admissible stencils per key; ~1.5 GB/worker; 0.9 ms/pick after caching.
- Run `runs10-21/inflow_smooth_peak0.5_same3_z25-75_lam1-1_seed1` (5M, 8 envs): best −2.99 at 250k, final −4.97, 3579 pts,
  bound 3.4e-3, same-level left/right 684/1189; map spans the full width at every level (48 × 75). Rules: lowest score
  left/right 1300/1763, greedy 497/1754. Stencil-usage figure made (same-level use only at the sides).
- Baseline `runs10-21/inflow_smooth_peak0.5_same0_z25-75_lam1-1_seed1` (SAME_LEVEL = 0, 5M): final −2.39 (snapshot),
  bound 2.2e-4; re-evaluated: −2.45, 3258 pts, bound 2.8e-4, error 1.4e-4.

## 8. Notebook 12.3.2 (built and executed in this chat)
`12.3.2 comparisons for single query ran in 20-10.ipynb`: IC D setup (11 cells, SAME_LEVEL switch = 0), then for the same0
baseline run: (1) training curve + table at (25, 75); (2) 12 queries it* 50/75/100/150 × ix* 12/25/38 with change and slope —
agent bound ×0.88 (7/11), error ×0.29 (10/11), points ×1.07; agent's top = best fixed = 5-point row: ×0.33 / ×0.33 (11/11, 10/11);
lowest score error ×0.11; (3) stencil-usage map (54 % centred row, 20 % right-shifted row); (4) classical solvers at (25, 75),
(12, 100), (38, 150): agent 1.4e-4 / 9.5e-5 / 4.1e-4, FD-3 2.7e-4 / 1.3e-3 / 1.4e-3, FD-5/WENO5 3e-8 – 2e-6.
Backup of the version before: scratch `backup_12.3.2.ipynb`.

## 9. 12.3.1 additions sent (old IC)
"Test 2 over seeds" (seeds 1, 2, 4), "error vs points: 3 seeds vs second-order schemes" (points to match FD-3/LW/CN), and
"stencils used by each seed". A `NameError`/`TypeError` occurred because an earlier cell overwrote `a`; fix: `a = lambda u: u`.

## 10. State features (12.3)
- `STATE_X = True` adds per window cell: ξ_q/dx and signed log of R_q/dx³, with R_q = P_p(Δx_q, Δt_q) − u_x ξ_q − u_xx(ξ_q²/2 + αΔt_q)
  (checked: |Σ w_q R_q| equals the score to 3e-10). N_OBS = 117.
- Run `runs10-21/inflow_smooth_peak0.5_same0_stateX_z25-75_lam1-1_seed1` (5M, default PPO): snapshots between −2.33 (bound 1.4e-4)
  and −15.5; final −4.83, 3285 pts, bound 2.8e-3, error 2.3e-3.
- Run `..._stateX_stable_...` started with PPO_KW = n_steps 4096, batch 4096, n_epochs 5, lr 1e-4, clip 0.1, target_kl 0.02:
  at 1.5M snapshots −4.94, −15.41, −9.48, −4.36, −3.72, −8.62 (running at the time of this handoff).

## 11. Discussions held (topics only)
Green's function / adjoint interpretation of β; per-point vs one-stencil behaviour; whether greedy is a baseline; cost per point;
multi-query vs single-query; finer grid as a convergence study; depth / level skipping; continuous-adjoint reward for a forward
(particle) formulation; a step-size formulation (agent chooses τ, neighbours placed at the characteristic foot with spread √(2ατ));
hp-adaptivity (order and depth as actions); candidate-library feature; advection notebook with sign-changing speed.
