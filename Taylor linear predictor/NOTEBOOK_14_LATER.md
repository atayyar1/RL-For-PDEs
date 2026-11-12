# Notebook 14 (`14. Backward Burgers Clean.ipynb`): kept for later

Started 2026-11-02. Things Ali asked to keep for later, plus open ideas. One change at a time, so we know what changed the results.

## Kept for later
1. **Extra Taylor terms**: use the 2 free degrees of freedom of the 5-point min-norm weights to cancel higher Taylor terms (to compete with FD-5 / WENO5).
2. **State ablation**: remove the guessed u, u_x, u_xx from the state (u_xx is noisy).
3. **N_LW in the state** → a fixed number (so testing needs no per-query LW map).
4. **Characteristic guess** (no grid, from the IC and inflow data only). Cell 5b result, median relative error vs the LW guess:
   u 0.043 vs 0.011, u_x 0.092 vs 0.023, u_xx 0.47 vs 0.036, u_xxxx 1.09 vs 0.079.
   Idea: speed / mask / β / state from it (training and test), LW only for the score in the training reward → no LW at test.
   Fallback: two-pass map (build, forward pass, rebuild with the map's own values as speed).
5. **Cone estimate N̂** (points still to create) as a state feature. Cheap (~10–50 µs per point).
6. **Exploration**: `ent_coef = 0.01` instead of 0.
7. **Depth-first action**: choose the depth (1, 2, 3 levels) first, then the cells.
8. **Point budget**: fix a budget B and tune λ automatically (dual ascent). The advisor's "global" cost, exactly.
9. **"height" shaping, column version** (N_q = t_q / mean Δt), as a comparison to the cone-union version (expected to over-stop).
10. **Advisor: terminal true-error reward** at the end of the episode. Concerns: sparse, rewards query-specific cancellation, needs the true value at z*.
    If tried: a switch, −w·log10(|e|/EPS), reference from a fine FD solve, compared on the 9 unseen queries. True error is logged as a measurement anyway.
11. **Comparisons**: Richardson-extrapolated guess baseline; flops column; the stencil library reported once; bound next to error; ≥ 3 seeds.
12. **Check**: score with the guess derivatives vs the true local error (never done).
13. **λ-conditioned policy**: one policy → an error-vs-points curve.
14. **GFDM angle**: parked.

## Found on the way (2026-11-02)
- **Score bug, fixed**: the weights used the speed rounded to 0.02, the score used the unrounded u → a fake first-order term that greedy / lowest score cancelled
  against the real error (score 14×–555× too optimistic). Fix: the score uses the rounded speed (`score_derivs`). Greedy bound now 2.5e-4 vs LW 8e-4.
  Every run before this fix trained on a broken accuracy term.
- **Picard**: 1 step is within 1% of converged → `PICARD = 1`.
- **Solves**: classical LW does 0 solves, so points stay the main metric. The stencil library (~1.4M solves) is a one-time table.
- **Greedy** is not expensive per point (~0.5M flops vs ~2M for the agent's mask): the agent has to beat it on points vs accuracy.
- **Stops** (reuse of existing lower-level points): greedy + stop at (25, 75) uses 936 points vs greedy's 1359, bound 4.3e-4 (LW 8.1e-4).

15. Greedy + lookahead rule (2026-11-03, not RL): cost = |beta| s / EPS + lam new / N_LW + mu lam d(cone estimate) / N_LW.
    mu = 1 gave 16-25% fewer points than greedy at the same bound on (25,75), (12,100), (38,150) (EPS 1e-3). Parked by Ali:
    greedy (a search over ~18k stencils per point) is not the target for the one-shot agent for now.
