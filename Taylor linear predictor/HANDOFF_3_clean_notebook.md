# Handoff 3: notebook 10 results, next step is the clean notebook (2026-10-02)

Read this first. It replaces `HANDOFF_next_session.md` and `HANDOFF_2_next_session.md` for current status; those two are still the history.
Supporting files in this folder:
- `local_reward_formulation.md`: the design doc (math of the identity, the score, the MDP).
- `where_we_are.md`: plain-language glossary and step-by-step explanation, written for the user.
- `notebook10_verified_cells/`: the verified source of every notebook 10 cell (copy from here, don't rewrite).

## The task for the new session

Write a **clean, presentable notebook for the advisor (Joseph Bakarji)**, from scratch, reusing the verified cells. The user builds it by pasting **one small cell at a time**. Number every cell, run it first and give the expected output, explain it in plain language, keep it simple, no fancy extras (see memory: cell-by-cell teaching style). Follow the thesis PDF's chapter order and notation, so it reads as the next version of `RL_for_PDEs__Ali_Tayyar.pdf`. Long training runs happen on the user's remote 12-core machine.

## The story the notebook should tell (agreed)

1. **Predictor.** Taylor rows with the PDE substituted (thesis §2.5), min-norm weights, admissibility: rank 3, cond ≤ 1e4, ‖w‖₁ ≤ 2, causal (earlier-time) neighbours only (ali: debatable, might change later). Use the **corrected** third row ½(Δx − cΔt)² + αΔt (3 points one level down = Lax–Wendroff exactly). The thesis PDF still shows the old row in eqs. 2.35–2.36 and §3.3.2; that needs fixing there.
2. **Backward map + exact identity.** Build the map from z* down to the IC/walls (latest-t-first heap). Influence β passed down: β_q += β_p·w_q. Exact identity: error(z*) = Σ_p β_p δ_p (verified to 1e-16). This is the adjoint/DWR representation (Becker & Rannacher 2001), the exact version of thesis Corollary 2.7.
3. **Geometry score (no exact solution).** s_p = |Σw T₃|/(6L³) + |Σw T₄|/(24L⁴), with heat polynomials T₃ = ξ³ + 6αΔt ξ, T₄ = ξ⁴ + 12αΔt ξ² + 12(αΔt)², ξ = Δx − cΔt, L = 1/(2π). It tracks the true local error (corr. about 0.75) and map totals track the true error at z* (corr. about 0.6–0.7).
4. **RL environment.** 5 picks per stencil, mask = admissible completions, reward per completed stencil −|β_p| s_p/B_ref − λ·new points/N_ref, forced points auto-played. State: window flags + reachable score per cell + β, level, offset, picks, points used + pending workload.(Ali: I am not sure what this is, we need to discuss)
5. **Training: failure, diagnosis, fix.** From scratch only 1/6 seeds beat the heuristics; failed runs drift to about 750 points (late costs of new points mis-credited, since GAE 0.95 looks about 20 steps ahead). Fix: pending workload in the state + `gae_lambda=1.0` → **3/3 seeds** (z* = (50,20), λ = 8: returns −7.29 to −8.10, 319–336 points vs FD 400, error bound 5–8e-6 vs FD error 5.2e-5, error spread evenly: 0.5–0.7 decades, 1–2% wasted points).
6. **Generalisation.** One multi-query model (15 training queries, t* 10–30) beats the heuristics on 6/6 unseen queries incl. t* = 35, 40; it discovered a **staggered/checkerboard grid** (98% of points on z*'s parity sublattice, about 0.52× FD's points).( Ali: also, not sure what this is)
7. **Honest check: brute force over fixed stencils.** On this homogeneous PDE the optimum is ONE stencil everywhere (the state is almost identical at every interior point, and the physics is uniform). Small window (240 stencils): PPO = the checkerboard stencil (rank 3); the best fixed stencil beats it. 21-cell window (|dix| ≤ 3, 3 levels down, 18,694 stencils, reward normalised by FD): PPO λ = 1: 305 points / bound 1.0e-5 (beats FD-5 on work and bound; vs FD-3: 5× lower bound, about 27% more work), but **75 fixed stencils beat it**. The best fixed stencil: 168 points, work 840, bound 1.1e-6 (FD-3: 400 / 1200 / 5.2e-5; FD-5: 780 / 3900 / 1.4e-5). The winners jump 3 levels (the F6 lesson: longer jumps win) (ali: everything related to the results needs to be checked again because I am not sure about the idea we are pushing for, here is the thing, I am not sure what benchmark we should test against, also, in these results we didn't do the videos we did before (how we started the training and how we are now, and how the integration is happening)).
8. **Long-time stability (the headline for the method).** The plain lowest-score rule (no RL) in the 21-cell window stays controlled to t* = 1000: bound 1.1e-5 vs FD-3 6.3e-4 (57× lower, about 1.7× the work) and vs FD-5 2.2e-4 at equal work (20× lower). At t* = 300: 3.9e-6 vs 5.8e-4 / 1.1e-4. Error grows slowly, with no geometric blow-up (cf. thesis ch. 5).
9. **Limits / next.** PPO trained at t* = 60–100 failed (episodes of 20–30k picks, only 669 episodes in 20M steps; bounds about 10× worse than FD-3 at t* = 300/1000). RL must be justified on inhomogeneous or nonlinear problems (walls, variable coefficients, Burgers / Buckley–Leverett) with **solution information in the state**, a **local score length scale**, and iterative build/evaluate (thesis Remark 2.10 needs values). Before more RL: a cheap "room test" (region-by-region best stencil vs one fixed stencil). (Ali: again what we said about the results and the interpretations of the old session, we need to just do the notbook now)

Always report **points, work (points × neighbours) and exact bound Σ|βδ|** (no cancellation luck), and include FD-3, FD-5, the heuristics and the best fixed stencil as baselines. Note that weight computation (pinv) is a real cost unless cached per geometry.

## Things that tripped us up (don't repeat)
- Weights must be computed at the **placed** (wall/IC-clamped) positions.
- Same-level or upward neighbours create loops → singular (I − W) (notebook 9's failure)( Ali: might touch on later).
- Compare returns only within one normalisation (reference map vs FD normalisation give different scales).
- The true error can be 10× smaller than the bound through cancellation: always show the exact bound.
- `sed` replacement strings with `\u` mangle LaTeX/Python; use the Edit tool.

## Saved models (on the remote machine's notebook folder) (Ali: lets have things more organized, lets have a file that has everything we need to save. and for the figures we need to discuss it along the way)
`local_reward_pending_gae1_50_20_lam8_seed{1,2,3}_*.zip`, `local_reward_multi_lam8_seed1_*.zip` (small window); `local_reward_big21_multi_lam{1,4}_seed1_*.zip`, `local_reward_big21_long_clip_lam1_seed7_*.zip` (21-cell window). Loading a small-window model needs the 10-cell window definitions (don't run Cell 32 in that session).

## Other open threads (not for the notebook)
- TU Delft visit (Voskov): research plan `Desktop/PhD 2027/Documents Needed/Research Plan TU Delft v2.tex` (2 pages, compiles). Ties: cascade = a space-time stencil, β for error accumulation, OBL for f′(S) in Buckley–Leverett.
- Literature check still pending: has anyone used an adjoint-weighted per-decision RL reward for goal-oriented point placement? (Ali: lets check this now(before anything))
- Thesis PDF updates: backward construction, "chooses stencils/dependencies", mesh-free caveat (lattice), admissibility changes (two-sided coverage no longer enforced), the u_xx row. (Ali: bakarji cares about this really, we need to report everything)
- Ali: lets add to git ignore the zip files the other sessions did.