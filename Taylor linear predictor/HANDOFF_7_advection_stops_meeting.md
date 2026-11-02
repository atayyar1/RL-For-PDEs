# Handoff 7: sign-changing advection, stops + shaping, the Oct 27 meeting (Oct 21 – Nov 2, 2026)

Written at the end of a long chat that was compacted once. To limit errors from the compaction, every number below
marked **[nb]** was re-read from the saved notebook outputs or run configs on Nov 2; numbers marked **[chat]** come from
the chat notes only and should be re-checked before they go into the thesis. Earlier work is in HANDOFF_4 (Burgers),
HANDOFF_5 (rebuild) and HANDOFF_6 (Oct 14–21: IC D, same-level, state features, 12.3.2).

Working style (unchanged): Ali runs every cell himself; cells (or only the changed lines) are sent in chat; scratch copies
are only for checking a cell before sending; no scratch runs while his training runs; keep cells minimal; never use the
exact solution in a reward; greedy is an internal gauge, not a comparison (too expensive: ~18k stencils scored per point).

---

## 0. The method in one paragraph (for a fresh session)

Query z* = (x*, t*). Starting at z*, each point p picks a 5-point stencil from a 21-cell window below it
(|Δix| ≤ 3, 1–3 levels down); weights from 3 Taylor rows (value, u_x, u_xx, with u_t substituted from the PDE), min-norm,
admissible if cond ≤ 1e4 and ‖w‖₁ ≤ 2. This continues down to the IC and the walls (known data). Values are then computed
forward on this DAG only. The influence β (discrete adjoint) makes the error identity exact:
$$e(z^*) = \sum_p \beta_p\,\delta_p .$$
Per-point reward: $r_p = -|\beta_p|\,s_p/\mathrm{EPS} - \lambda\,(\text{new points})/N_{LW}$, with $s_p$ the Taylor
score built from a coarse LW "guess" (run in the background, only to build the map). MaskablePPO, one action = one window cell.
PPO settings in 12.4 and 12.5 **[nb]**: `n_steps=1024, batch_size=1024, gamma=1.0, gae_lambda=0.97, lr=3e-4, ent_coef=0`, 8 envs.

---

## 1. Notebook 12.4: sign-changing advection–diffusion (`12.4 Robust Backward advection.ipynb`)

**Why.** On smooth Burgers (12.3.1/12.3.2) one fixed stencil matched the agent ("found one good stencil"). Needed a
problem where no single stencil works.

**Setup [nb].**
- $u_t + a(x)\,u_x = \alpha u_{xx}$, $a(x) = -3(x-0.5)$ (flow converges to x = 0.5, characteristics lean opposite ways
  on each side), α = 0.01, T = 0.5, nx = 50, nt = 56, dt = 9.09e-3, ν_max = 0.67, 2r + ν² = 0.88.
- IC 1 + 0.8 cos(2π(x − 0.3)/1.5) (peak at 0.3, run folder says `peak0.3`). Exact closed form (Ornstein–Uhlenbeck);
  PDE residual of the closed form 1.2e-6; FD converges ÷4.
- Changes needed: score Taylor terms by **total degree ≤ 4** (advection-dominated, Δt ~ Δx; corr 0.71 → 0.98), a′ term
  in the u_x weight row, `speed(x)` replaces `a`. Coarse guess 25 × 24 = 600 points. EPS = 1e-3, STATE_X = False.
- Errors hit while building: queries with it ≥ 56 (IndexError / BrokenPipe in SubprocVecEnv), EPS dict KeyError,
  STATE_X = True unstable → off.

**Run [nb].** `runs10-21/advection_peak0.3_same0_state0_stable_z25-45_lam1-1_seed1`, single query (25, 45), λ = 1,
5M steps (25 min) + continued to 10M (`ppo_final_10M`). Snapshot return −2.67 at 5M, −2.6 to −2.9 over 5–10M
(no further gain), ~1780 points.

**Comparisons [nb]** (9 unseen queries, geometric mean vs the LW map of the same query):

| method | points | bound | true error | wins |
|---|---|---|---|---|
| lowest score | ×1.15 | ×0.17 | ×0.08 | 9/9 |
| greedy | ×0.47 | ×0.18 | ×0.06 | 9/9 |
| best fixed stencil | ×1.15 | ×1.81 | ×1.48 | 3/9 |
| agent | ×1.21 | ×1.94 | ×1.06 | 5/9 |

- **No single stencil works here** (best fixed is worse than LW), while greedy is ~16× better: per-point adaptivity is
  genuinely needed on this problem. That is the point of 12.4.
- The agent is good at its training query (**[chat]** ~10× better than LW; re-check in P1 of 12.4) but **does not
  generalise** (unseen ≈ LW).
- Classical at (25, 45) **[chat, from a pasted table]**: LW map 1584 pts / 2.2e-3; lowest score 1.4e-4; greedy 772 pts /
  3.2e-4; FD-3 6.2e-3; LW 2.3e-3; CN 9.4e-3; FD-5+RK4 8268 pts / 1.2e-5; WENO5+RK3 10683 pts / 1.6e-4.
- Figures: P4 stencil usage with backward characteristic, P5 lean/depth per column, P6 Green's function.
- Green's function: the β on the IC matches the adjoint, but this is **forced by the Taylor conditions** (mass, centre,
  width are built in) — not evidence of learning. Observation that led to the next step: every method's map fills
  the whole domain, while the Green's function is a narrow band → most points have negligible influence → "stop".

## 2. Cloud exploration session (side work, kept separate; not merged)

Report: `C:\Users\amt35\Downloads\EXPLORE_REPORT.md`, branch `claude/nifty-lovelace-2ucxln`. Ali asked to keep this work
on the side. Main findings (from the report):
- **Stop rule** (a point takes the guess value when |β_p|·ĝ_p is small; identity stays exact). Rules only, no training.
  Advection, 9 unseen: LW + stop ×0.44 points at ×1.00 error; greedy + myopic stop ×0.24 points / ×0.06 error (greedy alone
  ×0.47 / ×0.06); fraction of points inside the Green band 0.40 → 0.80. Reuse stops (Ali's KD-tree idea) → greedy ×0.20 / ×0.09.
- Burgers **front** (α = 0.01): greedy is worse than LW (×1.87 error; the Taylor score is wrong there in both directions);
  greedy + stop ×0.38 / ×1.97; LW + stop ×0.53 / ×0.99. 3 Picard steps halve the error for free.
- PPO with a stop action learns **0 stops** without help; with **potential-based shaping** it learns to stop
  (advection (25, 45): 517 points, 141 stops, error 7.1e-4 vs greedy+stop 535 / 3.5e-4). Unseen: ×0.07 points, ×0.98 error.
  Multi-query shaped training over-stops (units problem). Shaped PPO on the Burgers front: 0 stops.

## 3. Notebook 12.5: smooth Burgers + stop action + shaping + Picard 3 (`12.5 Robust Backward Burger stops.ipynb`)

Built by me (30 cells, from scratch `nb125/build.py`), on top of 12.3 / IC D (peak 0.5), inflow_smooth, α = 0.1, nx = 50, nt = 276.

**What changed [nb].**
- Cell 8: two more coarse guesses (K = 3, 6, 12 → 629 + 117 + 30 points). Guess error estimate
  $\hat g_p = |g - g_2|/(R-1)$ with R observed from $g_2$ vs $g_4$: **R_OBS = 2.09**. Check: estimated/true guess error
  median 1.79, corr(log) 0.93.
- Cell 10: `stop_point(m)`, `build_map` treats `None` as stop, `evaluate(m, picard=3)` (stopped points take the guess value).
  Identity check with stops exact: at (10, 40) LW + stops (|β| < 1e-3) 660 points, 30 stops, e = Σβδ = −5.698898e-4.
  3 Picard steps at (10, 40): error 6.66e-4 → 2.91e-4.
- Cell 15 (env): STOP = extra action (allowed before the first pick of a point, never at z*); stop reward
  $-|\beta_p|\hat g_p/\mathrm{EPS}$; state + log of the current stop cost (N_OBS = 76).
  Shaping: $\Phi = -\sum_{q\ \rm pending}\big(|\beta_q|A_q/\mathrm{EPS} + \lambda N_q/N_{LW}\big)$ from the LW map's
  cost-to-go; each reward gets $\Phi(s')-\Phi(s)$; the extra terms sum to $-\Phi(s_0)$ (checked: difference 1.7165 = −Φ(start)).
  Return formula with stops: points + stops − 1.
- For evaluation `run_policy` must use `shaping=False` (else snapshot returns are offset by −Φ(start)).

**Runs [nb]** (5M steps, single query (25, 75), λ = 1): `runs10-22/inflow_smooth_peak0.5_stop_shaping{1,0}_picard3_z25-75_lam1-1_seed1`.

| at (25, 75) | return | points | stops | bound | true error |
|---|---|---|---|---|---|
| agent, shaping | −2.91 **[chat]** | 1674 | 109 | 1.5e-3 | 1.2e-3 |
| agent, no shaping | −3.25 | 3264 | 0 | 1.7e-3 | 3.8e-4 |
| LW map | −2.45 | 3024 | 0 | 8.0e-4 | 7.8e-4 |
| lowest score | −1.05 | 3178 | 0 | 1.2e-3 | 6.7e-5 |
| greedy | −0.53 | 1349 | 0 | 1.1e-3 | 6.1e-5 |

No-shaping run: stopped a little early (29 stops at 1.75M) then 0 stops from 2.25M on; last 1M return drifting down
(−3.13 → −3.56).

**9 unseen queries [nb]** (it ∈ {50, 100, 150} × ix ∈ {12, 25, 38}; vs LW map):

| method | points | bound | error | wins | stops/map |
|---|---|---|---|---|---|
| lowest score | ×1.06 | ×2.21 | ×0.09 | 9/9 | 0 |
| greedy | ×0.46 | ×2.18 | ×0.09 | 9/9 | 0 |
| agent, shaping | ×0.53 | ×3.34 | ×3.07 | 0/9 | 164 |
| agent, no shaping | ×1.10 | ×3.42 | ×0.62 | 6/9 | 0 |

Reading: shaping is what makes the agent stop; the **point saving generalises, the accuracy does not** (stops too
aggressively at unseen queries). P3 figure: footprint kept on the upwind side of the backward characteristic, stops on
the downwind side; the top stencils are ordered around the characteristic (right-shifted left of it, centred on it,
left-shifted right of it) **[chat, from the figure]**. This is the best "physical structure" evidence so far, and unlike
the Green's-function match it is not forced by consistency.

Options left open: continue the shaped run (it was still improving), or lower λ (e.g. 0.3) so stops are cheaper relative to accuracy.

## 4. Meeting with advisor, Oct 27

First meeting since the Burgers work started. A full bullet list of the work since Oct 8 was prepared for it (in the chat).
Outcome Ali reported: **the points cost in the reward should be global, not local.** Ali's framing: accuracy reward stays
per step (local), and the point cost is charged once at the end from the total number of points (global), so a map with
good accuracy but 3000 points gets a bad final reward. Ali's second idea: use the average time jump of the stencils,
avg(Δt), as a global measure. Ali wants to try both.

Discussion so far (my view, not yet agreed):
- The per-step new-point costs already sum exactly to the total: $\sum_{\rm steps}\lambda n_{\rm new}/N_{LW} = \lambda N_{\rm total}/N_{LW}$.
  So the terminal version has the **same return and same optimal policy**; it only changes when the agent hears about the cost.
- **Correction of something I said in chat:** I claimed PPO used γ = 0.99, so a terminal cost would be discounted to ~0.
  That was wrong — the runs use **γ = 1.0** (checked in 12.4/12.5 training cells, and HANDOFF_5 shows γ = 1 earlier too).
  With γ = 1 the terminal cost is not discounted away; it reaches earlier picks through the value function, while
  gae_lambda = 0.97 limits direct credit to ~30 steps. So a terminal cost is a fair experiment; the expected effect is
  noisier credit assignment, not invisibility.
- What the current reward misses is the future cost of a pick (a deep stencil and a shallow one both "cost 5 new points"
  now, but the deep one saves whole levels later). Ways to give that signal without changing the objective:
  potential-based shaping, either the LW cost-to-go of 12.5 or a height estimate $\hat N_q \propto t_q/\overline{\Delta t}$
  (Ali's avg(Δt) idea used as the potential). Using avg(Δt) directly as a reward term changes the objective (could be
  gamed by deep stencils, held back only by the score term).
- Proposed comparison at (25, 75), same budget: (a) current reward, (b) terminal points cost only, (c) Δt-height potential.
  Nothing built or run yet.

## 5. Other things discussed in this chat (no code)

- Literature: backward point evaluation exists stochastically (walk-on-spheres, Feynman–Kac, branching diffusions,
  multilevel Picard); a deterministic backward stencil DAG with learned stencil choice is not known to us (not a formal search).
- DWR (Becker & Rannacher 2001, Acta Numerica; Giles & Süli 2002): forward solve → adjoint (β) → indicators β × residual →
  refine mesh → repeat. Our identity is the same; we use it to build a single-query backward graph and choose stencils /
  stops, in one backward pass, instead of refining a global mesh after a full solve.
- Rejected by Ali: continuous velocity action ("tried, restrictive"); adjoint self-correction ("our problem is the point
  choice, not the error"); LW + stop / greedy + stop notebooks (dropped for now).
- Ali's idea to keep in mind: stopping mid-map using a KD-tree of nearest already-computed neighbours (reuse stops).
- 12.3.1 = comparisons for the Oct 15 single-query agent on the old sine IC (inflow BCs, no reach, nx = 50, live speed);
  12.3.2 = the same comparisons on IC D with the (25, 75) agent. Details in HANDOFF_6.

## 6. Open questions / next steps

1. Reward redesign from the meeting: build (b) and (c) above in 12.5 (or a copy), keep the exact points count.
2. Stops generalise in points but not accuracy: lower λ, or train on a few queries, or continue the shaped run.
3. Advection (12.4): add stop + shaping there (the cloud report says rules + stop already halve points at the same error).
4. Higher-order stencils (more Taylor rows) if we want to compete with FD-5 / WENO5 in accuracy.
5. Front problem (α = 0.01): the Taylor score is unreliable there; stops still help LW. Parked.

## 7. Files touched in this period

- `12.4 Robust Backward advection.ipynb` (Ali built from sent cells), runs in `runs10-21/advection_...`.
- `12.5 Robust Backward Burger stops.ipynb` (built by me), runs in `runs10-22/`.
- `12. Backward Burgers State change.ipynb` modified and `12.1 Backward Burgers Experiments.ipynb` deleted in the
  working tree (git status at the start of this chat; not by me as far as I know — check before committing).
- Memory: `meeting-2026-10-27.md` added; `rebuild-decisions-2026-10.md` updated with the advection setup.
